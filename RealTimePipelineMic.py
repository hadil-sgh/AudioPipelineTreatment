import asyncio
import logging
import wave
import numpy as np
import signal
import torch
import torchaudio
import time

from capture.audio_capture import AudioCapture
from transcription.speech_to_text import SpeechToText
from diarization.speaker_diarization import SpeakerDiarization
from sentiment.SentimentFromTrans import RealTimeSentimentAnalyzer
from sentiment.voice_emotion_recognizer import voice_emotion_recognizer

# ————————————————————————————————————————————————————————————————————————
# 1) Route all logs to file (pipeline.log); only transcripts print to console
# ————————————————————————————————————————————————————————————————————————
LOG_FILE = "pipeline.log"
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s"))
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(fh)
# remove default console log handlers
for h in list(logging.getLogger().handlers):
    if isinstance(h, logging.StreamHandler):
        logging.getLogger().removeHandler(h)

# ————————————————————————————————————————————————————————————————————————
# 2) AudioPipeline Class
# ————————————————————————————————————————————————————————————————————————
class AudioPipeline:
    def __init__(self, target_sr=16_000, chunk_duration=1.0):
        self.audio_cap = AudioCapture()
        self.target_sr = target_sr
        self.resampler = None  # Will be initialized in start()
        self.stt = SpeechToText(sample_rate=self.target_sr)
        self.diar = SpeakerDiarization(
            min_speech_duration=0.5,
            threshold=0.03,
            n_speakers=2,
            process_buffer_duration=5.0,
            device="cuda"  # Consider making this configurable
        )
        self.sentiment_analyzer = RealTimeSentimentAnalyzer(min_words=6)
        self.voice_analyzer = voice_emotion_recognizer()
        self.wav = None  # Optional: for saving output
        self.elapsed_time = 0.0
        self.chunk_duration = chunk_duration
        self.speaker_audio_buffers = {}
        self.logger = logging.getLogger(__name__)

    async def start(self):
        await self.audio_cap.start()
        device_sr = self.audio_cap.sample_rate
        if device_sr != self.target_sr:
            self.resampler = torchaudio.transforms.Resample(device_sr, self.target_sr)
        else:
            self.resampler = None

        # Optional: save to WAV
        self.wav = wave.open("output.wav", "wb")
        self.wav.setnchannels(1)
        self.wav.setsampwidth(2)
        self.wav.setframerate(self.target_sr)
        self.logger.info("AudioPipeline started.")

    async def process_audio_chunk(self):
        chunk = self.audio_cap.get_audio_chunk(min_samples=self.target_sr)
        if chunk is None:
            await asyncio.sleep(0.05)
            return None

        # Resample & normalize
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        tensor = torch.from_numpy(chunk).unsqueeze(0)

        if self.resampler:
            audio16 = self.resampler(tensor).squeeze(0).numpy()
        else:
            audio16 = tensor.squeeze(0).numpy()

        audio16 /= np.max(np.abs(audio16)) or 1.0

        # Write to WAV
        if self.wav:
            self.wav.writeframes((audio16 * 32767).astype(np.int16).tobytes())

        # Feed diarization & STT
        await self.diar.process_audio(audio16)
        await self.stt.process_audio(audio16)

        # Fetch new transcriptions & analyze
        new_segs = self.stt.get_all_transcriptions()
        results = []
        if new_segs:
            for text in new_segs:
                spk = self.diar.get_current_speaker()
                if spk is None:
                    self.logger.warning("No speaker identified for a segment.")
                    continue

                # Accumulate audio for current speaker
                if spk not in self.speaker_audio_buffers:
                    self.speaker_audio_buffers[spk] = []
                self.speaker_audio_buffers[spk].extend(audio16)

                # Analyze sentiment
                sentiment_result = self.sentiment_analyzer.analyze(text)

                # Analyze voice emotion
                if len(self.speaker_audio_buffers[spk]) >= self.target_sr:  # At least 1 second of audio
                    voice_result = self.voice_analyzer.analyze(np.array(self.speaker_audio_buffers[spk]))
                    self.speaker_audio_buffers[spk] = []  # Reset buffer
                else:
                    voice_result = {
                        "voice": {
                            "emotion": "NATURAL", # Default if not enough audio
                            "score": 0.5,
                            "features": {"pitch": 0.0, "energy": 0.0, "speaking_rate": 0.0}
                        }
                    }

                ts_str = time.strftime('%H:%M:%S', time.gmtime(self.elapsed_time))
                structured_output = {
                    "timestamp": ts_str,
                    "speaker": f"SPEAKER_{spk:02d}",
                    "text": text,
                    "sentiment": sentiment_result,
                    "voice_analysis": voice_result
                }
                results.append(structured_output)
                self.logger.info(f"Processed segment: {structured_output}")

            self.elapsed_time += self.chunk_duration # Increment elapsed time
            self.stt.reset() # Reset STT for next segment group

        return results if results else None

    async def stop(self):
        await self.audio_cap.stop()
        if self.wav:
            self.wav.close()
        if self.diar.device == "cuda":
            torch.cuda.empty_cache()
        self.logger.info("AudioPipeline stopped.")

# ————————————————————————————————————————————————————————————————————————
# 3) Main execution logic
# ————————————————————————————————————————————————————————————————————————
async def main():
    pipeline = AudioPipeline()
    await pipeline.start()

    stop = False
    def on_sigint(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, on_sigint)

    print("[Pipeline] Started — press Ctrl+C to stop")

    try:
        while not stop:
            processed_data = await pipeline.process_audio_chunk()
            if processed_data:
                for data_item in processed_data:
                    # Instead of printing, this data can be sent to an API, database, etc.
                    logging.info(f"Structured output: {data_item}")
            else:
                # Small sleep if no new data to prevent tight loop if get_audio_chunk returns None quickly
                await asyncio.sleep(0.01)
    except Exception as e:
        logging.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        await pipeline.stop()
        print("[Pipeline] Stopped — logs in pipeline.log")

if __name__ == "__main__":
    asyncio.run(main())
