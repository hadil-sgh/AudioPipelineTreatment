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
from sentiment.VoiceEmotionRecognizer import VoiceEmotionRecognizer

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
# 2) Real‑time pipeline with diarization + transcription + sentiment/emotion
# ————————————————————————————————————————————————————————————————————————
async def main():
    # 2.1 Initialize capture, STT, Diarization, and Analysis components
    audio_cap = AudioCapture()
    await audio_cap.start()
    device_sr = audio_cap.sample_rate

    # Resample to 16 kHz (model's expected rate)
    target_sr = 16_000
    resampler = torchaudio.transforms.Resample(device_sr, target_sr)
    stt = SpeechToText(sample_rate=target_sr)

    # Diarization: tune threshold & buffer for your use case
    diar = SpeakerDiarization(
        min_speech_duration=0.5,
        threshold=0.03,
        n_speakers=2,
        process_buffer_duration=5.0,
        device="cuda"
    )

    # Initialize sentiment and emotion analyzers
    sentiment_analyzer = RealTimeSentimentAnalyzer(min_words=6)
    voice_analyzer = VoiceEmotionRecognizer()

    # Optional: save to WAV
    wav = wave.open("output.wav", "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(target_sr)

    # Setup Ctrl+C handler
    stop = False
    def on_sigint(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, on_sigint)

    print("[Pipeline] Started — press Ctrl+C to stop")

    # Clock to track elapsed time
    elapsed = 0.0
    chunk_duration = 1.0  # seconds per chunk
    
    # Buffer for accumulating audio per speaker
    speaker_audio_buffers = {}

    try:
        while not stop:
            chunk = audio_cap.get_audio_chunk(min_samples=target_sr)
            if chunk is None:
                await asyncio.sleep(0.05)
                continue

            # 2.2 Resample & normalize
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            audio16 = resampler(tensor).squeeze(0).numpy()
            audio16 /= np.max(np.abs(audio16)) or 1.0

            # 2.3 Write to WAV
            wav.writeframes((audio16 * 32767).astype(np.int16).tobytes())

            # 2.4 Feed diarization & STT
            await diar.process_audio(audio16)
            await stt.process_audio(audio16)

            # 2.5 Fetch new transcriptions & analyze
            new_segs = stt.get_all_transcriptions()
            if new_segs:
                for text in new_segs:
                    spk = diar.get_current_speaker()
                    if spk is None:
                        continue
                        
                    # Accumulate audio for current speaker
                    if spk not in speaker_audio_buffers:
                        speaker_audio_buffers[spk] = []
                    speaker_audio_buffers[spk].extend(audio16)
                    
                    # Analyze sentiment
                    sentiment_result = sentiment_analyzer.analyze(text)
                    
                    # Analyze voice emotion if we have enough audio
                    if len(speaker_audio_buffers[spk]) >= target_sr:  # At least 1 second of audio
                        voice_result = voice_analyzer.analyze(np.array(speaker_audio_buffers[spk]))
                        speaker_audio_buffers[spk] = []  # Reset buffer
                    else:
                        voice_result = {
                            "voice": {
                                "emotion": "NATURAL",
                                "score": 0.5,
                                "features": {"pitch": 0.0, "energy": 0.0, "speaking_rate": 0.0}
                            }
                        }
                    
                    # Print results
                    ts_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
                    print(f"\n{ts_str} | SPEAKER_{spk:02d} | {text}")
                    print(f"⚠️ [{ts_str}] {voice_result['voice']['emotion']} (score={voice_result['voice']['score']:.2f})")
                    print(f"   Text: {sentiment_result['sentiment']}, Voice: {voice_result['voice']['emotion']}")
                    print(f"   Voice features - Pitch: {voice_result['voice']['features']['pitch']:.2f}, "
                          f"Energy: {voice_result['voice']['features']['energy']:.2f}, "
                          f"Rate: {voice_result['voice']['features']['speaking_rate']:.2f}")
                    
                elapsed += chunk_duration
                stt.reset()

    finally:
        await audio_cap.stop()
        wav.close()
        if diar.device == "cuda":
            torch.cuda.empty_cache()
        print("[Pipeline] Stopped — logs in pipeline.log")

if __name__ == "__main__":
    asyncio.run(main())
