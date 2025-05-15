import asyncio
import time
import numpy as np
import librosa

from diarization.speaker_diarization import SpeakerDiarization
from transcription.speech_to_text import SpeechToText

class RealTimeWavPipeline:
    def __init__(
        self,
        wav_path: str,
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        log_path: str = "transcription.log"
    ):
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.log_path = log_path

        # Queues for pipelines
        self.diar_queue = asyncio.Queue()
        self.stt_queue = asyncio.Queue()
        self.diar_results = asyncio.Queue()
        self.stt_results = asyncio.Queue()

        # Components
        self.diar = SpeakerDiarization(device="cuda", process_buffer_duration=5.0)
        self.stt = SpeechToText(device="cuda", sample_rate=sample_rate)

        # Open log file
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.log_file.write("Timestamp | Speaker | Transcription\n")
        self.log_file.flush()

        # State to avoid repeats
        self.last_speaker = None
        self.last_text = ""

    def load_wav_chunks(self):
        audio, sr = librosa.load(self.wav_path, sr=self.sample_rate, mono=True)
        total_samples = len(audio)
        chunk_size = int(self.sample_rate * self.chunk_duration)
        timestamp = 0.0
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            chunk = audio[start:end]
            yield chunk.astype(np.float32), timestamp
            timestamp += self.chunk_duration

    async def producer(self):
        for audio, ts in self.load_wav_chunks():
            packet = {"audio": audio, "timestamp": ts}
            await self.diar_queue.put(packet)
            await self.stt_queue.put(packet)
            await asyncio.sleep(self.chunk_duration)
        await self.diar_queue.put(None)
        await self.stt_queue.put(None)

    async def diar_consumer(self):
        while True:
            packet = await self.diar_queue.get()
            if packet is None:
                await self.diar.finalize_processing()
                break
            await self.diar.process_audio(packet['audio'])
            speaker = self.diar.get_current_speaker()
            ts = packet['timestamp']
            if speaker is not None:
                await self.diar_results.put({'start': ts, 'speaker': speaker})

    async def stt_consumer(self):
        while True:
            packet = await self.stt_queue.get()
            if packet is None:
                break
            await self.stt.process_audio(packet['audio'])
            text = self.stt.get_latest_transcription()
            ts = packet['timestamp']
            if text:
                await self.stt_results.put({'start': ts, 'text': text})

    async def aligner(self):
        while True:
            diar_evt = await self.diar_results.get()
            ts = diar_evt['start']
            speaker = diar_evt['speaker']
            # get best matching text event
            text = None
            while not self.stt_results.empty():
                stt_evt = self.stt_results.get_nowait()
                if abs(stt_evt['start'] - ts) < self.chunk_duration:
                    text = stt_evt['text']
            if text is None:
                continue
            # Only log if speaker or text changed
            if speaker != self.last_speaker or text != self.last_text:
                ts_str = time.strftime('%H:%M:%S', time.gmtime(ts))
                # If speaker changed, start new line
                if speaker != self.last_speaker:
                    line = f"{ts_str} | SPEAKER_{speaker:02d} | {text}\n"
                else:
                    # same speaker: only log new part
                    new_part = text.replace(self.last_text, '').strip()
                    line = f"{ts_str} | SPEAKER_{speaker:02d} | {new_part}\n"
                # Print and write
                print(line, end='')
                self.log_file.write(line)
                self.log_file.flush()
                self.last_speaker = speaker
                self.last_text = text

    async def run(self):
        try:
            await asyncio.gather(
                self.producer(),
                self.diar_consumer(),
                self.stt_consumer(),
                self.aligner()
            )
        finally:
            self.log_file.close()

if __name__ == "__main__":
    pipeline = RealTimeWavPipeline(
        "ordinary.wav",
        sample_rate=16000,
        chunk_duration=1.0,
        log_path="transcription.log"
    )
    asyncio.run(pipeline.run())