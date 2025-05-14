import asyncio
import time
import sounddevice as sd
import numpy as np
import wave
from capture.audio_capture import AudioCapture



class AudioCaptureTester:
    def __init__(self, duration_seconds=5, min_chunk_samples=16000):
        self.duration_seconds = duration_seconds
        self.min_chunk_samples = min_chunk_samples
        self.capture = AudioCapture()
        self.captured_chunks = []
        self.latencies = []
        self.dropped_chunks = 0

    async def run_test(self):
        print(f"🎤 Starting audio capture test for {self.duration_seconds} seconds...")
        await self.capture.start()
        start_time = time.time()

        while time.time() - start_time < self.duration_seconds:
            t1 = time.time()
            chunk = self.capture.get_audio_chunk(min_samples=self.min_chunk_samples)
            t2 = time.time()

            if chunk is not None:
                self.latencies.append(t2 - t1)
                self.captured_chunks.append(chunk)
                print(f"✅ Captured chunk of {len(chunk)} samples (Latency: {t2 - t1:.3f}s)")
            else:
                self.dropped_chunks += 1
                print("⚠️  No chunk available")

            await asyncio.sleep(0.1)

        await self.capture.stop()
        print("🛑 Capture stopped.")
        self._show_metrics()
        self._playback()
        self._save_wav("test_output.wav")

    def _show_metrics(self):
        total_captured = len(self.captured_chunks)
        total_samples = sum(len(c) for c in self.captured_chunks)
        avg_latency = np.mean(self.latencies) if self.latencies else 0
        max_latency = np.max(self.latencies) if self.latencies else 0

        print("\n📊 Test Metrics:")
        print(f"🧩 Chunks captured: {total_captured}")
        print(f"🚫 Chunks dropped: {self.dropped_chunks}")
        print(f"🎚️ Total samples: {total_samples}")
        print(f"⏱️ Avg latency: {avg_latency:.3f}s")
        print(f"⏱️ Max latency: {max_latency:.3f}s")

    def _playback(self):
        print("\n🔊 Playing back recorded audio...")
        audio = np.concatenate(self.captured_chunks)
        sd.play(audio, samplerate=self.capture.sample_rate)
        sd.wait()
        print("✅ Playback finished.")

    def _save_wav(self, filename):
        print(f"\n💾 Saving WAV to {filename}...")
        audio = np.concatenate(self.captured_chunks)
        scaled = np.int16(audio * 32767)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.capture.sample_rate)
            wf.writeframes(scaled.tobytes())
        print("✅ File saved.")


if __name__ == "__main__":
    asyncio.run(AudioCaptureTester().run_test())
