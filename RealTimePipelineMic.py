import asyncio
import logging
import wave
import numpy as np
import signal
import torch
import torchaudio

from capture.audio_capture import AudioCapture
from transcription.speech_to_text import SpeechToText

logging.basicConfig(level=logging.INFO)

async def main():
    # 1) Capture at device SR (likely 44100)
    audio_capture = AudioCapture()
    DEVICE_SR = audio_capture.sample_rate

    # 2) Resampler to 16 kHz
    TARGET_SR = 16000
    resampler = torchaudio.transforms.Resample(
        orig_freq=DEVICE_SR, new_freq=TARGET_SR
    )

    # 3) Initialize STT for 16 kHz (no auto_gain arg)
    stt = SpeechToText(sample_rate=TARGET_SR)

    # 4) Open WAV for saving (optional)
    wav_file = wave.open("output.wav", "wb")
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(TARGET_SR)

    stop_flag = False
    def on_sigint(sig, frame):
        nonlocal stop_flag
        print("\n[Main] Stopping…")
        stop_flag = True
    signal.signal(signal.SIGINT, on_sigint)

    await audio_capture.start()
    print("[Main] Capturing…")

    buffer = np.zeros((0,), dtype=np.float32)
    WINDOW_SAMPLES = TARGET_SR * 2  # 2 s windows

    try:
        while not stop_flag:
            chunk = audio_capture.get_audio_chunk(min_samples=1024)
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            # to float32
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)

            # resample to 16 kHz
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            tensor16 = resampler(tensor).squeeze(0).numpy()

            # accumulate into rolling 2 s buffer
            buffer = np.concatenate([buffer, tensor16])
            if buffer.shape[0] < WINDOW_SAMPLES:
                continue

            to_transcribe, buffer = buffer[:WINDOW_SAMPLES], buffer[WINDOW_SAMPLES//2:]
            mx = np.max(np.abs(to_transcribe)) or 1.0
            to_transcribe = to_transcribe / mx

            # run transcription with VAD off
            segments, _ = stt.model.transcribe(
                to_transcribe,
                beam_size=5,
                vad_filter=False
            )
            text = " ".join(s.text for s in segments).strip()
            if text:
                print("🗣️", text)

            # save to WAV
            ints = (to_transcribe * 32767).astype(np.int16)
            wav_file.writeframes(ints.tobytes())

    finally:
        await audio_capture.stop()
        wav_file.close()
        print("[Main] Done.")

if __name__ == "__main__":
    asyncio.run(main())
