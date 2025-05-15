import asyncio
import logging
import wave
import numpy as np
import signal
import torch
import torchaudio
import io
import contextlib

from capture.audio_capture import AudioCapture
from transcription.speech_to_text import SpeechToText

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configure logging to file only
# ──────────────────────────────────────────────────────────────────────────────
LOG_FILE = "pipeline.log"

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)

# remove any console handlers so nothing logs to stdout
for h in list(root_logger.handlers):
    if isinstance(h, logging.StreamHandler):
        root_logger.removeHandler(h)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Real‑time pipeline
# ──────────────────────────────────────────────────────────────────────────────
async def main():
    # 2.1) Instantiate capture and STT
    #    wrap capture startup in redirect_stdout to swallow its print()s
    with contextlib.redirect_stdout(io.StringIO()):
        audio_cap = AudioCapture()
        await audio_cap.start()

    DEVICE_SR = audio_cap.sample_rate
    TARGET_SR = 16000

    # 2.2) Set up resampler & STT
    resampler = torchaudio.transforms.Resample(orig_freq=DEVICE_SR, new_freq=TARGET_SR)
    stt = SpeechToText(sample_rate=TARGET_SR)

    # 2.3) (Optional) open WAV for saving
    wav_file = wave.open("output.wav", "wb")
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(TARGET_SR)

    # 2.4) Graceful shutdown via Ctrl+C
    stop_flag = False
    def on_sigint(sig, frame):
        nonlocal stop_flag
        stop_flag = True
    signal.signal(signal.SIGINT, on_sigint)

    # 2.5) Tell user we’re live (only this print goes to console)
    print("[Transcription] Pipeline started. Speak now…")

    # rolling buffer for 2s windows
    buffer = np.zeros((0,), dtype=np.float32)
    WINDOW_SAMPLES = TARGET_SR * 2

    try:
        while not stop_flag:
            chunk = audio_cap.get_audio_chunk(min_samples=1024)
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            # cast & resample
            chunk = chunk.astype(np.float32) if chunk.dtype != np.float32 else chunk
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            tensor16 = resampler(tensor).squeeze(0).numpy()

            # accumulate into rolling window
            buffer = np.concatenate([buffer, tensor16])
            if buffer.shape[0] < WINDOW_SAMPLES:
                continue

            to_transcribe, buffer = buffer[:WINDOW_SAMPLES], buffer[WINDOW_SAMPLES//2:]
            to_transcribe /= np.max(np.abs(to_transcribe)) or 1.0

            # transcribe with VAD off
            segments, _ = stt.model.transcribe(
                to_transcribe,
                beam_size=5,
                vad_filter=False
            )
            text = " ".join(s.text for s in segments).strip()
            if text:
                # **only** transcripts are printed
                print(text)

            # save audio window
            ints = (to_transcribe * 32767).astype(np.int16)
            wav_file.writeframes(ints.tobytes())

    finally:
        await audio_cap.stop()
        wav_file.close()
        print("[Transcription] Pipeline stopped. Logs in pipeline.log")

if __name__ == "__main__":
    asyncio.run(main())
