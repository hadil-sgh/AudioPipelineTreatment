import asyncio
import logging
import wave
import numpy as np
import signal
import torch
import torchaudio

from capture.audio_capture import AudioCapture
from transcription.speech_to_text import SpeechToText

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
# 2) Real‑time pipeline without echo
# ————————————————————————————————————————————————————————————————————————
async def main():
    # 2.1 Initialize capture and STT
    audio_cap = AudioCapture()
    await audio_cap.start()
    device_sr = audio_cap.sample_rate

    # resample to 16 kHz (model’s expected rate)
    target_sr = 16_000
    resampler = torchaudio.transforms.Resample(device_sr, target_sr)
    stt = SpeechToText(sample_rate=target_sr)

    # optional: save to WAV
    wav = wave.open("output.wav", "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(target_sr)

    stop = False
    def on_sigint(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, on_sigint)

    print("[Transcription] Started — press Ctrl+C to stop")

    try:
        while not stop:
            # grab 1 s of raw audio (or whatever min_samples you choose)
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

            # 2.3 Write to file
            wav.writeframes((audio16 * 32767).astype(np.int16).tobytes())

            # 2.4 Transcribe just this chunk
            await stt.process_audio(audio16)

            # 2.5 Fetch new segments, print them, then clear
            new_segs = stt.get_all_transcriptions()
            if new_segs:
                # print each in order
                for text in new_segs:
                    print(text)
                # reset buffer so we don't repeat
                stt.reset()

    finally:
        await audio_cap.stop()
        wav.close()
        print("[Transcription] Stopped — logs in pipeline.log")

if __name__ == "__main__":
    asyncio.run(main())
