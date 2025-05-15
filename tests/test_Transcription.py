import asyncio
import soundfile as sf
import numpy as np
import torch
import torchaudio
from transcription.speech_to_text import SpeechToText  # adjust import

async def transcribe_wav(path: str):
    # 1. Read the entire WAV (supports any sample rate)
    audio, sr = sf.read(path, always_2d=False)

    # 2. Convert to float32 and normalize
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    audio = audio / np.max(np.abs(audio))

    # 3. Resample to 16 kHz if needed
    target_sr = 16000
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz...")
        # waveform shape [samples], convert to tensor [1, samples]
        tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        tensor = resampler(tensor)
        audio = tensor.squeeze(0).numpy()
        sr = target_sr

    # 4. Initialize STT with correct sample rate
    stt = SpeechToText(sample_rate=sr)

    # 5. Process in chunks matching the buffer logic
    chunk_size = 1024
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        await stt.process_audio(chunk)
        # optional sleep to simulate real-time
        await asyncio.sleep(0.001)

    # 6. Output full transcript
    print("=== Full Transcript ===")
    for segment in stt.get_all_transcriptions():
        print(segment)

if __name__ == "__main__":
    import sys
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "output.wav"
    asyncio.run(transcribe_wav(wav_path))
