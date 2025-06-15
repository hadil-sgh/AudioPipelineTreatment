import soundfile as sf
import numpy as np
from scipy.signal import resample
from sentiment.VoiceSentiment import RealTimeEmotionRecognizer  # adjust import as needed

# 1. Load your WAV
audio, sr = sf.read("call.wav", dtype="float32")

# Convert to mono if stereo
if audio.ndim == 2:
    audio = np.mean(audio, axis=1)

target_sr = 16000
if sr != target_sr:
    # Resample audio to 16 kHz
    num_samples = int(len(audio) * target_sr / sr)
    audio = resample(audio, num_samples)
    sr = target_sr

# 2. Initialize the recognizer (buffer_duration in seconds)
recognizer = RealTimeEmotionRecognizer(buffer_duration=30.0)

# 3. Define a chunk size (e.g. 1 s)
chunk_size = int(sr * 1.0)

# 4. Stream through the file in chunks
for start in range(0, len(audio), chunk_size):
    chunk = audio[start : start + chunk_size]
    # if last chunk is shorter, you can pad with zeros:
    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
    emotion = recognizer.push_audio(chunk)
    if emotion is not None:
        print(f"Detected emotion at ~{start/sr:.1f}s: {emotion}")
