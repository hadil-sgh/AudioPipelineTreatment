import numpy as np
import pytest
import asyncio

from diarization.speaker_diarization import SpeakerDiarization  


def generate_sine_wave(freq=440, duration=3, sr=16000):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.mark.asyncio
async def test_process_audio_basic():
    diarizer = SpeakerDiarization(n_speakers=2)
    audio = generate_sine_wave(duration=5)  
   
    chunk_size = 16000  # 1 second
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        await diarizer.process_audio(chunk)
    
    speaker = diarizer.get_current_speaker()
    assert speaker in [0,1,None]  


def test_extract_features_output_shape():
    diarizer = SpeakerDiarization()
    audio = generate_sine_wave(duration=2)
    emb = diarizer._extract_features(audio)
    assert emb is not None
    assert emb.ndim == 2  


def test_detect_speech_no_speech():
    diarizer = SpeakerDiarization()
    silent_audio = np.zeros(int(diarizer.sample_rate * 3), dtype=np.float32)
    segments = diarizer._detect_speech(silent_audio)
    assert segments == []


def test_detect_speech_with_speech():
    diarizer = SpeakerDiarization()
   
    audio = np.concatenate([
        np.zeros(16000, dtype=np.float32),
        np.ones(16000, dtype=np.float32) * 0.1,
        np.zeros(16000, dtype=np.float32)
    ])
    segments = diarizer._detect_speech(audio)
    assert len(segments) >= 1


def test_reset_clears_buffers():
    diarizer = SpeakerDiarization()
    diarizer.raw_buffer = [0.1] * 10
    diarizer.speaker_embeddings = [np.array([1.0])]
    diarizer.speaker_labels = [0]
    diarizer.reset()
    assert diarizer.raw_buffer == []
    assert diarizer.speaker_embeddings == []
    assert diarizer.speaker_labels == []

