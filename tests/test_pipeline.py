import pytest
import numpy as np
import asyncio
from datetime import datetime

from capture.audio_capture import AudioCapture
from diarization.speaker_diarization import SpeakerDiarization
from transcription.speech_to_text import SpeechToText
from sentiment.voice_emotion_recognizer import SentimentAnalyzer

@pytest.fixture
def audio_capture():
    return AudioCapture()

@pytest.fixture
def diarization():
    return SpeakerDiarization()

@pytest.fixture
def transcription():
    return SpeechToText()

@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()

def create_test_audio(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Create a test audio signal"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

@pytest.mark.asyncio
async def test_audio_capture(audio_capture):
    """Test audio capture functionality"""
    # Start recording
    audio_capture.start()
    
    # Record for 2 seconds
    await asyncio.sleep(2)
    
    # Stop recording
    audio_capture.stop()
    
    # Check if we got any audio data
    chunk = audio_capture.get_audio_chunk()
    assert chunk is not None
    assert isinstance(chunk, np.ndarray)
    assert len(chunk) > 0

@pytest.mark.asyncio
async def test_diarization(diarization):
    """Test speaker diarization"""
    # Create test audio
    audio = create_test_audio()
    
    # Process audio
    await diarization.process_audio(audio)
    
    # Check if we got speaker labels
    speaker = diarization.get_current_speaker()
    assert speaker is not None
    assert isinstance(speaker, int)
    assert speaker in [0, 1]  # Should be one of two speakers

@pytest.mark.asyncio
async def test_transcription(transcription):
    """Test speech-to-text"""
    # Create test audio
    audio = create_test_audio()
    
    # Process audio
    await transcription.process_audio(audio)
    
    # Check if we got transcription
    text = transcription.get_latest_transcription()
    assert text is not None
    assert isinstance(text, str)

@pytest.mark.asyncio
async def test_sentiment_analysis(sentiment_analyzer):
    """Test sentiment analysis"""
    # Test text sentiment
    text = "I'm really happy with the service!"
    text_result = sentiment_analyzer.analyze_text(text)
    assert text_result is not None
    assert "emotion" in text_result
    assert "score" in text_result
    
    # Test voice sentiment
    audio = create_test_audio()
    voice_result = sentiment_analyzer.analyze_voice(audio)
    assert voice_result is not None
    assert "emotion" in voice_result
    assert "score" in voice_result
    assert "features" in voice_result
    
    # Test combined analysis
    combined_result = sentiment_analyzer.analyze(text, audio)
    assert combined_result is not None
    assert "text" in combined_result
    assert "voice" in combined_result

@pytest.mark.asyncio
async def test_full_pipeline():
    """Test the full pipeline integration"""
    # Initialize components
    audio_capture = AudioCapture()
    diarization = SpeakerDiarization()
    transcription = SpeechToText()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Create test audio
    audio = create_test_audio()
    
    # Process through pipeline
    await diarization.process_audio(audio)
    await transcription.process_audio(audio)
    
    # Get results
    speaker = diarization.get_current_speaker()
    text = transcription.get_latest_transcription()
    
    if text:
        sentiment = sentiment_analyzer.analyze(text, audio)
        
        # Verify output format
        assert speaker is not None
        assert text is not None
        assert sentiment is not None
        assert "text" in sentiment
        assert "voice" in sentiment

def test_voice_features(sentiment_analyzer):
    """Test voice feature extraction"""
    # Create test audio
    audio = create_test_audio()
    
    # Extract features
    features = sentiment_analyzer._extract_voice_features(audio)
    
    # Verify features
    assert "pitch" in features
    assert "energy" in features
    assert "speaking_rate" in features
    
    # Check value ranges
    assert 0 <= features["pitch"] <= 1000  # Typical pitch range
    assert 0 <= features["energy"] <= 1    # Normalized energy
    assert 0 <= features["speaking_rate"]  # Positive speaking rate 