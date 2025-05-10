from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import List, Dict, Optional
import json
import asyncio
from datetime import datetime
import numpy as np

from capture.audio_capture import AudioCapture
from diarization.speaker_diarization import SpeakerDiarization
from transcription.speech_to_text import SpeechToText
from sentiment.sentiment_analyzer import SentimentAnalyzer

router = APIRouter()

# Global state
pipeline_running = False
active_connections: List[WebSocket] = []

# Initialize components
audio_capture = AudioCapture()
diarization = SpeakerDiarization()
transcription = SpeechToText()
sentiment_analyzer = SentimentAnalyzer()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now - will be replaced with actual pipeline output
            await manager.broadcast(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.post("/start")
async def start_pipeline():
    global pipeline_running
    if pipeline_running:
        raise HTTPException(status_code=400, detail="Pipeline already running")
    
    pipeline_running = True
    asyncio.create_task(run_pipeline())
    return {"status": "Pipeline started"}

@router.post("/stop")
async def stop_pipeline():
    global pipeline_running
    if not pipeline_running:
        raise HTTPException(status_code=400, detail="Pipeline not running")
    
    pipeline_running = False
    return {"status": "Pipeline stopped"}

@router.get("/status")
async def get_status():
    return {"status": "running" if pipeline_running else "stopped"}

async def run_pipeline():
    """Main pipeline function that processes audio in real-time"""
    try:
        # Start audio capture
        audio_capture.start()
        
        while pipeline_running:
            # Get audio chunk
            audio_chunk = audio_capture.get_audio_chunk()
            if audio_chunk is None:
                await asyncio.sleep(0.001)
                continue
            
            # Process audio through pipeline
            await diarization.process_audio(audio_chunk)
            await transcription.process_audio(audio_chunk)
            
            # Get latest transcription
            text = transcription.get_latest_transcription()
            if text:
                # Get current speaker
                speaker = diarization.get_current_speaker()
                speaker_label = f"SPEAKER_{speaker:02d}" if speaker is not None else "UNKNOWN"
                
                # Analyze sentiment
                sentiment = sentiment_analyzer.analyze(text, audio_chunk)
                
                # Format output
                timestamp = datetime.now().strftime("%H:%M:%S")
                output = {
                    "timestamp": timestamp,
                    "speaker": speaker_label,
                    "text": text,
                    "sentiment": sentiment
                }
                
                # Broadcast to all connected clients
                await manager.broadcast(json.dumps(output))
            
            await asyncio.sleep(0.001)
            
    except Exception as e:
        print(f"Error in pipeline: {e}")
    finally:
        audio_capture.stop()
        pipeline_running = False

# Test endpoints
@router.post("/test/audio")
async def test_audio():
    """Test audio capture"""
    try:
        audio_capture.start()
        await asyncio.sleep(2)  # Record for 2 seconds
        audio_capture.stop()
        return {"status": "Audio test completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test/diarization")
async def test_diarization():
    """Test speaker diarization"""
    try:
        # Create test audio
        duration = 5  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        # Process audio
        await diarization.process_audio(audio)
        return {"status": "Diarization test completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test/transcription")
async def test_transcription():
    """Test speech-to-text"""
    try:
        # Create test audio
        duration = 5  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        # Process audio
        await transcription.process_audio(audio)
        return {"status": "Transcription test completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test/sentiment")
async def test_sentiment():
    """Test sentiment analysis"""
    try:
        # Test text
        text_result = sentiment_analyzer.analyze_text("I'm really happy with the service!")
        
        # Test audio
        duration = 5  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        voice_result = sentiment_analyzer.analyze_voice(audio)
        
        return {
            "text_sentiment": text_result,
            "voice_sentiment": voice_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 