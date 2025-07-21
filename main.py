from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional
import uvicorn
import numpy as np
import io
import wave
import base64
import logging
import torch
import torchaudio
from collections import deque
import time

# Import enhanced web pipeline
from enhanced_web_realtime_pipeline import get_pipeline, EnhancedWebRealTimePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Real-Time Audio Processing API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.pipelines: Dict[WebSocket, EnhancedWebRealTimePipeline] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Create a dedicated pipeline for this connection
        pipeline = await get_pipeline()
        pipeline.set_websocket(websocket)
        self.pipelines[websocket] = pipeline
        
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
            # Stop and cleanup the pipeline for this connection
            if websocket in self.pipelines:
                pipeline = self.pipelines[websocket]
                asyncio.create_task(pipeline.stop_session())
                del self.pipelines[websocket]
                
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def get_pipeline(self, websocket: WebSocket) -> Optional[EnhancedWebRealTimePipeline]:
        return self.pipelines.get(websocket)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    pipeline = await manager.get_pipeline(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                logger.debug(f"Received message type: {message_type}")
                
                if message_type == "start_session":
                    session_id = message.get("sessionId", f"session_{int(time.time())}")
                    success = await pipeline.start_session(session_id)
                    
                    response = {
                        "type": "session_response",
                        "success": success,
                        "sessionId": session_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(response))
                    
                elif message_type == "stop_session":
                    await pipeline.stop_session()
                    
                    response = {
                        "type": "session_stopped",
                        "sessionId": pipeline.session_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(response))
                    
                elif message_type == "audio_chunk":
                    # Decode base64 audio data
                    audio_data = base64.b64decode(message["data"])
                    
                    # Process audio through enhanced pipeline
                    result = await pipeline.process_audio_chunk(audio_data)
                    
                    if result:
                        await pipeline.send_result(result)
                        
                elif message_type == "test":
                    test_response = {
                        "type": "test_response",
                        "message": "Enhanced pipeline test successful",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "pipeline_info": pipeline.get_session_info()
                    }
                    await websocket.send_text(json.dumps(test_response))
                    
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    
            except json.JSONDecodeError:
                logger.error("Received invalid JSON data")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                error_response = {
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(error_response))
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.post("/api/start")
async def start_pipeline():
    """Legacy endpoint for backward compatibility"""
    return {
        "status": "Enhanced pipeline ready", 
        "message": "Use WebSocket connection for real-time processing",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/stop")
async def stop_pipeline():
    """Legacy endpoint for backward compatibility"""
    return {
        "status": "Use WebSocket stop_session message", 
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/status")
async def get_status():
    return {
        "status": "Enhanced pipeline running",
        "connections": len(manager.active_connections),
        "active_pipelines": len(manager.pipelines),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "Enhanced Real-Time Audio Processing API",
        "version": "2.0"
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Enhanced Real-Time Audio Processing API started")
    logger.info("WebSocket endpoint: /ws")
    logger.info("Health check: /health")

@app.on_event("shutdown")
async def shutdown_event():
    # Stop all active pipelines
    for pipeline in manager.pipelines.values():
        await pipeline.stop_session()
    logger.info("Enhanced Real-Time Audio Processing API shutting down")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )