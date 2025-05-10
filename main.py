from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional
import uvicorn

app = FastAPI(title="Real-Time Audio Processing API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Pipeline state
pipeline_running = False

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now - will be replaced with actual pipeline output
            await manager.broadcast(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/start")
async def start_pipeline():
    global pipeline_running
    if not pipeline_running:
        pipeline_running = True
        # Start the pipeline in a background task
        asyncio.create_task(run_pipeline())
        return {"status": "Pipeline started"}
    return {"status": "Pipeline already running"}

@app.post("/api/stop")
async def stop_pipeline():
    global pipeline_running
    pipeline_running = False
    return {"status": "Pipeline stopped"}

@app.get("/api/status")
async def get_status():
    return {"status": "running" if pipeline_running else "stopped"}

async def run_pipeline():
    """Main pipeline function that will be implemented with actual processing logic"""
    while pipeline_running:
        # This is a placeholder for the actual pipeline implementation
        # Will be replaced with real audio processing logic
        await asyncio.sleep(1)
        timestamp = datetime.now().strftime("%H:%M:%S")
        mock_output = {
            "timestamp": timestamp,
            "speaker": "SPEAKER_00",
            "text": "Processing audio...",
            "sentiment": {
                "text": "NATURAL",
                "voice": "NATURAL",
                "score": 0.92
            },
            "voice_features": {
                "pitch": 185.23,
                "energy": 0.12,
                "rate": 0.08
            }
        }
        await manager.broadcast(json.dumps(mock_output))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 