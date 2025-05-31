from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional
import uvicorn
import logging # Added import

# Assuming RealTimePipelineMic.py is in the same directory or accessible via PYTHONPATH
from RealTimePipelineMic import AudioPipeline

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
current_pipeline_task: Optional[asyncio.Task] = None
current_pipeline_instance: Optional[AudioPipeline] = None

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
async def start_pipeline_endpoint(): # Renamed for clarity
    global pipeline_running, current_pipeline_task, current_pipeline_instance
    if not pipeline_running:
        pipeline_running = True
        current_pipeline_instance = AudioPipeline()
        current_pipeline_task = asyncio.create_task(run_pipeline(current_pipeline_instance))
        return {"status": "Pipeline started"}
    return {"status": "Pipeline already running"}

@app.post("/api/stop")
async def stop_pipeline_endpoint(): # Renamed for clarity
    global pipeline_running, current_pipeline_task, current_pipeline_instance
    if pipeline_running and current_pipeline_task:
        pipeline_running = False # Signal run_pipeline to stop
        try:
            await asyncio.wait_for(current_pipeline_task, timeout=10.0) # Wait for task to finish
        except asyncio.TimeoutError:
            current_pipeline_task.cancel() # Force cancel if it doesn't stop gracefully
            if current_pipeline_instance: # Attempt to stop pipeline even if task cancelled
                await current_pipeline_instance.stop()
            return {"status": "Pipeline stop timed out, task cancelled"}
        finally:
            current_pipeline_task = None
            current_pipeline_instance = None
        return {"status": "Pipeline stopped"}
    return {"status": "Pipeline not running"}

@app.get("/api/status")
async def get_status():
    global current_pipeline_task
    status_str = "stopped"
    if pipeline_running:
        status_str = "running"
    if current_pipeline_task and current_pipeline_task.done():
        status_str = "finished_or_error" # Task is done but pipeline_running might not be false yet
    return {"status": status_str, "pipeline_running_flag": pipeline_running}


async def run_pipeline(pipeline: AudioPipeline):
    """Main pipeline function that processes audio using AudioPipeline"""
    global pipeline_running, current_pipeline_instance
    try:
        await pipeline.start()
        while pipeline_running:
            processed_data_list = await pipeline.process_audio_chunk()
            if processed_data_list:
                for data_item in processed_data_list:
                    # Format data as per requirements
                    ts = data_item.get("timestamp", "N/A")
                    speaker = data_item.get("speaker", "SPEAKER_UNKNOWN")
                    text = data_item.get("text", "")

                    sentiment_info = data_item.get("sentiment", {})
                    text_sentiment = sentiment_info.get("sentiment", "N/A") # Assuming 'sentiment' field inside 'sentiment' dict
                                                                          # Based on previous task, sentiment_result = {'sentiment': 'POSITIVE/NEGATIVE/NEUTRAL'}
                                                                          # This was changed in AudioPipeline to be just the string. Let's assume it's sentiment_info = "POSITIVE"

                    voice_analysis = data_item.get("voice_analysis", {})
                    voice_info = voice_analysis.get("voice", {})
                    voice_emotion = voice_info.get("emotion", "N/A")
                    voice_score = voice_info.get("score", 0.0)

                    features = voice_info.get("features", {})
                    pitch = features.get("pitch", 0.0)
                    energy = features.get("energy", 0.0)
                    s_rate = features.get("speaking_rate", 0.0)

                    # Correcting sentiment access based on AudioPipeline output structure
                    # The AudioPipeline returns:
                    # "sentiment": sentiment_result, -> where sentiment_result = RealTimeSentimentAnalyzer.analyze(text)
                    # RealTimeSentimentAnalyzer.analyze returns {'sentiment': 'POSITIVE', 'score': 0.99} or similar
                    # So text_sentiment should be sentiment_info.get("sentiment")
                    text_sentiment_val = "N/A"
                    if isinstance(sentiment_info, dict):
                        text_sentiment_val = sentiment_info.get("sentiment", "N/A")
                    elif isinstance(sentiment_info, str): # Fallback if it's directly a string
                        text_sentiment_val = sentiment_info


                    formatted_string = (
                        f"{ts} | {speaker} | {text}\n"
                        f"⚠️ [{ts}] {voice_emotion.upper()} (score={voice_score:.2f})\n"
                        f"   Text: {text_sentiment_val.upper()}, Voice: {voice_emotion.upper()}\n"
                        f"   Voice features - Pitch: {pitch:.2f}, Energy: {energy:.2f}, Rate: {s_rate:.2f}"
                    )
                    await manager.broadcast(formatted_string)
            else:
                await asyncio.sleep(0.01)  # Yield control if no data
    except asyncio.CancelledError:
        logging.info("Pipeline task cancelled.")
    except Exception as e:
        logging.error(f"Error in run_pipeline: {e}", exc_info=True)
        # Optionally broadcast an error message to clients
        await manager.broadcast(json.dumps({"error": str(e)}))
    finally:
        if pipeline: # ensure pipeline exists
            logging.info("Stopping pipeline instance...")
            await pipeline.stop()
        pipeline_running = False # Ensure this is set
        # current_pipeline_instance = None # This will be reset in stop_pipeline_endpoint
        logging.info("run_pipeline finished.")


if __name__ == "__main__":
    # Basic logging setup for the main application
    # The AudioPipeline itself logs to 'pipeline.log' via its own logger
    # This sets up logging for main.py itself.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 