import asyncio
import websockets
import json
from datetime import datetime
import sys

async def connect_and_listen():
    """Connect to the WebSocket server and listen for messages"""
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to server. Listening for messages...")
            
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Format and print the output
                    timestamp = data["timestamp"]
                    speaker = data["speaker"]
                    text = data["text"]
                    sentiment = data["sentiment"]
                    
                    # Print main line
                    print(f"\n{timestamp} | {speaker} | {text}")
                    
                    # Print sentiment information
                    text_emotion = sentiment["text"]["emotion"]
                    text_score = sentiment["text"]["score"]
                    voice_emotion = sentiment["voice"]["emotion"]
                    voice_score = sentiment["voice"]["score"]
                    features = sentiment["voice"]["features"]
                    
                    print(f"⚠️ [{timestamp}] {text_emotion} (score={text_score:.2f})")
                    print(f"   Text: {text_emotion}, Voice: {voice_emotion}")
                    print(f"   Voice features - Pitch: {features['pitch']:.2f}, "
                          f"Energy: {features['energy']:.2f}, "
                          f"Rate: {features['speaking_rate']:.2f}")
                    
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by server")
                    break
                except json.JSONDecodeError:
                    print("Received invalid JSON")
                except Exception as e:
                    print(f"Error: {e}")
                    
    except websockets.exceptions.ConnectionRefused:
        print("Could not connect to server. Make sure the server is running.")
    except KeyboardInterrupt:
        print("\nDisconnected by user")
    except Exception as e:
        print(f"Error: {e}")

async def start_pipeline():
    """Start the audio processing pipeline"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/api/start") as response:
            if response.status == 200:
                print("Pipeline started successfully")
            else:
                print(f"Failed to start pipeline: {response.status}")

async def stop_pipeline():
    """Stop the audio processing pipeline"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/api/stop") as response:
            if response.status == 200:
                print("Pipeline stopped successfully")
            else:
                print(f"Failed to stop pipeline: {response.status}")

async def main():
    """Main function"""
    # Start the pipeline
    await start_pipeline()
    
    try:
        # Connect and listen for messages
        await connect_and_listen()
    finally:
        # Stop the pipeline
        await stop_pipeline()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0) 