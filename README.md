# Real-Time Audio Processing API

A FastAPI-based real-time audio processing system for live customer calls that performs speaker diarization, transcription, and sentiment analysis.

## Features

- Real-time microphone input capture
- Speaker diarization (agent vs customer)
- Live speech-to-text transcription
- Sentiment analysis (both text and voice)
- Voice feature extraction (pitch, energy, speaking rate)
- WebSocket-based real-time output streaming

## Project Structure

```
.
├── api/                    # FastAPI routes and WebSocket handlers
├── capture/               # Audio capture module
├── diarization/          # Speaker diarization module
├── transcription/        # Speech-to-text module
├── sentiment/            # Sentiment analysis module
├── tests/                # Unit tests
├── main.py              # FastAPI application entry point
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

## API Endpoints

- `POST /api/start`: Start the audio processing pipeline
- `POST /api/stop`: Stop the audio processing pipeline
- `GET /api/status`: Get current pipeline status
- `WS /ws`: WebSocket endpoint for real-time output

## Testing

Run the test suite:
```bash
pytest
```

## Real-Time Output Format

The system outputs results in real-time with the following format:

```
12:01:23 | SPEAKER_00 | Hello, how can I help you today?
⚠️ [12:01:23] NATURAL (score=0.92)
   Text: NATURAL, Voice: NATURAL
   Voice features - Pitch: 185.23, Energy: 0.12, Rate: 0.08
```

## Requirements

- Python 3.8+
- Working microphone
- Sufficient CPU/GPU for real-time processing
- Internet connection for model downloads

## License

MIT 