

# 🎯 AI-Powered Real-Time Audio Analysis Pipeline

A comprehensive **FastAPI-based real-time audio processing system** designed for live customer service call analysis. This system provides real-time speaker diarization, transcription, sentiment analysis, and emotion recognition with WebSocket streaming capabilities.

## ✨ Key Features

### 🎤 **Audio Processing**
- Real-time microphone input capture with optimized audio buffers
- Voice Activity Detection (VAD) for efficient processing
- Multi-channel audio support with noise reduction
- Adaptive audio quality based on connection

### 👥 **Speaker Intelligence**
- Advanced speaker diarization (Agent vs Customer identification)
- Speaker embedding and clustering
- Real-time speaker switching detection
- Voice characteristic analysis

### 📝 **Transcription & Language**
- Live speech-to-text with **Faster-Whisper** models
- Multi-language support and automatic language detection
- Real-time transcription streaming
- Confidence scoring and error correction

### 💭 **Sentiment & Emotion Analysis**
- **Dual-mode sentiment analysis**: Text-based and Voice-based
- Real-time emotion recognition from speech patterns
- Voice feature extraction (pitch, energy, speaking rate, tone)
- Conversation flow analysis and emotional trajectory tracking

### 🌐 **Real-Time Communication**
- WebSocket-based real-time data streaming
- Angular frontend integration
- RESTful API endpoints for control and monitoring
- Real-time dashboard with live updates

## 🏗️ Project Architecture

```
AudioPipelineTreatment/
├── 📁 capture/                    # Audio capture and processing
│   ├── audio_capture.py          # Main audio capture logic
│   └── __init__.py
├── 📁 diarization/               # Speaker diarization
│   ├── speaker_diarization.py   # Speaker identification
│   ├── test_diarization.py      # Diarization tests
│   └── newdir.py
├── 📁 transcription/             # Speech-to-text
│   └── speech_to_text.py        # Whisper-based transcription
├── 📁 sentiment/                 # Sentiment analysis modules
│   ├── SentimentFromTrans.py    # Text-based sentiment
│   ├── VoiceEmotionRecognizer.py # Voice emotion analysis
│   ├── VoiceSentiment.py        # Voice sentiment analysis
│   └── TranscriptSentiment.py   # Transcript processing
├── 📁 tests/                     # Comprehensive test suite
│   ├── test_pipeline.py         # Pipeline integration tests
│   ├── test_realtime_sentiment.py
│   └── testdiarizatio.py
├── 📁 models/                    # Pre-trained model storage
│   ├── wav2vec2-lg-xlsr-en-speech-emotion-recognition/
│   └── faster-whisper-base.en/
├── 📁 logs/                      # Application logs
├── 📁 tmp/                       # Temporary processing files
├── 🐍 main.py                    # FastAPI application entry
├── 🐍 enhanced_web_realtime_pipeline.py  # Core pipeline
├── 🐍 RealTimePipelineMic.py     # Microphone pipeline
├── 🐍 RealTimeWavPipeline.py     # WAV file pipeline
├── 📋 requirements.txt           # Python dependencies
├── 🐳 dockerfile                 # Docker configuration
└── 📚 PROCESS_FLOW_DOCUMENTATION.md  # Detailed technical docs
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (3.12 recommended)
- **Working microphone** for real-time audio capture
- **NVIDIA GPU** (optional, for accelerated processing)
- **4GB+ RAM** for model loading
- **Internet connection** (for initial model downloads)

### Installation

1. **Clone and Navigate**
```powershell
git clone <repository-url>
cd AudioPipelineTreatment
```

2. **Create Virtual Environment**
```powershell
python -m venv venv
venv\Scripts\activate
```

3. **Install Dependencies**
```powershell
pip install -r requirements.txt
```

4. **Download Models** (Automatic on first run)
   - Faster-Whisper models will download automatically
   - Voice emotion models will be cached locally
   - Speaker diarization models will be loaded from HuggingFace

5. **Run the Application**
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 🌐 Access Points
- **API Documentation**: http://localhost:8000/docs
- **WebSocket Endpoint**: ws://localhost:8000/ws
- **Health Check**: http://localhost:8000/health

## 📡 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint with welcome message |
| `GET` | `/health` | System health and status check |
| `POST` | `/upload-audio` | Upload audio file for batch processing |
| `WS` | `/ws` | WebSocket for real-time audio streaming |

### WebSocket Communication

#### **Connection Flow**
1. Connect to `ws://localhost:8000/ws`
2. Send audio data as base64-encoded chunks
3. Receive real-time analysis results

#### **Message Format - Incoming (Audio Data)**
```json
{
  "type": "audio_chunk",
  "data": "base64-encoded-audio-chunk",
  "sampleRate": 16000,
  "channels": 1
}
```

#### **Message Format - Outgoing (Analysis Results)**
```json
{
  "timestamp": "2025-08-20T12:34:56",
  "speaker": "SPEAKER_00",
  "transcription": "Hello, how can I help you today?",
  "sentiment": {
    "text_sentiment": "POSITIVE",
    "text_confidence": 0.92,
    "voice_emotion": "HAPPY",
    "voice_confidence": 0.87,
    "combined_sentiment": "POSITIVE"
  },
  "voice_features": {
    "pitch": 185.23,
    "energy": 0.12,
    "speaking_rate": 0.08,
    "tone_stability": 0.76
  },
  "speaker_info": {
    "speaker_id": "SPEAKER_00",
    "speaker_type": "agent",
    "confidence": 0.94
  }
}
```

## 🧪 Testing

### Run Complete Test Suite
```powershell
pytest tests/ -v
```

### Run Specific Tests
```powershell
# Test real-time sentiment analysis
pytest tests/test_realtime_sentiment.py -v

# Test pipeline integration
pytest tests/test_pipeline.py -v

# Test speaker diarization
pytest tests/testdiarizatio.py -v
```

### Manual Testing Tools
```powershell
# Test audio capture directly
python AudioCaptureTester.py

# Test microphone pipeline
python RealTimePipelineMic.py

# Test WAV file processing
python RealTimeWavPipeline.py
```

## 📊 Real-Time Output Examples

### Console Output Format
```
🕐 12:01:23 | 👤 SPEAKER_00 | "Hello, how can I help you today?"
📊 [12:01:23] SENTIMENT: POSITIVE (confidence=0.92)
   📝 Text: POSITIVE | 🎵 Voice: HAPPY
   🎙️  Voice Features - Pitch: 185.23Hz, Energy: 0.12, Rate: 2.1 words/sec

🕐 12:01:28 | 👤 SPEAKER_01 | "I'm having trouble with my account."
📊 [12:01:28] SENTIMENT: FRUSTRATED (confidence=0.78)
   📝 Text: NEUTRAL | 🎵 Voice: FRUSTRATED  
   🎙️  Voice Features - Pitch: 165.45Hz, Energy: 0.18, Rate: 1.8 words/sec
```

### WebSocket JSON Stream
```json
{
  "session_id": "sess_1724150456",
  "timestamp": "12:01:23",
  "analysis": {
    "speaker": {
      "id": "SPEAKER_00",
      "type": "agent",
      "confidence": 0.94
    },
    "transcription": {
      "text": "Hello, how can I help you today?",
      "confidence": 0.96,
      "language": "en"
    },
    "sentiment": {
      "overall": "POSITIVE",
      "text_based": "POSITIVE",
      "voice_based": "HAPPY",
      "confidence": 0.92
    },
    "emotions": {
      "primary": "HAPPY",
      "secondary": "CONFIDENT",
      "arousal": 0.7,
      "valence": 0.8
    },
    "voice_analytics": {
      "pitch_hz": 185.23,
      "energy_db": -12.4,
      "speaking_rate_wps": 2.1,
      "pause_ratio": 0.15,
      "tone_stability": 0.76
    }
  }
}
```

## 🔧 Configuration & Customization

### Environment Variables
```powershell
# Model Configuration
WHISPER_MODEL_SIZE=base.en     # tiny, base, small, medium, large
EMOTION_MODEL_PATH=./models/   # Custom emotion model path
ENABLE_GPU=true                # Enable GPU acceleration

# Audio Settings  
SAMPLE_RATE=16000             # Audio sample rate
BUFFER_SIZE=1024              # Audio buffer size
VAD_THRESHOLD=0.5             # Voice activity detection threshold

# API Settings
HOST=0.0.0.0                  # API host
PORT=8000                     # API port
DEBUG=false                   # Debug mode
```

### Performance Tuning

#### For Real-Time Performance
- Use **GPU acceleration** when available
- Adjust `BUFFER_SIZE` based on your hardware
- Consider **smaller Whisper models** for faster processing
- Enable **VAD** to process only speech segments

#### For Accuracy
- Use **larger Whisper models** (medium/large)
- Increase **speaker diarization** sensitivity
- Enable **emotion model ensembling**

## 🐳 Docker Deployment

### Build and Run
```powershell
# Build the container
docker build -t audio-pipeline .

# Run with GPU support
docker run --gpus all -p 8000:8000 audio-pipeline

# Run CPU-only
docker run -p 8000:8000 audio-pipeline
```

### Docker Compose
```yaml
version: '3.8'
services:
  audio-pipeline:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENABLE_GPU=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 🛠️ Technical Stack

### Core Technologies
- **FastAPI** - Modern, fast web framework for APIs
- **WebSockets** - Real-time bidirectional communication
- **PyTorch** - Deep learning framework for ML models
- **Faster-Whisper** - Optimized speech recognition
- **PyAnnote.Audio** - Speaker diarization and audio analysis
- **NumPy & SciPy** - Numerical computing and signal processing

### Machine Learning Models
- **Wav2Vec2-XLS-R** - Voice emotion recognition
- **Faster-Whisper-Base/Small** - Speech-to-text transcription
- **PyAnnote Speaker Diarization** - Speaker identification
- **Custom Sentiment Models** - Text and voice sentiment analysis

### Audio Processing
- **PyAudio** - Real-time audio I/O
- **LibROSA** - Audio feature extraction
- **TorchAudio** - Audio processing with PyTorch
- **Voice Activity Detection** - Efficient audio segmentation

## 🔍 Troubleshooting

### Common Issues

#### **"No audio input detected"**
```powershell
# Check microphone permissions
# Verify audio device in Device Manager
# Test with: python AudioCaptureTester.py
```

#### **"Model download failed"**
```powershell
# Check internet connection
# Clear cache: rm -rf ~/.cache/huggingface/
# Manual download: huggingface-cli download Systran/faster-whisper-base.en
```

#### **"WebSocket connection refused"**
```powershell
# Verify server is running: curl http://localhost:8000/health
# Check firewall settings
# Ensure port 8000 is available
```

#### **"High CPU/Memory usage"**
```powershell
# Reduce model size in config
# Enable GPU acceleration
# Adjust buffer sizes
# Close other applications
```

### Performance Optimization

#### **For Low-Latency**
- Use `tiny` or `base` Whisper models
- Reduce audio buffer size
- Enable GPU processing
- Optimize VAD thresholds

#### **For High Accuracy**
- Use `medium` or `large` Whisper models
- Increase speaker diarization sensitivity
- Enable emotion model ensembling
- Use higher quality audio input

### Debug Mode
```powershell
# Enable detailed logging
export DEBUG=true
python main.py

# View logs
tail -f pipeline.log
```

## 📚 Documentation

- **[PROCESS_FLOW_DOCUMENTATION.md](PROCESS_FLOW_DOCUMENTATION.md)** - Detailed technical implementation
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI
- **[Model Documentation](models/README.md)** - ML model specifications
- **[Testing Guide](tests/README.md)** - Comprehensive testing procedures

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Standards
- Follow PEP 8 for Python code style
- Add docstrings to all functions and classes
- Write unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/hadil-sgh/AudioPipelineTreatment/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hadil-sgh/AudioPipelineTreatment/discussions)
- **Documentation**: [Process Flow Documentation](PROCESS_FLOW_DOCUMENTATION.md)

### System Requirements
- **Minimum**: Python 3.8, 4GB RAM, Intel i5 or equivalent
- **Recommended**: Python 3.12, 8GB RAM, NVIDIA GPU, Intel i7 or equivalent
- **Operating Systems**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

---

**🚀 Ready to get started?** Run `uvicorn main:app --reload` and visit http://localhost:8000/docs to explore the API!

**🎯 Need real-time analysis?** Connect to `ws://localhost:8000/ws` and start streaming audio data!

**📊 Want to see it in action?** Check out the [Process Flow Documentation](PROCESS_FLOW_DOCUMENTATION.md) for detailed examples!
