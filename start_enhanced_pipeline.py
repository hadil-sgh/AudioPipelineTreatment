#!/usr/bin/env python3
"""
Enhanced Real-Time Audio Pipeline Startup Script
Runs the enhanced pipeline with optimal settings for frontend integration
"""

import asyncio
import logging
import uvicorn
import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging():
    """Setup enhanced logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific log levels for different components
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('websockets').setLevel(logging.INFO)
    logging.getLogger('enhanced_web_realtime_pipeline').setLevel(logging.DEBUG)

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'torch', 'torchaudio', 'numpy', 'fastapi', 'uvicorn',
        'speechbrain', 'librosa', 'transformers', 'scipy'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    print("✓ All required dependencies are available")
    return True

def check_gpu_availability():
    """Check GPU availability for enhanced performance"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {device_name} ({device_count} device(s))")
            return True
        else:
            print("⚠️  GPU not available, using CPU (performance may be slower)")
            return False
    except Exception as e:
        print(f"⚠️  Error checking GPU: {e}")
        return False

def display_startup_info():
    """Display startup information"""
    print("=" * 80)
    print("🎤 ENHANCED REAL-TIME AUDIO PIPELINE")
    print("=" * 80)
    print("🔗 Frontend Integration: Optimized for Angular WebSocket connection")
    print("🎯 Audio Processing: RealTimePipelineMic.py components")
    print("📊 Features: Sentiment analysis, speaker diarization, emotion detection")
    print("💾 Recording: Automatic audio quality monitoring and recording")
    print("=" * 80)

def main():
    """Main startup function"""
    display_startup_info()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check GPU
    print("\n🔍 Checking GPU availability...")
    gpu_available = check_gpu_availability()
    
    # Set optimal configuration based on available hardware
    if gpu_available:
        os.environ['TORCH_DEVICE'] = 'cuda'
        print("🚀 Using GPU acceleration for optimal performance")
    else:
        os.environ['TORCH_DEVICE'] = 'cpu'
        print("🐌 Using CPU (consider GPU for better performance)")
    
    print("\n🌐 Starting Enhanced Audio Pipeline Server...")
    print("📡 WebSocket endpoint: ws://localhost:8000/ws")
    print("🔗 Frontend connection: Angular app should connect to this endpoint")
    print("📋 Health check: http://localhost:8000/health")
    print("📊 API status: http://localhost:8000/api/status")
    
    print("\n" + "=" * 80)
    print("🚀 SERVER STARTING - Ready for frontend connections!")
    print("=" * 80)
    
    try:
        # Import and run the FastAPI app
        from main import app
        
        # Run with optimal settings for real-time audio
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            # Optimizations for real-time audio processing
            ws_ping_interval=30,
            ws_ping_timeout=10,
            ws_max_size=16 * 1024 * 1024,  # 16MB max WebSocket message size
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        logger.info("Server shutdown requested")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        logger.exception("Server startup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
