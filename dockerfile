# Use Python 3.9 slim as base image for better compatibility with audio processing libraries
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing and ML libraries
RUN apt-get update && apt-get install -y \
    # Audio processing dependencies
    libsndfile1 \
    libsndfile1-dev \
    portaudio19-dev \
    libasound2-dev \
    # Build tools for Python packages
    build-essential \
    gcc \
    g++ \
    # Git for cloning repositories if needed
    git \
    # FFmpeg for audio/video processing
    ffmpeg \
    # Additional audio libraries
    libpulse-dev \
    libasound2-plugins \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Add user's local bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# First install PyTorch with CPU support (platform independent)
RUN pip install --user --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file (excluding torch dependencies)
COPY --chown=app:app requirements2.txt .

# Install other Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Install additional audio processing packages
RUN pip install --user --no-cache-dir \
    transformers \
    datasets \
    librosa \
    soundfile \
    pyaudio \
    webrtcvad \
    speechbrain \
    openai-whisper

# Copy the application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p logs temp models

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Set the default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]