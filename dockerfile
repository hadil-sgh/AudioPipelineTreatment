# Multi-stage build with GPU and CPU options

##############################
# GPU Stage
##############################
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu
WORKDIR /app

# Install system dependencies for GPU
RUN apt-get update -y && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libcudnn8 \
    libcudnn8-dev \
    libcublas-12-1 \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    libpulse-dev \
    libasound2-plugins \
    ffmpeg \
    curl \
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Global pip config
RUN mkdir -p /etc/pip && echo "[global]\nretries = 10\ntimeout = 100\nprefer-binary = yes" > /etc/pip.conf

# Create app user and set correct ownership for /app
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app && \
    chown -R app:app /app

# Set up virtualenv
RUN python3 -m venv /home/app/venv
ENV PATH="/home/app/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA
RUN pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Copy files and install requirements
COPY --chown=app:app requirements*.txt /app/
COPY --chown=app:app start_enhanced_pipeline.py /app/
RUN pip install --no-cache-dir -r /app/requirements2.txt
RUN if [ -f /app/requirements.enhanced.txt ]; then \
    pip install --no-cache-dir -r /app/requirements.enhanced.txt; \
    fi

# Install ML tools
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    librosa \
    soundfile \
    webrtcvad \
    speechbrain \
    openai-whisper \
    pyannote.audio \
    scipy \
    scikit-learn

# Copy full source
COPY --chown=app:app . /app/

# Switch to app user
USER app

# Create runtime directories and log file
RUN mkdir -p /app/logs /app/temp /app/models /app/cache && touch /app/enhanced_pipeline.log

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_DEVICE=cuda
ENV PIPELINE_ENV=docker

HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python3", "main.py"]

##############################
# CPU Stage
##############################
FROM ubuntu:22.04 AS cpu
WORKDIR /app

# Install system dependencies for CPU
RUN apt-get update -y && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    libpulse-dev \
    libasound2-plugins \
    ffmpeg \
    curl \
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Global pip config
RUN mkdir -p /etc/pip && echo "[global]\nretries = 10\ntimeout = 100\nprefer-binary = yes" > /etc/pip.conf

# Create app user and set proper ownership
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app && \
    chown -R app:app /app

# Set up virtualenv
RUN python3 -m venv /home/app/venv
ENV PATH="/home/app/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch CPU
RUN pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Copy files and install requirements
COPY --chown=app:app requirements*.txt /app/
COPY --chown=app:app start_enhanced_pipeline.py /app/
RUN pip install --no-cache-dir -r /app/requirements2.txt
RUN if [ -f /app/requirements.enhanced.txt ]; then \
    pip install --no-cache-dir -r /app/requirements.enhanced.txt; \
    fi

RUN pip install --no-cache-dir nvidia-cudnn-cu12==8.9.2.26
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    librosa \
    soundfile \
    webrtcvad \
    speechbrain \
    openai-whisper \
    pyannote.audio \
    scipy \
    scikit-learn

# Copy all source
COPY --chown=app:app . /app/

# Switch to app user
USER app

# Create runtime directories
RUN mkdir -p /app/logs /app/temp /app/models /app/cache && touch /app/enhanced_pipeline.log

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_DEVICE=cpu
ENV PIPELINE_ENV=docker

HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python3", "main.py"]
