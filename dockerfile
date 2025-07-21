# Multi-stage build with GPU and CPU options
# Build with: docker build --target gpu . (for GPU) or docker build --target cpu . (for CPU)

# GPU Stage
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as gpu
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
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Install PyTorch with CUDA support
RUN pip3 install --user --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install requirements
COPY --chown=app:app requirements2.txt /app/requirements2.txt
RUN pip3 install --user --no-cache-dir -r /app/requirements2.txt

# Install additional ML packages
RUN pip3 install --user --no-cache-dir \
    transformers \
    datasets \
    librosa \
    soundfile \
    webrtcvad \
    speechbrain \
    openai-whisper

# Copy application code
COPY --chown=app:app . /app/

# Create necessary directories
RUN mkdir -p logs temp models

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/app/.local/bin:${PATH}"
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# --------------------------------------------
# CPU Stage  
FROM ubuntu:22.04 as cpu
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
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Install PyTorch CPU version
RUN pip3 install --user --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements
COPY --chown=app:app requirements2.txt /app/requirements2.txt
RUN pip3 install --user --no-cache-dir -r /app/requirements2.txt

# Install additional ML packages
RUN pip3 install --user --no-cache-dir \
    transformers \
    datasets \
    librosa \
    soundfile \
    webrtcvad \
    speechbrain \
    openai-whisper

# Copy application code
COPY --chown=app:app . /app/

# Create necessary directories
RUN mkdir -p logs temp models

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/app/.local/bin:${PATH}"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]