# audio_capture.py (modified for your pipeline)
import asyncio
import sounddevice as sd
import numpy as np
from typing import Optional
import queue
import logging

logger = logging.getLogger(__name__)

class AudioCapture:
    def __init__(self):
        self.is_capturing = False
        self.audio_queue = queue.Queue()
        self.stream = None
        self.sample_rate = 16000  # Default for speech processing
        self.channels = 1
        self.chunk_size = 2048  # Increased chunk size for stability
        self.device_index = None
        self.audio_buffer = np.array([], dtype=np.float32)

    async def start(self):
        """Start audio capture with async support"""
        if self.is_capturing:
            return

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                device=self.device_index,
                callback=self._audio_callback,
                dtype='float32'
            )
            self.stream.start()
            self.is_capturing = True
            logger.info("Audio capture started")
        except Exception as e:
            logger.error(f"Error starting audio capture: {str(e)}")
            raise

    def _audio_callback(self, indata, frames, time, status):
        """Sounddevice callback for audio input"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())

    def get_audio_chunk(self, min_samples: int = 16000) -> Optional[np.ndarray]:
        """
        Get audio chunk with minimum sample requirement
        Returns numpy array of shape (samples, channels)
        """
        if not self.is_capturing:
            return None

        # Buffer until we have enough samples
        while self.audio_queue.qsize() > 0:
            self.audio_buffer = np.concatenate(
                (self.audio_buffer, self.audio_queue.get().flatten())
            )

        if len(self.audio_buffer) < min_samples:
            return None

        # Extract chunk and maintain buffer
        chunk = self.audio_buffer[:min_samples]
        self.audio_buffer = self.audio_buffer[min_samples:]
        return chunk

    async def stop(self):
        """Stop audio capture"""
        if not self.is_capturing:
            return

        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
            self.is_capturing = False
            self.audio_buffer = np.array([], dtype=np.float32)
            logger.info("Audio capture stopped")
        except Exception as e:
            logger.error(f"Error stopping audio capture: {str(e)}")
            raise

    def list_devices(self):
        """List available audio devices"""
        return sd.query_devices()