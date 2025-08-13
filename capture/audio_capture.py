import asyncio
import sounddevice as sd
import numpy as np
from typing import Optional
import queue
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)

class AudioCapture:
    def __init__(   
        self,
        sample_rate: int = 16000,  
        channels: int = 1,
        chunk_size: int = 2048,  # Changed to power of 2 for better performance
        device_index: Optional[int] = None,
        buffer_size: int = 100,
        gain: float = 1.0,
        auto_gain: bool = False,
        target_rms: float = 0.05,
        buffer_duration: float = 2.0  # Duration of circular buffer in seconds
    ):
        self.is_capturing = False
        self.audio_queue = queue.Queue(maxsize=buffer_size)
        self.stream = None
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.gain = gain
        self.auto_gain = auto_gain
        self.target_rms = target_rms
        self.lock = threading.Lock()
        
        # Circular buffer implementation
        self.buffer_duration = buffer_duration
        self.buffer_max_samples = int(sample_rate * buffer_duration)
        self.circular_buffer = deque(maxlen=self.buffer_max_samples)
        self.last_processed_index = 0

        self._validate_device()

    def _validate_device(self):
        try:
            index = self.device_index or sd.default.device[0]
            device_info = sd.query_devices(index)

            if device_info['max_input_channels'] < self.channels:
                logger.warning(
                    f"Device only supports {device_info['max_input_channels']} channels, adjusting..."
                )
                self.channels = device_info['max_input_channels']

            if self.sample_rate != int(device_info['default_samplerate']):
                logger.warning(
                    f"Device default sample rate is {device_info['default_samplerate']}, adjusting..."
                )
                self.sample_rate = int(device_info['default_samplerate'])

            self.device_index = index
            logger.info(f"Using sample rate: {self.sample_rate} Hz")

        except Exception as e:
            logger.error(f"Error validating device: {e}")
            raise

    async def start(self):
        logger.info(
            f"Audio capture started: {self.sample_rate}Hz, {self.channels}ch, {self.chunk_size}samples"
        )

        if self.is_capturing:
            return

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                device=self.device_index,
                callback=self._audio_callback,
                dtype='float32',
                latency='low'
            )
            self.stream.start()
            self.is_capturing = True
            
            # Reset buffer state
            self.circular_buffer.clear()
            self.last_processed_index = 0
            
        except Exception as e:
            logger.error(f"Error starting audio capture: {str(e)}")
            raise

    def _apply_auto_gain(self, audio: np.ndarray) -> np.ndarray:
        """Apply simple RMS-based auto gain to match target_rms."""
        rms = np.sqrt(np.mean(audio**2)) + 1e-8
        factor = self.target_rms / rms
        audio = audio * factor
        return np.clip(audio, -1.0, 1.0)

    def _audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        try:
            # Mix-down to mono if needed
            audio_data = (
                np.mean(indata, axis=1)
                if self.channels > 1
                else indata.flatten()
            )

            # Apply gain or auto_gain
            if self.auto_gain:
                audio_data = self._apply_auto_gain(audio_data)
            elif self.gain != 1.0:
                audio_data = audio_data * self.gain
                audio_data = np.clip(audio_data, -1.0, 1.0)

            # Add to circular buffer
            with self.lock:
                self.circular_buffer.extend(audio_data)
                
            # Also add to queue for immediate processing if needed
            self.audio_queue.put(audio_data, timeout=0.05)
            
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    def get_audio_chunk(self, min_samples: int = 16000) -> Optional[np.ndarray]:
        """Get audio chunk from circular buffer with overlap handling"""
        if not self.is_capturing:
            return None

        with self.lock:
            current_buffer_size = len(self.circular_buffer)
            
            # Not enough samples available yet
            if current_buffer_size < min_samples:
                return None
                
            # Calculate how much new audio we have
            new_samples = current_buffer_size - self.last_processed_index
            
            # If we have enough new samples, return them
            if new_samples >= min_samples:
                chunk = np.array(self.circular_buffer)[self.last_processed_index:self.last_processed_index+min_samples]
                self.last_processed_index += min_samples
                return chunk
                
            # If we're at the end of the buffer, return from the end with overlap
            chunk = np.array(self.circular_buffer)[-min_samples:]
            self.last_processed_index = current_buffer_size
            return chunk

    def get_recent_audio(self, duration: float = 1.0) -> np.ndarray:
        """Get most recent audio of specified duration"""
        with self.lock:
            samples_needed = int(self.sample_rate * duration)
            if len(self.circular_buffer) < samples_needed:
                return np.array([])
            return np.array(self.circular_buffer)[-samples_needed:]

    async def stop(self):
        if not self.is_capturing:
            return

        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
            self.is_capturing = False
            
            with self.lock:
                self.circular_buffer.clear()
                self.last_processed_index = 0
                
            logger.info("Audio capture stopped")
        except Exception as e:
            logger.error(f"Error stopping audio capture: {str(e)}")
            raise

    def list_devices(self):
        return sd.query_devices()

    def get_device_info(self):
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "device_index": self.device_index,
            "is_capturing": self.is_capturing,
            "gain": self.gain,
            "auto_gain": self.auto_gain,
            "target_rms": self.target_rms,
            "buffer_duration": self.buffer_duration,
            "buffer_samples": len(self.circular_buffer)
        }