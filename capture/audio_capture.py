# audio_capture.py 
import asyncio
import sounddevice as sd
import numpy as np
from typing import Optional
import queue
import logging
import threading

logger = logging.getLogger(__name__)

class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
        buffer_size: int = 50
    ):
        self.is_capturing = False
        self.audio_queue = queue.Queue(maxsize=buffer_size)
        self.stream = None
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.audio_buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
        
        # Validate device settings
        self._validate_device()

    def _validate_device(self):
        """Validate and adjust device settings"""
        try:
            # If device_index is None, use default input device
            index = self.device_index or sd.default.device[0]
            device_info = sd.query_devices(index)

            if device_info['max_input_channels'] < self.channels:
                logger.warning(f"Device only supports {device_info['max_input_channels']} channels, adjusting...")
                self.channels = device_info['max_input_channels']

            if self.sample_rate != int(device_info['default_samplerate']):
                logger.warning(f"Device default sample rate is {device_info['default_samplerate']}, adjusting...")
                self.sample_rate = int(device_info['default_samplerate'])

            # Store updated device index in case we used default
            self.device_index = index

        except Exception as e:
            logger.error(f"Error validating device: {e}")
            raise

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
                dtype='float32',
                latency='low'
            )
            self.stream.start()
            self.is_capturing = True
            logger.info(f"Audio capture started: {self.sample_rate}Hz, {self.channels}ch, {self.chunk_size}samples")
        except Exception as e:
            logger.error(f"Error starting audio capture: {str(e)}")
            raise

    def _audio_callback(self, indata, frames, time, status):
        """Sounddevice callback for audio input"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            # Convert to mono if needed
            if self.channels > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()

            # Apply preprocessing
            audio_data = self._preprocess_audio(audio_data)

            # Add to queue with timeout
            self.audio_queue.put(audio_data, timeout=0.1)
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Basic audio preprocessing"""
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        audio = audio - np.mean(audio)
        return audio

    def get_audio_chunk(self, min_samples: int = 16000) -> Optional[np.ndarray]:
        """Get audio chunk with minimum sample requirement"""
        if not self.is_capturing:
            return None

        with self.lock:
            while self.audio_queue.qsize() > 0:
                try:
                    chunk = self.audio_queue.get_nowait()
                    self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
                except queue.Empty:
                    break

            if len(self.audio_buffer) < min_samples:
                return None

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

    def get_device_info(self):
        """Get current device configuration"""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "device_index": self.device_index,
            "is_capturing": self.is_capturing
        }
