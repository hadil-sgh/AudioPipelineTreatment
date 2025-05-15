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
        chunk_size: int = 2048,  
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
        try:
            index = self.device_index or sd.default.device[0]
            device_info = sd.query_devices(index)

            if device_info['max_input_channels'] < self.channels:
                logger.warning(f"Device only supports {device_info['max_input_channels']} channels, adjusting...")
                self.channels = device_info['max_input_channels']

            if self.sample_rate != int(device_info['default_samplerate']):
                logger.warning(f"Device default sample rate is {device_info['default_samplerate']}, adjusting...")
                self.sample_rate = int(device_info['default_samplerate'])

            self.device_index = index
        except Exception as e:
            logger.error(f"Error validating device: {e}")
            raise

    async def start(self):
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
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            audio_data = np.mean(indata, axis=1) if self.channels > 1 else indata.flatten()
            audio_data = self._preprocess_audio(audio_data)
            self.audio_queue.put(audio_data, timeout=0.05)
        except queue.Full:
            logger.warning(" Audio queue full, dropping frame")
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio - np.mean(audio)

    def get_audio_chunk(self, min_samples: int = 16000) -> Optional[np.ndarray]:
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
        return sd.query_devices()

    def get_device_info(self):
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "device_index": self.device_index,
            "is_capturing": self.is_capturing
        }
