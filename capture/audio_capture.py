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
        chunk_size: int = 2084,  
        device_index: Optional[int] = None,
        buffer_size: int = 100,
        gain: float = 1.0,
        auto_gain: bool = False,
        target_rms: float = 0.05
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
        self.gain = gain
        self.auto_gain = auto_gain
        self.target_rms = target_rms

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
            print(f"[AudioCapture] Using sample rate: {self.sample_rate} Hz")
            logger.info(f"Using sample rate: {self.sample_rate} Hz")

        except Exception as e:
            logger.error(f"Error validating device: {e}")
            raise

    async def start(self):
        logger.info(
            f"Audio capture started: {self.sample_rate}Hz, {self.channels}ch, {self.chunk_size}samples"
        )
        print(f"[AudioCapture] Started with sample rate: {self.sample_rate} Hz")

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
            # mix-down to mono if needed
            audio_data = (
                np.mean(indata, axis=1)
                if self.channels > 1
                else indata.flatten()
            )

            # apply gain or auto_gain
            if self.auto_gain:
                audio_data = self._apply_auto_gain(audio_data)
            elif self.gain != 1.0:
                audio_data = audio_data * self.gain
                audio_data = np.clip(audio_data, -1.0, 1.0)

            self.audio_queue.put(audio_data, timeout=0.05)
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

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
            "is_capturing": self.is_capturing,
            "gain": self.gain,
            "auto_gain": self.auto_gain,
            "target_rms": self.target_rms
        }