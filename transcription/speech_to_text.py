import numpy as np
from typing import Optional, List, Dict
import torch
import torchaudio
from faster_whisper import WhisperModel
import asyncio
from queue import Queue
import json
import gc

class SpeechToText:
    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_size: int = 1024,
        model_size: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Initialize Whisper model
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root="./models"
        )
        
        # Buffer for accumulating audio
        self.audio_buffer = []
        self.transcription_buffer = []
        self.is_processing = False
        
        # Set GPU memory management if using CUDA
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.7)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for the model"""
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio

    async def process_audio(self, audio_chunk: np.ndarray):
        """Process incoming audio chunk"""
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        
        # Process if buffer is large enough (2 seconds of audio)
        if len(self.audio_buffer) >= self.sample_rate * 2:
            if self.is_processing:
                return
            
            self.is_processing = True
            try:
                # Convert buffer to numpy array
                audio = np.array(self.audio_buffer)
                
                # Preprocess audio
                audio = self._preprocess_audio(audio)
                
                # Transcribe using Whisper
                segments, _ = self.model.transcribe(
                    audio,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Get the transcription
                transcription = " ".join([segment.text for segment in segments])
                
                if transcription.strip():  # Only add non-empty transcriptions
                    self.transcription_buffer.append(transcription)
                
                # Clear audio buffer
                self.audio_buffer = []
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                print(f"Error in transcription: {e}")
            finally:
                self.is_processing = False

    def get_latest_transcription(self) -> Optional[str]:
        """Get the latest transcription"""
        if not self.transcription_buffer:
            return None
        return self.transcription_buffer[-1]

    def get_all_transcriptions(self) -> List[str]:
        """Get all transcriptions"""
        return self.transcription_buffer.copy()

    def reset(self):
        """Reset the transcription state"""
        self.audio_buffer = []
        self.transcription_buffer = []
        self.is_processing = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def __del__(self):
        """Cleanup when object is destroyed"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

# Example usage
if __name__ == "__main__":
    async def test_transcription():
        # Create a test audio signal
        duration = 5  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Initialize transcription
        stt = SpeechToText()
        
        # Process audio in chunks
        chunk_size = 1024
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            await stt.process_audio(chunk)
            await asyncio.sleep(0.001)
        
        print("Transcriptions:", stt.get_all_transcriptions())

    asyncio.run(test_transcription()) 