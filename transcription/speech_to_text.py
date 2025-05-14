import numpy as np
from typing import Optional, List, Dict
import torch_test
import torchaudio
from faster_whisper import WhisperModel
import asyncio
from queue import Queue
import json
import gc
from difflib import SequenceMatcher

class SpeechToText:
    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_size: int = 1024,
        model_size: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16",
        overlap_ratio: float = 0.5  # 50% overlap between chunks
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.overlap_ratio = overlap_ratio
        
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
        self.last_transcription = ""
        self.is_processing = False
        
        # Set GPU memory management if using CUDA
        if device == "cuda":
            torch_test.cuda.set_per_process_memory_fraction(0.7)
            torch_test.backends.cudnn.benchmark = True
            torch_test.cuda.empty_cache()

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for the model"""
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio

    def _stitch_transcriptions(self, new_transcription: str) -> str:
        """Stitch overlapping transcriptions using text similarity"""
        if not self.last_transcription:
            return new_transcription
            
        # Find the best overlap point
        matcher = SequenceMatcher(None, self.last_transcription, new_transcription)
        match = matcher.find_longest_match(0, len(self.last_transcription), 0, len(new_transcription))
        
        if match.size > 10:  # Minimum overlap threshold
            # Stitch at the overlap point
            return self.last_transcription[:match.a] + new_transcription[match.b:]
        else:
            # No significant overlap, append with space
            return self.last_transcription + " " + new_transcription

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
                
                if transcription.strip():  # Only process non-empty transcriptions
                    # Stitch with previous transcription
                    stitched_transcription = self._stitch_transcriptions(transcription)
                    self.transcription_buffer.append(stitched_transcription)
                    self.last_transcription = stitched_transcription
                
                # Keep overlap for next chunk
                overlap_samples = int(len(self.audio_buffer) * self.overlap_ratio)
                self.audio_buffer = self.audio_buffer[-overlap_samples:]
                
                # Clear GPU memory
                if torch_test.cuda.is_available():
                    torch_test.cuda.empty_cache()
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
        self.last_transcription = ""
        self.is_processing = False
        if torch_test.cuda.is_available():
            torch_test.cuda.empty_cache()
            gc.collect()

    def __del__(self):
        """Cleanup when object is destroyed"""
        if torch_test.cuda.is_available():
            torch_test.cuda.empty_cache()
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