import numpy as np
from typing import Optional, List
import torch
import gc
from faster_whisper import WhisperModel
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
        self.last_returned_index = 0  # Track how much text was already returned
        self.is_processing = False
        
        # Set GPU memory management if using CUDA
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.7)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for the model"""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        return audio

    def _stitch_transcriptions(self, new_transcription: str) -> str:
        """Stitch overlapping transcriptions using text similarity"""
        if not self.last_transcription:
            return new_transcription
            
        matcher = SequenceMatcher(None, self.last_transcription, new_transcription)
        match = matcher.find_longest_match(0, len(self.last_transcription), 0, len(new_transcription))
        
        if match.size > 10:  # Minimum overlap threshold
            return self.last_transcription[:match.a] + new_transcription[match.b:]
        else:
            return self.last_transcription + " " + new_transcription

    async def process_audio(self, audio_chunk: np.ndarray):
        """Process incoming audio chunk"""
        self.audio_buffer.extend(audio_chunk)
        
        if len(self.audio_buffer) >= self.sample_rate * 2:
            if self.is_processing:
                return
            
            self.is_processing = True
            try:
                audio = np.array(self.audio_buffer)
                audio = self._preprocess_audio(audio)
                
                segments, _ = self.model.transcribe(
                    audio,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                transcription = " ".join([segment.text for segment in segments]).strip()
                
                if transcription:
                    stitched = self._stitch_transcriptions(transcription)
                    self.transcription_buffer.append(stitched)
                    self.last_transcription = stitched
                
                overlap_samples = int(len(self.audio_buffer) * self.overlap_ratio)
                self.audio_buffer = self.audio_buffer[-overlap_samples:]
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            except Exception as e:
                print(f"Error in transcription: {e}")
            finally:
                self.is_processing = False

    def get_latest_transcription(self) -> Optional[str]:
        """Return only the NEW part of the transcription since last call"""
        if not self.transcription_buffer:
            return None
        
        full_text = self.transcription_buffer[-1]
        if len(full_text) > self.last_returned_index:
            new_text = full_text[self.last_returned_index:].strip()
            self.last_returned_index = len(full_text)
            return new_text if new_text else None
        return None

    def get_all_transcriptions(self) -> List[str]:
        """Get all transcriptions"""
        return self.transcription_buffer.copy()

    def reset(self):
        """Reset transcription state"""
        self.audio_buffer = []
        self.transcription_buffer = []
        self.last_transcription = ""
        self.last_returned_index = 0
        self.is_processing = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def __del__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
