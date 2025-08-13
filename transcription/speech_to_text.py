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
        min_audio_length: float = 3.0  # Minimum audio length in seconds to process
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.min_audio_length = min_audio_length
        
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

    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Simple but effective audio preprocessing"""
        try:
            # Ensure float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Check for actual audio content
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 0.001:  # Very quiet audio
                return np.zeros_like(audio_data)  # Return silence
            
            # Simple normalization
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                # Don't over-normalize quiet speech
                if max_val < 0.1:
                    audio_data = audio_data * (0.1 / max_val)
                else:
                    audio_data = audio_data * (0.8 / max_val)
            
            return audio_data
        except:
            return audio_data

    def _filter_repetitions(self, text: str) -> str:
        """Remove obvious repetitions that Whisper sometimes generates"""
        if not text or len(text) < 10:
            return text
        
        words = text.split()
        if len(words) < 3:
            return text
        
        # Check for word repeated many times
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        # If any single word appears more than 40% of the time, it's likely a repetition error
        total_words = len(words)
        for word, count in word_counts.items():
            if count > max(3, total_words * 0.4):  # More than 40% or more than 3 times
                # This is likely a Whisper hallucination
                return ""  # Return empty to skip this transcription
        
        # Check for phrase repetitions (like "is there is there is there")
        if len(words) > 6:
            # Look for 2-word phrases repeated
            for i in range(len(words) - 3):
                phrase = f"{words[i]} {words[i+1]}"
                phrase_count = 0
                for j in range(i, len(words) - 1):
                    if j + 1 < len(words) and f"{words[j]} {words[j+1]}" == phrase:
                        phrase_count += 1
                
                if phrase_count > 3:  # Same 2-word phrase appears 4+ times
                    return ""  # Skip this transcription
        
        return text

    async def process_audio(self, audio_chunk: np.ndarray):
        """Process incoming audio chunk"""
        self.audio_buffer.extend(audio_chunk)
        
        # Only process when we have enough audio
        if len(self.audio_buffer) >= self.sample_rate * self.min_audio_length:
            if self.is_processing:
                return
            
            self.is_processing = True
            try:
                audio = np.array(self.audio_buffer)
                audio = self._preprocess_audio(audio)
                
                # Check if audio has meaningful content
                if np.sqrt(np.mean(audio**2)) < 0.005:  # Too quiet
                    self.audio_buffer = []  # Clear buffer
                    return
                
                # Transcribe with conservative settings
                segments, _ = self.model.transcribe(
                    audio,
                    beam_size=1,  # Use beam_size=1 for less hallucination
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=1000,  # Longer silence detection
                        min_speech_duration_ms=500     # Minimum speech duration
                    ),
                    language="en",
                    condition_on_previous_text=False,  # DON'T condition on previous text
                    no_speech_threshold=0.6,  # Higher threshold to avoid hallucinations
                    temperature=0.0  # Deterministic output
                )
                
                transcription = " ".join([segment.text for segment in segments]).strip()
                
                # Apply repetition filter
                transcription = self._filter_repetitions(transcription)
                
                if transcription and len(transcription) > 5:
                    # Simple deduplication: only add if it's different from last result
                    if transcription != self.last_transcription:
                        self.transcription_buffer.append(transcription)
                        self.last_transcription = transcription
                
                # Clear buffer after processing
                self.audio_buffer = []
                
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