import numpy as np
from typing import List, Tuple, Optional
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.cluster import AgglomerativeClustering
import librosa
import asyncio
from queue import Queue

class SpeakerDiarization:
    def __init__(
        self,
        sample_rate: int = 16000,
        min_speech_duration: float = 0.5,
        min_silence_duration: float = 0.5,
        threshold: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.threshold = threshold
        
        # Initialize Wav2Vec2 model for speaker embeddings
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        # Buffer for accumulating audio
        self.audio_buffer = []
        self.speaker_embeddings = []
        self.speaker_labels = []
        
        # Clustering model
        self.clustering = AgglomerativeClustering(
            n_clusters=2,
            metric='cosine',
            linkage='average'
        )

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embeddings using Wav2Vec2"""
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Add batch dimension if needed
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Move to device
        audio_tensor = audio_tensor.to(self.device)
        
        # Extract features
        inputs = self.processor(
            audio_tensor,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        
        return embeddings.cpu().numpy()

    def _detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Detect speech segments using energy-based VAD"""
        # Calculate energy
        energy = librosa.feature.rms(y=audio)[0]
        
        # Find speech segments
        speech_segments = []
        is_speech = False
        start_idx = 0
        
        for i, e in enumerate(energy):
            if not is_speech and e > self.threshold:
                is_speech = True
                start_idx = i
            elif is_speech and e <= self.threshold:
                is_speech = False
                if i - start_idx >= self.min_speech_duration * self.sample_rate:
                    speech_segments.append((start_idx, i))
        
        return speech_segments

    async def process_audio(self, audio_chunk: np.ndarray):
        """Process incoming audio chunk"""
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        
        # Process if buffer is large enough
        if len(self.audio_buffer) >= self.sample_rate * 2:  # Process every 2 seconds
            audio = np.array(self.audio_buffer)
            segments = self._detect_speech_segments(audio)
            
            for start, end in segments:
                segment = audio[start:end]
                embedding = self._extract_features(segment)
                self.speaker_embeddings.append(embedding)
            
            # Clear buffer
            self.audio_buffer = []
            
            # Update speaker labels if we have enough embeddings
            if len(self.speaker_embeddings) >= 2:
                self._update_speaker_labels()

    def _update_speaker_labels(self):
        """Update speaker labels using clustering"""
        if len(self.speaker_embeddings) < 2:
            return
        
        # Perform clustering
        labels = self.clustering.fit_predict(self.speaker_embeddings)
        self.speaker_labels = labels

    def get_current_speaker(self) -> Optional[int]:
        """Get the current speaker label"""
        if not self.speaker_labels:
            return None
        return self.speaker_labels[-1]

    def reset(self):
        """Reset the diarization state"""
        self.audio_buffer = []
        self.speaker_embeddings = []
        self.speaker_labels = []

# Example usage
if __name__ == "__main__":
    async def test_diarization():
        # Create a test audio signal
        duration = 5  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Initialize diarization
        diarization = SpeakerDiarization()
        
        # Process audio in chunks
        chunk_size = 1024
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            await diarization.process_audio(chunk)
            await asyncio.sleep(0.001)
        
        print("Speaker labels:", diarization.speaker_labels)

    asyncio.run(test_diarization()) 