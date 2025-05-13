import numpy as np
from typing import List, Tuple, Optional
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
import librosa
import asyncio
from queue import Queue
import gc

class ECAPA_TDNN(nn.Module):
    def __init__(self, input_size=80, channels=512, emb_dim=192):
        super(ECAPA_TDNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, channels, kernel_size=5, dilation=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=2)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=3, dilation=3)
        self.conv4 = nn.Conv1d(channels*3, channels, kernel_size=1)
        self.conv5 = nn.Conv1d(channels, emb_dim, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Multi-scale feature fusion
        x = torch.cat([x, x.mean(dim=2, keepdim=True).expand_as(x)], dim=1)
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        
        # Global pooling
        x = torch.mean(x, dim=2)
        return F.normalize(x, p=2, dim=1)

class SpeakerDiarization:
    def __init__(
        self,
        sample_rate: int = 16000,
        min_speech_duration: float = 0.5,
        min_silence_duration: float = 0.5,
        threshold: float = 0.5,
        n_speakers: int = 2
    ):
        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.threshold = threshold
        self.n_speakers = n_speakers
        
        # Initialize ECAPA-TDNN model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ECAPA_TDNN().to(self.device)
        self.model.eval()
        
        # Initialize mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=80
        ).to(self.device)
        
        # Buffer for accumulating audio
        self.audio_buffer = []
        self.speaker_embeddings = []
        self.speaker_labels = []
        
        # Clustering model
        self.clustering = AgglomerativeClustering(
            n_clusters=n_speakers,
            metric='cosine',
            linkage='average'
        )
        
        # Set GPU memory management
        if self.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.7)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embeddings using ECAPA-TDNN"""
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        # Add batch dimension if needed
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(audio_tensor)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(mel_spec)
        
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
            if len(self.speaker_embeddings) >= self.n_speakers:
                self._update_speaker_labels()
                
            # Clear GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    def _update_speaker_labels(self):
        """Update speaker labels using clustering"""
        if len(self.speaker_embeddings) < self.n_speakers:
            return
        
        # Perform clustering
        embeddings = np.vstack(self.speaker_embeddings)
        labels = self.clustering.fit_predict(embeddings)
        self.speaker_labels = labels.tolist()

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
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

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