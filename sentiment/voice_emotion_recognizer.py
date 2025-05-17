import numpy as np
from typing import Dict
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification
)
import librosa
from scipy.signal import find_peaks
import gc

class voice_emotion_recognizer:
    def __init__(
        self,
        voice_model_name: str = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
    ):
        self.voice_model_name = voice_model_name

       
        self.voice_processor = None
        self.voice_model = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.7)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

        self.sample_rate = 16000
        self.hop_length = 512
        self.n_mels = 128


    def load_voice_model(self):
        if self.voice_processor is None or self.voice_model is None:
            self.voice_processor = AutoFeatureExtractor.from_pretrained(self.voice_model_name)
            self.voice_model = AutoModelForAudioClassification.from_pretrained(self.voice_model_name).to(self.device)


    def analyze_voice(self, audio: np.ndarray) -> Dict:
        self.load_voice_model()
        try:
            audio_tensor = torch.from_numpy(audio).float()
            inputs = self.voice_processor(audio_tensor, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.voice_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)

            emotion_idx = torch.argmax(scores).item()
            emotion = self.voice_model.config.id2label[emotion_idx]
            score = scores[0][emotion_idx].item()

            features = self._extract_features(audio)

            if self.device == "cuda":
                del inputs, outputs, scores
                torch.cuda.empty_cache()
                gc.collect()

            return {"emotion": emotion, "score": score, "features": features}
        except Exception as e:
            print(f"Error in voice analysis: {e}")
            return {
                "emotion": "UNKNOWN",
                "score": 0.0,
                "features": {"pitch": 0.0, "energy": 0.0, "speaking_rate": 0.0}
            }

    def _extract_features(self, audio: np.ndarray) -> Dict:
        """Extract voice features from audio."""
        try:
            # Extract pitch using librosa's piptrack
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
            pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])
            
            # Calculate energy (RMS)
            energy = np.mean(librosa.feature.rms(y=audio)[0])
            
            # Calculate speaking rate
            peaks, _ = find_peaks(np.abs(audio), height=np.std(audio))
            speaking_rate = len(peaks) / (len(audio) / self.sample_rate)
            
            return {
                "pitch": float(pitch),
                "energy": float(energy),
                "speaking_rate": float(speaking_rate)
            }
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return {
                "pitch": 0.0,
                "energy": 0.0,
                "speaking_rate": 0.0
            }

    def _classify_emotion(self, features: Dict) -> Dict:
        """Classify emotion based on voice features."""
        # Simple rule-based classification
        pitch = features["pitch"]
        energy = features["energy"]
        rate = features["speaking_rate"]
        
        # Default to NATURAL
        emotion = "NATURAL"
        score = 0.5
        
        # High pitch and energy might indicate excitement
        if pitch > 200 and energy > 0.1:
            emotion = "EXCITED"
            score = 0.8
        # Low pitch and energy might indicate sadness
        elif pitch < 100 and energy < 0.05:
            emotion = "SAD"
            score = 0.7
        # High speaking rate might indicate anxiety
        elif rate > 0.15:
            emotion = "ANXIOUS"
            score = 0.6
            
        return {
            "emotion": emotion,
            "score": score
        }

    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze voice emotion from audio."""
        features = self._extract_features(audio)
        emotion_result = self._classify_emotion(features)
        
        return {
            "voice": {
                "emotion": emotion_result["emotion"],
                "score": emotion_result["score"],
                "features": features
            }
        }

    def __del__(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
