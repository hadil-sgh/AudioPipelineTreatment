import numpy as np
from typing import Dict
import torch_test
import torchaudio
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoModelForAudioClassification
)
import librosa
from scipy.signal import find_peaks
import gc

class SentimentAnalyzer:
    def __init__(
        self,
        text_model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        voice_model_name: str = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
    ):
        self.text_model_name = text_model_name
        self.voice_model_name = voice_model_name

        self.text_tokenizer = None
        self.text_model = None
        self.voice_processor = None
        self.voice_model = None

        self.device = "cuda" if torch_test.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch_test.cuda.set_per_process_memory_fraction(0.7)
            torch_test.backends.cudnn.benchmark = True
            torch_test.cuda.empty_cache()

        self.sample_rate = 16000
        self.hop_length = 512
        self.n_mels = 128

    def load_text_model(self):
        if self.text_tokenizer is None or self.text_model is None:
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(self.text_model_name).to(self.device)

    def load_voice_model(self):
        if self.voice_processor is None or self.voice_model is None:
            self.voice_processor = AutoFeatureExtractor.from_pretrained(self.voice_model_name)
            self.voice_model = AutoModelForAudioClassification.from_pretrained(self.voice_model_name).to(self.device)

    def analyze_text(self, text: str) -> Dict:
        self.load_text_model()
        try:
            inputs = self.text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch_test.no_grad():
                outputs = self.text_model(**inputs)
                scores = torch_test.softmax(outputs.logits, dim=1)

            emotion_idx = torch_test.argmax(scores).item()
            emotion = self.text_model.config.id2label[emotion_idx]
            score = scores[0][emotion_idx].item()

            if self.device == "cuda":
                del inputs, outputs, scores
                torch_test.cuda.empty_cache()
                gc.collect()

            return {"emotion": emotion, "score": score}
        except Exception as e:
            print(f"Error in text analysis: {e}")
            return {"emotion": "UNKNOWN", "score": 0.0}

    def analyze_voice(self, audio: np.ndarray) -> Dict:
        self.load_voice_model()
        try:
            audio_tensor = torch_test.from_numpy(audio).float()
            inputs = self.voice_processor(audio_tensor, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)

            with torch_test.no_grad():
                outputs = self.voice_model(**inputs)
                scores = torch_test.softmax(outputs.logits, dim=1)

            emotion_idx = torch_test.argmax(scores).item()
            emotion = self.voice_model.config.id2label[emotion_idx]
            score = scores[0][emotion_idx].item()

            features = self._extract_voice_features(audio)

            if self.device == "cuda":
                del inputs, outputs, scores
                torch_test.cuda.empty_cache()
                gc.collect()

            return {"emotion": emotion, "score": score, "features": features}
        except Exception as e:
            print(f"Error in voice analysis: {e}")
            return {
                "emotion": "UNKNOWN",
                "score": 0.0,
                "features": {"pitch": 0.0, "energy": 0.0, "speaking_rate": 0.0}
            }

    def _extract_voice_features(self, audio: np.ndarray) -> Dict:
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
            pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])
            energy = np.mean(librosa.feature.rms(y=audio)[0])
            peaks, _ = find_peaks(np.abs(audio), height=np.std(audio))
            speaking_rate = len(peaks) / (len(audio) / self.sample_rate)

            return {"pitch": float(pitch), "energy": float(energy), "speaking_rate": float(speaking_rate)}
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return {"pitch": 0.0, "energy": 0.0, "speaking_rate": 0.0}

    def analyze(self, text: str, audio: np.ndarray) -> Dict:
        return {
            "text": self.analyze_text(text),
            "voice": self.analyze_voice(audio)
        }

    def __del__(self):
        if self.device == "cuda":
            torch_test.cuda.empty_cache()
            gc.collect()
