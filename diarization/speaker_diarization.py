import numpy as np
from typing import List, Tuple, Optional
import torch
from sklearn.cluster import AgglomerativeClustering
import librosa
import math

from speechbrain.inference.speaker import EncoderClassifier


class SpeakerDiarization:
    def __init__(
        self,
        min_speech_duration: float = 1.0, 
        threshold: float = 0.03,         
        n_speakers: int = 2,
        embedding_model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[str] = None,
        process_buffer_duration: float = 5.0
    ):
        self.sample_rate = 16000
        self.min_speech_duration = min_speech_duration
        self.threshold = threshold
        self.n_speakers = n_speakers
        self.process_buffer_duration = process_buffer_duration

        self.device = device or "cuda"

        try:
            self.embedding_model = EncoderClassifier.from_hparams(
                source=embedding_model_source,
                run_opts={"device": self.device}
            )
            self.embedding_model.eval()
        except Exception as e:
            raise

        self.vad_frame_length = 2048
        self.vad_hop_length = 512
        self.audio_buffer = []
        self.speaker_embeddings = []
        self.speaker_labels = []

        self.clustering = AgglomerativeClustering(
            n_clusters=self.n_speakers if self.n_speakers > 0 else None,
            distance_threshold=None if self.n_speakers > 0 else 0.6,
            metric='cosine',
            linkage='average'
        )

        if self.device == "cuda":
           torch.backends.cudnn.benchmark = True

    def _extract_features(self, audio_segment_np: np.ndarray) -> Optional[np.ndarray]:
        if audio_segment_np.ndim > 1:
            audio_segment_np = np.mean(audio_segment_np, axis=0)
        audio_tensor = torch.from_numpy(audio_segment_np).float().to(self.device)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        min_len_for_sb_model = 1600
        if audio_tensor.shape[1] < min_len_for_sb_model:
            padding_needed = min_len_for_sb_model - audio_tensor.shape[1]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding_needed), mode='reflect')

        with torch.no_grad():
            try:
                embedding_tensor = self.embedding_model.encode_batch(audio_tensor)
                if embedding_tensor.ndim == 3 and embedding_tensor.shape[1] == 1:
                    embedding_tensor = embedding_tensor.squeeze(1)
                if embedding_tensor.ndim == 1:
                    embedding_tensor = embedding_tensor.unsqueeze(0)
                return embedding_tensor.cpu().numpy()
            except Exception:
                return None

    def _detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        energy = librosa.feature.rms(y=audio, frame_length=self.vad_frame_length, hop_length=self.vad_hop_length)[0]
        if not energy.size:
            return []

        min_speech_samples = self.min_speech_duration * self.sample_rate
        min_speech_frames_vad = math.ceil(min_speech_samples / self.vad_hop_length)
        if min_speech_frames_vad < 1:
            min_speech_frames_vad = 1

        speech_segments_found_by_vad = []
        is_speech = False
        start_frame = 0
        for i, e_val in enumerate(energy):
            if not is_speech and e_val > self.threshold:
                is_speech = True
                start_frame = i
            elif is_speech and e_val <= self.threshold:
                is_speech = False
                segment_duration_frames = i - start_frame
                if segment_duration_frames >= min_speech_frames_vad:
                    speech_segments_found_by_vad.append((start_frame, i))

        if is_speech:
            end_frame = len(energy)
            segment_duration_frames = end_frame - start_frame
            if segment_duration_frames >= min_speech_frames_vad:
                speech_segments_found_by_vad.append((start_frame, end_frame))

        return speech_segments_found_by_vad

    async def process_audio_buffer(self):
        if not self.audio_buffer:
            return

        audio_combined = np.array(self.audio_buffer, dtype=np.float32)
        self.audio_buffer = []

        segments_frame_indices = self._detect_speech_segments(audio_combined)

        new_embeddings_added = False
        for start_frame, end_frame in segments_frame_indices:
            start_sample = int(start_frame * self.vad_hop_length)
            end_sample = int(end_frame * self.vad_hop_length)
            end_sample = min(end_sample, len(audio_combined))
            start_sample = min(start_sample, end_sample)
            segment_audio = audio_combined[start_sample:end_sample]

            if len(segment_audio) == 0:
                continue

            embedding = self._extract_features(segment_audio)
            if embedding is not None:
                self.speaker_embeddings.append(embedding)
                new_embeddings_added = True

        if new_embeddings_added:
            self._update_speaker_labels()

    async def process_audio(self, audio_chunk: np.ndarray):
        self.audio_buffer.extend(audio_chunk)
        min_buffer_len = int(self.sample_rate * self.process_buffer_duration)

        if len(self.audio_buffer) >= min_buffer_len:
            await self.process_audio_buffer()

    async def finalize_processing(self):
        await self.process_audio_buffer()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _update_speaker_labels(self):
        if not self.speaker_embeddings:
            return
        if self.n_speakers > 0 and len(self.speaker_embeddings) < self.n_speakers:
            return

        embeddings_matrix = np.vstack(self.speaker_embeddings)
        if embeddings_matrix.shape[0] < 2:
            return
        if self.clustering.n_clusters is not None and embeddings_matrix.shape[0] < self.clustering.n_clusters:
            return

        try:
            labels = self.clustering.fit_predict(embeddings_matrix)
            self.speaker_labels = labels.tolist()
        except ValueError:
            pass

    def get_current_speaker(self) -> Optional[int]:
        if not self.speaker_labels:
            return None
        return self.speaker_labels[-1]

    def reset(self):
        self.audio_buffer = []
        self.speaker_embeddings = []
        self.speaker_labels = []
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def __del__(self):
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
