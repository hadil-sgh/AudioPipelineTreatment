import numpy as np
from typing import List, Optional, Tuple
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
import librosa
import asyncio
import math
import matplotlib.pyplot as plt
from speechbrain.inference.speaker import EncoderClassifier
import soundfile as sf
from sklearn.manifold import TSNE  # For embedding visualization

class SpeakerDiarization:
    def __init__(
        self,
        min_speech_duration: float = 0.75,
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

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for diarization.")
        print(f"Loading embedding model from: {embedding_model_source}")

        self.embedding_model = EncoderClassifier.from_hparams(
            source=embedding_model_source,
            run_opts={"device": self.device}
        )
        self.embedding_model.eval()

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

        min_len = 1600
        if audio_tensor.shape[1] < min_len:
            padding = min_len - audio_tensor.shape[1]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding), mode='reflect')

        with torch.no_grad():
            try:
                embedding_tensor = self.embedding_model.encode_batch(audio_tensor)
                if embedding_tensor.ndim == 3 and embedding_tensor.shape[1] == 1:
                    embedding_tensor = embedding_tensor.squeeze(1)
                if embedding_tensor.ndim == 1:
                    embedding_tensor = embedding_tensor.unsqueeze(0)
                return embedding_tensor.cpu().numpy()
            except Exception as e:
                print(f"Error extracting embeddings: {e}")
                return None

    def _detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        energy = librosa.feature.rms(y=audio, frame_length=self.vad_frame_length, hop_length=self.vad_hop_length)[0]
        if energy.size == 0:
            return []

        min_speech_frames = math.ceil((self.min_speech_duration * self.sample_rate) / self.vad_hop_length)
        speech_segments = []
        is_speech = False
        start_frame = 0

        for i, val in enumerate(energy):
            if not is_speech and val > self.threshold:
                is_speech = True
                start_frame = i
            elif is_speech and val <= self.threshold:
                if (i - start_frame) >= min_speech_frames:
                    speech_segments.append((start_frame, i))
                is_speech = False
        if is_speech and (len(energy) - start_frame) >= min_speech_frames:
            speech_segments.append((start_frame, len(energy)))

        return speech_segments

    async def process_audio_buffer(self):
        if not self.audio_buffer:
            return
        audio_np = np.array(self.audio_buffer, dtype=np.float32)
        self.audio_buffer = []

        segments = self._detect_speech_segments(audio_np)
        for start_f, end_f in segments:
            start_sample = int(start_f * self.vad_hop_length)
            end_sample = int(end_f * self.vad_hop_length)
            segment_audio = audio_np[start_sample:end_sample]
            if len(segment_audio) == 0:
                continue
            embedding = self._extract_features(segment_audio)
            if embedding is not None:
                self.speaker_embeddings.append(embedding)

        self._update_speaker_labels()

    async def process_audio(self, audio_chunk: np.ndarray):
        self.audio_buffer.extend(audio_chunk)
        if len(self.audio_buffer) >= int(self.sample_rate * self.process_buffer_duration):
            await self.process_audio_buffer()

    async def finalize_processing(self):
        await self.process_audio_buffer()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _update_speaker_labels(self):
        if not self.speaker_embeddings:
            return
        embeddings_matrix = np.vstack(self.speaker_embeddings)
        if self.n_speakers > 0 and len(self.speaker_embeddings) < self.n_speakers:
            return
        if embeddings_matrix.shape[0] < 2:
            return
        try:
            labels = self.clustering.fit_predict(embeddings_matrix)
            self.speaker_labels = labels.tolist()
        except Exception as e:
            print(f"Clustering failed: {e}")

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
        print("Diarization reset.")

def visualize_diarization(audio: np.ndarray, sample_rate: int, diarization: SpeakerDiarization):
    energy = librosa.feature.rms(y=audio, frame_length=diarization.vad_frame_length, hop_length=diarization.vad_hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sample_rate, hop_length=diarization.vad_hop_length)

    plt.figure(figsize=(14, 4))
    plt.plot(times, energy, label='RMS Energy', alpha=0.5)
    plt.axhline(diarization.threshold, color='r', linestyle='--', label='VAD Threshold')

    if diarization.speaker_labels:
        segment_length = len(audio) / sample_rate / len(diarization.speaker_labels)
        for i, label in enumerate(diarization.speaker_labels):
            start = i * segment_length
            plt.axvspan(start, start + segment_length, alpha=0.3, color=f"C{label}")

    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title("Diarization: Energy with Speaker Labels")
    plt.legend()
    plt.show()

def visualize_embeddings(embeddings: np.ndarray, labels: List[int], method="tsne"):
    if embeddings.shape[0] == 0:
        print("No embeddings to visualize.")
        return

    plt.figure(figsize=(10, 7))
    if method == "tsne":
        perplexity = min(30, embeddings.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=perplexity)
        emb_2d = tsne.fit_transform(embeddings)
    else:
        emb_2d = embeddings[:, :2]

    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for i, label in enumerate(unique_labels):
        idxs = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(emb_2d[idxs, 0], emb_2d[idxs, 1], label=f"Speaker {label}", color=colors(i))

    plt.title("Speaker Embeddings Visualization")
    plt.legend()
    plt.show()

async def main():
    diarization = SpeakerDiarization(
        threshold=0.02,  # adjust this if needed to detect speech better
        n_speakers=2
    )

    # Load your audio file (make sure it's 16kHz mono)
    audio, sr = sf.read("output.wav")
    if sr != diarization.sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=diarization.sample_rate)

    # Feed the whole audio at once (or in chunks)
    await diarization.process_audio(audio.tolist())

    # Finalize to process remaining buffer
    await diarization.finalize_processing()

    if len(diarization.speaker_embeddings) == 0:
        print("No embeddings found after processing!")
        return

    print(f"Found {len(diarization.speaker_embeddings)} speaker embeddings.")
    print(f"Speaker labels: {diarization.speaker_labels}")

    # Uncomment to visualize embeddings and diarization
    # embeddings_np = np.vstack(diarization.speaker_embeddings)
    # visualize_embeddings(embeddings_np, diarization.speaker_labels)
    # visualize_diarization(audio, diarization.sample_rate, diarization)

if __name__ == "__main__":
    asyncio.run(main())
