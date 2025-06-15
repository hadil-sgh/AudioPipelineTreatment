import queue
import threading
import wave
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class SpeakerDiarizer:
    def __init__(
        self,
        embedding_extractor,
        two_speaker_threshold=17,
        silhouette_diff_threshold=0.0001,
        max_speakers=10,
    ):
        """
        embedding_extractor: callable(audio_path) -> np.ndarray
        thresholds control clustering behavior
        """
        self.embedding_extractor = embedding_extractor
        self.two_speaker_threshold = two_speaker_threshold
        self.silhouette_diff_threshold = silhouette_diff_threshold
        self.max_speakers = max_speakers

        self._sentences = []  # list of (text, embedding)
        self._speaker_labels = []

    def add_segment(self, text: str, audio_bytes: np.ndarray):
        """
        Add a transcribed segment and its raw audio bytes
        """
        # save bytes as wav
        temp = wave.open("_tmp.wav", 'wb')
        temp.setnchannels(1)
        temp.setsampwidth(2)
        temp.setframerate(16000)
        temp.writeframes((audio_bytes * 32767).astype(np.int16).tobytes())
        temp.close()

        emb = self.embedding_extractor("_tmp.wav")
        self._sentences.append((text, emb))
        self._update_clusters()

    def _determine_optimal_clusters(self, embeddings_scaled):
        n = len(embeddings_scaled)
        if n <= 1:
            return 1

        # quick 2-cluster check
        km = KMeans(n_clusters=2, random_state=0).fit(embeddings_scaled)
        dists = km.transform(embeddings_scaled)
        avg = np.mean(np.min(dists, axis=1))
        if avg < self.two_speaker_threshold:
            return 1

        # silhouette-based search
        scores = []
        candidates = range(2, min(self.max_speakers, n) + 1)
        for k in candidates:
            hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = hc.fit_predict(embeddings_scaled)
            uniq = set(labels)
            if 1 < len(uniq) < n:
                scores.append(silhouette_score(embeddings_scaled, labels))
            else:
                scores.append(-1)

        # find elbow
        opt = 2
        for i in range(1, len(scores)):
            if scores[i] < scores[i-1] + self.silhouette_diff_threshold:
                opt = candidates[i-1]
                break
        return opt

    def _update_clusters(self):
        embs = np.vstack([e for _, e in self._sentences])
        scaler = StandardScaler()
        embs_s = scaler.fit_transform(embs)

        k = self._determine_optimal_clusters(embs_s)
        if k == 1:
            self._speaker_labels = [0] * len(self._sentences)
            return

        hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = hc.fit_predict(embs_s)
        # remap labels to 0..k-1
        mapping = {}
        new_lbl = 0
        out = []
        for lbl in labels:
            if lbl not in mapping:
                mapping[lbl] = new_lbl
                new_lbl += 1
            out.append(mapping[lbl])

        self._speaker_labels = out

    def get_diarization(self):
        """
        Returns list of (text, speaker_label)
        """
        return list(zip([t for t,_ in self._sentences], self._speaker_labels))

# Usage example:
# from faster_whisper import WhisperModel
# model = WhisperModel(...)
# def extract_emb(path): return model.embed_audio(path)
# diar = SpeakerDiarizer(extract_emb)
# diar.add_segment(text, audio_bytes)
# for sentence, spk in diar.get_diarization(): print(spk, sentence)
