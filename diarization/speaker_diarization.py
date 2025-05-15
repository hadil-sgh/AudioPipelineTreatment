import numpy as np
from typing import List, Tuple, Optional
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
import librosa
import asyncio
import gc
import math

# --- SpeechBrain Import ---
# For SpeechBrain version 1.0 and later (recommended)
from speechbrain.inference.speaker import EncoderClassifier
# If using older SpeechBrain (0.5.x), you might need:
# from speechbrain.pretrained import EncoderClassifier

class SpeakerDiarization:
    def __init__(
        self,
        min_speech_duration: float = 0.1,   # Further reduced for shorter utterances
        threshold: float = 0.005,           # Lower threshold for better speech detection
        n_speakers: int = 2,               # We know there are exactly 2 speakers
        embedding_model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[str] = None,
        process_buffer_duration: float = 0.15  # Reduced for more frequent updates
    ):
        self.sample_rate = 16000
        self.min_speech_duration = min_speech_duration
        self.threshold = threshold
        self.n_speakers = n_speakers
        self.process_buffer_duration = process_buffer_duration

        if device is None:
            self.device = "cuda" 
        else:
            self.device = device
        print(f"Using device: {self.device} for diarization.")

        print(f"Loading embedding model from: {embedding_model_source}")
        try:
            self.embedding_model = EncoderClassifier.from_hparams(
                source=embedding_model_source,
                run_opts={"device": self.device}
            )
            self.embedding_model.eval()
        except Exception as e:
            print(f"Error loading SpeechBrain model: {e}")
            raise

        self.vad_frame_length = 512  # Reduced for better temporal resolution
        self.vad_hop_length = 128    # Reduced for better temporal resolution
        self.audio_buffer = []
        self.speaker_embeddings = []
        self.speaker_labels = []
        self.last_speaker_change_time = 0
        self.min_speaker_change_duration = 0.3  # Reduced minimum time between speaker changes

        self.clustering = AgglomerativeClustering(
            n_clusters=self.n_speakers,
            distance_threshold=None,
            metric='cosine',
            linkage='average'
        )

        if self.device == "cuda":
           torch.backends.cudnn.benchmark = True

    def _extract_features(self, audio_segment_np: np.ndarray) -> Optional[np.ndarray]:
        # Ensure audio is 1D
        if audio_segment_np.ndim > 1:
            audio_segment_np = np.mean(audio_segment_np, axis=0)
            
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_segment_np).float().to(self.device)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        min_len_for_sb_model = 1600  # ~100ms
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
            except RuntimeError as e:
                print(f"RuntimeError during SB embedding extraction: {e} on tensor shape {audio_tensor.shape}")
                return None
            except Exception as e:
                print(f"Unexpected error during SB embedding extraction: {e}")
                return None

    def _detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
            
        # Use multiple features for better speech detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=self.vad_hop_length)
        energy = librosa.feature.rms(y=audio, frame_length=self.vad_frame_length, hop_length=self.vad_hop_length)[0]
        zero_crossings = librosa.feature.zero_crossing_rate(audio, frame_length=self.vad_frame_length, hop_length=self.vad_hop_length)[0]
        
        # Ensure all features have the same length
        min_length = min(len(onset_env), len(energy), len(zero_crossings))
        onset_env = onset_env[:min_length]
        energy = energy[:min_length]
        zero_crossings = zero_crossings[:min_length]
        
        # Normalize features
        onset_env = (onset_env - np.min(onset_env)) / (np.max(onset_env) - np.min(onset_env) + 1e-8)
        energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
        zero_crossings = (zero_crossings - np.min(zero_crossings)) / (np.max(zero_crossings) - np.min(zero_crossings) + 1e-8)
        
        # Combine features with weights
        combined = (0.6 * onset_env + 0.3 * energy + 0.1 * zero_crossings)
        
        # Apply smoothing
        combined = np.convolve(combined, np.ones(5)/5, mode='same')
        
        min_speech_samples = self.min_speech_duration * self.sample_rate
        min_speech_frames_vad = math.ceil(min_speech_samples / self.vad_hop_length)
        if min_speech_frames_vad < 1: min_speech_frames_vad = 1

        speech_segments = []
        is_speech = False
        start_frame = 0
        
        # Add hysteresis to prevent rapid switching
        high_threshold = self.threshold
        low_threshold = self.threshold * 0.3  # More aggressive hysteresis
        
        for i, val in enumerate(combined):
            if not is_speech and val > high_threshold:
                is_speech = True
                start_frame = i
            elif is_speech and val < low_threshold:
                is_speech = False
                segment_duration_frames = i - start_frame
                if segment_duration_frames >= min_speech_frames_vad:
                    speech_segments.append((start_frame, i))

        if is_speech:
            end_frame = len(combined)
            segment_duration_frames = end_frame - start_frame
            if segment_duration_frames >= min_speech_frames_vad:
                speech_segments.append((start_frame, end_frame))

        return speech_segments

    async def process_audio_buffer(self):
        if not self.audio_buffer:
            return

        audio_combined = np.array(self.audio_buffer, dtype=np.float32)
        self.audio_buffer = []

        # Ensure audio is 1D
        if audio_combined.ndim > 1:
            audio_combined = np.mean(audio_combined, axis=0)

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

            # Only process segments with sufficient energy
            segment_energy = np.mean(np.abs(segment_audio))
            if segment_energy < self.threshold * 0.5:  # Skip very quiet segments
                continue

            # Ensure segment is long enough for the model
            if len(segment_audio) < 1600:  # Minimum length for SpeechBrain model
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
        print(f"\nFinalizing: Processing remaining {len(self.audio_buffer)/self.sample_rate:.2f}s in buffer...")
        await self.process_audio_buffer() # This will process if buffer is not empty
        if self.device == "cuda":
           torch.cuda.empty_cache()
            # gc.collect() # Python's gc, less critical for GPU usually

    def _update_speaker_labels(self):
        if not self.speaker_embeddings: return
        if len(self.speaker_embeddings) < 2: return

        embeddings_matrix = np.vstack(self.speaker_embeddings)
        
        try:
            # Normalize embeddings
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            embeddings_matrix = embeddings_matrix / (norms + 1e-8)
            
            # Use fixed number of clusters
            self.clustering.n_clusters = 2
            labels = self.clustering.fit_predict(embeddings_matrix)
            
            # Ensure consistent speaker labeling
            if len(self.speaker_labels) > 0:
                # Map new labels to maintain consistency with previous labels
                last_label = self.speaker_labels[-1]
                current_time = len(self.speaker_labels) * self.process_buffer_duration
                
                # Only allow speaker change if enough time has passed
                if current_time - self.last_speaker_change_time >= self.min_speaker_change_duration:
                    if labels[-1] != last_label:
                        # Only flip if we have enough confidence in the new label
                        if len(labels) >= 3:  # Need at least 3 embeddings for confidence
                            # Calculate similarity with previous speaker
                            prev_embeddings = embeddings_matrix[:-1]
                            curr_embedding = embeddings_matrix[-1:]
                            similarity = np.mean(np.dot(prev_embeddings, curr_embedding.T))
                            
                            if similarity < 0.7:  # Only change if similarity is low enough
                                labels = 1 - labels  # Flip labels if needed
                                self.last_speaker_change_time = current_time
                                print(f"Speaker change detected at {current_time:.2f}s: {last_label} -> {labels[-1]} (similarity: {similarity:.3f})")
                else:
                    # Force same speaker if not enough time has passed
                    labels = np.full_like(labels, last_label)
            
            self.speaker_labels = labels.tolist()
                
        except Exception as e:
            print(f"Clustering error: {e}")

    def get_current_speaker(self) -> Optional[int]:
        if not self.speaker_labels: return None
        return self.speaker_labels[-1]

    def reset(self):
        self.audio_buffer = []
        self.speaker_embeddings = []
        self.speaker_labels = []
        if self.device == "cuda":
           torch.cuda.empty_cache()
        print("Diarization system reset.")

    def __del__(self):
        if self.device == "cuda" and torch.cuda.is_available():
            try:
               torch.cuda.empty_cache()
            except RuntimeError: pass

# Example usage
if __name__ == "__main__":
    async def test_diarization():
        audio_path = "ordinary.wav"
        diarization_target_sr = 16000

        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            if sr != diarization_target_sr:
                print(f"Resampling audio from {sr} Hz to {diarization_target_sr} Hz.")
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=diarization_target_sr)
            print(f"Audio loaded: ~{len(audio)/diarization_target_sr:.2f}s, SR: {diarization_target_sr} Hz")
        except FileNotFoundError:
            print(f"ERROR: Audio file '{audio_path}' not found.")
            return
        except Exception as e:
            print(f"Error loading audio: {e}")
            return

        # <<< REVERTED VAD PARAMETERS FOR TESTING >>>
        diarization = SpeakerDiarization(
            n_speakers=2,
            threshold=0.02,              # More lenient threshold
            min_speech_duration=0.3,     # Shorter min speech duration
            process_buffer_duration=5.0  # Keep this for speed
        )
        print(f"Diarization params: VAD Thresh={diarization.threshold}, Min Speech={diarization.min_speech_duration}s, Buffer Proc Duration={diarization.process_buffer_duration}s")

        chunk_duration_s = 0.5
        chunk_size_samples = int(diarization_target_sr * chunk_duration_s)

        print(f"\nFeeding audio in {chunk_duration_s:.2f}s LPCM input chunks...")
        # total_chunks = math.ceil(len(audio) / chunk_size_samples) # Not used directly
        for i in range(0, len(audio), chunk_size_samples):
            chunk = audio[i:i + chunk_size_samples]
            if not chunk.any(): continue
            await diarization.process_audio(chunk)
            await asyncio.sleep(0.001)

        await diarization.finalize_processing()

        print("\n--- Final Diarization Results ---")
        print(f"Total segments processed for embeddings: {len(diarization.speaker_embeddings)}")
        print(f"Collected speaker labels for segments: {diarization.speaker_labels}")
        if not diarization.speaker_labels:
            print("No speaker labels generated. Check VAD parameters and audio content.")
        else:
            num_unique_speakers_found = len(set(diarization.speaker_labels))
            print(f"Number of unique speaker labels found: {num_unique_speakers_found}")
            if diarization.n_speakers > 0 and num_unique_speakers_found != diarization.n_speakers:
                print(f"Warning: Expected {diarization.n_speakers} speakers, found {num_unique_speakers_found}.")

        diarization.reset()

    # # Correcting SpeechBrain UserWarning for inspect module if possible, or acknowledging it
    # import warnings
    # # This warning comes from SpeechBrain's older import style interacting with Python's inspect
    # # It's generally safe to ignore for now if SpeechBrain itself functions.
    # # warnings.filterwarnings("ignore", category=UserWarning, module="inspect") # Can be too broad
    # # Or, more specifically if you can identify the exact message source
    # warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated", category=UserWarning)


    asyncio.run(test_diarization())