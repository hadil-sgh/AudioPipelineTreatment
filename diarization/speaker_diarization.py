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
        min_speech_duration: float = 0.75, # Default, can be overridden in __main__
        threshold: float = 0.03,         # Default, can be overridden in __main__
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
        if self.n_speakers <= 0:
            print(f"n_speakers <= 0, clustering will use distance_threshold={self.clustering.distance_threshold}")

        if self.device == "cuda":
           torch.backends.cudnn.benchmark = True

    def _extract_features(self, audio_segment_np: np.ndarray) -> Optional[np.ndarray]:
        if audio_segment_np.ndim > 1:
            audio_segment_np = np.mean(audio_segment_np, axis=0)
        audio_tensor =torch.from_numpy(audio_segment_np).float().to(self.device)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        min_len_for_sb_model = 1600 # ~100ms
        if audio_tensor.shape[1] < min_len_for_sb_model:
            padding_needed = min_len_for_sb_model - audio_tensor.shape[1]
            audio_tensor =torch.nn.functional.pad(audio_tensor, (0, padding_needed), mode='reflect')

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
        energy = librosa.feature.rms(y=audio, frame_length=self.vad_frame_length, hop_length=self.vad_hop_length)[0]
        if not energy.size:
            print("VAD Debug: No energy frames computed (audio segment likely too short for VAD analysis).")
            return []

        # <<< ENHANCED DEBUG PRINTS >>>
        print(f"\nVAD Debug (in _detect_speech_segments for {len(audio)/self.sample_rate:.2f}s audio buffer):")
        print(f"    Energy (min,max,mean,std): {np.min(energy):.4f}, {np.max(energy):.4f}, {np.mean(energy):.4f}, {np.std(energy):.4f}. Num Frames: {len(energy)}")
        print(f"    Using VAD threshold: {self.threshold:.4f}")

        min_speech_samples = self.min_speech_duration * self.sample_rate
        min_speech_frames_vad = math.ceil(min_speech_samples / self.vad_hop_length)
        if min_speech_frames_vad < 1: min_speech_frames_vad = 1
        print(f"    Min speech duration: {self.min_speech_duration:.2f}s == {min_speech_frames_vad} VAD frames")

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
                    # print(f"    VAD found segment: frames {start_frame}-{i} (duration {segment_duration_frames} frames)")
                else:
                    print(f"    VAD discarded short segment: frames {start_frame}-{i} (duration {segment_duration_frames} frames, needed {min_speech_frames_vad})")


        if is_speech:
            end_frame = len(energy)
            segment_duration_frames = end_frame - start_frame
            if segment_duration_frames >= min_speech_frames_vad:
                speech_segments_found_by_vad.append((start_frame, end_frame))
                # print(f"    VAD found segment at end: frames {start_frame}-{end_frame} (duration {segment_duration_frames} frames)")
            else:
                 print(f"    VAD discarded short segment at end: frames {start_frame}-{end_frame} (duration {segment_duration_frames} frames, needed {min_speech_frames_vad})")


        print(f"    _detect_speech_segments is returning {len(speech_segments_found_by_vad)} segments from this buffer.")
        return speech_segments_found_by_vad

    async def process_audio_buffer(self):
        if not self.audio_buffer:
            # print("process_audio_buffer called with empty self.audio_buffer") # Debug
            return

        audio_combined = np.array(self.audio_buffer, dtype=np.float32)
        self.audio_buffer = []

        print(f"\nProcessing accumulated buffer of {len(audio_combined)/self.sample_rate:.2f}s...")
        segments_frame_indices = self._detect_speech_segments(audio_combined)
        print(f"Found {len(segments_frame_indices)} segments from VAD to process for embeddings.")

        new_embeddings_added = False
        for start_frame, end_frame in segments_frame_indices:
            start_sample = int(start_frame * self.vad_hop_length)
            end_sample = int(end_frame * self.vad_hop_length)
            end_sample = min(end_sample, len(audio_combined))
            start_sample = min(start_sample, end_sample)
            segment_audio = audio_combined[start_sample:end_sample]

            if len(segment_audio) == 0:
                continue

            # print(f"Extracting features for segment of {len(segment_audio)/self.sample_rate:.2f}s") # Debug
            embedding = self._extract_features(segment_audio)
            if embedding is not None:
                self.speaker_embeddings.append(embedding)
                new_embeddings_added = True
            # else: # Debug
                # print(f"Feature extraction returned None for segment of {len(segment_audio)/self.sample_rate:.2f}s")


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
        if self.n_speakers > 0 and len(self.speaker_embeddings) < self.n_speakers:
            # print(f"Update labels: Not enough embeddings ({len(self.speaker_embeddings)}) for {self.n_speakers} clusters yet.")
            return

        embeddings_matrix = np.vstack(self.speaker_embeddings)
        if embeddings_matrix.shape[0] < 2:
            # print("Update labels: Less than 2 embeddings, cannot cluster.")
            return
        if self.clustering.n_clusters is not None and embeddings_matrix.shape[0] < self.clustering.n_clusters:
            # print(f"Update labels: Not enough embeddings ({embeddings_matrix.shape[0]}) for specified n_clusters ({self.clustering.n_clusters}).")
            return

        try:
            labels = self.clustering.fit_predict(embeddings_matrix)
            self.speaker_labels = labels.tolist()
            print(f"Speaker labels updated: {self.speaker_labels}")
        except ValueError as e:
            print(f"Clustering error: {e}. Embeddings shape: {embeddings_matrix.shape}")

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