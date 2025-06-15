import torch
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

class RealTimeEmotionRecognizer:
    """
    Real-time wrapper around firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3.
    """

    def __init__(
        self,
        model_id: str = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
        buffer_duration: float = 5.0,
    ):
        """
        Args:
            model_id: HF model identifier.
            buffer_duration: How many seconds of audio to accumulate before predicting.
        """
        # load model + feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        # map IDs↔labels
        self.id2label = self.model.config.id2label

        # device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # buffer settings
        self.sr = self.feature_extractor.sampling_rate
        self.buffer_size = int(self.sr * buffer_duration)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_ptr = 0

    def push_audio(self, chunk: np.ndarray) -> str | None:
        """
        Push a new audio chunk (1D float32 NumPy array at self.sr).
        When the internal buffer is full, runs inference and returns the emotion label;
        otherwise returns None.

        Args:
            chunk: audio samples, shape=(n_samples,)

        Returns:
            Predicted emotion label or None if buffer not yet full.
        """
        n = len(chunk)
        # if chunk larger than remaining space, we fill, predict, then start new buffer with the overflow
        end_ptr = self.write_ptr + n
        if end_ptr < self.buffer_size:
            self.buffer[self.write_ptr:end_ptr] = chunk
            self.write_ptr = end_ptr
            return None
        else:
            # fill to the end
            remaining = self.buffer_size - self.write_ptr
            self.buffer[self.write_ptr:] = chunk[:remaining]
            # predict on full buffer
            label = self._predict(self.buffer)
            # reset buffer and write overflow
            overflow = chunk[remaining:]
            self.buffer[:len(overflow)] = overflow
            self.write_ptr = len(overflow)
            return label

    def _predict(self, audio_array: np.ndarray) -> str:
        """
        Preprocesses the buffered audio and runs inference.
        """
        # feature extraction
        inputs = self.feature_extractor(audio_array,
                                        sampling_rate=self.sr,
                                        return_tensors="pt",
                                        truncation=True,
                                        padding="max_length",
                                        max_length=self.buffer_size)
        # move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()
        return self.id2label[pred_id]


# === Example usage ===
# recognizer = RealTimeEmotionRecognizer(buffer_duration=5.0)
# while streaming:
#     audio_chunk = get_next_audio_chunk()   # numpy array at 16 kHz
#     emotion = recognizer.push_audio(audio_chunk)
#     if emotion is not None:
#         print(f"Detected emotion: {emotion}")
