from dataclasses import dataclass

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor


@dataclass
class FrameLabelCollator:
    feature_extractor: Wav2Vec2FeatureExtractor
    num_labels: int = 2
    # Round padded audio length up to a multiple of this many samples (0.5s at 16kHz).
    # Batch-longest padding alone yields a unique tensor shape almost every batch, and
    # the MPS backend caches a compiled graph per shape — unbounded shapes leaked ~100GB
    # of host memory over 9 hours until jetsam killed the process. Bucketing collapses
    # thousands of shapes into a few dozen so the graph cache stays bounded.
    length_bucket_samples: int = 8_000

    def __call__(self, batch):
        # batch: list of dicts {"audio": np.ndarray, "labels": np.ndarray}
        audios = [b["audio"] for b in batch]
        labels = [b["labels"] for b in batch]

        # Pad audio + build attention_mask via the feature extractor
        inputs = self.feature_extractor(
            audios,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = self._pad_to_length_bucket(inputs)

        frame_lengths = [_wav2vec2_frame_count(len(audio)) for audio in audios]
        max_frames = _wav2vec2_frame_count(inputs["input_values"].shape[1])
        padded_labels = torch.zeros((len(labels), max_frames, self.num_labels), dtype=torch.float)
        for i, (label, frame_length) in enumerate(zip(labels, frame_lengths)):
            aligned = _align_labels(label, frame_length)
            if len(aligned) > 0:
                padded_labels[i, : len(aligned)] = torch.nn.functional.one_hot(
                    torch.from_numpy(aligned),
                    num_classes=self.num_labels,
                ).float()

        inputs["labels"] = padded_labels
        return inputs

    def _pad_to_length_bucket(self, inputs):
        if self.length_bucket_samples <= 0:
            return inputs
        current_length = inputs["input_values"].shape[1]
        bucket = self.length_bucket_samples
        target_length = ((current_length + bucket - 1) // bucket) * bucket
        padding = target_length - current_length
        if padding == 0:
            return inputs
        inputs["input_values"] = torch.nn.functional.pad(inputs["input_values"], (0, padding))
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.nn.functional.pad(inputs["attention_mask"], (0, padding))
        return inputs


def _wav2vec2_frame_count(input_length: int) -> int:
    """Return wav2vec2's convolutional feature length for a raw sample count."""
    for kernel_size, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        input_length = (input_length - kernel_size) // stride + 1
    return max(input_length, 0)


def _align_labels(labels: np.ndarray, target_length: int) -> np.ndarray:
    if len(labels) == target_length:
        return labels
    if len(labels) == 0 or target_length == 0:
        # No aligned labels: return an empty array (never np.empty(target_length),
        # which is uninitialized memory and would produce garbage one-hot targets).
        return np.empty(0, dtype=np.int64)

    source_positions = (np.arange(target_length) + 0.5) * len(labels) / target_length
    source_indices = np.clip(source_positions.astype(np.int64), 0, len(labels) - 1)
    return labels[source_indices].astype(np.int64, copy=False)
