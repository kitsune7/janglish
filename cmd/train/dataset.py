from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000
LABEL_TO_ID = {"e": 0, "j": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_CSV_PATH = DEFAULT_DATA_ROOT / "data-pairs.csv"


@dataclass(frozen=True)
class LidExample:
    audio_path: Path
    sidecar_path: Path
    sentence: str


def discover_examples(
    csv_path: Path = DEFAULT_CSV_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> list[LidExample]:
    examples: list[LidExample] = []
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="|")
        for row in reader:
            audio_rel = row.get("Audio file", "").strip()
            if not audio_rel:
                continue

            audio_path = data_root / audio_rel
            sidecar_path = audio_path.with_suffix(".json")
            if not audio_path.exists() or not sidecar_path.exists():
                continue

            examples.append(
                LidExample(
                    audio_path=audio_path,
                    sidecar_path=sidecar_path,
                    sentence=row.get("Sentence", "").strip(),
                )
            )

    return examples


def split_examples(
    examples: list[LidExample],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 13,
) -> tuple[list[LidExample], list[LidExample], list[LidExample]]:
    if not examples:
        return [], [], []

    test_ratio = 1.0 - train_ratio - validation_ratio
    if train_ratio <= 0 or validation_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio, validation_ratio, and test_ratio must be positive")

    groups = np.asarray([str(example.audio_path.parent) for example in examples])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("Need at least 3 speaker groups to create speaker-disjoint train, validation, and test splits")

    held_out_group_count = min(
        len(unique_groups) - 1,
        max(2, round(len(unique_groups) * (validation_ratio + test_ratio))),
    )
    indices = np.arange(len(examples))
    train_indices, held_out_indices = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=held_out_group_count,
            random_state=seed,
        ).split(indices, groups=groups)
    )

    held_out_groups = groups[held_out_indices]
    held_out_group_count = len(np.unique(held_out_groups))
    test_group_count = min(
        held_out_group_count - 1,
        max(1, round(held_out_group_count * test_ratio / (validation_ratio + test_ratio))),
    )
    validation_relative_indices, test_relative_indices = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=test_group_count,
            random_state=seed,
        ).split(held_out_indices, groups=held_out_groups)
    )

    validation_indices = held_out_indices[validation_relative_indices]
    test_indices = held_out_indices[test_relative_indices]

    return (
        [examples[index] for index in train_indices],
        [examples[index] for index in validation_indices],
        [examples[index] for index in test_indices],
    )


class LidFrameDataset(Dataset):
    def __init__(self, examples: list[LidExample], sample_rate: int = SAMPLE_RATE):
        self.examples = examples
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        return {
            "audio": _read_audio(example.audio_path, self.sample_rate),
            "labels": _read_labels(example.sidecar_path),
            "audio_path": str(example.audio_path),
            "sentence": example.sentence,
        }


def _read_audio(path: Path, target_sample_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != target_sample_rate:
        audio = _resample(audio, sample_rate, target_sample_rate)

    return np.asarray(audio, dtype=np.float32)


def _resample(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if audio.size == 0 or source_sample_rate == target_sample_rate:
        return np.asarray(audio, dtype=np.float32)

    common_factor = gcd(source_sample_rate, target_sample_rate)
    up = target_sample_rate // common_factor
    down = source_sample_rate // common_factor
    return resample_poly(audio, up, down).astype(np.float32)


def _read_labels(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as file:
        sidecar = json.load(file)

    frame_labels = sidecar.get("frame_labels", "")
    unknown = sorted(set(frame_labels) - set(LABEL_TO_ID))
    if unknown:
        raise ValueError(f"{path} contains unsupported frame labels: {unknown}")

    return np.asarray([LABEL_TO_ID[label] for label in frame_labels], dtype=np.int64)


all_examples = discover_examples()
train_examples, validation_examples, test_examples = split_examples(all_examples)

training_data = LidFrameDataset(train_examples)
validation_data = LidFrameDataset(validation_examples)
test_data = LidFrameDataset(test_examples)

# Kept for the existing training loop import.
evaluation_data = validation_data
