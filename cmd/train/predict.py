from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForAudioFrameClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = PROJECT_ROOT / "models" / "lid-wav2vec2"
SAMPLE_RATE = 16_000
DEFAULT_ID_TO_LABEL = {0: "e", 1: "j"}
AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}


@dataclass(frozen=True)
class Segment:
    label: str
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True)
class Prediction:
    segments: list[Segment]
    frame_labels: np.ndarray


@dataclass(frozen=True)
class SidecarComparison:
    sidecar_path: Path
    accuracy: float
    correct_frames: int
    compared_frames: int
    total_actual_frames: int
    aligned_frames: int
    ignored_frames: int


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    audio_paths = resolve_audio_paths(args.audio)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1
    if not audio_paths:
        print(f"No audio files found: {resolve_path(args.audio)}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(checkpoint_path).to(device)
    feature_extractor = load_feature_extractor(checkpoint_path)

    comparisons: list[SidecarComparison] = []
    for index, audio_path in enumerate(audio_paths):
        if index:
            print("--------------------------------")

        audio = read_audio(audio_path, SAMPLE_RATE)
        prediction = predict(model, feature_extractor, audio, device)
        comparison = compare_sidecar(audio_path, prediction.frame_labels)
        print_prediction(checkpoint_path, audio_path, prediction.segments, comparison)
        if comparison is not None:
            comparisons.append(comparison)

    if len(audio_paths) > 1:
        print_aggregate_comparison(comparisons)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict EN/JA language segments with a trained LID checkpoint.")
    parser.add_argument(
        "checkpoint",
        help="checkpoint number, checkpoint-* directory name, or explicit checkpoint path",
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="audio file or directory of audio files to classify",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="device to run inference on",
    )
    return parser.parse_args()


def resolve_checkpoint(checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.parts and checkpoint_path.exists():
        return checkpoint_path
    if checkpoint.isdigit():
        return CHECKPOINT_ROOT / f"checkpoint-{checkpoint}"
    if checkpoint.startswith("checkpoint-"):
        return CHECKPOINT_ROOT / checkpoint
    return checkpoint_path


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_audio_paths(audio_path: Path) -> list[Path]:
    resolved_path = resolve_path(audio_path)
    if resolved_path.is_file():
        return [resolved_path]
    if not resolved_path.is_dir():
        return []
    return sorted(
        path
        for path in resolved_path.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_audio(path: Path, target_sample_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != target_sample_rate:
        audio = resample(audio, sample_rate, target_sample_rate)

    return np.asarray(audio, dtype=np.float32)


def resample(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if audio.size == 0 or source_sample_rate == target_sample_rate:
        return np.asarray(audio, dtype=np.float32)

    common_factor = gcd(source_sample_rate, target_sample_rate)
    up = target_sample_rate // common_factor
    down = source_sample_rate // common_factor
    return resample_poly(audio, up, down).astype(np.float32)


def load_feature_extractor(checkpoint_path: Path) -> Wav2Vec2FeatureExtractor:
    try:
        return Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_path)
    except OSError:
        return Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=SAMPLE_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )


def predict(
    model: Wav2Vec2ForAudioFrameClassification,
    feature_extractor: Wav2Vec2FeatureExtractor,
    audio: np.ndarray,
    device: torch.device,
) -> Prediction:
    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        confidences, prediction_ids = probabilities.max(dim=-1)

    prediction_ids_np = prediction_ids.cpu().numpy()
    confidences_np = confidences.cpu().numpy()
    if len(prediction_ids_np) == 0:
        return Prediction(segments=[], frame_labels=np.asarray([], dtype=str))

    id_to_label = normalized_id_to_label(model)
    frame_duration = len(audio) / SAMPLE_RATE / len(prediction_ids_np)
    frame_labels = np.asarray([id_to_label.get(int(label_id), str(int(label_id))) for label_id in prediction_ids_np])
    return Prediction(
        segments=group_predictions(prediction_ids_np, confidences_np, id_to_label, frame_duration),
        frame_labels=frame_labels,
    )


def normalized_id_to_label(model: Wav2Vec2ForAudioFrameClassification) -> dict[int, str]:
    id_to_label = {}
    for key, value in model.config.id2label.items():
        label_id = int(key)
        label = str(value)
        id_to_label[label_id] = DEFAULT_ID_TO_LABEL.get(label_id, label) if label.startswith("LABEL_") else label
    return id_to_label


def group_predictions(
    prediction_ids: np.ndarray,
    confidences: np.ndarray,
    id_to_label: dict[int, str],
    frame_duration: float,
) -> list[Segment]:
    segments: list[Segment] = []
    start_index = 0
    current_id = int(prediction_ids[0])

    for index, prediction_id in enumerate(prediction_ids[1:], start=1):
        prediction_id = int(prediction_id)
        if prediction_id == current_id:
            continue

        segments.append(build_segment(current_id, start_index, index, confidences, id_to_label, frame_duration))
        start_index = index
        current_id = prediction_id

    segments.append(
        build_segment(current_id, start_index, len(prediction_ids), confidences, id_to_label, frame_duration)
    )
    return segments


def build_segment(
    label_id: int,
    start_index: int,
    end_index: int,
    confidences: np.ndarray,
    id_to_label: dict[int, str],
    frame_duration: float,
) -> Segment:
    return Segment(
        label=id_to_label.get(label_id, str(label_id)),
        start_seconds=start_index * frame_duration,
        end_seconds=end_index * frame_duration,
        confidence=float(confidences[start_index:end_index].mean()),
    )


def compare_sidecar(audio_path: Path, prediction_labels: np.ndarray) -> SidecarComparison | None:
    sidecar_path = audio_path.with_suffix(".json")
    if not sidecar_path.exists():
        return None

    with sidecar_path.open(encoding="utf-8") as file:
        sidecar = json.load(file)

    actual_labels = np.asarray(list(str(sidecar.get("frame_labels", ""))))
    aligned_actual_labels = align_labels(actual_labels, len(prediction_labels))
    comparable = np.isin(aligned_actual_labels, list(DEFAULT_ID_TO_LABEL.values()))
    compared_frames = int(comparable.sum())
    correct_frames = int((prediction_labels[comparable] == aligned_actual_labels[comparable]).sum())

    return SidecarComparison(
        sidecar_path=sidecar_path,
        accuracy=correct_frames / compared_frames if compared_frames else 0.0,
        correct_frames=correct_frames,
        compared_frames=compared_frames,
        total_actual_frames=len(actual_labels),
        aligned_frames=len(aligned_actual_labels),
        ignored_frames=len(aligned_actual_labels) - compared_frames,
    )


def align_labels(labels: np.ndarray, target_length: int) -> np.ndarray:
    if len(labels) == target_length:
        return labels
    if target_length == 0:
        return np.asarray([], dtype=str)
    if len(labels) == 0:
        return np.full(target_length, "", dtype=str)

    source_positions = (np.arange(target_length) + 0.5) * len(labels) / target_length
    source_indices = np.clip(source_positions.astype(np.int64), 0, len(labels) - 1)
    return labels[source_indices]


def print_prediction(
    checkpoint_path: Path,
    audio_path: Path,
    segments: list[Segment],
    comparison: SidecarComparison | None,
) -> None:
    print(f"Checkpoint: {display_path(checkpoint_path)}")
    print(f"Audio: {display_path(audio_path)}")
    if comparison is not None:
        print(f"Sidecar: {display_path(comparison.sidecar_path)}")
        print(
            "Frame Accuracy: "
            f"{comparison.accuracy:.2%} "
            f"({comparison.correct_frames}/{comparison.compared_frames} frames correct)"
        )
        if comparison.ignored_frames:
            print(f"Ignored Frames: {comparison.ignored_frames} unsupported sidecar labels")
        if comparison.total_actual_frames != comparison.aligned_frames:
            print(
                "Aligned Frames: "
                f"{comparison.total_actual_frames} sidecar frames -> {comparison.aligned_frames} prediction frames"
            )

    if not segments:
        print("No prediction frames returned.")
        return

    for segment in segments:
        print(
            f"{segment.start_seconds:06.2f}-{segment.end_seconds:06.2f} "
            f"{segment.label} confidence={segment.confidence:.2f}"
        )


def print_aggregate_comparison(comparisons: list[SidecarComparison]) -> None:
    print("--------------------------------")
    if not comparisons:
        print("Aggregate Frame Accuracy: no sidecars found")
        return

    compared_frames = sum(comparison.compared_frames for comparison in comparisons)
    correct_frames = sum(comparison.correct_frames for comparison in comparisons)
    if compared_frames == 0:
        print(f"Aggregate Frame Accuracy: no comparable frames across {len(comparisons)} sidecars")
        return

    print(
        "Aggregate Frame Accuracy: "
        f"{correct_frames / compared_frames:.2%} "
        f"({correct_frames}/{compared_frames} frames correct across {len(comparisons)} sidecars)"
    )


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


if __name__ == "__main__":
    raise SystemExit(main())
