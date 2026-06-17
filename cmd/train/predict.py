from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class Segment:
    label: str
    start_seconds: float
    end_seconds: float
    confidence: float


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    audio_path = resolve_audio_path(args.audio)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    audio = read_audio(audio_path, SAMPLE_RATE)
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(checkpoint_path).to(device)
    feature_extractor = load_feature_extractor(checkpoint_path)

    segments = predict_segments(model, feature_extractor, audio, device)
    print_segments(checkpoint_path, audio_path, segments)
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
        help="audio file to classify",
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


def resolve_audio_path(audio_path: Path) -> Path:
    audio_path = audio_path.expanduser()
    if audio_path.is_absolute():
        return audio_path
    return PROJECT_ROOT / audio_path


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


def predict_segments(
    model: Wav2Vec2ForAudioFrameClassification,
    feature_extractor: Wav2Vec2FeatureExtractor,
    audio: np.ndarray,
    device: torch.device,
) -> list[Segment]:
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
        return []

    id_to_label = normalized_id_to_label(model)
    frame_duration = len(audio) / SAMPLE_RATE / len(prediction_ids_np)
    return group_predictions(prediction_ids_np, confidences_np, id_to_label, frame_duration)


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


def print_segments(checkpoint_path: Path, audio_path: Path, segments: list[Segment]) -> None:
    print(f"Checkpoint: {display_path(checkpoint_path)}")
    print(f"Audio: {display_path(audio_path)}")

    if not segments:
        print("No prediction frames returned.")
        return

    for segment in segments:
        print(
            f"{segment.start_seconds:06.2f}-{segment.end_seconds:06.2f} "
            f"{segment.label} confidence={segment.confidence:.2f}"
        )


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


if __name__ == "__main__":
    raise SystemExit(main())
