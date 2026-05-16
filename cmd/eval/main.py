from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import jiwer
from faster_whisper import WhisperModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PAIRS = REPO_ROOT / "data" / "data-pairs.csv"


@dataclass(frozen=True)
class SentenceData:
    audio_path: Path
    text: str


@dataclass(frozen=True)
class EvalResult:
    sentence: SentenceData
    transcription: str
    word_error_rate: float
    character_error_rate: float


def main() -> int:
    args = parse_args()
    sentence_data = load_sentence_data(args.data_pairs, args.limit, args.prefix)
    if not sentence_data:
        print("No matching audio files found to evaluate.", file=sys.stderr)
        return 1

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    results = [
        evaluate(sentence, transcribe(model, sentence.audio_path, args.beam_size))
        for sentence in sentence_data
    ]

    print_results(results)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe Janglish audio and report WER/CER.",
    )
    parser.add_argument(
        "--data-pairs",
        type=Path,
        default=DEFAULT_DATA_PAIRS,
        help="pipe-delimited CSV with Audio file and Sentence columns",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="faster-whisper model size or local model path",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device passed to faster-whisper, such as cpu or cuda",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="compute type passed to faster-whisper, such as int8 or float16",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="beam size used during transcription",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="maximum number of existing audio rows to evaluate; 0 means all",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="only evaluate CSV audio paths with this prefix, e.g. manual/Johnny",
    )
    return parser.parse_args()


def load_sentence_data(data_pairs_path: Path, limit: int, prefix: str) -> list[SentenceData]:
    data_dir = data_pairs_path.parent
    rows: list[SentenceData] = []

    with data_pairs_path.open(newline="", encoding="utf-8") as data_pairs:
        reader = csv.DictReader(data_pairs, delimiter="|")
        for row in reader:
            relative_audio_path = row["Audio file"]
            if prefix and not relative_audio_path.startswith(prefix):
                continue

            audio_path = data_dir / relative_audio_path
            if not audio_path.exists():
                print(f"Skipping missing audio file: {audio_path}", file=sys.stderr)
                continue

            rows.append(SentenceData(audio_path=audio_path, text=row["Sentence"]))
            if limit > 0 and len(rows) >= limit:
                break

    return rows


def transcribe(model: WhisperModel, audio_path: Path, beam_size: int) -> str:
    segments, _ = model.transcribe(str(audio_path), beam_size=beam_size)
    return " ".join(segment.text.strip() for segment in segments).strip()


def evaluate(sentence: SentenceData, transcription: str) -> EvalResult:
    return EvalResult(
        sentence=sentence,
        transcription=transcription,
        word_error_rate=jiwer.wer(sentence.text, transcription),
        character_error_rate=jiwer.cer(sentence.text, transcription),
    )


def print_results(results: list[EvalResult]) -> None:
    for result in results:
        print(f"Audio: {result.sentence.audio_path.relative_to(REPO_ROOT)}")
        print(f"Sentence: {result.sentence.text}")
        print(f"Transcription: {result.transcription}")
        print(f"Word Error Rate: {result.word_error_rate:.6f}")
        print(f"Character Error Rate: {result.character_error_rate:.6f}")
        print("--------------------------------")

    references = [result.sentence.text for result in results]
    hypotheses = [result.transcription for result in results]
    print(f"Evaluated files: {len(results)}")
    print(f"Aggregate Word Error Rate: {jiwer.wer(references, hypotheses):.6f}")
    print(f"Aggregate Character Error Rate: {jiwer.cer(references, hypotheses):.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
