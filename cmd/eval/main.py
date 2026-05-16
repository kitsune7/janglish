from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from fugashi import Tagger
import jiwer
from faster_whisper import WhisperModel
from pykakasi import kakasi


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PAIRS = REPO_ROOT / "data" / "data-pairs.csv"
JAPANESE_TEXT_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
ROMAJI_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
JAPANESE_TAGGER = Tagger()
KAKASI_CONVERTER = kakasi()


@dataclass(frozen=True)
class SentenceData:
    audio_path: Path
    text: str


@dataclass(frozen=True)
class EvalResult:
    sentence: SentenceData
    transcription: str
    normalized_sentence: str
    normalized_transcription: str
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
    use_japanese_normalization = has_japanese_text(sentence.text) or has_japanese_text(
        transcription
    )
    normalized_sentence = normalize_for_error_rates(
        sentence.text,
        use_japanese_normalization,
    )
    normalized_transcription = normalize_for_error_rates(
        transcription,
        use_japanese_normalization,
    )

    return EvalResult(
        sentence=sentence,
        transcription=transcription,
        normalized_sentence=normalized_sentence,
        normalized_transcription=normalized_transcription,
        word_error_rate=jiwer.wer(normalized_sentence, normalized_transcription),
        character_error_rate=jiwer.cer(normalized_sentence, normalized_transcription),
    )


def has_japanese_text(text: str) -> bool:
    return bool(JAPANESE_TEXT_PATTERN.search(text))


def normalize_for_error_rates(
    text: str,
    use_japanese_normalization: bool | None = None,
) -> str:
    if use_japanese_normalization is None:
        use_japanese_normalization = has_japanese_text(text)

    if not use_japanese_normalization:
        return " ".join(text.split())

    if not has_japanese_text(text):
        return " ".join(ROMAJI_TOKEN_PATTERN.findall(text.lower()))

    tokens: list[str] = []
    for token in JAPANESE_TAGGER(text):
        romaji = "".join(
            item["hepburn"] for item in KAKASI_CONVERTER.convert(token.surface)
        )
        for romaji_part in romaji.split():
            append_romaji_tokens(tokens, romaji_part)

    return " ".join(tokens)


def append_romaji_tokens(tokens: list[str], text: str) -> None:
    token = text.lower()
    if token == "'" and tokens:
        tokens[-1] += "'"
        return

    matches = ROMAJI_TOKEN_PATTERN.findall(token)
    if not matches:
        return

    if tokens and tokens[-1].endswith("'"):
        tokens[-1] += matches[0]
        tokens.extend(matches[1:])
        return

    tokens.extend(matches)


def print_results(results: list[EvalResult]) -> None:
    for result in results:
        print(f"Audio: {result.sentence.audio_path.relative_to(REPO_ROOT)}")
        print(f"Sentence:\t{result.sentence.text}")
        print(f"Transcription:\t{result.transcription}")
        if (
            result.normalized_sentence != result.sentence.text
            or result.normalized_transcription != result.transcription
        ):
            print(f"Normalized Sentence:\t\t{result.normalized_sentence}")
            print(f"Normalized Transcription:\t{result.normalized_transcription}")
        print(f"Word Error Rate:\t{result.word_error_rate:.6f}")
        print(f"Character Error Rate:\t{result.character_error_rate:.6f}")
        print("--------------------------------")

    references = [result.normalized_sentence for result in results]
    hypotheses = [result.normalized_transcription for result in results]
    print(f"Evaluated files:\t\t{len(results)}")
    print(f"Aggregate Word Error Rate:\t{jiwer.wer(references, hypotheses):.6f}")
    print(f"Aggregate Character Error Rate:\t{jiwer.cer(references, hypotheses):.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
