from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import jiwer
from faster_whisper import WhisperModel
from fugashi import Tagger
from pykakasi import kakasi

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PAIRS = REPO_ROOT / "data" / "data-pairs.csv"
DEFAULT_LID_CHECKPOINT = REPO_ROOT / "models" / "lid-wav2vec2-f1-974"
SAMPLE_RATE = 16_000
LID_TO_WHISPER = {"e": "en", "j": "ja"}
JAPANESE_TEXT_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
ROMAJI_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
JAPANESE_TAGGER = Tagger()
KAKASI_CONVERTER = kakasi()

# Reuse the trained LID inference code rather than reimplementing it. The cmd/*
# scripts are not a package, so add cmd/train to the path before importing.
sys.path.insert(0, str(REPO_ROOT / "cmd" / "train"))
import predict as lid  # noqa: E402


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

    if args.baseline_only:
        results = [
            evaluate(sentence, transcribe(model, sentence.audio_path, args.beam_size)) for sentence in sentence_data
        ]
        print_results(results)
        return 0

    lid_model, feature_extractor, device = load_lid(args.lid_checkpoint, args.lid_device)
    if lid_model is None:
        return 1

    def run_pipeline(sentence: SentenceData) -> EvalResult:
        audio = lid.read_audio(sentence.audio_path, SAMPLE_RATE)
        transcription = transcribe_pipeline(
            model, lid_model, feature_extractor, audio, args.beam_size, args.min_segment_seconds, device
        )
        return evaluate(sentence, transcription)

    if args.pipeline_only:
        print_results([run_pipeline(sentence) for sentence in sentence_data])
        return 0

    pairs = [
        (
            evaluate(sentence, transcribe(model, sentence.audio_path, args.beam_size)),
            run_pipeline(sentence),
        )
        for sentence in sentence_data
    ]
    print_comparison(pairs)
    return 0


def load_lid(checkpoint: str, device_choice: str):
    checkpoint_path = lid.resolve_checkpoint(checkpoint)
    if not checkpoint_path.exists():
        print(f"LID checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return None, None, None
    device = lid.resolve_device(device_choice)
    model = lid.Wav2Vec2ForAudioFrameClassification.from_pretrained(checkpoint_path).to(device)
    return model, lid.load_feature_extractor(checkpoint_path), device


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
    parser.add_argument(
        "--lid-checkpoint",
        default=str(DEFAULT_LID_CHECKPOINT),
        help="LID checkpoint number, name, or path used to segment audio before Whisper",
    )
    parser.add_argument(
        "--lid-device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="device to run the LID model on",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=0.2,
        help="fold LID segments shorter than this into a neighbor before transcribing",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--baseline-only",
        action="store_true",
        help="only run plain base-Whisper transcription (no LID preprocessing)",
    )
    mode.add_argument(
        "--pipeline-only",
        action="store_true",
        help="only run the LID-preprocessing pipeline (no baseline comparison)",
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


def transcribe_pipeline(model, lid_model, feature_extractor, audio, beam_size, min_seconds, device) -> str:
    lid_segments = lid.predict(lid_model, feature_extractor, audio, device).segments
    coalesced = coalesce_segments(lid_segments, min_seconds)
    if not coalesced:
        segments, _ = model.transcribe(audio, beam_size=beam_size)
        return " ".join(segment.text.strip() for segment in segments).strip()

    texts: list[str] = []
    for segment in coalesced:
        language = LID_TO_WHISPER.get(segment.label)
        if language is None:
            continue
        clip = audio[int(segment.start_seconds * SAMPLE_RATE) : int(segment.end_seconds * SAMPLE_RATE)]
        if clip.size == 0:
            continue
        segments, _ = model.transcribe(clip, beam_size=beam_size, language=language)
        texts.append(" ".join(part.text.strip() for part in segments).strip())

    return " ".join(text for text in texts if text).strip()


def coalesce_segments(segments: list, min_seconds: float) -> list:
    # lid.predict already merges contiguous same-language runs; here we additionally
    # fold sub-threshold blips into their longer neighbor so Whisper does not
    # hallucinate on tiny clips, re-merging same-language runs after each fold.
    merged = merge_adjacent(segments)
    while len(merged) > 1:
        index = min(range(len(merged)), key=lambda i: duration(merged[i]))
        if duration(merged[index]) >= min_seconds:
            break

        before = merged[index - 1] if index > 0 else None
        after = merged[index + 1] if index + 1 < len(merged) else None
        neighbor = (
            before
            if after is None
            else after
            if before is None
            else (before if duration(before) >= duration(after) else after)
        )
        merged[index] = lid.Segment(
            label=neighbor.label,
            start_seconds=merged[index].start_seconds,
            end_seconds=merged[index].end_seconds,
            confidence=merged[index].confidence,
        )
        merged = merge_adjacent(merged)

    return merged


def merge_adjacent(segments: list) -> list:
    merged: list = []
    for segment in segments:
        if merged and merged[-1].label == segment.label:
            previous = merged[-1]
            merged[-1] = lid.Segment(
                label=previous.label,
                start_seconds=previous.start_seconds,
                end_seconds=segment.end_seconds,
                confidence=(previous.confidence + segment.confidence) / 2,
            )
        else:
            merged.append(segment)
    return merged


def duration(segment) -> float:
    return segment.end_seconds - segment.start_seconds


def evaluate(sentence: SentenceData, transcription: str) -> EvalResult:
    use_japanese_normalization = has_japanese_text(sentence.text) or has_japanese_text(transcription)
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
        romaji = "".join(item["hepburn"] for item in KAKASI_CONVERTER.convert(token.surface))
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


def print_comparison(pairs: list[tuple[EvalResult, EvalResult]]) -> None:
    improved = regressed = tied = 0
    for base, pipeline in pairs:
        print(f"Audio: {base.sentence.audio_path.relative_to(REPO_ROOT)}")
        print(f"Sentence:\t{base.sentence.text}")
        print(f"Base Transcription:\t{base.transcription}")
        print(f"Pipeline Transcription:\t{pipeline.transcription}")
        print(f"Base WER/CER:\t\t{base.word_error_rate:.6f} / {base.character_error_rate:.6f}")
        print(f"Pipeline WER/CER:\t{pipeline.word_error_rate:.6f} / {pipeline.character_error_rate:.6f}")
        wer_delta = pipeline.word_error_rate - base.word_error_rate
        cer_delta = pipeline.character_error_rate - base.character_error_rate
        print(f"Delta WER/CER:\t\t{wer_delta:+.6f} / {cer_delta:+.6f}")
        if wer_delta < 0:
            improved += 1
        elif wer_delta > 0:
            regressed += 1
        else:
            tied += 1
        print("--------------------------------")

    base_refs = [base.normalized_sentence for base, _ in pairs]
    base_hyps = [base.normalized_transcription for base, _ in pairs]
    pipe_hyps = [pipeline.normalized_transcription for _, pipeline in pairs]
    print(f"Evaluated files:\t\t{len(pairs)}")
    print(f"Base Aggregate WER/CER:\t\t{jiwer.wer(base_refs, base_hyps):.6f} / {jiwer.cer(base_refs, base_hyps):.6f}")
    print(f"Pipeline Aggregate WER/CER:\t{jiwer.wer(base_refs, pipe_hyps):.6f} / {jiwer.cer(base_refs, pipe_hyps):.6f}")
    print(f"Improved / Regressed / Tied:\t{improved} / {regressed} / {tied} (by WER)")


def _self_check() -> None:
    # ponytail: one assert that fails if the fold/re-merge logic breaks.
    segments = [
        lid.Segment(label="e", start_seconds=0.0, end_seconds=1.0, confidence=1.0),
        lid.Segment(label="j", start_seconds=1.0, end_seconds=1.05, confidence=1.0),  # sub-threshold blip
        lid.Segment(label="e", start_seconds=1.05, end_seconds=2.0, confidence=1.0),
    ]
    result = coalesce_segments(segments, min_seconds=0.2)
    assert [s.label for s in result] == ["e"], result
    assert result[0].start_seconds == 0.0 and result[0].end_seconds == 2.0, result
    print("self-check passed")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
