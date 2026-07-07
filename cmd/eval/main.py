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
DEFAULT_LID_CHECKPOINT = REPO_ROOT / "models" / "lid-wav2vec2-f1-98"
SAMPLE_RATE = 16_000
LID_TO_WHISPER = {"e": "en", "j": "ja"}
JAPANESE_TEXT_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
# Kana/kanji plus CJK punctuation and fullwidth forms, for space-free joining.
CJK_JOIN_PATTERN = re.compile(r"[\u3000-\u30ff\u3400-\u9fff\uff00-\uffef]")
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
class TimedWord:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass(frozen=True)
class SpliceOptions:
    # An English LID span in Japanese-dominant audio must clear all of these
    # gates before its ja-pass words are swapped for en-pass words; otherwise
    # the span keeps the ja-pass words (baseline behavior, never worse).
    min_seconds: float
    min_confidence: float
    min_probability: float
    max_words_per_second: float


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
            model,
            lid_model,
            feature_extractor,
            audio,
            args.beam_size,
            args.min_segment_seconds,
            device,
            args.strategy,
            SpliceOptions(
                min_seconds=args.splice_min_seconds,
                min_confidence=args.splice_min_confidence,
                min_probability=args.splice_min_probability,
                max_words_per_second=args.splice_max_words_per_second,
            ),
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
    parser.add_argument(
        "--strategy",
        default="splice",
        choices=["splice", "dual", "chop"],
        help=(
            "splice: one full-audio pass in the dominant language; for Japanese-dominant audio, "
            "splice English LID spans from a second English pass (default); "
            "dual: per-word majority vote between full en and ja passes; "
            "chop: cut the audio at LID boundaries and transcribe each clip separately"
        ),
    )
    parser.add_argument(
        "--splice-min-seconds",
        type=float,
        default=0.3,
        help="skip splicing English spans shorter than this (likely LID false positives)",
    )
    parser.add_argument(
        "--splice-min-confidence",
        type=float,
        default=0.85,
        help="skip splicing English spans whose mean LID confidence is below this",
    )
    parser.add_argument(
        "--splice-min-probability",
        type=float,
        default=0.5,
        help="skip splicing when the mean Whisper word probability of the English candidates is below this",
    )
    parser.add_argument(
        "--splice-max-words-per-second",
        type=float,
        default=3.0,
        help="skip splicing when the English candidates exceed this word density for the span",
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


def transcribe_pipeline(
    model, lid_model, feature_extractor, audio, beam_size, min_seconds, device, strategy, splice_options
) -> str:
    lid_segments = lid.predict(lid_model, feature_extractor, audio, device).segments
    coalesced = coalesce_segments(lid_segments, min_seconds)
    languages = {LID_TO_WHISPER[segment.label] for segment in coalesced if segment.label in LID_TO_WHISPER}
    if not languages:
        segments, _ = model.transcribe(audio, beam_size=beam_size)
        return " ".join(segment.text.strip() for segment in segments).strip()

    if len(languages) == 1:
        # Whole clip is one language: transcribe the full audio in one pass so
        # Whisper keeps the entire sentence context, with the language pinned.
        segments, _ = model.transcribe(audio, beam_size=beam_size, language=next(iter(languages)))
        return " ".join(segment.text.strip() for segment in segments).strip()

    if strategy == "chop":
        return transcribe_chopped(model, audio, beam_size, coalesced)
    if strategy == "dual":
        return transcribe_dual_pass(model, audio, beam_size, coalesced)
    return transcribe_spliced(model, audio, beam_size, coalesced, splice_options)


def transcribe_spliced(model, audio, beam_size: int, coalesced: list, options: SpliceOptions) -> str:
    # Pick the language that covers most of the audio and transcribe the FULL
    # audio once with that language pinned, so Whisper keeps sentence context.
    #
    # English-dominant audio: stop there. Whisper-en renders embedded Japanese
    # as romaji ("Setsuyaku"), which is a faithful romanization, while a forced
    # ja pass over mostly-English audio produces garbage in the same window.
    #
    # Japanese-dominant audio: Whisper-ja can only katakana-ize embedded
    # English ("プライマル" for "primer"), so splice English LID spans with words
    # from a second full-audio English pass using word timestamps.
    english_seconds = sum(duration(segment) for segment in coalesced if segment.label == "e")
    japanese_seconds = sum(duration(segment) for segment in coalesced if segment.label == "j")

    if english_seconds >= japanese_seconds:
        segments, _ = model.transcribe(audio, beam_size=beam_size, language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()

    japanese_words = transcribe_words(model, audio, beam_size, "ja")
    english_words = transcribe_words(model, audio, beam_size, "en")
    return join_words(splice_words(coalesced, english_words, japanese_words, options))


def splice_words(
    coalesced: list,
    english_words: list["TimedWord"],
    japanese_words: list["TimedWord"],
    options: SpliceOptions,
) -> list[str]:
    # Walk the LID spans in order; each span takes the words of the chosen
    # pass whose midpoints fall inside it. Span-by-span assembly keeps word
    # order coherent even when word timestamps jitter across boundaries.
    pieces: list[str] = []
    for span in coalesced:
        pieces.extend(word.text for word in choose_span_words(span, english_words, japanese_words, options))
    return pieces


def choose_span_words(
    span,
    english_words: list["TimedWord"],
    japanese_words: list["TimedWord"],
    options: SpliceOptions,
) -> list["TimedWord"]:
    inside_japanese = words_inside(japanese_words, span)
    if span.label != "e":
        return inside_japanese

    # Gate 1: tiny or low-confidence en spans are usually LID false positives
    # on Japanese audio; keep the ja-pass words.
    if duration(span) < options.min_seconds or span.confidence < options.min_confidence:
        return inside_japanese

    inside_english = words_inside(english_words, span)
    if not inside_english:
        return inside_japanese

    # Gate 2: the en pass over Japanese-dominant audio hallucinates phrases
    # with sloppy timestamps; an implausible word density for the span means
    # the words did not actually come from this stretch of audio.
    max_words = max(1, int(duration(span) * options.max_words_per_second) + 1)
    if len(inside_english) > max_words:
        return inside_japanese

    # Gate 3: low decoder confidence on the candidate words themselves.
    mean_probability = sum(word.probability for word in inside_english) / len(inside_english)
    if mean_probability < options.min_probability:
        return inside_japanese

    return inside_english


def words_inside(words: list["TimedWord"], span) -> list["TimedWord"]:
    return [word for word in words if span.start_seconds <= (word.start + word.end) / 2 < span.end_seconds]


def join_words(pieces: list[str]) -> str:
    # Join with spaces between Latin-script words only. Inserting spaces inside
    # Japanese text changes fugashi tokenization and therefore kanji readings
    # during normalization ("新 しい" -> "shin shii" instead of "atarashii").
    joined = ""
    for piece in pieces:
        if not piece:
            continue
        if joined and not (is_cjk(joined[-1]) or is_cjk(piece[0])):
            joined += " "
        joined += piece
    return joined


def is_cjk(character: str) -> bool:
    return bool(CJK_JOIN_PATTERN.match(character))


def transcribe_dual_pass(model, audio, beam_size: int, coalesced: list) -> str:
    # Transcribe the FULL audio once per detected language so Whisper always has
    # complete sentence context, then keep each timestamped word only where the
    # LID timeline says its language was spoken. This avoids the failure modes
    # of chopping: lost context, hallucination on sub-second clips, and clipped
    # phonemes at hard cut points.
    selected: list[TimedWord] = []
    for label, language in LID_TO_WHISPER.items():
        intervals = [
            (segment.start_seconds, segment.end_seconds) for segment in coalesced if segment.label == label
        ]
        if not intervals:
            continue
        selected.extend(
            word for word in transcribe_words(model, audio, beam_size, language) if majority_inside(word, intervals)
        )

    selected.sort(key=lambda word: (word.start, word.end))
    return " ".join(word.text for word in selected).strip()


def transcribe_words(model, audio, beam_size: int, language: str) -> list["TimedWord"]:
    segments, _ = model.transcribe(audio, beam_size=beam_size, language=language, word_timestamps=True)
    words: list[TimedWord] = []
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            if text:
                words.append(
                    TimedWord(
                        start=word.start,
                        end=word.end,
                        text=text,
                        probability=getattr(word, "probability", 1.0) or 1.0,
                    )
                )
    return words


def majority_inside(word: TimedWord, intervals: list[tuple[float, float]]) -> bool:
    duration_seconds = max(word.end - word.start, 1e-9)
    return overlap_seconds(word.start, word.end, intervals) / duration_seconds > 0.5


def overlap_seconds(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(end, interval_end) - max(start, interval_start)) for interval_start, interval_end in intervals)


def transcribe_chopped(model, audio, beam_size: int, coalesced: list) -> str:
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

    # Word selection: keep a word only when the majority of it overlaps the
    # language's LID spans, so each dual-pass transcript contributes exactly
    # the words spoken in its language.
    en_intervals = [(0.0, 1.66), (2.28, 5.98)]
    ja_intervals = [(1.66, 2.28)]
    assert majority_inside(TimedWord(start=0.1, end=0.5, text="I've"), en_intervals)
    assert not majority_inside(TimedWord(start=1.7, end=2.2, text="been"), en_intervals)  # ja span
    assert majority_inside(TimedWord(start=1.7, end=2.2, text="節約"), ja_intervals)
    assert majority_inside(TimedWord(start=1.5, end=2.3, text="setsuyaku"), ja_intervals)  # straddles, mostly ja
    assert not majority_inside(TimedWord(start=1.5, end=2.3, text="setsuyaku"), en_intervals)
    assert overlap_seconds(1.0, 3.0, en_intervals) == (1.66 - 1.0) + (3.0 - 2.28)

    # Splice assembly: ja-dominant sentence その[lip gloss]はどこで買いましたか.
    # Ja spans take ja-pass words, the en span takes en-pass words, in span order.
    spans = [
        lid.Segment(label="j", start_seconds=0.0, end_seconds=0.5, confidence=1.0),
        lid.Segment(label="e", start_seconds=0.5, end_seconds=1.4, confidence=1.0),
        lid.Segment(label="j", start_seconds=1.4, end_seconds=3.0, confidence=1.0),
    ]
    english_words = [
        TimedWord(start=0.05, end=0.45, text="Where"),  # en-pass mangling of その: dropped
        TimedWord(start=0.55, end=0.9, text="lip"),
        TimedWord(start=0.9, end=1.35, text="gloss"),
        TimedWord(start=1.5, end=2.9, text="where did you buy it"),  # dropped: ja span
    ]
    japanese_words = [
        TimedWord(start=0.0, end=0.5, text="その"),
        TimedWord(start=0.5, end=1.4, text="リップグロス"),  # katakana-ized en span: dropped
        TimedWord(start=1.4, end=2.2, text="はどこで"),
        TimedWord(start=2.2, end=3.0, text="買いましたか？"),
    ]
    options = SpliceOptions(min_seconds=0.3, min_confidence=0.85, min_probability=0.5, max_words_per_second=3.0)
    spliced = join_words(splice_words(spans, english_words, japanese_words, options))
    assert spliced == "そのlip glossはどこで買いましたか？", spliced

    # Gates: a bad en span must fall back to the ja-pass words, never drop text.
    en_span = lid.Segment(label="e", start_seconds=0.5, end_seconds=1.4, confidence=0.99)
    ja_fallback = [TimedWord(start=0.5, end=1.4, text="リップグロス")]
    good_en = [TimedWord(start=0.6, end=1.3, text="gloss", probability=0.9)]
    assert choose_span_words(en_span, good_en, ja_fallback, options) == good_en
    # Tiny span -> fallback.
    tiny = lid.Segment(label="e", start_seconds=0.5, end_seconds=0.7, confidence=0.99)
    assert choose_span_words(tiny, [TimedWord(start=0.5, end=0.7, text="What")], ja_fallback, options) == []
    # Low LID confidence -> fallback.
    unsure = lid.Segment(label="e", start_seconds=0.5, end_seconds=1.4, confidence=0.6)
    assert choose_span_words(unsure, good_en, ja_fallback, options) == ja_fallback
    # Implausible word density ("I painted a nude color manicure." in 0.9s) -> fallback.
    dense = [TimedWord(start=0.5 + i * 0.15, end=0.6 + i * 0.15, text=w, probability=0.9)
             for i, w in enumerate(["I", "painted", "a", "nude", "color", "manicure."])]
    assert choose_span_words(en_span, dense, ja_fallback, options) == ja_fallback
    # Low word probability -> fallback.
    low_prob = [TimedWord(start=0.6, end=1.3, text="wife", probability=0.2)]
    assert choose_span_words(en_span, low_prob, ja_fallback, options) == ja_fallback
    # No en candidates -> fallback keeps the ja words instead of dropping text.
    assert choose_span_words(en_span, [], ja_fallback, options) == ja_fallback

    # Space-free joining must only apply around CJK characters.
    assert join_words(["I've", "been", "節約", "all", "year"]) == "I've been節約all year"
    assert join_words(["hello", "world"]) == "hello world"
    print("self-check passed")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
