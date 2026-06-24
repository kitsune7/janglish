#!/usr/bin/env python3
"""
generate_lid_sidecars.py
========================

Generate per-clip LID (language-ID) sidecar JSON files for EN/JA
code-switched audio, one JSON next to every .flac (e.g. Johnny/001.flac ->
Johnny/001.json).

WHAT IT DOES
------------
For each audio file listed in the transcript CSV it writes a JSON of the form:

    {
      "sentence": "Hello, 元気ですか?",
      "sample_rate": 16000,
      "duration_ms": 2345,
      "spans": [
        {"lang": "en", "text": "Hello,", "start_ms": 0,   "end_ms": 850,
         "start_sample": 0,     "end_sample": 13600},
        {"lang": "ja", "text": "元気ですか?", "start_ms": 850, "end_ms": 2345,
         "start_sample": 13600, "end_sample": 37520}
      ],
      "frame_rate_hz": 50,
      "num_frames": 117,
      "frame_labels": "eeeeeeejjjjjjj"
    }

HOW THE TIMINGS ARE PRODUCED
----------------------------
This is an OFFLINE pipeline (no acoustic model / no network required):
  1. The CSV `sentence` is taken as ground truth.
  2. The sentence is split into contiguous English / Japanese runs by Unicode
     script (fullwidth Japanese punctuation stays with the surrounding text and
     does not trigger a switch).
  3. Each run's spoken length is estimated from linguistic units
     (Japanese mora via pykakasi; English syllables via a vowel-group heuristic).
  4. Provisional language boundaries are placed proportionally inside the
     detected speech region, then SNAPPED to the nearest low-energy frame
     (the natural pause / articulation valley at a code-switch) in the real audio.
  5. Spans are emitted tiling [0, duration] with no gaps, plus a 50 Hz
     frame-label string ('e'/'j', one char per 20 ms frame).

NOTE: boundaries are energy-guided estimates anchored to length priors, not
phoneme-level forced alignment. Expect a few tens of ms of jitter. For tighter
boundaries, replace `place_boundaries()` with a real forced aligner
(e.g. torchaudio MMS_FA or WhisperX) when GPU/torch + model downloads are available.

DEPENDENCIES
------------
    pip install numpy soundfile pykakasi
(ffmpeg is NOT required; soundfile reads FLAC directly via libsndfile.)

CSV FORMAT
----------
Pipe-delimited, one header row, columns: "Audio file|Sentence", e.g.

    Audio file|Sentence
    manual/Johnny/001.flac|Their 財布 was stolen right outside the currency exchange.
    manual/chris/s2.flac|こんにちは、お元気ですか？

Audio paths in the CSV are interpreted RELATIVE TO --root.

USAGE
-----
    # default: --root is the directory that the CSV path prefix is relative to.
    # If your CSV has paths like "manual/Johnny/001.flac" and your audio lives
    # under /path/to/janglish/data/manual/..., then --root is /path/to/janglish/data
    python3 generate_lid_sidecars.py --csv data-pairs.csv --root /path/to/janglish/data

    # only process a subset (substring match on the CSV path):
    python3 generate_lid_sidecars.py --csv data-pairs.csv --root .. --filter manual/

    # only manual/ rows (default true); set --all to also process synthetic/ etc.
    python3 generate_lid_sidecars.py --csv data-pairs.csv --root .. --all
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import pykakasi
import soundfile as sf

FRAME_HZ = 50
FRAME_MS = 1000 // FRAME_HZ  # 20 ms per frame
SNAP_WINDOW_MS = 120  # +/- search radius for the energy valley
JA_PER_MORA = 0.14  # relative duration prior (seconds / mora)
EN_PER_SYLL = 0.22  # relative duration prior (seconds / syllable)

_kks = pykakasi.kakasi()


# --------------------------------------------------------------------------
# Text -> language runs
# --------------------------------------------------------------------------
def char_lang(ch):
    """Classify a character as 'en', 'ja', or None (neutral: space/punct/digit)."""
    if ch.isspace():
        return None
    o = ord(ch)
    if (
        0x3040 <= o <= 0x30FF  # hiragana + katakana
        or 0x31F0 <= o <= 0x31FF  # katakana phonetic extensions
        or 0x3400 <= o <= 0x4DBF  # CJK extension A
        or 0x4E00 <= o <= 0x9FFF  # CJK unified ideographs
        or 0xF900 <= o <= 0xFAFF  # CJK compatibility ideographs
        or 0xFF66 <= o <= 0xFF9D
    ):  # halfwidth katakana
        return "ja"
    if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
        return "en"
    # Fullwidth/JA punctuation (、。「」？！...) is orthography only -> neutral.
    return None


def segment_runs(sentence):
    """Split sentence into ordered runs: [{'lang','text'}, ...].

    Neutral characters (spaces, punctuation incl. fullwidth JA punctuation,
    digits) attach to the currently open run, so the surface order of the
    sentence is preserved exactly and span texts concatenate back to it.
    """
    runs = []
    cur_lang = None
    buf = ""

    def flush():
        nonlocal buf, cur_lang
        if buf.strip():
            runs.append({"lang": cur_lang, "text": buf.strip()})
        buf = ""

    for ch in sentence:
        cl = char_lang(ch)
        if cl in ("en", "ja"):
            if cur_lang is None:
                cur_lang = cl
                buf += ch
            elif cl == cur_lang:
                buf += ch
            else:
                flush()
                cur_lang = cl
                buf = ch
        else:
            buf += ch  # neutral / punctuation -> stays with current buffer

    flush()

    # Merge any accidental adjacent same-language runs.
    merged = []
    for r in runs:
        if merged and merged[-1]["lang"] == r["lang"]:
            merged[-1]["text"] += " " + r["text"]
        else:
            merged.append(dict(r))
    return merged


# --------------------------------------------------------------------------
# Duration priors
# --------------------------------------------------------------------------
def ja_mora(text):
    hira = "".join(item["hira"] for item in _kks.convert(text))
    small = "ゃゅょゎぁぃぅぇぉ"
    n = sum(1 for c in hira if 0x3040 <= ord(c) <= 0x309F)
    n -= sum(1 for c in hira if c in small)
    return max(n, 1)


def en_syll(text):
    total = 0
    for w in text.split():
        letters = [c for c in w.lower() if "a" <= c <= "z"]
        groups, prev = 0, False
        for c in letters:
            v = c in "aeiouy"
            if v and not prev:
                groups += 1
            prev = v
        total += max(groups, 1)
    return max(total, 1)


def run_weight(r):
    if r["lang"] == "ja":
        return ja_mora(r["text"]) * JA_PER_MORA
    return en_syll(r["text"]) * EN_PER_SYLL


# --------------------------------------------------------------------------
# Audio / energy
# --------------------------------------------------------------------------
def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def frame_rms(wav, sr, num_frames):
    hop = sr / FRAME_HZ
    rms = np.zeros(num_frames, dtype=np.float64)
    for i in range(num_frames):
        a = int(round(i * hop))
        b = int(round((i + 1) * hop))
        seg = wav[a : min(b, len(wav))]
        rms[i] = math.sqrt(float(np.mean(seg**2))) if len(seg) else 0.0
    return rms


def speech_bounds(rms):
    """Return (first_voiced_frame, last_voiced_frame_exclusive)."""
    if rms.max() <= 0:
        return 0, len(rms)
    thr = max(rms.max() * 0.06, np.percentile(rms, 30) * 1.5, 1e-5)
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        return 0, len(rms)
    return int(voiced[0]), int(voiced[-1]) + 1


def place_boundaries(runs, rms, duration_ms):
    """Return list of internal boundary times in ms between consecutive runs."""
    k = len(runs)
    if k == 1:
        return []
    s_f, e_f = speech_bounds(rms)
    t0, t1 = s_f * FRAME_MS, e_f * FRAME_MS
    if t1 <= t0:
        t0, t1 = 0, duration_ms
    weights = [run_weight(r) for r in runs]
    tot = sum(weights)
    win = SNAP_WINDOW_MS // FRAME_MS
    bounds, acc = [], 0.0
    for i in range(k - 1):
        acc += weights[i]
        prov = t0 + (t1 - t0) * (acc / tot)
        pf = int(round(prov / FRAME_MS))
        lo = max((bounds[-1] // FRAME_MS + 1) if bounds else s_f + 1, pf - win)
        hi = min(e_f - 1, pf + win)
        snapped = prov if hi <= lo else (lo + int(np.argmin(rms[lo : hi + 1]))) * FRAME_MS
        if bounds and snapped <= bounds[-1]:
            snapped = bounds[-1] + FRAME_MS
        bounds.append(int(round(snapped)))
    return bounds


# --------------------------------------------------------------------------
# Per-file
# --------------------------------------------------------------------------
def process(audio_path, sentence):
    wav, sr = load_audio(audio_path)
    n = len(wav)
    duration_ms = int(round(n / sr * 1000))
    num_frames = max(int(round(n * FRAME_HZ / sr)), 1)
    rms = frame_rms(wav, sr, num_frames)

    runs = segment_runs(sentence)
    bounds_ms = place_boundaries(runs, rms, duration_ms)
    edges = [0] + bounds_ms + [duration_ms]

    spans = []
    for i, r in enumerate(runs):
        s_ms, e_ms = edges[i], edges[i + 1]
        s_smp = 0 if i == 0 else int(round(s_ms / 1000 * sr))
        e_smp = n if i == len(runs) - 1 else int(round(e_ms / 1000 * sr))
        spans.append(
            {
                "lang": r["lang"],
                "text": r["text"],
                "start_ms": s_ms,
                "end_ms": e_ms,
                "start_sample": s_smp,
                "end_sample": e_smp,
            }
        )
    for i in range(1, len(spans)):
        spans[i]["start_sample"] = spans[i - 1]["end_sample"]

    labels = []
    for f in range(num_frames):
        c = (f + 0.5) * FRAME_MS
        lab = spans[-1]["lang"]
        for sp in spans:
            if sp["start_ms"] <= c < sp["end_ms"]:
                lab = sp["lang"]
                break
        labels.append("j" if lab == "ja" else "e")

    return {
        "sentence": sentence,
        "sample_rate": sr,
        "duration_ms": duration_ms,
        "spans": spans,
        "frame_rate_hz": FRAME_HZ,
        "num_frames": num_frames,
        "frame_labels": "".join(labels),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate EN/JA LID sidecar JSON files.")
    ap.add_argument("--csv", required=True, help="pipe-delimited transcript CSV (Audio file|Sentence)")
    ap.add_argument("--root", default=".", help="directory the CSV audio paths are relative to (default: cwd)")
    ap.add_argument("--filter", default=None, help="only process CSV paths containing this substring")
    ap.add_argument("--delimiter", default="|", help="CSV delimiter (default: |)")
    ap.add_argument("--all", action="store_true", help='process all rows (default: only paths starting with "manual/")')
    args = ap.parse_args()

    rows = []
    with open(args.csv, encoding="utf-8") as f:
        rd = csv.reader(f, delimiter=args.delimiter)
        next(rd, None)  # header
        for row in rd:
            if len(row) < 2:
                continue
            rel, sent = row[0].strip(), row[1].strip()
            if not args.all and not rel.startswith("manual/"):
                continue
            if args.filter and args.filter not in rel:
                continue
            rows.append((rel, sent))

    done, missing = 0, 0
    for rel, sent in rows:
        audio = os.path.join(args.root, rel)
        if not os.path.exists(audio):
            print("MISSING", audio, file=sys.stderr)
            missing += 1
            continue
        data = process(audio, sent)
        out = os.path.splitext(audio)[0] + ".json"
        with open(out, "w", encoding="utf-8") as g:
            json.dump(data, g, ensure_ascii=False, indent=2)
        done += 1
        print(f"{rel}  spans={len(data['spans'])}  frames={data['num_frames']}")

    print(f"\nWROTE {done} sidecar JSON file(s)" + (f", {missing} missing audio" if missing else ""))


if __name__ == "__main__":
    main()
