#!/usr/bin/env -S uv run --with boto3 --with botocore python
"""Generate bilingual Japanese/English mixed sentences via Bedrock.

Half are primarily Japanese with 1-3 English words, half primarily English with
1-3 Japanese words. Output: data/mixed-sentences.csv (Sentence|Gloss).

Resumable: re-running continues where the previous run stopped.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import threading
import time
from pathlib import Path

import boto3
from botocore.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "data" / "mixed-sentences.csv"
HEADER = "Sentence|Gloss"

MODEL_ID = os.environ.get("GEN_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
REGION = os.environ.get("AWS_REGION", "us-east-2")

THEMES = [
    "food and cooking",
    "office work and meetings",
    "travel and sightseeing",
    "weather and seasons",
    "family and relatives",
    "hobbies and pastimes",
    "technology and computers",
    "home and apartments",
    "school and studying",
    "health and medicine",
    "shopping and errands",
    "public transport and commuting",
    "emotions and feelings",
    "movies, tv, and books",
    "music and concerts",
    "pets and animals",
    "sports and exercise",
    "restaurants and cafes",
    "holidays and festivals",
    "clothing and fashion",
    "phone calls and messaging",
    "banking and money",
    "driving and cars",
    "children and parenting",
    "dating and relationships",
    "news and politics (neutral)",
    "gardening and nature",
    "drinks and bars",
    "hotels and accommodation",
    "post office and deliveries",
    "beauty and grooming",
    "work from home",
    "job interviews",
    "emergencies and trouble",
    "directions and maps",
    "seasons and holidays",
    "photography",
    "art and museums",
    "gaming",
    "cleaning and chores",
]

BANNED_WORDS = [
    # Shared phonology / loanwords pronounced the same
    "sushi",
    "sashimi",
    "tempura",
    "ramen",
    "udon",
    "soba",
    "miso",
    "wasabi",
    "tofu",
    "mochi",
    "matcha",
    "sake",
    "bento",
    "onigiri",
    "edamame",
    "teriyaki",
    "anime",
    "manga",
    "karaoke",
    "karate",
    "judo",
    "sumo",
    "ninja",
    "samurai",
    "shogun",
    "geisha",
    "kimono",
    "futon",
    "tsunami",
    "typhoon",
    "sayonara",
    "sensei",
    "senpai",
    "kohai",
    "konnichiwa",
    "arigatou",
    "arigato",
    "pikachu",
    "pokemon",
    "nintendo",
    "tamagotchi",
    "emoji",
    "tycoon",
    "tofu",
    "yakuza",
    "zen",
    "banzai",
    "bonsai",
    "origami",
    "haiku",
    "konbini",
    "izakaya",
    "onsen",
    "ryokan",
    "shinkansen",
    "tatami",
    "katana",
    "wagyu",
    "hibachi",
    "teppanyaki",
    "sudoku",
    "daikon",
    "nori",
    "shiitake",
    "panko",
    "yuzu",
    "kombu",
    "dashi",
    "umami",
    "okonomiyaki",
    "takoyaki",
    "mochi",
    # Brand/proper nouns that are identical in both
    "toyota",
    "honda",
    "sony",
    "nintendo",
    "canon",
    "nissan",
    "mazda",
    "fuji",
    "tokyo",
    "osaka",
    "kyoto",
    "yokohama",
    "sapporo",
    "nagoya",
    "hokkaido",
    "okinawa",
]

SYSTEM_PROMPT = """You generate bilingual Japanese/English mixed ("janglish") sentences for language learners.

HARD RULES:
1. Output format: one sentence per line, then a pipe, then a plain English gloss.
   Example: 今日はmeetingが三つあります。|I have three meetings today.
2. The mixed sentence MUST switch naturally between Japanese and English mid-sentence.
   The switch must make sense in context.
3. The minority-language words must be NOUNS, ADJECTIVES, or ADVERBS only.
   NEVER use verbs that require conjugation.
   NEVER mix grammar/particles/inflections from one language onto a word of the other.
   Specifically for EN-primary sentences: the Japanese borrowed word must be a bare
   noun/na-adjective/i-adjective in dictionary form or an adverb.
   NEVER use Japanese verb forms like 感動した, 落ち着いた, 解放された, している, された, ってくる,
   になった, 驚いた, 黙って, 混雑した — these are VERBS and are forbidden.
   Use nouns (e.g., 感動, 余裕, 雰囲気, 本音, 建前) or na-adjectives
   (e.g., 静か, 丁寧, 新鮮, 豪華) instead.
4. NEVER use words that sound identical in both languages (loanwords). Examples to AVOID:
   sushi, ramen, anime, karaoke, karate, ninja, samurai, tsunami, emoji, tofu, matcha,
   sake, bento, mochi, pikachu, pokemon, nintendo, tokyo, kyoto, osaka, sensei,
   arigato, konnichiwa, kimono, futon, zen, origami, haiku, bonsai, katana, wagyu,
   hibachi, yuzu, panko, nori, miso, wasabi, udon, soba, tempura, sashimi, izakaya,
   konbini, ryokan, onsen, shinkansen, okonomiyaki, takoyaki, shiitake, dashi, umami,
   daikon.
5. The gloss must be natural plain English (translate any Japanese words).
6. Use real, common vocabulary. No romaji — write Japanese in actual Japanese script
   (hiragana/katakana/kanji with ひらがな readings where useful).
7. Do NOT number the lines. Do NOT add any header, explanation, markdown, or quoting.
   Output ONLY the sentence|gloss lines.
8. Do not repeat sentences you've produced before in this batch.
9. Do not use "san", "chan", "kun", "sama", or similar honorifics as the lone borrowed
   word — pick something more content-bearing.
"""

USER_TEMPLATE = """Generate exactly {n} bilingual sentences on the theme: {theme}.

DIRECTION: {direction}

{direction_detail}

Vary sentence structures (statements, questions, requests, exclamations,
past/present/future, first/second/third person).
Vary which English/Japanese words are borrowed — don't reuse the same borrowed word
across more than 2 lines in this batch.
Avoid clichés.

Output format, one per line:
<mixed sentence>|<plain English gloss>

Produce exactly {n} lines. No preamble, no numbering, no blank lines."""

DIRECTION_DETAILS = {
    "JP_BASE": (
        "Each sentence is PRIMARILY JAPANESE (subject, verb, particles, sentence structure in Japanese) "
        "with 1 to 3 ENGLISH words borrowed in as nouns/adjectives/adverbs. "
        "The English word should feel like a natural loanword-style insertion a bilingual speaker would use."
    ),
    "EN_BASE": (
        "Each sentence is PRIMARILY ENGLISH (subject, verb, grammar, structure in English) "
        "with 1 to 3 JAPANESE words borrowed in as nouns/adjectives/adverbs, written in Japanese script. "
        "The Japanese word should feel like a natural word a bilingual speaker would code-switch to."
    ),
}

LINE_RE = re.compile(r"^(.+?)\|(.+)$")


def make_client():
    cfg = Config(
        region_name=REGION,
        retries={"max_attempts": 8, "mode": "adaptive"},
        read_timeout=180,
        connect_timeout=30,
    )
    return boto3.client("bedrock-runtime", config=cfg)


def call_model(client, direction: str, theme: str, n: int, seed_salt: str) -> list[str]:
    user = USER_TEMPLATE.format(
        n=n,
        theme=theme,
        direction=direction,
        direction_detail=DIRECTION_DETAILS[direction],
    )
    # Give the model a little per-batch uniqueness hint to reduce repetition across calls
    user += f"\n\n[batch-id: {seed_salt}]"

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "temperature": 1.0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user}],
    }

    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())
    text = "".join(block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def clean_line(line: str) -> tuple[str, str] | None:
    # Strip numbering like "1. " or "1) "
    line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
    # Strip leading/trailing quotes
    line = line.strip().strip('"').strip("'")
    m = LINE_RE.match(line)
    if not m:
        return None
    sent, gloss = m.group(1).strip(), m.group(2).strip().rstrip("|").strip()
    if not sent or not gloss:
        return None
    if "|" in sent or "|" in gloss:
        return None
    return sent, gloss


def has_banned(sent: str) -> bool:
    low = sent.lower()
    for w in BANNED_WORDS:
        # word-boundary-ish check; most banned words are latin
        if re.search(rf"\b{re.escape(w)}\b", low):
            return True
    return False


def has_japanese(s: str) -> bool:
    return any("぀" <= c <= "ヿ" or "一" <= c <= "鿿" or "ｦ" <= c <= "ﾟ" for c in s)


def has_latin_word(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", s))


VERB_ENDINGS = (
    "した",
    "しました",
    "している",
    "していた",
    "された",
    "されている",
    "られた",
    "られる",
    "っている",
    "ってきた",
    "ってくる",
    "になった",
    "になる",
    "になって",
    "だった",
    "ている",
    "ていた",
    "えた",
    "いた",
    "った",
    "って",
    "せる",
    "せた",
)


def has_jp_verb_segment(sent: str) -> bool:
    for m in re.finditer(r"[ぁ-んァ-ン一-龥ー々]+", sent):
        run = m.group(0)
        if len(run) < 2:
            continue
        if run.endswith(VERB_ENDINGS):
            return True
    return False


def validate(direction: str, sent: str) -> bool:
    jp = has_japanese(sent)
    en = has_latin_word(sent)
    if not (jp and en):
        return False
    if has_banned(sent):
        return False
    jp_count = sum(1 for c in sent if "぀" <= c <= "ヿ" or "一" <= c <= "鿿")
    latin_count = sum(1 for c in sent if c.isascii() and c.isalpha())
    if direction == "JP_BASE" and jp_count <= latin_count:
        return False
    if direction == "EN_BASE" and latin_count <= jp_count:
        return False
    # For EN-primary sentences, the Japanese borrowed segment must not be
    # a conjugated verb form.
    if direction == "EN_BASE" and has_jp_verb_segment(sent):
        return False
    return True


class Dedup:
    def __init__(self):
        self.lock = threading.Lock()
        self.seen: set[str] = set()

    def add(self, sent: str) -> bool:
        key = re.sub(r"\s+", " ", sent.lower().strip())
        with self.lock:
            if key in self.seen:
                return False
            self.seen.add(key)
            return True


def load_existing(path: Path) -> tuple[int, int, Dedup]:
    dedup = Dedup()
    jp_count = en_count = 0
    if not path.exists():
        return 0, 0, dedup
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if header.strip() != HEADER:
            # treat as empty and rewrite
            return 0, 0, dedup
        for ln in f:
            parsed = clean_line(ln)
            if not parsed:
                continue
            sent, _ = parsed
            dedup.add(sent)
            # Rough classification by script dominance
            jp_c = sum(1 for c in sent if "぀" <= c <= "ヿ" or "一" <= c <= "鿿")
            la_c = sum(1 for c in sent if c.isascii() and c.isalpha())
            if jp_c > la_c:
                jp_count += 1
            else:
                en_count += 1
    return jp_count, en_count, dedup


def run(target_jp: int, target_en: int, batch_size: int, workers: int):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUTPUT_PATH.exists()
    if new_file:
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            f.write(HEADER + "\n")

    have_jp, have_en, dedup = load_existing(OUTPUT_PATH)
    need_jp = max(0, target_jp - have_jp)
    need_en = max(0, target_en - have_en)
    print(
        f"Have: JP={have_jp} EN={have_en} | Need: JP={need_jp} EN={need_en} | Total target: {target_jp + target_en}",
        flush=True,
    )
    if need_jp == 0 and need_en == 0:
        print("Already complete.")
        return

    # Build work queue: list of (direction, theme, seed_salt)
    jobs: list[tuple[str, str, str]] = []

    def enqueue(direction: str, need: int):
        remaining = need
        theme_cycle = THEMES.copy()
        random.shuffle(theme_cycle)
        i = 0
        while remaining > 0:
            theme = theme_cycle[i % len(theme_cycle)]
            salt = f"{direction}-{random.randint(10**9, 10**10 - 1)}"
            jobs.append((direction, theme, salt))
            remaining -= batch_size
            i += 1

    # Rough: we'll over-provision by ~20% because validation drops some lines
    enqueue("JP_BASE", int(need_jp * 1.2) + batch_size)
    enqueue("EN_BASE", int(need_en * 1.2) + batch_size)
    random.shuffle(jobs)

    client = make_client()
    file_lock = threading.Lock()
    counts = {"JP_BASE": have_jp, "EN_BASE": have_en}
    counts_lock = threading.Lock()
    targets = {"JP_BASE": target_jp, "EN_BASE": target_en}
    start = time.time()

    def worker(direction: str, theme: str, salt: str):
        with counts_lock:
            if counts[direction] >= targets[direction]:
                return 0
        try:
            raw_lines = call_model(client, direction, theme, batch_size, salt)
        except Exception as e:
            print(f"[ERR] {direction}/{theme}: {e}", flush=True)
            return 0

        accepted: list[tuple[str, str]] = []
        for ln in raw_lines:
            parsed = clean_line(ln)
            if not parsed:
                continue
            sent, gloss = parsed
            if not validate(direction, sent):
                continue
            if not dedup.add(sent):
                continue
            accepted.append((sent, gloss))

        if not accepted:
            return 0

        with counts_lock:
            remaining = targets[direction] - counts[direction]
            if remaining <= 0:
                return 0
            accepted = accepted[:remaining]
            counts[direction] += len(accepted)
            cur_jp = counts["JP_BASE"]
            cur_en = counts["EN_BASE"]

        with file_lock:
            with OUTPUT_PATH.open("a", encoding="utf-8") as f:
                for sent, gloss in accepted:
                    f.write(f"{sent}|{gloss}\n")

        elapsed = time.time() - start
        total = cur_jp + cur_en
        goal = targets["JP_BASE"] + targets["EN_BASE"]
        pct = total / goal * 100 if goal else 0
        rate = total / elapsed if elapsed > 0 else 0
        eta = (goal - total) / rate if rate > 0 else 0
        print(
            f"[{direction}/{theme[:24]:<24}] +{len(accepted):>2} "
            f"| JP={cur_jp} EN={cur_en} ({pct:.1f}%) "
            f"| {rate:.1f}/s ETA {eta / 60:.1f}m",
            flush=True,
        )
        return len(accepted)

    max_topups = 10
    topup_round = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, d, t, s) for (d, t, s) in jobs]
        # As jobs complete, if we're still short, queue more (bounded)
        pending = set(futures)
        while pending:
            done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for _ in done:
                pass
            with counts_lock:
                short_jp = targets["JP_BASE"] - counts["JP_BASE"]
                short_en = targets["EN_BASE"] - counts["EN_BASE"]
            if not pending and (short_jp > 0 or short_en > 0) and topup_round < max_topups:
                topup_round += 1
                extras: list[tuple[str, str, str]] = []
                if short_jp > 0:
                    for _ in range(max(1, (short_jp // batch_size) + 2)):
                        extras.append(
                            (
                                "JP_BASE",
                                random.choice(THEMES),
                                f"JP_BASE-top{topup_round}-{random.randint(10**9, 10**10 - 1)}",
                            )
                        )
                if short_en > 0:
                    for _ in range(max(1, (short_en // batch_size) + 2)):
                        extras.append(
                            (
                                "EN_BASE",
                                random.choice(THEMES),
                                f"EN_BASE-top{topup_round}-{random.randint(10**9, 10**10 - 1)}",
                            )
                        )
                print(
                    f"[top-up round {topup_round}] short JP={short_jp} EN={short_en}, "
                    f"queueing {len(extras)} more batches",
                    flush=True,
                )
                for d, t, s in extras:
                    pending.add(ex.submit(worker, d, t, s))

    have_jp, have_en, _ = load_existing(OUTPUT_PATH)
    print(f"\nFinal: JP={have_jp} EN={have_en} total={have_jp + have_en}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    target_jp = args.total // 2
    target_en = args.total - target_jp
    run(target_jp, target_en, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
