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
OUTPUT_PATH = REPO_ROOT / "data" / "cs-multi.csv"
HEADER = "cs_level|direction|sentence"

# Per-bucket targets, keyed by (direction, cs_level). cs_level = number of
# minority-language islands (code-switches). cs-4 means "4 or more".
TARGETS: dict[tuple[str, int], int] = {
    ("EN_BASE", 2): 3388,
    ("JP_BASE", 2): 3496,
    ("EN_BASE", 3): 1850,
    ("JP_BASE", 3): 1850,
    ("EN_BASE", 4): 500,
    ("JP_BASE", 4): 500,
}

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

{island_detail}

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
        "with ENGLISH words borrowed in as nouns/adjectives/adverbs. "
        "The English words should feel like natural loanword-style insertions a bilingual speaker would use."
    ),
    "EN_BASE": (
        "Each sentence is PRIMARILY ENGLISH (subject, verb, grammar, structure in English) "
        "with JAPANESE words borrowed in as nouns/adjectives/adverbs, written in Japanese script. "
        "The Japanese words should feel like natural words a bilingual speaker would code-switch to."
    ),
}

# Minority-language word for each direction, used in the island instructions.
_MINORITY = {"JP_BASE": "English", "EN_BASE": "Japanese"}


def island_detail(direction: str, cs_level: int) -> str:
    minority = _MINORITY[direction]
    if cs_level >= 4:
        count_phrase = "FOUR OR MORE separate"
        n_word = "four or more"
    else:
        words = {2: "TWO", 3: "THREE"}[cs_level]
        count_phrase = f"EXACTLY {words} separate"
        n_word = {2: "two", 3: "three"}[cs_level]
    return (
        f"CODE-SWITCH COUNT: Each sentence must contain {count_phrase} {minority} "
        f"island(s) — that is, {n_word} distinct runs of {minority} text separated by "
        f"the primary language. Each island is one or more {minority} words (nouns, "
        f"adjectives, or adverbs only) sitting at a DIFFERENT point in the sentence, "
        f"with primary-language text in between them. Do not let the islands touch or "
        f"merge into one run. Two adjacent {minority} words with nothing between them "
        f"count as ONE island, not two — keep them apart."
    )


LINE_RE = re.compile(r"^(.+?)\|(.+)$")


def make_client():
    cfg = Config(
        region_name=REGION,
        retries={"max_attempts": 8, "mode": "adaptive"},
        read_timeout=180,
        connect_timeout=30,
    )
    return boto3.client("bedrock-runtime", config=cfg)


def call_model(client, direction: str, cs_level: int, theme: str, n: int, seed_salt: str) -> list[str]:
    user = USER_TEMPLATE.format(
        n=n,
        theme=theme,
        direction=direction,
        direction_detail=DIRECTION_DETAILS[direction],
        island_detail=island_detail(direction, cs_level),
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


def _char_class(c: str) -> str | None:
    """j = Japanese script, e = Latin letter, None = ignored for island runs."""
    if "぀" <= c <= "ヿ" or "一" <= c <= "鿿" or "ｦ" <= c <= "ﾟ":
        return "j"
    if c.isascii() and c.isalpha():
        return "e"
    return None


def count_islands(sent: str, minority: str) -> int:
    """Number of `minority`-language islands (code-switches).

    Classify each char as Japanese ('j') / Latin ('e') / ignored, run-length
    encode the j/e sequence, and count the runs of the minority language.
    The minority is the inserted language, known from the generation direction
    (JP_BASE inserts English 'e', EN_BASE inserts Japanese 'j') — passing it in
    is more robust than inferring it from char counts on borderline sentences.
    """
    runs: list[str] = []
    for c in sent:
        k = _char_class(c)
        if k is None:
            continue
        if not runs or runs[-1] != k:
            runs.append(k)
    return runs.count(minority)


def has_hiragana(s: str) -> bool:
    """Hiragana signals Japanese grammar (particles は/が/を/に/と, okurigana)."""
    return any("぀" <= c <= "ゟ" for c in s)


# English function words: their presence signals English grammatical scaffolding.
_EN_FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "to",
    "of",
    "in",
    "on",
    "at",
    "and",
    "or",
    "but",
    "for",
    "with",
    "my",
    "your",
    "his",
    "her",
    "their",
    "our",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "it",
    "do",
    "did",
    "does",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "from",
    "as",
    "so",
    "if",
    "when",
}


def has_en_function_word(s: str) -> bool:
    return any(w in _EN_FUNCTION_WORDS for w in re.findall(r"[A-Za-z']+", s.lower()))


def validate(direction: str, cs_level: int, sent: str) -> bool:
    jp = has_japanese(sent)
    en = has_latin_word(sent)
    if not (jp and en):
        return False
    if has_banned(sent):
        return False
    # "Primarily X" means X provides the grammatical scaffolding (particles,
    # function words) — NOT that X has the most characters. Packing in several
    # short minority-language islands can flip the raw char count while the
    # sentence stays structurally in the base language, so gate on grammar.
    if direction == "JP_BASE" and not has_hiragana(sent):
        return False
    if direction == "EN_BASE" and not has_en_function_word(sent):
        return False
    # For EN-primary sentences, the Japanese borrowed segment must not be
    # a conjugated verb form.
    if direction == "EN_BASE" and has_jp_verb_segment(sent):
        return False
    # Switch-count gate: cs-2/3 require an exact island count; cs-4 means 4+.
    # Minority language = the inserted one: 'e' (English) for JP_BASE, 'j' for EN_BASE.
    minority = "e" if direction == "JP_BASE" else "j"
    islands = count_islands(sent, minority)
    if cs_level >= 4:
        if islands < 4:
            return False
    elif islands != cs_level:
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


def load_existing(path: Path) -> tuple[dict[tuple[str, int], int], Dedup]:
    """Return per-bucket counts {(direction, cs_level): n} and a primed Dedup."""
    dedup = Dedup()
    counts: dict[tuple[str, int], int] = {k: 0 for k in TARGETS}
    if not path.exists():
        return counts, dedup
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if header.strip() != HEADER:
            # treat as empty and rewrite
            return counts, dedup
        for ln in f:
            parts = ln.rstrip("\n").split("|", 2)
            if len(parts) != 3:
                continue
            level_s, direction, sent = parts
            try:
                key = (direction, int(level_s))
            except ValueError:
                continue
            dedup.add(sent)
            if key in counts:
                counts[key] += 1
    return counts, dedup


def run(batch_size: int, workers: int, max_topups: int):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            f.write(HEADER + "\n")

    counts, dedup = load_existing(OUTPUT_PATH)
    counts_lock = threading.Lock()
    file_lock = threading.Lock()
    goal = sum(TARGETS.values())
    start = time.time()

    need = {k: max(0, TARGETS[k] - counts[k]) for k in TARGETS}
    print("Per-bucket status:", flush=True)
    for k in TARGETS:
        print(f"  {k[0]} cs-{k[1]}: have {counts[k]} / {TARGETS[k]} (need {need[k]})", flush=True)
    if all(v == 0 for v in need.values()):
        print("Already complete.")
        return

    # Build initial work queue: list of (direction, cs_level, theme, salt).
    # Over-provision ~20% because validation (island count, verb, loanword) drops lines.
    jobs: list[tuple[str, int, str, str]] = []

    def enqueue(key: tuple[str, int], n_needed: int):
        direction, level = key
        remaining = int(n_needed * 1.2) + batch_size
        theme_cycle = THEMES.copy()
        random.shuffle(theme_cycle)
        i = 0
        while remaining > 0:
            theme = theme_cycle[i % len(theme_cycle)]
            salt = f"{direction}-cs{level}-{random.randint(10**9, 10**10 - 1)}"
            jobs.append((direction, level, theme, salt))
            remaining -= batch_size
            i += 1

    for k in TARGETS:
        if need[k] > 0:
            enqueue(k, need[k])
    random.shuffle(jobs)

    client = make_client()

    def worker(direction: str, level: int, theme: str, salt: str):
        key = (direction, level)
        with counts_lock:
            if counts[key] >= TARGETS[key]:
                return 0
        try:
            raw_lines = call_model(client, direction, level, theme, batch_size, salt)
        except Exception as e:
            print(f"[ERR] {direction}/cs-{level}/{theme}: {e}", flush=True)
            return 0

        accepted: list[str] = []
        for ln in raw_lines:
            parsed = clean_line(ln)
            if not parsed:
                continue
            sent, _gloss = parsed
            if not validate(direction, level, sent):
                continue
            if not dedup.add(sent):
                continue
            accepted.append(sent)

        if not accepted:
            return 0

        with counts_lock:
            remaining = TARGETS[key] - counts[key]
            if remaining <= 0:
                return 0
            accepted = accepted[:remaining]
            counts[key] += len(accepted)
            total = sum(counts.values())

        with file_lock:
            with OUTPUT_PATH.open("a", encoding="utf-8") as f:
                for sent in accepted:
                    f.write(f"{level}|{direction}|{sent}\n")

        elapsed = time.time() - start
        pct = total / goal * 100 if goal else 0
        rate = total / elapsed if elapsed > 0 else 0
        eta = (goal - total) / rate if rate > 0 else 0
        print(
            f"[{direction}/cs-{level}/{theme[:20]:<20}] +{len(accepted):>2} "
            f"| {total}/{goal} ({pct:.1f}%) | {rate:.1f}/s ETA {eta / 60:.1f}m",
            flush=True,
        )
        return len(accepted)

    topup_round = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        pending = {ex.submit(worker, d, lv, t, s) for (d, lv, t, s) in jobs}
        while pending:
            _done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            with counts_lock:
                short = {k: TARGETS[k] - counts[k] for k in TARGETS}
            if not pending and any(v > 0 for v in short.values()) and topup_round < max_topups:
                topup_round += 1
                extras: list[tuple[str, int, str, str]] = []
                for (direction, level), s in short.items():
                    if s <= 0:
                        continue
                    for _ in range(max(1, (s // batch_size) + 2)):
                        extras.append(
                            (
                                direction,
                                level,
                                random.choice(THEMES),
                                f"{direction}-cs{level}-top{topup_round}-{random.randint(10**9, 10**10 - 1)}",
                            )
                        )
                short_str = ", ".join(f"{k[0]}cs{k[1]}={v}" for k, v in short.items() if v > 0)
                print(f"[top-up round {topup_round}] short {short_str}, queueing {len(extras)} batches", flush=True)
                for d, lv, t, s in extras:
                    pending.add(ex.submit(worker, d, lv, t, s))

    counts, _ = load_existing(OUTPUT_PATH)
    print("\nFinal per-bucket:", flush=True)
    for k in TARGETS:
        print(f"  {k[0]} cs-{k[1]}: {counts[k]} / {TARGETS[k]}", flush=True)
    print(f"Total: {sum(counts.values())} / {goal}", flush=True)


def self_check():
    """Assert count_islands matches the definition on canonical examples."""
    # (sentence, minority language, expected island count)
    cases = [
        ("I went to 東京 and visited 京都 last summer.", "j", 2),  # EN cs-2
        ("She ate 寿司 and drank お茶 at the restaurant.", "j", 2),
        ("新しいlaptopとtabletを買いました。", "e", 2),  # JP cs-2
        ("My お母さん bought 野菜 and お肉 at the market.", "j", 3),  # EN cs-3
        ("今日はmeetingとlunchとpresentationがありました。", "e", 3),  # JP cs-3
        ("今日 I ate 寿司 for lunch and 天ぷら for dinner with お茶 afterward.", "j", 4),  # EN cs-4
        ("毎朝gymでworkoutしてからprotein shakeを飲んでofficeに行きます。", "e", 4),  # JP cs-4
        ("I really want to eat 寿司 tonight.", "j", 1),  # cs-1 sanity
        ("新しいlaptoptabletを買いました。", "e", 1),  # adjacent minority words = ONE island
    ]
    for sent, minority, expected in cases:
        got = count_islands(sent, minority)
        assert got == expected, f"count_islands({sent!r}, {minority!r}) = {got}, expected {expected}"

    # Structural gate: a JP-grammar sentence stays valid even when its 4+ English
    # islands make Latin the character-majority (the bug that stalled JP cs-4).
    jp4 = "彼はmeetingでpresentationをしてからclientにemailとreportを送りました。"
    assert count_islands(jp4, "e") >= 4, "expected 4+ English islands"
    assert validate("JP_BASE", 4, jp4), "JP-grammar cs-4 must validate despite Latin char-majority"
    # And an English-grammar sentence with many Japanese islands stays valid.
    en4 = "The 朝食 was 静か, the 景色 lovely, and the whole 雰囲気 felt 上品."
    assert validate("EN_BASE", 4, en4), "EN-grammar cs-4 must validate"
    print(f"self-check OK ({len(cases)} count cases + structural gates)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-topups", type=int, default=40)
    ap.add_argument("--self-check", action="store_true", help="Run count_islands asserts and exit.")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return
    run(args.batch_size, args.workers, args.max_topups)


if __name__ == "__main__":
    main()
