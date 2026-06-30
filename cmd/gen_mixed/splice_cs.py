#!/usr/bin/env -S uv run python
"""Splice generated cs-2/cs-3/cs-4 sentences into data/data-pairs.csv.

Reads data/cs-multi.csv (cs_level|direction|sentence), assigns each sentence a
synthetic/cs-N/csM.flac path continuing that block's numbering, and inserts the
rows at the END of the matching cs-N block in data-pairs.csv (preserving file
order: cs-2 block, then cs-3, then cs-4). Within each block, EN-primary rows are
written before JP-primary, mirroring the existing layout.

Idempotent: sentences already present in data-pairs.csv (by exact text) are
skipped, and csM numbering continues past the current max for each block.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRS_PATH = REPO_ROOT / "data" / "data-pairs.csv"
MULTI_PATH = REPO_ROOT / "data" / "cs-multi.csv"
MULTI_HEADER = "cs_level|direction|sentence"

PATH_RE = re.compile(r"^synthetic/cs-(\d+)/cs(\d+)\.flac$")


def read_pairs() -> list[str]:
    return PAIRS_PATH.read_text(encoding="utf-8").splitlines()


def existing_state(lines: list[str]) -> tuple[set[str], dict[int, int], dict[int, int]]:
    """Return (existing sentences, max csM index per level, last line index per level)."""
    seen: set[str] = set()
    max_idx: dict[int, int] = {}
    last_line: dict[int, int] = {}
    for i, line in enumerate(lines):
        path, _, sentence = line.partition("|")
        if not sentence:
            continue
        seen.add(sentence.strip())
        m = PATH_RE.match(path)
        if not m:
            continue
        level, idx = int(m.group(1)), int(m.group(2))
        max_idx[level] = max(max_idx.get(level, 0), idx)
        last_line[level] = i
    return seen, max_idx, last_line


def load_multi() -> dict[int, list[str]]:
    """Return {cs_level: [sentences]}, EN-primary first then JP-primary."""
    by_level: dict[int, list[tuple[bool, str]]] = {}
    with MULTI_PATH.open(encoding="utf-8") as f:
        header = f.readline().strip()
        if header != MULTI_HEADER:
            raise SystemExit(f"unexpected header in {MULTI_PATH}: {header!r}")
        for ln in f:
            parts = ln.rstrip("\n").split("|", 2)
            if len(parts) != 3:
                continue
            level_s, direction, sentence = parts
            sentence = sentence.strip()
            if not sentence:
                continue
            by_level.setdefault(int(level_s), []).append((direction == "EN_BASE", sentence))
    # EN-primary (True) sorts before JP-primary (False); stable within each.
    return {lv: [s for _, s in sorted(rows, key=lambda r: not r[0])] for lv, rows in by_level.items()}


def main() -> None:
    lines = read_pairs()
    seen, max_idx, last_line = existing_state(lines)
    by_level = load_multi()

    # Build the new rows per level, then splice each block's rows in at the end
    # of that block. Process levels high-to-low so earlier insertions don't shift
    # the line indices of later (lower-line-number) blocks.
    new_rows: dict[int, list[str]] = {}
    for level, sentences in by_level.items():
        if level not in last_line:
            raise SystemExit(f"no existing cs-{level} block in data-pairs.csv to append to")
        idx = max_idx.get(level, 0)
        rows: list[str] = []
        for s in sentences:
            if s in seen:
                continue
            seen.add(s)
            idx += 1
            rows.append(f"synthetic/cs-{level}/cs{idx}.flac|{s}")
        new_rows[level] = rows
        print(f"cs-{level}: {len(rows)} new rows (csM {max_idx.get(level, 0) + 1}..{idx})")

    for level in sorted(new_rows, reverse=True):
        rows = new_rows[level]
        if not rows:
            continue
        at = last_line[level] + 1
        lines[at:at] = rows

    PAIRS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(len(r) for r in new_rows.values())
    print(f"inserted {total} rows into {PAIRS_PATH}")


if __name__ == "__main__":
    main()
