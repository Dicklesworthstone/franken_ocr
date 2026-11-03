#!/usr/bin/env python3
"""Replay the runaway guard (src/native_engine/sampler.rs) on token streams.

A pure-Python port of `analyze_runaway_suffix`, `RunawayGuard::observe` and
`PageRunaway`'s numeral mapping, for counting where the guard fires on outputs
it never decoded (DISC-023, bd-gftg):

- From the 8,192-token floor, every 256 tokens, the last 2,048 are checked.
- A check is suspicious when some period up to 256 matches at least 15/16 of
  its lagged positions and at most 1/4 of the token 4-grams are distinct.
- Three suspicious checks in a row fire.
- `PageRunaway` (Unlimited-OCR-2, under FOCR_RUNAWAY_GUARD=1) runs the same
  test with every numeral piece read as one id, so "814 815 816 ..." loops.

The thresholds are named once below and checked against sampler.rs, so a
change there fails `--self-test` until this port follows it.

    scripts/runaway_replay.py --self-test
    scripts/runaway_replay.py DATA --v1-tokenizer PATH --v2-tokenizer PATH [--ndjson OUT]

DATA is a `uocr2_head_to_head.py` data directory (runs.ndjson, and
rerun.ndjson if present). Each output's text is re-encoded with its model's
tokenizer (`pip install tokenizers`, imported only in this mode), so ids and
counts approximate what the decoder emitted. Outputs are labelled by their
text, not by the guard: runaway when the second half of the text compresses
to <= 0.10 of its size, legit when it is >= 0.20 and the run finished. The
table reports four replays per output: raw and numeral-masked, each with no
floor (checks from 2,048) and at the production floor. A v1 single page's
default is raw at the floor; v2's opt-in page guard is masked at the floor.
"""

from __future__ import annotations

import argparse
import json
import operator
import re
import sys
import unicodedata
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLER = ROOT / "src" / "native_engine" / "sampler.rs"

# Every `RUNAWAY_GUARD_*: usize` constant in sampler.rs, by suffix.
CONSTANTS = {
    "MIN_TOKENS": 8_192,
    "CHECK_INTERVAL": 256,
    "WINDOW_TOKENS": 2_048,
    "MAX_PERIOD": 256,
    "MIN_PERIOD_CYCLES": 8,
    "MATCH_NUMERATOR": 15,
    "MATCH_DENOMINATOR": 16,
    "NGRAM_ORDER": 4,
    "NOVELTY_NUMERATOR": 1,
    "NOVELTY_DENOMINATOR": 4,
    "REQUIRED_HITS": 3,
}
MIN_TOKENS = CONSTANTS["MIN_TOKENS"]
INTERVAL = CONSTANTS["CHECK_INTERVAL"]
WINDOW = CONSTANTS["WINDOW_TOKENS"]
HITS = CONSTANTS["REQUIRED_HITS"]

# The numeral rule `PageRunaway::is_numeral_piece` implements, as source text.
NUMERAL_RULE = (
    "piece.trim_start_matches(['Ġ', '▁', ' '])",
    "!digits.is_empty() && digits.chars().all(char::is_numeric)",
)
NUMERAL = -1  # PAGE_RUNAWAY_NUMERAL: never a real id


def rust_constants(text: str) -> dict[str, int]:
    return {
        name: int(value.replace("_", ""))
        for name, value in re.findall(r"pub const RUNAWAY_GUARD_(\w+): usize = ([0-9_]+);", text)
    }


def drift(text: str) -> list[str]:
    """How sampler.rs's guard differs from this port; empty when in step."""
    found = rust_constants(text)
    problems = [
        f"RUNAWAY_GUARD_{name}: sampler.rs {found.get(name)}, runaway_replay.py {CONSTANTS.get(name)}"
        for name in sorted(set(found) | set(CONSTANTS))
        if found.get(name) != CONSTANTS.get(name)
    ]
    problems += [f"PageRunaway::is_numeral_piece no longer reads `{rule}`" for rule in NUMERAL_RULE if rule not in text]
    return problems


def is_numeral_piece(piece: str) -> bool:
    """`PageRunaway::is_numeral_piece`: Rust's `char::is_numeric` is general category N*."""
    digits = piece.lstrip("Ġ▁ ")
    return bool(digits) and all(unicodedata.category(c)[0] == "N" for c in digits)


def analyze(emitted: list[int], end: int | None = None) -> dict | None:
    """`analyze_runaway_suffix(&emitted[..end])`."""
    end = len(emitted) if end is None else end
    if end < WINDOW:
        return None
    w = emitted[end - WINDOW : end]
    n = len(w)
    max_period = min(CONSTANTS["MAX_PERIOD"], n // CONSTANTS["MIN_PERIOD_CYCLES"])
    best_period, best_matches, best_comparisons = 1, 0, n - 1
    for period in range(1, max_period + 1):
        comparisons = n - period
        matches = sum(map(operator.eq, w[period:], w[:comparisons]))
        lhs, rhs = matches * best_comparisons, best_matches * comparisons
        if lhs > rhs or (lhs == rhs and period < best_period):
            best_period, best_matches, best_comparisons = period, matches, comparisons
    order = CONSTANTS["NGRAM_ORDER"]
    total = n - order + 1
    unique = len(set(zip(*(w[i:] for i in range(order)))))
    suspicious = (
        best_matches * CONSTANTS["MATCH_DENOMINATOR"] >= best_comparisons * CONSTANTS["MATCH_NUMERATOR"]
        and unique * CONSTANTS["NOVELTY_DENOMINATOR"] <= total * CONSTANTS["NOVELTY_NUMERATOR"]
    )
    return {
        "emitted": end,
        "period": best_period,
        "matches": best_matches,
        "comparisons": best_comparisons,
        "unique_ngrams": unique,
        "total_ngrams": total,
        "suspicious": suspicious,
    }


def ppm(numerator: int, denominator: int) -> int:
    return numerator * 1_000_000 // denominator if denominator else 0


class Guard:
    """`RunawayGuard::new(true)`; `first_checkpoint=WINDOW` drops the floor."""

    def __init__(self, first_checkpoint: int = MIN_TOKENS) -> None:
        self.next_checkpoint = first_checkpoint
        self.hits = 0
        self.last: dict | None = None
        self.evidence: dict | None = None

    def observe(self, emitted: list[int]) -> dict | None:
        """The trigger's witness once it fires (and from then on), else None."""
        if self.evidence is not None:
            return self.evidence
        while len(emitted) >= self.next_checkpoint:
            checkpoint = self.next_checkpoint
            self.next_checkpoint += INTERVAL
            metrics = analyze(emitted, checkpoint)
            if metrics is None:
                continue
            self.last = metrics
            if not metrics["suspicious"]:
                self.hits = 0
                continue
            self.hits += 1
            if self.hits >= HITS:
                self.evidence = metrics
                return metrics
        return None


def fire_at(ids: list[int], floor: bool = True) -> int | None:
    """Tokens emitted when the guard fires on `ids`, or None."""
    evidence = Guard(MIN_TOKENS if floor else WINDOW).observe(ids)
    return evidence and evidence["emitted"]


def masked(ids: list[int], numerals: list[bool]) -> list[int]:
    """`PageRunaway`'s stream: a numeral id reads as one id; an id past the table is not a numeral."""
    return [NUMERAL if i < len(numerals) and numerals[i] else i for i in ids]


def periodic(n: int, period: int) -> list[int]:
    return [i % period for i in range(n)]


def near_periodic(n: int, period: int) -> list[int]:
    return [10_000 + i // period if i % period == 0 else i % period for i in range(n)]


def expect(ok: object, detail: object) -> None:
    """A self-test check that `python -O` cannot strip, as it strips `assert`."""
    if not ok:
        raise AssertionError(detail)


def self_test() -> None:
    """The sampler.rs unit vectors, then the constants against sampler.rs itself."""
    fired = MIN_TOKENS + 2 * INTERVAL

    m = analyze(periodic(WINDOW, 32))
    expect((m["period"], m["matches"], m["comparisons"]) == (32, WINDOW - 32, WINDOW - 32), m)
    expect((m["unique_ngrams"], m["total_ngrams"], ppm(m["unique_ngrams"], m["total_ngrams"])) == (32, WINDOW - 3, 15_647), m)
    expect(m["suspicious"], m)

    for period in (19, 23, 29, 32):
        tokens = periodic(fired, period)
        guard = Guard()
        expect(guard.observe(tokens[:MIN_TOKENS]) is None, f"period {period}: one check fired")
        expect(guard.observe(tokens[: MIN_TOKENS + INTERVAL]) is None, f"period {period}: two checks fired")
        evidence = guard.observe(tokens)
        expect(evidence and (evidence["emitted"], evidence["period"], guard.hits) == (fired, period, HITS), (period, evidence))

    m = analyze(near_periodic(fired, 32))
    expect(m["period"] == 32 and ppm(m["matches"], m["comparisons"]) >= 968_000, m)
    expect(ppm(m["unique_ngrams"], m["total_ngrams"]) < 250_000 and m["suspicious"], m)
    expect(fire_at(near_periodic(fired, 32)) == fired, "a repeated template with a changing field")

    guard = Guard()
    expect(guard.observe(list(range(32_768))) is None, "high-novelty ids fired")
    expect(ppm(guard.last["unique_ngrams"], guard.last["total_ngrams"]) == 1_000_000 and not guard.last["suspicious"], guard.last)

    broken = periodic(MIN_TOKENS, 32) + [1_000_000 + i for i in range(INTERVAL)]
    expect(analyze(broken, MIN_TOKENS)["suspicious"] and not analyze(broken)["suspicious"], "a broken run")
    expect(fire_at(broken) is None, "one suspicious check survived a normal one")

    tokens = near_periodic(fired, 47)
    one_shot, step = Guard().observe(tokens), Guard()
    for end in range(1, len(tokens) + 1):
        incremental = step.observe(tokens[:end])
        if incremental:
            break
    expect(incremental == one_shot, "the decision must not depend on how often the guard is polled")
    expect(step.observe(tokens + periodic(INTERVAL, 47)) is incremental, "a fire is terminal and keeps its first witness")

    # page_runaway_closes_a_counting_loop_the_raw_guard_misses
    numerals = [100 <= i < 1_100 for i in range(2_000)]
    counting = [100 + (k // 2) % 1_000 if k % 2 == 0 else 5 for k in range(MIN_TOKENS + 4 * INTERVAL)]
    expect(fire_at(counting) is None, "every number is a new id")
    expect(fire_at(masked(counting, numerals)) == fired, "the masked counting loop")
    expect(masked([2_000, 5, 150], numerals) == [2_000, 5, NUMERAL], "the numeral mapping")
    for piece in ("814", "Ġ814", "▁7", "0", "٣", "Ⅻ"):
        expect(is_numeral_piece(piece), piece)
    for piece in ("", "Ġ", "a1", "1.5", "8x", "Ġ-1", "814Ġ", "一"):
        expect(not is_numeral_piece(piece), piece)

    text = SAMPLER.read_text(encoding="utf-8")
    problems = drift(text)
    expect(not problems, problems)
    moved = text.replace("RUNAWAY_GUARD_MIN_TOKENS: usize = 8_192;", "RUNAWAY_GUARD_MIN_TOKENS: usize = 4_096;")
    expect(drift(moved) == ["RUNAWAY_GUARD_MIN_TOKENS: sampler.rs 4096, runaway_replay.py 8192"], drift(moved))
    expect(len(drift(text.replace("char::is_numeric", "char::is_ascii_digit"))) == 1, "the numeral rule")
    print(f"runaway_replay self-test ok: {len(CONSTANTS)} constants match {SAMPLER.relative_to(ROOT)}")


def outputs_of(path: Path):
    """(name, text, seconds) per page of an `ocr-batch` output, or one document's text."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        for result in data["results"]:
            yield Path(result.get("image", "?")).name, result.get("markdown") or "", result.get("seconds")
    elif isinstance(data, dict):
        yield None, data.get("markdown") or data.get("text") or "", None


def tail_ratio(text: str) -> float:
    tail = text.encode()[len(text.encode()) // 2 :]
    return round(len(zlib.compress(tail, 9)) / max(1, len(tail)), 3)


def replay(args: argparse.Namespace) -> int:
    from tokenizers import Tokenizer  # the self-test runs without it

    tokenizers, numerals = {}, {}
    for family, path in (("v1", args.v1_tokenizer), ("v2", args.v2_tokenizer)):
        tok = Tokenizer.from_file(str(path))
        tokenizers[family] = tok
        numerals[family] = [is_numeral_piece(tok.id_to_token(i) or "") for i in range(tok.get_vocab_size(with_added_tokens=True))]
    print({family: sum(table) for family, table in numerals.items()}, "numeral pieces in each vocabulary")

    runs = []
    for name in ("runs.ndjson", "rerun.ndjson"):
        path = args.data / name
        if path.exists():
            runs += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for run in runs:
        family = run["cell"][:2]
        budgeted = not run["run"].startswith("rerun/")
        for name, text, seconds in outputs_of(Path(run["output"])):
            ids = tokenizers[family].encode(text, add_special_tokens=False).ids
            seconds = seconds if seconds is not None else run.get("wall_s")
            finished = not (budgeted and run["cell"].endswith("single") and (seconds or 0) >= 590)
            z = tail_ratio(text)
            mask = masked(ids, numerals[family])
            rows.append({
                "cell": run["cell"], "run": run["run"], "name": name or run["run"], "family": family,
                "tokens": len(ids), "tail_z": z,
                "label": "runaway" if z <= 0.10 else ("legit" if z >= 0.20 and finished else "gray"),
                "raw": fire_at(ids, floor=False), "masked": fire_at(mask, floor=False),
                "raw_floor": fire_at(ids), "masked_floor": fire_at(mask),
            })
    if args.ndjson:
        args.ndjson.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(len(rows), "outputs ->", args.ndjson)

    columns = ("raw", "masked", "raw_floor", "masked_floor")
    long_ = [row for row in rows if row["tokens"] >= WINDOW + 2 * INTERVAL]
    print(f"outputs of {WINDOW + 2 * INTERVAL:,}+ tokens that fire: no floor (raw, masked) | at the {MIN_TOKENS:,} floor (raw, masked)")
    for family, label in sorted({(row["family"], row["label"]) for row in long_}):
        group = [row for row in long_ if (row["family"], row["label"]) == (family, label)]
        counts = [sum(row[c] is not None for row in group) for c in columns]
        print(f"  {family} {label:8} {len(group):>4}: {counts[0]:>3} {counts[1]:>3} | {counts[2]:>3} {counts[3]:>3}")
    print("fires on outputs not labelled runaway, and runaways nothing catches:")
    for row in sorted(long_, key=lambda r: (r["label"], r["family"], r["tokens"])):
        fires = any(row[c] is not None for c in columns)
        if fires != (row["label"] == "runaway"):
            print(f"  {row['label']:8} {row['family']} {row['tokens']:>6} z={row['tail_z']}", *(row[c] for c in columns), row["cell"], row["name"][:60])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--self-test", action="store_true", help="replay the sampler.rs unit vectors and check the constants")
    parser.add_argument("data", nargs="?", type=Path, help="a uocr2_head_to_head.py data directory")
    parser.add_argument("--v1-tokenizer", type=Path, help="Unlimited-OCR's tokenizer.json")
    parser.add_argument("--v2-tokenizer", type=Path, help="Unlimited-OCR-2's tokenizer.json")
    parser.add_argument("--ndjson", type=Path, help="write one row per output here")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not (args.data and args.v1_tokenizer and args.v2_tokenizer):
        parser.error("DATA, --v1-tokenizer and --v2-tokenizer are required without --self-test")
    return replay(args)


if __name__ == "__main__":
    sys.exit(main())
