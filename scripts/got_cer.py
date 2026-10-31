#!/usr/bin/env python3
"""GOT-OCR2.0 Character Error Rate (CER) harness — stdlib only, no deps.

Usage:
    python3 scripts/got_cer.py <hypothesis.txt> <ground_truth.txt>
    python3 scripts/got_cer.py --raw <hypothesis.txt> <ground_truth.txt>
    python3 scripts/got_cer.py --format-corpus
    python3 scripts/got_cer.py --selftest

Prints JSON: {"chars_gt", "edit_distance", "cer", "words_gt", "word_edit_distance", "wer"}

`--format-corpus` scores the frozen readings in tests/fixtures/got/format_corpus
against their sidecars and exits 1 if any score differs from `FORMAT_SCORES`.

Normalization (applied to BOTH hypothesis and ground truth so the comparison is
fair on a scanned historical book), in order:
  1. normalize curly quotes -> straight, en/em dashes -> "-"
  2. strip a leading running-header line (page-number + SMALL-CAPS title + [year])
  3. drop a trailing standalone page number
  4. lowercase
  5. join line-end hyphenation ("pene-\\ntrate" -> "penetrate")
  6. collapse all whitespace runs to a single space (and strip)
"""

import contextlib
import io
import json
import os
import re
import sys
import tempfile


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Curly quotes / primes -> straight ASCII quotes.
_QUOTE_MAP = {
    "‘": "'",  # left single quote
    "’": "'",  # right single quote / apostrophe
    "‚": "'",  # single low-9 quote
    "‛": "'",  # single high-reversed-9 quote
    "′": "'",  # prime
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "„": '"',  # double low-9 quote
    "‟": '"',  # double high-reversed-9 quote
    "″": '"',  # double prime
    "«": '"',  # <<
    "»": '"',  # >>
}

# En dash, em dash, figure dash, horizontal bar, minus sign -> "-".
_DASH_MAP = {
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "―": "-",  # horizontal bar
    "−": "-",  # minus sign
}


def _normalize_punct(text):
    """Curly quotes -> straight; en/em/other dashes -> '-'."""
    out = []
    for ch in text:
        if ch in _QUOTE_MAP:
            out.append(_QUOTE_MAP[ch])
        elif ch in _DASH_MAP:
            out.append(_DASH_MAP[ch])
        else:
            out.append(ch)
    return "".join(out)


# A running-header line on a scanned page typically looks like one of:
#     "42   THE HISTORY OF ROME   [1848]"
#     "THE HISTORY OF ROME   42"
#     "42   THE HISTORY OF ROME"
# i.e. a leading page number and/or a bracketed year around a SMALL-CAPS
# (all-uppercase) title. We only strip a *leading* line that matches this shape,
# and only when it is not the sole line of content.
_HEADER_RE = re.compile(
    r"""^\s*
        (?:\d{1,4}\s+)?                 # optional leading page number
        (?=.*[A-Z])                     # must contain some caps (a title)
        [A-Z0-9][A-Z0-9\s.,'"&:;()-]*   # SMALL-CAPS / uppercase title run
        (?:\s+\d{1,4})?                 # optional trailing page number
        (?:\s*\[\s*\d{3,4}\s*\])?       # optional bracketed [year]
        \s*$
    """,
    re.VERBOSE,
)

# A trailing standalone page number line: just digits (optionally bracketed).
_TRAILING_PAGENO_RE = re.compile(r"^\s*\[?\s*\d{1,4}\s*\]?\s*$")


def _strip_running_header(lines):
    """Drop a leading running-header line if one is present and there is more
    content after it."""
    if len(lines) >= 2 and _looks_like_header(lines[0]):
        return lines[1:]
    return lines


def _looks_like_header(line):
    stripped = line.strip()
    if not stripped:
        return False
    # Must have at least one uppercase letter and no lowercase letters in the
    # title portion — a real body line has lowercase text.
    if any(c.islower() for c in stripped):
        return False
    if not any(c.isupper() for c in stripped):
        return False
    return bool(_HEADER_RE.match(stripped))


def _drop_trailing_pageno(lines):
    """Drop a trailing standalone page-number line if present and there is more
    content before it."""
    if len(lines) >= 2 and _TRAILING_PAGENO_RE.match(lines[-1] or ""):
        return lines[:-1]
    return lines


# Line-end hyphenation: a word fragment ending in "-" at end of line, joined to
# the fragment starting the next line. "pene-\ntrate" -> "penetrate".
_HYPHEN_JOIN_RE = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")

_WS_RE = re.compile(r"\s+")


def normalize(text):
    """Apply the full normalization pipeline."""
    # 1. Punctuation normalization first (affects header/quote matching).
    text = _normalize_punct(text)

    # Split into lines for header / trailing-pageno stripping.
    lines = text.split("\n")
    # Note: header detection runs before lowercasing so SMALL-CAPS is visible.
    lines = _strip_running_header(lines)
    lines = _drop_trailing_pageno(lines)
    text = "\n".join(lines)

    # 5. Join line-end hyphenation (before lowercasing is fine; regex is
    # case-insensitive on the class). Do this before whitespace collapse so the
    # newline is still present.
    text = _HYPHEN_JOIN_RE.sub(r"\1\2", text)

    # 4. Lowercase.
    text = text.lower()

    # 6. Collapse all whitespace runs to a single space and strip.
    text = _WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Levenshtein edit distance (two-row DP, O(n*m) time, O(min) space)
# ---------------------------------------------------------------------------

def edit_distance(a, b):
    """Levenshtein edit distance between sequences a and b."""
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # Ensure b is the shorter dimension for the row width.
    if m > n:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(hyp, gt, raw=False):
    if not raw:
        hyp = normalize(hyp)
        gt = normalize(gt)

    gt_chars = list(gt)
    hyp_chars = list(hyp)
    char_ed = edit_distance(hyp_chars, gt_chars)
    cer = char_ed / len(gt_chars) if gt_chars else (0.0 if not hyp_chars else 1.0)

    gt_words = gt.split()
    hyp_words = hyp.split()
    word_ed = edit_distance(hyp_words, gt_words)
    wer = word_ed / len(gt_words) if gt_words else (0.0 if not hyp_words else 1.0)

    return {
        "chars_gt": len(gt_chars),
        "edit_distance": char_ed,
        "cer": cer,
        "words_gt": len(gt_words),
        "word_edit_distance": word_ed,
        "wer": wer,
    }


# ---------------------------------------------------------------------------
# The `--format` corpus (bd-fqn1)
# ---------------------------------------------------------------------------

FORMAT_CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "fixtures", "got", "format_corpus",
)

# asset -> (int8 CER, f32 CER, verdict). Exact on purpose, in both directions: a
# frozen reading that gets worse OR better has to change a number here in a
# reviewed diff. A misread asset is listed with its real CER and is not a pass.
FORMAT_SCORES = {
    "table": (0.0, 0.0, "read"),
    "molecule": (0.0, 0.0, "read"),
    "chart": (0.3, 0.1, "MISREAD by int8, bd-kd75"),
    "chart.task": (0.55, 0.55, "values read, title lost"),
    "music": (1.3023, 1.3023, "MISREAD, f32 too"),
    "formula": (4.8485, 4.8485, "MISREAD as SMILES, f32 too"),
}


def _squeeze(text):
    return _WS_RE.sub("", text)


def _table_cells(text):
    return re.sub(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\hline|&|\\\\", "", text)


# RDKit writes the sidecar's benzene ring Kekulé and GOT writes it aromatic. Both
# name the same ring, so one spelling is scored.
def _molecule(text):
    return re.sub(r"</?smiles>", "", text).replace("C1=CC=CC=C1", "c1ccccc1")


def format_reference(asset, sidecar):
    """Reduce a ground-truth sidecar to the content a reader would check."""
    if asset == "table":
        table = json.loads(sidecar)
        return "".join("".join(map(str, row)) for row in [table["headers"], *table["rows"]])
    if asset == "chart":
        chart = json.loads(sidecar)
        pairs = zip(chart["labels"], chart["values"])
        return chart["title"] + "".join(f"{k}{v}" for k, v in pairs)
    if asset == "molecule":
        return _molecule(sidecar)
    return sidecar


def format_hypothesis(asset, reading):
    """Reduce a frozen reading the same way, keeping everything the model said."""
    if asset == "table":
        return _table_cells(reading)
    if asset == "chart":
        chart = json.loads(reading)
        return chart["title"] + "".join(f"{k}{v}" for k, v in chart["values"].items())
    if asset == "molecule":
        return _molecule(reading)
    return reading


def format_cer(asset, reading, sidecar):
    hyp = _squeeze(format_hypothesis(asset, reading))
    ref = _squeeze(format_reference(asset, sidecar))
    return round(edit_distance(list(hyp), list(ref)) / len(ref), 4)


def format_corpus(root=FORMAT_CORPUS, scores=None):
    """Score every frozen reading. Returns the disagreements with `scores`."""
    scores = FORMAT_SCORES if scores is None else scores
    failures = []
    for asset, (int8, f32, verdict) in scores.items():
        base = asset.split(".")[0]
        sidecar = next(
            name for name in sorted(os.listdir(root))
            if name.startswith(f"{base}.gt.")
        )
        truth = _read(os.path.join(root, sidecar))
        for arm, recorded, suffix in (("int8", int8, "got"), ("f32", f32, "f32.got")):
            reading = _read(os.path.join(root, f"{asset}.{suffix}.txt"))
            cer = format_cer(base, reading, truth)
            print(json.dumps({"asset": asset, "arm": arm, "cer": cer, "verdict": verdict}))
            if cer != recorded:
                failures.append(f"{asset} {arm}: scored {cer}, recorded {recorded}")
    return failures


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def selftest():
    # CER("abc","abd") == 1/3 ; identical strings -> 0.0
    r = score("abc", "abd", raw=True)
    assert r["edit_distance"] == 1, r
    assert abs(r["cer"] - (1.0 / 3.0)) < 1e-9, r

    r = score("hello world", "hello world", raw=True)
    assert r["edit_distance"] == 0 and r["cer"] == 0.0, r
    assert r["word_edit_distance"] == 0 and r["wer"] == 0.0, r

    # edit_distance sanity
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], list("abc")) == 3
    assert edit_distance(list("abc"), []) == 3
    assert edit_distance(list("abc"), list("abc")) == 0

    # Normalization: hyphenation join.
    assert normalize("pene-\ntrate") == "penetrate"
    # Normalization: whitespace collapse + lowercase.
    assert normalize("  Hello   WORLD \n\t foo ") == "hello world foo"
    # Normalization: curly quotes + em dash.
    assert normalize("“Hi”—there") == '"hi"-there'
    # Normalization: running header stripped.
    assert normalize("42   THE HISTORY OF ROME   [1848]\nThe real body text.") == \
        "the real body text."
    # Normalization: trailing standalone page number dropped.
    assert normalize("Some body text here.\n107") == "some body text here."
    # A pure all-caps single line is NOT stripped (no following body).
    assert normalize("THE HISTORY OF ROME") == "the history of rome"

    # Post-normalization equivalence yields CER 0.
    r = score("pene-\ntrate the WALL", "penetrate the wall")
    assert r["cer"] == 0.0, r

    # The `--format` reductions: both spellings of a ring, a tabular, a chart.
    assert format_cer("molecule", "<smiles>Oc1ccccc1</smiles>", "OC1=CC=CC=C1\n") == 0.0
    table = '{"headers": ["Item", "Qty"], "rows": [["Nut", 4]]}'
    latex = "\\begin{tabular}{|c|c|}\n\\hline Item & Qty \\\\\n\\hline Nut & 4 \\\\\n\\end{tabular}"
    assert format_cer("table", latex, table) == 0.0
    chart = '{"title": "T", "labels": ["A", "B"], "values": [3, 7]}'
    assert format_cer("chart", '{"title": "T", "values": {"A": "3", "B": "7"}}', chart) == 0.0
    assert format_cer("chart", '{"title": "T", "values": {"A": "3", "B": "5"}}', chart) == 0.2

    # A frozen reading that moves either way fails against its recorded score.
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in (
            ("music.gt.krn", "4c 4d\n"), ("music.got.txt", "4c 4d\n"),
            ("music.f32.got.txt", "4c 4e\n"),
        ):
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        with contextlib.redirect_stdout(io.StringIO()):
            assert format_corpus(tmp, {"music": (0.0, 0.25, "x")}) == []
            assert len(format_corpus(tmp, {"music": (0.25, 0.25, "x")})) == 1
            assert len(format_corpus(tmp, {"music": (0.0, 0.0, "x")})) == 1

    print("selftest OK")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def main(argv):
    args = list(argv)
    if "--selftest" in args:
        return selftest()
    if "--format-corpus" in args:
        failures = format_corpus()
        for failure in failures:
            sys.stderr.write(f"got_cer.py: {failure}\n")
        return 1 if failures else 0

    raw = False
    if "--raw" in args:
        raw = True
        args.remove("--raw")

    if len(args) != 2:
        sys.stderr.write(
            "usage: got_cer.py [--raw] <hypothesis.txt> <ground_truth.txt>\n"
            "       got_cer.py --format-corpus\n"
            "       got_cer.py --selftest\n"
        )
        return 2

    hyp = _read(args[0])
    gt = _read(args[1])
    result = score(hyp, gt, raw=raw)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
