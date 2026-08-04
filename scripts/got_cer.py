#!/usr/bin/env python3
"""GOT-OCR2.0 Character Error Rate (CER) harness — stdlib only, no deps.

Usage:
    python3 scripts/got_cer.py <hypothesis.txt> <ground_truth.txt>
    python3 scripts/got_cer.py --raw <hypothesis.txt> <ground_truth.txt>
    python3 scripts/got_cer.py --selftest

Prints JSON: {"chars_gt", "edit_distance", "cer", "words_gt", "word_edit_distance", "wer"}

Normalization (applied to BOTH hypothesis and ground truth so the comparison is
fair on a scanned historical book), in order:
  1. normalize curly quotes -> straight, en/em dashes -> "-"
  2. strip a leading running-header line (page-number + SMALL-CAPS title + [year])
  3. drop a trailing standalone page number
  4. lowercase
  5. join line-end hyphenation ("pene-\\ntrate" -> "penetrate")
  6. collapse all whitespace runs to a single space (and strip)
"""

import json
import re
import sys


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

    raw = False
    if "--raw" in args:
        raw = True
        args.remove("--raw")

    if len(args) != 2:
        sys.stderr.write(
            "usage: got_cer.py [--raw] <hypothesis.txt> <ground_truth.txt>\n"
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
