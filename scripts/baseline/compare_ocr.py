#!/usr/bin/env python3
"""Score franken_ocr output against the baidu reference (the parity verdict).

Compares either two directories of per-page markdown (`page_XXXX.md`) or one
exact reference/hypothesis file pair:
  --ref  : baidu/Unlimited-OCR reference  (run_baidu_reference.py output)
  --hyp  : franken_ocr native output      (`focr ocr page.png`)

Reports per-page and aggregate Character Error Rate (CER, Levenshtein/len(ref))
and exact-match, with both raw and whitespace-normalized variants. CER is the
standard OCR quality metric; for a faithful port we expect aggregate CER well
under a small threshold (and ideally exact text on clean pages).

The JSON output is a versioned provenance receipt. It binds source and scored
UTF-8 bytes, the named reference transform and normalization, and the exact
integer edit-distance numerators/denominators used to derive CER.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


RECEIPT_SCHEMA = "focr-ocr-comparison/v1"
NORMALIZATION = "collapse-unicode-whitespace-v1"
IDENTITY_TRANSFORM = "identity-v1"
STRIP_DET_TRANSFORM = "strip-unlimited-det-spans-v1"
REFERENCE_TRANSFORMS = (IDENTITY_TRANSFORM, STRIP_DET_TRANSFORM)
DET_SPAN_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
DET_TOKEN_RE = re.compile(r"<\|/?det\|>")


def levenshtein(a: str, b: str) -> int:
    """Exact edit distance via Myers bit-vectors over Python arbitrary-width ints.

    This is the same unit-cost recurrence as the former two-row dynamic program,
    but advances every pattern column in parallel. Long OCR pages otherwise
    require hundreds of millions of interpreted inner-loop iterations.
    """
    if a == b:
        return 0
    # The shorter string is the bit-vector pattern, minimizing bigint width.
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    width = len(b)
    mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    char_masks = {}
    for index, char in enumerate(b):
        char_masks[char] = char_masks.get(char, 0) | (1 << index)

    positive = mask
    negative = 0
    distance = width
    for char in a:
        equal = char_masks.get(char, 0)
        vertical = equal | negative
        horizontal = ((((equal & positive) + positive) ^ positive) | equal) & mask
        positive_horizontal = (negative | ~(horizontal | positive)) & mask
        negative_horizontal = positive & horizontal
        if positive_horizontal & high_bit:
            distance += 1
        elif negative_horizontal & high_bit:
            distance -= 1
        positive_horizontal = ((positive_horizontal << 1) | 1) & mask
        negative_horizontal = (negative_horizontal << 1) & mask
        positive = (negative_horizontal | ~(vertical | positive_horizontal)) & mask
        negative = (positive_horizontal & vertical) & mask
    return distance


def norm_ws(s: str) -> str:
    """Collapse runs of whitespace and strip — tolerates layout-only differences."""
    return re.sub(r"\s+", " ", s).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_binding(
    text: str, raw: bytes | None = None, basename: str | None = None
) -> dict:
    """Bind both the exact UTF-8 representation and its named normalization."""
    encoded = text.encode("utf-8") if raw is None else raw
    if encoded.decode("utf-8") != text:
        raise ValueError("text binding bytes do not decode to the supplied UTF-8 text")
    normalized = norm_ws(text)
    normalized_bytes = normalized.encode("utf-8")
    binding = {
        "sha256": sha256_bytes(encoded),
        "bytes": len(encoded),
        "chars": len(text),
        "normalized_sha256": sha256_bytes(normalized_bytes),
        "normalized_bytes": len(normalized_bytes),
        "normalized_chars": len(normalized),
    }
    if basename is not None:
        binding = {"basename": basename, **binding}
    return binding


def transform_reference(text: str, transform: str) -> tuple[str, int]:
    if transform == IDENTITY_TRANSFORM:
        return text, 0
    if transform != STRIP_DET_TRANSFORM:
        raise ValueError(f"unsupported reference transform: {transform}")
    transformed, matches = DET_SPAN_RE.subn("", text)
    if DET_TOKEN_RE.search(transformed):
        raise ValueError("reference contains an unbalanced or nested det span")
    return transformed, matches


def read_utf8(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    try:
        return raw, raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not valid UTF-8: {error}") from error


def score_pair(
    page: str,
    reference_path: Path,
    hypothesis_path: Path | None,
    reference_transform: str,
) -> dict:
    reference_raw, reference_source = read_utf8(reference_path)
    scored_reference, transform_matches = transform_reference(
        reference_source, reference_transform
    )
    normalized_reference = norm_ws(scored_reference)
    reference = {
        "source": text_binding(
            reference_source, reference_raw, basename=reference_path.name
        ),
        "transform": {
            "name": reference_transform,
            "matches": transform_matches,
        },
        "scored": text_binding(scored_reference),
    }

    if hypothesis_path is None:
        raw_edit_distance = len(scored_reference)
        normalized_edit_distance = len(normalized_reference)
        return {
            "page": page,
            "status": "MISSING_HYP",
            "reference": reference,
            "hypothesis": None,
            "raw_edit_distance": raw_edit_distance,
            "normalized_edit_distance": normalized_edit_distance,
            "ref_chars": len(scored_reference),
            "cer_raw": raw_edit_distance / max(1, len(scored_reference)),
            "cer_norm": normalized_edit_distance / max(1, len(normalized_reference)),
            "exact": False,
            "exact_norm": False,
        }

    hypothesis_raw, hypothesis_source = read_utf8(hypothesis_path)
    normalized_hypothesis = norm_ws(hypothesis_source)
    raw_edit_distance = levenshtein(scored_reference, hypothesis_source)
    normalized_edit_distance = levenshtein(
        normalized_reference, normalized_hypothesis
    )
    return {
        "page": page,
        "status": "OK",
        "reference": reference,
        "hypothesis": {
            "source": text_binding(
                hypothesis_source, hypothesis_raw, basename=hypothesis_path.name
            ),
            "transform": {"name": IDENTITY_TRANSFORM, "matches": 0},
            "scored": text_binding(hypothesis_source),
        },
        "raw_edit_distance": raw_edit_distance,
        "normalized_edit_distance": normalized_edit_distance,
        # Retain the compact display fields while the integer distances and
        # bound counts above remain the authoritative evidence.
        "ref_chars": len(scored_reference),
        "hyp_chars": len(hypothesis_source),
        "cer_raw": raw_edit_distance / max(1, len(scored_reference)),
        "cer_norm": normalized_edit_distance / max(1, len(normalized_reference)),
        "exact": scored_reference == hypothesis_source,
        "exact_norm": normalized_reference == normalized_hypothesis,
    }


def receipt_from_rows(rows: list[dict]) -> dict:
    pages_with_hyp = [row for row in rows if row["status"] == "OK"]
    raw_edit_distance = sum(row["raw_edit_distance"] for row in rows)
    normalized_edit_distance = sum(row["normalized_edit_distance"] for row in rows)
    raw_reference_chars = sum(row["reference"]["scored"]["chars"] for row in rows)
    normalized_reference_chars = sum(
        row["reference"]["scored"]["normalized_chars"] for row in rows
    )
    aggregate = {
        "pages_total": len(rows),
        "pages_with_hyp": len(pages_with_hyp),
        "exact_raw": sum(bool(row["exact"]) for row in rows),
        "exact_norm": sum(bool(row["exact_norm"]) for row in rows),
        "raw_edit_distance": raw_edit_distance,
        "raw_reference_chars": raw_reference_chars,
        "normalized_edit_distance": normalized_edit_distance,
        "normalized_reference_chars": normalized_reference_chars,
        "reference_source_bytes": sum(
            row["reference"]["source"]["bytes"] for row in rows
        ),
        "reference_source_chars": sum(
            row["reference"]["source"]["chars"] for row in rows
        ),
        "hypothesis_source_bytes": sum(
            row["hypothesis"]["source"]["bytes"]
            for row in pages_with_hyp
        ),
        "hypothesis_source_chars": sum(
            row["hypothesis"]["source"]["chars"]
            for row in pages_with_hyp
        ),
        "cer_raw": raw_edit_distance / max(1, raw_reference_chars),
        "cer_norm": normalized_edit_distance / max(1, normalized_reference_chars),
    }
    return {
        "schema": RECEIPT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "normalization": NORMALIZATION,
        "metric_formulas": {
            "cer_raw": "raw_edit_distance / max(1, raw_reference_chars)",
            "cer_norm": (
                "normalized_edit_distance / max(1, normalized_reference_chars)"
            ),
        },
        "aggregate": aggregate,
        "pages": rows,
    }


def self_test() -> None:
    """Differentially prove the bit-vector kernel against the DP recurrence."""
    import itertools
    import random

    def reference(a: str, b: str) -> int:
        if not a:
            return len(b)
        if not b:
            return len(a)
        previous = list(range(len(b) + 1))
        for row, left in enumerate(a, 1):
            current = [row] + [0] * len(b)
            for column, right in enumerate(b, 1):
                current[column] = min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left != right),
                )
            previous = current
        return previous[-1]

    strings = [
        "".join(chars)
        for length in range(7)
        for chars in itertools.product("ab", repeat=length)
    ]
    for left in strings:
        for right in strings:
            assert levenshtein(left, right) == reference(left, right)

    rng = random.Random(0)
    alphabet = "abc xyzΩ"
    for _ in range(10_000):
        left = "".join(rng.choice(alphabet) for _ in range(rng.randrange(40)))
        right = "".join(rng.choice(alphabet) for _ in range(rng.randrange(40)))
        assert levenshtein(left, right) == reference(left, right)

    source = "A <|det|>bbox\nvalue<|/det|>  Ω\n"
    scored, matches = transform_reference(source, STRIP_DET_TRANSFORM)
    assert scored == "A   Ω\n"
    assert matches == 1
    source_binding = text_binding(source, source.encode("utf-8"), "reference.md")
    scored_binding = text_binding(scored)
    assert source_binding["basename"] == "reference.md"
    assert source_binding["bytes"] > source_binding["chars"]
    assert source_binding["normalized_chars"] == len(norm_ws(source))
    assert scored_binding["normalized_sha256"] == sha256_bytes(b"A \xce\xa9")
    hypothesis = "A Ω"
    row = {
        "page": "reference.md",
        "status": "OK",
        "reference": {
            "source": source_binding,
            "transform": {"name": STRIP_DET_TRANSFORM, "matches": matches},
            "scored": scored_binding,
        },
        "hypothesis": {
            "source": text_binding(
                hypothesis, hypothesis.encode("utf-8"), "run_001.stdout"
            ),
            "transform": {"name": IDENTITY_TRANSFORM, "matches": 0},
            "scored": text_binding(hypothesis),
        },
        "raw_edit_distance": levenshtein(scored, hypothesis),
        "normalized_edit_distance": levenshtein(norm_ws(scored), norm_ws(hypothesis)),
        "exact": scored == hypothesis,
        "exact_norm": norm_ws(scored) == norm_ws(hypothesis),
    }
    receipt = receipt_from_rows([row])
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["normalization"] == NORMALIZATION
    assert receipt["aggregate"]["raw_edit_distance"] == row["raw_edit_distance"]
    assert receipt["aggregate"]["normalized_edit_distance"] == 0
    assert receipt["aggregate"]["cer_norm"] == 0.0
    try:
        transform_reference("<|det|>unclosed", STRIP_DET_TRANSFORM)
    except ValueError:
        pass
    else:
        raise AssertionError("unbalanced det spans must fail closed")
    print(json.dumps({
        "check": "compare-ocr-levenshtein-differential",
        "receipt_schema": RECEIPT_SCHEMA,
        "exhaustive_pairs": len(strings) ** 2,
        "random_pairs": 10_000,
        "result": "pass",
    }))


def sha256_arg(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("expected 64 lowercase hexadecimal characters")
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref")
    ap.add_argument("--hyp")
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--reference-transform",
        choices=REFERENCE_TRANSFORMS,
        default=IDENTITY_TRANSFORM,
    )
    ap.add_argument("--expected-reference-sha256", type=sha256_arg)
    ap.add_argument("--expected-hypothesis-sha256", type=sha256_arg)
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.ref or not args.hyp:
        ap.error("--ref and --hyp are required unless --self-test is used")

    reference, hypothesis = Path(args.ref), Path(args.hyp)
    pairs: list[tuple[str, Path, Path | None]]
    if reference.is_file() and hypothesis.is_file():
        pairs = [(reference.name, reference, hypothesis)]
    elif reference.is_dir() and hypothesis.is_dir():
        ref_pages = sorted(reference.glob("page_*.md"))
        if not ref_pages:
            print(f"no reference pages in {reference}", file=sys.stderr)
            return 2
        pairs = [
            (rp.name, rp, hypothesis / rp.name if (hypothesis / rp.name).is_file() else None)
            for rp in ref_pages
        ]
    else:
        print(
            "--ref and --hyp must both be regular files or both be directories",
            file=sys.stderr,
        )
        return 2

    if (args.expected_reference_sha256 or args.expected_hypothesis_sha256) and len(
        pairs
    ) != 1:
        print("expected input hashes require a single-file comparison", file=sys.stderr)
        return 2

    try:
        rows = [
            score_pair(page, rp, hp, args.reference_transform)
            for page, rp, hp in pairs
        ]
    except (OSError, ValueError) as error:
        print(f"comparison input error: {error}", file=sys.stderr)
        return 2

    if args.expected_reference_sha256:
        actual = rows[0]["reference"]["source"]["sha256"]
        if actual != args.expected_reference_sha256:
            print(
                f"reference sha256 {actual} != expected {args.expected_reference_sha256}",
                file=sys.stderr,
            )
            return 2
    if args.expected_hypothesis_sha256:
        hypothesis_binding = rows[0].get("hypothesis")
        actual = hypothesis_binding and hypothesis_binding["source"]["sha256"]
        if actual != args.expected_hypothesis_sha256:
            print(
                f"hypothesis sha256 {actual} != expected {args.expected_hypothesis_sha256}",
                file=sys.stderr,
            )
            return 2

    receipt = receipt_from_rows(rows)
    agg = receipt["aggregate"]

    print(f"{'page':16} {'status':12} {'ref':>6} {'hyp':>6} {'CER_raw':>8} {'CER_norm':>8} {'exact':>6}")
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['page']:16} {r['status']:12} {r['ref_chars']:>6}")
            continue
        print(f"{r['page']:16} {r['status']:12} {r['ref_chars']:>6} {r['hyp_chars']:>6} "
              f"{r['cer_raw']:>8.4f} {r['cer_norm']:>8.4f} {str(r['exact']):>6}")
    print("-" * 72)
    print(f"AGGREGATE: pages={agg['pages_total']} with_hyp={agg['pages_with_hyp']} "
          f"exact={agg['exact_raw']} exact_norm={agg['exact_norm']} "
          f"CER_raw={agg['cer_raw']:.4f} CER_norm={agg['cer_norm']:.4f}")

    if args.require_complete:
        if agg["pages_with_hyp"] != agg["pages_total"]:
            print(
                "comparison is incomplete: one or more hypotheses are missing",
                file=sys.stderr,
            )
            return 2
        if any(
            row["reference"]["scored"]["chars"] == 0
            or row["reference"]["scored"]["normalized_chars"] == 0
            for row in rows
        ):
            print("comparison reference is empty after transformation", file=sys.stderr)
            return 2
    if args.json:
        Path(args.json).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
