#!/usr/bin/env python3
"""Generate Baidu Unlimited-OCR token-id-exact golden fixtures (OQ-16, bd-re8.8).

OFFLINE TOOLING ONLY. Loads NOTHING from model.safetensors and imports no torch.
Loads the pinned reference ``tokenizer.json`` (the LlamaTokenizerFast
serialization, SHA-256 pinned below) via the HF ``tokenizers`` library — the
EXACT Rust crate ``LlamaTokenizerFast`` wraps, so the ids are the reference ids
— encodes the committed conformance corpus with ``add_special_tokens=False``
(the inference path: ``modeling_unlimitedocr.py:259-268`` hardcodes BOS=0/EOS=1
itself), decodes the ids back with ``skip_special_tokens=False``, and freezes
``{text, ids, decoded}`` records. The pure-Rust byte-level BPE
(``src/tokenizer/mod.rs``) is held token-id-EXACT against this — the L0/L4
prerequisite parity gate for every downstream rung (AGENTS.md doctrine).

Corpus: ``tests/fixtures/tokenizer/corpus.txt`` (committed; one JSON-encoded
string per line — see the README there). It exercises ASCII prose, edge
whitespace/CR/LF, digit-grouping (the ``\\p{N}{1,3}`` pre-tokenizer stage),
math/LaTeX, code, CJK (Chinese/Japanese isolated by stage 2, Korean outside it),
RTL scripts, emoji/ZWJ, combining marks, the DeepSeek fullwidth-bar glyph
specials, the ASCII-pipe specials, all OCR/grounding/table glyphs, adjacency
stress, and realistic mixed-script OCR snippets.

Usage:
    # venv with:  pip install 'tokenizers>=0.15'
    python3 scripts/gen_token_id_fixtures.py

Reads ``docs/truth-pack/snapshots/tokenizer.json`` by default (9.9 MB,
gitignored — fetched out-of-band by scripts/fetch_sources.sh; override with
FOCR_TOKENIZER_JSON). Writes the committed golden at
``tests/fixtures/tokenizer_baidu/expected.json``. The Rust conformance gate
``tokenizer::tests::baidu_token_id_conformance_gate`` ``include_str!``s that
file and asserts our encoder/decoder reproduces every record exactly.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from tokenizers import Tokenizer
import tokenizers

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_JSON = Path(
    os.environ.get(
        "FOCR_TOKENIZER_JSON", REPO_ROOT / "docs/truth-pack/snapshots/tokenizer.json"
    )
)
CORPUS = Path(
    os.environ.get("FOCR_BAIDU_CORPUS", REPO_ROOT / "tests/fixtures/tokenizer/corpus.txt")
)
OUT = Path(
    os.environ.get(
        "FOCR_BAIDU_FIXTURES_OUT", REPO_ROOT / "tests/fixtures/tokenizer_baidu/expected.json"
    )
)

# SHA-256 pin of the reference tokenizer.json (docs/truth-pack/oq/tokenizer.md).
# Fixtures generated against a different serialization are NOT comparable.
PINNED_TOKENIZER_SHA256 = "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4"

# Anchor ids the model runtime hardcodes (src/tokenizer/mod.rs `special`,
# [SPEC-014/019]) — encode(surface) must be exactly [id] for each.
ANCHORS: "dict[str, int]" = {
    "<｜begin▁of▁sentence｜>": 0,
    "<｜end▁of▁sentence｜>": 1,
    "<｜▁pad▁｜>": 2,
    "<image>": 128815,
    "<|ref|>": 128816,
    "<|/ref|>": 128817,
    "<|det|>": 128818,
    "<|/det|>": 128819,
    "<|grounding|>": 128820,
    "<td>": 128821,
    "</td>": 128822,
    "<tr>": 128823,
    "</tr>": 128824,
    "<|User|>": 128825,
    "<|Assistant|>": 128826,
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_corpus(path: Path) -> "list[str]":
    """One JSON-encoded string per line; blank / raw-``#`` lines are comments."""
    cases: "list[str]" = []
    for lineno, raw in enumerate(path.open("r", encoding="utf-8"), start=1):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, str):
            raise SystemExit(f"ERROR: {path}:{lineno}: expected a JSON string literal")
        cases.append(value)
    return cases


def main() -> int:
    if not TOKENIZER_JSON.is_file():
        raise SystemExit(
            f"ERROR: tokenizer.json not found: {TOKENIZER_JSON}\n"
            f"Fetch it out-of-band (scripts/fetch_sources.sh) or set FOCR_TOKENIZER_JSON."
        )
    tj_sha = sha256_of(TOKENIZER_JSON)
    if tj_sha != PINNED_TOKENIZER_SHA256:
        raise SystemExit(
            f"ERROR: tokenizer.json SHA-256 drift.\n"
            f"  expected: {PINNED_TOKENIZER_SHA256}\n"
            f"  actual:   {tj_sha}\n"
            f"Re-pin deliberately (update this script + docs/truth-pack/oq/tokenizer.md) or fix the file."
        )

    tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    # Anchor sanity: every runtime-hardcoded special must encode to its one id.
    for surface, want in ANCHORS.items():
        got = tok.encode(surface, add_special_tokens=False).ids
        if got != [want]:
            raise SystemExit(f"ANCHOR FAIL: {surface!r} -> {got}, want [{want}]")

    cases = load_corpus(CORPUS)
    if len(cases) < 100:
        raise SystemExit(f"ERROR: corpus too small ({len(cases)} cases; need >= 100)")

    fixtures: "list[dict]" = []
    n_roundtrip = 0
    for text in cases:
        ids = tok.encode(text, add_special_tokens=False).ids
        # skip_special_tokens=False — literal round-trip of exactly these ids
        # (specials legitimately appear when the corpus contains their surfaces).
        decoded = tok.decode(ids, skip_special_tokens=False)
        if decoded == text:
            n_roundtrip += 1
        fixtures.append({"text": text, "ids": ids, "decoded": decoded})

    out_obj = {
        "_meta": {
            "purpose": "Baidu Unlimited-OCR tokenizer token-id-EXACT conformance golden fixtures (OQ-16)",
            "model_id": "unlimited-ocr",
            "tokenizer_class": "LlamaTokenizerFast (byte-level BPE tokenizer.json, NOT SentencePiece)",
            "tokenizer_json_sha256": tj_sha,
            "backend": "tokenizers.Tokenizer.from_file (the exact Rust crate LlamaTokenizerFast wraps)",
            "tokenizers_version": tokenizers.__version__,
            "transformers_version": None,
            "torch_loaded": False,
            "corpus": "tests/fixtures/tokenizer/corpus.txt",
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "bos_id": 0,
            "eos_id": 1,
            "pad_id": 2,
            "image_token_id": 128815,
            "anchor_ids": ANCHORS,
            "num_cases": len(fixtures),
            "num_exact_roundtrip": n_roundtrip,
            "note": (
                "ids use add_special_tokens=False (the inference path; the prompt "
                "builder owns BOS=0). decoded is the reference decode of exactly "
                "`ids` with skip_special_tokens=False; it may differ from `text` "
                "only where the reference itself is not surface-exact."
            ),
        },
        "fixtures": fixtures,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(fixtures)} cases ({n_roundtrip} exact text round-trips) to {OUT}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
