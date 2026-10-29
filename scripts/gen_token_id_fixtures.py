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

    # Unlimited-OCR-2 (U2-A4), in a venv pinned to the v2 reference stack
    # (tokenizers==0.21.4), over the v2 tokenizer.json:
    FOCR_BAIDU_MODEL=unlimited-ocr-2 FOCR_TOKENIZER_JSON=/path/to/v2/tokenizer.json \
        python3 scripts/gen_token_id_fixtures.py

``FOCR_BAIDU_MODEL`` selects the profile (default ``unlimited-ocr``): its
tokenizer pin, anchors, corpora and output. For v1 it reads
``docs/truth-pack/snapshots/tokenizer.json`` by default (9.9 MB, gitignored —
fetched out-of-band by scripts/fetch_sources.sh; override with
FOCR_TOKENIZER_JSON) and writes ``tests/fixtures/tokenizer_baidu/expected.json``.
v2 requires FOCR_TOKENIZER_JSON, encodes ``corpus.txt`` followed by
``corpus_uocr2.txt``, and writes ``tests/fixtures/tokenizer_baidu/expected_uocr2.json``.
The Rust gates ``tokenizer::tests::baidu_token_id_conformance_gate`` and
``tokenizer::tests::uocr2_token_id_conformance_gate`` ``include_str!`` those
files and assert our encoder/decoder reproduces every record exactly.
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

# Anchor ids the model runtime hardcodes (src/tokenizer/mod.rs `special`,
# [SPEC-014/019]) — encode(surface) must be exactly [id] for each.
V1_ANCHORS: "dict[str, int]" = {
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

# Unlimited-OCR-2's four added tokens: all special=false, so they survive
# skip_special_tokens=True. Only <PAGE> is used by inference.
V2_NEW_TOKENS: "dict[str, int]" = {
    "<PAGE>": 128827,
    "<BLOCK>": 128828,
    "<FORK>": 128829,
    "<CHILD>": 128830,
}

V1_CORPUS = "tests/fixtures/tokenizer/corpus.txt"

# Per-model profiles. `sha256` pins the reference tokenizer.json
# (docs/truth-pack/oq/tokenizer.md); fixtures generated against a different
# serialization are NOT comparable.
PROFILES: "dict[str, dict]" = {
    "unlimited-ocr": {
        "sha256": "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4",
        "purpose": "Baidu Unlimited-OCR tokenizer token-id-EXACT conformance golden fixtures (OQ-16)",
        "tokenizer_json": "docs/truth-pack/snapshots/tokenizer.json",
        "anchors": V1_ANCHORS,
        "non_special": {},
        "max_token_id": 128826,
        "corpora": [V1_CORPUS],
        "out": "tests/fixtures/tokenizer_baidu/expected.json",
    },
    "unlimited-ocr-2": {
        "sha256": "c55c165e32c5a67830362a53e1f548c8930dcc8469efa30ff3252a982799c589",
        "purpose": "Unlimited-OCR-2 tokenizer token-id-EXACT conformance golden fixtures (U2-A4, OQ-V2-8)",
        "tokenizer_json": None,
        "anchors": {**V1_ANCHORS, **V2_NEW_TOKENS},
        "non_special": V2_NEW_TOKENS,
        "max_token_id": 128830,
        "corpora": [V1_CORPUS, "tests/fixtures/tokenizer/corpus_uocr2.txt"],
        "out": "tests/fixtures/tokenizer_baidu/expected_uocr2.json",
    },
}

MODEL_ID = os.environ.get("FOCR_BAIDU_MODEL", "unlimited-ocr")
if MODEL_ID not in PROFILES:
    raise SystemExit(f"ERROR: FOCR_BAIDU_MODEL={MODEL_ID!r}; known: {sorted(PROFILES)}")
PROFILE = PROFILES[MODEL_ID]
PINNED_TOKENIZER_SHA256 = PROFILE["sha256"]
ANCHORS: "dict[str, int]" = PROFILE["anchors"]

_tokenizer_json = os.environ.get("FOCR_TOKENIZER_JSON") or PROFILE["tokenizer_json"]
if _tokenizer_json is None:
    raise SystemExit(f"ERROR: set FOCR_TOKENIZER_JSON to the pinned {MODEL_ID} tokenizer.json")
TOKENIZER_JSON = REPO_ROOT / _tokenizer_json
CORPORA = (
    [os.environ["FOCR_BAIDU_CORPUS"]] if "FOCR_BAIDU_CORPUS" in os.environ else PROFILE["corpora"]
)
OUT = REPO_ROOT / os.environ.get("FOCR_BAIDU_FIXTURES_OUT", PROFILE["out"])


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

    # Added-token flags and the id ceiling come from the file itself; a profile
    # that disagrees with them is stale.
    added = tok.get_added_tokens_decoder()
    for surface, want in PROFILE["non_special"].items():
        token = added.get(want)
        if token is None or token.content != surface or token.special:
            raise SystemExit(f"ADDED-TOKEN FAIL: id {want} is {token!r}, want non-special {surface!r}")
        if tok.decode([want], skip_special_tokens=True) != surface:
            raise SystemExit(f"SKIP-SPECIAL FAIL: {surface!r} dropped by skip_special_tokens=True")
    if max(added) != PROFILE["max_token_id"]:
        raise SystemExit(f"MAX-ID FAIL: max added id {max(added)}, want {PROFILE['max_token_id']}")

    cases = [case for corpus in CORPORA for case in load_corpus(REPO_ROOT / corpus)]
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
        record = {"text": text, "ids": ids, "decoded": decoded}
        if PROFILE["non_special"]:
            # The non-special added tokens must survive the skip path too.
            record["decoded_skip_special"] = tok.decode(ids, skip_special_tokens=True)
        fixtures.append(record)

    out_obj = {
        "_meta": {
            "purpose": PROFILE["purpose"],
            "model_id": MODEL_ID,
            "tokenizer_class": "LlamaTokenizerFast (byte-level BPE tokenizer.json, NOT SentencePiece)",
            "tokenizer_json_sha256": tj_sha,
            "backend": "tokenizers.Tokenizer.from_file (the exact Rust crate LlamaTokenizerFast wraps)",
            "tokenizers_version": tokenizers.__version__,
            "transformers_version": None,
            "torch_loaded": False,
            "corpus": " + ".join(str(corpus) for corpus in CORPORA),
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "bos_id": 0,
            "eos_id": 1,
            "pad_id": 2,
            "image_token_id": 128815,
            "anchor_ids": ANCHORS,
            **(
                {
                    "non_special_added_ids": PROFILE["non_special"],
                    "max_token_id": PROFILE["max_token_id"],
                }
                if PROFILE["non_special"]
                else {}
            ),
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
