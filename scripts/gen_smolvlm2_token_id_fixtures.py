#!/usr/bin/env python3
"""Generate SmolVLM2 (SmolLM2) token-id-exact golden fixtures (C6, bd-3jo6.3.6).

OFFLINE TOOLING ONLY. Loads NOTHING from model.safetensors and imports no torch.
Loads the pinned SmolVLM2-500M-Video-Instruct ``tokenizer.json`` (the
GPT2TokenizerFast serialization, SHA-256 pinned below) via the HF ``tokenizers``
library — the EXACT Rust crate the fast tokenizer wraps, so the ids are the
reference ids — encodes the committed conformance corpus with
``add_special_tokens=False`` (the post_processor is null, so True/False are
identical — spec §7; pinned False for symmetry with the Baidu generator),
decodes the ids back with ``skip_special_tokens=False``, and freezes
``{text, ids, decoded}`` records. The pure-Rust byte-level BPE
(``src/tokenizer/mod.rs``, ``PretokScheme::SmolLm2``) is held token-id-EXACT
against this — the L0a prerequisite for every downstream SmolVLM2 rung
(``docs/zoo/smolvlm2-spec.md`` §13).

Corpus: ``tests/fixtures/tokenizer_smolvlm2/corpus.txt`` (committed; one
JSON-encoded string per line). It stresses the two scheme deltas vs the Baidu
path — ``Digits(individual_digits=true)`` (dates, decimals, non-ASCII \\p{N})
and the GPT-2 word regex (contractions, no-\\p{M} letters, ``use_regex=true``
whitespace backtracking) — plus the SmolVLM2 specials and the exact §5
image-expansion strings whose ``"\\n"``/``"\\n\\n"`` BPE merges OQ-4 pins by
fixture.

Usage:
    # venv with:  pip install 'tokenizers>=0.15'   (the smolvlm2 oracle venv works)
    /private/tmp/smolvlm2_oracle_venv/bin/python scripts/gen_smolvlm2_token_id_fixtures.py

Reads ``$FOCR_SMOLVLM2_TOKENIZER_JSON``, else ``$FOCR_SMOLVLM2_DIR/tokenizer.json``,
else the default zoo mirror. Writes the committed golden at
``tests/fixtures/tokenizer_smolvlm2/expected.json``. The Rust conformance gate
``tokenizer::tests::smolvlm2_token_id_conformance_gate`` ``include_str!``s that
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
_default_dir = os.environ.get(
    "FOCR_SMOLVLM2_DIR", "/Volumes/USBNVME16TB/temp_agent_space/zoo/smolvlm2"
)
TOKENIZER_JSON = Path(
    os.environ.get("FOCR_SMOLVLM2_TOKENIZER_JSON", str(Path(_default_dir) / "tokenizer.json"))
)
CORPUS = Path(
    os.environ.get(
        "FOCR_SMOLVLM2_CORPUS", REPO_ROOT / "tests/fixtures/tokenizer_smolvlm2/corpus.txt"
    )
)
OUT = Path(
    os.environ.get(
        "FOCR_SMOLVLM2_FIXTURES_OUT",
        REPO_ROOT / "tests/fixtures/tokenizer_smolvlm2/expected.json",
    )
)

# SHA-256 pin of the reference tokenizer.json (HuggingFaceTB/
# SmolVLM2-500M-Video-Instruct, verified 2026-07-02). Fixtures generated
# against a different serialization are NOT comparable.
PINNED_TOKENIZER_SHA256 = "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a"

# Anchor ids pinned by docs/zoo/smolvlm2-spec.md §5 (src/tokenizer/mod.rs
# `special_smollm2`) — encode(surface) must be exactly [id] for each.
ANCHORS: "dict[str, int]" = {
    "<|endoftext|>": 0,
    "<|im_start|>": 1,
    "<|im_end|>": 2,
    "<global-img>": 49152,
    "<row_1_col_1>": 49153,
    "<row_6_col_6>": 49188,
    "<fake_token_around_image>": 49189,
    "<image>": 49190,
    "<end_of_utterance>": 49279,
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
            f"Set FOCR_SMOLVLM2_TOKENIZER_JSON or FOCR_SMOLVLM2_DIR."
        )
    tj_sha = sha256_of(TOKENIZER_JSON)
    if tj_sha != PINNED_TOKENIZER_SHA256:
        raise SystemExit(
            f"ERROR: tokenizer.json SHA-256 drift.\n"
            f"  expected: {PINNED_TOKENIZER_SHA256}\n"
            f"  actual:   {tj_sha}\n"
            f"Re-pin deliberately (update this script + docs/zoo/smolvlm2-spec.md) or fix the file."
        )

    tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    # Anchor sanity: every runtime-pinned special must encode to its one id.
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
            "purpose": "SmolVLM2 (SmolLM2) tokenizer token-id-EXACT conformance golden fixtures (C6, bd-3jo6.3.6)",
            "model_id": "smolvlm2",
            "tokenizer_class": "GPT2TokenizerFast (byte-level BPE tokenizer.json; pre_tokenizer = Digits(individual)+ByteLevel(use_regex))",
            "tokenizer_json_sha256": tj_sha,
            "backend": "tokenizers.Tokenizer.from_file (the exact Rust crate the fast tokenizer wraps)",
            "tokenizers_version": tokenizers.__version__,
            "transformers_version": None,
            "torch_loaded": False,
            "corpus": "tests/fixtures/tokenizer_smolvlm2/corpus.txt",
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "bos_id": 1,
            "eos_id": 49279,
            "pad_id": 2,
            "image_token_id": 49190,
            "anchor_ids": ANCHORS,
            "num_cases": len(fixtures),
            "num_exact_roundtrip": n_roundtrip,
            "note": (
                "ids use add_special_tokens=False (post_processor is null, so "
                "this equals True — spec §7; the prompt builder owns the "
                "template framing). decoded is the reference decode of exactly "
                "`ids` with skip_special_tokens=False; it may differ from "
                "`text` only where the reference itself is not surface-exact."
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
