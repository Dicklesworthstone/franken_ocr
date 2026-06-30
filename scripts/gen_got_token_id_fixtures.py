#!/usr/bin/env python3
"""Generate GOT-OCR2 token-id-exact golden fixtures from the Qwen tiktoken BPE.

OFFLINE TOOLING ONLY. Loads NOTHING from model.safetensors. Builds the Qwen
``QWenTokenizer`` encoder DIRECTLY as a ``tiktoken.Encoding`` over ``qwen.tiktoken``
(the QWenTokenizer recipe: Qwen PAT_STR + mergeable ranks + dual special sets),
encodes a diverse corpus, and freezes ``{string -> token-ids}``. The pure-Rust
tiktoken-BPE loader (``src/tokenizer/tiktoken.rs``, bead B6/A6) is held
token-id-EXACT against this — it is the L0a prerequisite parity gate for every
GOT-OCR2 downstream rung (AGENTS.md doctrine).

Reference behavior mirrored: ``QWenTokenizer.tokenize()`` default
``allowed_special="all", disallowed_special=()`` -> literal special-token surface
forms (``<|im_end|>``, ``<img>``, ...) become their control ids; and the
``unicodedata.normalize("NFC")`` pre-pass (every corpus string is NFC-stable).

Usage:
    pip install tiktoken
    # point FOCR_GOT_DIR at the dir holding qwen.tiktoken (out-of-band, ~2.4 MB)
    FOCR_GOT_DIR=/path/to/got-ocr2 python3 scripts/gen_got_token_id_fixtures.py

By default it writes the committed golden at
``tests/fixtures/tokenizer_got/expected.json`` (relative to the repo root). The
Rust conformance test ``tokenizer::tiktoken::tests::token_id_conformance_gate``
``include_str!``s that file and asserts our encoder reproduces every id stream.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path

import tiktoken

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("FOCR_GOT_DIR", "/Volumes/USBNVME16TB/temp_agent_space/zoo/got-ocr2"))
TIKTOKEN_FILE = Path(os.environ.get("FOCR_GOT_TIKTOKEN", MODEL_DIR / "qwen.tiktoken"))
OUT = Path(os.environ.get("FOCR_GOT_FIXTURES_OUT", REPO_ROOT / "tests/fixtures/tokenizer_got/expected.json"))

# --- QWenTokenizer recipe (tokenization_qwen.py) ----------------------------
# GPT-2/tiktoken regex but with single-char \p{N} (Qwen splits numerals).
PAT_STR = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
    r"""|[^\r\n\p{L}\p{N}]?\p{L}+"""
    r"""|\p{N}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*"""
    r"""|\s*[\r\n]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
EXTRAS = tuple(f"<|extra_{i}|>" for i in range(205))  # stock Qwen: extra_0..204
SPECIAL_TOKENS = (ENDOFTEXT, IMSTART, IMEND) + EXTRAS  # 208 -> ids 151643..151850
# GOT IMAGE_ST tuple, in the EXACT source order; enumerated contiguously after
# SPECIAL_TOKENS (start=len(mergeable_ranks)=151643), so these land at 151851..151859.
IMAGE_ST = (
    "<ref>", "</ref>",          # 151851, 151852
    "<box>", "</box>",          # 151853, 151854
    "<quad>", "</quad>",        # 151855, 151856
    "<img>", "</img>",          # 151857, 151858
    "<imgpad>",                 # 151859
)
SPECIAL_START_ID = 151643


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mergeable_ranks(path: Path) -> "dict[bytes, int]":
    ranks: "dict[bytes, int]" = {}
    with path.open("rb") as f:
        for line in f:
            line = line.rstrip(b"\n")
            if not line:
                continue
            tok_b64, rank = line.split(b" ")
            ranks[base64.b64decode(tok_b64)] = int(rank)
    return ranks


def build_encoder(ranks: "dict[bytes, int]") -> tiktoken.Encoding:
    # FAITHFUL to tokenization_qwen.py — enumerate(SPECIAL_TOKENS + IMAGE_ST,
    # start=len(mergeable_ranks)). No padding gap: every id 0..151859 is assigned.
    special_tokens: "dict[str, int]" = {
        tok: SPECIAL_START_ID + i
        for i, tok in enumerate(SPECIAL_TOKENS + IMAGE_ST)
    }
    enc = tiktoken.Encoding(
        name="Qwen",
        pat_str=PAT_STR,
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )
    # Mirror the source assert — n_vocab == 151860 (config.json vocab_size).
    assert len(ranks) + len(special_tokens) == enc.n_vocab, "n_vocab mismatch"
    assert enc.n_vocab == 151860, f"n_vocab {enc.n_vocab} != 151860 (config vocab_size)"
    return enc


# Diverse corpus: ascii, whitespace, digits, punctuation, code, CJK, Cyrillic,
# accents, math/symbol, emoji (incl. ZWJ + regional-indicator flag), URL, and
# the load-bearing GOT special-token surface forms / prompt fragments.
CORPUS: "list[str]" = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "OCR: ",
    "OCR with format: ",
    "1234567890",
    "Price: $1,234.56 (USD) -42%",
    "    leading and  multiple   spaces",
    "tab\there\tand\nnewline",
    "Line1\nLine2\nLine3\n",
    "snake_case camelCase PascalCase kebab-case",
    'fn main() { println!("{}", x + 1); }',
    "你好，世界！这是一个测试。",
    "日本語のテスト",
    "café naïve résumé Straße",
    "Здравствуй, мир",
    "emoji: \U0001f600\U0001f680\U0001f1fa\U0001f1f8\U0001f469‍\U0001f4bb",
    "x²+y²=z² ∀x∈ℝ, ∑_{i=1}^{n} i",
    "<|im_end|>",
    "<|im_start|>assistant\n",
    "<img><imgpad></img>\nOCR: ",
    "<ref>text</ref><box>(1,2),(3,4)</box>",
    "<quad>region</quad>",
    "https://example.com/path?q=1&r=2#frag-2",
    "MixedCASE123abc!@#",
]


def main() -> int:
    if not TIKTOKEN_FILE.is_file():
        raise SystemExit(
            f"ERROR: qwen.tiktoken not found: {TIKTOKEN_FILE}\n"
            f"Set FOCR_GOT_DIR or FOCR_GOT_TIKTOKEN to its location."
        )
    tt_sha = sha256_of(TIKTOKEN_FILE)
    ranks = load_mergeable_ranks(TIKTOKEN_FILE)
    enc = build_encoder(ranks)

    # Anchor sanity: control + ALL 9 GOT grounding/image ids (contiguous 151851..151859).
    anchors = {
        ENDOFTEXT: 151643, IMSTART: 151644, IMEND: 151645,
        "<ref>": 151851, "</ref>": 151852, "<box>": 151853, "</box>": 151854,
        "<quad>": 151855, "</quad>": 151856,
        "<img>": 151857, "</img>": 151858, "<imgpad>": 151859,
    }
    for surface, want in anchors.items():
        got = enc.encode(surface, allowed_special="all")
        if got != [want]:
            raise SystemExit(f"ANCHOR FAIL: {surface!r} -> {got}, want [{want}]")

    fixtures: "dict[str, list[int]]" = {}
    for s in CORPUS:
        # QWenTokenizer.tokenize applies NFC; all corpus strings are NFC-stable
        # (asserted) so raw==NFC and the ids are valid under both readings.
        nfc = unicodedata.normalize("NFC", s)
        ids = enc.encode(nfc, allowed_special="all")
        assert ids == enc.encode(s, allowed_special="all"), f"NFC drift: {s!r}"
        fixtures[s] = ids

    out_obj = {
        "_meta": {
            "purpose": "GOT-OCR2 (got-ocr2) tokenizer token-id-EXACT conformance golden fixtures",
            "model_id": "got-ocr2",
            "tokenizer_class": "QWenTokenizer (qwen.tiktoken byte-BPE, NOT a HF tokenizer.json)",
            "tokenizer_file_sha256": tt_sha,
            "backend": "tiktoken.Encoding (built directly from the QWenTokenizer recipe)",
            "tiktoken_version": tiktoken.__version__,
            "transformers_version": None,
            "torch_loaded": False,
            "pat_str": PAT_STR,
            "special_start_id": SPECIAL_START_ID,
            "num_extras": len(EXTRAS),
            "n_vocab": enc.n_vocab,
            "special_tokens_count": len(SPECIAL_TOKENS) + len(IMAGE_ST),
            "grounding_image_special_tokens": {
                "<ref>": 151851, "</ref>": 151852, "<box>": 151853, "</box>": 151854,
                "<quad>": 151855, "</quad>": 151856,
                "<img>": 151857, "</img>": 151858, "<imgpad>": 151859,
            },
            "encode_flags": {"allowed_special": "all", "disallowed_special": "()"},
            "nfc": "QWenTokenizer.tokenize applies unicodedata.normalize('NFC'); all corpus strings asserted NFC-stable",
            "num_cases": len(fixtures),
        },
        "fixtures": fixtures,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(fixtures)} cases to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
