#!/usr/bin/env python3
"""Generate golden tokenizer fixtures for franken_ocr's OQ-16 conformance gate.

This is a COMPLETE, runnable generator (NOT a skeleton). It loads the pinned
reference tokenizer (`docs/truth-pack/snapshots/tokenizer.json`, the Baidu
Unlimited-OCR `LlamaTokenizerFast` serialization) and emits one golden record
per corpus case to `tests/fixtures/tokenizer/expected.jsonl`. The pure-Rust
byte-level-BPE tokenizer is held token-id-exact against these records.

OFFLINE TOOLING ONLY. franken_ocr's Rust engine NEVER invokes this script and
NEVER unpickles or loads `tokenizer.json` at inference time via Python. This is a
deliberate, human-run, out-of-band step whose only job is to freeze the
reference tokenizer's behavior into a file the Rust conformance tests compare
against. It does NOT run the OCR model and imports no torch.

WHAT THIS SCRIPT DOES
---------------------
1. Verifies the SHA-256 of `tokenizer.json` matches the pin recorded in
   docs/truth-pack/oq/tokenizer.md (fixtures generated against a different
   tokenizer.json are NOT comparable, so we fail loudly on drift).
2. Loads `tokenizer.json` via the HF `tokenizers` Rust-backed library
   (Tokenizer.from_file), with an automatic fallback to
   `transformers.LlamaTokenizerFast` if only `transformers` is installed. Both
   load the same `tokenizer.json` and produce identical ids.
3. Reads `tests/fixtures/tokenizer/corpus.txt`. Each non-comment, non-blank line
   is a JSON-encoded string (the exact output of
   `json.dumps(text, ensure_ascii=False)`), which losslessly preserves
   whitespace, tabs, CR/LF and Unicode.
4. For each case, encodes with **add_special_tokens=False** (the inference path:
   `modeling_unlimitedocr.py:259-268` calls `tokenizer.encode(text,
   add_special_tokens=False)` and hardcodes BOS=0 / EOS=1 itself — the prompt
   builder, not the tokenizer, owns specials). It then decodes the ids back and
   records both, emitting one JSON object per line:
       {"text": <str>, "ids": [<int>, ...], "decoded": <str>}
5. Records provenance (tokenizer.json sha256, backend used + version, BOS/EOS/PAD
   ids, vocab size, add_special_tokens flag) as the FIRST line of the output,
   tagged with "_meta": true, so a stale fixture is auditable and detectable.

The Rust conformance test asserts: for every record, the Rust encoder produces
EXACTLY `ids` for `text` (with add_special_tokens=False semantics), and the Rust
decoder produces EXACTLY `decoded` for `ids`. The "_meta" line is skipped by the
test loader (it carries provenance, not a case).

REQUIREMENTS (install into an isolated venv; never runs in CI inference):
    python3 >= 3.9
    EITHER  tokenizers>=0.15   (preferred; the same Rust crate the model uses)
    OR      transformers>=4.40 (LlamaTokenizerFast path)

USAGE:
    python3 scripts/gen_tokenizer_fixtures.py
    python3 scripts/gen_tokenizer_fixtures.py \
        --tokenizer-json docs/truth-pack/snapshots/tokenizer.json \
        --corpus tests/fixtures/tokenizer/corpus.txt \
        --out tests/fixtures/tokenizer/expected.jsonl

This script only WRITES the fixture; it does not run the Rust tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# SHA-256 of the pinned reference tokenizer.json (docs/truth-pack/oq/tokenizer.md
# §"Source files read" and §UNBLOCKS). Fixtures are only comparable against this
# exact serialization; a mismatch is a hard error.
PINNED_TOKENIZER_SHA256 = (
    "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4"
)

# Hardcoded specials the MODEL applies (NOT the tokenizer post-processor) — see
# modeling_unlimitedocr.py:259-268 / :966-968. Recorded as provenance so the
# Rust prompt-builder bead can cross-check; encoding here uses
# add_special_tokens=False, so these ids do NOT appear in the per-case `ids`.
BOS_ID = 0
EOS_ID = 1
PAD_ID = 2

# Repo-root-relative defaults (resolved against this file's location so the
# script is runnable from any cwd).
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER_JSON = _REPO_ROOT / "docs" / "truth-pack" / "snapshots" / "tokenizer.json"
DEFAULT_CORPUS = _REPO_ROOT / "tests" / "fixtures" / "tokenizer" / "corpus.txt"
DEFAULT_OUT = _REPO_ROOT / "tests" / "fixtures" / "tokenizer" / "expected.jsonl"


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gen_tokenizer_fixtures.py",
        description="Emit golden tokenizer fixtures (text/ids/decoded) for the "
        "franken_ocr OQ-16 token-id-exact conformance gate.",
    )
    p.add_argument(
        "--tokenizer-json",
        type=Path,
        default=DEFAULT_TOKENIZER_JSON,
        help="Path to the pinned tokenizer.json (default: docs/truth-pack/snapshots/tokenizer.json).",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to corpus.txt (one JSON-encoded string per line; default: tests/fixtures/tokenizer/corpus.txt).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSONL path (default: tests/fixtures/tokenizer/expected.jsonl).",
    )
    p.add_argument(
        "--no-sha-check",
        action="store_true",
        help="Skip the tokenizer.json SHA-256 pin check (NOT recommended; for "
        "regenerating the pin only).",
    )
    return p.parse_args(argv)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Backend:
    """Uniform wrapper over the two acceptable loaders.

    Both load the SAME tokenizer.json and MUST yield identical ids. We prefer the
    `tokenizers` crate (the exact engine the reference model uses) and fall back
    to transformers.LlamaTokenizerFast only if `tokenizers` is unavailable.
    """

    def __init__(self, name: str, version: str, encode_fn, decode_fn):
        self.name = name
        self.version = version
        self._encode = encode_fn
        self._decode = decode_fn

    def encode(self, text: str) -> "list[int]":
        return self._encode(text)

    def decode(self, ids: "list[int]") -> str:
        return self._decode(ids)


def load_backend(tokenizer_json: Path) -> _Backend:
    """Load tokenizer.json, preferring `tokenizers`, falling back to transformers."""
    # --- Preferred: HF `tokenizers` Rust library (same crate as the model). ---
    try:
        import tokenizers  # type: ignore
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_file(str(tokenizer_json))

        def _encode(text: str) -> "list[int]":
            # add_special_tokens=False matches the inference path; the model
            # prepends BOS / appends EOS itself.
            return tok.encode(text, add_special_tokens=False).ids

        def _decode(ids: "list[int]") -> str:
            # skip_special_tokens=False: we want the literal round-trip of EXACTLY
            # the ids the encoder produced (specials may legitimately appear when
            # the corpus contains literal special-token strings).
            return tok.decode(ids, skip_special_tokens=False)

        version = getattr(tokenizers, "__version__", "unknown")
        return _Backend("tokenizers", version, _encode, _decode)
    except ImportError:
        pass

    # --- Fallback: transformers LlamaTokenizerFast over the same tokenizer.json. ---
    try:
        import transformers  # type: ignore
        from transformers import PreTrainedTokenizerFast  # type: ignore

        tok = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))

        def _encode(text: str) -> "list[int]":
            return tok.encode(text, add_special_tokens=False)

        def _decode(ids: "list[int]") -> str:
            return tok.decode(ids, skip_special_tokens=False)

        version = getattr(transformers, "__version__", "unknown")
        return _Backend("transformers.PreTrainedTokenizerFast", version, _encode, _decode)
    except ImportError:
        pass

    raise SystemExit(
        "ERROR: neither `tokenizers` nor `transformers` is importable.\n"
        "Install one into an isolated venv:\n"
        "    pip install 'tokenizers>=0.15'        # preferred\n"
        "    # or\n"
        "    pip install 'transformers>=4.40'\n"
    )


def iter_corpus_cases(corpus_path: Path) -> "list[str]":
    """Yield each decoded test string from the JSON-per-line corpus.

    Skips blank lines and comment lines. A line is a comment iff the RAW line
    (before JSON-decoding) starts with a literal '#'. Every other line MUST be a
    valid JSON string literal; anything else is a hard error (so a malformed
    corpus fails loudly rather than silently dropping cases).
    """
    cases: "list[str]" = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"ERROR: {corpus_path}:{lineno}: not a JSON-encoded string: {e}\n"
                    f"  line was: {line!r}"
                )
            if not isinstance(value, str):
                raise SystemExit(
                    f"ERROR: {corpus_path}:{lineno}: JSON value is {type(value).__name__}, "
                    f"expected a string literal."
                )
            cases.append(value)
    return cases


def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)

    tokenizer_json: Path = args.tokenizer_json
    corpus_path: Path = args.corpus
    out_path: Path = args.out

    if not tokenizer_json.is_file():
        raise SystemExit(f"ERROR: tokenizer.json not found: {tokenizer_json}")
    if not corpus_path.is_file():
        raise SystemExit(f"ERROR: corpus not found: {corpus_path}")

    actual_sha = sha256_of(tokenizer_json)
    if not args.no_sha_check and actual_sha != PINNED_TOKENIZER_SHA256:
        raise SystemExit(
            "ERROR: tokenizer.json SHA-256 does not match the pin.\n"
            f"  expected: {PINNED_TOKENIZER_SHA256}\n"
            f"  actual:   {actual_sha}\n"
            "Fixtures generated against a different tokenizer.json are not "
            "comparable. Pass --no-sha-check only if you are deliberately "
            "re-pinning (and update PINNED_TOKENIZER_SHA256 + the OQ doc)."
        )

    backend = load_backend(tokenizer_json)
    cases = iter_corpus_cases(corpus_path)

    # Sanity cross-check on the specials the model hardcodes, so a stale or
    # mismatched tokenizer.json is caught beyond the SHA pin.
    bos_ok = backend.encode("<｜begin▁of▁sentence｜>") == [BOS_ID]
    eos_ok = backend.encode("<｜end▁of▁sentence｜>") == [EOS_ID]
    image_ok = backend.encode("<image>") == [128815]
    if not (bos_ok and eos_ok and image_ok):
        raise SystemExit(
            "ERROR: special-token id sanity check failed (tokenizer.json may be "
            f"wrong): BOS->{backend.encode('<｜begin▁of▁sentence｜>')} "
            f"EOS->{backend.encode('<｜end▁of▁sentence｜>')} "
            f"<image>->{backend.encode('<image>')}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "_meta": True,
        "purpose": "franken_ocr OQ-16 tokenizer conformance golden fixtures",
        "tokenizer_json": str(tokenizer_json.relative_to(_REPO_ROOT))
        if tokenizer_json.is_relative_to(_REPO_ROOT)
        else str(tokenizer_json),
        "tokenizer_json_sha256": actual_sha,
        "backend": backend.name,
        "backend_version": backend.version,
        "add_special_tokens": False,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "pad_id": PAD_ID,
        "image_token_id": 128815,
        "note": (
            "ids are encoded with add_special_tokens=False (inference path). The "
            "model prepends BOS=0 / appends EOS=1 itself; the prompt-builder bead "
            "owns specials, not the tokenizer. Skip this _meta line when loading "
            "cases."
        ),
        "num_cases": len(cases),
    }

    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        out.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for text in cases:
            ids = list(backend.encode(text))
            decoded = backend.decode(ids)
            record = {"text": text, "ids": ids, "decoded": decoded}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"wrote {written} cases (+1 _meta line) to {out_path} "
        f"using backend={backend.name} v{backend.version}",
        file=sys.stderr,
    )
    print(f"  tokenizer.json sha256: {actual_sha}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
