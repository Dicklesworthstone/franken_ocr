#!/usr/bin/env python3
"""Characterize the Unlimited-OCR reference oracle's OWN nondeterminism floor.

This is the COMPLETE, runnable, offline tool that establishes the
**nondeterminism envelope** the franken_ocr conformance ladder derives every
L3/L4 tolerance from. It is the operational half of bead `VERIFY-nondeterminism-floor`
(`bd-re8.2`), and a sibling of `scripts/gen_reference_fixtures.py`: like that
script it is OFFLINE TOOLING ONLY (the Rust engine NEVER invokes it and NEVER
unpickles anything), and like that script it **requires the 6.67 GB weights +
the pinned torch/transformers stack** — it cannot run on a machine without them.

WHY THIS EXISTS (plan §8.2, AGENTS.md Testing Policy)
-----------------------------------------------------
The HF reference is FREQUENTLY non-deterministic across torch thread counts /
BLAS reduction order at the logit-tie level: bf16 arithmetic over a 129280-vocab
argmax means two runs (or two thread counts) can disagree on a tied token, and
that disagreement cascades through greedy decode. If we set parity tolerances
WITHOUT first measuring the oracle's own noise, we will either chase phantom
"bugs" that are just the oracle disagreeing with itself, or set tolerances so
loose that real franken_ocr drift hides inside them.

So, BEFORE any tolerance is set, this tool:

  1. Runs the reference oracle over the golden corpus **twice** (two repeats) and
     at **two thread counts** (default 8 and 32) — four runs per document.
  2. Captures, per document, the **decoded token-id sequence** and the
     **per-decode-step lm_head logits** (greedy / temperature=0.0).
  3. Diffs the four runs pairwise and records the **nondeterminism envelope** per
     document:
       - per-token divergence rate (fraction of compared positions whose argmax
         token differs across runs),
       - first-divergence position (the earliest decode step at which ANY pair of
         runs disagrees on the token) — this is the L4 "exact prefix" boundary,
       - per-logit max-abs spread at matching positions (the largest absolute
         logit difference across runs at positions before the first divergence) —
         this seeds the L3 logit tolerance.
  4. Derives and emits:
       - `tests/fixtures/oracle_nondeterminism_envelope.json` — the full per-doc
         envelope + provenance (the committed fixture the ladder reads).
       - `tolerances.toml` — the machine-readable L3 logit tolerance (derived from
         the measured spread, NOT the imported frankensearch 0.055) and the L4
         exact-prefix bounds, consumed by the conformance gates.
  5. Prints a `docs/DISCREPANCIES.md`-ready note to stderr if the floor reveals
     structural nondeterminism (a non-zero divergence rate inside the prefix, or a
     short reproducible prefix) so a human can ledger it as a DISC-NNN.

L4 "exact" is then defined ONLY over the reproducible prefix; the L3 logit
tolerance is the measured oracle variance. A franken_ocr int8 divergence INSIDE
this measured envelope is NOT a bug — and this tool is the proof of that claim.

WHAT IT DOES NOT DO
-------------------
It does NOT set the golden correctness fixtures (that is
`gen_reference_fixtures.py`, the GPU bf16 oracle). It only measures the oracle's
self-disagreement so the tolerances are defensible rather than guessed. It must
be RE-RUN if the pinned model source changes (CI-guarded against drift via the
E-PM1 census / SOURCE_HASHES.md).

Requires: python3, torch==2.10.0, transformers==4.57.1, numpy, Pillow.
Install into an isolated venv; this never runs in CI inference.

Usage:
    FOCR_MODEL_DIR=/path/to/model \\
        python3 oracle_nondeterminism_floor.py \\
            --corpus tests/fixtures/corpus \\
            --out tests/fixtures \\
            --tolerances tolerances.toml \\
            --thread-counts 8 32 --repeats 2 --mode gundam

Per OQ-17 the correctness oracle is CUDA-only (the official `infer()` path
hard-codes `.cuda()` + CUDA autocast), so the floor is measured on the SAME
device class the golden fixtures come from. A non-cuda device is allowed only for
the separate, must-be-proven CPU-equivalence experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ─────────────────────────────────────────────────────────────────────────────
# Pins the finished version MUST assert against (Unlimited-OCR README runtime).
# Source: docs/truth-pack/PINNED_SOURCES.md. Shared verbatim with
# gen_reference_fixtures.py — fixtures/envelopes generated against a different
# stack are NOT comparable.
# ─────────────────────────────────────────────────────────────────────────────
PIN_TORCH = "2.10.0"
PIN_TRANSFORMERS = "4.57.1"

# The HF repo + the immutable commit the whole truth-pack is pinned to.
HF_REPO = "baidu/Unlimited-OCR"
HF_COMMIT = "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"

# Canonical filenames inside $FOCR_MODEL_DIR (see scripts/fetch_model.sh).
WEIGHTS_FILE = "model-00001-of-000001.safetensors"
INDEX_FILE = "model.safetensors.index.json"
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "config.json"

# Vocabulary size — the lm_head logits' last dim (config.json vocab_size). Used
# as a sanity check on captured logits (CENSUS quick-reference: 129280).
VOCAB_SIZE = 129280

# Recommended per-mode generation config (docs/truth-pack EXISTING_..._STRUCTURE.md
# §8 SPEC-100..103, OQ-18). Mirrors gen_reference_fixtures.py so the floor is
# measured under the SAME generation semantics the golden fixtures use.
MODE_PRESETS: dict[str, dict[str, Any]] = {
    "gundam": dict(
        base_size=1024,
        image_size=640,
        crop_mode=True,
        no_repeat_ngram_size=35,
        ngram_window=128,
    ),
    "base": dict(
        base_size=1024,
        image_size=1024,
        crop_mode=False,
        no_repeat_ngram_size=35,
        ngram_window=128,
    ),
}

# Default prompt per mode (the de-facto document-parsing prompt; OQ-8).
DEFAULT_PROMPT = "<image>document parsing."

# Image suffixes we treat as corpus documents (v1 is image-only).
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Output artifact names.
ENVELOPE_FILE = "oracle_nondeterminism_envelope.json"
DEFAULT_TOLERANCES_FILE = "tolerances.toml"
# Unlimited-OCR-2 (U2-A6): the committed envelope of the v2 oracle's runs.
UOCR2_FLOOR_FILE = "tests/fixtures/uocr2_oracle/FLOOR.json"
# The v2 oracle keeps logits for every one of the first 64 steps.
UOCR2_HEAD_STEPS = 64

# Schema version for the committed envelope/tolerances — bumped on any layout
# change so a stale fixture is detectable (mirrors the gen_reference_fixtures
# schema_version discipline).
ENVELOPE_SCHEMA_VERSION = 1

# A small safety margin applied to the measured logit spread when deriving the L3
# tolerance, so a tolerance derived from a finite sample is not razor-thin. This
# is a documented multiplier, NOT a hand-guessed absolute epsilon — the absolute
# number still comes entirely from the measured spread.
L3_TOLERANCE_MARGIN = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="oracle_nondeterminism_floor.py",
        description="Run the Unlimited-OCR reference twice and at two thread "
        "counts over a corpus; emit the per-token nondeterminism envelope "
        "(divergence rate, first-divergence position, logit spread) as the floor "
        "for L3/L4 tolerances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Dir holding the Unlimited-OCR safetensors + tokenizer.json + "
        "config.json (defaults to $FOCR_MODEL_DIR). See scripts/fetch_model.sh.",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path("tests/fixtures/corpus"),
        help="Dir of input document images to run through the oracle.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures"),
        help="Output dir for the envelope JSON (oracle_nondeterminism_envelope.json).",
    )
    p.add_argument(
        "--tolerances",
        type=Path,
        default=Path(DEFAULT_TOLERANCES_FILE),
        help="Path to write the derived tolerances.toml (consumed by the ladder).",
    )
    p.add_argument(
        "--mode",
        choices=sorted(MODE_PRESETS.keys()),
        default="gundam",
        help="Inference mode preset (geometry + generation config).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Free-text prompt passed to the oracle (must contain <image>).",
    )
    p.add_argument(
        "--thread-counts",
        type=int,
        nargs="+",
        default=[8, 32],
        help="The torch CPU thread counts to vary (>=2 values). The plan calls "
        "for two thread counts; more is allowed. (Even on CUDA these change CPU-"
        "side reduction order; the variation is the point.)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Repeats per (thread-count, document). The plan calls for run-twice; "
        ">=2 required.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Generation cap (matches the reference max_length).",
    )
    p.add_argument(
        "--max-compare-tokens",
        type=int,
        default=2048,
        help="Cap the number of decode steps whose logits are captured + compared "
        "(0 = no cap). Captured logits are 129280-wide f32; capping bounds memory "
        "on very long parses without affecting the first-divergence measurement "
        "for typical documents.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N corpus documents (0 = no limit).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for the oracle. Per OQ-17 the correctness oracle is "
        "CUDA-only; a non-cuda value is allowed ONLY for the separate, "
        "must-be-proven CPU-equivalence experiment.",
    )
    p.add_argument(
        "--uocr2-group",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR,RUN_DIR[,...]",
        help="Unlimited-OCR-2 (U2-A6): compare runs already written by "
        "scripts/gen_reference_fixtures_uocr2.py (each RUN_DIR is <out>/<dtype>/<tag>) "
        "instead of running the v1 oracle. Repeatable; numpy only, no torch.",
    )
    p.add_argument(
        "--uocr2-out",
        type=Path,
        default=Path(UOCR2_FLOOR_FILE),
        help="Where --uocr2-group writes the envelope JSON.",
    )
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Stack assertions (shared shape with gen_reference_fixtures.py)
# ─────────────────────────────────────────────────────────────────────────────
def _assert_pinned_stack() -> tuple[Any, Any]:
    """Import + assert torch/transformers match the pinned reference stack.

    An envelope measured against a different stack is not comparable, so we refuse
    to proceed on a version mismatch.
    """
    try:
        import torch  # noqa: WPS433 (deliberate late import — offline tool)
        import transformers  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "oracle_nondeterminism_floor: torch / transformers are not installed. "
            f"Install the pinned oracle stack (torch=={PIN_TORCH}, "
            f"transformers=={PIN_TRANSFORMERS}) into an isolated venv. "
            f"Underlying import error: {exc}"
        )

    torch_v = torch.__version__.split("+", 1)[0]  # drop "+cu128" build tag
    tf_v = transformers.__version__
    problems = []
    if torch_v != PIN_TORCH:
        problems.append(f"torch=={torch_v} (need {PIN_TORCH})")
    if tf_v != PIN_TRANSFORMERS:
        problems.append(f"transformers=={tf_v} (need {PIN_TRANSFORMERS})")
    if problems:
        raise SystemExit(
            "oracle_nondeterminism_floor: pinned-stack mismatch — an envelope "
            "measured against a different stack is not comparable. Found: "
            + ", ".join(problems)
            + ". Recreate the venv with the exact pins from "
            "docs/truth-pack/PINNED_SOURCES.md."
        )
    return torch, transformers


def _resolve_model_dir(cli_dir: Path | None) -> Path:
    raw = cli_dir or os.environ.get("FOCR_MODEL_DIR")
    if not raw:
        raise SystemExit(
            "oracle_nondeterminism_floor: no model dir. Pass --model-dir or set "
            "$FOCR_MODEL_DIR (see scripts/fetch_model.sh)."
        )
    model_dir = Path(raw).expanduser()
    if not model_dir.is_dir():
        raise SystemExit(
            f"oracle_nondeterminism_floor: model dir not found: {model_dir}"
        )
    missing = [
        name
        for name in (WEIGHTS_FILE, TOKENIZER_FILE, CONFIG_FILE)
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise SystemExit(
            f"oracle_nondeterminism_floor: model dir {model_dir} is missing "
            f"{', '.join(missing)}. Run scripts/fetch_model.sh first."
        )
    return model_dir


def _sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    """Streaming SHA-256 — never loads the 6.67 GB shard into memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (mirrors gen_reference_fixtures.py)
# ─────────────────────────────────────────────────────────────────────────────
def _load_model_and_tokenizer(torch: Any, model_dir: Path, device: str):
    """Load Unlimited-OCR + tokenizer in bf16, eval, on `device`."""
    from transformers import AutoModel, AutoTokenizer  # noqa: WPS433

    print(f"[floor] loading tokenizer from {model_dir} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print(
        f"[floor] loading model from {model_dir} (bf16, {device}) ...",
        file=sys.stderr,
    )
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval()
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "oracle_nondeterminism_floor: --device cuda but no CUDA device is "
                "available. The correctness oracle is GPU-only (OQ-17). Run on a "
                "CUDA host, or use --device cpu ONLY for the separate "
                "CPU-equivalence experiment."
            )
        model = model.cuda()
    else:
        model = model.to(device)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Per-decode-step logit capture
# ─────────────────────────────────────────────────────────────────────────────
class LogitTrace:
    """Capture the lm_head logits of EACH decode step.

    `infer()` calls `generate()`, which runs one prefill pass (seq_len > 1)
    followed by one forward per generated token (seq_len == 1). We record the
    single-position logits at every DECODE step (skipping the prefill, whose
    logits are the L3-prefill seam handled by gen_reference_fixtures.py). The
    captured per-step argmax IS the greedy token, and the stacked logits feed the
    per-position logit-spread measurement.
    """

    def __init__(self, torch: Any, model: Any, max_steps: int):
        self.torch = torch
        self.model = model
        self.max_steps = max_steps  # 0 = unbounded
        self._handle: Any | None = None
        self._logits: list[Any] = []  # list of (vocab,) float32 numpy arrays

    @staticmethod
    def _primary_tensor(out: Any) -> Any | None:
        import torch as _t  # local alias for isinstance checks

        if isinstance(out, _t.Tensor):
            return out
        if isinstance(out, (tuple, list)) and out:
            head = out[0]
            if isinstance(head, _t.Tensor):
                return head
        return None

    def _hook(self, _module: Any, _inp: Any, out: Any) -> None:
        import numpy as np  # noqa: WPS433

        t = self._primary_tensor(out)
        if t is None:
            return
        if t.dim() < 2:
            return
        # Decode steps have a single query position (seq_len == 1). Skip prefill
        # (seq_len > 1) — its logits are a separate seam.
        if t.shape[-2] != 1:
            return
        if self.max_steps and len(self._logits) >= self.max_steps:
            return
        # (..., 1, vocab) -> (vocab,) float32 on CPU.
        last = t.detach()
        if last.dtype == self.torch.bfloat16:
            last = last.to(self.torch.float32)
        vec = last.reshape(-1, last.shape[-1])[-1].to("cpu").numpy()
        self._logits.append(vec.astype(np.float32, copy=False))

    def register(self) -> None:
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            raise SystemExit(
                "oracle_nondeterminism_floor: model has no lm_head to hook; cannot "
                "capture per-step logits (the floor needs them)."
            )
        self._handle = lm_head.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def stacked(self) -> Any:
        """(steps, vocab) float32 array of the captured decode-step logits."""
        import numpy as np  # noqa: WPS433

        if not self._logits:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(self._logits, axis=0)

    def __enter__(self) -> "LogitTrace":
        self.register()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()


# ─────────────────────────────────────────────────────────────────────────────
# One oracle run (one document, one thread-count, one repeat)
# ─────────────────────────────────────────────────────────────────────────────
def _set_thread_count(torch: Any, n_threads: int) -> None:
    """Pin torch's CPU intra-op threads. Varying this changes BLAS reduction
    order, which is exactly the nondeterminism source we are measuring."""
    try:
        torch.set_num_threads(int(n_threads))
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(
            f"[floor] warning: torch.set_num_threads({n_threads}) failed: {exc}",
            file=sys.stderr,
        )
    os.environ["OMP_NUM_THREADS"] = str(n_threads)


def _enable_determinism(torch: Any) -> bool:
    """Enable deterministic algorithms where it does not error (plan §8.2).

    Returns whether it was enabled — recorded in provenance. Even with this on,
    bf16 + a 129280-vocab argmax can still tie-break differently across thread
    counts; that residual is precisely the floor we measure.
    """
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        return True
    except Exception as exc:  # noqa: BLE001 - older/newer API or unsupported op
        print(
            f"[floor] note: use_deterministic_algorithms unavailable/failed "
            f"({exc}); proceeding (the residual nondeterminism is the measurement).",
            file=sys.stderr,
        )
        return False


def _run_once(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    doc: Path,
    out_scratch: Path,
    prompt: str,
    preset: dict[str, Any],
    max_length: int,
    max_compare_tokens: int,
) -> dict[str, Any]:
    """Run the oracle on one document; return decoded text, token ids, and the
    captured per-decode-step logits."""
    out_scratch.mkdir(parents=True, exist_ok=True)
    trace = LogitTrace(torch, model, max_steps=max_compare_tokens)
    trace.register()
    t0 = time.time()
    try:
        decoded = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(doc),
            output_path=str(out_scratch),
            base_size=preset["base_size"],
            image_size=preset["image_size"],
            crop_mode=preset["crop_mode"],
            max_length=max_length,
            no_repeat_ngram_size=preset["no_repeat_ngram_size"],
            ngram_window=preset["ngram_window"],
            temperature=0.0,  # greedy / deterministic-as-possible oracle
            save_results=False,
        )
    finally:
        trace.remove()
    elapsed = time.time() - t0

    decoded_text = decoded if isinstance(decoded, str) else ""
    logits = trace.stacked()  # (steps, vocab) f32

    # The greedy token at each captured step is the argmax of that step's logits.
    if logits.shape[0] > 0:
        import numpy as np  # noqa: WPS433

        token_ids = np.argmax(logits, axis=1).astype("int64").tolist()
        vocab_dim = int(logits.shape[1])
    else:
        token_ids = []
        vocab_dim = 0

    return {
        "decoded_text": decoded_text,
        "decoded_sha256": hashlib.sha256(decoded_text.encode("utf-8")).hexdigest(),
        "token_ids": token_ids,
        "n_tokens": len(token_ids),
        "vocab_dim": vocab_dim,
        "logits": logits,  # numpy, NOT serialized; consumed by the diff then dropped
        "elapsed_seconds": round(elapsed, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Envelope computation (the diff)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_doc_envelope(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Diff every run of one document and compute its nondeterminism envelope.

    `runs` is the list of per-run dicts for ONE document across all
    (thread-count, repeat) combinations. Each carries `token_ids` and a
    `(steps, vocab)` `logits` numpy array.

    Envelope fields:
      - first_divergence_pos: the earliest decode step at which ANY pair of runs
        disagrees on the argmax token. == min compared length when fully
        reproducible (i.e. "no divergence within the compared prefix"). This is
        the L4 exact-prefix boundary.
      - divergence_rate: fraction of compared positions (over the common length)
        where the runs are not unanimous on the token.
      - logit_spread_in_prefix: the max over positions < first_divergence_pos of
        the per-position max-abs logit difference across runs. Seeds L3.
      - logit_spread_overall: the same max over ALL compared positions (audit).
      - decoded_text_identical: whether every run's decoded text is byte-identical.
    """
    import numpy as np  # noqa: WPS433

    n_runs = len(runs)
    common_len = min(r["n_tokens"] for r in runs) if runs else 0

    # Token unanimity per position over the common prefix.
    first_div = common_len  # default: reproducible across the whole common prefix
    n_divergent_positions = 0
    if n_runs >= 2 and common_len > 0:
        token_matrix = np.stack(
            [np.asarray(r["token_ids"][:common_len], dtype=np.int64) for r in runs],
            axis=0,
        )  # (n_runs, common_len)
        # A position diverges if not all runs agree with run 0.
        disagree = np.any(token_matrix != token_matrix[0:1, :], axis=0)  # (common_len,)
        n_divergent_positions = int(disagree.sum())
        div_positions = np.nonzero(disagree)[0]
        if div_positions.size > 0:
            first_div = int(div_positions[0])

    divergence_rate = (
        float(n_divergent_positions) / float(common_len) if common_len > 0 else 0.0
    )

    # Per-position max-abs logit spread across runs (only where vocab dims match).
    logit_spread_in_prefix = 0.0
    logit_spread_overall = 0.0
    vocab_dims = {int(r["vocab_dim"]) for r in runs if r["vocab_dim"] > 0}
    if n_runs >= 2 and common_len > 0 and len(vocab_dims) == 1:
        vdim = vocab_dims.pop()
        stack = np.stack(
            [r["logits"][:common_len, :vdim] for r in runs], axis=0
        )  # (n_runs, common_len, vocab)
        per_pos_spread = (stack.max(axis=0) - stack.min(axis=0)).max(
            axis=1
        )  # (common_len,)
        logit_spread_overall = float(per_pos_spread.max())
        if first_div > 0:
            logit_spread_in_prefix = float(per_pos_spread[:first_div].max())
        else:
            logit_spread_in_prefix = 0.0

    decoded_shas = {r["decoded_sha256"] for r in runs}
    return {
        "n_runs": n_runs,
        "n_tokens_per_run": [r["n_tokens"] for r in runs],
        "common_compared_len": common_len,
        "first_divergence_pos": first_div,
        "n_divergent_positions": n_divergent_positions,
        "divergence_rate": round(divergence_rate, 8),
        "logit_spread_in_prefix": round(logit_spread_in_prefix, 8),
        "logit_spread_overall": round(logit_spread_overall, 8),
        "decoded_text_identical": len(decoded_shas) == 1,
        "decoded_text_sha256_set": sorted(decoded_shas),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tolerance derivation + tolerances.toml emission
# ─────────────────────────────────────────────────────────────────────────────
def _derive_tolerances(per_doc: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Derive the L3 logit tolerance + the L4 exact-prefix policy from the
    measured envelope across the whole corpus.

    - L3 logit tolerance = (max over documents of the in-prefix logit spread)
      times a documented safety margin. This is the measured oracle variance —
      explicitly NOT the imported frankensearch 0.055.
    - L4 exact-prefix bound = per document, the first_divergence_pos: text/token
      bit-exactness is asserted ONLY over [0, first_divergence_pos). Beyond it the
      oracle is non-deterministic and the gate falls back to CER-within-budget.
    """
    max_in_prefix_spread = 0.0
    max_overall_spread = 0.0
    max_divergence_rate = 0.0
    min_first_divergence = None
    for env in per_doc.values():
        max_in_prefix_spread = max(max_in_prefix_spread, env["logit_spread_in_prefix"])
        max_overall_spread = max(max_overall_spread, env["logit_spread_overall"])
        max_divergence_rate = max(max_divergence_rate, env["divergence_rate"])
        fd = env["first_divergence_pos"]
        min_first_divergence = fd if min_first_divergence is None else min(
            min_first_divergence, fd
        )

    l3_logit_tolerance = round(max_in_prefix_spread * L3_TOLERANCE_MARGIN, 8)
    return {
        "l3_logit_tolerance": l3_logit_tolerance,
        "l3_tolerance_margin": L3_TOLERANCE_MARGIN,
        "measured_max_in_prefix_logit_spread": round(max_in_prefix_spread, 8),
        "measured_max_overall_logit_spread": round(max_overall_spread, 8),
        "measured_max_token_divergence_rate": round(max_divergence_rate, 8),
        "min_first_divergence_pos": int(min_first_divergence or 0),
    }


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _write_tolerances_toml(
    path: Path,
    *,
    derived: dict[str, Any],
    per_doc: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    """Emit a hand-written TOML (no toml-writer dependency) the ladder consumes.

    The format is intentionally simple + line-oriented so it diffs cleanly and is
    parseable by the Rust gates (a tiny `toml` crate read).
    """
    lines: list[str] = []
    lines.append("# tolerances.toml — DERIVED from the measured oracle")
    lines.append("# nondeterminism floor (scripts/oracle_nondeterminism_floor.py,")
    lines.append("# bead VERIFY-nondeterminism-floor / bd-re8.2). DO NOT hand-edit:")
    lines.append("# regenerate by re-running the floor tool. Every number here is")
    lines.append("# MEASURED, not guessed — the L3 logit tolerance is the oracle's")
    lines.append("# own bf16 variance, NOT the imported frankensearch 0.055.")
    lines.append("")
    lines.append(f'schema_version = {ENVELOPE_SCHEMA_VERSION}')
    lines.append(f'generated_utc = "{provenance["generated_utc"]}"')
    lines.append(f'hf_commit = "{provenance["hf_commit"]}"')
    lines.append(
        f'model_weights_sha256 = "{provenance["model_weights_sha256"]}"'
    )
    lines.append(f'envelope_fixture = "{ENVELOPE_FILE}"')
    lines.append("")
    lines.append("[l3]")
    lines.append("# pre-sampling logit tolerance: |focr_logit - oracle_logit| must")
    lines.append("# be <= this, AND argmax must match, where the reference is")
    lines.append("# deterministic. Derived = measured in-prefix spread x margin.")
    lines.append(f'logit_tolerance = {derived["l3_logit_tolerance"]}')
    lines.append(f'tolerance_margin = {derived["l3_tolerance_margin"]}')
    lines.append(
        f'measured_max_in_prefix_logit_spread = '
        f'{derived["measured_max_in_prefix_logit_spread"]}'
    )
    lines.append(
        f'measured_max_overall_logit_spread = '
        f'{derived["measured_max_overall_logit_spread"]}'
    )
    lines.append("")
    lines.append("[l4]")
    lines.append("# token/text bit-exactness is asserted ONLY over the per-document")
    lines.append("# reproducible prefix [0, first_divergence_pos). Beyond it the bf16")
    lines.append("# oracle is non-deterministic => fall back to CER-within-budget.")
    lines.append(
        f'measured_max_token_divergence_rate = '
        f'{derived["measured_max_token_divergence_rate"]}'
    )
    lines.append(f'min_first_divergence_pos = {derived["min_first_divergence_pos"]}')
    lines.append("")
    lines.append("# Per-document exact-prefix boundaries (the L4 gate reads these).")
    for doc_name in sorted(per_doc):
        env = per_doc[doc_name]
        lines.append(f"[l4.exact_prefix.\"{_toml_escape(doc_name)}\"]")
        lines.append(f'first_divergence_pos = {env["first_divergence_pos"]}')
        lines.append(f'common_compared_len = {env["common_compared_len"]}')
        lines.append(f'divergence_rate = {env["divergence_rate"]}')
        lines.append(
            f'decoded_text_identical = '
            f'{"true" if env["decoded_text_identical"] else "false"}'
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Provenance + corpus listing (shared shape with gen_reference_fixtures.py)
# ─────────────────────────────────────────────────────────────────────────────
def _list_corpus(corpus: Path) -> list[Path]:
    if not corpus.is_dir():
        raise SystemExit(
            f"oracle_nondeterminism_floor: corpus dir not found: {corpus}. "
            "Populate it with document images (see tests/fixtures/corpus)."
        )
    docs = sorted(
        p
        for p in corpus.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not docs:
        raise SystemExit(
            f"oracle_nondeterminism_floor: no images "
            f"({', '.join(sorted(IMAGE_SUFFIXES))}) found in corpus dir {corpus}."
        )
    return docs


def _build_provenance(
    *,
    torch: Any,
    transformers: Any,
    model_dir: Path,
    preset: dict[str, Any],
    prompt: str,
    mode: str,
    device: str,
    max_length: int,
    thread_counts: list[int],
    repeats: int,
    deterministic_enabled: bool,
) -> dict[str, Any]:
    weights = model_dir / WEIGHTS_FILE
    print(
        f"[floor] hashing {weights.name} (reads the full 6.67 GB once) ...",
        file=sys.stderr,
    )
    model_sha = _sha256_file(weights)

    index_total: int | None = None
    index_path = model_dir / INDEX_FILE
    if index_path.is_file():
        try:
            index_total = int(
                json.loads(index_path.read_text(encoding="utf-8"))["metadata"][
                    "total_size"
                ]
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            index_total = None

    cuda_info: dict[str, Any] = {"available": bool(torch.cuda.is_available())}
    if cuda_info["available"]:
        try:
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
            cuda_info["capability"] = list(torch.cuda.get_device_capability(0))
            cuda_info["cuda_runtime"] = torch.version.cuda
        except Exception:  # noqa: BLE001 - diagnostics only
            pass

    return {
        "generator": "scripts/oracle_nondeterminism_floor.py",
        "bead": "VERIFY-nondeterminism-floor (bd-re8.2)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hf_repo": HF_REPO,
        "hf_commit": HF_COMMIT,
        "model_dir": str(model_dir),
        "model_weights_file": WEIGHTS_FILE,
        "model_weights_sha256": model_sha,
        "model_weights_bytes": weights.stat().st_size,
        "model_index_total_size": index_total,
        "device": device,
        "cuda": cuda_info,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "pinned_torch": PIN_TORCH,
        "pinned_transformers": PIN_TRANSFORMERS,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "deterministic_algorithms_enabled": deterministic_enabled,
        "thread_counts": list(thread_counts),
        "repeats": repeats,
        "generation_config": {
            "mode": mode,
            "prompt": prompt,
            "max_length": max_length,
            "temperature": 0.0,
            "do_sample": False,
            **preset,
        },
        # OQ-17: the floor is measured on the same device class as the golden
        # fixtures; a non-cuda device is the separate, must-be-proven experiment.
        "is_correctness_device_class": device == "cuda",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Unlimited-OCR-2 floor (U2-A6): compare runs of gen_reference_fixtures_uocr2.py
# ─────────────────────────────────────────────────────────────────────────────
def _uocr2_layer_delta(np: Any, a: Any, b: Any) -> dict[str, float]:
    """max |a - b| and that maximum relative to max |a| (normwise, so rows near
    zero cannot inflate it)."""
    diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    return {"max_abs": diff, "max_rel": diff / scale if scale > 0 else 0.0}


# The router's "not remote" score (`_SCORE_NEG`, MD); remote entries are above half of it.
UOCR2_SCORE_NEG = -1.0e9


def _uocr2_selected_sets(topi: Any, valid: Any) -> list[frozenset[int]]:
    """Each row's selected blocks: its first-K slots that are candidates. The
    order of the slots is not part of the selection."""
    return [frozenset(int(b) for b, v in zip(t, m) if v) for t, m in zip(topi, valid)]


def _uocr2_compare_fixture(np: Any, run_dirs: list[Path], fixture: str) -> dict[str, Any]:
    refs, acts = [], []
    for run_dir in run_dirs:
        refs.append(json.loads((run_dir / f"{fixture}_reference.json").read_text()))
        acts.append(run_dir / "activations" / fixture)

    def load(i: int, stage: str) -> Any:
        return np.load(acts[i] / f"{stage}.npy", allow_pickle=False)

    if any(r["prompt_token_ids"] != refs[0]["prompt_token_ids"] for r in refs):
        raise SystemExit(f"{fixture}: the runs' prompt ids differ")
    runs = []
    for i, ref in enumerate(refs):
        steps = load(i, "steps.index")
        logits = load(i, "steps.logits")[steps < UOCR2_HEAD_STEPS]
        runs.append(
            {
                "token_ids": ref["token_ids"],
                "n_tokens": len(ref["token_ids"]),
                "logits": logits,
                "vocab_dim": int(logits.shape[1]),
                "decoded_sha256": ref["decoded_text_sha256"],
            }
        )
    rows = min(r["logits"].shape[0] for r in runs)
    for r in runs:
        r["logits"] = r["logits"][:rows]
    envelope = _compute_doc_envelope(runs)
    envelope["token_ids_identical"] = len({json.dumps(r["token_ids"]) for r in refs}) == 1
    envelope["logit_rows_compared"] = rows

    n_layers = len(refs[0]["routing"]["infer_params"])
    layers = []
    for layer in range(n_layers):
        entry: dict[str, Any] = {"layer": layer}

        def stage_of(i: int, name: str) -> Any:
            return load(i, f"prefill.L{layer:02d}.{name}")

        for stage in ("hidden_in", "iq", "ik", "kbar", "attn_pre_o", "attn_out"):
            base = stage_of(0, stage)
            worst = {"max_abs": 0.0, "max_rel": 0.0}
            for i in range(1, len(runs)):
                d = _uocr2_layer_delta(np, base, stage_of(i, stage))
                worst = {k: max(worst[k], d[k]) for k in worst}
            entry[stage] = worst
        # Scores on the entries every run marks remote; the prior on the rows
        # whose top-M window holds the same blocks in every run.
        scores0 = stage_of(0, "scores")
        window0 = [frozenset(r.tolist()) for r in stage_of(0, "topm_idx")]
        prior0 = stage_of(0, "prior")
        scores = {"max_abs": 0.0, "max_rel": 0.0}
        prior = {"max_abs": 0.0, "max_rel": 0.0}
        for i in range(1, len(runs)):
            scores_i = stage_of(i, "scores")
            remote = (scores0 > UOCR2_SCORE_NEG * 0.5) & (scores_i > UOCR2_SCORE_NEG * 0.5)
            d = _uocr2_layer_delta(np, scores0[remote], scores_i[remote])
            scores = {k: max(scores[k], d[k]) for k in scores}
            same = np.array([a == frozenset(b.tolist()) for a, b in zip(window0, stage_of(i, "topm_idx"))], dtype=bool)
            d = _uocr2_layer_delta(np, prior0[same], stage_of(i, "prior")[same])
            prior = {k: max(prior[k], d[k]) for k in prior}
        entry["scores"] = scores
        entry["prior"] = prior
        topi0 = stage_of(0, "topi")
        valid0 = stage_of(0, "valid")
        sets0 = _uocr2_selected_sets(topi0, valid0)
        differing = np.zeros(topi0.shape[0], dtype=bool)
        set_differing = np.zeros(topi0.shape[0], dtype=bool)
        for i in range(1, len(runs)):
            topi_i, valid_i = stage_of(i, "topi"), stage_of(i, "valid")
            differing |= np.any(topi0 != topi_i, axis=-1)
            differing |= np.any(valid0 != valid_i, axis=-1)
            set_differing |= np.array([a != b for a, b in zip(sets0, _uocr2_selected_sets(topi_i, valid_i))], dtype=bool)
        # Rows whose topi/valid arrays differ, slot order included, and the
        # subset whose selected block sets differ.
        entry["routing_rows_differing"] = int(differing.sum())
        entry["routing_set_rows_differing"] = int(set_differing.sum())
        layers.append(entry)
    final0 = load(0, "prefill.final_hidden")
    final = {"max_abs": 0.0, "max_rel": 0.0}
    for i in range(1, len(runs)):
        d = _uocr2_layer_delta(np, final0, load(i, "prefill.final_hidden"))
        final = {k: max(final[k], d[k]) for k in final}
    return {
        "runs": [
            {"dtype": r["provenance"]["dtype"], "threads": r["provenance"]["threads"], "tokens": len(r["token_ids"]), "wall_seconds": r["wall_seconds"]}
            for r in refs
        ],
        "prefill_rows_compared": int(final0.shape[0]),
        "envelope": envelope,
        "layers": layers,
        "final_hidden": final,
    }


def _uocr2_floor(groups: list[str], out: Path) -> int:
    import numpy as np  # noqa: WPS433

    result: dict[str, Any] = {
        "schema_version": 1,
        "generator": "scripts/oracle_nondeterminism_floor.py --uocr2-group",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "groups": {},
    }
    for spec in groups:
        name, _, dirs = spec.partition("=")
        run_dirs = [Path(d) for d in dirs.split(",") if d]
        if not name or len(run_dirs) < 2:
            raise SystemExit(f"--uocr2-group {spec!r}: need NAME=RUN_DIR,RUN_DIR[,...]")
        names = [{p.name.removesuffix("_reference.json") for p in d.glob("*_reference.json")} for d in run_dirs]
        fixtures = sorted(set.intersection(*names))
        if not fixtures:
            raise SystemExit(f"--uocr2-group {name}: the runs share no fixture")
        per_fixture = {}
        for fixture in fixtures:
            cmp = _uocr2_compare_fixture(np, run_dirs, fixture)
            per_fixture[fixture] = cmp
            env = cmp["envelope"]
            print(
                json.dumps(
                    {
                        "test": "uocr2_floor",
                        "group": name,
                        "fixture": fixture,
                        "token_ids_identical": env["token_ids_identical"],
                        "first_divergence_pos": env["first_divergence_pos"],
                        "logit_spread_in_prefix": env["logit_spread_in_prefix"],
                        "max_hidden_rel": max(l["hidden_in"]["max_rel"] for l in cmp["layers"]),
                        "routing_rows_differing": sum(l["routing_rows_differing"] for l in cmp["layers"]),
                        "routing_set_rows_differing": sum(l["routing_set_rows_differing"] for l in cmp["layers"]),
                    }
                ),
                flush=True,
            )
        summary = {
            "fixtures": len(fixtures),
            "all_token_ids_identical": all(c["envelope"]["token_ids_identical"] for c in per_fixture.values()),
            "max_logit_spread_in_prefix": max(c["envelope"]["logit_spread_in_prefix"] for c in per_fixture.values()),
            "max_hidden_in": {
                k: max(l["hidden_in"][k] for c in per_fixture.values() for l in c["layers"]) for k in ("max_abs", "max_rel")
            },
            "max_attn_out": {
                k: max(l["attn_out"][k] for c in per_fixture.values() for l in c["layers"]) for k in ("max_abs", "max_rel")
            },
            "max_final_hidden": {k: max(c["final_hidden"][k] for c in per_fixture.values()) for k in ("max_abs", "max_rel")},
            "routing_rows_differing": sum(l["routing_rows_differing"] for c in per_fixture.values() for l in c["layers"]),
            "routing_set_rows_differing": sum(
                l["routing_set_rows_differing"] for c in per_fixture.values() for l in c["layers"]
            ),
        }
        for stage in ("iq", "ik", "kbar", "scores", "prior", "attn_pre_o"):
            summary[f"max_{stage}"] = {
                k: max(l[stage][k] for c in per_fixture.values() for l in c["layers"]) for k in ("max_abs", "max_rel")
            }
        result["groups"][name] = {"run_dirs": [str(d) for d in run_dirs], "summary": summary, "fixtures": per_fixture}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"test": "uocr2_floor", "event": "wrote", "file": str(out)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.uocr2_group:
        return _uocr2_floor(args.uocr2_group, args.uocr2_out)

    if len(args.thread_counts) < 2:
        raise SystemExit(
            "oracle_nondeterminism_floor: --thread-counts needs >=2 values "
            "(the plan calls for two thread counts to expose BLAS reduction-order "
            "nondeterminism)."
        )
    if args.repeats < 2:
        raise SystemExit(
            "oracle_nondeterminism_floor: --repeats must be >=2 (run-twice, "
            "plan §8.2)."
        )

    # 1. Assert the pinned oracle stack BEFORE touching anything expensive.
    torch, transformers = _assert_pinned_stack()

    # 2. Resolve + validate the model dir and corpus.
    model_dir = _resolve_model_dir(args.model_dir)
    docs = _list_corpus(args.corpus)
    if args.limit > 0:
        docs = docs[: args.limit]

    preset = MODE_PRESETS[args.mode]
    args.out.mkdir(parents=True, exist_ok=True)
    scratch_root = args.out / "_floor_scratch"

    deterministic_enabled = _enable_determinism(torch)

    print(
        f"[floor] mode={args.mode} prompt={args.prompt!r} device={args.device} "
        f"docs={len(docs)} thread_counts={args.thread_counts} "
        f"repeats={args.repeats} det_algos={'on' if deterministic_enabled else 'off'}",
        file=sys.stderr,
    )

    # 3. Provenance (hashes the weights once).
    provenance = _build_provenance(
        torch=torch,
        transformers=transformers,
        model_dir=model_dir,
        preset=preset,
        prompt=args.prompt,
        mode=args.mode,
        device=args.device,
        max_length=args.max_length,
        thread_counts=args.thread_counts,
        repeats=args.repeats,
        deterministic_enabled=deterministic_enabled,
    )

    # 4. Load the model + tokenizer once (reused across every run).
    model, tokenizer = _load_model_and_tokenizer(torch, model_dir, args.device)
    try:
        transformers.set_seed(0)
    except Exception:  # noqa: BLE001 - older/newer API shape
        torch.manual_seed(0)

    # 5. Run the cross-product: for each document, every (thread-count, repeat).
    per_doc_env: dict[str, dict[str, Any]] = {}
    per_doc_runs_meta: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        stem = doc.stem
        runs: list[dict[str, Any]] = []
        runs_meta: list[dict[str, Any]] = []
        for n_threads in args.thread_counts:
            _set_thread_count(torch, n_threads)
            for rep in range(args.repeats):
                scratch = scratch_root / stem / f"t{n_threads}_r{rep}"
                with torch.no_grad():
                    run = _run_once(
                        torch=torch,
                        model=model,
                        tokenizer=tokenizer,
                        doc=doc,
                        out_scratch=scratch,
                        prompt=args.prompt,
                        preset=preset,
                        max_length=args.max_length,
                        max_compare_tokens=args.max_compare_tokens,
                    )
                runs.append(run)
                runs_meta.append(
                    {
                        "thread_count": n_threads,
                        "repeat": rep,
                        "n_tokens": run["n_tokens"],
                        "vocab_dim": run["vocab_dim"],
                        "decoded_sha256": run["decoded_sha256"],
                        "elapsed_seconds": run["elapsed_seconds"],
                    }
                )
                print(
                    f"[floor]   {doc.name} t={n_threads} r={rep}: "
                    f"{run['n_tokens']} tokens, {run['elapsed_seconds']:.1f}s",
                    file=sys.stderr,
                )

        env = _compute_doc_envelope(runs)
        per_doc_env[doc.name] = env
        per_doc_runs_meta[doc.name] = runs_meta
        # Drop the heavy logits arrays now that the envelope is computed.
        for run in runs:
            run.pop("logits", None)
        print(
            f"[floor]   {doc.name}: first_divergence_pos="
            f"{env['first_divergence_pos']} (of {env['common_compared_len']}), "
            f"divergence_rate={env['divergence_rate']}, "
            f"logit_spread_in_prefix={env['logit_spread_in_prefix']}, "
            f"text_identical={env['decoded_text_identical']}",
            file=sys.stderr,
        )

    # 6. Derive corpus-level tolerances from the per-doc envelopes.
    derived = _derive_tolerances(per_doc_env)

    # 7. Emit the committed envelope fixture.
    envelope = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "provenance": provenance,
        "derived_tolerances": derived,
        "per_document": {
            name: {**per_doc_env[name], "runs": per_doc_runs_meta[name]}
            for name in sorted(per_doc_env)
        },
        "n_documents": len(per_doc_env),
    }
    envelope_path = args.out / ENVELOPE_FILE
    envelope_path.write_text(
        json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # 8. Emit tolerances.toml.
    _write_tolerances_toml(
        args.tolerances,
        derived=derived,
        per_doc=per_doc_env,
        provenance=provenance,
    )

    print(
        f"[floor] wrote {envelope_path} and {args.tolerances} "
        f"(L3 logit_tolerance={derived['l3_logit_tolerance']}, "
        f"max_divergence_rate={derived['measured_max_token_divergence_rate']})",
        file=sys.stderr,
    )

    # 9. Structural-nondeterminism note for docs/DISCREPANCIES.md (stderr only;
    #    a human ledgers a DISC-NNN if warranted).
    structural = derived["measured_max_token_divergence_rate"] > 0.0 or any(
        env["common_compared_len"] > 0
        and env["first_divergence_pos"] < env["common_compared_len"]
        for env in per_doc_env.values()
    )
    if structural:
        print(
            "[floor] DISCREPANCIES.md NOTE — the oracle exhibits structural "
            "nondeterminism: at least one document diverges before the end of its "
            "compared prefix (max token divergence_rate="
            f"{derived['measured_max_token_divergence_rate']}, "
            f"min first_divergence_pos={derived['min_first_divergence_pos']}). "
            "L4 'exact' is defined ONLY over each document's reproducible prefix; "
            "ledger this as a DISC-NNN (reference behavior=bf16 tie nondeterminism, "
            "measured impact=the envelope above, kill-switch=n/a, review date set).",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
