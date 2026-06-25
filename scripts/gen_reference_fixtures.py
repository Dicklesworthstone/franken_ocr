#!/usr/bin/env python3
"""Generate reference (golden) fixtures for franken_ocr from the PyTorch oracle.

STATUS: SKELETON / TODO. This script does NOT yet run the model. It defines the
argparse surface and documents exactly what the finished version WILL do; the
actual torch / transformers calls are stubbed with `raise NotImplementedError`.

OFFLINE TOOLING ONLY. franken_ocr's Rust engine NEVER invokes this script and
NEVER unpickles anything. This is a deliberate, human-run, out-of-band step whose
only job is to freeze the reference model's behavior into files the Rust
conformance tests compare against.

WHAT THE FINISHED VERSION WILL DO
---------------------------------
1. Pin the exact reference stack the Unlimited-OCR README specifies:
       torch == 2.10.0
       transformers == 4.57.1
   and assert at runtime that the installed versions match (fail loudly on drift
   — fixtures generated against a different stack are not comparable).

2. Load the Baidu Unlimited-OCR model + tokenizer from $FOCR_MODEL_DIR
   (see scripts/fetch_model.sh) in bf16, eval mode, deterministic /
   no-sampling generation config matching what `focr` will replicate.
   NOTE (plan OQ-17, §8.1): the official model.infer() path is CUDA-oriented
   (.cuda() + CUDA autocast), so the CORRECTNESS oracle runs on a CUDA host
   (device='cuda'); a CPU-patched run (device='cpu', autocast off) is a SEPARATE
   step that must first be PROVEN to reproduce the GPU oracle's tokens within the
   nondeterminism floor. The golden fixtures (correctness) never depend on CPU HF.

3. For each input document in the fixture corpus:
     a. Run model.infer() (the reference end-to-end path) and dump the
        END-TO-END golden output (parsed text / markdown + structured JSON) to
        tests/fixtures/native/<doc>_reference.json — the bar `focr ocr --json`
        must match (after canonicalization: strip timing, sort bboxes).
     b. With forward hooks, capture PER-LAYER / per-stage activations at the
        seams the Rust engine is unit-tested against — DeepEncoder (SAM out,
        CLIP out, fused-2048, projector-1280), decoder per-layer hidden states,
        router logits / top-6 selections, lm_head logits — and dump each as a
        .npy under tests/fixtures/native/activations/<doc>/<stage>.npy for
        bit-/tolerance-level differential tests.

4. Record provenance alongside every artifact: model sha256, torch/transformers
   versions, generation config, and a timestamp, so a fixture is auditable and a
   stale one is detectable.

These fixtures are the reference oracle for docs/DISCREPANCIES.md (measured
divergence) and the int8/int4 accuracy curves — they are the source of truth the
quantized engine is held against.

Requires (when implemented): python3, torch==2.10.0, transformers==4.57.1,
numpy, safetensors. Install into an isolated venv; this never runs in CI inference.

Usage (skeleton):
    python3 gen_reference_fixtures.py --model-dir DIR --corpus DIR --out DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Pins the finished version MUST assert against (Unlimited-OCR README runtime).
PIN_TORCH = "2.10.0"
PIN_TRANSFORMERS = "4.57.1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gen_reference_fixtures.py",
        description="Dump Unlimited-OCR reference fixtures (golden outputs + "
        "per-layer activations) for franken_ocr conformance tests. SKELETON.",
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
        help="Dir of input document images to run through the oracle. PDF fixtures "
        "must be rasterized explicitly and kept separate from v1 native image tests.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/native"),
        help="Output dir for golden outputs (.json) and activations (.npy).",
    )
    p.add_argument(
        "--activations",
        action="store_true",
        help="Also dump per-layer/per-stage activations (.npy), not just the "
        "end-to-end golden output.",
    )
    return p.parse_args(argv)


def _assert_pinned_stack() -> None:
    """Fail loudly unless torch/transformers match the pinned reference stack.

    Fixtures generated against a different stack are not comparable, so the
    finished version refuses to proceed on a version mismatch.
    """
    raise NotImplementedError(
        f"TODO: import torch / transformers and assert torch=={PIN_TORCH}, "
        f"transformers=={PIN_TRANSFORMERS} before generating any fixtures."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- SKELETON: nothing below is implemented yet. ---
    print("gen_reference_fixtures.py — SKELETON / TODO; no fixtures generated.",
          file=sys.stderr)
    print(f"  would assert: torch=={PIN_TORCH}, transformers=={PIN_TRANSFORMERS}",
          file=sys.stderr)
    print(f"  model-dir:    {args.model_dir or '$FOCR_MODEL_DIR'}", file=sys.stderr)
    print(f"  corpus:       {args.corpus}", file=sys.stderr)
    print(f"  out:          {args.out}", file=sys.stderr)
    print(f"  activations:  {args.activations}", file=sys.stderr)
    print(file=sys.stderr)
    print("TODO:", file=sys.stderr)
    print("  1. _assert_pinned_stack()", file=sys.stderr)
    print("  2. load model + tokenizer (bf16, eval, deterministic; oracle device=cuda — CPU run is a separate, proven step, OQ-17)", file=sys.stderr)
    print("  3. per doc: model.infer() -> <doc>_reference.json (end-to-end golden)",
          file=sys.stderr)
    print("  4. forward hooks -> per-stage activations/<doc>/<stage>.npy", file=sys.stderr)
    print("  5. write provenance (model sha256, versions, gen config, timestamp)",
          file=sys.stderr)

    # Make the skeleton's assertion path reachable/visible without running torch.
    try:
        _assert_pinned_stack()
    except NotImplementedError as e:
        print(f"\nnot implemented: {e}", file=sys.stderr)

    return 1  # non-zero: the skeleton did not produce fixtures.


if __name__ == "__main__":
    raise SystemExit(main())
