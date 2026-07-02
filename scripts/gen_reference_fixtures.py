#!/usr/bin/env python3
"""Generate reference (golden) fixtures for franken_ocr from the PyTorch oracle.

This is the COMPLETE, runnable, offline fixture generator. It is OFFLINE TOOLING
ONLY: franken_ocr's Rust engine NEVER invokes this script and NEVER unpickles
anything. This is a deliberate, human-run, out-of-band step whose only job is to
freeze the reference model's behavior into files the Rust conformance tests
compare against.

It CANNOT run as the CORRECTNESS oracle on a machine without a CUDA GPU and the
6.67 GB weights — per plan OQ-17 (docs/truth-pack/oq/preprocess-infer.md) the
official `model.infer()` path hard-codes `.cuda()` +
`torch.autocast("cuda", dtype=bfloat16)`, so the CORRECTNESS oracle is GPU-only.
A CPU-patched run (`--device cpu --cpu-patch`, the same shims
scripts/baseline/run_baidu_reference.py proved on the 20-page corpus) is a
SEPARATE step that must first be PROVEN to reproduce the GPU oracle's tokens
within the nondeterminism floor; its provenance records
`oracle_is_correctness_golden: false` and the golden fixtures (correctness)
never depend on CPU HF.

WHAT THIS DOES
--------------
1. Pin the exact reference stack the Unlimited-OCR README specifies:
       torch == 2.10.0
       transformers == 4.57.1
   and assert at runtime that the installed versions match (fail loudly on drift
   — fixtures generated against a different stack are not comparable).

2. Load the Baidu Unlimited-OCR model + tokenizer from $FOCR_MODEL_DIR
   (see scripts/fetch_model.sh) in bf16, eval mode, on device='cuda'.

3. For each input document in the fixture corpus:
     a. Run model.infer() (the reference end-to-end path) and dump the
        END-TO-END golden output (decoded text / markdown + structured JSON) to
        <out>/<doc>_reference.json — the bar `focr ocr --json` must match (after
        canonicalization: strip timing, sort bboxes).
     b. With forward hooks, capture PER-STAGE activations at the seams the Rust
        engine is unit-tested against — DeepEncoder (SAM out, CLIP out,
        projector-1280), decoder per-layer hidden states, and lm_head logits —
        PLUS the seam INPUTS the ladder needs to isolate the remaining rungs
        (bd-3s7v): `sam_input` (the exact preprocessed tensor entering SAM;
        unlocks L0-exact and SAM/CLIP isolation) and `inputs_embeds` (the
        post-splice hidden entering decoder layer 0; unlocks decoder-from-embeds
        isolation) — and dump each as a .npy under
        <out>/activations/<doc>/<stage>.npy for bit-/tolerance-level
        differential tests.
     c. Wrap model.generate to freeze the decoded TOKEN-ID stream + decode
        metadata into the golden JSON (`token_stream`, bd-3s7v) — the L4
        exact-token bar — without changing any generate() argument.

4. Record provenance alongside every artifact: model sha256, the pinned HF
   commit, torch/transformers versions, generation config, and a timestamp, so a
   fixture is auditable and a stale one is detectable.

These fixtures are the reference oracle for docs/DISCREPANCIES.md (measured
divergence) and the int8/int4 accuracy curves — the source of truth the quantized
engine is held against.

Requires: python3, torch==2.10.0, transformers==4.57.1, numpy, safetensors,
Pillow. Install into an isolated venv; this never runs in CI inference.

Usage:
    FOCR_MODEL_DIR=/path/to/model \\
        python3 gen_reference_fixtures.py --corpus tests/fixtures/corpus \\
                                          --out tests/fixtures/native --activations

    # Establish the oracle nondeterminism floor (plan §8.1) by running twice and
    # diffing — pass --run-tag to keep both runs side by side:
    python3 gen_reference_fixtures.py ... --run-tag a
    python3 gen_reference_fixtures.py ... --run-tag b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ─────────────────────────────────────────────────────────────────────────────
# Pins the finished version MUST assert against (Unlimited-OCR README runtime).
# Source: docs/truth-pack/PINNED_SOURCES.md (the README runtime pin, NOT the
# config.json export tag transformers_version=4.46.3).
# ─────────────────────────────────────────────────────────────────────────────
PIN_TORCH = "2.10.0"
PIN_TRANSFORMERS = "4.57.1"

# The HF repo + the immutable commit the whole truth-pack is pinned to.
HF_REPO = "baidu/Unlimited-OCR"
HF_COMMIT = "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"
GITHUB_REPO = "https://github.com/baidu/Unlimited-OCR"
GITHUB_COMMIT = "7e98affeacba24e95562fbaa234ddb89b856874a"

# Canonical filenames inside $FOCR_MODEL_DIR (see scripts/fetch_model.sh).
WEIGHTS_FILE = "model-00001-of-000001.safetensors"
INDEX_FILE = "model.safetensors.index.json"
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "config.json"

# Recommended per-mode generation config (docs/truth-pack EXISTING_..._STRUCTURE.md
# §8 SPEC-100..103, OQ-18). Greedy (temperature=0 -> do_sample=False), the sliding
# window no-repeat-ngram blocker on, max_length 32768.
MODE_PRESETS: dict[str, dict[str, Any]] = {
    # Single image, Gundam tiling (base_size=1024, image_size=640, crop_mode=True).
    "gundam": dict(
        base_size=1024,
        image_size=640,
        crop_mode=True,
        no_repeat_ngram_size=35,
        ngram_window=128,
    ),
    # Single image, base (no crop): base_size=1024, image_size=1024, crop_mode=False.
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

# Image suffixes we treat as corpus documents (v1 is image-only; PDFs are
# rasterized out-of-band and dropped here as separate images).
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Stop string appended by the tokenizer at the end of the decoded output.
EOS_STOP_STRING = "<｜end▁of▁sentence｜>"
DEFAULT_RNG_SEED = 0
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gen_reference_fixtures.py",
        description="Dump Unlimited-OCR reference fixtures (golden outputs + "
        "per-stage activations) for franken_ocr conformance tests.",
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
        help="Dir of input document images to run through the oracle. PDF "
        "fixtures must be rasterized explicitly and kept separate from v1 native "
        "image tests.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/native"),
        help="Output dir for golden outputs (.json) and activations (.npy).",
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
        help="Free-text prompt passed to model.infer() (must contain <image>).",
    )
    p.add_argument(
        "--activations",
        action="store_true",
        help="Also dump per-stage activations (.npy), not just the end-to-end "
        "golden output. Adds forward hooks (SAM/CLIP/projector/per-layer/"
        "lm_head) captured on the prefill pass.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Generation cap (matches the reference max_length).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N corpus documents (0 = no limit). Useful to "
        "establish the nondeterminism floor cheaply.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help="RNG seed captured into every fixture replay contract.",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional suffix on the run-provenance file so two runs (e.g. for "
        "the nondeterminism floor) do not clobber each other.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for the oracle. Per OQ-17 the correctness oracle is "
        "CUDA-only; a non-cuda value is allowed ONLY for the separate, "
        "must-be-proven CPU-equivalence experiment, never for golden fixtures.",
    )
    p.add_argument(
        "--cpu-patch",
        action="store_true",
        help="Monkeypatch the reference's hard-coded .cuda()/autocast('cuda') "
        "to no-ops so infer() runs with --device cpu (the separate "
        "CPU-equivalence path, OQ-17; provenance stays "
        "oracle_is_correctness_golden=false). Same shims "
        "scripts/baseline/run_baidu_reference.py proved on the 20-page corpus. "
        "Rejected with --device cuda.",
    )
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Stack assertions
# ─────────────────────────────────────────────────────────────────────────────
def _assert_pinned_stack() -> tuple[Any, Any]:
    """Import + assert torch/transformers match the pinned reference stack.

    Fixtures generated against a different stack are NOT comparable, so we refuse
    to proceed on a version mismatch. Returns the imported torch + transformers
    modules so callers do not re-import.
    """
    try:
        import torch  # noqa: WPS433 (deliberate late import — offline tool)
        import transformers  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "gen_reference_fixtures: torch / transformers are not installed. "
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
            "gen_reference_fixtures: pinned-stack mismatch — fixtures generated "
            "against a different stack are not comparable. Found: "
            + ", ".join(problems)
            + ". Recreate the venv with the exact pins from "
            "docs/truth-pack/PINNED_SOURCES.md."
        )
    return torch, transformers


def _install_cpu_patches(torch: Any) -> None:
    """Make the CUDA-hard-coded `infer()` runnable on CPU (--cpu-patch only).

    The reference modeling code calls `.cuda()` and `torch.autocast("cuda",
    dtype=bfloat16)` unconditionally (OQ-17). These are the exact shims
    scripts/baseline/run_baidu_reference.py used for the proven CPU baseline:
    `.cuda()` becomes identity, the cuda autocast becomes a nullcontext (the
    graph stays in the loaded bf16 dtype), and CUDA is reported unavailable so
    flash-attn style branches stay off.
    """
    import contextlib  # noqa: WPS433 (only needed on the patched path)

    torch.Tensor.cuda = lambda self, *a, **k: self
    _orig_autocast = torch.autocast

    class _AutocastShim(contextlib.AbstractContextManager):
        def __init__(self, device_type: str, *a: Any, **k: Any):
            if device_type == "cuda":
                self._cm: Any = contextlib.nullcontext()
            else:
                self._cm = _orig_autocast(device_type, *a, **k)

        def __enter__(self) -> Any:
            return self._cm.__enter__()

        def __exit__(self, *exc: Any) -> Any:
            return self._cm.__exit__(*exc)

    torch.autocast = _AutocastShim
    torch.cuda.is_available = lambda: False


def _resolve_model_dir(cli_dir: Path | None) -> Path:
    raw = cli_dir or os.environ.get("FOCR_MODEL_DIR")
    if not raw:
        raise SystemExit(
            "gen_reference_fixtures: no model dir. Pass --model-dir or set "
            "$FOCR_MODEL_DIR (see scripts/fetch_model.sh)."
        )
    model_dir = Path(raw).expanduser()
    if not model_dir.is_dir():
        raise SystemExit(f"gen_reference_fixtures: model dir not found: {model_dir}")
    missing = [
        name
        for name in (WEIGHTS_FILE, TOKENIZER_FILE, CONFIG_FILE)
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise SystemExit(
            f"gen_reference_fixtures: model dir {model_dir} is missing "
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
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_model_and_tokenizer(torch: Any, model_dir: Path, device: str):
    """Load Unlimited-OCR + tokenizer in bf16, eval, on `device`.

    Mirrors the README runtime (trust_remote_code=True, use_safetensors=True,
    torch_dtype=bfloat16, .eval().cuda()). trust_remote_code is required because
    the model class (UnlimitedOCRForCausalLM) lives in the repo's
    modeling_unlimitedocr.py — this is offline tooling the human runs knowingly.
    """
    from transformers import AutoModel, AutoTokenizer  # noqa: WPS433

    print(f"[fixtures] loading tokenizer from {model_dir} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )

    print(
        f"[fixtures] loading model from {model_dir} (bf16, {device}) ...",
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
                "gen_reference_fixtures: --device cuda but no CUDA device is "
                "available. The correctness oracle is GPU-only (OQ-17). Run on a "
                "CUDA host, or use --device cpu ONLY for the separate "
                "CPU-equivalence experiment (never for golden fixtures)."
            )
        model = model.cuda()
    else:
        model = model.to(device)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Activation hooks
# ─────────────────────────────────────────────────────────────────────────────
class ActivationCapture:
    """Register forward hooks that capture per-stage activations on the PREFILL
    pass only, then save them as .npy.

    The seams (docs/truth-pack EXISTING_UNLIMITED_OCR_STRUCTURE.md):
      - model.sam_model     -> SAM ViT-B output  (SPEC-040..046)
      - model.vision_model  -> CLIP-L output     (SPEC-047..050)
      - model.projector     -> 2048->1280 linear (SPEC-051..052)
      - model.layers[i]     -> per-decoder-layer hidden states (SPEC-070..072)
      - lm_head             -> logits            (SPEC-081)

    Seam INPUTS (forward_pre_hooks, bd-3s7v) — the tensors the parity ladder
    needs to ISOLATE the seams the output dumps alone cannot:
      - sam_input     : the exact preprocessed tensor entering model.sam_model
                        (L0-exact; vision_sam::forward(sam_input) -> sam_output;
                        vision_clip::forward(sam_input, sam_output) -> clip_output)
      - inputs_embeds : the post-splice hidden entering model.layers[0] on the
                        PREFILL pass (embedding + vision masked_scatter already
                        applied; decoder::forward(inputs_embeds) -> final hidden)

    `infer()` calls `generate()`, which runs many forward passes (one prefill +
    one per generated token). We only want the prefill activations (the seam the
    Rust unit tests target), so every hook records the FIRST call whose primary
    output has sequence length > 1 (prefill) and ignores subsequent single-token
    decode calls. SAM/CLIP/projector run once per image inside the prefill, so we
    keep their first invocation. lm_head additionally has its full prefill logits
    recorded (all positions), the load-bearing seam for L4 (logits parity).
    """

    def __init__(self, torch: Any, model: Any):
        self.torch = torch
        self.model = model
        self._handles: list[Any] = []
        self._captured: dict[str, "Any"] = {}  # stage -> numpy array
        self._seen_prefill = False

    # --- helpers ---------------------------------------------------------------
    def _to_numpy(self, t: Any) -> Any:
        import numpy as np  # noqa: WPS433

        # bf16 has no numpy dtype; upcast to float32 for storage + comparison.
        t = t.detach()
        if t.dtype == self.torch.bfloat16:
            t = t.to(self.torch.float32)
        return t.to("cpu").numpy().astype(np.float32, copy=False)

    @staticmethod
    def _primary_tensor(out: Any) -> Any | None:
        """Pull the load-bearing tensor out of a hook's `output`."""
        import torch as _t  # local alias only for isinstance checks

        if isinstance(out, _t.Tensor):
            return out
        if isinstance(out, (tuple, list)) and out:
            head = out[0]
            if isinstance(head, _t.Tensor):
                return head
        # transformers ModelOutput-like (has last_hidden_state)
        lhs = getattr(out, "last_hidden_state", None)
        if isinstance(lhs, _t.Tensor):
            return lhs
        return None

    @staticmethod
    def _primary_input(args: Any, kwargs: Any) -> Any | None:
        """Pull the load-bearing tensor out of a pre-hook's `(args, kwargs)`.

        The reference calls SAM positionally with the pixel tensor and the
        decoder layers with `hidden_states` (positionally or by keyword), so
        scan positional args first, then the `hidden_states` kwarg, then any
        tensor kwarg.
        """
        import torch as _t  # local alias only for isinstance checks

        for a in args or ():
            if isinstance(a, _t.Tensor):
                return a
        if kwargs:
            hs = kwargs.get("hidden_states")
            if isinstance(hs, _t.Tensor):
                return hs
            for v in kwargs.values():
                if isinstance(v, _t.Tensor):
                    return v
        return None

    def _store_once(self, stage: str, tensor: Any) -> None:
        if stage in self._captured:
            return
        self._captured[stage] = self._to_numpy(tensor)

    # --- hook factories --------------------------------------------------------
    def _make_input_hook(
        self, stage: str, *, prefill_only: bool = False
    ) -> Callable[..., None]:
        """forward_pre_hook capturing a module's INPUT tensor (bd-3s7v)."""

        def hook(_module: Any, args: Any, kwargs: Any = None) -> None:
            t = self._primary_input(args, kwargs)
            if t is None:
                return
            # Decoder pre-hooks must skip the seq_len == 1 decode steps (same
            # gate as the layer OUTPUT hooks); vision inputs have no causal seq
            # dim, so they record the first invocation only.
            if prefill_only and not (t.dim() >= 2 and t.shape[-2] > 1):
                return
            self._store_once(stage, t)

        return hook

    def _make_stage_hook(self, stage: str) -> Callable[..., None]:
        def hook(_module: Any, _inp: Any, out: Any) -> None:
            t = self._primary_tensor(out)
            if t is None:
                return
            # SAM/CLIP/projector run on image tensors (no causal seq dim concept);
            # record the first invocation only.
            self._store_once(stage, t)

        return hook

    def _make_layer_hook(self, idx: int) -> Callable[..., None]:
        stage = f"decoder_layer_{idx:02d}_hidden"

        def hook(_module: Any, _inp: Any, out: Any) -> None:
            t = self._primary_tensor(out)
            if t is None:
                return
            # Only the prefill pass (seq_len > 1). Decode steps have seq_len == 1.
            if t.dim() >= 2 and t.shape[-2] > 1:
                self._store_once(stage, t)

        return hook

    def _make_lm_head_hook(self) -> Callable[..., None]:
        def hook(_module: Any, _inp: Any, out: Any) -> None:
            t = self._primary_tensor(out)
            if t is None:
                return
            # Prefill logits over all positions (seq_len > 1).
            if t.dim() >= 2 and t.shape[-2] > 1:
                self._store_once("lm_head_logits", t)
                self._seen_prefill = True

        return hook

    # --- public API ------------------------------------------------------------
    def register(self) -> None:
        m = self.model
        # The decoder/vision submodules hang off `model.model` (UnlimitedOCRModel),
        # the lm_head off the top-level CausalLM. Resolve defensively.
        inner = getattr(m, "model", m)

        sam = getattr(inner, "sam_model", None)
        if sam is not None:
            self._handles.append(
                sam.register_forward_pre_hook(
                    self._make_input_hook("sam_input"), with_kwargs=True
                )
            )
            self._handles.append(
                sam.register_forward_hook(self._make_stage_hook("sam_output"))
            )
        clip = getattr(inner, "vision_model", None)
        if clip is not None:
            self._handles.append(
                clip.register_forward_hook(self._make_stage_hook("clip_output"))
            )
        projector = getattr(inner, "projector", None)
        if projector is not None:
            self._handles.append(
                projector.register_forward_hook(
                    self._make_stage_hook("projector_output")
                )
            )
        layers = getattr(inner, "layers", None)
        if layers is not None:
            for idx, layer in enumerate(layers):
                if idx == 0:
                    # The post-splice hidden entering layer 0 == the decoder's
                    # inputs_embeds (embedding + vision masked_scatter applied).
                    self._handles.append(
                        layer.register_forward_pre_hook(
                            self._make_input_hook("inputs_embeds", prefill_only=True),
                            with_kwargs=True,
                        )
                    )
                self._handles.append(
                    layer.register_forward_hook(self._make_layer_hook(idx))
                )
        lm_head = getattr(m, "lm_head", None)
        if lm_head is not None:
            self._handles.append(
                lm_head.register_forward_hook(self._make_lm_head_hook())
            )

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def dump(self, out_dir: Path) -> dict[str, dict[str, Any]]:
        """Write every captured activation as <out_dir>/<stage>.npy.

        Returns a manifest {stage: {shape, dtype, sha256, file}} for provenance.
        """
        import numpy as np  # noqa: WPS433

        out_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, dict[str, Any]] = {}
        for stage, arr in sorted(self._captured.items()):
            fpath = out_dir / f"{stage}.npy"
            np.save(fpath, arr, allow_pickle=False)
            manifest[stage] = {
                "file": fpath.name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "file_sha256": _sha256_file(fpath),
            }
        return manifest

    def __enter__(self) -> "ActivationCapture":
        self.register()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Token-stream capture (bd-3s7v) — the L4 exact-token bar.
# ─────────────────────────────────────────────────────────────────────────────
class TokenStreamCapture:
    """Wrap `model.generate` to freeze the token-id stream + decode metadata.

    `infer()` calls `generate()` exactly once per document with `input_ids`
    (the prompt, image-pad tokens included) and receives the FULL sequence ids
    back; recording both sides gives L4 its exact greedy token bar without
    re-deriving ids from decoded text (lossy under the stop-string strip). The
    wrapper passes every argument through UNCHANGED — the decode itself is
    untouched.
    """

    # JSON-safe scalars copied verbatim into decode metadata when present.
    _META_KWARGS = (
        "max_length",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "eos_token_id",
        "pad_token_id",
        "no_repeat_ngram_size",
        "use_cache",
    )

    def __init__(self, torch: Any, model: Any):
        self.torch = torch
        self.model = model
        self._orig: Any = None
        self.calls: list[dict[str, Any]] = []

    def _ids(self, t: Any) -> list[int] | None:
        if not isinstance(t, self.torch.Tensor):
            return None
        arr = t.detach().to("cpu")
        if arr.dim() == 2:  # [batch, seq] — infer() always runs batch 1
            arr = arr[0]
        return [int(x) for x in arr.tolist()]

    def register(self) -> None:
        self._orig = self.model.generate

        def wrapped(*g_args: Any, **g_kwargs: Any) -> Any:
            out = self._orig(*g_args, **g_kwargs)
            record: dict[str, Any] = {
                "prompt_ids": self._ids(g_kwargs.get("input_ids")),
                "generate_kwargs": {
                    k: g_kwargs[k]
                    for k in self._META_KWARGS
                    if k in g_kwargs
                    and isinstance(g_kwargs[k], (bool, int, float, str, type(None)))
                },
                # infer() routes the sliding-window ngram blocker through a
                # custom LogitsProcessor, invisible in the kwarg scalars.
                "has_logits_processor": "logits_processor" in g_kwargs,
            }
            mask = g_kwargs.get("images_seq_mask")
            if isinstance(mask, self.torch.Tensor):
                record["images_seq_mask_true_count"] = int(mask.sum().item())
            seq = out.sequences if hasattr(out, "sequences") else out
            record["full_sequence_ids"] = self._ids(seq)
            self.calls.append(record)
            return out

        self.model.generate = wrapped

    def remove(self) -> None:
        if self._orig is not None:
            self.model.generate = self._orig
            self._orig = None

    def golden_block(self) -> dict[str, Any] | None:
        """The `token_stream` block for the golden JSON. `None` if generate()
        never ran (a loud absence — the golden then shows the decode was never
        reached, not an empty stream)."""
        if not self.calls:
            return None
        call = self.calls[0]
        prompt = call.get("prompt_ids")
        full = call.get("full_sequence_ids")
        generated: list[int] | None = None
        if (
            full is not None
            and prompt is not None
            and full[: len(prompt)] == prompt
        ):
            generated = full[len(prompt) :]

        def _sha(ids: list[int] | None) -> str | None:
            # Over the compact JSON list ("[1,2,3]") — platform/endian-neutral.
            if ids is None:
                return None
            return hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode("ascii")
            ).hexdigest()

        return {
            "schema_version": 1,
            "captured_from": "model.generate wrapper (ids as returned; prompt = input_ids kwarg)",
            "n_generate_calls": len(self.calls),
            "prompt_ids": prompt,
            "n_prompt": len(prompt) if prompt is not None else None,
            "generated_ids": generated,
            "n_generated": len(generated) if generated is not None else None,
            "n_full": len(full) if full is not None else None,
            "prompt_ids_sha256": _sha(prompt),
            "generated_ids_sha256": _sha(generated),
            "ids_sha256_domain": "sha256 over json.dumps(ids, separators=(',', ':'))",
            "decode_metadata": {
                k: call[k]
                for k in (
                    "generate_kwargs",
                    "has_logits_processor",
                    "images_seq_mask_true_count",
                )
                if k in call
            },
        }

    def __enter__(self) -> "TokenStreamCapture":
        self.register()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Per-document oracle run
# ─────────────────────────────────────────────────────────────────────────────
def _list_corpus(corpus: Path) -> list[Path]:
    if not corpus.is_dir():
        raise SystemExit(
            f"gen_reference_fixtures: corpus dir not found: {corpus}. Populate it "
            "with document images (see tests/fixtures/corpus)."
        )
    docs = sorted(
        p
        for p in corpus.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        # Skip hidden files — macOS writes AppleDouble sidecars ("._page.png",
        # not decodable images) next to every file on external volumes.
        and not p.name.startswith(".")
    )
    if not docs:
        raise SystemExit(
            f"gen_reference_fixtures: no images ({', '.join(sorted(IMAGE_SUFFIXES))}) "
            f"found in corpus dir {corpus}."
        )
    return docs


def _image_meta(doc: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageOps  # noqa: WPS433

        with Image.open(doc) as im:
            im = ImageOps.exif_transpose(im)  # match the model's load_image
            width, height = im.size
            pil_mode = im.mode
        return {"width": width, "height": height, "pil_mode": pil_mode}
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort
        return {"error": f"could not read image dims: {exc}"}


def _run_one(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    doc: Path,
    out_dir: Path,
    prompt: str,
    preset: dict[str, Any],
    max_length: int,
    want_activations: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Run model.infer() on one document, dump golden JSON + (optional)
    activations, return a per-doc summary record."""
    stem = doc.stem
    doc_out_path = out_dir / "_infer_out" / stem  # save_results artifacts land here
    doc_out_path.mkdir(parents=True, exist_ok=True)

    capture: ActivationCapture | None = None
    if want_activations:
        capture = ActivationCapture(torch, model)
        capture.register()
    # Token stream is captured on EVERY run (not only --activations): it is part
    # of the e2e golden record (the L4 bar) and costs nothing extra.
    tokens = TokenStreamCapture(torch, model)
    tokens.register()

    t0 = time.time()
    try:
        # model.infer() runs the full reference end-to-end path (preprocess ->
        # tile -> vision tower -> connector -> R-SWA decoder -> greedy generate ->
        # postprocess) and returns the decoded markdown/text. save_results=True
        # also writes result.md / result_with_boxes.jpg / images/ under
        # output_path, which we keep as golden side-artifacts.
        decoded = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(doc),
            output_path=str(doc_out_path),
            base_size=preset["base_size"],
            image_size=preset["image_size"],
            crop_mode=preset["crop_mode"],
            max_length=max_length,
            no_repeat_ngram_size=preset["no_repeat_ngram_size"],
            ngram_window=preset["ngram_window"],
            temperature=0.0,  # greedy / deterministic oracle
            save_results=True,
        )
    finally:
        tokens.remove()
        if capture is not None:
            capture.remove()
    elapsed = time.time() - t0
    token_stream = tokens.golden_block()

    # `infer` may return None (it primarily writes files). Capture decoded text
    # both from its return value and from the written result.md, preferring the
    # explicit return.
    decoded_text = decoded if isinstance(decoded, str) else None
    result_md = doc_out_path / "result.md"
    md_text = result_md.read_text(encoding="utf-8") if result_md.is_file() else None
    if decoded_text is None:
        decoded_text = md_text

    activations_manifest: dict[str, Any] = {}
    if capture is not None:
        act_dir = out_dir / "activations" / stem
        activations_manifest = capture.dump(act_dir)

    decoded_text_sha256 = (
        hashlib.sha256(decoded_text.encode("utf-8")).hexdigest()
        if decoded_text is not None
        else None
    )
    replay_contract = {
        "schema_version": 1,
        "rng_seed": provenance["determinism"]["seed"],
        "requires_cuda": provenance["device"] == "cuda",
        "expected_prefix_kind": "full_decoded_text",
        "expected_prefix_chars": len(decoded_text or ""),
        "expected_prefix_sha256": decoded_text_sha256,
        "expected_decoded_text_sha256": decoded_text_sha256,
        "replay_command_argv": provenance["command_argv"],
    }

    golden = {
        "schema_version": 1,
        "doc": doc.name,
        "image": _image_meta(doc),
        "mode": preset,
        "prompt": prompt,
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "max_length": max_length,
            "no_repeat_ngram_size": preset["no_repeat_ngram_size"],
            "ngram_window": preset["ngram_window"],
            "eos_stop_string": EOS_STOP_STRING,
        },
        # The END-TO-END golden text the Rust `focr ocr --json` must match after
        # canonicalization (strip timing; sort bboxes). Timing is intentionally
        # NOT part of the comparison surface.
        "decoded_text": decoded_text,
        "decoded_text_sha256": decoded_text_sha256,
        # The exact token-id stream generate() produced (bd-3s7v) — L4 compares
        # the engine's greedy ids against this, id-for-id, over the §2
        # reproducible prefix.
        "token_stream": token_stream,
        "deterministic_replay": replay_contract,
        "result_md_present": md_text is not None,
        "activations": activations_manifest,
        "non_comparable": {
            # Excluded from parity comparison; recorded for human audit only.
            "elapsed_seconds": round(elapsed, 4),
        },
        "provenance": provenance,
    }

    golden_path = out_dir / f"{stem}_reference.json"
    golden_path.write_text(
        json.dumps(golden, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    n_generated = (token_stream or {}).get("n_generated")
    print(
        f"[fixtures]   {doc.name}: {len(decoded_text or '')} chars, "
        f"{n_generated} generated ids, "
        f"{len(activations_manifest)} activations, {elapsed:.1f}s "
        f"-> {golden_path}",
        file=sys.stderr,
    )
    return {
        "doc": doc.name,
        "golden": golden_path.name,
        "decoded_text_sha256": golden["decoded_text_sha256"],
        "n_activations": len(activations_manifest),
        "n_generated_ids": n_generated,
        "elapsed_seconds": round(elapsed, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────
def _build_provenance(
    *,
    torch: Any,
    transformers: Any,
    model_dir: Path,
    preset: dict[str, Any],
    prompt: str,
    mode: str,
    device: str,
    cpu_patch: bool,
    max_length: int,
    determinism: dict[str, Any],
    command_argv: list[str],
) -> dict[str, Any]:
    """Audit record written into every fixture. The expensive bit (the model
    sha256) streams the 6.67 GB shard once."""
    weights = model_dir / WEIGHTS_FILE
    print(
        f"[fixtures] hashing {weights.name} (this reads the full 6.67 GB once) ...",
        file=sys.stderr,
    )
    model_sha = _sha256_file(weights)

    # Expected total_size from the index for a cross-check (no download).
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
        "generator": "scripts/gen_reference_fixtures.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hf_repo": HF_REPO,
        "hf_commit": HF_COMMIT,
        "github_repo": GITHUB_REPO,
        "github_commit": GITHUB_COMMIT,
        "command_argv": command_argv,
        "exact_command": " ".join(shlex.quote(arg) for arg in command_argv),
        "model_dir": str(model_dir),
        "model_weights_file": WEIGHTS_FILE,
        "model_weights_sha256": model_sha,
        "model_weights_bytes": weights.stat().st_size,
        "model_index_total_size": index_total,
        "device": device,
        # True ⇒ infer() ran through the .cuda()/autocast no-op shims (the
        # separate CPU-equivalence path) — never a correctness golden.
        "cpu_patched_infer": cpu_patch,
        "cuda": cuda_info,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "pinned_torch": PIN_TORCH,
        "pinned_transformers": PIN_TRANSFORMERS,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "determinism": determinism,
        "generation_config": {
            "mode": mode,
            "prompt": prompt,
            "max_length": max_length,
            "temperature": 0.0,
            "do_sample": False,
            **preset,
        },
        # OQ-17: the correctness oracle is GPU/bf16; a non-cuda device is the
        # separate, must-be-proven experiment and its fixtures are NOT golden.
        "oracle_is_correctness_golden": device == "cuda",
    }


def _provenance_markdown(manifest: dict[str, Any]) -> str:
    provenance = manifest["provenance"]
    lines = [
        "# Oracle Fixture Provenance",
        "",
        "Generated by `scripts/gen_reference_fixtures.py`.",
        "",
        "## Pinned Stack",
        "",
        f"- torch=={provenance['pinned_torch']}",
        f"- transformers=={provenance['pinned_transformers']}",
        f"- Hugging Face commit: `{provenance['hf_commit']}`",
        f"- GitHub commit: `{provenance['github_commit']}`",
        f"- Exact command: `{provenance['exact_command']}`",
        f"- RNG seed: `{provenance['determinism']['seed']}`",
        f"- Model weights SHA-256: `{provenance['model_weights_sha256']}`",
        f"- Correctness golden: `{provenance['oracle_is_correctness_golden']}`",
        "",
        "## Documents",
        "",
        "| Document | Golden JSON | Decoded text SHA-256 | Activations |",
        "|----------|-------------|----------------------|-------------|",
    ]
    for doc in manifest["documents"]:
        lines.append(
            "| {doc} | `{golden}` | `{sha}` | {acts} |".format(
                doc=doc["doc"],
                golden=doc["golden"],
                sha=doc["decoded_text_sha256"],
                acts=doc["n_activations"],
            )
        )
    return "\n".join(lines) + "\n"


def _apply_determinism(torch: Any, transformers: Any, seed: int) -> dict[str, Any]:
    """Best-effort deterministic setup captured into fixture provenance.

    The oracle still needs a separate nondeterminism-floor run; this only makes a
    replay attempt carry the same explicit seed and deterministic-algorithm knobs.
    """
    if seed < 0:
        raise SystemExit("gen_reference_fixtures: --seed must be non-negative")

    record: dict[str, Any] = {
        "seed": seed,
        "transformers_set_seed": False,
        "torch_manual_seed": False,
        "torch_cuda_manual_seed_all": False,
        "torch_deterministic_algorithms": False,
        "torch_deterministic_warn_only": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    try:
        transformers.set_seed(seed)
        record["transformers_set_seed"] = True
    except Exception as exc:  # noqa: BLE001 - older/newer API shape
        record["transformers_set_seed_error"] = str(exc)

    torch.manual_seed(seed)
    record["torch_manual_seed"] = True

    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        record["torch_cuda_manual_seed_all"] = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        record["torch_deterministic_algorithms"] = True
    except TypeError:
        torch.use_deterministic_algorithms(True)
        record["torch_deterministic_algorithms"] = True
        record["torch_deterministic_warn_only"] = False
    except Exception as exc:  # noqa: BLE001 - environment/backend dependent
        record["torch_deterministic_algorithms_error"] = str(exc)

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG

    # 1. Assert the pinned oracle stack BEFORE touching anything expensive.
    torch, transformers = _assert_pinned_stack()

    # 1b. The CPU-equivalence shims must be in place BEFORE the model loads
    # (from_pretrained probes torch.cuda). Contradiction with --device cuda is
    # refused loudly — the patch exists ONLY for the separate CPU experiment.
    if args.cpu_patch:
        if args.device == "cuda":
            raise SystemExit(
                "gen_reference_fixtures: --cpu-patch contradicts --device cuda; "
                "the shims exist only for the separate CPU-equivalence run "
                "(OQ-17), never for golden fixtures."
            )
        _install_cpu_patches(torch)

    # 2. Resolve + validate the model dir and corpus.
    model_dir = _resolve_model_dir(args.model_dir)
    docs = _list_corpus(args.corpus)
    if args.limit > 0:
        docs = docs[: args.limit]

    preset = MODE_PRESETS[args.mode]
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"[fixtures] mode={args.mode} prompt={args.prompt!r} "
        f"device={args.device} docs={len(docs)} "
        f"activations={'on' if args.activations else 'off'}",
        file=sys.stderr,
    )

    # 3. Load the model + tokenizer (bf16, eval, device). This may allocate and
    # consume RNG internally, so the replay seed is applied immediately after it.
    model, tokenizer = _load_model_and_tokenizer(torch, model_dir, args.device)

    # 4. Make generation as deterministic as the runtime allows (the oracle's own
    # nondeterminism floor is established by running this script twice — plan §8.1
    # — NOT by pretending bf16 CUDA matmul is bitwise reproducible).
    determinism = _apply_determinism(torch, transformers, args.seed)

    # 5. Provenance (hashes the weights once, shared by every fixture this run).
    provenance = _build_provenance(
        torch=torch,
        transformers=transformers,
        model_dir=model_dir,
        preset=preset,
        prompt=args.prompt,
        mode=args.mode,
        device=args.device,
        cpu_patch=args.cpu_patch,
        max_length=args.max_length,
        determinism=determinism,
        command_argv=[sys.executable, *sys.argv],
    )

    # 6. Per-document oracle runs.
    records: list[dict[str, Any]] = []
    for doc in docs:
        with torch.no_grad():
            rec = _run_one(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                doc=doc,
                out_dir=args.out,
                prompt=args.prompt,
                preset=preset,
                max_length=args.max_length,
                want_activations=args.activations,
                provenance=provenance,
            )
        records.append(rec)

    # 7. Run-level provenance manifest (one per run; --run-tag keeps two side by
    # side for the nondeterminism floor).
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_manifest = {
        "schema_version": 1,
        "provenance": provenance,
        "documents": records,
        "n_documents": len(records),
    }
    manifest_path = args.out / f"PROVENANCE{tag}.json"
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    markdown_path = args.out / f"PROVENANCE{tag}.md"
    markdown_path.write_text(_provenance_markdown(run_manifest), encoding="utf-8")
    print(
        f"[fixtures] wrote {len(records)} fixtures + {manifest_path} + {markdown_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
