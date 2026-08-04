#!/usr/bin/env python3
"""Roofline floors for the head-to-head gauntlet (bd-re8.17; plan §9.1).

For each pipeline stage, computes the **compute floor** (int8 GEMM MACs ÷ the
machine's peak int8 throughput, plus f32 terms ÷ peak f32) and the **memory
floor** (bytes streamed ÷ DRAM bandwidth) and takes the **max** — what a
perfect kernel would hit. `docs/PERF_LEDGER.md` records that floor next to
every measured ratio: `dist_above_floor = focr_ms / floor_ms`, and a stage
>~1.3× above its floor is a named attackable lever.

The model shapes are transcribed from the source of truth in `src/` (each
constant cites its file); token counts and view counts are READ FROM the real
measurement file produced by `scripts/gauntlet_focr.sh` — a token-dependent
floor without a measurement file is REFUSED, never assumed. Floors are
analytic lower bounds: terms that are hard to bound tightly (norms, softmax,
embeds, activations) are *excluded*, which can only LOWER the floor and
therefore never flatters `dist_above_floor`.

The machine profile is explicit, echoed into the output, and overridable —
a floor is only meaningful relative to the recorded profile.

Usage:
  gauntlet_roofline.py --arch unlimited-ocr --precision mixed-ffn-int8 \
      --stages-json artifacts/perf/bd-re8.17/focr/focr_stages.json \
      [--stage decode_per_token ...] [--profile m4|m4-pro|m4-max] \
      [--dram-gb-s X] [--peak-int8-gmacs X] [--peak-f32-gflops X] --out FILE
  gauntlet_roofline.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import time

SCHEMA = "focr-gauntlet-roofline/v1"
DECODE_MIXED_FFN_INT8 = "mixed-ffn-int8"
DECODE_FULL_INT8 = "full-int8"
UNLIMITED_DECODE_MODES = (DECODE_MIXED_FFN_INT8, DECODE_FULL_INT8)
RUNTIME_PRECISION_BY_DECODE_MODE = {
    DECODE_MIXED_FFN_INT8: "focr-mixed-ffn-int8",
    DECODE_FULL_INT8: "focr-full-int8",
}
UNLIMITED_QUANT_RECIPE = "unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1"

STAGES = ("preprocess", "vision_encode", "prefill", "decode_per_token", "end_to_end")

STAGES_SCHEMA = "focr-gauntlet-stages/v1"
STAGE_SCHEMA = "focr-gauntlet-stage/v1"
MAX_STAGES_JSON_BYTES = 4 * 1024 * 1024
MAX_TOP_LEVEL_KEYS = 128
MAX_STAGE_RECORDS = 128
MAX_STAGE_RECORD_KEYS = 64
MAX_STAGE_SAMPLES = 4096
MAX_JSON_DEPTH = 32
MAX_JSON_CONTAINER_ITEMS = 4096
MAX_JSON_TOTAL_NODES = 100_000
MAX_JSON_STRING_CHARS = 1_048_576
MAX_JSON_ABS_NUMBER = 1e18
MAX_MODEL_BYTES = 1 << 50
MAX_MEASURED_TOKENS = 1_000_000
MAX_STAGE_OCCURRENCES = 1_000_000
MAX_VIEWS = 4096
MAX_DURATION_MS = 31_536_000_000.0
MAX_PROFILE_RATE = 1_000_000_000.0
STAGE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")

# ── model shapes (transcribed from src; the src constants are the truth) ────

UNLIMITED = {
    # src/native_engine/decoder.rs::config
    "hidden": 1280,
    "layers": 12,
    "heads": 10,
    "head_dim": 128,
    "vocab": 129_280,
    "dense_inter": 6848,  # layer 0 (FIRST_K_DENSE_REPLACE = 1)
    # src/native_engine/moe.rs::config — layers 1..11 are MoE
    "moe_layers": 11,
    "routed_experts": 64,
    "experts_per_tok": 6,
    "moe_inter": 896,
    "shared_inter": 1792,  # MOE_INTERMEDIATE_SIZE * N_SHARED_EXPERTS (fused)
    # src/native_engine/rswa.rs — decode attention reads a 128-token ring
    "kv_window": 128,
}

GOT = {
    # src/native_engine/decoder_qwen2.rs::DecoderConfig::got_ocr2 (dense, no MoE)
    "hidden": 1024,
    "layers": 24,
    "heads": 16,
    "head_dim": 64,
    "vocab": 151_860,
    "dense_inter": 2816,
    "kv_window": None,  # full causal: the window is the whole context
}

SMOLVLM2 = {
    # src/native_engine/decoder_qwen2.rs::DecoderConfig::smolvlm2 (dense, GQA 15/5)
    "hidden": 960,
    "layers": 32,
    "heads": 15,
    "head_dim": 64,
    "kv_dim": 5 * 64,  # num_key_value_heads 5 × head_dim
    "vocab": 49_280,
    "dense_inter": 2560,  # SwiGLU: gate/up/down = 3 GEMMs
    # UNTIED f32 lm_head (model_arch.rs: lm_head_stored_int8: false)
    "head_f32": True,
    "kv_window": None,  # full causal
}

ONECHART = {
    # src/native_engine/decoder_qwen2.rs::DecoderConfig::onechart (OPT-125M)
    "hidden": 768,
    "layers": 12,
    "heads": 12,
    "head_dim": 64,
    "kv_dim": 768,  # MHA
    "vocab": 50_269,
    "relu_inter": 3072,  # OPT ReLU MLP: fc1/fc2 = 2 GEMMs (no gate)
    # TIED head runs against the f32 embed matrix (lm_head omitted from .focrq)
    "head_f32": True,
    "kv_window": None,  # full causal
}

VISION_SIGLIP = {
    # src/native_engine/vision_siglip.rs — SigLIP-B/16 per 512² frame
    "dim": 768,
    "depth": 12,
    "mlp": 3072,
    "tokens": 1024,  # (512/16)² patches, full (non-windowed) attention
    # token_compress.rs — pixel_shuffle is a free gather; the connector GEMM is
    # modality_projection Linear(12288→960) over 64 shuffled tokens per frame.
    "connector_macs": 64 * 12_288 * 960,
    "params": 12 * (4 * 768 * 768 + 2 * 768 * 3072) + 768 * 3 * 16 * 16 + 12_288 * 960,
}

VISION_SAM = {
    # src/native_engine/vision_sam.rs — SAM-ViT-B at 1024px / patch 16
    "dim": 768,
    "depth": 12,
    "mlp": 3072,
    "tokens": 64 * 64,
    "window": 14 * 14,
    "global_blocks": 4,
    # Vary neck + conv compressor (768→256 1×1, 256 3×3, 256→512 s2, 512→1024 s2)
    "neck_convs_macs": (
        64 * 64 * 768 * 256
        + 64 * 64 * 9 * 256 * 256
        + 32 * 32 * 9 * 256 * 512
        + 16 * 16 * 9 * 512 * 1024
    ),
    "params": 12 * (4 * 768 * 768 + 2 * 768 * 3072)
    + 768 * 3 * 16 * 16
    + 768 * 256
    + 9 * 256 * 256
    + 9 * 256 * 512
    + 9 * 512 * 1024,
}

VISION_CLIP = {
    # src/native_engine/vision_clip.rs — CLIP-L/14 style NoTP tower ([SPEC-047])
    "dim": 1024,
    "depth": 24,
    "ffn": 4096,
    "tokens": 257,  # 256 patch embeds from SAM + CLS
}

BRIDGE = {"tokens_per_view": 256, "proj_in": 2048, "proj_out": 1280}  # vision_bridge.rs

# ── machine profiles (explicit; every field overridable) ────────────────────
#
# peak_int8_gmacs: NEON SDOT = 16 int8 MACs/instr × 4 SIMD pipes = 64
# MACs/cycle/P-core (SMMLA is half-rate on M4 — perf playbook). peak_f32_gflops:
# 128-bit FMA = 4 MACs = 8 FLOPs/instr × 4 pipes = 32 FLOPs/cycle/P-core.
# E-cores are excluded (conservative capacity), which RAISES floor_ms slightly;
# the profile is recorded in the output so a row's floor is reproducible.
PROFILES = {
    "m4": {"dram_gb_s": 120.0, "p_cores": 4, "ghz": 4.4},
    "m4-pro": {"dram_gb_s": 273.0, "p_cores": 10, "ghz": 4.4},
    "m4-max": {"dram_gb_s": 546.0, "p_cores": 12, "ghz": 4.4},
}


def resolve_profile(args: argparse.Namespace) -> dict:
    base = PROFILES[args.profile]
    p_cores, ghz = base["p_cores"], base["ghz"]
    profile = {
        "name": args.profile,
        "dram_gb_s": args.dram_gb_s if args.dram_gb_s is not None else base["dram_gb_s"],
        "peak_int8_gmacs": (
            args.peak_int8_gmacs if args.peak_int8_gmacs is not None else p_cores * ghz * 64.0
        ),
        "peak_f32_gflops": (
            args.peak_f32_gflops if args.peak_f32_gflops is not None else p_cores * ghz * 32.0
        ),
        "note": "P-cores only, SDOT 64 int8-MAC/cycle/core, FMA 32 f32-FLOP/cycle/core",
    }
    for key in ("dram_gb_s", "peak_int8_gmacs", "peak_f32_gflops"):
        value = profile[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 < value <= MAX_PROFILE_RATE
        ):
            raise RooflineError(
                f"machine profile field {key} must be finite and in "
                f"(0, {MAX_PROFILE_RATE:g}]"
            )
    return profile


class RooflineError(ValueError):
    """A floor cannot be computed honestly from the available inputs."""


# ── per-stage cost models (MACs + bytes; all terms are lower bounds) ─────────


def decode_per_token_cost(arch: str, precision: str, kv_len: int | None) -> dict:
    """Active weight traffic + MACs for ONE decode step (the G2 gating stage)."""
    if arch == "unlimited-ocr":
        if precision not in UNLIMITED_DECODE_MODES:
            raise RooflineError(
                "Unlimited-OCR decode precision must be bound to runtime mode "
                f"{DECODE_MIXED_FFN_INT8!r} or {DECODE_FULL_INT8!r}, got {precision!r}"
            )
        m = UNLIMITED
        h = m["hidden"]
        attn = m["layers"] * 4 * h * h
        dense0 = 3 * m["dense_inter"] * h
        moe = m["moe_layers"] * 3 * h * (m["experts_per_tok"] * m["moe_inter"] + m["shared_inter"])
        lm_head = m["vocab"] * h
        router_f32 = m["moe_layers"] * m["routed_experts"] * h  # f32 gate MACs
        ffn = dense0 + moe
        ffn_scale_rows = (2 * m["dense_inter"] + h) + m["moe_layers"] * (
            m["experts_per_tok"] * (2 * m["moe_inter"] + h)
            + (2 * m["shared_inter"] + h)
        )
        full_scale_rows = (
            m["layers"] * 4 * h
            + ffn_scale_rows
            + m["vocab"]
        )
        kv = m["kv_window"] if kv_len is None else min(kv_len, m["kv_window"])
        kv_bytes = m["layers"] * 2 * kv * h * 4  # f32 K+V ring reads
        kv_f32_macs = m["layers"] * 2 * kv * h
        if precision == DECODE_FULL_INT8:
            int8_macs = attn + ffn + lm_head
            f32_macs = router_f32 + kv_f32_macs
            weight_bytes = int8_macs + full_scale_rows * 4 + router_f32 * 4
            assumptions = [
                "experimental full-int8 cache: attn qkvo + FFN/experts + lm_head int8",
                "per-output-channel f32 scales counted; router remains f32",
                f"R-SWA ring window {kv} K+V rows read per layer in f32",
            ]
        else:
            int8_macs = ffn
            f32_macs = attn + lm_head + router_f32 + kv_f32_macs
            weight_bytes = (
                ffn
                + ffn_scale_rows * 4
                + (attn + lm_head + router_f32) * 4
            )
            assumptions = [
                "conservative cache: dense/MoE FFNs int8; attn qkvo + lm_head f32",
                "FFN per-output-channel f32 scales counted; router remains f32",
                f"R-SWA ring window {kv} K+V rows read per layer in f32",
            ]
        return {
            "int8_macs": int8_macs,
            "f32_flops": 2 * f32_macs,
            "bytes": weight_bytes + kv_bytes,
            "assumptions": assumptions,
        }
    if precision != "int8":
        raise RooflineError(
            f"{arch} decode floor is modeled for its int8 artifact, got {precision!r}"
        )
    if arch == "got-ocr2":
        m = GOT
        h = m["hidden"]
        int8_macs = m["layers"] * (4 * h * h + 3 * m["dense_inter"] * h) + m["vocab"] * h
        scale_rows = m["layers"] * (4 * h + 2 * m["dense_inter"] + h) + m["vocab"]
        if kv_len is None:
            raise RooflineError(
                "got-ocr2 decode attention is full-causal: kv_len must come from the "
                "measurement file (prefill + decoded tokens), not a guess"
            )
        kv_bytes = m["layers"] * 2 * kv_len * h * 4
        kv_f32_macs = m["layers"] * 2 * kv_len * h
        return {
            "int8_macs": int8_macs,
            "f32_flops": 2 * kv_f32_macs,
            "bytes": int8_macs + scale_rows * 4 + kv_bytes,
            "assumptions": [
                "dense Qwen2: attn qkvo + SwiGLU MLP + lm_head, all int8",
                f"full-causal K+V read at mean context {kv_len} in f32",
                "norms/embeds/biases/activations excluded (lower bound)",
            ],
        }
    if arch in ("smolvlm2", "onechart"):
        m = SMOLVLM2 if arch == "smolvlm2" else ONECHART
        h, kvd = m["hidden"], m["kv_dim"]
        if arch == "smolvlm2":
            # GQA: q/o are h×h, k/v are h×kv_dim; SwiGLU = 3 GEMMs.
            gemm_per_layer = 2 * h * h + 2 * h * kvd + 3 * m["dense_inter"] * h
            scale_rows = m["layers"] * (2 * h + 2 * kvd + 2 * m["dense_inter"] + h)
            mlp_note = "SwiGLU gate/up/down"
        else:
            # OPT MHA qkvo + ReLU fc1/fc2 (2 GEMMs).
            gemm_per_layer = 4 * h * h + 2 * m["relu_inter"] * h
            scale_rows = m["layers"] * (4 * h + m["relu_inter"] + h)
            mlp_note = "ReLU fc1/fc2"
        int8_macs = m["layers"] * gemm_per_layer
        head_macs = m["vocab"] * h  # f32 head (untied-f32 / tied-embed)
        if kv_len is None:
            raise RooflineError(
                f"{arch} decode attention is full-causal: kv_len must come from the "
                "measurement file (prefill + decoded tokens), not a guess"
            )
        kv_bytes = m["layers"] * 2 * kv_len * kvd * 4
        kv_f32_macs = m["layers"] * 2 * kv_len * h  # scores + context over all q heads
        return {
            "int8_macs": int8_macs,
            "f32_flops": 2 * (head_macs + kv_f32_macs),
            "bytes": int8_macs + scale_rows * 4 + head_macs * 4 + kv_bytes,
            "assumptions": [
                f"dense {arch}: attn qkvo + {mlp_note} int8; the head is an f32 GEMM "
                "(model_arch.rs: lm_head_stored_int8 false / tied f32 embed)",
                f"full-causal K+V read at mean context {kv_len} in f32 (kv_dim {kvd})",
                "norms/embeds/biases/activations excluded (lower bound)",
            ],
        }
    raise RooflineError(f"unknown arch {arch!r}")


def prefill_cost(arch: str, precision: str, prefill_tokens: int) -> dict:
    """One prefill pass over `prefill_tokens` positions (lm_head excluded)."""
    if prefill_tokens <= 0:
        raise RooflineError("prefill floor needs the measured prefill token count")
    n = prefill_tokens
    if arch == "unlimited-ocr":
        if precision not in UNLIMITED_DECODE_MODES:
            raise RooflineError(
                "Unlimited-OCR prefill precision must be bound to runtime mode "
                f"{DECODE_MIXED_FFN_INT8!r} or {DECODE_FULL_INT8!r}, got {precision!r}"
            )
        m = UNLIMITED
        h = m["hidden"]
        attn_proj = m["layers"] * 4 * h * h
        dense0 = 3 * m["dense_inter"] * h
        moe = m["moe_layers"] * 3 * h * (
            m["experts_per_tok"] * m["moe_inter"] + m["shared_inter"]
        )
        ffn = dense0 + moe
        router_f32 = m["moe_layers"] * m["routed_experts"] * h
        ffn_scale_rows = (2 * m["dense_inter"] + h) + m["moe_layers"] * (
            m["experts_per_tok"] * (2 * m["moe_inter"] + h)
            + (2 * m["shared_inter"] + h)
        )
        attn_f32 = m["layers"] * 2 * n * n * h  # QK^T + AV, causal lower bound
        if precision == DECODE_FULL_INT8:
            int8_macs = n * (attn_proj + ffn)
            f32_macs = n * router_f32 + attn_f32
            scale_rows = m["layers"] * 4 * h + ffn_scale_rows
            weight_bytes = attn_proj + ffn + scale_rows * 4 + router_f32 * 4
            assumptions = [
                f"{n} prefill tokens: attn qkvo + active FFNs int8; router f32",
                "per-output-channel f32 scales counted; lm_head excluded",
            ]
        else:
            int8_macs = n * ffn
            f32_macs = n * (attn_proj + router_f32) + attn_f32
            weight_bytes = ffn + ffn_scale_rows * 4 + (attn_proj + router_f32) * 4
            assumptions = [
                f"{n} prefill tokens: active FFNs int8; attn qkvo + router f32",
                "FFN per-output-channel f32 scales counted; lm_head excluded",
            ]
        # Weight bytes at least once (identical-routing lower bound for MoE).
        return {
            "int8_macs": int8_macs,
            "f32_flops": 2 * f32_macs,
            "bytes": weight_bytes,
            "assumptions": assumptions
            + [
                "memory floor = active weights streamed once (identical-routing MoE lower bound)",
                "attention scores/context counted full-causal in f32",
            ],
        }
    if precision != "int8":
        raise RooflineError(f"prefill floor modeled for int8 weights only, got {precision!r}")
    if arch == "got-ocr2":
        m = GOT
        h = m["hidden"]
        per_tok = m["layers"] * (4 * h * h + 3 * m["dense_inter"] * h)
        attn_f32 = m["layers"] * 2 * n * n * h
        return {
            "int8_macs": n * per_tok,
            "f32_flops": 2 * attn_f32,
            "bytes": per_tok,
            "assumptions": [
                f"{n} prefill tokens × dense per-token GEMM MACs (lm_head excluded)",
                "memory floor = weights streamed once",
            ],
        }
    if arch in ("smolvlm2", "onechart"):
        m = SMOLVLM2 if arch == "smolvlm2" else ONECHART
        h, kvd = m["hidden"], m["kv_dim"]
        if arch == "smolvlm2":
            per_tok = m["layers"] * (2 * h * h + 2 * h * kvd + 3 * m["dense_inter"] * h)
        else:
            per_tok = m["layers"] * (4 * h * h + 2 * m["relu_inter"] * h)
        attn_f32 = m["layers"] * 2 * n * n * h
        return {
            "int8_macs": n * per_tok,
            "f32_flops": 2 * attn_f32,
            "bytes": per_tok,
            "assumptions": [
                f"{n} prefill tokens × dense per-token GEMM MACs (head excluded)",
                "memory floor = int8 weights streamed once",
            ],
        }
    raise RooflineError(f"unknown arch {arch!r}")


def vision_cost(arch: str, views: int) -> dict:
    """The f32 vision tower(s), per page = `views` × per-view GEMM terms."""
    if views <= 0:
        raise RooflineError("vision floor needs the measured view count (vision_sam occurrences)")
    if arch == "smolvlm2":
        # SigLIP-B/16 per 512² frame (views = frames), full attention, then the
        # pixel-shuffle connector GEMM (token_compress.rs).
        g = VISION_SIGLIP
        per_frame = (
            g["depth"] * g["tokens"] * (4 * g["dim"] * g["dim"] + 2 * g["dim"] * g["mlp"])
            + g["depth"] * 2 * g["tokens"] * g["tokens"] * g["dim"]
            + g["connector_macs"]
        )
        return {
            "int8_macs": 0,
            "f32_flops": views * 2 * per_frame,
            "bytes": g["params"] * 4,
            "assumptions": [
                f"SigLIP-B/16 blocks + full attention over 1024 tokens × {views} frames "
                "+ the 12288→960 modality_projection",
                "f32 weights streamed once; norms/softmax/pos-embeds excluded (lower bound)",
            ],
        }
    s = VISION_SAM
    sam_gemm = s["depth"] * s["tokens"] * (4 * s["dim"] * s["dim"] + 2 * s["dim"] * s["mlp"])
    sam_attn = s["tokens"] * s["dim"] * 2 * (
        (s["depth"] - s["global_blocks"]) * s["window"] + s["global_blocks"] * s["tokens"]
    )
    macs = sam_gemm + sam_attn + s["neck_convs_macs"]
    bytes_ = s["params"] * 4
    assumptions = ["SAM-ViT-B blocks + windowed/global attention + Vary neck/compressor convs"]
    if arch == "unlimited-ocr":
        c = VISION_CLIP
        macs += c["depth"] * c["tokens"] * (4 * c["dim"] * c["dim"] + 2 * c["dim"] * c["ffn"])
        macs += BRIDGE["tokens_per_view"] * BRIDGE["proj_in"] * BRIDGE["proj_out"]
        bytes_ += (
            c["depth"] * (4 * c["dim"] * c["dim"] + 2 * c["dim"] * c["ffn"])
            + BRIDGE["proj_in"] * BRIDGE["proj_out"]
        ) * 4
        assumptions.append("CLIP-L NoTP tower (24×1024/4096 over 257 tokens) + 2048→1280 bridge")
    elif arch == "got-ocr2":
        macs += BRIDGE["tokens_per_view"] * 1024 * 1024  # mm_projector_vary
        bytes_ += 1024 * 1024 * 4
        assumptions.append("GOT mm_projector_vary 1024→1024 over 256 tokens")
    elif arch == "onechart":
        macs += BRIDGE["tokens_per_view"] * 1024 * 768  # mm_projector 1024→768
        bytes_ += 1024 * 768 * 4
        assumptions.append("OneChart mm_projector 1024→768 over 256 tokens (same SAM tower)")
    else:
        raise RooflineError(f"unknown arch {arch!r}")
    assumptions.append("f32 weights streamed once; norms/softmax/pos-embeds excluded (lower bound)")
    return {
        "int8_macs": 0,
        "f32_flops": views * 2 * macs,
        "bytes": bytes_,  # weights dominate; activations excluded (lower bound)
        "assumptions": assumptions,
    }


def preprocess_cost(views: int) -> dict:
    """Producing the model input tensor(s) is the only safely boundable term."""
    if views <= 0:
        raise RooflineError("preprocess floor needs the measured view count")
    return {
        "int8_macs": 0,
        "f32_flops": 0,
        "bytes": views * 3 * 1024 * 1024 * 4,  # normalized f32 CHW tensor write per view
        "assumptions": [
            "memory floor only: the normalized 3×1024×1024 f32 input tensor must be written",
            "source decode/resize excluded (image dims not in the timing evidence)",
        ],
    }


def floor_from_cost(cost: dict, profile: dict) -> dict:
    compute_s = cost["int8_macs"] / (profile["peak_int8_gmacs"] * 1e9) + cost["f32_flops"] / (
        profile["peak_f32_gflops"] * 1e9
    )
    memory_s = cost["bytes"] / (profile["dram_gb_s"] * 1e9)
    kind = "compute" if compute_s >= memory_s else "memory"
    return {
        "floor_kind": kind,
        "floor_ms": round(max(compute_s, memory_s) * 1000.0, 6),
        "compute_floor_ms": round(compute_s * 1000.0, 6),
        "memory_floor_ms": round(memory_s * 1000.0, 6),
        "int8_macs": cost["int8_macs"],
        "f32_flops": cost["f32_flops"],
        "bytes": cost["bytes"],
        "assumptions": cost["assumptions"],
    }


# ── measurement-file plumbing ────────────────────────────────────────────────


def _stat_fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_bounded_regular_file(path: str, max_bytes: int) -> bytes:
    """Read a stable regular file without following a final-component symlink."""
    if not isinstance(path, str) or not path:
        raise RooflineError("stages path must be a non-empty string")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise RooflineError("internal file-size bound must be a positive integer")

    before_path = os.lstat(path)
    if stat.S_ISLNK(before_path.st_mode):
        raise RooflineError(f"{path}: symlink evidence is refused")
    if not stat.S_ISREG(before_path.st_mode):
        raise RooflineError(f"{path}: evidence must be a regular file")
    if before_path.st_size > max_bytes:
        raise RooflineError(
            f"{path}: evidence is {before_path.st_size} bytes; limit is {max_bytes}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before_fd = os.fstat(fd)
        if not stat.S_ISREG(before_fd.st_mode):
            raise RooflineError(f"{path}: opened evidence is not a regular file")
        if _stat_fingerprint(before_fd) != _stat_fingerprint(before_path):
            raise RooflineError(f"{path}: evidence changed before it could be opened")
        if before_fd.st_size > max_bytes:
            raise RooflineError(
                f"{path}: evidence is {before_fd.st_size} bytes; limit is {max_bytes}"
            )

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(65_536, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise RooflineError(f"{path}: evidence grew past the {max_bytes}-byte limit")
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)

    after_path = os.lstat(path)
    expected = _stat_fingerprint(before_fd)
    if _stat_fingerprint(after_fd) != expected or _stat_fingerprint(after_path) != expected:
        raise RooflineError(f"{path}: evidence changed while it was read")
    raw = b"".join(chunks)
    if len(raw) != before_fd.st_size:
        raise RooflineError(
            f"{path}: stable read length {len(raw)} disagrees with stat size {before_fd.st_size}"
        )
    return raw


def _reject_json_constant(value: str) -> None:
    raise RooflineError(f"non-finite JSON number {value!r} is refused")


def _duplicate_free_object(pairs: list[tuple[str, object]]) -> dict:
    if len(pairs) > MAX_JSON_CONTAINER_ITEMS:
        raise RooflineError(
            f"JSON object has {len(pairs)} entries; limit is {MAX_JSON_CONTAINER_ITEMS}"
        )
    result = {}
    for key, value in pairs:
        if key in result:
            raise RooflineError(f"duplicate JSON key {key!r} is refused")
        result[key] = value
    return result


def _validate_json_limits(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_TOTAL_NODES:
            raise RooflineError(f"JSON document exceeds {MAX_JSON_TOTAL_NODES} values")
        if depth > MAX_JSON_DEPTH:
            raise RooflineError(f"JSON nesting exceeds {MAX_JSON_DEPTH} levels")
        if isinstance(item, dict):
            if len(item) > MAX_JSON_CONTAINER_ITEMS:
                raise RooflineError(
                    f"JSON object has {len(item)} entries; limit is {MAX_JSON_CONTAINER_ITEMS}"
                )
            for key, child in item.items():
                if not isinstance(key, str) or len(key) > MAX_JSON_STRING_CHARS:
                    raise RooflineError("JSON object key is not a bounded string")
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            if len(item) > MAX_JSON_CONTAINER_ITEMS:
                raise RooflineError(
                    f"JSON array has {len(item)} entries; limit is {MAX_JSON_CONTAINER_ITEMS}"
                )
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            if len(item) > MAX_JSON_STRING_CHARS:
                raise RooflineError(
                    f"JSON string has {len(item)} characters; limit is {MAX_JSON_STRING_CHARS}"
                )
        elif isinstance(item, bool) or item is None:
            continue
        elif isinstance(item, int):
            if abs(item) > MAX_JSON_ABS_NUMBER:
                raise RooflineError(f"JSON integer magnitude exceeds {MAX_JSON_ABS_NUMBER:g}")
        elif isinstance(item, float):
            if not math.isfinite(item) or abs(item) > MAX_JSON_ABS_NUMBER:
                raise RooflineError(
                    f"JSON float must be finite with magnitude <= {MAX_JSON_ABS_NUMBER:g}"
                )
        else:
            raise RooflineError(f"unsupported JSON value type {type(item).__name__}")


def _require_string(value: object, label: str, max_chars: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > max_chars:
        raise RooflineError(f"{label} must be a non-empty string of at most {max_chars} characters")
    return value


def _require_positive_int(value: object, label: str, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 < value <= maximum
    ):
        raise RooflineError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _require_positive_number(value: object, label: str, maximum: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise RooflineError(f"{label} must be finite and in (0, {maximum:g}]")
    return float(value)


def _validate_stages_document(doc: object, path: str) -> dict:
    _validate_json_limits(doc)
    if not isinstance(doc, dict):
        raise RooflineError(f"{path}: top-level JSON value must be an object")
    if len(doc) > MAX_TOP_LEVEL_KEYS:
        raise RooflineError(
            f"{path}: top-level object has {len(doc)} keys; limit is {MAX_TOP_LEVEL_KEYS}"
        )
    if doc.get("schema") != STAGES_SCHEMA or doc.get("source") != "focr":
        raise RooflineError(f"{path}: not a {STAGES_SCHEMA} focr measurement file")

    precision = _require_string(doc.get("precision"), "timing precision", 64)
    synthetic = doc.get("synthetic", False)
    if not isinstance(synthetic, bool):
        raise RooflineError("timing synthetic marker must be a boolean")
    for key in ("decode_mode", "quant_recipe"):
        if doc.get(key) is not None:
            _require_string(doc[key], f"timing {key}", 256)
    if doc.get("model_sha256") is not None:
        model_sha256 = doc["model_sha256"]
        if not isinstance(model_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", model_sha256) is None:
            raise RooflineError("timing model_sha256 must be 64 lowercase hexadecimal characters")
    if doc.get("model_size") is not None:
        _require_positive_int(doc["model_size"], "timing model_size", MAX_MODEL_BYTES)
    gate_states = doc.get("precision_gate_states")
    if gate_states is not None:
        if not isinstance(gate_states, dict) or len(gate_states) > 64:
            raise RooflineError(
                "timing precision_gate_states must be an object with at most 64 keys"
            )
        for key, value in gate_states.items():
            _require_string(key, "precision gate name", 128)
            _require_string(value, f"precision gate {key!r} value", 256)

    records = doc.get("stages")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_STAGE_RECORDS:
        raise RooflineError(
            f"timing stages must be a non-empty array of at most {MAX_STAGE_RECORDS} records"
        )
    seen: set[str] = set()
    for index, record in enumerate(records):
        label = f"timing stages[{index}]"
        if not isinstance(record, dict):
            raise RooflineError(f"{label} must be an object")
        if len(record) > MAX_STAGE_RECORD_KEYS:
            raise RooflineError(
                f"{label} has {len(record)} keys; limit is {MAX_STAGE_RECORD_KEYS}"
            )
        if record.get("schema") != STAGE_SCHEMA or record.get("source") != "focr":
            raise RooflineError(f"{label} must be a {STAGE_SCHEMA} focr record")
        if record.get("unit") != "ms":
            raise RooflineError(f"{label} unit must be 'ms'")
        stage = _require_string(record.get("stage"), f"{label} stage", 64)
        if STAGE_NAME_RE.fullmatch(stage) is None:
            raise RooflineError(f"{label} stage {stage!r} is not a valid stage name")
        if stage in seen:
            raise RooflineError(f"duplicate timing stage {stage!r} is refused")
        seen.add(stage)
        if record.get("precision") != precision:
            raise RooflineError(
                f"{label} precision {record.get('precision')!r} disagrees with {precision!r}"
            )
        _require_positive_number(record.get("best_ms"), f"{label} best_ms", MAX_DURATION_MS)
        samples = record.get("samples_ms")
        if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_STAGE_SAMPLES:
            raise RooflineError(
                f"{label} samples_ms must contain 1..{MAX_STAGE_SAMPLES} measurements"
            )
        for sample_index, sample in enumerate(samples):
            _require_positive_number(
                sample,
                f"{label} samples_ms[{sample_index}]",
                MAX_DURATION_MS,
            )
        sample_count = _require_positive_int(record.get("n"), f"{label} n", MAX_STAGE_SAMPLES)
        if sample_count != len(samples):
            raise RooflineError(
                f"{label} n={sample_count} disagrees with {len(samples)} samples_ms entries"
            )
        if "tokens" in record:
            _require_positive_int(record["tokens"], f"{label} tokens", MAX_MEASURED_TOKENS)
        if "occurrences" in record:
            _require_positive_int(
                record["occurrences"],
                f"{label} occurrences",
                MAX_STAGE_OCCURRENCES,
            )
        if "views" in record:
            _require_positive_int(record["views"], f"{label} views", MAX_VIEWS)
        for key in ("ledger_stage", "synthetic", "tokens_consistent"):
            if key in record and not isinstance(record[key], bool):
                raise RooflineError(f"{label} {key} must be a boolean")
    return doc


def _decode_stages_json(raw: object, path: str) -> dict:
    try:
        doc = json.loads(
            raw,
            object_pairs_hook=_duplicate_free_object,
            parse_constant=_reject_json_constant,
        )
        return _validate_stages_document(doc, path)
    except RooflineError:
        raise
    except (AttributeError, TypeError, RecursionError, UnicodeDecodeError, ValueError) as err:
        raise RooflineError(
            f"{path}: malformed JSON evidence ({type(err).__name__}: {err})"
        ) from err


def load_stages(path: str) -> tuple[dict, str]:
    try:
        raw = _read_bounded_regular_file(path, MAX_STAGES_JSON_BYTES)
        doc = _decode_stages_json(raw, path)
    except RooflineError:
        raise
    except (AttributeError, TypeError, RecursionError, OSError) as err:
        raise RooflineError(
            f"{path}: unable to read stable timing evidence ({type(err).__name__}: {err})"
        ) from err
    return doc, hashlib.sha256(raw).hexdigest()


def validate_precision_contract(arch: str, precision: str, doc: dict) -> None:
    """Bind an Unlimited floor to the capture's artifact and runtime identity."""
    if arch != "unlimited-ocr":
        if precision != "int8":
            raise RooflineError(f"{arch} roofline precision must be 'int8', got {precision!r}")
        if doc.get("precision") != "focr-int8":
            raise RooflineError(
                f"{arch} roofline requires timing precision 'focr-int8', "
                f"got {doc.get('precision')!r}"
            )
        return
    expected_runtime = RUNTIME_PRECISION_BY_DECODE_MODE.get(precision)
    if expected_runtime is None:
        raise RooflineError(
            "Unlimited-OCR roofline precision must be "
            f"{DECODE_MIXED_FFN_INT8!r} or {DECODE_FULL_INT8!r}, got {precision!r}"
        )
    if doc.get("precision") != expected_runtime:
        raise RooflineError(
            f"roofline precision {precision!r} requires timing precision "
            f"{expected_runtime!r}, got {doc.get('precision')!r}"
        )
    if doc.get("decode_mode") != precision:
        raise RooflineError(
            f"roofline precision {precision!r} contradicts timing decode_mode "
            f"{doc.get('decode_mode')!r}"
        )
    if doc.get("quant_recipe") != UNLIMITED_QUANT_RECIPE:
        raise RooflineError(
            "Unlimited-OCR roofline requires exact conservative artifact recipe "
            f"{UNLIMITED_QUANT_RECIPE!r}, got {doc.get('quant_recipe')!r}"
        )
    model_sha256 = doc.get("model_sha256")
    model_size = doc.get("model_size")
    if not isinstance(model_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", model_sha256) is None:
        raise RooflineError("timing evidence has no valid model_sha256")
    if not isinstance(model_size, int) or isinstance(model_size, bool) or model_size <= 0:
        raise RooflineError("timing evidence has no positive integer model_size")
    records = doc.get("stages")
    if not isinstance(records, list) or not records:
        raise RooflineError("timing evidence has no stage records")
    mismatched = [
        str(record.get("stage", "<unnamed>"))
        for record in records
        if record.get("precision") != expected_runtime
    ]
    if mismatched:
        raise RooflineError(
            "timing stage precision disagrees with the runtime identity: " + ", ".join(mismatched)
        )


def stage_record(doc: dict, stage: str) -> dict | None:
    for record in doc.get("stages", []):
        if record.get("stage") == stage:
            return record
    return None


def measured_tokens(doc: dict, stage: str) -> int:
    record = stage_record(doc, stage)
    if record is None or "tokens" not in record:
        raise RooflineError(f"measurement file has no token count for stage {stage!r}")
    return record["tokens"]


def measured_views(doc: dict) -> int:
    # SmolVLM2: the frame count rides the vision+splice record as `views`.
    record = stage_record(doc, "smolvlm2_vision_splice")
    if record is not None:
        views = record.get("views")
        if views is None:
            raise RooflineError("smolvlm2_vision_splice record carries no frame count")
        return views
    for name in ("vision_sam", "got_vision_splice", "onechart_vision_splice", "vision_encode"):
        record = stage_record(doc, name)
        if record is not None:
            return record.get("occurrences", 1)
    raise RooflineError("measurement file has no vision stage to derive the view count from")


def compute_stage_floor(stage: str, arch: str, precision: str, doc: dict, profile: dict) -> dict:
    if stage == "decode_per_token":
        kv_len = None
        if arch in ("got-ocr2", "smolvlm2", "onechart"):
            # Full-causal lanes: mean decode context = prefill + half the stream.
            prefill = measured_tokens(doc, "prefill") if _has_tokens(doc, "prefill") else 0
            decoded = measured_tokens(doc, "decode_total")
            kv_len = prefill + decoded // 2
            if prefill == 0:
                raise RooflineError(
                    f"{arch} decode floor needs the measured prefill token count"
                )
        cost = decode_per_token_cost(arch, precision, kv_len)
    elif stage == "prefill":
        cost = prefill_cost(arch, precision, measured_tokens(doc, "prefill"))
    elif stage == "vision_encode":
        cost = vision_cost(arch, measured_views(doc))
    elif stage == "preprocess":
        cost = preprocess_cost(measured_views(doc))
    elif stage == "end_to_end":
        parts = {}
        for sub in ("preprocess", "vision_encode", "prefill", "decode_per_token"):
            if (
                sub in ("preprocess",)
                and arch in ("got-ocr2", "smolvlm2", "onechart")
                and stage_record(doc, sub) is None
            ):
                continue  # zoo paths emit no separate preprocess line (vision+splice includes it)
            parts[sub] = compute_stage_floor(sub, arch, precision, doc, profile)
        decoded = measured_tokens(doc, "decode_total")
        floor_ms = sum(
            p["floor_ms"] * (decoded if name == "decode_per_token" else 1)
            for name, p in parts.items()
        )
        dominant = max(
            parts.items(),
            key=lambda kv: kv[1]["floor_ms"] * (decoded if kv[0] == "decode_per_token" else 1),
        )
        return {
            "floor_kind": dominant[1]["floor_kind"],
            "floor_ms": round(floor_ms, 6),
            "compute_floor_ms": None,
            "memory_floor_ms": None,
            "int8_macs": None,
            "f32_flops": None,
            "bytes": None,
            "assumptions": [
                f"sum of modeled stage floors with decode × {decoded} measured tokens",
                f"floor_kind = dominant component ({dominant[0]})",
            ],
            "components": parts,
        }
    else:
        raise RooflineError(f"no floor model for stage {stage!r}")
    return floor_from_cost(cost, profile)


def _has_tokens(doc: dict, stage: str) -> bool:
    record = stage_record(doc, stage)
    return record is not None and bool(record.get("tokens"))


def run(args: argparse.Namespace) -> int:
    profile = resolve_profile(args)
    doc, stages_json_sha256 = load_stages(args.stages_json)
    validate_precision_contract(args.arch, args.precision, doc)
    if bool(doc.get("synthetic")):
        # Floors over synthetic measurements are for plumbing tests only.
        print("WARNING: stages file is synthetic; output is stamped synthetic", file=sys.stderr)

    stages = args.stage or [
        s for s in STAGES if stage_record(doc, s) is not None or s == "decode_per_token"
    ]
    if len(stages) != len(set(stages)):
        raise RooflineError("duplicate requested --stage values are refused")
    floors = []
    for stage in stages:
        try:
            floor = compute_stage_floor(stage, args.arch, args.precision, doc, profile)
        except RooflineError as err:
            print(f"ERROR: {stage}: {err}", file=sys.stderr)
            return 1
        record = stage_record(doc, stage)
        best_ms = record["best_ms"] if record else None
        floor["stage"] = stage
        floor["focr_best_ms"] = best_ms
        floor["dist_above_floor"] = (
            round(best_ms / floor["floor_ms"], 4)
            if best_ms is not None and floor["floor_ms"] > 0
            else None
        )
        floors.append(floor)

    out = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": args.arch,
        "precision": args.precision,
        "timing_precision": doc.get("precision"),
        "decode_mode": doc.get("decode_mode"),
        "quant_recipe": doc.get("quant_recipe"),
        "model_sha256": doc.get("model_sha256"),
        "model_size": doc.get("model_size"),
        "precision_gate_states": doc.get("precision_gate_states"),
        "machine_profile": profile,
        "stages_json": args.stages_json,
        "stages_json_sha256": stages_json_sha256,
        "floors": floors,
        "synthetic": bool(doc.get("synthetic")),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                "event": "roofline_written",
                "out": args.out,
                "floors": {f["stage"]: f["floor_ms"] for f in floors},
                "synthetic": out["synthetic"],
            }
        )
    )
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def _self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, **fields: object) -> None:
        print(json.dumps({"check": name, "result": "pass" if ok else "fail", **fields}))
        if not ok:
            failures.append(name)

    # Independent arithmetic for the Unlimited decode active set (the numbers a
    # reviewer would compute by hand from the src constants).
    attn = 12 * 4 * 1280 * 1280  # 78_643_200
    dense0 = 3 * 1280 * 6848  # 26_296_320
    moe = 11 * (6 * 3 * 1280 * 896 + 3 * 1280 * 1792)  # 302_776_320
    lm_head = 129_280 * 1280  # 165_478_400
    expected_full_macs = attn + dense0 + moe + lm_head
    expected_mixed_macs = dense0 + moe
    full = decode_per_token_cost("unlimited-ocr", DECODE_FULL_INT8, None)
    mixed = decode_per_token_cost("unlimited-ocr", DECODE_MIXED_FFN_INT8, None)
    check(
        "unlimited-full-decode-macs",
        full["int8_macs"] == expected_full_macs,
        macs=expected_full_macs,
    )
    check(
        "unlimited-mixed-decode-macs",
        mixed["int8_macs"] == expected_mixed_macs,
        macs=expected_mixed_macs,
    )
    check(
        "unlimited-full-decode-exact-bytes",
        full["bytes"] == 594_375_168,
        bytes=full["bytes"],
    )
    check(
        "unlimited-mixed-decode-exact-bytes",
        mixed["bytes"] == 1_325_977_088,
        bytes=mixed["bytes"],
    )

    profile = {"dram_gb_s": 120.0, "peak_int8_gmacs": 1126.4, "peak_f32_gflops": 563.2}
    full_floor = floor_from_cost(full, profile)
    mixed_floor = floor_from_cost(mixed, profile)
    check("unlimited-full-decode-memory-bound", full_floor["floor_kind"] == "memory")
    check(
        "unlimited-full-decode-floor-exact",
        math.isclose(full_floor["memory_floor_ms"], 4.953126, abs_tol=1e-6),
        floor_ms=full_floor["memory_floor_ms"],
    )
    check("unlimited-mixed-decode-memory-bound", mixed_floor["floor_kind"] == "memory")
    check(
        "unlimited-mixed-decode-floor-exact",
        math.isclose(mixed_floor["memory_floor_ms"], 11.049809, abs_tol=1e-6),
        floor_ms=mixed_floor["memory_floor_ms"],
    )
    # With an absurdly fast DRAM the same stage flips to compute-bound.
    flipped = floor_from_cost(full, profile | {"dram_gb_s": 1e6})
    check("floor-kind-flips-with-profile", flipped["floor_kind"] == "compute")
    check(
        "compute-floor-consistent",
        math.isclose(flipped["compute_floor_ms"], full_floor["compute_floor_ms"]),
    )

    # GOT decode refuses without a measured context length (full causal).
    try:
        decode_per_token_cost("got-ocr2", "int8", None)
        check("got-refuses-unmeasured-kv", False)
    except RooflineError:
        check("got-refuses-unmeasured-kv", True)

    # Prefill refuses a fabricated token count; accepts a measured one.
    try:
        prefill_cost("unlimited-ocr", DECODE_MIXED_FFN_INT8, 0)
        check("prefill-refuses-no-tokens", False)
    except RooflineError:
        check("prefill-refuses-no-tokens", True)
    pf_mixed = prefill_cost("unlimited-ocr", DECODE_MIXED_FFN_INT8, 289)
    pf_full = prefill_cost("unlimited-ocr", DECODE_FULL_INT8, 289)
    check(
        "mixed-prefill-macs-scale-with-tokens",
        pf_mixed["int8_macs"] == 289 * (dense0 + moe),
    )
    check(
        "full-prefill-macs-scale-with-tokens",
        pf_full["int8_macs"] == 289 * (attn + dense0 + moe),
    )

    # Vision floor: per-view scaling.
    v1 = vision_cost("unlimited-ocr", 1)
    v3 = vision_cost("unlimited-ocr", 3)
    check("vision-flops-scale-with-views", v3["f32_flops"] == 3 * v1["f32_flops"])
    check("vision-weights-once", v3["bytes"] == v1["bytes"])

    # int8 floors refuse a non-int8 precision rather than mislabeling the row.
    try:
        decode_per_token_cost("unlimited-ocr", "f32", None)
        check("refuses-f32-decode-model", False)
    except RooflineError:
        check("refuses-f32-decode-model", True)

    contract_doc = {
        "schema": STAGES_SCHEMA,
        "source": "focr",
        "precision": "focr-mixed-ffn-int8",
        "decode_mode": DECODE_MIXED_FFN_INT8,
        "quant_recipe": UNLIMITED_QUANT_RECIPE,
        "model_sha256": "a" * 64,
        "model_size": 4_157_448_783,
        "stages": [
            {
                "schema": STAGE_SCHEMA,
                "source": "focr",
                "stage": "decode_per_token",
                "unit": "ms",
                "samples_ms": [12.0],
                "best_ms": 12.0,
                "n": 1,
                "precision": "focr-mixed-ffn-int8",
                "tokens": 16,
            }
        ],
        "synthetic": True,
    }
    try:
        validate_precision_contract("unlimited-ocr", DECODE_MIXED_FFN_INT8, contract_doc)
        check("accepts-bound-mixed-contract", True)
    except RooflineError as err:
        check("accepts-bound-mixed-contract", False, error=str(err))
    for name, requested, mutate in (
        ("refuses-roofline-mode-mismatch", DECODE_FULL_INT8, lambda doc: None),
        (
            "refuses-timing-stage-precision-mismatch",
            DECODE_MIXED_FFN_INT8,
            lambda doc: doc["stages"][0].update(precision="focr-full-int8"),
        ),
        (
            "refuses-roofline-recipe-mismatch",
            DECODE_MIXED_FFN_INT8,
            lambda doc: doc.update(quant_recipe="legacy-or-unknown"),
        ),
    ):
        bad = json.loads(json.dumps(contract_doc))
        mutate(bad)
        try:
            validate_precision_contract("unlimited-ocr", requested, bad)
            check(name, False)
        except RooflineError:
            check(name, True)

    def refuses(name: str, operation) -> None:
        try:
            operation()
            check(name, False, error="accepted malformed evidence")
        except RooflineError as err:
            check(name, True, error=str(err))
        except Exception as err:  # pragma: no cover - reports an unclean error boundary
            check(name, False, error=f"unclean {type(err).__name__}: {err}")

    encoded_contract = json.dumps(contract_doc).encode("utf-8")
    try:
        decoded_contract = _decode_stages_json(encoded_contract, "<self-test>")
        check(
            "accepts-bounded-stage-document",
            decoded_contract["stages"][0]["stage"] == "decode_per_token",
        )
    except RooflineError as err:
        check("accepts-bounded-stage-document", False, error=str(err))

    try:
        script_bytes = _read_bounded_regular_file(__file__, MAX_STAGES_JSON_BYTES)
        check("stable-regular-file-read", script_bytes.startswith(b"#!/usr/bin/env python3"))
    except (RooflineError, OSError) as err:
        check("stable-regular-file-read", False, error=str(err))
    refuses(
        "refuses-oversize-regular-file",
        lambda: _read_bounded_regular_file(__file__, 1),
    )
    refuses(
        "refuses-final-component-symlink",
        lambda: _read_bounded_regular_file("/dev/stdin", MAX_STAGES_JSON_BYTES),
    )
    refuses(
        "refuses-special-file",
        lambda: _read_bounded_regular_file("/dev/null", MAX_STAGES_JSON_BYTES),
    )

    refuses(
        "refuses-non-object-document",
        lambda: _decode_stages_json(b"[]", "<self-test>"),
    )
    refuses(
        "refuses-duplicate-json-key",
        lambda: _decode_stages_json(b'{"schema":"a","schema":"b"}', "<self-test>"),
    )
    refuses(
        "refuses-json-nan",
        lambda: _decode_stages_json(b'{"value":NaN}', "<self-test>"),
    )
    refuses(
        "type-error-becomes-roofline-error",
        lambda: _decode_stages_json(123, "<self-test>"),
    )
    refuses(
        "recursion-error-becomes-roofline-error",
        lambda: _decode_stages_json(b"[" * 1200 + b"]" * 1200, "<self-test>"),
    )

    duplicate_stage = json.loads(json.dumps(contract_doc))
    duplicate_stage["stages"].append(json.loads(json.dumps(duplicate_stage["stages"][0])))
    refuses(
        "refuses-duplicate-stage-record",
        lambda: _validate_stages_document(duplicate_stage, "<self-test>"),
    )
    scalar_stage = json.loads(json.dumps(contract_doc))
    scalar_stage["stages"] = [1]
    refuses(
        "attribute-error-shape-becomes-roofline-error",
        lambda: _validate_stages_document(scalar_stage, "<self-test>"),
    )
    too_many_stages = json.loads(json.dumps(contract_doc))
    too_many_stages["stages"] = [
        {
            **json.loads(json.dumps(contract_doc["stages"][0])),
            "stage": f"stage_{index}",
        }
        for index in range(MAX_STAGE_RECORDS + 1)
    ]
    refuses(
        "refuses-too-many-stage-records",
        lambda: _validate_stages_document(too_many_stages, "<self-test>"),
    )
    bad_schema = json.loads(json.dumps(contract_doc))
    bad_schema["stages"][0]["schema"] = "focr-gauntlet-stage/v999"
    refuses(
        "refuses-stage-schema-mismatch",
        lambda: _validate_stages_document(bad_schema, "<self-test>"),
    )
    nonfinite = json.loads(json.dumps(contract_doc))
    nonfinite["stages"][0]["best_ms"] = math.inf
    refuses(
        "refuses-nonfinite-stage-duration",
        lambda: _validate_stages_document(nonfinite, "<self-test>"),
    )
    boolean_tokens = json.loads(json.dumps(contract_doc))
    boolean_tokens["stages"][0]["tokens"] = True
    refuses(
        "refuses-boolean-token-count",
        lambda: _validate_stages_document(boolean_tokens, "<self-test>"),
    )
    too_many_tokens = json.loads(json.dumps(contract_doc))
    too_many_tokens["stages"][0]["tokens"] = MAX_MEASURED_TOKENS + 1
    refuses(
        "refuses-excessive-token-count",
        lambda: _validate_stages_document(too_many_tokens, "<self-test>"),
    )
    oversized_samples = json.loads(json.dumps(contract_doc))
    oversized_samples["stages"][0]["samples_ms"] = [1.0] * (MAX_STAGE_SAMPLES + 1)
    oversized_samples["stages"][0]["n"] = MAX_STAGE_SAMPLES + 1
    refuses(
        "refuses-excessive-sample-count",
        lambda: _validate_stages_document(oversized_samples, "<self-test>"),
    )
    nested: object = "leaf"
    for _ in range(MAX_JSON_DEPTH + 1):
        nested = {"next": nested}
    refuses("refuses-excessive-json-depth", lambda: _validate_json_limits(nested))
    bad_profile_args = argparse.Namespace(
        profile="m4",
        dram_gb_s=math.inf,
        peak_int8_gmacs=None,
        peak_f32_gflops=None,
    )
    refuses("refuses-nonfinite-machine-profile", lambda: resolve_profile(bad_profile_args))

    if failures:
        print(f"SELF-TEST FAILED: {failures}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "gauntlet-roofline-self-test", "result": "pass"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--arch",
        choices=("unlimited-ocr", "got-ocr2", "smolvlm2", "onechart"),
        default="unlimited-ocr",
    )
    parser.add_argument(
        "--precision",
        choices=("int8", *UNLIMITED_DECODE_MODES),
        default="int8",
    )
    parser.add_argument("--stages-json", default=None, help="focr_stages.json (required)")
    parser.add_argument("--stage", action="append", choices=STAGES, default=None)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="m4")
    parser.add_argument("--dram-gb-s", type=float, default=None)
    parser.add_argument("--peak-int8-gmacs", type=float, default=None)
    parser.add_argument("--peak-f32-gflops", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if not args.stages_json or not args.out:
        parser.error("--stages-json and --out are required (floors bind to real measurements)")
    try:
        try:
            return run(args)
        except RooflineError:
            raise
        except (AttributeError, TypeError, RecursionError) as err:
            raise RooflineError(
                f"malformed timing evidence ({type(err).__name__}: {err})"
            ) from err
    except (RooflineError, OSError, json.JSONDecodeError, UnicodeDecodeError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
