#!/usr/bin/env python3
"""E2 (bd-3jo6.5.2): the OFFLINE Polyphonic-TrOMR checkpoint export —
`img2score_epoch47.pth` (torch pickle) → `model.safetensors`, with the
convert-time Weight-Standardization fold (tromr-spec §10.3/§11).

What it does, in census order:

1. **Provenance gate**: refuses unless the input .pth matches the census pin
   (86,254,711 bytes, sha256 02925259ef…, spec §Sources) — the 261-tensor
   inventory in §12 was extracted from exactly this file.
2. **WS fold**: every ResNetV2 backbone conv (`encoder.patch_embed.backbone.
   *conv*.weight`) is stored PRE-STANDARDIZED by INVOKING the pinned
   `timm==0.6.5` `StdConv2dSame` arithmetic itself (`F.batch_norm(..., eps
   1e-6)`, population variance — census §16), so the stored weight is
   bit-faithful to upstream's runtime WS by construction. Runtime then runs
   plain `nn::conv2d` — no WS kernel exists in Rust (§15/E3 delta).
3. **WS proof (L1)**: per conv, (a) a determinism re-run must `torch.equal`
   the fold, and (b) the analytic `(w-mean)/sqrt(var+eps)` formulation must
   agree within 1e-5 (guards a wrong-axis/eps/shape invocation; the analytic
   form itself differs from the fused kernel by ~5e-7 rounding, measured
   2026-07-05). Any violation refuses the export.
4. **Drop `decoder.note_mask`** (train-only, census §12) — everything else
   (260 tensors) carries over byte-identical (the fold touches ONLY backbone
   conv weights; norms/biases/ViT/decoder/heads are untouched f32).
5. Writes `model.safetensors` + `TROMR_EXPORT_MANIFEST.json` (per-tensor
   sha256 of the folded set, source pin, counts) beside the output.

Usage:
    python scripts/gen_tromr_safetensors.py \
        --pth  <zoo>/tromr-upstream/tromr/workspace/checkpoints/img2score_epoch47.pth \
        --out  <zoo>/tromr/model.safetensors

Requires: torch, safetensors, timm==0.6.5 (the pinned reference for the proof).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

PIN_BYTES = 86_254_711
PIN_SHA256 = "02925259ef59f5578a8c9e954ac363bb15538ea38ce73090b861c1519179f910"
WS_EPS = 1e-6  # census §16: population variance, eps 1e-6
EXPECTED_TENSORS = 261  # census §12 (incl. the dropped note_mask)
DROP = ("decoder.note_mask",)  # train-only (census §12)
BACKBONE_PREFIX = "encoder.patch_embed.backbone."


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_ws_conv(name: str, shape) -> bool:
    """The WS-folded set: backbone conv WEIGHTS only (4-D), never norms.

    ResNetV2's StdConv2dSame standardizes stem.conv, every blocks.*.conv{1,2,3}
    and every downsample.conv — i.e. every 4-D `.weight` under the backbone.
    GN weights/biases are 1-D and stay untouched.
    """
    return name.startswith(BACKBONE_PREFIX) and name.endswith(".weight") and len(shape) == 4


def ws_fold(w):
    """The fold IS the pinned timm 0.6.5 StdConv2dSame arithmetic — the stored
    weight must be BIT-IDENTICAL to what upstream's runtime WS computes (so our
    plain conv reproduces their standardized conv bitwise):

        weight = F.batch_norm(self.weight.reshape(1, out, -1), None, None,
                              training=True, momentum=0., eps=self.eps)
                  .reshape_as(self.weight)

    (The analytic (w-mean)/sqrt(var+eps) form differs from this fused kernel
    by ~5e-7 float rounding — measured 2026-07-05; hence invoke, don't mimic.)
    """
    import torch.nn.functional as F

    return F.batch_norm(
        w.reshape(1, w.shape[0], -1), None, None, training=True, momentum=0.0, eps=WS_EPS
    ).reshape_as(w)


def analytic_cross_check(w, folded) -> float:
    """Guard against invoking the reference WRONGLY (axis/eps/shape bugs): the
    analytic population-variance formulation must agree to ~float-rounding.
    Returns the maxabs delta (caller enforces the 1e-5 sanity bound)."""
    import torch

    v = w.reshape(w.shape[0], -1)
    mean = v.mean(dim=1, keepdim=True)
    var = v.var(dim=1, unbiased=False, keepdim=True)
    analytic = ((v - mean) / torch.sqrt(var + WS_EPS)).reshape_as(w)
    return (analytic - folded).abs().max().item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    size = os.path.getsize(args.pth)
    digest = sha256_file(args.pth)
    if size != PIN_BYTES or digest != PIN_SHA256:
        print(
            f"FATAL: {args.pth} does not match the census pin "
            f"(size {size} vs {PIN_BYTES}, sha256 {digest[:16]}… vs {PIN_SHA256[:16]}…)",
            file=sys.stderr,
        )
        return 1

    import torch
    from safetensors.torch import save_file

    state = torch.load(args.pth, map_location="cpu", weights_only=True)
    if len(state) != EXPECTED_TENSORS:
        print(f"FATAL: {len(state)} tensors, census expects {EXPECTED_TENSORS}", file=sys.stderr)
        return 1

    out_state: dict = {}
    folded_names: list[str] = []
    for name, tensor in state.items():
        if name in DROP:
            continue
        tensor = tensor.contiguous()
        if tensor.dtype != torch.float32:
            print(f"FATAL: {name} is {tensor.dtype}, census says all-fp32", file=sys.stderr)
            return 1
        if is_ws_conv(name, tensor.shape):
            folded = ws_fold(tensor)
            # Determinism proof: the reference arithmetic must reproduce itself
            # bit-exactly (a nondeterministic kernel could not be blessed).
            if not torch.equal(folded, ws_fold(tensor)):
                print(f"FATAL: WS fold nondeterministic for {name}", file=sys.stderr)
                return 1
            delta = analytic_cross_check(tensor, folded)
            if delta > 1e-5:
                print(
                    f"FATAL: WS fold sanity FAILED for {name} (analytic delta {delta:.3e} "
                    "> 1e-5 — wrong axis/eps/shape?)",
                    file=sys.stderr,
                )
                return 1
            out_state[name] = folded
            folded_names.append(name)
        else:
            out_state[name] = tensor

    # Census cross-checks: 3 stages × {2,3,7} blocks × conv{1,2,3} + 3
    # downsamples + the stem = 40 folded convs.
    if len(folded_names) != 40:
        print(f"FATAL: folded {len(folded_names)} convs, census layout expects 40", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out_state, args.out)

    manifest = {
        "purpose": "TrOMR E2 offline export (bd-3jo6.5.2) — WS-folded, note_mask dropped",
        "script": "scripts/gen_tromr_safetensors.py",
        "source_pth": {"bytes": PIN_BYTES, "sha256": PIN_SHA256},
        "ws_fold": {
            "eps": WS_EPS,
            "variance": "population (unbiased=False)",
            "proof": "fold INVOKES timm==0.6.5 StdConv2dSame F.batch_norm arithmetic (bit-faithful by construction); determinism re-run torch.equal + analytic cross-check <= 1e-5 per conv",
            "folded_convs": folded_names,
        },
        "dropped": list(DROP),
        "tensors_out": len(out_state),
        "model_safetensors_sha256": sha256_file(args.out),
        "license": "Apache-2.0 (NetEase Polyphonic-TrOMR — NOTICE carried to distribution)",
    }
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "TROMR_EXPORT_MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                "event": "tromr_export",
                "result": "pass",
                "tensors": len(out_state),
                "ws_folded": len(folded_names),
                "out": args.out,
                "manifest": manifest_path,
                "sha256": manifest["model_safetensors_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
