#!/usr/bin/env python3
"""E3 (bd-3jo6.5.3): TrOMR ENCODER reference-oracle fixtures — establish the
oracle's own nondeterminism floor FIRST, then dump the seams the Rust encoder
certs compare against (LADDER_HARNESS.md §9 recipe; tromr-spec §2a/§2b/§6).

Loads the REAL upstream model (tromr-upstream clone: pinned timm==0.6.5 +
x-transformers==0.29.2 code paths, the census-pinned checkpoint) and runs the
committed example staff `examples/1.png` through:

1. **readimg preprocess** (spec §6, reproduced here from the pinned sources:
   cv2.imread → BGR2RGB → resize(h=128, w floored to ×16, INTER_LINEAR) →
   cv2 RGB2GRAY fixed-point luma → uint8 round → replicate ×3 →
   `(px − 0.7931·255)/(0.1738·255)` → channel 0). albumentations 1.2.0 itself
   is NOT importable on this python (scikit-image 0.18.3 has no wheels); its
   two transforms used here (ToGray, Normalize) are exactly the cv2.cvtColor +
   the linear normalize above (albumentations/augmentations/functional.py at
   1.2.0 — OQ-T3 pinned by delegating the fixed-point step to cv2 itself).
2. **encoder seams** via forward hooks: backbone stem, stage 0/1/2, the 1×1
   patch proj, each of the 4 ViT blocks, the final encoder LayerNorm output.
3. **floor**: the full encoder runs twice @1 torch thread and once @2 threads;
   the fixture records the same-thread and cross-thread maxabs of the FINAL
   output — the L1/L2 tolerances derive from these, never guessed.
4. **decoder leg (E4)**: `torch.multinomial` is monkeypatched to argmax (the
   port's deterministic default — spec §5 port decision; upstream sampling is
   the FOCR_TROMR_SAMPLE kill-switch) and the full `model.generate` runs on
   the same staff: the three id streams (POSITIONAL rhythm/pitch/lift — the
   §4 naming-swap trap cancels) and each head's step-0 logits are dumped.

Outputs (beside the zoo model, NOT committed — multi-MB):
    <zoo>/tromr_preproc.bin          f32 LE, the (1,128,W) readimg tensor
    <zoo>/tromr_seam_<name>.bin      one flat f32-LE file per hooked seam
                                     (flat .bin like every other lane — the
                                     Rust certs have no npz reader)
    <zoo>/tromr_oracle_fixtures.json shapes + floor + provenance + sha256s

Usage:  gen_reference_fixtures_tromr.py  [--upstream DIR] [--zoo DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def readimg(cv2, np, path: str):
    """spec §6, byte-faithful: the L0 reference preprocess."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"FATAL: cannot read {path}")
    if img.ndim == 3 and img.shape[2] == 4 and img[:, :, 3].min() < 255:
        # Inverted alpha = ink (rendered-PNG convention) — ONLY when the alpha
        # channel actually varies. Upstream applies 255−alpha to EVERY
        # 4-channel input, which BLANKS fully-opaque PNGs (their own
        # examples/*.png are opaque RGBA: alpha ≡ 255 ⇒ ink ≡ 0 — measured
        # 2026-07-06, DISC-007). Deliberate, documented divergence.
        img = 255 - img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise SystemExit(f"FATAL: unsupported channel count {img.shape}")
    h, w, _ = img.shape
    new_h = 128
    new_w = int(new_h / h * w) // 16 * 16
    img = cv2.resize(img, (new_w, new_h))  # INTER_LINEAR default
    # albumentations-1.2.0 ToGray: cv2 fixed-point luma, uint8, replicate ×3.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # Normalize(mean=0.7931, std=0.1738, max_pixel_value=255) then CHW ch-0.
    x = (img.astype(np.float32) - 0.7931 * 255.0) / (0.1738 * 255.0)
    return x.transpose(2, 0, 1)[:1]  # (1, 128, W)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream", default="/Volumes/USBNVME16TB/temp_agent_space/zoo/tromr-upstream"
    )
    parser.add_argument("--zoo", default="/Volumes/USBNVME16TB/temp_agent_space/zoo/tromr")
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, os.path.join(args.upstream, "tromr"))
    from configs import getconfig  # noqa: PLC0415 — upstream module
    from model.tromr_arch import TrOMR  # noqa: PLC0415

    cfg_path = os.path.join(args.upstream, "tromr", "workspace", "config.yaml")
    ckpt = os.path.join(args.upstream, "tromr", "workspace", "checkpoints", "img2score_epoch47.pth")
    conf = getconfig(cfg_path)
    model = TrOMR(conf)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.train(False)

    page = os.path.join(args.upstream, "examples", "1.png")
    x = readimg(cv2, np, page)
    xt = torch.from_numpy(x).unsqueeze(0)  # (1, 1, 128, W)

    # ── seam hooks over the encoder ──────────────────────────────────────
    enc = model.encoder
    seams: dict = {}

    def grab(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            seams[name] = t.detach().float().numpy()

        return hook

    backbone = enc.patch_embed.backbone
    hooks = [
        backbone.stem.register_forward_hook(grab("stem")),
        backbone.stages[0].register_forward_hook(grab("stage0")),
        backbone.stages[1].register_forward_hook(grab("stage1")),
        backbone.stages[2].register_forward_hook(grab("stage2")),
        enc.patch_embed.register_forward_hook(grab("patch_embed")),
        enc.norm.register_forward_hook(grab("encoder_norm")),
    ]
    for i, blk in enumerate(enc.blocks):
        hooks.append(blk.register_forward_hook(grab(f"vit_block{i}")))

    def run():
        with torch.inference_mode():
            return enc(xt).detach().float().numpy()

    # ── E4 decoder leg: argmax-forced full generate + step-0 head logits ──
    head_step0: dict = {}

    def grab_head(name):
        def hook(_m, _i, out):
            if name not in head_step0:  # step 0 only
                head_step0[name] = out.detach().float().numpy()

        return hook

    head_hooks = [
        getattr(model.decoder.net, f"to_logits_{h}").register_forward_hook(grab_head(h))
        for h in ("rhythm", "pitch", "lift", "note")
    ]
    real_multinomial = torch.multinomial
    torch.multinomial = lambda probs, n, **kw: probs.argmax(-1, keepdim=True)
    try:
        torch.set_num_threads(1)
        with torch.inference_mode():
            g_rhythm, g_pitch, g_lift = model.generate(xt, temperature=0.2)
        with torch.inference_mode():
            g2_rhythm, g2_pitch, g2_lift = model.generate(xt, temperature=0.2)
    finally:
        torch.multinomial = real_multinomial
        for h in head_hooks:
            h.remove()
    streams = {
        "rhythm": [int(v) for v in g_rhythm[0].tolist()],
        "pitch": [int(v) for v in g_pitch[0].tolist()],
        "lift": [int(v) for v in g_lift[0].tolist()],
    }
    argmax_deterministic = (
        g_rhythm.equal(g2_rhythm) and g_pitch.equal(g2_pitch) and g_lift.equal(g2_lift)
    )

    # ── the oracle's own floor FIRST (two runs @1 thread, one @2) ────────
    torch.set_num_threads(1)
    out1 = run()
    out2 = run()
    torch.set_num_threads(2)
    out3 = run()
    torch.set_num_threads(1)
    seams.clear()
    final = run()  # the blessed pass (hooks fill `seams`)
    for h in hooks:
        h.remove()
    floor_same = float(np.max(np.abs(out1 - out2)))
    floor_threads = float(np.max(np.abs(out1 - out3)))

    os.makedirs(args.zoo, exist_ok=True)
    pre_path = os.path.join(args.zoo, "tromr_preproc.bin")
    x.astype("<f4").tofile(pre_path)
    seams["encoder_out"] = final
    for name, arr in head_step0.items():
        seams[f"head0_{name}"] = arr
    seam_files = {}
    for name, arr in seams.items():
        p = os.path.join(args.zoo, f"tromr_seam_{name}.bin")
        arr.astype("<f4").tofile(p)
        seam_files[name] = p

    meta = {
        "_meta": {
            "purpose": "TrOMR encoder oracle fixtures (E3 seams + floor)",
            "script": "scripts/gen_reference_fixtures_tromr.py",
            "page": page,
            "page_sha256": sha256_file(page),
            "checkpoint_sha256": sha256_file(ckpt),
            "torch": torch.__version__,
            "pins": "timm==0.6.5, x-transformers==0.29.2 (upstream code paths)",
        },
        "preproc": {"shape": list(x.shape), "file": os.path.basename(pre_path)},
        "seams": {k: list(v.shape) for k, v in seams.items()},
        "nondeterminism_floor": {
            "encoder_out_maxabs_same_thread": floor_same,
            "encoder_out_maxabs_cross_thread": floor_threads,
            "argmax_generate_deterministic": argmax_deterministic,
        },
        "argmax_generate": streams,
        "files_sha256": {
            os.path.basename(p): sha256_file(p)
            for p in [pre_path, *seam_files.values()]
        },
    }
    fx_path = os.path.join(args.zoo, "tromr_oracle_fixtures.json")
    with open(fx_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    print(
        json.dumps(
            {
                "event": "tromr_encoder_fixtures",
                "result": "pass",
                "preproc_shape": list(x.shape),
                "encoder_out_shape": list(final.shape),
                "floor_same": floor_same,
                "floor_threads": floor_threads,
                "argmax_deterministic": argmax_deterministic,
                "stream_lens": {k: len(v) for k, v in streams.items()},
                "seams": sorted(seams.keys()),
                "out": fx_path,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
