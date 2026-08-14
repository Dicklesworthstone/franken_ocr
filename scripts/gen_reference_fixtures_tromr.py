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
   the same staff. The three id streams (POSITIONAL rhythm/pitch/lift — the
   §4 naming-swap trap cancels), every head's step-0 logits, and the complete
   per-step logits for all four heads are dumped. The latter are conditioned
   on the exact free-running argmax prefix and are the parity oracle for the
   retained Spohr-row diagnostic in `native_engine::tromr`.

Outputs (beside the zoo model, NOT committed — multi-MB):
    <zoo>/tromr_preproc.bin          f32 LE, the (1,128,W) readimg tensor
    <zoo>/tromr_seam_<name>.bin      one flat f32-LE file per hooked seam
                                     (flat .bin like every other lane — the
                                     Rust certs have no npz reader)
    <zoo>/tromr_oracle_fixtures.json shapes + floor + provenance + sha256s

Usage:  gen_reference_fixtures_tromr.py  [--upstream DIR] [--zoo DIR]
            [--page PNG] [--artifact-prefix NAME] [--fixture-name NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys


CHECKPOINT_SHA256 = "02925259ef59f5578a8c9e954ac363bb15538ea38ce73090b861c1519179f910"
HEAD_VOCABULARIES = {"rhythm": 260, "pitch": 71, "lift": 7, "note": 2}
FULL_DECODE_SCHEMA = "franken_ocr.tromr.upstream_free_argmax_full_logits.v1"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f32le_bytes(np, values) -> bytes:
    """Canonical contiguous little-endian f32 tensor bytes."""
    return np.asarray(values, dtype="<f4", order="C").tobytes(order="C")


def write_f32le_tensor(np, out_dir: str, filename: str, values) -> dict:
    array = np.asarray(values, dtype=np.float32, order="C")
    payload = f32le_bytes(np, array)
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        f.write(payload)
    return {
        "file": filename,
        "shape": list(array.shape),
        "dtype": "f32",
        "byte_order": "little",
        "layout": "row_major_c_contiguous",
        "byte_len": len(payload),
        "sha256": sha256_bytes(payload),
    }


def append_last_head_logits(np, captured: dict, name: str, values) -> None:
    """Retain exactly the final sequence position from one decoder step."""
    array = np.asarray(values, dtype=np.float32)
    expected = HEAD_VOCABULARIES[name]
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != expected:
        raise RuntimeError(
            f"{name} head emitted shape {array.shape}, expected (1, steps, {expected})"
        )
    row = np.asarray(array[0, -1, :], dtype=np.float32, order="C")
    if not np.isfinite(row).all():
        raise RuntimeError(f"{name} head emitted non-finite logits")
    captured[name].append(row)


def self_test() -> int:
    import numpy as np

    captured = {name: [] for name in HEAD_VOCABULARIES}
    for name, width in HEAD_VOCABULARIES.items():
        values = np.arange(2 * width, dtype=np.float32).reshape(1, 2, width)
        append_last_head_logits(np, captured, name, values)
        assert captured[name][0].shape == (width,)
        assert float(captured[name][0][0]) == float(width)
        assert float(captured[name][0][-1]) == float(2 * width - 1)
    probe = np.asarray([0.0, 1.0, -2.5], dtype=np.float32)
    assert f32le_bytes(np, probe).hex() == "000000000000803f000020c0"
    assert sha256_bytes(f32le_bytes(np, probe)) == (
        "4356516ed57de986ba8080c557e8856871336d6a17b170fb946df125605466c9"
    )
    print(json.dumps({"event": "tromr_fixture_self_test", "result": "pass"}))
    return 0


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
    parser.add_argument(
        "--page",
        help="exact staff PNG; defaults to the pinned upstream examples/1.png",
    )
    parser.add_argument(
        "--expected-page-sha256",
        help="fail closed unless --page has this exact SHA-256",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="tromr",
        help="filename prefix for binary artifacts (default preserves legacy names)",
    )
    parser.add_argument(
        "--fixture-name",
        default="tromr_oracle_fixtures.json",
        help="JSON manifest filename under --zoo",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.artifact_prefix or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for ch in args.artifact_prefix
    ):
        raise SystemExit("FATAL: --artifact-prefix must be a non-empty portable filename token")
    if os.path.basename(args.fixture_name) != args.fixture_name:
        raise SystemExit("FATAL: --fixture-name must be a basename, not a path")

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

    page = args.page or os.path.join(args.upstream, "examples", "1.png")
    page_sha256 = sha256_file(page)
    if args.expected_page_sha256 and page_sha256 != args.expected_page_sha256:
        raise SystemExit(
            "FATAL: exact page SHA-256 mismatch: "
            f"got {page_sha256}, expected {args.expected_page_sha256}"
        )
    checkpoint_sha256 = sha256_file(ckpt)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise SystemExit(
            f"FATAL: checkpoint SHA-256 {checkpoint_sha256} != pin {CHECKPOINT_SHA256}"
        )
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

    # ── E4 decoder leg: argmax-forced full generate + every head/step logit ──
    head_step0: dict = {}
    head_steps = {name: [] for name in HEAD_VOCABULARIES}

    def grab_head(name):
        def hook(_m, _i, out):
            values = out.detach().cpu().float().numpy()
            if name not in head_step0:
                head_step0[name] = values
            append_last_head_logits(np, head_steps, name, values)

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
        for hook in head_hooks:
            hook.remove()
        head_hooks.clear()
        with torch.inference_mode():
            g2_rhythm, g2_pitch, g2_lift = model.generate(xt, temperature=0.2)
    finally:
        torch.multinomial = real_multinomial
        for hook in head_hooks:
            hook.remove()
    streams = {
        "rhythm": [int(v) for v in g_rhythm[0].tolist()],
        "pitch": [int(v) for v in g_pitch[0].tolist()],
        "lift": [int(v) for v in g_lift[0].tolist()],
    }
    argmax_deterministic = (
        g_rhythm.equal(g2_rhythm) and g_pitch.equal(g2_pitch) and g_lift.equal(g2_lift)
    )
    step_count = len(streams["rhythm"])
    if any(len(rows) != step_count for rows in head_steps.values()):
        raise RuntimeError(
            "full-logit capture count does not match the free argmax stream: "
            f"steps={step_count}, captures={ {k: len(v) for k, v in head_steps.items()} }"
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
    pre_name = f"{args.artifact_prefix}_preproc.bin"
    pre_path = os.path.join(args.zoo, pre_name)
    x.astype("<f4").tofile(pre_path)
    seams["encoder_out"] = final
    for name, arr in head_step0.items():
        seams[f"head0_{name}"] = arr
    seam_files = {}
    for name, arr in seams.items():
        p = os.path.join(args.zoo, f"{args.artifact_prefix}_seam_{name}.bin")
        arr.astype("<f4").tofile(p)
        seam_files[name] = p

    full_logits = {}
    for name, rows in head_steps.items():
        values = np.stack(rows, axis=0)
        full_logits[name] = write_f32le_tensor(
            np,
            args.zoo,
            f"{args.artifact_prefix}_free_argmax_{name}_logits.f32le.bin",
            values,
        )

    preproc_descriptor = {
        "file": pre_name,
        "shape": list(x.shape),
        "dtype": "f32",
        "byte_order": "little",
        "layout": "chw_c_contiguous",
        "byte_len": os.path.getsize(pre_path),
        "sha256": sha256_file(pre_path),
    }
    try:
        upstream_commit = subprocess.run(
            ["git", "-C", args.upstream, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        upstream_commit = None

    meta = {
        "_meta": {
            "purpose": "TrOMR encoder oracle fixtures (E3 seams + floor)",
            "script": "scripts/gen_reference_fixtures_tromr.py",
            "page": page,
            "page_sha256": page_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "upstream_commit": upstream_commit,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pins": "timm==0.6.5, x-transformers==0.29.2 (upstream code paths)",
        },
        "preproc": preproc_descriptor,
        "seams": {k: list(v.shape) for k, v in seams.items()},
        "nondeterminism_floor": {
            "encoder_out_maxabs_same_thread": floor_same,
            "encoder_out_maxabs_cross_thread": floor_threads,
            "argmax_generate_deterministic": argmax_deterministic,
        },
        "argmax_generate": streams,
        "free_argmax_full_logits": {
            "schema": FULL_DECODE_SCHEMA,
            "prefix_contract": (
                "seed_rhythm_1_pitch_0_lift_0_then_free_argmax_previous_tokens_v1"
            ),
            "step_count": step_count,
            "streams": streams,
            "heads": full_logits,
        },
        "files_sha256": {
            os.path.basename(p): sha256_file(p) for p in [pre_path, *seam_files.values()]
        },
    }
    meta["files_sha256"].update(
        {descriptor["file"]: descriptor["sha256"] for descriptor in full_logits.values()}
    )
    fx_path = os.path.join(args.zoo, args.fixture_name)
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
                "full_logit_shapes": {
                    k: descriptor["shape"] for k, descriptor in full_logits.items()
                },
                "page_sha256": page_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "seams": sorted(seams.keys()),
                "out": fx_path,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
