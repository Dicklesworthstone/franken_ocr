#!/usr/bin/env python3
"""Differential validator + golden generator for `src/preprocess/pil_resample.rs`
(bd-30me, DISC-001 in docs/DISCREPANCIES.md).

Contains a pure-Python mirror of Pillow's src/libImaging/Resample.c 8bpc
BICUBIC path — the exact same steps the Rust module ports:

  * bicubic filter, a = -0.5, support 2.0
  * precompute_coeffs: f64 coeffs, window clamped BEFORE weighting,
    per-window renormalization by the f64 sum
  * normalize_coeffs_8bpc: round-half-away-from-zero to i32 with
    PRECISION_BITS = 32 - 8 - 2 = 22
  * two passes (horizontal THEN vertical), u8 clip between passes
  * accumulate from 1 << (PRECISION_BITS-1); clip8 = (>> 22, clamp 0/255)

Every float step is IEEE double in the C evaluation order and every integer
step is exact, so agreement here transfers to the Rust port bit-for-bit.

Run (the oracle stack pins Pillow 12.1.1 — docs/truth-pack/PINNED_SOURCES.md;
the script refuses any other version):

    uv venv /private/tmp/pilvenv --python 3.12
    uv pip install --python /private/tmp/pilvenv/bin/python 'pillow==12.1.1' numpy
    /private/tmp/pilvenv/bin/python scripts/gen_pil_bicubic_goldens.py            # differential only
    /private/tmp/pilvenv/bin/python scripts/gen_pil_bicubic_goldens.py --goldens  # + Rust constants

2026-07-01 result vs Pillow 12.1.1: 370/370 differential cases bit-exact
(sources 1x1..640x480 resized to 1x1..1024x1024, random + solid-extreme
pixels). The `--goldens` output is embedded in pil_resample.rs's tests.
"""

import math
import sys

import numpy as np
import PIL
from PIL import Image

PRECISION_BITS = 32 - 8 - 2  # Resample.c: #define PRECISION_BITS (32 - 8 - 2)
SUPPORT = 2.0  # bicubic support


def bicubic_filter(x: float) -> float:
    # Resample.c bicubic_filter, a = -0.5, exact C expression order.
    a = -0.5
    if x < 0.0:
        x = -x
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


def precompute_coeffs(in_size: int, out_size: int):
    """Resample.c precompute_coeffs for a full box (in0=0, in1=in_size)."""
    scale = in_size / out_size
    filterscale = scale if scale >= 1.0 else 1.0
    support = SUPPORT * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    ss = 1.0 / filterscale
    bounds = []
    prekk = []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        # C (int) casts truncate toward zero; Python int() matches.
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        k = [0.0] * ksize
        ww = 0.0
        for x in range(xmax):
            w = bicubic_filter((x + xmin - center + 0.5) * ss)
            k[x] = w
            ww += w
        if ww != 0.0:
            for x in range(xmax):
                k[x] /= ww
        bounds.append((xmin, xmax))
        prekk.append(k)
    # normalize_coeffs_8bpc: round half away from zero, truncating (int) cast.
    kk = []
    for k in prekk:
        row = []
        for w in k:
            if w < 0:
                row.append(int(-0.5 + w * (1 << PRECISION_BITS)))
            else:
                row.append(int(0.5 + w * (1 << PRECISION_BITS)))
        kk.append(row)
    return ksize, bounds, kk


def pass_along_width(arr: np.ndarray, out_w: int) -> np.ndarray:
    """One 8bpc resample pass along axis=1 (width). arr: uint8 [h, w, 3]."""
    _, in_w, _ = arr.shape
    _, bounds, kk = precompute_coeffs(in_w, out_w)
    # Integer coefficient matrix K[out, in] (exact; i64 sums cannot overflow).
    kmat = np.zeros((out_w, in_w), dtype=np.int64)
    for xx, (xmin, xmax) in enumerate(bounds):
        kmat[xx, xmin : xmin + xmax] = kk[xx][:xmax]
    acc = np.einsum("hwc,ow->hoc", arr.astype(np.int64), kmat)
    acc += 1 << (PRECISION_BITS - 1)
    out = np.where(
        acc <= 0, 0, np.where(acc >= (1 << PRECISION_BITS << 8), 255, acc >> PRECISION_BITS)
    )
    return out.astype(np.uint8)


def resize_pil_bicubic(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Full two-pass resize mirroring ImagingResampleInner (full box)."""
    h, w, _ = arr.shape
    out = arr
    if out_w != w:  # need_horizontal
        out = pass_along_width(out, out_w)
    if out_h != h:  # need_vertical
        out = pass_along_width(out.transpose(1, 0, 2), out_h).transpose(1, 0, 2)
    return out.copy()


def pillow_resize(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    img = Image.fromarray(arr, mode="RGB")
    return np.asarray(img.resize((out_w, out_h), Image.Resampling.BICUBIC))


def differential() -> bool:
    sizes = [
        (1, 1), (1, 7), (7, 1), (2, 2), (3, 3), (4, 4), (5, 3), (3, 5),
        (16, 16), (31, 17), (100, 40), (101, 40), (40, 100), (640, 480),
        (123, 77),
    ]
    targets = [
        (1, 1), (2, 2), (3, 7), (7, 3), (64, 26), (64, 64), (26, 64),
        (2, 8), (8, 2), (100, 40), (256, 256), (1024, 1024),
    ]
    rng = np.random.default_rng(0xF0C4)
    n = 0
    fails = 0
    for (w, h) in sizes:
        for (tw, th) in targets:
            if tw * th > 300_000 and w * h > 300_000:
                continue
            for _ in range(2):
                arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
                mine = resize_pil_bicubic(arr, tw, th)
                ref = pillow_resize(arr, tw, th)
                n += 1
                if not np.array_equal(mine, ref):
                    fails += 1
                    d = np.abs(mine.astype(int) - ref.astype(int))
                    print(
                        f"MISMATCH {w}x{h} -> {tw}x{th}: "
                        f"maxdiff={d.max()} n={np.count_nonzero(d)}"
                    )
    # Extreme-value images (clip8 saturation paths).
    for arr in (
        np.zeros((5, 9, 3), np.uint8),
        np.full((5, 9, 3), 255, np.uint8),
        np.tile(np.array([[[0, 255, 127]]], np.uint8), (9, 5, 1)),
    ):
        for (tw, th) in [(3, 3), (17, 2), (2, 17), (40, 40)]:
            mine = resize_pil_bicubic(arr, tw, th)
            ref = pillow_resize(arr, tw, th)
            n += 1
            if not np.array_equal(mine, ref):
                fails += 1
                print(f"MISMATCH extreme -> {tw}x{th}")
    print(f"Pillow {PIL.__version__}: {n} differential cases, {fails} mismatches")
    return fails == 0


def rust_bytes(arr: np.ndarray, per_line: int = 12) -> str:
    flat = arr.reshape(-1)
    lines = []
    for i in range(0, len(flat), per_line):
        lines.append("    " + ", ".join(str(int(v)) for v in flat[i : i + per_line]) + ",")
    return "\n".join(lines)


def emit_goldens() -> None:
    rng = np.random.default_rng(30_1466)  # bd-30me
    cases = [
        ("4X4_TO_2X2", (4, 4), (2, 2)),   # symmetric 2x downscale
        ("5X3_TO_7X2", (5, 3), (7, 2)),   # asymmetric: up-x, down-y
        ("3X5_TO_2X8", (3, 5), (2, 8)),   # asymmetric: down-x, up-y
        ("4X4_TO_6X6", (4, 4), (6, 6)),   # upscale (negative lobes + edge clamp)
        ("8X5_TO_3X5", (8, 5), (3, 5)),   # horizontal-only (vertical pass skipped)
        ("5X8_TO_5X3", (5, 8), (5, 3)),   # vertical-only (horizontal pass skipped)
    ]
    print(f"// Goldens generated with Pillow {PIL.__version__} (truth-pack pin)")
    for name, (w, h), (tw, th) in cases:
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        ref = pillow_resize(arr, tw, th)
        mine = resize_pil_bicubic(arr, tw, th)
        assert np.array_equal(mine, ref), name
        print(f"\nconst SRC_{name}: [u8; {h*w*3}] = [\n{rust_bytes(arr)}\n];")
        print(f"const PIL_{name}: [u8; {th*tw*3}] = [\n{rust_bytes(ref)}\n];")


if __name__ == "__main__":
    assert PIL.__version__ == "12.1.1", f"truth-pack pins Pillow 12.1.1, got {PIL.__version__}"
    if not differential():
        sys.exit(1)
    if "--goldens" in sys.argv:
        emit_goldens()
