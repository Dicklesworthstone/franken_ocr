//! Pillow-bit-exact resampler — **BICUBIC** (the `FOCR_RESAMPLE=pil-bicubic`
//! reference path for the Baidu/GOT sites, bd-30me, DISC-001) and **LANCZOS**
//! (the SmolVLM2 preprocess default, spec §6 — [`resize_lanczos`], C7). Both
//! run the same `Resample.c` fixed-point pipeline; only the
//! `(filter, support)` table entry differs.
//!
//! The oracle preprocess resizes with PIL BICUBIC: `ImageOps.pad` at
//! `modeling_unlimitedocr.py:872`, `dynamic_preprocess`'s
//! `image.resize((W, H))` at `:197` (BICUBIC is `Image.resize`'s default),
//! and GOT-OCR2's `GOTImageEvalProcessor` squash resize (spec §13b). The
//! shipped default routes those sites to the `image` crate's CatmullRom —
//! the same `a = -0.5` continuous cubic, but with clamp-at-edge sampling and
//! float accumulation, so it is **not** bit-identical to PIL. This module
//! reimplements Pillow's `src/libImaging/Resample.c` 8-bit fixed-point
//! pipeline step for step so the L0 EXACT gate can compare preprocessed
//! tensors byte-for-byte against the torch/PIL oracle:
//!
//! * **Two passes** — horizontal then vertical, each an independent 1-D
//!   convolution; the intermediate image is clipped back to `u8` between
//!   passes (this inter-pass quantization is load-bearing for exactness).
//! * **Precomputed `f64` coefficients** (`precompute_coeffs`): the sample
//!   window is clamped to the image *before* weights are computed, and the
//!   surviving weights are renormalized by their own `f64` sum — edge
//!   semantics that differ from the crate's clamp-at-edge sampling.
//! * **Fixed-point conversion** (`normalize_coeffs_8bpc`): each coefficient
//!   is scaled by `1 << 22` (`PRECISION_BITS = 32 - 8 - 2`) and rounded half
//!   away from zero with a truncating cast.
//! * **Accumulation** starts at `1 << 21` (a pre-added rounding half) in
//!   `i32`; `clip8` shifts down by 22 and saturates to `[0, 255]`.
//!
//! Every float step keeps the exact C expression/evaluation order (IEEE f64,
//! no reassociation) and every integer step is exact, so the port is
//! bit-identical by construction. Evidence: a pure-Python mirror of exactly
//! these steps matched the pinned oracle **Pillow 12.1.1**
//! (`docs/truth-pack/PINNED_SOURCES.md` runtime pin) on **370/370**
//! randomized differential cases — sources 1×1..640×480 resized to
//! 1×1..1024×1024, random plus solid-extreme pixels
//! (`scripts/gen_pil_bicubic_goldens.py`, 2026-07-01) — and the goldens in
//! the tests below are Pillow 12.1.1 outputs embedded as constants.

use image::RgbImage;

/// Fixed-point precision of Pillow's 8-bit resample path
/// (`Resample.c`: `#define PRECISION_BITS (32 - 8 - 2)`).
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Pillow's `bicubic_filter`, `a = -0.5` (the Keys cubic). The expression
/// order is the C source's, so the doubles are IEEE-identical.
fn bicubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Pillow's `sinc_filter` (`Resample.c`), operand order preserved.
fn sinc_filter(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

/// Pillow's `lanczos_filter` (`Resample.c`: truncated sinc, `{lanczos_filter,
/// 3.0}`) — the SmolVLM2 preprocess resample (`preprocessor_config.json
/// resample: 1` = `PILImageResampling.LANCZOS`). The asymmetric window test
/// (`-3.0 <= x && x < 3.0`) is the C source's, kept verbatim.
fn lanczos_filter(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc_filter(x) * sinc_filter(x / 3.0)
    } else {
        0.0
    }
}

/// A Pillow resample filter: the `(filter fn, support)` pair from
/// `Resample.c`'s filter table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PilFilter {
    /// `{bicubic_filter, 2.0}` — the Baidu/GOT oracle resample (DISC-001).
    Bicubic,
    /// `{lanczos_filter, 3.0}` — the SmolVLM2 oracle resample.
    Lanczos,
}

impl PilFilter {
    fn support(self) -> f64 {
        match self {
            PilFilter::Bicubic => 2.0,
            PilFilter::Lanczos => 3.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        match self {
            PilFilter::Bicubic => bicubic_filter(x),
            PilFilter::Lanczos => lanczos_filter(x),
        }
    }
}

/// One pass's precomputed integer coefficients (`precompute_coeffs` +
/// `normalize_coeffs_8bpc`): for output index `i`, the source window starts
/// at `bounds[i].0` and spans `bounds[i].1` taps whose fixed-point weights
/// are `kk[i * ksize..][..bounds[i].1]` (tail slots zero-filled, as in C).
struct PassCoeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    kk: Vec<i32>,
}

/// `Resample.c precompute_coeffs` for a full box (`in0 = 0`, `in1 = in_size`
/// — `Image.resize` always passes the whole image as the box in the oracle
/// preprocess), followed by `normalize_coeffs_8bpc`.
fn precompute_coeffs(in_size: u32, out_size: u32, filter: PilFilter) -> PassCoeffs {
    let scale = f64::from(in_size) / f64::from(out_size);
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = filter.support() * filterscale;
    // C: `ksize = (int)ceil(support) * 2 + 1`.
    let ksize = (support.ceil() as usize) * 2 + 1;
    let ss = 1.0 / filterscale;

    let mut bounds = Vec::with_capacity(out_size as usize);
    let mut prekk = vec![0.0f64; out_size as usize * ksize];
    for xx in 0..out_size as usize {
        let center = (xx as f64 + 0.5) * scale;
        // C truncating `(int)` casts (toward zero) — Rust `as` matches.
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = ((center + support + 0.5) as i64).min(i64::from(in_size));
        let taps = usize::try_from((xmax - xmin).max(0))
            .expect("pil_resample: window tap count fits usize");
        let xmin = usize::try_from(xmin).expect("pil_resample: window start fits usize");
        let k = &mut prekk[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0f64;
        for (x, slot) in k.iter_mut().enumerate().take(taps) {
            // C: `filter((x + xmin - center + 0.5) * ss)` — integer add
            // first, then the f64 subtraction chain, left to right.
            let w = filter.eval(((x + xmin) as f64 - center + 0.5) * ss);
            *slot = w;
            ww += w;
        }
        if ww != 0.0 {
            for slot in k.iter_mut().take(taps) {
                *slot /= ww;
            }
        }
        // Slots past `taps` stay 0.0 — Resample.c zero-fills them explicitly.
        bounds.push((xmin, taps));
    }

    // `normalize_coeffs_8bpc`: scale by 2^22 and round half away from zero
    // via a truncating cast, keeping the C operand order.
    let fixed_one = f64::from(1i32 << PRECISION_BITS);
    let kk = prekk
        .iter()
        .map(|&w| {
            if w < 0.0 {
                (-0.5 + w * fixed_one) as i32
            } else {
                (0.5 + w * fixed_one) as i32
            }
        })
        .collect();
    PassCoeffs { ksize, bounds, kk }
}

/// `Resample.c clip8`: shift the fixed-point accumulator down and saturate
/// to `u8`. `1 << PRECISION_BITS << 8` is `1 << 30`, safely inside `i32`.
fn clip8(v: i32) -> u8 {
    if v >= 1 << PRECISION_BITS << 8 {
        255
    } else if v <= 0 {
        0
    } else {
        (v >> PRECISION_BITS) as u8
    }
}

/// One horizontal pass (`ImagingResampleHorizontal_8bpc`): every output
/// pixel of every row is the clipped fixed-point dot product of its window.
fn resample_horizontal(src: &RgbImage, out_w: u32, filter: PilFilter) -> RgbImage {
    let coeffs = precompute_coeffs(src.width(), out_w, filter);
    let mut out = RgbImage::new(out_w, src.height());
    for yy in 0..src.height() {
        for xx in 0..out_w {
            let (xmin, taps) = coeffs.bounds[xx as usize];
            let k = &coeffs.kk[xx as usize * coeffs.ksize..];
            // Accumulators start at the pre-added rounding half (1 << 21).
            let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
            for (x, &w) in k.iter().enumerate().take(taps) {
                let p = src.get_pixel((xmin + x) as u32, yy).0;
                for c in 0..3 {
                    ss[c] += i32::from(p[c]) * w;
                }
            }
            out.put_pixel(
                xx,
                yy,
                image::Rgb([clip8(ss[0]), clip8(ss[1]), clip8(ss[2])]),
            );
        }
    }
    out
}

/// One vertical pass (`ImagingResampleVertical_8bpc`) — the same dot product
/// down columns of the (already horizontally resampled, u8-clipped) image.
fn resample_vertical(src: &RgbImage, out_h: u32, filter: PilFilter) -> RgbImage {
    let coeffs = precompute_coeffs(src.height(), out_h, filter);
    let mut out = RgbImage::new(src.width(), out_h);
    for yy in 0..out_h {
        let (ymin, taps) = coeffs.bounds[yy as usize];
        let k = &coeffs.kk[yy as usize * coeffs.ksize..];
        for xx in 0..src.width() {
            let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
            for (y, &w) in k.iter().enumerate().take(taps) {
                let p = src.get_pixel(xx, (ymin + y) as u32).0;
                for c in 0..3 {
                    ss[c] += i32::from(p[c]) * w;
                }
            }
            out.put_pixel(
                xx,
                yy,
                image::Rgb([clip8(ss[0]), clip8(ss[1]), clip8(ss[2])]),
            );
        }
    }
    out
}

/// Resize `src` to `out_w × out_h` exactly as the pinned oracle Pillow
/// 12.1.1's `Image.resize((out_w, out_h), Image.Resampling.BICUBIC)` does on
/// an RGB image (full box) — bit-identical output bytes.
///
/// Mirrors `ImagingResampleInner`: a pass is skipped when its dimension is
/// unchanged (`need_horizontal` / `need_vertical`), horizontal runs first,
/// and an unchanged size returns a plain copy (PIL's `self.copy()`). A
/// degenerate zero-sized `src` (unreachable from any decoder) yields a black
/// output: every clamped window is empty, so `clip8(1 << 21)` = 0.
///
/// # Panics
/// If `out_w` or `out_h` is 0 (Pillow raises `ValueError`; every preprocess
/// caller validates its target size long before this point).
#[must_use]
pub fn resize_bicubic(src: &RgbImage, out_w: u32, out_h: u32) -> RgbImage {
    resize_with(src, out_w, out_h, PilFilter::Bicubic)
}

/// Resize `src` to `out_w × out_h` exactly as Pillow's
/// `Image.resize((out_w, out_h), Image.Resampling.LANCZOS)` does on an RGB
/// image — the SmolVLM2 preprocess resample (spec §6; every resize site is
/// LANCZOS there, `resample: 1`). Same pass structure and fixed-point math as
/// [`resize_bicubic`], only the `(filter, support)` pair differs.
///
/// # Panics
/// If `out_w` or `out_h` is 0 (Pillow raises `ValueError`).
#[must_use]
pub fn resize_lanczos(src: &RgbImage, out_w: u32, out_h: u32) -> RgbImage {
    resize_with(src, out_w, out_h, PilFilter::Lanczos)
}

fn resize_with(src: &RgbImage, out_w: u32, out_h: u32, filter: PilFilter) -> RgbImage {
    assert!(
        out_w > 0 && out_h > 0,
        "pil_resample::resize ({filter:?}): zero output dimension {out_w}x{out_h}"
    );
    let need_horizontal = out_w != src.width();
    let need_vertical = out_h != src.height();
    match (need_horizontal, need_vertical) {
        (false, false) => src.clone(),
        (true, false) => resample_horizontal(src, out_w, filter),
        (false, true) => resample_vertical(src, out_h, filter),
        (true, true) => resample_vertical(&resample_horizontal(src, out_w, filter), out_h, filter),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One Pillow-generated golden: `src` (row-major RGB bytes) resized to
    /// `dst_size` must reproduce `expected` byte-for-byte.
    struct Golden {
        name: &'static str,
        src_size: (u32, u32),
        dst_size: (u32, u32),
        src: &'static [u8],
        expected: &'static [u8],
    }

    // ── Pillow 12.1.1 goldens (the truth-pack runtime pin) ──────────────────
    // Generated by `scripts/gen_pil_bicubic_goldens.py --goldens` (seed
    // 301466) on 2026-07-01: `expected = np.asarray(Image.fromarray(src,
    // "RGB").resize(dst, Image.Resampling.BICUBIC))`. The generator also
    // differentially validates this module's exact algorithm (in Python)
    // against Pillow on 370 randomized cases before emitting anything.

    /// 4×4 → 2×2: symmetric 2× downscale.
    const SRC_4X4_TO_2X2: [u8; 48] = [
        31, 48, 152, 201, 113, 39, 41, 31, 63, 112, 250, 222, //
        62, 244, 162, 134, 48, 248, 90, 11, 69, 195, 75, 134, //
        192, 146, 79, 19, 253, 111, 48, 156, 234, 82, 21, 18, //
        204, 231, 236, 36, 233, 82, 240, 64, 236, 221, 216, 35,
    ];
    const PIL_4X4_TO_2X2: [u8; 12] = [99, 108, 135, 108, 88, 124, 113, 203, 152, 134, 116, 124];

    /// 5×3 → 7×2: asymmetric, upscale-x + downscale-y.
    const SRC_5X3_TO_7X2: [u8; 45] = [
        127, 17, 136, 243, 179, 60, 214, 68, 55, 219, 34, 195, //
        82, 195, 208, 110, 94, 198, 69, 124, 206, 8, 228, 152, //
        49, 228, 213, 13, 42, 223, 5, 214, 111, 33, 94, 240, //
        214, 255, 215, 121, 62, 187, 218, 47, 99,
    ];
    const PIL_5X3_TO_7X2: [u8; 42] = [
        122, 32, 164, 161, 110, 134, 175, 161, 97, 134, 123, 86, //
        157, 109, 172, 111, 122, 217, 43, 143, 219, 41, 179, 139, //
        34, 124, 197, 63, 137, 232, 134, 252, 197, 99, 169, 199, //
        110, 81, 173, 147, 34, 140,
    ];

    /// 3×5 → 2×8: asymmetric, downscale-x + upscale-y.
    const SRC_3X5_TO_2X8: [u8; 45] = [
        220, 250, 235, 172, 14, 122, 254, 41, 31, 116, 56, 100, //
        234, 174, 252, 239, 239, 202, 0, 77, 179, 56, 191, 60, //
        167, 125, 49, 181, 113, 86, 23, 144, 213, 27, 84, 119, //
        237, 243, 10, 128, 188, 167, 251, 190, 187,
    ];
    const PIL_3X5_TO_2X8: [u8; 48] = [
        203, 171, 202, 222, 8, 46, 190, 137, 182, 236, 104, 133, //
        150, 94, 153, 239, 224, 224, 44, 110, 141, 169, 182, 94, //
        32, 120, 138, 85, 132, 64, 120, 123, 136, 18, 105, 152, //
        173, 186, 91, 125, 153, 180, 199, 231, 59, 217, 193, 188,
    ];

    /// 4×4 → 6×6: upscale — exercises the negative lobes, the clamped +
    /// renormalized edge windows, and both `clip8` saturation branches (the
    /// expected bytes contain exact 0 and 255 overshoot clips).
    const SRC_4X4_TO_6X6: [u8; 48] = [
        165, 238, 191, 208, 37, 197, 187, 220, 18, 177, 201, 214, //
        32, 237, 131, 229, 43, 161, 112, 35, 222, 33, 184, 169, //
        179, 237, 184, 109, 167, 154, 148, 166, 80, 143, 137, 226, //
        79, 55, 209, 238, 137, 73, 42, 52, 68, 8, 250, 246,
    ];
    const PIL_4X4_TO_6X6: [u8; 108] = [
        171, 251, 195, 190, 132, 208, 207, 49, 174, 195, 208, 16, //
        188, 228, 105, 186, 200, 231, 85, 251, 158, 160, 136, 173, //
        224, 31, 174, 163, 108, 126, 120, 168, 155, 100, 201, 194, //
        32, 251, 132, 130, 155, 143, 213, 44, 168, 132, 33, 213, //
        68, 111, 198, 35, 185, 171, 169, 254, 177, 141, 203, 171, //
        118, 149, 150, 145, 155, 100, 144, 145, 159, 135, 135, 226, //
        133, 140, 206, 156, 152, 161, 163, 153, 98, 104, 114, 60, //
        81, 154, 155, 77, 195, 250, 61, 37, 220, 166, 92, 143, //
        231, 127, 57, 64, 44, 55, 3, 152, 163, 0, 255, 255,
    ];

    /// 8×5 → 3×5: horizontal-only (the vertical pass must be skipped, as
    /// `ImagingResampleInner`'s `need_vertical` does).
    const SRC_8X5_TO_3X5: [u8; 120] = [
        146, 142, 132, 113, 159, 47, 3, 202, 9, 245, 255, 139, //
        40, 88, 135, 100, 174, 228, 110, 254, 171, 82, 191, 217, //
        182, 148, 105, 138, 253, 65, 176, 239, 82, 123, 104, 150, //
        14, 195, 134, 179, 207, 178, 23, 231, 145, 57, 171, 149, //
        178, 89, 215, 18, 172, 227, 103, 172, 233, 185, 209, 3, //
        56, 233, 136, 11, 164, 20, 115, 252, 109, 50, 126, 80, //
        163, 0, 52, 164, 114, 228, 35, 250, 6, 55, 177, 15, //
        239, 219, 225, 93, 127, 23, 249, 180, 101, 90, 40, 250, //
        244, 126, 127, 233, 9, 11, 168, 49, 222, 232, 66, 77, //
        174, 216, 86, 11, 131, 41, 206, 83, 4, 82, 167, 113,
    ];
    const PIL_8X5_TO_3X5: [u8; 45] = [
        105, 172, 64, 111, 179, 128, 93, 204, 203, 164, 209, 83, //
        104, 177, 138, 68, 206, 156, 102, 146, 218, 97, 208, 88, //
        62, 190, 78, 123, 114, 104, 121, 203, 83, 166, 122, 141, //
        226, 53, 105, 166, 120, 95, 113, 132, 47,
    ];

    /// 5×8 → 5×3: vertical-only (the horizontal pass must be skipped).
    const SRC_5X8_TO_5X3: [u8; 120] = [
        112, 88, 248, 229, 125, 49, 8, 215, 212, 158, 94, 169, //
        8, 30, 36, 2, 217, 8, 218, 161, 133, 204, 137, 181, //
        33, 212, 155, 150, 15, 224, 223, 182, 180, 80, 182, 32, //
        176, 144, 235, 88, 62, 188, 133, 89, 79, 2, 233, 5, //
        80, 37, 112, 52, 92, 182, 108, 140, 251, 123, 52, 81, //
        125, 43, 152, 58, 234, 252, 129, 36, 10, 32, 179, 39, //
        176, 48, 194, 129, 45, 21, 22, 101, 54, 198, 225, 159, //
        79, 198, 4, 38, 243, 59, 5, 71, 21, 1, 228, 192, //
        167, 119, 19, 221, 143, 145, 101, 203, 84, 179, 116, 204, //
        125, 155, 75, 203, 109, 234, 210, 136, 224, 39, 53, 101,
    ];
    const PIL_5X8_TO_5X3: [u8; 45] = [
        90, 175, 127, 185, 147, 77, 126, 162, 210, 91, 131, 180, //
        100, 36, 121, 97, 132, 76, 58, 139, 139, 126, 103, 125, //
        70, 152, 125, 133, 93, 118, 97, 70, 85, 46, 177, 127, //
        187, 138, 116, 174, 159, 124, 68, 162, 90,
    ];

    fn img(w: u32, h: u32, bytes: &[u8]) -> RgbImage {
        RgbImage::from_raw(w, h, bytes.to_vec()).expect("golden byte count matches dims")
    }

    /// Every golden: `resize_bicubic` must reproduce the pinned Pillow
    /// 12.1.1 output byte-for-byte (bd-30me / DISC-001 reference path).
    #[test]
    fn matches_pillow_12_1_1_goldens() {
        let goldens = [
            Golden {
                name: "4x4 -> 2x2 (symmetric downscale)",
                src_size: (4, 4),
                dst_size: (2, 2),
                src: &SRC_4X4_TO_2X2,
                expected: &PIL_4X4_TO_2X2,
            },
            Golden {
                name: "5x3 -> 7x2 (up-x, down-y)",
                src_size: (5, 3),
                dst_size: (7, 2),
                src: &SRC_5X3_TO_7X2,
                expected: &PIL_5X3_TO_7X2,
            },
            Golden {
                name: "3x5 -> 2x8 (down-x, up-y)",
                src_size: (3, 5),
                dst_size: (2, 8),
                src: &SRC_3X5_TO_2X8,
                expected: &PIL_3X5_TO_2X8,
            },
            Golden {
                name: "4x4 -> 6x6 (upscale, overshoot clips)",
                src_size: (4, 4),
                dst_size: (6, 6),
                src: &SRC_4X4_TO_6X6,
                expected: &PIL_4X4_TO_6X6,
            },
            Golden {
                name: "8x5 -> 3x5 (horizontal-only)",
                src_size: (8, 5),
                dst_size: (3, 5),
                src: &SRC_8X5_TO_3X5,
                expected: &PIL_8X5_TO_3X5,
            },
            Golden {
                name: "5x8 -> 5x3 (vertical-only)",
                src_size: (5, 8),
                dst_size: (5, 3),
                src: &SRC_5X8_TO_5X3,
                expected: &PIL_5X8_TO_5X3,
            },
        ];
        for g in &goldens {
            let out = resize_bicubic(
                &img(g.src_size.0, g.src_size.1, g.src),
                g.dst_size.0,
                g.dst_size.1,
            );
            assert_eq!(
                out.dimensions(),
                g.dst_size,
                "{}: wrong output size",
                g.name
            );
            assert_eq!(
                out.as_raw().as_slice(),
                g.expected,
                "{}: diverged from Pillow 12.1.1",
                g.name
            );
        }
    }

    /// Same-size resize is a plain copy (PIL returns `self.copy()`; both
    /// `need_*` flags are false).
    #[test]
    fn identity_size_is_copy() {
        let src = img(4, 4, &SRC_4X4_TO_2X2);
        let out = resize_bicubic(&src, 4, 4);
        assert_eq!(out.as_raw(), src.as_raw());
    }

    // ── Pillow LANCZOS goldens (SmolVLM2 preprocess, C7) ─────────────────────
    // Generated 2026-07-02 with Pillow 12.3.0 (the smolvlm2 oracle venv),
    // seed 301466: `expected = np.asarray(Image.fromarray(src, "RGB")
    // .resize(dst, Image.Resampling.LANCZOS))`. Resample.c's fixed-point
    // pipeline is unchanged between 12.1.1 and 12.3.0.

    /// 9×7 → 4×3: downscale on both axes (support-6 windows engage).
    const LZ_SRC_9X7_TO_4X3: [u8; 189] = [
        31, 48, 152, 201, 113, 39, 41, 31, 63, 112, 250, 222, 62, 244, 162, 134, 48, 248, 90, 11,
        69, 195, 75, 134, 192, 146, 79, 19, 253, 111, 48, 156, 234, 82, 21, 18, 204, 231, 236, 36,
        233, 82, 240, 64, 236, 221, 216, 35, 127, 17, 136, 243, 179, 60, 214, 68, 55, 219, 34, 195,
        82, 195, 208, 110, 94, 198, 69, 124, 206, 8, 228, 152, 49, 228, 213, 13, 42, 223, 5, 214,
        111, 33, 94, 240, 214, 255, 215, 121, 62, 187, 218, 47, 99, 18, 198, 196, 220, 250, 235,
        172, 14, 122, 254, 41, 31, 116, 56, 100, 234, 174, 252, 239, 239, 202, 0, 77, 179, 56, 191,
        60, 167, 125, 49, 181, 113, 86, 23, 144, 213, 27, 84, 119, 237, 243, 10, 128, 188, 167,
        251, 190, 187, 185, 87, 181, 165, 238, 191, 208, 37, 197, 187, 220, 18, 177, 201, 214, 32,
        237, 131, 229, 43, 161, 112, 35, 222, 33, 184, 169, 179, 237, 184, 109, 167, 154, 148, 166,
        80, 143, 137, 226, 79, 55, 209, 238, 137, 73, 42, 52, 68,
    ];
    const LZ_PIL_9X7_TO_4X3: [u8; 36] = [
        92, 98, 112, 103, 165, 166, 120, 139, 172, 154, 102, 106, 171, 148, 203, 115, 120, 153,
        120, 164, 160, 115, 106, 112, 151, 167, 201, 151, 165, 142, 153, 141, 141, 131, 131, 121,
    ];

    /// 4×3 → 7×9: upscale (unit filterscale, overshoot clipping engages).
    const LZ_SRC_4X3_TO_7X9: [u8; 36] = [
        146, 142, 132, 113, 159, 47, 3, 202, 9, 245, 255, 139, 40, 88, 135, 100, 174, 228, 110,
        254, 171, 82, 191, 217, 182, 148, 105, 138, 253, 65, 176, 239, 82, 123, 104, 150,
    ];
    const LZ_PIL_4X3_TO_7X9: [u8; 189] = [
        172, 154, 142, 168, 153, 100, 130, 159, 21, 4, 166, 0, 3, 199, 0, 181, 244, 80, 255, 255,
        141, 146, 142, 140, 148, 144, 112, 123, 157, 56, 19, 175, 0, 15, 207, 17, 166, 243, 99,
        255, 255, 155, 103, 121, 136, 115, 128, 133, 112, 152, 114, 42, 189, 67, 34, 221, 64, 142,
        242, 131, 205, 243, 179, 53, 96, 130, 75, 110, 157, 99, 151, 185, 76, 210, 152, 65, 239,
        126, 110, 236, 171, 137, 221, 208, 33, 81, 122, 57, 106, 167, 95, 165, 225, 118, 235, 206,
        108, 252, 170, 90, 215, 198, 78, 181, 224, 74, 92, 115, 84, 127, 148, 107, 195, 191, 146,
        251, 181, 145, 250, 161, 98, 184, 188, 67, 137, 209, 138, 116, 112, 129, 156, 117, 125,
        225, 123, 156, 255, 114, 165, 238, 122, 122, 158, 158, 91, 107, 180, 188, 135, 110, 165,
        178, 93, 139, 245, 68, 158, 255, 58, 175, 228, 88, 142, 141, 132, 116, 89, 157, 219, 147,
        109, 187, 192, 78, 148, 255, 35, 160, 255, 24, 182, 222, 68, 154, 130, 117, 130, 77, 143,
    ];

    /// 8×5 → 5×5: the aspect-changing squash (the SmolVLM2 global-frame
    /// resize shape — horizontal down, vertical identity).
    const LZ_SRC_8X5_TO_5X5: [u8; 120] = [
        14, 195, 134, 179, 207, 178, 23, 231, 145, 57, 171, 149, 178, 89, 215, 18, 172, 227, 103,
        172, 233, 185, 209, 3, 56, 233, 136, 11, 164, 20, 115, 252, 109, 50, 126, 80, 163, 0, 52,
        164, 114, 228, 35, 250, 6, 55, 177, 15, 239, 219, 225, 93, 127, 23, 249, 180, 101, 90, 40,
        250, 244, 126, 127, 233, 9, 11, 168, 49, 222, 232, 66, 77, 174, 216, 86, 11, 131, 41, 206,
        83, 4, 82, 167, 113, 112, 88, 248, 229, 125, 49, 8, 215, 212, 158, 94, 169, 8, 30, 36, 2,
        217, 8, 218, 161, 133, 204, 137, 181, 33, 212, 155, 150, 15, 224, 223, 182, 180, 80, 182,
        32,
    ];
    const LZ_PIL_8X5_TO_5X5: [u8; 75] = [
        89, 199, 154, 69, 225, 151, 106, 128, 175, 67, 151, 249, 151, 199, 91, 36, 203, 86, 66,
        221, 75, 120, 54, 82, 138, 127, 144, 36, 221, 4, 181, 187, 134, 164, 134, 100, 180, 85,
        188, 225, 41, 78, 199, 56, 145, 103, 182, 70, 124, 108, 14, 121, 120, 176, 148, 145, 131,
        89, 150, 186, 0, 108, 15, 177, 180, 108, 122, 162, 176, 146, 91, 210, 148, 185, 89,
    ];

    /// LANCZOS matches Pillow byte-for-byte on down/up/squash cases (C7's
    /// preprocess resample; same fixed-point pipeline as bicubic, filter
    /// table entry `{lanczos_filter, 3.0}`).
    #[test]
    fn lanczos_matches_pillow_goldens() {
        let goldens = [
            Golden {
                name: "lanczos 9x7 -> 4x3 (downscale)",
                src_size: (9, 7),
                dst_size: (4, 3),
                src: &LZ_SRC_9X7_TO_4X3,
                expected: &LZ_PIL_9X7_TO_4X3,
            },
            Golden {
                name: "lanczos 4x3 -> 7x9 (upscale)",
                src_size: (4, 3),
                dst_size: (7, 9),
                src: &LZ_SRC_4X3_TO_7X9,
                expected: &LZ_PIL_4X3_TO_7X9,
            },
            Golden {
                name: "lanczos 8x5 -> 5x5 (squash)",
                src_size: (8, 5),
                dst_size: (5, 5),
                src: &LZ_SRC_8X5_TO_5X5,
                expected: &LZ_PIL_8X5_TO_5X5,
            },
        ];
        for g in &goldens {
            let out = resize_lanczos(
                &img(g.src_size.0, g.src_size.1, g.src),
                g.dst_size.0,
                g.dst_size.1,
            );
            assert_eq!(
                out.dimensions(),
                g.dst_size,
                "{}: wrong output size",
                g.name
            );
            assert_eq!(
                out.as_raw().as_slice(),
                g.expected,
                "{}: diverged from Pillow LANCZOS",
                g.name
            );
        }
    }

    /// A solid color survives any resize exactly: every renormalized window
    /// sums to ~1 in fixed point, and the pre-added rounding half absorbs
    /// the ±few-ULP coefficient rounding. Hand-verified against Pillow
    /// 12.1.1 on these exact sizes (2026-07-01).
    #[test]
    fn solid_color_is_preserved_exactly() {
        let src = RgbImage::from_pixel(9, 5, image::Rgb([3, 200, 77]));
        for (w, h) in [(4, 7), (2, 2), (17, 3), (9, 8)] {
            let out = resize_bicubic(&src, w, h);
            assert!(
                out.pixels().all(|p| p.0 == [3, 200, 77]),
                "solid color drifted at {w}x{h}"
            );
        }
    }

    /// 1×1 upscale replicates the single source pixel: the clamped window is
    /// always that one pixel with a renormalized weight of exactly 1.
    /// Hand-verified against Pillow 12.1.1 (2026-07-01).
    #[test]
    fn one_pixel_upscale_replicates() {
        let src = RgbImage::from_pixel(1, 1, image::Rgb([42, 0, 255]));
        let out = resize_bicubic(&src, 3, 3);
        assert_eq!(out.dimensions(), (3, 3));
        assert!(out.pixels().all(|p| p.0 == [42, 0, 255]));
    }

    /// Zero output dims are a caller bug — loudly rejected, matching PIL's
    /// `ValueError` rather than fabricating an empty image.
    #[test]
    fn zero_output_dimension_panics() {
        let panic = std::panic::catch_unwind(|| {
            let src = RgbImage::from_pixel(2, 2, image::Rgb([1, 2, 3]));
            resize_bicubic(&src, 0, 2)
        })
        .expect_err("zero width must panic");
        let msg = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default();
        assert!(
            msg.contains("zero output dimension"),
            "unexpected panic: {msg}"
        );
    }
}
