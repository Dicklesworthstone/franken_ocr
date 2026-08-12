//! wasm32 **simd128** int8/int4 kernel island (bd task K, browser decode).
//!
//! This is the wasm twin of [`arm`](crate::simd::arm) / [`x86`](crate::simd::x86):
//! an audited `unsafe` intrinsic island that is **bit-identical** to the
//! [`scalar`](crate::simd::scalar) oracle, selected by
//! [`dispatch`](crate::simd::dispatch) when the module is compiled with
//! `-C target-feature=+simd128` (which `site/build.sh` always does).
//!
//! ## Why a hand island exists here at all (the INVERTED per-target doctrine)
//!
//! AGENTS.md doctrine #3 — "never hand-roll wide SIMD over scalar inner loops"
//! — is a **native** law: on aarch64/x86 LLVM autovectorizes the scalar dot
//! product into the same shape a hand kernel would emit, and hand loops
//! measured ~5x slower. On `wasm32` the law inverts for exactly one reason:
//! **wasm SIMD128 has no int8 dot-product instruction**, so for the int4 path
//! (nibble extraction inside the inner loop) the autovectorizer has nothing to
//! pattern-match and emits byte-at-a-time scalar code. The island is written
//! only where measurement showed LLVM was blind:
//!
//! | kernel | wasm scalar baseline (Node 25/V8, M4, +simd128) | this island |
//! |---|---|---|
//! | `igemm_s8s8` m=1 k=1280 n=16384 (lm_head class) | 2.07 GMAC/s | see PERF ledger |
//! | `igemm_s4s8_packed` m=1 k=1280 n=896 g16 (expert class) | 1.11 GMAC/s | see PERF ledger |
//!
//! The int4 baseline is ~4x below the int8 one **despite moving half the
//! bytes** — the `kk & 1` nibble branch in the scalar reference is what the
//! autovectorizer cannot lift. That gap, not an instruction-count fantasy, is
//! the lever. Budget for the int8 kernels stays ≤~2x per the sibling-project
//! measurement (frankentts NE-003: 1.47–1.71x, never the predicted 4–8x —
//! 4 widenings + 2 `i32x4.dot_i16x8_s` + 2 adds per 16 bytes ≈ 2.0 MACs/op).
//!
//! ## Bit-identity (the parity contract, doctrine #1 + #8)
//!
//! Every kernel here accumulates in **i32**, and integer addition is
//! associative, so lane order is irrelevant: the result is *equal* to the
//! scalar oracle, not merely close. The int4 path additionally reproduces the
//! scalar f32 epilogue **in increasing group order** (`acc_f += scale[g] *
//! (group_dot as f32)`), so even the non-associative f32 sum matches
//! bit-for-bit. Proven on the BUILT artifact by
//! [`dispatch::selftest`](crate::simd::dispatch::selftest) (int8, 44 cases incl.
//! K=6848) and by the int4 parity battery in the packed entrypoint's tests.
//!
//! ## i32 overflow (doctrine #6)
//!
//! `i32x4.dot_i16x8_s` forms `a·b + c·d` from i16 lanes: with i8 operands each
//! product is ≤ `127·128 = 16256` and the pair ≤ `32512`, so the instruction
//! itself cannot overflow. Each of the four i32 lanes then accumulates at most
//! `K/4` such pairs: at the model worst case `K = 6848` that is
//! `1712 · 32512 ≈ 55.7 M`, ~38x inside `i32::MAX`. U8S8 doubles the per-pair
//! bound (`255·127·2 = 64770`) to ≈ 111 M — still ~19x inside. The int4 path is
//! bounded far tighter (`|w| ≤ 8`, one group ≤ 32 elements).

// SAFETY (module-wide): the only `unsafe` here is `v128_load`/`v128_load64_zero`
// on pointers derived from `&[T]` slices whose length was checked by the caller
// (`dispatch`/`int4` assert the shape contract) and by the `while p + 16 <= k`
// loop bounds immediately above each load. wasm has no alignment requirement for
// `v128.load` (unaligned loads are architecturally defined), so a `*const i8`
// cast to `*const v128` is well-formed for any offset. No aliasing: reads only.
#![allow(unsafe_code, unsafe_op_in_unsafe_fn)]

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::*;

/// True when this module's accelerated kernels are compiled in. simd128 is a
/// **module-level** wasm feature: a module built with it either instantiates on
/// an engine that has it or is refused outright, so there is nothing to detect
/// at runtime the way `FEAT_DotProd` must be — the refusal IS the detection.
#[must_use]
pub const fn simd128_enabled() -> bool {
    cfg!(target_feature = "simd128")
}

// ── horizontal i32x4 reduction ──────────────────────────────────────────────

/// Exact horizontal sum of an `i32x4` accumulator. Integer addition is
/// associative, so any reduction order equals the scalar oracle's.
#[cfg(target_feature = "simd128")]
#[inline(always)]
fn hsum_i32x4(v: v128) -> i32 {
    let s = i32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
    let s = i32x4_add(s, i32x4_shuffle::<1, 0, 3, 2>(s, s));
    i32x4_extract_lane::<0>(s)
}

// ── int8 dot products ───────────────────────────────────────────────────────

/// `Σ a[p]·b[p]` over `k` signed-int8 elements, in exact i32.
///
/// Four independent accumulator streams over 64-byte blocks so one stream's
/// dependent-add latency does not stall the next; the `<64` remainder runs
/// 16 bytes at a time and the `<16` tail is scalar (identical arithmetic).
#[cfg(target_feature = "simd128")]
#[inline]
unsafe fn dot_s8s8(a: &[i8], b: &[i8], k: usize) -> i32 {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = i32x4_splat(0);
    let mut acc1 = i32x4_splat(0);
    let mut acc2 = i32x4_splat(0);
    let mut acc3 = i32x4_splat(0);
    let mut p = 0usize;
    while p + 64 <= k {
        for (i, acc) in [&mut acc0, &mut acc1, &mut acc2, &mut acc3]
            .into_iter()
            .enumerate()
        {
            let off = p + i * 16;
            let va = v128_load(ap.add(off).cast());
            let vb = v128_load(bp.add(off).cast());
            let lo = i32x4_dot_i16x8(i16x8_extend_low_i8x16(va), i16x8_extend_low_i8x16(vb));
            let hi = i32x4_dot_i16x8(i16x8_extend_high_i8x16(va), i16x8_extend_high_i8x16(vb));
            *acc = i32x4_add(*acc, i32x4_add(lo, hi));
        }
        p += 64;
    }
    while p + 16 <= k {
        let va = v128_load(ap.add(p).cast());
        let vb = v128_load(bp.add(p).cast());
        let lo = i32x4_dot_i16x8(i16x8_extend_low_i8x16(va), i16x8_extend_low_i8x16(vb));
        let hi = i32x4_dot_i16x8(i16x8_extend_high_i8x16(va), i16x8_extend_high_i8x16(vb));
        acc0 = i32x4_add(acc0, i32x4_add(lo, hi));
        p += 16;
    }
    let acc = i32x4_add(i32x4_add(acc0, acc1), i32x4_add(acc2, acc3));
    let mut sum = hsum_i32x4(acc);
    while p < k {
        sum += i32::from(a[p]) * i32::from(b[p]);
        p += 1;
    }
    sum
}

/// `Σ a[p]·b[p]` over `k` elements with **unsigned** activations.
///
/// `u16x8.extend_*_u8x16` places each `u8` in `[0, 255]` into an i16 lane, so
/// `i32x4.dot_i16x8_s` computes the signed `u8 · i8` products exactly (both
/// operands are inside the i16 range; see the module overflow note).
#[cfg(target_feature = "simd128")]
#[inline]
unsafe fn dot_u8s8(a: &[u8], b: &[i8], k: usize) -> i32 {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = i32x4_splat(0);
    let mut acc1 = i32x4_splat(0);
    let mut acc2 = i32x4_splat(0);
    let mut acc3 = i32x4_splat(0);
    let mut p = 0usize;
    while p + 64 <= k {
        for (i, acc) in [&mut acc0, &mut acc1, &mut acc2, &mut acc3]
            .into_iter()
            .enumerate()
        {
            let off = p + i * 16;
            let va = v128_load(ap.add(off).cast());
            let vb = v128_load(bp.add(off).cast());
            let lo = i32x4_dot_i16x8(u16x8_extend_low_u8x16(va), i16x8_extend_low_i8x16(vb));
            let hi = i32x4_dot_i16x8(u16x8_extend_high_u8x16(va), i16x8_extend_high_i8x16(vb));
            *acc = i32x4_add(*acc, i32x4_add(lo, hi));
        }
        p += 64;
    }
    while p + 16 <= k {
        let va = v128_load(ap.add(p).cast());
        let vb = v128_load(bp.add(p).cast());
        let lo = i32x4_dot_i16x8(u16x8_extend_low_u8x16(va), i16x8_extend_low_i8x16(vb));
        let hi = i32x4_dot_i16x8(u16x8_extend_high_u8x16(va), i16x8_extend_high_i8x16(vb));
        acc0 = i32x4_add(acc0, i32x4_add(lo, hi));
        p += 16;
    }
    let acc = i32x4_add(i32x4_add(acc0, acc1), i32x4_add(acc2, acc3));
    let mut sum = hsum_i32x4(acc);
    while p < k {
        sum += i32::from(a[p]) * i32::from(b[p]);
        p += 1;
    }
    sum
}

// ── public int8 GEMM entrypoints (the dispatch targets) ─────────────────────

/// `C[M,N] += A[M,K]·B[N,K]` (S8S8), simd128 where available, else the scalar
/// oracle. Bit-identical to [`scalar::igemm_s8s8`](crate::simd::scalar::igemm_s8s8).
///
/// # Panics
/// As the scalar oracle (length-contract violations).
pub fn igemm_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    #[cfg(not(target_feature = "simd128"))]
    {
        super::scalar::igemm_s8s8(a, b, m, k, n, out);
    }
    #[cfg(target_feature = "simd128")]
    {
        super::scalar::assert_gemm_shapes("igemm_s8s8", a.len(), b.len(), out.len(), m, k, n);
        for i in 0..m {
            let a_row = &a[i * k..i * k + k];
            let out_row = &mut out[i * n..i * n + n];
            for o in 0..n {
                let b_row = &b[o * k..o * k + k];
                // SAFETY: `a_row`/`b_row` are exactly `k` elements (asserted
                // above); every vector load below is bounded by `p + 16 <= k`.
                out_row[o] += unsafe { dot_s8s8(a_row, b_row, k) };
            }
        }
    }
}

/// `C[M,N] += A[M,K]·B[N,K]` (U8S8, unsigned activations), simd128 where
/// available. Bit-identical to
/// [`scalar::igemm_u8s8`](crate::simd::scalar::igemm_u8s8).
///
/// # Panics
/// As the scalar oracle.
pub fn igemm_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    #[cfg(not(target_feature = "simd128"))]
    {
        super::scalar::igemm_u8s8(a, b, m, k, n, out);
    }
    #[cfg(target_feature = "simd128")]
    {
        super::scalar::assert_gemm_shapes("igemm_u8s8", a.len(), b.len(), out.len(), m, k, n);
        for i in 0..m {
            let a_row = &a[i * k..i * k + k];
            let out_row = &mut out[i * n..i * n + n];
            for o in 0..n {
                let b_row = &b[o * k..o * k + k];
                // SAFETY: as `igemm_s8s8` above.
                out_row[o] += unsafe { dot_u8s8(a_row, b_row, k) };
            }
        }
    }
}

// ── packed int4 (the expert-FFN bandwidth path) ─────────────────────────────

/// One 32-k-element packed-int4 block: returns the exact i32 dot of k-elements
/// `[0,16)` and of `[16,32)` **separately**, so a `group_size` of 16 gets its
/// two per-group sums from one pass and a `group_size` of 32 just adds them.
///
/// Layout (pinned by `quant::int4`): weight `kk` lives in packed byte `kk/2`,
/// the LOW nibble for even `kk` and the HIGH nibble for odd `kk`. So a single
/// 16-byte load carries 32 consecutive weights: the sign-extended low nibbles
/// are the even-`kk` weights and the high nibbles the odd-`kk` ones. Rather
/// than re-interleave the weights, we **de-interleave the activations** with two
/// `i8x16.shuffle`s — one instruction each, no memory traffic.
#[cfg(target_feature = "simd128")]
#[inline]
unsafe fn dot_s4s8_block32(a: *const i8, w: *const u8) -> (i32, i32) {
    let wv = v128_load(w.cast());
    // Sign-extend each nibble: `(x << 4) >> 4` arithmetic, per byte lane.
    let lo_nib = i8x16_shr(i8x16_shl(wv, 4), 4); // even-kk weights
    let hi_nib = i8x16_shr(wv, 4); // odd-kk weights
    let a0 = v128_load(a.cast());
    let a1 = v128_load(a.add(16).cast());
    let a_even = i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(a0, a1);
    let a_odd = i8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31>(a0, a1);
    // Lane j of the low halves covers kk = 2j and 2j+1 (kk < 16 → first group);
    // lane j of the high halves covers kk = 16 + 2j, 17 + 2j (second group).
    let first = i32x4_add(
        i32x4_dot_i16x8(
            i16x8_extend_low_i8x16(a_even),
            i16x8_extend_low_i8x16(lo_nib),
        ),
        i32x4_dot_i16x8(
            i16x8_extend_low_i8x16(a_odd),
            i16x8_extend_low_i8x16(hi_nib),
        ),
    );
    let second = i32x4_add(
        i32x4_dot_i16x8(
            i16x8_extend_high_i8x16(a_even),
            i16x8_extend_high_i8x16(lo_nib),
        ),
        i32x4_dot_i16x8(
            i16x8_extend_high_i8x16(a_odd),
            i16x8_extend_high_i8x16(hi_nib),
        ),
    );
    (hsum_i32x4(first), hsum_i32x4(second))
}

/// Native packed-int4 GEMM on simd128 — B is never materialized as dense int8.
///
/// Bit-identical to
/// [`int4::igemm_s4s8_packed_scalar`](crate::simd::int4::igemm_s4s8_packed_scalar):
/// the per-group contraction is the same exact i32 dot (lane order irrelevant),
/// and the f32 dequant-and-sum runs in the same increasing group order.
///
/// Preconditions are the packed entrypoint's (already asserted by the caller):
/// `k` even, `group ∈ {16, 32}` divides `k`, buffers sized to `m/k/n/group`.
/// `out` is **overwritten**, as in the scalar reference.
#[allow(clippy::too_many_arguments)]
pub fn igemm_s4s8_packed(
    a: &[i8],
    b_packed: &[u8],
    scales: &[f32],
    group: usize,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    #[cfg(not(target_feature = "simd128"))]
    {
        super::int4::s4s8_packed_kernel_scalar(a, b_packed, scales, group, m, k, n, out);
    }
    #[cfg(target_feature = "simd128")]
    {
        let groups = k / group;
        let kbytes = k / 2;
        // Whole 32-element blocks only; a ragged remainder (k not a multiple of
        // 32, or an odd group count at group=16) falls to the scalar reference
        // for those groups — same arithmetic, so still bit-identical.
        let blocks = k / 32;
        let vec_groups = blocks * (32 / group);
        for mi in 0..m {
            let a_row = &a[mi * k..mi * k + k];
            for ni in 0..n {
                let wbase = ni * kbytes;
                let sbase = ni * groups;
                let mut acc_f = 0.0f32;
                for blk in 0..blocks {
                    let kk = blk * 32;
                    // SAFETY: `a_row` holds `k` elements and `kk + 32 <= k`;
                    // the packed row holds `kbytes` bytes and
                    // `wbase + kk/2 + 16 <= wbase + kbytes`.
                    let (d0, d1) = unsafe {
                        dot_s4s8_block32(
                            a_row.as_ptr().add(kk),
                            b_packed.as_ptr().add(wbase + kk / 2),
                        )
                    };
                    if group == 32 {
                        acc_f += scales[sbase + blk] * (d0 + d1) as f32;
                    } else {
                        let g = blk * 2;
                        acc_f += scales[sbase + g] * d0 as f32;
                        acc_f += scales[sbase + g + 1] * d1 as f32;
                    }
                }
                // Ragged tail groups (never taken at the model's shapes, which
                // are all multiples of 32): the scalar reference, verbatim.
                for g in vec_groups..groups {
                    let lo = g * group;
                    let mut acc_i: i32 = 0;
                    for kk in lo..lo + group {
                        let byte = b_packed[wbase + kk / 2];
                        let w = if kk & 1 == 0 {
                            super::int4::sign_extend_nibble(byte & 0x0F)
                        } else {
                            super::int4::sign_extend_nibble(byte >> 4)
                        };
                        acc_i += i32::from(a_row[kk]) * i32::from(w);
                    }
                    acc_f += scales[sbase + g] * acc_i as f32;
                }
                out[mi * n + ni] = acc_f;
            }
        }
    }
}
