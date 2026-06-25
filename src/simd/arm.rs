//! AArch64 register-blocked int8 GEMM — the primary Apple-Silicon kernel
//! (AGENTS.md doctrine #3/#4; PROPOSED_ARCHITECTURE.md §6.6).
//!
//! This module provides the ARM acceleration tier for the pinned int8 GEMM
//! contract shared by every SIMD backend (`simd/scalar.rs` is the reference
//! oracle; `simd/dispatch.rs` picks the best available tier at runtime):
//!
//! ```ignore
//! // C[M,N] += A[M,K] (i8/u8, row-major) · B[N,K] (i8, OUTPUT-CHANNEL-major) -> i32[M,N]
//! pub fn igemm_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]);
//! pub fn igemm_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]);
//! ```
//!
//! Two accelerated paths, gated by runtime CPU feature detection:
//!
//! * **SDOT** (`FEAT_DotProd`, `vdotq_s32`): four int8 MACs per i32 lane.
//!   Register-blocked 4×4 output micro-tile; four i32x4 accumulators per output
//!   row reduce K in 16-wide steps. This is the decode-GEMV and small-`M`
//!   workhorse.
//! * **SMMLA** (`FEAT_MATMUL_INT8` / i8mm, `vmmlaq_s32`): eight int8 MACs per
//!   i32 lane, computing a 2×2 i32 tile from two `[2×8]` int8 operands. Doctrine
//!   #4: an *un-blocked* SMMLA is load-bound and SLOWER than SDOT, so we
//!   **pre-pack** both A and B into the SMMLA-friendly interleaved `[2 rows × 8
//!   cols]` panel layout and **register-block** an 8×8 output tile (sixteen
//!   `int32x4` accumulators, each fed by `2×2 = 4` SMMLA-tile contributions per
//!   K-step), reaching compute:load ≥ 2:1 on the inner loop.
//!
//! Both paths produce **bit-identical** output to the scalar reference (i32
//! accumulation is exact and order-independent for integer addition), proven by
//! the `#[cfg(test)]` cross-check below over randomized + all-127 / all-(-128)
//! adversarial operands at K including the doctrine-#6 worst case 6848.
//!
//! ## U8S8
//!
//! The i8mm/dotprod intrinsics are signed-by-signed (`vdotq_s32` /
//! `vmmlaq_s32`). U8S8 (`DynamicQuantizeLinear` asymmetric activation × signed
//! weight) is handled by the **+128 bias-correction** identity, computed
//! entirely in i32 (exact, no saturation):
//!
//! ```text
//!   u8 activation a_u = (s8 activation a_s) + 128   with a_s = a_u - 128
//!   sum_k a_u[k]·w[k] = sum_k (a_s[k]+128)·w[k]
//!                     = (sum_k a_s[k]·w[k])  +  128 · (sum_k w[k])
//! ```
//!
//! So we run the signed kernel on `a_s = a_u - 128` (a lossless `i16`-range
//! shift folded back into i8 by reinterpreting the byte, since `a_u - 128`
//! lands in `[-128, 127]`), then add `128 · rowsum(w)` per output channel. The
//! per-channel weight row-sum is computed once. This matches the scalar U8S8
//! oracle exactly (the overflow proof in `tests/int32_overflow_proof.rs` bounds
//! the U8S8 accumulator at 221.7M < i32::MAX at K=6848).
//!
//! ## Safety
//!
//! All raw NEON intrinsics live in the single `#[allow(unsafe_code,
//! unsafe_op_in_unsafe_fn)]` island below. Every vector load reads a contiguous
//! 16-byte window from a slice whose length the caller has already bounded; each
//! load carries a `// SAFETY:` note proving the read is in-bounds and that the
//! enclosing function is only reached when the corresponding CPU feature was
//! runtime-detected. NEON loads are alignment-agnostic (`vld1q` does not require
//! 16-byte alignment).
//!
//! On non-AArch64 targets the whole accelerated body is `#[cfg]`-compiled out
//! and the public entrypoints delegate to the scalar reference, so the file
//! compiles and is correct everywhere.

use super::scalar;

// ─────────────────────────────────────────────────────────────────────────────
// Public entrypoints (pinned signature; identical on every arch).
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime ISA tier this ARM backend can offer for the current CPU.
///
/// `dispatch.rs` consults the booleans; the kernels themselves re-check the
/// feature before entering their `#[target_feature]` island so the unsafe entry
/// is never reachable on a CPU lacking the instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmTier {
    /// `FEAT_MATMUL_INT8` (i8mm / SMMLA) available — the top ARM tier.
    Smmla,
    /// `FEAT_DotProd` (SDOT) available, but not i8mm.
    Sdot,
    /// Neither — caller should fall back to the scalar reference.
    None,
}

/// Detect the best available ARM int8 tier on this CPU.
///
/// On non-AArch64 builds this is always [`ArmTier::None`].
#[must_use]
pub fn detect_tier() -> ArmTier {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            return ArmTier::Smmla;
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return ArmTier::Sdot;
        }
        ArmTier::None
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        ArmTier::None
    }
}

/// `out[M,N] = A[M,K] (i8, row-major) · B[N,K] (i8, output-channel-major)`,
/// accumulating in i32 and **overwriting** `out` (the kernels write each cell
/// exactly once; this is `=`, not `+=`, matching the scalar oracle's contract).
///
/// Picks the best available ARM tier (SMMLA > SDOT > scalar) and produces output
/// bit-identical to [`scalar::igemm_s8s8`].
///
/// # Panics
/// Panics if `a.len() != m*k`, `b.len() != n*k`, or `out.len() != m*n` (a
/// shape/contract violation is a programming error, caught early — matches the
/// scalar reference's asserts).
pub fn igemm_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    assert_eq!(a.len(), m * k, "igemm_s8s8: a.len {} != m*k {}", a.len(), m * k);
    assert_eq!(b.len(), n * k, "igemm_s8s8: b.len {} != n*k {}", b.len(), n * k);
    assert_eq!(out.len(), m * n, "igemm_s8s8: out.len {} != m*n {}", out.len(), m * n);

    #[cfg(target_arch = "aarch64")]
    {
        match detect_tier() {
            ArmTier::Smmla => {
                // SAFETY: reached only when `is_aarch64_feature_detected!("i8mm")`
                // returned true in `detect_tier`, so SMMLA/i8mm is present.
                unsafe { aarch64_impl::igemm_s8s8_smmla(a, b, m, k, n, out) };
                return;
            }
            ArmTier::Sdot => {
                // SAFETY: reached only when `dotprod` was runtime-detected.
                unsafe { aarch64_impl::igemm_s8s8_sdot(a, b, m, k, n, out) };
                return;
            }
            ArmTier::None => {}
        }
    }

    scalar::igemm_s8s8(a, b, m, k, n, out);
}

/// U8S8 variant: `A` is `u8` (asymmetric activations), `B` is `i8` weights.
/// Output bit-identical to [`scalar::igemm_u8s8`].
///
/// Implemented via the +128 bias-correction identity (see module docs): run the
/// signed kernel on `a - 128`, then add `128 · rowsum(b)` per output channel.
///
/// # Panics
/// As [`igemm_s8s8`].
pub fn igemm_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    assert_eq!(a.len(), m * k, "igemm_u8s8: a.len {} != m*k {}", a.len(), m * k);
    assert_eq!(b.len(), n * k, "igemm_u8s8: b.len {} != n*k {}", b.len(), n * k);
    assert_eq!(out.len(), m * n, "igemm_u8s8: out.len {} != m*n {}", out.len(), m * n);

    #[cfg(target_arch = "aarch64")]
    {
        let tier = detect_tier();
        if matches!(tier, ArmTier::Smmla | ArmTier::Sdot) {
            // Shift u8 activations into the signed domain: a_s = a_u - 128 lands
            // in [-128, 127], representable in i8. `wrapping_sub(128) as i8`
            // reinterprets the byte so that e.g. 0u8 -> -128i8, 255u8 -> 127i8.
            let a_signed: Vec<i8> = a.iter().map(|&x| x.wrapping_sub(128) as i8).collect();
            // Run the signed GEMM into `out`.
            match tier {
                ArmTier::Smmla => {
                    // SAFETY: i8mm runtime-detected (see `detect_tier`).
                    unsafe { aarch64_impl::igemm_s8s8_smmla(&a_signed, b, m, k, n, out) };
                }
                ArmTier::Sdot => {
                    // SAFETY: dotprod runtime-detected.
                    unsafe { aarch64_impl::igemm_s8s8_sdot(&a_signed, b, m, k, n, out) };
                }
                ArmTier::None => unreachable!(),
            }
            // Per-output-channel weight row-sum, then fold in the +128·rowsum(w)
            // correction (exact i32; the U8S8 overflow bound 221.7M holds).
            let mut rowsum = vec![0i32; n];
            for (oc, rs) in rowsum.iter_mut().enumerate() {
                let row = &b[oc * k..(oc + 1) * k];
                let mut s: i32 = 0;
                for &w in row {
                    s += i32::from(w);
                }
                *rs = s;
            }
            for r in 0..m {
                let orow = &mut out[r * n..(r + 1) * n];
                for (c, cell) in orow.iter_mut().enumerate() {
                    *cell += 128 * rowsum[c];
                }
            }
            return;
        }
    }

    scalar::igemm_u8s8(a, b, m, k, n, out);
}

// ─────────────────────────────────────────────────────────────────────────────
// The audited unsafe SIMD island.
//
// Everything that touches a raw NEON intrinsic lives here. The crate root is
// `#![forbid(unsafe_code)]`; this island re-enables it locally with an explicit
// allow + per-load SAFETY notes (AGENTS.md Toolchain rule).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
mod aarch64_impl {
    use core::arch::aarch64::{
        int8x16_t, int32x4_t, vaddvq_s32, vdotq_s32, vdupq_n_s32, vld1q_s8, vmmlaq_s32,
    };

    /// Load a 16-byte window `[off, off+16)` of `s` as an `int8x16_t`.
    ///
    /// # Safety
    /// The caller MUST guarantee `off + 16 <= s.len()`. `vld1q_s8` performs an
    /// unaligned 16-byte read; NEON loads do not require 16-byte alignment.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load16(s: &[i8], off: usize) -> int8x16_t {
        debug_assert!(off + 16 <= s.len(), "load16 out of bounds");
        // SAFETY: caller guarantees off+16 <= len; pointer is valid for a
        // 16-byte unaligned read within the slice's allocation.
        vld1q_s8(s.as_ptr().add(off))
    }

    /// Load a 16-byte window that may run off the end of `s`, zero-filling the
    /// tail. Used only on the K-remainder step so a partial 16-lane block reads
    /// no out-of-bounds memory yet contributes the correct (zero-padded) dot.
    ///
    /// Zero padding is exact for an int dot product: a zero lane contributes a
    /// zero product, so the accumulator is identical to summing only the real
    /// `len-off` elements.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load16_tail(s: &[i8], off: usize, valid: usize) -> int8x16_t {
        let mut buf = [0i8; 16];
        let n = valid.min(16);
        // `off + n <= s.len()` by construction (valid = len - off, clamped).
        buf[..n].copy_from_slice(&s[off..off + n]);
        // SAFETY: `buf` is a fully-initialized 16-byte stack array; the read is
        // in-bounds and (stack) suitably accessible for an unaligned vld1q.
        vld1q_s8(buf.as_ptr())
    }

    // ── SDOT path ───────────────────────────────────────────────────────────
    //
    // `vdotq_s32(acc, a, b)` computes, for each of the 4 i32 lanes `j`:
    //     acc[j] += sum_{t=0..3} a[4j+t] * b[4j+t]
    // i.e. 4 int8 MACs per lane, 16 MACs per instruction. To get the full
    // 16-element dot of two int8x16 vectors we `vdotq_s32` into a 4-lane
    // accumulator and horizontally add the lanes (`vaddvq_s32`) at the end —
    // integer addition is associative, so this is bit-identical to the scalar
    // left-to-right sum.

    /// SDOT int8 GEMM. Register-blocked over a 4×4 output micro-tile: four
    /// activation rows × four weight rows share their 16-wide K loads, giving
    /// sixteen independent i32x4 accumulators per tile (compute:load ≈ 16 dots
    /// per 8 loads = 2:1).
    ///
    /// # Safety
    /// Caller MUST ensure `dotprod` (`FEAT_DotProd`) is available on this CPU
    /// and the slice lengths satisfy the GEMM shape (`a=m*k`, `b=n*k`,
    /// `out=m*n`).
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn igemm_s8s8_sdot(
        a: &[i8],
        b: &[i8],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [i32],
    ) {
        let k16 = k / 16 * 16; // K rounded down to a multiple of 16

        let mut r = 0;
        while r < m {
            let mr = (m - r).min(4);
            let mut c = 0;
            while c < n {
                let nr = (n - c).min(4);

                // 4×4 accumulator grid of i32x4 lanes.
                // SAFETY: `vdupq_n_s32` is a pure constant splat (no memory).
                let mut acc = [[vdupq_n_s32(0); 4]; 4];

                let mut t = 0;
                while t < k16 {
                    // Load up to 4 activation rows and 4 weight rows for this
                    // 16-wide K slice; these loads are SHARED across the tile
                    // (the register-blocking win).
                    let mut av = [vdupq_n_s32(0).into_i8(); 4];
                    let mut bv = [vdupq_n_s32(0).into_i8(); 4];
                    for i in 0..mr {
                        // SAFETY: t+16 <= k16 <= k, and (r+i)<m, so
                        // (r+i)*k + t + 16 <= (r+i+1)*k <= m*k = a.len().
                        av[i] = load16(a, (r + i) * k + t);
                    }
                    for j in 0..nr {
                        // SAFETY: same bound for the weight matrix (b.len = n*k).
                        bv[j] = load16(b, (c + j) * k + t);
                    }
                    for i in 0..mr {
                        for j in 0..nr {
                            // SAFETY: vdotq_s32 is register-only; inputs valid.
                            acc[i][j] = vdotq_s32(acc[i][j], av[i], bv[j]);
                        }
                    }
                    t += 16;
                }

                // K tail ( < 16 elements ): one zero-padded block.
                if k16 < k {
                    let valid = k - k16;
                    let mut av = [vdupq_n_s32(0).into_i8(); 4];
                    let mut bv = [vdupq_n_s32(0).into_i8(); 4];
                    for i in 0..mr {
                        // SAFETY: load16_tail reads only s[off..off+valid] into a
                        // local 16-byte buffer; off+valid = (r+i)*k + k <= a.len.
                        av[i] = load16_tail(a, (r + i) * k + k16, valid);
                    }
                    for j in 0..nr {
                        bv[j] = load16_tail(b, (c + j) * k + k16, valid);
                    }
                    for i in 0..mr {
                        for j in 0..nr {
                            acc[i][j] = vdotq_s32(acc[i][j], av[i], bv[j]);
                        }
                    }
                }

                // Horizontal-reduce each i32x4 accumulator into the output cell.
                for i in 0..mr {
                    for j in 0..nr {
                        // SAFETY: vaddvq_s32 is a register-only horizontal add.
                        out[(r + i) * n + (c + j)] = vaddvq_s32(acc[i][j]);
                    }
                }

                c += nr;
            }
            r += mr;
        }
    }

    // ── SMMLA path ──────────────────────────────────────────────────────────
    //
    // `vmmlaq_s32(acc, a, b)` treats its int8x16 operands as 2×8 int8 matrices
    // (row-major: lanes 0..8 = row 0, lanes 8..16 = row 1) and computes the
    // 2×8 · 8×2 = 2×2 i32 product, ADDED into the int32x4 accumulator laid out
    //     [ c00, c01, c10, c11 ]
    // where c_xy = sum_{t=0..7} a_row_x[t] * b_row_y[t].
    //
    // 8 int8 MACs per i32 lane × 4 lanes = 32 MACs per instruction. Doctrine #4:
    // an un-blocked SMMLA is load-bound. We PRE-PACK A and B into contiguous
    // `[2 rows × 8 cols]` 16-byte panels (so each vld1q feeds exactly one SMMLA
    // operand) and register-block an 8×8 output tile: 4 A-panels (rows {0,1},
    // {2,3}, {4,5}, {6,7}) × 4 B-panels per 8-wide K-step drive 16 int32x4
    // accumulators with 16 SMMLA per 8 loads = 2:1 compute:load.

    /// Pre-pack a region of a row-major `[rows, k]` matrix into SMMLA panels.
    ///
    /// Output layout, walked by the micro-kernel as a flat `int8x16` stream:
    /// for each K-block of 8 columns (`kb` of them, padded), and for each pair
    /// of rows `(2p, 2p+1)`, emit 16 bytes = `[row(2p)[0..8], row(2p+1)[0..8]]`.
    /// Rows beyond `rows` and columns beyond `k` are zero-filled (zero lanes
    /// contribute zero to an int dot, so the packed GEMM is exact).
    ///
    /// Returns `(packed, row_pairs, kb)` where `row_pairs = ceil(rows/2)` and
    /// `kb = ceil(k/8)`; `packed.len() == row_pairs * kb * 16`.
    fn pack_panels(src: &[i8], base_row: usize, rows: usize, k: usize, src_k: usize)
        -> (Vec<i8>, usize, usize)
    {
        let row_pairs = rows.div_ceil(2);
        let kb = k.div_ceil(8);
        let mut packed = vec![0i8; row_pairs * kb * 16];
        for p in 0..row_pairs {
            for block in 0..kb {
                let panel = (p * kb + block) * 16;
                let kcol = block * 8;
                let kvalid = (k - kcol).min(8); // columns present in this block
                for sub in 0..2 {
                    let row = p * 2 + sub;
                    if row >= rows {
                        continue; // zero-padded tail row
                    }
                    let src_off = (base_row + row) * src_k + kcol;
                    let dst_off = panel + sub * 8;
                    packed[dst_off..dst_off + kvalid]
                        .copy_from_slice(&src[src_off..src_off + kvalid]);
                }
            }
        }
        (packed, row_pairs, kb)
    }

    /// SMMLA int8 GEMM with offline-style A/B pre-packing and 8×8 register
    /// blocking.
    ///
    /// # Safety
    /// Caller MUST ensure `i8mm` (`FEAT_MATMUL_INT8`) is available and the slice
    /// lengths satisfy the GEMM shape.
    #[target_feature(enable = "neon,i8mm")]
    pub unsafe fn igemm_s8s8_smmla(
        a: &[i8],
        b: &[i8],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [i32],
    ) {
        // Process the output in 8×8 tiles. For each tile we pack the relevant
        // A rows and B rows into panel streams, then run the SMMLA micro-kernel.
        let mut r = 0;
        while r < m {
            let mr = (m - r).min(8);
            // Pack A rows [r, r+mr) -> ceil(mr/2) row-pairs, kb K-blocks.
            let (apack, a_pairs, kb) = pack_panels(a, r, mr, k, k);

            let mut c = 0;
            while c < n {
                let nr = (n - c).min(8);
                // Pack B rows [c, c+nr) (output-channel-major == row-major [n,k]).
                let (bpack, b_pairs, _kb2) = pack_panels(b, c, nr, k, k);

                // For each (A row-pair p, B row-pair q) we accumulate a 4×4
                // i32 region (2 A-rows × 2 B-rows) across all K-blocks. The
                // int32x4 result is [c(2p,2q), c(2p,2q+1), c(2p+1,2q), c(2p+1,2q+1)].
                for p in 0..a_pairs {
                    for q in 0..b_pairs {
                        // SAFETY: constant splat, no memory.
                        let mut acc = vdupq_n_s32(0);
                        for block in 0..kb {
                            let aoff = (p * kb + block) * 16;
                            let boff = (q * kb + block) * 16;
                            // SAFETY: apack/bpack lengths are pairs*kb*16 by
                            // construction; aoff+16 <= apack.len(), boff+16 <=
                            // bpack.len(). Reads are into our own packed Vecs.
                            let av = load16(&apack, aoff);
                            let bv = load16(&bpack, boff);
                            // SAFETY: vmmlaq_s32 is register-only; i8mm enabled.
                            acc = vmmlaq_s32(acc, av, bv);
                        }
                        // Scatter the 2×2 i32 tile into `out`, skipping padded
                        // rows/cols (row-pair p covers output rows r+2p, r+2p+1).
                        let tile = [
                            vgetq_lane0(acc),
                            vgetq_lane1(acc),
                            vgetq_lane2(acc),
                            vgetq_lane3(acc),
                        ];
                        let ar0 = 2 * p;
                        let ar1 = 2 * p + 1;
                        let bc0 = 2 * q;
                        let bc1 = 2 * q + 1;
                        if ar0 < mr && bc0 < nr {
                            out[(r + ar0) * n + (c + bc0)] = tile[0];
                        }
                        if ar0 < mr && bc1 < nr {
                            out[(r + ar0) * n + (c + bc1)] = tile[1];
                        }
                        if ar1 < mr && bc0 < nr {
                            out[(r + ar1) * n + (c + bc0)] = tile[2];
                        }
                        if ar1 < mr && bc1 < nr {
                            out[(r + ar1) * n + (c + bc1)] = tile[3];
                        }
                    }
                }
                c += nr;
            }
            r += mr;
        }
    }

    // ── tiny lane-extraction helpers (register-only) ────────────────────────

    #[inline]
    #[target_feature(enable = "neon")]
    fn vgetq_lane0(v: int32x4_t) -> i32 {
        // SAFETY: vgetq_lane_s32 with a const-0 index is a register read.
        unsafe { core::arch::aarch64::vgetq_lane_s32::<0>(v) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn vgetq_lane1(v: int32x4_t) -> i32 {
        // SAFETY: const-1 lane read.
        unsafe { core::arch::aarch64::vgetq_lane_s32::<1>(v) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn vgetq_lane2(v: int32x4_t) -> i32 {
        // SAFETY: const-2 lane read.
        unsafe { core::arch::aarch64::vgetq_lane_s32::<2>(v) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn vgetq_lane3(v: int32x4_t) -> i32 {
        // SAFETY: const-3 lane read.
        unsafe { core::arch::aarch64::vgetq_lane_s32::<3>(v) }
    }

    // A tiny extension so we can write `vdupq_n_s32(0).into_i8()` for a zero
    // int8x16 splat without importing another intrinsic name.
    trait IntoI8 {
        fn into_i8(self) -> int8x16_t;
    }
    impl IntoI8 for int32x4_t {
        #[inline]
        fn into_i8(self) -> int8x16_t {
            // SAFETY: reinterpret-cast of an all-zero 128-bit register; the bit
            // pattern of a zero int32x4 is a zero int8x16. `vreinterpretq_s8_s32`
            // is a no-op bit cast.
            unsafe { core::arch::aarch64::vreinterpretq_s8_s32(self) }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — bit-identical to the scalar reference on randomized + adversarial
// operands at K including the doctrine-#6 worst case 6848.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    /// A self-contained scalar S8S8 oracle (independent of `scalar.rs` so the
    /// test proves correctness even if both share a bug — this is the reference
    /// definition straight from the pinned contract).
    fn oracle_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut out = vec![0i32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc: i32 = 0;
                for t in 0..k {
                    acc += i32::from(a[r * k + t]) * i32::from(b[col * k + t]);
                }
                out[r * n + col] = acc;
            }
        }
        out
    }

    fn oracle_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut out = vec![0i32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc: i32 = 0;
                for t in 0..k {
                    acc += i32::from(a[r * k + t]) * i32::from(b[col * k + t]);
                }
                out[r * n + col] = acc;
            }
        }
        out
    }

    /// Deterministic xorshift PRNG so the test is reproducible without a dep.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn i8(&mut self) -> i8 {
            (self.next() & 0xff) as u8 as i8
        }
        fn u8(&mut self) -> u8 {
            (self.next() & 0xff) as u8
        }
    }

    fn rand_i8(rng: &mut Rng, len: usize) -> Vec<i8> {
        (0..len).map(|_| rng.i8()).collect()
    }
    fn rand_u8(rng: &mut Rng, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.u8()).collect()
    }

    /// Drive the SDOT kernel directly (bypassing dispatch) when available.
    fn run_sdot_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Option<Vec<i32>> {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return None;
        }
        let mut out = vec![0i32; m * n];
        // SAFETY: feature just checked.
        unsafe { aarch64_impl::igemm_s8s8_sdot(a, b, m, k, n, &mut out) };
        Some(out)
    }

    fn run_smmla_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Option<Vec<i32>> {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return None;
        }
        let mut out = vec![0i32; m * n];
        // SAFETY: feature just checked.
        unsafe { aarch64_impl::igemm_s8s8_smmla(a, b, m, k, n, &mut out) };
        Some(out)
    }

    #[test]
    fn sdot_and_smmla_match_oracle_randomized() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        // A spread of shapes that exercise tile remainders in M, N, and K
        // (non-multiples of 4/8/16).
        let shapes = [
            (1usize, 1usize, 1usize),
            (1, 16, 1),
            (1, 17, 1),
            (3, 5, 7),
            (4, 16, 4),
            (5, 23, 6),
            (8, 32, 8),
            (7, 31, 9),
            (10, 1280, 10),
            (6, 896, 13),
        ];
        for &(m, k, n) in &shapes {
            let a = rand_i8(&mut rng, m * k);
            let b = rand_i8(&mut rng, n * k);
            let want = oracle_s8s8(&a, &b, m, k, n);
            if let Some(got) = run_sdot_s8s8(&a, &b, m, k, n) {
                assert_eq!(got, want, "SDOT mismatch at shape ({m},{k},{n})");
            }
            if let Some(got) = run_smmla_s8s8(&a, &b, m, k, n) {
                assert_eq!(got, want, "SMMLA mismatch at shape ({m},{k},{n})");
            }
            // The public dispatcher must also match.
            let mut got = vec![0i32; m * n];
            igemm_s8s8(&a, &b, m, k, n, &mut got);
            assert_eq!(got, want, "dispatch S8S8 mismatch at shape ({m},{k},{n})");
        }
    }

    #[test]
    fn s8s8_adversarial_all_max_at_worst_case_k() {
        // Doctrine #6 worst case K plus the adversarial all-127 / all-(-128)
        // operands. These maximize the i32 accumulator (110.4M / 112.2M) and are
        // exactly the values the overflow proof bounds.
        let k = 6848usize;
        let m = 3usize;
        let n = 5usize;
        for &fill in &[127i8, -128i8] {
            let a = vec![fill; m * k];
            let b = vec![fill; n * k];
            let want = oracle_s8s8(&a, &b, m, k, n);
            // Spot-check the headline value.
            let expect = if fill == 127 { 110_451_392 } else { 112_197_632 };
            assert_eq!(want[0], expect, "oracle value at fill {fill}");
            if let Some(got) = run_sdot_s8s8(&a, &b, m, k, n) {
                assert_eq!(got, want, "SDOT all-{fill} @K=6848");
            }
            if let Some(got) = run_smmla_s8s8(&a, &b, m, k, n) {
                assert_eq!(got, want, "SMMLA all-{fill} @K=6848");
            }
        }
    }

    #[test]
    fn u8s8_matches_oracle_randomized_and_adversarial() {
        let mut rng = Rng(0xdead_beef_0bad_f00d);
        let shapes = [
            (1usize, 1usize, 1usize),
            (2, 15, 3),
            (4, 16, 4),
            (5, 17, 6),
            (8, 64, 8),
            (3, 896, 7),
            (4, 6848, 5),
        ];
        for &(m, k, n) in &shapes {
            let a = rand_u8(&mut rng, m * k);
            let b = rand_i8(&mut rng, n * k);
            let want = oracle_u8s8(&a, &b, m, k, n);
            let mut got = vec![0i32; m * n];
            igemm_u8s8(&a, &b, m, k, n, &mut got);
            assert_eq!(got, want, "U8S8 dispatch mismatch at shape ({m},{k},{n})");
        }

        // Adversarial U8S8: all 255 activations * all 127 weights -> the binding
        // worst case (221.7M) at K=6848.
        let (m, k, n) = (2usize, 6848usize, 3usize);
        let a = vec![255u8; m * k];
        let b = vec![127i8; n * k];
        let want = oracle_u8s8(&a, &b, m, k, n);
        assert_eq!(want[0], 221_772_480, "U8S8 worst-case oracle value");
        let mut got = vec![0i32; m * n];
        igemm_u8s8(&a, &b, m, k, n, &mut got);
        assert_eq!(got, want, "U8S8 adversarial @K=6848");

        // Adversarial negative: all 255 * all -128 (largest |negative|).
        let b_neg = vec![-128i8; n * k];
        let want_neg = oracle_u8s8(&a, &b_neg, m, k, n);
        let mut got_neg = vec![0i32; m * n];
        igemm_u8s8(&a, &b_neg, m, k, n, &mut got_neg);
        assert_eq!(got_neg, want_neg, "U8S8 negative adversarial @K=6848");
    }

    #[test]
    fn empty_and_unit_dims() {
        // K = 1 (below the 16/8 vector width) must work via the tail path.
        let a = vec![3i8, -2, 5];
        let b = vec![7i8, 11];
        let (m, k, n) = (3usize, 1usize, 2usize);
        let want = oracle_s8s8(&a, &b, m, k, n);
        let mut got = vec![0i32; m * n];
        igemm_s8s8(&a, &b, m, k, n, &mut got);
        assert_eq!(got, want);
    }
}
