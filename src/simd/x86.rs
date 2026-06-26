//! x86-64 register-blocked int8 GEMM — the AVX2 / AVX-VNNI / AVX-512-VNNI
//! tiers of the perf wedge (AGENTS.md doctrine #3/#4, PROPOSED_ARCHITECTURE.md
//! §6.6).
//!
//! This is the x86 backend behind the runtime ISA dispatch (`simd::dispatch`,
//! owned by the simd-scalar agent). It implements the **pinned cross-module
//! GEMM entrypoint** identically to the scalar oracle and the ARM backend:
//!
//! ```text
//! // C[M,N] += A[M,K] (i8, row-major) · B[N,K] (i8, OUTPUT-CHANNEL-major) -> i32
//! pub fn igemm_s8s8(a: &[i8], b: &[i8], m, k, n, out: &mut [i32]);
//! pub fn igemm_u8s8(a: &[u8], b: &[i8], m, k, n, out: &mut [i32]);
//! ```
//!
//! `out` is **accumulated into** (`C += A·B`), matching the `+=` in the
//! contract. Every output cell `(r, c)` is the i32 dot product of the
//! contiguous K-vector `A[r, :]` with the contiguous K-vector `B[c, :]`; because
//! B is stored output-channel-major (`[N, K]`) with each output row contiguous
//! over the contraction, B is **already in dot-product packing** for this
//! formulation — no transpose/repack is needed, and the register blocking reuses
//! each loaded A K-vector across an `NR`-wide strip of B rows (and each loaded B
//! K-vector across an `MR`-tall strip of A rows) to reach the doctrine-#4
//! compute:load ≥ 2:1 ratio.
//!
//! ## Three feature tiers (each a separate `#[target_feature]` fn the dispatcher
//! calls ONLY after `is_x86_feature_detected!`)
//!
//! * **AVX2** (`avx2`): no VNNI. We deliberately do **not** use the saturating
//!   `vpmaddubsw` (`_mm256_maddubs_epi16`) pair-multiply — its i16 signed
//!   saturation (`255*127 + 255*127 = 64770 > i16::MAX`) is the bd-2mo.9.1
//!   hazard and would silently diverge from the i32 oracle. Instead we
//!   sign/zero-extend the int8 lanes to i16 and use the **non-saturating**
//!   `vpmaddwd` (`_mm256_madd_epi16`, two i16 products summed into an i32 lane),
//!   accumulating in i32 lanes. This is **bit-identical** to the scalar i32
//!   accumulation (the per-lane i32 accumulator is bounded by the doctrine-#6
//!   proof, `tests/int32_overflow_proof.rs`: U8S8 ≤ 221.7M at K=6848 ≪ i32::MAX).
//! * **AVX-VNNI** (`avxvnni`): `vpdpbusd` (`_mm256_dpbusd_avx_epi32`) — native
//!   u8·s8, 4 MACs per i32 lane. U8S8 maps directly. For S8S8 we use the **+128
//!   offset correction**: with `a' = a + 128` (now u8), `Σ a'·b = Σ a·b + 128·Σ
//!   b`, so we subtract `128 · rowsum(b)` (precomputed once per B row). i32
//!   accumulation is exact, so this is bit-identical to the oracle.
//! * **AVX-512-VNNI** (`avx512vnni` + `avx512vl`/`avx512bw`/`avx512f`):
//!   `_mm512_dpbusd_epi32`, the 64-wide form of the same scheme.
//!
//! ## Memory safety
//!
//! All intrinsics live in the single audited island below
//! (`#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]`), each load/store carrying a
//! `// SAFETY:` note. Every accelerated tier has a **bit-identical scalar
//! fallback** ([`scalar_s8s8`] / [`scalar_u8s8`]) used (a) directly when no
//! feature is detected, (b) on non-x86 targets (this file cross-compiles to
//! aarch64 — the dev machine — where the public entrypoints delegate to the
//! scalar reference so the crate still type-checks and the tests still run), and
//! (c) as the test oracle the SIMD tiers are asserted against.
//!
//! Vectorized loads use **unaligned** moves (`_mm256_loadu_si256` /
//! `_mm512_loadu_si512`); the K-tails are handled by a masked/scalar epilogue so
//! arbitrary K (incl. the worst-case K=6848) is correct without alignment
//! assumptions on the caller's slices.

// This module is the named, audited SIMD island. `unsafe` is permitted ONLY
// here (the crate root is `#![deny(unsafe_code)]`); every intrinsic call is
// annotated with a `// SAFETY:` note and is reachable only after the dispatcher
// has confirmed the corresponding CPU feature via `is_x86_feature_detected!`.
#![allow(unsafe_code, unsafe_op_in_unsafe_fn)]

// ─────────────────────────────────────────────────────────────────────────────
// Public entrypoints — the pinned cross-module GEMM signature.
//
// On x86-64 these are the runtime-dispatched fast paths (the dispatcher in
// `simd::dispatch` may instead call a specific `*_avx2` / `*_vnni` /
// `*_avx512vnni` tier directly after feature detection; these top-level fns
// perform the detection themselves so the backend is also usable standalone and
// always lands on the best available tier with a scalar floor).
//
// On every other architecture they delegate to the bit-identical scalar
// reference so the file cross-compiles (e.g. to aarch64, this dev machine).
// ─────────────────────────────────────────────────────────────────────────────

/// `C[M,N] += A[M,K] (i8) · B[N,K] (i8, OC-major)` into `out` (i32, row-major
/// `[M,N]`). Best available x86 tier at runtime; bit-identical to the scalar
/// oracle.
///
/// # Panics
/// Panics if `a.len() != m*k`, `b.len() != n*k`, or `out.len() != m*n` (a
/// shape/length contract violation is a programming error).
pub fn igemm_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    let a_len = super::scalar::checked_len("igemm_s8s8", m, k, "m*k");
    let b_len = super::scalar::checked_len("igemm_s8s8", n, k, "n*k");
    let out_len = super::scalar::checked_len("igemm_s8s8", m, n, "m*n");
    assert_eq!(
        a.len(),
        a_len,
        "igemm_s8s8: a.len {} != m*k {}",
        a.len(),
        a_len
    );
    assert_eq!(
        b.len(),
        b_len,
        "igemm_s8s8: b.len {} != n*k {}",
        b.len(),
        b_len
    );
    assert_eq!(
        out.len(),
        out_len,
        "igemm_s8s8: out.len {} != m*n {}",
        out.len(),
        out_len
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f")
        {
            // SAFETY: guarded by the feature detection immediately above; the
            // fn enables exactly these features via `#[target_feature]`.
            unsafe {
                x86_avx512vnni::igemm_s8s8_avx512vnni(a, b, m, k, n, out);
            }
            return;
        }
        if is_x86_feature_detected!("avxvnni") {
            // SAFETY: `avxvnni` confirmed present by the guard above.
            unsafe {
                x86_avxvnni::igemm_s8s8_avxvnni(a, b, m, k, n, out);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` confirmed present by the guard above.
            unsafe {
                x86_avx2::igemm_s8s8_avx2(a, b, m, k, n, out);
            }
            return;
        }
    }

    scalar_s8s8(a, b, m, k, n, out);
}

/// `C[M,N] += A[M,K] (u8) · B[N,K] (i8, OC-major)` into `out` (i32, row-major
/// `[M,N]`) — the `DynamicQuantizeLinear` U8S8 path. Best available x86 tier at
/// runtime; bit-identical to the scalar oracle.
///
/// # Panics
/// Panics if `a.len() != m*k`, `b.len() != n*k`, or `out.len() != m*n`.
pub fn igemm_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    let a_len = super::scalar::checked_len("igemm_u8s8", m, k, "m*k");
    let b_len = super::scalar::checked_len("igemm_u8s8", n, k, "n*k");
    let out_len = super::scalar::checked_len("igemm_u8s8", m, n, "m*n");
    assert_eq!(
        a.len(),
        a_len,
        "igemm_u8s8: a.len {} != m*k {}",
        a.len(),
        a_len
    );
    assert_eq!(
        b.len(),
        b_len,
        "igemm_u8s8: b.len {} != n*k {}",
        b.len(),
        b_len
    );
    assert_eq!(
        out.len(),
        out_len,
        "igemm_u8s8: out.len {} != m*n {}",
        out.len(),
        out_len
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f")
        {
            // SAFETY: guarded by the feature detection immediately above.
            unsafe {
                x86_avx512vnni::igemm_u8s8_avx512vnni(a, b, m, k, n, out);
            }
            return;
        }
        if is_x86_feature_detected!("avxvnni") {
            // SAFETY: `avxvnni` confirmed present by the guard above.
            unsafe {
                x86_avxvnni::igemm_u8s8_avxvnni(a, b, m, k, n, out);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` confirmed present by the guard above.
            unsafe {
                x86_avx2::igemm_u8s8_avx2(a, b, m, k, n, out);
            }
            return;
        }
    }

    scalar_u8s8(a, b, m, k, n, out);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit-identical scalar reference (the oracle every tier must reproduce exactly).
//
// These mirror the i32-accumulating dot product of `tests/int32_overflow_proof.rs`
// EXACTLY: per output cell, a single i32 accumulator over the contiguous K-vectors
// `A[r,:]` and `B[c,:]`, added into `out[r*n + c]`. The accumulation order is the
// natural ascending-k order; the SIMD tiers split K across lanes but, because i32
// integer addition is associative and exact (no overflow at any model K — the
// doctrine-#6 proof), the lane-reduced result is bit-identical regardless of
// reduction order.
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar S8S8 reference: `out[r*n+c] += Σ_k A[r,k]·B[c,k]` in i32.
pub fn scalar_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    for r in 0..m {
        let arow = &a[r * k..r * k + k];
        for c in 0..n {
            let brow = &b[c * k..c * k + k];
            let mut acc: i32 = 0;
            for t in 0..k {
                acc += i32::from(arow[t]) * i32::from(brow[t]);
            }
            out[r * n + c] += acc;
        }
    }
}

/// Scalar U8S8 reference: `out[r*n+c] += Σ_k A[r,k]·B[c,k]` in i32 (u8·s8).
pub fn scalar_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    for r in 0..m {
        let arow = &a[r * k..r * k + k];
        for c in 0..n {
            let brow = &b[c * k..c * k + k];
            let mut acc: i32 = 0;
            for t in 0..k {
                acc += i32::from(arow[t]) * i32::from(brow[t]);
            }
            out[r * n + c] += acc;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 tier (no VNNI).
//
// Strategy (bit-exact, non-saturating): for each contiguous 16-element K-chunk,
// widen the int8 lanes to i16 (sign-extend for s8, zero-extend for u8) and use
// `_mm256_madd_epi16` (vpmaddwd) — which computes, per i32 output lane,
// `a16[2i]*b16[2i] + a16[2i+1]*b16[2i+1]` with NO saturation — accumulating into
// eight i32 lanes. This sidesteps the saturating `vpmaddubsw` (bd-2mo.9.1
// hazard). The K-tail (< 16) is handled scalar.
//
// Register blocking: we tile the output in an MR×NR block (2×2) so each loaded A
// K-chunk is reused across NR B rows and each loaded B K-chunk across MR A rows,
// reaching compute:load ≥ 2:1 (doctrine #4). The accumulators are kept in
// vector registers across the whole K loop and horizontally reduced once at the
// end of each block.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86_avx2 {
    // The kernel's load loops use the index for BOTH the destination vector slot
    // `av[i]`/`bv[j]` AND the source pointer arithmetic `a[(r0+i)*k + t]`; an
    // `enumerate()` over one array cannot carry the row/col offset, so the
    // range loop is the correct idiom here.
    #![allow(clippy::needless_range_loop)]
    use core::arch::x86_64::*;

    /// Micro-kernel tile heights/widths (rows of A / rows of B per block).
    const MR: usize = 2;
    const NR: usize = 2;

    /// Horizontally sum the eight i32 lanes of `v` to a scalar i32.
    ///
    /// # Safety
    /// Requires the `avx2` feature (checked by the caller's `#[target_feature]`).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_i32_avx2(v: __m256i) -> i32 {
        // SAFETY: extracting the two 128-bit halves and reducing is defined for
        // any __m256i; no memory access.
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let s = _mm_add_epi32(lo, hi); // 4 i32 lanes
        let s = _mm_hadd_epi32(s, s); // 2 meaningful lanes
        let s = _mm_hadd_epi32(s, s); // lane 0 = total
        _mm_cvtsi128_si32(s)
    }

    /// Sign-extend 16 i8 bytes at `p` to a `__m256i` of 16 i16.
    ///
    /// # Safety
    /// `p` must point to at least 16 readable bytes. `avx2` required.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_i8x16_to_i16(p: *const i8) -> __m256i {
        // SAFETY: caller guarantees 16 readable bytes at `p`; unaligned load.
        let lo = _mm_loadu_si128(p.cast::<__m128i>());
        _mm256_cvtepi8_epi16(lo)
    }

    /// Zero-extend 16 u8 bytes at `p` to a `__m256i` of 16 i16.
    ///
    /// # Safety
    /// `p` must point to at least 16 readable bytes. `avx2` required.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_u8x16_to_i16(p: *const u8) -> __m256i {
        // SAFETY: caller guarantees 16 readable bytes at `p`; unaligned load.
        let lo = _mm_loadu_si128(p.cast::<__m128i>());
        _mm256_cvtepu8_epi16(lo)
    }

    /// S8S8 AVX2 GEMM. See module docs for the non-saturating scheme.
    ///
    /// # Safety
    /// Requires `avx2`. Slices must satisfy `a.len()==m*k`, `b.len()==n*k`,
    /// `out.len()==m*n` (the public entrypoint asserts this before dispatch).
    #[target_feature(enable = "avx2")]
    pub unsafe fn igemm_s8s8_avx2(
        a: &[i8],
        b: &[i8],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [i32],
    ) {
        let k16 = k - (k % 16);
        let mut r0 = 0;
        while r0 < m {
            let mr = MR.min(m - r0);
            let mut c0 = 0;
            while c0 < n {
                let nr = NR.min(n - c0);
                // MR×NR i32 vector accumulators.
                let mut acc = [[_mm256_setzero_si256(); NR]; MR];
                let mut t = 0;
                while t < k16 {
                    // Load the MR A chunks and NR B chunks for this k-window
                    // once, reuse across the tile (compute:load >= 2:1).
                    let mut av = [_mm256_setzero_si256(); MR];
                    for i in 0..mr {
                        // SAFETY: r0+i < m and t+16 <= k16 <= k, so the 16 bytes
                        // at a[(r0+i)*k + t] are in-bounds.
                        av[i] = load_i8x16_to_i16(a.as_ptr().add((r0 + i) * k + t));
                    }
                    let mut bv = [_mm256_setzero_si256(); NR];
                    for j in 0..nr {
                        // SAFETY: c0+j < n and t+16 <= k, so 16 bytes at
                        // b[(c0+j)*k + t] are in-bounds.
                        bv[j] = load_i8x16_to_i16(b.as_ptr().add((c0 + j) * k + t));
                    }
                    for i in 0..mr {
                        for j in 0..nr {
                            // vpmaddwd: non-saturating i16·i16 -> i32 pair-sum.
                            let prod = _mm256_madd_epi16(av[i], bv[j]);
                            acc[i][j] = _mm256_add_epi32(acc[i][j], prod);
                        }
                    }
                    t += 16;
                }
                // Reduce vector accumulators + handle the scalar K-tail.
                for i in 0..mr {
                    for j in 0..nr {
                        let mut s = hsum_i32_avx2(acc[i][j]);
                        let arow = &a[(r0 + i) * k..(r0 + i) * k + k];
                        let brow = &b[(c0 + j) * k..(c0 + j) * k + k];
                        for tt in k16..k {
                            s += i32::from(arow[tt]) * i32::from(brow[tt]);
                        }
                        out[(r0 + i) * n + (c0 + j)] += s;
                    }
                }
                c0 += nr;
            }
            r0 += mr;
        }
    }

    /// U8S8 AVX2 GEMM (a zero-extended, b sign-extended; same non-saturating
    /// `vpmaddwd` accumulation). Worst-case per-lane i32 fits (doctrine #6).
    ///
    /// # Safety
    /// Requires `avx2`. Slice length contract as [`igemm_s8s8_avx2`].
    #[target_feature(enable = "avx2")]
    pub unsafe fn igemm_u8s8_avx2(
        a: &[u8],
        b: &[i8],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [i32],
    ) {
        let k16 = k - (k % 16);
        let mut r0 = 0;
        while r0 < m {
            let mr = MR.min(m - r0);
            let mut c0 = 0;
            while c0 < n {
                let nr = NR.min(n - c0);
                let mut acc = [[_mm256_setzero_si256(); NR]; MR];
                let mut t = 0;
                while t < k16 {
                    let mut av = [_mm256_setzero_si256(); MR];
                    for i in 0..mr {
                        // SAFETY: in-bounds as in the s8s8 path; 16 u8 bytes.
                        av[i] = load_u8x16_to_i16(a.as_ptr().add((r0 + i) * k + t));
                    }
                    let mut bv = [_mm256_setzero_si256(); NR];
                    for j in 0..nr {
                        // SAFETY: in-bounds; 16 i8 bytes.
                        bv[j] = load_i8x16_to_i16(b.as_ptr().add((c0 + j) * k + t));
                    }
                    for i in 0..mr {
                        for j in 0..nr {
                            let prod = _mm256_madd_epi16(av[i], bv[j]);
                            acc[i][j] = _mm256_add_epi32(acc[i][j], prod);
                        }
                    }
                    t += 16;
                }
                for i in 0..mr {
                    for j in 0..nr {
                        let mut s = hsum_i32_avx2(acc[i][j]);
                        let arow = &a[(r0 + i) * k..(r0 + i) * k + k];
                        let brow = &b[(c0 + j) * k..(c0 + j) * k + k];
                        for tt in k16..k {
                            s += i32::from(arow[tt]) * i32::from(brow[tt]);
                        }
                        out[(r0 + i) * n + (c0 + j)] += s;
                    }
                }
                c0 += nr;
            }
            r0 += mr;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX-VNNI tier — `vpdpbusd` (u8·s8 -> i32, 4 MACs/lane).
//
// U8S8 is the native mode. S8S8 uses the +128 offset correction:
//   a' = a + 128  (now u8 in [0,255])
//   Σ a'·b = Σ (a+128)·b = Σ a·b + 128·Σ b
//   => Σ a·b = dpbusd(a', b) - 128·rowsum(b)
// The `128·rowsum(b)` correction is computed once per B row in i32 and is exact,
// so the result is bit-identical to the scalar oracle. The 32-element K-chunk is
// processed with `_mm256_dpbusd_avx_epi32`; the K-tail is scalar.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86_avxvnni {
    // See `x86_avx2`: the load-loop index doubles as a pointer-arithmetic
    // offset, so the range loop is intentional.
    #![allow(clippy::needless_range_loop)]
    use core::arch::x86_64::*;

    const MR: usize = 2;
    const NR: usize = 2;

    /// Horizontally sum the eight i32 lanes of `v`.
    ///
    /// # Safety
    /// Requires `avx2` (a strict subset of `avxvnni`'s prerequisites).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_i32(v: __m256i) -> i32 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_hadd_epi32(s, s);
        let s = _mm_hadd_epi32(s, s);
        _mm_cvtsi128_si32(s)
    }

    /// U8S8 AVX-VNNI GEMM via `vpdpbusd`.
    ///
    /// # Safety
    /// Requires `avxvnni`. Slice length contract as the public entrypoint.
    #[target_feature(enable = "avxvnni")]
    pub unsafe fn igemm_u8s8_avxvnni(
        a: &[u8],
        b: &[i8],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [i32],
    ) {
        let k32 = k - (k % 32);
        let mut r0 = 0;
        while r0 < m {
            let mr = MR.min(m - r0);
            let mut c0 = 0;
            while c0 < n {
                let nr = NR.min(n - c0);
                let mut acc = [[_mm256_setzero_si256(); NR]; MR];
                let mut t = 0;
                while t < k32 {
                    let mut av = [_mm256_setzero_si256(); MR];
                    for i in 0..mr {
                        // SAFETY: r0+i<m, t+32<=k; 32 u8 bytes in-bounds.
                        av[i] =
                            _mm256_loadu_si256(a.as_ptr().add((r0 + i) * k + t).cast::<__m256i>());
                    }
                    let mut bv = [_mm256_setzero_si256(); NR];
                    for j in 0..nr {
                        // SAFETY: c0+j<n, t+32<=k; 32 i8 bytes in-bounds.
                        bv[j] =
                            _mm256_loadu_si256(b.as_ptr().add((c0 + j) * k + t).cast::<__m256i>());
                    }
                    for i in 0..mr {
                        for j in 0..nr {
                            // vpdpbusd: u8(av)·s8(bv), 4 MACs/i32 lane, exact.
                            acc[i][j] = _mm256_dpbusd_avx_epi32(acc[i][j], av[i], bv[j]);
                        }
                    }
                    t += 32;
                }
                for i in 0..mr {
                    for j in 0..nr {
                        let mut s = hsum_i32(acc[i][j]);
                        let arow = &a[(r0 + i) * k..(r0 + i) * k + k];
                        let brow = &b[(c0 + j) * k..(c0 + j) * k + k];
                        for tt in k32..k {
                            s += i32::from(arow[tt]) * i32::from(brow[tt]);
                        }
                        out[(r0 + i) * n + (c0 + j)] += s;
                    }
                }
                c0 += nr;
            }
            r0 += mr;
        }
    }

    /// S8S8 AVX-VNNI GEMM via the +128 offset correction (see module docs).
    ///
    /// # Safety
    /// Requires `avxvnni`. Slice length contract as the public entrypoint.
    #[target_feature(enable = "avxvnni")]
    pub unsafe fn igemm_s8s8_avxvnni(
        a: &[i8],
        b: &[i8],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [i32],
    ) {
        let k32 = k - (k % 32);
        let bias = _mm256_set1_epi8(-128i8); // XOR-add to map s8 -> u8 via +128.
        let mut r0 = 0;
        while r0 < m {
            let mr = MR.min(m - r0);
            let mut c0 = 0;
            while c0 < n {
                let nr = NR.min(n - c0);
                let mut acc = [[_mm256_setzero_si256(); NR]; MR];
                let mut t = 0;
                while t < k32 {
                    let mut av = [_mm256_setzero_si256(); MR];
                    for i in 0..mr {
                        // SAFETY: in-bounds; 32 i8 bytes.
                        let raw =
                            _mm256_loadu_si256(a.as_ptr().add((r0 + i) * k + t).cast::<__m256i>());
                        // a + 128: adding 128 to a signed byte == XOR 0x80,
                        // reinterpreting the result as u8 in [0,255]. We use
                        // add_epi8 with -128 (== +128 mod 256) so the byte
                        // pattern is identical to the unsigned (a as u16 + 128).
                        av[i] = _mm256_add_epi8(raw, bias);
                    }
                    let mut bv = [_mm256_setzero_si256(); NR];
                    for j in 0..nr {
                        // SAFETY: in-bounds; 32 i8 bytes.
                        bv[j] =
                            _mm256_loadu_si256(b.as_ptr().add((c0 + j) * k + t).cast::<__m256i>());
                    }
                    for i in 0..mr {
                        for j in 0..nr {
                            // dpbusd((a+128), b) = Σ(a+128)·b over this chunk.
                            acc[i][j] = _mm256_dpbusd_avx_epi32(acc[i][j], av[i], bv[j]);
                        }
                    }
                    t += 32;
                }
                for i in 0..mr {
                    for j in 0..nr {
                        let arow = &a[(r0 + i) * k..(r0 + i) * k + k];
                        let brow = &b[(c0 + j) * k..(c0 + j) * k + k];
                        // Σ(a+128)·b over the vectorized chunk, then SUBTRACT the
                        // 128·Σb correction over the SAME [0,k32) range, then add
                        // the exact scalar tail.
                        let mut s = hsum_i32(acc[i][j]);
                        let mut bsum_vec: i32 = 0;
                        for tt in 0..k32 {
                            bsum_vec += i32::from(brow[tt]);
                        }
                        s -= 128 * bsum_vec;
                        for tt in k32..k {
                            s += i32::from(arow[tt]) * i32::from(brow[tt]);
                        }
                        out[(r0 + i) * n + (c0 + j)] += s;
                    }
                }
                c0 += nr;
            }
            r0 += mr;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX-512-VNNI tier — `_mm512_dpbusd_epi32` (64-wide form of the above).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86_avx512vnni {
    // See `x86_avx2`: the load-loop index doubles as a pointer-arithmetic
    // offset, so the range loop is intentional.
    #![allow(clippy::needless_range_loop)]
    use core::arch::x86_64::*;

    const MR: usize = 2;
    const NR: usize = 2;

    /// Horizontally sum the sixteen i32 lanes of a `__m512i`.
    ///
    /// # Safety
    /// Requires `avx512f`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn hsum_i32_512(v: __m512i) -> i32 {
        _mm512_reduce_add_epi32(v)
    }

    /// U8S8 AVX-512-VNNI GEMM via `vpdpbusd` (zmm).
    ///
    /// # Safety
    /// Requires `avx512vnni`+`avx512bw`+`avx512f`. Slice length contract as the
    /// public entrypoint.
    #[target_feature(enable = "avx512vnni,avx512bw,avx512f")]
    pub unsafe fn igemm_u8s8_avx512vnni(
        a: &[u8],
        b: &[i8],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [i32],
    ) {
        let k64 = k - (k % 64);
        let mut r0 = 0;
        while r0 < m {
            let mr = MR.min(m - r0);
            let mut c0 = 0;
            while c0 < n {
                let nr = NR.min(n - c0);
                let mut acc = [[_mm512_setzero_si512(); NR]; MR];
                let mut t = 0;
                while t < k64 {
                    let mut av = [_mm512_setzero_si512(); MR];
                    for i in 0..mr {
                        // SAFETY: r0+i<m, t+64<=k; 64 u8 bytes in-bounds.
                        av[i] =
                            _mm512_loadu_si512(a.as_ptr().add((r0 + i) * k + t).cast::<__m512i>());
                    }
                    let mut bv = [_mm512_setzero_si512(); NR];
                    for j in 0..nr {
                        // SAFETY: c0+j<n, t+64<=k; 64 i8 bytes in-bounds.
                        bv[j] =
                            _mm512_loadu_si512(b.as_ptr().add((c0 + j) * k + t).cast::<__m512i>());
                    }
                    for i in 0..mr {
                        for j in 0..nr {
                            acc[i][j] = _mm512_dpbusd_epi32(acc[i][j], av[i], bv[j]);
                        }
                    }
                    t += 64;
                }
                for i in 0..mr {
                    for j in 0..nr {
                        let mut s = hsum_i32_512(acc[i][j]);
                        let arow = &a[(r0 + i) * k..(r0 + i) * k + k];
                        let brow = &b[(c0 + j) * k..(c0 + j) * k + k];
                        for tt in k64..k {
                            s += i32::from(arow[tt]) * i32::from(brow[tt]);
                        }
                        out[(r0 + i) * n + (c0 + j)] += s;
                    }
                }
                c0 += nr;
            }
            r0 += mr;
        }
    }

    /// S8S8 AVX-512-VNNI GEMM via the +128 offset correction (zmm).
    ///
    /// # Safety
    /// Requires `avx512vnni`+`avx512bw`+`avx512f`. Slice length contract as the
    /// public entrypoint.
    #[target_feature(enable = "avx512vnni,avx512bw,avx512f")]
    pub unsafe fn igemm_s8s8_avx512vnni(
        a: &[i8],
        b: &[i8],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [i32],
    ) {
        let k64 = k - (k % 64);
        let bias = _mm512_set1_epi8(-128i8);
        let mut r0 = 0;
        while r0 < m {
            let mr = MR.min(m - r0);
            let mut c0 = 0;
            while c0 < n {
                let nr = NR.min(n - c0);
                let mut acc = [[_mm512_setzero_si512(); NR]; MR];
                let mut t = 0;
                while t < k64 {
                    let mut av = [_mm512_setzero_si512(); MR];
                    for i in 0..mr {
                        // SAFETY: in-bounds; 64 i8 bytes.
                        let raw =
                            _mm512_loadu_si512(a.as_ptr().add((r0 + i) * k + t).cast::<__m512i>());
                        av[i] = _mm512_add_epi8(raw, bias); // a + 128 (mod 256)
                    }
                    let mut bv = [_mm512_setzero_si512(); NR];
                    for j in 0..nr {
                        // SAFETY: in-bounds; 64 i8 bytes.
                        bv[j] =
                            _mm512_loadu_si512(b.as_ptr().add((c0 + j) * k + t).cast::<__m512i>());
                    }
                    for i in 0..mr {
                        for j in 0..nr {
                            acc[i][j] = _mm512_dpbusd_epi32(acc[i][j], av[i], bv[j]);
                        }
                    }
                    t += 64;
                }
                for i in 0..mr {
                    for j in 0..nr {
                        let arow = &a[(r0 + i) * k..(r0 + i) * k + k];
                        let brow = &b[(c0 + j) * k..(c0 + j) * k + k];
                        let mut s = hsum_i32_512(acc[i][j]);
                        let mut bsum_vec: i32 = 0;
                        for tt in 0..k64 {
                            bsum_vec += i32::from(brow[tt]);
                        }
                        s -= 128 * bsum_vec;
                        for tt in k64..k {
                            s += i32::from(arow[tt]) * i32::from(brow[tt]);
                        }
                        out[(r0 + i) * n + (c0 + j)] += s;
                    }
                }
                c0 += nr;
            }
            r0 += mr;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — bit-identical-vs-scalar over randomized + adversarial operands,
// including the doctrine-#6 worst-case K = 6848.
//
// On x86-64 (the CI dist matrix) the tiers actually execute when the host CPU
// has the feature, and are asserted bit-equal to the scalar oracle; where a
// feature is absent the tier is skipped (logged) and only the dispatched path +
// scalar floor are exercised. On aarch64 (this dev machine) only the scalar
// reference and the dispatch delegation are exercised — that is expected and
// documented (the file compile-checks here; the SIMD tiers run on x86 CI).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic xorshift PRNG so the tests need no `rand` dep and
    /// are fully reproducible across arches.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn i8(&mut self) -> i8 {
            (self.next_u64() & 0xFF) as u8 as i8
        }
        fn u8(&mut self) -> u8 {
            (self.next_u64() & 0xFF) as u8
        }
    }

    /// Independent i64 oracle (cannot wrap at any model K) — the second witness
    /// the scalar reference is itself checked against, so the whole tower of
    /// equalities is anchored to an overflow-proof computation.
    fn oracle_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut out = vec![0i32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc: i64 = 0;
                for t in 0..k {
                    acc += i64::from(a[r * k + t]) * i64::from(b[c * k + t]);
                }
                out[r * n + c] = acc as i32;
            }
        }
        out
    }

    fn oracle_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut out = vec![0i32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc: i64 = 0;
                for t in 0..k {
                    acc += i64::from(a[r * k + t]) * i64::from(b[c * k + t]);
                }
                out[r * n + c] = acc as i32;
            }
        }
        out
    }

    fn rand_a_s8(rng: &mut Rng, len: usize) -> Vec<i8> {
        (0..len).map(|_| rng.i8()).collect()
    }
    fn rand_a_u8(rng: &mut Rng, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.u8()).collect()
    }
    fn rand_b(rng: &mut Rng, len: usize) -> Vec<i8> {
        (0..len).map(|_| rng.i8()).collect()
    }

    // ── scalar reference is anchored to the i64 oracle ─────────────────────────

    #[test]
    fn scalar_s8s8_matches_i64_oracle() {
        let mut rng = Rng::new(1);
        for &(m, k, n) in &[(1, 1, 1), (2, 16, 3), (3, 17, 4), (4, 33, 5), (2, 6848, 2)] {
            let a = rand_a_s8(&mut rng, m * k);
            let b = rand_b(&mut rng, n * k);
            let mut out = vec![0i32; m * n];
            scalar_s8s8(&a, &b, m, k, n, &mut out);
            assert_eq!(out, oracle_s8s8(&a, &b, m, k, n), "m={m} k={k} n={n}");
        }
    }

    #[test]
    fn scalar_u8s8_matches_i64_oracle() {
        let mut rng = Rng::new(2);
        for &(m, k, n) in &[(1, 1, 1), (2, 16, 3), (3, 17, 4), (4, 33, 5), (2, 6848, 2)] {
            let a = rand_a_u8(&mut rng, m * k);
            let b = rand_b(&mut rng, n * k);
            let mut out = vec![0i32; m * n];
            scalar_u8s8(&a, &b, m, k, n, &mut out);
            assert_eq!(out, oracle_u8s8(&a, &b, m, k, n), "m={m} k={k} n={n}");
        }
    }

    /// The `+=` accumulation contract: a pre-seeded `out` is added to, not
    /// overwritten.
    #[test]
    fn accumulation_adds_into_out() {
        let a = vec![1i8, 2, 3];
        let b = vec![1i8, 1, 1, 2, 0, 1]; // n=2, k=3
        let mut out = vec![100i32, 200];
        scalar_s8s8(&a, &b, 1, 3, 2, &mut out);
        // dots: [1+2+3, 2+0+3] = [6, 5]; out += => [106, 205].
        assert_eq!(out, vec![106, 205]);
    }

    // ── public dispatch entrypoint == scalar oracle on every host ──────────────

    #[test]
    fn dispatch_s8s8_matches_oracle_random_and_adversarial() {
        let mut rng = Rng::new(3);
        let shapes = [
            (1, 1, 1),
            (2, 16, 2),
            (3, 31, 4),
            (5, 64, 3),
            (2, 100, 2),
            (1, 6848, 1),
        ];
        for &(m, k, n) in &shapes {
            let a = rand_a_s8(&mut rng, m * k);
            let b = rand_b(&mut rng, n * k);
            let mut out = vec![0i32; m * n];
            igemm_s8s8(&a, &b, m, k, n, &mut out);
            assert_eq!(
                out,
                oracle_s8s8(&a, &b, m, k, n),
                "random m={m} k={k} n={n}"
            );
        }
        // Adversarial all-max operands at the worst-case K (doctrine #6).
        for &(m, k, n) in &[(2, 6848, 2), (1, 6848, 3)] {
            // all +127
            let a = vec![127i8; m * k];
            let b = vec![127i8; n * k];
            let mut out = vec![0i32; m * n];
            igemm_s8s8(&a, &b, m, k, n, &mut out);
            assert_eq!(out, oracle_s8s8(&a, &b, m, k, n), "+127 m={m} k={k} n={n}");
            // all -128 (max |product|)
            let a = vec![-128i8; m * k];
            let b = vec![-128i8; n * k];
            let mut out = vec![0i32; m * n];
            igemm_s8s8(&a, &b, m, k, n, &mut out);
            assert_eq!(out, oracle_s8s8(&a, &b, m, k, n), "-128 m={m} k={k} n={n}");
            // mixed extreme: a all -128, b all +127 (largest |negative| sum)
            let a = vec![-128i8; m * k];
            let b = vec![127i8; n * k];
            let mut out = vec![0i32; m * n];
            igemm_s8s8(&a, &b, m, k, n, &mut out);
            assert_eq!(out, oracle_s8s8(&a, &b, m, k, n), "mixed m={m} k={k} n={n}");
        }
    }

    #[test]
    fn dispatch_u8s8_matches_oracle_random_and_adversarial() {
        let mut rng = Rng::new(4);
        let shapes = [
            (1, 1, 1),
            (2, 16, 2),
            (3, 31, 4),
            (5, 64, 3),
            (2, 100, 2),
            (1, 6848, 1),
        ];
        for &(m, k, n) in &shapes {
            let a = rand_a_u8(&mut rng, m * k);
            let b = rand_b(&mut rng, n * k);
            let mut out = vec![0i32; m * n];
            igemm_u8s8(&a, &b, m, k, n, &mut out);
            assert_eq!(
                out,
                oracle_u8s8(&a, &b, m, k, n),
                "random m={m} k={k} n={n}"
            );
        }
        // Adversarial worst case: u8 all 255 * s8 all 127 (binding U8S8 worst).
        for &(m, k, n) in &[(2, 6848, 2), (1, 6848, 3)] {
            let a = vec![255u8; m * k];
            let b = vec![127i8; n * k];
            let mut out = vec![0i32; m * n];
            igemm_u8s8(&a, &b, m, k, n, &mut out);
            assert_eq!(
                out,
                oracle_u8s8(&a, &b, m, k, n),
                "255*127 m={m} k={k} n={n}"
            );
            // u8 all 255 * s8 all -128 (largest |negative|).
            let b = vec![-128i8; n * k];
            let mut out = vec![0i32; m * n];
            igemm_u8s8(&a, &b, m, k, n, &mut out);
            assert_eq!(
                out,
                oracle_u8s8(&a, &b, m, k, n),
                "255*-128 m={m} k={k} n={n}"
            );
        }
    }

    // ── per-tier bit-identity (executes only where the feature is present) ─────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_tiers_bit_identical_to_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("[skip] avx2 not present on this host");
            return;
        }
        let mut rng = Rng::new(10);
        let shapes = [
            (1, 1, 1),
            (2, 15, 3),
            (3, 16, 2),
            (4, 17, 5),
            (2, 64, 4),
            (3, 6848, 2),
        ];
        for &(m, k, n) in &shapes {
            // s8s8
            let a = rand_a_s8(&mut rng, m * k);
            let b = rand_b(&mut rng, n * k);
            let mut got = vec![0i32; m * n];
            let mut want = vec![0i32; m * n];
            // SAFETY: avx2 confirmed present by the guard above.
            unsafe { super::x86_avx2::igemm_s8s8_avx2(&a, &b, m, k, n, &mut got) };
            scalar_s8s8(&a, &b, m, k, n, &mut want);
            assert_eq!(got, want, "avx2 s8s8 m={m} k={k} n={n}");
            // u8s8
            let au = rand_a_u8(&mut rng, m * k);
            let mut got = vec![0i32; m * n];
            let mut want = vec![0i32; m * n];
            // SAFETY: avx2 confirmed present.
            unsafe { super::x86_avx2::igemm_u8s8_avx2(&au, &b, m, k, n, &mut got) };
            scalar_u8s8(&au, &b, m, k, n, &mut want);
            assert_eq!(got, want, "avx2 u8s8 m={m} k={k} n={n}");
        }
        // adversarial all-max at worst-case K
        let (m, k, n) = (2, 6848, 2);
        let a = vec![127i8; m * k];
        let b = vec![127i8; n * k];
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        // SAFETY: avx2 present.
        unsafe { super::x86_avx2::igemm_s8s8_avx2(&a, &b, m, k, n, &mut got) };
        scalar_s8s8(&a, &b, m, k, n, &mut want);
        assert_eq!(got, want, "avx2 s8s8 adversarial");
        let au = vec![255u8; m * k];
        let b = vec![127i8; n * k];
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        // SAFETY: avx2 present.
        unsafe { super::x86_avx2::igemm_u8s8_avx2(&au, &b, m, k, n, &mut got) };
        scalar_u8s8(&au, &b, m, k, n, &mut want);
        assert_eq!(got, want, "avx2 u8s8 adversarial");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avxvnni_tiers_bit_identical_to_scalar() {
        if !is_x86_feature_detected!("avxvnni") {
            eprintln!("[skip] avxvnni not present on this host");
            return;
        }
        let mut rng = Rng::new(11);
        let shapes = [
            (1, 1, 1),
            (2, 31, 3),
            (3, 32, 2),
            (4, 33, 5),
            (2, 96, 4),
            (3, 6848, 2),
        ];
        for &(m, k, n) in &shapes {
            let a = rand_a_s8(&mut rng, m * k);
            let b = rand_b(&mut rng, n * k);
            let mut got = vec![0i32; m * n];
            let mut want = vec![0i32; m * n];
            // SAFETY: avxvnni confirmed present.
            unsafe { super::x86_avxvnni::igemm_s8s8_avxvnni(&a, &b, m, k, n, &mut got) };
            scalar_s8s8(&a, &b, m, k, n, &mut want);
            assert_eq!(got, want, "avxvnni s8s8 m={m} k={k} n={n}");
            let au = rand_a_u8(&mut rng, m * k);
            let mut got = vec![0i32; m * n];
            let mut want = vec![0i32; m * n];
            // SAFETY: avxvnni present.
            unsafe { super::x86_avxvnni::igemm_u8s8_avxvnni(&au, &b, m, k, n, &mut got) };
            scalar_u8s8(&au, &b, m, k, n, &mut want);
            assert_eq!(got, want, "avxvnni u8s8 m={m} k={k} n={n}");
        }
        // adversarial (exercises the +128 correction at the extreme)
        let (m, k, n) = (2, 6848, 2);
        let a = vec![-128i8; m * k];
        let b = vec![127i8; n * k];
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        // SAFETY: avxvnni present.
        unsafe { super::x86_avxvnni::igemm_s8s8_avxvnni(&a, &b, m, k, n, &mut got) };
        scalar_s8s8(&a, &b, m, k, n, &mut want);
        assert_eq!(got, want, "avxvnni s8s8 adversarial -128*127");
        let au = vec![255u8; m * k];
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        // SAFETY: avxvnni present.
        unsafe { super::x86_avxvnni::igemm_u8s8_avxvnni(&au, &b, m, k, n, &mut got) };
        scalar_u8s8(&au, &b, m, k, n, &mut want);
        assert_eq!(got, want, "avxvnni u8s8 adversarial 255*127");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512vnni_tiers_bit_identical_to_scalar() {
        if !(is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f"))
        {
            eprintln!("[skip] avx512vnni/bw/f not present on this host");
            return;
        }
        let mut rng = Rng::new(12);
        let shapes = [
            (1, 1, 1),
            (2, 63, 3),
            (3, 64, 2),
            (4, 65, 5),
            (2, 192, 4),
            (3, 6848, 2),
        ];
        for &(m, k, n) in &shapes {
            let a = rand_a_s8(&mut rng, m * k);
            let b = rand_b(&mut rng, n * k);
            let mut got = vec![0i32; m * n];
            let mut want = vec![0i32; m * n];
            // SAFETY: avx512vnni/bw/f confirmed present.
            unsafe { super::x86_avx512vnni::igemm_s8s8_avx512vnni(&a, &b, m, k, n, &mut got) };
            scalar_s8s8(&a, &b, m, k, n, &mut want);
            assert_eq!(got, want, "avx512vnni s8s8 m={m} k={k} n={n}");
            let au = rand_a_u8(&mut rng, m * k);
            let mut got = vec![0i32; m * n];
            let mut want = vec![0i32; m * n];
            // SAFETY: avx512vnni/bw/f present.
            unsafe { super::x86_avx512vnni::igemm_u8s8_avx512vnni(&au, &b, m, k, n, &mut got) };
            scalar_u8s8(&au, &b, m, k, n, &mut want);
            assert_eq!(got, want, "avx512vnni u8s8 m={m} k={k} n={n}");
        }
        let (m, k, n) = (2, 6848, 2);
        let a = vec![-128i8; m * k];
        let b = vec![127i8; n * k];
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        // SAFETY: avx512vnni/bw/f present.
        unsafe { super::x86_avx512vnni::igemm_s8s8_avx512vnni(&a, &b, m, k, n, &mut got) };
        scalar_s8s8(&a, &b, m, k, n, &mut want);
        assert_eq!(got, want, "avx512vnni s8s8 adversarial");
        let au = vec![255u8; m * k];
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        // SAFETY: avx512vnni/bw/f present.
        unsafe { super::x86_avx512vnni::igemm_u8s8_avx512vnni(&au, &b, m, k, n, &mut got) };
        scalar_u8s8(&au, &b, m, k, n, &mut want);
        assert_eq!(got, want, "avx512vnni u8s8 adversarial");
    }
}
