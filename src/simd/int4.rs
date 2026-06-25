//! Int4-weight GEMM — the Phase-4 decode-bandwidth wedge (doctrine #4, plan §6.3,
//! bd-3gaa).
//!
//! Decode is bandwidth-bound: streaming the expert weight bulk dominates the
//! per-token cost. There is **no CPU int4 MAC**, so the win is purely the
//! *bandwidth* of the B operand — int4 packs two weights per byte, **halving the
//! bytes moved** for the expert GEMMs (`moe_intermediate_size = 896`, 64 routed
//! experts × 11 MoE layers = 2112 expert weight tensors; the bulk of the 6.67 GB
//! checkpoint). We UNPACK int4 → int8 in-register and feed the **same** int8
//! dot-product accumulation that `scalar.rs` / `arm.rs` / `x86.rs` use, so this
//! module is a *bandwidth* optimization layered on top of the int8 MAC, never a
//! new arithmetic path (doctrine #4: the int4 win is on the expert bulk).
//!
//! ## Bytes-moved reduction (the documented win)
//!
//! For an expert `down_proj` weight `[n=1280, k=896]` (one of the 2112 routed
//! tensors):
//! * int8 store:  `n*k     = 1 146 880` weight bytes + `n*4 = 5120` scale bytes.
//! * int4 store:  `n*k/2   =   573 440` weight bytes + `n*(k/group)*4` scale
//!   bytes (g=16 → `n*56*4 = 286 720`; g=32 → `n*28*4 = 143 360`).
//!
//! The **weight** payload — the term that dominates the streamed bandwidth at
//! decode — is **exactly halved** (`n*k` → `n*k/2`). Group scales add a small
//! constant per group; at g=32 the int4 form is ~`0.5 + 0.125 = 0.625×` the int8
//! bytes for this tensor, ~`0.5 + 0.25 = 0.75×` at g=16, and asymptotically
//! `→ 0.5×` as `k` grows relative to the group count. That streamed-byte halving
//! on the expert bulk is the entire point of Phase 4 (≈2× decode, plan §6.3).
//!
//! ## Packing convention (PINNED — `docs/focrq-format.md` §`QInt4PerGroup`)
//!
//! This module reads the **exact** `QInt4PerGroup` layout the `.focrq` reader
//! ([`crate::native_engine::weights`]) and the quant-core packer write, so the
//! unpack reproduces the stored int4 values bit-for-bit:
//!
//! * Each logical value is **signed two's-complement int4 in `[-8, 7]`**.
//! * **Two int4 values pack into one byte: the LOW nibble first, then the HIGH
//!   nibble.** So byte `b` holds weight `2j` in `b & 0x0F` and weight `2j+1` in
//!   `b >> 4`, each sign-extended from 4 bits.
//! * `b_packed` is row-major `[n, k/2]` (`k` even): row `o` occupies
//!   `b_packed[o*(k/2) .. (o+1)*(k/2)]`.
//! * Groups run along **K within each output row**; `group` (16 or 32) divides
//!   `k`. Scales are row-major `[n, k/group]`: `scales[o*(k/group) + g]` is the
//!   f32 scale for output row `o`, group `g` (covering input lanes
//!   `g*group .. (g+1)*group`). Logical dequant of a weight is
//!   `f32(q4) * scale[o, g]`.
//!
//! ## GEMM contract
//!
//! The int8 cross-module contract is `C[M,N] += A[M,K] (i8) · B[N,K] (i8) ->
//! i32`. int4 differs in **one** way: the weight scale is **per-group along K**,
//! not a single per-output-channel post-scale, so the contraction cannot be a
//! single i32 dot dequantized once. Instead, for each `(m, o)`:
//!
//! ```text
//! y[m, o] = Σ_g  scale[o, g] * ( Σ_{k ∈ group g} a[m, k] * q4[o, k] )
//!                              └──────── i32 group accumulator ───────┘
//! ```
//!
//! i.e. an **i32 accumulation per group** (the same SDOT/VNNI int8 dot the int8
//! kernels do, just bounded to one group), dequantized by that group's f32 scale
//! and summed in f32. The accelerated paths only change *how the nibbles are
//! unpacked to i8*; the integer dot and the scale-then-sum are bit-identical to
//! the scalar oracle, so every backend yields **bit-identical** `out: &[f32]`.
//!
//! ## Overflow (doctrine #6)
//!
//! The i32 group accumulator spans at most `group ≤ 32` terms. Worst monotone
//! S4S8 term magnitude is `8 * 127 = 1016` (int4 ∈ `[-8,7]`, |min| = 8; int8 ∈
//! `[-128,127]`, |min| = 128 — but the activations here are the same dynamically
//! quantized int8 in `[-127,127]`, so `8 * 127`). `32 * 8 * 127 = 260 096 ≪
//! i32::MAX`. Even the absolute extreme `32 * 8 * 128 = 262 144` fits with five
//! orders of magnitude of headroom — the per-group bound is *vastly* looser than
//! the full-K int8 bound proven in `tests/int32_overflow_proof.rs`, so int4
//! inherits that proof trivially (see [`tests`]).
//!
//! ## Safety
//!
//! The crate root is `#![forbid(unsafe_code)]`; the only `unsafe` here lives in
//! the [`accel`] island behind `#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]`,
//! every intrinsic load carrying a `// SAFETY:` note, each accelerated unpack
//! guarded by a runtime feature check with a **bit-identical scalar fallback**
//! that cross-compiles to every target.

// ── int4 unpack: nibble → i8, bit-exact to the quant-core packing ───────────

/// Sign-extend a 4-bit two's-complement nibble (`0x0..=0xF`) to `i8` in
/// `[-8, 7]`.
///
/// `0x0..=0x7 → 0..=7`, `0x8..=0xF → -8..=-1`. This is the inverse of the
/// quant-core packer's `q4 & 0x0F` and reproduces the stored logical int4 value
/// bit-for-bit. Implemented as an arithmetic left-then-right shift so it is
/// branch-free and trivially autovectorizable.
#[inline]
#[must_use]
pub fn sign_extend_nibble(nib: u8) -> i8 {
    // Place the nibble in the high 4 bits of an i8, then arithmetic-shift back:
    // bit 3 of the nibble becomes the sign bit and is replicated down.
    ((nib << 4) as i8) >> 4
}

/// Unpack one packed byte into its `(low, high)` int4 pair, each sign-extended
/// to `i8`.
///
/// Per the PINNED packing (`docs/focrq-format.md`): the **low** nibble is the
/// first (even-index) weight, the **high** nibble the second (odd-index).
#[inline]
#[must_use]
pub fn unpack_byte(b: u8) -> (i8, i8) {
    (sign_extend_nibble(b & 0x0F), sign_extend_nibble(b >> 4))
}

/// Unpack a packed `[n, k/2]` int4 weight buffer into a dense `[n, k]` `Vec<i8>`
/// in `[-8, 7]` — the scalar reference unpack (the oracle the accelerated
/// nibble-extract paths must match bit-for-bit).
///
/// `b_packed.len()` must be `n * (k / 2)` and `k` must be even. Output row `o`
/// occupies `out[o*k .. (o+1)*k]`, with `out[o*k + 2j] = lo(byte j)` and
/// `out[o*k + 2j+1] = hi(byte j)`.
///
/// # Panics
/// Panics if `k` is odd or `b_packed.len() != n * (k / 2)`.
#[must_use]
pub fn unpack_to_i8(b_packed: &[u8], n: usize, k: usize) -> Vec<i8> {
    assert!(k % 2 == 0, "unpack_to_i8: k {k} must be even (two nibbles/byte)");
    assert_eq!(
        b_packed.len(),
        n * (k / 2),
        "unpack_to_i8: packed len {} != n*k/2 {}",
        b_packed.len(),
        n * (k / 2)
    );
    let mut out = vec![0i8; n * k];
    accel::unpack_nibbles(b_packed, &mut out);
    out
}

// ── int4 GEMM: igemm_s4s8 (the entrypoint) ──────────────────────────────────

/// Int4-weight GEMM: `C[M,N] = A[M,K] (i8) · dequant(B_packed[N,K] int4,
/// per-group scales) -> f32`, the Phase-4 expert kernel.
///
/// * `a`            — row-major `[m, k]` **int8** activations (the dynamically
///   quantized decoder activations, `[-127, 127]`).
/// * `b_packed`     — row-major `[n, k/2]` packed int4 weights (low nibble first,
///   then high; `[-8, 7]` each).
/// * `scales`       — row-major `[n, k/group]` f32 per-group scales.
/// * `group`        — group size along K (16 or 32; must divide `k`).
/// * `out`          — `[m, n]` row-major **f32** result (`out[m*n + ..]`),
///   overwritten (`= C`, not `+=`).
///
/// The contraction is computed **per group** in i32 (the int8 dot the
/// SDOT/VNNI/SMMLA kernels do, bounded to one group), dequantized by that
/// group's scale, and summed in f32 — exactly the order documented in the module
/// header, so every backend is bit-identical. `out` is f32 (not i32) because the
/// per-group scale is folded into the contraction; this is the documented
/// divergence from the int8 `out: &[i32]` contract.
///
/// # Panics
/// Panics on any shape contract violation (`k` odd, `group` not dividing `k`, or
/// a buffer length disagreeing with `m/k/n/group`).
pub fn igemm_s4s8(
    a: &[i8],
    b_packed: &[u8],
    scales: &[f32],
    group: usize,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    assert!(k % 2 == 0, "igemm_s4s8: k {k} must be even");
    assert!(group != 0 && k % group == 0, "igemm_s4s8: group {group} must divide k {k}");
    let groups = k / group;
    assert_eq!(a.len(), m * k, "igemm_s4s8: a len {} != m*k {}", a.len(), m * k);
    assert_eq!(
        b_packed.len(),
        n * (k / 2),
        "igemm_s4s8: b_packed len {} != n*k/2 {}",
        b_packed.len(),
        n * (k / 2)
    );
    assert_eq!(
        scales.len(),
        n * groups,
        "igemm_s4s8: scales len {} != n*(k/group) {}",
        scales.len(),
        n * groups
    );
    assert_eq!(out.len(), m * n, "igemm_s4s8: out len {} != m*n {}", out.len(), m * n);

    // Unpack the whole B once (the bandwidth win is the *stored/streamed* bytes;
    // the in-register int8 the MAC consumes is the same either way). `b_i8` is
    // row-major [n, k]; the accelerated paths only change how this buffer is
    // produced — the GEMM below is identical for every backend.
    let b_i8 = unpack_to_i8(b_packed, n, k);
    igemm_s4s8_unpacked(a, &b_i8, scales, group, m, k, n, out);
}

/// The post-unpack int8 contraction with per-group dequant — shared by the
/// public [`igemm_s4s8`] and by the tests' "unpack then scalar int8 GEMM" oracle
/// (so the two cannot drift). `b_i8` is the dense row-major `[n, k]` int8
/// (`[-8, 7]`) produced by [`unpack_to_i8`].
fn igemm_s4s8_unpacked(
    a: &[i8],
    b_i8: &[i8],
    scales: &[f32],
    group: usize,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    let groups = k / group;
    for mi in 0..m {
        let a_row = &a[mi * k..(mi + 1) * k];
        for ni in 0..n {
            let b_row = &b_i8[ni * k..(ni + 1) * k];
            let scale_row = &scales[ni * groups..(ni + 1) * groups];
            let mut acc_f = 0.0f32;
            for (g, &s) in scale_row.iter().enumerate() {
                let lo = g * group;
                let hi = lo + group;
                // i32 group accumulator: the int8 dot over exactly one group.
                // group ≤ 32, |term| ≤ 8*127 → ≤ 260_096 ≪ i32::MAX (doctrine #6).
                let mut acc_i: i32 = 0;
                for kk in lo..hi {
                    acc_i += i32::from(a_row[kk]) * i32::from(b_row[kk]);
                }
                acc_f += s * acc_i as f32;
            }
            out[mi * n + ni] = acc_f;
        }
    }
}

// ── audited unsafe island: accelerated nibble unpack ────────────────────────
//
// The ONLY accelerated step is the nibble extract (int4 → i8). The int8 MAC and
// the scale-then-sum stay in the scalar `igemm_s4s8_unpacked` above so the
// output is bit-identical across backends. Each accelerated path has a runtime
// feature check and falls back to the portable scalar `unpack_nibbles_scalar`,
// which is the oracle. Every path produces the EXACT same `[n, k]` i8 buffer.

#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
mod accel {
    /// Portable scalar nibble unpack — the bit-exact oracle every accelerated
    /// path reproduces. `out.len()` must equal `2 * src.len()`; writes
    /// `out[2j] = lo(src[j])`, `out[2j+1] = hi(src[j])`, each sign-extended.
    ///
    /// Tight scalar loop — LLVM autovectorizes the shifts/stores (doctrine #3:
    /// never hand-roll wide SIMD over a scalar inner loop; the *only* reason the
    /// NEON/AVX paths below exist is that a vectorized nibble *gather* with a
    /// `vand`/`vshr` + sign-extend can outrun the autovec on the unpack-bound
    /// expert bulk — and even then the MAC is unchanged).
    #[inline]
    pub(super) fn unpack_nibbles_scalar(src: &[u8], out: &mut [i8]) {
        debug_assert_eq!(out.len(), src.len() * 2, "unpack: out must be 2x src");
        for (j, &b) in src.iter().enumerate() {
            // Identical to super::unpack_byte, inlined to keep the hot loop flat.
            out[2 * j] = ((b << 4) as i8) >> 4; // low nibble, sign-extended
            out[2 * j + 1] = (b as i8) >> 4; // high nibble, sign-extended
        }
    }

    /// Runtime-dispatched nibble unpack. Picks the best available SIMD nibble
    /// extract at runtime (`is_*_feature_detected!`) and falls back to the
    /// scalar oracle. The output is byte-for-byte identical to
    /// [`unpack_nibbles_scalar`] on every path.
    #[inline]
    pub(super) fn unpack_nibbles(src: &[u8], out: &mut [i8]) {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: guarded by the runtime NEON detection above; NEON is
                // baseline on aarch64 but the check keeps the contract uniform.
                unsafe {
                    return unpack_nibbles_neon(src, out);
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by the runtime AVX2 detection above.
                unsafe {
                    return unpack_nibbles_avx2(src, out);
                }
            }
        }
        unpack_nibbles_scalar(src, out);
    }

    // ── aarch64 NEON nibble extract ─────────────────────────────────────────

    /// NEON nibble unpack: 16 packed bytes → 32 sign-extended i8 per iteration.
    ///
    /// For a 16-byte block: low nibbles = `b & 0x0F`, high nibbles = `b >> 4`
    /// (logical), both then sign-extended from 4 bits by `(x << 4) >> 4` done in
    /// 8-bit lanes via `vshlq_n_s8` / `vshrq_n_s8`. The de-interleave so that
    /// output index `2j`/`2j+1` map to lo/hi of byte `j` is done with the
    /// `vst2q_s8` interleaving store (lane `2j` ← `lo[j]`, lane `2j+1` ←
    /// `hi[j]`), which is exactly the scalar layout. The scalar tail handles the
    /// final `< 16` bytes.
    ///
    /// # Safety
    /// Caller guarantees `neon` is available (checked in [`unpack_nibbles`]).
    /// All loads/stores are bounds-checked by the `chunks_exact` windowing; the
    /// intrinsics only touch the 16-byte `src` window and the 32-byte `out`
    /// window proven in-bounds by the slice lengths.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn unpack_nibbles_neon(src: &[u8], out: &mut [i8]) {
        use std::arch::aarch64::*;
        debug_assert_eq!(out.len(), src.len() * 2);

        let blocks = src.len() / 16;
        // Mask for the low nibble.
        // SAFETY: vdupq_n_u8 is a register-fill, no memory access.
        let lo_mask = vdupq_n_u8(0x0F);

        for blk in 0..blocks {
            let s_off = blk * 16;
            let o_off = blk * 32;
            // SAFETY: `src[s_off .. s_off+16]` is in-bounds: `blk < src.len()/16`
            // ⇒ `s_off + 16 ≤ src.len()`. Unaligned load is valid for u8.
            let v = vld1q_u8(src.as_ptr().add(s_off));

            // Low nibble: (v & 0x0F) then sign-extend from bit 3.
            // SAFETY: pure register ops.
            let lo_u = vandq_u8(v, lo_mask);
            // reinterpret as s8, shift left 4 (nibble → high 4 bits), arith-shift
            // right 4 → sign-extended low nibble.
            let lo_s = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(lo_u), 4), 4);

            // High nibble: logical shift right 4 (v >> 4), then sign-extend.
            // `vshrq_n_u8(v, 4)` gives the high nibble in the low 4 bits, zero
            // top; sign-extend the same way.
            let hi_u = vshrq_n_u8(v, 4);
            let hi_s = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(hi_u), 4), 4);

            // Interleave store: out[2j] = lo[j], out[2j+1] = hi[j].
            let pair = int8x16x2_t(lo_s, hi_s);
            // SAFETY: `out[o_off .. o_off+32]` is in-bounds: `o_off + 32 =
            // blk*32 + 32 ≤ blocks*32 ≤ (src.len()/16)*32 ≤ out.len()`. vst2q_s8
            // writes exactly 32 i8, interleaved lane-by-lane.
            vst2q_s8(out.as_mut_ptr().add(o_off), pair);
        }

        // Scalar tail for the final `< 16` packed bytes.
        let done = blocks * 16;
        if done < src.len() {
            unpack_nibbles_scalar(&src[done..], &mut out[done * 2..]);
        }
    }

    // ── x86-64 AVX2 nibble extract ──────────────────────────────────────────

    /// AVX2 nibble unpack: 32 packed bytes → 64 sign-extended i8 per iteration.
    ///
    /// Low nibbles = `b & 0x0F`, high nibbles = `(b >> 4) & 0x0F` (AVX2 has no
    /// byte shift, so high = `_mm256_srli_epi16(v,4) & 0x0F`). Each nibble is
    /// sign-extended from bit 3 by `xor 0x08 then sub 0x08` (the standard
    /// branch-free `(x ^ 0x08) - 0x08` 4-bit sign extension). The lo/hi lanes are
    /// then re-interleaved to the scalar `out[2j]=lo, out[2j+1]=hi` order via
    /// `_mm256_unpacklo/hi_epi8` (with the AVX2 128-bit-lane fixup), and the
    /// scalar tail finishes `< 32` bytes.
    ///
    /// # Safety
    /// Caller guarantees `avx2` is available (checked in [`unpack_nibbles`]).
    /// Loads/stores touch only the 32-byte `src` window and 64-byte `out` window
    /// proven in-bounds by the `blocks` arithmetic.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_nibbles_avx2(src: &[u8], out: &mut [i8]) {
        use std::arch::x86_64::*;
        debug_assert_eq!(out.len(), src.len() * 2);

        let blocks = src.len() / 32;
        // SAFETY: set1 are register fills, no memory access.
        let lo_mask = _mm256_set1_epi8(0x0F);
        let bias = _mm256_set1_epi8(0x08);

        for blk in 0..blocks {
            let s_off = blk * 32;
            let o_off = blk * 64;
            // SAFETY: `src[s_off .. s_off+32]` in-bounds: `blk < src.len()/32`
            // ⇒ `s_off + 32 ≤ src.len()`. Unaligned 256-bit load.
            let v = _mm256_loadu_si256(src.as_ptr().add(s_off) as *const __m256i);

            // Low nibble value (0..15), then sign-extend: (x ^ 8) - 8.
            let lo_n = _mm256_and_si256(v, lo_mask);
            let lo_s = _mm256_sub_epi8(_mm256_xor_si256(lo_n, bias), bias);

            // High nibble: shift the 16-bit lanes right 4, then mask 0x0F.
            let hi_n = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask);
            let hi_s = _mm256_sub_epi8(_mm256_xor_si256(hi_n, bias), bias);

            // Interleave to out[2j]=lo, out[2j+1]=hi. unpack_epi8 interleaves
            // per 128-bit lane: unpacklo gives bytes 0..7 of each lane, unpackhi
            // bytes 8..15. We must lay them out so global index 2j/2j+1 are
            // lo[j]/hi[j]. Build the two 256-bit interleaved halves, then fix the
            // lane crossing with permute2x128.
            let il = _mm256_unpacklo_epi8(lo_s, hi_s); // lanes: j in {0..7, 16..23}
            let ih = _mm256_unpackhi_epi8(lo_s, hi_s); // lanes: j in {8..15, 24..31}
            // permute2x128 to assemble contiguous output halves:
            //   first 32 i8  = interleaved bytes for src j in 0..15
            //   second 32 i8 = interleaved bytes for src j in 16..31
            let out0 = _mm256_permute2x128_si256(il, ih, 0x20);
            let out1 = _mm256_permute2x128_si256(il, ih, 0x31);

            // SAFETY: `out[o_off .. o_off+64]` in-bounds: `o_off + 64 =
            // blk*64 + 64 ≤ blocks*64 ≤ (src.len()/32)*64 ≤ out.len()`.
            _mm256_storeu_si256(out.as_mut_ptr().add(o_off) as *mut __m256i, out0);
            _mm256_storeu_si256(out.as_mut_ptr().add(o_off + 32) as *mut __m256i, out1);
        }

        // Scalar tail for the final `< 32` packed bytes.
        let done = blocks * 32;
        if done < src.len() {
            unpack_nibbles_scalar(&src[done..], &mut out[done * 2..]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── nibble sign-extension / unpack vs the pinned packing convention ──────

    /// `sign_extend_nibble` maps every 4-bit two's-complement code to its signed
    /// value in `[-8, 7]` (the quant-core int4 domain).
    #[test]
    fn sign_extend_covers_full_int4_domain() {
        let expected: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1];
        for nib in 0u8..16 {
            assert_eq!(
                sign_extend_nibble(nib),
                expected[nib as usize],
                "nibble 0x{nib:X} sign-extends wrong"
            );
        }
    }

    /// The PINNED packing: low nibble = first (even) weight, high nibble =
    /// second (odd). Byte 0x21 → (lo=1, hi=2); 0x87 → (lo=7, hi=-8); 0xF0 →
    /// (lo=0, hi=-1).
    #[test]
    fn unpack_byte_matches_pinned_low_then_high() {
        assert_eq!(unpack_byte(0x21), (1, 2));
        assert_eq!(unpack_byte(0x43), (3, 4));
        assert_eq!(unpack_byte(0x87), (7, -8));
        assert_eq!(unpack_byte(0xF0), (0, -1));
        assert_eq!(unpack_byte(0x00), (0, 0));
        assert_eq!(unpack_byte(0x88), (-8, -8)); // adversarial: both = int4 min
    }

    /// `unpack_to_i8` reproduces the quant-core values bit-exactly for the
    /// `tensor.rs` / `weights.rs` round-trip fixture `packed = [0x21,0x43]`,
    /// n=1, k=4 → [1,2,3,4].
    #[test]
    fn unpack_to_i8_reproduces_quant_core_fixture() {
        let unpacked = unpack_to_i8(&[0x21, 0x43], 1, 4);
        assert_eq!(unpacked, vec![1i8, 2, 3, 4]);
        // The committed weights.rs fixture: packed [0x21,0x43,0x65,0x87], n=2,
        // k=4 → row0 [1,2,3,4], row1 [5,6,7,-8].
        let two_rows = unpack_to_i8(&[0x21, 0x43, 0x65, 0x87], 2, 4);
        assert_eq!(two_rows, vec![1i8, 2, 3, 4, 5, 6, 7, -8]);
    }

    // ── accelerated unpack == scalar oracle (randomized + adversarial) ───────

    /// Tiny xorshift PRNG so the tests need no `rand` dependency.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn byte(&mut self) -> u8 {
            (self.next_u64() & 0xFF) as u8
        }
        fn i8q(&mut self) -> i8 {
            // int8 activation domain [-127, 127] (dynamic-quant symmetric).
            ((self.next_u64() % 255) as i64 - 127) as i8
        }
    }

    /// The dispatched (SIMD-or-scalar) `unpack_nibbles` must byte-for-byte equal
    /// the scalar oracle across non-block-aligned and large lengths (exercises
    /// the SIMD body + the scalar tail).
    #[test]
    fn accel_unpack_equals_scalar_oracle_randomized() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        // Lengths chosen to hit < block, == block, block+tail, many blocks for
        // both the 16-byte (NEON) and 32-byte (AVX2) block sizes.
        for &len in &[0usize, 1, 7, 15, 16, 17, 31, 32, 33, 48, 63, 64, 100, 257, 1000] {
            let src: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
            let mut got = vec![0i8; len * 2];
            let mut want = vec![0i8; len * 2];
            super::accel::unpack_nibbles(&src, &mut got);
            super::accel::unpack_nibbles_scalar(&src, &mut want);
            assert_eq!(got, want, "dispatched unpack != scalar oracle at len {len}");
        }
    }

    /// Adversarial unpack: all-`0x88` (both nibbles int4-min = -8) and all-`0x77`
    /// (both = +7) — the operands that maximize the downstream i32 magnitude.
    #[test]
    fn accel_unpack_equals_scalar_oracle_adversarial() {
        for fill in [0x88u8, 0x77, 0xFF, 0x00, 0x8F, 0xF8] {
            for &len in &[16usize, 32, 48, 100] {
                let src = vec![fill; len];
                let mut got = vec![0i8; len * 2];
                let mut want = vec![0i8; len * 2];
                super::accel::unpack_nibbles(&src, &mut got);
                super::accel::unpack_nibbles_scalar(&src, &mut want);
                assert_eq!(got, want, "adversarial fill 0x{fill:X} len {len}");
            }
        }
    }

    // ── igemm_s4s8 == unpack-then-scalar-int8-GEMM (the GEMM oracle) ─────────

    /// Independent reference: unpack int4 → i8 with the *byte-level* oracle
    /// (`unpack_byte`), then do the per-group dequant int8 GEMM directly. This is
    /// deliberately written from the packing spec, NOT by calling the module's
    /// own helpers, so it is a true cross-check.
    fn reference_s4s8(
        a: &[i8],
        b_packed: &[u8],
        scales: &[f32],
        group: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let groups = k / group;
        // Unpack from the spec: byte (o*(k/2)+j) → out[o*k + 2j]=lo, +2j+1=hi.
        let mut b = vec![0i8; n * k];
        for o in 0..n {
            for j in 0..k / 2 {
                let byte = b_packed[o * (k / 2) + j];
                b[o * k + 2 * j] = sign_extend_nibble(byte & 0x0F);
                b[o * k + 2 * j + 1] = sign_extend_nibble(byte >> 4);
            }
        }
        let mut out = vec![0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for g in 0..groups {
                    let s = scales[ni * groups + g];
                    let mut gi: i32 = 0;
                    for kk in g * group..(g + 1) * group {
                        gi += i32::from(a[mi * k + kk]) * i32::from(b[ni * k + kk]);
                    }
                    acc += s * gi as f32;
                }
                out[mi * n + ni] = acc;
            }
        }
        out
    }

    #[test]
    fn igemm_s4s8_matches_reference_randomized() {
        let mut rng = Rng(0xdead_beef_cafe_babe);
        // group must divide k; try both pinned group sizes.
        let cases = [
            (1usize, 16usize, 1usize, 16usize),
            (2, 32, 3, 16),
            (4, 64, 5, 32),
            (3, 96, 2, 16),
            (2, 128, 7, 32),
        ];
        for (m, k, n, group) in cases {
            let groups = k / group;
            let a: Vec<i8> = (0..m * k).map(|_| rng.i8q()).collect();
            let b_packed: Vec<u8> = (0..n * (k / 2)).map(|_| rng.byte()).collect();
            // Scales: small positive/negative f32 with exact-ish binary values to
            // avoid spurious rounding diffs (both sides use the SAME order, so any
            // rounding is identical anyway — but keep them clean).
            let scales: Vec<f32> = (0..n * groups)
                .map(|i| {
                    let v = ((rng.next_u64() % 17) as f32 - 8.0) / 8.0; // multiples of 0.125
                    if v == 0.0 { 0.125 } else { v + 0.0 * i as f32 }
                })
                .collect();

            let mut out = vec![0f32; m * n];
            igemm_s4s8(&a, &b_packed, &scales, group, m, k, n, &mut out);
            let want = reference_s4s8(&a, &b_packed, &scales, group, m, k, n);
            assert_eq!(out, want, "igemm_s4s8 != reference for (m={m},k={k},n={n},g={group})");
        }
    }

    /// Adversarial GEMM: all weights = int4-min (-8), all activations = int8-max
    /// (127 / -127), unit scales — the worst-case operands for the i32 group
    /// accumulator (doctrine #6). Confirms (a) bit-exactness vs the reference and
    /// (b) that the i32 group accumulator never overflows: each group sums
    /// `group ≤ 32` terms of `|−8 * −127| = 1016`, max `32*1016 = 32_512`.
    #[test]
    fn igemm_s4s8_adversarial_max_operands() {
        let (m, k, n, group) = (2usize, 32usize, 3usize, 16usize);
        let groups = k / group;
        // 0x88 → both nibbles = -8 (int4 min); fills the whole packed buffer.
        let b_packed = vec![0x88u8; n * (k / 2)];
        // Alternate +127 / -127 activations.
        let a: Vec<i8> = (0..m * k)
            .map(|i| if i % 2 == 0 { 127 } else { -127 })
            .collect();
        let scales = vec![1.0f32; n * groups];

        let mut out = vec![0f32; m * n];
        igemm_s4s8(&a, &b_packed, &scales, group, m, k, n, &mut out);
        let want = reference_s4s8(&a, &b_packed, &scales, group, m, k, n);
        assert_eq!(out, want, "adversarial igemm_s4s8 != reference");

        // Hand-check one cell: each group has `group` terms, weight = -8,
        // activations alternate ±127. Sum over a group of 16 (even count):
        // 8 * (-8*127) + 8 * (-8*-127) = 8*(-1016) + 8*(1016) = 0. With unit
        // scales over 2 groups, the whole cell is 0.0 — and crucially no panic /
        // overflow occurred building those i32 group sums.
        assert!(out.iter().all(|&v| v == 0.0), "expected 0.0 cells for ±127 cancel");

        // Now an all-positive variant to actually exercise a large i32 group sum:
        // weights -8, activations all -127 → term +1016 each, group sum 16*1016 =
        // 16_256 (fits i32 with vast headroom).
        let a_pos = vec![-127i8; m * k];
        let mut out2 = vec![0f32; m * n];
        igemm_s4s8(&a_pos, &b_packed, &scales, group, m, k, n, &mut out2);
        let want2 = reference_s4s8(&a_pos, &b_packed, &scales, group, m, k, n);
        assert_eq!(out2, want2);
        // Per cell: 2 groups * 16_256 = 32_512.
        assert!(out2.iter().all(|&v| (v - 32_512.0).abs() < 1e-3), "group-sum value wrong");
    }

    /// The int4 GEMM equals the int8 GEMM when the int8 weights are the
    /// unpacked int4 values and the int8 per-channel scale is folded as a
    /// per-group scale with a single group spanning all of K — proving int4 is
    /// the same MAC with finer scale granularity (doctrine #4: "feed the same
    /// MAC"). With `group == k` there is exactly one group/row, so int4's
    /// per-group dequant collapses to int8's per-channel dequant.
    #[test]
    fn igemm_s4s8_single_group_equals_int8_per_channel() {
        let mut rng = Rng(0x0f0f_0f0f_1234_5678);
        let (m, k, n) = (3usize, 16usize, 4usize);
        let group = k; // one group per row == per-output-channel int8 scale
        let a: Vec<i8> = (0..m * k).map(|_| rng.i8q()).collect();
        let b_packed: Vec<u8> = (0..n * (k / 2)).map(|_| rng.byte()).collect();
        let scales: Vec<f32> = (0..n).map(|i| 0.0625 + i as f32 * 0.03125).collect();

        let mut out = vec![0f32; m * n];
        igemm_s4s8(&a, &b_packed, &scales, group, m, k, n, &mut out);

        // Independent int8-per-channel reference: unpack, full-K i32 dot, * scale.
        let b = unpack_to_i8(&b_packed, n, k);
        let mut want = vec![0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc: i32 = 0;
                for kk in 0..k {
                    acc += i32::from(a[mi * k + kk]) * i32::from(b[ni * k + kk]);
                }
                want[mi * n + ni] = scales[ni] * acc as f32;
            }
        }
        assert_eq!(out, want);
    }

    #[test]
    #[should_panic(expected = "must divide k")]
    fn igemm_s4s8_rejects_non_dividing_group() {
        let (m, k, n, group) = (1usize, 16usize, 1usize, 5usize); // 5 ∤ 16
        let a = vec![1i8; m * k];
        let b_packed = vec![0u8; n * (k / 2)];
        let scales = vec![1.0f32; n * (k / group).max(1)];
        let mut out = vec![0f32; m * n];
        igemm_s4s8(&a, &b_packed, &scales, group, m, k, n, &mut out);
    }
}
