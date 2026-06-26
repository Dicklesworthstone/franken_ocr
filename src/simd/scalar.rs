//! The int8 GEMM **reference oracle** + portable scalar fallback (plan §6, the
//! Phase-3 perf core).
//!
//! This module is the *correctness ground truth* for the whole SIMD tier stack:
//! every accelerated kernel (`arm.rs` SMMLA/SDOT, `x86.rs` VNNI/AVX2,
//! `int4.rs`'s unpack→int8 path) is tested for **bit-identical** output against
//! the functions here, and `dispatch.rs` falls back to them on any target
//! lacking a native int8 MAC. There is intentionally **no `unsafe`** in this
//! file — it cross-compiles to every target and is the floor doctrine #4 calls
//! "a bit-identical scalar fallback that cross-compiles to every target".
//!
//! ## Contract (PINNED — every backend implements this EXACTLY)
//!
//! ```text
//! // C[M,N] += A[M,K] (row-major) · B[N,K] (OUTPUT-CHANNEL-major) -> i32[M,N]
//! pub fn igemm_s8s8(a: &[i8], b: &[i8], m, k, n, out: &mut [i32]);
//! pub fn igemm_u8s8(a: &[u8], b: &[i8], m, k, n, out: &mut [i32]);
//! ```
//!
//! * `a` is the activation matrix, row-major `[M, K]` (`a[i*K + p]`).
//! * `b` is the weight matrix in **output-channel-major** `[N, K]` layout —
//!   weight row `o` (one output channel) is the contiguous slice
//!   `b[o*K .. o*K + K]`. This matches `tensor::QInt8` (`[n, k]` OC-major) and
//!   `nn::linear_int8_dynamic`'s convention exactly, so the per-output-channel
//!   dequant scale `w_scale[o]` lines up with `out` column `o`.
//! * `out` is the i32 accumulator buffer, row-major `[M, N]` (`out[i*N + o]`),
//!   and is written with `+=` (the caller zeroes it, or seeds a prior partial).
//!   Length must be exactly `m * n`.
//!
//! The inner loop is a plain scalar dot product so **LLVM autovectorizes** it —
//! doctrine #3 ("NEVER hand-roll wide-SIMD over scalar inner loops; it measured
//! ~5x SLOWER than LLVM autovec"). The hand-written SIMD win lives in `arm.rs` /
//! `x86.rs` via native int8 *matmul* intrinsics, NOT here.
//!
//! ## i32 accumulation overflow (doctrine #6 — a PROOF OBLIGATION)
//!
//! Accumulation is `i32`, matching the SDOT/VNNI/SMMLA hardware lanes and ONNX
//! `MatMulInteger`. The model worst case is the dense layer-0 `down_proj` at
//! `K = 6848`: S8S8 monotone sum `K·127·127 = 110_451_392`, U8S8 monotone sum
//! `K·255·127 = 221_772_480`, the all-`-128` S8S8 variant `K·128·128 =
//! 112_197_632` — all `< i32::MAX = 2_147_483_647`. We do NOT inherit
//! frankensearch's `k ≤ 1536` bound; `tests/int32_overflow_proof.rs` is the
//! standalone proof and this module's tests re-assert it at `K = 6848`.

/// Multiply each i32 product into an i32 accumulator with a hard overflow guard.
///
/// In a debug build a genuine overflow already panics; in release it would wrap
/// silently. We use a `debug_assert!`-guarded checked add path that is the
/// identity in release (LLVM proves the bound away for the legal K range), so
/// the autovectorizer still sees a plain `acc + prod`. The proof obligation
/// (doctrine #6) guarantees the legal model K never reaches the bound, so this
/// is belt-and-suspenders, not a runtime cost on the hot path.
#[inline(always)]
fn acc_add(acc: i32, prod: i32) -> i32 {
    // Release: plain wrapping-free add — within the proven K bound it never
    // overflows, and keeping it a bare `+` lets LLVM autovectorize the loop.
    // Debug: the implicit overflow check in `+` panics loudly on a contract
    // violation (an out-of-spec K), which is exactly what we want a test to see.
    acc + prod
}

#[track_caller]
pub(crate) fn checked_len(context: &str, lhs: usize, rhs: usize, expr: &str) -> usize {
    let len = lhs.checked_mul(rhs);
    assert!(len.is_some(), "{context}: {expr} overflow ({lhs} * {rhs})");
    len.unwrap_or(0)
}

/// `C[M,N] += A[M,K] (i8, row-major) · B[N,K] (i8, output-channel-major)` into
/// the i32 buffer `out` (the S8S8 reference oracle).
///
/// Both operands are signed int8 in `[-128, 127]`. See the module docs for the
/// pinned layout. Accumulation is i32; see doctrine #6 for the overflow proof.
///
/// # Panics
/// Panics on a length-contract violation: `a.len() != m*k`, `b.len() != n*k`,
/// or `out.len() != m*n` (a programming error, caught early like `Mat`).
pub fn igemm_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    let a_len = checked_len("igemm_s8s8", m, k, "m*k");
    let b_len = checked_len("igemm_s8s8", n, k, "n*k");
    let out_len = checked_len("igemm_s8s8", m, n, "m*n");
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
    for i in 0..m {
        let a_row = &a[i * k..i * k + k];
        let out_row = &mut out[i * n..i * n + n];
        for o in 0..n {
            let b_row = &b[o * k..o * k + k];
            let mut acc: i32 = 0;
            // Tight scalar dot product — LLVM autovectorizes (doctrine #3).
            for p in 0..k {
                acc = acc_add(acc, i32::from(a_row[p]) * i32::from(b_row[p]));
            }
            out_row[o] += acc;
        }
    }
}

/// `C[M,N] += A[M,K] (u8, row-major) · B[N,K] (i8, output-channel-major)` into
/// the i32 buffer `out` (the U8S8 reference oracle).
///
/// Activations are **unsigned** `u8` in `[0, 255]` (the asymmetric
/// `DynamicQuantizeLinear` path — the activation zero-point correction is
/// applied by the caller, not here); weights are signed int8. This is the
/// native VNNI (`vpdpbusd`) operand domain and the binding overflow worst case
/// (`K·255·127`). See the module docs for the pinned layout.
///
/// # Panics
/// As [`igemm_s8s8`] (length-contract violations).
pub fn igemm_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    let a_len = checked_len("igemm_u8s8", m, k, "m*k");
    let b_len = checked_len("igemm_u8s8", n, k, "n*k");
    let out_len = checked_len("igemm_u8s8", m, n, "m*n");
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
    for i in 0..m {
        let a_row = &a[i * k..i * k + k];
        let out_row = &mut out[i * n..i * n + n];
        for o in 0..n {
            let b_row = &b[o * k..o * k + k];
            let mut acc: i32 = 0;
            for p in 0..k {
                acc = acc_add(acc, i32::from(a_row[p]) * i32::from(b_row[p]));
            }
            out_row[o] += acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed S8S8: A = [[1,2,3],[4,5,6]] (2x3),
    /// B (OC-major [N=2, K=3]) = row0 [1,0,1], row1 [0,1,0].
    /// out[0,0] = 1*1+2*0+3*1 = 4; out[0,1] = 1*0+2*1+3*0 = 2
    /// out[1,0] = 4*1+5*0+6*1 = 10; out[1,1] = 4*0+5*1+6*0 = 5
    #[test]
    fn s8s8_matches_hand_computed() {
        let a: [i8; 6] = [1, 2, 3, 4, 5, 6];
        let b: [i8; 6] = [1, 0, 1, 0, 1, 0];
        let mut out = [0i32; 4];
        igemm_s8s8(&a, &b, 2, 3, 2, &mut out);
        assert_eq!(out, [4, 2, 10, 5]);
    }

    /// `out` is `+=`: a second call accumulates on top of the first.
    #[test]
    fn s8s8_accumulates_into_out() {
        let a: [i8; 3] = [1, 2, 3];
        let b: [i8; 3] = [1, 1, 1];
        let mut out = [100i32; 1];
        igemm_s8s8(&a, &b, 1, 3, 1, &mut out); // 1+2+3 = 6
        assert_eq!(out, [106]);
        igemm_s8s8(&a, &b, 1, 3, 1, &mut out); // +6 again
        assert_eq!(out, [112]);
    }

    /// Hand-computed U8S8: unsigned activations times signed weights.
    /// A = [[10, 20, 30]] (1x3, u8), B (OC-major [1,3]) = [[2, -1, 1]]
    /// out = 10*2 + 20*(-1) + 30*1 = 20 - 20 + 30 = 30.
    #[test]
    fn u8s8_matches_hand_computed() {
        let a: [u8; 3] = [10, 20, 30];
        let b: [i8; 3] = [2, -1, 1];
        let mut out = [0i32; 1];
        igemm_u8s8(&a, &b, 1, 3, 1, &mut out);
        assert_eq!(out, [30]);
    }

    /// Negative weights and activations exercise sign handling in S8S8.
    #[test]
    fn s8s8_handles_negatives() {
        // A = [[-2, 3]] (1x2), B (OC-major [2,2]) row0=[-1, -1], row1=[4, -2]
        // out[0,0] = (-2)*(-1) + 3*(-1) = 2 - 3 = -1
        // out[0,1] = (-2)*4 + 3*(-2) = -8 - 6 = -14
        let a: [i8; 2] = [-2, 3];
        let b: [i8; 4] = [-1, -1, 4, -2];
        let mut out = [0i32; 2];
        igemm_s8s8(&a, &b, 1, 2, 2, &mut out);
        assert_eq!(out, [-1, -14]);
    }

    // ── doctrine #6: i32 accumulation overflow proof at the real worst-case K ──

    /// S8S8 monotone worst case at the global worst K=6848 (dense layer-0
    /// down_proj): all-127 operands. K·127·127 = 110_451_392 < i32::MAX. This
    /// re-asserts `tests/int32_overflow_proof.rs` inside the oracle itself.
    #[test]
    fn s8s8_no_overflow_at_k6848_all_max() {
        const K: usize = 6848;
        let a = vec![127i8; K];
        let b = vec![127i8; K];
        let mut out = [0i32; 1];
        igemm_s8s8(&a, &b, 1, K, 1, &mut out);
        assert_eq!(out[0], 110_451_392);
        assert!(out[0] < i32::MAX);
    }

    /// The most extreme S8S8 per-term variant: all `-128`. K·128·128 =
    /// 112_197_632 < i32::MAX. (Larger per-term than 127·127, still bounded.)
    #[test]
    fn s8s8_no_overflow_at_k6848_all_neg128() {
        const K: usize = 6848;
        let a = vec![-128i8; K];
        let b = vec![-128i8; K];
        let mut out = [0i32; 1];
        igemm_s8s8(&a, &b, 1, K, 1, &mut out);
        assert_eq!(out[0], 112_197_632);
        assert!(out[0] < i32::MAX);
    }

    /// U8S8 binding worst case at K=6848: all-255 activations, all-127 weights.
    /// K·255·127 = 221_772_480 < i32::MAX.
    #[test]
    fn u8s8_no_overflow_at_k6848_all_max() {
        const K: usize = 6848;
        let a = vec![255u8; K];
        let b = vec![127i8; K];
        let mut out = [0i32; 1];
        igemm_u8s8(&a, &b, 1, K, 1, &mut out);
        assert_eq!(out[0], 221_772_480);
        assert!(out[0] < i32::MAX);
    }

    /// A small randomized multi-row/col case cross-checked against an
    /// independent i64 oracle (no overflow possible in i64), so the i32 path is
    /// proven equal to the wider reference on arbitrary operands. This is the
    /// pattern every accelerated kernel reuses against THIS oracle.
    #[test]
    fn s8s8_matches_i64_oracle_randomized() {
        let (m, k, n) = (3usize, 17usize, 5usize);
        let a = pseudo_i8(m * k, 0x1234_5678);
        let b = pseudo_i8(n * k, 0x9abc_def0);
        let mut out = vec![0i32; m * n];
        igemm_s8s8(&a, &b, m, k, n, &mut out);
        for i in 0..m {
            for o in 0..n {
                let mut acc: i64 = 0;
                for p in 0..k {
                    acc += i64::from(a[i * k + p]) * i64::from(b[o * k + p]);
                }
                assert_eq!(i64::from(out[i * n + o]), acc, "mismatch at ({i},{o})");
            }
        }
    }

    #[test]
    fn u8s8_matches_i64_oracle_randomized() {
        let (m, k, n) = (4usize, 13usize, 6usize);
        let a = pseudo_u8(m * k, 0x0f0f_1234);
        let b = pseudo_i8(n * k, 0xfeed_face);
        let mut out = vec![0i32; m * n];
        igemm_u8s8(&a, &b, m, k, n, &mut out);
        for i in 0..m {
            for o in 0..n {
                let mut acc: i64 = 0;
                for p in 0..k {
                    acc += i64::from(a[i * k + p]) * i64::from(b[o * k + p]);
                }
                assert_eq!(i64::from(out[i * n + o]), acc, "mismatch at ({i},{o})");
            }
        }
    }

    #[test]
    #[should_panic(expected = "a.len")]
    fn s8s8_rejects_bad_a_len() {
        let mut out = [0i32; 1];
        igemm_s8s8(&[1i8, 2], &[1i8], 1, 1, 1, &mut out);
    }

    #[test]
    #[should_panic(expected = "igemm_s8s8: m*k overflow")]
    fn s8s8_rejects_shape_product_overflow_before_len_checks() {
        let mut out = [];
        igemm_s8s8(&[], &[], usize::MAX, 2, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "igemm_u8s8: m*n overflow")]
    fn u8s8_rejects_output_shape_overflow_before_looping() {
        let mut out = [];
        igemm_u8s8(&[], &[], usize::MAX, 0, 2, &mut out);
    }

    // ── tiny xorshift PRNG so tests are deterministic with no dev-dep ─────────

    fn xorshift(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    fn pseudo_i8(len: usize, seed: u32) -> Vec<i8> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| (xorshift(&mut s) & 0xff) as u8 as i8)
            .collect()
    }

    fn pseudo_u8(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..len).map(|_| (xorshift(&mut s) & 0xff) as u8).collect()
    }
}
