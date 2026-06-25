//! The int8/int4 SIMD perf core (Phase 3, plan §6) — the runtime-dispatched
//! GEMM tier stack.
//!
//! This module owns the model's int8 matrix-multiply kernels and the runtime
//! ISA dispatch that selects the fastest one for the host CPU, with a portable
//! scalar oracle as the always-present floor. The public surface is exactly the
//! two GEMM entrypoints re-exported below ([`igemm_s8s8`] / [`igemm_u8s8`]) plus
//! the capability-reflection helpers for `focr robot backends`.
//!
//! ## Layout
//!
//! * [`scalar`] — the **reference oracle** and portable fallback. No `unsafe`;
//!   a tight scalar dot product LLVM autovectorizes (doctrine #3). Every
//!   accelerated kernel is tested bit-identical against it.
//! * [`arm`] — aarch64 NEON kernels: SMMLA/i8mm (the register-blocked wedge) and
//!   SDOT/dotprod. Compiled only on `target_arch = "aarch64"`; the audited
//!   `unsafe` intrinsic island lives there.
//! * [`x86`] — x86-64 kernels: AVX-512-VNNI, AVX-VNNI, AVX2. Compiled only on
//!   `target_arch = "x86_64"`.
//! * [`int4`] — int4 (2 nibbles/byte, per-group scales) unpack-to-int8 path; the
//!   decode-bandwidth wedge (doctrine #4). Portable (the unpack is scalar; it
//!   feeds the same dispatched int8 GEMM), so it is built on every target.
//! * [`dispatch`] — runtime feature detection (`is_aarch64_feature_detected!` /
//!   `is_x86_feature_detected!`), the cached [`IsaTier`] selection, and the
//!   public GEMM entrypoint that routes to the best available kernel (else
//!   scalar). The dispatch contains no `unsafe` — it only *selects* which safe
//!   wrapper to call, and only ever selects a tier whose CPU feature it has
//!   confirmed present (the safety precondition for the intrinsics).
//!
//! ## Contract (PINNED — identical across every backend)
//!
//! ```text
//! // C[M,N] += A[M,K] (row-major) · B[N,K] (OUTPUT-CHANNEL-major) -> i32[M,N]
//! pub fn igemm_s8s8(a: &[i8], b: &[i8], m, k, n, out: &mut [i32]);
//! pub fn igemm_u8s8(a: &[u8], b: &[i8], m, k, n, out: &mut [i32]);
//! ```
//!
//! `b` is **output-channel-major** `[N, K]` (weight row `o` is `b[o*K..o*K+K]`),
//! matching `tensor::QInt8` and `nn::linear_int8_dynamic`. `out` is `+=` into an
//! i32 buffer of length `m*n`. Accumulation is i32; the worst-case-K overflow
//! proof is doctrine #6 (`tests/int32_overflow_proof.rs` + the `scalar` tests).
//!
//! Crate-root `#![deny(unsafe_code)]` holds by default; only the named
//! `arm`/`x86` islands relax it behind
//! `#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]`, each with `// SAFETY:` notes
//! and a bit-identical scalar fallback.

pub mod dispatch;
pub mod int4;
pub mod scalar;

// Arch-specific intrinsic modules. Each is compiled only on its native arch so
// the crate builds on every target (an aarch64-only intrinsic module would not
// type-check under an x86 build, and vice versa). `dispatch.rs` references their
// entrypoints only inside the matching `#[cfg(target_arch = ...)]` match arm, so
// gating the module declaration the same way keeps the whole stack coherent:
// on a non-native arch the module simply does not exist and dispatch falls back
// to the always-present `scalar` floor.
#[cfg(target_arch = "aarch64")]
pub mod arm;
#[cfg(target_arch = "x86_64")]
pub mod x86;

// ── Public SIMD API ─────────────────────────────────────────────────────────
//
// The rest of the engine calls these (and `nn::linear_int8_dynamic` slots its
// int8 path under them). They are the runtime-dispatched entrypoints; callers
// never name a tier.

pub use dispatch::{
    Caps, IsaTier, available_tiers, caps, detected_tier, igemm_s8s8, igemm_u8s8, tier_string,
};

#[cfg(test)]
mod tests {
    //! Module-level coherence tests: the re-exported public API routes to the
    //! dispatcher and stays bit-identical to the scalar oracle on this host.

    use super::{igemm_s8s8, igemm_u8s8, scalar};

    /// The re-exported `igemm_s8s8` is the dispatched entrypoint and equals the
    /// scalar oracle (hand-computed value).
    #[test]
    fn public_s8s8_routes_to_dispatch_and_matches_oracle() {
        let a: [i8; 6] = [1, 2, 3, 4, 5, 6];
        let b: [i8; 6] = [1, 0, 1, 0, 1, 0]; // OC-major [2,3]
        let mut got = [0i32; 4];
        let mut want = [0i32; 4];
        igemm_s8s8(&a, &b, 2, 3, 2, &mut got);
        scalar::igemm_s8s8(&a, &b, 2, 3, 2, &mut want);
        assert_eq!(got, want);
        assert_eq!(got, [4, 2, 10, 5]);
    }

    /// The re-exported `igemm_u8s8` likewise matches the oracle.
    #[test]
    fn public_u8s8_routes_to_dispatch_and_matches_oracle() {
        let a: [u8; 3] = [10, 20, 30];
        let b: [i8; 3] = [2, -1, 1];
        let mut got = [0i32; 1];
        let mut want = [0i32; 1];
        igemm_u8s8(&a, &b, 1, 3, 1, &mut got);
        scalar::igemm_u8s8(&a, &b, 1, 3, 1, &mut want);
        assert_eq!(got, want);
        assert_eq!(got, [30]);
    }

    /// The capability reflection is reachable through the public API and offers
    /// a non-empty available set (for `robot backends`).
    #[test]
    fn public_caps_are_reflectable() {
        let c = super::caps();
        assert!(!c.available.is_empty());
        assert!(!super::tier_string().is_empty());
        assert!(c.available.contains(&super::detected_tier()));
    }
}
