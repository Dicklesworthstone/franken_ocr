//! Batched-M int8 GEMM parity gate (bd-1azu.2 — the deepest lossless leaf of the
//! Phase-6 continuous-batch decode spine, bd-1azu).
//!
//! The whole batched spine (Lever B: GEMV→GEMM over B in-flight page-streams)
//! rests on ONE invariant: stacking B streams' hidden rows into a single
//! `[B, K]` activation matrix and running ONE `M=B` GEMM must produce, for every
//! row `r`, output **byte-for-byte identical** to running that row alone as an
//! `m=1` GEMV. M-blocking changes only LOAD SCHEDULING; the per-output i32
//! contraction is identical for any M, so this is LOSSLESS by construction —
//! this file is the executing proof of that, the parity gate Doctrine #1 demands
//! BEFORE any forward driver (bd-1azu.3) batches on top of it.
//!
//! Two independent checks per (shape, M, tier):
//!   1. **Batched invariant** — `igemm(M=B)[r] == igemm(m=1 of row r)`, the
//!      kernel-vs-itself property the spine needs (no cross-row contamination,
//!      correct M-block REMAINDER handling when B is not a multiple of the
//!      register-block height MR=4/8).
//!   2. **i64 absolute oracle** — `igemm(M=B)[cell]` equals the true dot product
//!      accumulated in i64, AND that value fits in i32. A kernel-vs-kernel parity
//!      check alone cannot catch a SHARED i32-accumulator overflow (both sides
//!      would wrap identically); the i64 oracle at the Doctrine-#6 worst case
//!      K=6848 (dense layer-0 `down_proj`) does (plan §5.4).
//!
//! Tier coverage: the public dispatch entrypoint runs the host's selected tier
//! (SDOT on Apple Silicon, AVX2/VNNI on x86, …). The whole binary is re-run under
//! `FOCR_FORCE_ARCH=scalar|sdot|smmla|avx2|avxvnni|avx512vnni` to sweep EVERY
//! tier the host can dispatch (the `FOCR_FORCE_ARCH` knob — dispatch.rs — is the
//! parity-sweep kill-switch named "FOCR_ISA_TIER" in the bead; it is cached in a
//! OnceLock, so the sweep is by process re-exec, not in-process). When the env
//! var is set to a tier present on the host, [`forced_tier_takes_effect`] asserts
//! the force actually changed the dispatched kernel, so the sweep is
//! self-verifying rather than silently testing the same tier N times.

use franken_ocr::simd::{IsaTier, detected_tier, effective_dense_route, igemm_s8s8, igemm_u8s8};

/// Deterministic xorshift64 PRNG — reproducible, no dev-dependency.
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
    fn fill_i8(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.i8()).collect()
    }
    fn fill_u8(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.u8()).collect()
    }
}

/// True S8S8 dot products accumulated in i64 (the absolute oracle, independent of
/// any kernel). `a` is `[m,k]` row-major, `b` is `[n,k]` output-channel-major.
fn oracle_i64_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i64> {
    let mut out = vec![0i64; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc: i64 = 0;
            for t in 0..k {
                acc += i64::from(a[r * k + t]) * i64::from(b[col * k + t]);
            }
            out[r * n + col] = acc;
        }
    }
    out
}

/// True U8S8 dot products in i64 (unsigned activations · signed weights).
fn oracle_i64_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i64> {
    let mut out = vec![0i64; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc: i64 = 0;
            for t in 0..k {
                acc += i64::from(a[r * k + t]) * i64::from(b[col * k + t]);
            }
            out[r * n + col] = acc;
        }
    }
    out
}

/// Assert every i64-oracle cell fits in i32 (no accumulator overflow at this K)
/// and the i32 kernel result equals it exactly.
fn assert_fits_and_matches(got: &[i32], oracle: &[i64], label: &str) {
    for (idx, (&g, &o)) in got.iter().zip(oracle.iter()).enumerate() {
        assert!(
            o >= i64::from(i32::MIN) && o <= i64::from(i32::MAX),
            "{label}: oracle cell {idx} = {o} overflows i32 (the proof obligation is violated)"
        );
        assert_eq!(
            i64::from(g),
            o,
            "{label}: kernel cell {idx} = {g} != i64-oracle {o}"
        );
    }
}

/// M values that straddle the register-block heights used by the kernels
/// (SDOT MR=4, SMMLA 8-row panels, x86 MR=4): exact multiples, multiples±1, and
/// a large batch (B=256, a realistic continuous-batch in-flight count).
const M_SWEEP: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256,
];

/// N values that straddle the N register-block (NR=4) and the K-tail boundary.
const N_SWEEP: &[usize] = &[1, 4, 5, 7, 10, 64, 65];

/// The decode-relevant K values plus the Doctrine-#6 worst case. Small for the
/// exhaustive M×N sweep; K=6848 is exercised separately at a few M.
const K_SWEEP: &[usize] = &[1, 16, 17, 128];

#[test]
fn batched_m_equals_per_row_gemv_s8s8() {
    let mut rng = Rng(0x5eed_1a2b_3c4d_5e6f);
    for &k in K_SWEEP {
        for &n in N_SWEEP {
            let b = rng.fill_i8(n * k);
            for &m in M_SWEEP {
                let a = rng.fill_i8(m * k);

                // ONE batched M=B GEMM.
                let mut batched = vec![0i32; m * n];
                igemm_s8s8(&a, &b, m, k, n, &mut batched);

                // Same operands, but each row run ALONE as an m=1 GEMV.
                for r in 0..m {
                    let row = &a[r * k..(r + 1) * k];
                    let mut single = vec![0i32; n];
                    igemm_s8s8(row, &b, 1, k, n, &mut single);
                    assert_eq!(
                        &batched[r * n..(r + 1) * n],
                        &single[..],
                        "S8S8 batched row {r} != standalone m=1 GEMV (tier {}, m={m} k={k} n={n})",
                        effective_dense_route().tag()
                    );
                }

                // Absolute correctness vs the i64 oracle.
                let oracle = oracle_i64_s8s8(&a, &b, m, k, n);
                assert_fits_and_matches(
                    &batched,
                    &oracle,
                    &format!(
                        "S8S8 m={m} k={k} n={n} route={}",
                        effective_dense_route().tag()
                    ),
                );
            }
        }
    }
}

#[test]
fn batched_m_equals_per_row_gemv_u8s8() {
    let mut rng = Rng(0xa11c_e0ba_df00_d042);
    for &k in K_SWEEP {
        for &n in N_SWEEP {
            let b = rng.fill_i8(n * k);
            for &m in M_SWEEP {
                let a = rng.fill_u8(m * k);

                let mut batched = vec![0i32; m * n];
                igemm_u8s8(&a, &b, m, k, n, &mut batched);

                for r in 0..m {
                    let row = &a[r * k..(r + 1) * k];
                    let mut single = vec![0i32; n];
                    igemm_u8s8(row, &b, 1, k, n, &mut single);
                    assert_eq!(
                        &batched[r * n..(r + 1) * n],
                        &single[..],
                        "U8S8 batched row {r} != standalone m=1 GEMV (tier {}, m={m} k={k} n={n})",
                        effective_dense_route().tag()
                    );
                }

                let oracle = oracle_i64_u8s8(&a, &b, m, k, n);
                assert_fits_and_matches(
                    &batched,
                    &oracle,
                    &format!(
                        "U8S8 m={m} k={k} n={n} route={}",
                        effective_dense_route().tag()
                    ),
                );
            }
        }
    }
}

/// `+=` semantics survive batching: a non-zero seed in `out` must be added to,
/// not overwritten, identically whether run batched or per-row (the forward
/// driver accumulates bias/residual into the same buffer).
#[test]
fn batched_accumulates_into_seeded_out_s8s8() {
    let mut rng = Rng(0x0ddb_a11c_0ffe_e099);
    let (k, n) = (96usize, 13usize);
    let b = rng.fill_i8(n * k);
    for &m in &[1usize, 4, 7, 33, 128] {
        let a = rng.fill_i8(m * k);
        let seed: Vec<i32> = (0..m * n).map(|i| (i as i32 * 7) - 23).collect();

        let mut batched = seed.clone();
        igemm_s8s8(&a, &b, m, k, n, &mut batched);

        for r in 0..m {
            let row = &a[r * k..(r + 1) * k];
            let mut single = seed[r * n..(r + 1) * n].to_vec();
            igemm_s8s8(row, &b, 1, k, n, &mut single);
            assert_eq!(
                &batched[r * n..(r + 1) * n],
                &single[..],
                "S8S8 seeded batched row {r} != standalone (m={m})"
            );
        }
    }
}

#[test]
fn batched_worst_case_k_i64_oracle_s8s8() {
    let k = 6848usize; // Doctrine #6 worst case (dense layer-0 down_proj).
    let mut rng = Rng(0xdead_beef_cafe_0001);
    for &m in &[64usize, 129] {
        for &n in &[5usize, 65] {
            // Randomized.
            let a = rng.fill_i8(m * k);
            let b = rng.fill_i8(n * k);
            let mut got = vec![0i32; m * n];
            igemm_s8s8(&a, &b, m, k, n, &mut got);
            let oracle = oracle_i64_s8s8(&a, &b, m, k, n);
            assert_fits_and_matches(
                &got,
                &oracle,
                &format!(
                    "S8S8 worst-K random m={m} n={n} tier={}",
                    effective_dense_route().tag()
                ),
            );

            // Adversarial: all-127 and all-(-128) maximize |accumulator|.
            for &fill in &[127i8, -128i8] {
                let a = vec![fill; m * k];
                let b = vec![fill; n * k];
                let mut got = vec![0i32; m * n];
                igemm_s8s8(&a, &b, m, k, n, &mut got);
                let oracle = oracle_i64_s8s8(&a, &b, m, k, n);
                assert_fits_and_matches(
                    &got,
                    &oracle,
                    &format!("S8S8 worst-K all-{fill} m={m} n={n}"),
                );
            }
        }
    }
}

#[test]
fn batched_worst_case_k_i64_oracle_u8s8() {
    let k = 6848usize;
    let mut rng = Rng(0xdead_beef_cafe_0002);
    for &m in &[64usize, 129] {
        for &n in &[5usize, 65] {
            let a = rng.fill_u8(m * k);
            let b = rng.fill_i8(n * k);
            let mut got = vec![0i32; m * n];
            igemm_u8s8(&a, &b, m, k, n, &mut got);
            let oracle = oracle_i64_u8s8(&a, &b, m, k, n);
            assert_fits_and_matches(
                &got,
                &oracle,
                &format!(
                    "U8S8 worst-K random m={m} n={n} tier={}",
                    effective_dense_route().tag()
                ),
            );

            // The binding U8S8 worst case: all-255 activations · all-127 (and
            // all-(-128)) weights = 221.7M / 223.5M at K=6848, < i32::MAX.
            for &fill in &[127i8, -128i8] {
                let a = vec![255u8; m * k];
                let b = vec![fill; n * k];
                let mut got = vec![0i32; m * n];
                igemm_u8s8(&a, &b, m, k, n, &mut got);
                let oracle = oracle_i64_u8s8(&a, &b, m, k, n);
                assert_fits_and_matches(
                    &got,
                    &oracle,
                    &format!("U8S8 worst-K 255x{fill} m={m} n={n}"),
                );
            }
        }
    }
}

/// Self-verification of the per-tier sweep: when `FOCR_FORCE_ARCH` names a tier
/// that is actually present on this host, the dispatched tier MUST become that
/// tier — otherwise the CI sweep would silently test the same tier repeatedly and
/// the "all tiers" claim would be hollow. Unknown/absent forces are ignored by
/// design (you cannot run VNNI on ARM), so this only asserts when the force is
/// host-valid. With no env var set, it just reports the natural tier.
#[test]
fn forced_tier_takes_effect() {
    let tier = detected_tier();
    let route = effective_dense_route();
    eprintln!(
        "[batched_igemm_parity] hardware tier = {}; effective route = {}",
        tier.tag(),
        route.tag()
    );
    let Ok(force) = std::env::var("FOCR_FORCE_ARCH") else {
        return; // natural selection — nothing to assert.
    };
    let want = force.trim().to_ascii_lowercase();
    // Scalar is always present, so a scalar force must always take effect.
    if want == "scalar" {
        assert_eq!(
            tier,
            IsaTier::Scalar,
            "FOCR_FORCE_ARCH=scalar must dispatch the scalar floor"
        );
        assert_eq!(
            route.tag(),
            "scalar",
            "FOCR_FORCE_ARCH=scalar must execute the scalar floor"
        );
        return;
    }
    // For an accelerated force, only assert if that tier is one this host
    // advertises; otherwise the override is correctly a no-op.
    let host_has = franken_ocr::simd::available_tiers()
        .iter()
        .any(|t| t.tag() == want);
    if host_has {
        assert_eq!(
            tier.tag(),
            want,
            "FOCR_FORCE_ARCH={want} is host-available but was not dispatched"
        );
        assert_eq!(
            route.tag(),
            want,
            "FOCR_FORCE_ARCH={want} is available but did not reach that implementation"
        );
    }
}
