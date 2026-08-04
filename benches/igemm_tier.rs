//! `igemm_tier` — A/B micro-benchmark of the committed `src/simd` int8 GEMM
//! across its ARM ISA tiers (SDOT vs SMMLA) at the SHAPES THIS MODEL ACTUALLY
//! USES.
//!
//! This is the kernel-level companion to [`nn_facade`]: where `nn_facade`
//! benches the `nn.rs` facade (which currently delegates to `ft-kernel-cpu`),
//! this benches franken_ocr's OWN [`franken_ocr::simd::dispatch::igemm_s8s8`] —
//! the project-mandated custom int8 kernel (no general ML framework) that the
//! forward will dispatch to once wired.
//!
//! ──────────────────────────────────────────────────────────────────────────
//! THE A/B PROTOCOL (why this file exists)
//! ──────────────────────────────────────────────────────────────────────────
//! On Apple Silicon, `SMMLA`/i8mm issues at HALF the rate of `SDOT`, so its 2x
//! MACs/instruction cancel (measured on M4: 0.994x SDOT MACs/s at the raw-
//! intrinsic level) AND it pays a 2x2 operand repack the dot path skips.
//! [`simd::arm::detect_tier`] reports SDOT capability on macOS, while ordinary
//! dense GEMM defaults to the measured-faster LLVM-autovec route. This bench can
//! force each intrinsic tier and compare ns/iter at the committed-kernel level:
//!
//! ```text
//! FOCR_FORCE_ARCH=sdot  cargo bench --bench igemm_tier -- --nocapture
//! FOCR_FORCE_ARCH=smmla cargo bench --bench igemm_tier -- --nocapture
//! ```
//!
//! Each process reads `FOCR_FORCE_ARCH` ONCE (the tier is cached in a OnceLock),
//! so every `#[bench]` in a given run dispatches to the same forced tier. The
//! `[igemm_tier] EFFECTIVE dense int8 route = ...` line (under `--nocapture`)
//! confirms which implementation was timed and records hardware selection
//! separately. The two forced runs give the committed SDOT-vs-SMMLA ratio.
//!
//! Every tier is BIT-IDENTICAL to the scalar oracle (i32 accumulation is exact),
//! so this measures pure throughput — correctness is held by the `src/simd`
//! unit tests (`run_sdot_s8s8` / `run_smmla_s8s8` vs `scalar::igemm_s8s8`).
//!
//! ──────────────────────────────────────────────────────────────────────────
//! SHAPE RATIONALE (every (M,K,N) is a real decoder dim — cited as in nn_facade)
//! ──────────────────────────────────────────────────────────────────────────
//!   * hidden_size = 1280; dense layer-0 intermediate_size = 6848
//!     (gate/up: 1280->6848, down: 6848->1280); attn q/k/v/o: 1280->1280.
//!   * K=6848 down_proj is the int8 i32-accumulation worst case (doctrine #6).
//!   * M=1 is the decode step (latency-critical GEMV); M=16 a small prefill
//!     batch (where SMMLA's 2x2 tiling would, on paper, help most — yet does
//!     not on M4, which is the point).
//!
//! PROVENANCE: one `// PROV:` line per bench (kernel · shape · regime). This
//! file MEASURES — the numbers come out of `cargo bench`, not out of this file.

#![feature(test)]
extern crate test;

use franken_ocr::simd::dispatch;
use std::sync::Once;
use test::{Bencher, black_box};

// ── Real config dims (cite: config.json / CENSUS.md; same as nn_facade) ────────
const HIDDEN: usize = 1280; // hidden_size
const DENSE_INTER: usize = 6848; // dense layer-0 intermediate_size (worst-case K)

// One-time report of the effective route (visible under `--nocapture`), so the
// A/B log is self-documenting about which implementation each run timed.
static TIER_ONCE: Once = Once::new();
fn report_tier() {
    TIER_ONCE.call_once(|| {
        eprintln!(
            "[igemm_tier] EFFECTIVE dense int8 route = {} ({})  hardware = {} ({}) \
             [override via FOCR_FORCE_ARCH]",
            dispatch::effective_dense_route().tag(),
            dispatch::effective_dense_route().feature_string(),
            dispatch::detected_tier().tag(),
            dispatch::tier_string(),
        );
    });
}

// Deterministic i8 filler in [-127, 127] (the symmetric per-OC quant range;
// excludes -128 to mirror the real clamp). Cheap LCG, built ONCE per bench.
fn filler_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // high bits -> [-127, 127]
        let v = ((s >> 40) % 255) as i64 - 127;
        out.push(v as i8);
    }
    out
}

// Run an `igemm_s8s8` bench at (m, k, n): activations a[m*k], weights b[n*k]
// (output-channel-major), output i32[m*n]. The out buffer is re-zeroed each
// iter (the kernel ADDS into it, so reuse without reset would overflow over the
// harness's millions of iters); the fill cost is negligible vs the GEMM.
fn bench_s8s8(b: &mut Bencher, m: usize, k: usize, n: usize) {
    let a = filler_i8(m * k, 0x11);
    let w = filler_i8(n * k, 0x22);
    let mut out = vec![0i32; m * n];
    dispatch::igemm_s8s8(&a, &w, m, k, n, &mut out); // pre-roll (first-touch)
    report_tier();
    b.iter(|| {
        out.fill(0);
        dispatch::igemm_s8s8(black_box(&a), black_box(&w), m, k, n, black_box(&mut out));
        black_box(&out);
    });
}

// ════════════════════════════════════════════════════════════════════════════
// down_proj  (K = 6848 worst case, N = 1280)
// ════════════════════════════════════════════════════════════════════════════

// PROV: simd::igemm_s8s8 · dense layer-0 down_proj [M=1,K=6848,N=1280] · decode step
#[bench]
fn s8s8_down_decode_m1(b: &mut Bencher) {
    bench_s8s8(b, 1, DENSE_INTER, HIDDEN);
}

// PROV: simd::igemm_s8s8 · dense layer-0 down_proj [M=16,K=6848,N=1280] · prefill16
#[bench]
fn s8s8_down_prefill_m16(b: &mut Bencher) {
    bench_s8s8(b, 16, DENSE_INTER, HIDDEN);
}

// ════════════════════════════════════════════════════════════════════════════
// gate/up  (K = 1280, N = 6848)
// ════════════════════════════════════════════════════════════════════════════

// PROV: simd::igemm_s8s8 · dense layer-0 gate/up [M=1,K=1280,N=6848] · decode step
#[bench]
fn s8s8_gate_up_decode_m1(b: &mut Bencher) {
    bench_s8s8(b, 1, HIDDEN, DENSE_INTER);
}

// PROV: simd::igemm_s8s8 · dense layer-0 gate/up [M=16,K=1280,N=6848] · prefill16
#[bench]
fn s8s8_gate_up_prefill_m16(b: &mut Bencher) {
    bench_s8s8(b, 16, HIDDEN, DENSE_INTER);
}

// ════════════════════════════════════════════════════════════════════════════
// attn q/k/v/o proj  (K = 1280, N = 1280)
// ════════════════════════════════════════════════════════════════════════════

// PROV: simd::igemm_s8s8 · attn q/k/v/o proj [M=1,K=1280,N=1280] · decode step
#[bench]
fn s8s8_attn_proj_decode_m1(b: &mut Bencher) {
    bench_s8s8(b, 1, HIDDEN, HIDDEN);
}

// PROV: simd::igemm_s8s8 · attn q/k/v/o proj [M=16,K=1280,N=1280] · prefill16
#[bench]
fn s8s8_attn_proj_prefill_m16(b: &mut Bencher) {
    bench_s8s8(b, 16, HIDDEN, HIDDEN);
}
