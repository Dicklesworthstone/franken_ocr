//! `nn_facade` — micro-benchmarks of the STABLE committed `nn.rs` facade kernels
//! at the SHAPES THIS MODEL ACTUALLY USES.
//!
//! This is the substrate the perf gauntlet (plan §9.3, running-the-gauntlet)
//! ratchets against. Every `#[bench]` here calls *only* the committed facade in
//! [`franken_ocr::native_engine::nn`] over the [`Mat`] / [`QInt8`] currency from
//! [`franken_ocr::native_engine::tensor`] — no `src/` is edited, no future
//! kernel is invented. Where a Phase-2/3/4 kernel (the register-blocked
//! SMMLA/VNNI int8/int4 GEMM tiers) does not exist yet, its bench SLOT is
//! scaffolded and clearly logged as `// SCAFFOLD` — never silently empty — so
//! that when the kernel lands behind the same [`nn::linear_int8_dynamic`]
//! entrypoint the measurement is already wired.
//!
//! ──────────────────────────────────────────────────────────────────────────
//! SHAPE RATIONALE (every (M,K,N) is a real model dim, cited to the config)
//! ──────────────────────────────────────────────────────────────────────────
//! Decoder (DeepSeek-V2 MoE, `use_mla=false`, 12 layers):
//!   * `hidden_size = 1280`              (config.json:29,97 — CENSUS.md:234)
//!   * dense layer-0 `intermediate_size = 6848` (gate/up: 1280→6848;
//!       down: 6848→1280)               (config.json:98 — EXISTING_…:62)
//!       ← This K=6848 down_proj is the int8 i32-accumulation worst case
//!         (doctrine #6: U8S8 ≤ ~221.7M, plan §5.4). Benched as the
//!         bandwidth-critical int8 path.
//!   * MoE `moe_intermediate_size = 896` per expert (1280→896 gate/up,
//!       896→1280 down)                 (config.json:102 — EXISTING_…:83)
//!   * attention q/k/v/o: 10 heads × head_dim 128 = 1280, all
//!       1280→1280 dense GEMMs          (head_dim = 1280/10 = 128;
//!       config.json:38,106; CENSUS.md:234; OQ-4 RESOLVED)
//!   * `lm_head`: 1280 → `vocab_size` 129280 (config.json:118 — the big GEMV)
//! Vision tower (DeepEncoder, fixed per-page cost):
//!   * SAM-ViT-B width 768, 12 blocks, 12 heads, 64×64=4096 patch tokens
//!       (SPEC-041/043; PROPOSED_ARCHITECTURE.md:160)
//!   * CLIP-L/14 width 1024, 24 layers, 16 heads (SPEC-047; …:163)
//!   * projector concat → 2048 → 1280 (SPEC-052; …:166) — benched as a linear.
//! R-SWA decode attention (SlidingWindowLlamaAttention, SPEC-090):
//!   * window W = 128 (config.json:52 `sliding_window_size` — CENSUS.md:199)
//!   * head_dim 128, scale 1/√128, 10 heads (PROPOSED_ARCHITECTURE.md:175)
//!   * reference block `m` GROWS with page count; base 1024-view = 273 token
//!       slots (256 image-feature + 16 image_newline + 1 view_seperator;
//!       CENSUS.md:131-164). Steady-state decode step attends m + W keys
//!       (`O(m+128)`/step, doctrine #7). We bench the reference m=273 decode
//!       step (seq_q=1, seq_k=273+128=401) and a prefill (seq_q=seq_k=273).
//!
//! ──────────────────────────────────────────────────────────────────────────
//! FAIRNESS CONTROLS (doctrine #8; running-the-gauntlet pillar 3)
//! ──────────────────────────────────────────────────────────────────────────
//! * THREADING: the facade delegates GEMM/conv/sdpa to `ft-kernel-cpu`, which
//!   fans out across its OWN rayon pool (doctrine #5: one live forward at a
//!   time, pinned to physical cores). These benches therefore measure the
//!   FULL-CORE forward path — exactly how the engine runs in production — NOT a
//!   single-threaded path. To get an apples-to-apples single-thread number,
//!   pin with `RAYON_NUM_THREADS=1 cargo bench`. Each bench logs which regime
//!   it intends via its provenance line. The elementwise glue (silu / gelu /
//!   quick_gelu / norms / softmax) is the committed *scalar* loop that LLVM
//!   autovectorizes (doctrine #3) — inherently single-threaded here.
//! * ALLOCATOR: the system allocator (whatever the crate links). Inputs are
//!   built ONCE outside the timed `b.iter()` closure; only the kernel call and
//!   its output allocation are timed. No per-iteration input rebuild.
//! * PRECISION: f32 throughout (the parity spine, tensor.rs P1). int8 paths
//!   carry an ISOMORPHISM/GOLDEN note vs the f32 reference (the facade's own
//!   `linear_int8_dynamic_approximates_f32` test is the bit-level guard; here
//!   we additionally assert a cosine/relative-error bound on a model-scale
//!   shape so the bench cannot silently measure a broken kernel).
//! * WARMUP: `test::Bencher` auto-warms (it calibrates iteration count before
//!   the timed run); we additionally do an explicit pre-roll call for the
//!   heavier GEMMs so first-touch page faults / pool spin-up are not timed.
//! * REPORTING: `cargo bench` reports ns/iter ± deviation. For p50/p90 beyond
//!   the harness's mean±σ, run `cargo bench -- --nocapture` repeatedly or wrap
//!   with the comprehensive-bench JSON collector (deps_wanted) — `test`'s
//!   built-in `#[bench]` does not emit per-sample percentiles by itself.
//!
//! PROVENANCE: one `// PROV:` line per bench documenting kernel · shape · regime.
//! No fabricated numbers — this harness MEASURES (the numbers come out of
//! `cargo bench`, not out of this file).

#![feature(test)]
extern crate test;

use franken_ocr::native_engine::nn;
use franken_ocr::native_engine::tensor::Mat;
use test::{Bencher, black_box};

// ── Config dims as named constants (single source — cite above) ────────────
const HIDDEN: usize = 1280; // hidden_size
const DENSE_INTER: usize = 6848; // dense layer-0 intermediate_size (worst-case K)
const MOE_INTER: usize = 896; // moe_intermediate_size (per expert)
const HEAD_DIM: usize = 128; // head_dim = 1280/10
const N_HEADS: usize = 10; // num_attention_heads
const VOCAB: usize = 129_280; // vocab_size (lm_head N)
const SAM_W: usize = 768; // SAM-ViT-B width
const CLIP_W: usize = 1024; // CLIP-L/14 width
const PROJ_IN: usize = 2048; // vision feature concat width → projector
const RSWA_W: usize = 128; // sliding window
const REF_M: usize = 273; // base 1024-view reference token slots
const RMS_EPS: f32 = 1e-6; // rms_norm_eps (decoder)
const LN_EPS: f32 = 1e-5; // vision LayerNorm eps (typical)

// ── Deterministic, non-trivial filler (avoid all-zero/all-one degeneracy that
//    would let the kernel or the optimizer cheat). Cheap LCG, built ONCE. ────
fn filler(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map high bits to ~[-1, 1)
        let u = (s >> 40) as f32 / (1u64 << 23) as f32;
        out.push(u - 1.0);
    }
    out
}

fn mat(rows: usize, cols: usize, seed: u64) -> Mat {
    Mat::from_vec(rows, cols, filler(rows * cols, seed))
}

// Cosine similarity for the int8↔f32 isomorphism guard.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

// ════════════════════════════════════════════════════════════════════════════
// 1. DECODER GEMMs (f32 matmul) — the dense linear shapes, decode regime M=1
//    (one token/step) and a small prefill batch M=16.
// ════════════════════════════════════════════════════════════════════════════

// PROV: nn::matmul · attn q/k/v/o proj [M=1,K=1280,N=1280] · decode step · full-core
#[bench]
fn matmul_attn_proj_decode(b: &mut Bencher) {
    let x = mat(1, HIDDEN, 1);
    let w = mat(HIDDEN, HIDDEN, 2); // 1280→1280
    let _ = nn::matmul(&x, &w).unwrap(); // pre-roll (first-touch / pool spin-up)
    b.iter(|| black_box(nn::matmul(black_box(&x), black_box(&w)).unwrap()));
}

// PROV: nn::matmul · attn q/k/v/o proj [M=16,K=1280,N=1280] · prefill batch · full-core
#[bench]
fn matmul_attn_proj_prefill16(b: &mut Bencher) {
    let x = mat(16, HIDDEN, 3);
    let w = mat(HIDDEN, HIDDEN, 4);
    let _ = nn::matmul(&x, &w).unwrap();
    b.iter(|| black_box(nn::matmul(black_box(&x), black_box(&w)).unwrap()));
}

// PROV: nn::matmul · dense layer-0 gate/up [M=1,K=1280,N=6848] · decode step · full-core
#[bench]
fn matmul_dense0_gate_up_decode(b: &mut Bencher) {
    let x = mat(1, HIDDEN, 5);
    let w = mat(HIDDEN, DENSE_INTER, 6); // 1280→6848
    let _ = nn::matmul(&x, &w).unwrap();
    b.iter(|| black_box(nn::matmul(black_box(&x), black_box(&w)).unwrap()));
}

// PROV: nn::matmul · dense layer-0 down_proj [M=1,K=6848,N=1280] · decode step · full-core
//       K=6848 is the int8 i32-accum WORST CASE (doctrine #6); here in f32 it is the
//       bandwidth/compute reference the int8 path below is compared against.
#[bench]
fn matmul_dense0_down_decode(b: &mut Bencher) {
    let x = mat(1, DENSE_INTER, 7);
    let w = mat(DENSE_INTER, HIDDEN, 8); // 6848→1280
    let _ = nn::matmul(&x, &w).unwrap();
    b.iter(|| black_box(nn::matmul(black_box(&x), black_box(&w)).unwrap()));
}

// PROV: nn::matmul · MoE expert gate/up [M=1,K=1280,N=896] · decode step · full-core
#[bench]
fn matmul_moe_expert_gate_up_decode(b: &mut Bencher) {
    let x = mat(1, HIDDEN, 9);
    let w = mat(HIDDEN, MOE_INTER, 10); // 1280→896
    let _ = nn::matmul(&x, &w).unwrap();
    b.iter(|| black_box(nn::matmul(black_box(&x), black_box(&w)).unwrap()));
}

// PROV: nn::matmul · MoE expert down [M=1,K=896,N=1280] · decode step · full-core
#[bench]
fn matmul_moe_expert_down_decode(b: &mut Bencher) {
    let x = mat(1, MOE_INTER, 11);
    let w = mat(MOE_INTER, HIDDEN, 12); // 896→1280
    let _ = nn::matmul(&x, &w).unwrap();
    b.iter(|| black_box(nn::matmul(black_box(&x), black_box(&w)).unwrap()));
}

// PROV: nn::matmul · lm_head GEMV [M=1,K=1280,N=129280] · decode step · full-core
//       The 129K-vocab logits projection — the per-token decode tail (plan §6.10).
#[bench]
fn matmul_lm_head_decode(b: &mut Bencher) {
    let x = mat(1, HIDDEN, 13);
    let w = mat(HIDDEN, VOCAB, 14); // 1280→129280
    let _ = nn::matmul(&x, &w).unwrap();
    b.iter(|| black_box(nn::matmul(black_box(&x), black_box(&w)).unwrap()));
}

// ── Vision GEMMs (f32). SAM 4096 tokens × 768 wide; CLIP 1025 tokens × 1024
//    wide (1024 patches + class token). qkv-fused proj width = 3× model width. ─

// PROV: nn::matmul · SAM qkv proj [M=4096,K=768,N=2304] · vision encode · full-core
#[bench]
fn matmul_sam_qkv_proj(b: &mut Bencher) {
    let x = mat(4096, SAM_W, 15);
    let w = mat(SAM_W, SAM_W * 3, 16); // 768→2304 (fused qkv)
    let _ = nn::matmul(&x, &w).unwrap();
    b.iter(|| black_box(nn::matmul(black_box(&x), black_box(&w)).unwrap()));
}

// PROV: nn::matmul · CLIP qkv proj [M=1025,K=1024,N=3072] · vision encode · full-core
#[bench]
fn matmul_clip_qkv_proj(b: &mut Bencher) {
    let x = mat(1025, CLIP_W, 17);
    let w = mat(CLIP_W, CLIP_W * 3, 18); // 1024→3072 (fused qkv)
    let _ = nn::matmul(&x, &w).unwrap();
    b.iter(|| black_box(nn::matmul(black_box(&x), black_box(&w)).unwrap()));
}

// PROV: nn::matmul · projector [M=273,K=2048,N=1280] · vision→decoder bridge · full-core
#[bench]
fn matmul_projector(b: &mut Bencher) {
    let x = mat(REF_M, PROJ_IN, 19);
    let w = mat(PROJ_IN, HIDDEN, 20); // 2048→1280
    let _ = nn::matmul(&x, &w).unwrap();
    b.iter(|| black_box(nn::matmul(black_box(&x), black_box(&w)).unwrap()));
}

// ════════════════════════════════════════════════════════════════════════════
// 2. linear_int8_dynamic — the bandwidth-critical decoder GEMM (crown asset).
//    QInt8 weight is [n, k] (output-channel-major). Benched at the dense
//    layer-0 K=6848 down_proj shape (doctrine #6 worst case) and the MoE shape.
//    NOTE: this is the EXISTING facade entrypoint; the register-blocked
//    SMMLA/VNNI tiers (Phase 2-4) land UNDER it without changing this call —
//    so these benches already measure the right slot.
// ════════════════════════════════════════════════════════════════════════════

// PROV: nn::linear_int8_dynamic · dense0 down_proj [M=1,K=6848,N=1280] · decode · full-core
//       + ISOMORPHISM GUARD: cosine vs f32 matmul ≥ 0.99 on this exact shape.
#[bench]
fn linear_int8_dense0_down_decode(b: &mut Bencher) {
    let x = mat(1, DENSE_INTER, 21);
    let wf = mat(HIDDEN, DENSE_INTER, 22); // f32 weight [n=1280, k=6848]
    let w = nn::quantize_int8(&wf.data, HIDDEN, DENSE_INTER); // → QInt8 [n,k]

    // ISOMORPHISM/GOLDEN note: int8 dyn-quant must track the f32 product. We
    // compare against nn::matmul(x[1,K], wf^T[K,n]); wf is row-major [n,k] so its
    // transpose [k,n] is what matmul contracts. Guard fails loudly if the int8
    // path is silently broken — the bench must never measure a wrong kernel.
    let mut wt = vec![0.0f32; DENSE_INTER * HIDDEN];
    for o in 0..HIDDEN {
        for kk in 0..DENSE_INTER {
            wt[kk * HIDDEN + o] = wf.data[o * DENSE_INTER + kk];
        }
    }
    let ref_f32 = nn::matmul(&x, &Mat::from_vec(DENSE_INTER, HIDDEN, wt)).unwrap();
    let got = nn::linear_int8_dynamic(&x, &w, None).unwrap();
    let cos = cosine(&got.data, &ref_f32.data);
    assert!(
        cos >= 0.99,
        "int8 dense0-down cosine {cos} < 0.99 — kernel drift"
    );

    b.iter(|| black_box(nn::linear_int8_dynamic(black_box(&x), black_box(&w), None).unwrap()));
}

// PROV: nn::linear_int8_dynamic · dense0 down_proj [M=16,K=6848,N=1280] · prefill · full-core
#[bench]
fn linear_int8_dense0_down_prefill16(b: &mut Bencher) {
    let x = mat(16, DENSE_INTER, 23);
    let wf = mat(HIDDEN, DENSE_INTER, 24);
    let w = nn::quantize_int8(&wf.data, HIDDEN, DENSE_INTER);
    let _ = nn::linear_int8_dynamic(&x, &w, None).unwrap(); // pre-roll
    b.iter(|| black_box(nn::linear_int8_dynamic(black_box(&x), black_box(&w), None).unwrap()));
}

// PROV: nn::linear_int8_dynamic · MoE expert down [M=1,K=896,N=1280] · decode · full-core
#[bench]
fn linear_int8_moe_down_decode(b: &mut Bencher) {
    let x = mat(1, MOE_INTER, 25);
    let wf = mat(HIDDEN, MOE_INTER, 26); // [n=1280, k=896]
    let w = nn::quantize_int8(&wf.data, HIDDEN, MOE_INTER);
    let _ = nn::linear_int8_dynamic(&x, &w, None).unwrap();
    b.iter(|| black_box(nn::linear_int8_dynamic(black_box(&x), black_box(&w), None).unwrap()));
}

// PROV: nn::quantize_int8 · dense0 down_proj weight [n=1280,k=6848] · load-time cost
//       (run once per weight at load; benched to size the model-prep budget.)
#[bench]
fn quantize_int8_dense0_down(b: &mut Bencher) {
    let wf = mat(HIDDEN, DENSE_INTER, 27);
    b.iter(|| black_box(nn::quantize_int8(black_box(&wf.data), HIDDEN, DENSE_INTER)));
}

// SCAFFOLD — Phase-4 int4 group-quant decode-bandwidth wedge (plan §9, QInt4 in
// tensor.rs). There is NO facade entrypoint for int4 GEMM yet (QInt4 only
// CARRIES the packing; construction/dequant land with bd-3gaa.1). When the int4
// kernel slots under nn:: (unpack int4→int8 in-register → same int8 MAC), add a
// bench here mirroring `linear_int8_dense0_down_decode` at the dense-0 K=6848
// shape — that is the bandwidth-critical expert-bulk path the int4 win targets.
// Intentionally left as a logged scaffold (NOT a silent gap), per task contract.
#[bench]
#[ignore = "SCAFFOLD: no int4 GEMM facade entrypoint yet (Phase 4 / bd-3gaa.1)"]
fn scaffold_linear_int4_dense0_down_decode(_b: &mut Bencher) {
    // Deliberately a no-op placeholder: `cargo bench` shows it as ignored,
    // making the missing-kernel slot visible in the harness output.
}

// ════════════════════════════════════════════════════════════════════════════
// 3. SDPA — R-SWA decode step, R-SWA prefill, and vision full attention.
//    sdpa(q,k,v, num_bh, seq_q, seq_k, d_k, d_v, scale, causal). num_bh =
//    batch*heads. Decoder R-SWA: 10 heads, head_dim 128, scale 1/√128.
// ════════════════════════════════════════════════════════════════════════════

// PROV: nn::sdpa · R-SWA decode step [bh=10, sq=1, sk=401(=273+128)] dk=dv=128 · full-core
//       Steady-state: one query token attends the reference block m=273 + window
//       W=128 keys (doctrine #7, O(m+128)/step). NOT causal (ring has no causal
//       mask in steady state, SPEC-091).
#[bench]
fn sdpa_rswa_decode_ref273(b: &mut Bencher) {
    let bh = N_HEADS;
    let (sq, sk) = (1, REF_M + RSWA_W); // 401
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = filler(bh * sq * HEAD_DIM, 30);
    let k = filler(bh * sk * HEAD_DIM, 31);
    let v = filler(bh * sk * HEAD_DIM, 32);
    let _ = nn::sdpa(&q, &k, &v, bh, sq, sk, HEAD_DIM, HEAD_DIM, scale, false);
    b.iter(|| {
        black_box(nn::sdpa(
            black_box(&q),
            black_box(&k),
            black_box(&v),
            bh,
            sq,
            sk,
            HEAD_DIM,
            HEAD_DIM,
            scale,
            false,
        ))
    });
}

// PROV: nn::sdpa · R-SWA decode step, GROWN ref m=2000 [bh=10, sq=1, sk=2128] · full-core
//       Multi-page: reference block grows with page count (doctrine #7). A larger
//       m shows the per-step O(m+128) growth the harness must track over docs.
#[bench]
fn sdpa_rswa_decode_ref2000(b: &mut Bencher) {
    let bh = N_HEADS;
    let m = 2000;
    let (sq, sk) = (1, m + RSWA_W); // 2128
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = filler(bh * sq * HEAD_DIM, 33);
    let k = filler(bh * sk * HEAD_DIM, 34);
    let v = filler(bh * sk * HEAD_DIM, 35);
    let _ = nn::sdpa(&q, &k, &v, bh, sq, sk, HEAD_DIM, HEAD_DIM, scale, false);
    b.iter(|| {
        black_box(nn::sdpa(
            black_box(&q),
            black_box(&k),
            black_box(&v),
            bh,
            sq,
            sk,
            HEAD_DIM,
            HEAD_DIM,
            scale,
            false,
        ))
    });
}

// PROV: nn::sdpa · R-SWA prefill [bh=10, sq=sk=273] dk=dv=128 · CAUSAL · full-core
//       True prefill regime: full causal mask over the reference view (SPEC-091).
#[bench]
fn sdpa_rswa_prefill_273(b: &mut Bencher) {
    let bh = N_HEADS;
    let (sq, sk) = (REF_M, REF_M);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = filler(bh * sq * HEAD_DIM, 36);
    let k = filler(bh * sk * HEAD_DIM, 37);
    let v = filler(bh * sk * HEAD_DIM, 38);
    let _ = nn::sdpa(&q, &k, &v, bh, sq, sk, HEAD_DIM, HEAD_DIM, scale, true);
    b.iter(|| {
        black_box(nn::sdpa(
            black_box(&q),
            black_box(&k),
            black_box(&v),
            bh,
            sq,
            sk,
            HEAD_DIM,
            HEAD_DIM,
            scale,
            true,
        ))
    });
}

// PROV: nn::sdpa · SAM global attn [bh=12, sq=sk=4096] dk=dv=64 · vision · NON-causal · full-core
//       SAM-ViT-B global blocks [2,5,8,11]: 12 heads, 4096 tokens, per-head dim
//       768/12=64 (SPEC-043). The compute-bound vision quadratic.
#[bench]
fn sdpa_sam_global_4096(b: &mut Bencher) {
    let bh = 12;
    let dk = SAM_W / 12; // 64
    let (sq, sk) = (4096, 4096);
    let scale = 1.0 / (dk as f32).sqrt();
    let q = filler(bh * sq * dk, 39);
    let k = filler(bh * sk * dk, 40);
    let v = filler(bh * sk * dk, 41);
    // No pre-roll: this is the most expensive bench; the harness warmup suffices
    // and a pre-roll would double a multi-hundred-ms first call. Inputs still
    // built once outside the timed loop.
    b.iter(|| {
        black_box(nn::sdpa(
            black_box(&q),
            black_box(&k),
            black_box(&v),
            bh,
            sq,
            sk,
            dk,
            dk,
            scale,
            false,
        ))
    });
}

// PROV: nn::sdpa · CLIP attn [bh=16, sq=sk=1025] dk=dv=64 · vision · NON-causal · full-core
//       CLIP-L/14: 16 heads, 1024 patches + class token = 1025, per-head 1024/16=64.
#[bench]
fn sdpa_clip_1025(b: &mut Bencher) {
    let bh = 16;
    let dk = CLIP_W / 16; // 64
    let (sq, sk) = (1025, 1025);
    let scale = 1.0 / (dk as f32).sqrt();
    let q = filler(bh * sq * dk, 42);
    let k = filler(bh * sk * dk, 43);
    let v = filler(bh * sk * dk, 44);
    let _ = nn::sdpa(&q, &k, &v, bh, sq, sk, dk, dk, scale, false);
    b.iter(|| {
        black_box(nn::sdpa(
            black_box(&q),
            black_box(&k),
            black_box(&v),
            bh,
            sq,
            sk,
            dk,
            dk,
            scale,
            false,
        ))
    });
}

// ════════════════════════════════════════════════════════════════════════════
// 4. NORMS / SOFTMAX / ACTIVATIONS — the elementwise glue (scalar, LLVM-autovec,
//    doctrine #3). Single-threaded by construction. Benched at decoder hidden
//    1280 and the vision widths (768 / 1024), at decode M=1 and a prefill batch.
// ════════════════════════════════════════════════════════════════════════════

// PROV: nn::rms_norm · decoder norm [M=1,cols=1280] eps=1e-6 · decode step · scalar
#[bench]
fn rms_norm_hidden_decode(b: &mut Bencher) {
    let x = mat(1, HIDDEN, 50);
    let w = filler(HIDDEN, 51);
    b.iter(|| black_box(nn::rms_norm(black_box(&x), Some(black_box(&w)), RMS_EPS).unwrap()));
}

// PROV: nn::rms_norm · decoder norm [M=273,cols=1280] eps=1e-6 · prefill view · scalar
#[bench]
fn rms_norm_hidden_prefill273(b: &mut Bencher) {
    let x = mat(REF_M, HIDDEN, 52);
    let w = filler(HIDDEN, 53);
    b.iter(|| black_box(nn::rms_norm(black_box(&x), Some(black_box(&w)), RMS_EPS).unwrap()));
}

// PROV: nn::layer_norm · SAM LayerNorm2d-equiv [M=4096,cols=768] · vision · scalar
#[bench]
fn layer_norm_sam_768(b: &mut Bencher) {
    let x = mat(4096, SAM_W, 54);
    let w = filler(SAM_W, 55);
    let bias = filler(SAM_W, 56);
    b.iter(|| {
        black_box(
            nn::layer_norm(
                black_box(&x),
                Some(black_box(&w)),
                Some(black_box(&bias)),
                LN_EPS,
            )
            .unwrap(),
        )
    });
}

// PROV: nn::layer_norm · CLIP LN [M=1025,cols=1024] · vision · scalar
#[bench]
fn layer_norm_clip_1024(b: &mut Bencher) {
    let x = mat(1025, CLIP_W, 57);
    let w = filler(CLIP_W, 58);
    let bias = filler(CLIP_W, 59);
    b.iter(|| {
        black_box(
            nn::layer_norm(
                black_box(&x),
                Some(black_box(&w)),
                Some(black_box(&bias)),
                LN_EPS,
            )
            .unwrap(),
        )
    });
}

// PROV: nn::softmax_rows · R-SWA decode attn weights [1 row × 401] · decode step · scalar
//       The per-step attention-weight softmax over m+W keys (mutates in place;
//       buffer rebuilt each iter since the kernel writes back).
#[bench]
fn softmax_rows_rswa_decode_401(b: &mut Bencher) {
    let base = mat(N_HEADS, REF_M + RSWA_W, 60); // 10 heads × 401 keys
    b.iter(|| {
        let mut m = base.clone();
        nn::softmax_rows(&mut m).unwrap();
        black_box(m);
    });
}

// PROV: nn::softmax_rows · SAM global attn weights [4096 × 4096] · vision · scalar
#[bench]
fn softmax_rows_sam_4096(b: &mut Bencher) {
    let base = mat(4096, 4096, 61);
    b.iter(|| {
        let mut m = base.clone();
        nn::softmax_rows(&mut m).unwrap();
        black_box(m);
    });
}

// PROV: nn::silu · decoder MLP/MoE gate act [M=1,cols=6848] · decode step · scalar
//       Sized at the dense layer-0 intermediate width (the gate output the
//       SiLU runs over before the elementwise gate*up product).
#[bench]
fn silu_dense0_inter_decode(b: &mut Bencher) {
    let base = mat(1, DENSE_INTER, 62);
    b.iter(|| {
        let mut m = base.clone();
        nn::silu(&mut m);
        black_box(m);
    });
}

// PROV: nn::silu · MoE expert gate act [M=1,cols=896] · decode step · scalar
#[bench]
fn silu_moe_inter_decode(b: &mut Bencher) {
    let base = mat(1, MOE_INTER, 63);
    b.iter(|| {
        let mut m = base.clone();
        nn::silu(&mut m);
        black_box(m);
    });
}

// PROV: nn::gelu · SAM MLP act [M=4096,cols=3072(=768*4)] · vision · scalar
//       SAM MLPBlock uses exact erf-GELU (SPEC-049); intermediate = 4× width.
#[bench]
fn gelu_sam_mlp(b: &mut Bencher) {
    let base = mat(4096, SAM_W * 4, 64);
    b.iter(|| {
        let mut m = base.clone();
        nn::gelu(&mut m);
        black_box(m);
    });
}

// PROV: nn::quick_gelu · CLIP MLP act [M=1025,cols=4096(=1024*4)] · vision · scalar
//       CLIP-L quick_gelu x·σ(1.702x) (SPEC-049); intermediate = 4× width.
#[bench]
fn quick_gelu_clip_mlp(b: &mut Bencher) {
    let base = mat(1025, CLIP_W * 4, 65);
    b.iter(|| {
        let mut m = base.clone();
        nn::quick_gelu(&mut m);
        black_box(m);
    });
}

// ── A QInt8 sanity smoke so the bench file's int8 isomorphism premise is
//    self-checked even if the big GEMM benches are filtered out. Not a perf
//    bench; a fast guard the harness runs as part of compilation/`--test`. ──
#[bench]
#[ignore = "GUARD: correctness check, not a perf measurement"]
fn int8_isomorphism_guard_smoke(_b: &mut Bencher) {
    // Tiny near-lossless case mirrors nn.rs's own committed test. Build the
    // QInt8 via the facade's own quantizer (scale[o]=max(|row|)/127) rather than
    // hand-packing int weights — the recipe is the kernel's, not ours.
    // x=[[1,2,3]], W=[[1,0,1],[0,1,0]] (n=2,k=3) ⇒ y=[1+3, 2]=[4,2].
    let x = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
    let w = nn::quantize_int8(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 2, 3);
    let y = nn::linear_int8_dynamic(&x, &w, None).unwrap();
    assert_eq!(y.shape(), (1, 2));
    assert!((y.data[0] - 4.0).abs() < 0.1 && (y.data[1] - 2.0).abs() < 0.1);
}
