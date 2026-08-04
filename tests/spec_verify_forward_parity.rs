//! Batched K-token R-SWA VERIFY-forward parity gate (Lever D/K, bd-1azu.30 — the
//! verify half of speculative decode).
//!
//! The verify forward must compute the `K` next-token logits rows — one per draft
//! position — in ONE forward that is BIT-EXACT to running `K` sequential
//! single-token decode steps. The win is QUERY-dim batching ONLY: the `K` draft
//! *queries* share each layer's projection weight panel (one `M=K` GEMM), while
//! the attention stays a per-position fold over the ring; the keys are NEVER
//! speculatively gathered into a shared block (the rejected key-batch lever,
//! `docs/NEGATIVE_EVIDENCE.md`).
//!
//! This file proves the claim WITHOUT a model. The fixed-config decoder weight
//! cache (12 layers × 64 experts × vocab 129280) is multi-GB, so — exactly as
//! `tests/chunked_prefill_parity.rs` does for the prefill tiling — we exercise the
//! REAL novel kernel directly and rebuild the per-layer forward out of the SAME
//! PUBLIC kernels the driver uses, over small deterministic synthetic weights at
//! the REAL R-SWA head shape (10 heads × 128 = hidden 1280).
//!
//! What is proven, with byte-for-byte (`to_bits`) equality:
//!  * `verify_attention_matches_sequential_and_is_causal` — the bd-1azu.30 kernel
//!    [`rswa::verify_attention`] returns, for every draft position `i`, exactly the
//!    context `decode_attention` produces over reference ++ ring ++ `draft[0..=i]`
//!    (an independent fresh-clone oracle), it is genuinely CAUSAL (query 0 does NOT
//!    see `draft[1..]`), and it does NOT mutate the caller's cache.
//!  * `verify_forward_schedule_matches_sequential_decode` — a layer-major verify
//!    forward built on `verify_attention` equals a token-major sequence of
//!    single-token decode steps built on [`rswa::attention`], for BOTH the f32 and
//!    the int8 projection paths, over several `K` (incl. a non-aligned `K=3`).
//!  * `int8_projection_stacking_is_byte_identical` — the `M=K`-vs-`m=1` projection
//!    batching the int8 verify forward relies on is lossless at the verify shapes.
//!
//! The xorshift RNG / `to_bits` idiom mirrors `tests/chunked_prefill_parity.rs`.

// Parallel stride-array indexing (rings[layer] / x[i] / q_rows[i]) is the natural
// shape of a layer-major forward; `needless_range_loop` is a false positive here,
// matching the same allow in src/native_engine/{decoder,rswa}.rs.
#![allow(clippy::needless_range_loop)]

use franken_ocr::native_engine::decoder::{RopeTable, add_residual, apply_rope, dense_mlp};
use franken_ocr::native_engine::nn;
use franken_ocr::native_engine::rswa::{self, RingCache};
use franken_ocr::native_engine::tensor::{Mat, QInt8};

// REAL R-SWA head shape so the seeded `RingCache` (hardwired 10×128) takes our
// K/V; hidden = heads*head_dim. A tiny dense-MLP intermediate, few layers, short
// reference block, and a partially-filled (no-eviction) ring keep it small/fast.
const NUM_HEADS: usize = rswa::NUM_HEADS; // 10
const HEAD_DIM: usize = rswa::HEAD_DIM; // 128
const QKV_DIM: usize = NUM_HEADS * HEAD_DIM; // 1280
const HIDDEN: usize = QKV_DIM; // 1280
const INTER: usize = 8; // tiny dense-MLP intermediate
const N_LAYERS: usize = 2;
const VOCAB_T: usize = 16; // tiny lm_head substitute (logits-row parity)
const EPS: f32 = 1e-6;
const THETA: f32 = 10000.0;

const P_REF: usize = 5; // reference-block rows (non-empty)
const BASE_RING_WRITES: usize = 3; // prior generated tokens already in the ring
// Draft positions are appended AFTER the base ring; BASE_RING_WRITES + max(K) stays
// well under RING_WINDOW (128), so the verify window holds every draft token (no
// eviction — the regime bd-1azu.30 targets).
const KS: &[usize] = &[1, 3, 4];

/// Deterministic xorshift64 PRNG — reproducible, no dev-dependency (the idiom in
/// `tests/chunked_prefill_parity.rs` / `tests/batched_forward_parity.rs`).
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
    /// An f32 in roughly `[-1, 1)` — small, signed, dense.
    fn f32(&mut self) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        u * 2.0 - 1.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
    /// Norm weights centred near 1.0 so RMSNorm is well-conditioned.
    fn norm(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 1.0 + 0.1 * self.f32()).collect()
    }
}

fn bits(s: &[f32]) -> Vec<u32> {
    s.iter().map(|f| f.to_bits()).collect()
}

/// f32 GEMV `y[o] = Σ_j x[j]·w[o,j]` over a row-major `[n, k]` (out, in) panel —
/// the f32 projection kernel, used identically in BOTH schedules so the only
/// observable difference is the SCHEDULE (layer-major vs token-major).
fn gemv_f32(x: &[f32], w: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|o| {
            w[o * k..o * k + k]
                .iter()
                .zip(x.iter())
                .map(|(&wj, &xj)| wj * xj)
                .sum::<f32>()
        })
        .collect()
}

/// One synthetic decoder layer: f32 attention projections + their per-output
/// symmetric int8 quantizations (so the same layer drives both numeric paths),
/// plus a dense SwiGLU MLP (row-independent, like the real MoE).
struct Layer {
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    q_i8: QInt8,
    k_i8: QInt8,
    v_i8: QInt8,
    o_i8: QInt8,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

impl Layer {
    fn synth(rng: &mut Rng) -> Self {
        let q = rng.fill(QKV_DIM * HIDDEN);
        let k = rng.fill(QKV_DIM * HIDDEN);
        let v = rng.fill(QKV_DIM * HIDDEN);
        let o = rng.fill(HIDDEN * QKV_DIM);
        Layer {
            input_ln: rng.norm(HIDDEN),
            post_attn_ln: rng.norm(HIDDEN),
            q_i8: nn::quantize_int8(&q, QKV_DIM, HIDDEN),
            k_i8: nn::quantize_int8(&k, QKV_DIM, HIDDEN),
            v_i8: nn::quantize_int8(&v, QKV_DIM, HIDDEN),
            o_i8: nn::quantize_int8(&o, HIDDEN, QKV_DIM),
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            gate: rng.fill(INTER * HIDDEN),
            up: rng.fill(INTER * HIDDEN),
            down: rng.fill(HIDDEN * INTER),
        }
    }

    /// Project a single `[k]` activation row to `[n]` with the selected numeric
    /// path — the SAME call in both schedules, so it never injects a difference.
    fn project(&self, use_i8: bool, x: &[f32], which: Which) -> Vec<f32> {
        let (wf, wi, n, k) = match which {
            Which::Q => (&self.q_proj, &self.q_i8, QKV_DIM, HIDDEN),
            Which::K => (&self.k_proj, &self.k_i8, QKV_DIM, HIDDEN),
            Which::V => (&self.v_proj, &self.v_i8, QKV_DIM, HIDDEN),
            Which::O => (&self.o_proj, &self.o_i8, HIDDEN, QKV_DIM),
        };
        if use_i8 {
            let row = Mat::from_vec(1, k, x.to_vec());
            nn::linear_int8_dynamic(&row, wi, None).unwrap().data
        } else {
            gemv_f32(x, wf, n, k)
        }
    }

    /// Post-attention block (RMSNorm → dense SwiGLU → residual), shared by both
    /// schedules.
    fn mlp_block(&self, h: &Mat) -> Mat {
        let normed2 = nn::rms_norm(h, Some(&self.post_attn_ln), EPS).unwrap();
        let mlp = dense_mlp(&normed2, &self.gate, &self.up, &self.down, HIDDEN, INTER).unwrap();
        add_residual(h, &mlp).unwrap()
    }
}

#[derive(Clone, Copy)]
enum Which {
    Q,
    K,
    V,
    O,
}

/// Build the starting cache state per layer: a non-empty reference block plus a
/// few prior generated tokens already in the ring (the partially-filled,
/// no-eviction regime).
fn base_rings(rng: &mut Rng) -> Vec<RingCache> {
    (0..N_LAYERS)
        .map(|_| {
            let mut c = RingCache::new(P_REF);
            let kref = rng.fill(NUM_HEADS * P_REF * HEAD_DIM);
            let vref = rng.fill(NUM_HEADS * P_REF * HEAD_DIM);
            c.record_prefill(&kref, &vref, P_REF).unwrap();
            for _ in 0..BASE_RING_WRITES {
                let ks = rng.fill(NUM_HEADS * HEAD_DIM);
                let vs = rng.fill(NUM_HEADS * HEAD_DIM);
                c.write_decode_step(&ks, &vs).unwrap();
            }
            c
        })
        .collect()
}

/// First absolute position of the draft block (just past the base ring); both
/// schedules RoPE draft token `i` at `base_position() + i`.
fn base_position() -> usize {
    P_REF + BASE_RING_WRITES
}

/// SEQUENTIAL ORACLE: one single-token decode step over a MUTATING ring (writes
/// this token's K/V, then attends) — the literal per-step path the verify forward
/// must reproduce. Uses the public [`rswa::attention`] convenience (write + decode
/// attention) exactly as the real `decode_step_with_cache`.
fn decode_step(layer: &Layer, use_i8: bool, x: &Mat, ring: &mut RingCache, pos: usize) -> Mat {
    let normed = nn::rms_norm(x, Some(&layer.input_ln), EPS).unwrap();
    let nrow = normed.row(0);
    let mut q = Mat::from_vec(1, QKV_DIM, layer.project(use_i8, nrow, Which::Q));
    let mut k = Mat::from_vec(1, QKV_DIM, layer.project(use_i8, nrow, Which::K));
    let v = Mat::from_vec(1, QKV_DIM, layer.project(use_i8, nrow, Which::V));
    let rope = RopeTable::build(&[pos], HEAD_DIM, THETA);
    apply_rope(&mut q, &rope).unwrap();
    apply_rope(&mut k, &rope).unwrap();
    let ctx = rswa::attention(ring, &q, &k, &v, &[pos]).unwrap();
    let attn_out = Mat::from_vec(1, HIDDEN, layer.project(use_i8, ctx.row(0), Which::O));
    let h = add_residual(x, &attn_out).unwrap();
    layer.mlp_block(&h)
}

/// Token-major sequential decode of all `K` draft tokens through the layer stack
/// over a fresh clone of `rings` (so each draft token's K/V is visible to the
/// next, exactly as production decode). Returns the `K` final hidden rows.
fn run_sequential(layers: &[Layer], rings: &[RingCache], embeds: &[Mat], use_i8: bool) -> Vec<Mat> {
    let mut rings = rings.to_vec();
    let base = base_position();
    embeds
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut x = e.clone();
            for layer in 0..N_LAYERS {
                x = decode_step(&layers[layer], use_i8, &x, &mut rings[layer], base + i);
            }
            x
        })
        .collect()
}

/// Layer-major batched VERIFY forward: per layer, project all `K` draft queries,
/// then [`rswa::verify_attention`] over the READ-ONLY `rings` (causal among the
/// draft, no mutation), then the per-position MLP. The structural twin of
/// `decoder::verify_forward` over the public kernels.
fn run_verify(layers: &[Layer], rings: &[RingCache], embeds: &[Mat], use_i8: bool) -> Vec<Mat> {
    let k = embeds.len();
    let base = base_position();
    let mut x: Vec<Mat> = embeds.to_vec();
    for layer in 0..N_LAYERS {
        let l = &layers[layer];
        // 1. Per-position RMSNorm + q/k/v projection + RoPE at the true position.
        let mut q_rows: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut k_rows: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut v_rows: Vec<Vec<f32>> = Vec::with_capacity(k);
        for i in 0..k {
            let normed = nn::rms_norm(&x[i], Some(&l.input_ln), EPS).unwrap();
            let nrow = normed.row(0);
            let mut q = Mat::from_vec(1, QKV_DIM, l.project(use_i8, nrow, Which::Q));
            let mut kk = Mat::from_vec(1, QKV_DIM, l.project(use_i8, nrow, Which::K));
            let v = l.project(use_i8, nrow, Which::V);
            let rope = RopeTable::build(&[base + i], HEAD_DIM, THETA);
            apply_rope(&mut q, &rope).unwrap();
            apply_rope(&mut kk, &rope).unwrap();
            q_rows.push(q.data);
            k_rows.push(kk.data);
            v_rows.push(v);
        }
        // 2. Query-batched verify attention (no mutation of `rings`).
        let q_refs: Vec<&[f32]> = q_rows.iter().map(|r| r.as_slice()).collect();
        let k_refs: Vec<&[f32]> = k_rows.iter().map(|r| r.as_slice()).collect();
        let v_refs: Vec<&[f32]> = v_rows.iter().map(|r| r.as_slice()).collect();
        let contexts = rswa::verify_attention(&rings[layer], &q_refs, &k_refs, &v_refs).unwrap();
        // 3. Per-position o_proj, residual, MLP, residual.
        for i in 0..k {
            let o = l.project(use_i8, contexts[i].row(0), Which::O);
            let attn_out = Mat::from_vec(1, HIDDEN, o);
            let h = add_residual(&x[i], &attn_out).unwrap();
            x[i] = l.mlp_block(&h);
        }
    }
    x
}

/// Final RMSNorm + tiny lm_head over one decode hidden — the per-row, M-invariant
/// logits projection (equal hiddens ⇒ equal logits, so this only makes the
/// "logits-row" parity literal).
fn logits_of(h: &Mat, final_norm: &[f32], lm_head: &[f32]) -> Vec<f32> {
    let normed = nn::rms_norm(h, Some(final_norm), EPS).unwrap();
    gemv_f32(normed.row(0), lm_head, VOCAB_T, HIDDEN)
}

/// CORE GATE: [`rswa::verify_attention`] reproduces, byte-for-byte, the `K`
/// sequential `decode_attention` contexts over reference ++ ring ++ `draft[0..=i]`;
/// it is genuinely causal among the draft; and it leaves the caller's cache
/// unmutated.
#[test]
fn verify_attention_matches_sequential_and_is_causal() {
    for &k in KS {
        let mut rng = Rng(0x51EC_0030_1A2B_3C4D ^ (k as u64).wrapping_mul(0x9E37_79B9));
        // A single layer's starting cache (reference block + partial ring).
        let cache = base_rings(&mut rng).pop().unwrap();
        // K draft tokens' already-RoPE'd-equivalent q/k/v rows (opaque to attention).
        let qd: Vec<Vec<f32>> = (0..k).map(|_| rng.fill(QKV_DIM)).collect();
        let kd: Vec<Vec<f32>> = (0..k).map(|_| rng.fill(QKV_DIM)).collect();
        let vd: Vec<Vec<f32>> = (0..k).map(|_| rng.fill(QKV_DIM)).collect();
        let q_refs: Vec<&[f32]> = qd.iter().map(|r| r.as_slice()).collect();
        let k_refs: Vec<&[f32]> = kd.iter().map(|r| r.as_slice()).collect();
        let v_refs: Vec<&[f32]> = vd.iter().map(|r| r.as_slice()).collect();

        // No-mutation snapshot: a probe query over the full reference ++ ring, plus
        // the reference blocks and ring cursors.
        let probe = rng.fill(QKV_DIM);
        let probe_before = rswa::decode_attention(&cache, &probe).unwrap();
        let ref_before: Vec<Vec<u32>> = (0..NUM_HEADS)
            .flat_map(|h| [bits(cache.reference_k(h)), bits(cache.reference_v(h))])
            .collect();
        let state_before = (
            cache.prefill_len(),
            cache.ring_len(),
            cache.ring_pos(),
            cache.effective_len(),
        );

        let got = rswa::verify_attention(&cache, &q_refs, &k_refs, &v_refs).unwrap();
        assert_eq!(
            got.len(),
            k,
            "K={k}: verify_attention returned {} rows",
            got.len()
        );

        // Independent oracle: for each i, a FRESH clone with ONLY draft[0..=i]
        // written, then decode_attention(q_i). Reconstructs the causal key set per
        // position without reusing verify_attention's incremental write loop.
        for i in 0..k {
            let mut work = cache.clone();
            for j in 0..=i {
                work.write_decode_step(&kd[j], &vd[j]).unwrap();
            }
            let oracle = rswa::decode_attention(&work, &qd[i]).unwrap();
            assert_eq!(
                bits(&got[i].data),
                bits(&oracle.data),
                "K={k} pos {i}: verify_attention context != sequential decode_attention",
            );
        }

        // Causal TEETH: query 0 must NOT see draft[1..]. Writing ALL K draft tokens
        // before attending with q_0 changes the context (random, non-degenerate
        // K/V), so the batched verify — which folds only draft[0] for query 0 —
        // must DIFFER from that "saw the future" result.
        if k > 1 {
            let mut all = cache.clone();
            for j in 0..k {
                all.write_decode_step(&kd[j], &vd[j]).unwrap();
            }
            let saw_future = rswa::decode_attention(&all, &qd[0]).unwrap();
            assert_ne!(
                bits(&got[0].data),
                bits(&saw_future.data),
                "K={k}: query 0 leaked future draft keys (verify_attention is not causal)",
            );
        }

        // No mutation: probe, reference blocks, and cursors are byte-for-byte intact.
        let probe_after = rswa::decode_attention(&cache, &probe).unwrap();
        assert_eq!(
            bits(&probe_before.data),
            bits(&probe_after.data),
            "K={k}: verify_attention mutated the cache (probe attention changed)",
        );
        let ref_after: Vec<Vec<u32>> = (0..NUM_HEADS)
            .flat_map(|h| [bits(cache.reference_k(h)), bits(cache.reference_v(h))])
            .collect();
        assert_eq!(ref_before, ref_after, "K={k}: reference block was mutated");
        let state_after = (
            cache.prefill_len(),
            cache.ring_len(),
            cache.ring_pos(),
            cache.effective_len(),
        );
        assert_eq!(
            state_before, state_after,
            "K={k}: ring cursors were mutated"
        );
    }
}

/// END-TO-END GATE: the layer-major batched verify forward equals the token-major
/// sequential decode, in BOTH the final hidden rows AND the lm_head logits rows,
/// byte-for-byte, for both numeric paths and several `K` (incl. non-aligned 3).
#[test]
fn verify_forward_schedule_matches_sequential_decode() {
    for &use_i8 in &[false, true] {
        for &k in KS {
            let mut rng = Rng(0x0B5E_55ED_4F0C_2230 ^ (k as u64).wrapping_mul(0xD1B5_4A32));
            let layers: Vec<Layer> = (0..N_LAYERS).map(|_| Layer::synth(&mut rng)).collect();
            let rings = base_rings(&mut rng);
            let embeds: Vec<Mat> = (0..k)
                .map(|_| Mat::from_vec(1, HIDDEN, rng.fill(HIDDEN)))
                .collect();
            let final_norm = rng.norm(HIDDEN);
            let lm_head = rng.fill(VOCAB_T * HIDDEN);

            let seq = run_sequential(&layers, &rings, &embeds, use_i8);
            let ver = run_verify(&layers, &rings, &embeds, use_i8);
            assert_eq!(seq.len(), k);
            assert_eq!(ver.len(), k);

            for i in 0..k {
                assert_eq!(
                    bits(&ver[i].data),
                    bits(&seq[i].data),
                    "i8={use_i8} K={k} pos {i}: verify hidden != sequential decode hidden",
                );
                let lo_seq = logits_of(&seq[i], &final_norm, &lm_head);
                let lo_ver = logits_of(&ver[i], &final_norm, &lm_head);
                assert_eq!(
                    bits(&lo_ver),
                    bits(&lo_seq),
                    "i8={use_i8} K={k} pos {i}: verify logits row != sequential decode logits row",
                );
            }
        }
    }
}

/// INT8 PROJECTION GATE: the `M=K`-vs-`m=1` projection batching the int8 verify
/// forward relies on (one shared weight panel reused across all `K` draft queries)
/// is byte-for-byte lossless, at the real qkv-fused / qkv / o_proj shapes —
/// affirming, via the public int8 linear, the property `decoder::gemv_i8_batched`
/// provides (bd-1azu.2).
#[test]
fn int8_projection_stacking_is_byte_identical() {
    // (label, n=out, k=in): the fused q/k/v stack, a single q/k/v, and o_proj.
    let shapes: [(&str, usize, usize); 3] = [
        ("qkv_fused", 3 * QKV_DIM, HIDDEN),
        ("q_proj", QKV_DIM, HIDDEN),
        ("o_proj", HIDDEN, QKV_DIM),
    ];
    let mut rng = Rng(0xA11C_E0BA_DF00_D042);
    for (label, n, kdim) in shapes {
        let wf = rng.fill(n * kdim);
        let qw = nn::quantize_int8(&wf, n, kdim);
        for &k in &[2usize, 3, 4] {
            let rows: Vec<Vec<f32>> = (0..k).map(|_| rng.fill(kdim)).collect();
            // Stacked M=K GEMM.
            let mut stacked_in = Vec::with_capacity(k * kdim);
            for r in &rows {
                stacked_in.extend_from_slice(r);
            }
            let xin = Mat::from_vec(k, kdim, stacked_in);
            let stacked = nn::linear_int8_dynamic(&xin, &qw, None).unwrap();
            // Each row must equal the standalone m=1 GEMV.
            for (r, row) in rows.iter().enumerate() {
                let rin = Mat::from_vec(1, kdim, row.clone());
                let single = nn::linear_int8_dynamic(&rin, &qw, None).unwrap();
                assert_eq!(
                    bits(&stacked.data[r * n..(r + 1) * n]),
                    bits(&single.data),
                    "{label} K={k}: stacked row {r} != standalone m=1 int8 projection",
                );
            }
        }
    }
}
