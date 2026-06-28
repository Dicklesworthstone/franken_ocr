//! Chunked-prefill lossless parity gate (bd-1azu.9 — the chunked prefill
//! co-scheduled into the decode batch).
//!
//! `FOCR_PREFILL_CHUNK` tiles the 12-layer R-SWA prefill into `C`-token chunks:
//! each chunk is pushed through ALL layers (growing the per-layer reference K/V)
//! before the next, so a prefill slice is processed like a decode step and can
//! later share the batched per-layer GEMMs with in-flight decode rows. The claim
//! is that this is a pure TILING of the SAME causal attention — byte-for-byte the
//! monolithic prefill in BOTH the final hidden states AND every layer's ring K/V.
//!
//! This file proves that claim WITHOUT a model. The fixed-config decoder weight
//! cache is multi-GB, so instead of driving the real `prefill_with_cache`, we
//! rebuild its EXACT per-layer prefill loop — monolithic vs chunked — out of the
//! same PUBLIC kernels the driver uses (`qkv_with_rope`, `prefill_attention` /
//! `chunk_prefill_attention`, `attn_output_proj`, `dense_mlp`, `add_residual`,
//! `nn::rms_norm`), over small deterministic synthetic weights at the REAL R-SWA
//! head shape (10 heads × 128 = hidden 1280), and assert byte-for-byte equality —
//! including seeding actual `RingCache`s and comparing their reference K/V via the
//! public `reference_k` / `reference_v` accessors. Chunk sizes cover {1, 2, prime,
//! full}. The xorshift RNG / `to_bits` idiom mirrors `tests/batched_forward_parity.rs`.

use franken_ocr::native_engine::decoder::{
    self, LayerWeights, RopeTable, add_residual, attn_output_proj, chunk_prefill_attention,
    dense_mlp, prefill_attention, qkv_with_rope,
};
use franken_ocr::native_engine::nn;
use franken_ocr::native_engine::rswa::{self, RingCache};
use franken_ocr::native_engine::tensor::Mat;

// The REAL R-SWA head shape so the seeded `RingCache` (hardwired to 10×128) takes
// our K/V; hidden = heads*head_dim. A deliberately tiny dense-MLP intermediate +
// few layers + short sequence keep the synthetic weights small and the test fast.
const NUM_HEADS: usize = rswa::NUM_HEADS; // 10
const HEAD_DIM: usize = rswa::HEAD_DIM; // 128
const QKV_DIM: usize = NUM_HEADS * HEAD_DIM; // 1280
const HIDDEN: usize = QKV_DIM; // 1280
const INTER: usize = 8; // tiny dense-MLP intermediate
const N_LAYERS: usize = 2;
const EPS: f32 = 1e-6;
const THETA: f32 = 10000.0;

/// Deterministic xorshift64 PRNG — reproducible, no dev-dependency (the idiom in
/// `tests/batched_forward_parity.rs` / `tests/batched_igemm_parity.rs`).
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
    /// An f32 in roughly `[-1, 1)` — small, signed, dense (exercises RMSNorm /
    /// projections / RoPE / softmax without overflow).
    fn f32(&mut self) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        u * 2.0 - 1.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

/// One synthetic decoder layer's f32 weights (dense MLP for all layers — the
/// chunking lossless property holds for any ROW-INDEPENDENT MLP, and dense is the
/// public one; the real MoE is per-token-row too).
struct Layer {
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

impl Layer {
    fn synth(rng: &mut Rng) -> Self {
        Layer {
            // Norm weights centred near 1.0 so RMSNorm is well-conditioned.
            input_ln: (0..HIDDEN).map(|_| 1.0 + 0.1 * rng.f32()).collect(),
            post_attn_ln: (0..HIDDEN).map(|_| 1.0 + 0.1 * rng.f32()).collect(),
            q_proj: rng.fill(QKV_DIM * HIDDEN),
            k_proj: rng.fill(QKV_DIM * HIDDEN),
            v_proj: rng.fill(QKV_DIM * HIDDEN),
            o_proj: rng.fill(HIDDEN * QKV_DIM),
            gate: rng.fill(INTER * HIDDEN),
            up: rng.fill(INTER * HIDDEN),
            down: rng.fill(HIDDEN * INTER),
        }
    }
    fn weights(&self) -> LayerWeights<'_> {
        LayerWeights {
            input_ln: &self.input_ln,
            post_attn_ln: &self.post_attn_ln,
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            o_proj: &self.o_proj,
            // qkv_with_rope only reads q/k/v_proj; the dense MLP runs via dense_mlp.
            gate_w: &[],
            up_w: &[],
            down_w: &[],
        }
    }
}

/// The shared per-layer post-attention block (RMSNorm → dense SwiGLU → residual),
/// identical between the monolithic and chunked schedules.
fn mlp_block(layer: &Layer, h: &Mat) -> Mat {
    let normed2 = nn::rms_norm(h, Some(&layer.post_attn_ln), EPS).unwrap();
    let mlp = dense_mlp(&normed2, &layer.gate, &layer.up, &layer.down, HIDDEN, INTER).unwrap();
    add_residual(h, &mlp).unwrap()
}

/// Captured prefill output: the final hidden plus each layer's token-major K/V
/// (exactly the bytes `record_prefill` ingests into the ring).
struct Prefill {
    hidden: Mat,
    k: Vec<Vec<f32>>, // per layer, token-major [seq, QKV_DIM]
    v: Vec<Vec<f32>>,
}

/// Monolithic schedule — one SDPA over the whole sequence per layer (the
/// `prefill_with_cache` default path, rebuilt over the public kernels).
fn run_monolithic(layers: &[Layer], embeds: &Mat) -> Prefill {
    let seq = embeds.rows;
    let rope = RopeTable::build(&(0..seq).collect::<Vec<_>>(), HEAD_DIM, THETA);
    let mut x = embeds.clone();
    let mut k_all = Vec::with_capacity(layers.len());
    let mut v_all = Vec::with_capacity(layers.len());
    for layer in layers {
        let lw = layer.weights();
        let normed = nn::rms_norm(&x, Some(&layer.input_ln), EPS).unwrap();
        let (q, k, v) = qkv_with_rope(&normed, &lw, &rope, HIDDEN, QKV_DIM).unwrap();
        k_all.push(k.data.clone());
        v_all.push(v.data.clone());
        let ctx = prefill_attention(&q, &k, &v, NUM_HEADS, HEAD_DIM).unwrap();
        let attn = attn_output_proj(&ctx, &layer.o_proj, HIDDEN, QKV_DIM).unwrap();
        let h = add_residual(&x, &attn).unwrap();
        x = mlp_block(layer, &h);
    }
    Prefill {
        hidden: x,
        k: k_all,
        v: v_all,
    }
}

/// Chunked schedule — push each `chunk`-token slice through ALL layers, growing
/// the per-layer reference K/V, then attend over the running prefix (the
/// `prefill_with_cache` chunked path, rebuilt over the public kernels).
fn run_chunked(layers: &[Layer], embeds: &Mat, chunk: usize) -> Prefill {
    let seq = embeds.rows;
    let mut out = Mat::zeros(seq, HIDDEN);
    let mut k_full: Vec<Mat> = (0..layers.len())
        .map(|_| Mat::zeros(seq, QKV_DIM))
        .collect();
    let mut v_full: Vec<Mat> = (0..layers.len())
        .map(|_| Mat::zeros(seq, QKV_DIM))
        .collect();
    let mut c0 = 0usize;
    while c0 < seq {
        let c1 = (c0 + chunk).min(seq);
        let rope = RopeTable::build(&(c0..c1).collect::<Vec<_>>(), HEAD_DIM, THETA);
        let mut x = Mat::from_vec(
            c1 - c0,
            HIDDEN,
            embeds.data[c0 * HIDDEN..c1 * HIDDEN].to_vec(),
        );
        for (li, layer) in layers.iter().enumerate() {
            let lw = layer.weights();
            let normed = nn::rms_norm(&x, Some(&layer.input_ln), EPS).unwrap();
            let (q, k, v) = qkv_with_rope(&normed, &lw, &rope, HIDDEN, QKV_DIM).unwrap();
            k_full[li].data[c0 * QKV_DIM..c1 * QKV_DIM].copy_from_slice(&k.data);
            v_full[li].data[c0 * QKV_DIM..c1 * QKV_DIM].copy_from_slice(&v.data);
            let kpre = Mat::from_vec(c1, QKV_DIM, k_full[li].data[..c1 * QKV_DIM].to_vec());
            let vpre = Mat::from_vec(c1, QKV_DIM, v_full[li].data[..c1 * QKV_DIM].to_vec());
            let ctx = chunk_prefill_attention(&q, &kpre, &vpre, NUM_HEADS, HEAD_DIM, c0).unwrap();
            let attn = attn_output_proj(&ctx, &layer.o_proj, HIDDEN, QKV_DIM).unwrap();
            let h = add_residual(&x, &attn).unwrap();
            x = mlp_block(layer, &h);
        }
        out.data[c0 * HIDDEN..c1 * HIDDEN].copy_from_slice(&x.data);
        c0 = c1;
    }
    Prefill {
        hidden: out,
        k: k_full.into_iter().map(|m| m.data).collect(),
        v: v_full.into_iter().map(|m| m.data).collect(),
    }
}

/// Repack token-major `[seq, QKV_DIM]` K/V into the head-major
/// `[NUM_HEADS, seq, HEAD_DIM]` flat layout `RingCache::record_prefill` consumes.
fn token_to_head_major(tm: &[f32], seq: usize) -> Vec<f32> {
    let mut hm = vec![0.0f32; NUM_HEADS * seq * HEAD_DIM];
    for s in 0..seq {
        for h in 0..NUM_HEADS {
            let src = s * QKV_DIM + h * HEAD_DIM;
            let dst = h * seq * HEAD_DIM + s * HEAD_DIM;
            hm[dst..dst + HEAD_DIM].copy_from_slice(&tm[src..src + HEAD_DIM]);
        }
    }
    hm
}

/// Seed one `RingCache` per layer from token-major K/V (exactly what the driver
/// does after the prefill loop).
fn seed_rings(k: &[Vec<f32>], v: &[Vec<f32>], seq: usize) -> Vec<RingCache> {
    k.iter()
        .zip(v.iter())
        .map(|(kl, vl)| {
            let mut cache = RingCache::new(seq.max(1));
            let kh = token_to_head_major(kl, seq);
            let vh = token_to_head_major(vl, seq);
            cache.record_prefill(&kh, &vh, seq).unwrap();
            cache
        })
        .collect()
}

fn bits(s: &[f32]) -> Vec<u32> {
    s.iter().map(|f| f.to_bits()).collect()
}

/// Chunked prefill (final hidden + every layer's token-major reference K/V) is
/// BYTE-FOR-BYTE the monolithic prefill, for chunk sizes {1, 2, prime, full}.
#[test]
fn chunked_prefill_matches_monolithic_hidden_and_kv() {
    let seq = 7usize; // prime length
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let layers: Vec<Layer> = (0..N_LAYERS).map(|_| Layer::synth(&mut rng)).collect();
    let embeds = Mat::from_vec(seq, HIDDEN, rng.fill(seq * HIDDEN));

    let mono = run_monolithic(&layers, &embeds);

    // 1, 2, a prime (3) that does not divide 7, and the full length (single chunk).
    for &chunk in &[1usize, 2, 3, seq] {
        let ch = run_chunked(&layers, &embeds, chunk);
        assert_eq!(
            bits(&mono.hidden.data),
            bits(&ch.hidden.data),
            "chunk={chunk}: final hidden != monolithic"
        );
        for li in 0..N_LAYERS {
            assert_eq!(
                bits(&mono.k[li]),
                bits(&ch.k[li]),
                "chunk={chunk} layer {li}: K (ring input) != monolithic"
            );
            assert_eq!(
                bits(&mono.v[li]),
                bits(&ch.v[li]),
                "chunk={chunk} layer {li}: V (ring input) != monolithic"
            );
        }

        // End-to-end through the ACTUAL RingCache: seed both schedules' K/V and
        // compare the live reference blocks the decode phase will read.
        let mono_rings = seed_rings(&mono.k, &mono.v, seq);
        let ch_rings = seed_rings(&ch.k, &ch.v, seq);
        for li in 0..N_LAYERS {
            assert_eq!(mono_rings[li].prefill_len(), ch_rings[li].prefill_len());
            for h in 0..NUM_HEADS {
                assert_eq!(
                    bits(mono_rings[li].reference_k(h)),
                    bits(ch_rings[li].reference_k(h)),
                    "chunk={chunk} layer {li} head {h}: ring reference K != monolithic"
                );
                assert_eq!(
                    bits(mono_rings[li].reference_v(h)),
                    bits(ch_rings[li].reference_v(h)),
                    "chunk={chunk} layer {li} head {h}: ring reference V != monolithic"
                );
            }
        }
    }
}

/// The chunked-attention KERNEL alone is byte-for-byte the monolithic
/// `prefill_attention` over the whole sequence — the crux of the tiling claim,
/// isolated from the rest of the layer and exercised at a non-trivial head shape.
#[test]
fn chunk_prefill_attention_kernel_matches_monolithic() {
    let (num_heads, head_dim, seq) = (4usize, 6usize, 13usize);
    let dim = num_heads * head_dim;
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mk = |rng: &mut Rng| Mat::from_vec(seq, dim, rng.fill(seq * dim));
    let q = mk(&mut rng);
    let k = mk(&mut rng);
    let v = mk(&mut rng);
    let mono = prefill_attention(&q, &k, &v, num_heads, head_dim).unwrap();
    for &chunk in &[1usize, 2, 5, seq] {
        let mut assembled = Mat::zeros(seq, dim);
        let mut c0 = 0usize;
        while c0 < seq {
            let c1 = (c0 + chunk).min(seq);
            let q_chunk = Mat::from_vec(c1 - c0, dim, q.data[c0 * dim..c1 * dim].to_vec());
            let kpre = Mat::from_vec(c1, dim, k.data[..c1 * dim].to_vec());
            let vpre = Mat::from_vec(c1, dim, v.data[..c1 * dim].to_vec());
            let ctx =
                chunk_prefill_attention(&q_chunk, &kpre, &vpre, num_heads, head_dim, c0).unwrap();
            assembled.data[c0 * dim..c1 * dim].copy_from_slice(&ctx.data);
            c0 = c1;
        }
        assert_eq!(
            bits(&mono.data),
            bits(&assembled.data),
            "chunk={chunk}: chunked attention != monolithic prefill_attention"
        );
    }
}

/// The kill-switch defaults to the monolithic schedule (`None`) when unset.
#[test]
fn prefill_chunk_size_defaults_off() {
    if std::env::var_os("FOCR_PREFILL_CHUNK").is_none() {
        assert_eq!(decoder::prefill_chunk_size(), None);
    }
}
