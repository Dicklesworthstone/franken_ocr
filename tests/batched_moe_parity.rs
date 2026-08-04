//! Batched MoE decode-dispatch parity gate (bd-1azu.6 — the Phase-6 MoE stage of
//! the continuous-batch decode spine, bd-1azu).
//!
//! The batched MoE stage MUST be LOSSLESS (Doctrine #1): for every in-flight
//! page-stream `s`, the batched call's output `result[s]` is **byte-for-byte**
//! the single-stream `moe_block(row s)`. Unlike the per-stream attention stage
//! (bd-1azu.5, which never key-batches across streams), the MoE GROUPS all B
//! stacked tokens — ONE f32 router GEMM over B, then ONE SwiGLU GEMM per ACTIVE
//! expert over the rows that selected it, scattered back to stream order, plus
//! the shared experts. This is the real Lever B throughput win AND bit-exact,
//! because the grouping only M-batches the GEMMs: every output row's per-key f32
//! reduction order is fixed regardless of how many rows are stacked (the same
//! M-independence the int8 spine proves in `batched_igemm_parity`, bd-1azu.2),
//! and a token's 6 routed contributions are restored to the pinned torch-2.10
//! `topk(sorted=False)` slots and combined in that same order identically batched
//! vs. standalone (bd-1waa). This file is the executing proof of that invariant.
//!
//! The oracle is the SAME tested `moe_block` the batched dispatch wraps, fed the
//! SAME gate / 64 experts / shared expert and the SAME stacked rows — only the
//! row count M differs — so parity isolates exactly the batched-vs-standalone
//! M-dimension and nothing else.
//!
//! Coverage: random router/expert weights across a sweep of batch sizes B (the
//! natural case, partially-overlapping top-6 sets per stream); an all-zero gate
//! that forces EVERY stream onto experts 0..5 (the maximal `m = B` grouped-GEMM
//! case); the `norm_topk_prob = true` + scaled gate branch; the `B = 1` identity;
//! and the two argument-validation errors (wrong expert count, empty `Weights`).

use franken_ocr::FocrError;
use franken_ocr::native_engine::moe::{
    MlpWeights, batched_forward, batched_moe_block, batched_moe_block_default, config, moe_block,
    moe_block_default,
};
use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::weights::Weights;

/// Deterministic xorshift64 PRNG — reproducible, no dev-dependency (mirrors the
/// `batched_igemm_parity` idiom).
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
    /// A value in `[-1, 1)` from the top 24 bits (a clean f32 mantissa).
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

/// Owns the gate / 64 routed experts / fused shared expert backing storage so the
/// borrowing [`MlpWeights`] views handed to both the batched and the per-stream
/// calls reference the IDENTICAL bytes (the only thing that varies between the
/// two paths is the row count M).
struct Fixture {
    hidden: usize,
    inter: usize,
    shared_inter: usize,
    gate: Vec<f32>,
    routed: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    sg: Vec<f32>,
    su: Vec<f32>,
    sd: Vec<f32>,
}

impl Fixture {
    fn new(rng: &mut Rng, hidden: usize, inter: usize, shared_inter: usize) -> Self {
        let n = config::N_ROUTED_EXPERTS;
        let gate = rng.fill(n * hidden);
        let routed = (0..n)
            .map(|_| {
                (
                    rng.fill(inter * hidden), // gate_proj [inter, hidden]
                    rng.fill(inter * hidden), // up_proj   [inter, hidden]
                    rng.fill(hidden * inter), // down_proj [hidden, inter]
                )
            })
            .collect();
        let sg = rng.fill(shared_inter * hidden);
        let su = rng.fill(shared_inter * hidden);
        let sd = rng.fill(hidden * shared_inter);
        Self {
            hidden,
            inter,
            shared_inter,
            gate,
            routed,
            sg,
            su,
            sd,
        }
    }

    fn experts(&self) -> Vec<MlpWeights<'_>> {
        self.routed
            .iter()
            .map(|(g, u, d)| MlpWeights {
                gate_proj: g,
                up_proj: u,
                down_proj: d,
                hidden: self.hidden,
                intermediate: self.inter,
            })
            .collect()
    }

    fn shared(&self) -> MlpWeights<'_> {
        MlpWeights {
            gate_proj: &self.sg,
            up_proj: &self.su,
            down_proj: &self.sd,
            hidden: self.hidden,
            intermediate: self.shared_inter,
        }
    }
}

/// THE invariant: each stream's batched output is byte-for-byte the single-stream
/// `moe_block` over that stream's own `[1, hidden]` row, across a sweep of batch
/// sizes B with random routing (partially-overlapping top-6 sets per stream).
#[test]
fn batched_moe_block_matches_per_stream_moe_block() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (hidden, inter, shared_inter) = (8usize, 4usize, 6usize);
    let fx = Fixture::new(&mut rng, hidden, inter, shared_inter);
    let experts = fx.experts();
    let shared = fx.shared();

    for &b in &[1usize, 2, 3, 5, 8, 16, 17, 32] {
        let hb = Mat::from_vec(b, hidden, rng.fill(b * hidden));

        // ONE batched grouped MoE pass over all B stacked streams.
        let batched =
            batched_moe_block(&hb, &fx.gate, &experts, &shared, false, 1.0).expect("batched moe");
        assert_eq!(batched.len(), b, "one output per stream (B={b})");

        for (s, out) in batched.iter().enumerate() {
            // Oracle: stream s alone through the existing per-token moe_block.
            let row = Mat::from_vec(1, hidden, hb.row(s).to_vec());
            let single =
                moe_block(&row, &fx.gate, &experts, &shared, false, 1.0).expect("single moe");
            assert_eq!(out.shape(), (1, hidden), "output s{s} shape (B={b})");
            assert_eq!(
                out.data, single.data,
                "stream {s} (B={b}): batched grouped MoE != single-stream \
                 moe_block (LOSSLESS invariant broken)"
            );
        }
    }
}

/// The maximal grouped-GEMM case: an all-zero gate makes every stream's pinned
/// torch-2.10 CPU top-6 exactly experts 0..5 (the already-equivalent input range
/// exits libc++'s partition without swaps),
/// so experts 0..5 each group ALL B streams (`m = B`) into one `[B, hidden]`
/// SwiGLU GEMM. Row `s` of that grouped GEMM MUST still byte-match the standalone
/// `[1, hidden]` GEMM for stream `s` — the strongest M-independence assertion.
#[test]
fn batched_moe_block_all_streams_same_experts_bit_exact() {
    let mut rng = Rng(0xd1b5_4a32_0c7e_91af);
    let (hidden, inter, shared_inter) = (8usize, 4usize, 6usize);
    let mut fx = Fixture::new(&mut rng, hidden, inter, shared_inter);
    // Force a uniform router so the pinned top-6 set is identical (experts 0..5)
    // for every stream; experts/shared stay random so outputs differ per stream.
    fx.gate = vec![0.0f32; config::N_ROUTED_EXPERTS * hidden];
    let experts = fx.experts();
    let shared = fx.shared();

    for &b in &[2usize, 4, 7, 16] {
        let hb = Mat::from_vec(b, hidden, rng.fill(b * hidden));
        let batched =
            batched_moe_block_default(&hb, &fx.gate, &experts, &shared).expect("batched moe");
        for (s, out) in batched.iter().enumerate() {
            let row = Mat::from_vec(1, hidden, hb.row(s).to_vec());
            let single = moe_block_default(&row, &fx.gate, &experts, &shared).expect("single moe");
            assert_eq!(
                out.data, single.data,
                "all-same-experts grouping: stream {s} (B={b}) batched != single"
            );
        }
    }
}

/// The `norm_topk_prob = true` (renormalize the 6 weights to sum 1) + scaled gate
/// branch is batched-lossless too — the gate normalization is per-token, so it is
/// M-independent like the rest.
#[test]
fn batched_moe_block_norm_topk_scaled_matches_per_stream() {
    let mut rng = Rng(0x0ddb_a11c_0ffe_e099);
    let (hidden, inter, shared_inter) = (8usize, 4usize, 6usize);
    let fx = Fixture::new(&mut rng, hidden, inter, shared_inter);
    let experts = fx.experts();
    let shared = fx.shared();

    for &b in &[1usize, 4, 8, 13] {
        let hb = Mat::from_vec(b, hidden, rng.fill(b * hidden));
        let batched =
            batched_moe_block(&hb, &fx.gate, &experts, &shared, true, 2.5).expect("batched moe");
        for (s, out) in batched.iter().enumerate() {
            let row = Mat::from_vec(1, hidden, hb.row(s).to_vec());
            let single =
                moe_block(&row, &fx.gate, &experts, &shared, true, 2.5).expect("single moe");
            assert_eq!(
                out.data, single.data,
                "norm_topk+scale: stream {s} (B={b}) batched != single"
            );
        }
    }
}

/// `batched_moe_block_default` is the pinned-config wrapper: it must equal the
/// per-stream `moe_block_default` exactly (and so the explicit-config call with
/// `norm_topk_prob = false, scale = 1.0`).
#[test]
fn batched_moe_block_default_matches_per_stream_default() {
    let mut rng = Rng(0xa11c_e0ba_df00_d042);
    let (hidden, inter, shared_inter) = (8usize, 4usize, 6usize);
    let fx = Fixture::new(&mut rng, hidden, inter, shared_inter);
    let experts = fx.experts();
    let shared = fx.shared();

    for &b in &[1usize, 3, 9] {
        let hb = Mat::from_vec(b, hidden, rng.fill(b * hidden));
        let batched =
            batched_moe_block_default(&hb, &fx.gate, &experts, &shared).expect("batched moe");
        for (s, out) in batched.iter().enumerate() {
            let row = Mat::from_vec(1, hidden, hb.row(s).to_vec());
            let single = moe_block_default(&row, &fx.gate, &experts, &shared).expect("single moe");
            assert_eq!(
                out.data, single.data,
                "default: stream {s} (B={b}) batched != single"
            );
        }
    }
}

/// `B = 1` edge case: the batched dispatch returns exactly one output, byte-for-byte
/// the single-stream `moe_block` over the same `[1, hidden]` row.
#[test]
fn batched_moe_block_single_stream_identity() {
    let mut rng = Rng(0x5eed_1a2b_3c4d_5e6f);
    let (hidden, inter, shared_inter) = (8usize, 4usize, 6usize);
    let fx = Fixture::new(&mut rng, hidden, inter, shared_inter);
    let experts = fx.experts();
    let shared = fx.shared();

    let hb = Mat::from_vec(1, hidden, rng.fill(hidden));
    let batched =
        batched_moe_block(&hb, &fx.gate, &experts, &shared, false, 1.0).expect("batched moe");
    assert_eq!(batched.len(), 1, "B=1 yields exactly one output");
    let single = moe_block(&hb, &fx.gate, &experts, &shared, false, 1.0).expect("single moe");
    assert_eq!(batched[0].data, single.data, "B=1 batched != single");
}

/// Argument validation: a wrong routed-expert count is a clean error (propagated
/// from `moe_block`), not a panic.
#[test]
fn batched_moe_block_rejects_wrong_expert_count() {
    let mut rng = Rng(0xdead_beef_cafe_0006);
    let (hidden, inter, shared_inter) = (8usize, 4usize, 6usize);
    let fx = Fixture::new(&mut rng, hidden, inter, shared_inter);
    let experts = fx.experts();
    let shared = fx.shared();

    let experts_short = experts[..1].to_vec(); // 1, not N_ROUTED_EXPERTS=64
    let hb = Mat::from_vec(2, hidden, vec![0.0f32; 2 * hidden]);
    assert!(
        batched_moe_block(&hb, &fx.gate, &experts_short, &shared, false, 1.0).is_err(),
        "wrong expert count must error"
    );
}

/// `batched_forward` is wired like `forward`: an empty `Weights` (no tensors) must
/// surface a clean `FormatMismatch` (tensor not found) for the batched dispatch
/// too, never a panic or garbage — mirrors the per-token `forward` shim's error
/// path.
#[test]
fn batched_forward_errors_cleanly_on_empty_weights() {
    let w = Weights::default();
    let h = config::HIDDEN_SIZE;
    let x = Mat::from_vec(2, h, vec![0.0f32; 2 * h]); // B=2 streams
    assert!(matches!(
        batched_forward(&w, &x, 1),
        Err(FocrError::FormatMismatch(_))
    ));
}
