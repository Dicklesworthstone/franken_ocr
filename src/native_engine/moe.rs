//! MoE: greedy-softmax top-6 gate + grouped SiLU-gated experts, plus the
//! dense layer-0 MLP ([SPEC-074..077], PROPOSED_ARCHITECTURE.md §6.9).
//!
//! This module realizes three beads:
//!
//! * **P1-moe-router** — [`route`]: `logits = linear(hidden.f32, gate.f32)` ->
//!   `softmax(f32, dim=-1)` -> greedy top-6 (partial select, unsorted). The
//!   norm step follows [SPEC-077]: because `num_experts_per_tok = 6 (>1)` and
//!   `norm_topk_prob = false`, the gate takes the `else` branch and
//!   `topk_weight *= routed_scaling_factor (= 1.0)` — i.e. the raw softmax
//!   top-k probs, NOT renormalized to sum 1. The renormalizing branch
//!   (`norm_topk_prob = true`) is implemented too and selected by a flag, so
//!   the routing primitive is parity-correct for either config. The router is
//!   NEVER quantized (it stays f32).
//! * **P1-moe-experts** — [`expert_mlp`] / [`moe_block`]: each routed expert is
//!   a SwiGLU MLP `down_proj(silu(gate_proj(x)) * up_proj(x))` with
//!   `gate_proj/up_proj : 1280 -> 896` and `down_proj : 896 -> 1280`. The
//!   per-token output is the router-weighted sum over its 6 routed experts plus
//!   the 2 always-on shared experts (a single fused `DeepseekV2MLP` with
//!   intermediate `896 * 2 = 1792`), added at weight 1.0 ([SPEC-076]).
//! * **P1-dense-mlp0** — [`dense_mlp`]: the layer-0 dense MLP, the same SwiGLU
//!   but with `intermediate_size = 6848` ([SPEC-074/075]). `first_k_dense_replace
//!   = 1`, so only layer 0 is dense; layers 1..11 are MoE.
//!
//! ## Weight layout
//!
//! All linear weights are PyTorch `nn.Linear.weight`, i.e. row-major
//! `[out_features, in_features]`. `F.linear(x, w)` computes `x @ w.T`, so for an
//! `[n_tok, in]` activation and an `[out, in]` weight the result is
//! `[n_tok, out]`. The frankentorch facade [`nn::matmul`] computes
//! `[m, k] x [k, n]`, so we transpose the weight to `[in, out]` once (see
//! [`linear_no_bias`]) and matmul. No projection in this block has a bias
//! (`DeepseekV2MLP` Linears are `bias=False`).
//!
//! The gate weight is `[n_routed_experts, hidden] = [64, 1280]` ([SPEC-077]).
//!
//! ## `Weights`-backed entry points
//!
//! The `.focrq` reader / weight directory (`super::weights::Weights`) is still a
//! Phase-2 stub with no tensor accessors, so the public [`forward`] /
//! [`dense_forward`] shims that take a `&Weights` return
//! [`FocrError::NotImplemented`] until that lands; the real, fully-tested math
//! lives in the slice-typed primitives below ([`route`], [`expert_mlp`],
//! [`dense_mlp`], [`moe_block`]) which the decoder driver will call once it can
//! hand over the per-layer weight slices.

use super::nn;
use super::tensor::Mat;
use super::weights::Weights;
use crate::error::{FocrError, FocrResult};

/// MoE config constants ([SPEC-010/012]).
pub mod config {
    /// Decoder hidden size ([SPEC-010]).
    pub const HIDDEN_SIZE: usize = 1280;
    /// Routed experts ([SPEC-012]).
    pub const N_ROUTED_EXPERTS: usize = 64;
    /// Shared experts ([SPEC-012]).
    pub const N_SHARED_EXPERTS: usize = 2;
    /// Experts per token (top-k) ([SPEC-012]).
    pub const NUM_EXPERTS_PER_TOK: usize = 6;
    /// MoE expert intermediate size ([SPEC-012]).
    pub const MOE_INTERMEDIATE_SIZE: usize = 896;
    /// Fused shared-expert intermediate size = `MOE_INTERMEDIATE_SIZE *
    /// N_SHARED_EXPERTS = 1792` ([SPEC-076]).
    pub const SHARED_INTERMEDIATE_SIZE: usize = MOE_INTERMEDIATE_SIZE * N_SHARED_EXPERTS;
    /// Dense MLP intermediate size (layer 0) ([SPEC-010/075]).
    pub const DENSE_INTERMEDIATE_SIZE: usize = 6848;
    /// Routed scaling factor ([SPEC-013]).
    pub const ROUTED_SCALING_FACTOR: f32 = 1.0;
    /// `norm_topk_prob` ([SPEC-013/077]). False => raw top-k probs, NOT
    /// renormalized; the gate multiplies by `ROUTED_SCALING_FACTOR` only.
    pub const NORM_TOPK_PROB: bool = false;
    /// Layers `< FIRST_K_DENSE_REPLACE` are dense MLP ([SPEC-012/074]).
    pub const FIRST_K_DENSE_REPLACE: usize = 1;
}

fn checked_shape_mul(context: &str, lhs: usize, rhs: usize, expression: &str) -> FocrResult<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: usize overflow computing {expression} ({lhs} * {rhs})"
        ))
    })
}

/// A SwiGLU MLP's three weight matrices, all PyTorch `[out, in]` row-major.
///
/// `gate_proj` and `up_proj` are `[intermediate, hidden]`; `down_proj` is
/// `[hidden, intermediate]`. Used uniformly for the dense layer-0 MLP, the
/// fused shared expert, and each routed expert — only the `intermediate`
/// dimension differs (6848 / 1792 / 896 respectively).
#[derive(Debug, Clone, Copy)]
pub struct MlpWeights<'a> {
    /// `gate_proj.weight`, row-major `[intermediate, hidden]`.
    pub gate_proj: &'a [f32],
    /// `up_proj.weight`, row-major `[intermediate, hidden]`.
    pub up_proj: &'a [f32],
    /// `down_proj.weight`, row-major `[hidden, intermediate]`.
    pub down_proj: &'a [f32],
    /// Hidden (input/output) dimension.
    pub hidden: usize,
    /// Intermediate (SwiGLU) dimension.
    pub intermediate: usize,
}

/// The result of the gate: for each token, the chosen expert ids and the
/// matching router weights ([SPEC-077]).
///
/// `indices[t]` and `weights[t]` are each length `NUM_EXPERTS_PER_TOK = 6` and
/// positionally aligned (`weights[t][j]` is the router weight for expert
/// `indices[t][j]`). Selection is greedy top-k (largest softmax prob first);
/// ordering within the 6 is by descending probability.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    /// Per-token chosen expert ids, `n_tok x top_k`.
    pub indices: Vec<[usize; config::NUM_EXPERTS_PER_TOK]>,
    /// Per-token router weights, `n_tok x top_k`, aligned with `indices`.
    pub weights: Vec<[f32; config::NUM_EXPERTS_PER_TOK]>,
}

// ── linear helper (PyTorch `F.linear`, no bias) ─────────────────────────────

/// `y = x @ w.T` for a PyTorch `[out, in]` weight and `[n_tok, in]` activation,
/// returning `[n_tok, out]`. No bias (every projection in this block is
/// `bias=False`). The router gate and all expert/dense projections route through
/// here; the gate stays f32 (never quantized) and the experts use the f32 rail
/// for the parity spine (int8 is an additive kill-switched layer, not this path).
///
/// # Errors
/// [`FocrError::Other`] if `x.cols != in_`, `out * in_` overflows, or
/// `w.len() != out * in_`.
fn linear_no_bias(x: &Mat, w: &[f32], out: usize, in_: usize) -> FocrResult<Mat> {
    if x.cols != in_ {
        return Err(FocrError::Other(anyhow::anyhow!(
            "moe::linear_no_bias: x.cols {} != in {}",
            x.cols,
            in_
        )));
    }
    let expected_weight_len = checked_shape_mul("moe::linear_no_bias", out, in_, "out*in")?;
    if w.len() != expected_weight_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "moe::linear_no_bias: weight len {} != out*in {}",
            w.len(),
            expected_weight_len
        )));
    }
    // Transpose [out, in] -> [in, out] so matmul([n_tok, in], [in, out]) works.
    let mut wt = vec![0.0f32; expected_weight_len];
    for i in 0..in_ {
        let dst = &mut wt[i * out..(i + 1) * out];
        for (o, slot) in dst.iter_mut().enumerate() {
            *slot = w[o * in_ + i];
        }
    }
    let wt_mat = Mat::from_vec(in_, out, wt);
    nn::matmul(x, &wt_mat)
}

// ── P1-moe-router ───────────────────────────────────────────────────────────

/// Greedy top-k router ([SPEC-077]).
///
/// `hidden` is `[n_tok, HIDDEN_SIZE]`; `gate` is the gate weight, row-major
/// `[N_ROUTED_EXPERTS, HIDDEN_SIZE] = [64, 1280]` (NEVER quantized). Steps:
///
/// 1. `logits = linear(hidden, gate)` -> `[n_tok, 64]` (f32).
/// 2. `scores = softmax(logits, dim=-1)` over the 64 experts (f32,
///    `scoring_func = 'softmax'`).
/// 3. Greedy top-`k` (`k = NUM_EXPERTS_PER_TOK = 6`) by descending score —
///    `torch.topk(scores, k, sorted=False)`; we partial-select then order the 6
///    by descending prob for determinism.
/// 4. Norm: if `norm_topk_prob` the 6 weights are renormalized to sum 1 (`w /=
///    sum(w) + 1e-20`) then `*= routed_scaling_factor`; otherwise (the pinned
///    config, `norm_topk_prob = false`) they are just `*= routed_scaling_factor`
///    — the raw softmax probs.
///
/// # Errors
/// [`FocrError::Other`] on a dimension mismatch, or if `N_ROUTED_EXPERTS <
/// NUM_EXPERTS_PER_TOK` (cannot select 6 of fewer than 6).
pub fn route(
    hidden: &Mat,
    gate: &[f32],
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> FocrResult<Routing> {
    const K: usize = config::NUM_EXPERTS_PER_TOK;
    let n_experts = config::N_ROUTED_EXPERTS;
    if n_experts < K {
        return Err(FocrError::Other(anyhow::anyhow!(
            "moe::route: n_routed_experts {} < top_k {}",
            n_experts,
            K
        )));
    }

    // 1. logits [n_tok, 64]. The gate is [n_experts, hidden]; read the input
    //    dimension from the activation itself so the router works for any
    //    configured hidden size (the full model uses HIDDEN_SIZE = 1280, the
    //    small-MoE tests use a tiny hidden), rather than hardcoding the constant.
    let mut scores = linear_no_bias(hidden, gate, n_experts, hidden.cols)?;
    // 2. softmax over the expert axis (dim=-1) — numerically stable, in place.
    nn::softmax_rows(&mut scores)?;

    let n_tok = scores.rows;
    let mut indices = Vec::with_capacity(n_tok);
    let mut weights = Vec::with_capacity(n_tok);

    for t in 0..n_tok {
        let row = scores.row(t);

        // 3. Greedy top-k via a small partial selection: repeatedly pick the
        //    argmax over not-yet-chosen experts. K=6 over 64 — a tight scalar
        //    loop, and ties break to the lower index (stable, matches a
        //    deterministic topk on equal scores).
        let mut chosen_idx = [0usize; K];
        let mut chosen_w = [0.0f32; K];
        let mut taken = [false; config::N_ROUTED_EXPERTS];
        for slot in 0..K {
            let (best, best_v) = next_router_expert(row, &taken, t, slot)?;
            taken[best] = true;
            chosen_idx[slot] = best;
            chosen_w[slot] = best_v;
        }

        // 4. norm_topk_prob branch ([SPEC-077]).
        if norm_topk_prob {
            // top_k > 1 guaranteed here; renormalize the 6 to sum 1.
            let denom: f32 = chosen_w.iter().sum::<f32>() + 1e-20;
            for w in &mut chosen_w {
                *w /= denom;
            }
        }
        for w in &mut chosen_w {
            *w *= routed_scaling_factor;
        }

        indices.push(chosen_idx);
        weights.push(chosen_w);
    }

    Ok(Routing { indices, weights })
}

fn next_router_expert(
    row: &[f32],
    taken: &[bool; config::N_ROUTED_EXPERTS],
    token_idx: usize,
    slot: usize,
) -> FocrResult<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for (e, &v) in row.iter().enumerate() {
        if taken[e] || v.is_nan() {
            continue;
        }
        match best {
            Some((_, best_v)) if v <= best_v => {}
            _ => best = Some((e, v)),
        }
    }
    best.ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "moe::route: no finite router score for token {token_idx} top-k slot {slot}"
        ))
    })
}

/// [`route`] with the pinned-config defaults (`norm_topk_prob = false`,
/// `routed_scaling_factor = 1.0`) — the raw softmax top-6 probs ([SPEC-013/077]).
///
/// # Errors
/// Propagates [`route`].
pub fn route_default(hidden: &Mat, gate: &[f32]) -> FocrResult<Routing> {
    route(
        hidden,
        gate,
        config::NORM_TOPK_PROB,
        config::ROUTED_SCALING_FACTOR,
    )
}

// ── P1-moe-experts / P1-dense-mlp0: the SwiGLU MLP ─────────────────────────

/// SwiGLU MLP forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`
/// ([SPEC-075]). Shared by the dense layer-0 MLP, the fused shared expert, and
/// every routed expert (only `intermediate` differs).
///
/// `x` is `[n_tok, hidden]`; returns `[n_tok, hidden]`.
///
/// # Errors
/// [`FocrError::Other`] on any dimension mismatch in `w`.
pub fn expert_mlp(x: &Mat, w: &MlpWeights<'_>) -> FocrResult<Mat> {
    // gate = silu(x @ gate_proj.T)  -> [n_tok, intermediate]
    let mut gate = linear_no_bias(x, w.gate_proj, w.intermediate, w.hidden)?;
    nn::silu(&mut gate);
    // up = x @ up_proj.T            -> [n_tok, intermediate]
    let up = linear_no_bias(x, w.up_proj, w.intermediate, w.hidden)?;
    // elementwise silu(gate) * up
    if gate.data.len() != up.data.len() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "moe::expert_mlp: gate/up shape mismatch ({} vs {})",
            gate.data.len(),
            up.data.len()
        )));
    }
    for (g, &u) in gate.data.iter_mut().zip(up.data.iter()) {
        *g *= u;
    }
    // down = (silu(gate)*up) @ down_proj.T -> [n_tok, hidden]
    linear_no_bias(&gate, w.down_proj, w.hidden, w.intermediate)
}

/// The dense layer-0 MLP ([SPEC-074/075]) — [`expert_mlp`] with the dense
/// intermediate (`6848`). A thin named wrapper so call sites read intent.
///
/// `x` is `[n_tok, HIDDEN_SIZE]`; returns `[n_tok, HIDDEN_SIZE]`.
///
/// # Errors
/// [`FocrError::Other`] if `w.intermediate != DENSE_INTERMEDIATE_SIZE` or any
/// dimension mismatch.
pub fn dense_mlp(x: &Mat, w: &MlpWeights<'_>) -> FocrResult<Mat> {
    if w.intermediate != config::DENSE_INTERMEDIATE_SIZE {
        return Err(FocrError::Other(anyhow::anyhow!(
            "moe::dense_mlp: intermediate {} != dense {}",
            w.intermediate,
            config::DENSE_INTERMEDIATE_SIZE
        )));
    }
    expert_mlp(x, w)
}

/// The full MoE block over a layer's hidden states ([SPEC-076]):
///
/// `y = moe_infer(hidden) + shared_experts(hidden)` where
///
/// * `moe_infer` routes each token to its top-6 routed experts (via [`route`]),
///   runs each expert's [`expert_mlp`], and sums them weighted by the router
///   weights — `y_routed[t] = Σ_j w[t][j] · expert_{idx[t][j]}(hidden[t])`.
/// * the shared experts are a single fused `DeepseekV2MLP` with intermediate
///   `1792`, added at weight 1.0 over every token.
///
/// `experts` must be exactly `N_ROUTED_EXPERTS` long (one [`MlpWeights`] per
/// routed expert, each with `intermediate = MOE_INTERMEDIATE_SIZE = 896`);
/// `shared` is the fused shared expert (`intermediate = 1792`). `gate` is the
/// `[64, 1280]` router weight.
///
/// This computes each expert over **all** tokens and masks by the per-token
/// routing — the simple dense path. It is mathematically identical to the
/// reference's sort-by-expert `moe_infer` (a token contributes `w` of an expert
/// iff that expert is in its top-6, else 0); the gather/scatter optimization is
/// a later perf lever (plan §9), not a correctness concern for the parity spine.
///
/// # Errors
/// [`FocrError::Other`] on a wrong expert count or any dimension mismatch.
#[allow(clippy::needless_range_loop)]
pub fn moe_block(
    hidden: &Mat,
    gate: &[f32],
    experts: &[MlpWeights<'_>],
    shared: &MlpWeights<'_>,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> FocrResult<Mat> {
    if experts.len() != config::N_ROUTED_EXPERTS {
        return Err(FocrError::Other(anyhow::anyhow!(
            "moe::moe_block: expected {} routed experts, got {}",
            config::N_ROUTED_EXPERTS,
            experts.len()
        )));
    }
    let n_tok = hidden.rows;
    let h = hidden.cols;

    let routing = route(hidden, gate, norm_topk_prob, routed_scaling_factor)?;

    // For each token, accumulate its routed-expert contributions. Build a
    // per-expert token mask so each expert MLP runs once over the tokens that
    // selected it (dense over the batch when a token routes to it, weighted).
    let mut out = Mat::zeros(n_tok, h);

    // expert -> list of (token, weight)
    let mut per_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); config::N_ROUTED_EXPERTS];
    for t in 0..n_tok {
        for j in 0..config::NUM_EXPERTS_PER_TOK {
            let e = routing.indices[t][j];
            let w = routing.weights[t][j];
            per_expert[e].push((t, w));
        }
    }

    for (e, members) in per_expert.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        // Gather the rows that selected expert e into a compact [m, h] activation.
        let m = members.len();
        let mut sub = Mat::zeros(m, h);
        for (r, &(t, _w)) in members.iter().enumerate() {
            sub.row_mut(r).copy_from_slice(hidden.row(t));
        }
        let y = expert_mlp(&sub, &experts[e])?; // [m, h]
        // Scatter back, scaled by the router weight.
        for (r, &(t, w)) in members.iter().enumerate() {
            let yr = y.row(r);
            let outr = out.row_mut(t);
            for c in 0..h {
                outr[c] += w * yr[c];
            }
        }
    }

    // Shared experts: identity (weight 1.0) over every token.
    let shared_out = expert_mlp(hidden, shared)?;
    for (o, &s) in out.data.iter_mut().zip(shared_out.data.iter()) {
        *o += s;
    }

    Ok(out)
}

/// [`moe_block`] with the pinned-config gate defaults ([SPEC-013/077]).
///
/// # Errors
/// Propagates [`moe_block`].
pub fn moe_block_default(
    hidden: &Mat,
    gate: &[f32],
    experts: &[MlpWeights<'_>],
    shared: &MlpWeights<'_>,
) -> FocrResult<Mat> {
    moe_block(
        hidden,
        gate,
        experts,
        shared,
        config::NORM_TOPK_PROB,
        config::ROUTED_SCALING_FACTOR,
    )
}

// ── `Weights`-backed shims (wired once the .focrq reader lands) ─────────────

/// Run the MoE block (layers 1..11) over a layer's hidden states.
///
/// Shim: the `.focrq` weight directory has no tensor accessors yet (Phase 2
/// stub), so this returns [`FocrError::NotImplemented`]. The real math is in
/// [`moe_block`] — the decoder driver will fetch the per-layer gate / expert /
/// shared slices and call it directly once the reader lands.
///
/// # Errors
/// Always [`FocrError::NotImplemented`] until `Weights` exposes tensor slices.
pub fn forward(_weights: &Weights, _hidden: &Mat) -> FocrResult<Mat> {
    Err(FocrError::NotImplemented(
        "native_engine::moe::forward — wire moe_block once Weights exposes tensor slices (bd-1es.3)"
            .into(),
    ))
}

/// Run the dense layer-0 MLP over a layer's hidden states.
///
/// Shim around [`dense_mlp`]; same `Weights`-accessor blocker as [`forward`].
///
/// # Errors
/// Always [`FocrError::NotImplemented`] until `Weights` exposes tensor slices.
pub fn dense_forward(_weights: &Weights, _hidden: &Mat) -> FocrResult<Mat> {
    Err(FocrError::NotImplemented(
        "native_engine::moe::dense_forward — wire dense_mlp once Weights exposes tensor slices (bd-1es.3)"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A trivial [out, in] weight builder for small hand-checkable cases.
    fn linrow(rows: Vec<Vec<f32>>) -> (Vec<f32>, usize, usize) {
        let out = rows.len();
        let in_ = rows[0].len();
        let mut flat = Vec::with_capacity(out * in_);
        for r in &rows {
            assert_eq!(r.len(), in_);
            flat.extend_from_slice(r);
        }
        (flat, out, in_)
    }

    #[test]
    fn linear_no_bias_matches_pytorch_linear() -> FocrResult<()> {
        // x = [[1,2,3]] (1x3); W = [[1,0,0],[0,1,1]] (out=2,in=3)
        // y = x @ W.T = [[1, 2+3]] = [[1, 5]]
        let x = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let (w, out, in_) = linrow(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 1.0]]);
        let y = linear_no_bias(&x, &w, out, in_)?;
        assert_eq!(y.shape(), (1, 2));
        assert_eq!(y.data, vec![1.0, 5.0]);
        Ok(())
    }

    #[test]
    fn linear_no_bias_matches_pytorch_linear_multirow_nonsquare() -> FocrResult<()> {
        let x = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (w, out, in_) = linrow(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ]);
        let y = linear_no_bias(&x, &w, out, in_)?;
        assert_eq!(y.shape(), (2, 4));
        assert_eq!(y.data, vec![1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
        Ok(())
    }

    #[test]
    fn linear_no_bias_rejects_bad_in() {
        let x = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let (w, out, in_) = linrow(vec![vec![1.0, 0.0]]); // in=2 != x.cols=3
        assert!(linear_no_bias(&x, &w, out, in_).is_err());
    }

    #[test]
    fn linear_no_bias_rejects_weight_shape_overflow_without_panicking() {
        let x = Mat::zeros(1, 2);
        let result = linear_no_bias(&x, &[], usize::MAX, 2);
        assert!(matches!(
            &result,
            Err(err) if err.to_string().contains("out*in")
        ));
    }

    /// Hand-check the full router on tiny shapes: hidden=2, n_experts shrunk
    /// conceptually but the real code routes over 64. We instead build a 64-wide
    /// gate where only a few experts get nonzero logits so the top-6 set is
    /// known, then verify selection + softmax-prob weights.
    #[test]
    fn route_selects_greedy_top6_unnormalized() -> FocrResult<()> {
        // hidden = [[1.0]] -> but HIDDEN_SIZE is 1280; build a 1-token hidden of
        // width 1280 with a single 1.0 in column 0, rest 0. gate row e dotted
        // with that hidden = gate[e][0]. So we control each expert's logit by
        // gate[e][0].
        let h = config::HIDDEN_SIZE;
        let n = config::N_ROUTED_EXPERTS;
        let mut hid = vec![0.0f32; h];
        hid[0] = 1.0;
        let hidden = Mat::from_vec(1, h, hid);

        // gate [64, 1280]; set column-0 logits: experts 10,11,12,13,14,15 get
        // descending big values, everyone else gets 0.
        let mut gate = vec![0.0f32; n * h];
        let big = [10.0f32, 9.0, 8.0, 7.0, 6.0, 5.0];
        let want = [10usize, 11, 12, 13, 14, 15];
        for (k, &e) in want.iter().enumerate() {
            gate[e * h] = big[k];
        }

        let r = route(&hidden, &gate, false, 1.0)?;
        assert_eq!(r.indices.len(), 1);
        // Greedy top-6, descending: exactly the six experts we boosted, in order.
        assert_eq!(r.indices[0], want);

        // Weights are raw softmax probs (not renormalized). Recompute softmax
        // over all 64 logits and compare the 6 selected.
        let mut denom = 0.0f64;
        for e in 0..n {
            denom += (gate[e * h] as f64).exp();
        }
        for (k, &e) in want.iter().enumerate() {
            let p = ((gate[e * h] as f64).exp() / denom) as f32;
            assert!(
                (r.weights[0][k] - p).abs() < 1e-5,
                "weight[{k}] {} != softmax prob {p}",
                r.weights[0][k]
            );
        }
        // Unnormalized: the 6 weights do NOT sum to 1.
        let s: f32 = r.weights[0].iter().sum();
        assert!(
            s < 0.9999,
            "top-6 should not sum to 1 when unnormalized: {s}"
        );
        Ok(())
    }

    #[test]
    fn route_norm_topk_renormalizes_to_one() -> FocrResult<()> {
        let h = config::HIDDEN_SIZE;
        let n = config::N_ROUTED_EXPERTS;
        let mut hid = vec![0.0f32; h];
        hid[0] = 1.0;
        let hidden = Mat::from_vec(1, h, hid);
        let mut gate = vec![0.0f32; n * h];
        for (k, e) in (0..6usize).enumerate() {
            gate[e * h] = 6.0 - k as f32;
        }
        // norm_topk_prob = true, scaling = 1.0 -> the 6 weights sum to 1.
        let r = route(&hidden, &gate, true, 1.0)?;
        let s: f32 = r.weights[0].iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-5,
            "renormalized top-6 must sum to 1: {s}"
        );
        // And descending order preserved.
        for k in 1..config::NUM_EXPERTS_PER_TOK {
            assert!(r.weights[0][k] <= r.weights[0][k - 1]);
        }
        Ok(())
    }

    #[test]
    fn route_rejects_nan_router_scores_without_panicking() {
        let h = config::HIDDEN_SIZE;
        let n = config::N_ROUTED_EXPERTS;
        let mut hid = vec![0.0f32; h];
        hid[0] = f32::NAN;
        let hidden = Mat::from_vec(1, h, hid);
        let gate = vec![0.0f32; n * h];

        let result = route(&hidden, &gate, false, 1.0);
        assert!(matches!(
            &result,
            Err(err) if err.to_string().contains("no finite router score")
        ));
    }

    #[test]
    fn route_applies_scaling_factor() -> FocrResult<()> {
        let h = config::HIDDEN_SIZE;
        let n = config::N_ROUTED_EXPERTS;
        let mut hid = vec![0.0f32; h];
        hid[0] = 1.0;
        let hidden = Mat::from_vec(1, h, hid);
        let mut gate = vec![0.0f32; n * h];
        for (k, e) in (0..6usize).enumerate() {
            gate[e * h] = 6.0 - k as f32;
        }
        let base = route(&hidden, &gate, false, 1.0)?;
        let scaled = route(&hidden, &gate, false, 2.5)?;
        for k in 0..config::NUM_EXPERTS_PER_TOK {
            assert!((scaled.weights[0][k] - 2.5 * base.weights[0][k]).abs() < 1e-5);
        }
        Ok(())
    }

    /// expert_mlp on a hand-computable 1->2->1 SwiGLU.
    /// x = [[2.0]] (hidden=1, intermediate=2)
    /// gate_proj = [[1],[ -1]] -> pre = [2, -2]; silu(2)=1.7615942, silu(-2)=-0.23840584
    /// up_proj   = [[3],[ 1]]  -> up = [6, 2]
    /// h = silu(gate)*up = [1.7615942*6, -0.23840584*2] = [10.569565, -0.47681168]
    /// down_proj = [[1, 1]]    -> y = 10.569565 + (-0.47681168) = 10.092754
    #[test]
    fn expert_mlp_matches_hand_computed_swiglu() -> FocrResult<()> {
        let x = Mat::from_vec(1, 1, vec![2.0]);
        let gate_proj = vec![1.0f32, -1.0]; // [intermediate=2, hidden=1]
        let up_proj = vec![3.0f32, 1.0];
        let down_proj = vec![1.0f32, 1.0]; // [hidden=1, intermediate=2]
        let w = MlpWeights {
            gate_proj: &gate_proj,
            up_proj: &up_proj,
            down_proj: &down_proj,
            hidden: 1,
            intermediate: 2,
        };
        let y = expert_mlp(&x, &w)?;
        assert_eq!(y.shape(), (1, 1));
        let silu2 = 2.0f32 / (1.0 + (-2.0f32).exp());
        let silum2 = -2.0f32 / (1.0 + (2.0f32).exp());
        let expect = silu2 * 6.0 + silum2 * 2.0;
        assert!(
            (y.data[0] - expect).abs() < 1e-5,
            "{} != {expect}",
            y.data[0]
        );
        Ok(())
    }

    #[test]
    fn dense_mlp_rejects_wrong_intermediate() {
        let x = Mat::from_vec(1, 1, vec![1.0]);
        let g = vec![1.0f32];
        let w = MlpWeights {
            gate_proj: &g,
            up_proj: &g,
            down_proj: &g,
            hidden: 1,
            intermediate: 1, // != 6848
        };
        assert!(dense_mlp(&x, &w).is_err());
    }

    /// moe_block: with an all-zero gate every expert gets logit 0 -> uniform
    /// softmax (1/64 each). Greedy top-6 picks experts 0..5 (ties -> lower idx),
    /// each weighted 1/64. If every routed expert is the identity-ish MLP that
    /// outputs a known per-expert constant, the routed sum is predictable; we
    /// keep it simple by making all experts produce the SAME output, so the
    /// routed contribution is (sum of 6 weights) * expert_out, plus shared.
    #[test]
    fn moe_block_routes_weights_and_adds_shared() -> FocrResult<()> {
        let h = 2usize;
        let inter = 2usize;
        let n_tok = 1usize;

        // hidden = [[1, 1]]
        let hidden = Mat::from_vec(n_tok, h, vec![1.0, 1.0]);
        // all-zero gate -> uniform softmax -> each prob = 1/64; top-6 weight sum.
        let gate = vec![0.0f32; config::N_ROUTED_EXPERTS * h];

        // Build one shared weight-set used for every routed expert: a SwiGLU
        // that we can evaluate. gate_proj=up_proj=I (2x2), down_proj=I (2x2).
        // pre = x = [1,1]; silu([1,1]) = [0.7310586, 0.7310586]; up = [1,1];
        // hmid = silu*up = [0.7310586, 0.7310586]; down=I -> y = same.
        let eye = vec![1.0f32, 0.0, 0.0, 1.0]; // [2,2] identity, row-major
        let mk = || MlpWeights {
            gate_proj: &eye,
            up_proj: &eye,
            down_proj: &eye,
            hidden: h,
            intermediate: inter,
        };
        let experts: Vec<MlpWeights> = (0..config::N_ROUTED_EXPERTS).map(|_| mk()).collect();
        // Shared expert: zero weights so it contributes nothing (intermediate
        // must be 1792 per spec, but moe_block does not enforce that — keep it
        // small & zero so the shared add is exactly 0 and the routed term is
        // isolated). down_proj all zeros => shared_out = 0.
        let zshared_gate = vec![0.0f32; 2 * h]; // intermediate 2
        let zshared_down = vec![0.0f32; h * 2];
        let shared = MlpWeights {
            gate_proj: &zshared_gate,
            up_proj: &zshared_gate,
            down_proj: &zshared_down,
            hidden: h,
            intermediate: 2,
        };

        let y = moe_block(&hidden, &gate, &experts, &shared, false, 1.0)?;
        assert_eq!(y.shape(), (n_tok, h));

        // Each routed expert outputs silu(1) = 0.7310586 per channel. Six of
        // them are selected, each weighted 1/64 (uniform softmax). Shared = 0.
        let silu1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let w_each = 1.0f32 / config::N_ROUTED_EXPERTS as f32;
        let expect = 6.0 * w_each * silu1;
        assert!(
            (y.data[0] - expect).abs() < 1e-5,
            "{} != {expect}",
            y.data[0]
        );
        assert!(
            (y.data[1] - expect).abs() < 1e-5,
            "{} != {expect}",
            y.data[1]
        );
        Ok(())
    }

    #[test]
    fn moe_block_shared_contributes_at_weight_one() -> FocrResult<()> {
        let h = 2usize;
        let hidden = Mat::from_vec(1, h, vec![1.0, 1.0]);
        let gate = vec![0.0f32; config::N_ROUTED_EXPERTS * h];

        // Zero routed experts (down_proj=0) so routed term is 0.
        let zgate = vec![0.0f32; 2 * h];
        let zdown = vec![0.0f32; h * 2];
        let mk = || MlpWeights {
            gate_proj: &zgate,
            up_proj: &zgate,
            down_proj: &zdown,
            hidden: h,
            intermediate: 2,
        };
        let experts: Vec<MlpWeights> = (0..config::N_ROUTED_EXPERTS).map(|_| mk()).collect();

        // Shared expert = identity SwiGLU -> outputs silu(1) per channel, weight 1.
        let eye = vec![1.0f32, 0.0, 0.0, 1.0];
        let shared = MlpWeights {
            gate_proj: &eye,
            up_proj: &eye,
            down_proj: &eye,
            hidden: h,
            intermediate: 2,
        };
        let y = moe_block(&hidden, &gate, &experts, &shared, false, 1.0)?;
        let silu1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!((y.data[0] - silu1).abs() < 1e-5);
        assert!((y.data[1] - silu1).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn moe_block_rejects_wrong_expert_count() {
        let h = 2usize;
        let hidden = Mat::from_vec(1, h, vec![1.0, 1.0]);
        let gate = vec![0.0f32; config::N_ROUTED_EXPERTS * h];
        let eye = vec![1.0f32, 0.0, 0.0, 1.0];
        let w = MlpWeights {
            gate_proj: &eye,
            up_proj: &eye,
            down_proj: &eye,
            hidden: h,
            intermediate: 2,
        };
        let experts = vec![w]; // only 1, not 64
        let shared = w;
        assert!(moe_block(&hidden, &gate, &experts, &shared, false, 1.0).is_err());
    }

    #[test]
    fn config_constants_match_spec() {
        assert_eq!(config::N_ROUTED_EXPERTS, 64);
        assert_eq!(config::N_SHARED_EXPERTS, 2);
        assert_eq!(config::NUM_EXPERTS_PER_TOK, 6);
        assert_eq!(config::MOE_INTERMEDIATE_SIZE, 896);
        assert_eq!(config::SHARED_INTERMEDIATE_SIZE, 1792);
        assert_eq!(config::DENSE_INTERMEDIATE_SIZE, 6848);
        assert_eq!(config::HIDDEN_SIZE, 1280);
        assert_eq!(config::FIRST_K_DENSE_REPLACE, 1);
        const _: () = assert!(!config::NORM_TOPK_PROB);
        assert_eq!(config::ROUTED_SCALING_FACTOR, 1.0);
    }

    #[test]
    fn forward_shims_are_not_implemented_until_reader_lands() {
        let w = Weights::default();
        let x = Mat::from_vec(1, config::HIDDEN_SIZE, vec![0.0; config::HIDDEN_SIZE]);
        assert!(matches!(forward(&w, &x), Err(FocrError::NotImplemented(_))));
        assert!(matches!(
            dense_forward(&w, &x),
            Err(FocrError::NotImplemented(_))
        ));
    }
}
