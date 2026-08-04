//! MoE: greedy-softmax top-6 gate + grouped SiLU-gated experts, plus the
//! dense layer-0 MLP ([SPEC-074..077], PROPOSED_ARCHITECTURE.md §6.9).
//!
//! This module realizes three beads:
//!
//! * **P1-moe-router** — [`route`]: `logits = linear(hidden.f32, gate.f32)` ->
//!   `softmax(f32, dim=-1)` -> the pinned torch-2.10 CPU top-6 permutation. The
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
//! The public [`forward`] (MoE layers 1..11, takes the absolute `layer` index)
//! and [`dense_forward`] (the layer-0 dense MLP) shims pull the per-layer router
//! / 64 routed experts / fused shared expert (or the dense gate/up/down) straight
//! out of [`super::weights::Weights`] via `mat()` (BF16→f32 at the boundary) and
//! delegate to the fully-tested slice-typed primitives below ([`route`],
//! [`expert_mlp`], [`dense_mlp`], [`moe_block`]). The decoder driver
//! ([`super::decoder::forward`]) calls these per layer.

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
/// `indices[t][j]`). By default their order exactly reproduces the pinned
/// torch-2.10 macOS CPU `topk(sorted=False)` implementation. The
/// `FOCR_MOE_SCORE_ORDER` rollback restores the previous descending-score,
/// lower-expert-id tie policy.
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

/// Torch-2.10-compatible top-k router ([SPEC-077]).
///
/// `hidden` is `[n_tok, HIDDEN_SIZE]`; `gate` is the gate weight, row-major
/// `[N_ROUTED_EXPERTS, HIDDEN_SIZE] = [64, 1280]` (NEVER quantized). Steps:
///
/// 1. `logits = linear(hidden, gate)` -> `[n_tok, 64]` (f32).
/// 2. `scores = softmax(logits, dim=-1)` over the 64 experts (f32,
///    `scoring_func = 'softmax'`).
/// 3. Top-`k` (`k = NUM_EXPERTS_PER_TOK = 6`) via the pinned torch-2.10 CPU
///    `topk(scores, k, sorted=False)` permutation. For 64 experts, torch calls
///    libc++ `nth_element(begin, begin + 5, end)` and returns the first six
///    slots without sorting them.
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

        // 3. Reproduce the exact pinned torch-2.10 CPU slot permutation. This is
        //    deterministic Rust, not a call to the host standard library, so it
        //    is identical on every architecture we ship.
        let (chosen_idx, mut chosen_w) = select_router_experts(row, t)?;

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

const MOE_SCORE_ORDER_ENV: &str = "FOCR_MOE_SCORE_ORDER";

fn parse_moe_score_order(value: Option<&str>) -> bool {
    value.is_some_and(crate::quant::recipe::is_truthy)
}

fn moe_score_order_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        let value = std::env::var(MOE_SCORE_ORDER_ENV).ok();
        parse_moe_score_order(value.as_deref())
    })
}

#[derive(Clone, Copy, Debug)]
struct RouterCandidate {
    score: f32,
    expert: usize,
}

/// Exact comparator from torch-2.10 `TopKImpl.h` for `largest=true`.
///
/// NaNs precede every non-NaN value. Two NaNs, equal finite values, and signed
/// zero are comparator-equivalent; their deterministic permutation is then
/// entirely determined by the pinned libc++ partition below.
#[inline]
fn torch_topk_precedes(lhs: RouterCandidate, rhs: RouterCandidate) -> bool {
    (lhs.score.is_nan() && !rhs.score.is_nan()) || lhs.score > rhs.score
}

/// Stable median-of-three helper used by libc++'s `nth_element`.
fn libcxx_sort3(values: &mut [RouterCandidate], x: usize, y: usize, z: usize) -> usize {
    if !torch_topk_precedes(values[y], values[x]) {
        if !torch_topk_precedes(values[z], values[y]) {
            return 0;
        }
        values.swap(y, z);
        if torch_topk_precedes(values[y], values[x]) {
            values.swap(x, y);
            return 2;
        }
        return 1;
    }
    if torch_topk_precedes(values[z], values[y]) {
        values.swap(x, z);
        return 1;
    }
    values.swap(x, y);
    if torch_topk_precedes(values[z], values[y]) {
        values.swap(y, z);
        return 2;
    }
    1
}

fn libcxx_selection_sort(values: &mut [RouterCandidate], mut first: usize, last: usize) {
    while first + 1 < last {
        let mut best = first;
        for index in first + 1..last {
            if torch_topk_precedes(values[index], values[best]) {
                best = index;
            }
        }
        if best != first {
            values.swap(first, best);
        }
        first += 1;
    }
}

/// Safe Rust transcription of the libc++-15 `std::nth_element` algorithm whose
/// behavior is pinned by the torch-2.10 macOS arm64 oracle corpus.
///
/// Sources (retrieved 2026-07-10): LLVM `llvmorg-15.0.7`
/// `libcxx/include/__algorithm/{nth_element,sort}.h` and torch commit
/// `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`
/// `aten/src/ATen/native/TopKImpl.h`. The fixture records their SHA-256 hashes.
/// LLVM libc++ is licensed Apache-2.0 WITH LLVM-exception; this transcription
/// retains that provenance and contains no copied C++ or unsafe code.
/// The port deliberately retains the exact median-of-three, guard, partition,
/// and small-range selection-sort behavior because equal values expose those
/// otherwise-unspecified permutations.
fn libcxx_15_nth_element(values: &mut [RouterCandidate], nth: usize) {
    const SELECTION_SORT_LIMIT: usize = 7;
    let mut first = 0usize;
    let mut last = values.len();

    loop {
        if nth == last {
            return;
        }
        let len = last - first;
        match len {
            0 | 1 => return,
            2 => {
                if torch_topk_precedes(values[last - 1], values[first]) {
                    values.swap(first, last - 1);
                }
                return;
            }
            3 => {
                libcxx_sort3(values, first, first + 1, last - 1);
                return;
            }
            _ => {}
        }
        if len <= SELECTION_SORT_LIMIT {
            libcxx_selection_sort(values, first, last);
            return;
        }

        let mut middle = first + len / 2;
        let last_minus_one = last - 1;
        let mut swaps = libcxx_sort3(values, first, middle, last_minus_one);
        let mut up = first;
        let mut down = last_minus_one;

        if !torch_topk_precedes(values[up], values[middle]) {
            let guard_found = loop {
                down -= 1;
                if up == down {
                    break false;
                }
                if torch_topk_precedes(values[down], values[middle]) {
                    break true;
                }
            };
            if guard_found {
                values.swap(up, down);
                swaps += 1;
            } else {
                up += 1;
                down = last - 1;
                if !torch_topk_precedes(values[first], values[down]) {
                    loop {
                        if up == down {
                            return;
                        }
                        if torch_topk_precedes(values[first], values[up]) {
                            values.swap(up, down);
                            swaps += 1;
                            up += 1;
                            break;
                        }
                        up += 1;
                    }
                }
                if up == down {
                    return;
                }
                loop {
                    while !torch_topk_precedes(values[first], values[up]) {
                        up += 1;
                    }
                    loop {
                        down -= 1;
                        if !torch_topk_precedes(values[first], values[down]) {
                            break;
                        }
                    }
                    if up >= down {
                        break;
                    }
                    values.swap(up, down);
                    swaps += 1;
                    up += 1;
                }
                if nth < up {
                    return;
                }
                first = up;
                continue;
            }
        }

        up += 1;
        if up < down {
            loop {
                while torch_topk_precedes(values[up], values[middle]) {
                    up += 1;
                }
                loop {
                    down -= 1;
                    if torch_topk_precedes(values[down], values[middle]) {
                        break;
                    }
                }
                if up >= down {
                    break;
                }
                values.swap(up, down);
                swaps += 1;
                if middle == up {
                    middle = down;
                }
                up += 1;
            }
        }
        if up != middle && torch_topk_precedes(values[middle], values[up]) {
            values.swap(up, middle);
            swaps += 1;
        }
        if nth == up {
            return;
        }
        if swaps == 0 {
            if nth < up {
                down = first;
                middle = first;
                loop {
                    down += 1;
                    if down == up {
                        return;
                    }
                    if torch_topk_precedes(values[down], values[middle]) {
                        break;
                    }
                    middle = down;
                }
            } else {
                down = up;
                middle = up;
                loop {
                    down += 1;
                    if down == last {
                        return;
                    }
                    if torch_topk_precedes(values[down], values[middle]) {
                        break;
                    }
                    middle = down;
                }
            }
        }
        if nth < up {
            last = up;
        } else {
            first = up + 1;
        }
    }
}

fn torch_2_10_cpu_topk_unsorted(
    row: &[f32; config::N_ROUTED_EXPERTS],
) -> [RouterCandidate; config::NUM_EXPERTS_PER_TOK] {
    let mut candidates: [RouterCandidate; config::N_ROUTED_EXPERTS] =
        std::array::from_fn(|expert| RouterCandidate {
            score: row[expert],
            expert,
        });
    libcxx_15_nth_element(&mut candidates, config::NUM_EXPERTS_PER_TOK - 1);
    std::array::from_fn(|slot| candidates[slot])
}

fn score_ordered_topk(
    row: &[f32; config::N_ROUTED_EXPERTS],
) -> FocrResult<[RouterCandidate; config::NUM_EXPERTS_PER_TOK]> {
    let mut chosen = [RouterCandidate {
        score: 0.0,
        expert: 0,
    }; config::NUM_EXPERTS_PER_TOK];
    let mut taken = [false; config::N_ROUTED_EXPERTS];
    for (slot, selected) in chosen.iter_mut().enumerate() {
        let Some(first_available) = taken.iter().position(|is_taken| !is_taken) else {
            return Err(FocrError::Other(anyhow::anyhow!(
                "moe::route: no candidate remains for top-k slot {slot}"
            )));
        };
        let mut best = RouterCandidate {
            score: row[first_available],
            expert: first_available,
        };
        for (expert, &score) in row.iter().enumerate().skip(first_available + 1) {
            if taken[expert] {
                continue;
            }
            if score > best.score {
                best = RouterCandidate { score, expert };
            }
        }
        taken[best.expert] = true;
        *selected = best;
    }
    Ok(chosen)
}

fn select_router_experts_with_policy(
    row: &[f32],
    token_idx: usize,
    score_order_rollback: bool,
) -> FocrResult<(
    [usize; config::NUM_EXPERTS_PER_TOK],
    [f32; config::NUM_EXPERTS_PER_TOK],
)> {
    let row: &[f32; config::N_ROUTED_EXPERTS] = row.try_into().map_err(|_| {
        FocrError::Other(anyhow::anyhow!(
            "moe::route: token {token_idx} router row has {} scores; expected {}",
            row.len(),
            config::N_ROUTED_EXPERTS
        ))
    })?;
    if row.iter().any(|score| !score.is_finite()) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "moe::route: non-finite router score for token {token_idx}"
        )));
    }
    let selected = if score_order_rollback {
        score_ordered_topk(row)?
    } else {
        torch_2_10_cpu_topk_unsorted(row)
    };
    Ok((
        selected.map(|candidate| candidate.expert),
        selected.map(|candidate| candidate.score),
    ))
}

fn select_router_experts(
    row: &[f32],
    token_idx: usize,
) -> FocrResult<(
    [usize; config::NUM_EXPERTS_PER_TOK],
    [f32; config::NUM_EXPERTS_PER_TOK],
)> {
    select_router_experts_with_policy(row, token_idx, moe_score_order_enabled())
}

/// Return the routed-expert reduction order for every MoE execution path.
///
/// Default reduction follows the pinned torch/libc++ slot permutation exactly.
/// `FOCR_MOE_SCORE_ORDER` rolls selection back to descending score and reduction
/// back to ascending expert id, the deterministic policy present immediately
/// before the torch-order port. Both policies apply uniformly to f32/int8 and
/// prefill/batched/decode paths.
fn routed_reduction_slots_with_policy(
    indices: &[usize; config::NUM_EXPERTS_PER_TOK],
    score_order_rollback: bool,
) -> [usize; config::NUM_EXPERTS_PER_TOK] {
    let mut slots = std::array::from_fn(|slot| slot);
    if score_order_rollback {
        slots.sort_unstable_by_key(|&slot| indices[slot]);
    }
    slots
}

/// Combine six already-router-weighted expert rows with one shared reduction
/// primitive across every execution path.
///
/// At the production `[tokens, 6, hidden=1280]` geometry, torch-2.10 CPU
/// `sum(dim=1)` is bit-identical to a slot-ordered f32 left fold. The scalar
/// `[1, 6, 1]` reduction takes a different vector-reduction tree and is not the
/// model kernel; both behaviors are pinned in the oracle fixture. The default
/// slot permutation comes from [`torch_2_10_cpu_topk_unsorted`].
///
/// # Errors
/// [`FocrError::Other`] if a contribution row has the wrong hidden width.
pub(crate) fn combine_routed_rows(
    contributions: [&[f32]; config::NUM_EXPERTS_PER_TOK],
    indices: &[usize; config::NUM_EXPERTS_PER_TOK],
    out: &mut [f32],
) -> FocrResult<()> {
    combine_routed_rows_with_policy(contributions, indices, out, moe_score_order_enabled())
}

fn combine_routed_rows_with_policy(
    contributions: [&[f32]; config::NUM_EXPERTS_PER_TOK],
    indices: &[usize; config::NUM_EXPERTS_PER_TOK],
    out: &mut [f32],
    score_order_rollback: bool,
) -> FocrResult<()> {
    for (slot, row) in contributions.iter().enumerate() {
        if row.len() != out.len() {
            return Err(FocrError::Other(anyhow::anyhow!(
                "moe::combine_routed_rows: slot {slot} width {} != output width {}",
                row.len(),
                out.len()
            )));
        }
    }
    for slot in routed_reduction_slots_with_policy(indices, score_order_rollback) {
        for (dst, &value) in out.iter_mut().zip(contributions[slot].iter()) {
            *dst += value;
        }
    }
    Ok(())
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

/// `FOCR_FUSE_SWIGLU` (bd-1azu.54, Lever 2): fuse the SwiGLU activation
/// `silu(gate)·up` into the expert FFN GEMM epilogue — apply it in a SINGLE pass
/// over the gate GEMM output, in place, before the down-proj, instead of a
/// separate `nn::silu` pass followed by an elementwise multiply over the same
/// materialized buffer. DEFAULT OFF — unset reproduces the two-pass path
/// byte-for-byte. Read ONCE into a process-wide bool.
const FUSE_SWIGLU_ENV: &str = "FOCR_FUSE_SWIGLU";

fn fuse_swiglu_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os(FUSE_SWIGLU_ENV).is_some())
}

/// FUSED SwiGLU epilogue (FOCR_FUSE_SWIGLU): `gate[i] = silu(gate[i]) * up[i]` in
/// ONE pass over the gate GEMM output. Byte-for-byte identical to the separate
/// `nn::silu(gate)` then `gate[i] *= up[i]`: each lane computes `s/(1+e^-s)` — the
/// exact `nn::silu` scalar — then multiplies by `up[i]`, so nothing is
/// reassociated (the divide and the multiply stay distinct rounding steps, as in
/// the two-pass order).
#[inline]
fn swiglu_elemwise_fused(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        let s = *g;
        *g = (s / (1.0 + (-s).exp())) * u;
    }
}

/// SwiGLU MLP forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`
/// ([SPEC-075]). Shared by the dense layer-0 MLP, the fused shared expert, and
/// every routed expert (only `intermediate` differs).
///
/// `x` is `[n_tok, hidden]`; returns `[n_tok, hidden]`.
///
/// The `silu(gate)·up` activation runs as two passes (`nn::silu` then an
/// elementwise multiply, the default) or, under [`FUSE_SWIGLU_ENV`], as a single
/// fused epilogue pass ([`swiglu_elemwise_fused`]) — byte-for-byte identical.
///
/// # Errors
/// [`FocrError::Other`] on any dimension mismatch in `w`.
pub fn expert_mlp(x: &Mat, w: &MlpWeights<'_>) -> FocrResult<Mat> {
    // gate = x @ gate_proj.T  -> [n_tok, intermediate]
    let mut gate = linear_no_bias(x, w.gate_proj, w.intermediate, w.hidden)?;
    let fused = fuse_swiglu_enabled();
    if !fused {
        // Default: silu in place FIRST (today's exact two-pass ordering).
        nn::silu(&mut gate);
    }
    // up = x @ up_proj.T            -> [n_tok, intermediate]
    let up = linear_no_bias(x, w.up_proj, w.intermediate, w.hidden)?;
    if gate.data.len() != up.data.len() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "moe::expert_mlp: gate/up shape mismatch ({} vs {})",
            gate.data.len(),
            up.data.len()
        )));
    }
    if fused {
        // FOCR_FUSE_SWIGLU: silu(gate)·up folded into one epilogue pass.
        swiglu_elemwise_fused(&mut gate.data, &up.data);
    } else {
        // elementwise silu(gate) * up (the second of the two default passes).
        for (g, &u) in gate.data.iter_mut().zip(up.data.iter()) {
            *g *= u;
        }
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
/// Like the reference's `moe_infer`, this groups only selected tokens by expert,
/// executes one compact GEMM per active expert, restores outputs to their
/// original top-k slots, and then performs the pinned slot-order reduction.
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

    // Build each routed contribution through expert-grouped GEMMs, then restore
    // the original top-k slot axis before the shared combine primitive. This is
    // the reference's `new_x[idxs] = outs; view(..., 6, hidden); sum(dim=1)`
    // structure and preserves the pinned f32 order without giving up grouping.
    let mut out = Mat::zeros(n_tok, h);
    let route_rows = checked_shape_mul(
        "moe::moe_block",
        n_tok,
        config::NUM_EXPERTS_PER_TOK,
        "n_tok*top_k",
    )?;
    let contribution_len =
        checked_shape_mul("moe::moe_block", route_rows, h, "n_tok*top_k*hidden")?;
    let mut contributions = vec![0.0f32; contribution_len];

    // expert -> list of (token, original route slot, weight)
    let mut per_expert: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); config::N_ROUTED_EXPERTS];
    for t in 0..n_tok {
        for j in 0..config::NUM_EXPERTS_PER_TOK {
            let e = routing.indices[t][j];
            let w = routing.weights[t][j];
            per_expert[e].push((t, j, w));
        }
    }

    for (e, members) in per_expert.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        // Gather the rows that selected expert e into a compact [m, h] activation.
        let m = members.len();
        let mut sub = Mat::zeros(m, h);
        for (r, &(t, _slot, _w)) in members.iter().enumerate() {
            sub.row_mut(r).copy_from_slice(hidden.row(t));
        }
        let y = expert_mlp(&sub, &experts[e])?; // [m, h]
        // Restore each expert output to its original top-k slot, scaled by the
        // matching router weight. The reduction happens only after all slots
        // are populated.
        for (r, &(t, slot, w)) in members.iter().enumerate() {
            let yr = y.row(r);
            let base = (t * config::NUM_EXPERTS_PER_TOK + slot) * h;
            let dst = &mut contributions[base..base + h];
            for (value, &expert_value) in dst.iter_mut().zip(yr.iter()) {
                *value = w * expert_value;
            }
        }
    }

    for t in 0..n_tok {
        let token_base = t * config::NUM_EXPERTS_PER_TOK * h;
        let rows: [&[f32]; config::NUM_EXPERTS_PER_TOK] = std::array::from_fn(|slot| {
            let base = token_base + slot * h;
            &contributions[base..base + h]
        });
        combine_routed_rows(rows, &routing.indices[t], out.row_mut(t))?;
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

// ── bd-1azu.6: batched MoE decode dispatch (Phase-6 continuous-batch spine) ──

/// Split a `[B, hidden]` batched MoE result into B per-stream `[1, hidden]`
/// outputs in input (stream-row) order — the "returns B outputs" half of the
/// batched dispatch. Pure reshaping; each `combined.row(s)` is copied, so the
/// returned [`Mat`]s own their data and alias nothing.
fn split_stream_rows(combined: &Mat) -> Vec<Mat> {
    let h = combined.cols;
    (0..combined.rows)
        .map(|s| Mat::from_vec(1, h, combined.row(s).to_vec()))
        .collect()
}

/// Batched MoE decode dispatch over B in-flight page-streams (bd-1azu.6 — the
/// MoE stage of the Phase-6 continuous-batch decode spine, bd-1azu).
///
/// `hidden` is the B streams' decode hidden states stacked row-major as
/// `[B, HIDDEN]` (row `s` is stream `s`'s single decode token). Runs ONE batched
/// MoE pass over the whole stack via the tested [`moe_block`] — a SINGLE f32
/// router GEMM over all B rows, the per-expert counting-sort grouping already
/// inside [`moe_block`] (each ACTIVE expert runs its SwiGLU GEMM ONCE over the
/// grouped rows that selected it, instead of B separate per-stream passes), the
/// per-token router-weighted combine scattered back to stream order, and the
/// shared experts — then splits the `[B, HIDDEN]` result into B per-stream
/// `[1, HIDDEN]` outputs ([`split_stream_rows`]).
///
/// ## Losslessness (Doctrine #1 — the bd-1azu parity invariant)
///
/// Output `s` is **byte-for-byte identical** to running stream `s` alone through
/// [`moe_block`] (`batched_moe_block(stack)[s] == moe_block(row s)`), so the
/// grouping is the real Lever B throughput win **and** bit-exact:
///
/// * the router GEMM ([`linear_no_bias`] -> [`nn::matmul`]) is M-independent —
///   each output row's per-key reduction order is fixed regardless of how many
///   rows are stacked (the same property the int8 spine proves in
///   `tests/batched_igemm_parity.rs`, bd-1azu.2), and `softmax_rows` + the greedy
///   top-k are per-row;
/// * each expert's [`expert_mlp`] is likewise M-independent, so a stream's
///   grouped row yields the same output whether it shares the expert's group with
///   other streams' rows or runs alone;
/// * the per-token weighted combine restores the pinned torch top-k slot axis
///   and uses [`combine_routed_rows`] — identical batched vs. standalone — then
///   the shared add follows, so the f32 reduction order is preserved exactly.
///
/// The per-stream parity is the executing proof in `tests/batched_moe_parity.rs`.
/// Wiring this into the decode driver is gated by the `FOCR_BATCHED_MOE`
/// kill-switch at the call site (default OFF); the function itself is a pure,
/// lossless API.
///
/// # Errors
/// [`FocrError::Other`] on a wrong expert count or any dimension mismatch,
/// propagated from [`moe_block`].
pub fn batched_moe_block(
    hidden: &Mat,
    gate: &[f32],
    experts: &[MlpWeights<'_>],
    shared: &MlpWeights<'_>,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> FocrResult<Vec<Mat>> {
    let combined = moe_block(
        hidden,
        gate,
        experts,
        shared,
        norm_topk_prob,
        routed_scaling_factor,
    )?;
    Ok(split_stream_rows(&combined))
}

/// [`batched_moe_block`] with the pinned-config gate defaults ([SPEC-013/077]).
///
/// # Errors
/// Propagates [`batched_moe_block`].
pub fn batched_moe_block_default(
    hidden: &Mat,
    gate: &[f32],
    experts: &[MlpWeights<'_>],
    shared: &MlpWeights<'_>,
) -> FocrResult<Vec<Mat>> {
    batched_moe_block(
        hidden,
        gate,
        experts,
        shared,
        config::NORM_TOPK_PROB,
        config::ROUTED_SCALING_FACTOR,
    )
}

// ── `Weights`-backed shims (now wired to the safetensors/`.focrq` accessors) ──

/// Run the MoE block (layers 1..11) over a layer's `post_attention_layernorm`'d
/// hidden states, pulling the per-layer router / 64 routed experts / fused
/// shared expert straight out of [`Weights`] ([SPEC-076/077]).
///
/// Tensor names (verified against the real checkpoint, `forward_wiring_intel.md`):
/// router `model.layers.{layer}.mlp.gate.weight` `[64, 1280]`; routed expert `e`
/// `model.layers.{layer}.mlp.experts.{e}.{gate,up,down}_proj.weight`
/// (`intermediate = 896`); fused shared expert
/// `model.layers.{layer}.mlp.shared_experts.{gate,up,down}_proj.weight`
/// (singular `shared_experts`, `intermediate = 1792`). Delegates to the tested
/// [`moe_block_default`] (pinned gate: `norm_topk_prob = false`,
/// `routed_scaling_factor = 1.0`).
///
/// `layer` is the **absolute** decoder layer index (1..=11); the dense layer 0
/// must use [`dense_forward`] instead.
///
/// # Errors
/// [`FocrError::Other`] if any tensor is absent / wrong-shaped, or on a kernel
/// dimension mismatch (propagated from [`moe_block`]).
pub fn forward(weights: &Weights, hidden: &Mat, layer: usize) -> FocrResult<Mat> {
    let prefix = format!("model.layers.{layer}.mlp");
    // Router gate [N_ROUTED_EXPERTS, HIDDEN_SIZE] = [64, 1280] (NEVER quantized).
    let gate = weights.mat(&format!("{prefix}.gate.weight"))?;

    // Load all 64 routed experts as owned f32 Mats (kept alive while the
    // borrowing MlpWeights slices reference them).
    let mut routed: Vec<(Mat, Mat, Mat)> = Vec::with_capacity(config::N_ROUTED_EXPERTS);
    for e in 0..config::N_ROUTED_EXPERTS {
        let g = weights.mat(&format!("{prefix}.experts.{e}.gate_proj.weight"))?;
        let u = weights.mat(&format!("{prefix}.experts.{e}.up_proj.weight"))?;
        let d = weights.mat(&format!("{prefix}.experts.{e}.down_proj.weight"))?;
        routed.push((g, u, d));
    }
    let experts: Vec<MlpWeights<'_>> = routed
        .iter()
        .map(|(g, u, d)| MlpWeights {
            gate_proj: &g.data,
            up_proj: &u.data,
            down_proj: &d.data,
            hidden: config::HIDDEN_SIZE,
            intermediate: config::MOE_INTERMEDIATE_SIZE,
        })
        .collect();

    // Fused shared expert (intermediate 2 * 896 = 1792).
    let sg = weights.mat(&format!("{prefix}.shared_experts.gate_proj.weight"))?;
    let su = weights.mat(&format!("{prefix}.shared_experts.up_proj.weight"))?;
    let sd = weights.mat(&format!("{prefix}.shared_experts.down_proj.weight"))?;
    let shared = MlpWeights {
        gate_proj: &sg.data,
        up_proj: &su.data,
        down_proj: &sd.data,
        hidden: config::HIDDEN_SIZE,
        intermediate: config::SHARED_INTERMEDIATE_SIZE,
    };

    moe_block_default(hidden, &gate.data, &experts, &shared)
}

/// Run the dense layer-0 MLP over a layer's `post_attention_layernorm`'d hidden
/// states ([SPEC-074/075]).
///
/// `first_k_dense_replace = 1`, so layer 0 is the ONLY dense MLP; this shim is
/// hardcoded to it. Tensor names: `model.layers.0.mlp.{gate,up,down}_proj.weight`
/// (`intermediate = 6848`). Delegates to the tested [`dense_mlp`].
///
/// # Errors
/// [`FocrError::Other`] if any tensor is absent / wrong-shaped, or on a kernel
/// dimension mismatch (propagated from [`dense_mlp`]).
pub fn dense_forward(weights: &Weights, hidden: &Mat) -> FocrResult<Mat> {
    let prefix = "model.layers.0.mlp";
    let g = weights.mat(&format!("{prefix}.gate_proj.weight"))?;
    let u = weights.mat(&format!("{prefix}.up_proj.weight"))?;
    let d = weights.mat(&format!("{prefix}.down_proj.weight"))?;
    let w = MlpWeights {
        gate_proj: &g.data,
        up_proj: &u.data,
        down_proj: &d.data,
        hidden: config::HIDDEN_SIZE,
        intermediate: config::DENSE_INTERMEDIATE_SIZE,
    };
    dense_mlp(hidden, &w)
}

/// `Weights`-backed batched MoE dispatch (layers 1..11) over B stacked stream
/// hidden states — the decode-spine entry point that loads the per-layer router /
/// 64 routed experts / fused shared expert ONCE and runs them grouped over all B
/// streams (bd-1azu.6).
///
/// `hidden` is `[B, HIDDEN]` (one decode token per in-flight stream); `layer` is
/// the **absolute** decoder layer index (1..=11; dense layer 0 uses
/// [`dense_forward`]). Delegates to [`forward`] (identical tensor wiring and the
/// tested [`moe_block_default`]) and splits the `[B, HIDDEN]` result into B
/// per-stream `[1, HIDDEN]` outputs, so `batched_forward(w, stack, L)[s]` is
/// byte-for-byte `forward(w, row s, L)` (the M-independence argument on
/// [`batched_moe_block`]). Wiring into the decode driver is gated by the
/// `FOCR_BATCHED_MOE` kill-switch at the call site (default OFF).
///
/// # Errors
/// [`FocrError::FormatMismatch`] if any tensor is absent / wrong-shaped, or
/// [`FocrError::Other`] on a kernel dimension mismatch — propagated from
/// [`forward`].
pub fn batched_forward(weights: &Weights, hidden: &Mat, layer: usize) -> FocrResult<Vec<Mat>> {
    let combined = forward(weights, hidden, layer)?;
    Ok(split_stream_rows(&combined))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn moe_oracle_fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../../tests/fixtures/moe_torch_2_10_cpu.json"))
            .expect("valid pinned torch MoE oracle fixture")
    }

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = self.0;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^ (value >> 31)
        }
    }

    fn fixture_hex_u64(value: &serde_json::Value, field: &str) -> u64 {
        u64::from_str_radix(value[field].as_str().expect("hex fixture field"), 16)
            .expect("valid fixture u64 hex")
    }

    fn digest_hex(hasher: Sha256) -> String {
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    const POLICY_CHILD_ENV: &str = "FOCR_MOE_POLICY_TEST_CASE";

    fn fixture_usize_array(
        value: &serde_json::Value,
        field: &str,
    ) -> [usize; config::NUM_EXPERTS_PER_TOK] {
        value[field]
            .as_array()
            .expect("fixture usize array")
            .iter()
            .map(|item| item.as_u64().expect("fixture usize") as usize)
            .collect::<Vec<_>>()
            .try_into()
            .expect("fixture top-k width")
    }

    fn policy_child_case() -> Option<(&'static str, bool)> {
        match std::env::var(POLICY_CHILD_ENV).ok().as_deref() {
            None => None,
            Some("unset") => Some(("unset", false)),
            Some("zero") => Some(("zero", false)),
            Some("one") => Some(("one", true)),
            Some(other) => {
                eprintln!("unknown MoE policy subprocess case {other:?}");
                None
            }
        }
    }

    fn policy_route_fixture(
        rollback: bool,
    ) -> (Mat, Vec<f32>, Routing, [usize; config::NUM_EXPERTS_PER_TOK]) {
        let fixture = moe_oracle_fixture();
        let unique = &fixture["cases"][0];
        let scores = unique["scores_f32_bits"]
            .as_array()
            .expect("fixture score bits")
            .iter()
            .map(|bits| f32::from_bits(bits.as_u64().expect("fixture f32 bits") as u32))
            .collect::<Vec<_>>();
        assert_eq!(scores.len(), config::N_ROUTED_EXPERTS);

        let mut hidden_data = vec![0.0f32; config::HIDDEN_SIZE];
        hidden_data[0] = 1.0;
        let hidden = Mat::from_vec(1, config::HIDDEN_SIZE, hidden_data);
        let mut gate = vec![0.0f32; config::N_ROUTED_EXPERTS * config::HIDDEN_SIZE];
        for (expert, score) in scores.into_iter().enumerate() {
            gate[expert * config::HIDDEN_SIZE] = score;
        }

        let routing = route_default(&hidden, &gate).expect("public route_default succeeds");
        let expected = fixture_usize_array(
            unique,
            if rollback {
                "torch_sorted_indices"
            } else {
                "torch_unsorted_indices"
            },
        );
        assert_eq!(routing.indices, vec![expected]);
        (hidden, gate, routing, expected)
    }

    fn fold_contributions(
        contributions: &[Vec<f32>; config::NUM_EXPERTS_PER_TOK],
        indices: &[usize; config::NUM_EXPERTS_PER_TOK],
        ascending_expert: bool,
    ) -> Vec<f32> {
        let mut slots: [usize; config::NUM_EXPERTS_PER_TOK] = std::array::from_fn(|slot| slot);
        if ascending_expert {
            slots.sort_unstable_by_key(|&slot| indices[slot]);
        }
        let mut out = vec![0.0f32; config::HIDDEN_SIZE];
        for slot in slots {
            for (dst, &value) in out.iter_mut().zip(contributions[slot].iter()) {
                *dst += value;
            }
        }
        out
    }

    fn assert_public_f32_policy_paths(rollback: bool) -> FocrResult<()> {
        const TARGETS: [f32; config::NUM_EXPERTS_PER_TOK] =
            [16_777_216.0, 1.0, -16_777_216.0, 1.0, 1.0, 1.0];
        let (hidden, gate, routing, indices) = policy_route_fixture(rollback);
        let fixture = moe_oracle_fixture();
        let combine = &fixture["weighted_combine"];
        let combine_indices = fixture_usize_array(combine, "slot_expert_indices");
        let scalar_contributions = combine["contribution_f32_bits"]
            .as_array()
            .expect("combine contribution bits")
            .iter()
            .map(|bits| f32::from_bits(bits.as_u64().expect("combine f32 bits") as u32))
            .collect::<Vec<_>>();
        let direct_contributions: [Vec<f32>; config::NUM_EXPERTS_PER_TOK] =
            std::array::from_fn(|slot| vec![scalar_contributions[slot]; config::HIDDEN_SIZE]);
        let direct_rows: [&[f32]; config::NUM_EXPERTS_PER_TOK] =
            std::array::from_fn(|slot| direct_contributions[slot].as_slice());
        let mut direct = vec![0.0f32; config::HIDDEN_SIZE];
        combine_routed_rows(direct_rows, &combine_indices, &mut direct)?;
        let expected_direct_bits = combine[if rollback {
            "rust_ascending_left_fold_f32_bits"
        } else {
            "torch_production_shape_1x6x1280_sum_dim_1_f32_bits"
        }]
        .as_u64()
        .expect("combine expected bits") as u32;
        assert!(
            direct
                .iter()
                .all(|value| value.to_bits() == expected_direct_bits)
        );

        let mut gate_proj = vec![0.0f32; config::HIDDEN_SIZE];
        gate_proj[0] = 1.0;
        let up_proj = gate_proj.clone();
        let silu_one = 1.0f32 / (1.0 + (-1.0f32).exp());
        let mut downs = vec![vec![0.0f32; config::HIDDEN_SIZE]; config::N_ROUTED_EXPERTS];
        for (slot, &expert) in indices.iter().enumerate() {
            let expert_output = TARGETS[slot] / routing.weights[0][slot];
            downs[expert].fill(expert_output / silu_one);
        }
        let experts = downs
            .iter()
            .map(|down_proj| MlpWeights {
                gate_proj: &gate_proj,
                up_proj: &up_proj,
                down_proj,
                hidden: config::HIDDEN_SIZE,
                intermediate: 1,
            })
            .collect::<Vec<_>>();
        let shared_gate = vec![0.0f32; config::HIDDEN_SIZE];
        let shared_down = vec![0.0f32; config::HIDDEN_SIZE];
        let shared = MlpWeights {
            gate_proj: &shared_gate,
            up_proj: &shared_gate,
            down_proj: &shared_down,
            hidden: config::HIDDEN_SIZE,
            intermediate: 1,
        };

        let mut contributions: [Vec<f32>; config::NUM_EXPERTS_PER_TOK] =
            std::array::from_fn(|_| vec![0.0; config::HIDDEN_SIZE]);
        for (slot, &expert) in indices.iter().enumerate() {
            let expert_out = expert_mlp(&hidden, &experts[expert])?;
            for (dst, &value) in contributions[slot].iter_mut().zip(expert_out.data.iter()) {
                *dst = routing.weights[0][slot] * value;
            }
        }
        let expected = fold_contributions(&contributions, &indices, rollback);
        let alternate = fold_contributions(&contributions, &indices, !rollback);
        assert_ne!(
            expected[0].to_bits(),
            alternate[0].to_bits(),
            "fixture must distinguish slot and ascending-expert reductions"
        );

        let standalone = moe_block_default(&hidden, &gate, &experts, &shared)?;
        assert_eq!(standalone.data, expected, "public f32 MoE policy path");

        let mut batched_data = hidden.data.clone();
        batched_data.extend_from_slice(&hidden.data);
        let batched_hidden = Mat::from_vec(2, config::HIDDEN_SIZE, batched_data);
        let batched = batched_moe_block_default(&batched_hidden, &gate, &experts, &shared)?;
        assert_eq!(batched.len(), 2);
        for row in batched {
            assert_eq!(row.data, expected, "public batched f32 MoE policy path");
        }
        Ok(())
    }

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

    /// FOCR_FUSE_SWIGLU (Lever 2, bd-1azu.54): the fused one-pass `silu(gate)·up`
    /// epilogue must reproduce, BYTE-FOR-BYTE, the default two-pass path —
    /// production `nn::silu` over the gate buffer THEN an elementwise multiply by
    /// `up`. `inter` is deliberately not a tidy width so the activation tail is
    /// exercised; the row spans negatives/positives (the silu sigmoid both ways).
    #[test]
    fn fused_swiglu_epilogue_is_byte_identical_to_two_pass() {
        let inter = 257usize;
        let gate0: Vec<f32> = (0..inter)
            .map(|i| (i as f32 * 0.17).sin() * 4.0 - 1.3)
            .collect();
        let up: Vec<f32> = (0..inter)
            .map(|i| (i as f32 * 0.23).cos() * 2.0 + 0.5)
            .collect();
        let bits = |s: &[f32]| s.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();

        // Default two-pass: the production `nn::silu`, then elementwise multiply.
        let mut sep = Mat::from_vec(1, inter, gate0.clone());
        nn::silu(&mut sep);
        for (g, &u) in sep.data.iter_mut().zip(up.iter()) {
            *g *= u;
        }

        // Fused one-pass epilogue (the production helper).
        let mut fused = gate0.clone();
        swiglu_elemwise_fused(&mut fused, &up);

        assert_eq!(bits(&sep.data), bits(&fused), "swiglu fused != two-pass");
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
        // `sorted=False` exposes the pinned partition permutation, but the
        // selected set is exactly the six experts we boosted.
        let mut selected = r.indices[0];
        selected.sort_unstable();
        assert_eq!(selected, want);

        // Weights are raw softmax probs (not renormalized). Recompute softmax
        // over all 64 logits and compare the 6 selected.
        let mut denom = 0.0f64;
        for e in 0..n {
            denom += (gate[e * h] as f64).exp();
        }
        for (k, &e) in r.indices[0].iter().enumerate() {
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
        assert_eq!(
            r.indices[0]
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            6
        );
        Ok(())
    }

    #[test]
    fn route_rejects_nonfinite_router_scores_without_panicking() {
        let h = config::HIDDEN_SIZE;
        let n = config::N_ROUTED_EXPERTS;
        let mut gate = vec![0.0f32; n * h];
        for expert in 0..n {
            gate[expert * h] = 1.0;
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut hid = vec![0.0f32; h];
            hid[0] = value;
            let hidden = Mat::from_vec(1, h, hid);
            let result = route_default(&hidden, &gate);
            assert!(matches!(
                &result,
                Err(err) if err.to_string().contains("non-finite router score for token 0")
            ));
        }
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

    #[test]
    fn moe_score_order_parser_is_truthy_only_and_fail_closed() {
        for value in ["1", "true", "on", "yes", " TRUE ", "On"] {
            assert!(parse_moe_score_order(Some(value)), "{value:?}");
        }
        for value in [
            "", "0", "false", "off", "no", "default", "2", "enabled", "garbage",
        ] {
            assert!(!parse_moe_score_order(Some(value)), "{value:?}");
        }
        assert!(!parse_moe_score_order(None));
    }

    #[test]
    fn moe_policy_subprocess_probe_f32() -> FocrResult<()> {
        let Some((case, rollback)) = policy_child_case() else {
            return Ok(());
        };
        assert_eq!(moe_score_order_enabled(), rollback);
        assert_public_f32_policy_paths(rollback)?;
        eprintln!("FOCR_MOE_POLICY_PROBE_F32={case}");
        Ok(())
    }

    #[test]
    fn moe_policy_env_subprocess_matrix() {
        let executable = std::env::current_exe().expect("current Rust test executable");
        for (case, value) in [("unset", None), ("zero", Some("0")), ("one", Some("1"))] {
            let mut command = std::process::Command::new(&executable);
            command
                .arg("moe_policy_subprocess_probe")
                .arg("--nocapture")
                .arg("--test-threads=1")
                .env(POLICY_CHILD_ENV, case);
            match value {
                Some(value) => {
                    command.env(MOE_SCORE_ORDER_ENV, value);
                }
                None => {
                    command.env_remove(MOE_SCORE_ORDER_ENV);
                }
            }
            let output = command.output().expect("spawn isolated MoE policy test");
            let transcript = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                output.status.success(),
                "MoE policy subprocess {case} failed:\n{transcript}"
            );
            assert!(
                transcript.contains(&format!("FOCR_MOE_POLICY_PROBE_F32={case}")),
                "f32 child probe did not execute for {case}:\n{transcript}"
            );
            assert!(
                transcript.contains(&format!("FOCR_MOE_POLICY_PROBE_INT8={case}")),
                "int8 child probe did not execute for {case}:\n{transcript}"
            );
        }
    }

    #[test]
    fn torch_2_10_topk_fixture_matches_exact_slot_permutation() -> FocrResult<()> {
        let fixture = moe_oracle_fixture();
        let cases = fixture["cases"]
            .as_array()
            .expect("fixture cases must be an array");

        let unique = &cases[0];
        let unique_scores = unique["scores_f32_bits"]
            .as_array()
            .expect("unique scores bits")
            .iter()
            .map(|bits| f32::from_bits(bits.as_u64().expect("u32 bits") as u32))
            .collect::<Vec<_>>();
        let (local_idx, local_values) =
            select_router_experts_with_policy(&unique_scores, 0, false)?;
        let torch_unsorted = unique["torch_unsorted_indices"]
            .as_array()
            .expect("torch unsorted indices")
            .iter()
            .map(|index| index.as_u64().expect("usize index") as usize)
            .collect::<Vec<_>>();
        let torch_sorted = unique["torch_sorted_indices"]
            .as_array()
            .expect("torch sorted indices")
            .iter()
            .map(|index| index.as_u64().expect("usize index") as usize)
            .collect::<Vec<_>>();

        assert_eq!(local_idx.as_slice(), torch_unsorted.as_slice());
        assert_ne!(local_idx.as_slice(), torch_sorted.as_slice());
        for (slot, &expert) in local_idx.iter().enumerate() {
            assert_eq!(
                local_values[slot].to_bits(),
                unique_scores[expert].to_bits()
            );
        }

        let tied = &cases[1];
        let tied_scores = (0..config::N_ROUTED_EXPERTS)
            .map(|index| ((index * 17) % 7) as f32)
            .collect::<Vec<_>>();
        let (tied_idx, tied_values) = select_router_experts_with_policy(&tied_scores, 0, false)?;
        assert!(tied_values.iter().all(|value| *value == 6.0));
        let torch_tied = tied["torch_unsorted_indices"]
            .as_array()
            .expect("torch tied indices")
            .iter()
            .map(|index| index.as_u64().expect("usize index") as usize)
            .collect::<Vec<_>>();
        assert_eq!(tied_idx.as_slice(), torch_tied.as_slice());

        let (rollback_unique, _) = select_router_experts_with_policy(&unique_scores, 0, true)?;
        assert_eq!(rollback_unique.as_slice(), torch_sorted.as_slice());
        let (rollback_tied, _) = select_router_experts_with_policy(&tied_scores, 0, true)?;
        assert_eq!(rollback_tied, [2, 9, 16, 23, 30, 37]);
        Ok(())
    }

    #[test]
    fn torch_2_10_topk_matches_2048_case_oracle_corpus() {
        let fixture = moe_oracle_fixture();
        let corpus = &fixture["topk_corpus"];
        let case_count = corpus["case_count"].as_u64().expect("case count") as usize;
        assert!(
            case_count >= 512,
            "oracle corpus must contain hundreds of cases"
        );
        let mut rng = SplitMix64(fixture_hex_u64(corpus, "seed_hex"));
        let spots = corpus["spot_indices"].as_object().expect("spot indices");
        let mut output = Vec::with_capacity(case_count * config::NUM_EXPERTS_PER_TOK);

        for case_index in 0..case_count {
            let mode = case_index % 8;
            let scores: [f32; config::N_ROUTED_EXPERTS] = std::array::from_fn(|expert| {
                let random = rng.next();
                match mode {
                    0 => f32::from_bits(0x3f00_0000 | (random as u32 & 0x007f_ffff)),
                    1 => (random % 2) as f32,
                    2 => (random % 3) as f32,
                    3 => (random % 7) as f32,
                    4 => match random % 11 {
                        0 => f32::INFINITY,
                        1 => f32::NEG_INFINITY,
                        2 => -0.0,
                        _ => (random % 19) as f32 - 9.0,
                    },
                    5 if random.is_multiple_of(13) => f32::NAN,
                    5 => (random % 17) as f32,
                    6 => ((expert * 17 + case_index * 13) % 64) as f32,
                    7 if (expert + case_index).is_multiple_of(9) => 100.0,
                    7 => (random % 5) as f32 - 2.0,
                    _ => unreachable!(),
                }
            });
            let selected = torch_2_10_cpu_topk_unsorted(&scores);
            let indices = selected.map(|candidate| candidate.expert);
            output.extend(indices.map(|expert| expert as u8));

            if let Some(expected) = spots.get(&case_index.to_string()) {
                let expected = expected
                    .as_array()
                    .expect("spot array")
                    .iter()
                    .map(|index| index.as_u64().expect("spot index") as usize)
                    .collect::<Vec<_>>();
                assert_eq!(indices.as_slice(), expected.as_slice(), "case {case_index}");
            }
        }

        assert_eq!(
            output.len(),
            corpus["torch_output_bytes"].as_u64().expect("output bytes") as usize
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&output)),
            corpus["torch_output_sha256"].as_str().expect("topk sha256")
        );
    }

    #[test]
    fn production_six_term_combine_matches_256_case_torch_oracle() -> FocrResult<()> {
        let fixture = moe_oracle_fixture();
        let corpus = &fixture["reduction_corpus"];
        let case_count = corpus["case_count"].as_u64().expect("case count") as usize;
        let hidden = corpus["hidden"].as_u64().expect("hidden") as usize;
        assert_eq!(hidden, config::HIDDEN_SIZE);
        let mut rng = SplitMix64(fixture_hex_u64(corpus, "seed_hex"));
        let mut hasher = Sha256::new();
        let indices = [0, 1, 2, 3, 4, 5];
        const CANCELLATION: [f32; 6] = [16_777_216.0, 1.0, -16_777_216.0, 1.0, 1.0, 1.0];

        for case_index in 0..case_count {
            let mode = case_index % 4;
            let mut contributions: [Vec<f32>; config::NUM_EXPERTS_PER_TOK] =
                std::array::from_fn(|_| vec![0.0; hidden]);
            // Keep the channel-major RNG order byte-identical to the pinned
            // Python fixture generator; transposing this loop changes the oracle.
            #[allow(clippy::needless_range_loop)]
            for channel in 0..hidden {
                if mode == 1 {
                    let rotation = (rng.next() % 6) as usize;
                    for slot in 0..config::NUM_EXPERTS_PER_TOK {
                        contributions[slot][channel] = CANCELLATION[(slot + rotation) % 6];
                    }
                    continue;
                }
                for slot in 0..config::NUM_EXPERTS_PER_TOK {
                    let random = rng.next();
                    contributions[slot][channel] = match mode {
                        0 => {
                            let sign = ((random >> 63) as u32) << 31;
                            let exponent = (125 + ((random >> 60) & 3) as u32) << 23;
                            f32::from_bits(sign | exponent | (random as u32 & 0x007f_ffff))
                        }
                        2 => {
                            let sign = ((random >> 63) as u32) << 31;
                            let exponent = (1 + ((random >> 32) % 253) as u32) << 23;
                            f32::from_bits(sign | exponent | (random as u32 & 0x007f_ffff))
                        }
                        3 => (random % 17) as f32 - 8.0,
                        _ => unreachable!(),
                    };
                }
            }
            let rows: [&[f32]; config::NUM_EXPERTS_PER_TOK] =
                std::array::from_fn(|slot| contributions[slot].as_slice());
            let mut out = vec![0.0f32; hidden];
            combine_routed_rows_with_policy(rows, &indices, &mut out, false)?;
            for value in out {
                hasher.update(value.to_le_bytes());
            }
        }

        assert_eq!(
            digest_hex(hasher),
            corpus["torch_output_sha256"]
                .as_str()
                .expect("reduction sha256")
        );
        Ok(())
    }

    #[test]
    fn production_combine_fixture_pins_scalar_tree_difference_and_rollback() -> FocrResult<()> {
        let fixture = moe_oracle_fixture();
        let combine = &fixture["weighted_combine"];
        let indices: [usize; config::NUM_EXPERTS_PER_TOK] = combine["slot_expert_indices"]
            .as_array()
            .expect("combine expert indices")
            .iter()
            .map(|index| index.as_u64().expect("usize index") as usize)
            .collect::<Vec<_>>()
            .try_into()
            .expect("six expert indices");
        let scalar_contributions = combine["contribution_f32_bits"]
            .as_array()
            .expect("combine contribution bits")
            .iter()
            .map(|bits| f32::from_bits(bits.as_u64().expect("u32 bits") as u32))
            .collect::<Vec<_>>();
        let contributions: [Vec<f32>; config::NUM_EXPERTS_PER_TOK] =
            std::array::from_fn(|slot| vec![scalar_contributions[slot]; config::HIDDEN_SIZE]);
        let rows: [&[f32]; config::NUM_EXPERTS_PER_TOK] =
            std::array::from_fn(|slot| contributions[slot].as_slice());
        let mut local = vec![0.0f32; config::HIDDEN_SIZE];
        combine_routed_rows_with_policy(rows, &indices, &mut local, false)?;
        assert_eq!(
            local[0].to_bits(),
            combine["torch_production_shape_1x6x1280_sum_dim_1_f32_bits"]
                .as_u64()
                .expect("Rust fold bits") as u32
        );
        assert!(
            local
                .iter()
                .all(|value| value.to_bits() == 3.0f32.to_bits())
        );

        let rows: [&[f32]; config::NUM_EXPERTS_PER_TOK] =
            std::array::from_fn(|slot| contributions[slot].as_slice());
        let mut rollback = vec![0.0f32; config::HIDDEN_SIZE];
        combine_routed_rows_with_policy(rows, &indices, &mut rollback, true)?;
        let rollback_bits = combine["rust_ascending_left_fold_f32_bits"]
            .as_u64()
            .expect("rollback sum bits") as u32;
        assert_eq!(rollback_bits, 4.0f32.to_bits());
        assert_eq!(rollback[0].to_bits(), rollback_bits);
        assert_ne!(
            local[0].to_bits(),
            combine["torch_scalar_shape_1x6x1_sum_dim_1_f32_bits"]
                .as_u64()
                .expect("scalar torch sum bits") as u32
        );
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
    /// softmax (1/64 each). Torch's pinned unsorted top-6 selects six slots, each
    /// weighted 1/64. If every routed expert is the identity-ish MLP that outputs
    /// a known per-expert constant, the routed sum is predictable; we keep it
    /// simple by making all experts produce the SAME output, so the routed
    /// contribution is (sum of 6 weights) * expert_out, plus shared.
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
    fn forward_shims_error_cleanly_on_empty_weights() {
        // The shims are now wired: they look tensors up by name and delegate to
        // the tested `moe_block`/`dense_mlp`. An empty `Weights::default()` has no
        // tensors, so they must surface a clean `FormatMismatch` (tensor not
        // found) rather than panic or return garbage.
        let w = Weights::default();
        let x = Mat::from_vec(1, config::HIDDEN_SIZE, vec![0.0; config::HIDDEN_SIZE]);
        assert!(matches!(
            forward(&w, &x, 1),
            Err(FocrError::FormatMismatch(_))
        ));
        assert!(matches!(
            dense_forward(&w, &x),
            Err(FocrError::FormatMismatch(_))
        ));
    }
}
