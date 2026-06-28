//! Sampler + the autoregressive decode loop ([SPEC-100..103],
//! PROPOSED_ARCHITECTURE.md §6.10). Greedy fp32 decode.
//!
//! Greedy by default (`do_sample = temperature > 0`, default `temperature=0.0`
//! => argmax over the `vocab_size = 129280` lm_head logits — [SPEC-100],
//! [SPEC-081], `config.json:118`). EOS=1, `max_length=32768`, `use_cache`
//! ([SPEC-101]). `no_repeat_ngram_size=35` with `ngram_window` 128 single-image
//! / 1024 multi-image (OQ-18, `oq/preprocess-infer.md`) realized as the custom
//! [`SlidingWindowNoRepeatNgramProcessor`] ([SPEC-102/103]).
//!
//! Under greedy temperature=0 there is **no full softmax**: argmax of the logits
//! equals argmax of the softmax, so we skip the (vocab-wide) normalization and
//! just scan for the max. The n-gram blocker bans a token by setting its logit
//! to `-inf` *before* the argmax scan, which is exactly the HF `LogitsProcessor`
//! contract (`scores[batch, banned] = float('-inf')`,
//! `modeling_unlimitedocr.py:382`).
//!
//! (The architecture names this `decode.rs`; the substrate skeleton keeps it as
//! `sampler` per the engine module list, with the AR loop living here.)

use super::tensor::Mat;
use crate::error::{FocrError, FocrResult};

/// Vocabulary size — the lm_head output width and logits-row length
/// (`config.json:118` `"vocab_size": 129280`, [SPEC-081]). Kept as a named
/// constant so call sites and shape checks agree on the one true width.
pub const VOCAB_SIZE: usize = 129_280;

/// Default end-of-sentence token id (`<｜end▁of▁sentence｜>`), [SPEC-101].
pub const DEFAULT_EOS_TOKEN_ID: u32 = 1;

/// Default no-repeat n-gram size (README single/multi both use 35; OQ-18 (f)).
pub const DEFAULT_NO_REPEAT_NGRAM_SIZE: usize = 35;

/// Default sliding-window lookback for single-image decode (OQ-18 (f),
/// `README.md:96`). Multi-image uses [`NGRAM_WINDOW_MULTI`].
pub const NGRAM_WINDOW_SINGLE: usize = 128;

/// Sliding-window lookback for multi-image / PDF decode (OQ-18 (f),
/// `README.md:108`).
pub const NGRAM_WINDOW_MULTI: usize = 1024;

/// Generation length cap in every reference path ([SPEC-101],
/// `modeling_unlimitedocr.py:787/1011/1139/1249`).
pub const DEFAULT_MAX_LENGTH: usize = 32_768;

/// Decode-time sampling parameters (the frozen contract). Greedy when
/// `temperature == 0.0`.
#[derive(Debug, Clone)]
pub struct DecodeParams {
    /// Sampling temperature; `0.0` => greedy argmax ([SPEC-100]).
    pub temperature: f32,
    /// EOS token id ([SPEC-101]).
    pub eos_token_id: u32,
    /// Maximum generated length ([SPEC-101]).
    pub max_length: usize,
    /// No-repeat n-gram size; `0` disables ([SPEC-102]).
    pub no_repeat_ngram_size: usize,
    /// Sliding window for the custom n-gram processor; `0` => HF builtin
    /// behavior ([SPEC-102/103]).
    pub ngram_window: usize,
}

impl Default for DecodeParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            eos_token_id: DEFAULT_EOS_TOKEN_ID,
            max_length: DEFAULT_MAX_LENGTH,
            no_repeat_ngram_size: DEFAULT_NO_REPEAT_NGRAM_SIZE,
            ngram_window: NGRAM_WINDOW_SINGLE,
        }
    }
}

impl DecodeParams {
    /// Single-image / Gundam greedy decode (`ngram_window = 128`), OQ-18 (f).
    #[must_use]
    pub fn single_image() -> Self {
        Self::default()
    }

    /// Multi-image / PDF greedy decode (`ngram_window = 1024`), OQ-18 (f).
    #[must_use]
    pub fn multi_image() -> Self {
        Self {
            ngram_window: NGRAM_WINDOW_MULTI,
            ..Self::default()
        }
    }

    /// Whether sampling is greedy (`do_sample = temperature > 0`, [SPEC-100]).
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        // Negation is intentional: `do_sample = temperature > 0`, so greedy is its
        // exact logical negation. This also maps a NaN `temperature` to greedy
        // (`!(NaN > 0.0)` == true), which `temperature <= 0.0` would not preserve.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        !(self.temperature > 0.0)
    }

    /// Whether the custom sliding-window n-gram blocker is active — both
    /// `no_repeat_ngram_size > 0` and `ngram_window > 0` ([SPEC-102]).
    #[must_use]
    pub fn sliding_ngram_active(&self) -> bool {
        self.no_repeat_ngram_size > 0 && self.ngram_window > 0
    }
}

/// One step's decode result (the frozen output contract).
///
/// `is_eos` is `true` when `token_id == eos_token_id`; the AR loop uses it to halt
/// ([SPEC-101]). The `token_id` is always the chosen id even when `is_eos` (the
/// EOS id itself is the produced token, matching HF where EOS is appended then
/// generation stops).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOutput {
    /// The chosen next-token id.
    pub token_id: u32,
    /// Whether the chosen token is EOS (caller should stop after appending).
    pub is_eos: bool,
}

impl DecodeOutput {
    /// Build a [`DecodeOutput`], computing `is_eos` from `params`.
    #[must_use]
    pub fn new(token_id: u32, params: &DecodeParams) -> Self {
        Self {
            token_id,
            is_eos: token_id == params.eos_token_id,
        }
    }
}

/// Greedy argmax over a single `[1, vocab]` logits row, returning the
/// lowest-index maximal token id.
///
/// This matches `torch.argmax` semantics used by HF greedy decode: on ties the
/// **first** (lowest-index) maximum wins. `NaN` logits never compare greater, so
/// a token banned to `-inf` (or any finite value) is preferred over `NaN`; an
/// all-`NaN` row falls back to id 0.
///
/// # Errors
/// Returns [`FocrError::Other`] if the row is empty (`vocab == 0`).
///
/// `pub(crate)` so the speculative-decode verifier ([`super::spec`], bd-1azu.32)
/// reuses the EXACT argmax/tie-break the production decode loop runs — sharing
/// this one function is what makes the verifier byte-for-byte greedy.
pub(crate) fn argmax_row(logits: &[f32]) -> FocrResult<u32> {
    if logits.is_empty() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "sampler::argmax_row: empty logits row"
        )));
    }
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some((_, best_val)) if v <= best_val => {}
            _ => best = Some((i, v)),
        }
    }
    let best_idx = best.map_or(0, |(i, _)| i);
    Ok(best_idx as u32)
}

/// Visit every in-vocab token id that the sliding-window no-repeat-n-gram
/// processor would ban. `window == 0` means the HF built-in global
/// no-repeat-ngram fallback: scan the whole generated history.
fn for_each_sliding_window_ngram_ban(
    sequence: &[u32],
    ngram_size: usize,
    window: usize,
    whitelist: &[u32],
    vocab: usize,
    mut visit: impl FnMut(usize),
) {
    if ngram_size == 0 {
        return;
    }
    let len = sequence.len();
    if len < ngram_size {
        return;
    }
    let search_start = if window == 0 {
        0
    } else {
        len.saturating_sub(window)
    };
    // len - ngram_size + 1; safe because len >= ngram_size >= 1.
    let search_end = len - ngram_size + 1;
    if search_end <= search_start {
        return;
    }

    // current_prefix = last (ngram_size - 1) tokens (empty for ngram_size==1).
    let prefix_len = ngram_size - 1;
    let current_prefix = &sequence[len - prefix_len..];

    for idx in search_start..search_end {
        let ngram = &sequence[idx..idx + ngram_size];
        let prefix_matches = ngram_size == 1 || &ngram[..prefix_len] == current_prefix;
        if prefix_matches {
            let banned = ngram[ngram_size - 1];
            if whitelist.contains(&banned) {
                continue;
            }
            let bi = banned as usize;
            if bi < vocab {
                visit(bi);
            }
        }
    }
}

/// Return a masked logits copy only when the blocker actually bans at least one
/// in-vocab token. The common no-ban decode step returns `None`, avoiding a
/// full-vocab copy.
///
/// `pub(crate)` so the speculative-decode verifier ([`super::spec`], bd-1azu.32)
/// applies the IDENTICAL sliding-window n-gram ban the production decode loop
/// runs before argmax — the verifier reuses this exact masking, never a re-derived
/// copy, so its per-position greedy token matches sequential decode bit-for-bit.
pub(crate) fn masked_sliding_window_logits_if_needed(
    row: &[f32],
    sequence: &[u32],
    ngram_size: usize,
    window: usize,
    whitelist: &[u32],
) -> Option<Vec<f32>> {
    let mut masked: Option<Vec<f32>> = None;
    for_each_sliding_window_ngram_ban(sequence, ngram_size, window, whitelist, row.len(), |bi| {
        let row = masked.get_or_insert_with(|| row.to_vec());
        row[bi] = f32::NEG_INFINITY;
    });
    masked
}

/// Apply the custom sliding-window no-repeat-n-gram blocker in place over a
/// single batch row's `logits`, given the already-generated `sequence`
/// ([SPEC-103], `modeling_unlimitedocr.py:354-383`).
///
/// Exact port of `SlidingWindowNoRepeatNgramProcessor.__call__` for one batch
/// row (we always run with `batch == 1`), plus the reference generation
/// fallback where `ngram_window == 0` and `no_repeat_ngram_size > 0` uses HF's
/// global no-repeat-ngram processor over the whole sequence:
///
/// * `ngram_size == 0` is a no-op (HF builtin path / disabled — handled by the
///   caller, included here for safety).
/// * if `sequence.len() < ngram_size`: nothing to ban.
/// * `search_start = max(0, len - window)` when `window > 0`, or `0` when
///   `window == 0`; `search_end = len - ngram_size + 1`; if
///   `search_end <= search_start`: nothing to ban.
/// * `current_prefix = last (ngram_size - 1) tokens` (empty when
///   `ngram_size == 1`).
/// * for each window position `idx` in `[search_start, search_end)`: take the
///   `ngram = sequence[idx .. idx+ngram_size]`; if `ngram_size == 1` or the
///   ngram's prefix (`ngram[..ngram_size-1]`) equals `current_prefix`, ban its
///   last token by setting `logits[last] = -inf`.
///
/// `whitelist` tokens are never banned (matches `banned.difference_update`).
///
/// Banning a token whose id is out of range for `logits` is silently skipped
/// (a malformed sequence shouldn't panic the decode loop).
#[cfg(test)]
fn apply_sliding_window_ngram_block(
    logits: &mut [f32],
    sequence: &[u32],
    ngram_size: usize,
    window: usize,
    whitelist: &[u32],
) {
    let vocab = logits.len();
    for_each_sliding_window_ngram_ban(sequence, ngram_size, window, whitelist, vocab, |bi| {
        logits[bi] = f32::NEG_INFINITY;
    });
}

/// Pick the next token id from a `[1, vocab]` logits row under `params`.
///
/// Greedy fp32 decode ([SPEC-100]): when `params.is_greedy()` (temperature 0)
/// we argmax the logits — **no softmax**, since `argmax(softmax(x)) ==
/// argmax(x)`. Before the argmax we run the custom sliding-window n-gram blocker
/// over a scratch copy of the row when `no_repeat_ngram_size > 0`
/// ([SPEC-102/103]). With `ngram_window > 0` this is the custom sliding-window
/// processor; with `ngram_window == 0` it is the reference fallback to HF's
/// global no-repeat-ngram behavior. Banned tokens get `-inf` and so can never be
/// selected.
///
/// `generated` is the full sequence decoded so far (prompt + emitted tokens);
/// the n-gram blocker reads its tail. The logits row is borrowed read-only — the
/// `-inf` masking happens on an internal copy only when a token actually needs
/// banning, so the common no-ban step does zero extra allocation.
///
/// Non-greedy (`temperature > 0`) sampling is not part of the greedy fp32 spine
/// and returns [`FocrError::NotImplemented`].
///
/// # Errors
/// * [`FocrError::Other`] if `logits` is not a single row (`rows != 1`) or the
///   row width is `0`.
/// * [`FocrError::NotImplemented`] for `temperature > 0` (sampling path).
pub fn sample(logits: &Mat, generated: &[u32], params: &DecodeParams) -> FocrResult<u32> {
    if logits.rows != 1 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "sampler::sample expects a single [1, vocab] logits row, got [{}, {}]",
            logits.rows,
            logits.cols
        )));
    }
    if !params.is_greedy() {
        return Err(FocrError::NotImplemented(
            "native_engine::sampler::sample — temperature>0 sampling is outside the greedy fp32 spine"
                .into(),
        ));
    }
    let expected_len = logits.rows.checked_mul(logits.cols).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "sampler::sample: logits shape product overflow for [{}, {}]",
            logits.rows,
            logits.cols
        ))
    })?;
    if logits.data.len() != expected_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "sampler::sample: logits data len {} != rows*cols {} for shape [{}, {}]",
            logits.data.len(),
            expected_len,
            logits.rows,
            logits.cols
        )));
    }

    let row = logits.row(0);

    // Fast path: no blocker active, or nothing can be banned yet.
    if params.no_repeat_ngram_size == 0 || generated.len() < params.no_repeat_ngram_size {
        return argmax_row(row);
    }

    if let Some(masked) = masked_sliding_window_logits_if_needed(
        row,
        generated,
        params.no_repeat_ngram_size,
        params.ngram_window,
        &[],
    ) {
        return argmax_row(&masked);
    }

    argmax_row(row)
}

/// Full single-step greedy decode returning the frozen [`DecodeOutput`]
/// (token + EOS flag). Thin wrapper over [`sample`] that also classifies EOS so
/// the AR loop can branch on one value ([SPEC-101]).
///
/// # Errors
/// Propagates [`sample`]'s errors.
pub fn decode_step(
    logits: &Mat,
    generated: &[u32],
    params: &DecodeParams,
) -> FocrResult<DecodeOutput> {
    let token_id = sample(logits, generated, params)?;
    Ok(DecodeOutput::new(token_id, params))
}

/// Greedy fp32 decode for `B` in-flight page-streams at once (bd-1azu.7 — the
/// Phase-6 continuous-batch decode spine, bd-1azu).
///
/// `logits` is the stacked `[B, vocab]` lm_head output: row `s` is stream `s`'s
/// next-token logits, exactly the `[1, vocab]` row the single-stream [`sample`]
/// consumes. `histories[s]` is stream `s`'s OWN generated sequence so far
/// (prompt and emitted tokens); each stream's sliding-window n-gram blocker reads
/// only its own tail, so two streams with different histories ban different
/// tokens off otherwise-identical logits ([SPEC-102/103]). Returns one chosen
/// token id per stream — `result[s]` is the id the single-stream path picks for
/// `(row s, histories[s])`.
///
/// LOSSLESS by construction: this is a per-stream loop that calls the existing
/// [`sample`] on each stream's `[1, vocab]` row with that stream's history, so
/// `batched_sample(logits, histories, params)[s]` is byte-for-byte identical to
/// `sample(row s as a [1, vocab] Mat, histories[s], params)`. Greedy argmax +
/// the ngram ban is a per-row reduction with no cross-stream interaction, so there
/// is no shared reduction order to preserve (unlike attention, bd-1waa) — the
/// batched API only amortizes the caller's dispatch, it does not change the math.
///
/// PERF SEAM (bd-1azu, OFF here): each stream's row is copied into a temporary
/// `[1, vocab]` [`Mat`] to reuse [`sample`] verbatim. A future lossless
/// optimization can drop the copy by argmax-ing `logits.row(s)` in place against a
/// per-stream ngram mask, but that is a perf-only change and MUST keep this
/// per-stream == single-stream invariant.
///
/// # Errors
/// * [`FocrError::Other`] if `histories.len() != logits.rows` (one history per
///   stream), or the backing data length disagrees with `rows * cols`.
/// * Propagates [`sample`]'s per-stream errors ([`FocrError::NotImplemented`] for
///   `temperature > 0`, [`FocrError::Other`] for an empty row).
pub fn batched_sample(
    logits: &Mat,
    histories: &[&[u32]],
    params: &DecodeParams,
) -> FocrResult<Vec<u32>> {
    if histories.len() != logits.rows {
        return Err(FocrError::Other(anyhow::anyhow!(
            "sampler::batched_sample: {} histories for {} logits rows (need one history per stream)",
            histories.len(),
            logits.rows
        )));
    }
    let expected_len = logits.rows.checked_mul(logits.cols).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "sampler::batched_sample: logits shape product overflow for [{}, {}]",
            logits.rows,
            logits.cols
        ))
    })?;
    if logits.data.len() != expected_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "sampler::batched_sample: logits data len {} != rows*cols {} for shape [{}, {}]",
            logits.data.len(),
            expected_len,
            logits.rows,
            logits.cols
        )));
    }

    let mut tokens = Vec::with_capacity(logits.rows);
    for (s, hist) in histories.iter().enumerate() {
        // PERF SEAM: per-stream `[1, vocab]` row copy so we can reuse the
        // single-stream `sample` byte-for-byte (same fn, same args) — lossless by
        // construction. The optimization that removes this copy lives behind the
        // bd-1azu batched-decode kill-switch, not here.
        let row = Mat::from_vec(1, logits.cols, logits.row(s).to_vec());
        tokens.push(sample(&row, hist, params)?);
    }
    Ok(tokens)
}

/// Batched [`decode_step`]: greedy-decode `B` streams and classify EOS per stream
/// (bd-1azu.7). `result[s]` is byte-for-byte the [`DecodeOutput`] the
/// single-stream [`decode_step`] returns for `(row s, histories[s])`. Thin wrapper
/// over [`batched_sample`] plus per-stream EOS classification ([SPEC-101]).
///
/// # Errors
/// Propagates [`batched_sample`]'s errors.
pub fn batched_decode_step(
    logits: &Mat,
    histories: &[&[u32]],
    params: &DecodeParams,
) -> FocrResult<Vec<DecodeOutput>> {
    let tokens = batched_sample(logits, histories, params)?;
    Ok(tokens
        .into_iter()
        .map(|token_id| DecodeOutput::new(token_id, params))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(v: Vec<f32>) -> Mat {
        let n = v.len();
        Mat::from_vec(1, n, v)
    }

    fn logits_preferring_35gram_banned_token() -> Mat {
        let mut logits = vec![0.0; 128];
        logits[7] = 10.0; // raw argmax, banned when the old 35-gram is in-window
        logits[6] = 9.0; // next-best fallback when token 7 is banned
        row(logits)
    }

    fn repeat_35gram_sequence(total_len: usize) -> Vec<u32> {
        const NGRAM: usize = 35;
        const PREFIX_LEN: usize = NGRAM - 1;
        const BANNED: u32 = 7;
        let prefix: Vec<u32> = (20..20 + PREFIX_LEN as u32).collect();
        let min_len = PREFIX_LEN + 1 + PREFIX_LEN;
        assert!(total_len >= min_len);

        let mut seq = Vec::with_capacity(total_len);
        seq.extend_from_slice(&prefix);
        seq.push(BANNED);
        seq.extend(std::iter::repeat_n(99, total_len - min_len));
        seq.extend_from_slice(&prefix);
        seq
    }

    fn params_with_window(window: usize) -> DecodeParams {
        DecodeParams {
            no_repeat_ngram_size: 35,
            ngram_window: window,
            ..DecodeParams::default()
        }
    }

    #[test]
    fn defaults_match_frozen_contract() {
        let p = DecodeParams::default();
        assert_eq!(p.temperature, 0.0);
        assert_eq!(p.eos_token_id, 1);
        assert_eq!(p.max_length, 32_768);
        assert_eq!(p.no_repeat_ngram_size, 35);
        assert_eq!(p.ngram_window, 128);
        assert!(p.is_greedy());
        assert!(p.sliding_ngram_active());
    }

    #[test]
    fn single_and_multi_windows() {
        assert_eq!(DecodeParams::single_image().ngram_window, 128);
        assert_eq!(DecodeParams::multi_image().ngram_window, 1024);
        // both keep ngram_size 35 and greedy temperature
        assert_eq!(DecodeParams::multi_image().no_repeat_ngram_size, 35);
        assert!(DecodeParams::multi_image().is_greedy());
    }

    #[test]
    fn vocab_size_constant() {
        assert_eq!(VOCAB_SIZE, 129_280);
    }

    #[test]
    fn argmax_picks_max() {
        let r = row(vec![0.1, -2.0, 3.5, 3.4, 0.0]);
        assert_eq!(sample(&r, &[], &DecodeParams::default()).unwrap(), 2);
    }

    #[test]
    fn argmax_ties_pick_lowest_index() {
        // two equal maxima at idx 1 and 3 -> torch.argmax returns the FIRST (1)
        let r = row(vec![0.0, 5.0, 1.0, 5.0]);
        // disable blocker so we test pure argmax tie semantics
        let p = DecodeParams {
            no_repeat_ngram_size: 0,
            ngram_window: 0,
            ..DecodeParams::default()
        };
        assert_eq!(sample(&r, &[], &p).unwrap(), 1);
    }

    #[test]
    fn argmax_skips_nan_and_neg_inf() {
        let r = row(vec![f32::NAN, f32::NEG_INFINITY, 2.0, 1.0]);
        let p = DecodeParams {
            no_repeat_ngram_size: 0,
            ngram_window: 0,
            ..DecodeParams::default()
        };
        assert_eq!(sample(&r, &[], &p).unwrap(), 2);
    }

    #[test]
    fn argmax_all_nan_falls_back_to_zero() {
        let r = row(vec![f32::NAN, f32::NAN, f32::NAN]);
        let p = DecodeParams {
            no_repeat_ngram_size: 0,
            ngram_window: 0,
            ..DecodeParams::default()
        };
        assert_eq!(sample(&r, &[], &p).unwrap(), 0);
    }

    #[test]
    fn rejects_multi_row_logits() {
        let m = Mat::zeros(2, 4);
        assert!(sample(&m, &[], &DecodeParams::default()).is_err());
    }

    #[test]
    fn rejects_empty_row() {
        let m = Mat::from_vec(1, 0, vec![]);
        assert!(sample(&m, &[], &DecodeParams::default()).is_err());
    }

    #[test]
    fn rejects_malformed_logits_backing_data_without_panicking() {
        let m = Mat {
            rows: 1,
            cols: 4,
            data: vec![0.0, 1.0, 2.0],
        };
        assert!(matches!(
            sample(&m, &[], &DecodeParams::default()),
            Err(err) if err.to_string().contains("logits data len 3 != rows*cols 4")
        ));
    }

    #[test]
    fn temperature_sampling_not_implemented() {
        let r = row(vec![1.0, 2.0, 3.0]);
        let p = DecodeParams {
            temperature: 0.7,
            ..DecodeParams::default()
        };
        let e = sample(&r, &[], &p).unwrap_err();
        assert!(matches!(e, FocrError::NotImplemented(_)));
    }

    #[test]
    fn decode_step_flags_eos() {
        // logits favor id 1 (= default EOS)
        let r = row(vec![0.0, 9.0, 0.0]);
        let p = DecodeParams {
            no_repeat_ngram_size: 0,
            ngram_window: 0,
            ..DecodeParams::default()
        };
        let out = decode_step(&r, &[], &p).unwrap();
        assert_eq!(out.token_id, 1);
        assert!(out.is_eos);
    }

    #[test]
    fn decode_step_non_eos() {
        let r = row(vec![0.0, 0.0, 9.0]);
        let p = DecodeParams {
            no_repeat_ngram_size: 0,
            ngram_window: 0,
            ..DecodeParams::default()
        };
        let out = decode_step(&r, &[], &p).unwrap();
        assert_eq!(out.token_id, 2);
        assert!(!out.is_eos);
    }

    // ── n-gram blocker semantics ──────────────────────────────────────────

    /// With ngram_size=1 every token that appears anywhere in the window is
    /// banned (prefix is empty, always "matches"). Sequence [0,0] over vocab 3
    /// with window 8: positions [0,2) (search_end = 2-1+1 = 2, start = 0) ban
    /// token 0; argmax over [0:-inf, hi, hi] -> first remaining max.
    #[test]
    fn ngram_size_one_bans_window_tokens() {
        let r = row(vec![10.0, 5.0, 5.0]); // raw argmax would be 0
        let p = DecodeParams {
            no_repeat_ngram_size: 1,
            ngram_window: 8,
            ..DecodeParams::default()
        };
        // generated = [0, 0]; token 0 banned -> first of the remaining (idx 1)
        let got = sample(&r, &[0, 0], &p).unwrap();
        assert_eq!(got, 1);
    }

    /// ngram_size=2: ban the token that would complete a repeated bigram whose
    /// prefix == the last (ngram_size-1)=1 generated token.
    /// sequence = [7, 3, 7]; current_prefix = [7]. Window scan finds bigram
    /// (7,3) at idx 0 whose prefix [7] matches -> ban token 3.
    #[test]
    fn ngram_size_two_bans_repeat_completion() {
        // vocab 5; raw argmax would be token 3 (highest)
        let r = row(vec![0.0, 0.0, 0.0, 9.0, 1.0]);
        let p = DecodeParams {
            no_repeat_ngram_size: 2,
            ngram_window: 16,
            ..DecodeParams::default()
        };
        let got = sample(&r, &[7, 3, 7], &p).unwrap();
        // token 3 banned -> next best is token 4 (logit 1.0)
        assert_eq!(got, 4);
    }

    #[test]
    fn ngram_window_zero_uses_global_no_repeat_fallback() {
        // Reference generation uses the HF builtin no-repeat processor when
        // no_repeat_ngram_size > 0 and ngram_window == 0. That scans the whole
        // history, so [5,0,5] with ngram_size=2 bans token 0 from completing a
        // repeated [5,0] bigram even though the custom sliding window is off.
        let r = row(vec![9.0, 1.0, 0.0]);
        let p = DecodeParams {
            no_repeat_ngram_size: 2,
            ngram_window: 0,
            ..DecodeParams::default()
        };
        assert!(!p.sliding_ngram_active());
        let got = sample(&r, &[5, 0, 5], &p).unwrap();
        assert_eq!(got, 1);
    }

    #[test]
    fn ngram_35_single_window_boundary_127_128_129() {
        let r = logits_preferring_35gram_banned_token();
        let p = params_with_window(NGRAM_WINDOW_SINGLE);
        for (total_len, expected) in [(127usize, 6u32), (128, 6), (129, 7)] {
            let seq = repeat_35gram_sequence(total_len);
            assert_eq!(
                sample(&r, &seq, &p).unwrap(),
                expected,
                "total_len={total_len} should map to token {expected}"
            );
        }
    }

    #[test]
    fn ngram_35_multi_window_boundary_1023_1024_1025() {
        let r = logits_preferring_35gram_banned_token();
        let p = params_with_window(NGRAM_WINDOW_MULTI);
        for (total_len, expected) in [(1023usize, 6u32), (1024, 6), (1025, 7)] {
            let seq = repeat_35gram_sequence(total_len);
            assert_eq!(
                sample(&r, &seq, &p).unwrap(),
                expected,
                "total_len={total_len} should map to token {expected}"
            );
        }
    }

    #[test]
    fn ngram_all_banned_falls_back_to_lowest_token() {
        let r = row(vec![3.0, 2.0, 1.0]);
        let p = DecodeParams {
            no_repeat_ngram_size: 1,
            ngram_window: 8,
            ..DecodeParams::default()
        };
        assert_eq!(sample(&r, &[0, 1, 2], &p).unwrap(), 0);
    }

    #[test]
    fn sampler_boundary_masking_is_deterministic() {
        let r = logits_preferring_35gram_banned_token();
        let p = params_with_window(NGRAM_WINDOW_SINGLE);
        let seq = repeat_35gram_sequence(128);
        let first = sample(&r, &seq, &p).unwrap();
        for _ in 0..8 {
            assert_eq!(sample(&r, &seq, &p).unwrap(), first);
        }
    }

    /// The prefix must match: a bigram in the window whose prefix != last token
    /// does NOT ban. sequence = [1, 2, 9]; current_prefix=[9]; the only bigram
    /// in scan range with prefix 9 — none (bigrams are (1,2),(2,9)); (2,9)
    /// prefix is [2] != [9]; so nothing banned, raw argmax stands.
    #[test]
    fn ngram_two_no_ban_when_prefix_differs() {
        let r = row(vec![0.0, 0.0, 9.0, 0.0]); // argmax token 2
        let p = DecodeParams {
            no_repeat_ngram_size: 2,
            ngram_window: 16,
            ..DecodeParams::default()
        };
        let got = sample(&r, &[1, 2, 9], &p).unwrap();
        assert_eq!(got, 2);
    }

    #[test]
    fn ngram_mask_is_absent_when_scan_bans_nothing() {
        let r = row(vec![0.0, 0.0, 9.0, 0.0]);
        let masked = masked_sliding_window_logits_if_needed(r.row(0), &[1, 2, 9], 2, 16, &[]);
        assert!(masked.is_none());
        assert_eq!(sample(&r, &[1, 2, 9], &DecodeParams::default()).unwrap(), 2);
    }

    #[test]
    fn ngram_mask_materializes_on_first_real_ban() {
        let r = row(vec![0.0, 0.0, 9.0, 1.0]);
        let masked = masked_sliding_window_logits_if_needed(r.row(0), &[0, 2, 0], 2, 16, &[])
            .expect("token 2 should be banned");
        assert_eq!(masked[2], f32::NEG_INFINITY);
        assert_eq!(masked[3], 1.0);
        let p = DecodeParams {
            no_repeat_ngram_size: 2,
            ngram_window: 16,
            ..DecodeParams::default()
        };
        assert_eq!(sample(&r, &[0, 2, 0], &p).unwrap(), 3);
    }

    /// search window bounds: tokens older than `window` are not scanned.
    /// sequence = [3, <12 filler>, 3] won't reach the early (filler,3) bigram if
    /// the window is small. Here we check that a too-old repeat is NOT banned.
    #[test]
    fn ngram_respects_window_lookback() {
        // ngram_size 2, window 2 => search_start = len-2, only the most recent
        // bigram boundary is considered. sequence=[5,0,5]; len=3, window=2 =>
        // search_start=1, search_end=3-2+1=2 => idx in [1,2): bigram (0,5),
        // prefix [0] vs current_prefix [5] -> no match -> nothing banned.
        let r = row(vec![0.0, 0.0, 0.0, 0.0, 0.0, 9.0]); // argmax token 5
        let p = DecodeParams {
            no_repeat_ngram_size: 2,
            ngram_window: 2,
            ..DecodeParams::default()
        };
        let got = sample(&r, &[5, 0, 5], &p).unwrap();
        assert_eq!(got, 5);
    }

    /// short sequence (len < ngram_size) => no banning, raw argmax.
    #[test]
    fn ngram_skips_when_sequence_too_short() {
        let r = row(vec![9.0, 0.0, 0.0]);
        let p = DecodeParams {
            no_repeat_ngram_size: 35,
            ngram_window: 128,
            ..DecodeParams::default()
        };
        // only 3 tokens generated, far below ngram_size 35 -> no ban
        let got = sample(&r, &[0, 0, 0], &p).unwrap();
        assert_eq!(got, 0);
    }

    /// out-of-range banned id is skipped without panic (defensive).
    #[test]
    fn ngram_block_ignores_out_of_range_ban() {
        let mut logits = vec![1.0, 2.0, 3.0];
        // sequence references token id 99 (>= vocab 3); ngram_size 1, window 8.
        apply_sliding_window_ngram_block(&mut logits, &[99, 99], 1, 8, &[]);
        // nothing banned in-range -> logits unchanged
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    /// whitelist tokens are never banned.
    #[test]
    fn ngram_block_respects_whitelist() {
        let mut logits = vec![1.0, 2.0, 3.0];
        // ngram_size 1 would ban token 1, but it's whitelisted.
        apply_sliding_window_ngram_block(&mut logits, &[1, 1], 1, 8, &[1]);
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    /// direct check of the -inf masking on the completing token.
    #[test]
    fn ngram_block_sets_neg_inf_on_banned() {
        let mut logits = vec![0.0, 0.0, 0.0];
        // sequence [0,2,0]; ngram_size 2; current_prefix [0]; bigram (0,2) at
        // idx 0 has prefix [0] -> ban token 2.
        apply_sliding_window_ngram_block(&mut logits, &[0, 2, 0], 2, 16, &[]);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[1], 0.0);
    }
}
