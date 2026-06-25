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
/// `is_eos` is `true` when `token == eos_token_id`; the AR loop uses it to halt
/// ([SPEC-101]). `at_max_length` reflects whether the caller has reached the
/// generation cap. The `token` is always the chosen id even when `is_eos` (the
/// EOS id itself is the produced token, matching HF where EOS is appended then
/// generation stops).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOutput {
    /// The chosen next-token id.
    pub token: u32,
    /// Whether the chosen token is EOS (caller should stop after appending).
    pub is_eos: bool,
}

impl DecodeOutput {
    /// Build a [`DecodeOutput`], computing `is_eos` from `params`.
    #[must_use]
    pub fn new(token: u32, params: &DecodeParams) -> Self {
        Self {
            token,
            is_eos: token == params.eos_token_id,
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
fn argmax_row(logits: &[f32]) -> FocrResult<u32> {
    if logits.is_empty() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "sampler::argmax_row: empty logits row"
        )));
    }
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        // Strict `>` keeps the FIRST max on ties (torch.argmax). The compare is
        // false when `v` is NaN, so NaN is skipped; if `best_val` is NaN we
        // adopt the first non-NaN we meet (which is `> NaN` is also false, so we
        // additionally promote when best is NaN).
        if v > best_val || best_val.is_nan() {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

/// Apply the custom sliding-window no-repeat-n-gram blocker in place over a
/// single batch row's `logits`, given the already-generated `sequence`
/// ([SPEC-103], `modeling_unlimitedocr.py:354-383`).
///
/// Exact port of `SlidingWindowNoRepeatNgramProcessor.__call__` for one batch
/// row (we always run with `batch == 1`):
///
/// * `ngram_size == 0` is a no-op (HF builtin path / disabled — handled by the
///   caller, included here for safety).
/// * if `sequence.len() < ngram_size`: nothing to ban.
/// * `search_start = max(0, len - window)`, `search_end = len - ngram_size + 1`;
///   if `search_end <= search_start`: nothing to ban.
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
fn apply_sliding_window_ngram_block(
    logits: &mut [f32],
    sequence: &[u32],
    ngram_size: usize,
    window: usize,
    whitelist: &[u32],
) {
    if ngram_size == 0 {
        return;
    }
    let len = sequence.len();
    if len < ngram_size {
        return;
    }
    let search_start = len.saturating_sub(window);
    // len - ngram_size + 1; safe because len >= ngram_size >= 1.
    let search_end = len - ngram_size + 1;
    if search_end <= search_start {
        return;
    }

    // current_prefix = last (ngram_size - 1) tokens (empty for ngram_size==1).
    let prefix_len = ngram_size - 1;
    let current_prefix = &sequence[len - prefix_len..];

    let vocab = logits.len();
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
                logits[bi] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Pick the next token id from a `[1, vocab]` logits row under `params`.
///
/// Greedy fp32 decode ([SPEC-100]): when `params.is_greedy()` (temperature 0)
/// we argmax the logits — **no softmax**, since `argmax(softmax(x)) ==
/// argmax(x)`. Before the argmax we run the custom sliding-window n-gram blocker
/// over a scratch copy of the row when [`DecodeParams::sliding_ngram_active`]
/// ([SPEC-102/103]); banned tokens get `-inf` and so can never be selected.
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

    let row = logits.row(0);

    // Fast path: no blocker active, or nothing in the window can be banned yet.
    if !params.sliding_ngram_active() || generated.len() < params.no_repeat_ngram_size {
        return argmax_row(row);
    }

    // Mask banned tokens on a scratch copy, then argmax.
    let mut masked = row.to_vec();
    apply_sliding_window_ngram_block(
        &mut masked,
        generated,
        params.no_repeat_ngram_size,
        params.ngram_window,
        &[],
    );
    argmax_row(&masked)
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
    let token = sample(logits, generated, params)?;
    Ok(DecodeOutput::new(token, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(v: Vec<f32>) -> Mat {
        let n = v.len();
        Mat::from_vec(1, n, v)
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
        assert_eq!(out.token, 1);
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
        assert_eq!(out.token, 2);
        assert!(!out.is_eos);
    }

    // ── n-gram blocker semantics ──────────────────────────────────────────

    /// With ngram_size=1 every token that appears anywhere in the window is
    /// banned (prefix is empty, always "matches"). Sequence [0,0] over vocab 3
    /// with window 8: positions [0,1) (search_end = 2-1+1 = 2, start = 0) ban
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
