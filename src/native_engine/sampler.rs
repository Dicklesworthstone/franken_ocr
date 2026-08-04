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

/// Opt-in diagnostic guard for deterministic low-novelty periodic decode
/// trajectories (bd-2mo.30.12). Unset and `0` are OFF; only `1` arms it.
///
/// The guard remains default-off until its false-positive posterior and corpus
/// tail impact have been calibrated. When armed, a trigger is a typed timeout,
/// never a synthetic EOS or a successful truncated result.
pub const RUNAWAY_GUARD_ENV: &str = "FOCR_RUNAWAY_GUARD";

/// Do not inspect ordinary short outputs. This provisional threshold has not
/// yet been calibrated against the pinned BF16 token stream and cannot become a
/// default until that evidence exists.
pub const RUNAWAY_GUARD_MIN_TOKENS: usize = 8_192;

/// Inspect one fixed suffix every 256 emitted tokens. Sparse checkpoints bound
/// detector work without making the decision depend on caller polling cadence.
pub const RUNAWAY_GUARD_CHECK_INTERVAL: usize = 256;

/// Token suffix used by the periodicity and novelty statistics.
pub const RUNAWAY_GUARD_WINDOW_TOKENS: usize = 2_048;

/// Longest candidate token period. Together with the 2,048-token window this
/// guarantees at least eight observed cycles for every candidate.
pub const RUNAWAY_GUARD_MAX_PERIOD: usize = 256;

/// Minimum complete cycles represented by a candidate period.
pub const RUNAWAY_GUARD_MIN_PERIOD_CYCLES: usize = 8;

/// A candidate period must agree with its lagged copy on at least 15/16 token
/// positions. Integer ratios keep the trigger bit-deterministic.
pub const RUNAWAY_GUARD_MATCH_NUMERATOR: usize = 15;
pub const RUNAWAY_GUARD_MATCH_DENOMINATOR: usize = 16;

/// Exact token 4-gram novelty must be no more than 1/4. This catches a repeated
/// table-row template whose small numeric fields change while avoiding string,
/// markup, or language-specific heuristics.
pub const RUNAWAY_GUARD_NGRAM_ORDER: usize = 4;
pub const RUNAWAY_GUARD_NOVELTY_NUMERATOR: usize = 1;
pub const RUNAWAY_GUARD_NOVELTY_DENOMINATOR: usize = 4;

/// Hysteresis: three consecutive suspicious checkpoints are required before a
/// request fails. One anomalous 2,048-token suffix cannot terminate a decode.
pub const RUNAWAY_GUARD_REQUIRED_HITS: usize = 3;

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

    /// Whether these params are exactly the FROZEN single-image ban the
    /// speculative-decode verifier's chooser hardwires (bd-1azu.32/.35/.36):
    /// `no_repeat_ngram_size == 35` over the `ngram_window == 128` lookback.
    ///
    /// This is the params half of the `FOCR_SPEC_DECODE` dispatch guard in
    /// `OcrModel::generate_cached_i8` — `spec::accept_longest` recomputes each
    /// per-position greedy token with [`DEFAULT_NO_REPEAT_NGRAM_SIZE`] /
    /// [`NGRAM_WINDOW_SINGLE`] baked in, so ANY override of either knob (e.g.
    /// `--no-repeat-ngram 20`, `--ngram-window 1024`, `FOCR_NO_REPEAT_NGRAM`)
    /// MUST keep speculative decode disengaged: this predicate returning `false`
    /// means the sequential greedy loop runs untouched, byte-for-byte today's
    /// path. Extracted here so the gate is testable from the public surface
    /// (`tests/spec_decode_gate.rs`) — a pure read, no numerics.
    #[must_use]
    pub fn matches_frozen_spec_ban(&self) -> bool {
        self.no_repeat_ngram_size == DEFAULT_NO_REPEAT_NGRAM_SIZE
            && self.ngram_window == NGRAM_WINDOW_SINGLE
    }
}

/// Pure token-stream statistics from one runaway-guard checkpoint.
///
/// These fields are deliberately public so calibration/evidence harnesses can
/// record the exact integer witness. No decoded strings or model precision
/// choices participate in the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunawayMetrics {
    /// Number of emitted tokens at this checkpoint.
    pub emitted_tokens: usize,
    /// Fixed suffix length analyzed.
    pub window_tokens: usize,
    /// Candidate lag with the greatest exact token agreement.
    pub best_period: usize,
    /// Equal token positions at `i` and `i - best_period`.
    pub period_matches: usize,
    /// Total lagged token comparisons.
    pub period_comparisons: usize,
    /// Distinct exact token 4-grams in the suffix.
    pub unique_ngrams: usize,
    /// Total (overlapping) token 4-grams in the suffix.
    pub total_ngrams: usize,
}

impl RunawayMetrics {
    /// Exact period-match ratio in parts per million for evidence logs.
    #[must_use]
    pub fn period_match_ppm(self) -> u32 {
        ratio_ppm(self.period_matches, self.period_comparisons)
    }

    /// Exact token 4-gram novelty ratio in parts per million for evidence logs.
    #[must_use]
    pub fn ngram_novelty_ppm(self) -> u32 {
        ratio_ppm(self.unique_ngrams, self.total_ngrams)
    }

    /// Whether this checkpoint crosses both fixed integer thresholds.
    #[must_use]
    pub fn is_suspicious(self) -> bool {
        (self.period_matches as u128) * (RUNAWAY_GUARD_MATCH_DENOMINATOR as u128)
            >= (self.period_comparisons as u128) * (RUNAWAY_GUARD_MATCH_NUMERATOR as u128)
            && (self.unique_ngrams as u128) * (RUNAWAY_GUARD_NOVELTY_DENOMINATOR as u128)
                <= (self.total_ngrams as u128) * (RUNAWAY_GUARD_NOVELTY_NUMERATOR as u128)
    }
}

/// Auditable witness carried by the typed timeout when hysteresis fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunawayEvidence {
    /// Metrics at the checkpoint that completed the trigger.
    pub metrics: RunawayMetrics,
    /// Consecutive suspicious checkpoints observed.
    pub consecutive_hits: usize,
}

impl RunawayEvidence {
    /// Convert a trigger into the stable budget/timeout error class (exit 5).
    ///
    /// This intentionally does not return generated tokens or synthesize EOS:
    /// the caller must reject the run rather than silently bless truncation.
    #[must_use]
    pub fn timeout_error(self) -> FocrError {
        FocrError::Timeout(format!(
            "runaway token guard triggered at {} emitted tokens after {} consecutive \
             checkpoints: period={} match={}/{} ({} ppm), token-4gram novelty={}/{} \
             ({} ppm); output rejected rather than silently truncated",
            self.metrics.emitted_tokens,
            self.consecutive_hits,
            self.metrics.best_period,
            self.metrics.period_matches,
            self.metrics.period_comparisons,
            self.metrics.period_match_ppm(),
            self.metrics.unique_ngrams,
            self.metrics.total_ngrams,
            self.metrics.ngram_novelty_ppm(),
        ))
    }
}

/// Deterministic action selected by [`RunawayGuard::observe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunawayDecision {
    /// Preserve the ordinary decode path unchanged.
    Continue,
    /// Reject the run with [`RunawayEvidence::timeout_error`].
    Abort(RunawayEvidence),
}

/// Opt-in finite-state detector for sustained periodic, low-novelty token tails.
///
/// State is only the next fixed checkpoint, consecutive suspicious-hit count,
/// last metrics, and an optional terminal evidence witness. Abort is sticky:
/// once selected, every later observation returns the identical evidence. The
/// token stream is caller-owned and read-only, so an armed guard cannot change
/// sampling, token order, or EOS behavior before selecting Abort.
#[derive(Debug, Clone)]
pub struct RunawayGuard {
    enabled: bool,
    next_checkpoint: Option<usize>,
    consecutive_hits: usize,
    last_metrics: Option<RunawayMetrics>,
    terminal_evidence: Option<RunawayEvidence>,
}

impl RunawayGuard {
    /// Construct an explicitly enabled or disabled guard.
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            next_checkpoint: Some(RUNAWAY_GUARD_MIN_TOKENS),
            consecutive_hits: 0,
            last_metrics: None,
            terminal_evidence: None,
        }
    }

    /// Parse the strict public environment contract.
    ///
    /// `None` and `Some("0")` preserve today's exact token stream. Only
    /// `Some("1")` arms the uncalibrated diagnostic controller; every other
    /// value is a usage error rather than a silently guessed policy.
    pub fn from_env_value(raw: Option<&str>) -> FocrResult<Self> {
        let enabled = match raw {
            None | Some("0") => false,
            Some("1") => true,
            Some(other) => {
                return Err(FocrError::Usage(format!(
                    "{RUNAWAY_GUARD_ENV} must be exactly 0 or 1, got {other:?}"
                )));
            }
        };
        Ok(Self::new(enabled))
    }

    /// Read [`RUNAWAY_GUARD_ENV`] once for one decode invocation.
    pub fn from_env() -> FocrResult<Self> {
        match std::env::var(RUNAWAY_GUARD_ENV) {
            Ok(raw) => Self::from_env_value(Some(&raw)),
            Err(std::env::VarError::NotPresent) => Self::from_env_value(None),
            Err(std::env::VarError::NotUnicode(_)) => Err(FocrError::Usage(format!(
                "{RUNAWAY_GUARD_ENV} must be valid UTF-8 containing exactly 0 or 1"
            ))),
        }
    }

    /// Whether this invocation is armed.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Metrics from the most recent analyzed checkpoint, if any.
    #[must_use]
    pub fn last_metrics(&self) -> Option<RunawayMetrics> {
        self.last_metrics
    }

    /// Observe an append-only emitted-token history.
    ///
    /// Checkpoints are analyzed at their exact prefix even if a caller commits
    /// several tokens between calls. The decision is therefore invariant to
    /// polling cadence. Disabled guards do not inspect or allocate.
    #[must_use]
    pub fn observe(&mut self, emitted: &[u32]) -> RunawayDecision {
        if !self.enabled {
            return RunawayDecision::Continue;
        }
        if let Some(evidence) = self.terminal_evidence {
            return RunawayDecision::Abort(evidence);
        }

        while let Some(checkpoint) = self.next_checkpoint {
            if emitted.len() < checkpoint {
                break;
            }
            self.next_checkpoint = checkpoint.checked_add(RUNAWAY_GUARD_CHECK_INTERVAL);

            let Some(metrics) = analyze_runaway_suffix(&emitted[..checkpoint]) else {
                continue;
            };
            self.last_metrics = Some(metrics);
            if metrics.is_suspicious() {
                self.consecutive_hits += 1;
                if self.consecutive_hits >= RUNAWAY_GUARD_REQUIRED_HITS {
                    let evidence = RunawayEvidence {
                        metrics,
                        consecutive_hits: self.consecutive_hits,
                    };
                    self.terminal_evidence = Some(evidence);
                    return RunawayDecision::Abort(evidence);
                }
            } else {
                self.consecutive_hits = 0;
            }
        }

        RunawayDecision::Continue
    }

    /// Shared production commit hook: check one just-appended token and
    /// propagate a terminal trigger as a typed timeout.
    ///
    /// A real EOS observed before a terminal trigger bypasses analysis; the
    /// guard exists only for sustained **no-EOS** trajectories. Once selected,
    /// Abort remains sticky. Every production AR loop calls this after appending
    /// its token and before committing more decoder state.
    pub fn check_after_emit(&mut self, emitted: &[u32], is_eos: bool) -> FocrResult<()> {
        if let Some(evidence) = self.terminal_evidence {
            return Err(evidence.timeout_error());
        }
        if is_eos {
            return Ok(());
        }
        match self.observe(emitted) {
            RunawayDecision::Continue => Ok(()),
            RunawayDecision::Abort(evidence) => Err(evidence.timeout_error()),
        }
    }
}

impl Default for RunawayGuard {
    fn default() -> Self {
        Self::new(false)
    }
}

/// Analyze the fixed token suffix used by the runaway guard.
///
/// This pure API exposes the exact integer witness for offline calibration and
/// replay. It returns `None` until a full 2,048-token window exists.
#[must_use]
pub fn analyze_runaway_suffix(emitted: &[u32]) -> Option<RunawayMetrics> {
    if emitted.len() < RUNAWAY_GUARD_WINDOW_TOKENS {
        return None;
    }
    let window = &emitted[emitted.len() - RUNAWAY_GUARD_WINDOW_TOKENS..];
    let max_period = RUNAWAY_GUARD_MAX_PERIOD.min(window.len() / RUNAWAY_GUARD_MIN_PERIOD_CYCLES);

    let mut best_period = 1usize;
    let mut best_matches = 0usize;
    let mut best_comparisons = window.len() - 1;
    for period in 1..=max_period {
        let comparisons = window.len() - period;
        let matches = (period..window.len())
            .filter(|&i| window[i] == window[i - period])
            .count();
        let lhs = (matches as u128) * (best_comparisons as u128);
        let rhs = (best_matches as u128) * (comparisons as u128);
        if lhs > rhs || (lhs == rhs && period < best_period) {
            best_period = period;
            best_matches = matches;
            best_comparisons = comparisons;
        }
    }

    let total_ngrams = window.len() - RUNAWAY_GUARD_NGRAM_ORDER + 1;
    let mut unique = std::collections::BTreeSet::new();
    for ngram in window.windows(RUNAWAY_GUARD_NGRAM_ORDER) {
        unique.insert(ngram);
    }

    Some(RunawayMetrics {
        emitted_tokens: emitted.len(),
        window_tokens: window.len(),
        best_period,
        period_matches: best_matches,
        period_comparisons: best_comparisons,
        unique_ngrams: unique.len(),
        total_ngrams,
    })
}

fn ratio_ppm(numerator: usize, denominator: usize) -> u32 {
    if denominator == 0 {
        return 0;
    }
    u32::try_from(((numerator as u128) * 1_000_000) / (denominator as u128)).unwrap_or(u32::MAX)
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
/// This matches the pinned `torch.argmax` semantics used by HF greedy decode:
/// on ties the **first** (lowest-index) maximum wins, and the first `NaN` wins
/// as soon as one is present. Banned tokens are set to `-inf` before this scan,
/// so a banned `NaN` no longer participates.
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
            return Ok(i as u32);
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

/// Collect every in-vocab token id the sliding-window no-repeat-ngram processor
/// would ban for `sequence`, as a flat list — the ban SET the
/// `FOCR_FUSE_NGRAM_LMHEAD` lm_head epilogue masks to -inf as the logits are
/// produced ([`super::decoder::lm_head_cached_i8_ngram_masked`]). Reuses the EXACT
/// [`for_each_sliding_window_ngram_ban`] scan that
/// [`masked_sliding_window_logits_if_needed`] uses, so the ban set is identical;
/// ids may repeat when the same completion is reachable from several window
/// positions (the epilogue mask is idempotent). `window == 0` is the HF global
/// no-repeat-ngram fallback.
pub(crate) fn collect_sliding_window_ngram_bans(
    sequence: &[u32],
    ngram_size: usize,
    window: usize,
    whitelist: &[u32],
    vocab: usize,
) -> Vec<u32> {
    let mut banned = Vec::new();
    for_each_sliding_window_ngram_ban(sequence, ngram_size, window, whitelist, vocab, |bi| {
        banned.push(bi as u32);
    });
    banned
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

/// Argmax + EOS over a `[1, vocab]` logits row whose sliding-window
/// no-repeat-ngram ban has ALREADY been folded into the lm_head epilogue
/// (`FOCR_FUSE_NGRAM_LMHEAD`, [`super::decoder::lm_head_cached_i8_ngram_masked`]):
/// the banned tokens are already `-inf`, so this argmaxes directly with NO masking
/// pass. For a row produced from those bans (via
/// [`collect_sliding_window_ngram_bans`]), the chosen token is byte-for-byte the
/// one [`decode_step`] returns for the UNMASKED logits + the same sequence — the
/// row the argmax sees is identical either way (banned channels `-inf`, the rest
/// the same lm_head dot products), and [`argmax_row`] is the same tie/NaN scan.
///
/// # Errors
/// * [`FocrError::Other`] if `logits` is not a single row, or the backing length
///   disagrees with `rows * cols`.
/// * [`FocrError::NotImplemented`] for `temperature > 0` (sampling path).
pub fn decode_step_premasked(logits: &Mat, params: &DecodeParams) -> FocrResult<DecodeOutput> {
    if logits.rows != 1 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "sampler::decode_step_premasked expects a single [1, vocab] logits row, got [{}, {}]",
            logits.rows,
            logits.cols
        )));
    }
    if !params.is_greedy() {
        return Err(FocrError::NotImplemented(
            "native_engine::sampler::decode_step_premasked — temperature>0 sampling is outside the greedy fp32 spine"
                .into(),
        ));
    }
    let expected_len = logits.rows.checked_mul(logits.cols).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "sampler::decode_step_premasked: logits shape product overflow for [{}, {}]",
            logits.rows,
            logits.cols
        ))
    })?;
    if logits.data.len() != expected_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "sampler::decode_step_premasked: logits data len {} != rows*cols {} for shape [{}, {}]",
            logits.data.len(),
            expected_len,
            logits.rows,
            logits.cols
        )));
    }
    let token_id = argmax_row(logits.row(0))?;
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

    fn periodic_tokens(len: usize, period: usize) -> Vec<u32> {
        (0..len).map(|i| (i % period) as u32).collect()
    }

    fn near_periodic_tokens(len: usize, period: usize) -> Vec<u32> {
        (0..len)
            .map(|i| {
                if i % period == 0 {
                    10_000 + (i / period) as u32
                } else {
                    (i % period) as u32
                }
            })
            .collect()
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
    fn runaway_guard_is_default_off_and_strictly_opt_in() {
        let default_guard = RunawayGuard::default();
        assert!(!default_guard.enabled());
        assert!(!RunawayGuard::from_env_value(None).unwrap().enabled());
        assert!(!RunawayGuard::from_env_value(Some("0")).unwrap().enabled());
        assert!(RunawayGuard::from_env_value(Some("1")).unwrap().enabled());

        for invalid in ["", "true", "on", "2", " 1 "] {
            assert!(matches!(
                RunawayGuard::from_env_value(Some(invalid)),
                Err(FocrError::Usage(message))
                    if message.contains("must be exactly 0 or 1")
            ));
        }
    }

    #[test]
    fn runaway_metrics_are_exact_token_level_witnesses() {
        let tokens = periodic_tokens(RUNAWAY_GUARD_WINDOW_TOKENS, 32);
        let metrics = analyze_runaway_suffix(&tokens).expect("full analysis window");
        assert_eq!(metrics.emitted_tokens, RUNAWAY_GUARD_WINDOW_TOKENS);
        assert_eq!(metrics.window_tokens, RUNAWAY_GUARD_WINDOW_TOKENS);
        assert_eq!(metrics.best_period, 32);
        assert_eq!(metrics.period_matches, RUNAWAY_GUARD_WINDOW_TOKENS - 32);
        assert_eq!(metrics.period_comparisons, RUNAWAY_GUARD_WINDOW_TOKENS - 32);
        assert_eq!(metrics.period_match_ppm(), 1_000_000);
        assert_eq!(metrics.unique_ngrams, 32);
        assert_eq!(metrics.total_ngrams, RUNAWAY_GUARD_WINDOW_TOKENS - 3);
        assert_eq!(metrics.ngram_novelty_ppm(), 15_647);
        assert!(metrics.is_suspicious());
    }

    #[test]
    fn runaway_guard_requires_three_consecutive_checkpoints() {
        let tokens = periodic_tokens(
            RUNAWAY_GUARD_MIN_TOKENS + 2 * RUNAWAY_GUARD_CHECK_INTERVAL,
            32,
        );
        let mut guard = RunawayGuard::new(true);

        assert_eq!(
            guard.observe(&tokens[..RUNAWAY_GUARD_MIN_TOKENS]),
            RunawayDecision::Continue
        );
        assert_eq!(
            guard.observe(&tokens[..RUNAWAY_GUARD_MIN_TOKENS + RUNAWAY_GUARD_CHECK_INTERVAL]),
            RunawayDecision::Continue
        );
        let RunawayDecision::Abort(evidence) = guard.observe(&tokens) else {
            unreachable!("third suspicious checkpoint must abort");
        };
        assert_eq!(evidence.consecutive_hits, RUNAWAY_GUARD_REQUIRED_HITS);
        assert_eq!(evidence.metrics.emitted_tokens, tokens.len());
        assert!(evidence.metrics.is_suspicious());
    }

    #[test]
    fn runaway_guard_catches_repeated_template_with_changing_field() {
        let tokens = near_periodic_tokens(
            RUNAWAY_GUARD_MIN_TOKENS + 2 * RUNAWAY_GUARD_CHECK_INTERVAL,
            32,
        );
        let metrics = analyze_runaway_suffix(&tokens).expect("full analysis window");
        assert_eq!(metrics.best_period, 32);
        assert!(metrics.period_match_ppm() >= 968_000);
        assert!(metrics.ngram_novelty_ppm() < 250_000);
        assert!(metrics.is_suspicious());

        let mut guard = RunawayGuard::new(true);
        assert!(matches!(guard.observe(&tokens), RunawayDecision::Abort(_)));
    }

    #[test]
    fn runaway_guard_rejects_with_typed_timeout_not_synthetic_eos() {
        let tokens = periodic_tokens(
            RUNAWAY_GUARD_MIN_TOKENS + 2 * RUNAWAY_GUARD_CHECK_INTERVAL,
            19,
        );
        let mut guard = RunawayGuard::new(true);
        let RunawayDecision::Abort(evidence) = guard.observe(&tokens) else {
            unreachable!("periodic stream must produce an evidence witness");
        };
        let error = evidence.timeout_error();
        assert_eq!(error.kind(), "timeout");
        assert_eq!(error.exit_code(), crate::error::EXIT_TIMEOUT);
        let message = error.to_string();
        assert!(message.contains("output rejected rather than silently truncated"));
        assert!(message.contains("period=19"));
        assert!(message.contains("token-4gram novelty"));
    }

    #[test]
    fn shared_commit_hook_propagates_timeout_but_never_overrides_real_eos() {
        let tokens = periodic_tokens(
            RUNAWAY_GUARD_MIN_TOKENS + 2 * RUNAWAY_GUARD_CHECK_INTERVAL,
            29,
        );

        let mut eos_guard = RunawayGuard::new(true);
        assert!(
            eos_guard
                .check_after_emit(&tokens[..RUNAWAY_GUARD_MIN_TOKENS], false)
                .is_ok()
        );
        assert!(
            eos_guard
                .check_after_emit(
                    &tokens[..RUNAWAY_GUARD_MIN_TOKENS + RUNAWAY_GUARD_CHECK_INTERVAL],
                    false,
                )
                .is_ok()
        );
        assert!(eos_guard.check_after_emit(&tokens, true).is_ok());

        let mut runaway_guard = RunawayGuard::new(true);
        let error = runaway_guard
            .check_after_emit(&tokens, false)
            .expect_err("third no-EOS checkpoint must fail");
        assert_eq!(error.kind(), "timeout");
        assert_eq!(error.exit_code(), crate::error::EXIT_TIMEOUT);
    }

    #[test]
    fn runaway_abort_is_terminal_and_preserves_its_first_witness() {
        let mut tokens = periodic_tokens(
            RUNAWAY_GUARD_MIN_TOKENS + 2 * RUNAWAY_GUARD_CHECK_INTERVAL,
            23,
        );
        let mut guard = RunawayGuard::new(true);
        let first = guard.observe(&tokens);
        let RunawayDecision::Abort(first_evidence) = first else {
            unreachable!("periodic stream must abort");
        };

        assert_eq!(guard.observe(&tokens), first);
        tokens.extend(periodic_tokens(RUNAWAY_GUARD_CHECK_INTERVAL, 23));
        assert_eq!(
            guard.observe(&tokens),
            RunawayDecision::Abort(first_evidence)
        );
        let sticky_error = guard
            .check_after_emit(&tokens, true)
            .expect_err("a terminal Abort cannot transition back to Continue on later EOS");
        let first_error = first_evidence.timeout_error();
        assert_eq!(sticky_error.kind(), first_error.kind());
        assert_eq!(sticky_error.to_string(), first_error.to_string());
    }

    #[test]
    fn runaway_guard_does_not_fire_on_high_novelty_tokens() {
        let tokens: Vec<u32> = (0..DEFAULT_MAX_LENGTH as u32).collect();
        let mut guard = RunawayGuard::new(true);
        assert_eq!(guard.observe(&tokens), RunawayDecision::Continue);
        let metrics = guard.last_metrics().expect("long stream was analyzed");
        assert_eq!(metrics.ngram_novelty_ppm(), 1_000_000);
        assert!(!metrics.is_suspicious());
    }

    #[test]
    fn one_suspicious_checkpoint_cannot_survive_a_normal_checkpoint() {
        let mut tokens = periodic_tokens(RUNAWAY_GUARD_MIN_TOKENS, 32);
        let mut guard = RunawayGuard::new(true);
        assert_eq!(guard.observe(&tokens), RunawayDecision::Continue);
        assert!(guard.last_metrics().unwrap().is_suspicious());

        tokens.extend((0..RUNAWAY_GUARD_CHECK_INTERVAL).map(|i| 1_000_000 + i as u32));
        assert_eq!(guard.observe(&tokens), RunawayDecision::Continue);
        assert!(!guard.last_metrics().unwrap().is_suspicious());
    }

    #[test]
    fn runaway_decision_is_invariant_to_observer_polling_cadence() {
        let tokens = near_periodic_tokens(
            RUNAWAY_GUARD_MIN_TOKENS + 2 * RUNAWAY_GUARD_CHECK_INTERVAL,
            47,
        );
        let mut one_shot = RunawayGuard::new(true);
        let one_shot_decision = one_shot.observe(&tokens);

        let mut incremental = RunawayGuard::new(true);
        let mut incremental_decision = RunawayDecision::Continue;
        for end in 1..=tokens.len() {
            incremental_decision = incremental.observe(&tokens[..end]);
            if matches!(incremental_decision, RunawayDecision::Abort(_)) {
                break;
            }
        }
        assert_eq!(incremental_decision, one_shot_decision);
    }

    #[test]
    fn disabled_runaway_guard_never_inspects_or_changes_tokens() {
        let tokens = periodic_tokens(DEFAULT_MAX_LENGTH, 7);
        let original = tokens.clone();
        let mut guard = RunawayGuard::default();
        assert_eq!(guard.observe(&tokens), RunawayDecision::Continue);
        assert!(guard.last_metrics().is_none());
        assert_eq!(tokens, original);
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
    fn argmax_selects_first_nan_like_pinned_torch() {
        let r = row(vec![7.0, f32::NAN, f32::NEG_INFINITY, f32::NAN, 9.0]);
        let p = DecodeParams {
            no_repeat_ngram_size: 0,
            ngram_window: 0,
            ..DecodeParams::default()
        };
        assert_eq!(sample(&r, &[], &p).unwrap(), 1);
    }

    #[test]
    fn argmax_all_nan_selects_first_index() {
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

/// bd-1azu.36 (LINEAR half) — speculative-decode gate FAULT-INJECTION battery: an
/// UNTRUSTED drafter can never change the emitted stream.
///
/// Lives here rather than in `tests/spec_decode_gate.rs` (its integration-side
/// companion) because the seams under fault —
/// `native_engine::spec::{accept_longest, resolve_round}` — are `pub(crate)`, so
/// integration tests (public API only) cannot reach them, and `spec.rs`/`mod.rs`
/// are owner-frozen this wave. The sampler is the natural in-crate home: the
/// verifier's chooser under fault IS this module's [`sample`]/[`decode_step`],
/// and every assertion replays it as the ground truth.
///
/// The model-free abstraction mirrors `spec.rs`'s own loop-parity tests: the
/// decoder is a pure token-sequence -> `[1, V]` logits ORACLE (the property the
/// real verify forward preserves bit-exactly, gated by
/// `tests/spec_verify_forward_parity.rs`), and the loop skeleton mirrors
/// `OcrModel::spec_decode_i8` — but the DRAFTER is an injected ADVERSARY instead
/// of `spec::draft_ngram`: garbage ids, out-of-vocab/wild ids, forged EOS,
/// oversized blocks far past `SPEC_DRAFT_MAX`, and always-empty proposals. The
/// gate holds iff the emitted stream is byte-for-byte sequential greedy in every
/// case (a draft is a PROPOSAL — the verifier accepts only tokens equal to the
/// per-position greedy choice, and fails CLOSED on malformed verify rows).
///
/// TREE-verify clauses (tree-attention node parity, longest-path accept,
/// `FOCR_SPEC_TREE_W=1` collapse) stay parked behind bd-1azu.34.
#[cfg(test)]
mod spec_gate_fault_injection {
    use super::{DecodeParams, decode_step, sample};
    use crate::native_engine::spec::{SPEC_DRAFT_MAX, accept_longest, resolve_round};
    use crate::native_engine::tensor::Mat;

    /// Vocabulary width for the synthetic logits rows (above every id used,
    /// including the distinct-id oversized-draft targets).
    const V: usize = 128;
    /// EOS id under test == the frozen default ([SPEC-101]).
    const EOS: u32 = 1;

    /// A `[1, V]` logits row whose unique argmax is `token`.
    fn peak_row(token: u32) -> Mat {
        let mut r = vec![0.0f32; V];
        r[token as usize] = 10.0;
        Mat::from_vec(1, V, r)
    }

    /// A `[1, V]` row peaked at `peak` with a distinct runner-up at `runner_up`,
    /// so a ban on `peak` flips the greedy token (the spec.rs ban-fixture idiom).
    fn row_peaked(peak: u32, runner_up: u32) -> Mat {
        let mut r = vec![0.0f32; V];
        r[peak as usize] = 10.0;
        r[runner_up as usize] = 9.0;
        Mat::from_vec(1, V, r)
    }

    /// Single-image greedy params (the frozen 35/128 ban) with a `max_length` cap.
    fn params(max_length: usize) -> DecodeParams {
        let mut p = DecodeParams::single_image();
        p.max_length = max_length;
        p
    }

    /// Deterministic xorshift64 step (the house PRNG idiom — reproducible, no
    /// dev-dependency).
    fn xs(s: &mut u64) -> u64 {
        let mut x = *s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *s = x;
        x
    }

    /// A deterministic 3rd-order content oracle: the next token is a hash of the
    /// last three tokens, in `2..=6`, with EOS firing intermittently once there
    /// is some history. Content-keyed, so verify rows are genuinely sensitive to
    /// the draft tokens; the small alphabet lets plausible garbage both agree and
    /// diverge. (Alphabet is `{EOS} ∪ 2..=6` — ids outside it are NEVER greedy.)
    fn content_oracle(seq: &[u32]) -> Mat {
        let start = seq.len().saturating_sub(3);
        let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
        for &t in &seq[start..] {
            h ^= u64::from(t).wrapping_add(0x517C_C1B7_2722_0A95);
            h = h.rotate_left(23).wrapping_mul(0x2545_F491_4F6C_DD1D);
        }
        let pick = if seq.len() >= 5 && (h & 7) == 0 {
            EOS
        } else {
            2 + (h % 5) as u32
        };
        peak_row(pick)
    }

    /// Reference SEQUENTIAL greedy decode — the literal `generate_cached_i8` loop
    /// with the decoder abstracted as `oracle`: choose via [`decode_step`] (the
    /// production chooser), append, halt at EOS or `max_length`. No draft, no
    /// verify assembly — the independent ground truth.
    fn seq_generate(
        oracle: &dyn Fn(&[u32]) -> Mat,
        prompt: &[u32],
        params: &DecodeParams,
    ) -> Vec<u32> {
        let mut generated = prompt.to_vec();
        let mut emitted = Vec::new();
        while emitted.len() < params.max_length {
            let logits = oracle(&generated);
            let step = decode_step(&logits, &generated, params).expect("seq decode_step");
            generated.push(step.token_id);
            emitted.push(step.token_id);
            if step.is_eos {
                break;
            }
        }
        emitted
    }

    /// What each adversarial drafter actually hit — TEETH for the battery (the
    /// parity assertions must not pass vacuously).
    #[derive(Default)]
    struct RoundStats {
        /// Speculative rounds run (a non-empty draft reached the verifier).
        rounds: usize,
        /// Empty-draft fallback steps (one sequential step, no verify).
        fallbacks: usize,
        /// Total accepted draft tokens across all rounds.
        accepted_tokens: usize,
        /// Rounds where the verifier rejected at least one proposed token.
        rejected_rounds: usize,
    }

    /// The `OcrModel::spec_decode_i8` loop skeleton with the DRAFTER injected as
    /// an arbitrary (adversarial) closure: `verify_logits[i]` plays the batched
    /// verify forward (`oracle(generated ++ draft[0..i])` — the contract
    /// `decoder::verify_forward_i8` upholds bit-exactly), the REAL
    /// [`resolve_round`] accepts + corrects, and committing a token is appending
    /// it (the oracle is a pure function of the token sequence). Honors
    /// EOS/`max_length` exactly as the live loop; mirrors `spec.rs`'s own
    /// `spec_generate` step for step.
    fn spec_generate_with_drafter(
        oracle: &dyn Fn(&[u32]) -> Mat,
        prompt: &[u32],
        params: &DecodeParams,
        drafter: &mut dyn FnMut(&[u32]) -> Vec<u32>,
        stats: &mut RoundStats,
    ) -> Vec<u32> {
        let mut generated = prompt.to_vec();
        let mut emitted = Vec::new();
        while emitted.len() < params.max_length {
            let draft = drafter(&generated);
            if draft.is_empty() {
                stats.fallbacks += 1;
                let logits = oracle(&generated);
                let step = decode_step(&logits, &generated, params).expect("spec fallback step");
                generated.push(step.token_id);
                emitted.push(step.token_id);
                if step.is_eos {
                    break;
                }
                continue;
            }
            // verify_logits[i] conditions on generated ++ draft[0..i] (i in 0..=K).
            let mut verify_logits: Vec<Mat> = Vec::with_capacity(draft.len() + 1);
            for i in 0..=draft.len() {
                let mut ctx = generated.clone();
                ctx.extend_from_slice(&draft[..i]);
                verify_logits.push(oracle(&ctx));
            }
            let emit =
                resolve_round(&generated, &draft, &verify_logits, params).expect("resolve_round");
            stats.rounds += 1;
            stats.accepted_tokens += emit.accepted;
            if emit.accepted < draft.len() {
                stats.rejected_rounds += 1;
            }
            let mut stopped = false;
            for &token in &draft[..emit.accepted] {
                generated.push(token);
                emitted.push(token);
                if params.eos_token_id == token {
                    stopped = true;
                    break;
                }
                if emitted.len() >= params.max_length {
                    stopped = true;
                    break;
                }
            }
            if stopped {
                break;
            }
            match emit.correction {
                None => break,
                Some(c) => {
                    generated.push(c.token_id);
                    emitted.push(c.token_id);
                    if c.is_eos {
                        break;
                    }
                }
            }
        }
        emitted
    }

    /// The next up-to-`k` tokens sequential greedy WOULD emit from `seq` — the
    /// "perfect drafter" (teeth: full agreement must actually be accepted).
    fn greedy_lookahead(
        oracle: &dyn Fn(&[u32]) -> Mat,
        seq: &[u32],
        k: usize,
        params: &DecodeParams,
    ) -> Vec<u32> {
        let mut ctx = seq.to_vec();
        let mut out = Vec::new();
        for _ in 0..k {
            let step = decode_step(&oracle(&ctx), &ctx, params).expect("lookahead step");
            out.push(step.token_id);
            if step.is_eos {
                break;
            }
            ctx.push(step.token_id);
        }
        out
    }

    /// Assert one (oracle, prompt, drafter) case: the speculative stream equals
    /// the sequential greedy stream byte-for-byte, whatever the drafter proposed.
    fn assert_drafter_harmless(
        label: &str,
        oracle: &dyn Fn(&[u32]) -> Mat,
        prompt: &[u32],
        max_length: usize,
        drafter: &mut dyn FnMut(&[u32]) -> Vec<u32>,
        stats: &mut RoundStats,
    ) {
        let p = params(max_length);
        let seq = seq_generate(oracle, prompt, &p);
        let spec = spec_generate_with_drafter(oracle, prompt, &p, drafter, stats);
        assert_eq!(
            spec, seq,
            "{label}: adversarial drafter changed the emitted stream \
             (prompt={prompt:?} ml={max_length})"
        );
    }

    /// GATE (bd-1azu.36 fault-injection): NO drafter behavior — plausible garbage,
    /// out-of-vocab/wild ids, EOS spam, oversized blocks far past
    /// [`SPEC_DRAFT_MAX`], always-empty proposals, or the true greedy continuation
    /// — changes the emitted stream: it is byte-for-byte sequential greedy in
    /// every case. Prompts + caps stay under the 35-gram window so this battery
    /// runs ban-free (the ban path has its own dedicated fixture below); teeth
    /// assertions prove accepts, rejects, AND fallbacks all actually ran.
    #[test]
    fn adversarial_drafters_never_change_the_emitted_stream() {
        let oracle: fn(&[u32]) -> Mat = content_oracle;
        let prompts: [&[u32]; 3] = [&[2, 3, 4], &[5, 5, 5, 5], &[2, 6, 2, 6, 3]];

        let mut garbage = RoundStats::default();
        let mut wild = RoundStats::default();
        let mut spam = RoundStats::default();
        let mut oversized = RoundStats::default();
        let mut empty = RoundStats::default();
        let mut echo = RoundStats::default();
        let mut seed: u64 = 0x5EC6_A7E0_D00D_F00D;

        for prompt in prompts {
            for ml in [8usize, 20] {
                // (a) plausible garbage: random-length drafts of random ids in
                // 0..8 (the oracle's alphabet ∪ EOS ∪ two never-emitted ids) —
                // agreement is possible but never trusted.
                let mut s = xs(&mut seed);
                assert_drafter_harmless(
                    "garbage",
                    &oracle,
                    prompt,
                    ml,
                    &mut |_: &[u32]| {
                        let len = 1 + (xs(&mut s) % 6) as usize;
                        (0..len).map(|_| (xs(&mut s) % 8) as u32).collect()
                    },
                    &mut garbage,
                );

                // (b) wild out-of-vocab ids: can never equal a greedy token, and
                // must be rejected without panicking (the verifier never indexes
                // by a draft id).
                assert_drafter_harmless(
                    "wild-ids",
                    &oracle,
                    prompt,
                    ml,
                    &mut |_: &[u32]| vec![u32::MAX, V as u32, 0x7FFF_FFFF],
                    &mut wild,
                );

                // (c) EOS spam: a forged-termination attempt every round.
                assert_drafter_harmless(
                    "eos-spam",
                    &oracle,
                    prompt,
                    ml,
                    &mut |_: &[u32]| vec![EOS; 4],
                    &mut spam,
                );

                // (d) oversized: 64 tokens (>> SPEC_DRAFT_MAX) of id 30, which the
                // content oracle never emits — the budget is a proposal knob, not
                // a safety boundary the verifier relies on.
                assert_drafter_harmless(
                    "oversized",
                    &oracle,
                    prompt,
                    ml,
                    &mut |_: &[u32]| vec![30u32; 64],
                    &mut oversized,
                );

                // (e) always-empty: the loop must ride the sequential fallback.
                assert_drafter_harmless(
                    "empty",
                    &oracle,
                    prompt,
                    ml,
                    &mut |_: &[u32]| Vec::new(),
                    &mut empty,
                );

                // (f) echo of the true greedy continuation: full accepts.
                let p_look = params(ml);
                assert_drafter_harmless(
                    "echo",
                    &oracle,
                    prompt,
                    ml,
                    &mut |g: &[u32]| greedy_lookahead(&oracle, g, 4, &p_look),
                    &mut echo,
                );
            }
        }

        // TEETH — each guaranteed by construction (pure oracles, fixed seeds):
        assert!(
            garbage.rounds > 0,
            "garbage drafter never reached the verifier"
        );
        assert!(
            wild.rounds > 0,
            "wild-id drafter never reached the verifier"
        );
        assert_eq!(
            wild.rejected_rounds, wild.rounds,
            "an out-of-vocab id can never equal a greedy token"
        );
        assert_eq!(wild.accepted_tokens, 0, "wild ids must never be accepted");
        assert!(
            spam.rounds > 0,
            "EOS-spam drafter never reached the verifier"
        );
        assert!(
            oversized.rounds > 0,
            "oversized drafter never reached the verifier"
        );
        assert_eq!(
            oversized.accepted_tokens, 0,
            "an id the oracle never emits must never be accepted"
        );
        assert_eq!(
            empty.rounds, 0,
            "an empty draft must not reach the verifier"
        );
        assert!(empty.fallbacks > 0, "empty-draft fallback never exercised");
        assert!(echo.rounds > 0, "echo drafter never reached the verifier");
        assert!(
            echo.accepted_tokens > 0,
            "full agreement was never accepted — the harness has no teeth"
        );
        assert_eq!(
            echo.rejected_rounds, 0,
            "the true greedy continuation must never be rejected"
        );
    }

    /// Direct verifier-level fault: a draft far beyond the [`SPEC_DRAFT_MAX`]
    /// budget is verified position by position — truncated at the first
    /// divergence with the true greedy correction, or fully accepted when it
    /// genuinely agrees — never trusted, never panicking. Distinct-id targets
    /// keep the 35-gram ban silent (no 34-gram ever recurs), so greedy is the raw
    /// per-row argmax throughout.
    #[test]
    fn oversized_draft_is_verified_position_by_position_never_trusted() {
        const K: usize = 64;
        let p = params(1000);
        let target: Vec<u32> = (0..=K).map(|i| i as u32 + 2).collect();
        let rows: Vec<Mat> = target.iter().map(|&t| peak_row(t)).collect();

        // (a) agrees for 5 positions then diverges: truncated exactly there, the
        // correction is the true greedy token, the oversized tail is discarded.
        let mut draft: Vec<u32> = target[..K].to_vec();
        draft[5] = 99;
        assert!(
            draft.len() > SPEC_DRAFT_MAX,
            "the fault draft must dwarf the live proposal budget"
        );
        let emit = resolve_round(&[], &draft, &rows, &p).unwrap();
        assert_eq!(
            emit.accepted, 5,
            "first divergence truncates; budget ignored"
        );
        assert_eq!(
            emit.correction.expect("correction at divergence").token_id,
            target[5]
        );

        // (b) fully-agreeing oversized draft: all K accepted + the bonus token.
        let emit = resolve_round(&[], &target[..K], &rows, &p).unwrap();
        assert_eq!(emit.accepted, K, "genuine agreement is accepted in full");
        assert_eq!(
            emit.correction.expect("bonus after full accept").token_id,
            target[K]
        );
    }

    /// The 69-token spec.rs ban fixture: the trailing 34 tokens repeat an earlier
    /// 34-gram whose observed completion was token 7, so the frozen 35/128
    /// blocker bans 7 at the next position.
    fn history_banning_token_7() -> Vec<u32> {
        let prefix: Vec<u32> = (20u32..54).collect();
        assert_eq!(prefix.len(), 34);
        let mut h = Vec::with_capacity(69);
        h.extend_from_slice(&prefix); // leading 34-gram
        h.push(7); // its observed completion
        h.extend_from_slice(&prefix); // current prefix == leading 34-gram
        h
    }

    /// A malicious drafter cannot SMUGGLE a banned token past the frozen 35-gram
    /// ban: the verifier recomputes greedy WITH the ban, rejects the raw-argmax
    /// token the ban forbids, and corrects to the ban-aware choice.
    #[test]
    fn drafter_cannot_smuggle_a_banned_token() {
        let history = history_banning_token_7();
        // Raw argmax is 7 (banned in this context); runner-up is 6.
        let rows = vec![row_peaked(7, 6), peak_row(8)];
        let p = params(1000);
        // Ground truth via the production chooser: the ban flips greedy to 6.
        let g = sample(&rows[0], &history, &p).expect("production chooser");
        assert_eq!(g, 6, "the 35-gram ban must flip greedy from 7 to 6");

        let emit = resolve_round(&history, &[7], &rows, &p).unwrap();
        assert_eq!(emit.accepted, 0, "the banned token must not be accepted");
        assert_eq!(
            emit.correction.expect("ban-aware correction").token_id,
            6,
            "the correction must be the ban-aware greedy token"
        );
    }

    /// A drafter cannot FORGE termination: a proposed EOS where greedy would not
    /// pick EOS is rejected, and the stream continues with the true token.
    #[test]
    fn drafter_cannot_forge_eos_termination() {
        let rows = vec![peak_row(5), peak_row(6)];
        let p = params(1000);
        let emit = resolve_round(&[], &[EOS], &rows, &p).unwrap();
        assert_eq!(emit.accepted, 0, "a forged EOS must be rejected");
        let c = emit.correction.expect("correction after forged EOS");
        assert_eq!(c.token_id, 5);
        assert!(!c.is_eos, "the stream must not terminate on a forged EOS");
    }

    /// An empty draft resolves to exactly one pure sequential step: nothing
    /// accepted, and the correction equals the production chooser's decision.
    #[test]
    fn empty_draft_resolves_to_the_pure_sequential_step() {
        let rows = vec![peak_row(9)];
        let p = params(1000);
        let emit = resolve_round(&[2, 3], &[], &rows, &p).unwrap();
        assert_eq!(emit.accepted, 0);
        let c = emit
            .correction
            .expect("the round still yields the sequential token");
        assert_eq!(c.token_id, 9);
        assert_eq!(
            c.token_id,
            sample(&rows[0], &[2, 3], &p).expect("production chooser"),
            "the empty-draft round must equal the sequential chooser"
        );
    }

    /// Verify rows SHORTER than the `draft.len() + 1` contract: only verified
    /// positions are ever emitted — acceptance caps at the available rows and a
    /// missing correction row yields NO token, never a fabricated one.
    #[test]
    fn short_verify_rows_fail_closed() {
        let p = params(1000);
        let draft = [3u32, 4, 2];
        // 2 rows for a 3-token draft: positions 0 and 1 verifiable, 2 is not.
        let rows = vec![peak_row(3), peak_row(4)];
        let emit = resolve_round(&[], &draft, &rows, &p).unwrap();
        assert_eq!(emit.accepted, 2, "the unverifiable tail is not accepted");
        assert!(
            emit.correction.is_none(),
            "no verify row to correct from -> no token"
        );
    }

    /// A MALFORMED (empty) verify row fails CLOSED: acceptance stops before the
    /// unverifiable position, and a malformed correction row is a hard error —
    /// the round never emits an unverified token.
    #[test]
    fn malformed_verify_row_never_emits_unverified_tokens() {
        let p = params(1000);
        // Empty row at the live position: nothing is verifiable and the
        // correction row itself is malformed -> error, no fabricated token.
        let r = resolve_round(&[], &[3], &[Mat::from_vec(1, 0, vec![]), peak_row(4)], &p);
        assert!(r.is_err(), "a malformed correction row must fail closed");

        // Empty row mid-draft: acceptance stops BEFORE the unverifiable position
        // (position 0 verifies against a well-formed row and is accepted).
        let good = peak_row(3);
        let bad = Mat::from_vec(1, 0, vec![]);
        let rows: Vec<&[f32]> = vec![good.row(0), bad.row(0)];
        assert_eq!(
            accept_longest(&[], &[3, 4], &rows, EOS),
            1,
            "acceptance must stop at the first unverifiable position"
        );
    }
}
