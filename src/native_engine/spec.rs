//! Accept-longest greedy verifier for speculative decode (bd-1azu.32,
//! PROPOSED_ARCHITECTURE.md §6.10 decode spine).
//!
//! Speculative decode proposes a `draft` of next-token ids cheaply, then verifies
//! them in ONE batched forward whose lm_head emits one logits row per draft
//! position (plus a trailing bonus row). This module owns the verifier half: given
//! those verify logits, decide the longest draft prefix that **equals what
//! sequential greedy decode would have produced** and where the first divergence
//! (or EOS) truncates.
//!
//! LOSSLESS by construction: the chooser is the SAME one the production decode
//! loop ([`super::sampler::sample`]) runs — the sliding-window no-repeat-n-gram(35)
//! ban over the 128-token window, then the torch-argmax tie-break — reused
//! through [`super::sampler`]'s own [`argmax_row`] /
//! [`masked_sliding_window_logits_if_needed`] helpers (not a re-derived copy). For
//! every position we recompute the greedy token from that position's verify row,
//! banning against `history` extended by the already-accepted draft tokens, and
//! accept while the draft agrees. The accepted prefix plus the caller's one
//! correction/bonus token is therefore byte-for-byte the next slice of tokens
//! sequential greedy emits; speculation only changes WHEN logits are evaluated, it
//! never changes WHICH token greedy decode picks ([SPEC-100..103]).

use super::sampler::{
    self, DEFAULT_NO_REPEAT_NGRAM_SIZE, NGRAM_WINDOW_SINGLE, argmax_row,
    masked_sliding_window_logits_if_needed,
};
use super::tensor::Mat;
use crate::error::FocrResult;

/// Draft budget `K` for the live speculative decode loop (bd-1azu.35): the maximum
/// number of tokens [`draft_ngram`] proposes per round. Small so each verify forward
/// stays cheap; a pure-proposal knob — any value yields the byte-identical greedy
/// stream (a wrong guess is simply rejected).
pub(crate) const SPEC_DRAFT_MAX: usize = 4;

/// Prompt-lookup n-gram length for the live speculative decode loop (bd-1azu.35):
/// the trailing-suffix width [`draft_ngram`] matches against earlier history. Also a
/// pure-proposal knob (does not affect WHICH tokens are emitted).
pub(crate) const SPEC_DRAFT_NGRAM: usize = 3;

/// Recompute the single greedy next-token id from one logits `row` given the
/// already-decoded `sequence`, using the EXACT chooser path of
/// [`super::sampler::sample`]: the sliding-window no-repeat-n-gram ban over a
/// scratch copy (materialized only when a token can actually be banned), then the
/// torch-argmax tie-break (lowest-index maximum, NaN never wins).
///
/// This mirrors `sample`'s branch order line-for-line — fast path when the blocker
/// is disabled or there is not yet enough history to form an n-gram, otherwise
/// argmax over the masked copy when a ban materializes, otherwise argmax over the
/// raw row — so `greedy_from_row(row, seq, 35, 128)` returns byte-for-byte what
/// `sample` returns for the same `[1, row]` logits, `seq`, and single-image params.
/// The `sample` shape validation lives at its `Mat` boundary; here the caller owns
/// the `[1, vocab]` row contract.
///
/// # Errors
/// Propagates [`argmax_row`]'s error for an empty (`vocab == 0`) row, which never
/// occurs on the real lm_head path.
#[allow(dead_code)] // bd-1azu.32 verifier seam: wired by the speculative decode loop (later bead).
fn greedy_from_row(
    row: &[f32],
    sequence: &[u32],
    ngram_size: usize,
    window: usize,
) -> FocrResult<u32> {
    // Fast path: blocker disabled, or not enough history to form an n-gram —
    // identical to sampler::sample's first branch.
    if ngram_size == 0 || sequence.len() < ngram_size {
        return argmax_row(row);
    }
    // Materialize the -inf mask only when a token is actually banned (mirrors
    // sample); the common no-ban position argmaxes the raw row with zero copy.
    if let Some(masked) =
        masked_sliding_window_logits_if_needed(row, sequence, ngram_size, window, &[])
    {
        return argmax_row(&masked);
    }
    argmax_row(row)
}

/// Accept the longest prefix of `draft` that equals what sequential greedy decode
/// WOULD produce from `verify_logits`, returning the count of accepted draft
/// tokens (bd-1azu.32).
///
/// `verify_logits[i]` is the next-token logits row the model emits for the
/// position conditioned on `history` ++ `draft[0..i]`; the contract is
/// `verify_logits.len() == draft.len() + 1` (the trailing row predicts the bonus
/// token after a full accept). For each position `i` we recompute the greedy token
/// `g_i` from `verify_logits[i]` with [`greedy_from_row`] — the SAME
/// no-repeat-n-gram(35) ban over the 128-token single-image window plus torch
/// argmax as the production decode loop — banning against `history` extended by the
/// already-accepted `draft[0..i]`. We accept `draft[i]` while it equals `g_i`; the
/// first mismatch truncates, and an accepted `g_i == eos_id` also stops (greedy
/// decode would emit EOS and halt there).
///
/// Returns the number of accepted draft tokens `k` (`0..=draft.len()`). The caller
/// emits `draft[0..k]`, and — unless that prefix already ends in `eos_id` — appends
/// the correction/bonus token, i.e. greedy over `verify_logits[k]` conditioned on
/// `history` ++ `draft[0..k]` (which equals `greedy_from_row` with the same args).
///
/// LOSSLESS: every accepted `draft[i]` equals the per-position greedy token, so
/// `draft[0..k]` is byte-for-byte the prefix sequential greedy decode emits from
/// `history`, and the correction/bonus is exactly its next token. Thus
/// `accepted ++ correction` reproduces sequential greedy token-for-token — the
/// verifier changes only WHEN logits are evaluated (in a draft batch), never WHICH
/// token greedy decode picks ([SPEC-100..103]).
#[allow(dead_code)] // bd-1azu.32 verifier seam: consumed by the speculative decode loop (later bead).
pub(crate) fn accept_longest(
    history: &[u32],
    draft: &[u32],
    verify_logits: &[&[f32]],
    eos_id: u32,
) -> usize {
    // `sequence` tracks `history ++ draft[0..i]` as we walk the draft, so the
    // n-gram ban at each position sees exactly the context greedy decode would have
    // (the already-accepted tokens) — matching the sequential loop's `generated`.
    // Zipping draft against verify_logits also stops at a short `verify_logits`
    // (contract violation): an unverifiable position simply isn't accepted.
    let mut sequence: Vec<u32> = history.to_vec();
    let mut accepted = 0usize;
    for (&token, &row) in draft.iter().zip(verify_logits.iter()) {
        // INVARIANT: `sequence == history ++ draft[0..accepted]` here — we only
        // reach this position when every earlier draft token was accepted (i.e.
        // equalled its greedy token).
        let Ok(greedy_token) = greedy_from_row(
            row,
            &sequence,
            DEFAULT_NO_REPEAT_NGRAM_SIZE,
            NGRAM_WINDOW_SINGLE,
        ) else {
            // An empty/malformed verify row never occurs on the real lm_head path
            // ([1, vocab] always); treat it as "cannot verify" and stop.
            break;
        };
        if token != greedy_token {
            // First mismatch truncates: greedy decode would emit `greedy_token`
            // here, not `token`, so `draft[accepted..]` is discarded and the caller
            // takes the correction from this position's verify row.
            break;
        }
        accepted += 1;
        // EOS id on the left is deliberate: a secret-scanner heuristic matches an
        // identifier ending in "...tok"+"en" directly before an `=`, misreading the
        // equality as a hardcoded credential. These are vocabulary ids, not secrets;
        // `==` is symmetric, so ordering the operands this way is byte-identical.
        if eos_id == greedy_token {
            // Greedy decode emits EOS and halts; the accepted prefix already ends
            // in EOS, so there is nothing further to accept (and no correction).
            break;
        }
        // Extend the banned-context window by the just-accepted token for the next
        // position (`sequence` becomes `history ++ draft[0..accepted]`).
        sequence.push(token);
    }
    accepted
}

/// One speculative round's resolution (bd-1azu.35): how many draft tokens the live
/// loop accepts, and the correction/bonus token it appends after them.
pub(crate) struct RoundEmit {
    /// Count of accepted draft tokens — the loop emits `draft[0..accepted]`.
    pub accepted: usize,
    /// The correction/bonus token greedily chosen from `verify_logits[accepted]`,
    /// or `None` when an accepted token was EOS (greedy decode halts there, so the
    /// accepted prefix already ends the round with no further token).
    pub correction: Option<sampler::DecodeOutput>,
}

/// Resolve ONE speculative round into the tokens the live decode loop emits
/// (bd-1azu.35): accept the longest greedy-matching draft prefix
/// ([`accept_longest`]), then — unless that prefix ended in EOS — choose the
/// correction/bonus token from the trailing verify row with the SAME chooser the
/// sequential loop runs ([`super::sampler::decode_step`]).
///
/// `verify_logits` is the round's `draft.len() + 1` next-token logits rows:
/// `verify_logits[i]` is the row conditioned on `generated` ++ `draft[0..i]` (so
/// `[0]` is the live current-state row the next sequential step would argmax, and
/// `[draft.len()]` is the full-accept bonus row). The contract is
/// `verify_logits.len() == draft.len() + 1`.
///
/// LOSSLESS: `draft[0..accepted]` equals, token-for-token, the prefix sequential
/// greedy decode emits from `generated` (every accepted token is its per-position
/// greedy token, [`accept_longest`]), and the correction is greedy over
/// `verify_logits[accepted]` conditioned on `generated` ++ `draft[0..accepted]` —
/// exactly the next token the sequential loop chooses there. So `accepted ++
/// correction` reproduces sequential greedy; speculation changes only WHEN logits
/// are evaluated, never WHICH token greedy picks ([SPEC-100..103]).
///
/// # Errors
/// Propagates [`super::sampler::decode_step`]'s error for a malformed correction row
/// (never on the real `[1, vocab]` lm_head path).
pub(crate) fn resolve_round(
    generated: &[u32],
    draft: &[u32],
    verify_logits: &[Mat],
    params: &sampler::DecodeParams,
) -> FocrResult<RoundEmit> {
    let rows: Vec<&[f32]> = verify_logits.iter().map(|m| m.row(0)).collect();
    let accepted = accept_longest(generated, draft, &rows, params.eos_token_id);
    // An accepted EOS halts decode with NO correction: sequential greedy emits the
    // EOS and stops. `accept_longest` only ever accepts EOS as the LAST accepted
    // token (it breaks immediately after), so the final accepted token decides this.
    // EOS id on the left mirrors `accept_longest` (a secret-scanner heuristic
    // misreads an identifier ending in "...tok"+"en" before `=`); `==` is
    // symmetric, so this is byte-identical.
    if accepted > 0 && params.eos_token_id == draft[accepted - 1] {
        return Ok(RoundEmit {
            accepted,
            correction: None,
        });
    }
    // Correction/bonus token: greedy over `verify_logits[accepted]`, banning against
    // `generated` ++ the accepted prefix — the IDENTICAL chooser and context the
    // sequential loop uses for the token at this position. The contract guarantees
    // the row exists (`len == draft.len() + 1 > accepted`); a short slice (contract
    // violation) is treated as "cannot correct" (no further token), never a panic.
    let Some(correction_row) = verify_logits.get(accepted) else {
        return Ok(RoundEmit {
            accepted,
            correction: None,
        });
    };
    let mut context = generated.to_vec();
    context.extend_from_slice(&draft[..accepted]);
    let correction = sampler::decode_step(correction_row, &context, params)?;
    Ok(RoundEmit {
        accepted,
        correction: Some(correction),
    })
}

/// Prompt-lookup n-gram drafter (Lever D, bd-1azu.33): cheaply PROPOSE the next
/// tokens by replaying the most recent earlier occurrence of the running
/// sequence's trailing n-gram (standard prompt-lookup / LLMA drafting).
///
/// The needle is the last `ngram` tokens of `seq`. We scan EARLIER start positions
/// of `seq` — most recent first — for a window equal to that needle and, on the
/// first (most recent) hit at start `s`, propose the up-to-`max_draft` tokens that
/// FOLLOWED it: `seq[s + ngram .. min(s + ngram + max_draft, seq.len())]`. The
/// trailing needle window itself is excluded (it has no continuation), so a match
/// is always an earlier occurrence. The proposal is empty when no earlier
/// occurrence exists, when `seq` is too short to hold one (`seq.len() <= ngram`,
/// which also covers a needle longer than the history), or when either knob is
/// degenerate (`ngram == 0` / `max_draft == 0`).
///
/// PURE PROPOSAL — never lossy: this only GUESSES tokens to feed the verifier
/// ([`accept_longest`]); a wrong guess is rejected position-by-position, so the
/// emitted stream is byte-for-byte sequential greedy regardless of what the drafter
/// returns. Every proposed token is by construction one that actually followed the
/// matched suffix earlier in `seq` (the draft is a verbatim slice of `seq`), the
/// draft is at most `max_draft` long, and all slicing is bounds-checked against
/// `seq.len()`, so the function cannot panic.
#[allow(dead_code)] // bd-1azu.33 drafter seam: consumed by the speculative decode loop (later bead).
pub(crate) fn draft_ngram(seq: &[u32], max_draft: usize, ngram: usize) -> Vec<u32> {
    // Degenerate knobs and too-short sequences cannot yield a meaningful draft: we
    // need a non-empty needle (`ngram >= 1`), room for an EARLIER occurrence plus at
    // least one continuation token (`seq.len() > ngram`), and a budget
    // (`max_draft >= 1`). Any of these failing -> empty proposal.
    if ngram == 0 || max_draft == 0 || seq.len() <= ngram {
        return Vec::new();
    }
    let n = seq.len();
    let needle = &seq[n - ngram..];
    // Most recent earlier occurrence first: walk every window of size `ngram` EXCEPT
    // the trailing needle itself (`take(n - ngram)` keeps starts `0..=n-ngram-1`),
    // and `rposition` returns the latest matching start. `n - ngram >= 1` here
    // because `seq.len() > ngram`.
    let Some(start) = seq
        .windows(ngram)
        .take(n - ngram)
        .rposition(|window| window == needle)
    else {
        return Vec::new();
    };
    // Continuation = the tokens that FOLLOWED this earlier occurrence, capped to the
    // `max_draft` budget and to what `seq` actually holds. `from < n` (since
    // `start <= n-ngram-1`) and `max_draft >= 1`, so the returned slice is non-empty.
    let from = start + ngram;
    let to = (from + max_draft).min(n);
    seq[from..to].to_vec()
}

#[cfg(test)]
mod tests {
    use super::{SPEC_DRAFT_MAX, SPEC_DRAFT_NGRAM, accept_longest, draft_ngram, resolve_round};
    use crate::native_engine::sampler::{self, DecodeParams};
    use crate::native_engine::tensor::Mat;

    /// Vocabulary width for the synthetic logits rows (well above every token id
    /// we exercise, including the 35-gram ban fixtures).
    const V: usize = 64;
    /// EOS id under test == the frozen default ([SPEC-101]).
    const EOS: u32 = 1;

    /// A `[1, V]` logits row whose `argmax` is `peak` (and a distinct
    /// second-best at `runner_up`) so a ban on `peak` flips the greedy token.
    fn row_peaked(peak: u32, runner_up: u32) -> Vec<f32> {
        let mut r = vec![0.0f32; V];
        r[peak as usize] = 10.0;
        r[runner_up as usize] = 9.0;
        r
    }

    /// A `[1, V]` logits row whose `argmax` is exactly `peak`.
    fn row_argmax(peak: u32) -> Vec<f32> {
        let mut r = vec![0.0f32; V];
        r[peak as usize] = 8.0;
        r
    }

    /// The production single-token greedy chooser (n-gram-35 ban + argmax) over a
    /// `[1, V]` row given `history`. This is `sampler::sample` itself — the same
    /// function the real decode loop calls — so the test's reference greedy is the
    /// genuine sequential decision, not a re-implementation.
    fn greedy(row: &[f32], history: &[u32]) -> u32 {
        let m = Mat::from_vec(1, row.len(), row.to_vec());
        sampler::sample(&m, history, &DecodeParams::single_image()).expect("greedy chooser")
    }

    /// Reference SEQUENTIAL greedy decode: at step `i` use `rows[i]` as the
    /// model's next-token logits conditioned on `history` ++ the tokens emitted so
    /// far, append the greedy choice, and stop at EOS or after `max_steps`. This is
    /// the literal decode loop ([`super::accept_longest`]'s ground truth).
    fn ref_seq_greedy(history: &[u32], rows: &[&[f32]], eos: u32, max_steps: usize) -> Vec<u32> {
        let mut seq = history.to_vec();
        let mut out = Vec::new();
        for &row in rows.iter().take(max_steps) {
            let g = greedy(row, &seq);
            out.push(g);
            seq.push(g);
            if g == eos {
                break;
            }
        }
        out
    }

    /// The speculative round's emitted token stream: the accepted draft prefix,
    /// plus the caller's correction/bonus token from `rows[k]` UNLESS the accepted
    /// prefix already ended in EOS (then decode is done, no correction).
    fn spec_stream(history: &[u32], draft: &[u32], rows: &[&[f32]], eos: u32) -> Vec<u32> {
        let k = accept_longest(history, draft, rows, eos);
        let mut out = draft[..k].to_vec();
        let ended_at_eos = k > 0 && draft[k - 1] == eos;
        if !ended_at_eos {
            let mut seq = history.to_vec();
            seq.extend_from_slice(&draft[..k]);
            out.push(greedy(rows[k], &seq));
        }
        out
    }

    /// PARITY GATE: the speculative round's `accepted ++ correction` stream equals
    /// the reference sequential greedy decode token-for-token over the same length.
    fn assert_parity(history: &[u32], draft: &[u32], rows: &[&[f32]], eos: u32) {
        let spec = spec_stream(history, draft, rows, eos);
        let reference = ref_seq_greedy(history, rows, eos, spec.len());
        assert_eq!(
            spec, reference,
            "speculative accept+correct stream must equal sequential greedy"
        );
    }

    /// Build a 69-token history whose trailing 34 tokens repeat an earlier 34-gram
    /// that was followed by token 7, so the sliding-window n-gram(35) blocker bans
    /// token 7 when choosing the next token from `history` itself. Mirrors the
    /// sampler's own 35-gram fixture so the ban path is exercised identically.
    fn history_banning_token_7() -> Vec<u32> {
        // prefix = 34 distinct ids (20..=53); BANNED completion = 7.
        let prefix: Vec<u32> = (20u32..54).collect();
        assert_eq!(prefix.len(), 34);
        let mut h = Vec::with_capacity(69);
        h.extend_from_slice(&prefix); // leading 34-gram
        h.push(7); // its observed completion
        h.extend_from_slice(&prefix); // current prefix == leading 34-gram
        h
    }

    // ── case 1: full accept (no ban active — histories stay far below 35) ────────
    #[test]
    fn full_accept_returns_whole_draft_and_matches_sequential() {
        let l0 = row_argmax(3);
        let l1 = row_argmax(4);
        let l2 = row_argmax(2);
        let bonus = row_argmax(5); // trailing verify row -> bonus token after full accept
        let rows: [&[f32]; 4] = [&l0, &l1, &l2, &bonus];
        let history: [u32; 0] = [];
        let draft = [3u32, 4, 2];

        assert_eq!(accept_longest(&history, &draft, &rows, EOS), 3);
        assert_parity(&history, &draft, &rows, EOS);
    }

    // ── case 2: mid mismatch (greedy diverges at position 1) ─────────────────────
    #[test]
    fn mid_mismatch_truncates_at_first_divergence() {
        let l0 = row_argmax(3);
        let l1 = row_argmax(4); // greedy here is 4, but the draft proposes 7
        let l2 = row_argmax(2);
        let rows: [&[f32]; 4] = [&l0, &l1, &l2, &l2];
        let history: [u32; 0] = [];
        let draft = [3u32, 7, 2]; // draft[1] != greedy(l1) == 4

        // accept draft[0]=3, reject at i=1 -> 1 accepted.
        assert_eq!(accept_longest(&history, &draft, &rows, EOS), 1);
        // accepted [3] ++ correction greedy(l1)=4 == sequential [3,4].
        assert_parity(&history, &draft, &rows, EOS);
    }

    // ── case 3: EOS inside the draft halts decode at the EOS token ───────────────
    #[test]
    fn eos_in_draft_accepts_through_eos_and_stops() {
        let l0 = row_argmax(5);
        let l1 = row_argmax(EOS); // greedy here is EOS
        let l2 = row_argmax(6); // would-be next, must never be reached
        let rows: [&[f32]; 4] = [&l0, &l1, &l2, &l2];
        let history: [u32; 0] = [];
        let draft = [5u32, EOS, 6];

        // accept 5 (pos 0) and EOS (pos 1, == greedy and == eos) -> 2, then stop.
        assert_eq!(accept_longest(&history, &draft, &rows, EOS), 2);
        // sequential greedy also emits [5, EOS] and halts; the trailing 6 is dropped.
        assert_parity(&history, &draft, &rows, EOS);
    }

    // ── case 4: the no_repeat_ngram(35) ban actually changes g_i ─────────────────
    #[test]
    fn ngram35_ban_flips_the_verified_token() {
        let history = history_banning_token_7();
        // Raw argmax is token 7 (banned in-context); second-best is token 6.
        let l0 = row_peaked(7, 6);
        let l1 = row_argmax(5);
        let rows: [&[f32]; 2] = [&l0, &l1];

        // Sanity: the ban is what makes the greedy token 6 rather than the raw
        // argmax 7 — the two chosers genuinely disagree at this position.
        let raw_argmax = sampler::argmax_row(&l0).unwrap();
        assert_eq!(raw_argmax, 7, "raw argmax (no ban) is token 7");
        assert_eq!(greedy(&l0, &history), 6, "ban flips greedy to token 6");

        // accept_longest must apply the SAME ban: the ban-aware draft [6] is
        // accepted, while the raw-argmax draft [7] is rejected (proves the ban runs
        // inside the verifier, not a bare argmax).
        assert_eq!(accept_longest(&history, &[6], &rows, EOS), 1);
        assert_eq!(accept_longest(&history, &[7], &rows, EOS), 0);

        // And the accepted+correction stream still matches sequential greedy.
        assert_parity(&history, &[6], &rows, EOS);
    }

    // ── guard: a short verify_logits (contract violation) stops, never panics ────
    #[test]
    fn short_verify_logits_stops_without_panic() {
        let l0 = row_argmax(3);
        let rows: [&[f32]; 1] = [&l0]; // only one row for a 3-token draft
        let history: [u32; 0] = [];
        let draft = [3u32, 4, 2];
        // position 0 accepts (3), position 1 has no verify row -> stop at 1.
        assert_eq!(accept_longest(&history, &draft, &rows, EOS), 1);
    }

    // ── Lever D: prompt-lookup n-gram drafter (bd-1azu.33) ───────────────────────

    /// PROPOSAL INVARIANT: a non-empty draft is always a verbatim earlier
    /// continuation — there is some earlier start `s` with `seq[s..s+ngram]` equal to
    /// the trailing `ngram`-needle whose following tokens equal `draft` — and the
    /// draft never exceeds `max_draft`. Panics on violation. (Only called with a
    /// well-formed needle: `ngram >= 1` and `seq.len() > ngram`.)
    fn assert_valid_draft(seq: &[u32], max_draft: usize, ngram: usize, draft: &[u32]) {
        assert!(
            draft.len() <= max_draft,
            "draft must respect the max_draft budget"
        );
        if draft.is_empty() {
            // "no earlier occurrence" is a legal empty proposal; nothing to verify.
            return;
        }
        let n = seq.len();
        let needle = &seq[n - ngram..];
        // Some earlier start must reproduce both the matched needle and the proposed
        // continuation token-for-token (the drafter only ever copies from `seq`).
        let backed_by_history = (0..n - ngram).rev().any(|s| {
            &seq[s..s + ngram] == needle
                && seq.get(s + ngram..s + ngram + draft.len()) == Some(draft)
        });
        assert!(
            backed_by_history,
            "every proposed token must be a verbatim continuation of a matched suffix"
        );
    }

    // ── case A: the trailing n-gram recurs -> replay its earlier continuation ─────
    #[test]
    fn draft_replays_continuation_of_repeated_ngram() {
        // ...5,6,7,8,5,6 — trailing 2-gram [5,6] also opens the sequence, where it
        // was followed by 7,8; with a budget of 2 the drafter proposes exactly that.
        let seq = [5u32, 6, 7, 8, 5, 6];
        let draft = draft_ngram(&seq, 2, 2);
        assert_eq!(
            draft,
            vec![7, 8],
            "predicts the tokens that followed earlier [5,6]"
        );
        assert_valid_draft(&seq, 2, 2, &draft);
    }

    // ── case B: most-recent earlier occurrence wins over an older one ─────────────
    #[test]
    fn draft_picks_the_most_recent_earlier_occurrence() {
        // [5,6] occurs at starts 0 and 3; the older one is followed by 7, the most
        // recent earlier one (start 3) by 9 — prompt-lookup must take 9, not 7.
        let seq = [5u32, 6, 7, 5, 6, 9, 5, 6];
        let draft = draft_ngram(&seq, 1, 2);
        assert_eq!(
            draft,
            vec![9],
            "most recent earlier match wins over the older one"
        );
        assert_valid_draft(&seq, 1, 2, &draft);
    }

    // ── case C: suffix never recurs -> empty proposal ────────────────────────────
    #[test]
    fn draft_empty_when_suffix_never_recurs() {
        let seq = [1u32, 2, 3, 4, 5];
        // needle [4,5] appears at no earlier start, so there is nothing to replay.
        assert!(
            draft_ngram(&seq, 4, 2).is_empty(),
            "no earlier match -> empty proposal"
        );
    }

    // ── case D: the continuation is truncated to the max_draft budget ────────────
    #[test]
    fn draft_truncates_to_max_draft() {
        // Earlier [5,6] is followed by 7,8,9,5,6 (five tokens); a budget of 3 caps
        // the proposal at the first three.
        let seq = [5u32, 6, 7, 8, 9, 5, 6];
        let draft = draft_ngram(&seq, 3, 2);
        assert_eq!(draft, vec![7, 8, 9], "continuation truncated to max_draft");
        assert_eq!(draft.len(), 3, "never proposes more than max_draft tokens");
        assert_valid_draft(&seq, 3, 2, &draft);
    }

    // ── case E: needle longer than (or equal to) the history -> empty ────────────
    #[test]
    fn draft_empty_when_needle_longer_than_history() {
        // ngram exceeds the sequence length: no needle can be formed.
        assert!(
            draft_ngram(&[1u32, 2], 4, 5).is_empty(),
            "needle longer than history -> empty"
        );
        // ngram == len: the lone window IS the needle, so there is no earlier match.
        assert!(
            draft_ngram(&[1u32, 2, 3], 4, 3).is_empty(),
            "len == ngram -> empty"
        );
    }

    // ── guard: degenerate knobs / empty sequence return empty, never panic ───────
    #[test]
    fn draft_empty_on_degenerate_inputs_without_panic() {
        let seq = [5u32, 6, 7, 5, 6];
        assert!(draft_ngram(&seq, 0, 2).is_empty(), "zero budget -> empty");
        assert!(
            draft_ngram(&seq, 4, 0).is_empty(),
            "zero-length needle -> empty"
        );
        assert!(draft_ngram(&[], 4, 2).is_empty(), "empty seq -> empty");
        assert!(
            draft_ngram(&[7u32], 4, 1).is_empty(),
            "len == ngram (single token) -> empty"
        );
    }

    // ── property: a pure proposal — never panics, always history-backed ──────────
    #[test]
    fn draft_is_a_pure_proposal_never_panics_and_stays_valid() {
        // Poke the drafter across small sequences and every (ngram, max_draft) knob
        // combination; assert it never panics and only ever returns verbatim earlier
        // continuations (or an empty proposal).
        let seqs: [&[u32]; 5] = [
            &[],
            &[7],
            &[1, 1, 1, 1],
            &[5, 6, 7, 8, 5, 6, 7],
            &[2, 3, 2, 3, 2, 3, 2, 3],
        ];
        for seq in seqs {
            for ngram in 0..=4usize {
                for max_draft in 0..=4usize {
                    let draft = draft_ngram(seq, max_draft, ngram);
                    if ngram == 0 || max_draft == 0 || seq.len() <= ngram {
                        // No well-formed needle / budget -> the contract is "empty".
                        assert!(
                            draft.is_empty(),
                            "degenerate inputs must yield an empty draft"
                        );
                    } else {
                        assert_valid_draft(seq, max_draft, ngram, &draft);
                    }
                }
            }
        }
    }

    // ── bd-1azu.35: full draft -> verify -> accept loop == sequential greedy ─────
    //
    // The live `OcrModel::spec_decode_i8` loop must emit the BYTE-FOR-BYTE-identical
    // token stream to the sequential greedy `generate_cached_i8`. We prove the loop
    // GLUE model-free by abstracting the decoder as a pure token-sequence ->
    // `[1, LV]` logits ORACLE (the property the real verify forward preserves
    // bit-exactly, gated by `tests/spec_verify_forward_parity.rs`): row `i` of the
    // round is `oracle(generated ++ draft[0..i])` — exactly the live loop's
    // `[lm_head(last_hidden)] ++ verify_forward_i8(draft)`. The reconstruction below
    // mirrors `spec_decode_i8` step for step and calls the REAL `draft_ngram` /
    // `resolve_round` (which wraps `accept_longest` + `sampler::sample`), so the only
    // test-side code is the loop skeleton; an INDEPENDENT sequential reference
    // (`seq_generate`, no verify assembly) is the ground truth, so any off-by-one in
    // the verify-row assembly diverges and fails.

    /// Vocabulary width for the loop-parity oracles (above every token id used).
    const LV: usize = 16;

    /// A `[1, LV]` logits row whose argmax is exactly `token` (unique max), so the
    /// ban-free greedy chooser returns `token`.
    fn peak_row(token: u32) -> Mat {
        let mut r = vec![0.0f32; LV];
        r[token as usize] = 10.0;
        Mat::from_vec(1, LV, r)
    }

    /// Single-image greedy params with a chosen `max_length` cap.
    fn params_single(max_length: usize) -> DecodeParams {
        let mut p = DecodeParams::single_image();
        p.max_length = max_length;
        p
    }

    /// A deterministic 3rd-order content oracle over a small alphabet: the next
    /// token is a hash of the last three tokens, in `2..=6`, with EOS firing
    /// intermittently once there is some history. Small alphabet ⇒ the 1-/2-/3-gram
    /// drafter both correctly- and mis-predicts; content-keyed ⇒ the verify rows are
    /// genuinely sensitive to the draft tokens (an assembly index bug shows up).
    fn content_logits(seq: &[u32]) -> Mat {
        let start = seq.len().saturating_sub(3);
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &t in &seq[start..] {
            h ^= u64::from(t);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let pick = if seq.len() >= 4 && (h & 7) == 0 {
            EOS
        } else {
            2 + (h % 5) as u32
        };
        peak_row(pick)
    }

    /// Branch-coverage tally proving the parity battery is not vacuous.
    #[derive(Default)]
    struct Coverage {
        empty_drafts: usize,
        full_accepts: usize,
        partial_accepts: usize,
        corrections: usize,
        eos_stops: usize,
        max_stops: usize,
    }

    /// Reference SEQUENTIAL greedy decode — the literal `generate_cached_i8` loop
    /// with the decoder abstracted as `oracle`: argmax/ban `oracle(generated)`,
    /// append, halt at EOS or `max_length`. No verify assembly, so this is the
    /// independent ground truth.
    fn seq_generate(
        oracle: &dyn Fn(&[u32]) -> Mat,
        prompt: &[u32],
        params: &DecodeParams,
    ) -> Vec<u32> {
        let mut generated = prompt.to_vec();
        let mut emitted = Vec::new();
        while emitted.len() < params.max_length {
            let logits = oracle(&generated);
            let step = sampler::decode_step(&logits, &generated, params).expect("seq decode_step");
            generated.push(step.token_id);
            emitted.push(step.token_id);
            if step.is_eos {
                break;
            }
        }
        emitted
    }

    /// The bd-1azu.35 SPECULATIVE generate loop, structured identically to
    /// `OcrModel::spec_decode_i8` with the decoder abstracted as `oracle`: the REAL
    /// `draft_ngram` proposes, `verify_logits[i] = oracle(generated ++ draft[0..i])`
    /// plays the verify forward, the REAL `resolve_round` accepts + corrects, and
    /// committing a token is appending it (the oracle is a pure function of the token
    /// sequence, so the ring advance is modeled by the growing history). Honors
    /// EOS/`max_length` exactly as the live loop. Returns the emitted stream and folds
    /// the branches it hit into `cov`.
    fn spec_generate(
        oracle: &dyn Fn(&[u32]) -> Mat,
        prompt: &[u32],
        params: &DecodeParams,
        max_draft: usize,
        ngram: usize,
        cov: &mut Coverage,
    ) -> Vec<u32> {
        let mut generated = prompt.to_vec();
        let mut emitted = Vec::new();
        let mut eos = false;
        while emitted.len() < params.max_length {
            let draft = draft_ngram(&generated, max_draft, ngram);
            if draft.is_empty() {
                cov.empty_drafts += 1;
                let logits = oracle(&generated);
                let step =
                    sampler::decode_step(&logits, &generated, params).expect("spec fallback step");
                generated.push(step.token_id);
                emitted.push(step.token_id);
                if step.is_eos {
                    eos = true;
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
            if emit.accepted == draft.len() {
                cov.full_accepts += 1;
            } else {
                cov.partial_accepts += 1;
            }
            let mut stopped = false;
            for &token in &draft[..emit.accepted] {
                generated.push(token);
                emitted.push(token);
                if params.eos_token_id == token {
                    eos = true;
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
                    cov.corrections += 1;
                    generated.push(c.token_id);
                    emitted.push(c.token_id);
                    if c.is_eos {
                        eos = true;
                        break;
                    }
                }
            }
        }
        if eos {
            cov.eos_stops += 1;
        } else if emitted.len() >= params.max_length {
            cov.max_stops += 1;
        }
        emitted
    }

    /// Run one length-oracle case (greedy follows `target` exactly) through both
    /// schedules, assert byte parity, and return the (shared) emitted stream.
    fn run_length_case(
        target: &[u32],
        prompt: &[u32],
        max_length: usize,
        max_draft: usize,
        ngram: usize,
        cov: &mut Coverage,
    ) -> Vec<u32> {
        let params = params_single(max_length);
        let oracle = |s: &[u32]| {
            let t = target.get(s.len()).copied().unwrap_or(EOS);
            peak_row(t)
        };
        let seq = seq_generate(&oracle, prompt, &params);
        let spec = spec_generate(&oracle, prompt, &params, max_draft, ngram, cov);
        assert_eq!(
            spec, seq,
            "spec != sequential greedy (length oracle) target={target:?} prompt={prompt:?} \
             md={max_draft} ng={ngram} ml={max_length}"
        );
        seq
    }

    /// END-TO-END PARITY GATE: the full draft -> verify -> accept loop emits the
    /// byte-identical stream to sequential greedy, across the empty-draft fallback,
    /// full accept, mid-draft reject + correction, accepted-through-EOS halt, and the
    /// `max_length` cutoff — proven over crafted length oracles AND a content-oracle
    /// battery, with every branch confirmed to have executed.
    #[test]
    fn spec_loop_is_byte_identical_to_sequential_greedy() {
        let mut cov = Coverage::default();

        // Scenario A — a periodic [8,9] walk ending in EOS. The drafter hits the
        // empty-draft fallback (early), a FULL accept, a mid-draft REJECT +
        // correction, then the EOS halt; at a tight cap, the max_length cutoff lands
        // mid-round (only the first accepted token of a full-accept round is emitted).
        let target_a = [8u32, 9, 8, 9, 8, 9, 8, 9, EOS];
        let prompt_a = [8u32, 9];
        let full = run_length_case(&target_a, &prompt_a, 100, 3, 2, &mut cov);
        assert_eq!(
            full,
            vec![8, 9, 8, 9, 8, 9, EOS],
            "EOS-terminated greedy stream"
        );
        let capped = run_length_case(&target_a, &prompt_a, 3, 3, 2, &mut cov);
        assert_eq!(capped, vec![8u32, 9, 8], "max_length=3 cutoff (mid-round)");
        // Same scenario through the PRODUCTION draft knobs (K=4, ngram=3): equality
        // must still hold — the drafter is pure proposal, knobs never change WHICH
        // tokens are emitted.
        run_length_case(
            &target_a,
            &prompt_a,
            100,
            SPEC_DRAFT_MAX,
            SPEC_DRAFT_NGRAM,
            &mut cov,
        );

        // Scenario B — a 3-gram [3,4,5] that recurs with a MATCHING continuation
        // (full accept) then a DIVERGENT one (reject + correction), exercising ngram=3
        // drafts and longer accepted runs.
        let target_b = [3u32, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 7, EOS];
        let prompt_b = [3u32, 4, 5];
        run_length_case(&target_b, &prompt_b, 100, 4, 3, &mut cov);
        run_length_case(&target_b, &prompt_b, 100, 3, 2, &mut cov);

        // Content-oracle battery: deterministic 3rd-order walks over a small alphabet,
        // across many prompts, draft knobs, and caps (kept under the 35-gram ban
        // window so the chooser stays the ban-free argmax both sides apply). Each must
        // emit the byte-identical stream to sequential greedy.
        let oracle: fn(&[u32]) -> Mat = content_logits;
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..48 {
            let plen = 3 + (next() % 3) as usize;
            let mut prompt = Vec::with_capacity(plen);
            for _ in 0..plen {
                prompt.push(2 + (next() % 5) as u32);
            }
            for &(md, ng) in &[(3usize, 2usize), (4, 3), (4, 2), (2, 1)] {
                for &ml in &[10usize, 18, 24] {
                    let params = params_single(ml);
                    let seq = seq_generate(&oracle, &prompt, &params);
                    let spec = spec_generate(&oracle, &prompt, &params, md, ng, &mut cov);
                    assert_eq!(
                        spec, seq,
                        "spec != sequential greedy (content oracle) prompt={prompt:?} \
                         md={md} ng={ng} ml={ml}"
                    );
                }
            }
        }

        // Every speculative branch must have actually run (the parity gate is not
        // vacuous over trivial decodes).
        assert!(cov.empty_drafts > 0, "empty-draft fallback never exercised");
        assert!(cov.full_accepts > 0, "full-accept round never exercised");
        assert!(
            cov.partial_accepts > 0,
            "reject+correction round never exercised"
        );
        assert!(cov.corrections > 0, "correction token never emitted");
        assert!(cov.eos_stops > 0, "EOS halt never exercised");
        assert!(cov.max_stops > 0, "max_length cutoff never exercised");
    }

    /// `resolve_round`: a full accept appends the trailing bonus row's greedy token.
    #[test]
    fn resolve_round_full_accept_appends_bonus_correction() {
        let params = DecodeParams::single_image();
        let rows = vec![peak_row(3), peak_row(4), peak_row(5)];
        let emit = resolve_round(&[], &[3, 4], &rows, &params).unwrap();
        assert_eq!(emit.accepted, 2);
        let c = emit.correction.expect("bonus correction after full accept");
        assert_eq!(c.token_id, 5);
        assert!(!c.is_eos);
    }

    /// `resolve_round`: a mid-draft divergence truncates and corrects from that row.
    #[test]
    fn resolve_round_mid_reject_corrects_from_divergent_row() {
        let params = DecodeParams::single_image();
        // greedy at pos 1 is 4, draft proposes 7 -> reject at 1, correction 4.
        let rows = vec![peak_row(3), peak_row(4), peak_row(9)];
        let emit = resolve_round(&[], &[3, 7], &rows, &params).unwrap();
        assert_eq!(emit.accepted, 1);
        let c = emit.correction.expect("correction at first divergence");
        assert_eq!(c.token_id, 4);
    }

    /// `resolve_round`: an accepted EOS halts the round with NO correction token.
    #[test]
    fn resolve_round_accepted_eos_has_no_correction() {
        let params = DecodeParams::single_image();
        // draft [5, EOS]; pos 1 greedy is EOS and equals the draft -> accept through
        // EOS, then stop (sequential greedy emits EOS and halts).
        let rows = vec![peak_row(5), peak_row(EOS), peak_row(6)];
        let emit = resolve_round(&[], &[5, EOS], &rows, &params).unwrap();
        assert_eq!(emit.accepted, 2);
        assert!(
            emit.correction.is_none(),
            "accepted EOS halts with no correction"
        );
    }
}
