//! bd-1azu.36 (LINEAR half) — speculative-decode single-stream GATE hardening:
//! the dispatch-guard surface that keeps `FOCR_SPEC_DECODE` from engaging
//! anywhere outside its proven regime.
//!
//! TREE-verify clauses (per-node logits parity vs the sequential root-to-node
//! path, longest-path accept on branching drafts, the `FOCR_SPEC_TREE_W=1`
//! collapse) are PARKED behind bd-1azu.34 (tree verification) and intentionally
//! absent from this file.
//!
//! The bead's LINEAR gate splits three ways (visibility forces the split —
//! `native_engine::spec` is `pub(crate)`, and an integration test sees only the
//! public API, exactly like `tests/spec_verify_forward_parity.rs`, which rebuilds
//! the verify forward from PUBLIC kernels rather than touching `spec::`):
//!
//! 1. **Fault injection** (drafter is UNTRUSTED input: garbage / oversized /
//!    forged-EOS / empty drafts never change the accepted stream) needs the
//!    `pub(crate)` seams `spec::{accept_longest, resolve_round}`, so it lives
//!    in-crate: `src/native_engine/sampler.rs::spec_gate_fault_injection`
//!    (`spec.rs` itself is owner-frozen this wave; the sampler is the chooser
//!    the verifier reuses, the nearest editable seam).
//! 2. **Dispatch-guard non-engagement** (this file): `FOCR_SPEC_DECODE` may only
//!    arm the spec loop when the decode params are EXACTLY the frozen
//!    single-image ban (`no_repeat_ngram_size == 35`, `ngram_window == 128`) the
//!    verifier's chooser hardwires. The guard's params half is the public
//!    [`DecodeParams::matches_frozen_spec_ban`]; the tests below pin the frozen
//!    constants and prove every override shape (`--no-repeat-ngram 20`,
//!    `--ngram-window 1024`, multi-image, disabled blocker) fails it. NOTE: the
//!    live dispatch in `OcrModel::generate_cached_i8` still spells the same two
//!    comparisons inline (`src/native_engine/mod.rs` is owner-frozen this wave);
//!    a one-hunk adoption diff replacing the inline expression with this
//!    predicate was returned to the hot-cluster owner so guard and gate cannot
//!    drift apart.
//! 3. **Model-gated ON-vs-OFF byte identity** is `scripts/spec_gate_e2e.sh`:
//!    `FOCR_SPEC_DECODE` is a presence kill-switch read ONCE into a process-wide
//!    `OnceLock`, and edition-2024 `set_var` is `unsafe` (this crate denies
//!    unsafe), so an in-process env flip is impossible — the script drives TWO
//!    `focr` processes (env removed vs `=1`) over real pages and sha256-compares
//!    stdout, the same two-process A/B discipline as the `FOCR_BATCH_VISION`
//!    on/off parity runs. Default-OFF ("unset never engages") is likewise a
//!    process-level property and is exercised there, not here.

use franken_ocr::native_engine::sampler::{
    DEFAULT_NO_REPEAT_NGRAM_SIZE, DecodeParams, NGRAM_WINDOW_MULTI, NGRAM_WINDOW_SINGLE,
};

/// The frozen single-image ban is literally 35-gram over a 128-token window.
/// `spec::greedy_from_row` (the verifier's chooser) and the `FOCR_SPEC_DECODE`
/// dispatch guard both hardwire THESE values; if either constant moves, the
/// spec-decode lossless proof (bd-1azu.32/.35/.36) must be re-run before this
/// pin is updated — the failure is the tripwire, not an inconvenience.
#[test]
fn frozen_spec_ban_constants_are_pinned() {
    assert_eq!(
        DEFAULT_NO_REPEAT_NGRAM_SIZE, 35,
        "the verifier's chooser hardwires the 35-gram ban (README single/multi, OQ-18)"
    );
    assert_eq!(
        NGRAM_WINDOW_SINGLE, 128,
        "the verifier's chooser hardwires the 128-token single-image window"
    );
    assert_eq!(
        NGRAM_WINDOW_MULTI, 1024,
        "the multi-image window the spec gate must keep REJECTING (1024 != 128)"
    );
}

/// The default single-image params — the ONLY regime speculative decode is
/// proven lossless in — satisfy the guard's params half.
#[test]
fn single_image_params_match_the_frozen_spec_ban() {
    let p = DecodeParams::single_image();
    assert!(
        p.matches_frozen_spec_ban(),
        "single-image params must be exactly the frozen 35/128 spec ban"
    );
    assert_eq!(p.no_repeat_ngram_size, DEFAULT_NO_REPEAT_NGRAM_SIZE);
    assert_eq!(p.ngram_window, NGRAM_WINDOW_SINGLE);
}

/// NON-REVIVAL GATE: any n-gram override shape MUST fail the guard's params
/// half, so `FOCR_SPEC_DECODE` cannot engage the spec loop when the CLI/env
/// re-shapes the ban (`--no-repeat-ngram` / `--ngram-window` /
/// `FOCR_NO_REPEAT_NGRAM` → `DecodeOverrides` → the model's `DecodeParams`).
/// The chooser inside `spec::accept_longest` hardwires 35/128 — with any other
/// ban it would verify against the WRONG greedy rule, so "disengage" is the only
/// lossless answer, and the sequential greedy loop must run untouched.
#[test]
fn non_frozen_ngram_params_must_not_engage_spec() {
    // (no_repeat_ngram_size, ngram_window, why it exists)
    let cases: &[(usize, usize, &str)] = &[
        (
            20,
            128,
            "the bead's example: `--no-repeat-ngram 20` on the single window",
        ),
        (34, 128, "one below the frozen ban size"),
        (36, 128, "one above the frozen ban size"),
        (0, 128, "blocker disabled by size"),
        (
            35,
            1024,
            "`--ngram-window 1024` (the multi-image window) on ban 35",
        ),
        (35, 127, "one below the frozen window"),
        (35, 129, "one above the frozen window"),
        (
            35,
            0,
            "window 0 = the HF-builtin whole-history fallback, NOT 128",
        ),
        (0, 0, "blocker fully off"),
        (20, 1024, "both knobs overridden"),
    ];
    for &(n, w, why) in cases {
        let mut p = DecodeParams::single_image();
        p.no_repeat_ngram_size = n;
        p.ngram_window = w;
        assert!(
            !p.matches_frozen_spec_ban(),
            "params ({n}/{w}) must NOT engage spec decode — {why}"
        );
    }
    // The multi-image constructor is a non-frozen shape by construction.
    assert!(
        !DecodeParams::multi_image().matches_frozen_spec_ban(),
        "multi-image params (window 1024) must NOT engage spec decode"
    );
}

/// The guard is EXACTLY the two n-gram knobs: `max_length` / `temperature` /
/// `eos_token_id` overrides do not disarm speculation (they parameterize both
/// the sequential and the spec loop identically — `resolve_round` takes them
/// from the same `DecodeParams`), matching the inline dispatch expression in
/// `OcrModel::generate_cached_i8`.
#[test]
fn frozen_ban_predicate_ignores_non_ngram_fields() {
    let mut p = DecodeParams::single_image();
    p.max_length = 7;
    p.temperature = 0.0;
    p.eos_token_id = 2;
    assert!(
        p.matches_frozen_spec_ban(),
        "non-ngram decode fields must not affect the spec dispatch guard"
    );
}
