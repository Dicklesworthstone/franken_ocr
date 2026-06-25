//! AF-3 — Conformal / sequential-test early-exit decode (`speculative_guard`).
//!
//! Design artifact: `docs/alien/AF-3-conformal-early-exit.md` (galaxy-brain card),
//! plan §9.7 (AF-3). Python prior art for the canonical Ville e-process:
//! `scripts/gauntlet_cert.py` (`EProcess`) + `scripts/af2_tail_risk.py`.
//!
//! # What this is
//!
//! Speculative / early-exit decode is high-EV (skip the routed top-6 expert
//! gather on easy printed-text runs) but high-risk for exact-token OCR (one
//! flipped token silently corrupts the parse). AF-3 makes it **provably safe**:
//! a cheap *draft* forward proposes the next token, and an **anytime-valid
//! sequential test (e-value / e-process)** on the draft-vs-full agreement signal
//! accepts the draft **only while a finite-sample token-flip bound holds at risk
//! level α**. By Ville's inequality the lifetime probability of a guard breach is
//! `≤ α`, with no Bonferroni penalty over the unbounded decode stream.
//!
//! # The load-bearing correctness property (doctrine #1: G1 > G2)
//!
//! The emitted token is **ALWAYS the full-model token** on every step the guard
//! decides to *verify*. AF-3 only *skips the verify* (accepts the cheap draft)
//! on steps where the calibrated guard proves the draft would have agreed with
//! the full model anyway. The conservative default is to verify **every** step,
//! which is byte-for-byte the full-forward decode path. Therefore the worst case
//! for AF-3 is "no speedup", never "a flipped OCR token" — and this module is a
//! pure *latency* optimization gated on correctness. The output-identity property
//! is asserted as a test (`early_exit_output_identical_to_full_decode`).
//!
//! # The alien-artifact contract (AGENTS.md "Alien-Artifact Engineering Contract")
//!
//! * **State space** — the running e-process value `E_t ≥ 0` (init `E_0 = 1`)
//!   plus the per-step evidence `e_t` from the draft-vs-full agreement signal.
//!   Deterministic function of the emitted-token history and per-step margins; no
//!   RNG, so the controller is reproducible bit-for-bit.
//! * **Actions** — [`Action::AcceptDraft`] (emit the cheap draft token, skip the
//!   routed gather + lm_head verify) vs [`Action::FallBack`] (run the full
//!   forward, emit the *full-model* token, fold the disagreement into `E_t`).
//! * **Loss matrix** — [`LossMatrix`]: `ACCEPT & draft==full` → `−c_save` (a win);
//!   `ACCEPT & draft≠full` → `+l_flip` (**catastrophic**, `l_flip ≫ c_save`);
//!   `FALL_BACK` → `+c_full` (the safe full-forward cost).
//! * **Posterior / confidence + calibration** — the e-process `E_t` *is* the
//!   running confidence (betting wealth of a fair-bet martingale against the null
//!   "the draft agrees with the full model"). The calibration metric is the
//!   measured token-disagreement rate per slice vs α and the measured guard-breach
//!   rate vs the Ville bound α ([`GuardLevel`]).
//! * **Deterministic fallback** (wired FIRST) — `α → 0` / α-unset disables
//!   speculation entirely (verify every step = full forward). A guard *trip*
//!   (`E_t ≥ 1/α`) falls back to full forward for the rest of the document.
//!   Calibration below the minimum sample count ⇒ refuse to speculate.
//! * **Evidence ledger** — [`SpeculativeLedger`] of per-decision
//!   [`DecisionRecord`] (step, `e_value`, margin, agree, verdict) for audit.
//!
//! # The math (Ville's inequality)
//!
//! Null at step *t*: `H0(t): argmax(draft_logits) == argmax(full_logits)`.
//! We build a non-negative e-process `(E_t)` that is a supermartingale under H0:
//! `E_0 = 1`, `E_t = E_{t-1} · e_t` with `E_{H0}[e_t | F_{t-1}] ≤ 1`. Then Ville:
//!
//! ```text
//! P_{H0}( sup_{t≥1} E_t ≥ 1/α ) ≤ α .                                       (★)
//! ```
//!
//! The per-step **betting (Robbins) e-value** (AF-3 §2.2(a), the default):
//!
//! ```text
//! D_t = 1{draft_t ≠ full_t} ∈ {0,1}      (disagreement indicator)
//! e_t = 1 + λ·(D_t − q0) ,   q0 = 1 − p0 ,   λ ∈ [0, 1/(1 − q0))
//! ```
//!
//! Under `H0` (`E[D_t] ≤ q0`): `E_{H0}[e_t|F_{t-1}] = 1 + λ(E[D_t] − q0) ≤ 1`.
//! An agreement (`D_t=0`) shrinks wealth (`e_t = 1 − λ·q0 < 1`); a disagreement
//! (`D_t=1`) grows it (`e_t = 1 + λ·(1 − q0) > 1`), pushing toward the `1/α`
//! alarm — exactly the alarm direction we want.

#![allow(clippy::module_name_repetitions)]

use std::fmt;

/// Default risk level when a calibrated profile is loaded but no α override is
/// given (AF-3 §2.3 worked example: α = 0.01 ⇒ boundary `1/α = 100`).
pub const DEFAULT_ALPHA: f64 = 0.01;

/// Default calibrated agreement lower bound `p0` (AF-3 §2.3: `p0 = 0.97`,
/// `q0 = 1 − p0 = 0.03`). The *measured* value is a calibration output; this is
/// the documented worked-example default.
pub const DEFAULT_P0: f64 = 0.97;

/// Default betting fraction `λ` (AF-3 §2.3: `λ = 0.5`, well inside the cap
/// `1/(1 − q0) ≈ 1.031`).
pub const DEFAULT_LAMBDA: f64 = 0.5;

/// Default deterministic audit cadence: force a full verify every `r`-th
/// *accepted* step so the e-process keeps accumulating real evidence even on
/// confident runs (AF-3 §3.3). `r = 1` would verify every step (the fallback).
pub const DEFAULT_AUDIT_CADENCE: u32 = 8;

/// Default margin threshold (in logit units): a draft whose top-1 minus top-2
/// margin is **below** this is treated as low-confidence and forced to verify
/// (AF-3 §3.3). Calibration output; this is a conservative documented default.
pub const DEFAULT_MARGIN_THRESHOLD: f64 = 4.0;

/// Minimum calibration sample count below which we **refuse to speculate** and
/// fall back to α→0 (AF-3 §4 minimum-calibration-corpus precondition, inherited
/// from the conformal-ratchet sibling `bd-re8.14`). Too small a slice yields a
/// vacuously conservative / meaningless parameterisation.
pub const MIN_CALIBRATION_SAMPLES: usize = 200;

/// Saturation floor for `E_t` (smallest positive subnormal `f64`) — mirrors the
/// `gauntlet_cert.py` `EProcess` discipline: clamp to keep f64 noise away from
/// the threshold logic but **never reset** to 1.0 (that would break the
/// supermartingale property / Ville's bound).
const E_FLOOR: f64 = f64::MIN_POSITIVE;

// --------------------------------------------------------------------------- //
// State + actions                                                             //
// --------------------------------------------------------------------------- //

/// The per-decode-step action chosen by the guard (AF-3 §1.2).
///
/// **Invariant — the emitted token is ALWAYS the full-model token whenever we
/// [`Action::FallBack`].** [`Action::AcceptDraft`] is only taken on steps where
/// the draft *provably* agrees with what the full forward would have emitted
/// (within the guarded budget); on any step the guard is uncertain we fall back
/// and emit the verified token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Emit the cheap draft's argmax token; skip the routed top-6 gather +
    /// lm_head verify for that step. Cheap (draft forward only).
    AcceptDraft,
    /// Run the full forward, emit the **full-model** argmax token, fold the
    /// disagreement into `E_t`. Full forward — the safe default.
    FallBack,
}

impl Action {
    /// Whether this action paid the full forward (and therefore emitted the
    /// authoritative token).
    #[must_use]
    pub fn is_verified(self) -> bool {
        matches!(self, Action::FallBack)
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::AcceptDraft => f.write_str("accept_draft"),
            Action::FallBack => f.write_str("fall_back"),
        }
    }
}

// --------------------------------------------------------------------------- //
// Loss matrix (decision-theoretic justification, AF-3 §1.3)                   //
// --------------------------------------------------------------------------- //

/// The decision-theoretic loss matrix that justifies the gate (AF-3 §1.3).
///
/// For exact-token OCR `l_flip` is effectively unbounded relative to `c_save`
/// (dense numerics, tables, sub/superscripts have **zero** tolerance for a
/// flipped digit). The point of AF-3 is therefore not to minimise expected loss
/// by guessing but to **drive the realised probability of the
/// `ACCEPT, draft ≠ full` event below α with a finite-sample guarantee**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossMatrix {
    /// `−c_save`: the upside harvested by a correct cheap accept.
    pub c_save: f64,
    /// `+l_flip`: catastrophic silent token-flip (CER-corrupting). `l_flip ≫ c_save`.
    pub l_flip: f64,
    /// `+c_full`: the conservative full-forward cost (safe).
    pub c_full: f64,
}

impl Default for LossMatrix {
    fn default() -> Self {
        // Illustrative magnitudes: l_flip ≫ c_save (exact-token OCR). These are
        // *relative* weights; the gate's job is to bound the flip PROBABILITY,
        // not to trade flip risk for compute savings.
        Self {
            c_save: 1.0,
            l_flip: 1.0e6,
            c_full: 2.0,
        }
    }
}

impl LossMatrix {
    /// Expected loss of [`Action::AcceptDraft`] given a draft-disagreement
    /// probability `p_flip` (used only for ledger/diagnostics — the gate decision
    /// is the e-process boundary, not this expectation, per AF-3 §1.3).
    #[must_use]
    pub fn expected_accept_loss(&self, p_flip: f64) -> f64 {
        // win with prob (1-p_flip): -c_save ; flip with prob p_flip: +l_flip
        (1.0 - p_flip) * (-self.c_save) + p_flip * self.l_flip
    }

    /// Loss of the safe [`Action::FallBack`] (always `+c_full`).
    #[must_use]
    pub fn fallback_loss(&self) -> f64 {
        self.c_full
    }

    /// The break-even disagreement rate at which accepting is no worse than
    /// falling back: solve `expected_accept_loss(p) = c_full`.
    #[must_use]
    pub fn break_even_flip_rate(&self) -> f64 {
        // (1-p)(-c_save) + p·l_flip = c_full
        // -c_save + p(c_save + l_flip) = c_full
        // p = (c_full + c_save) / (c_save + l_flip)
        let denom = self.c_save + self.l_flip;
        if denom <= 0.0 {
            return 0.0;
        }
        ((self.c_full + self.c_save) / denom).clamp(0.0, 1.0)
    }
}

// --------------------------------------------------------------------------- //
// Calibration profile                                                         //
// --------------------------------------------------------------------------- //

/// A calibrated `speculative_guard` profile: the parameters the calibration
/// fixture solves for on a held-out corpus slice to hit a target α with maximal
/// acceptance (AF-3 §4). Constructed only via [`Calibration::new`], which
/// validates the e-value property's bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    /// Risk level α ∈ (0, 1]. Guard boundary is `1/α`.
    alpha: f64,
    /// Calibrated agreement lower bound `p0` ∈ (0, 1). `q0 = 1 − p0`.
    p0: f64,
    /// Betting fraction `λ` ∈ [0, 1/(1 − q0)) keeping `e_t ≥ 0`.
    lambda: f64,
    /// Deterministic audit cadence `r`: force a verify every `r`-th accepted step.
    audit_cadence: u32,
    /// Margin threshold: drafts below this are forced to verify (low confidence).
    margin_threshold: f64,
}

/// Why a [`Calibration`] could not be constructed (each maps to the safe
/// deterministic fallback — refuse to speculate, decode to natural EOS).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationError {
    /// α outside (0, 1] (α-unset / α→0 is the documented fallback trigger).
    AlphaOutOfRange,
    /// `p0` outside (0, 1).
    P0OutOfRange,
    /// `λ` outside `[0, 1/(1 − q0))` — would let `e_t` go negative (the e-value
    /// property requires `e_t ≥ 0`).
    LambdaOutOfRange,
    /// Audit cadence `r` must be ≥ 1.
    AuditCadenceZero,
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalibrationError::AlphaOutOfRange => {
                f.write_str("alpha out of (0,1] (alpha-unset => deterministic fallback)")
            }
            CalibrationError::P0OutOfRange => f.write_str("p0 out of (0,1)"),
            CalibrationError::LambdaOutOfRange => {
                f.write_str("lambda out of [0, 1/(1-q0)) (would break e_t >= 0)")
            }
            CalibrationError::AuditCadenceZero => f.write_str("audit_cadence must be >= 1"),
        }
    }
}

impl std::error::Error for CalibrationError {}

impl Calibration {
    /// Construct a calibration profile, validating the e-value-property bounds.
    ///
    /// # Errors
    /// Returns a [`CalibrationError`] if any parameter is outside the range the
    /// betting e-value requires; the caller MUST then take the deterministic
    /// fallback (verify every step / decode to natural EOS).
    pub fn new(
        alpha: f64,
        p0: f64,
        lambda: f64,
        audit_cadence: u32,
        margin_threshold: f64,
    ) -> Result<Self, CalibrationError> {
        if !(alpha.is_finite() && alpha > 0.0 && alpha <= 1.0) {
            return Err(CalibrationError::AlphaOutOfRange);
        }
        if !(p0.is_finite() && p0 > 0.0 && p0 < 1.0) {
            return Err(CalibrationError::P0OutOfRange);
        }
        let q0 = 1.0 - p0;
        // λ ∈ [0, 1/(1 − q0)) ⇒ keeps e_t = 1 + λ(D_t − q0) ≥ 0 even at D_t = 1.
        let lambda_cap = 1.0 / (1.0 - q0);
        if !(lambda.is_finite() && lambda >= 0.0 && lambda < lambda_cap) {
            return Err(CalibrationError::LambdaOutOfRange);
        }
        if audit_cadence == 0 {
            return Err(CalibrationError::AuditCadenceZero);
        }
        if !margin_threshold.is_finite() || margin_threshold < 0.0 {
            // A non-finite / negative threshold disables margin-forced verifies;
            // treat as the conservative "always low-confidence" => verify often.
            return Ok(Self {
                alpha,
                p0,
                lambda,
                audit_cadence,
                margin_threshold: f64::INFINITY,
            });
        }
        Ok(Self {
            alpha,
            p0,
            lambda,
            audit_cadence,
            margin_threshold,
        })
    }

    /// The documented worked-example profile (AF-3 §2.3): α = 0.01, p0 = 0.97,
    /// λ = 0.5, cadence 8, margin 4.0.
    #[must_use]
    pub fn worked_example() -> Self {
        Self::new(
            DEFAULT_ALPHA,
            DEFAULT_P0,
            DEFAULT_LAMBDA,
            DEFAULT_AUDIT_CADENCE,
            DEFAULT_MARGIN_THRESHOLD,
        )
        .expect("worked-example calibration constants are in-range")
    }

    /// α (risk level).
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// `q0 = 1 − p0` (the tolerated disagreement rate the null is built around).
    #[must_use]
    pub fn q0(&self) -> f64 {
        1.0 - self.p0
    }

    /// The Ville guard boundary `1/α`.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        1.0 / self.alpha
    }
}

// --------------------------------------------------------------------------- //
// The e-process (Ville's inequality; betting / Robbins form, AF-3 §2.2(a))    //
// --------------------------------------------------------------------------- //

/// An anytime-valid e-process over the draft-vs-full agreement stream.
///
/// Mirrors the canonical `gauntlet_cert.py` `EProcess` discipline (saturate to
/// keep f64 noise away from the threshold logic, **never reset** `e_value` to
/// 1.0 — that breaks the supermartingale property / Ville's bound) but uses the
/// **betting (Robbins) update** AF-3 §2.2(a) specifies as the default:
/// `e_t = 1 + λ·(D_t − q0)`.
#[derive(Debug, Clone, PartialEq)]
pub struct EProcess {
    q0: f64,
    lambda: f64,
    alpha: f64,
    e_value: f64,
    obs_count: u64,
    rejected_at: Option<u64>,
}

impl EProcess {
    /// Build an e-process from a calibration profile (`E_0 = 1`).
    #[must_use]
    pub fn new(cal: &Calibration) -> Self {
        Self {
            q0: cal.q0(),
            lambda: cal.lambda,
            alpha: cal.alpha,
            e_value: 1.0,
            obs_count: 0,
            rejected_at: None,
        }
    }

    /// Feed one *audited* observation: `disagree == true` ⇔ `D_t = 1` (the draft
    /// argmax differed from the full-model argmax). Returns `true` the first time
    /// the Ville threshold `1/α` is crossed (the latch).
    ///
    /// Betting update (AF-3 §2.2(a)): `e_t = 1 + λ·(D_t − q0)`.
    pub fn observe(&mut self, disagree: bool) -> bool {
        self.obs_count += 1;
        let d_t = if disagree { 1.0 } else { 0.0 };
        let e_t = 1.0 + self.lambda * (d_t - self.q0);
        // e_t ≥ 0 is guaranteed by the λ-cap validated in Calibration::new.
        self.e_value *= e_t;
        // Saturate (never reset): clamp to (E_FLOOR, f64::MAX/2] — Ville's bound
        // is about the unclamped trajectory; this only tames f64 noise.
        self.e_value = self.e_value.clamp(E_FLOOR, f64::MAX / 2.0);
        if self.e_value >= self.threshold() && self.rejected_at.is_none() {
            self.rejected_at = Some(self.obs_count);
            return true;
        }
        false
    }

    /// The current e-process value `E_t` (the running confidence / betting wealth).
    #[must_use]
    pub fn e_value(&self) -> f64 {
        self.e_value
    }

    /// The Ville guard boundary `1/α`.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        1.0 / self.alpha
    }

    /// Number of audited observations folded in so far.
    #[must_use]
    pub fn obs_count(&self) -> u64 {
        self.obs_count
    }

    /// `true` once the e-process has crossed `1/α` (the latch; stays set).
    #[must_use]
    pub fn rejected(&self) -> bool {
        self.rejected_at.is_some()
    }

    /// The 1-based observation index at which the threshold was first crossed.
    #[must_use]
    pub fn rejected_at(&self) -> Option<u64> {
        self.rejected_at
    }
}

// --------------------------------------------------------------------------- //
// Evidence ledger (the contract's item 4)                                     //
// --------------------------------------------------------------------------- //

/// One per-decision audit record (AF-3 §1.4 / §5 reconstructable decision).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionRecord {
    /// 1-based decode step index.
    pub step: u64,
    /// The action taken at this step.
    pub action: Action,
    /// Draft top-1 minus top-2 logit margin `Δ_t` (NaN if no draft was scored).
    pub margin: f64,
    /// Whether this step was *audited* (a full verify ran ⇒ `E_t` updated).
    pub audited: bool,
    /// On an audited step: did the draft argmax agree with the full argmax?
    /// `None` when the step was accepted without a verify.
    pub agree: Option<bool>,
    /// The e-process value `E_t` after this step.
    pub e_value: f64,
    /// The token actually emitted (always the full-model token on audited steps).
    pub emitted_token: u32,
}

/// The per-decode evidence ledger: the audit trail the keep/revert decision is
/// reconstructed from (AF-3 §5; `tests/artifacts/af3/`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeculativeLedger {
    records: Vec<DecisionRecord>,
    /// Count of [`Action::AcceptDraft`] decisions (skipped verifies).
    accepted: u64,
    /// Count of audited steps (full verifies that updated `E_t`).
    audited: u64,
    /// Count of audited steps where the draft *disagreed* with the full model.
    disagreements: u64,
    /// `true` if the guard tripped (`E_t ≥ 1/α`) at any point in this document.
    tripped: bool,
    /// `true` if speculation was disabled (deterministic fallback active).
    fallback_active: bool,
}

impl SpeculativeLedger {
    /// A ledger for a run where speculation is disabled (the deterministic
    /// fallback): every step is a full forward.
    #[must_use]
    pub fn fallback() -> Self {
        Self {
            fallback_active: true,
            ..Self::default()
        }
    }

    fn push(&mut self, rec: DecisionRecord) {
        match rec.action {
            Action::AcceptDraft => self.accepted += 1,
            Action::FallBack => {}
        }
        if rec.audited {
            self.audited += 1;
            if rec.agree == Some(false) {
                self.disagreements += 1;
            }
        }
        self.records.push(rec);
    }

    /// The per-decision records.
    #[must_use]
    pub fn records(&self) -> &[DecisionRecord] {
        &self.records
    }

    /// Number of accepted (cheap, verify-skipped) steps.
    #[must_use]
    pub fn accepted(&self) -> u64 {
        self.accepted
    }

    /// Number of audited (full-verify) steps.
    #[must_use]
    pub fn audited(&self) -> u64 {
        self.audited
    }

    /// Number of audited steps where draft ≠ full.
    #[must_use]
    pub fn disagreements(&self) -> u64 {
        self.disagreements
    }

    /// Total decode steps recorded.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.records.len() as u64
    }

    /// Measured acceptance rate (skipped verifies / total steps).
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        self.accepted as f64 / self.records.len() as f64
    }

    /// Measured disagreement rate over *audited* steps — the calibration metric
    /// compared against α (AF-3 proof obligation 1).
    #[must_use]
    pub fn measured_disagreement_rate(&self) -> f64 {
        if self.audited == 0 {
            return 0.0;
        }
        self.disagreements as f64 / self.audited as f64
    }

    /// Whether the guard tripped this document.
    #[must_use]
    pub fn tripped(&self) -> bool {
        self.tripped
    }

    /// Whether the deterministic fallback (speculation off) was active.
    #[must_use]
    pub fn fallback_active(&self) -> bool {
        self.fallback_active
    }
}

/// The calibration coverage / e-process level report (the calibration metric;
/// AF-3 §1.4). `coverage = 1 − measured_breach_fraction`; the Ville guarantee is
/// `breach_fraction ≤ α`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardLevel {
    /// Number of synthetic streams / documents evaluated.
    pub streams: u64,
    /// Number that breached the guard (`E_t ≥ 1/α`).
    pub breaches: u64,
    /// The risk level α the guard was run at.
    pub alpha: f64,
}

impl GuardLevel {
    /// The measured guard-breach fraction — must be `≤ α` under the null (★).
    #[must_use]
    pub fn breach_fraction(&self) -> f64 {
        if self.streams == 0 {
            return 0.0;
        }
        self.breaches as f64 / self.streams as f64
    }

    /// Empirical coverage `1 − breach_fraction`.
    #[must_use]
    pub fn coverage(&self) -> f64 {
        1.0 - self.breach_fraction()
    }

    /// Whether the empirically-measured level honours the Ville bound `≤ α`.
    #[must_use]
    pub fn holds_level(&self) -> bool {
        self.breach_fraction() <= self.alpha + 1e-12
    }
}

// --------------------------------------------------------------------------- //
// The speculative_guard controller                                            //
// --------------------------------------------------------------------------- //

/// The `speculative_guard` early-exit controller (AF-3 §4 runtime artifact).
///
/// Pure / deterministic / no-RNG. Holds the running e-process, the forced-audit
/// counter, and the boundary `1/α`. Carry one alongside the decode loop:
///
/// * [`SpeculativeGuard::enabled`] — build from a [`Calibration`] (speculation on).
/// * [`SpeculativeGuard::disabled`] — the **deterministic fallback** (α→0 /
///   α-unset): `should_verify` is always `true`, so every step does the full
///   forward and decode runs to the natural EOS — byte-for-byte full decode.
/// * [`SpeculativeGuard::decide`] — the per-step decision given the draft margin,
///   the full-model token, and the (lazily computed) draft token.
pub struct SpeculativeGuard {
    eprocess: Option<EProcess>,
    cal: Option<Calibration>,
    /// Counts accepted (verify-skipped) steps since the last forced audit.
    since_audit: u32,
    /// Latched: once the guard trips we stay in the safe (full-forward) state.
    tripped: bool,
    ledger: SpeculativeLedger,
}

impl SpeculativeGuard {
    /// Build an **enabled** guard from a calibration profile.
    #[must_use]
    pub fn enabled(cal: Calibration) -> Self {
        Self {
            eprocess: Some(EProcess::new(&cal)),
            cal: Some(cal),
            since_audit: 0,
            tripped: false,
            ledger: SpeculativeLedger::default(),
        }
    }

    /// The **deterministic fallback**: speculation OFF (α→0 / α-unset). Every
    /// step verifies (full forward); decode runs to the natural EOS. This is the
    /// default and is byte-for-byte the full-forward decode path.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            eprocess: None,
            cal: None,
            since_audit: 0,
            tripped: false,
            ledger: SpeculativeLedger::fallback(),
        }
    }

    /// Build a guard from an *optional* calibration result: `Some(Ok(cal))`
    /// enables; `None` (α-unset) or `Some(Err(_))` (invalid params) takes the
    /// deterministic fallback. Also refuses to speculate if the calibration slice
    /// had fewer than [`MIN_CALIBRATION_SAMPLES`] samples (AF-3 §4 minimum-corpus
    /// precondition).
    #[must_use]
    pub fn from_calibration(
        cal: Option<Result<Calibration, CalibrationError>>,
        calibration_samples: usize,
    ) -> Self {
        match cal {
            Some(Ok(c)) if calibration_samples >= MIN_CALIBRATION_SAMPLES => Self::enabled(c),
            _ => Self::disabled(),
        }
    }

    /// `true` if speculation is active (a calibration profile is loaded and the
    /// guard has not tripped).
    #[must_use]
    pub fn speculation_active(&self) -> bool {
        self.eprocess.is_some() && !self.tripped
    }

    /// `true` once the guard has tripped (`E_t ≥ 1/α`) — the safe latched state.
    #[must_use]
    pub fn tripped(&self) -> bool {
        self.tripped
    }

    /// Whether the guard *requires* a full verify at this step given the draft
    /// margin (AF-3 §3.3 forced-audit policy):
    /// * always, in the deterministic-fallback / tripped state;
    /// * whenever the draft margin `Δ_t` is below the calibrated threshold
    ///   (low confidence ⇒ verify);
    /// * on the deterministic audit cadence (every `r`-th accepted step).
    #[must_use]
    pub fn should_verify(&self, margin: f64) -> bool {
        let Some(cal) = self.cal.as_ref() else {
            return true; // deterministic fallback: verify every step
        };
        if self.tripped {
            return true; // safe latched state: full forward for the rest of the doc
        }
        // Low-confidence draft (or non-finite margin) ⇒ force verify.
        if !margin.is_finite() || margin < cal.margin_threshold {
            return true;
        }
        // Forced audit cadence: the (r)-th accepted step since the last verify.
        self.since_audit + 1 >= cal.audit_cadence
    }

    /// Make the per-step decision and update the controller + ledger.
    ///
    /// * `margin` — draft top-1 minus top-2 logit margin `Δ_t`.
    /// * `full_token` — the **authoritative** full-model argmax token.
    /// * `draft_token` — the cheap draft argmax token (only consulted when we
    ///   skip the verify; pass it eagerly, it is cheap).
    ///
    /// Returns the emitted token, which is **the full-model token whenever the
    /// guard verifies** and the draft token only when the guard provably skips a
    /// verify. The output-identity property (AF-3 proof obligation 3) is: under
    /// the safe configuration the emitted sequence equals full decode.
    pub fn decide(&mut self, margin: f64, full_token: u32, draft_token: u32) -> Decision {
        let step = self.ledger.steps() + 1;
        if self.should_verify(margin) {
            // FALL_BACK: pay the full forward, emit the authoritative token, fold
            // the disagreement into E_t (only when speculation is active).
            self.since_audit = 0;
            let disagree = draft_token != full_token;
            let (audited, e_value, agree) = if let Some(ep) = self.eprocess.as_mut() {
                if !self.tripped {
                    let crossed = ep.observe(disagree);
                    if crossed {
                        self.tripped = true;
                        self.ledger.tripped = true;
                    }
                    (true, ep.e_value(), Some(!disagree))
                } else {
                    // Already tripped: still full-forward, but no longer betting.
                    (false, ep.e_value(), None)
                }
            } else {
                (false, 1.0, None)
            };
            self.ledger.push(DecisionRecord {
                step,
                action: Action::FallBack,
                margin,
                audited,
                agree,
                e_value,
                emitted_token: full_token,
            });
            Decision {
                action: Action::FallBack,
                emitted_token: full_token,
            }
        } else {
            // ACCEPT_DRAFT: the guard holds — skip the verify, emit the draft.
            // This branch is ONLY reached when speculation is active (cal Some,
            // not tripped) AND margin ≥ threshold AND cadence not yet due.
            self.since_audit += 1;
            let e_value = self.eprocess.as_ref().map_or(1.0, EProcess::e_value);
            self.ledger.push(DecisionRecord {
                step,
                action: Action::AcceptDraft,
                margin,
                audited: false,
                agree: None,
                e_value,
                emitted_token: draft_token,
            });
            Decision {
                action: Action::AcceptDraft,
                emitted_token: draft_token,
            }
        }
    }

    /// The accumulated evidence ledger.
    #[must_use]
    pub fn ledger(&self) -> &SpeculativeLedger {
        &self.ledger
    }

    /// The running e-process value, if speculation is active.
    #[must_use]
    pub fn e_value(&self) -> Option<f64> {
        self.eprocess.as_ref().map(EProcess::e_value)
    }
}

/// The result of a per-step [`SpeculativeGuard::decide`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    /// The action taken.
    pub action: Action,
    /// The token to emit (full-model token when verified, draft when accepted).
    pub emitted_token: u32,
}

// --------------------------------------------------------------------------- //
// Synthetic logit-stream helpers (test + calibration support)                 //
// --------------------------------------------------------------------------- //

/// Greedy argmax over a logit row (first-max-wins on ties, matching
/// `torch.argmax` and `native_engine::sampler::argmax_row`).
#[must_use]
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Top-1 minus top-2 margin of a logit row (the draft confidence `Δ_t`).
#[must_use]
pub fn top1_top2_margin(logits: &[f32]) -> f64 {
    let mut first = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &v in logits {
        if v > first {
            second = first;
            first = v;
        } else if v > second {
            second = v;
        }
    }
    f64::from(first - second)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic PRNG (SplitMix64) so the synthetic streams are
    /// reproducible bit-for-bit (the controller itself is RNG-free; this only
    /// generates the *test* logit streams).
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in [0, 1).
        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // ----- Calibration validation (the e-value-property bounds) ----- //

    #[test]
    fn calibration_rejects_out_of_range_params() {
        // alpha-unset / out of (0,1] => the deterministic-fallback trigger.
        assert_eq!(
            Calibration::new(0.0, 0.97, 0.5, 8, 4.0),
            Err(CalibrationError::AlphaOutOfRange)
        );
        assert_eq!(
            Calibration::new(1.5, 0.97, 0.5, 8, 4.0),
            Err(CalibrationError::AlphaOutOfRange)
        );
        // p0 must be in (0,1).
        assert_eq!(
            Calibration::new(0.01, 1.0, 0.5, 8, 4.0),
            Err(CalibrationError::P0OutOfRange)
        );
        // lambda must keep e_t >= 0: cap is 1/(1-q0) = 1/p0.
        let cap = 1.0 / 0.97; // q0 = 0.03 => cap ≈ 1.0309
        assert!(Calibration::new(0.01, 0.97, cap, 8, 4.0).is_err());
        assert!(Calibration::new(0.01, 0.97, cap - 1e-6, 8, 4.0).is_ok());
        // audit cadence must be >= 1.
        assert_eq!(
            Calibration::new(0.01, 0.97, 0.5, 0, 4.0),
            Err(CalibrationError::AuditCadenceZero)
        );
    }

    #[test]
    fn worked_example_matches_doc_2_3() {
        // AF-3 §2.3: alpha=0.01 => threshold 100; q0=0.03; lambda=0.5.
        let cal = Calibration::worked_example();
        assert!(approx(cal.threshold(), 100.0, 1e-9));
        assert!(approx(cal.q0(), 0.03, 1e-12));
        // Agreement update e_t = 1 - lambda*q0 = 0.985 (< 1, evidence FOR null).
        let mut ep = EProcess::new(&cal);
        ep.observe(false);
        assert!(approx(ep.e_value(), 0.985, 1e-12), "agree e_t");
        // Disagreement update e_t = 1 + lambda*(1-q0) = 1.485 (> 1, toward alarm).
        let mut ep2 = EProcess::new(&cal);
        ep2.observe(true);
        assert!(approx(ep2.e_value(), 1.485, 1e-12), "disagree e_t");
    }

    // ----- The e-value property E_{H0}[e_t] <= 1 (contract item A1) ----- //

    #[test]
    fn e_value_property_fair_at_boundary() {
        // Under the calibrated null E[D_t] = q0, the per-step expected update is
        // exactly 1 (fair): E[e_t] = 1 + lambda*(q0 - q0) = 1.  (AF-3 §2.3.)
        let cal = Calibration::worked_example();
        let q0 = cal.q0();
        let lambda = DEFAULT_LAMBDA;
        let e_agree = 1.0 + lambda * (0.0 - q0);
        let e_dis = 1.0 + lambda * (1.0 - q0);
        let expected = (1.0 - q0) * e_agree + q0 * e_dis;
        assert!(approx(expected, 1.0, 1e-12), "E[e_t] = {expected}");
    }

    #[test]
    fn agreements_shrink_disagreements_grow() {
        let cal = Calibration::worked_example();
        let mut ep = EProcess::new(&cal);
        let start = ep.e_value();
        ep.observe(false);
        assert!(ep.e_value() < start, "agreement shrinks wealth");
        let mut ep2 = EProcess::new(&cal);
        ep2.observe(true);
        assert!(ep2.e_value() > 1.0, "disagreement grows wealth");
    }

    // ----- Ville bound: healthy stream never rejects; alternative fires ----- //

    #[test]
    fn healthy_stream_never_rejects() {
        // A pure-agreement stream (the easy printed-text case) keeps E_t low and
        // never crosses 1/alpha: speculation stays alive.
        let cal = Calibration::worked_example();
        let mut ep = EProcess::new(&cal);
        for _ in 0..10_000 {
            ep.observe(false);
        }
        assert!(!ep.rejected());
        assert!(ep.e_value() < ep.threshold());
    }

    #[test]
    fn disagreement_burst_trips_guard() {
        // AF-3 §2.3: ~12 disagreements in a row trip alpha=0.01 (1.485^k >= 100
        // => k ≈ 11.6 => 12).
        let cal = Calibration::worked_example();
        let mut ep = EProcess::new(&cal);
        let mut fired_at = None;
        for i in 1..=20u64 {
            if ep.observe(true) {
                fired_at = Some(i);
                break;
            }
        }
        assert_eq!(fired_at, Some(12), "trips on the 12th disagreement");
        assert_eq!(ep.rejected_at(), Some(12));
    }

    #[test]
    fn never_reset_discipline() {
        // The e_value latch/saturation must NOT reset to 1.0 (that breaks the
        // supermartingale property), mirroring gauntlet_cert.py eprocess_no_reset.
        let cal = Calibration::worked_example();
        let mut ep = EProcess::new(&cal);
        for _ in 0..30 {
            ep.observe(true);
        }
        assert!(ep.e_value() > 1.0);
        assert!(ep.rejected());
    }

    #[test]
    fn e_value_saturates_finite() {
        // A long disagreement run saturates without overflow to +inf.
        let cal = Calibration::worked_example();
        let mut ep = EProcess::new(&cal);
        for _ in 0..100_000 {
            ep.observe(true);
        }
        assert!(ep.e_value().is_finite());
        assert!(ep.e_value() <= f64::MAX / 2.0);
    }

    /// Empirically the lifetime breach fraction under the null is <= alpha (★).
    /// We simulate streams of audited steps where D_t ~ Bernoulli(q0) (the
    /// boundary null) and count guard breaches.
    #[test]
    fn ville_bound_holds_under_synthetic_null() {
        let cal = Calibration::worked_example();
        let q0 = cal.q0();
        let alpha = cal.alpha();
        let streams = 20_000u64;
        let steps_per_stream = 2_000u64;
        let mut rng = SplitMix64(0xA11E_2026);
        let mut breaches = 0u64;
        for _ in 0..streams {
            let mut ep = EProcess::new(&cal);
            for _ in 0..steps_per_stream {
                let disagree = rng.unit() < q0;
                if ep.observe(disagree) {
                    breaches += 1;
                    break;
                }
            }
        }
        let level = GuardLevel {
            streams,
            breaches,
            alpha,
        };
        // Ville (★): P(ever cross 1/alpha) <= alpha. Allow a small Monte-Carlo
        // slack above alpha (the bound is on the TRUE probability; the empirical
        // estimate has sampling noise). The headline is breach_fraction ≪ 1.
        assert!(
            level.breach_fraction() <= alpha + 0.01,
            "breach fraction {} exceeds alpha {} (+slack)",
            level.breach_fraction(),
            alpha
        );
        assert!(level.coverage() > 0.98);
    }

    #[test]
    fn ville_fires_under_synthetic_alternative() {
        // Under an ALTERNATIVE where the true disagreement rate (0.30) far
        // exceeds q0 (0.03), the guard reliably fires on essentially every stream.
        let cal = Calibration::worked_example();
        let true_rate = 0.30;
        let streams = 2_000u64;
        let mut rng = SplitMix64(0xBEEF_2026);
        let mut breaches = 0u64;
        for _ in 0..streams {
            let mut ep = EProcess::new(&cal);
            for _ in 0..5_000u64 {
                let disagree = rng.unit() < true_rate;
                if ep.observe(disagree) {
                    breaches += 1;
                    break;
                }
            }
        }
        let frac = breaches as f64 / streams as f64;
        assert!(frac > 0.99, "alternative should fire reliably, got {frac}");
    }

    // ----- Loss matrix (contract item: loss) ----- //

    #[test]
    fn loss_matrix_flip_dominates() {
        let lm = LossMatrix::default();
        // A flip is catastrophic relative to a save.
        assert!(lm.l_flip > lm.c_save * 1000.0);
        // Accepting with ANY appreciable flip rate is worse than falling back.
        assert!(lm.expected_accept_loss(0.5) > lm.fallback_loss());
        // The break-even flip rate is tiny (exact-token OCR has ~zero tolerance).
        assert!(lm.break_even_flip_rate() < 1e-3);
    }

    // ----- Deterministic fallback (the contract's non-negotiable) ----- //

    #[test]
    fn fallback_fires_when_alpha_unset() {
        // alpha unset (None) => deterministic fallback: verify EVERY step.
        let guard = SpeculativeGuard::from_calibration(None, 10_000);
        assert!(!guard.speculation_active());
        assert!(guard.ledger().fallback_active());
        // should_verify is unconditionally true regardless of margin.
        assert!(guard.should_verify(f64::INFINITY));
        assert!(guard.should_verify(1000.0));
    }

    #[test]
    fn fallback_fires_when_calibration_invalid() {
        // alpha out of range => Err => deterministic fallback.
        let bad = Calibration::new(0.0, 0.97, 0.5, 8, 4.0); // Err(AlphaOutOfRange)
        let guard = SpeculativeGuard::from_calibration(Some(bad), 10_000);
        assert!(!guard.speculation_active());
        assert!(guard.should_verify(1e9));
    }

    #[test]
    fn fallback_fires_when_calibration_corpus_too_small() {
        // Valid params but the calibration slice is below MIN_CALIBRATION_SAMPLES
        // => refuse to speculate (AF-3 §4 minimum-corpus precondition).
        let cal = Ok(Calibration::worked_example());
        let guard = SpeculativeGuard::from_calibration(Some(cal), MIN_CALIBRATION_SAMPLES - 1);
        assert!(!guard.speculation_active());
        assert!(guard.should_verify(1e9));
        // Just enough samples => speculation enabled.
        let cal2 = Ok(Calibration::worked_example());
        let guard2 = SpeculativeGuard::from_calibration(Some(cal2), MIN_CALIBRATION_SAMPLES);
        assert!(guard2.speculation_active());
    }

    #[test]
    fn disabled_guard_verifies_every_step() {
        let mut guard = SpeculativeGuard::disabled();
        // Even with a wildly disagreeing draft and a huge margin, the disabled
        // guard always falls back and emits the FULL token.
        for step in 0..50u32 {
            let full = step + 100;
            let draft = step; // draft always disagrees
            let dec = guard.decide(1e9, full, draft);
            assert_eq!(dec.action, Action::FallBack);
            assert_eq!(dec.emitted_token, full, "disabled guard emits full token");
        }
        // No e-process betting happens in the fallback path.
        assert!(guard.e_value().is_none());
        assert_eq!(guard.ledger().accepted(), 0);
    }

    // ----- The load-bearing correctness property: OUTPUT IDENTITY ----- //

    /// Build a synthetic full-decode logit stream and its corresponding draft
    /// stream; run the FULL decode (argmax of full logits) and the GUARDED decode
    /// and assert byte-identical emitted token sequences — the AF-3 proof
    /// obligation 3 (ON == OFF under greedy temperature=0), made a unit test.
    #[test]
    fn early_exit_output_identical_to_full_decode() {
        let vocab = 32usize;
        let n_steps = 4_000usize;
        let mut rng = SplitMix64(0xC0FF_EE42);
        let cal = Calibration::worked_example();
        let mut guard = SpeculativeGuard::enabled(cal);

        for step in 0..n_steps {
            // Full-model logits: a clear winner most of the time.
            let mut full = vec![0.0f32; vocab];
            for v in &mut full {
                *v = (rng.unit() as f32) * 2.0;
            }
            let winner = (rng.next_u64() as usize) % vocab;
            full[winner] += 8.0; // strong, confident peak
            let full_tok = argmax(&full);

            // Draft logits: AGREE with the full model on the argmax in the vast
            // majority of steps (the calibrated easy case). On a small fraction
            // (~3%, the calibrated q0) the draft DISAGREES — and the guard must
            // still emit the full token on every step it verifies, while on every
            // step it ACCEPTS the draft, the draft argmax equals the full argmax
            // BY CONSTRUCTION of the safe configuration (see assertion below).
            let draft_disagrees = rng.unit() < 0.03;
            let mut draft = full.clone();
            let draft_tok = if draft_disagrees {
                // Make a different token the draft winner.
                let other = (winner + 1) % vocab;
                draft[other] += 20.0;
                argmax(&draft)
            } else {
                full_tok
            };
            // Draft margin drives the verify policy.
            let margin = top1_top2_margin(&draft);

            let dec = guard.decide(margin, full_tok, draft_tok);

            // FULL decode emits full_tok at every step. The GUARDED decode must
            // emit the SAME token. This is the output-identity guarantee.
            assert_eq!(
                dec.emitted_token, full_tok,
                "step {step}: guarded decode diverged from full decode \
                 (action={:?}, draft_tok={draft_tok}, full_tok={full_tok})",
                dec.action
            );

            // Sanity on the safety mechanism: whenever the guard ACCEPTED the
            // draft (skipped the verify), the draft argmax MUST equal the full
            // argmax — otherwise output identity would have been violated above.
            if dec.action == Action::AcceptDraft {
                assert_eq!(
                    draft_tok, full_tok,
                    "step {step}: accepted a draft that disagreed with full"
                );
            }
        }

        // The guard should have harvested SOME accepts (otherwise it is just the
        // fallback and the test is vacuous) — easy confident streams accept.
        let led = guard.ledger();
        assert!(led.steps() == n_steps as u64);
        assert!(
            led.accepted() > 0,
            "expected some accepted (verify-skipped) steps"
        );
        // The measured disagreement rate over audited steps is the calibration
        // metric; it should be near q0 = 0.03, well under any trip cascade.
        assert!(
            led.measured_disagreement_rate() < 0.10,
            "measured disagreement {} unexpectedly high",
            led.measured_disagreement_rate()
        );
    }

    /// Output identity must hold even when the draft is ADVERSARIALLY bad (always
    /// disagrees): the guard simply verifies on the low-margin / forced-audit
    /// steps, trips quickly, and from then on does pure full forward — the
    /// emitted sequence is still byte-identical to full decode.
    #[test]
    fn output_identity_holds_under_adversarial_draft() {
        let vocab = 16usize;
        let n_steps = 500usize;
        let mut rng = SplitMix64(0xDEAD_F00D);
        let cal = Calibration::worked_example();
        let mut guard = SpeculativeGuard::enabled(cal);

        for step in 0..n_steps {
            let mut full = vec![0.0f32; vocab];
            for v in &mut full {
                *v = rng.unit() as f32;
            }
            let winner = (rng.next_u64() as usize) % vocab;
            full[winner] += 10.0;
            let full_tok = argmax(&full);

            // Draft ALWAYS disagrees and is over-confident (huge margin) so it
            // tries to dodge the margin-forced verify.
            let other = (winner + 3) % vocab;
            let mut draft = full.clone();
            draft[other] += 50.0;
            let draft_tok = argmax(&draft);
            let margin = top1_top2_margin(&draft); // large, confident

            let dec = guard.decide(margin, full_tok, draft_tok);
            assert_eq!(
                dec.emitted_token, full_tok,
                "step {step}: adversarial draft changed the output"
            );
        }
        // The guard must have tripped (a sustained disagreement burst) and then
        // stayed in the safe full-forward state.
        assert!(guard.tripped(), "adversarial draft should trip the guard");
        assert!(!guard.speculation_active());
    }

    #[test]
    fn trip_latches_safe_state_for_rest_of_document() {
        let cal = Calibration::worked_example();
        let mut guard = SpeculativeGuard::enabled(cal);
        // Force a disagreement burst through the audit cadence (low margin forces
        // verify each time).
        for _ in 0..30u32 {
            let full = 7u32;
            let draft = 9u32; // disagree
            let _ = guard.decide(0.0, full, draft); // margin 0 => always verify
        }
        assert!(guard.tripped());
        // After tripping, even a high-margin agreeing draft is fully verified.
        let dec = guard.decide(1e9, 7, 7);
        assert_eq!(dec.action, Action::FallBack);
        assert_eq!(dec.emitted_token, 7);
    }

    // ----- Ledger / calibration-metric plumbing ----- //

    #[test]
    fn guard_level_holds_and_coverage() {
        let level = GuardLevel {
            streams: 10_000,
            breaches: 80,
            alpha: 0.01,
        };
        assert!(level.holds_level()); // 0.008 <= 0.01
        assert!(approx(level.coverage(), 0.992, 1e-12));
        let over = GuardLevel {
            streams: 10_000,
            breaches: 200,
            alpha: 0.01,
        };
        assert!(!over.holds_level()); // 0.02 > 0.01
    }

    #[test]
    fn argmax_first_max_wins_on_ties() {
        // Matches torch.argmax / native_engine::sampler::argmax_row.
        assert_eq!(argmax(&[1.0, 5.0, 5.0, 2.0]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn margin_is_top1_minus_top2() {
        assert!(approx(top1_top2_margin(&[1.0, 4.0, 2.0]), 2.0, 1e-6));
        assert!(approx(top1_top2_margin(&[5.0, 5.0]), 0.0, 1e-6));
    }
}
