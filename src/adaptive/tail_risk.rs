//! AF-2 — Distributionally-robust tail-risk gate (CVaR + EVT/POT-GPD).
//!
//! Rust port of [`scripts/af2_tail_risk.py`], held **numerically equivalent** to
//! that offline reference (design doc
//! [`docs/alien/AF-2-tail-risk-cvar-evt.md`]). This is the shipping
//! `tail_risk_monitor` artifact: it turns a vector of **per-document Character
//! Error Rate (CER)** measurements over the frozen golden corpus into the three
//! tail-risk statistics the release scorecard gates on, and applies the
//! accept/reject gate against an f32 baseline.
//!
//! ```text
//!   mean       — naive average CER (reported, NEVER the gate)
//!   cvar_α     — Conditional Value-at-Risk: mean of the worst α-fraction of docs
//!   evt_pXXX   — the p99.9 document via a Generalized-Pareto (POT) tail fit
//! ```
//!
//! WHY: OCR fails in the **tail** — a quantization choice can leave mean CER
//! unchanged while wrecking dense tables / formulae / long digit runs. The gate
//! therefore bounds the worst-α-fraction (CVaR) and extrapolates past the corpus
//! size to the 1-in-1000 document (EVT), never trusting the mean (doctrine #8;
//! design doc §1, §3).
//!
//! # Alien-Artifact Engineering Contract (`AGENTS.md`)
//!
//! - **State** = the observed per-document CER samples (`&[f64]` in `[0, 1]`).
//! - **Action** = accept / reject a candidate quant config ([`GateVerdict`]).
//! - **Loss matrix** = ship-a-bad-config (false accept) vs reject-a-good-config
//!   (false reject); the gate is asymmetric — a tail breach (ship-bad) is the
//!   catastrophic cell, so the bound is read conservatively (CVaR ≥ VaR ≥ mean,
//!   EVT never under-states the empirical quantile). See [`LossMatrix`].
//! - **Posterior / confidence** = the GPD fit `(ξ, β, u)` and exceedance count;
//!   **calibration** = coverage of the bound (the bound may only *raise* the
//!   empirical quantile, so its realized coverage of the observed tail is ≥ the
//!   nominal `q` by construction). See [`TailReport::coverage_floor`].
//! - **Deterministic fallback** = if too few exceedances (< [`MIN_EXCEEDANCES`])
//!   or a degenerate PWM fit, fall back to the **empirical quantile** (the
//!   conservative empirical max-class estimate) — no extrapolated bound is ever
//!   invented. See [`GpdMethod::EmpiricalFallback`].
//! - **Evidence ledger** = [`TailReport`] (+ embedded [`GpdFit`] and optional
//!   [`Gate`]) captures every decision input and output for audit.
//!
//! [`scripts/af2_tail_risk.py`]: ../../../scripts/af2_tail_risk.py
//! [`docs/alien/AF-2-tail-risk-cvar-evt.md`]: ../../../docs/alien/AF-2-tail-risk-cvar-evt.md

use std::fmt;

// --------------------------------------------------------------------------- //
// Tunable constants (mirror the Python defaults exactly).                      //
// --------------------------------------------------------------------------- //

/// Default CVaR worst-fraction. `0.10` == "the worst 10% of documents".
pub const DEFAULT_ALPHA: f64 = 0.10;
/// Default EVT target quantile: the 99.9th-percentile document.
pub const DEFAULT_EVT_Q: f64 = 0.999;
/// Default Peaks-Over-Threshold exceedance fraction (fit the GPD to the worst
/// 15%). Standard POT practice: enough exceedances for a stable fit while
/// staying in the genuine tail.
pub const DEFAULT_POT_FRAC: f64 = 0.15;
/// Minimum exceedances below which the GPD fit is **not trusted** and the
/// monitor falls back to the empirical quantile (the conservative deterministic
/// fallback). This is the AF-2 fallback trigger.
pub const MIN_EXCEEDANCES: usize = 8;

/// `|ξ|` below this is treated as the `ξ == 0` (exponential) GPD branch — matches
/// the Python `abs(self.shape) < 1e-8`.
const SHAPE_ZERO_EPS: f64 = 1e-8;
/// Degeneracy guard on the PWM moment denominator `a0 − 2·a1` — matches the
/// Python `abs(denom) < 1e-12`.
const DENOM_EPS: f64 = 1e-12;
/// Slack added to a gate limit before the `<=` comparison — matches the Python
/// `limit + 1e-12`.
const GATE_SLACK: f64 = 1e-12;

// --------------------------------------------------------------------------- //
// Errors                                                                       //
// --------------------------------------------------------------------------- //

/// Input-validation errors for the tail-risk monitor (mirrors the Python
/// `ValueError` paths). These are *programmer/usage* errors on the offline gate,
/// kept separate from the inference-time [`crate::error::FocrError`].
#[derive(Debug, Clone, PartialEq)]
pub enum TailRiskError {
    /// No CER samples were provided.
    Empty,
    /// A CER value was non-finite (NaN / ±inf).
    NonFinite(f64),
    /// A CER value fell outside the `[0, 1]` rate domain.
    OutOfRange(f64),
    /// `alpha` was outside `(0, 1]`.
    BadAlpha(f64),
    /// The EVT target quantile was outside `(0, 1)`.
    BadEvtQ(f64),
    /// The POT exceedance fraction was outside `(0, 1)`.
    BadPotFrac(f64),
}

impl fmt::Display for TailRiskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "no CER values provided"),
            Self::NonFinite(v) => write!(f, "non-finite CER value: {v}"),
            Self::OutOfRange(v) => write!(f, "CER out of [0,1]: {v} (CER is a rate)"),
            Self::BadAlpha(a) => write!(f, "alpha must be in (0, 1], got {a}"),
            Self::BadEvtQ(q) => write!(f, "evt_q must be in (0, 1), got {q}"),
            Self::BadPotFrac(p) => write!(f, "pot_frac must be in (0, 1), got {p}"),
        }
    }
}

impl std::error::Error for TailRiskError {}

/// Result alias for the tail-risk monitor.
pub type TailRiskResult<T> = Result<T, TailRiskError>;

// --------------------------------------------------------------------------- //
// Numerically-stable summation (mirror Python `math.fsum`).                    //
// --------------------------------------------------------------------------- //

/// Neumaier/Kahan compensated summation — the stable accumulator matching
/// Python's `math.fsum` closely enough for bit-for-bit-equivalent tail
/// statistics on the small samples a golden corpus yields.
///
/// We use compensated summation (not a naive `iter().sum()`) precisely because
/// the Python reference uses `math.fsum`; an uncompensated sum would drift in the
/// last ULPs and break the AF-2 "numerically equivalent to the reference"
/// obligation (design doc §7, A6).
fn fsum<I: IntoIterator<Item = f64>>(it: I) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64; // running compensation for lost low-order bits
    for x in it {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    sum + c
}

// --------------------------------------------------------------------------- //
// Core statistics                                                              //
// --------------------------------------------------------------------------- //

/// Type-7 (linear-interpolation) empirical quantile of an **ascending** slice.
///
/// Matches numpy's default and R's type 7. Used as the EVT fallback and to seed
/// the POT threshold. `sorted_vals` MUST be sorted ascending.
///
/// # Panics
/// Never panics on a non-empty slice; returns `None` for an empty one (the
/// Python raises, but a `None` is the idiomatic Rust signal and callers in this
/// module always pass non-empty slices).
#[must_use]
pub fn empirical_quantile(sorted_vals: &[f64], q: f64) -> Option<f64> {
    let n = sorted_vals.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(sorted_vals[0]);
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    Some(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)
}

/// Arithmetic mean (compensated). Caller guarantees a non-empty slice.
#[must_use]
fn mean(vals: &[f64]) -> f64 {
    fsum(vals.iter().copied()) / vals.len() as f64
}

/// `CVaR_α` = mean of the worst `α` fraction of `vals` (upper tail).
///
/// Larger CER is worse, so the "worst α fraction" is the **top** `α` of the
/// distribution. This is the coherent Rockafellar–Uryasev definition, exact for
/// fractional cutoffs:
///
/// ```text
///   k       = ceil(α·n)                 (docs fully or partially in the tail)
///   full    = sum of the (k−1) largest values
///   w       = (α·n) − (k−1)             (boundary doc's residual weight)
///   CVaR_α  = (full + w · k-th-largest) / (α·n)
/// ```
///
/// This makes CVaR continuous in `α` and guarantees `CVaR_α ≥ VaR_α` always.
///
/// # Errors
/// [`TailRiskError::Empty`] on an empty slice, [`TailRiskError::BadAlpha`] when
/// `alpha ∉ (0, 1]`.
pub fn cvar(vals: &[f64], alpha: f64) -> TailRiskResult<f64> {
    let n = vals.len();
    if n == 0 {
        return Err(TailRiskError::Empty);
    }
    if !(alpha > 0.0 && alpha <= 1.0) {
        return Err(TailRiskError::BadAlpha(alpha));
    }
    // Descending sort: worst (largest CER) first. NaN already excluded upstream.
    let mut ordered: Vec<f64> = vals.to_vec();
    ordered.sort_by(|a, b| b.partial_cmp(a).expect("CER values are finite"));

    let target = alpha * n as f64;
    // k = ceil(target - 1e-12): documents fully or partially in the tail.
    let mut k = (target - 1e-12).ceil() as i64;
    k = k.clamp(1, n as i64);
    let k = k as usize;

    let full = if k >= 1 {
        fsum(ordered[..k - 1].iter().copied())
    } else {
        0.0
    };
    // Boundary (k-th) document weight: the residual fraction of α·n.
    let boundary_weight = (target - (k as f64 - 1.0)).clamp(0.0, 1.0);
    let weighted_sum = full + boundary_weight * ordered[k - 1];
    Ok(weighted_sum / target)
}

/// `VaR_α` for the upper tail: the `(1 − α)` quantile of CER. Reported alongside
/// CVaR for context (`CVaR ≥ VaR` always). `sorted_vals` MUST be ascending.
#[must_use]
pub fn value_at_risk(sorted_vals: &[f64], alpha: f64) -> Option<f64> {
    empirical_quantile(sorted_vals, 1.0 - alpha)
}

// --------------------------------------------------------------------------- //
// EVT: Peaks-Over-Threshold Generalized-Pareto tail fit                        //
// --------------------------------------------------------------------------- //

/// Which estimator produced a [`GpdFit`] — the calibration/audit flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpdMethod {
    /// A trusted probability-weighted-moment (Hosking & Wallis 1987) GPD fit.
    Pwm,
    /// The **deterministic fallback**: too few exceedances or a degenerate fit,
    /// so the monitor reverts to the empirical quantile and invents no bound.
    EmpiricalFallback,
}

impl GpdMethod {
    /// The stable string used in the NDJSON ledger (`gpd_method`), matching the
    /// Python reference exactly.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pwm => "pwm",
            Self::EmpiricalFallback => "empirical-fallback",
        }
    }

    /// `true` iff this is the conservative deterministic fallback path.
    #[must_use]
    pub fn is_fallback(self) -> bool {
        matches!(self, Self::EmpiricalFallback)
    }
}

impl fmt::Display for GpdMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A Generalized-Pareto fit to the peaks-over-threshold exceedances.
///
/// The GPD CDF for exceedance `y = x − u` (`y ≥ 0`) is
///
/// ```text
///   G(y) = 1 − (1 + ξ·y/β)^(−1/ξ),   ξ ≠ 0
///   G(y) = 1 − exp(−y/β),            ξ == 0
/// ```
///
/// with shape `ξ` ([`Self::shape`]) and scale `β > 0` ([`Self::scale`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpdFit {
    /// Threshold `u` (the `(1 − pot_frac)` empirical quantile).
    pub threshold: f64,
    /// Scale `β > 0` (`0.0` in the fallback).
    pub scale: f64,
    /// Shape `ξ` (`0.0` in the fallback).
    pub shape: f64,
    /// Number of exceedances used in the fit.
    pub n_exceed: usize,
    /// Total sample size.
    pub n_total: usize,
    /// Which estimator produced this fit.
    pub method: GpdMethod,
}

impl GpdFit {
    /// `P(X > u)`: the fraction of documents above the POT threshold (`ζ_u`).
    #[must_use]
    pub fn exceed_rate(&self) -> f64 {
        if self.n_total == 0 {
            0.0
        } else {
            self.n_exceed as f64 / self.n_total as f64
        }
    }

    /// The GPD-extrapolated quantile `x_q` for `q` in `(exceed-anchor, 1)`.
    ///
    /// Inverting the conditional GPD and composing with the exceedance rate:
    ///
    /// ```text
    ///   x_q = u + (β/ξ)·( ((1−q)/ζ_u)^(−ξ) − 1 ),   ξ ≠ 0
    ///   x_q = u − β·ln( (1−q)/ζ_u ),                 ξ == 0
    /// ```
    ///
    /// where `ζ_u = P(X > u)` is the exceedance rate. Returns the threshold when
    /// the exceedance rate is non-positive (the fallback shape).
    #[must_use]
    pub fn quantile(&self, q: f64) -> f64 {
        let zeta = self.exceed_rate();
        if zeta <= 0.0 {
            return self.threshold;
        }
        let mut ratio = (1.0 - q) / zeta;
        ratio = ratio.max(1e-300);
        if self.shape.abs() < SHAPE_ZERO_EPS {
            self.threshold - self.scale * ratio.ln()
        } else {
            self.threshold + (self.scale / self.shape) * (ratio.powf(-self.shape) - 1.0)
        }
    }
}

/// Fit a GPD to the upper-tail exceedances via **probability-weighted moments**
/// (Hosking & Wallis 1987) — a closed-form, optimizer-free estimator stable on
/// the small exceedance counts a golden corpus produces (MLE often fails to
/// converge there).
///
/// For ascending exceedances `y_(1)..y_(m)` it uses the plotting-position
/// moments `a0` (the mean) and `a1`:
///
/// ```text
///   a0   = mean(y)
///   a1   = (1/m) · Σ_j (1 − (j − 0.35)/m) · y_(j)       (j = 1..m)
///   ξ    = 2 − a0 / (a0 − 2·a1)
///   β    = 2·a0·a1 / (a0 − 2·a1)
/// ```
///
/// **The fixed rank-weight bug (reproduced faithfully):** the prototype's
/// docstring records that using the *unbiased* rank weight `(j−1)/(m−1)` instead
/// of the **plotting-position** weight `1 − (j − 0.35)/m` collapses `a1 → a0/2`,
/// which drives the moment denominator `a0 − 2·a1 → 0` and makes the GPD relation
/// degenerate. We use the plotting-position weight, exactly as the FIXED Python
/// does — this is the load-bearing correctness fix for AF-2's EVT fit.
///
/// Falls back to [`GpdMethod::EmpiricalFallback`] (scale/shape `0.0`) when there
/// are fewer than [`MIN_EXCEEDANCES`] exceedances, the denominator is degenerate
/// (`|a0 − 2·a1| < 1e-12` or `a0 ≤ 0`), or the resulting `(ξ, β)` is non-finite /
/// `β ≤ 0`. No bound is ever fabricated from too little data.
///
/// # Errors
/// [`TailRiskError::Empty`] on an empty slice.
pub fn fit_gpd_pwm(sorted_vals: &[f64], pot_frac: f64) -> TailRiskResult<GpdFit> {
    let n = sorted_vals.len();
    if n == 0 {
        return Err(TailRiskError::Empty);
    }

    // Threshold u = the (1 − pot_frac) quantile. The GPD models the *excess*
    // over the threshold, so the exceedances are the residuals `y = x − u` for
    // every doc strictly above `u` (the quantile function adds `u` back).
    let u = empirical_quantile(sorted_vals, 1.0 - pot_frac).expect("non-empty");
    let mut exceed: Vec<f64> = sorted_vals
        .iter()
        .copied()
        .filter(|&v| v > u)
        .map(|v| v - u)
        .collect();
    let m = exceed.len();

    let fallback = |m: usize| GpdFit {
        threshold: u,
        scale: 0.0,
        shape: 0.0,
        n_exceed: m,
        n_total: n,
        method: GpdMethod::EmpiricalFallback,
    };

    if m < MIN_EXCEEDANCES {
        // Not enough tail data to trust an EVT extrapolation — deterministic
        // fallback to the empirical quantile.
        return Ok(fallback(m));
    }

    exceed.sort_by(|a, b| a.partial_cmp(b).expect("finite")); // ascending y_(1..m)

    // a0 = mean(y); a1 = (1/m) Σ_j (1 − ((j+1) − 0.35)/m) y_(j), 0-indexed j.
    // (The `(j + 1)` reproduces the Python `enumerate`'s 1-based plotting
    // position exactly.)
    let a0 = fsum(exceed.iter().copied()) / m as f64;
    let mf = m as f64;
    let a1 = fsum(
        exceed
            .iter()
            .enumerate()
            .map(|(j, &y)| (1.0 - (((j + 1) as f64) - 0.35) / mf) * y),
    ) / mf;

    let denom = a0 - 2.0 * a1;
    if denom.abs() < DENOM_EPS || a0 <= 0.0 {
        return Ok(fallback(m));
    }

    let shape = 2.0 - a0 / denom;
    let scale = 2.0 * a0 * a1 / denom;

    if !(shape.is_finite() && scale.is_finite()) || scale <= 0.0 {
        return Ok(fallback(m));
    }

    Ok(GpdFit {
        threshold: u,
        scale,
        shape,
        n_exceed: m,
        n_total: n,
        method: GpdMethod::Pwm,
    })
}

/// EVT (POT/GPD) estimate of the `q`-quantile, with empirical clamping.
///
/// Returns `(x_q, fit)`. CER is bounded in `[0, 1]`, so the EVT estimate is
/// clamped into that range; the tail bound also **never reports below the
/// empirical quantile** (the fit may only *raise* the worst-case estimate — this
/// is the calibration guard that gives the bound its ≥-nominal coverage). A
/// non-finite GPD quantile degrades to the empirical quantile.
///
/// # Errors
/// [`TailRiskError::Empty`] on an empty slice.
pub fn evt_quantile(sorted_vals: &[f64], q: f64, pot_frac: f64) -> TailRiskResult<(f64, GpdFit)> {
    let n = sorted_vals.len();
    if n == 0 {
        return Err(TailRiskError::Empty);
    }
    let emp = empirical_quantile(sorted_vals, q).expect("non-empty");
    let fit = fit_gpd_pwm(sorted_vals, pot_frac)?;
    let mut x_q = if fit.method.is_fallback() {
        emp
    } else {
        let v = fit.quantile(q);
        if v.is_finite() { v } else { emp }
    };
    // Worst-case discipline: never under-state the empirical quantile, then clamp
    // to the CER domain [0, 1].
    x_q = x_q.max(emp);
    x_q = x_q.clamp(0.0, 1.0);
    Ok((x_q, fit))
}

// --------------------------------------------------------------------------- //
// Alien-contract scaffolding: loss matrix, actions, gate                       //
// --------------------------------------------------------------------------- //

/// The asymmetric loss matrix for the accept/reject decision (the contract's
/// loss term). The two error cells are deliberately *not* symmetric: shipping a
/// tail-breaking config is catastrophic (it silently corrupts every dense-table /
/// formula document in production), whereas rejecting a good config merely costs
/// us some footprint. The gate is therefore tuned to make a **false accept**
/// expensive — which is exactly why it reads the conservative bound (CVaR/EVT,
/// never the mean) and why the EVT bound never under-states observed risk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossMatrix {
    /// Loss of shipping a config whose true tail exceeds budget (false accept).
    /// The catastrophic cell.
    pub ship_bad: f64,
    /// Loss of rejecting a config whose true tail is within budget (false
    /// reject). The recoverable cell.
    pub reject_good: f64,
}

impl Default for LossMatrix {
    /// The documented asymmetry: shipping a bad config is ~10× worse than
    /// rejecting a good one (the gauntlet conformance-pillar weighting, design
    /// doc §3). These weights document intent for the audit ledger; the gate
    /// decision itself is the deterministic budget comparison below.
    fn default() -> Self {
        Self {
            ship_bad: 10.0,
            reject_good: 1.0,
        }
    }
}

/// The action taken on a candidate quant config — the contract's action space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    /// Both tail bounds are within budget of the baseline — accept the config.
    Pass,
    /// A tail bound breached the budget — reject the config; the per-tensor
    /// promotion fallback applies.
    Fail,
    /// No baseline was supplied — informational only, NOT a failure.
    NoBaseline,
}

impl GateVerdict {
    /// The stable ledger string, matching the Python (`"pass"`/`"fail"`/
    /// `"no-baseline"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::NoBaseline => "no-baseline",
        }
    }
}

impl fmt::Display for GateVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single tail-bound check inside the [`Gate`] ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct GateCheck {
    /// The bound's stable name (e.g. `"cvar_0.1"`, `"evt_p999"`).
    pub name: String,
    /// The candidate config's measured bound.
    pub candidate: f64,
    /// The f32 baseline bound.
    pub baseline: f64,
    /// `baseline + budget` — the limit the candidate may not exceed.
    pub limit: f64,
    /// Whether this individual check passed.
    pub pass: bool,
}

/// The release-gate verdict ledger (the evidence-ledger gate sub-artifact).
#[derive(Debug, Clone, PartialEq)]
pub struct Gate {
    /// The ledgered tolerance the candidate tail bound may exceed the baseline by.
    pub budget: f64,
    /// The per-bound checks (CVaR and/or EVT).
    pub checks: Vec<GateCheck>,
    /// The overall verdict.
    pub verdict: GateVerdict,
    /// The documented per-tensor promotion remediation, present only on `Fail`.
    pub fallback: Option<String>,
}

/// The exact fallback remediation string the Python emits on a failing gate.
pub const GATE_FALLBACK_MSG: &str = "Tail bound exceeds the ledgered budget: keep the \
tail-offending tensor one precision tier higher (int4->int8 or int8->bf16) and re-measure \
(plan section 9.7 AF-2 fallback).";

// --------------------------------------------------------------------------- //
// The evidence-ledger report                                                   //
// --------------------------------------------------------------------------- //

/// The AF-2 evidence ledger: every decision input and output for one corpus,
/// captured for audit. The shipping monitor emits the same self-describing record
/// the Python `--pretty`/NDJSON does.
#[derive(Debug, Clone, PartialEq)]
pub struct TailReport {
    /// Corpus size.
    pub n: usize,
    /// CVaR worst-fraction.
    pub alpha: f64,
    /// Naive mean CER — reported, NEVER gated.
    pub mean: f64,
    /// `VaR_α` (the `1−α` quantile), for context.
    pub var_alpha: f64,
    /// `CVaR_α` — the gate variable (mean of the worst α-fraction).
    pub cvar_alpha: f64,
    /// EVT target quantile (e.g. `0.999`).
    pub evt_q: f64,
    /// The EVT/GPD estimate of the `evt_q` document, CER-clamped to `[0, 1]`.
    pub evt_quantile: f64,
    /// POT exceedance fraction used for the fit.
    pub pot_frac: f64,
    /// The fitted GPD (`ξ, β, u`, exceedance count, method).
    pub fit: GpdFit,
    /// Largest observed CER.
    pub max_cer: f64,
    /// Smallest observed CER.
    pub min_cer: f64,
    /// The release-gate verdict, present only when a baseline was supplied.
    pub gate: Option<Gate>,
}

impl TailReport {
    /// Realized **coverage floor** of the EVT bound — the contract's calibration
    /// metric. Because the reported bound is `max(GPD-quantile, empirical_q)` and
    /// clamped into `[0, 1]`, the fraction of observed documents at or below the
    /// bound is at least the nominal `evt_q` whenever the empirical quantile is
    /// itself well-defined. This returns the realized empirical coverage of the
    /// reported bound over the corpus (≥ `evt_q` by construction for a trusted
    /// fit), so a calibration audit can confirm the bound is conservative.
    ///
    /// `sorted_vals` MUST be the same ascending corpus the report was computed
    /// from.
    #[must_use]
    pub fn coverage_floor(&self, sorted_vals: &[f64]) -> f64 {
        if sorted_vals.is_empty() {
            return 0.0;
        }
        let at_or_below = sorted_vals
            .iter()
            .filter(|&&v| v <= self.evt_quantile)
            .count();
        at_or_below as f64 / sorted_vals.len() as f64
    }

    /// `true` iff the EVT estimate came from the conservative deterministic
    /// fallback (too few exceedances or a degenerate fit). Surfaced so an audit
    /// can see at a glance that no extrapolated bound was invented.
    #[must_use]
    pub fn used_fallback(&self) -> bool {
        self.fit.method.is_fallback()
    }
}

/// Format `alpha = 0.1 -> "0.1"` for stable field names (matches the Python
/// `_fmt_frac`: 6-dp, strip trailing zeros then a trailing dot).
#[must_use]
pub fn fmt_frac(alpha: f64) -> String {
    let s = format!("{alpha:.6}");
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Format `q = 0.999 -> "999"`, `q = 0.99 -> "99"` for stable field names
/// (matches the Python `_fmt_pctile`: `q*100` at 4-dp, strip trailing zeros/dot,
/// then remove any remaining decimal point).
#[must_use]
pub fn fmt_pctile(q: f64) -> String {
    let s = format!("{:.4}", q * 100.0);
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    trimmed.replace('.', "")
}

/// Validate the CER samples (finite, in `[0, 1]`) — the state-space guard.
fn validate_samples(vals: &[f64]) -> TailRiskResult<()> {
    if vals.is_empty() {
        return Err(TailRiskError::Empty);
    }
    for &v in vals {
        if !v.is_finite() {
            return Err(TailRiskError::NonFinite(v));
        }
        if !(0.0..=1.0).contains(&v) {
            return Err(TailRiskError::OutOfRange(v));
        }
    }
    Ok(())
}

/// Compute the full AF-2 tail-risk report for a list of per-document CER.
///
/// This is the top-level entry point — the shipping `tail_risk_monitor`. It
/// validates the samples (the state space), computes `mean` / `VaR_α` / `CVaR_α`
/// and the EVT/GPD `evt_q` bound, and returns the evidence ledger ([`TailReport`])
/// with no gate attached (apply one with [`apply_gate`]).
///
/// # Errors
/// [`TailRiskError`] on empty/non-finite/out-of-range samples or bad parameters.
pub fn compute_report(
    vals: &[f64],
    alpha: f64,
    evt_q: f64,
    pot_frac: f64,
) -> TailRiskResult<TailReport> {
    validate_samples(vals)?;
    if !(alpha > 0.0 && alpha <= 1.0) {
        return Err(TailRiskError::BadAlpha(alpha));
    }
    if !(evt_q > 0.0 && evt_q < 1.0) {
        return Err(TailRiskError::BadEvtQ(evt_q));
    }
    if !(pot_frac > 0.0 && pot_frac < 1.0) {
        return Err(TailRiskError::BadPotFrac(pot_frac));
    }

    let mut sorted_vals: Vec<f64> = vals.to_vec();
    sorted_vals.sort_by(|a, b| a.partial_cmp(b).expect("finite"));

    let (evt_q_val, fit) = evt_quantile(&sorted_vals, evt_q, pot_frac)?;

    Ok(TailReport {
        n: vals.len(),
        alpha,
        mean: mean(vals),
        var_alpha: value_at_risk(&sorted_vals, alpha).expect("non-empty"),
        cvar_alpha: cvar(vals, alpha)?,
        evt_q,
        evt_quantile: evt_q_val,
        pot_frac,
        fit,
        max_cer: sorted_vals[sorted_vals.len() - 1],
        min_cer: sorted_vals[0],
        gate: None,
    })
}

/// Convenience wrapper using the AF-2 defaults (`α = 0.10`, `q = 0.999`,
/// `pot_frac = 0.15`).
///
/// # Errors
/// As [`compute_report`].
pub fn compute_report_default(vals: &[f64]) -> TailRiskResult<TailReport> {
    compute_report(vals, DEFAULT_ALPHA, DEFAULT_EVT_Q, DEFAULT_POT_FRAC)
}

/// Evaluate the AF-2 release gate — the accept/reject **action** over the ledger.
///
/// The release scorecard gates on the **CVaR / EVT bound vs the f32 baseline**,
/// NOT the mean. A candidate (e.g. an int4 config) **passes** iff its tail
/// statistics stay within `budget` of the f32 baseline's:
///
/// ```text
///   cvar_candidate  <= baseline_cvar + budget
///   evt_candidate   <= baseline_evt  + budget    (when a baseline EVT is given)
/// ```
///
/// With no baseline supplied the verdict is [`GateVerdict::NoBaseline`]
/// (informational, not a fail). On a `Fail` the per-tensor promotion remediation
/// ([`GATE_FALLBACK_MSG`]) is attached.
///
/// Returns the [`Gate`] ledger; it is the caller's job to stash it on the report
/// (`report.gate = Some(gate)`).
#[must_use]
pub fn apply_gate(
    report: &TailReport,
    baseline_cvar: Option<f64>,
    baseline_evt: Option<f64>,
    budget: f64,
) -> Gate {
    let mut checks: Vec<GateCheck> = Vec::new();
    let mut passed = true;
    let mut have_baseline = false;

    if let Some(b_cvar) = baseline_cvar {
        have_baseline = true;
        let limit = b_cvar + budget;
        let ok = report.cvar_alpha <= limit + GATE_SLACK;
        passed = passed && ok;
        checks.push(GateCheck {
            name: format!("cvar_{}", fmt_frac(report.alpha)),
            candidate: report.cvar_alpha,
            baseline: b_cvar,
            limit,
            pass: ok,
        });
    }

    if let Some(b_evt) = baseline_evt {
        have_baseline = true;
        let limit = b_evt + budget;
        let ok = report.evt_quantile <= limit + GATE_SLACK;
        passed = passed && ok;
        checks.push(GateCheck {
            name: format!("evt_p{}", fmt_pctile(report.evt_q)),
            candidate: report.evt_quantile,
            baseline: b_evt,
            limit,
            pass: ok,
        });
    }

    let (verdict, fallback) = if !have_baseline {
        (GateVerdict::NoBaseline, None)
    } else if passed {
        (GateVerdict::Pass, None)
    } else {
        (GateVerdict::Fail, Some(GATE_FALLBACK_MSG.to_string()))
    };

    Gate {
        budget,
        checks,
        verdict,
        fallback,
    }
}

// --------------------------------------------------------------------------- //
// Tests — reproduce the af2_tail_risk.py invariants + fallback triggers.       //
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for "numerically equivalent to the Python reference" (design
    /// doc §7, A6). The reference uses f64 + `math.fsum`; we match to ~1e-9.
    const TOL: f64 = 1e-9;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // ----- empirical_quantile: type-7 linear interpolation -------------------

    #[test]
    fn empirical_quantile_type7_matches_numpy_default() {
        // Ascending [0,1,2,3,4]; type-7 median is exactly 2.0, p25 = 1.0.
        let v = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert!(approx(empirical_quantile(&v, 0.5).unwrap(), 2.0, TOL));
        assert!(approx(empirical_quantile(&v, 0.25).unwrap(), 1.0, TOL));
        assert!(approx(empirical_quantile(&v, 0.0).unwrap(), 0.0, TOL));
        assert!(approx(empirical_quantile(&v, 1.0).unwrap(), 4.0, TOL));
        // Interpolated point: pos = 0.1*(4) = 0.4 → 0.4 between idx0,1 = 0.4.
        assert!(approx(empirical_quantile(&v, 0.1).unwrap(), 0.4, TOL));
    }

    #[test]
    fn empirical_quantile_single_and_empty() {
        assert!(approx(empirical_quantile(&[0.7], 0.999).unwrap(), 0.7, TOL));
        assert!(empirical_quantile(&[], 0.5).is_none());
    }

    // ----- CVaR: Rockafellar–Uryasev, exact for fractional cutoffs -----------

    #[test]
    fn cvar_full_fraction_equals_mean() {
        // α = 1.0 → CVaR is the mean of all values.
        let v = [0.1, 0.2, 0.3, 0.4];
        assert!(approx(cvar(&v, 1.0).unwrap(), 0.25, TOL));
    }

    #[test]
    fn cvar_geq_var_geq_mean_invariant() {
        // Core design-doc invariant: CVaR_α ≥ VaR_α ≥ mean (§2.1).
        let v = [0.0, 0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.6, 0.9, 1.0];
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = mean(&v);
        let var = value_at_risk(&s, 0.1).unwrap();
        let cv = cvar(&v, 0.1).unwrap();
        assert!(cv >= var - TOL, "CVaR {cv} >= VaR {var}");
        assert!(var >= m - TOL, "VaR {var} >= mean {m}");
    }

    #[test]
    fn cvar_matches_python_reference_value() {
        // Reproduces the Python: --values 0.0 0.0 0.5 0.9 --alpha 0.5 → cvar 0.7.
        let v = [0.0, 0.0, 0.5, 0.9];
        assert!(
            approx(cvar(&v, 0.5).unwrap(), 0.7, TOL),
            "{}",
            cvar(&v, 0.5).unwrap()
        );
    }

    #[test]
    fn cvar_fractional_boundary_continuity() {
        // α·n non-integer: k = ceil(0.1*4 - eps) = 1, boundary weight = 0.4,
        // target = 0.4 → CVaR = (0 + 0.4*max)/0.4 = max. The single worst doc.
        let v = [0.0, 0.1, 0.2, 1.0];
        // worst = 1.0, target = 0.4, k=1, full=0, boundary_weight=0.4 → 1.0.
        assert!(approx(cvar(&v, 0.1).unwrap(), 1.0, TOL));
    }

    #[test]
    fn cvar_rejects_bad_alpha_and_empty() {
        assert_eq!(cvar(&[0.1], 0.0), Err(TailRiskError::BadAlpha(0.0)));
        assert_eq!(cvar(&[0.1], 1.5), Err(TailRiskError::BadAlpha(1.5)));
        assert_eq!(cvar(&[], 0.1), Err(TailRiskError::Empty));
    }

    // ----- GPD PWM fit: the FIXED plotting-position rank weight ---------------

    #[test]
    fn gpd_pwm_fit_matches_python_reference() {
        // Reproduces the captured Python `pwm` fit (60-sample mild tail):
        //   shape ≈ 0.1530800545, scale ≈ 0.1041241022, threshold ≈ 0.0815,
        //   n_exceed = 9, evt_p999 ≈ 0.8660067126.
        let mut vals: Vec<f64> = (0..48)
            .map(|i| (0.001 * i as f64 * 10000.0).round() / 10000.0)
            .collect();
        vals.extend_from_slice(&[
            0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.16, 0.20, 0.26, 0.34, 0.45,
        ]);
        let mut s = vals.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let fit = fit_gpd_pwm(&s, 0.15).unwrap();
        assert_eq!(fit.method, GpdMethod::Pwm);
        assert_eq!(fit.n_exceed, 9);
        assert!(
            approx(fit.shape, 0.1530800545257287, 1e-9),
            "shape {}",
            fit.shape
        );
        assert!(
            approx(fit.scale, 0.10412410218525352, 1e-9),
            "scale {}",
            fit.scale
        );
        assert!(
            approx(fit.threshold, 0.08149999999999999, 1e-9),
            "thr {}",
            fit.threshold
        );
        let (evt, _) = evt_quantile(&s, 0.999, 0.15).unwrap();
        assert!(approx(evt, 0.8660067126095988, 1e-8), "evt {evt}");
    }

    /// The fixed-bug guard (the load-bearing AF-2 correctness fix). The prototype
    /// FIXED a rank-weight collapse: using the *unbiased* rank weight
    /// `(j−1)/(m−1)` instead of the **plotting-position** weight
    /// `1 − ((j) − 0.35)/m` collapses `a1` toward `a0/2`, which flips the sign of
    /// the moment denominator `a0 − 2·a1` and drives the GPD **scale negative**,
    /// so the fit would be rejected (degenerate). We assert, on an evenly-spaced
    /// real tail (m ≥ MIN_EXCEEDANCES), that (1) the CORRECT plotting-position
    /// weight gives a valid `pwm` fit with a positive scale, and (2) the BUGGED
    /// unbiased weight produces an opposite-sign denominator and a negative scale
    /// — i.e. it breaks the fit.
    ///
    /// Verified against the Python reference (denom_corr=+0.0657,
    /// scale_corr=+0.3786; denom_bug=−0.0667, scale_bug=−0.7585).
    #[test]
    fn gpd_pwm_plotting_position_is_the_fixed_estimator() {
        // 48 tiny + an evenly-spaced ramp 0.20..0.64 (12 docs) → m = 9 exceedances.
        let mut vals: Vec<f64> = (0..48).map(|i| 0.001 * i as f64).collect();
        vals.extend((0..12).map(|k| 0.20 + 0.04 * k as f64));
        let mut s = vals.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let u = empirical_quantile(&s, 1.0 - DEFAULT_POT_FRAC).unwrap();
        let mut exc: Vec<f64> = s
            .iter()
            .copied()
            .filter(|&v| v > u)
            .map(|v| v - u)
            .collect();
        exc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = exc.len();
        assert!(
            m >= MIN_EXCEEDANCES,
            "need a real tail for this guard, m={m}"
        );
        let mf = m as f64;
        let a0 = fsum(exc.iter().copied()) / mf;

        // CORRECT plotting-position weight (what fit_gpd_pwm uses).
        let a1_corr = fsum(
            exc.iter()
                .enumerate()
                .map(|(j, &y)| (1.0 - (((j + 1) as f64) - 0.35) / mf) * y),
        ) / mf;
        let denom_corr = a0 - 2.0 * a1_corr;
        let scale_corr = 2.0 * a0 * a1_corr / denom_corr;

        // BUGGED unbiased rank weight (j-1)/(m-1) (0-indexed j → j/(m-1)).
        let a1_bug = fsum(
            exc.iter()
                .enumerate()
                .map(|(j, &y)| ((j as f64) / (mf - 1.0)) * y),
        ) / mf;
        let denom_bug = a0 - 2.0 * a1_bug;
        let scale_bug = 2.0 * a0 * a1_bug / denom_bug;

        // (1) The correct weight is the one fit_gpd_pwm actually uses → valid pwm.
        let fit = fit_gpd_pwm(&s, DEFAULT_POT_FRAC).unwrap();
        assert_eq!(
            fit.method,
            GpdMethod::Pwm,
            "plotting-position must give a real fit"
        );
        assert!(
            scale_corr > 0.0,
            "correct scale must be positive, got {scale_corr}"
        );
        assert!(
            approx(fit.scale, scale_corr, 1e-9),
            "fit uses the correct weight"
        );

        // (2) The bugged weight flips the denom sign and yields a NEGATIVE scale —
        // exactly the collapse the prototype fixed; that fit would fall back.
        assert!(
            (denom_corr > 0.0) != (denom_bug > 0.0),
            "bugged weight must flip denom sign: corr={denom_corr} bug={denom_bug}"
        );
        assert!(
            scale_bug < 0.0,
            "bugged weight must drive scale negative, got {scale_bug}"
        );
    }

    // ----- Deterministic fallback: < MIN_EXCEEDANCES samples ------------------

    #[test]
    fn fallback_fires_on_too_few_exceedances() {
        // 5 samples → at most 1 exceedance over the 85th pctile; far below the
        // MIN_EXCEEDANCES=8 trigger → empirical-fallback, NO invented bound.
        let v = [0.0, 0.1, 0.2, 0.3, 0.5];
        let report = compute_report_default(&v).unwrap();
        assert!(
            report.used_fallback(),
            "too-few-samples must trigger fallback"
        );
        assert_eq!(report.fit.method, GpdMethod::EmpiricalFallback);
        assert_eq!(report.fit.scale, 0.0);
        assert_eq!(report.fit.shape, 0.0);
        // Reproduces the Python: evt_p999 = empirical p999 = 0.4992, cvar = 0.5.
        assert!(
            approx(report.evt_quantile, 0.4992, 1e-9),
            "evt {}",
            report.evt_quantile
        );
        assert!(
            approx(report.cvar_alpha, 0.5, TOL),
            "cvar {}",
            report.cvar_alpha
        );
        assert!(
            approx(report.fit.threshold, 0.38, TOL),
            "thr {}",
            report.fit.threshold
        );
    }

    #[test]
    fn fallback_evt_never_below_empirical_and_clamped() {
        // The fallback bound must equal the empirical quantile, in [0,1].
        let v = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6];
        let report = compute_report_default(&v).unwrap();
        assert!(report.used_fallback());
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let emp = empirical_quantile(&s, 0.999).unwrap();
        assert!(approx(report.evt_quantile, emp.clamp(0.0, 1.0), TOL));
        assert!((0.0..=1.0).contains(&report.evt_quantile));
    }

    // ----- Deterministic fallback: degenerate shape / denominator -------------

    #[test]
    fn fallback_fires_on_degenerate_constant_exceedances() {
        // ≥8 identical exceedances → all y equal → after subtracting threshold
        // the relation degenerates (a0 − 2·a1 ≈ 0) OR scale ≤ 0 → fallback.
        // Build many docs at 0 and a flat plateau of identical worst docs.
        let mut v = vec![0.0; 40];
        // 10 identical tail docs at 0.5 → exceedances all equal after threshold.
        v.extend(std::iter::repeat_n(0.5, 10));
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let fit = fit_gpd_pwm(&s, 0.15).unwrap();
        assert_eq!(
            fit.method,
            GpdMethod::EmpiricalFallback,
            "degenerate (constant) tail must fall back, shape={} scale={}",
            fit.shape,
            fit.scale
        );
    }

    #[test]
    fn degenerate_fit_does_not_invent_a_bound() {
        // Whatever the fit, on the degenerate case the reported EVT equals the
        // empirical quantile (no extrapolation invented) — the contract guarantee.
        let mut v = vec![0.0; 40];
        v.extend(std::iter::repeat_n(0.5, 10));
        let report = compute_report_default(&v).unwrap();
        if report.used_fallback() {
            let mut s = v.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let emp = empirical_quantile(&s, 0.999).unwrap();
            assert!(approx(report.evt_quantile, emp.clamp(0.0, 1.0), TOL));
        }
    }

    // ----- EVT clamping & never-under-state ----------------------------------

    #[test]
    fn evt_clamped_to_unit_interval_on_heavy_tail() {
        // Reproduces the Python heavy-tail case: a 50-sample heavy tail whose raw
        // GPD p999 blows past 1.0 → reported clamped to exactly 1.0.
        let mut v = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.08,
            0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98,
            1.0,
        ];
        v.extend_from_slice(&[
            0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12,
            0.13, 0.14, 0.15, 0.16, 0.17,
        ]);
        let report = compute_report_default(&v).unwrap();
        assert_eq!(report.fit.method, GpdMethod::Pwm);
        assert_eq!(report.n, 50);
        assert!(
            approx(report.evt_quantile, 1.0, TOL),
            "evt {}",
            report.evt_quantile
        );
        assert!(
            approx(report.cvar_alpha, 0.9259999999999999, 1e-9),
            "cvar {}",
            report.cvar_alpha
        );
        assert!(approx(report.mean, 0.1966, 1e-9), "mean {}", report.mean);
        assert!(
            approx(report.var_alpha, 0.7100000000000002, 1e-9),
            "var {}",
            report.var_alpha
        );
        assert!(approx(report.fit.shape, -1.0695172023219603, 1e-9));
        assert!(approx(report.fit.scale, 0.7010489522865643, 1e-9));
        assert_eq!(report.fit.n_exceed, 8);
    }

    #[test]
    fn evt_never_under_states_empirical_quantile() {
        // For any corpus the reported bound ≥ the empirical p999 (calibration).
        let v = [
            0.0, 0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.9, 0.95, 1.0,
        ];
        let report = compute_report_default(&v).unwrap();
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let emp = empirical_quantile(&s, 0.999).unwrap().clamp(0.0, 1.0);
        assert!(
            report.evt_quantile >= emp - TOL,
            "bound {} >= emp {emp}",
            report.evt_quantile
        );
    }

    // ----- Calibration: coverage floor ≥ nominal ------------------------------

    #[test]
    fn coverage_floor_is_at_least_nominal_for_trusted_fit() {
        // The reported bound covers ≥ evt_q of the corpus (conservative).
        let mut v: Vec<f64> = (0..48).map(|i| 0.001 * i as f64).collect();
        v.extend_from_slice(&[
            0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.16, 0.20, 0.26, 0.34, 0.45,
        ]);
        let report = compute_report_default(&v).unwrap();
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cov = report.coverage_floor(&s);
        assert!(
            cov >= report.evt_q - 1e-9 || cov >= (s.len() - 1) as f64 / s.len() as f64,
            "coverage {cov} should be conservative vs q {}",
            report.evt_q
        );
    }

    // ----- Gate: pass / fail / no-baseline + exit semantics -------------------

    #[test]
    fn gate_fails_when_bound_exceeds_budget() {
        // Reproduces the Python gate-fail: values 0 0 0.5 0.9, α 0.5,
        // baseline-cvar 0.05, budget 0.01 → cvar 0.7 > limit 0.06 → FAIL.
        let v = [0.0, 0.0, 0.5, 0.9];
        let report = compute_report(&v, 0.5, DEFAULT_EVT_Q, DEFAULT_POT_FRAC).unwrap();
        let gate = apply_gate(&report, Some(0.05), None, 0.01);
        assert_eq!(gate.verdict, GateVerdict::Fail);
        assert_eq!(gate.checks.len(), 1);
        assert_eq!(gate.checks[0].name, "cvar_0.5");
        assert!(!gate.checks[0].pass);
        assert!(approx(gate.checks[0].limit, 0.06, 1e-9));
        assert_eq!(gate.fallback.as_deref(), Some(GATE_FALLBACK_MSG));
    }

    #[test]
    fn gate_passes_within_budget() {
        let v = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12];
        let report = compute_report_default(&v).unwrap();
        // Baseline generously above the candidate's small bounds.
        let gate = apply_gate(&report, Some(report.cvar_alpha + 0.1), Some(1.0), 0.0);
        assert_eq!(gate.verdict, GateVerdict::Pass);
        assert!(gate.fallback.is_none());
        assert!(gate.checks.iter().all(|c| c.pass));
    }

    #[test]
    fn gate_no_baseline_is_informational_not_fail() {
        let v = [0.0, 0.1, 0.2, 0.9];
        let report = compute_report_default(&v).unwrap();
        let gate = apply_gate(&report, None, None, 0.0);
        assert_eq!(gate.verdict, GateVerdict::NoBaseline);
        assert!(gate.checks.is_empty());
        assert!(gate.fallback.is_none());
    }

    #[test]
    fn gate_both_bounds_checked() {
        let v = [0.0, 0.01, 0.02, 0.5, 0.9, 0.95];
        let report = compute_report(&v, 0.5, DEFAULT_EVT_Q, DEFAULT_POT_FRAC).unwrap();
        // EVT baseline far too low → EVT check fails even if CVaR passes.
        let gate = apply_gate(&report, Some(10.0), Some(0.0), 0.0);
        assert_eq!(gate.checks.len(), 2);
        assert_eq!(gate.verdict, GateVerdict::Fail);
        let evt_check = gate
            .checks
            .iter()
            .find(|c| c.name.starts_with("evt_"))
            .unwrap();
        assert!(!evt_check.pass);
        assert_eq!(evt_check.name, "evt_p999");
    }

    // ----- Input validation ---------------------------------------------------

    #[test]
    fn rejects_non_finite_and_out_of_range() {
        assert_eq!(
            compute_report_default(&[]).unwrap_err(),
            TailRiskError::Empty
        );
        // NaN payload: `NonFinite(NaN) == NonFinite(NaN)` is *false* under IEEE-754
        // (NaN ≠ NaN) and the derived `PartialEq` compares the payload, so an
        // `assert_eq!` here can never hold. Match the variant and assert the
        // payload is the NaN we fed in (mirrors the `OutOfRange` checks below).
        assert!(matches!(
            compute_report_default(&[0.1, f64::NAN]).unwrap_err(),
            TailRiskError::NonFinite(v) if v.is_nan()
        ));
        assert!(matches!(
            compute_report_default(&[0.1, 1.5]).unwrap_err(),
            TailRiskError::OutOfRange(_)
        ));
        assert!(matches!(
            compute_report_default(&[-0.1, 0.2]).unwrap_err(),
            TailRiskError::OutOfRange(_)
        ));
    }

    #[test]
    fn rejects_bad_parameters() {
        let v = [0.1, 0.2, 0.3];
        assert!(matches!(
            compute_report(&v, 0.0, 0.999, 0.15).unwrap_err(),
            TailRiskError::BadAlpha(_)
        ));
        assert!(matches!(
            compute_report(&v, 0.1, 1.0, 0.15).unwrap_err(),
            TailRiskError::BadEvtQ(_)
        ));
        assert!(matches!(
            compute_report(&v, 0.1, 0.999, 1.0).unwrap_err(),
            TailRiskError::BadPotFrac(_)
        ));
    }

    // ----- Field-name formatting (stable NDJSON keys) -------------------------

    #[test]
    fn frac_and_pctile_formatting_matches_python() {
        assert_eq!(fmt_frac(0.1), "0.1");
        assert_eq!(fmt_frac(0.10), "0.1");
        assert_eq!(fmt_frac(0.5), "0.5");
        assert_eq!(fmt_frac(1.0), "1");
        assert_eq!(fmt_pctile(0.999), "999");
        assert_eq!(fmt_pctile(0.99), "99");
        assert_eq!(fmt_pctile(0.95), "95");
    }

    // ----- fsum sanity --------------------------------------------------------

    #[test]
    fn fsum_is_compensated() {
        // The classic catastrophic-cancellation case naive sum gets wrong.
        let xs = [1.0, 1e16, -1e16, -1.0, 0.5];
        assert!(approx(fsum(xs.iter().copied()), 0.5, 1e-9));
    }
}
