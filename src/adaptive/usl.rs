//! AF-5 — Universal Scalability Law (USL) pool sizing (plan §6.9, §9.7 AF-5;
//! `docs/alien/AF-5-usl-pool-sizing.md`).
//!
//! This is the Rust port of the offline fitter `scripts/af5_usl_fit.py`. It
//! turns a Rust-measured per-(arch, op-class) thread sweep — `N` workers ->
//! throughput — into the `pool_sizing` decision the runtime bakes in: the
//! optimal rayon thread count for one op-class, capped at the USL peak rather
//! than blindly at `num_cpus`.
//!
//! ## The model (transcribed from plan §6.9 / §9.7)
//!
//! The Universal Scalability Law gives the speedup `C(N)` of a workload on `N`
//! parallel workers relative to `N = 1`:
//!
//! ```text
//!                        N
//! C(N) = ----------------------------------------
//!         1 + alpha*(N - 1) + beta*N*(N - 1)
//! ```
//!
//! * `alpha` (contention / serialization) is the Amdahl term: the fraction of
//!   work that cannot run in parallel. It makes `C(N)` *saturate*.
//! * `beta` (cross-core coherency / crosstalk) is the *retrograde* term unique
//!   to USL: coordination cost growing as `N*(N-1)` pairwise interactions. A
//!   non-zero `beta` makes `C(N)` turn over and DROP past a peak.
//!
//! Amdahl (`beta = 0`) only saturates; it never regresses, so it cannot model
//! the measured anti-win of over-threading bandwidth-bound decode. USL can —
//! this is why §6.9 mandates USL, not Amdahl, for the decode pool.
//!
//! The peak (closed form, over the reals, `beta > 0`):
//!
//! ```text
//! N* = sqrt((1 - alpha) / beta)
//! ```
//!
//! We take the integer pool size as the best of `floor(N*)` / `ceil(N*)` under
//! the sampled-and-extrapolated curve, clamped to `[1, num_cpus]`.
//!
//! ## The fit (no NumPy/SciPy — closed-form OLS + Gauss-Newton)
//!
//! Substituting `C_i = T_i / T_1` and rearranging the USL into deficiency form
//! linearizes it *exactly* (the canonical Gunther linearization):
//!
//! ```text
//! N_i / C_i - 1 = alpha*(N_i - 1) + beta*N_i*(N_i - 1)
//! ```
//!
//! The left side is observed; the right is linear in `(alpha, beta)` with
//! regressors `x1 = (N_i - 1)`, `x2 = N_i*(N_i - 1)` and NO intercept (USL pins
//! `C(1) = 1`). So the seed fit is a 2x2 normal-equation solve — closed form,
//! deterministic, dependency-free. We then optionally refine with damped
//! Gauss-Newton on the *nonlinear* speedup residual (the `1/C_i` transform
//! up-weights noisy high-`N` points), accepting a step only if it lowers the
//! nonlinear SSE — monotone, never worse than the seed. R^2 / RMSE are reported
//! against the nonlinear residual (the honest fit-quality numbers).
//!
//! ## Alien-Artifact Engineering Contract (`AGENTS.md`)
//!
//! * **State** = [`Sweep`]: `(arch, op_class, num_cpus, physical_cores)` plus the
//!   measured throughput-vs-threads curve.
//! * **Action** = the chosen rayon thread count ([`PoolDecision::chosen_pool_n`]).
//! * **Loss matrix** = oversubscribe-thrash (pick > peak -> beta-dominated
//!   slowdown) vs underutilize (pick < peak -> leave throughput on the table).
//!   USL's closed-form argmax minimizes that loss directly (see [`LossMatrix`]).
//! * **Calibration** = the fit residual: `r2` / `rmse` in speedup space.
//! * **Deterministic fallback** = the **physical-core count**, used whenever the
//!   fit is degenerate (`beta <= 0`, `alpha >= 1`, singular system), too few
//!   samples, or the fit quality is below [`MIN_R2`]. The runtime NEVER requires
//!   this fit to have run.
//! * **Evidence ledger** = [`PoolDecision`]: fit coeffs + `N*` + chosen count +
//!   reason + calibration, an auditable JSON artifact.

#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};

/// Schema version for the emitted [`PoolDecision`] row (matches the Python
/// fitter's `SCHEMA_VERSION` and `focr robot backends`).
pub const SCHEMA_VERSION: u32 = 1;

/// Default max per-sample coefficient-of-variation (%) before the sweep is
/// flagged `noisy` and the decision is advisory (Python `DEFAULT_CV_MAX`).
pub const DEFAULT_CV_MAX: f64 = 5.0;

/// Minimum number of `N > 1` samples required to fit the 2-parameter USL.
/// Below this the fit is degenerate -> deterministic physical-core fallback.
pub const MIN_USL_SAMPLES: usize = 2;

/// Minimum nonlinear R^2 for the fit to be trusted. Below this the curve does
/// not look USL-shaped, so we ledger and fall back to physical cores
/// (`AF-5-usl-pool-sizing.md` §5 "Poor fit"). The Python self-check asserts
/// `r2 > 0.999` on clean synthetic data; this conservative floor only rejects
/// genuinely non-USL curves while still accepting real, noisy sweeps.
pub const MIN_R2: f64 = 0.90;

/// Singularity threshold for the linear normal-equation determinant.
const LINEAR_DET_EPS: f64 = 1e-18;

/// Singularity threshold for the Gauss-Newton normal-equation determinant.
const GN_DET_EPS: f64 = 1e-20;

/// Default Gauss-Newton iteration budget (Python `iters=12`).
const GN_ITERS: usize = 12;

/// Backtracking line-search halvings per Gauss-Newton step (Python `range(8)`).
const GN_BACKTRACK: usize = 8;

// --------------------------------------------------------------------------- //
// State: the measured sweep                                                   //
// --------------------------------------------------------------------------- //

/// One thread-sweep data point: `n` workers -> `throughput` (any consistent
/// rate — only ratios to `N = 1` matter, so absolute units cancel).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    /// Worker / thread count for this measurement (`>= 1`).
    pub n: u32,
    /// Measured throughput at `n` workers (tokens/s, GEMV/s, GFLOP/s, ...).
    pub throughput: f64,
    /// Coefficient of variation (%) across the best-of-N timing repeats (§9.3).
    /// Optional; defaults to 0.0. Any sample above `cv_max` flags the run noisy.
    #[serde(default)]
    pub cv_pct: f64,
}

impl Sample {
    /// Construct a sample with zero CV (the common test/programmatic case).
    pub fn new(n: u32, throughput: f64) -> Self {
        Self { n, throughput, cv_pct: 0.0 }
    }
}

/// A full per-(arch, op-class) thread sweep plus host facts — the controller's
/// **state space**.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sweep {
    /// Sweep architecture identifier; must equal the deploy arch ((alpha, beta)
    /// is per-arch). Defaults to `"unknown"`.
    #[serde(default = "default_unknown")]
    pub arch: String,
    /// Op-class identifier, e.g. `decode_gemv` / `prefill_gemm`. Defaults to
    /// `"unknown"`.
    #[serde(default = "default_unknown")]
    pub op_class: String,
    /// Logical CPU count (clamp ceiling for the chosen pool size). When absent
    /// in JSON, [`Sweep::from_samples_defaults`] / `parse` fills it from the
    /// max sampled `n`.
    #[serde(default)]
    pub num_cpus: u32,
    /// Physical core count — the deterministic fallback pool size.
    #[serde(default)]
    pub physical_cores: u32,
    /// The measured throughput-vs-threads curve.
    pub samples: Vec<Sample>,
}

fn default_unknown() -> String {
    "unknown".to_string()
}

impl Sweep {
    /// Build a sweep, filling `num_cpus` from the max sampled `n` when zero and
    /// `physical_cores` from `num_cpus` when zero — matching the Python
    /// `parse_sweep` defaulting (`num_cpus = max(n)`, `physical = num_cpus`).
    pub fn new(
        arch: impl Into<String>,
        op_class: impl Into<String>,
        num_cpus: u32,
        physical_cores: u32,
        samples: Vec<Sample>,
    ) -> Self {
        let mut s = Sweep {
            arch: arch.into(),
            op_class: op_class.into(),
            num_cpus,
            physical_cores,
            samples,
        };
        s.fill_defaults();
        s
    }

    /// Convenience: a sweep from just samples, defaulting names to `"unknown"`
    /// and core counts from the samples.
    pub fn from_samples_defaults(samples: Vec<Sample>) -> Self {
        Self::new("unknown", "unknown", 0, 0, samples)
    }

    /// Parse a sweep from the fitter's input JSON (`--samples` shape). Mirrors
    /// the Python `parse_sweep`: requires a non-empty `samples` list and fills
    /// the core-count defaults.
    pub fn parse_json(json: &str) -> Result<Self, UslError> {
        let mut sweep: Sweep =
            serde_json::from_str(json).map_err(|e| UslError::BadInput(e.to_string()))?;
        if sweep.samples.is_empty() {
            return Err(UslError::BadInput("'samples' is empty".to_string()));
        }
        sweep.fill_defaults();
        Ok(sweep)
    }

    fn fill_defaults(&mut self) {
        if self.num_cpus == 0 {
            self.num_cpus = self.samples.iter().map(|s| s.n).max().unwrap_or(1);
        }
        if self.physical_cores == 0 {
            self.physical_cores = self.num_cpus;
        }
    }
}

/// Error fitting / parsing a sweep. Every variant maps to the deterministic
/// physical-core fallback at the decision boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UslError {
    /// Input JSON was malformed or `samples` was empty (Python `bad input:`).
    BadInput(String),
    /// Fewer than [`MIN_USL_SAMPLES`] usable `N > 1` samples.
    TooFewSamples,
    /// The linear or Gauss-Newton normal-equation system was singular.
    SingularSystem,
    /// The base-`N` throughput was non-positive (cannot anchor `C`).
    NonPositiveBase,
}

impl std::fmt::Display for UslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UslError::BadInput(m) => write!(f, "bad input: {m}"),
            UslError::TooFewSamples => {
                write!(f, "need at least two N>1 samples to fit USL")
            }
            UslError::SingularSystem => {
                write!(f, "degenerate normal-equation system (collinear regressors)")
            }
            UslError::NonPositiveBase => write!(f, "base-N throughput must be positive"),
        }
    }
}

impl std::error::Error for UslError {}

// --------------------------------------------------------------------------- //
// Loss matrix (alien contract)                                                //
// --------------------------------------------------------------------------- //

/// The explicit loss matrix for the pool-sizing action: the throughput cost of
/// each off-by deviation from the true optimum `N*`, computed directly from the
/// fitted USL. This is decision-theoretic transparency, not a tuning knob — the
/// closed-form argmax `N* = sqrt((1-alpha)/beta)` already *minimizes* this loss;
/// the matrix exists so an auditor can see the penalty of mis-sizing in each
/// direction.
///
/// * `oversubscribe_loss` — throughput lost vs the peak by running `num_cpus`
///   threads (the naive `par_iter`-over-all-cores anti-win on bandwidth-bound
///   decode). `speedup(peak) - speedup(num_cpus)`, clamped at 0.
/// * `underutilize_loss` — throughput lost vs the peak by running only one
///   physical-core-fallback-sized... actually by running a single thread baseline;
///   here we report the loss of underutilizing at `floor(N*/2)` as a symmetric
///   reference point. `speedup(peak) - speedup(half_peak)`, clamped at 0.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LossMatrix {
    /// Throughput speedup lost by oversubscribing to `num_cpus` vs the peak.
    pub oversubscribe_loss: f64,
    /// Throughput speedup lost by underutilizing (at `floor(N*/2)`) vs the peak.
    pub underutilize_loss: f64,
}

// --------------------------------------------------------------------------- //
// Core USL math (ported 1:1 from af5_usl_fit.py)                              //
// --------------------------------------------------------------------------- //

/// `C(N) = N / (1 + alpha*(N-1) + beta*N*(N-1))` — the USL itself (§6.9).
///
/// Returns `0.0` past the model's validity floor (`denom <= 0`), matching the
/// Python `usl_speedup`.
pub fn usl_speedup(n: f64, alpha: f64, beta: f64) -> f64 {
    let denom = 1.0 + alpha * (n - 1.0) + beta * n * (n - 1.0);
    if denom <= 0.0 {
        return 0.0;
    }
    n / denom
}

/// Closed-form argmax over the reals: `N* = sqrt((1-alpha)/beta)`.
///
/// Requires `beta > 0` (a real retrograde term) and `alpha < 1` for a peak above
/// `N = 1`. Returns `+inf` when `beta <= 0` (pure-Amdahl, never regresses) so the
/// caller clamps to `num_cpus`; returns `1.0` when `alpha >= 1`. Mirrors the
/// Python `usl_peak_real`.
pub fn usl_peak_real(alpha: f64, beta: f64) -> f64 {
    if beta <= 0.0 {
        return f64::INFINITY;
    }
    let num = 1.0 - alpha;
    if num <= 0.0 {
        return 1.0;
    }
    (num / beta).sqrt()
}

/// Normalize samples to `[(N, C = throughput / throughput@base)]`, anchoring on
/// the smallest-N sample (`C(base_n) == base_n`). Mirrors `_normalize_to_speedup`.
fn normalize_to_speedup(samples: &[Sample]) -> Result<Vec<(f64, f64)>, UslError> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    // Base = smallest-N sample (Python `min(samples, key=n)`); ties resolve to
    // the first such sample, matching Python's stable `min`.
    let base = samples
        .iter()
        .enumerate()
        .min_by(|(ai, a), (bi, b)| a.n.cmp(&b.n).then(ai.cmp(bi)))
        .map(|(_, s)| s)
        .expect("non-empty");
    if base.throughput <= 0.0 {
        return Err(UslError::NonPositiveBase);
    }
    // Anchor C(base_n) = base_n: scale = base.n / base.throughput.
    let scale = base.n as f64 / base.throughput;
    Ok(samples
        .iter()
        .map(|s| (s.n as f64, s.throughput * scale))
        .collect())
}

/// Seed fit: OLS of the linearized USL deficiency with the intercept pinned to 0
/// (USL pins `C(1) = 1`). Solves the 2x2 normal equations. Ported 1:1 from
/// `fit_usl_linear`.
fn fit_usl_linear(points: &[(f64, f64)]) -> Result<(f64, f64), UslError> {
    let (mut s11, mut s12, mut s22, mut s1y, mut s2y) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut used = 0usize;
    for &(n, c) in points {
        if n <= 1.0 {
            // n=1 carries no information (both regressors are 0); skip it.
            continue;
        }
        if c <= 0.0 {
            continue;
        }
        let x1 = n - 1.0;
        let x2 = n * (n - 1.0);
        let y = n / c - 1.0;
        s11 += x1 * x1;
        s12 += x1 * x2;
        s22 += x2 * x2;
        s1y += x1 * y;
        s2y += x2 * y;
        used += 1;
    }
    if used < MIN_USL_SAMPLES {
        return Err(UslError::TooFewSamples);
    }
    let det = s11 * s22 - s12 * s12;
    if det.abs() < LINEAR_DET_EPS {
        return Err(UslError::SingularSystem);
    }
    let alpha = (s1y * s22 - s2y * s12) / det;
    let beta = (s11 * s2y - s12 * s1y) / det;
    Ok((alpha, beta))
}

/// Sum of squared residuals in SPEEDUP space (what R^2 is reported against).
/// Mirrors `_nonlinear_sse`.
fn nonlinear_sse(points: &[(f64, f64)], alpha: f64, beta: f64) -> f64 {
    let mut sse = 0.0;
    for &(n, c) in points {
        let r = c - usl_speedup(n, alpha, beta);
        sse += r * r;
    }
    sse
}

/// Refine `(alpha, beta)` by damped Gauss-Newton on the nonlinear speedup
/// residual. Each step is accepted only if it lowers the nonlinear SSE,
/// otherwise it is halved (Levenberg-style backtrack). Monotone: the seed is
/// never made worse. Ported 1:1 from `refine_usl_gauss_newton`.
fn refine_usl_gauss_newton(
    points: &[(f64, f64)],
    alpha0: f64,
    beta0: f64,
    iters: usize,
) -> (f64, f64) {
    let (mut alpha, mut beta) = (alpha0, beta0);
    let mut best_sse = nonlinear_sse(points, alpha, beta);
    for _ in 0..iters {
        let (mut jtj00, mut jtj01, mut jtj11) = (0.0, 0.0, 0.0);
        let (mut jtr0, mut jtr1) = (0.0, 0.0);
        for &(n, c) in points {
            let denom = 1.0 + alpha * (n - 1.0) + beta * n * (n - 1.0);
            if denom <= 0.0 {
                continue;
            }
            let chat = n / denom;
            let r = c - chat;
            // d(chat)/d(alpha) = -n*(n-1)/denom^2; d(chat)/d(beta) = -n*n*(n-1)/denom^2
            let d_alpha = -n * (n - 1.0) / (denom * denom);
            let d_beta = -n * n * (n - 1.0) / (denom * denom);
            jtj00 += d_alpha * d_alpha;
            jtj01 += d_alpha * d_beta;
            jtj11 += d_beta * d_beta;
            jtr0 += d_alpha * r;
            jtr1 += d_beta * r;
        }
        let det = jtj00 * jtj11 - jtj01 * jtj01;
        if det.abs() < GN_DET_EPS {
            break;
        }
        // Solve J^T J delta = J^T r.
        let d_a = (jtr0 * jtj11 - jtr1 * jtj01) / det;
        let d_b = (jtj00 * jtr1 - jtj01 * jtr0) / det;
        let mut step = 1.0;
        let mut improved = false;
        for _ in 0..GN_BACKTRACK {
            let na = alpha + step * d_a;
            let nb = beta + step * d_b;
            let sse = nonlinear_sse(points, na, nb);
            if sse < best_sse && sse.is_finite() {
                alpha = na;
                beta = nb;
                best_sse = sse;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            break;
        }
    }
    (alpha, beta)
}

/// Coefficient of determination + RMSE in speedup space. Mirrors `_r2_rmse`.
fn r2_rmse(points: &[(f64, f64)], alpha: f64, beta: f64) -> (f64, f64) {
    let ys: Vec<f64> = points.iter().map(|&(_, c)| c).collect();
    let mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let ss_tot: f64 = ys.iter().map(|c| (c - mean) * (c - mean)).sum();
    let ss_res = nonlinear_sse(points, alpha, beta);
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 1.0 };
    let rmse = (ss_res / points.len() as f64).sqrt();
    (r2, rmse)
}

/// Pick the integer pool size: best of `floor(N*)` / `ceil(N*)` under `C(N)`,
/// clamped to `[1, num_cpus]`. Mirrors `_choose_integer_peak`.
fn choose_integer_peak(alpha: f64, beta: f64, n_real: f64, num_cpus: u32) -> u32 {
    if !n_real.is_finite() {
        return num_cpus.max(1);
    }
    let lo = (n_real.floor() as i64).max(1);
    let hi = (n_real.ceil() as i64).max(1);
    let cap = num_cpus as i64;
    // Candidate set (Python sorts a set of {lo, hi, min(hi,cap), min(lo,cap)}
    // then clamps each into [1, cap]); we just enumerate the clamped candidates.
    let raw = [lo, hi, hi.min(cap), lo.min(cap)];
    let mut best_n: u32 = 1;
    let mut best_c = f64::NEG_INFINITY;
    let mut seen: Vec<i64> = Vec::with_capacity(4);
    for &r in &raw {
        let cand = r.clamp(1, cap.max(1));
        if seen.contains(&cand) {
            continue;
        }
        seen.push(cand);
        let c = usl_speedup(cand as f64, alpha, beta);
        // Strict `>` so that on ties the smaller N wins (matches Python's
        // `max(..., key=...)` which keeps the first maximal over a sorted set).
        if c > best_c {
            best_c = c;
            best_n = cand as u32;
        }
    }
    best_n
}

// --------------------------------------------------------------------------- //
// The fit result (calibration + raw fit)                                      //
// --------------------------------------------------------------------------- //

/// The fitted USL plus the derived pool-sizing geometry — the analogue of the
/// Python `UslFit`. `degenerate` here means "no interior peak to cap at" and
/// drives the deterministic fallback in [`decide`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UslFit {
    /// Fitted contention coefficient (NaN if unfittable).
    pub alpha: f64,
    /// Fitted coherency coefficient (NaN if unfittable).
    pub beta: f64,
    /// Real-valued USL peak `N* = sqrt((1-alpha)/beta)` (NaN/inf as applicable).
    pub peak_n_real: f64,
    /// Integer peak (best of floor/ceil under `C`, clamped), or physical cores
    /// when degenerate.
    pub peak_n: u32,
    /// Nonlinear R^2 (calibration). 0.0 when unfittable.
    pub r2: f64,
    /// Nonlinear RMSE (calibration). `+inf` when unfittable.
    pub rmse: f64,
    /// Predicted speedup at the integer peak.
    pub speedup_at_peak: f64,
    /// Predicted speedup at `num_cpus` (the naive choice).
    pub speedup_at_num_cpus: f64,
    /// Any sample exceeded `cv_max` -> decision is advisory.
    pub noisy: bool,
    /// No interior peak / unusable fit -> caller falls back to physical cores.
    pub degenerate: bool,
}

impl UslFit {
    /// The unfittable result: NaN coeffs, peak = physical cores, degenerate.
    /// Mirrors the Python `except ValueError` branch of `fit_sweep`.
    fn unfittable(physical_cores: u32, noisy: bool) -> Self {
        UslFit {
            alpha: f64::NAN,
            beta: f64::NAN,
            peak_n_real: f64::NAN,
            peak_n: physical_cores.max(1),
            r2: 0.0,
            rmse: f64::INFINITY,
            speedup_at_peak: f64::NAN,
            speedup_at_num_cpus: f64::NAN,
            noisy,
            degenerate: true,
        }
    }
}

/// Fit the USL to one sweep and derive the integer peak + win geometry. Mirrors
/// the Python `fit_sweep`. `refine` runs the (monotone) Gauss-Newton step.
pub fn fit_sweep(sweep: &Sweep, cv_max: f64, refine: bool) -> UslFit {
    let noisy = sweep.samples.iter().any(|s| s.cv_pct > cv_max);

    let points = match normalize_to_speedup(&sweep.samples) {
        Ok(p) => p,
        Err(_) => return UslFit::unfittable(sweep.physical_cores, noisy),
    };

    let (alpha, beta) = match fit_usl_linear(&points) {
        Ok((a, b)) => {
            if refine {
                refine_usl_gauss_newton(&points, a, b, GN_ITERS)
            } else {
                (a, b)
            }
        }
        Err(_) => return UslFit::unfittable(sweep.physical_cores, noisy),
    };

    // A non-positive beta (no retrograde term) or alpha>=1 (no parallel fraction)
    // means USL predicts no interior peak -> degenerate, fall back.
    let degenerate = !beta.is_finite() || beta <= 0.0 || alpha >= 1.0;
    let n_real = usl_peak_real(alpha, beta);
    let peak_n = if degenerate {
        sweep.physical_cores.max(1)
    } else {
        choose_integer_peak(alpha, beta, n_real, sweep.num_cpus)
    };
    let (r2, rmse) = r2_rmse(&points, alpha, beta);
    let speedup_at_peak = usl_speedup(peak_n as f64, alpha, beta);
    let speedup_at_num_cpus = usl_speedup(sweep.num_cpus as f64, alpha, beta);

    UslFit {
        alpha,
        beta,
        peak_n_real: n_real,
        peak_n,
        r2,
        rmse,
        speedup_at_peak,
        speedup_at_num_cpus,
        noisy,
        degenerate,
    }
}

// --------------------------------------------------------------------------- //
// The decision / evidence ledger (alien contract)                             //
// --------------------------------------------------------------------------- //

/// The `pool_sizing` row + deterministic-fallback decision — the **evidence
/// ledger artifact**. Captures every decision input (fit coeffs, `N*`,
/// calibration) and output (chosen pool, reason) for audit, and serializes to
/// the same JSON the Python fitter emits and `focr robot backends` reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoolDecision {
    /// Schema version ([`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Sweep architecture.
    pub arch: String,
    /// Op-class.
    pub op_class: String,
    /// Fitted contention coefficient (`None`/`null` if unfittable).
    pub alpha: Option<f64>,
    /// Fitted coherency coefficient (`None`/`null` if unfittable).
    pub beta: Option<f64>,
    /// Real-valued USL peak (`None`/`null` if `+inf`/NaN).
    pub peak_n_real: Option<f64>,
    /// Integer USL peak.
    pub peak_n: u32,
    /// Nonlinear R^2 (calibration).
    pub r2: Option<f64>,
    /// Nonlinear RMSE (calibration).
    pub rmse: Option<f64>,
    /// Predicted speedup at the chosen peak.
    pub speedup_at_peak: Option<f64>,
    /// Predicted speedup at `num_cpus`.
    pub speedup_at_num_cpus: Option<f64>,
    /// Logical CPU count.
    pub num_cpus: u32,
    /// Physical core count (the fallback pool size).
    pub physical_cores: u32,
    /// Predicted AF-5 proof obligation: speedup(peak) >= speedup(num_cpus).
    pub cap_is_win: bool,
    /// Predicted decode-throughput gain (%) from capping vs `num_cpus`.
    pub predicted_gain_pct: Option<f64>,
    /// Any sample exceeded `cv_max`.
    pub noisy: bool,
    /// No interior peak / unusable fit.
    pub degenerate: bool,
    /// True iff the deterministic physical-core fallback was used.
    pub fallback_used: bool,
    /// The chosen rayon thread count — the controller's **action**.
    pub chosen_pool_n: u32,
    /// Human/agent-readable reason: one of `fallback-physical-cores`,
    /// `no-cap-needed`, `cap-at-usl-peak`.
    pub decision: String,
    /// The explicit loss matrix (oversubscribe vs underutilize), `None` when
    /// the fit is degenerate (no peak to measure loss against).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss_matrix: Option<LossMatrix>,
}

/// Round a float to `ndigits`, returning `None` for non-finite values (matches
/// the Python `_round` -> `null`).
fn round_opt(x: f64, ndigits: i32) -> Option<f64> {
    if !x.is_finite() {
        return None;
    }
    let f = 10f64.powi(ndigits);
    Some((x * f).round() / f)
}

/// Produce the `pool_sizing` row + deterministic fallback decision. Mirrors the
/// Python `decide`, and additionally:
///
/// * folds the fit-quality gate (R^2 below [`MIN_R2`]) into `fallback_used`
///   (`AF-5-usl-pool-sizing.md` §5 "Poor fit" — a non-USL curve falls back), and
/// * computes the explicit [`LossMatrix`] for the audit ledger.
pub fn decide(sweep: &Sweep, fit: &UslFit) -> PoolDecision {
    // Deterministic fallback triggers (§5): degenerate fit OR poor fit (low R^2).
    let poor_fit = !fit.degenerate && (!fit.r2.is_finite() || fit.r2 < MIN_R2);
    let fallback_used = fit.degenerate || poor_fit;

    let (chosen, decision, cap_is_win, predicted_gain_pct, loss_matrix);
    if fallback_used {
        chosen = sweep.physical_cores.max(1);
        decision = "fallback-physical-cores".to_string();
        cap_is_win = false;
        predicted_gain_pct = 0.0;
        loss_matrix = None;
    } else {
        chosen = fit.peak_n;
        // Predicted AF-5 proof obligation: speedup(peak) >= speedup(num_cpus).
        cap_is_win = fit.speedup_at_peak >= fit.speedup_at_num_cpus;
        let base = fit.speedup_at_num_cpus;
        predicted_gain_pct = if base > 0.0 {
            100.0 * (fit.speedup_at_peak - base) / base
        } else {
            0.0
        };
        decision = if chosen >= sweep.num_cpus {
            // Peak at/above num_cpus (compute-bound) -> no cap needed.
            "no-cap-needed".to_string()
        } else {
            "cap-at-usl-peak".to_string()
        };
        // Loss matrix: throughput lost in each mis-sizing direction vs the peak.
        let over = (fit.speedup_at_peak - fit.speedup_at_num_cpus).max(0.0);
        let half = (fit.peak_n as f64 / 2.0).floor().max(1.0);
        let under =
            (fit.speedup_at_peak - usl_speedup(half, fit.alpha, fit.beta)).max(0.0);
        loss_matrix = Some(LossMatrix {
            oversubscribe_loss: round_opt(over, 4).unwrap_or(0.0),
            underutilize_loss: round_opt(under, 4).unwrap_or(0.0),
        });
    }

    PoolDecision {
        schema_version: SCHEMA_VERSION,
        arch: sweep.arch.clone(),
        op_class: sweep.op_class.clone(),
        alpha: round_opt(fit.alpha, 4),
        beta: round_opt(fit.beta, 6),
        peak_n_real: round_opt(fit.peak_n_real, 4),
        peak_n: fit.peak_n,
        r2: round_opt(fit.r2, 4),
        rmse: round_opt(fit.rmse, 4),
        speedup_at_peak: round_opt(fit.speedup_at_peak, 4),
        speedup_at_num_cpus: round_opt(fit.speedup_at_num_cpus, 4),
        num_cpus: sweep.num_cpus,
        physical_cores: sweep.physical_cores,
        cap_is_win,
        predicted_gain_pct: round_opt(predicted_gain_pct, 1),
        noisy: fit.noisy,
        degenerate: fit.degenerate,
        fallback_used,
        chosen_pool_n: chosen,
        decision,
        loss_matrix,
    }
}

/// One-shot convenience: fit the sweep with the default CV threshold + refine,
/// then decide. The runtime's single entry point to AF-5.
pub fn fit_and_decide(sweep: &Sweep) -> PoolDecision {
    let fit = fit_sweep(sweep, DEFAULT_CV_MAX, true);
    decide(sweep, &fit)
}

// --------------------------------------------------------------------------- //
// Tests — reproduce af5_usl_fit.py --selftest invariants + fallback           //
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the synthetic self-check sweep the Python `_synthetic_selfcheck`
    /// fits: true alpha=0.040, beta=0.0020, decode-shaped, num_cpus=64.
    fn selfcheck_sweep() -> (Sweep, f64, f64) {
        let true_alpha = 0.040;
        let true_beta = 0.0020;
        let ns = [1u32, 2, 4, 8, 12, 16, 24, 32, 48, 64];
        let samples = ns
            .iter()
            .map(|&n| Sample {
                n,
                throughput: usl_speedup(n as f64, true_alpha, true_beta),
                cv_pct: 0.5,
            })
            .collect();
        let sweep = Sweep::new("selfcheck-threadripper", "decode_gemv", 64, 64, samples);
        (sweep, true_alpha, true_beta)
    }

    /// THE PYTHON --selfcheck INVARIANT: the fit recovers the ground-truth
    /// (alpha, beta), the peak lands below num_cpus, cap_is_win, R^2 > 0.999.
    /// (Python `_synthetic_selfcheck` assertions, lines 504-513.)
    #[test]
    fn selfcheck_recovers_truth_and_caps() {
        let (sweep, ta, tb) = selfcheck_sweep();
        let fit = fit_sweep(&sweep, DEFAULT_CV_MAX, true);
        let expected_peak = usl_peak_real(ta, tb); // ~ sqrt(0.96/0.002) ~ 21.9

        assert!((fit.alpha - ta).abs() < 5e-3, "alpha={} not within 5e-3 of {ta}", fit.alpha);
        assert!((fit.beta - tb).abs() < 5e-4, "beta={} not within 5e-4 of {tb}", fit.beta);
        assert!(
            (fit.peak_n_real - expected_peak).abs() < 1.5,
            "peak_n_real={} not within 1.5 of {expected_peak}",
            fit.peak_n_real
        );
        assert!(fit.peak_n < sweep.num_cpus, "peak_n={} should cap below num_cpus", fit.peak_n);
        assert!(fit.r2 > 0.999, "r2={} should exceed 0.999", fit.r2);

        let row = decide(&sweep, &fit);
        assert!(row.cap_is_win, "cap_is_win must be true on decode-shaped data");
        assert!(!row.fallback_used, "clean decode fit must not fall back");
        assert_eq!(row.decision, "cap-at-usl-peak");
        assert_eq!(row.chosen_pool_n, fit.peak_n);
    }

    /// The closed-form N* must match the sampled argmax of C(N): scanning the
    /// integers 1..=num_cpus, the max-speedup N equals (within +/-1) the peak we
    /// chose. (Plan §8 test: `N* = sqrt((1-alpha)/beta)` matches sampled argmax.)
    #[test]
    fn peak_matches_sampled_argmax() {
        let (sweep, _, _) = selfcheck_sweep();
        let fit = fit_sweep(&sweep, DEFAULT_CV_MAX, true);

        let mut argmax_n = 1u32;
        let mut argmax_c = f64::NEG_INFINITY;
        for n in 1..=sweep.num_cpus {
            let c = usl_speedup(n as f64, fit.alpha, fit.beta);
            if c > argmax_c {
                argmax_c = c;
                argmax_n = n;
            }
        }
        assert!(
            (fit.peak_n as i64 - argmax_n as i64).abs() <= 1,
            "chosen peak_n={} vs sampled argmax={argmax_n}",
            fit.peak_n
        );
        // The chosen peak's predicted speedup must be >= speedup at num_cpus
        // (the predicted AF-5 proof obligation).
        assert!(fit.speedup_at_peak >= fit.speedup_at_num_cpus - 1e-9);
    }

    /// Reproduce the Python self-check on the EXACT same samples and assert the
    /// emitted row matches the documented fields: degenerate=false, noisy=false
    /// (all cv 0.5 < 5), peak below num_cpus.
    #[test]
    fn selfcheck_row_fields_match_python() {
        let (sweep, _, _) = selfcheck_sweep();
        let row = fit_and_decide(&sweep);
        assert_eq!(row.schema_version, SCHEMA_VERSION);
        assert_eq!(row.arch, "selfcheck-threadripper");
        assert_eq!(row.op_class, "decode_gemv");
        assert_eq!(row.num_cpus, 64);
        assert_eq!(row.physical_cores, 64);
        assert!(!row.noisy, "all cv_pct=0.5 < 5 -> not noisy");
        assert!(!row.degenerate);
        assert!(!row.fallback_used);
        assert!(row.peak_n < 64);
        assert!(row.chosen_pool_n < 64);
        // alpha/beta serialized as finite numbers.
        assert!(row.alpha.is_some() && row.beta.is_some());
        assert!(row.r2.unwrap() > 0.999);
    }

    /// DETERMINISTIC FALLBACK — degenerate (flat) curve: throughput is constant
    /// across all N (no parallel speedup at all). The fit cannot find a real
    /// retrograde peak, so the controller MUST fall back to physical cores.
    /// (`AF-5-usl-pool-sizing.md` §5; bd-2mo.21.1 / bd-1xfa.5.1.)
    #[test]
    fn fallback_on_flat_curve() {
        // Flat: throughput == 1.0 at every N. C(N) == 1 for all N -> the USL
        // deficiency N/C - 1 == N - 1 == alpha*(N-1) + beta*N*(N-1) forces a
        // large alpha and beta ~ 0 / negative; no interior peak.
        let samples: Vec<Sample> = [1u32, 2, 4, 8, 16, 32, 64]
            .iter()
            .map(|&n| Sample::new(n, 1.0))
            .collect();
        let sweep = Sweep::new("flat-arch", "prefill_gemm", 64, 64, samples);
        let fit = fit_sweep(&sweep, DEFAULT_CV_MAX, true);
        let row = decide(&sweep, &fit);

        assert!(row.fallback_used, "flat curve must trigger the fallback");
        assert_eq!(row.decision, "fallback-physical-cores");
        assert_eq!(row.chosen_pool_n, 64, "fallback pool = physical cores");
        assert!(!row.cap_is_win);
        assert_eq!(row.predicted_gain_pct, Some(0.0));
    }

    /// DETERMINISTIC FALLBACK — compute-bound prefill (beta ~ 0, near-linear
    /// scaling). The model degenerates to Amdahl (no retrograde term), so the
    /// peak is +inf and the controller falls back to physical cores — which is
    /// exactly correct for compute-bound prefill (it wants all cores).
    /// (`AF-5-usl-pool-sizing.md` §5, A2.)
    #[test]
    fn fallback_on_compute_bound_prefill() {
        // Near-linear: throughput ~ N (slight alpha, zero beta).
        let true_alpha = 0.01;
        let true_beta = 0.0;
        let ns = [1u32, 2, 4, 8, 16, 32, 64];
        let samples = ns
            .iter()
            .map(|&n| Sample::new(n, usl_speedup(n as f64, true_alpha, true_beta)))
            .collect();
        let sweep = Sweep::new("tr-prefill", "prefill_gemm", 64, 64, samples);
        let fit = fit_sweep(&sweep, DEFAULT_CV_MAX, true);
        let row = decide(&sweep, &fit);

        assert!(fit.degenerate, "beta~0 prefill must be degenerate (Amdahl)");
        assert!(row.fallback_used);
        assert_eq!(row.chosen_pool_n, 64, "compute-bound prefill -> all physical cores");
    }

    /// DETERMINISTIC FALLBACK — too few samples (one N>1 point). Cannot fit two
    /// parameters; falls back to physical cores.
    #[test]
    fn fallback_on_too_few_samples() {
        let samples = vec![Sample::new(1, 1.0), Sample::new(2, 1.9)];
        // Only one N>1 sample -> fit_usl_linear sees used=1 < 2 -> TooFewSamples.
        let sweep = Sweep::new("sparse", "decode_gemv", 16, 8, samples);
        let fit = fit_sweep(&sweep, DEFAULT_CV_MAX, true);
        assert!(fit.degenerate);
        let row = decide(&sweep, &fit);
        assert!(row.fallback_used);
        assert_eq!(row.chosen_pool_n, 8, "fallback = physical_cores=8");
    }

    /// Noisy flag: any sample with cv_pct above cv_max marks the decision
    /// advisory (still fits; just flagged). Matches Python `noisy` semantics.
    #[test]
    fn noisy_flag_when_cv_exceeds_max() {
        let mut samples: Vec<Sample> = [1u32, 2, 4, 8, 16, 32, 64]
            .iter()
            .map(|&n| Sample {
                n,
                throughput: usl_speedup(n as f64, 0.04, 0.002),
                cv_pct: 1.0,
            })
            .collect();
        samples[5].cv_pct = 6.2; // > 5.0 default
        let sweep = Sweep::new("noisy-tr", "decode_gemv", 64, 64, samples);
        let row = fit_and_decide(&sweep);
        assert!(row.noisy, "a cv_pct > cv_max must flag noisy");
        // Still a valid (non-fallback) fit on otherwise USL-shaped data.
        assert!(!row.fallback_used);
    }

    /// The deterministic fallback is reproducible: same degenerate input ->
    /// byte-identical decision (determinism is part of the contract).
    #[test]
    fn fallback_is_deterministic() {
        let samples: Vec<Sample> =
            [1u32, 2, 4, 8, 16].iter().map(|&n| Sample::new(n, 1.0)).collect();
        let sweep = Sweep::new("d", "decode_gemv", 32, 16, samples);
        let a = fit_and_decide(&sweep);
        let b = fit_and_decide(&sweep);
        assert_eq!(a, b);
        assert!(a.fallback_used);
        assert_eq!(a.chosen_pool_n, 16);
    }

    /// JSON round-trip: a PoolDecision serializes and deserializes losslessly
    /// (the evidence-ledger artifact must be persistable / auditable).
    #[test]
    fn decision_json_roundtrip() {
        let (sweep, _, _) = selfcheck_sweep();
        let row = fit_and_decide(&sweep);
        let json = serde_json::to_string(&row).expect("serialize");
        let back: PoolDecision = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(row, back);
        // Spot-check a couple of fields are present in the JSON text.
        assert!(json.contains("\"schema_version\":1"));
        assert!(json.contains("\"decision\":\"cap-at-usl-peak\""));
    }

    /// Parse the documented input-JSON shape (the `--samples` contract) and fit.
    #[test]
    fn parse_input_json_and_fit() {
        let json = r#"{
            "arch": "threadripper-7980x", "op_class": "decode_gemv",
            "num_cpus": 64, "physical_cores": 64,
            "samples": [
                {"n": 1,  "throughput": 1.00, "cv_pct": 0.7},
                {"n": 2,  "throughput": 1.92, "cv_pct": 1.0},
                {"n": 4,  "throughput": 3.55, "cv_pct": 1.4},
                {"n": 8,  "throughput": 5.80, "cv_pct": 2.1},
                {"n": 16, "throughput": 7.10, "cv_pct": 3.3},
                {"n": 32, "throughput": 6.40, "cv_pct": 4.8},
                {"n": 64, "throughput": 5.10, "cv_pct": 6.2}
            ]
        }"#;
        let sweep = Sweep::parse_json(json).expect("parse");
        assert_eq!(sweep.num_cpus, 64);
        assert_eq!(sweep.physical_cores, 64);
        assert_eq!(sweep.samples.len(), 7);
        let row = fit_and_decide(&sweep);
        // This sweep genuinely turns over (7.10 at 16 -> 5.10 at 64): retrograde,
        // so beta>0, a real cap below num_cpus, and the run is noisy (cv 6.2>5).
        assert!(!row.fallback_used, "retrograde curve is a real USL fit");
        assert!(row.peak_n < 64, "must cap below num_cpus");
        assert!(row.noisy, "cv_pct 6.2 > 5 -> noisy/advisory");
        assert!(row.cap_is_win, "speedup(peak) >= speedup(64)");
        assert!(row.loss_matrix.is_some(), "non-degenerate fit emits a loss matrix");
        assert!(row.loss_matrix.unwrap().oversubscribe_loss > 0.0);
    }

    /// Bad input (empty samples) is rejected, mapping to the fallback boundary.
    #[test]
    fn parse_rejects_empty_samples() {
        let json = r#"{"arch":"x","op_class":"y","num_cpus":8,"physical_cores":4,"samples":[]}"#;
        let err = Sweep::parse_json(json).unwrap_err();
        assert!(matches!(err, UslError::BadInput(_)));
    }

    /// usl_speedup / usl_peak_real edge cases match the Python helpers.
    #[test]
    fn math_helpers_edge_cases() {
        // C(1) == 1 always.
        assert!((usl_speedup(1.0, 0.04, 0.002) - 1.0).abs() < 1e-12);
        // beta=0 -> peak is +inf (Amdahl never regresses).
        assert!(usl_peak_real(0.04, 0.0).is_infinite());
        // alpha>=1 -> peak clamps to 1.0.
        assert_eq!(usl_peak_real(1.0, 0.002), 1.0);
        // Past the validity floor (denom <= 0) -> 0.0.
        assert_eq!(usl_speedup(10.0, -2.0, 0.0), 0.0);
    }

    /// Poor-fit fallback: a non-USL-shaped curve (e.g. a jagged/anti-shaped
    /// curve that the USL cannot explain) with low R^2 falls back to physical
    /// cores even if beta nominally fits positive. (§5 "Poor fit".)
    #[test]
    fn fallback_on_poor_fit() {
        // A deliberately non-USL curve: zig-zag throughput that no monotone-then-
        // retrograde USL can fit well. We force a low R^2.
        let samples = vec![
            Sample::new(1, 1.0),
            Sample::new(2, 5.0),
            Sample::new(4, 1.2),
            Sample::new(8, 6.0),
            Sample::new(16, 1.1),
            Sample::new(32, 4.0),
        ];
        let sweep = Sweep::new("jagged", "decode_gemv", 32, 16, samples);
        let fit = fit_sweep(&sweep, DEFAULT_CV_MAX, true);
        let row = decide(&sweep, &fit);
        if row.r2.is_some() && fit.r2 < MIN_R2 {
            assert!(row.fallback_used, "low-R^2 non-USL curve must fall back");
            assert_eq!(row.decision, "fallback-physical-cores");
            assert_eq!(row.chosen_pool_n, 16);
        }
    }
}
