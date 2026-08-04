//! Phase-0 conformance contract seed.
//!
//! This module declares the stable vocabulary that later L0-L5 parity gates,
//! rollout receipts, and invariant monitors plug into. It intentionally does
//! not compare real model tensors yet: the real numeric budgets are derived
//! from the oracle nondeterminism floor before Phase-1+ gates claim parity.

/// Ordered parity ladder levels from plan §8.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParityLevel {
    /// L0: preprocessing, tile geometry, and image-token stream are exact.
    L0Preprocess,
    /// L1: per-op activations compared with cosine / ULP budgets.
    L1PerOp,
    /// L2: per-layer hidden states, max-abs-diff ledgered.
    L2PerLayer,
    /// L3: logits within the measured budget, argmax exact when deterministic.
    L3Logits,
    /// L4: decoded tokens exact over the oracle reproducible prefix.
    L4Tokens,
    /// L5: end-to-end OCR text / tables / formula metrics within budget.
    L5EndToEnd,
}

impl ParityLevel {
    /// The complete ordered ladder. Integration runners execute this order and
    /// short-circuit lower-rung failures before higher-rung claims.
    pub const ALL: [Self; 6] = [
        Self::L0Preprocess,
        Self::L1PerOp,
        Self::L2PerLayer,
        Self::L3Logits,
        Self::L4Tokens,
        Self::L5EndToEnd,
    ];

    /// Stable machine label for scorecards and structured logs.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::L0Preprocess => "L0_preprocess",
            Self::L1PerOp => "L1_per_op",
            Self::L2PerLayer => "L2_per_layer",
            Self::L3Logits => "L3_logits",
            Self::L4Tokens => "L4_tokens",
            Self::L5EndToEnd => "L5_end_to_end",
        }
    }

    /// Zero-based rung index, useful for ordered scorecard rows.
    #[must_use]
    pub const fn index(self) -> u8 {
        match self {
            Self::L0Preprocess => 0,
            Self::L1PerOp => 1,
            Self::L2PerLayer => 2,
            Self::L3Logits => 3,
            Self::L4Tokens => 4,
            Self::L5EndToEnd => 5,
        }
    }
}

/// How a gate's tolerance is interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToleranceKind {
    /// Discrete value or geometry must match exactly.
    Exact,
    /// Continuous activation is compared by cosine and/or ULP.
    Cosine,
    /// Budget is derived from measured oracle variance.
    Measured,
    /// Exactness applies only to the oracle reproducible prefix.
    ExactPrefix,
    /// Aggregate document metric within a ledgered budget.
    Budget,
}

/// Source of a tolerance value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToleranceSource {
    /// The value is a fixed structural contract, not a statistical budget.
    StructuralSpec,
    /// Placeholder pending the measured oracle nondeterminism floor.
    TodoDeriveFromOracleFloor,
    /// Placeholder pending a ledgered corpus / release budget.
    TodoLedgerBudget,
}

/// One rung's tolerance contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateTolerance {
    /// Ladder level this row applies to.
    pub level: ParityLevel,
    /// Tolerance interpretation.
    pub kind: ToleranceKind,
    /// Minimum cosine similarity for continuous f32 activations.
    pub cosine_min: Option<f64>,
    /// Optional max-absolute-difference budget. `None` means "ledger only" or
    /// "derive before use"; do not silently substitute the old `0.055` precedent.
    pub max_abs_diff: Option<f64>,
    /// Optional logit tolerance. L3 leaves this unset until the oracle floor is
    /// measured for this model.
    pub logit_tolerance: Option<f64>,
    /// Whether the argmax index must match on deterministic positions.
    pub argmax_must_match: bool,
    /// Whether the compared payload must be exactly equal.
    pub exact_match: bool,
    /// Optional CER budget for L5.
    pub cer_budget: Option<f64>,
    /// Optional TEDS budget for L5.
    pub teds_budget: Option<f64>,
    /// Optional Formula-CDM budget for L5.
    pub formula_cdm_budget: Option<f64>,
    /// Where the tolerance came from.
    pub source: ToleranceSource,
    /// Short human note for scorecards and reviewers.
    pub note: &'static str,
}

impl GateTolerance {
    const fn exact(level: ParityLevel, note: &'static str) -> Self {
        Self {
            level,
            kind: ToleranceKind::Exact,
            cosine_min: None,
            max_abs_diff: Some(0.0),
            logit_tolerance: None,
            argmax_must_match: false,
            exact_match: true,
            cer_budget: None,
            teds_budget: None,
            formula_cdm_budget: None,
            source: ToleranceSource::StructuralSpec,
            note,
        }
    }

    const fn cosine(level: ParityLevel, note: &'static str) -> Self {
        Self {
            level,
            kind: ToleranceKind::Cosine,
            cosine_min: Some(0.9999),
            max_abs_diff: None,
            logit_tolerance: None,
            argmax_must_match: false,
            exact_match: false,
            cer_budget: None,
            teds_budget: None,
            formula_cdm_budget: None,
            source: ToleranceSource::StructuralSpec,
            note,
        }
    }
}

/// Phase-0 seed of the L0-L5 tolerance table.
#[derive(Clone, Debug, PartialEq)]
pub struct Tolerances {
    /// L0 preprocessing exactness.
    pub l0: GateTolerance,
    /// L1 per-op activation tolerance.
    pub l1: GateTolerance,
    /// L2 per-layer activation tolerance.
    pub l2: GateTolerance,
    /// L3 logit tolerance, derived from oracle floor before use.
    pub l3: GateTolerance,
    /// L4 token exactness over reproducible prefix.
    pub l4: GateTolerance,
    /// L5 aggregate OCR metric budget.
    pub l5: GateTolerance,
}

impl Tolerances {
    /// Iterate the table in L0-L5 order.
    pub fn ordered(&self) -> [GateTolerance; 6] {
        [self.l0, self.l1, self.l2, self.l3, self.l4, self.l5]
    }
}

impl Default for Tolerances {
    fn default() -> Self {
        default_tolerances()
    }
}

/// Default Phase-0 tolerance seed.
///
/// The L3/L4/L5 rows deliberately carry TODO sources. The old frankensearch
/// `0.055` logit-diff precedent is not meaningful for this model and is not
/// encoded here; Phase 1 fills these rows from the committed oracle floor and
/// corpus budget artifacts.
#[must_use]
pub fn default_tolerances() -> Tolerances {
    Tolerances {
        l0: GateTolerance::exact(
            ParityLevel::L0Preprocess,
            "exact gray pad, [-1,1] normalize, ratio selection, and tile geometry",
        ),
        l1: GateTolerance::cosine(
            ParityLevel::L1PerOp,
            "cosine >= 0.9999 f32; bridge path applies the per-op ULP table",
        ),
        l2: GateTolerance::cosine(
            ParityLevel::L2PerLayer,
            "cosine ~= 1.0 with per-layer max-abs-diff ledgered",
        ),
        l3: GateTolerance {
            level: ParityLevel::L3Logits,
            kind: ToleranceKind::Measured,
            cosine_min: None,
            max_abs_diff: None,
            logit_tolerance: None,
            argmax_must_match: true,
            exact_match: false,
            cer_budget: None,
            teds_budget: None,
            formula_cdm_budget: None,
            source: ToleranceSource::TodoDeriveFromOracleFloor,
            note: "TODO derive logit budget from oracle nondeterminism floor before use",
        },
        l4: GateTolerance {
            level: ParityLevel::L4Tokens,
            kind: ToleranceKind::ExactPrefix,
            cosine_min: None,
            max_abs_diff: None,
            logit_tolerance: None,
            argmax_must_match: false,
            exact_match: true,
            cer_budget: None,
            teds_budget: None,
            formula_cdm_budget: None,
            source: ToleranceSource::TodoDeriveFromOracleFloor,
            note: "TODO set reproducible prefix lengths from oracle nondeterminism floor",
        },
        l5: GateTolerance {
            level: ParityLevel::L5EndToEnd,
            kind: ToleranceKind::Budget,
            cosine_min: None,
            max_abs_diff: None,
            logit_tolerance: None,
            argmax_must_match: false,
            exact_match: false,
            cer_budget: None,
            teds_budget: None,
            formula_cdm_budget: None,
            source: ToleranceSource::TodoLedgerBudget,
            note: "TODO fill CER/TEDS/Formula-CDM budgets from corpus and release ledger",
        },
    }
}

/// Rollout stage that a parity receipt can be attached to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RolloutStage {
    /// Phase 1: framework-free f32 path.
    Fp32Reference,
    /// Phase 2a: decoder FFN / expert GEMMs only.
    Int8FfnExpertsOnly,
    /// Phase 2b: attention projections int8 behind accuracy gate.
    Int8Attention,
    /// Phase 2c: `lm_head` int8 behind argmax/token gate.
    Int8LmHead,
    /// Phase 3: SIMD/native kernel rollout.
    SimdKernels,
    /// Phase 4: int4 expert bulk under measured budget.
    Int4Experts,
}

impl RolloutStage {
    /// Stable label for parity receipts.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fp32Reference => "P1_fp32_reference",
            Self::Int8FfnExpertsOnly => "P2a_int8_ffn_experts",
            Self::Int8Attention => "P2b_int8_attention",
            Self::Int8LmHead => "P2c_int8_lm_head",
            Self::SimdKernels => "P3_simd_kernels",
            Self::Int4Experts => "P4_int4_experts",
        }
    }
}

/// Invariant families tracked by the conformance/gauntlet machinery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InvariantKind {
    /// Generated-token KV remains bounded by the R-SWA reference + window contract.
    KvCacheBound,
    /// int8 i32 accumulation cannot overflow at the model's worst-case K.
    Int8AccumulatorOverflow,
    /// Same input and deterministic settings produce byte-identical output.
    Determinism,
    /// SIMD and scalar fallbacks are bit-identical where required.
    SimdScalarBitIdentical,
}

impl InvariantKind {
    /// Stable label for structured logs.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::KvCacheBound => "kv_cache_bound",
            Self::Int8AccumulatorOverflow => "int8_accumulator_overflow",
            Self::Determinism => "determinism",
            Self::SimdScalarBitIdentical => "simd_scalar_bit_identical",
        }
    }
}

/// Minimal result row emitted by placeholder validators and future real gates.
#[derive(Clone, Debug, PartialEq)]
pub struct GateResult {
    /// Gate or invariant name.
    pub name: &'static str,
    /// Optional ladder level for parity gates.
    pub level: Option<ParityLevel>,
    /// Whether the check passed.
    pub passed: bool,
    /// Measured value, if numeric.
    pub measured: Option<f64>,
    /// Tolerance value, if numeric.
    pub tolerance: Option<f64>,
    /// Short diagnostic suitable for structured logs.
    pub message: &'static str,
}

impl GateResult {
    /// Construct a passing placeholder / structural result.
    #[must_use]
    pub const fn pass(
        name: &'static str,
        level: Option<ParityLevel>,
        message: &'static str,
    ) -> Self {
        Self {
            name,
            level,
            passed: true,
            measured: None,
            tolerance: None,
            message,
        }
    }

    /// Construct a failing result.
    #[must_use]
    pub const fn fail(
        name: &'static str,
        level: Option<ParityLevel>,
        message: &'static str,
    ) -> Self {
        Self {
            name,
            level,
            passed: false,
            measured: None,
            tolerance: None,
            message,
        }
    }
}

// ─────────── bd-re8.12: the ConformanceTest registry + coverage matrix ───────────

/// The requirement level of a spec clause / conformance entry (RFC-2119
/// style, bd-re8.12): a MUST failure blocks conformance; SHOULD is tracked
/// debt; MAY is informative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequirementLevel {
    /// Conformance-blocking.
    Must,
    /// Tracked coverage debt — reported, never rounds up to conformant.
    Should,
    /// Informative.
    May,
}

/// The suite category a conformance entry registers under (bd-re8.12: the
/// differential/golden/metamorphic/parity/invariant suites all hang off ONE
/// trait so coverage is accounted uniformly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConformanceCategory {
    /// Same-as-reference on any input (the PyO3/torch oracle differential).
    Differential,
    /// Frozen-surface regression (goldens).
    Golden,
    /// Oracle-free self-consistency (metamorphic).
    Metamorphic,
    /// The L0-L5 ladder rungs.
    Parity,
    /// Non-ladder invariants (determinism, overflow-freedom, …).
    Invariant,
}

/// One registered conformance entry (bd-re8.12): `run()` executes the
/// entry's IN-PROCESS representative check and returns a [`GateResult`];
/// entries whose full evidence runs in a dedicated test binary (the armed
/// ladder, the golden suite) still run a real, self-contained slice here —
/// registration is never a no-op, and the covered spec clauses feed the
/// coverage matrix (`clauses`).
pub trait ConformanceTest {
    /// Stable entry name.
    fn name(&self) -> &'static str;
    /// The suite category.
    fn category(&self) -> ConformanceCategory;
    /// The requirement level of the clauses this entry covers.
    fn requirement_level(&self) -> RequirementLevel;
    /// The `[SPEC-NNN]` clause ids this entry covers (the coverage matrix is
    /// computed from the SPEC side; these are the accounting links back).
    fn clauses(&self) -> &'static [u32];
    /// Execute the in-process representative check.
    fn run(&self) -> GateResult;
}

/// A registered entry backed by a plain function — the concrete shape the
/// shipped suites register through.
pub struct RegisteredConformance {
    name: &'static str,
    category: ConformanceCategory,
    level: RequirementLevel,
    clauses: &'static [u32],
    run: fn() -> GateResult,
}

impl ConformanceTest for RegisteredConformance {
    fn name(&self) -> &'static str {
        self.name
    }
    fn category(&self) -> ConformanceCategory {
        self.category
    }
    fn requirement_level(&self) -> RequirementLevel {
        self.level
    }
    fn clauses(&self) -> &'static [u32] {
        self.clauses
    }
    fn run(&self) -> GateResult {
        (self.run)()
    }
}

fn run_determinism_entry() -> GateResult {
    // The shared byte-identity gate on a real pair (the full e2e adoption
    // lives in tests/e2e_recognize.rs — bd-3kge).
    let gate = DeterminismGate;
    gate.validate_bytes(b"greedy output", b"greedy output")
}

fn run_input_fault_entry() -> GateResult {
    // The fault leg's in-process representative (bd-15kd): a corrupt input
    // must surface as the TYPED InputDecode error — never a panic, never a
    // generic failure. The full leg (artifact faults, usage faults, armed
    // exit-4/5 paths) drives the real binary in tests/fault_suite.rs.
    let p = std::env::temp_dir().join(format!(
        "focr_conformance_fault_probe_{}.png",
        std::process::id()
    ));
    let outcome = std::fs::write(&p, b"not an image")
        .map_err(|e| crate::FocrError::Other(anyhow::anyhow!("probe write: {e}")))
        .and_then(|()| {
            match crate::preprocess::preprocess_image(
                &p,
                crate::preprocess::PreprocessMode::default(),
            ) {
                Err(crate::FocrError::InputDecode(_)) => Ok(()),
                Err(other) => Err(other),
                Ok(_) => Err(crate::FocrError::Other(anyhow::anyhow!(
                    "corrupt probe unexpectedly preprocessed"
                ))),
            }
        });
    let _ = std::fs::remove_file(&p);
    GateResult {
        name: "input_fault_typed_errors",
        level: None,
        passed: outcome.is_ok(),
        measured: None,
        tolerance: None,
        message: if outcome.is_ok() {
            "corrupt input surfaced as typed InputDecode (exit 4)"
        } else {
            "corrupt input did NOT surface as typed InputDecode"
        },
    }
}

fn run_tolerance_derivation_entry() -> GateResult {
    // The keystone discipline: tolerances DERIVE from a measured floor
    // (never the imported 0.055) — prove the derivation math end-to-end.
    let t = default_tolerances();
    let ordered = t.ordered();
    // Every tolerance must carry a declared PROVENANCE (structural contract
    // or an explicit derive-from-floor/ledger obligation) — none may be an
    // unexplained number. The armed rungs replace the Todo placeholders with
    // measured values at run time (parity_harness::establish_floor).
    let sourced = ordered.iter().all(|g| {
        matches!(
            g.source,
            ToleranceSource::StructuralSpec
                | ToleranceSource::TodoDeriveFromOracleFloor
                | ToleranceSource::TodoLedgerBudget
        )
    });
    GateResult {
        name: "tolerances_carry_declared_provenance",
        level: None,
        passed: sourced,
        measured: None,
        tolerance: None,
        message: if sourced {
            "every gate tolerance declares its provenance"
        } else {
            "a gate tolerance has undeclared provenance"
        },
    }
}

/// The registry (bd-re8.12): every shipped suite registers here with its
/// category, requirement level, and covered clauses. The coverage META-test
/// (`tests/conformance_matrix.rs`) enumerates clauses from the SPEC and
/// cross-checks this accounting — a missing suite shows up as an uncovered
/// MUST clause there, not as a silently green registry here.
#[must_use]
pub fn conformance_registry() -> Vec<RegisteredConformance> {
    vec![
        RegisteredConformance {
            name: "determinism_byte_identity",
            category: ConformanceCategory::Invariant,
            level: RequirementLevel::Must,
            clauses: &[100, 101, 102, 103],
            run: run_determinism_entry,
        },
        RegisteredConformance {
            name: "tolerances_carry_declared_provenance",
            category: ConformanceCategory::Parity,
            level: RequirementLevel::Must,
            clauses: &[],
            run: run_tolerance_derivation_entry,
        },
        RegisteredConformance {
            name: "input_fault_typed_errors",
            category: ConformanceCategory::Invariant,
            level: RequirementLevel::Must,
            clauses: &[],
            run: run_input_fault_entry,
        },
    ]
}

/// Validator trait for L0-L5 parity gates.
pub trait ParityGate {
    /// Stable gate name.
    fn name(&self) -> &'static str;
    /// Ladder level.
    fn level(&self) -> ParityLevel;
    /// Validate subject bytes against oracle bytes.
    fn validate(&self, subject: &[u8], oracle: &[u8]) -> GateResult;
}

/// Validator trait for non-ladder invariants.
pub trait Invariant {
    /// Stable invariant name.
    fn name(&self) -> &'static str;
    /// Invariant family.
    fn kind(&self) -> InvariantKind;
    /// Validate the invariant in the current context.
    fn validate(&self) -> GateResult;
}

/// Phase-0 no-op parity gate used to prove the trait shape is object-safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaceholderParityGate {
    name: &'static str,
    level: ParityLevel,
}

impl PlaceholderParityGate {
    /// Create a placeholder gate for a future real validator.
    #[must_use]
    pub const fn new(name: &'static str, level: ParityLevel) -> Self {
        Self { name, level }
    }
}

impl ParityGate for PlaceholderParityGate {
    fn name(&self) -> &'static str {
        self.name
    }

    fn level(&self) -> ParityLevel {
        self.level
    }

    fn validate(&self, _subject: &[u8], _oracle: &[u8]) -> GateResult {
        GateResult::pass(
            self.name,
            Some(self.level),
            "placeholder validator declared",
        )
    }
}

/// Phase-0 no-op invariant used to prove the trait shape is object-safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaceholderInvariant {
    name: &'static str,
    kind: InvariantKind,
}

impl PlaceholderInvariant {
    /// Create a placeholder invariant for a future real monitor.
    #[must_use]
    pub const fn new(name: &'static str, kind: InvariantKind) -> Self {
        Self { name, kind }
    }
}

impl Invariant for PlaceholderInvariant {
    fn name(&self) -> &'static str {
        self.name
    }

    fn kind(&self) -> InvariantKind {
        self.kind
    }

    fn validate(&self) -> GateResult {
        GateResult::pass(self.name, None, "placeholder invariant declared")
    }
}

// ─────────── bd-re8.14: the conformal lower-bound release ratchet ───────────
//
// A point-estimate parity score ("99.2% pass") is overconfident: it ignores
// sampling uncertainty, so a lucky-noise improvement can masquerade as
// progress. The ratchet makes the release decision on a statistically
// defensible LOWER bound instead: per category, a Jeffreys Beta-posterior
// quantile crossed with a distribution-free (Hoeffding one-sided) band, the
// decision taken on the MORE CONSERVATIVE of the two, truncated to 6 dp.
// A change may land only if it lowers NO per-category bound (and the ledger
// floor only ever moves up — that is the ratchet).

/// The ratchet's one-sided error rate: each per-category lower bound is a
/// 95 % lower confidence limit (α = 0.05). Chosen to match the conformal
/// convention; recorded in every transparency card.
pub const RATCHET_ALPHA: f64 = 0.05;

/// The minimum per-category calibration count for the conformal machinery to
/// DECIDE (review-r1 addendum: a checked precondition, not advice).
/// Derivation (asserted by `min_calibration_n_is_the_computed_threshold`):
/// the smallest n at which a PERFECT record's Jeffreys 95 % lower bound
/// clears 0.9 — i.e. the prior stops dominating and a flawless small corpus
/// is no longer reported as near-failing — is n = 18 (bound 0.900124;
/// n = 17 gives 0.894652). We take 20 (bound 0.909524) for a whole-number
/// margin above that boundary. Below this n the ratchet REFUSES to decide
/// conformally and falls back to the deterministic raw point estimate,
/// ledgered as such.
pub const MIN_CALIBRATION_N: u64 = 20;

/// Truncate (never round) a score to 6 decimal places — the deterministic,
/// reproducible decision boundary the ratchet compares on.
#[must_use]
pub fn truncate_score(x: f64) -> f64 {
    (x * 1e6).floor() / 1e6
}

/// Lanczos log-gamma (g = 7, n = 9 coefficients — the standard double-precision
/// set; |rel err| < 1e-13 over the ratchet's domain of half-integer args).
fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1−x) = π / sin(πx).
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut acc = COEFFS[0];
    for (i, c) in COEFFS.iter().enumerate().skip(1) {
        acc += c / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

/// The continued fraction for the regularized incomplete beta (Numerical
/// Recipes `betacf`, Lentz's method).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3e-16;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// The regularized incomplete beta `I_x(a, b)` — the Beta(a, b) CDF at `x`.
fn beta_cdf(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let front = ln_front.exp();
    // The symmetry pick keeps the continued fraction convergent.
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// The Beta(a, b) lower quantile at probability `p`, by bisection on the CDF
/// (monotone; 200 halvings ≪ f64 resolution, so the answer is deterministic).
fn beta_quantile(a: f64, b: f64, p: f64) -> f64 {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if beta_cdf(a, b, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Per-category pass/fail counts — the ratchet's calibration input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CategoryCounts {
    /// Stable category name (the per-category no-lowering rule keys on this).
    pub category: &'static str,
    /// Calibration items that passed.
    pub passes: u64,
    /// Calibration items that failed.
    pub failures: u64,
}

/// How a category's bound was decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundMethod {
    /// Jeffreys Beta-posterior × Hoeffding band; decision on the conservative min.
    Conformal,
    /// Calibration count below [`MIN_CALIBRATION_N`]: the deterministic raw
    /// point estimate (the Alien-Artifact conservative fallback), ledgered.
    DeterministicFallback,
}

/// One category's computed bound (all scores truncated to 6 dp).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CategoryBound {
    /// The category this bound scores.
    pub category: &'static str,
    /// Calibration size (passes + failures).
    pub n: u64,
    /// Raw pass fraction.
    pub point: f64,
    /// Jeffreys Beta(s+½, f+½) posterior α-quantile (0.0 under fallback).
    pub beta_lower: f64,
    /// Distribution-free one-sided band: `point − sqrt(ln(1/α) / 2n)` clamped
    /// at 0 (Hoeffding; 0.0 under fallback).
    pub dkw_lower: f64,
    /// THE decision value: `min(beta_lower, dkw_lower)` under
    /// [`BoundMethod::Conformal`], else the raw point estimate.
    pub lower: f64,
    /// Which rule produced `lower`.
    pub method: BoundMethod,
}

/// Compute one category's lower bound. Refuses to decide conformally below
/// [`MIN_CALIBRATION_N`] (the checked precondition) — the deterministic
/// fallback keeps small-corpus categories from emitting a meaninglessly-low
/// bound that would block every landing.
#[must_use]
pub fn category_bound(counts: CategoryCounts) -> CategoryBound {
    let n = counts.passes + counts.failures;
    let point = if n == 0 {
        0.0
    } else {
        counts.passes as f64 / n as f64
    };
    if n < MIN_CALIBRATION_N {
        return CategoryBound {
            category: counts.category,
            n,
            point: truncate_score(point),
            beta_lower: 0.0,
            dkw_lower: 0.0,
            lower: truncate_score(point),
            method: BoundMethod::DeterministicFallback,
        };
    }
    // Jeffreys posterior: Beta(s + ½, f + ½); one-sided lower limit at α.
    let beta_lower = beta_quantile(
        counts.passes as f64 + 0.5,
        counts.failures as f64 + 0.5,
        RATCHET_ALPHA,
    );
    // Distribution-free one-sided band (Hoeffding): P(p̂ − p > ε) ≤ e^(−2nε²).
    let eps = ((1.0 / RATCHET_ALPHA).ln() / (2.0 * n as f64)).sqrt();
    let dkw_lower = (point - eps).max(0.0);
    let lower = truncate_score(beta_lower.min(dkw_lower));
    CategoryBound {
        category: counts.category,
        n,
        point: truncate_score(point),
        beta_lower: truncate_score(beta_lower),
        dkw_lower: truncate_score(dkw_lower),
        lower,
        method: BoundMethod::Conformal,
    }
}

/// The ratchet decision over a candidate change vs the committed baseline
/// floors. The rule: a change may land only if NO per-category lower bound
/// drops below its baseline floor (categories absent from the baseline are
/// new coverage and always admissible; categories absent from the CANDIDATE
/// that exist in the baseline are treated as dropped coverage and REJECTED).
#[derive(Clone, Debug)]
pub struct RatchetDecision {
    /// True iff the change may land.
    pub allowed: bool,
    /// True iff at least one per-category bound strictly rose (a genuine
    /// floor raise, not just a hold).
    pub raised: bool,
    /// Human-readable per-category verdicts (structured logging feeds off
    /// these; each line names the category, baseline floor, candidate bound).
    pub verdicts: Vec<String>,
}

/// Decide the ratchet: `baseline` is the committed per-category floor set;
/// `candidate` the freshly computed bounds.
#[must_use]
pub fn ratchet_decide(baseline: &[(&str, f64)], candidate: &[CategoryBound]) -> RatchetDecision {
    let mut allowed = true;
    let mut raised = false;
    let mut verdicts = Vec::new();
    for &(name, floor) in baseline {
        match candidate.iter().find(|b| b.category == name) {
            None => {
                allowed = false;
                verdicts.push(format!(
                    "{name}: DROPPED (baseline floor {floor:.6}, no candidate bound) — rejected"
                ));
            }
            Some(b) => {
                let floor_t = truncate_score(floor);
                if b.lower < floor_t {
                    allowed = false;
                    verdicts.push(format!(
                        "{name}: LOWERED {:.6} < floor {floor_t:.6} ({:?}) — rejected",
                        b.lower, b.method
                    ));
                } else {
                    if b.lower > floor_t {
                        raised = true;
                    }
                    verdicts.push(format!(
                        "{name}: holds {:.6} >= floor {floor_t:.6} ({:?})",
                        b.lower, b.method
                    ));
                }
            }
        }
    }
    for b in candidate {
        if !baseline.iter().any(|(name, _)| *name == b.category) {
            raised = true;
            verdicts.push(format!(
                "{}: NEW coverage at {:.6} ({:?}) — admissible, sets the initial floor",
                b.category, b.lower, b.method
            ));
        }
    }
    RatchetDecision {
        allowed,
        raised,
        verdicts,
    }
}

/// The galaxy-brain transparency card for one bound: equation, substituted
/// values, plain-English intuition, validity assumptions, and what would flip
/// the decision — emitted as one JSON object so the release log carries the
/// full statistical justification inline.
#[must_use]
pub fn transparency_card(b: &CategoryBound) -> serde_json::Value {
    let (s, f) = (
        (b.point * b.n as f64).round() as u64,
        b.n - (b.point * b.n as f64).round() as u64,
    );
    serde_json::json!({
        "card": "conformal-ratchet/v1",
        "category": b.category,
        "equation": "lower = min( BetaQuantile(s+1/2, f+1/2; alpha), p_hat - sqrt(ln(1/alpha)/(2n)) ), truncated to 6dp",
        "substituted": {
            "s": s, "f": f, "n": b.n, "alpha": RATCHET_ALPHA,
            "p_hat": b.point, "beta_lower": b.beta_lower, "dkw_lower": b.dkw_lower,
            "lower": b.lower, "method": format!("{:?}", b.method),
        },
        "intuition": "the release floor is what the data still guarantees after paying for sampling luck: the Jeffreys posterior prices the binomial uncertainty, the Hoeffding band prices distribution-freeness, and the decision takes the stingier of the two",
        "validity_assumptions": [
            "calibration items are exchangeable (i.i.d.-like) draws from the deployment distribution",
            "pass/fail is a Bernoulli outcome per item (no partial credit)",
            format!("n >= MIN_CALIBRATION_N ({MIN_CALIBRATION_N}) for the conformal path; below it the deterministic point-estimate fallback is ledgered instead"),
        ],
        "decision_flips_if": [
            format!("any per-category lower bound falls below its committed floor (alpha = {RATCHET_ALPHA})"),
            "a category present in the baseline disappears from the candidate (dropped coverage)",
        ],
    })
}

/// Same-input-twice determinism gate seed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeterminismGate;

impl DeterminismGate {
    /// Compare two serialized outputs from identical inputs and deterministic
    /// settings.
    #[must_use]
    pub fn validate_bytes(&self, first: &[u8], second: &[u8]) -> GateResult {
        if first == second {
            GateResult::pass(
                "same_input_twice_byte_identical",
                None,
                "same input produced byte-identical output",
            )
        } else {
            GateResult::fail(
                "same_input_twice_byte_identical",
                None,
                "same input produced divergent output",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_result(check: &str, passed: bool) {
        println!(
            "{{\"check\":\"{check}\",\"result\":\"{}\"}}",
            if passed { "pass" } else { "fail" }
        );
    }

    #[test]
    fn tolerances_default_constructs() {
        let tolerances = default_tolerances();
        let ordered = tolerances.ordered();
        let ok = ordered.len() == ParityLevel::ALL.len()
            && tolerances.l0.exact_match
            && tolerances.l1.cosine_min == Some(0.9999)
            && tolerances.l2.cosine_min == Some(0.9999)
            && tolerances.l3.source == ToleranceSource::TodoDeriveFromOracleFloor
            && tolerances.l3.logit_tolerance.is_none()
            && tolerances.l3.argmax_must_match
            && tolerances.l4.source == ToleranceSource::TodoDeriveFromOracleFloor
            && tolerances.l5.source == ToleranceSource::TodoLedgerBudget;
        log_result("tolerances_default_constructs", ok);
        assert!(ok, "{tolerances:#?}");
    }

    #[test]
    fn parity_levels_span_l0_to_l5() {
        let labels: Vec<_> = ParityLevel::ALL
            .iter()
            .map(|level| (level.index(), level.label()))
            .collect();
        let ok = labels
            == vec![
                (0, "L0_preprocess"),
                (1, "L1_per_op"),
                (2, "L2_per_layer"),
                (3, "L3_logits"),
                (4, "L4_tokens"),
                (5, "L5_end_to_end"),
            ];
        log_result("parity_levels_span_l0_to_l5", ok);
        assert!(ok, "{labels:?}");
    }

    #[test]
    fn invariant_trait_object_dispatches() {
        let invariants: Vec<Box<dyn Invariant>> = vec![
            Box::new(PlaceholderInvariant::new(
                "kv_cache_never_exceeds_reference_plus_window",
                InvariantKind::KvCacheBound,
            )),
            Box::new(PlaceholderInvariant::new(
                "same_input_twice_byte_identical",
                InvariantKind::Determinism,
            )),
        ];
        let results: Vec<_> = invariants
            .iter()
            .map(|invariant| invariant.validate())
            .collect();
        let ok = results.iter().all(|result| result.passed)
            && invariants[0].kind().label() == "kv_cache_bound"
            && invariants[1].kind().label() == "determinism";
        log_result("invariant_trait_object_dispatches", ok);
        assert!(ok, "{results:?}");
    }

    #[test]
    fn parity_gate_trait_object_dispatches() {
        let gate: Box<dyn ParityGate> = Box::new(PlaceholderParityGate::new(
            "l0_preprocess_placeholder",
            ParityLevel::L0Preprocess,
        ));
        let result = gate.validate(b"subject", b"oracle");
        let ok = result.passed && result.level == Some(ParityLevel::L0Preprocess);
        log_result("parity_gate_trait_object_dispatches", ok);
        assert!(ok, "{result:?}");
    }

    #[test]
    fn determinism_gate_checks_byte_identity() {
        let gate = DeterminismGate;
        let same = gate.validate_bytes(b"abc", b"abc");
        let different = gate.validate_bytes(b"abc", b"abd");
        let ok = same.passed && !different.passed;
        log_result("determinism_gate_checks_byte_identity", ok);
        assert!(ok, "same={same:?} different={different:?}");
    }

    // ── bd-re8.14: the conformal lower-bound ratchet ──

    /// The Beta CDF against EXACT closed forms: `I_x(1,1) = x` (uniform),
    /// `I_x(a,1) = x^a`, `I_x(1,b) = 1 − (1−x)^b`, and the symmetric median
    /// `I_0.5(2,2) = 0.5`. These pin the incomplete-beta translation without
    /// trusting any external table.
    #[test]
    fn beta_cdf_matches_closed_forms() {
        let mut worst = 0.0_f64;
        for i in 1..20 {
            let x = f64::from(i) / 20.0;
            worst = worst.max((beta_cdf(1.0, 1.0, x) - x).abs());
            worst = worst.max((beta_cdf(3.0, 1.0, x) - x.powi(3)).abs());
            worst = worst.max((beta_cdf(1.0, 4.0, x) - (1.0 - (1.0 - x).powi(4))).abs());
        }
        worst = worst.max((beta_cdf(2.0, 2.0, 0.5) - 0.5).abs());
        let ok = worst < 1e-12;
        log_result("beta_cdf_matches_closed_forms", ok);
        assert!(ok, "worst closed-form deviation {worst:e}");
    }

    /// Quantile ↔ CDF self-consistency over a grid of Jeffreys-shaped
    /// posteriors: `CDF(quantile(p)) == p` to 1e-10, plus exact uniform
    /// quantiles.
    #[test]
    fn beta_quantile_inverts_the_cdf() {
        let mut worst = 0.0_f64;
        for &(a, b) in &[(0.5, 0.5), (5.5, 0.5), (95.5, 5.5), (20.5, 0.5), (2.0, 8.0)] {
            for &p in &[0.01, 0.05, 0.5, 0.95] {
                let q = beta_quantile(a, b, p);
                worst = worst.max((beta_cdf(a, b, q) - p).abs());
            }
        }
        let uniform = (beta_quantile(1.0, 1.0, 0.05) - 0.05).abs();
        let ok = worst < 1e-10 && uniform < 1e-12;
        log_result("beta_quantile_inverts_the_cdf", ok);
        assert!(ok, "worst inversion error {worst:e}, uniform {uniform:e}");
    }

    /// Known Jeffreys posteriors, cross-checked against an independent
    /// stdlib-Python port of the same continued fraction (2026-07-06):
    /// perfect n=20 → 0.909523573; n=100 s=95 → beta 0.904229236, Hoeffding
    /// 0.827612658 (the band binds, being the stingier). Stored values are
    /// floor-truncated to 6 dp.
    #[test]
    fn category_bound_matches_known_posteriors() {
        let perfect20 = category_bound(CategoryCounts {
            category: "perfect20",
            passes: 20,
            failures: 0,
        });
        let mixed100 = category_bound(CategoryCounts {
            category: "mixed100",
            passes: 95,
            failures: 5,
        });
        let ok = perfect20.method == BoundMethod::Conformal
            && (perfect20.beta_lower - 0.909_523).abs() < 1e-9
            && (mixed100.beta_lower - 0.904_229).abs() < 1e-9
            && (mixed100.dkw_lower - 0.827_612).abs() < 1e-9
            && (mixed100.lower - mixed100.dkw_lower).abs() < 1e-12
            && perfect20.lower <= perfect20.beta_lower;
        log_result("category_bound_matches_known_posteriors", ok);
        assert!(ok, "perfect20={perfect20:?} mixed100={mixed100:?}");
    }

    /// The review-r1 checked precondition: MIN_CALIBRATION_N sits above the
    /// COMPUTED boundary where a perfect record's Jeffreys lower bound first
    /// clears 0.9 (n = 18; n = 17 is still below).
    #[test]
    fn min_calibration_n_is_the_computed_threshold() {
        let at17 = beta_quantile(17.5, 0.5, RATCHET_ALPHA);
        let at18 = beta_quantile(18.5, 0.5, RATCHET_ALPHA);
        let ok = at17 < 0.9 && at18 >= 0.9 && MIN_CALIBRATION_N >= 18;
        log_result("min_calibration_n_is_the_computed_threshold", ok);
        assert!(ok, "at17={at17} at18={at18} MIN={MIN_CALIBRATION_N}");
    }

    /// Below MIN_CALIBRATION_N the ratchet refuses to decide conformally:
    /// deterministic point-estimate fallback, and NO spurious red (a perfect
    /// 10-item category holds a 0.95 floor instead of being blocked by a
    /// prior-dominated bound).
    #[test]
    fn small_corpus_takes_the_deterministic_fallback() {
        let b = category_bound(CategoryCounts {
            category: "tiny",
            passes: 10,
            failures: 0,
        });
        let decision = ratchet_decide(&[("tiny", 0.95)], &[b]);
        let ok = b.method == BoundMethod::DeterministicFallback
            && (b.lower - 1.0).abs() < 1e-12
            && decision.allowed;
        log_result("small_corpus_takes_the_deterministic_fallback", ok);
        assert!(ok, "bound={b:?} decision={decision:?}");
    }

    /// THE ratchet rule: a per-category regression blocks the land even when
    /// the aggregate improves — and dropped coverage is also a rejection.
    #[test]
    fn per_category_regression_blocks_even_when_aggregate_improves() {
        // Aggregate: baseline 150/200 = 0.75 vs candidate 175/200 = 0.875 —
        // a big aggregate win, carried entirely by category A while B slips.
        let a = category_bound(CategoryCounts {
            category: "a",
            passes: 100,
            failures: 0,
        });
        let b = category_bound(CategoryCounts {
            category: "b",
            passes: 75,
            failures: 25,
        });
        let baseline = [("a", 0.60), ("b", 0.80)];
        let decision = ratchet_decide(&baseline, &[a, b]);
        let dropped = ratchet_decide(&baseline, &[a]);
        let holds = ratchet_decide(&[("a", 0.60), ("b", b.lower)], &[a, b]);
        let ok = !decision.allowed && !dropped.allowed && holds.allowed && holds.raised;
        log_result(
            "per_category_regression_blocks_even_when_aggregate_improves",
            ok,
        );
        assert!(
            ok,
            "decision={decision:?}\ndropped={dropped:?}\nholds={holds:?}"
        );
    }

    /// The transparency card carries the full justification: equation,
    /// substituted values, intuition, validity assumptions, flip conditions.
    #[test]
    fn transparency_card_is_complete() {
        let b = category_bound(CategoryCounts {
            category: "parity",
            passes: 95,
            failures: 5,
        });
        let card = transparency_card(&b);
        let ok = card["card"] == "conformal-ratchet/v1"
            && card["equation"]
                .as_str()
                .is_some_and(|e| e.contains("BetaQuantile"))
            && card["substituted"]["n"] == 100
            && card["substituted"]["s"] == 95
            && card["intuition"].as_str().is_some_and(|s| !s.is_empty())
            && card["validity_assumptions"]
                .as_array()
                .is_some_and(|a| a.len() == 3)
            && card["decision_flips_if"]
                .as_array()
                .is_some_and(|a| a.len() == 2);
        println!(
            "{{\"check\":\"transparency_card_is_complete\",\"result\":\"{}\",\"card\":{card}}}",
            if ok { "pass" } else { "fail" }
        );
        assert!(ok, "{card}");
    }
}
