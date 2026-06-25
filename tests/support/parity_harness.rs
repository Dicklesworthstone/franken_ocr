//! Shared parity-ladder comparator infrastructure (pure infra — compiles and
//! self-tests **without** the 6.67 GB weights or any oracle fixture).
//!
//! This module is the comparator kernel the L0–L5 ladder
//! (`tests/parity_ladder.rs`) and the differential / golden legs reuse. Its
//! design-of-record is `docs/conformance/PARITY_LADDER.md` (§2 the
//! nondeterminism floor, §3 the per-rung tolerances) and
//! `docs/gauntlet/METHODOLOGY.md` §1.3 (the per-op ULP table). It deliberately
//! depends on **nothing** in `src/` that is mid-flux: it works on plain
//! `&[f32]` / `&[u8]` and `serde_json::Value`, so the comparator MATH is
//! unit-tested inline with synthetic vectors while the oracle RUNS stay gated.
//!
//! What lives here (the task contract):
//!   * [`NormalizedValue`] / [`TensorSpec`] — shape+dtype normalization so a
//!     shape/dtype mismatch is caught before the numeric compare runs (the
//!     `TensorSpec`-normalized comparison of METHODOLOGY §1.3).
//!   * [`cosine`] — cosine-similarity comparator (the L1/L2 ≥ 0.9999 gate).
//!   * [`ulp_table`] / [`OpFamily`] / [`ulp_compare`] — the per-op ULP-tolerance
//!     table (4 ULP f32 matmul, 2 ULP elementwise, documented per family).
//!   * [`scrub_volatile`] — non-determinism scrubbers for robot/JSON artifacts.
//!   * [`FixtureLoader`] — reads `scripts/gen_reference_fixtures.py` JSON output
//!     (the `<doc>_reference.json` golden + the `.npy` activation manifest) and
//!     reports presence so a rung can skip-with-SUCCESS when fixtures are absent.
//!   * [`OracleFloor`] / [`establish_floor`] — the establish-the-oracle-
//!     nondeterminism-floor helper: compare two oracle runs BEFORE setting
//!     tolerances (PARITY_LADDER §2; the keystone gate `bd-re8.2`).
//!   * [`Logger`] — structured NDJSON emission conforming to the frozen
//!     `tests/fixtures/test_log_schema.json` contract (detailed-logging is a
//!     first-class requirement).
//!
//! Hand-rolled, no dev-deps (per the harness constraints): the golden-diff loop
//! is `UPDATE_GOLDENS`-driven with `*.actual` on mismatch, JSON shape assertions
//! are manual `serde_json` walks, and the `.npy` reader is a minimal byte-scanner
//! for the exact subset the oracle emits (little-endian `<f4`, C-order). The dev
//! deps we WANTED but hand-rolled instead are recorded in the RET `deps_wanted`.
//!
//! NOTE: not every item is consumed by `parity_ladder.rs` yet (some rungs are
//! still fixture-gated stubs); `#![allow(dead_code)]` keeps the shared infra
//! whole without per-item churn as the ladder fills in.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

// ─────────────────────────────────────────────────────────────────────────────
// 0. Provenance constants (line-backed to the pinned oracle — see
//    docs/truth-pack/PINNED_SOURCES.md and gen_reference_fixtures.py).
//    Every measured parity result must be traceable to THIS model version
//    (PARITY_LADDER §8). A fixture whose provenance does not resolve to these is
//    incomplete and may NOT be recorded as a pass.
// ─────────────────────────────────────────────────────────────────────────────

/// The pinned Hugging Face model commit the whole truth-pack (and every fixture)
/// is measured against. Verified 2026-06-25.
pub const HF_COMMIT: &str = "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5";
/// The pinned oracle runtime stack (a fixture from any other stack is not
/// comparable — `gen_reference_fixtures.py` asserts this at generation time).
pub const PIN_TORCH: &str = "2.10.0";
/// Pinned transformers version.
pub const PIN_TRANSFORMERS: &str = "4.57.1";

// ─────────────────────────────────────────────────────────────────────────────
// 1. TensorSpec / NormalizedValue — shape+dtype normalization.
//    METHODOLOGY §1.3: normalize both sides BEFORE the numeric compare so a
//    shape or dtype mismatch is caught first, never hidden inside a fuzzy cosine.
// ─────────────────────────────────────────────────────────────────────────────

/// The dtype of a value as it crosses the comparator boundary. The oracle dumps
/// bf16 upcast to f32 (`ActivationCapture._to_numpy`), so f32 is the wire dtype;
/// the quantized subject path additionally produces i8/i4 we compare against the
/// MEASURED budget (not the ULP table — METHODOLOGY §1.3 non-default #1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    /// IEEE-754 single — the oracle wire dtype and the f32 reference forward.
    F32,
    /// int8 — the quantized decoder forward (compared within the measured budget).
    I8,
    /// int4 — the int4-group experts (compared within the small ledgered budget).
    I4,
    /// raw bytes (preprocessed tensor / image-token id stream — L0 EXACT).
    U8,
}

impl DType {
    /// Human label for the structured log (matches the `dtype` enum in
    /// `tests/fixtures/test_log_schema.json`).
    pub fn label(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::I8 => "i8",
            DType::I4 => "i4",
            DType::U8 => "u8",
        }
    }
}

/// Shape + dtype of a tensor, used to reject a mismatch before the numeric
/// compare. The `data_len` is the flattened element count (the BLAKE3 data hash
/// of METHODOLOGY §1.3 is hand-rolled away — we compare values directly here and
/// record the SHA-256 of the *fixture file* in provenance instead, the
/// equivalent guard without a hashing dep).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    /// Row-major (C-order) shape, outermost dim first — the oracle `.npy` layout.
    pub shape: Vec<usize>,
    /// Element dtype crossing the comparator.
    pub dtype: DType,
}

impl TensorSpec {
    /// Construct a spec; `shape` is C-order.
    pub fn new(shape: impl Into<Vec<usize>>, dtype: DType) -> Self {
        Self {
            shape: shape.into(),
            dtype,
        }
    }

    /// Flattened element count (product of the shape; empty shape ⇒ scalar = 1).
    pub fn numel(&self) -> usize {
        self.shape
            .iter()
            .product::<usize>()
            .max(usize::from(self.shape.is_empty()))
    }

    /// Reject a shape/dtype mismatch with a self-diagnosing message naming BOTH
    /// sides (the mismatched-field-printed discipline). Returns `Ok(())` only on
    /// an exact spec match.
    pub fn check_against(&self, other: &TensorSpec) -> Result<(), String> {
        if self.dtype != other.dtype {
            return Err(format!(
                "dtype mismatch: subject={:?} oracle={:?}",
                self.dtype, other.dtype
            ));
        }
        if self.shape != other.shape {
            return Err(format!(
                "shape mismatch: subject={:?} (numel {}) oracle={:?} (numel {})",
                self.shape,
                self.numel(),
                other.shape,
                other.numel()
            ));
        }
        Ok(())
    }
}

/// A normalized comparator value: the spec plus its flat f32 view. Both sides are
/// normalized to this before any numeric compare so shape/dtype are checked once,
/// at one chokepoint, for every rung.
#[derive(Clone, Debug)]
pub struct NormalizedValue {
    /// Shape + dtype.
    pub spec: TensorSpec,
    /// Row-major flat data (oracle `.npy` is C-order; quantized paths dequant to
    /// f32 for the cosine/ULP compare, with the integer budget applied separately).
    pub data: Vec<f32>,
}

impl NormalizedValue {
    /// Build from a flat f32 slice and an explicit spec, asserting the element
    /// count matches the spec (a programmer error otherwise).
    pub fn from_f32(spec: TensorSpec, data: Vec<f32>) -> Self {
        assert_eq!(
            spec.numel(),
            data.len(),
            "NormalizedValue: spec numel {} != data len {}",
            spec.numel(),
            data.len()
        );
        Self { spec, data }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Cosine-similarity comparator (the L1/L2 ≥ 0.9999 gate).
// ─────────────────────────────────────────────────────────────────────────────

/// The L1/L2 f32 cosine-parity threshold (PARITY_LADDER §3, line 160/161:
/// "cosine ≥ 0.9999 (f32)"). A continuous activation is held to this, not to
/// bit-exactness, because f32-vs-bf16 is a continuous-value divergence.
pub const COSINE_F32_THRESHOLD: f64 = 0.9999;

/// Cosine similarity between two equal-length vectors, computed in f64 so the
/// accumulation order does not perturb the metric itself (a comparator must be
/// more stable than the thing it measures). Returns a value in `[-1, 1]`.
///
/// Edge cases (documented, not panicking): a zero-norm vector vs a zero-norm
/// vector is treated as identical (`1.0`) — two all-zero activations agree; a
/// zero-norm vs a non-zero vector is maximally dissimilar (`0.0`).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "cosine: length mismatch {} != {}",
        a.len(),
        b.len()
    );
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Max absolute element-wise difference (the per-layer max-abs ledger of
/// PARITY_LADDER §3.2: "max-abs-diff LEDGERED per layer" so slow cross-layer
/// drift is visible even when the final cosine still looks fine).
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "max_abs_diff: length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold(0.0, f64::max)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. The per-op ULP-tolerance table — the L1/L2 comparator (METHODOLOGY §1.3).
//    Units in the last place of IEEE-754 f32. Anchored to torch's own
//    torch.testing.assert_close posture. This is the contract — a tolerance
//    change is a bead, not a knob (the 🎚 Raise-ULP-Tolerance gate).
// ─────────────────────────────────────────────────────────────────────────────

/// The op family a tensor was produced by — selects its ULP budget. The bite
/// sites are line-backed in METHODOLOGY §1.3's table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpFamily {
    /// f32 matmul (SAM/CLIP GEMMs, projector 2048→1280, lm_head 1280→129280).
    MatmulF32,
    /// Elementwise (residual adds, RoPE rotation, masked-scatter fusion).
    Elementwise,
    /// Reductions (RMSNorm variance, MoE router softmax normalizer).
    Reduction,
    /// Transcendentals (softmax exp, SiLU/quick_gelu sigmoid, RoPE sin/cos).
    Transcendental,
}

/// The per-op ULP budget for a family (METHODOLOGY §1.3 defaults, verbatim):
/// matmul 4 ULP, elementwise 2 ULP, reductions 8 ULP, transcendentals 8 ULP.
/// These are the **f32-Subject vs f32-Oracle** tolerances; the int8 forward drift
/// is a SEPARATE measured budget (non-default #1), never a ULP question.
pub fn ulp_table(family: OpFamily) -> u32 {
    match family {
        OpFamily::MatmulF32 => 4,
        OpFamily::Elementwise => 2,
        OpFamily::Reduction => 8,
        OpFamily::Transcendental => 8,
    }
}

/// ULP distance between two finite f32s (the count of representable f32 values
/// between them). Uses the standard monotone-ordinal transform of the IEEE-754
/// bit pattern so that adjacent floats are 1 ULP apart across the sign boundary.
///
/// Non-finite handling (documented): two equal NaNs (same bits) ⇒ 0; a NaN vs
/// anything else ⇒ `u32::MAX` (maximally far — a NaN where the oracle has a
/// finite value is always a failure). `±0.0` are treated as identical (0 ULP).
pub fn ulp_distance(a: f32, b: f32) -> u32 {
    if a.to_bits() == b.to_bits() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    if a == b {
        // +0.0 vs -0.0
        return 0;
    }
    // Map the i32 bit pattern to a monotone ordinal: flip all-but-sign for
    // negatives, set the sign bit for positives. Adjacent floats then differ by 1.
    let ord = |x: f32| -> i64 {
        let bits = x.to_bits() as i32;
        let ordinal = if bits < 0 {
            i32::MIN.wrapping_sub(bits)
        } else {
            bits
        };
        i64::from(ordinal)
    };
    (ord(a) - ord(b)).unsigned_abs() as u32
}

/// Outcome of a per-op ULP comparison: the worst ULP seen, the family budget,
/// the offending flat index, the max-abs-diff, and pass/fail. The offending
/// index is reported so a failure is self-diagnosing (`mismatched-field-printed`).
#[derive(Clone, Debug)]
pub struct UlpReport {
    /// The op family whose budget was applied.
    pub family: OpFamily,
    /// The budget (from [`ulp_table`]).
    pub budget_ulp: u32,
    /// The worst per-element ULP distance observed.
    pub max_ulp: u32,
    /// Flat index of the worst element.
    pub worst_index: usize,
    /// Max absolute difference (the ledgered companion metric).
    pub max_abs_diff: f64,
    /// `true` iff every element is within the family budget.
    pub pass: bool,
}

/// Compare a subject tensor to an oracle tensor through the per-op ULP table.
/// Lengths must match (call after [`TensorSpec::check_against`]). Returns the
/// full report — the caller logs it whether it passes or fails.
pub fn ulp_compare(subject: &[f32], oracle: &[f32], family: OpFamily) -> UlpReport {
    assert_eq!(subject.len(), oracle.len(), "ulp_compare: length mismatch");
    let budget = ulp_table(family);
    let mut max_ulp = 0u32;
    let mut worst_index = 0usize;
    let mut mad = 0.0f64;
    for (i, (&s, &o)) in subject.iter().zip(oracle.iter()).enumerate() {
        let u = ulp_distance(s, o);
        if u > max_ulp {
            max_ulp = u;
            worst_index = i;
        }
        let d = (f64::from(s) - f64::from(o)).abs();
        if d > mad {
            mad = d;
        }
    }
    UlpReport {
        family,
        budget_ulp: budget,
        max_ulp,
        worst_index,
        max_abs_diff: mad,
        pass: max_ulp <= budget,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Non-determinism scrubbers (robot NDJSON / --json artifacts).
//    GOLDEN.md §2D scrub list: replace volatile fields with stable placeholders
//    BEFORE snapshotting; SCRUB, do not delete (a dropped field must still be
//    caught). The scrubbable fields are also named in test_log_schema.json.
// ─────────────────────────────────────────────────────────────────────────────

/// The fields whose *value* is volatile but whose *presence* is contract
/// (GOLDEN.md §2D + `test_log_schema.json` `scrubbable_fields`). Any key ending
/// in `_ms`, `_us`, `_seconds` is additionally scrubbed (timing suffixes).
pub const SCRUB_KEYS: &[&str] = &[
    "ts",
    "elapsed_us",
    "elapsed_ms",
    "elapsed_seconds",
    "run_id",
    "trace_id",
    "started_at",
    "finished_at",
    "finished_utc",
    "generated_utc",
    "timestamp",
    "duration",
];

/// Replace volatile leaves in a JSON value with stable placeholders, recursively.
/// Timing leaves → `"[ms]"`, ids → `"[id]"`, timestamps → `"[ts]"`, absolute
/// paths (string leaves beginning with `/`) → `"[path]"`. The *shape* is
/// preserved so a dropped field is still caught (GOLDEN.md §2D, §5).
pub fn scrub_volatile(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                let scrubbed = if is_scrub_key(k) {
                    placeholder_for(k)
                } else {
                    scrub_volatile(v)
                };
                out.insert(k.clone(), scrubbed);
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(scrub_volatile).collect()),
        Value::String(s) if s.starts_with('/') && s.len() > 1 => Value::String("[path]".into()),
        other => other.clone(),
    }
}

fn is_scrub_key(k: &str) -> bool {
    SCRUB_KEYS.contains(&k)
        || k.ends_with("_ms")
        || k.ends_with("_us")
        || k.ends_with("_seconds")
        || k.ends_with("_utc")
}

fn placeholder_for(k: &str) -> Value {
    if k.contains("id") {
        Value::String("[id]".into())
    } else if k.contains("ts") || k.contains("_at") || k.contains("utc") || k.contains("time") {
        Value::String("[ts]".into())
    } else {
        Value::String("[ms]".into())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Fixture loader — reads gen_reference_fixtures.py JSON output.
//    The oracle dumps two artifact families (PARITY_LADDER §1 table):
//      <out>/<doc>_reference.json     — e2e golden (L4/L5)
//      <out>/activations/<doc>/<stage>.npy + the manifest in the json (L1/L2/L3)
//    The loader reports PRESENCE so a rung can skip-with-SUCCESS when absent.
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve the native-fixtures root. Honors `$FOCR_FIXTURES_DIR` (so a CUDA host
/// that generated them can point here) else the committed
/// `tests/fixtures/native` of GOLDEN.md §1. Relative to `CARGO_MANIFEST_DIR` so
/// the path is correct regardless of the test's CWD.
pub fn fixtures_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("FOCR_FIXTURES_DIR") {
        return PathBuf::from(dir);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/native")
}

/// A loaded end-to-end golden (`<doc>_reference.json`) — the L4/L5 bar.
#[derive(Clone, Debug)]
pub struct ReferenceGolden {
    /// The document name (`golden["doc"]`).
    pub doc: String,
    /// The decoded text (`golden["decoded_text"]`) — may be `None` if the oracle
    /// only wrote `result.md`.
    pub decoded_text: Option<String>,
    /// SHA-256 of the decoded text (`golden["decoded_text_sha256"]`) — the L5
    /// exact-where-deterministic anchor.
    pub decoded_text_sha256: Option<String>,
    /// The per-stage activation manifest (`golden["activations"]`): stage → file
    /// + shape + dtype + sha256.
    pub activations: BTreeMap<String, ActivationEntry>,
    /// The fixture's own provenance block (the artifact-graph contract, §8).
    pub provenance: Value,
    /// The raw parsed value (for ad-hoc field assertions a rung may need).
    pub raw: Value,
}

/// One entry in the activation manifest.
#[derive(Clone, Debug)]
pub struct ActivationEntry {
    /// The `.npy` filename (relative to `activations/<doc>/`).
    pub file: String,
    /// C-order shape.
    pub shape: Vec<usize>,
    /// dtype string as the oracle wrote it (e.g. `"float32"`).
    pub dtype: String,
    /// SHA-256 of the array bytes (for provenance cross-check).
    pub sha256: Option<String>,
}

/// Loads + reports presence of the oracle fixtures under [`fixtures_root`].
pub struct FixtureLoader {
    root: PathBuf,
}

impl FixtureLoader {
    /// Construct against the resolved [`fixtures_root`].
    pub fn new() -> Self {
        Self {
            root: fixtures_root(),
        }
    }

    /// Construct against an explicit root (tests that synthesize a fixture tree).
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The resolved fixtures root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Are ANY end-to-end golden fixtures present? Used by the model/fixture gate
    /// so a rung skips-with-SUCCESS (never a silent fake pass) when the CUDA-host
    /// fixtures have not been generated/committed.
    pub fn any_present(&self) -> bool {
        self.list_goldens().map(|g| !g.is_empty()).unwrap_or(false)
    }

    /// List the `<doc>_reference.json` golden files (sorted, deterministic).
    pub fn list_goldens(&self) -> std::io::Result<Vec<PathBuf>> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let path = entry?.path();
            if path.is_file()
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("_reference.json"))
            {
                out.push(path);
            }
        }
        out.sort();
        Ok(out)
    }

    /// Load + parse one golden JSON into a [`ReferenceGolden`].
    pub fn load_golden(&self, path: &Path) -> Result<ReferenceGolden, String> {
        let text =
            fs::read_to_string(path).map_err(|e| format!("read golden {}: {e}", path.display()))?;
        let raw: Value = serde_json::from_str(&text)
            .map_err(|e| format!("parse golden {}: {e}", path.display()))?;
        Self::golden_from_value(raw)
    }

    /// Parse a [`ReferenceGolden`] out of an already-decoded `Value` (factored so
    /// the parser is unit-testable on a synthetic value without touching disk).
    pub fn golden_from_value(raw: Value) -> Result<ReferenceGolden, String> {
        let doc = raw
            .get("doc")
            .and_then(Value::as_str)
            .ok_or("golden missing `doc`")?
            .to_string();
        let decoded_text = raw
            .get("decoded_text")
            .and_then(Value::as_str)
            .map(str::to_string);
        let decoded_text_sha256 = raw
            .get("decoded_text_sha256")
            .and_then(Value::as_str)
            .map(str::to_string);
        let mut activations = BTreeMap::new();
        if let Some(map) = raw.get("activations").and_then(Value::as_object) {
            for (stage, entry) in map {
                let file = entry
                    .get("file")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let shape = entry
                    .get("shape")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let dtype = entry
                    .get("dtype")
                    .and_then(Value::as_str)
                    .unwrap_or("float32")
                    .to_string();
                let sha256 = entry
                    .get("sha256")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                activations.insert(
                    stage.clone(),
                    ActivationEntry {
                        file,
                        shape,
                        dtype,
                        sha256,
                    },
                );
            }
        }
        let provenance = raw.get("provenance").cloned().unwrap_or(Value::Null);
        Ok(ReferenceGolden {
            doc,
            decoded_text,
            decoded_text_sha256,
            activations,
            provenance,
            raw,
        })
    }

    /// Assert a golden's provenance resolves to the pinned model (§8: a fixture
    /// whose provenance cannot be resolved to a pinned source is incomplete and
    /// may NOT count as a pass). Returns the mismatched field on failure.
    pub fn check_provenance(golden: &ReferenceGolden) -> Result<(), String> {
        let p = &golden.provenance;
        let commit = p.get("hf_commit").and_then(Value::as_str).unwrap_or("");
        if commit != HF_COMMIT {
            return Err(format!("hf_commit {commit:?} != pinned {HF_COMMIT:?}"));
        }
        let torch = p.get("pinned_torch").and_then(Value::as_str).unwrap_or("");
        if torch != PIN_TORCH {
            return Err(format!("pinned_torch {torch:?} != {PIN_TORCH:?}"));
        }
        Ok(())
    }

    /// Load one activation `.npy` for a doc+stage into a [`NormalizedValue`].
    /// The oracle writes C-order little-endian `<f4` arrays (`np.save(..,
    /// astype(np.float32))`), so [`read_npy_f32`] handles exactly that subset.
    pub fn load_activation(&self, doc: &str, stage: &str) -> Result<NormalizedValue, String> {
        let path = self
            .root
            .join("activations")
            .join(doc)
            .join(format!("{stage}.npy"));
        let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let (shape, data) =
            read_npy_f32(&bytes).map_err(|e| format!("parse npy {}: {e}", path.display()))?;
        Ok(NormalizedValue::from_f32(
            TensorSpec::new(shape, DType::F32),
            data,
        ))
    }
}

impl Default for FixtureLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimal `.npy` v1/v2 reader for the exact subset `gen_reference_fixtures.py`
/// emits: little-endian `<f4` (float32), C-contiguous (`fortran_order: False`).
/// Hand-rolled in place of the `ndarray`/`npyz` dev-deps (recorded in
/// `deps_wanted`). Returns `(shape, flat_c_order_data)`. Rejects anything outside
/// that subset with a self-diagnosing error rather than silently misreading.
pub fn read_npy_f32(bytes: &[u8]) -> Result<(Vec<usize>, Vec<f32>), String> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("not a .npy file (bad magic)".into());
    }
    let major = bytes[6];
    // v1.0: u16 header len at [8..10]; v2.0: u32 header len at [8..12].
    let (header_start, header_len) = if major == 1 {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (10usize, len)
    } else {
        if bytes.len() < 12 {
            return Err("truncated v2 header".into());
        }
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (12usize, len)
    };
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err("truncated header".into());
    }
    let header =
        std::str::from_utf8(&bytes[header_start..header_end]).map_err(|_| "non-utf8 npy header")?;
    // The header is a Python-dict literal: {'descr':'<f4','fortran_order':False,'shape':(.. ,)}.
    if !(header.contains("<f4") || header.contains("|f4") || header.contains("'<f4'")) {
        return Err(format!(
            "unsupported descr (need little-endian <f4): {header}"
        ));
    }
    if header.contains("'fortran_order': True") || header.contains("'fortran_order':True") {
        return Err("fortran_order True unsupported (need C-order)".into());
    }
    let shape = parse_npy_shape(header)?;
    let numel: usize = shape
        .iter()
        .product::<usize>()
        .max(usize::from(shape.is_empty()));
    let data_bytes = &bytes[header_end..];
    if data_bytes.len() < numel * 4 {
        return Err(format!(
            "data truncated: have {} bytes, need {} ({} elems × 4)",
            data_bytes.len(),
            numel * 4,
            numel
        ));
    }
    let mut data = Vec::with_capacity(numel);
    for chunk in data_bytes[..numel * 4].chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((shape, data))
}

/// Extract the `shape` tuple from a `.npy` header dict literal.
fn parse_npy_shape(header: &str) -> Result<Vec<usize>, String> {
    let key = "'shape':";
    let idx = header
        .find(key)
        .or_else(|| header.find("'shape' :"))
        .ok_or("npy header missing 'shape'")?;
    let after = &header[idx + key.len()..];
    let open = after.find('(').ok_or("npy shape: no '('")?;
    let close = after[open..].find(')').ok_or("npy shape: no ')'")? + open;
    let inner = &after[open + 1..close];
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        let n: usize = t.parse().map_err(|_| format!("npy shape: bad dim {t:?}"))?;
        dims.push(n);
    }
    Ok(dims)
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. The keystone: establish the oracle's OWN nondeterminism floor FIRST.
//    PARITY_LADDER §2 / bd-re8.2: compare TWO oracle runs (two thread counts)
//    BEFORE setting any L3/L4 tolerance. "A franken_ocr int8 divergence inside
//    the oracle's own bf16 noise is not a bug."
// ─────────────────────────────────────────────────────────────────────────────

/// The measured nondeterminism envelope between two oracle runs (the committed
/// `tests/fixtures/oracle_nondeterminism_envelope.json` of §2). The two
/// tolerances every downstream gate uses are DERIVED from this, never guessed,
/// and never the imported frankensearch `0.055`.
#[derive(Clone, Debug)]
pub struct OracleFloor {
    /// Per-logit max-abs spread at matching positions across the two runs — the
    /// L3 logit tolerance is derived from this (§2 step 3).
    pub per_logit_max_abs_spread: f64,
    /// The longest decoded-token prefix the oracle reproduces IDENTICALLY across
    /// the two runs — the L4 "exact" prefix is defined ONLY over this (§2).
    pub reproducible_prefix_len: usize,
    /// The total token count compared (so the prefix can be read as a fraction).
    pub total_tokens: usize,
    /// Per-token divergence rate across the two runs (§2 step 2 metric).
    pub per_token_divergence_rate: f64,
    /// Position of the first divergence (`None` ⇒ the runs were identical).
    pub first_divergence_pos: Option<usize>,
}

impl OracleFloor {
    /// The derived L3 logit tolerance: a continuous franken_ocr divergence at or
    /// below the oracle's own per-logit spread is inside the noise floor (§2). We
    /// expose the measured spread directly — the L3 gate compares against it, not
    /// a hand-guessed epsilon.
    pub fn l3_logit_tolerance(&self) -> f64 {
        self.per_logit_max_abs_spread
    }

    /// The L4 reproducible-prefix length — bit-exact token comparison is asserted
    /// ONLY over `[0, reproducible_prefix_len)` (§2).
    pub fn l4_exact_prefix(&self) -> usize {
        self.reproducible_prefix_len
    }
}

/// Establish the floor from two oracle runs over the SAME corpus entry: two
/// per-position logit rows (run A, run B) and the two decoded token id streams.
/// This is the in-process realization of the §2 procedure on already-captured
/// run pairs (the runs themselves come from `gen_reference_fixtures.py
/// --run-tag a/b`).
///
/// `logits_a`/`logits_b` are per-position logit rows (outer = position, inner =
/// vocab); `tokens_a`/`tokens_b` are the decoded greedy token streams.
pub fn establish_floor(
    logits_a: &[Vec<f32>],
    logits_b: &[Vec<f32>],
    tokens_a: &[u32],
    tokens_b: &[u32],
) -> OracleFloor {
    // Per-logit max-abs spread at matching positions.
    let mut spread = 0.0f64;
    for (ra, rb) in logits_a.iter().zip(logits_b.iter()) {
        let n = ra.len().min(rb.len());
        for i in 0..n {
            let d = (f64::from(ra[i]) - f64::from(rb[i])).abs();
            if d > spread {
                spread = d;
            }
        }
    }
    // Reproducible prefix + first divergence + divergence rate.
    let total = tokens_a.len().min(tokens_b.len());
    let mut prefix = total;
    let mut first_div = None;
    let mut diverged = 0usize;
    for i in 0..total {
        if tokens_a[i] != tokens_b[i] {
            diverged += 1;
            if first_div.is_none() {
                first_div = Some(i);
                prefix = i;
            }
        }
    }
    // A length mismatch is itself a divergence at the shorter length.
    if tokens_a.len() != tokens_b.len() && first_div.is_none() {
        first_div = Some(total);
        prefix = total;
    }
    let rate = if total == 0 {
        0.0
    } else {
        diverged as f64 / total as f64
    };
    OracleFloor {
        per_logit_max_abs_spread: spread,
        reproducible_prefix_len: prefix,
        total_tokens: total,
        per_token_divergence_rate: rate,
        first_divergence_pos: first_div,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Golden-diff loop (hand-rolled, UPDATE_GOLDENS-driven; GOLDEN.md §4/§5).
//    No insta dep — on mismatch we write `<golden>.actual` next to the committed
//    golden and fail with the diff; UPDATE_GOLDENS=1 rewrites the committed file.
// ─────────────────────────────────────────────────────────────────────────────

/// Is the deliberate golden-update flag set? `UPDATE_GOLDENS=1` is the ONLY way
/// to rewrite a committed golden (GOLDEN.md §4 rule 1). CI never sets it (§4 rule
/// 3) — a contract test in the golden suite asserts that.
pub fn update_goldens() -> bool {
    matches!(
        std::env::var("UPDATE_GOLDENS").ok().as_deref(),
        Some("1") | Some("true")
    )
}

/// Compare `actual` against the committed golden file at `golden_path`.
///   * golden missing & `UPDATE_GOLDENS` set ⇒ write it, return Ok.
///   * golden missing & not set ⇒ write `<golden>.actual`, return Err (so a
///     missing golden is a visible failure, never a silent pass).
///   * mismatch & `UPDATE_GOLDENS` set ⇒ rewrite, return Ok.
///   * mismatch & not set ⇒ write `<golden>.actual`, return Err with a line diff.
///   * match ⇒ remove any stale `.actual`, return Ok.
/// Content is normalized to `\n` line endings (GOLDEN.md §2E canonicalization).
pub fn golden_diff(golden_path: &Path, actual: &str) -> Result<(), String> {
    let actual = actual.replace("\r\n", "\n");
    let actual_path = actual_sidecar(golden_path);
    let existing = fs::read_to_string(golden_path)
        .ok()
        .map(|s| s.replace("\r\n", "\n"));
    match existing {
        Some(ref committed) if committed == &actual => {
            let _ = fs::remove_file(&actual_path);
            Ok(())
        }
        _ if update_goldens() => {
            if let Some(parent) = golden_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            fs::write(golden_path, &actual).map_err(|e| format!("write golden: {e}"))?;
            let _ = fs::remove_file(&actual_path);
            Ok(())
        }
        Some(committed) => {
            let _ = fs::write(&actual_path, &actual);
            Err(format!(
                "golden mismatch: {}\n{}\n(wrote {} — review, then UPDATE_GOLDENS=1 to bless)",
                golden_path.display(),
                first_line_diff(&committed, &actual),
                actual_path.display()
            ))
        }
        None => {
            let _ = fs::write(&actual_path, &actual);
            Err(format!(
                "golden missing: {} (wrote {} — review, then UPDATE_GOLDENS=1 to create)",
                golden_path.display(),
                actual_path.display()
            ))
        }
    }
}

/// The `<golden>.actual` sidecar path (gitignored per GOLDEN.md §5).
pub fn actual_sidecar(golden_path: &Path) -> PathBuf {
    let mut s = golden_path.as_os_str().to_os_string();
    s.push(".actual");
    PathBuf::from(s)
}

/// First differing line, with line number, for a self-diagnosing golden failure.
fn first_line_diff(expected: &str, actual: &str) -> String {
    for (i, (e, a)) in expected.lines().zip(actual.lines()).enumerate() {
        if e != a {
            return format!("  L{}: expected {:?}\n       actual {:?}", i + 1, e, a);
        }
    }
    let (el, al) = (expected.lines().count(), actual.lines().count());
    if el != al {
        return format!("  line count differs: expected {el}, actual {al}");
    }
    "  (no line-level diff — trailing whitespace / final newline?)".into()
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Structured NDJSON logger — conforms to tests/fixtures/test_log_schema.json.
//    Detailed logging is a first-class requirement: every test emits a clear
//    structured line on what it exercised, inputs, expected-vs-actual, and on a
//    model-gated skip a SUCCESS line explaining why it skipped.
// ─────────────────────────────────────────────────────────────────────────────

/// The fixed `/nonexistent` fallback the native-path-proof contract requires:
/// pointing the model fallback here PROVES the native path ran (a silent skip is
/// never mistaken for a pass — `test_log_schema.json` `native_path_proof`).
pub const NATIVE_PATH_FALLBACK: &str = "/nonexistent";

/// The robot schema version the log lines carry (mirrors `ROBOT_SCHEMA_VERSION`
/// in `src/robot.rs` — kept as a literal so this test infra does not depend on
/// the mid-flux lib build; a contract test asserts they agree).
pub const LOG_SCHEMA_VERSION: u64 = 1;

/// A monotonically-increasing sequence so each test's lines are ordered
/// (`run_seq` in the schema). Per-`Logger` so tests stay independent.
pub struct Logger {
    test: String,
    case: String,
    seq: u64,
}

impl Logger {
    /// Begin a structured-log scope for `test` (the rung, e.g. `"L0_preprocess"`)
    /// and `case` (the corpus entry, e.g. `"doc01"` or `"synthetic"`).
    pub fn new(test: &str, case: &str) -> Self {
        Self {
            test: test.to_string(),
            case: case.to_string(),
            seq: 0,
        }
    }

    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    /// Emit one NDJSON line on stderr (so it does not pollute any stdout the
    /// harness captures) with the required common fields filled in.
    fn emit(&mut self, event: &str, result: &str, mut extra: serde_json::Map<String, Value>) {
        let seq = self.next_seq();
        let mut obj = serde_json::Map::new();
        obj.insert("schema_version".into(), json!(LOG_SCHEMA_VERSION));
        obj.insert("ts".into(), json!(now_micros()));
        obj.insert("test".into(), json!(self.test));
        obj.insert("case".into(), json!(self.case));
        obj.insert("run_seq".into(), json!(seq));
        obj.insert("event".into(), json!(event));
        obj.insert("result".into(), json!(result));
        obj.append(&mut extra);
        eprintln!("{}", Value::Object(obj));
    }

    /// `setup` event — records the seed (required-by-event `setup` ⇒ `seed`).
    pub fn setup(&mut self, seed: u64) {
        self.emit("setup", "pass", map(&[("seed", json!(seed))]));
    }

    /// `parity` event — the load-bearing line: which gate, the metric, the
    /// measured value vs tolerance, the oracle fixture it ran against, and the
    /// nondeterminism envelope the tolerance was derived from. Required-by-event
    /// `parity` fields are all supplied (`test_log_schema.json`).
    #[allow(clippy::too_many_arguments)]
    pub fn parity(
        &mut self,
        gate: &str,
        metric: &str,
        value: f64,
        tolerance: f64,
        oracle_fixture: &str,
        oracle_sha256: &str,
        nondeterminism_envelope: Value,
        pass: bool,
    ) {
        self.emit(
            "parity",
            if pass { "pass" } else { "fail" },
            map(&[
                ("gate", json!(gate)),
                ("metric", json!(metric)),
                ("value", json!(value)),
                ("tolerance", json!(tolerance)),
                ("oracle_fixture", json!(oracle_fixture)),
                ("oracle_sha256", json!(oracle_sha256)),
                ("nondeterminism_envelope", nondeterminism_envelope),
                ("pass", json!(pass)),
            ]),
        );
    }

    /// `assert` event — a structural (non-parity) assertion with its description.
    pub fn assertion(&mut self, assertion: &str, pass: bool) {
        self.emit(
            "assert",
            if pass { "pass" } else { "fail" },
            map(&[("assertion", json!(assertion)), ("pass", json!(pass))]),
        );
    }

    /// `skip` event with the `skip_no_model` result — the model/fixture-gated
    /// SUCCESS line. `reason` explains WHY it skipped; `native_path_ran` +
    /// `fallback_target` satisfy the native-path-proof contract so a skip is
    /// never confused for a pass.
    pub fn skip_no_model(&mut self, reason: &str) {
        self.emit(
            "skip",
            "skip_no_model",
            map(&[
                ("reason", json!(reason)),
                ("native_path_ran", json!(false)),
                ("fallback_target", json!(NATIVE_PATH_FALLBACK)),
            ]),
        );
    }

    /// `result` event — the rung's terminal line. `elapsed_us` is required;
    /// scrubbed in goldens.
    pub fn result(&mut self, outcome: &str, elapsed_us: u128) {
        self.emit(
            "result",
            outcome,
            map(&[("elapsed_us", json!(elapsed_us as u64))]),
        );
    }

    /// `error` event — a self-diagnosing failure line carrying the `diag` block
    /// (`error_kind`, `focr_exit_code`, `message` — `required_diag_fields`).
    pub fn error(&mut self, kind: &str, exit_code: i32, message: &str) {
        self.emit(
            "error",
            "fail",
            map(&[(
                "diag",
                json!({
                    "error_kind": kind,
                    "focr_exit_code": exit_code,
                    "message": message,
                }),
            )]),
        );
    }
}

fn map(pairs: &[(&str, Value)]) -> serde_json::Map<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

fn now_micros() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Load the frozen log schema (`tests/fixtures/test_log_schema.json`) so the
/// ladder can self-validate its emitted events against the committed contract
/// (the contract-test-vs-frozen-schema discipline).
pub fn load_log_schema() -> Result<Value, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_log_schema.json");
    let text = fs::read_to_string(&path).map_err(|e| format!("read log schema: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("parse log schema: {e}"))
}

/// Validate a single emitted event object against the frozen schema's
/// `required_common` + `required_by_event` lists. Returns the FIRST missing
/// field (self-diagnosing) or `Ok(())`. This is the hand-rolled equivalent of a
/// `jsonschema` validate (recorded in `deps_wanted`).
pub fn validate_event(schema: &Value, event: &Value) -> Result<(), String> {
    let obj = event.as_object().ok_or("event is not an object")?;
    let required_common = schema
        .get("required_common")
        .and_then(Value::as_array)
        .ok_or("schema missing required_common")?;
    for f in required_common {
        let key = f.as_str().unwrap_or_default();
        if !obj.contains_key(key) {
            return Err(format!("event missing required_common field {key:?}"));
        }
    }
    let event_kind = obj
        .get("event")
        .and_then(Value::as_str)
        .ok_or("event has no `event` kind")?;
    // Validate the `event` value is an allowed enum member.
    if let Some(enums) = schema
        .get("enums")
        .and_then(|e| e.get("event"))
        .and_then(Value::as_array)
    {
        let allowed: Vec<&str> = enums.iter().filter_map(Value::as_str).collect();
        if !allowed.contains(&event_kind) {
            return Err(format!("event kind {event_kind:?} not in {allowed:?}"));
        }
    }
    if let Some(per) = schema
        .get("required_by_event")
        .and_then(|r| r.get(event_kind))
        .and_then(Value::as_array)
    {
        for f in per {
            let key = f.as_str().unwrap_or_default();
            if !obj.contains_key(key) {
                return Err(format!(
                    "`{event_kind}` event missing required field {key:?}"
                ));
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Inline unit tests — the comparator MATH, on SYNTHETIC vectors only.
//    These run with no weights and no fixtures (the task contract: "the
//    comparator MATH is unit-tested inline with synthetic vectors").
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let v = [1.0f32, 2.0, 3.0, -4.0];
        assert!(
            (cosine(&v, &v) - 1.0).abs() < 1e-12,
            "identical vectors ⇒ cosine 1.0"
        );
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-12, "orthogonal ⇒ 0");
    }

    #[test]
    fn cosine_above_threshold_for_tiny_perturbation() {
        // A 1e-5 relative perturbation must stay above the L1/L2 0.9999 gate.
        let a: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = a.iter().map(|&x| x * (1.0 + 1e-5)).collect();
        let c = cosine(&a, &b);
        assert!(
            c >= COSINE_F32_THRESHOLD,
            "cosine {c} below {COSINE_F32_THRESHOLD}"
        );
    }

    #[test]
    fn cosine_zero_vs_zero_is_one() {
        let z = [0.0f32, 0.0, 0.0];
        assert_eq!(cosine(&z, &z), 1.0);
    }

    #[test]
    fn ulp_distance_adjacent_is_one() {
        let a = 1.0f32;
        let b = f32::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_distance(a, b), 1, "adjacent floats ⇒ 1 ULP");
    }

    #[test]
    fn ulp_distance_across_zero_sign() {
        // -0.0 and +0.0 are the same value ⇒ 0 ULP.
        assert_eq!(ulp_distance(-0.0, 0.0), 0);
        // The two smallest-magnitude floats of opposite sign are 2 ULP apart
        // (±MIN_POSITIVE-subnormal straddling zero), proving the sign-flip map.
        let tiny = f32::from_bits(1); // smallest positive subnormal
        let neg_tiny = -tiny;
        assert_eq!(ulp_distance(tiny, neg_tiny), 2);
    }

    #[test]
    fn ulp_distance_nan_is_max() {
        assert_eq!(ulp_distance(f32::NAN, 1.0), u32::MAX);
    }

    #[test]
    fn ulp_table_matches_methodology() {
        assert_eq!(
            ulp_table(OpFamily::MatmulF32),
            4,
            "matmul 4 ULP (METHODOLOGY §1.3)"
        );
        assert_eq!(ulp_table(OpFamily::Elementwise), 2, "elementwise 2 ULP");
        assert_eq!(ulp_table(OpFamily::Reduction), 8);
        assert_eq!(ulp_table(OpFamily::Transcendental), 8);
    }

    #[test]
    fn ulp_compare_passes_within_budget_fails_outside() {
        let oracle: Vec<f32> = (0..64).map(|i| 0.5 + i as f32).collect();
        // Within budget: every element +1 ULP ⇒ ≤ 4 (matmul) passes.
        let near: Vec<f32> = oracle
            .iter()
            .map(|&x| f32::from_bits(x.to_bits() + 1))
            .collect();
        let r = ulp_compare(&near, &oracle, OpFamily::MatmulF32);
        assert!(r.pass, "1 ULP within 4-ULP matmul budget; report {r:?}");
        assert_eq!(r.max_ulp, 1);
        // Outside elementwise budget (2 ULP): +5 ULP fails and names the index.
        let far: Vec<f32> = oracle
            .iter()
            .map(|&x| f32::from_bits(x.to_bits() + 5))
            .collect();
        let r2 = ulp_compare(&far, &oracle, OpFamily::Elementwise);
        assert!(!r2.pass, "5 ULP exceeds 2-ULP elementwise budget");
        assert_eq!(r2.max_ulp, 5);
        assert!(r2.max_abs_diff > 0.0);
    }

    #[test]
    fn tensorspec_rejects_shape_and_dtype_mismatch() {
        let a = TensorSpec::new([2, 3], DType::F32);
        let b = TensorSpec::new([2, 3], DType::F32);
        assert!(a.check_against(&b).is_ok());
        let c = TensorSpec::new([3, 2], DType::F32);
        let err = a.check_against(&c).unwrap_err();
        assert!(err.contains("shape mismatch"), "got {err}");
        let d = TensorSpec::new([2, 3], DType::I8);
        assert!(a.check_against(&d).unwrap_err().contains("dtype mismatch"));
    }

    #[test]
    fn scrub_removes_timing_keeps_shape() {
        let raw = json!({
            "schema_version": 1,
            "event": "stage",
            "stage": "vision_sam",
            "elapsed_us": 1432,
            "run_id": "abc-123",
            "path": "/home/user/doc.png",
            "shapes": [[1, 256, 1280]]
        });
        let s = scrub_volatile(&raw);
        assert_eq!(s["elapsed_us"], json!("[ms]"), "timing scrubbed");
        assert_eq!(s["run_id"], json!("[id]"), "id scrubbed");
        assert_eq!(s["path"], json!("[path]"), "abs path scrubbed");
        // Non-volatile content untouched (presence + value preserved).
        assert_eq!(s["stage"], json!("vision_sam"));
        assert_eq!(s["shapes"], json!([[1, 256, 1280]]));
        // The scrubbed field is still PRESENT (a dropped field must still fail).
        assert!(s.as_object().unwrap().contains_key("elapsed_us"));
    }

    #[test]
    fn establish_floor_finds_prefix_and_spread() {
        // Two runs agree for 5 tokens then diverge; logits spread by 0.03 max.
        let la = vec![vec![1.0f32, 2.0, 3.0]; 6];
        let lb = vec![vec![1.0f32, 2.0, 3.03]; 6];
        let ta = [10u32, 11, 12, 13, 14, 99];
        let tb = [10u32, 11, 12, 13, 14, 77];
        let floor = establish_floor(&la, &lb, &ta, &tb);
        assert_eq!(
            floor.reproducible_prefix_len, 5,
            "prefix is the agreeing run"
        );
        assert_eq!(floor.first_divergence_pos, Some(5));
        assert!(
            (floor.l3_logit_tolerance() - 0.03).abs() < 1e-6,
            "spread {}",
            floor.l3_logit_tolerance()
        );
        assert!(floor.per_token_divergence_rate > 0.0);
    }

    #[test]
    fn establish_floor_identical_runs_full_prefix() {
        let la = vec![vec![1.0f32; 4]; 3];
        let ta = [1u32, 2, 3];
        let floor = establish_floor(&la, &la, &ta, &ta);
        assert_eq!(floor.reproducible_prefix_len, 3);
        assert_eq!(
            floor.first_divergence_pos, None,
            "identical ⇒ no divergence"
        );
        assert_eq!(floor.l3_logit_tolerance(), 0.0);
    }

    #[test]
    fn read_npy_roundtrip_synthetic_v1() {
        // Hand-build a minimal v1.0 .npy: magic, header, C-order f32 payload.
        let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        // Pad header so total (10 + header) is a multiple of 64, terminated by \n.
        let mut hdr = header.to_string();
        let base = 10 + hdr.len() + 1; // +1 for trailing \n
        let pad = (64 - base % 64) % 64;
        hdr.push_str(&" ".repeat(pad));
        hdr.push('\n');
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1); // major
        bytes.push(0); // minor
        bytes.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
        bytes.extend_from_slice(hdr.as_bytes());
        for &x in &data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let (shape, parsed) = read_npy_f32(&bytes).expect("parse synthetic npy");
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(parsed, data.to_vec());
    }

    #[test]
    fn read_npy_rejects_non_npy() {
        assert!(read_npy_f32(b"not a numpy file at all").is_err());
    }

    #[test]
    fn golden_from_value_parses_manifest() {
        let raw = json!({
            "doc": "doc01.png",
            "decoded_text": "# Title\nbody",
            "decoded_text_sha256": "deadbeef",
            "activations": {
                "sam_output": { "file": "sam_output.npy", "shape": [1, 256, 1280],
                                "dtype": "float32", "sha256": "aa" }
            },
            "provenance": { "hf_commit": HF_COMMIT, "pinned_torch": PIN_TORCH }
        });
        let g = FixtureLoader::golden_from_value(raw).expect("parse golden");
        assert_eq!(g.doc, "doc01.png");
        assert_eq!(g.decoded_text.as_deref(), Some("# Title\nbody"));
        assert_eq!(g.activations["sam_output"].shape, vec![1, 256, 1280]);
        assert!(
            FixtureLoader::check_provenance(&g).is_ok(),
            "pinned provenance resolves"
        );
    }

    #[test]
    fn check_provenance_rejects_wrong_commit() {
        let raw = json!({
            "doc": "x", "provenance": { "hf_commit": "wrong", "pinned_torch": PIN_TORCH }
        });
        let g = FixtureLoader::golden_from_value(raw).unwrap();
        assert!(
            FixtureLoader::check_provenance(&g)
                .unwrap_err()
                .contains("hf_commit")
        );
    }

    #[test]
    fn fixture_loader_reports_absence_on_empty_root() {
        // A non-existent root reports no goldens (so a rung skips-with-SUCCESS).
        let loader = FixtureLoader::at("/nonexistent/franken_ocr/fixtures");
        assert!(!loader.any_present(), "absent root ⇒ no goldens present");
        assert!(loader.list_goldens().unwrap().is_empty());
    }

    #[test]
    fn emitted_events_conform_to_frozen_schema() {
        // The logger's lines must validate against tests/fixtures/test_log_schema.json
        // (contract-test-vs-frozen-schema). We re-create the exact event objects the
        // Logger emits and validate each.
        let schema = load_log_schema().expect("load frozen log schema");
        let parity = json!({
            "schema_version": 1, "ts": 0, "test": "L1", "case": "doc01", "run_seq": 0,
            "event": "parity", "result": "pass",
            "gate": "L1", "metric": "cosine", "value": 0.99999, "tolerance": 0.9999,
            "oracle_fixture": "sam_output.npy", "oracle_sha256": "aa",
            "nondeterminism_envelope": {}, "pass": true
        });
        validate_event(&schema, &parity).expect("parity event conforms");
        let skip = json!({
            "schema_version": 1, "ts": 0, "test": "L5", "case": "doc01", "run_seq": 0,
            "event": "skip", "result": "skip_no_model", "reason": "no fixtures"
        });
        validate_event(&schema, &skip).expect("skip event conforms");
        // A malformed event (missing required `gate`) is caught with the field named.
        let bad = json!({
            "schema_version": 1, "ts": 0, "test": "L1", "case": "x", "run_seq": 0,
            "event": "parity", "result": "pass", "metric": "cosine"
        });
        assert!(validate_event(&schema, &bad).unwrap_err().contains("gate"));
    }

    #[test]
    fn log_schema_version_matches_robot() {
        // Guard: this infra's literal LOG_SCHEMA_VERSION must equal the lib's
        // ROBOT_SCHEMA_VERSION (which is 1). Asserted against the frozen schema
        // fixture's own schema_version so all three agree.
        let schema = load_log_schema().expect("schema");
        assert_eq!(schema["schema_version"].as_u64(), Some(LOG_SCHEMA_VERSION));
        assert_eq!(
            franken_ocr::robot::ROBOT_SCHEMA_VERSION as u64,
            LOG_SCHEMA_VERSION
        );
    }
}
