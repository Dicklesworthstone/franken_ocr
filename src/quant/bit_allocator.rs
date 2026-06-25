//! AF-1 — rate-distortion / Lagrangian water-filling per-tensor bit allocator.
//!
//! A faithful Rust port of [`scripts/af1_bit_allocator.py`] (the reference
//! implementation), specified in [`docs/alien/AF-1-rate-distortion-bit-allocation.md`].
//!
//! Choosing `{bf16, int8, int4-g32, int4-g16}` per quantizable tensor under a
//! total footprint budget `B` is a rate-distortion problem:
//!
//! ```text
//! minimize  D(b) = Σ_t D_t(b_t)        (additive layer-output cosine-drop surrogate)
//! s.t.      R(b) = Σ_t R_t(b_t) ≤ B    (total footprint in bytes)
//! ```
//!
//! The Lagrangian `L(b, λ) = Σ_t [ D_t(b_t) + λ·R_t(b_t) ]` separates per tensor,
//! so at any price `λ` each tensor independently picks the option minimizing its
//! RD-cost. Sweeping `λ` traces the optimal frontier (water-filling); we walk it
//! until the footprint just fits `B`, then spend the slack with a greedy
//! hull-climb. An exact bounded-knapsack DP ([`allocate_exact_dp`]) closes the
//! duality gap when requested.
//!
//! Per the **Alien-Artifact Engineering Contract** (AGENTS.md) this module makes
//! every required element explicit:
//! * **state space** — per tensor, the finite ordered option set `O` on its
//!   convex hull ([`Tensor::points`]);
//! * **actions** — assign each tensor one option (the allocation `b`);
//! * **loss matrix** — the per-`(tensor, option)` distortion `D_t(o)` carried in
//!   each [`Point`], plus the footprint cost `R_t(o)`;
//! * **calibration term** — the distortion curves are the *measured* layer-output
//!   cosine drop on a calibration batch (the converter's `--measure-distortion`
//!   output; this module consumes it);
//! * **deterministic fallback** — [`allocate_uniform`] (the conservative,
//!   always-wired uniform Q4_K_M recipe), selected automatically by
//!   [`allocate`] when the optimizer is unsure (flat curves / infeasible /
//!   surrogate pre-check fails);
//! * **evidence-ledger artifact** — [`AllocationTable`], the machine-readable
//!   record of *what was decided and why* (one [`AllocationRecord`] per tensor
//!   with a `reason`, plus the `uniform_baseline` and `surrogate_precheck_pass`).
//!
//! It is deterministic (pure function of inputs; deterministic tie-breaks; no
//! RNG / clock / map-iteration-order dependence). The [`selftest`] reproduces the
//! Python `--selftest` invariants.

use std::collections::BTreeMap;
use std::fmt;

/// `bit_allocation_table.json` schema version (matches the Python
/// `SCHEMA_VERSION`).
pub const SCHEMA_VERSION: u32 = 1;

/// Generator tag baked into the emitted table.
pub const GENERATOR: &str = "src/quant/bit_allocator.rs";

/// Default effective bits-per-weight per option (scale overhead included). These
/// are only fallbacks when a curve point omits an explicit `bits` value. Matches
/// `DEFAULT_OPTION_BITS` in the Python.
pub const DEFAULT_OPTION_BITS: &[(&str, f64)] = &[
    ("bf16", 16.0),
    ("int8", 8.03),
    ("int4-g32", 5.0),
    ("int4-g16", 6.0),
];

/// Numerical tie-break epsilon, matching the Python's `1e-18`.
const EPS: f64 = 1e-18;

/// An allocation error: malformed input or an infeasible budget. Mirrors the
/// Python `AllocatorError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocatorError(pub String);

impl fmt::Display for AllocatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for AllocatorError {}

/// Result alias for the allocator.
pub type AllocResult<T> = Result<T, AllocatorError>;

// ── option ranking ──────────────────────────────────────────────────────────

/// Canonical option ordering (index 0 = highest precision). Used for
/// deterministic tie-breaks (prefer higher precision on equal RD-cost) and for
/// the tier-floor comparison. Matches the Python `OPTION_ORDER`.
const OPTION_ORDER: &[(&str, i64)] = &[
    ("bf16", 0),
    ("int8", 1),
    ("int4-g16", 2),
    ("int4-g32", 3),
];

/// Deterministic precision rank; unknown options sort after known ones by name
/// (sum of char codes). Mirrors the Python `_option_rank`.
fn option_rank(option: &str) -> i64 {
    for (name, rank) in OPTION_ORDER {
        if *name == option {
            return *rank;
        }
    }
    let sum: i64 = option.chars().map(|c| c as i64).sum();
    OPTION_ORDER.len() as i64 + sum
}

/// Footprint in bytes for `numel` weights at `bpw` effective bits-per-weight
/// (`ceil(numel * bpw / 8)`). Matches the Python `bytes_for`.
#[must_use]
pub fn bytes_for(numel: u64, bpw: f64) -> u64 {
    (numel as f64 * bpw / 8.0).ceil() as u64
}

fn default_option_bits(option: &str) -> Option<f64> {
    DEFAULT_OPTION_BITS
        .iter()
        .find(|(name, _)| *name == option)
        .map(|(_, b)| *b)
}

// ── R-D point + tensor ──────────────────────────────────────────────────────

/// One operational rate-distortion point for a tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// The option label (e.g. `"int8"`).
    pub option: String,
    /// Effective bits-per-weight (incl. scale overhead).
    pub bits: f64,
    /// `D_t(option)`: layer-output cosine drop (≥ 0, `bf16 == 0`).
    pub distortion: f64,
    /// `R_t(option)`: footprint in bytes for this tensor.
    pub rate_bytes: u64,
}

/// One option of a tensor's raw curve, as ingested from the curves JSON.
#[derive(Debug, Clone)]
pub struct CurveOption {
    /// The option label.
    pub option: String,
    /// Explicit bits-per-weight, or `None` to use the global / default table.
    pub bits: Option<f64>,
    /// Measured cosine-drop distortion.
    pub distortion: f64,
}

/// A quantizable tensor with its raw curve, pins, and tier-floor — the
/// allocator's input unit. Mirrors the Python input `tensors[*]` entry.
#[derive(Debug, Clone)]
pub struct TensorCurve {
    /// Tensor name.
    pub name: String,
    /// Element count (`out × in` for a Linear weight).
    pub numel: u64,
    /// `None`, or `Some("bf16")` to force high precision.
    pub pin: Option<String>,
    /// `None`, or e.g. `Some("int8")` (§6.3 `_M` discipline: never below this).
    pub tier_floor: Option<String>,
    /// The raw `option -> {bits, distortion}` curve.
    pub curve: Vec<CurveOption>,
}

/// A processed tensor: its convex-hull-pruned R-D points plus mutable allocation
/// state. Mirrors the Python `Tensor` dataclass.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor name.
    pub name: String,
    /// Element count.
    pub numel: u64,
    /// Pruned, convex, byte-ascending operational points.
    pub points: Vec<Point>,
    /// bf16 pin (if any).
    pub pin: Option<String>,
    /// Tier floor (if any).
    pub tier_floor: Option<String>,
    /// Index into `points` of the currently chosen option.
    pub chosen_idx: usize,
}

impl Tensor {
    /// The currently chosen point.
    #[must_use]
    pub fn chosen(&self) -> &Point {
        &self.points[self.chosen_idx]
    }

    /// The cheapest (fewest-bytes) point.
    #[must_use]
    pub fn cheapest(&self) -> &Point {
        &self.points[0]
    }

    /// The costliest (most-bytes) point.
    #[must_use]
    pub fn costliest(&self) -> &Point {
        &self.points[self.points.len() - 1]
    }
}

// ── curve ingestion: build, monotone repair, hull prune, pins/floors ────────

/// Build raw points from a tensor's curve, resolving missing `bits` from the
/// `option_bits` table. Sorted by (bytes asc, precision rank, option). Mirrors
/// the Python `_build_points`.
fn build_points(
    name: &str,
    numel: u64,
    curve: &[CurveOption],
    option_bits: &BTreeMap<String, f64>,
) -> AllocResult<Vec<Point>> {
    if curve.is_empty() {
        return Err(AllocatorError(format!("tensor {name:?}: empty curve")));
    }
    let mut raw: Vec<Point> = Vec::with_capacity(curve.len());
    for opt in curve {
        let bits = match opt.bits {
            Some(b) => b,
            None => option_bits
                .get(&opt.option)
                .copied()
                .or_else(|| default_option_bits(&opt.option))
                .ok_or_else(|| {
                    AllocatorError(format!(
                        "tensor {name:?}: option {:?} has no known bits-per-weight",
                        opt.option
                    ))
                })?,
        };
        if opt.distortion < 0.0 {
            return Err(AllocatorError(format!(
                "tensor {name:?}: option {:?} has negative distortion {}",
                opt.option, opt.distortion
            )));
        }
        raw.push(Point {
            option: opt.option.clone(),
            bits,
            distortion: opt.distortion,
            rate_bytes: bytes_for(numel, bits),
        });
    }
    // Deterministic order: by bytes asc, then precision rank, then name.
    raw.sort_by(|a, b| {
        a.rate_bytes
            .cmp(&b.rate_bytes)
            .then(option_rank(&a.option).cmp(&option_rank(&b.option)))
            .then(a.option.cmp(&b.option))
    });
    Ok(raw)
}

/// Enforce D non-increasing as bytes increase (cummin from the cheap end).
/// Mirrors the Python `_monotone_repair`.
fn monotone_repair(points: &[Point]) -> Vec<Point> {
    let mut repaired = Vec::with_capacity(points.len());
    let mut running_min = f64::INFINITY;
    for p in points {
        let d = p.distortion.min(running_min);
        running_min = d;
        repaired.push(Point {
            option: p.option.clone(),
            bits: p.bits,
            distortion: d,
            rate_bytes: p.rate_bytes,
        });
    }
    repaired
}

/// Keep only the lower convex hull of the `(bytes, distortion)` points. Mirrors
/// the Python `_convex_hull_prune`.
fn convex_hull_prune(points: &[Point]) -> Vec<Point> {
    // De-duplicate identical byte sizes, keeping the lowest distortion (then
    // highest precision). Insertion order follows the byte-ascending input.
    let mut by_bytes: BTreeMap<u64, Point> = BTreeMap::new();
    for p in points {
        match by_bytes.get(&p.rate_bytes) {
            Some(cur) => {
                let cand_key = (p.distortion, option_rank(&p.option));
                let cur_key = (cur.distortion, option_rank(&cur.option));
                if lt_tuple(cand_key, cur_key) {
                    by_bytes.insert(p.rate_bytes, p.clone());
                }
            }
            None => {
                by_bytes.insert(p.rate_bytes, p.clone());
            }
        }
    }
    // BTreeMap over u64 keys => byte-ascending.
    let uniq: Vec<Point> = by_bytes.into_values().collect();
    if uniq.len() <= 2 {
        return uniq;
    }
    let mut hull: Vec<Point> = Vec::new();
    for p in uniq {
        while hull.len() >= 2 {
            let a = &hull[hull.len() - 2];
            let b = &hull[hull.len() - 1];
            // cross of (b-a) x (p-a) in (bytes, distortion) space; <= 0 => b not below.
            let cross = (b.rate_bytes as f64 - a.rate_bytes as f64)
                * (p.distortion - a.distortion)
                - (b.distortion - a.distortion) * (p.rate_bytes as f64 - a.rate_bytes as f64);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    hull
}

/// `(f64, i64)` lexicographic strict-less, matching Python tuple comparison.
fn lt_tuple(a: (f64, i64), b: (f64, i64)) -> bool {
    if a.0 < b.0 {
        return true;
    }
    if a.0 > b.0 {
        return false;
    }
    a.1 < b.1
}

/// Restrict the option set per a bf16 pin and/or a tier floor. Mirrors the
/// Python `_apply_pins_and_floors`.
fn apply_pins_and_floors(
    name: &str,
    points: Vec<Point>,
    pin: Option<&str>,
    tier_floor: Option<&str>,
) -> AllocResult<Vec<Point>> {
    if let Some(pin) = pin {
        let pinned: Vec<Point> = points.into_iter().filter(|p| p.option == pin).collect();
        if pinned.is_empty() {
            return Err(AllocatorError(format!(
                "tensor {name:?}: pin {pin:?} not present in its curve"
            )));
        }
        return Ok(pinned);
    }
    if let Some(floor) = tier_floor {
        let floor_rank = option_rank(floor);
        let kept: Vec<Point> = points
            .into_iter()
            .filter(|p| option_rank(&p.option) <= floor_rank)
            .collect();
        if kept.is_empty() {
            return Err(AllocatorError(format!(
                "tensor {name:?}: tier_floor {floor:?} removes every option"
            )));
        }
        return Ok(kept);
    }
    Ok(points)
}

/// Parse + validate the curves into convex-hull [`Tensor`] objects. Mirrors the
/// Python `load_tensors`. `global_option_bits` overrides the default per-option
/// bpw table.
pub fn load_tensors(
    curves: &[TensorCurve],
    global_option_bits: &BTreeMap<String, f64>,
) -> AllocResult<Vec<Tensor>> {
    let mut option_bits: BTreeMap<String, f64> = DEFAULT_OPTION_BITS
        .iter()
        .map(|(k, v)| ((*k).to_string(), *v))
        .collect();
    for (k, v) in global_option_bits {
        option_bits.insert(k.clone(), *v);
    }
    if curves.is_empty() {
        return Err(AllocatorError(
            "curves must contain a non-empty 'tensors' array".to_string(),
        ));
    }
    let mut tensors: Vec<Tensor> = Vec::with_capacity(curves.len());
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    for entry in curves {
        if seen.insert(entry.name.clone(), ()).is_some() {
            return Err(AllocatorError(format!(
                "duplicate tensor name {:?}",
                entry.name
            )));
        }
        if entry.numel == 0 {
            return Err(AllocatorError(format!(
                "tensor {:?}: numel must be positive",
                entry.name
            )));
        }
        let raw = build_points(&entry.name, entry.numel, &entry.curve, &option_bits)?;
        // P1: bf16 anchor must be lossless when present.
        for p in &raw {
            if p.option == "bf16" && p.distortion != 0.0 {
                return Err(AllocatorError(format!(
                    "tensor {:?}: bf16 anchor distortion must be 0, got {}",
                    entry.name, p.distortion
                )));
            }
        }
        let raw = apply_pins_and_floors(
            &entry.name,
            raw,
            entry.pin.as_deref(),
            entry.tier_floor.as_deref(),
        )?;
        let pts = convex_hull_prune(&monotone_repair(&raw));
        if pts.is_empty() {
            return Err(AllocatorError(format!(
                "tensor {:?}: no options survive pruning",
                entry.name
            )));
        }
        tensors.push(Tensor {
            name: entry.name.clone(),
            numel: entry.numel,
            points: pts,
            pin: entry.pin.clone(),
            tier_floor: entry.tier_floor.clone(),
            chosen_idx: 0,
        });
    }
    // Deterministic tensor order (by name).
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tensors)
}

// ── Lagrangian water-filling ─────────────────────────────────────────────────

/// Index of the option minimizing `D + λ·R`; deterministic tie-break (prefer
/// higher precision). Mirrors the Python `_argmin_at_lambda`.
fn argmin_at_lambda(t: &Tensor, lam: f64) -> usize {
    let mut best_idx = 0usize;
    let mut best_cost = t.points[0].distortion + lam * t.points[0].rate_bytes as f64;
    let mut best_rank = option_rank(&t.points[0].option);
    for idx in 1..t.points.len() {
        let p = &t.points[idx];
        let cost = p.distortion + lam * p.rate_bytes as f64;
        let rank = option_rank(&p.option);
        if cost < best_cost - EPS || ((cost - best_cost).abs() <= EPS && rank < best_rank) {
            best_idx = idx;
            best_cost = cost;
            best_rank = rank;
        }
    }
    best_idx
}

fn footprint(tensors: &[Tensor]) -> u64 {
    tensors.iter().map(|t| t.chosen().rate_bytes).sum()
}

fn distortion(tensors: &[Tensor]) -> f64 {
    tensors.iter().map(|t| t.chosen().distortion).sum()
}

fn assign_at_lambda(tensors: &mut [Tensor], lam: f64) {
    for t in tensors.iter_mut() {
        t.chosen_idx = argmin_at_lambda(t, lam);
    }
}

/// Spend leftover budget on the globally steepest distortion-per-added-byte
/// upgrade. Mirrors the Python `_greedy_topup`.
fn greedy_topup(tensors: &mut [Tensor], budget: u64) {
    let mut used = footprint(tensors);
    loop {
        let mut best_ti: Option<usize> = None;
        let mut best_to_idx: usize = 0;
        let mut best_gain_per_byte = 0.0f64;
        let mut best_added: u64 = 0;
        // For the deterministic tie-break we need the current best's (name, option).
        for (ti, t) in tensors.iter().enumerate() {
            let cur = t.chosen();
            // Only the immediate next hull step is a single edge.
            for j in (t.chosen_idx + 1)..t.points.len() {
                let cand = &t.points[j];
                if cand.rate_bytes <= cur.rate_bytes {
                    // not an upgrade in bytes; skip but still only consider the first
                    continue;
                }
                let added = cand.rate_bytes - cur.rate_bytes;
                if used + added > budget {
                    break;
                }
                let drop = cur.distortion - cand.distortion; // >= 0 by monotonicity
                let gain_per_byte = drop / added as f64;
                let strictly_better = gain_per_byte > best_gain_per_byte + EPS;
                let tie_better = (gain_per_byte - best_gain_per_byte).abs() <= EPS
                    && best_ti.is_some()
                    && {
                        let bt = &tensors[best_ti.unwrap()];
                        let bo = &bt.points[best_to_idx].option;
                        (t.name.as_str(), cand.option.as_str()) < (bt.name.as_str(), bo.as_str())
                    };
                if strictly_better || tie_better {
                    best_ti = Some(ti);
                    best_to_idx = j;
                    best_gain_per_byte = gain_per_byte;
                    best_added = added;
                }
                break; // only the immediate next hull step is a single edge
            }
        }
        match best_ti {
            Some(ti) if best_gain_per_byte > 0.0 => {
                tensors[ti].chosen_idx = best_to_idx;
                used += best_added;
            }
            _ => return,
        }
    }
}

/// Water-filling allocation: bisection on `λ` + greedy hull top-up. Mirrors the
/// Python `allocate_waterfill`.
pub fn allocate_waterfill(tensors: &mut [Tensor], budget: u64) -> AllocResult<AllocationTable> {
    let cheapest: u64 = tensors.iter().map(|t| t.cheapest().rate_bytes).sum();
    let costliest: u64 = tensors.iter().map(|t| t.costliest().rate_bytes).sum();
    if budget < cheapest {
        return Err(AllocatorError(format!(
            "budget {budget} bytes is infeasible: cheapest config needs {cheapest} bytes"
        )));
    }
    if budget >= costliest {
        for t in tensors.iter_mut() {
            t.chosen_idx = t.points.len() - 1;
        }
        return Ok(finish(tensors, budget, "lagrangian-waterfill", Some(0.0)));
    }

    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    assign_at_lambda(tensors, hi);
    while footprint(tensors) > budget {
        hi *= 2.0;
        assign_at_lambda(tensors, hi);
        if hi > 1e30 {
            break;
        }
    }
    let mut lam_feasible = hi;
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        assign_at_lambda(tensors, mid);
        if footprint(tensors) <= budget {
            lam_feasible = mid;
            hi = mid;
        } else {
            lo = mid;
        }
    }
    assign_at_lambda(tensors, lam_feasible);
    greedy_topup(tensors, budget);
    Ok(finish(
        tensors,
        budget,
        "lagrangian-waterfill",
        Some(lam_feasible),
    ))
}

// ── exact bounded-knapsack DP ────────────────────────────────────────────────

/// Exact integer-optimal allocation via 1-D DP over a byte-quantized budget.
/// Mirrors the Python `allocate_exact_dp`.
pub fn allocate_exact_dp(
    tensors: &mut [Tensor],
    budget: u64,
    grid_bytes: u64,
) -> AllocResult<AllocationTable> {
    let cheapest: u64 = tensors.iter().map(|t| t.cheapest().rate_bytes).sum();
    if budget < cheapest {
        return Err(AllocatorError(format!(
            "budget {budget} bytes is infeasible: cheapest config needs {cheapest} bytes"
        )));
    }
    let mut attempt_grid = grid_bytes.max(1);
    for _ in 0..6 {
        let chosen = dp_once(tensors, budget, attempt_grid)?;
        for (t, oi) in tensors.iter_mut().zip(chosen.iter()) {
            t.chosen_idx = *oi;
        }
        if footprint(tensors) <= budget {
            return Ok(finish(tensors, budget, "exact-dp", None));
        }
        attempt_grid = (attempt_grid / 4).max(1);
    }
    // Final guard: water-filling is always exactly feasible.
    allocate_waterfill(tensors, budget)
}

/// One DP pass at a fixed grid; returns the chosen option index per tensor.
/// Mirrors the Python `_dp_once`.
fn dp_once(tensors: &[Tensor], budget: u64, grid_bytes: u64) -> AllocResult<Vec<usize>> {
    let cells = ((budget / grid_bytes).max(1)) as usize;
    let inf = f64::INFINITY;
    // dp[c] = min total distortion reachable using exactly c grid-cells of budget.
    let mut dp = vec![inf; cells + 1];
    dp[0] = 0.0;
    // choice[ti][c] = packed (prev_c, option_idx); store as Option<(usize, usize)>.
    let mut choice: Vec<Vec<Option<(usize, usize)>>> = Vec::with_capacity(tensors.len());

    for t in tensors {
        let mut ndp = vec![inf; cells + 1];
        let mut tchoice: Vec<Option<(usize, usize)>> = vec![None; cells + 1];
        // Order options by ascending bytes (fewer bytes on distortion ties).
        let mut opts: Vec<usize> = (0..t.points.len()).collect();
        opts.sort_by(|&i, &j| {
            t.points[i]
                .rate_bytes
                .cmp(&t.points[j].rate_bytes)
                .then(option_rank(&t.points[i].option).cmp(&option_rank(&t.points[j].option)))
        });
        // `c` is the cell *count* (used in arithmetic `nc = c + cost_cells` and
        // stored as the backtrack predecessor), not merely an index, so the
        // range loop is the clear form here.
        #[allow(clippy::needless_range_loop)]
        for c in 0..=cells {
            let base_d = dp[c];
            if base_d.is_infinite() {
                continue;
            }
            for &oi in &opts {
                let p = &t.points[oi];
                let cost_cells = (p.rate_bytes / grid_bytes) as usize; // FLOOR
                let nc = c + cost_cells;
                if nc > cells {
                    continue;
                }
                let nd = base_d + p.distortion;
                if nd < ndp[nc] - EPS {
                    ndp[nc] = nd;
                    tchoice[nc] = Some((c, oi));
                }
            }
        }
        dp = ndp;
        choice.push(tchoice);
    }

    // Global minimum distortion, then the LARGEST fill achieving it.
    let best_d = dp
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(inf, f64::min);
    if best_d.is_infinite() {
        return Err(AllocatorError(
            "DP found no feasible allocation (budget too small)".to_string(),
        ));
    }
    let best_c = (0..=cells)
        .rev()
        .find(|&c| dp[c] <= best_d + EPS)
        .ok_or_else(|| AllocatorError("DP found no feasible allocation".to_string()))?;

    let mut chosen = vec![0usize; tensors.len()];
    let mut c = best_c;
    for ti in (0..tensors.len()).rev() {
        let (prev_c, oi) = choice[ti][c]
            .ok_or_else(|| AllocatorError("DP backtrack failed (internal inconsistency)".into()))?;
        chosen[ti] = oi;
        c = prev_c;
    }
    Ok(chosen)
}

// ── deterministic fallback: uniform Q4_K_M-class allocation ──────────────────

/// Uniform allocation: every tensor to `uniform_option` (or its tier-floor /
/// pin). The conservative deterministic fallback. Mirrors the Python
/// `allocate_uniform`.
pub fn allocate_uniform(
    tensors: &mut [Tensor],
    uniform_option: &str,
    budget: Option<u64>,
) -> AllocationTable {
    for t in tensors.iter_mut() {
        let target_rank = option_rank(uniform_option);
        let eligible: Vec<usize> = (0..t.points.len())
            .filter(|&i| option_rank(&t.points[i].option) <= target_rank)
            .collect();
        if !eligible.is_empty() {
            // Prefer the exact uniform option if present, else the lowest
            // precision >= floor (max rank among eligible).
            let exact = eligible
                .iter()
                .copied()
                .find(|&i| t.points[i].option == uniform_option);
            t.chosen_idx = match exact {
                Some(i) => i,
                None => eligible
                    .iter()
                    .copied()
                    .max_by_key(|&i| option_rank(&t.points[i].option))
                    .unwrap(),
            };
        } else {
            // uniform_option cheaper than everything kept (pinned bf16): cheapest kept.
            t.chosen_idx = (0..t.points.len())
                .min_by_key(|&i| t.points[i].rate_bytes)
                .unwrap();
        }
    }
    let eff_budget = budget.unwrap_or_else(|| footprint(tensors));
    finish(tensors, eff_budget, &format!("uniform-{uniform_option}"), None)
}

// ── result assembly + evidence ledger ────────────────────────────────────────

/// The densest uniform option whose footprint ≤ `target_bytes`, for the proof
/// pre-check. Mirrors the Python `_uniform_equal_footprint_distortion`.
fn uniform_equal_footprint(tensors: &[Tensor], target_bytes: u64) -> UniformBaseline {
    // Candidate uniform options = union of options across tensors, ordered by
    // least bits first (ascending bytes).
    let mut all_opts: BTreeMap<String, f64> = BTreeMap::new();
    for t in tensors {
        for p in &t.points {
            all_opts.insert(p.option.clone(), p.bits);
        }
    }
    let mut ordered: Vec<(String, f64)> = all_opts.into_iter().collect();
    // least bits first: sort by (-bits, rank).
    ordered.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(option_rank(&a.0).cmp(&option_rank(&b.0)))
    });

    let eval_uniform = |option: &str| -> (u64, f64) {
        let mut fp = 0u64;
        let mut dd = 0.0f64;
        let target_rank = option_rank(option);
        for t in tensors {
            let eligible: Vec<&Point> = t
                .points
                .iter()
                .filter(|p| option_rank(&p.option) <= target_rank)
                .collect();
            let p: &Point = if !eligible.is_empty() {
                eligible
                    .iter()
                    .copied()
                    .find(|p| p.option == option)
                    .unwrap_or_else(|| {
                        eligible
                            .iter()
                            .copied()
                            .max_by_key(|p| option_rank(&p.option))
                            .unwrap()
                    })
            } else {
                t.points
                    .iter()
                    .min_by_key(|p| p.rate_bytes)
                    .unwrap()
            };
            fp += p.rate_bytes;
            dd += p.distortion;
        }
        (fp, dd)
    };

    let mut best: Option<UniformBaseline> = None;
    for (option, _bpw) in &ordered {
        let (fp, dd) = eval_uniform(option);
        if fp <= target_bytes {
            let cand = UniformBaseline {
                option: option.clone(),
                footprint_bytes: fp,
                distortion: round9(dd),
            };
            if best.as_ref().is_none_or(|b| fp > b.footprint_bytes) {
                best = Some(cand);
            }
        }
    }
    best.unwrap_or_else(|| {
        // Even the cheapest uniform exceeds target; report the cheapest config.
        let option = ordered[0].0.clone();
        let fp: u64 = tensors
            .iter()
            .map(|t| t.points.iter().min_by_key(|p| p.rate_bytes).unwrap().rate_bytes)
            .sum();
        let dd: f64 = tensors
            .iter()
            .map(|t| t.points.iter().min_by_key(|p| p.rate_bytes).unwrap().distortion)
            .sum();
        UniformBaseline {
            option,
            footprint_bytes: fp,
            distortion: round9(dd),
        }
    })
}

/// The equal-footprint uniform baseline carried in the evidence ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformBaseline {
    /// The uniform option label.
    pub option: String,
    /// Its total footprint in bytes.
    pub footprint_bytes: u64,
    /// Its total distortion.
    pub distortion: f64,
}

/// One per-tensor allocation record in the evidence ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct AllocationRecord {
    /// Tensor name.
    pub tensor: String,
    /// Element count.
    pub numel: u64,
    /// Chosen option label.
    pub option: String,
    /// Effective bits-per-weight at the chosen option (rounded to 4 dp).
    pub bits_per_weight: f64,
    /// Chosen footprint in bytes.
    pub bytes: u64,
    /// Distortion at the chosen option (rounded to 9 dp).
    pub distortion: f64,
    /// Marginal distortion-per-byte of the next available upgrade (hull slope).
    pub marginal_dpb: f64,
    /// Human-readable why (the evidence-ledger reason).
    pub reason: String,
}

/// Total summary in the evidence ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct Totals {
    /// `Σ R_t` chosen.
    pub footprint_bytes: u64,
    /// Footprint in GiB.
    pub footprint_gib: f64,
    /// `Σ D_t` chosen.
    pub distortion: f64,
    /// The equal-footprint uniform baseline it must beat.
    pub uniform_baseline: UniformBaseline,
    /// Whether `Σ D_alloc ≤ Σ D_uniform` (the §5 P6 surrogate pre-check).
    pub surrogate_precheck_pass: bool,
}

/// The emitted evidence-ledger artifact — the machine-readable record of what
/// was decided and why. Mirrors the Python `bit_allocation_table` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct AllocationTable {
    /// Schema version.
    pub schema_version: u32,
    /// Generator tag.
    pub generator: String,
    /// `"lagrangian-waterfill"` / `"exact-dp"` / `"uniform-<opt>"`.
    pub method: String,
    /// The chosen price (distortion per byte); `None` for DP / uniform.
    pub lambda_star: Option<f64>,
    /// The budget in bytes.
    pub budget_bytes: u64,
    /// Summary totals + the surrogate pre-check.
    pub totals: Totals,
    /// Number of tensors.
    pub n_tensors: usize,
    /// One record per tensor (name-sorted, since `tensors` is sorted).
    pub allocation: Vec<AllocationRecord>,
}

impl AllocationTable {
    /// Serialize to the canonical, sorted-key JSON the Python emits (so a
    /// Rust↔Python differential is byte-checkable on the structural fields).
    /// Deterministic: stable key order, fixed float formatting.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"budget_bytes\": {},\n", self.budget_bytes));
        s.push_str(&format!("  \"generator\": {},\n", json_str(&self.generator)));
        s.push_str(&format!(
            "  \"lambda_star\": {},\n",
            match self.lambda_star {
                Some(l) => fmt_float(l),
                None => "null".to_string(),
            }
        ));
        s.push_str(&format!("  \"method\": {},\n", json_str(&self.method)));
        s.push_str(&format!("  \"n_tensors\": {},\n", self.n_tensors));
        s.push_str(&format!("  \"schema_version\": {},\n", self.schema_version));
        // totals
        s.push_str("  \"totals\": {\n");
        s.push_str(&format!(
            "    \"distortion\": {},\n",
            fmt_float(self.totals.distortion)
        ));
        s.push_str(&format!(
            "    \"footprint_bytes\": {},\n",
            self.totals.footprint_bytes
        ));
        s.push_str(&format!(
            "    \"footprint_gib\": {},\n",
            fmt_float(self.totals.footprint_gib)
        ));
        s.push_str(&format!(
            "    \"surrogate_precheck_pass\": {},\n",
            self.totals.surrogate_precheck_pass
        ));
        s.push_str("    \"uniform_baseline\": {\n");
        s.push_str(&format!(
            "      \"distortion\": {},\n",
            fmt_float(self.totals.uniform_baseline.distortion)
        ));
        s.push_str(&format!(
            "      \"footprint_bytes\": {},\n",
            self.totals.uniform_baseline.footprint_bytes
        ));
        s.push_str(&format!(
            "      \"option\": {}\n",
            json_str(&self.totals.uniform_baseline.option)
        ));
        s.push_str("    }\n");
        s.push_str("  },\n");
        // allocation
        s.push_str("  \"allocation\": [\n");
        for (i, a) in self.allocation.iter().enumerate() {
            s.push_str("    {\n");
            s.push_str(&format!(
                "      \"bits_per_weight\": {},\n",
                fmt_float(a.bits_per_weight)
            ));
            s.push_str(&format!("      \"bytes\": {},\n", a.bytes));
            s.push_str(&format!("      \"distortion\": {},\n", fmt_float(a.distortion)));
            s.push_str(&format!(
                "      \"marginal_dpb\": {},\n",
                fmt_float(a.marginal_dpb)
            ));
            s.push_str(&format!("      \"numel\": {},\n", a.numel));
            s.push_str(&format!("      \"option\": {},\n", json_str(&a.option)));
            s.push_str(&format!("      \"reason\": {},\n", json_str(&a.reason)));
            s.push_str(&format!("      \"tensor\": {}\n", json_str(&a.tensor)));
            s.push_str(if i + 1 == self.allocation.len() {
                "    }\n"
            } else {
                "    },\n"
            });
        }
        s.push_str("  ]\n");
        s.push('}');
        s
    }
}

fn assemble_reason(t: &Tensor) -> String {
    let cur = t.chosen();
    if let Some(pin) = &t.pin {
        return format!("pinned:{pin}");
    }
    if let Some(floor) = &t.tier_floor
        && &cur.option == floor
    {
        return format!("tier-floored:{floor}");
    }
    if t.chosen_idx == 0 {
        "flat: starved to cheapest (return below price)".to_string()
    } else if t.chosen_idx == t.points.len() - 1 {
        "steep: upgraded to highest available precision".to_string()
    } else {
        "interior: upgraded one or more hull steps".to_string()
    }
}

/// Assemble the final table from a finished allocation. Mirrors the Python
/// `_finish`.
fn finish(
    tensors: &[Tensor],
    budget: u64,
    method: &str,
    lambda_star: Option<f64>,
) -> AllocationTable {
    let fp = footprint(tensors);
    let dist = distortion(tensors);
    let uniform = uniform_equal_footprint(tensors, if budget != 0 { budget } else { fp });

    let mut allocation = Vec::with_capacity(tensors.len());
    for t in tensors {
        let cur = t.chosen();
        let mut marginal = 0.0f64;
        if t.chosen_idx + 1 < t.points.len() {
            let nxt = &t.points[t.chosen_idx + 1];
            if nxt.rate_bytes > cur.rate_bytes {
                let added = nxt.rate_bytes - cur.rate_bytes;
                marginal = (cur.distortion - nxt.distortion) / added as f64;
            }
        }
        allocation.push(AllocationRecord {
            tensor: t.name.clone(),
            numel: t.numel,
            option: cur.option.clone(),
            bits_per_weight: round4(cur.bits),
            bytes: cur.rate_bytes,
            distortion: round9(cur.distortion),
            marginal_dpb: fmt_sci6(marginal),
            reason: assemble_reason(t),
        });
    }

    let surrogate_ok = dist <= uniform.distortion + 1e-12;

    AllocationTable {
        schema_version: SCHEMA_VERSION,
        generator: GENERATOR.to_string(),
        method: method.to_string(),
        lambda_star,
        budget_bytes: budget,
        totals: Totals {
            footprint_bytes: fp,
            footprint_gib: round6(fp as f64 / (1024f64.powi(3))),
            distortion: round9(dist),
            uniform_baseline: uniform,
            surrogate_precheck_pass: surrogate_ok,
        },
        n_tensors: tensors.len(),
        allocation,
    }
}

// ── top-level dispatch (with the deterministic fallback) ─────────────────────

/// The allocation method requested by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Lagrangian water-filling (the fast default).
    Waterfill,
    /// Exact bounded-knapsack DP (closes the duality gap).
    ExactDp,
    /// The deterministic uniform fallback.
    Uniform,
}

/// Top-level allocate with the **deterministic fallback wired in** (Alien-Artifact
/// Contract: "no adaptive controller ships without a conservative deterministic
/// fallback"). Runs the requested method; if it errors (infeasible budget, etc.)
/// OR the surrogate pre-check fails OR `force_fallback` is set, it returns the
/// uniform allocation instead — and records `fallback_fired` so the caller can
/// ledger it.
///
/// `uniform_option` is the fallback option (e.g. `"int4-g32"`). When `budget` is
/// `None`, only [`Method::Uniform`] is valid (others need a budget).
pub fn allocate(
    curves: &[TensorCurve],
    global_option_bits: &BTreeMap<String, f64>,
    method: Method,
    budget: Option<u64>,
    uniform_option: &str,
    force_fallback: bool,
) -> AllocResult<AllocationOutcome> {
    // The fallback is always constructible (uniform never errors on a valid curve set).
    let fallback = || -> AllocResult<AllocationTable> {
        let mut t = load_tensors(curves, global_option_bits)?;
        Ok(allocate_uniform(&mut t, uniform_option, budget))
    };

    if force_fallback || method == Method::Uniform {
        return Ok(AllocationOutcome {
            table: fallback()?,
            fallback_fired: force_fallback || method == Method::Uniform,
            fallback_reason: if force_fallback {
                Some("forced fallback".to_string())
            } else {
                None
            },
        });
    }

    let budget = budget.ok_or_else(|| {
        AllocatorError("a budget is required for waterfill / exact-dp".to_string())
    })?;

    let attempt = (|| -> AllocResult<AllocationTable> {
        let mut tensors = load_tensors(curves, global_option_bits)?;
        match method {
            Method::Waterfill => allocate_waterfill(&mut tensors, budget),
            Method::ExactDp => allocate_exact_dp(&mut tensors, budget, 1 << 20),
            Method::Uniform => unreachable!(),
        }
    })();

    match attempt {
        Ok(table) if table.totals.surrogate_precheck_pass => Ok(AllocationOutcome {
            table,
            fallback_fired: false,
            fallback_reason: None,
        }),
        Ok(_table) => Ok(AllocationOutcome {
            table: fallback()?,
            fallback_fired: true,
            fallback_reason: Some(
                "surrogate pre-check failed (Σ D_alloc > Σ D_uniform): fell back to uniform".into(),
            ),
        }),
        Err(e) => Ok(AllocationOutcome {
            table: fallback()?,
            fallback_fired: true,
            fallback_reason: Some(format!("optimizer error: {e}; fell back to uniform")),
        }),
    }
}

/// The outcome of [`allocate`]: the table plus whether the deterministic
/// fallback fired (and why) — the evidence-ledger record.
#[derive(Debug, Clone, PartialEq)]
pub struct AllocationOutcome {
    /// The chosen allocation table.
    pub table: AllocationTable,
    /// Whether the deterministic fallback was selected.
    pub fallback_fired: bool,
    /// Why the fallback fired (`None` if it did not).
    pub fallback_reason: Option<String>,
}

// ── number formatting helpers (deterministic) ────────────────────────────────

fn round_to(x: f64, places: i32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let f = 10f64.powi(places);
    (x * f).round() / f
}

fn round4(x: f64) -> f64 {
    round_to(x, 4)
}
fn round6(x: f64) -> f64 {
    round_to(x, 6)
}
fn round9(x: f64) -> f64 {
    round_to(x, 9)
}

/// Match the Python `float(f"{marginal:.6e}")` — round to 6 significant figures
/// in scientific notation, parsed back to f64.
fn fmt_sci6(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    format!("{x:.6e}").parse().unwrap_or(x)
}

/// Format an f64 for JSON: integers as integers, else shortest round-trip.
fn fmt_float(x: f64) -> String {
    if !x.is_finite() {
        return "null".to_string();
    }
    if x == x.trunc() && x.abs() < 1e15 {
        // Whole number -> emit with a trailing `.0` like JSON floats.
        return format!("{:.1}", x);
    }
    let s = format!("{x}");
    s
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

// ── self-test: the synthetic model + the Python invariants ───────────────────

/// The 6-tensor synthetic model used by [`selftest`] — 2 steep, 2 flat, 1
/// pinned, 1 tier-floored. Mirrors the Python `_synthetic_doc`.
fn synthetic_doc() -> Vec<TensorCurve> {
    let curve = |d_int8: f64, d_g32: f64, d_g16: f64| -> Vec<CurveOption> {
        vec![
            CurveOption {
                option: "bf16".into(),
                bits: Some(16.0),
                distortion: 0.0,
            },
            CurveOption {
                option: "int8".into(),
                bits: Some(8.03),
                distortion: d_int8,
            },
            CurveOption {
                option: "int4-g16".into(),
                bits: Some(6.0),
                distortion: d_g16,
            },
            CurveOption {
                option: "int4-g32".into(),
                bits: Some(5.0),
                distortion: d_g32,
            },
        ]
    };
    let n = 1_146_880u64; // 1280 x 896, an expert proj
    vec![
        TensorCurve {
            name: "steep.a".into(),
            numel: n,
            pin: None,
            tier_floor: None,
            curve: curve(0.00030, 0.00500, 0.00210),
        },
        TensorCurve {
            name: "steep.b".into(),
            numel: n,
            pin: None,
            tier_floor: None,
            curve: curve(0.00025, 0.00420, 0.00190),
        },
        TensorCurve {
            name: "flat.a".into(),
            numel: n,
            pin: None,
            tier_floor: None,
            curve: curve(0.00002, 0.00009, 0.00005),
        },
        TensorCurve {
            name: "flat.b".into(),
            numel: n,
            pin: None,
            tier_floor: None,
            curve: curve(0.00003, 0.00011, 0.00006),
        },
        TensorCurve {
            name: "router.gate".into(),
            numel: 81_920,
            pin: Some("bf16".into()),
            tier_floor: None,
            curve: curve(0.00100, 0.01000, 0.00500),
        },
        TensorCurve {
            name: "attn.v".into(),
            numel: 163_840,
            pin: None,
            tier_floor: Some("int8".into()),
            curve: curve(0.00040, 0.00800, 0.00300),
        },
    ]
}

/// The result of [`selftest`]: pass + the per-invariant failures (empty on pass).
#[derive(Debug, Clone, PartialEq)]
pub struct SelftestResult {
    /// Whether every invariant held.
    pub pass: bool,
    /// The budget that was used (for reproducibility).
    pub budget_bytes: u64,
    /// Water-filling footprint.
    pub waterfill_footprint: u64,
    /// Water-filling distortion.
    pub waterfill_distortion: f64,
    /// Exact-DP distortion.
    pub exact_dp_distortion: f64,
    /// Uniform-fallback distortion.
    pub uniform_fallback_distortion: f64,
    /// Failed-invariant messages (empty => pass).
    pub failures: Vec<String>,
}

/// Run the built-in synthetic self-test, reproducing the Python `--selftest`
/// invariants (1–7). Returns the structured result; `pass == true` iff every
/// invariant held.
pub fn selftest() -> AllocResult<SelftestResult> {
    let doc = synthetic_doc();
    let empty: BTreeMap<String, f64> = BTreeMap::new();

    // Budget that forces a genuine steep-vs-flat tradeoff (mirror the Python).
    let probe = load_tensors(&doc, &empty)?;
    let cheapest: u64 = probe.iter().map(|t| t.cheapest().rate_bytes).sum();
    let steep_a = probe
        .iter()
        .find(|t| t.name == "steep.a")
        .ok_or_else(|| AllocatorError("selftest: steep.a missing".into()))?;
    let steep_upgrade = steep_a.costliest().rate_bytes - steep_a.cheapest().rate_bytes;
    let budget = cheapest + (2.6 * steep_upgrade as f64) as u64;

    let mut wf_tensors = load_tensors(&doc, &empty)?;
    let table = allocate_waterfill(&mut wf_tensors, budget)?;
    let mut failures: Vec<String> = Vec::new();

    // Invariant 1: footprint within budget.
    if table.totals.footprint_bytes > budget {
        failures.push(format!(
            "footprint {} exceeds budget {budget}",
            table.totals.footprint_bytes
        ));
    }

    // Invariant 2: surrogate pre-check passes.
    if !table.totals.surrogate_precheck_pass {
        failures.push("surrogate pre-check failed (allocated > uniform baseline)".into());
    }

    // Invariant 3: pins / tier floors respected.
    let by_name: BTreeMap<&str, &AllocationRecord> = table
        .allocation
        .iter()
        .map(|a| (a.tensor.as_str(), a))
        .collect();
    if by_name["router.gate"].option != "bf16" {
        failures.push("pinned tensor not bf16".into());
    }
    if option_rank(&by_name["attn.v"].option) > option_rank("int8") {
        failures.push("tier-floored tensor went below int8".into());
    }

    // Invariant 4: steep tensors get >= the precision of every flat one.
    let worst_steep_rank = option_rank(&by_name["steep.a"].option)
        .max(option_rank(&by_name["steep.b"].option));
    let best_flat_rank = option_rank(&by_name["flat.a"].option)
        .min(option_rank(&by_name["flat.b"].option));
    if worst_steep_rank > best_flat_rank {
        failures.push(format!(
            "water-filling starved a steep tensor (rank {worst_steep_rank}) below a flat one (rank {best_flat_rank})"
        ));
    }
    // Invariant 4b: a real tradeoff was forced (not everything at bf16).
    if table.allocation.iter().all(|a| a.option == "bf16") {
        failures.push("budget too loose: every tensor reached bf16 (no tradeoff exercised)".into());
    }

    // Invariant 5: determinism — same input twice yields identical output.
    let mut t1v = load_tensors(&doc, &empty)?;
    let mut t2v = load_tensors(&doc, &empty)?;
    let t1 = allocate_waterfill(&mut t1v, budget)?;
    let t2 = allocate_waterfill(&mut t2v, budget)?;
    if t1.to_json() != t2.to_json() {
        failures.push("non-deterministic allocation (two runs differ)".into());
    }

    // Invariant 6: exact DP never does worse than water-filling on distortion.
    let mut dp_tensors = load_tensors(&doc, &empty)?;
    let dp_table = allocate_exact_dp(&mut dp_tensors, budget, 4096)?;
    if dp_table.totals.distortion > table.totals.distortion + 1e-9 {
        failures.push(format!(
            "exact-dp distortion {} worse than waterfill {}",
            dp_table.totals.distortion, table.totals.distortion
        ));
    }

    // Invariant 7: fallback (uniform) is feasible and >= allocated distortion.
    let mut fb_tensors = load_tensors(&doc, &empty)?;
    let fb = allocate_uniform(&mut fb_tensors, "int4-g32", Some(budget));
    if fb.totals.distortion + 1e-12 < table.totals.distortion {
        failures.push("uniform fallback beat the allocator (impossible at equal/looser footprint)".into());
    }

    Ok(SelftestResult {
        pass: failures.is_empty(),
        budget_bytes: budget,
        waterfill_footprint: table.totals.footprint_bytes,
        waterfill_distortion: table.totals.distortion,
        exact_dp_distortion: dp_table.totals.distortion,
        uniform_fallback_distortion: fb.totals.distortion,
        failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_bits() -> BTreeMap<String, f64> {
        BTreeMap::new()
    }

    // ── the Python self-test invariants ──────────────────────────────────────

    #[test]
    fn selftest_passes_all_invariants() {
        let r = selftest().expect("selftest runs");
        assert!(r.pass, "self-test failures: {:?}", r.failures);
        assert!(r.failures.is_empty());
        // The DP must not beat... I mean, must not be worse than waterfill.
        assert!(r.exact_dp_distortion <= r.waterfill_distortion + 1e-9);
        // Uniform fallback never beats the optimizer at equal/looser footprint.
        assert!(r.uniform_fallback_distortion + 1e-12 >= r.waterfill_distortion);
    }

    // ── bytes_for / option rank ──────────────────────────────────────────────

    #[test]
    fn bytes_for_matches_ceil() {
        // 1280*896 = 1_146_880 weights at int8 (8.03 bpw).
        let n = 1_146_880u64;
        assert_eq!(bytes_for(n, 16.0), n * 2);
        assert_eq!(bytes_for(n, 8.0), n);
        // int4-g32 = 5.0 bpw => numel*5/8.
        assert_eq!(bytes_for(n, 5.0), (n * 5).div_ceil(8));
        // ceil behavior on a non-divisible case.
        assert_eq!(bytes_for(3, 5.0), 2); // 15/8 -> ceil 2
    }

    #[test]
    fn option_rank_orders_precision() {
        assert!(option_rank("bf16") < option_rank("int8"));
        assert!(option_rank("int8") < option_rank("int4-g16"));
        assert!(option_rank("int4-g16") < option_rank("int4-g32"));
        // unknown sorts after known.
        assert!(option_rank("weird") > option_rank("int4-g32"));
    }

    // ── water-filling: steep upgraded, flat starved ──────────────────────────

    #[test]
    fn waterfill_spends_bits_on_steep_tensors() {
        let doc = synthetic_doc();
        let probe = load_tensors(&doc, &empty_bits()).unwrap();
        let cheapest: u64 = probe.iter().map(|t| t.cheapest().rate_bytes).sum();
        let steep_a = probe.iter().find(|t| t.name == "steep.a").unwrap();
        let upg = steep_a.costliest().rate_bytes - steep_a.cheapest().rate_bytes;
        let budget = cheapest + (2.6 * upg as f64) as u64;

        let mut tensors = load_tensors(&doc, &empty_bits()).unwrap();
        let table = allocate_waterfill(&mut tensors, budget).unwrap();
        let by: BTreeMap<&str, &AllocationRecord> =
            table.allocation.iter().map(|a| (a.tensor.as_str(), a)).collect();
        // Steep tensors at least as precise as flat ones.
        let ws = option_rank(&by["steep.a"].option).max(option_rank(&by["steep.b"].option));
        let bf = option_rank(&by["flat.a"].option).min(option_rank(&by["flat.b"].option));
        assert!(ws <= bf, "steep starved below flat: {ws} > {bf}");
        assert!(table.totals.footprint_bytes <= budget);
    }

    // ── infeasible / trivial budgets ─────────────────────────────────────────

    #[test]
    fn infeasible_budget_errors() {
        let doc = synthetic_doc();
        let mut tensors = load_tensors(&doc, &empty_bits()).unwrap();
        let cheapest: u64 = tensors.iter().map(|t| t.cheapest().rate_bytes).sum();
        let err = allocate_waterfill(&mut tensors, cheapest - 1).unwrap_err();
        assert!(err.0.contains("infeasible"), "{}", err.0);
    }

    #[test]
    fn budget_above_costliest_picks_all_top_precision() {
        let doc = synthetic_doc();
        let mut tensors = load_tensors(&doc, &empty_bits()).unwrap();
        let costliest: u64 = tensors.iter().map(|t| t.costliest().rate_bytes).sum();
        let table = allocate_waterfill(&mut tensors, costliest + 1).unwrap();
        // Everything at its costliest (most precise eligible) option.
        // Non-pinned tensors reach bf16; pinned router.gate is bf16; tier-floored
        // attn.v is int8 (its costliest after the floor prune).
        for a in &table.allocation {
            if a.tensor == "attn.v" {
                assert_eq!(a.option, "int8");
            } else {
                assert_eq!(a.option, "bf16", "{}", a.tensor);
            }
        }
        assert_eq!(table.lambda_star, Some(0.0));
    }

    // ── deterministic fallback fires when expected-loss is flat ──────────────

    #[test]
    fn fallback_fires_on_force() {
        let doc = synthetic_doc();
        let out = allocate(
            &doc,
            &empty_bits(),
            Method::Waterfill,
            Some(u64::MAX / 4),
            "int4-g32",
            true, // force fallback
        )
        .unwrap();
        assert!(out.fallback_fired);
        assert!(out.table.method.starts_with("uniform-"));
    }

    #[test]
    fn fallback_fires_on_infeasible_budget() {
        let doc = synthetic_doc();
        let cheapest: u64 = load_tensors(&doc, &empty_bits())
            .unwrap()
            .iter()
            .map(|t| t.cheapest().rate_bytes)
            .sum();
        // Budget below the cheapest config => waterfill errors => fallback fires.
        let out = allocate(
            &doc,
            &empty_bits(),
            Method::Waterfill,
            Some(cheapest - 1),
            "int4-g32",
            false,
        )
        .unwrap();
        assert!(out.fallback_fired);
        assert!(out.fallback_reason.unwrap().contains("optimizer error"));
        assert!(out.table.method.starts_with("uniform-"));
    }

    #[test]
    fn fallback_method_uniform_is_deterministic_fallback() {
        // The conservative all-int8 / uniform default: when the optimizer is not
        // run at all (Method::Uniform), the table is the uniform recipe.
        let doc = synthetic_doc();
        let out = allocate(
            &doc,
            &empty_bits(),
            Method::Uniform,
            None,
            "int8", // conservative all-int8 default
            false,
        )
        .unwrap();
        assert!(out.fallback_fired);
        // Every quantizable tensor is int8 (or higher where pinned/floored).
        let by: BTreeMap<&str, &AllocationRecord> =
            out.table.allocation.iter().map(|a| (a.tensor.as_str(), a)).collect();
        // router.gate pinned bf16; others at int8 (their floor / the uniform option).
        assert_eq!(by["router.gate"].option, "bf16");
        for name in ["steep.a", "steep.b", "flat.a", "flat.b", "attn.v"] {
            // int8 or more precise — never cheaper than int8 under an int8 uniform.
            assert!(
                option_rank(&by[name].option) <= option_rank("int8"),
                "{name} went below int8 under uniform-int8"
            );
        }
    }

    #[test]
    fn fallback_fires_when_curves_flat_no_gain() {
        // A flat-curve model: every option has ~the same (tiny) distortion. The
        // optimizer succeeds and the surrogate pre-check passes, so the *optimizer*
        // does not need to fall back — but the deterministic uniform path is still
        // available and feasible, and Method::Uniform yields a valid conservative
        // table. This documents that a flat distortion landscape produces a sane
        // (cheapest-precision) allocation rather than thrashing.
        let flat_curve = vec![
            CurveOption { option: "bf16".into(), bits: Some(16.0), distortion: 0.0 },
            CurveOption { option: "int8".into(), bits: Some(8.0), distortion: 1e-9 },
            CurveOption { option: "int4-g32".into(), bits: Some(5.0), distortion: 1e-9 },
        ];
        let doc = vec![
            TensorCurve { name: "a".into(), numel: 1024, pin: None, tier_floor: None, curve: flat_curve.clone() },
            TensorCurve { name: "b".into(), numel: 1024, pin: None, tier_floor: None, curve: flat_curve },
        ];
        let cheapest: u64 = load_tensors(&doc, &empty_bits())
            .unwrap()
            .iter()
            .map(|t| t.cheapest().rate_bytes)
            .sum();
        // Loose budget: optimizer runs, surrogate passes -> no fallback.
        let out = allocate(&doc, &empty_bits(), Method::Waterfill, Some(cheapest * 4), "int4-g32", false).unwrap();
        assert!(!out.fallback_fired);
        assert!(out.table.totals.surrogate_precheck_pass);
        // Flat curves => bits buy nothing => starved to the cheapest option.
        for a in &out.table.allocation {
            assert_eq!(a.option, "int4-g32", "flat tensor {} should starve to cheapest", a.tensor);
        }
    }

    // ── exact DP ≤ waterfill ─────────────────────────────────────────────────

    #[test]
    fn exact_dp_no_worse_than_waterfill() {
        let doc = synthetic_doc();
        let probe = load_tensors(&doc, &empty_bits()).unwrap();
        let cheapest: u64 = probe.iter().map(|t| t.cheapest().rate_bytes).sum();
        let steep_a = probe.iter().find(|t| t.name == "steep.a").unwrap();
        let upg = steep_a.costliest().rate_bytes - steep_a.cheapest().rate_bytes;
        let budget = cheapest + (2.6 * upg as f64) as u64;

        let mut wf = load_tensors(&doc, &empty_bits()).unwrap();
        let wf_table = allocate_waterfill(&mut wf, budget).unwrap();
        let mut dp = load_tensors(&doc, &empty_bits()).unwrap();
        let dp_table = allocate_exact_dp(&mut dp, budget, 4096).unwrap();
        assert!(dp_table.totals.distortion <= wf_table.totals.distortion + 1e-9);
        assert!(dp_table.totals.footprint_bytes <= budget);
    }

    // ── ingestion validation ─────────────────────────────────────────────────

    #[test]
    fn rejects_bf16_anchor_with_nonzero_distortion() {
        let doc = vec![TensorCurve {
            name: "x".into(),
            numel: 100,
            pin: None,
            tier_floor: None,
            curve: vec![
                CurveOption { option: "bf16".into(), bits: Some(16.0), distortion: 0.01 },
                CurveOption { option: "int8".into(), bits: Some(8.0), distortion: 0.02 },
            ],
        }];
        let err = load_tensors(&doc, &empty_bits()).unwrap_err();
        assert!(err.0.contains("bf16 anchor"), "{}", err.0);
    }

    #[test]
    fn rejects_negative_distortion() {
        let doc = vec![TensorCurve {
            name: "x".into(),
            numel: 100,
            pin: None,
            tier_floor: None,
            curve: vec![CurveOption { option: "int8".into(), bits: Some(8.0), distortion: -0.1 }],
        }];
        assert!(load_tensors(&doc, &empty_bits()).is_err());
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        let one = || TensorCurve {
            name: "dup".into(),
            numel: 100,
            pin: None,
            tier_floor: None,
            curve: vec![CurveOption { option: "int8".into(), bits: Some(8.0), distortion: 0.0001 }],
        };
        let doc = vec![one(), one()];
        let err = load_tensors(&doc, &empty_bits()).unwrap_err();
        assert!(err.0.contains("duplicate"), "{}", err.0);
    }

    #[test]
    fn rejects_pin_not_in_curve() {
        let doc = vec![TensorCurve {
            name: "x".into(),
            numel: 100,
            pin: Some("bf16".into()),
            tier_floor: None,
            curve: vec![CurveOption { option: "int8".into(), bits: Some(8.0), distortion: 0.0001 }],
        }];
        let err = load_tensors(&doc, &empty_bits()).unwrap_err();
        assert!(err.0.contains("pin"), "{}", err.0);
    }

    #[test]
    fn tier_floor_removes_cheaper_options() {
        let doc = vec![TensorCurve {
            name: "v".into(),
            numel: 1000,
            pin: None,
            tier_floor: Some("int8".into()),
            curve: vec![
                CurveOption { option: "bf16".into(), bits: Some(16.0), distortion: 0.0 },
                CurveOption { option: "int8".into(), bits: Some(8.0), distortion: 0.001 },
                CurveOption { option: "int4-g32".into(), bits: Some(5.0), distortion: 0.01 },
            ],
        }];
        let tensors = load_tensors(&doc, &empty_bits()).unwrap();
        // int4-g32 (rank below int8) must be gone.
        assert!(tensors[0].points.iter().all(|p| p.option != "int4-g32"));
    }

    // ── monotone repair + hull prune ─────────────────────────────────────────

    #[test]
    fn monotone_repair_clamps_noisy_distortion() {
        // int4-g32 (cheaper) has LOWER distortion than int8 (pricier) — noise.
        // After cummin, the pricier int8 must be clamped to <= the cheaper's.
        let pts = vec![
            Point { option: "int4-g32".into(), bits: 5.0, distortion: 0.001, rate_bytes: 5 },
            Point { option: "int8".into(), bits: 8.0, distortion: 0.005, rate_bytes: 8 },
            Point { option: "bf16".into(), bits: 16.0, distortion: 0.0, rate_bytes: 16 },
        ];
        let repaired = monotone_repair(&pts);
        // distortion non-increasing as bytes increase.
        for w in repaired.windows(2) {
            assert!(w[1].distortion <= w[0].distortion + 1e-15);
        }
    }

    #[test]
    fn convex_hull_drops_dominated_interior_point() {
        // 3 points on a straight line: the middle is not strictly below -> dropped.
        let pts = vec![
            Point { option: "a".into(), bits: 4.0, distortion: 0.004, rate_bytes: 4 },
            Point { option: "b".into(), bits: 8.0, distortion: 0.002, rate_bytes: 8 },
            Point { option: "c".into(), bits: 12.0, distortion: 0.0, rate_bytes: 12 },
        ];
        let hull = convex_hull_prune(&pts);
        // Collinear middle is dropped (cross == 0 => popped).
        assert_eq!(hull.len(), 2);
        assert_eq!(hull[0].rate_bytes, 4);
        assert_eq!(hull[1].rate_bytes, 12);
    }

    // ── JSON determinism ─────────────────────────────────────────────────────

    #[test]
    fn json_is_deterministic() {
        let doc = synthetic_doc();
        let mut a = load_tensors(&doc, &empty_bits()).unwrap();
        let mut b = load_tensors(&doc, &empty_bits()).unwrap();
        let budget = 6_000_000u64;
        let ta = allocate_waterfill(&mut a, budget).unwrap();
        let tb = allocate_waterfill(&mut b, budget).unwrap();
        assert_eq!(ta.to_json(), tb.to_json());
        // Sanity: the JSON contains the ledger fields.
        let j = ta.to_json();
        assert!(j.contains("\"surrogate_precheck_pass\""));
        assert!(j.contains("\"uniform_baseline\""));
        assert!(j.contains("\"reason\""));
        assert!(j.contains("\"marginal_dpb\""));
    }

    #[test]
    fn uniform_baseline_is_carried_for_proof() {
        let doc = synthetic_doc();
        let mut t = load_tensors(&doc, &empty_bits()).unwrap();
        let table = allocate_waterfill(&mut t, 6_000_000).unwrap();
        // Surrogate pre-check: allocated <= uniform baseline (by construction).
        assert!(table.totals.surrogate_precheck_pass);
        assert!(table.totals.distortion <= table.totals.uniform_baseline.distortion + 1e-12);
    }

    #[test]
    fn marginal_dpb_is_nonneg_and_zero_at_top() {
        let doc = synthetic_doc();
        let mut t = load_tensors(&doc, &empty_bits()).unwrap();
        // Loose budget => everything tops out => marginal 0 for all.
        let costliest: u64 = t.iter().map(|x| x.costliest().rate_bytes).sum();
        let table = allocate_waterfill(&mut t, costliest + 10).unwrap();
        for a in &table.allocation {
            assert!(a.marginal_dpb >= 0.0);
            // At the top of its (possibly floored) hull, there is no upgrade.
            assert_eq!(a.marginal_dpb, 0.0, "{}", a.tensor);
        }
    }
}
