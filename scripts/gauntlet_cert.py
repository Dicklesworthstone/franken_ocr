#!/usr/bin/env python3
"""Reference implementation of the franken_ocr three-pillar release gauntlet math.

This is the executable backing for ``docs/gauntlet/METHODOLOGY.md`` (beads
VERIFY-three-pillar-cert / VERIFY-conformal-ratchet / VERIFY-eprocess-invariants,
= bd-re8.13/14/15). It is stdlib-only (no numpy, no torch) so it runs in CI
without model weights, mirroring scripts/check_ledgers.py.

It implements, exactly as the methodology specifies:

  (1) The conformance parity score: a per-category Beta posterior, a
      distribution-free conformal band, ``truncate_score`` to 6 dp, and the
      release-decision rule that uses the LOWER bound (never the point estimate).

  (2) The ratchet state machine: Allow / Block / Quarantine / Waiver against a
      persisted, monotone, per-category high-water mark.

  (3) The Ville e-process monitor for the four load-bearing invariants
      (KV-cap L*(m+128), i32 no-overflow at K=6848, same-input determinism,
      SIMD==scalar bit-identical), with the hardware/software p0/lambda/alpha
      calibration split and the arithmetic-mean global e-value.

Nothing here ships in the ``focr`` binary; like the PyO3 oracle bridge this is a
verification-only artifact (G3's no-FFI runtime claim is preserved).

Usage:
    python3 scripts/gauntlet_cert.py --self-test     # validate the math (CI gate)
    python3 scripts/gauntlet_cert.py --demo          # print a worked franken_ocr cert
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------- #
# K-5 / §3.4 — truncate_score: cross-platform LSB determinism.
# x86 vs ARM vs WASM differ at the LSB of IEEE-754 double; truncate (NOT round)
# to 6 dp at every release boundary so the byte-wise ratchet diff never flickers.
# --------------------------------------------------------------------------- #

SCORE_DECIMALS = 6
_SCORE_SCALE = 10.0**SCORE_DECIMALS


def truncate_score(x: float) -> float:
    """Truncate to 6 decimal places (associative across the ULP; round-mode-free)."""
    return math.floor(x * _SCORE_SCALE) / _SCORE_SCALE


# --------------------------------------------------------------------------- #
# §3.1 — Beta posterior per category (Jeffreys-like uniform prior).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BetaParams:
    """A Beta(alpha, beta) posterior over a per-category pass rate."""

    alpha: float
    beta: float

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1.0))

    def quantile(self, p: float) -> float:
        """Quantile of Beta(alpha, beta) via bisection on the regularized
        incomplete beta function (stdlib-only; no scipy)."""
        if not 0.0 < p < 1.0:
            return 0.0 if p <= 0.0 else 1.0
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _reg_inc_beta(mid, self.alpha, self.beta) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def credible_interval(self, confidence: float) -> tuple[float, float]:
        tail = (1.0 - confidence) / 2.0
        return (self.quantile(tail), self.quantile(1.0 - tail))


def _reg_inc_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) via the Lentz continued fraction.

    Standard Numerical-Recipes ``betai`` translated to pure Python; accurate to
    well past the 6-dp truncation floor we round to anyway.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    # Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) for fast CF convergence.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


# --------------------------------------------------------------------------- #
# §3.1 — per-category evidence (present/partial/missing/excluded weighting).
# --------------------------------------------------------------------------- #

PRIOR = BetaParams(alpha=1.0, beta=1.0)  # uniform on [0,1]; declared in the score contract


@dataclass
class CategoryEvidence:
    """Weighted outcome tallies for one FeatureUniverse category."""

    category_id: str
    category_weight: float  # release weight; categories sum to 1.0 (loader-enforced)
    weighted_successes: float = 0.0  # Σ 1.0*w over present rows
    weighted_partials: float = 0.0  # Σ 0.5*w over partial rows (counts in BOTH alpha and beta)
    weighted_failures: float = 0.0  # Σ 1.0*w over missing rows
    # excluded rows are skipped here but their weight stays in the denominator for
    # a strict-100% claim (see §2.1.2); tracked separately for coverage debt.
    weighted_excluded: float = 0.0

    def posterior(self) -> BetaParams:
        return BetaParams(
            alpha=PRIOR.alpha + self.weighted_successes + self.weighted_partials,
            beta=PRIOR.beta + self.weighted_failures + self.weighted_partials,
        )


def add_outcome(ev: CategoryEvidence, status: str, weight: float) -> None:
    """Apply one feature outcome to a category's evidence (§3.1 scoring rule)."""
    if status == "present":
        ev.weighted_successes += 1.0 * weight
    elif status == "partial":
        ev.weighted_partials += 0.5 * weight
    elif status == "missing":
        ev.weighted_failures += 1.0 * weight
    elif status == "excluded":
        ev.weighted_excluded += 1.0 * weight
    else:
        raise ValueError(f"unknown status {status!r}")


# --------------------------------------------------------------------------- #
# §3.2/§3.3 — distribution-free conformal band; release uses the LOWER bound.
# --------------------------------------------------------------------------- #


def conformal_halfwidth(residuals: Iterable[float], confidence: float) -> float:
    """(1 - alpha) empirical quantile of held-out nonconformity residuals
    (Vovk-Gammerman-Shafer 2005). Distribution-free finite-sample coverage."""
    rs = sorted(residuals)
    if not rs:
        # Bootstrap with a wide band before residuals exist (§3.2 pitfall guard).
        return 1.0
    n = len(rs)
    k = math.ceil(confidence * (n + 1)) - 1  # 0-indexed (1-α)-quantile, α = 1-confidence
    k = max(0, min(k, n - 1))
    return rs[k]


@dataclass
class Scorecard:
    """The output of the conformance parity score for one gauntlet round."""

    point_estimate: float  # S_mean (dashboards only)
    lower_bound: float  # S_lower = truncate_score(S_mean - q); THE release number
    conformal_halfwidth: float
    per_category_lower: dict[str, float]  # truncate_score'd; ratchet compares these
    per_category_mean: dict[str, float]


def compute_scorecard(
    categories: list[CategoryEvidence],
    residuals: Iterable[float],
    confidence: float = 0.95,
) -> Scorecard:
    """Beta posterior per category -> weighted mean -> conformal band -> lower bound."""
    total_w = sum(c.category_weight for c in categories)
    if abs(total_w - 1.0) > 1e-9:
        raise ValueError(
            f"category weights must sum to 1.0 (loader invariant); got {total_w!r}"
        )
    q = conformal_halfwidth(residuals, confidence)
    per_mean: dict[str, float] = {}
    per_lower: dict[str, float] = {}
    s_mean = 0.0
    for c in categories:
        post = c.posterior()
        mean_c = post.mean()
        per_mean[c.category_id] = truncate_score(mean_c)
        per_lower[c.category_id] = truncate_score(max(0.0, mean_c - q))
        s_mean += c.category_weight * mean_c
    s_lower = truncate_score(max(0.0, s_mean - q))
    return Scorecard(
        point_estimate=truncate_score(s_mean),
        lower_bound=s_lower,
        conformal_halfwidth=q,
        per_category_lower=per_lower,
        per_category_mean=per_mean,
    )


# --------------------------------------------------------------------------- #
# §3.5 — the ratchet state machine (Allow | Block | Quarantine | Waiver).
# --------------------------------------------------------------------------- #

QUARANTINE_DELTA = 0.005  # a single per-category dip ≤ this -> Quarantine, not Block


@dataclass
class RatchetState:
    current_lower_bound: float
    per_category_bounds: dict[str, float]


@dataclass
class RatchetVerdict:
    decision: str  # "Allow" | "Block" | "Quarantine" | "Waiver"
    reason: str
    new_state: RatchetState | None = None


def apply_ratchet(
    scorecard: Scorecard,
    state: RatchetState,
    waived_categories: frozenset[str] = frozenset(),
) -> RatchetVerdict:
    """Decide whether this round may land, per §3.5. Release uses the LOWER bound."""
    dipped: list[str] = []
    worst_dip = 0.0
    for cat, floor in state.per_category_bounds.items():
        cur = scorecard.per_category_lower.get(cat, 0.0)
        if cur < floor and cat not in waived_categories:
            dipped.append(cat)
            worst_dip = max(worst_dip, floor - cur)

    global_ok = scorecard.lower_bound >= state.current_lower_bound

    if waived_categories and dipped == []:
        # all dips covered by waivers (none remain after filtering)
        pass

    if not dipped and global_ok:
        new = RatchetState(
            current_lower_bound=max(scorecard.lower_bound, state.current_lower_bound),
            per_category_bounds={
                cat: max(scorecard.per_category_lower.get(cat, 0.0), floor)
                for cat, floor in state.per_category_bounds.items()
            },
        )
        return RatchetVerdict("Allow", "raises lower bound; no per-category regression", new)

    if dipped and all(c in waived_categories for c in dipped):
        return RatchetVerdict(
            "Waiver", f"regression in {dipped} covered by active waiver", None
        )

    if global_ok and len(dipped) == 1 and worst_dip <= QUARANTINE_DELTA:
        return RatchetVerdict(
            "Quarantine",
            f"single per-category dip {dipped[0]} by {worst_dip:.6f} ≤ {QUARANTINE_DELTA}",
            None,
        )

    return RatchetVerdict(
        "Block",
        f"lower bound or per-category bound regressed (global_ok={global_ok}, dipped={dipped})",
        None,
    )


# --------------------------------------------------------------------------- #
# §6 — Ville e-processes for the four load-bearing invariants.
# Howard-Ramdas-McAuliffe-Sekhon 2021; Ville's inequality gives
# P_{H_0}(exists t: E_t >= 1/alpha) <= alpha  (anytime-valid; no Bonferroni).
# --------------------------------------------------------------------------- #

# Calibration split (§6.2): hardware-enforced (CPU integer semantics / determinism
# flag; a violation is a CPU/logic bug, tight prior) vs software-enforced (code
# path; rare but plausible, looser prior).
HARDWARE = dict(p0=1e-9, lam=0.999, alpha=1e-6)  # threshold 1/alpha = 1_000_000
SOFTWARE = dict(p0=1e-6, lam=0.9, alpha=1e-3)  # threshold 1/alpha = 1_000

# The four franken_ocr invariants, line-backed to the truth pack / plan §5.4.
INVARIANT_CALIBRATION = {
    "INV-KV-CAP": SOFTWARE,  # KV never exceeds L*(m+128); m_max=32896, W=128, L=12 (CENSUS d)
    "INV-I32-NOOVERFLOW": SOFTWARE,  # i32 acc never overflows at K_max=6848 (plan §5.4, ≥9x headroom)
    "INV-DETERMINISM": HARDWARE,  # same input twice -> byte-identical output
    "INV-SIMD-SCALAR": HARDWARE,  # SDOT/SMMLA/VNNI == scalar bit-identical (integer add associative)
}


@dataclass
class EProcess:
    """An anytime-valid e-process over one invariant's observation stream."""

    p0: float
    lam: float
    alpha: float
    e_value: float = 1.0
    obs_count: int = 0
    rejected_at: int | None = None

    @classmethod
    def for_invariant(cls, name: str) -> "EProcess":
        cal = INVARIANT_CALIBRATION[name]
        return cls(p0=cal["p0"], lam=cal["lam"], alpha=cal["alpha"])

    def observe(self, alarm: bool) -> bool:
        """Feed one observation (alarm == invariant violated). Returns True the
        first time the Ville threshold is crossed."""
        self.obs_count += 1
        if alarm:
            factor = (1.0 - self.lam) + self.lam * 1.0 / self.p0
        else:
            factor = 1.0 - self.lam  # (1-lam) + lam*0/p0
        self.e_value *= factor
        # Saturate to keep f64 noise away from the threshold logic; do NOT reset
        # to 1.0 (that breaks the supermartingale property / Ville's bound).
        self.e_value = min(max(self.e_value, 5e-324), sys.float_info.max / 2.0)
        if self.e_value >= 1.0 / self.alpha and self.rejected_at is None:
            self.rejected_at = self.obs_count
            return True
        return False

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha


def global_e_value(processes: Iterable[EProcess]) -> float:
    """Arithmetic mean of per-invariant e-values -- itself an e-process under the
    global null REGARDLESS of dependence (§6.1). Never max, never product."""
    ps = list(processes)
    if not ps:
        return 1.0
    return sum(p.e_value for p in ps) / len(ps)


# --------------------------------------------------------------------------- #
# A worked franken_ocr scorecard (illustrative; matches METHODOLOGY §3.6).
# --------------------------------------------------------------------------- #


def _demo_categories() -> list[CategoryEvidence]:
    """A plausible mid-maturity round: most categories strong, R-SWA mid-climb.

    Weights mirror METHODOLOGY §2.2. Tallies are illustrative (the real numbers
    come from FEATURE_PARITY.md once kernels land), but they demonstrate the
    full Beta + conformal + ratchet pipeline end to end.
    """
    return [
        _cat("Preprocess", 0.12, present=17, partial=0, missing=1),
        _cat("Tokenizer", 0.10, present=4, partial=0, missing=0),
        _cat("VisionSAM", 0.10, present=9, partial=1, missing=0),
        _cat("VisionCLIP", 0.08, present=8, partial=0, missing=1),
        _cat("Connector", 0.08, present=7, partial=0, missing=0),
        _cat("DecoderMoE", 0.16, present=13, partial=1, missing=0),
        _cat("RSWA", 0.14, present=7, partial=1, missing=1),
        _cat("SamplerPost", 0.10, present=8, partial=0, missing=1),
        _cat("OpMapFacade", 0.06, present=19, partial=1, missing=0),
        _cat("QuantRecipe", 0.06, present=8, partial=0, missing=1),
    ]


def _cat(name: str, weight: float, present: int, partial: int, missing: int) -> CategoryEvidence:
    # equal-weight within the category (default rubric) so per-row weight = 1/n_rows
    n = present + partial + missing
    w = 1.0 / n if n else 0.0
    ev = CategoryEvidence(name, weight)
    for _ in range(present):
        add_outcome(ev, "present", w)
    for _ in range(partial):
        add_outcome(ev, "partial", w)
    for _ in range(missing):
        add_outcome(ev, "missing", w)
    return ev


def _demo() -> int:
    cats = _demo_categories()
    # Held-out residuals from prior cycles (|observed - Beta-mean|); illustrative.
    residuals = [0.012, 0.018, 0.021, 0.027, 0.031, 0.038, 0.041, 0.047, 0.052, 0.061]
    sc = compute_scorecard(cats, residuals, confidence=0.95)
    # A prior round's persisted high-water mark below this round's bounds, so this
    # round is an Allow that advances the ratchet (the instructive default case).
    state = RatchetState(
        current_lower_bound=0.540000,
        per_category_bounds={c.category_id: 0.540000 for c in cats},
    )
    verdict = apply_ratchet(sc, state)
    print(
        json.dumps(
            {
                "artifact": "franken_ocr.gauntlet.scorecard.v1",
                "point_estimate": sc.point_estimate,
                "parity_score_lower_bound": sc.lower_bound,
                "conformal_halfwidth": round(sc.conformal_halfwidth, 6),
                "per_category_lower": sc.per_category_lower,
                "ratchet_decision": verdict.decision,
                "ratchet_reason": verdict.reason,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


# --------------------------------------------------------------------------- #
# Self-test (CI gate; mirrors scripts/check_test_logs.py --self-test style).
# --------------------------------------------------------------------------- #


def emit(check: str, ok: bool, **fields: object) -> None:
    print(json.dumps({"check": check, "result": "pass" if ok else "fail", **fields}, sort_keys=True))


def _approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, **fields: object) -> None:
        emit(name, cond, **fields)
        if not cond:
            failures.append(name)

    # --- truncate_score: truncates, never rounds; idempotent; cross-arch stable.
    check("truncate_score_truncates", _approx(truncate_score(0.8472919), 0.847291))
    check("truncate_score_no_round_up", _approx(truncate_score(0.9999999), 0.999999))
    check("truncate_score_idempotent", truncate_score(truncate_score(0.123456789)) == truncate_score(0.123456789))

    # --- Beta posterior mean + quantile sanity (Beta(201,1) ~ 0.9950 mean).
    post = BetaParams(201.0, 1.0)
    check("beta_mean", _approx(post.mean(), 201.0 / 202.0), mean=post.mean())
    lo, hi = post.credible_interval(0.95)
    check("beta_ci_orders", 0.0 < lo < post.mean() < hi <= 1.0, lo=round(lo, 6), hi=round(hi, 6))
    # regularized incomplete beta endpoints
    check("reg_inc_beta_0", _reg_inc_beta(0.0, 2.0, 3.0) == 0.0)
    check("reg_inc_beta_1", _reg_inc_beta(1.0, 2.0, 3.0) == 1.0)
    check("reg_inc_beta_half_symmetry", _approx(_reg_inc_beta(0.5, 3.0, 3.0), 0.5, 1e-6))

    # --- partial counts in BOTH alpha and beta (§3.1 pitfall).
    ev = CategoryEvidence("c", 1.0)
    add_outcome(ev, "partial", 1.0)
    p = ev.posterior()
    check("partial_both_sides", _approx(p.alpha, 1.5) and _approx(p.beta, 1.5), alpha=p.alpha, beta=p.beta)
    # a present-only category beats a missing-only category
    ev_pass = _cat("pass", 1.0, present=10, partial=0, missing=0)
    ev_fail = _cat("fail", 1.0, present=0, partial=0, missing=10)
    check("present_beats_missing", ev_pass.posterior().mean() > ev_fail.posterior().mean())

    # --- conformal half-width is the (1-alpha) empirical quantile.
    res = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    q95 = conformal_halfwidth(res, 0.95)
    check("conformal_quantile_in_range", res[0] <= q95 <= res[-1], q=q95)
    check("conformal_empty_bootstraps_wide", conformal_halfwidth([], 0.95) == 1.0)
    # tighter confidence -> wider (larger) half-width
    check("conformal_tighter_wider", conformal_halfwidth(res, 0.99) >= conformal_halfwidth(res, 0.90))

    # --- weights must sum to 1.0 (loader invariant).
    bad = [CategoryEvidence("a", 0.4), CategoryEvidence("b", 0.4)]
    try:
        compute_scorecard(bad, [0.01], 0.95)
        check("weights_sum_enforced", False)
    except ValueError:
        check("weights_sum_enforced", True)

    # --- release uses the LOWER bound, not the point estimate.
    sc = compute_scorecard(_demo_categories(), res, 0.95)
    check("lower_below_point", sc.lower_bound <= sc.point_estimate, lower=sc.lower_bound, point=sc.point_estimate)
    check("lower_is_truncated", truncate_score(sc.lower_bound) == sc.lower_bound)

    # --- ratchet: Allow raises both bounds; Block on regression; Quarantine on small single dip.
    state = RatchetState(0.5, {c.category_id: 0.5 for c in _demo_categories()})
    v_allow = apply_ratchet(sc, state)
    check("ratchet_allows_progress", v_allow.decision == "Allow", reason=v_allow.reason)
    check("ratchet_monotone", v_allow.new_state.current_lower_bound >= state.current_lower_bound)

    high = RatchetState(0.999999, {c.category_id: 0.999999 for c in _demo_categories()})
    v_block = apply_ratchet(sc, high)
    check("ratchet_blocks_regression", v_block.decision == "Block", reason=v_block.reason)

    # single tiny per-category dip with global holding -> Quarantine
    qstate = RatchetState(
        current_lower_bound=0.0,
        per_category_bounds={c.category_id: sc.per_category_lower[c.category_id] for c in _demo_categories()},
    )
    one = next(iter(qstate.per_category_bounds))
    qstate.per_category_bounds[one] = sc.per_category_lower[one] + 0.003  # dip 0.003 ≤ 0.005
    v_q = apply_ratchet(sc, qstate)
    check("ratchet_quarantines_small_dip", v_q.decision == "Quarantine", reason=v_q.reason)
    # ...unless a waiver covers it
    v_w = apply_ratchet(sc, qstate, waived_categories=frozenset({one}))
    check("ratchet_waiver_covers_dip", v_w.decision in ("Allow", "Waiver"), decision=v_w.decision)

    # --- e-process calibration: all four invariants registered, split correct.
    check("four_invariants_registered", set(INVARIANT_CALIBRATION) == {
        "INV-KV-CAP", "INV-I32-NOOVERFLOW", "INV-DETERMINISM", "INV-SIMD-SCALAR"
    })
    check("hardware_invariants_tight", INVARIANT_CALIBRATION["INV-SIMD-SCALAR"]["p0"] == 1e-9
          and INVARIANT_CALIBRATION["INV-DETERMINISM"]["p0"] == 1e-9)
    check("software_invariants_loose", INVARIANT_CALIBRATION["INV-KV-CAP"]["p0"] == 1e-6
          and INVARIANT_CALIBRATION["INV-I32-NOOVERFLOW"]["p0"] == 1e-6)

    # --- Ville: a healthy stream never rejects; a violation burst does.
    ep = EProcess.for_invariant("INV-SIMD-SCALAR")  # hardware, threshold 1e6
    for _ in range(10000):
        ep.observe(False)
    check("eprocess_healthy_no_reject", ep.rejected_at is None and ep.e_value < ep.threshold)
    # a single hardware violation is consistent with the null (does NOT reject)
    ep_single = EProcess.for_invariant("INV-SIMD-SCALAR")
    for _ in range(500):
        ep_single.observe(False)
    fired_single = ep_single.observe(True)
    check("eprocess_single_hw_violation_no_reject", not fired_single and ep_single.rejected_at is None)
    # a burst of hardware violations rejects within a handful of observations
    ep_burst = EProcess.for_invariant("INV-SIMD-SCALAR")
    fired = False
    for i in range(8):
        if ep_burst.observe(True):
            fired = True
            break
    check("eprocess_burst_rejects", fired and ep_burst.rejected_at is not None and ep_burst.rejected_at <= 4)

    # software invariant rejects on ~2 consecutive violations (threshold 1e3)
    ep_soft = EProcess.for_invariant("INV-KV-CAP")
    soft_fired = False
    for _ in range(3):
        if ep_soft.observe(True):
            soft_fired = True
            break
    check("eprocess_software_rejects_fast", soft_fired and ep_soft.rejected_at is not None and ep_soft.rejected_at <= 2)
    # ...and the e-VALUE decays back toward 0 under sustained health: 2 sporadic
    # violations diluted by 18 healthy observations leave E_t far below 1.0 (the
    # §4 worked figure ~8.1e-7), even though rejected_at latches the first crossing.
    # (A single software alarm DOES cross 1e3 -- 9e5 > 1e3 -- so the latch fires;
    # the anytime-valid guarantee is about the e-VALUE trajectory, which decays.)
    ep_dilute = EProcess.for_invariant("INV-KV-CAP")
    ep_dilute.observe(True)
    for _ in range(18):
        ep_dilute.observe(False)
    ep_dilute.observe(True)
    check("eprocess_software_evalue_decays", ep_dilute.e_value < 1e-3, e=ep_dilute.e_value)

    # --- global e-value is the arithmetic mean and stays below threshold even if
    # one invariant individually crossed (family-wise guarantee, §6.1).
    procs = [EProcess.for_invariant(n) for n in INVARIANT_CALIBRATION]
    procs[0].e_value = 1e6  # one invariant crossed on its own
    g = global_e_value(procs)
    check("global_e_arithmetic_mean", _approx(g, (1e6 + 1.0 + 1.0 + 1.0) / 4.0, 1.0), g=g)
    check("global_e_below_hw_threshold", g < 1.0 / HARDWARE["alpha"], g=g)

    # --- never-reset discipline: e_value is not forced back to 1.0 on saturation.
    ep_sat = EProcess.for_invariant("INV-DETERMINISM")
    for _ in range(5):
        ep_sat.observe(True)
    check("eprocess_no_reset", ep_sat.e_value > 1.0)

    if failures:
        emit("gauntlet-cert-self-test", False, failed=failures)
        return 1
    emit("gauntlet-cert-self-test", True, checks_passed=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="validate the gauntlet math (CI gate)")
    parser.add_argument("--demo", action="store_true", help="print a worked franken_ocr scorecard + ratchet verdict")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.demo:
        return _demo()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
