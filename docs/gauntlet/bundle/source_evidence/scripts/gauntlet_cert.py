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
    # bd-re8.13 real-data modes:
    python3 scripts/gauntlet_cert.py --from-parity docs/FEATURE_PARITY.md \
        --scorecard-out docs/gauntlet/RELEASE_SCORECARD.json   # score the REAL scoreboard
    python3 scripts/gauntlet_cert.py --convergence   # >=10 rounds / >=2 clean gate (exit 1 until met)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

# --------------------------------------------------------------------------- #
# K-5 / §3.4 — truncate_score: cross-platform LSB determinism.
# x86 vs ARM vs WASM differ at the LSB of IEEE-754 double; truncate (NOT round)
# to 6 dp at every release boundary so the byte-wise ratchet diff never flickers.
# --------------------------------------------------------------------------- #

SCORE_DECIMALS = 6
_SCORE_SCALE = 10.0**SCORE_DECIMALS

CERTIFICATION_MAX_EVIDENCE_AGE_HOURS = 24.0
UNLIMITED_OCR_MODEL_COMMIT = "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"
STRICT_CERTIFICATE_SCHEMA = "strict-conformant-release.v1"
STRICT_CERTIFICATE_ARTIFACT = "franken_ocr.release_certificate.v1"
STRICT_BUNDLE_SCHEMA = "gauntlet.certification_bundle_manifest.v1"
STRICT_BUNDLE_ARTIFACT = "franken_ocr.certification_bundle.v1"
CERTIFICATION_REQUIRED_SIGNERS = 3
CERTIFICATION_REQUIRED_SIGNATURE_ROLES = {
    "producer",
    "independent-reviewer",
    "release-authorizer",
}
MAX_CORRECTNESS_CER = 0.25
CI_ARTIFACT_BINDING_SCHEMA = "gauntlet.ci_artifact_file.v1"
CERTIFICATION_GITHUB_REPOSITORY = "Dicklesworthstone/franken_ocr"
CERTIFICATION_GITHUB_WORKFLOW = "CI"
CERTIFICATION_GITHUB_EVENT = "push"
TRUSTED_SIGNERS_PATH = "docs/gauntlet/TRUSTED_SIGNERS.json"
TRUSTED_KEYRING_PATH = "docs/gauntlet/signing/trusted-release-keys.gpg"
HYPOTHESIS_LEDGER_PATHS = (
    "docs/gauntlet/GAUNTLET_EXPERIMENT_DESIGNS.md",
    "docs/gauntlet/PERF_HYPOTHESIS_LEDGER.md",
    "docs/gauntlet/CONFORMANCE_HYPOTHESIS_LEDGER.md",
    "docs/gauntlet/SURFACE_PARITY_HYPOTHESIS_LEDGER.md",
)
CERTIFICATION_READINESS_CELLS = (
    "parity_l0_l5",
    "surface_parity",
    "perf_vs_reference",
    "determinism",
    "deadlock_watchdog",
    "robot_schema",
    "build_matrix",
    "installer",
    "ledger_completeness",
    "hypothesis_ledgers",
    "agent_ergonomics",
    "doctor",
    "certification_bundle",
    "gauntlet_convergence",
)
CERTIFICATION_EXTERNAL_READINESS_CELLS = tuple(
    cell for cell in CERTIFICATION_READINESS_CELLS if cell != "certification_bundle"
)
CERTIFICATION_READINESS_EVIDENCE_PATHS = {
    "parity_l0_l5": ("tests/fixtures/ladder_scorecard/scorecard_armed.json",),
    "surface_parity": ("docs/FEATURE_PARITY.md",),
    "perf_vs_reference": ("docs/PERF_LEDGER.md", "scripts/check_ledgers.py"),
    "determinism": ("docs/gauntlet/EPROCESS_STATE.json",),
    "deadlock_watchdog": (
        "tests/many_pages_without_deadlock.rs",
        "tests/cancel_and_panic_faults.rs",
    ),
    "robot_schema": (
        "tests/fixtures/robot_schema_v1.json",
        "tests/fixtures/runs_schema.json",
        "tests/cli_robot_golden.rs",
        "tests/surface_matrix.rs",
    ),
    "build_matrix": (".github/workflows/dist.yml",),
    "installer": ("install.sh",),
    "ledger_completeness": (
        "scripts/check_ledgers.py",
        "docs/DISCREPANCIES.md",
        "docs/NEGATIVE_EVIDENCE.md",
        "docs/PERF_LEDGER.md",
    ),
    "hypothesis_ledgers": HYPOTHESIS_LEDGER_PATHS,
    "agent_ergonomics": (
        "docs/ergonomics/AUDIT.md",
        "tests/agent_ergonomics_regression.rs",
    ),
    "doctor": ("src/doctor.rs", "tests/doctor_fixtures.rs"),
    "gauntlet_convergence": ("docs/gauntlet/ROUNDS.jsonl", *HYPOTHESIS_LEDGER_PATHS),
}
CERTIFICATION_PROOF_EVIDENCE_PATHS = tuple(
    dict.fromkeys(
        path
        for cell in CERTIFICATION_EXTERNAL_READINESS_CELLS
        for path in CERTIFICATION_READINESS_EVIDENCE_PATHS[cell]
    )
)
CERTIFICATION_FEATURE_UNIVERSE = (
    {
        "feature_id": "conformance",
        "required": True,
        "proof_obligations": ["parity_l0_l5", "surface_parity"],
    },
    {
        "feature_id": "performance",
        "required": True,
        "proof_obligations": ["perf_vs_reference"],
    },
    {
        "feature_id": "runtime-invariants",
        "required": True,
        "proof_obligations": ["determinism", "deadlock_watchdog"],
    },
    {
        "feature_id": "agent-contract",
        "required": True,
        "proof_obligations": ["robot_schema", "agent_ergonomics", "doctor"],
    },
    {
        "feature_id": "distribution",
        "required": True,
        "proof_obligations": ["build_matrix", "installer"],
    },
    {
        "feature_id": "evidence-integrity",
        "required": True,
        "proof_obligations": [
            "ledger_completeness",
            "hypothesis_ledgers",
            "gauntlet_convergence",
        ],
    },
)
CERTIFICATION_READINESS_CELL_SET_SHA256 = hashlib.sha256(
    json.dumps(
        CERTIFICATION_READINESS_CELLS,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()
CERTIFICATION_FEATURE_UNIVERSE_SHA256 = hashlib.sha256(
    json.dumps(
        CERTIFICATION_FEATURE_UNIVERSE,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()
CERTIFICATION_CONSTANTS = {
    "CERTIFICATION_MIN_VERIFICATION_PCT": 100.0,
    "CERTIFICATION_REQUIRED_SUITE_PASS_RATE_PCT": 100.0,
    "CERTIFICATION_MAX_HIGH_SEVERITY_COUNTEREXAMPLES": 0,
    "CERTIFICATION_MAX_EVIDENCE_AGE_HOURS": CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
}
CORE_EVIDENCE_PATHS = (
    "docs/gauntlet/RELEASE_READINESS.json",
    "docs/gauntlet/ROUNDS.jsonl",
    "docs/gauntlet/EPROCESS_STATE.json",
    "docs/gauntlet/RELEASE_SCORECARD.json",
    "tests/fixtures/ladder_scorecard/scorecard_armed.json",
    "benches/.bench-history/baseline.json",
    "docs/DISCREPANCIES.md",
    "docs/NEGATIVE_EVIDENCE.md",
    "docs/PERF_LEDGER.md",
    "docs/FEATURE_PARITY.md",
    "scripts/gauntlet_cert.py",
    "scripts/gauntlet_row.py",
    "scripts/gauntlet_reference.py",
    *HYPOTHESIS_LEDGER_PATHS,
)
EVIDENCE_MANIFEST_NAMES = (
    "SHA256SUMS",
    "SHA256SUMS.txt",
    "sha256sums.txt",
    "sha256.txt",
    "manifest.sha256",
    "manifest.json",
)
STRICT_BUNDLE_CLASSES = {
    "confidence_gate": ("confidence_gate.json", "confidence_gate.v1"),
    "verification_contract": (
        "verification_contract.json",
        "gauntlet.verification_contract.v1",
    ),
    "ci_manifest": ("ci_manifest.json", "gauntlet.ci_artifact_manifest.v1"),
    "benchmark_summary": ("benchmark_summary.json", "gauntlet.benchmark_summary.v1"),
    "scorecards": ("scorecards.json", "gauntlet.scorecards.v1"),
    "critical_path_report": (
        "critical_path_report.json",
        "gauntlet.critical_path_report.v1",
    ),
    "ratchet_state": ("ratchet_state.json", "ratchet_state.v1"),
}
_RETRY_CONDITION_PREFIXES = (
    "retry only if a profiler attributes a clearly-above-noise share to ",
    "reconsider only inside the broader ",
    "worth reconsidering when ",
    "not worth retrying as a standalone patch",
    "do not retry from a cold read; use comprehensive-bench attribution instead",
    "retry condition not applicable",
    "retry only if this workload class exhibits measurable ",
    "blocked until ",
)
_CORRECTNESS_CLAIM_RE = re.compile(
    r"receipt=correctness_receipt\.json "
    r"sha256=(?P<sha256>[0-9a-f]{64}) "
    r"metric=cer_norm value=(?P<value>[0-9]+\.[0-9]{6}) "
    r"max=(?P<maximum>[0-9]+\.[0-9]{6}) result=pass"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_output(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_repo_path(root: Path, relative: str) -> Path | None:
    try:
        candidate = (root / relative).resolve()
    except (OSError, RuntimeError):
        return None
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _safe_output_dir(root: Path, out_dir: str) -> tuple[Path | None, list[str]]:
    reasons: list[str] = []
    raw = Path(out_dir)
    candidate = raw if raw.is_absolute() else root / raw
    current = candidate
    while current != root and current != current.parent:
        if current.is_symlink():
            reasons.append(f"bundle output path traverses a symlink: {current}")
            break
        current = current.parent
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as error:
        reasons.append(f"bundle output path cannot be resolved: {error}")
        return None, reasons
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        reasons.append("bundle output path escapes the repository")
    git_dir = (root / ".git").resolve()
    if resolved == root.resolve() or resolved == git_dir or git_dir in resolved.parents:
        reasons.append("bundle output path overlaps the repository root or .git")
    if resolved.exists() and not resolved.is_dir():
        reasons.append("bundle output path collides with a non-directory")
    elif resolved.is_dir():
        try:
            for child in resolved.rglob("*"):
                if child.is_symlink():
                    reasons.append(f"bundle output contains a symlink: {child}")
        except OSError as error:
            reasons.append(f"bundle output directory cannot be inspected: {error}")
    for relative in CORE_EVIDENCE_PATHS:
        evidence = _safe_repo_path(root, relative)
        if evidence is not None and (
            resolved == evidence or resolved in evidence.parents
        ):
            reasons.append(
                f"bundle output path would overwrite input evidence: {relative}"
            )
    return (None if reasons else resolved), reasons


def _git_worktree_state(root: Path) -> dict:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        message = f"git status failed: {error}"
        return {
            "status_ok": False,
            "clean": False,
            "dirty_path_count": None,
            "status_porcelain": [],
            "status_sha256": _sha256_text(message),
            "status_error": message,
        }
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "git status failed").strip()
        return {
            "status_ok": False,
            "clean": False,
            "dirty_path_count": None,
            "status_porcelain": [],
            "status_sha256": _sha256_text(message),
            "status_error": message,
        }
    lines = result.stdout.splitlines()
    return {
        "status_ok": True,
        "clean": not lines,
        "dirty_path_count": len(lines),
        "status_porcelain": lines,
        "status_sha256": _sha256_text("\n".join(lines) + ("\n" if lines else "")),
    }


def _cargo_package_version(root: Path) -> str | None:
    try:
        with (root / "Cargo.toml").open("rb") as handle:
            package = tomllib.load(handle).get("package")
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError):
        return None
    if not isinstance(package, dict):
        return None
    version = package.get("version")
    return version if isinstance(version, str) and version else None


def _valid_git_head(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _timestamp_text(parsed: datetime) -> str:
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _artifact_native_timestamp(
    path: Path, content: bytes | None = None
) -> tuple[datetime, str]:
    """Return a native structured timestamp, using mtime only for plain files."""
    timestamp_keys = (
        "issued_at",
        "generated_at_utc",
        "generated_utc",
        "created_utc",
        "timestamp",
        "date",
    )
    if path.suffix == ".json":
        try:
            payload = json.loads(
                content.decode("utf-8")
                if content is not None
                else path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"JSON artifact is unreadable: {error}") from error
        if isinstance(payload, dict):
            native_timestamps: list[tuple[str, datetime]] = []
            for key in timestamp_keys:
                if key not in payload:
                    continue
                parsed = _parse_utc_timestamp(payload.get(key))
                if parsed is None:
                    raise ValueError(
                        f"JSON artifact has invalid native timestamp: {key}"
                    )
                native_timestamps.append((key, parsed))
            if native_timestamps:
                distinct = {parsed for _key, parsed in native_timestamps}
                if len(distinct) != 1:
                    raise ValueError("JSON artifact has conflicting native timestamps")
                keys = "+".join(key for key, _parsed in native_timestamps)
                return native_timestamps[0][1], f"json:{keys}"
        raise ValueError("JSON artifact has no valid native timestamp")
    if path.suffix == ".jsonl":
        try:
            native: list[datetime] = []
            text = (
                content.decode("utf-8")
                if content is not None
                else path.read_text(encoding="utf-8")
            )
            for line in text.splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("JSONL row is not an object")
                row_timestamps: list[tuple[str, datetime]] = []
                for key in timestamp_keys:
                    if key not in payload:
                        continue
                    parsed = _parse_utc_timestamp(payload.get(key))
                    if parsed is None:
                        raise ValueError(
                            f"JSONL row has invalid native timestamp: {key}"
                        )
                    row_timestamps.append((key, parsed))
                if not row_timestamps:
                    raise ValueError("JSONL row has no valid native timestamp")
                if len({parsed for _key, parsed in row_timestamps}) != 1:
                    raise ValueError("JSONL row has conflicting native timestamps")
                native.append(row_timestamps[0][1])
            if native:
                return max(native), "jsonl:native"
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"JSONL artifact is invalid: {error}") from error
        raise ValueError("JSONL artifact contains no timestamped rows")
    return datetime.fromtimestamp(
        path.stat().st_mtime, timezone.utc
    ), "filesystem:mtime"


def _age_reason(
    timestamp: datetime | None,
    now: datetime,
    label: str,
    max_age_hours: float = CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
) -> str | None:
    if timestamp is None:
        return f"{label} has no valid timestamp"
    age_hours = (now - timestamp).total_seconds() / 3600.0
    if age_hours < -5.0 / 60.0:
        return f"{label} timestamp is in the future"
    if age_hours > max_age_hours:
        return f"{label} is stale ({age_hours:.1f}h > {max_age_hours:.0f}h)"
    return None


def _correctness_claim(value: object) -> tuple[dict | None, str | None]:
    match = _CORRECTNESS_CLAIM_RE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return None, "correctness proof is not a structured hash-bound pass receipt"
    measured = float(match.group("value"))
    maximum = float(match.group("maximum"))
    if (
        not math.isfinite(measured)
        or not math.isfinite(maximum)
        or maximum != MAX_CORRECTNESS_CER
        or not 0.0 <= measured <= maximum
    ):
        return None, "correctness receipt claims an invalid or out-of-budget CER"
    return {
        "sha256": match.group("sha256"),
        "cer_norm": measured,
        "maximum": maximum,
    }, None


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

PRIOR = BetaParams(
    alpha=1.0, beta=1.0
)  # uniform on [0,1]; declared in the score contract


@dataclass
class CategoryEvidence:
    """Weighted outcome tallies for one FeatureUniverse category."""

    category_id: str
    category_weight: float  # release weight; categories sum to 1.0 (loader-enforced)
    weighted_successes: float = 0.0  # Σ 1.0*w over present rows
    weighted_partials: float = (
        0.0  # Σ 0.5*w over partial rows (counts in BOTH alpha and beta)
    )
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
    k = (
        math.ceil(confidence * (n + 1)) - 1
    )  # 0-indexed (1-α)-quantile, α = 1-confidence
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
        return RatchetVerdict(
            "Allow", "raises lower bound; no per-category regression", new
        )

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
# bd-re8.13 — the REAL-data modes: score FEATURE_PARITY.md, track convergence.
# --------------------------------------------------------------------------- #

STATUS_TOKENS = ("present", "partial", "missing", "n/a", "excluded")


def parse_feature_parity(md_text: str) -> dict[str, dict[str, int]]:
    """Parse FEATURE_PARITY.md into per-section status tallies.

    A row is status-bearing when one of its pipe-cells is EXACTLY a status
    token — the same rule `tests/surface_matrix.rs` uses, so the two parsers
    cannot diverge on what counts as a scoreboard row. Returns
    {section_title: {status: count}} for every `## N.` section with rows.
    """
    import re

    sections: dict[str, dict[str, int]] = {}
    current = None
    for line in md_text.splitlines():
        m = re.match(r"^## (\d+)\.\s*(.*)", line)
        if m:
            current = f"§{m.group(1)} {m.group(2).split('(')[0].split('—')[0].strip()}"
            continue
        if current is None or not line.startswith("|") or line.startswith("|-"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 6:
            continue
        for c in cells:
            if c.lower() in STATUS_TOKENS:
                sections.setdefault(current, {})
                sections[current][c.lower()] = sections[current].get(c.lower(), 0) + 1
                break
    return {k: v for k, v in sections.items() if v}


def categories_from_parity(md_text: str) -> list[CategoryEvidence]:
    """One CategoryEvidence per scoreboard section, equal category weights
    (normalized to the loader's sum-to-1.0 invariant), equal row weights
    within a section. `n/a` rows are skipped entirely; `excluded` rows keep
    their weight in the coverage-debt tally per §2.1.2."""
    sections = parse_feature_parity(md_text)
    if not sections:
        raise ValueError("no scoreboard sections parsed from FEATURE_PARITY.md")
    cat_w = 1.0 / len(sections)
    cats: list[CategoryEvidence] = []
    for title, tallies in sections.items():
        scored = {s: n for s, n in tallies.items() if s != "n/a"}
        n_rows = sum(scored.values())
        row_w = 1.0 / n_rows if n_rows else 0.0
        ev = CategoryEvidence(title, cat_w)
        for status, count in scored.items():
            for _ in range(count):
                add_outcome(ev, status, row_w)
        cats.append(ev)
    return cats


def surface_must_verdict(md_text: str) -> dict:
    """Require every release-surface MUST row (§12-§15) to be present.

    `partial`, `missing`, `excluded`, `n/a`, and unrecognized statuses are all
    release debt. The methodology explicitly forbids rounding partial upward,
    and a strict certificate cannot silently waive excluded MUST rows.
    """
    section = 0
    must_rows = 0
    debt: list[dict[str, str]] = []
    for line in md_text.splitlines():
        heading = re.match(r"^## (\d+)\.", line)
        if heading:
            section = int(heading.group(1))
            continue
        if not (12 <= section <= 15) or not line.startswith("|"):
            continue
        cells = _markdown_cells(line)
        if "MUST" not in cells:
            continue
        must_rows += 1
        status = next(
            (cell.lower() for cell in cells if cell.lower() in STATUS_TOKENS),
            "unrecognized",
        )
        if status != "present":
            debt.append(
                {
                    "surface": cells[0][:80] if cells else "<unnamed>",
                    "status": status,
                }
            )
    return {
        "ok": must_rows > 0 and not debt,
        "must_rows": must_rows,
        "debt": debt,
    }


_ANY_EXPERIMENT_HEADING_RE = re.compile(r"^##\s+Experiment\b.*$", re.MULTILINE)
_HYPOTHESIS_ENTRY_RE = re.compile(
    r"^##\s+Experiment\s+`(EXP-[0-9]{4})`(?:\s+[-\N{EM DASH}]\s+.+)?\s*$",
    re.MULTILINE,
)
_HYPOTHESIS_TERMINAL_STATES = {"CONFIRMED_GAP", "NO_EVIDENCE"}


def _clean_experiment_value(value: str) -> str:
    return value.strip().strip("`").strip().strip('"').strip("'")


def _experiment_field_values(section: str, field_name: str) -> list[str]:
    escaped = re.escape(field_name)
    patterns = (
        rf"^\|\s*`{escaped}`\s*\|\s*(.*?)\s*\|\s*$",
        rf"^\s*{escaped}:\s*(.*?)\s*$",
        rf"^\s*-\s*\*\*{escaped}:\*\*\s*(.*?)\s*$",
        rf"^\s*-\s*\*\*{escaped}\*\*:\s*(.*?)\s*$",
    )
    values: list[str] = []
    for pattern in patterns:
        values.extend(re.findall(pattern, section, re.IGNORECASE | re.MULTILINE))
    return [_clean_experiment_value(value) for value in values]


def _experiment_table_field_values(section: str, field_name: str) -> list[str]:
    values = re.findall(
        rf"^\|\s*`{re.escape(field_name)}`\s*\|\s*(.*?)\s*\|\s*$",
        section,
        re.IGNORECASE | re.MULTILINE,
    )
    return [_clean_experiment_value(value) for value in values]


def _experiment_list_field(section: str, field_name: str) -> list[str]:
    match = re.search(
        rf"^\s*{re.escape(field_name)}:\s*$\n(?P<body>(?:^[ \t]+-\s+.*(?:\n|$))*)",
        section,
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return []
    return [
        _clean_experiment_value(value)
        for value in re.findall(
            r"^[ \t]+-\s+(.+?)\s*$", match.group("body"), re.MULTILINE
        )
    ]


def _experiment_section_body(section: str, heading: str) -> str | None:
    match = re.search(
        rf"^###\s+{re.escape(heading)}\s*$\n(?P<body>.*?)(?=^###\s+|\Z)",
        section,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not match:
        return None
    body = match.group("body").strip()
    return body if body and "<" not in body else None


def _experiment_section_present(section: str, heading: str) -> bool:
    return _experiment_section_body(section, heading) is not None


def _experiment_results_fence(section: str) -> str | None:
    body = _experiment_section_body(section, "Results Inline")
    if body is None:
        return None
    fences = re.findall(
        r"^```(?:yaml|yml)?\s*$\n(?P<fields>.*?)^```\s*$",
        body,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return fences[0] if len(fences) == 1 else None


def _retry_condition_valid(value: str) -> bool:
    normalized = _clean_experiment_value(value).lower().rstrip(".")
    return any(normalized.startswith(prefix) for prefix in _RETRY_CONDITION_PREFIXES)


def _evidence_path_verdict(root: Path | None, relative: str) -> str | None:
    if not relative or "<" in relative or relative.upper() in {"N/A", "NONE", "[]"}:
        return "evidence path is empty or a placeholder"
    if root is None:
        return "evidence root is unavailable"
    path = _safe_repo_path(root, relative)
    if path is None or not path.exists():
        return f"evidence path is missing or unsafe: {relative}"
    if path.is_dir():
        manifest_ok, reasons, _covered = _verify_sha256_manifest(path)
        if not manifest_ok:
            return f"evidence directory is not hash-verified: {relative}: {', '.join(reasons)}"
        return None
    manifest_ok, reasons, covered = _verify_sha256_manifest(path.parent)
    if not manifest_ok or path.name not in covered:
        detail = ", ".join(reasons) if reasons else "manifest omits file"
        return f"evidence file is not hash-verified: {relative}: {detail}"
    return None


def _proof_pack_path(
    root: Path | None, section: str, evidence_paths: Sequence[str]
) -> str | None:
    explicit = _experiment_field_values(section, "proof_pack_path")
    candidates = explicit + [path for path in evidence_paths if "proof_pack" in path]
    for candidate in candidates:
        if root is None:
            continue
        path = _safe_repo_path(root, candidate)
        if path is not None and path.is_dir() and any(path.iterdir()):
            manifest_ok, _reasons, _covered = _verify_sha256_manifest(path)
            if manifest_ok:
                return candidate
    return None


def _remediation_bead_failure(root: Path | None, bead_id: str) -> str | None:
    if root is None:
        return "remediation bead registry is unavailable"
    issues_path = root / ".beads/issues.jsonl"
    try:
        issues = {
            issue["id"]: issue
            for line in issues_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for issue in [json.loads(line)]
            if isinstance(issue, dict) and isinstance(issue.get("id"), str)
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return f"remediation bead registry is unreadable: {error}"
    issue = issues.get(bead_id)
    if issue is None:
        return f"remediation bead does not exist: {bead_id}"
    labels = issue.get("labels")
    if not isinstance(labels, list) or "kind:remediation" not in labels:
        return f"remediation bead lacks kind:remediation label: {bead_id}"
    dependencies = issue.get("dependencies")
    if not isinstance(dependencies, list):
        return f"remediation bead has no dependency graph: {bead_id}"
    dependency_texts: list[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        dependency_id = dependency.get("depends_on_id") or dependency.get("id")
        dependency_issue = issues.get(dependency_id)
        if not isinstance(dependency_issue, dict):
            continue
        dependency_texts.append(
            " ".join(
                [
                    str(dependency_issue.get("id", "")),
                    str(dependency_issue.get("title", "")),
                    str(dependency_issue.get("issue_type", "")),
                    " ".join(map(str, dependency_issue.get("labels", []))),
                ]
            ).lower()
        )
    required_kinds = {
        "test": re.compile(
            r"\b(?:test|oracle|metamorphic|sanitizer|fuzz|property|golden)\b"
        ),
        "bench": re.compile(
            r"\b(?:bench|benchmark|criterion|hyperfine|comprehensive-bench)\b"
        ),
        "doc": re.compile(r"\b(?:doc|docs|documentation|safety|contract|agents\.md)\b"),
    }
    missing = [
        kind
        for kind, pattern in required_kinds.items()
        if not any(pattern.search(text) for text in dependency_texts)
    ]
    if missing:
        return f"remediation bead lacks test+bench+doc dependencies ({', '.join(missing)}): {bead_id}"
    return None


def _proof_pack_failure(root: Path | None, bead_id: str, pillar: str) -> str | None:
    if root is None:
        return "proof-pack root is unavailable"
    relative = f"artifacts/{bead_id}/proof_pack"
    path = _safe_repo_path(root, relative)
    if path is None or not path.is_dir():
        return f"required proof pack is missing: {relative}"
    manifest_ok, reasons, covered = _verify_sha256_manifest(path)
    if not manifest_ok:
        return f"proof pack is not exhaustively hash-verified: {', '.join(reasons)}"
    required = {"README.md", "delta_summary.json", "rerun.sh", "rollback.md"}
    if pillar == "perf":
        required.update(
            {
                "baseline_profile.flame.svg",
                "baseline_profile.samply.json",
                "selections_byte_identical.txt",
                "concurrent_mode_default_guard.txt",
                "cv_pct.json",
            }
        )
    elif pillar == "conformance":
        required.update(
            {"selections_byte_identical.txt", "concurrent_mode_default_guard.txt"}
        )
    missing = sorted(required - covered)
    return f"proof pack omits required files: {', '.join(missing)}" if missing else None


def hypothesis_texts_verdict(
    contents: dict[str, str | None],
    ledger_paths: Sequence[str] = HYPOTHESIS_LEDGER_PATHS,
    root: Path | None = None,
) -> dict:
    """Validate canonical experiment closure and proof obligations."""
    missing: list[str] = []
    unresolved: list[dict[str, str]] = []
    entries = 0
    seen: set[str] = set()
    for ledger in ledger_paths:
        text = contents.get(ledger)
        if text is None:
            missing.append(ledger)
            continue
        matches = list(_HYPOTHESIS_ENTRY_RE.finditer(text))
        all_headings = list(_ANY_EXPERIMENT_HEADING_RE.finditer(text))
        recognized_headings = {match.group(0).strip() for match in matches}
        for heading in all_headings:
            if heading.group(0).strip() not in recognized_headings:
                unresolved.append(
                    {
                        "ledger": ledger,
                        "hypothesis": heading.group(0)[:120],
                        "state": "NONCANONICAL_EXPERIMENT_HEADING",
                    }
                )
        if not matches:
            unresolved.append(
                {
                    "ledger": ledger,
                    "hypothesis": "<ledger>",
                    "state": "NO_CANONICAL_EXPERIMENT_ENTRIES",
                }
            )
            continue
        for index, match in enumerate(matches):
            hypothesis_id = match.group(1)
            if hypothesis_id in seen:
                unresolved.append(
                    {
                        "ledger": ledger,
                        "hypothesis": hypothesis_id,
                        "state": "DUPLICATE_EXPERIMENT_ID",
                    }
                )
                continue
            seen.add(hypothesis_id)
            entries += 1
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section = text[match.start() : end]
            failures: list[str] = []
            required_fields = (
                "experiment_id",
                "pillar",
                "created_at_utc",
                "created_by_agent",
                "bead_id",
                "parent_hypothesis_id",
                "status",
            )
            fields: dict[str, str] = {}
            for field_name in required_fields:
                values = _experiment_table_field_values(section, field_name)
                if len(values) != 1:
                    failures.append(f"{field_name} must occur exactly once")
                else:
                    fields[field_name] = values[0]
            if fields.get("experiment_id", "").strip("<>") != hypothesis_id:
                failures.append("experiment_id does not match heading")
            if fields.get("status", "").strip("`").upper() != "CLOSED":
                failures.append("status is not CLOSED")
            if _parse_utc_timestamp(fields.get("created_at_utc")) is None:
                failures.append("created_at_utc is invalid")
            for name in ("created_by_agent", "bead_id", "parent_hypothesis_id"):
                if not fields.get(name) or "<" in fields.get(name, ""):
                    failures.append(f"{name} is empty or a placeholder")
            expected_pillar = {
                "PERF_HYPOTHESIS_LEDGER.md": "perf",
                "CONFORMANCE_HYPOTHESIS_LEDGER.md": "conformance",
                "SURFACE_PARITY_HYPOTHESIS_LEDGER.md": "surface",
            }.get(Path(ledger).name)
            pillar = fields.get("pillar", "").strip("`").lower()
            if pillar not in {"perf", "conformance", "surface"}:
                failures.append("pillar is invalid")
            elif expected_pillar is not None and pillar != expected_pillar:
                failures.append(f"pillar {pillar!r} does not match ledger")

            for heading in (
                "Hypothesis",
                "Motivation",
                "Minimal Reproducer",
                "Expected Signal",
                "Falsifiability Criteria",
                "One-Line Invocation",
                "Results Inline",
                "Closure Predicate",
            ):
                if not _experiment_section_present(section, heading):
                    failures.append(f"missing or placeholder section: {heading}")

            results_fence = _experiment_results_fence(section)
            if results_fence is None:
                failures.append(
                    "Results Inline must contain exactly one fenced YAML field block"
                )
                results_fence = ""
            result_values = _experiment_field_values(results_fence, "result_status")
            state = (
                result_values[0].upper()
                if len(result_values) == 1
                else "RESULT_MISSING_OR_DUPLICATE"
            )
            if state not in _HYPOTHESIS_TERMINAL_STATES:
                failures.append(f"result_status is not terminal: {state}")
            summary_values = _experiment_field_values(results_fence, "result_summary")
            if (
                len(summary_values) != 1
                or not summary_values[0]
                or "<" in summary_values[0]
            ):
                failures.append("result_summary is missing or a placeholder")
            closed_values = _experiment_field_values(results_fence, "closed_at_utc")
            closed_at = (
                _parse_utc_timestamp(closed_values[0])
                if len(closed_values) == 1
                else None
            )
            if closed_at is None:
                failures.append("closed_at_utc is missing or invalid")
            created_at = _parse_utc_timestamp(fields.get("created_at_utc"))
            if (
                closed_at is not None
                and created_at is not None
                and closed_at < created_at
            ):
                failures.append("closed_at_utc predates created_at_utc")

            evidence_paths = _experiment_list_field(
                results_fence, "result_evidence_paths"
            )
            if not evidence_paths:
                failures.append("result_evidence_paths is empty")
            elif len(set(evidence_paths)) != len(evidence_paths):
                failures.append("result_evidence_paths contains duplicates")
            for evidence_path in evidence_paths:
                evidence_reason = _evidence_path_verdict(root, evidence_path)
                if evidence_reason:
                    failures.append(evidence_reason)

            if state == "CONFIRMED_GAP":
                bead_values = _experiment_field_values(
                    results_fence, "spawned_remediation_bead"
                )
                remediation_id = bead_values[0] if len(bead_values) == 1 else ""
                if (
                    re.fullmatch(r"bd-[A-Za-z0-9][A-Za-z0-9.-]*", remediation_id)
                    is None
                ):
                    failures.append("CONFIRMED_GAP has no remediation bead")
                    failures.append("CONFIRMED_GAP has no hash-verified proof pack")
                else:
                    bead_failure = _remediation_bead_failure(root, remediation_id)
                    if bead_failure:
                        failures.append(bead_failure)
                    expected_pack = f"artifacts/{remediation_id}/proof_pack"
                    declared_pack = _proof_pack_path(
                        root, results_fence, evidence_paths
                    )
                    if declared_pack != expected_pack:
                        failures.append(
                            f"CONFIRMED_GAP must declare proof_pack_path: {expected_pack}"
                        )
                    pack_failure = _proof_pack_failure(root, remediation_id, pillar)
                    if pack_failure:
                        failures.append(pack_failure)
                impact_values = _experiment_field_values(results_fence, "result_impact")
                if (
                    len(impact_values) != 1
                    or not impact_values[0]
                    or "<" in impact_values[0]
                ):
                    failures.append("CONFIRMED_GAP has no measured result_impact")
            elif state == "NO_EVIDENCE":
                retry_values = _experiment_field_values(
                    results_fence, "retry_condition_predicate"
                )
                if len(retry_values) != 1 or not _retry_condition_valid(
                    retry_values[0]
                ):
                    failures.append(
                        "NO_EVIDENCE has no allowed retry_condition_predicate"
                    )

            if failures:
                unresolved.append(
                    {
                        "ledger": ledger,
                        "hypothesis": hypothesis_id,
                        "state": state,
                        "failures": failures,
                    }
                )
    resolved = not missing and not unresolved
    return {
        "resolved": resolved,
        "missing": missing,
        "unresolved": unresolved,
        "entries": entries,
        "required_ledgers": list(ledger_paths),
    }


def hypothesis_ledger_verdict(
    root: Path, ledger_paths: Sequence[str] = HYPOTHESIS_LEDGER_PATHS
) -> dict:
    contents: dict[str, str | None] = {}
    for ledger in ledger_paths:
        path = _safe_repo_path(root, ledger)
        try:
            contents[ledger] = path.read_text(encoding="utf-8") if path else None
        except OSError:
            contents[ledger] = None
    return hypothesis_texts_verdict(contents, ledger_paths, root)


def _markdown_cells(line: str) -> list[str]:
    return [
        cell.strip().replace(r"\|", "|")
        for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))
    ]


def _perf_ledger_rows(md_text: str) -> list[dict[str, str]]:
    header = next(
        (
            line
            for line in md_text.splitlines()
            if line.startswith("| date | claim_id | evidence_id |")
        ),
        "",
    )
    if not header:
        return []
    columns = _markdown_cells(header)
    rows: list[dict[str, str]] = []
    for line in md_text.splitlines():
        if not line.startswith("|") or line == header or "------" in line:
            continue
        cells = _markdown_cells(line)
        if len(cells) == len(columns):
            rows.append(dict(zip(columns, cells, strict=True)))
    return rows


def _verify_sha256_manifest(evidence_dir: Path) -> tuple[bool, list[str], set[str]]:
    manifest = next(
        (
            evidence_dir / name
            for name in EVIDENCE_MANIFEST_NAMES
            if (evidence_dir / name).is_file()
        ),
        None,
    )
    if manifest is None:
        return False, ["missing SHA-256 manifest"], set()
    if manifest.is_symlink():
        return False, ["SHA-256 manifest must not be a symlink"], set()
    if manifest.suffix == ".json":
        return (
            False,
            ["JSON evidence manifests are not yet a supported strict format"],
            set(),
        )
    reasons: list[str] = []
    covered: set[str] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return False, [f"unreadable manifest: {error}"], set()
    for line in lines:
        if not line.strip():
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+[* ]?(.+?)\s*$", line)
        if not match:
            reasons.append(f"malformed manifest line: {line[:80]}")
            continue
        expected, relative = match.group(1).lower(), match.group(2)
        relative_path = Path(relative)
        canonical = relative_path.as_posix()
        if (
            relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or canonical != relative
        ):
            reasons.append(f"noncanonical manifest target: {relative}")
            continue
        if relative in covered:
            reasons.append(f"duplicate manifest target: {relative}")
            continue
        target = _safe_repo_path(evidence_dir, relative)
        if (
            target is None
            or not target.is_file()
            or any(
                evidence_dir.joinpath(*relative_path.parts[:index]).is_symlink()
                for index in range(1, len(relative_path.parts) + 1)
            )
        ):
            reasons.append(
                f"manifest target missing, symlinked, or escapes evidence dir: {relative}"
            )
            continue
        covered.add(relative)
        if _sha256_file(target) != expected:
            reasons.append(f"manifest hash mismatch: {relative}")
    if not covered:
        reasons.append("manifest covers no evidence files")
    actual_files = {
        path.relative_to(evidence_dir).as_posix()
        for path in evidence_dir.rglob("*")
        if path.is_file() and path != manifest and not path.name.startswith("._")
    }
    symlinks = sorted(
        path.relative_to(evidence_dir).as_posix()
        for path in evidence_dir.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        reasons.append("evidence directory contains symlinks: " + ", ".join(symlinks))
    uncovered = sorted(actual_files - covered)
    if uncovered:
        reasons.append("manifest omits evidence files: " + ", ".join(uncovered))
    return not reasons, reasons, covered


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _stage_record(payload: dict, stage: str) -> dict | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), list):
        return None
    wanted = stage.replace("-", "_")
    for record in payload.get("stages", []):
        if (
            isinstance(record, dict)
            and str(record.get("stage", "")).replace("-", "_") == wanted
        ):
            return record
    return None


def _fixture_identity(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = re.search(
        r"(?:^|;\s*)page=(?P<page>[^\s;]+)\s+sha256=(?P<sha>[0-9a-f]{64})(?:;|\s|$)",
        value,
    )
    return (match.group("page"), match.group("sha")) if match else None


def _measurement_contract(
    focr_payload: object,
    ref_payload: object,
    row: dict,
    threads: int | None,
) -> tuple[list[str], dict | None, dict | None]:
    reasons: list[str] = []
    if not isinstance(focr_payload, dict) or not isinstance(ref_payload, dict):
        return ["stage payloads must both be JSON objects"], None, None
    for side, payload, source in (
        ("focr", focr_payload, "focr"),
        ("reference", ref_payload, "reference"),
    ):
        if payload.get("schema") != "focr-gauntlet-stages/v1":
            reasons.append(f"{side} stage payload has an unsupported schema")
        if payload.get("source") != source:
            reasons.append(f"{side} stage payload has the wrong source identity")
        if payload.get("synthetic") is not False:
            reasons.append(f"{side} stage payload is synthetic or unstamped")
    if focr_payload.get("stdout_identical_across_runs") is not True:
        reasons.append("focr stdout is not deterministic across runs")

    fixture = _fixture_identity(row.get("fixture_hash"))
    if fixture is None:
        reasons.append("ledger fixture_hash lacks a page SHA-256 binding")
        fixture_page, fixture_sha = "", ""
    else:
        fixture_page, fixture_sha = fixture
    focr_page = Path(str(focr_payload.get("page", ""))).name
    ref_page = Path(str(ref_payload.get("page", ""))).name
    if not focr_page or focr_page != ref_page or focr_page != fixture_page:
        reasons.append("focr/reference/ledger fixture pages do not match")
    if focr_payload.get("page_sha256") != fixture_sha:
        reasons.append("focr page_sha256 does not match the ledger fixture hash")
    ref_page_sha = ref_payload.get("page_sha256")
    if ref_page_sha != fixture_sha:
        reasons.append("reference page_sha256 does not match the ledger fixture hash")

    focr_stage = _stage_record(focr_payload, "decode-per-token")
    ref_stage = _stage_record(ref_payload, "decode-per-token")
    if focr_stage is None or ref_stage is None:
        reasons.append("decode-per-token stage evidence is missing")
        return reasons, focr_stage, ref_stage

    for side, record, source in (
        ("focr", focr_stage, "focr"),
        ("reference", ref_stage, "reference"),
    ):
        if record.get("schema") != "focr-gauntlet-stage/v1":
            reasons.append(f"{side} stage record has an unsupported schema")
        if record.get("source") != source:
            reasons.append(f"{side} stage record has the wrong source identity")
        if str(record.get("stage", "")).replace("-", "_") != "decode_per_token":
            reasons.append(f"{side} stage record has the wrong stage")
        if record.get("ledger_stage") is not True:
            reasons.append(f"{side} stage record is not ledger-eligible")
        if record.get("synthetic") is not False:
            reasons.append(f"{side} stage record is synthetic or unstamped")
        if record.get("unit") != "ms":
            reasons.append(f"{side} stage record unit is not ms")
        samples = record.get("samples_ms")
        n = record.get("n")
        if (
            not isinstance(samples, list)
            or len(samples) < 2
            or not isinstance(n, int)
            or isinstance(n, bool)
            or n != len(samples)
        ):
            reasons.append(f"{side} stage samples/n are incomplete")
            continue
        try:
            numeric_samples = [float(sample) for sample in samples]
            if not all(
                math.isfinite(sample) and sample > 0.0 for sample in numeric_samples
            ):
                raise ValueError("non-positive sample")
            best_ms = float(record["best_ms"])
            cv_pct = float(record["cv_pct"])
            recomputed_cv = round(
                statistics.stdev(numeric_samples)
                / statistics.fmean(numeric_samples)
                * 100.0,
                3,
            )
            if not math.isclose(best_ms, min(numeric_samples), rel_tol=1e-9):
                reasons.append(f"{side} best_ms is not the best measured sample")
            if (
                not math.isfinite(cv_pct)
                or not 0.0 <= cv_pct <= 5.0
                or not math.isclose(cv_pct, recomputed_cv, abs_tol=0.001)
            ):
                reasons.append(f"{side} cv_pct is invalid or not sample-derived")
        except (KeyError, TypeError, ValueError, statistics.StatisticsError):
            reasons.append(f"{side} stage timing samples are invalid")
        if record.get("warmup_discarded", 0) < 1:
            reasons.append(f"{side} stage did not discard a warmup")
        if not isinstance(record.get("precision"), str) or not record.get("precision"):
            reasons.append(f"{side} stage lacks precision identity")
        if not isinstance(record.get("backend"), str) or not record.get("backend"):
            reasons.append(f"{side} stage lacks backend identity")
        if not isinstance(record.get("allocator"), str) or not record.get("allocator"):
            reasons.append(f"{side} stage lacks allocator identity")
        if threads is None or record.get("threads") != threads:
            reasons.append(f"{side} stage does not preserve thread parity")

    if focr_stage.get("backend") != "focr":
        reasons.append("focr stage backend identity is not focr")
    if focr_stage.get("tokens_consistent") is not True:
        reasons.append("focr decode token count is not deterministic")
    proof = ref_stage.get("thread_proof")
    if not isinstance(proof, dict) or (
        proof.get("budget") != threads or proof.get("torch_num_threads") != threads
    ):
        reasons.append("reference stage lacks a matching thread proof")
    if ref_stage.get("backend") != row.get("reference_backend"):
        reasons.append("reference backend does not match the ledger row")
    expected_precision = (
        f"{focr_stage.get('precision')} vs "
        f"{ref_stage.get('backend')}-{ref_stage.get('precision')}"
    )
    if row.get("precision (focr vs ref)") != expected_precision:
        reasons.append("stage precision identities do not match the ledger row")
    if focr_stage.get("allocator") != ref_stage.get("allocator") or focr_stage.get(
        "allocator"
    ) != row.get("allocator"):
        reasons.append("stage allocator identities do not match the ledger row")
    return reasons, focr_stage, ref_stage


def _correctness_receipt_reasons(path: Path, claim: dict | None) -> list[str]:
    if claim is None:
        return ["structured correctness claim is unavailable"]
    reasons: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return [f"correctness receipt is unreadable: {error}"]
    if not isinstance(payload, dict):
        return ["correctness receipt is not a JSON object"]
    aggregate = payload.get("aggregate")
    pages = payload.get("pages")
    if not isinstance(aggregate, dict) or not isinstance(pages, list) or not pages:
        return ["correctness receipt lacks aggregate/pages evidence"]
    measured = aggregate.get("cer_norm")
    pages_total = aggregate.get("pages_total")
    if (
        _sha256_file(path) != claim.get("sha256")
        or not isinstance(measured, (int, float))
        or isinstance(measured, bool)
        or not math.isfinite(float(measured))
        or not math.isclose(
            float(measured), claim.get("cer_norm", math.nan), abs_tol=5e-7
        )
        or not 0.0 <= float(measured) <= MAX_CORRECTNESS_CER
        or not isinstance(pages_total, int)
        or isinstance(pages_total, bool)
        or pages_total <= 0
        or aggregate.get("pages_with_hyp") != pages_total
        or len(pages) != pages_total
    ):
        reasons.append(
            "correctness receipt does not match the structured in-budget claim"
        )
    if any(not isinstance(page, dict) or page.get("status") != "OK" for page in pages):
        reasons.append("correctness receipt contains a non-OK page")
    return reasons


def perf_evidence_verdict(
    perf_text: str,
    root: Path,
    current_head: str,
    now: datetime | None = None,
    max_age_hours: float = CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
) -> dict:
    """Find a current, eligible Unlimited-OCR decode-per-token proof row."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    candidates: list[dict[str, object]] = []
    for row in _perf_ledger_rows(perf_text):
        stage = row.get("stage", "").replace("_", "-")
        if (
            row.get("model_commit") != UNLIMITED_OCR_MODEL_COMMIT
            or stage != "decode-per-token"
        ):
            continue
        reasons: list[str] = []
        claim_id = row.get("claim_id", "<missing-claim>")
        try:
            focr_ms = float(row.get("focr_ms", "nan"))
            ref_ms = float(row.get("ref_ms", "nan"))
            ratio = float(row.get("ratio", "nan"))
            expected_ratio = ref_ms / focr_ms
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (focr_ms, ref_ms, ratio)
            ):
                reasons.append("non-positive or non-finite timing/ratio")
            elif ratio <= 1.0:
                reasons.append(f"decode ratio does not beat reference: {ratio:.3f}")
            elif abs(ratio - expected_ratio) / expected_ratio > 0.01:
                reasons.append("ratio does not match ref_ms / focr_ms")
        except (TypeError, ValueError, ZeroDivisionError):
            reasons.append("invalid timing/ratio cells")

        command = row.get("command/env", "")
        if "release-perf" not in command:
            reasons.append("subject was not measured from the release-perf profile")
        precision = row.get("precision (focr vs ref)", "")
        if "focr-" not in precision or " vs " not in precision:
            reasons.append("missing precision comparison")
        thread_match = re.fullmatch(
            r"focr=ref=(\d+)", row.get("threads (focr=ref N)", "")
        )
        if (
            not thread_match
            or int(thread_match.group(1)) <= 0
            or int(thread_match.group(1)) == 64
        ):
            reasons.append("invalid or unfair thread-parity proof")
        correctness_claim, correctness_reason = _correctness_claim(
            row.get("correctness_proof")
        )
        if correctness_reason:
            reasons.append(correctness_reason)
        if row.get("reference_backend", "").lower() in {"", "unknown", "n/a"}:
            reasons.append("missing reference backend identity")

        evidence_id = row.get("evidence_id", "").rstrip("/")
        evidence_dir = _safe_repo_path(root, evidence_id)
        if evidence_dir is None or not evidence_dir.is_dir():
            reasons.append(f"missing or unsafe evidence directory: {evidence_id}")
        else:
            manifest_ok, manifest_reasons, covered = _verify_sha256_manifest(
                evidence_dir
            )
            if not manifest_ok:
                reasons.extend(manifest_reasons)
            try:
                row_doc = json.loads(
                    (evidence_dir / "row.json").read_text(encoding="utf-8")
                )
                if not isinstance(row_doc, dict):
                    raise TypeError("row.json is not a JSON object")
                if row_doc.get("schema") != "focr-gauntlet-row/v2":
                    reasons.append("row.json schema is not focr-gauntlet-row/v2")
                age_reason = _age_reason(
                    _parse_utc_timestamp(row_doc.get("created_utc")),
                    now,
                    "row.json",
                    max_age_hours,
                )
                if age_reason:
                    reasons.append(age_reason)
                if not _valid_git_head(current_head):
                    reasons.append("current git HEAD is unavailable")
                elif row_doc.get("git_head") != current_head:
                    reasons.append(
                        f"performance evidence git_head {row_doc.get('git_head')!r} != current HEAD {current_head}"
                    )
                local_rows = row_doc.get("rows")
                if not isinstance(local_rows, list):
                    local_rows = []
                    reasons.append("row.json rows is not a list")
                matching_rows = [
                    item
                    for item in local_rows
                    if isinstance(item, dict)
                    and item.get("claim_id") == claim_id
                    and str(item.get("stage", "")).replace("_", "-")
                    == "decode-per-token"
                ]
                bound_correctness_claim: dict | None = None
                if len(matching_rows) != 1:
                    reasons.append(
                        f"row.json must contain exactly one ledger claim/stage row; found {len(matching_rows)}"
                    )
                    bound_row: dict = {}
                else:
                    bound_row = matching_rows[0]
                    mismatched_columns = [
                        column
                        for column, value in row.items()
                        if str(bound_row.get(column, "")) != value
                    ]
                    if mismatched_columns:
                        reasons.append(
                            "ledger row differs from hashed row.json columns: "
                            + ", ".join(mismatched_columns)
                        )
                    bound_correctness_claim, bound_correctness_reason = (
                        _correctness_claim(bound_row.get("correctness_proof"))
                    )
                    if bound_correctness_reason:
                        reasons.append("hashed row: " + bound_correctness_reason)

                inputs = row_doc.get("inputs")
                if not isinstance(inputs, dict):
                    inputs = {}
                    reasons.append("row.json inputs is not an object")
                expected_inputs = {
                    "focr_stages",
                    "ref_stages",
                    "roofline",
                    "correctness_receipt",
                }
                if set(inputs) != expected_inputs:
                    reasons.append(
                        "row.json inputs must bind exactly: "
                        + ", ".join(sorted(expected_inputs))
                    )

                def bound_input(name: str, fallback: str) -> tuple[Path, dict]:
                    record = inputs.get(name)
                    if not isinstance(record, dict):
                        reasons.append(f"row.json inputs.{name} is missing")
                        record = {}
                    bundle_path = record.get("bundle_path")
                    if bundle_path != fallback:
                        reasons.append(
                            f"row.json inputs.{name}.bundle_path is not canonical: {bundle_path!r}"
                        )
                        bundle_path = fallback
                    if bundle_path not in covered:
                        reasons.append(f"manifest omits bound input: {bundle_path}")
                    path = _safe_repo_path(evidence_dir, bundle_path)
                    if path is None or not path.is_file():
                        reasons.append(
                            f"bound input is missing or unsafe: {bundle_path}"
                        )
                        path = evidence_dir / "__missing__"
                    expected_sha = record.get("sha256")
                    if path.is_file() and (
                        not isinstance(expected_sha, str)
                        or expected_sha != _sha256_file(path)
                    ):
                        reasons.append(
                            f"row.json inputs.{name}.sha256 does not bind {bundle_path}"
                        )
                    return path, record

                if "row.json" not in covered:
                    reasons.append("manifest omits: row.json")
                focr_path, _focr_input = bound_input("focr_stages", "focr_stages.json")
                ref_path, _ref_input = bound_input("ref_stages", "ref_stages.json")
                roofline_path, _roofline_input = bound_input(
                    "roofline", "roofline.json"
                )
                correctness_path, _correctness_input = bound_input(
                    "correctness_receipt", "correctness_receipt.json"
                )
                required_bundle_files = {"PERF_LEDGER_ROW.md", "row.json"}
                for required_file in required_bundle_files:
                    if required_file not in covered:
                        reasons.append(
                            f"manifest omits required evidence: {required_file}"
                        )
                raw_stdout = {
                    path
                    for path in covered
                    if path.startswith("raw/") and path.endswith(".stdout")
                }
                raw_stderr = {
                    path
                    for path in covered
                    if path.startswith("raw/") and path.endswith(".stderr")
                }
                if not raw_stdout or not raw_stderr:
                    reasons.append("manifest lacks mandatory raw stdout/stderr logs")
                if bound_row:
                    expected_row_line = (
                        "| "
                        + " | ".join(str(bound_row.get(column, "")) for column in row)
                        + " |"
                    )
                    try:
                        emitted_rows = (
                            (evidence_dir / "PERF_LEDGER_ROW.md")
                            .read_text(encoding="utf-8")
                            .splitlines()
                        )
                    except OSError as error:
                        emitted_rows = []
                        reasons.append(f"PERF_LEDGER_ROW.md is unreadable: {error}")
                    if emitted_rows.count(expected_row_line) != 1:
                        reasons.append(
                            "PERF_LEDGER_ROW.md does not contain exactly one bound claim row"
                        )
                focr_payload = json.loads(focr_path.read_text(encoding="utf-8"))
                ref_payload = json.loads(ref_path.read_text(encoding="utf-8"))
                for side, payload in (
                    ("focr", focr_payload),
                    ("reference", ref_payload),
                ):
                    if isinstance(payload, dict):
                        stage_age_reason = _age_reason(
                            _parse_utc_timestamp(payload.get("created_utc")),
                            now,
                            f"{side} stage measurement",
                            max_age_hours,
                        )
                        if stage_age_reason:
                            reasons.append(stage_age_reason)
                contract_reasons, focr_stage, ref_stage = _measurement_contract(
                    focr_payload,
                    ref_payload,
                    bound_row,
                    int(thread_match.group(1)) if thread_match else None,
                )
                reasons.extend(contract_reasons)
                if focr_stage is not None and ref_stage is not None:
                    try:
                        stage_focr_ms = float(focr_stage["best_ms"])
                        stage_ref_ms = float(ref_stage["best_ms"])
                        expected_focr = f"{stage_focr_ms:.3f}"
                        expected_ref = f"{stage_ref_ms:.3f}"
                        expected_ratio_text = f"{stage_ref_ms / stage_focr_ms:.3f}"
                        if bound_row.get("focr_ms") != expected_focr:
                            reasons.append(
                                "hashed row focr_ms does not match stage best_ms"
                            )
                        if bound_row.get("ref_ms") != expected_ref:
                            reasons.append(
                                "hashed row ref_ms does not match stage best_ms"
                            )
                        if bound_row.get("ratio") != expected_ratio_text:
                            reasons.append(
                                "hashed row ratio does not match stage-derived ratio"
                            )
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        reasons.append("stage timing evidence is missing or invalid")
                    required_raw_count = max(
                        int(focr_stage.get("n", 0)), int(ref_stage.get("n", 0))
                    )
                    if (
                        len(raw_stdout) < required_raw_count
                        or len(raw_stderr) < required_raw_count
                    ):
                        reasons.append("raw logs do not cover every measured run")

                    try:
                        roofline_payload = json.loads(
                            roofline_path.read_text(encoding="utf-8")
                        )
                    except (
                        OSError,
                        ValueError,
                        TypeError,
                        json.JSONDecodeError,
                    ) as error:
                        roofline_payload = {}
                        reasons.append(f"roofline evidence is unreadable: {error}")
                    if not isinstance(roofline_payload, dict):
                        roofline_payload = {}
                        reasons.append("roofline evidence is not a JSON object")
                    if roofline_payload.get("schema") != "focr-gauntlet-roofline/v1":
                        reasons.append("roofline evidence has an unsupported schema")
                    if roofline_payload.get("synthetic") is not False:
                        reasons.append("roofline evidence is synthetic or unstamped")
                    floors = roofline_payload.get("floors")
                    matching_floors = (
                        [
                            floor
                            for floor in floors
                            if isinstance(floor, dict)
                            and str(floor.get("stage", "")).replace("-", "_")
                            == "decode_per_token"
                        ]
                        if isinstance(floors, list)
                        else []
                    )
                    if len(matching_floors) != 1:
                        reasons.append(
                            "roofline must contain exactly one decode-per-token floor"
                        )
                    else:
                        floor = matching_floors[0]
                        try:
                            floor_ms = float(floor["floor_ms"])
                            if not math.isfinite(floor_ms) or floor_ms <= 0.0:
                                raise ValueError("invalid floor")
                            if bound_row.get("floor_kind") != floor.get("floor_kind"):
                                reasons.append(
                                    "hashed row floor_kind does not match roofline"
                                )
                            if bound_row.get("floor_ms") != f"{floor_ms:.3f}":
                                reasons.append(
                                    "hashed row floor_ms does not match roofline"
                                )
                            if (
                                bound_row.get("dist_above_floor")
                                != f"{stage_focr_ms / floor_ms:.2f}"
                            ):
                                reasons.append(
                                    "hashed row dist_above_floor does not match roofline"
                                )
                        except (KeyError, TypeError, ValueError, ZeroDivisionError):
                            reasons.append("roofline floor is missing or invalid")

                reasons.extend(
                    _correctness_receipt_reasons(
                        correctness_path,
                        bound_correctness_claim or correctness_claim,
                    )
                )
            except (
                AttributeError,
                OSError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                reasons.append(f"unreadable structured evidence: {error}")
        candidates.append(
            {"claim_id": claim_id, "eligible": not reasons, "reasons": reasons}
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    return {
        "ok": bool(eligible),
        "eligible_claims": [candidate["claim_id"] for candidate in eligible],
        "candidates": candidates,
        "reason": (
            "eligible current Unlimited-OCR decode evidence found"
            if eligible
            else "no eligible current Unlimited-OCR decode-per-token evidence"
        ),
    }


def _bundle_root_sha256(manifest: Sequence[dict]) -> str | None:
    paths: set[str] = set()
    bindings: list[tuple[str, str, str, str]] = []
    for entry in manifest:
        if not isinstance(entry, dict) or "error" in entry:
            return None
        relative = entry.get("artifact")
        digest = entry.get("sha256")
        timestamp = entry.get("timestamp_utc")
        timestamp_source = entry.get("timestamp_source")
        if (
            not isinstance(relative, str)
            or relative in paths
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or _parse_utc_timestamp(timestamp) is None
            or not isinstance(timestamp_source, str)
            or not timestamp_source
        ):
            return None
        paths.add(relative)
        bindings.append((relative, digest, str(timestamp), timestamp_source))
    if not bindings:
        return None
    canonical = "".join(
        f"{relative}\0{digest}\0{timestamp}\0{source}\n"
        for relative, digest, timestamp, source in sorted(bindings)
    )
    return _sha256_text(canonical)


def _trusted_signer_fingerprints(
    root: Path, content: bytes | None = None
) -> dict[str, str]:
    path = _safe_repo_path(root, TRUSTED_SIGNERS_PATH)
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(content if content is not None else path.read_bytes())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "gauntlet.trusted_signers.v1"
        or not isinstance(payload.get("signers"), list)
    ):
        return {}
    trusted: dict[str, str] = {}
    for signer in payload["signers"]:
        if not isinstance(signer, dict) or signer.get("active") is not True:
            continue
        identity = signer.get("identity")
        fingerprint = str(signer.get("fingerprint", "")).upper()
        if (
            isinstance(identity, str)
            and identity
            and re.fullmatch(r"[0-9A-F]{40,64}", fingerprint)
        ):
            trusted[identity] = fingerprint
    return trusted


def _certificate_signed_claim_sha256(
    certificate: dict, bundle_root: str | None
) -> str | None:
    if (
        not isinstance(bundle_root, str)
        or re.fullmatch(r"[0-9a-f]{64}", bundle_root) is None
    ):
        return None
    signatures = certificate.get("detached_signatures")
    if not isinstance(signatures, list):
        return None
    declarations: list[dict] = []
    for signature in signatures:
        if not isinstance(signature, dict):
            return None
        declarations.append(
            {
                "signer": signature.get("signer"),
                "role": signature.get("role"),
                "fingerprint": signature.get("fingerprint"),
                "scheme": signature.get("scheme"),
                "signature_path": signature.get("signature_path"),
            }
        )
    claim = {
        key: value
        for key, value in certificate.items()
        if key not in {"detached_signatures", "signed_claim_sha256"}
    }
    claim["detached_signature_declarations"] = declarations
    claim["evidence_bundle_sha256"] = bundle_root
    try:
        canonical = json.dumps(
            claim,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return None
    return _sha256_text(canonical)


def _default_signature_verifier(
    root: Path,
    signature: dict,
    signed_claim_sha256: str,
    *,
    signature_bytes: bytes | None = None,
    keyring_bytes: bytes | None = None,
) -> bool:
    verifier = shutil.which("gpgv")
    signature_path = _safe_repo_path(root, str(signature.get("signature_path", "")))
    keyring_path = _safe_repo_path(root, TRUSTED_KEYRING_PATH)
    expected_fingerprint = str(signature.get("fingerprint", "")).upper()
    if (
        verifier is None
        or signature_path is None
        or keyring_path is None
        or not signature_path.is_file()
        or not keyring_path.is_file()
        or re.fullmatch(r"[0-9A-F]{40,64}", expected_fingerprint) is None
    ):
        return False
    try:
        detached = (
            signature_bytes
            if signature_bytes is not None
            else signature_path.read_bytes()
        )
        trusted_keyring = (
            keyring_bytes if keyring_bytes is not None else keyring_path.read_bytes()
        )
        with tempfile.TemporaryDirectory(prefix="focr-cert-signature-") as tmp:
            payload = Path(tmp) / "bundle-root.txt"
            signature_snapshot = Path(tmp) / "signature.asc"
            keyring_snapshot = Path(tmp) / "trusted-release-keys.gpg"
            payload.write_text(signed_claim_sha256 + "\n", encoding="utf-8")
            signature_snapshot.write_bytes(detached)
            keyring_snapshot.write_bytes(trusted_keyring)
            result = subprocess.run(
                [
                    verifier,
                    "--status-fd=1",
                    "--keyring",
                    str(keyring_snapshot),
                    str(signature_snapshot),
                    str(payload),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
    except (OSError, subprocess.SubprocessError):
        return False
    valid_fingerprints = {
        match.group(1).upper()
        for match in re.finditer(
            r"^\[GNUPG:\] VALIDSIG ([0-9A-F]+)\b", result.stdout, re.MULTILINE
        )
    }
    return result.returncode == 0 and valid_fingerprints == {expected_fingerprint}


def _default_ci_run_verifier(
    root: Path,
    run_id: str,
    current_head: str,
    artifacts: Sequence[dict],
) -> bool:
    gh = shutil.which("gh")
    if gh is None or re.fullmatch(r"[1-9][0-9]*", run_id) is None:
        return False
    try:
        result = subprocess.run(
            [
                gh,
                "run",
                "view",
                run_id,
                "--repo",
                CERTIFICATION_GITHUB_REPOSITORY,
                "--json",
                "databaseId,headSha,status,conclusion,workflowName,event",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else {}
    except (
        OSError,
        ValueError,
        TypeError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ):
        return False
    run_ok = (
        isinstance(payload, dict)
        and str(payload.get("databaseId")) == run_id
        and payload.get("headSha") == current_head
        and payload.get("status") == "completed"
        and payload.get("conclusion") == "success"
        and payload.get("workflowName") == CERTIFICATION_GITHUB_WORKFLOW
        and payload.get("event") == CERTIFICATION_GITHUB_EVENT
    )
    if not run_ok or not artifacts:
        return False
    grouped: dict[str, list[dict]] = {}
    for item in artifacts:
        artifact_name = item.get("source_ci_artifact_name")
        member = item.get("source_ci_artifact_path")
        digest = item.get("sha256")
        member_path = Path(member) if isinstance(member, str) else None
        if (
            not isinstance(artifact_name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", artifact_name) is None
            or member_path is None
            or member_path.is_absolute()
            or member_path.as_posix() != member
            or any(part in {"", ".", ".."} for part in member_path.parts)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            return False
        grouped.setdefault(artifact_name, []).append(item)
    try:
        with tempfile.TemporaryDirectory(prefix="focr-cert-ci-artifacts-") as tmp:
            download_root = Path(tmp)
            for index, (artifact_name, bound_files) in enumerate(
                sorted(grouped.items())
            ):
                destination = download_root / f"artifact-{index}"
                result = subprocess.run(
                    [
                        gh,
                        "run",
                        "download",
                        run_id,
                        "--repo",
                        CERTIFICATION_GITHUB_REPOSITORY,
                        "--name",
                        artifact_name,
                        "--dir",
                        str(destination),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode != 0:
                    return False
                for item in bound_files:
                    member = str(item["source_ci_artifact_path"])
                    member_path = Path(member)
                    candidate = _safe_repo_path(destination, member)
                    if (
                        candidate is None
                        or not candidate.is_file()
                        or any(
                            destination.joinpath(
                                *member_path.parts[:part_count]
                            ).is_symlink()
                            for part_count in range(1, len(member_path.parts) + 1)
                        )
                        or _sha256_file(candidate) != item["sha256"]
                    ):
                        return False
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _strict_evidence_reasons(
    root: Path,
    bundle_dir: Path,
    certificate: dict,
    entries: dict[str, dict],
    *,
    provenance_root: Path | None = None,
    ci_run_verifier: Callable[[Path, str, str, Sequence[dict]], bool] | None = None,
) -> list[str]:
    reasons: list[str] = []
    evidence_classes = certificate.get("evidence_classes")
    if not isinstance(evidence_classes, dict):
        return ["certificate evidence_classes is missing or not an object"]
    documents: dict[str, dict] = {}
    for class_name, (expected_name, expected_schema) in STRICT_BUNDLE_CLASSES.items():
        relative = evidence_classes.get(class_name)
        if not isinstance(relative, str) or not relative:
            reasons.append(f"certificate omits evidence class: {class_name}")
            continue
        if Path(relative).name != expected_name:
            reasons.append(
                f"evidence class {class_name} has noncanonical filename: {relative}"
            )
        if relative not in entries:
            reasons.append(
                f"bundle manifest omits evidence class: {class_name} ({relative})"
            )
            continue
        path = _safe_repo_path(root, relative)
        if path is None or not path.is_file():
            reasons.append(
                f"evidence class {class_name} is missing or unsafe: {relative}"
            )
            continue
        try:
            path.relative_to(bundle_dir)
        except ValueError:
            reasons.append(
                f"evidence class {class_name} is outside the selected bundle: {relative}"
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            reasons.append(f"evidence class {class_name} is unreadable JSON: {error}")
            continue
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != expected_schema
        ):
            reasons.append(
                f"evidence class {class_name} schema is not {expected_schema}"
            )
            continue
        if _parse_utc_timestamp(payload.get("generated_at_utc")) is None:
            reasons.append(f"evidence class {class_name} has no valid generated_at_utc")
            continue
        documents[class_name] = payload

    confidence = documents.get("confidence_gate", {})
    if confidence:
        if confidence.get("release_decision") != "Allow":
            reasons.append("confidence gate release_decision is not Allow")
        if confidence.get("min_verification_pct_observed") != 100.0:
            reasons.append("confidence gate verification percentage is not 100.0")
        if confidence.get("required_suite_pass_rate_pct_observed") != 100.0:
            reasons.append("confidence gate suite pass rate is not 100.0")
        if confidence.get("high_severity_counterexample_count") != 0:
            reasons.append("confidence gate has high-severity counterexamples")
        if confidence.get("constants_enforced") != list(CERTIFICATION_CONSTANTS):
            reasons.append(
                "confidence gate does not enumerate the exact strict constants"
            )
        actuals = certificate.get("required_pass_actuals")
        expected_actuals = {
            "min_verification_pct_observed": confidence.get(
                "min_verification_pct_observed"
            ),
            "required_suite_pass_rate_pct_observed": confidence.get(
                "required_suite_pass_rate_pct_observed"
            ),
            "high_severity_counterexample_count": confidence.get(
                "high_severity_counterexample_count"
            ),
            "max_evidence_age_hours_observed": (
                actuals.get("max_evidence_age_hours_observed")
                if isinstance(actuals, dict)
                else None
            ),
        }
        if actuals != expected_actuals:
            reasons.append(
                "certificate required-pass actuals do not bind the confidence gate"
            )
        if certificate.get("high_severity_counterexamples") != confidence.get(
            "high_severity_counterexample_count"
        ):
            reasons.append(
                "certificate counterexample count does not bind the confidence gate"
            )

    verification = documents.get("verification_contract", {})
    if verification:
        rows = verification.get("rows")
        universe_relative = verification.get("feature_universe_path")
        universe_entry = (
            entries.get(universe_relative)
            if isinstance(universe_relative, str)
            else None
        )
        universe_path = (
            _safe_repo_path(root, universe_relative)
            if isinstance(universe_relative, str)
            else None
        )
        try:
            universe_inside_bundle = (
                universe_path is not None
                and universe_path.is_file()
                and universe_path.relative_to(bundle_dir) is not None
            )
        except ValueError:
            universe_inside_bundle = False
        try:
            universe = (
                json.loads(universe_path.read_text(encoding="utf-8"))
                if universe_inside_bundle
                else None
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            universe = None
        if (
            not isinstance(universe_entry, dict)
            or not isinstance(universe, dict)
            or universe.get("schema_version") != "gauntlet.feature_universe.v1"
            or _parse_utc_timestamp(universe.get("generated_at_utc")) is None
            or universe.get("definition_sha256")
            != CERTIFICATION_FEATURE_UNIVERSE_SHA256
            or universe.get("features") != list(CERTIFICATION_FEATURE_UNIVERSE)
            or verification.get("feature_universe_sha256")
            != universe_entry.get("sha256")
            or verification.get("feature_universe_definition_sha256")
            != CERTIFICATION_FEATURE_UNIVERSE_SHA256
            or certificate.get("feature_universe_sha256")
            != universe_entry.get("sha256")
            or certificate.get("feature_universe_definition_sha256")
            != CERTIFICATION_FEATURE_UNIVERSE_SHA256
        ):
            reasons.append(
                "verification contract lacks the canonical hash-bound in-bundle feature universe"
            )
            expected_pairs: set[tuple[str, str]] = set()
        else:
            features = universe.get("features")
            expected_pairs = set()
            if not isinstance(features, list) or not features:
                reasons.append("feature universe has no required features")
            else:
                for feature in features:
                    if (
                        not isinstance(feature, dict)
                        or not isinstance(feature.get("feature_id"), str)
                        or feature.get("required") is not True
                        or not isinstance(feature.get("proof_obligations"), list)
                        or not feature.get("proof_obligations")
                    ):
                        reasons.append("feature universe contains a malformed feature")
                        continue
                    expected_pairs.update(
                        (feature["feature_id"], obligation)
                        for obligation in feature["proof_obligations"]
                        if isinstance(obligation, str) and obligation
                    )
        if not isinstance(rows, list) or not rows:
            reasons.append("verification contract has no proof-obligation rows")
        else:
            observed_pairs: set[tuple[str, str]] = set()
            claim_sources = certificate.get("claim_sources")
            readiness_evidence = (
                claim_sources.get("readiness_evidence")
                if isinstance(claim_sources, dict)
                and isinstance(claim_sources.get("readiness_evidence"), dict)
                else {}
            )
            for row in rows:
                if not isinstance(row, dict):
                    reasons.append("verification contract contains a malformed row")
                    continue
                pair = (row.get("feature_id"), row.get("proof_obligation"))
                if (
                    not all(isinstance(value, str) and value for value in pair)
                    or pair in observed_pairs
                    or row.get("status") != "pass"
                    or row.get("gate") != "allowed"
                ):
                    reasons.append(
                        "verification contract contains a duplicate or non-pass row"
                    )
                    continue
                observed_pairs.add(pair)
                obligation = str(row["proof_obligation"])
                expected_evidence_paths = [
                    readiness_evidence.get(original)
                    for original in CERTIFICATION_READINESS_EVIDENCE_PATHS.get(
                        obligation, ()
                    )
                ]
                evidence_paths = row.get("evidence_paths")
                evidence_sha256s = row.get("evidence_sha256s")
                expected_hashes = {
                    relative: entries[relative]["sha256"]
                    for relative in expected_evidence_paths
                    if isinstance(relative, str)
                    and isinstance(entries.get(relative), dict)
                    and isinstance(entries[relative].get("sha256"), str)
                }
                if (
                    not expected_evidence_paths
                    or any(
                        not isinstance(relative, str) or not relative
                        for relative in expected_evidence_paths
                    )
                    or evidence_paths != expected_evidence_paths
                    or evidence_sha256s != expected_hashes
                    or len(expected_hashes) != len(expected_evidence_paths)
                ):
                    reasons.append(
                        f"verification row {pair!r} lacks hash-bound manifest evidence"
                    )
            if observed_pairs != expected_pairs:
                reasons.append(
                    "verification contract does not exhaust the feature universe"
                )

    ci_manifest = documents.get("ci_manifest", {})
    if ci_manifest:
        if (
            ci_manifest.get("repository") != CERTIFICATION_GITHUB_REPOSITORY
            or ci_manifest.get("workflow") != CERTIFICATION_GITHUB_WORKFLOW
            or ci_manifest.get("event") != CERTIFICATION_GITHUB_EVENT
        ):
            reasons.append("CI manifest does not pin the canonical repository workflow")
        artifacts = ci_manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            reasons.append("CI manifest has no reconstructable artifacts")
        else:
            ci_relative = evidence_classes.get("ci_manifest")
            expected_ci_paths = set(entries) - {ci_relative}
            observed_ci_paths: set[str] = set()
            run_artifacts: dict[str, list[dict]] = {}
            for item in artifacts:
                if not isinstance(item, dict):
                    reasons.append("CI manifest contains a malformed artifact")
                    continue
                relative = item.get("artifact")
                run_id = str(item.get("source_ci_run_id", ""))
                entry = entries.get(relative) if isinstance(relative, str) else None
                if (
                    not isinstance(relative, str)
                    or relative in observed_ci_paths
                    or not isinstance(entry, dict)
                    or item.get("sha256") != entry.get("sha256")
                    or re.fullmatch(r"[1-9][0-9]*", run_id) is None
                    or item.get("schema_version") != CI_ARTIFACT_BINDING_SCHEMA
                    or re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
                        str(item.get("source_ci_artifact_name", "")),
                    )
                    is None
                    or not isinstance(item.get("source_ci_artifact_path"), str)
                    or not item.get("source_ci_artifact_path")
                ):
                    reasons.append("CI manifest contains an untraceable artifact")
                    continue
                observed_ci_paths.add(relative)
                run_artifacts.setdefault(run_id, []).append(item)
            if observed_ci_paths != expected_ci_paths:
                reasons.append(
                    "CI manifest does not exhaustively cover the evidence manifest"
                )
            verify_ci_run = ci_run_verifier or _default_ci_run_verifier
            for run_id, bound_artifacts in sorted(run_artifacts.items()):
                if not verify_ci_run(
                    provenance_root or root,
                    run_id,
                    str(certificate.get("git_head", "")),
                    bound_artifacts,
                ):
                    reasons.append(
                        f"CI run artifact provenance could not be verified: {run_id}"
                    )

    benchmark = documents.get("benchmark_summary", {})
    if benchmark:
        gates = benchmark.get("pass_over_pass_gates")
        required_gates = {
            "primary_score",
            "geomean",
            "category_geomean",
            "p90",
            "throughput_drop",
        }
        if not isinstance(gates, dict) or set(gates) != required_gates:
            reasons.append(
                "benchmark summary does not carry all five pass-over-pass gates"
            )
        else:
            thresholds = {
                "primary_score": -3.0,
                "geomean": -5.0,
                "category_geomean": -10.0,
                "p90": -15.0,
                "throughput_drop": -5.0,
            }
            for gate_name, minimum in thresholds.items():
                gate = gates[gate_name]
                observed = (
                    gate.get("regression_pct") if isinstance(gate, dict) else None
                )
                if (
                    not isinstance(gate, dict)
                    or gate.get("minimum_pct") != minimum
                    or not isinstance(observed, (int, float))
                    or isinstance(observed, bool)
                    or not math.isfinite(float(observed))
                    or gate.get("passed") is not (float(observed) >= minimum)
                    or float(observed) < minimum
                ):
                    reasons.append(
                        f"benchmark pass-over-pass gate failed or is unbound: {gate_name}"
                    )

    scorecards = documents.get("scorecards", {})
    ratchet = documents.get("ratchet_state", {})
    if scorecards and ratchet:
        lower_bound = scorecards.get("parity_score_lower_bound")
        current_bound = ratchet.get("current_lower_bound")
        if (
            not isinstance(lower_bound, (int, float))
            or isinstance(lower_bound, bool)
            or not math.isfinite(float(lower_bound))
            or not isinstance(current_bound, (int, float))
            or isinstance(current_bound, bool)
            or not math.isfinite(float(current_bound))
            or truncate_score(float(lower_bound)) < float(current_bound)
        ):
            reasons.append("scorecard lower bound does not satisfy the ratchet")
        if certificate.get("parity_score") != lower_bound:
            reasons.append(
                "certificate parity_score does not equal the scorecard lower bound"
            )
        previous = ratchet.get("previous_bound")
        if (
            not isinstance(previous, (int, float))
            or isinstance(previous, bool)
            or not math.isfinite(float(previous))
            or not isinstance(current_bound, (int, float))
            or isinstance(current_bound, bool)
            or not math.isfinite(float(current_bound))
            or float(previous) > float(current_bound)
        ):
            reasons.append("ratchet state is not monotone")
        if ratchet.get("commit_sha") != certificate.get("git_head"):
            reasons.append("ratchet state commit_sha does not match certificate HEAD")
        per_category = ratchet.get("per_category_bounds")
        scorecard_categories = scorecards.get("per_category_lower")
        if (
            not isinstance(per_category, dict)
            or not per_category
            or not isinstance(scorecard_categories, dict)
            or set(per_category) != set(scorecard_categories)
            or any(
                not isinstance(bound, (int, float))
                or isinstance(bound, bool)
                or not math.isfinite(float(bound))
                or not isinstance(scorecard_categories.get(category), (int, float))
                or isinstance(scorecard_categories.get(category), bool)
                or not math.isfinite(float(scorecard_categories[category]))
                or float(scorecard_categories[category]) < float(bound)
                for category, bound in per_category.items()
            )
            or _parse_utc_timestamp(ratchet.get("timestamp")) is None
            or not isinstance(ratchet.get("advance_reason"), str)
            or not ratchet.get("advance_reason")
        ):
            reasons.append("ratchet state lacks valid per-category history invariants")

    critical = documents.get("critical_path_report", {})
    if critical:
        findings = critical.get("findings")
        open_count = critical.get("open_high_critical")
        waived_count = critical.get("waived_high_critical")
        if (
            not isinstance(open_count, int)
            or isinstance(open_count, bool)
            or open_count != 0
            or not isinstance(waived_count, int)
            or isinstance(waived_count, bool)
            or waived_count != 0
            or not isinstance(findings, list)
            or any(
                not isinstance(finding, dict)
                or finding.get("severity") not in {"High", "Critical"}
                or finding.get("status") != "resolved"
                or not isinstance(finding.get("evidence_path"), str)
                or not isinstance(entries.get(finding.get("evidence_path")), dict)
                or finding.get("evidence_sha256")
                != entries[finding["evidence_path"]].get("sha256")
                for finding in findings
            )
        ):
            reasons.append(
                "critical-path report has open, waived, or unproven High/Critical findings"
            )
    return reasons


def _hypothesis_dependency_files(
    bundle_dir: Path,
    contents: dict[str, str | None],
    ledger_paths: Sequence[str],
) -> tuple[set[Path], list[str]]:
    """Return every file whose bytes affect hypothesis closure."""
    dependencies: set[Path] = set()
    reasons: list[str] = []

    def include_tree(path: Path, declared: str) -> None:
        scope = path if path.is_dir() else path.parent
        if not scope.is_dir():
            reasons.append(f"hypothesis dependency is missing: {declared}")
            return
        for candidate in scope.rglob("*"):
            if candidate.is_file() and not candidate.name.startswith("._"):
                dependencies.add(candidate.resolve())

    for ledger in ledger_paths:
        text = contents.get(ledger)
        if not isinstance(text, str):
            continue
        matches = list(_HYPOTHESIS_ENTRY_RE.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section = text[match.start() : end]
            results_fence = _experiment_results_fence(section) or ""
            for relative in _experiment_list_field(
                results_fence, "result_evidence_paths"
            ):
                path = _safe_repo_path(bundle_dir, relative)
                if path is None or not path.exists():
                    reasons.append(
                        f"hypothesis dependency is missing or unsafe: {relative}"
                    )
                    continue
                include_tree(path, relative)
            result_status = _experiment_field_values(results_fence, "result_status")
            if len(result_status) != 1 or result_status[0].upper() != "CONFIRMED_GAP":
                continue
            remediation = _experiment_field_values(
                results_fence, "spawned_remediation_bead"
            )
            if len(remediation) != 1:
                continue
            issues = _safe_repo_path(bundle_dir, ".beads/issues.jsonl")
            if issues is None or not issues.is_file():
                reasons.append(
                    "confirmed-gap hypothesis dependency is missing: "
                    ".beads/issues.jsonl"
                )
            else:
                dependencies.add(issues.resolve())
            proof_relative = f"artifacts/{remediation[0]}/proof_pack"
            proof_pack = _safe_repo_path(bundle_dir, proof_relative)
            if proof_pack is None or not proof_pack.is_dir():
                reasons.append(
                    f"confirmed-gap hypothesis dependency is missing: {proof_relative}"
                )
            else:
                include_tree(proof_pack, proof_relative)
    return dependencies, reasons


def _certificate_source_claim_reasons(
    root: Path, bundle_dir: Path, certificate: dict, entries: dict[str, dict]
) -> list[str]:
    reasons: list[str] = []
    claim_sources = certificate.get("claim_sources")
    readiness_relative = (
        claim_sources.get("release_readiness")
        if isinstance(claim_sources, dict)
        else None
    )
    rounds_relative = (
        claim_sources.get("rounds") if isinstance(claim_sources, dict) else None
    )
    ledger_paths = (
        claim_sources.get("hypothesis_ledgers")
        if isinstance(claim_sources, dict)
        else None
    )
    core_evidence = (
        claim_sources.get("core_evidence") if isinstance(claim_sources, dict) else None
    )
    readiness_evidence = (
        claim_sources.get("readiness_evidence")
        if isinstance(claim_sources, dict)
        else None
    )
    if (
        not isinstance(readiness_relative, str)
        or not isinstance(rounds_relative, str)
        or not isinstance(ledger_paths, list)
        or len(ledger_paths) != len(HYPOTHESIS_LEDGER_PATHS)
        or any(not isinstance(path, str) or not path for path in ledger_paths)
        or {Path(path).name for path in ledger_paths}
        != {Path(path).name for path in HYPOTHESIS_LEDGER_PATHS}
        or not isinstance(core_evidence, dict)
        or set(core_evidence) != set(CORE_EVIDENCE_PATHS)
        or any(
            not isinstance(relative, str) or not relative
            for relative in core_evidence.values()
        )
        or len(set(core_evidence.values())) != len(core_evidence)
        or not isinstance(readiness_evidence, dict)
        or set(readiness_evidence) != set(CERTIFICATION_PROOF_EVIDENCE_PATHS)
        or any(
            not isinstance(relative, str) or not relative
            for relative in readiness_evidence.values()
        )
        or len(set(readiness_evidence.values())) != len(readiness_evidence)
    ):
        return ["certificate claim_sources are missing or noncanonical"]
    expected_core_claims = {
        "docs/gauntlet/RELEASE_READINESS.json": readiness_relative,
        "docs/gauntlet/ROUNDS.jsonl": rounds_relative,
        **{
            original: next(
                path for path in ledger_paths if Path(path).name == Path(original).name
            )
            for original in HYPOTHESIS_LEDGER_PATHS
        },
    }
    if any(
        core_evidence.get(original) != relative
        for original, relative in expected_core_claims.items()
    ):
        reasons.append(
            "canonical claim sources do not match the core-evidence snapshot mapping"
        )
    source_paths = tuple(
        dict.fromkeys((*core_evidence.values(), *readiness_evidence.values()))
    )
    for relative in source_paths:
        if relative not in entries:
            reasons.append(f"bundle manifest omits signed claim source: {relative}")
        path = _safe_repo_path(root, relative)
        try:
            inside_bundle = (
                path is not None
                and path.is_file()
                and path.relative_to(bundle_dir) is not None
            )
        except ValueError:
            inside_bundle = False
        if not inside_bundle:
            reasons.append(
                f"signed claim source is outside the selected bundle: {relative}"
            )
    readiness_path = _safe_repo_path(root, readiness_relative)
    try:
        readiness = (
            json.loads(readiness_path.read_text(encoding="utf-8"))
            if readiness_path
            else None
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        readiness = None
        reasons.append(f"release-readiness claim source is unreadable: {error}")
    if not isinstance(readiness, dict):
        reasons.append("release-readiness claim source is not a JSON object")
    else:
        cells = readiness.get("cells")
        canonical_cells = (
            isinstance(cells, list)
            and all(isinstance(cell, dict) for cell in cells)
            and tuple(cell.get("cell") for cell in cells)
            == CERTIFICATION_READINESS_CELLS
            and all(cell.get("status") in {"green", "red"} for cell in cells)
            and readiness.get("schema_version") == "gauntlet.release_readiness.v1"
            and readiness.get("artifact") == "franken_ocr.release_readiness.v1"
            and _parse_utc_timestamp(readiness.get("generated_at_utc")) is not None
            and readiness.get("generated_by")
            == "scripts/gauntlet_cert.py --release-readiness"
            and readiness.get("cell_set_sha256")
            == CERTIFICATION_READINESS_CELL_SET_SHA256
            and certificate.get("readiness_cell_set_sha256")
            == CERTIFICATION_READINESS_CELL_SET_SHA256
        )
        if not canonical_cells:
            reasons.append(
                "release-readiness does not contain the exact canonical cell set"
            )
        else:
            external_cells = [
                cell for cell in cells if cell["cell"] != "certification_bundle"
            ]
            for cell in external_cells:
                expected_evidence = [
                    readiness_evidence[original]
                    for original in CERTIFICATION_READINESS_EVIDENCE_PATHS[cell["cell"]]
                ]
                if cell.get("evidence_paths") != expected_evidence:
                    reasons.append(
                        "release-readiness cell lacks canonical manifest evidence: "
                        f"{cell['cell']}"
                    )
            all_blocking = [cell["cell"] for cell in cells if cell["status"] == "red"]
            if (
                readiness.get("green")
                != sum(cell["status"] == "green" for cell in cells)
                or readiness.get("red") != len(all_blocking)
                or readiness.get("blocking_cells") != all_blocking
                or readiness.get("ship") != (not all_blocking)
            ):
                reasons.append(
                    "release-readiness aggregate fields do not match canonical cells"
                )
            blocking = [
                cell["cell"] for cell in external_cells if cell["status"] == "red"
            ]
            derived_readiness = {
                "green": sum(cell["status"] == "green" for cell in external_cells),
                "red": len(blocking),
                "blocking_cells": blocking,
                "ship": not blocking,
            }
            if certificate.get("readiness") != derived_readiness:
                reasons.append(
                    "certificate readiness does not match the hashed readiness cells"
                )

    rounds_path = _safe_repo_path(root, rounds_relative)
    try:
        rounds = (
            [
                json.loads(line)
                for line in rounds_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if rounds_path
            else []
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        rounds = []
        reasons.append(f"round-history claim source is unreadable: {error}")
    for round_reason in _round_history_reasons(rounds):
        reasons.append(f"round-history claim source is invalid: {round_reason}")
    contents: dict[str, str | None] = {}
    for relative in ledger_paths:
        path = _safe_repo_path(root, relative)
        try:
            contents[relative] = path.read_text(encoding="utf-8") if path else None
        except OSError:
            contents[relative] = None
    hypotheses = hypothesis_texts_verdict(contents, ledger_paths, bundle_dir)
    dependency_files, dependency_reasons = _hypothesis_dependency_files(
        bundle_dir, contents, ledger_paths
    )
    reasons.extend(dependency_reasons)
    repository_root = root.resolve()
    for dependency in sorted(dependency_files):
        try:
            relative = dependency.relative_to(repository_root).as_posix()
        except ValueError:
            reasons.append(
                f"hypothesis dependency escapes the repository: {dependency}"
            )
            continue
        if relative not in entries:
            reasons.append(f"bundle manifest omits hypothesis dependency: {relative}")
    derived_convergence = convergence_verdict(rounds, hypotheses)
    if certificate.get("convergence") != derived_convergence:
        reasons.append(
            "certificate convergence does not match hashed rounds and hypothesis ledgers"
        )
    return reasons


def certificate_bundle_verdict(
    root: Path,
    current_head: str,
    required_artifacts: Sequence[str] = (),
    *,
    bundle_dir: Path | None = None,
    now: datetime | None = None,
    worktree_state: dict | None = None,
    signature_verifier: Callable[[Path, dict, str], bool] | None = None,
    trusted_signers: dict[str, str] | None = None,
    ci_run_verifier: Callable[[Path, str, str, Sequence[dict]], bool] | None = None,
) -> dict:
    """Verify the complete strict certificate against live source and evidence."""
    reasons: list[str] = []
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    requested_bundle = bundle_dir or root / "docs/gauntlet/bundle"
    bundle_dir, output_reasons = _safe_output_dir(root, str(requested_bundle))
    if bundle_dir is None:
        return {"ok": False, "reasons": output_reasons}
    certificate_path = bundle_dir / "release_certificate.json"
    bundle_path = bundle_dir / "certification_bundle.json"
    try:
        certificate_bytes = certificate_path.read_bytes()
        bundle_bytes = bundle_path.read_bytes()
        certificate = json.loads(certificate_bytes)
        bundle = json.loads(bundle_bytes)
        control_file_hashes = {
            certificate_path: hashlib.sha256(certificate_bytes).hexdigest(),
            bundle_path: hashlib.sha256(bundle_bytes).hexdigest(),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return {
            "ok": False,
            "reasons": [f"missing or unreadable certificate bundle: {error}"],
        }
    if not isinstance(certificate, dict) or not isinstance(bundle, dict):
        return {
            "ok": False,
            "reasons": ["certificate and bundle must both be JSON objects"],
        }

    if certificate.get("schema_version") != STRICT_CERTIFICATE_SCHEMA:
        reasons.append(f"certificate schema_version is not {STRICT_CERTIFICATE_SCHEMA}")
    if certificate.get("artifact") != STRICT_CERTIFICATE_ARTIFACT:
        reasons.append(f"certificate artifact is not {STRICT_CERTIFICATE_ARTIFACT}")
    if certificate.get("template") != STRICT_CERTIFICATE_SCHEMA:
        reasons.append(f"certificate template is not {STRICT_CERTIFICATE_SCHEMA}")
    if certificate.get("certified") is not True:
        reasons.append("certificate verdict is not certified")
    if certificate.get("constants") != CERTIFICATION_CONSTANTS:
        reasons.append("certificate required-pass constants are missing or altered")
    if certificate.get("project") != "franken_ocr":
        reasons.append("certificate project identity is missing or incorrect")
    package_version = _cargo_package_version(root)
    if package_version is None or certificate.get("version") != package_version:
        reasons.append("certificate version does not match Cargo.toml")
    if not _valid_git_head(current_head):
        reasons.append("current git HEAD is unavailable")
    elif certificate.get("git_head") != current_head:
        reasons.append(
            f"certificate git_head {certificate.get('git_head')!r} != current HEAD {current_head}"
        )
    reference = certificate.get("reference")
    if (
        not isinstance(reference, dict)
        or reference.get("model_commit") != UNLIMITED_OCR_MODEL_COMMIT
    ):
        reasons.append("certificate reference identity is missing or unpinned")
    issued_at = _parse_utc_timestamp(certificate.get("issued_at"))
    issued_reason = _age_reason(
        issued_at, now, "certificate", CERTIFICATION_MAX_EVIDENCE_AGE_HOURS
    )
    if issued_reason:
        reasons.append(issued_reason)

    live_worktree = worktree_state or _git_worktree_state(root)
    live_describe = _git_output(root, "describe", "--tags", "--always", "--dirty")
    if live_worktree.get("status_ok") is not True:
        reasons.append("git worktree status could not be verified")
    if live_worktree.get("clean") is not True:
        reasons.append(
            f"git worktree is dirty ({live_worktree.get('dirty_path_count', '?')} status entries)"
        )
    recorded_worktree = certificate.get("git_worktree")
    if not isinstance(recorded_worktree, dict) or recorded_worktree != live_worktree:
        reasons.append(
            "certificate git_worktree state does not match the live worktree"
        )
    if worktree_state is None and certificate.get("git_describe") != live_describe:
        reasons.append(
            "certificate git_describe does not match the live dirty-state description"
        )

    readiness = certificate.get("readiness")
    readiness_green = readiness.get("green") if isinstance(readiness, dict) else None
    if not isinstance(readiness, dict) or not (
        readiness.get("ship") is True
        and readiness.get("red") == 0
        and readiness.get("blocking_cells") == []
        and isinstance(readiness_green, int)
        and not isinstance(readiness_green, bool)
        and readiness_green > 0
    ):
        reasons.append("certificate readiness is not an all-green ship decision")
    convergence = certificate.get("convergence")
    convergence_rounds = (
        convergence.get("rounds") if isinstance(convergence, dict) else None
    )
    if not isinstance(convergence, dict) or not (
        convergence.get("converged") is True
        and convergence.get("result") == "pass"
        and isinstance(convergence_rounds, int)
        and not isinstance(convergence_rounds, bool)
        and convergence_rounds >= MIN_ROUNDS
        and convergence.get("tail_clean") is True
        and convergence.get("hypotheses_resolved") is True
    ):
        reasons.append("certificate convergence claim is incomplete or false")
    actuals = certificate.get("required_pass_actuals")
    if not isinstance(actuals, dict) or not (
        actuals.get("min_verification_pct_observed") == 100.0
        and actuals.get("required_suite_pass_rate_pct_observed") == 100.0
        and actuals.get("high_severity_counterexample_count") == 0
        and isinstance(actuals.get("max_evidence_age_hours_observed"), (int, float))
        and not isinstance(actuals.get("max_evidence_age_hours_observed"), bool)
        and actuals.get("max_evidence_age_hours_observed")
        <= CERTIFICATION_MAX_EVIDENCE_AGE_HOURS
    ):
        reasons.append(
            "certificate observed required-pass values do not satisfy the strict constants"
        )
    if certificate.get("high_severity_counterexamples") != 0:
        reasons.append(
            "certificate has a nonzero or unknown high-severity counterexample count"
        )
    parity_score = certificate.get("parity_score")
    if (
        not isinstance(parity_score, (int, float))
        or isinstance(parity_score, bool)
        or not math.isfinite(float(parity_score))
        or truncate_score(float(parity_score)) != parity_score
    ):
        reasons.append(
            "certificate parity_score is not a truncated conformal lower bound"
        )
    if certificate.get("refusal_reasons") != []:
        reasons.append("a certified certificate must have no refusal reasons")

    if bundle.get("schema_version") != STRICT_BUNDLE_SCHEMA:
        reasons.append(f"bundle schema_version is not {STRICT_BUNDLE_SCHEMA}")
    if bundle.get("artifact") != STRICT_BUNDLE_ARTIFACT:
        reasons.append(f"bundle artifact is not {STRICT_BUNDLE_ARTIFACT}")
    manifest = bundle.get("manifest")
    if not isinstance(manifest, list):
        reasons.append("bundle manifest is missing or not a list")
        manifest = []
    entries: dict[str, dict] = {}
    entry_snapshots: dict[str, bytes] = {}
    max_age = 0.0
    for entry in manifest:
        if not isinstance(entry, dict) or not isinstance(entry.get("artifact"), str):
            reasons.append("bundle manifest contains a malformed entry")
            continue
        relative = entry["artifact"]
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            reasons.append(
                f"bundle manifest contains a noncanonical artifact: {relative}"
            )
            continue
        if relative in entries:
            reasons.append(f"bundle manifest contains duplicate artifact: {relative}")
            continue
        entries[relative] = entry
    declared_signature_paths = (
        {
            signature.get("signature_path")
            for signature in certificate.get("detached_signatures", [])
            if isinstance(signature, dict)
            and isinstance(signature.get("signature_path"), str)
        }
        if isinstance(certificate.get("detached_signatures"), list)
        else set()
    )
    allowed_unmanifested = {
        str(certificate_path.relative_to(root.resolve())),
        str(bundle_path.relative_to(root.resolve())),
        str((bundle_dir / "FINAL_GAUNTLET_REPORT.md").relative_to(root.resolve())),
        *declared_signature_paths,
    }
    actual_bundle_files = {
        str(path.relative_to(root.resolve()))
        for path in bundle_dir.rglob("*")
        if path.is_file() and not path.name.startswith("._")
    }
    unmanifested = sorted(actual_bundle_files - set(entries) - allowed_unmanifested)
    if unmanifested:
        reasons.append(
            "selected bundle contains unmanifested evidence files: "
            + ", ".join(unmanifested)
        )
    for required in required_artifacts:
        if required not in entries:
            reasons.append(f"bundle manifest omits required artifact: {required}")
    for relative, entry in entries.items():
        if "error" in entry:
            reasons.append(f"bundle artifact error for {relative}: {entry['error']}")
            continue
        path = _safe_repo_path(root, relative)
        expected = entry.get("sha256")
        if path is None or not path.is_file():
            reasons.append(f"bundle artifact is missing or unsafe: {relative}")
            continue
        try:
            path.relative_to(bundle_dir)
        except ValueError:
            reasons.append(
                f"bundle artifact is outside the selected bundle: {relative}"
            )
            continue
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            reasons.append(f"bundle artifact has invalid SHA-256: {relative}")
            continue
        try:
            artifact_bytes = path.read_bytes()
        except OSError as error:
            reasons.append(f"bundle artifact is unreadable: {relative}: {error}")
            continue
        entry_snapshots[relative] = artifact_bytes
        if hashlib.sha256(artifact_bytes).hexdigest() != expected:
            reasons.append(f"bundle artifact hash mismatch: {relative}")
        try:
            timestamp, timestamp_source = _artifact_native_timestamp(
                path, artifact_bytes
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            reasons.append(
                f"bundle artifact timestamp unreadable for {relative}: {error}"
            )
            continue
        recorded_timestamp = _parse_utc_timestamp(entry.get("timestamp_utc"))
        if (
            recorded_timestamp != timestamp
            or entry.get("timestamp_source") != timestamp_source
        ):
            reasons.append(f"bundle artifact timestamp binding mismatch: {relative}")
        age_hours = (now - timestamp).total_seconds() / 3600.0
        max_age = max(max_age, age_hours)
        age_reason = _age_reason(timestamp, now, f"bundle artifact {relative}")
        if age_reason:
            reasons.append(age_reason)

    root_hash = _bundle_root_sha256(manifest)
    if root_hash is None:
        reasons.append("bundle root cannot be computed from the manifest")
    else:
        if bundle.get("bundle_root_sha256") != root_hash:
            reasons.append("bundle root hash does not match the manifest")
        if certificate.get("evidence_bundle_sha256") != root_hash:
            reasons.append(
                "certificate evidence_bundle_sha256 does not match the manifest"
            )
    bundle_relative = bundle_dir.relative_to(root.resolve())
    with tempfile.TemporaryDirectory(prefix="focr-cert-evidence-snapshot-") as tmp:
        snapshot_root = Path(tmp)
        for relative, artifact_bytes in entry_snapshots.items():
            snapshot_path = _safe_repo_path(snapshot_root, relative)
            if snapshot_path is None:
                reasons.append(f"bundle artifact cannot be snapshotted: {relative}")
                continue
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(artifact_bytes)
        snapshot_bundle_dir = snapshot_root / bundle_relative
        reasons.extend(
            _strict_evidence_reasons(
                snapshot_root,
                snapshot_bundle_dir,
                certificate,
                entries,
                provenance_root=root,
                ci_run_verifier=ci_run_verifier,
            )
        )
        reasons.extend(
            _certificate_source_claim_reasons(
                snapshot_root, snapshot_bundle_dir, certificate, entries
            )
        )

    signers = certificate.get("signers")
    signatures = certificate.get("detached_signatures")
    if (
        not isinstance(signers, list)
        or any(not isinstance(signer, str) or not signer for signer in signers)
        or len(signers) != CERTIFICATION_REQUIRED_SIGNERS
        or len(set(signers)) != CERTIFICATION_REQUIRED_SIGNERS
    ):
        reasons.append("certificate must declare exactly three distinct signers")
        signers = []
    if (
        not isinstance(signatures, list)
        or len(signatures) != CERTIFICATION_REQUIRED_SIGNERS
    ):
        reasons.append("certificate must declare exactly three detached signatures")
        signatures = []
    signer_roles: set[str] = set()
    signature_signers: set[str] = set()
    signature_fingerprints: set[str] = set()
    signature_paths: set[str] = set()
    signature_file_hashes: dict[Path, str] = {}
    trust_file_hashes: dict[Path, str] = {}
    trusted_registry_bytes: bytes | None = None
    keyring_bytes: bytes | None = None
    if trusted_signers is None:
        for relative in (TRUSTED_SIGNERS_PATH, TRUSTED_KEYRING_PATH):
            path = _safe_repo_path(root, relative)
            try:
                content = path.read_bytes() if path is not None else None
            except OSError:
                content = None
            if path is None or content is None:
                reasons.append(f"release trust artifact is missing: {relative}")
                continue
            trust_file_hashes[path] = hashlib.sha256(content).hexdigest()
            if relative == TRUSTED_SIGNERS_PATH:
                trusted_registry_bytes = content
            else:
                keyring_bytes = content
    trusted = {
        identity: fingerprint.upper()
        for identity, fingerprint in (
            trusted_signers
            if trusted_signers is not None
            else _trusted_signer_fingerprints(root, trusted_registry_bytes)
        ).items()
        if isinstance(identity, str)
        and isinstance(fingerprint, str)
        and re.fullmatch(r"[0-9A-Fa-f]{40,64}", fingerprint)
    }
    signed_claim_sha256 = _certificate_signed_claim_sha256(certificate, root_hash)
    if signed_claim_sha256 is None:
        reasons.append("certificate signed claim cannot be canonicalized")
    elif certificate.get("signed_claim_sha256") != signed_claim_sha256:
        reasons.append(
            "certificate signed_claim_sha256 does not authenticate its claims"
        )
    for signature in signatures:
        if not isinstance(signature, dict):
            reasons.append("certificate contains a malformed detached signature")
            continue
        signer = signature.get("signer")
        role = signature.get("role")
        fingerprint = str(signature.get("fingerprint", "")).upper()
        signature_relative = signature.get("signature_path")
        if signer not in signers or not isinstance(role, str) or not role:
            reasons.append("detached signature signer/role is not declared")
            continue
        if trusted.get(signer) != fingerprint:
            reasons.append(
                f"detached signature signer is not pinned to a trusted fingerprint: {signer}"
            )
            continue
        if not isinstance(signature_relative, str) or not signature_relative:
            reasons.append(f"detached signature path is missing for {signer}")
            continue
        signature_relative_path = Path(signature_relative)
        if (
            signature_relative_path.is_absolute()
            or signature_relative_path.as_posix() != signature_relative
            or any(part in {"", ".", ".."} for part in signature_relative_path.parts)
        ):
            reasons.append(f"detached signature path is noncanonical for {signer}")
            continue
        if (
            signer in signature_signers
            or role in signer_roles
            or fingerprint in signature_fingerprints
            or signature_relative in signature_paths
        ):
            reasons.append(
                "detached signatures do not have distinct signers, roles, fingerprints, and files"
            )
            continue
        signature_path = (
            _safe_repo_path(root, signature_relative)
            if isinstance(signature_relative, str)
            else None
        )
        try:
            signature_inside_bundle = (
                signature_path is not None
                and signature_path.is_file()
                and signature_path.relative_to(bundle_dir) is not None
            )
        except ValueError:
            signature_inside_bundle = False
        if not signature_inside_bundle:
            reasons.append(
                f"detached signature file is missing or outside the selected bundle: {signer}"
            )
            continue
        try:
            signature_bytes = signature_path.read_bytes()
        except OSError:
            reasons.append(f"detached signature file is unreadable for {signer}")
            continue
        signature_file_hashes[signature_path] = hashlib.sha256(
            signature_bytes
        ).hexdigest()
        signature_signers.add(signer)
        signer_roles.add(role)
        signature_fingerprints.add(fingerprint)
        signature_paths.add(str(signature_relative))
        if signature.get("scheme") != "openpgp-detached":
            reasons.append(f"detached signature for {signer} has the wrong scheme")
        elif signed_claim_sha256 is None:
            reasons.append(f"detached signature verification failed for {signer}")
        elif signature_verifier is not None:
            if not signature_verifier(root, signature, signed_claim_sha256):
                reasons.append(f"detached signature verification failed for {signer}")
        elif not _default_signature_verifier(
            root,
            signature,
            signed_claim_sha256,
            signature_bytes=signature_bytes,
            keyring_bytes=keyring_bytes,
        ):
            reasons.append(f"detached signature verification failed for {signer}")
    if len(signature_signers) != CERTIFICATION_REQUIRED_SIGNERS:
        reasons.append("fewer than three independent detached signatures verified")
    if signer_roles != CERTIFICATION_REQUIRED_SIGNATURE_ROLES:
        reasons.append("detached signatures do not cover the exact required roles")
    if len(signature_fingerprints) != CERTIFICATION_REQUIRED_SIGNERS:
        reasons.append("detached signatures do not cover three trusted fingerprints")
    if set(signers) != signature_signers:
        reasons.append("certificate signer list does not exactly match signatures")

    recorded_max_age = (
        actuals.get("max_evidence_age_hours_observed")
        if isinstance(actuals, dict)
        else None
    )
    if isinstance(recorded_max_age, (int, float)) and not isinstance(
        recorded_max_age, bool
    ):
        if abs(float(recorded_max_age) - max_age) > 0.02:
            reasons.append(
                "certificate max evidence age does not match live verification"
            )
    for control_path, initial_digest in control_file_hashes.items():
        try:
            control_changed = (
                not control_path.is_file()
                or _sha256_file(control_path) != initial_digest
            )
        except OSError:
            control_changed = True
        if control_changed:
            reasons.append(
                f"bundle control file changed during verification: {control_path.name}"
            )
    for protected_path, initial_digest in {
        **trust_file_hashes,
        **signature_file_hashes,
    }.items():
        try:
            protected_changed = (
                not protected_path.is_file()
                or _sha256_file(protected_path) != initial_digest
            )
        except OSError:
            protected_changed = True
        if protected_changed:
            reasons.append(
                "release trust/signature file changed during verification: "
                f"{protected_path}"
            )
    for relative, entry in entries.items():
        if "error" in entry:
            continue
        path = _safe_repo_path(root, relative)
        try:
            artifact_changed = (
                path is None
                or not path.is_file()
                or _sha256_file(path) != entry.get("sha256")
            )
        except OSError:
            artifact_changed = True
        if artifact_changed:
            reasons.append(f"bundle artifact changed during verification: {relative}")
            continue
        try:
            final_timestamp, final_timestamp_source = _artifact_native_timestamp(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            reasons.append(f"bundle artifact changed during verification: {relative}")
            continue
        if (
            _parse_utc_timestamp(entry.get("timestamp_utc")) != final_timestamp
            or entry.get("timestamp_source") != final_timestamp_source
        ):
            reasons.append(f"bundle artifact changed during verification: {relative}")
    _, final_output_reasons = _safe_output_dir(root, str(bundle_dir))
    reasons.extend(
        f"bundle output changed during verification: {reason}"
        for reason in final_output_reasons
    )
    if worktree_state is None:
        final_head = _git_output(root, "rev-parse", "HEAD")
        final_worktree = _git_worktree_state(root)
        final_describe = _git_output(root, "describe", "--tags", "--always", "--dirty")
        if final_head != current_head:
            reasons.append("git HEAD changed during certificate verification")
        if final_worktree != live_worktree:
            reasons.append("git worktree changed during certificate verification")
        if final_describe != live_describe:
            reasons.append("git describe changed during certificate verification")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "bundle_dir": str(bundle_dir),
        "max_evidence_age_hours": round(max_age, 2),
    }


def build_evidence_manifest(
    root: Path,
    now_timestamp: float,
    evidence_paths: Sequence[str] = CORE_EVIDENCE_PATHS,
    max_age_hours: float = CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
) -> tuple[list[dict], list[str]]:
    manifest: list[dict] = []
    stale: list[str] = []
    for relative in evidence_paths:
        path = _safe_repo_path(root, relative)
        if path is None:
            manifest.append({"artifact": relative, "error": "path escapes repository"})
            stale.append(relative)
            continue
        try:
            timestamp, timestamp_source = _artifact_native_timestamp(path)
            age_hours = (
                datetime.fromtimestamp(now_timestamp, timezone.utc) - timestamp
            ).total_seconds() / 3600.0
            entry = {
                "artifact": relative,
                "sha256": _sha256_file(path),
                "age_hours": round(age_hours, 2),
                "timestamp_utc": _timestamp_text(timestamp),
                "timestamp_source": timestamp_source,
            }
            manifest.append(entry)
            if age_hours > max_age_hours or age_hours < -5.0 / 60.0:
                stale.append(relative)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            manifest.append({"artifact": relative, "error": str(error)})
            stale.append(relative)
    return manifest, stale


def from_parity(md_path: str, residuals_path: str | None, out_path: str | None) -> int:
    """Emit the surface-pillar scorecard computed from the REAL scoreboard.

    With no residual history the conformal band bootstraps WIDE (halfwidth
    1.0 -> lower bound 0.0) per the §3.2 pitfall guard — the artifact says so
    explicitly rather than inventing confidence it does not have. Residual
    history accrues one entry per gauntlet round (|observed - Beta-mean|).
    """
    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()
    cats = categories_from_parity(md_text)
    residuals: list[float] = []
    if residuals_path and os.path.exists(residuals_path):
        with open(residuals_path, encoding="utf-8") as f:
            residuals = json.load(f)
    sc = compute_scorecard(cats, residuals, confidence=0.95)
    debt = {
        c.category_id: round(c.weighted_excluded, 6)
        for c in cats
        if c.weighted_excluded > 0
    }
    artifact = {
        "artifact": "franken_ocr.gauntlet.scorecard.v1",
        "source": md_path,
        "pillar": "surface",
        "point_estimate": sc.point_estimate,
        "parity_score_lower_bound": sc.lower_bound,
        "conformal_halfwidth": round(sc.conformal_halfwidth, 6),
        "residual_history_n": len(residuals),
        "band_note": (
            "no residual history yet: bootstrap band 1.0 -> lower bound 0 (maximally conservative, §3.2)"
            if not residuals
            else "band from held-out round residuals"
        ),
        "per_category_mean": sc.per_category_mean,
        "per_category_lower": sc.per_category_lower,
        "excluded_weight_debt": debt,
    }
    text = json.dumps(artifact, sort_keys=True, indent=1)
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


# Convergence (METHODOLOGY §7): >=10 full rounds, the last >=2 consecutive
# rounds each produced <3 new genuine findings, AND every hypothesis ledger is
# present with no unresolved entry. Exits nonzero until all three conditions
# hold — this is the gate bd-wp8.8 runs behind.
MIN_ROUNDS = 10
CLEAN_ROUNDS = 2
CLEAN_FINDING_CEILING = 3


def _round_history_reasons(rounds: object) -> list[str]:
    if not isinstance(rounds, list):
        return ["round history is not a list"]
    reasons: list[str] = []
    required_keys = {"round", "date", "new_findings", "pillars", "notes"}
    required_pillars = {"conformance", "surface", "perf"}
    for expected_round, record in enumerate(rounds, start=1):
        if not isinstance(record, dict) or set(record) != required_keys:
            reasons.append(f"round {expected_round} has a noncanonical schema")
            continue
        round_number = record.get("round")
        findings = record.get("new_findings")
        pillars = record.get("pillars")
        if (
            not isinstance(round_number, int)
            or isinstance(round_number, bool)
            or round_number != expected_round
        ):
            reasons.append(f"round {expected_round} has a nonsequential round id")
        if not isinstance(findings, int) or isinstance(findings, bool) or findings < 0:
            reasons.append(f"round {expected_round} has invalid new_findings")
        if (
            not isinstance(record.get("date"), str)
            or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", record["date"]) is None
            or _parse_utc_timestamp(record["date"]) is None
        ):
            reasons.append(f"round {expected_round} has an invalid date")
        if (
            not isinstance(pillars, dict)
            or set(pillars) != required_pillars
            or any(
                not isinstance(value, str) or not value for value in pillars.values()
            )
        ):
            reasons.append(f"round {expected_round} has invalid pillar evidence")
        if not isinstance(record.get("notes"), str) or not record["notes"]:
            reasons.append(f"round {expected_round} has invalid notes")
    return reasons


def convergence_verdict(rounds: list[dict], hypotheses: dict | None = None) -> dict:
    round_errors = _round_history_reasons(rounds)
    n = len(rounds) if isinstance(rounds, list) else 0
    tail = (
        rounds[-CLEAN_ROUNDS:]
        if isinstance(rounds, list) and not round_errors and n >= CLEAN_ROUNDS
        else []
    )
    tail_clean = len(tail) == CLEAN_ROUNDS and all(
        r["new_findings"] < CLEAN_FINDING_CEILING for r in tail
    )
    hypotheses_resolved = bool(hypotheses and hypotheses.get("resolved") is True)
    converged = (
        not round_errors and n >= MIN_ROUNDS and tail_clean and hypotheses_resolved
    )
    return {
        "check": "gauntlet-convergence",
        "rounds": n,
        "min_rounds": MIN_ROUNDS,
        "tail_clean": tail_clean,
        "tail_findings": [int(r.get("new_findings", -1)) for r in tail],
        "clean_ceiling": CLEAN_FINDING_CEILING,
        "round_history_errors": round_errors,
        "hypotheses_resolved": hypotheses_resolved,
        "hypothesis_ledgers": hypotheses
        or {
            "resolved": False,
            "missing": list(HYPOTHESIS_LEDGER_PATHS),
            "unresolved": [],
            "entries": 0,
        },
        "converged": converged,
        "result": "pass" if converged else "fail",
    }


def convergence(rounds_path: str) -> int:
    root = _repo_root()
    rounds: list[dict] = []
    if os.path.exists(rounds_path):
        with open(rounds_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rounds.append(json.loads(line))
    verdict = convergence_verdict(rounds, hypothesis_ledger_verdict(root))
    verdict["rounds_file"] = rounds_path
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["converged"] else 1


# --------------------------------------------------------------------------- #
# bd-re8.15 — wiring the e-processes to the LIVE invariant streams.
#
# The four load-bearing invariants already EMIT structured NDJSON wherever
# they are exercised (the determinism gates, `robot selftest`, the i32
# overflow proofs, the R-SWA/KV bound gates). The fold below maps those
# real lines onto e-process observations and persists the per-invariant
# state ACROSS runs — the e-value only ever multiplies (Ville's inequality
# holds over the whole history; resetting would forge the guarantee).
# --------------------------------------------------------------------------- #

# invariant -> lowercase substrings matched against the line's identifying
# fields. The mapping is deliberately explicit and versioned here: adding an
# invariant means adding its emission-site vocabulary alongside.
EPROCESS_MATCHERS: list[tuple[str, tuple[str, ...]]] = [
    ("INV-DETERMINISM", ("determin",)),
    ("INV-SIMD-SCALAR", ("selftest", "simd")),
    ("INV-I32-NOOVERFLOW", ("overflow",)),
    ("INV-KV-CAP", ("kv_cap", "kv_bound", "kvcache_bound", "rswa_bound")),
]

_EPROCESS_ID_FIELDS = (
    "test",
    "case",
    "check",
    "gate",
    "suite",
    "relation",
    "assertion",
    "invariant",
    "command",
)


def classify_invariant_line(obj: dict) -> tuple[str, bool] | None:
    """Map one structured log line to (invariant, alarm) or None.

    Only lines carrying an explicit pass/fail verdict count as observations —
    skips (`skip_no_model`) are NOT evidence in either direction."""
    result = str(obj.get("result", obj.get("verdict", "")))
    if result not in ("pass", "fail"):
        return None
    hay = " ".join(str(obj.get(f, "")) for f in _EPROCESS_ID_FIELDS).lower()
    for invariant, needles in EPROCESS_MATCHERS:
        if any(n in hay for n in needles):
            return (invariant, result == "fail")
    return None


def _eprocess_from_state(name: str, saved: dict | None) -> EProcess:
    ep = EProcess.for_invariant(name)
    if saved:
        ep.e_value = float(saved["e_value"])
        ep.obs_count = int(saved["obs_count"])
        ep.rejected_at = saved.get("rejected_at")
    return ep


# --------------------------------------------------------------------------- #
# bd-wp8.10 — the release-readiness scorecard (the all-green ship gate).
#
# Reads each cell's EVIDENCE ARTIFACT and asserts green; a missing or stale
# artifact is a RED cell, and ANY red cell exits nonzero — the gate is hard,
# not advisory (G7). Cells whose delivering bead is still open are honestly
# RED with the owning bead named; the gate goes green when the work exists,
# never before.
# --------------------------------------------------------------------------- #


def _cell(name: str, status: str, evidence: str, detail: str = "") -> dict:
    out = {"cell": name, "status": status, "evidence": evidence}
    if detail:
        out["detail"] = detail
    return out


def release_readiness(
    out_path: str | None,
    *,
    bundle_dir: Path | None = None,
    now: datetime | None = None,
    worktree_state: dict | None = None,
    evidence_path_mapping: dict[str, str] | None = None,
) -> int:
    root = _repo_root()
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    current_head = _git_output(root, "rev-parse", "HEAD")

    def p(rel: str) -> str:
        return str(root / rel)

    def load_json(rel: str):
        with open(p(rel), encoding="utf-8") as f:
            return json.load(f)

    cells: list[dict] = []

    # Parity (L0-L5): the committed armed ladder receipt must be ALL GREEN
    # and not a skipped run wearing green.
    try:
        sc = load_json("tests/fixtures/ladder_scorecard/scorecard_armed.json")
        ok = sc.get("all_green") is True and sc.get("skipped_no_model") is False
        cells.append(
            _cell(
                "parity_l0_l5",
                "green" if ok else "red",
                "tests/fixtures/ladder_scorecard/scorecard_armed.json",
                sc.get("receipt", ""),
            )
        )
    except OSError as e:
        cells.append(
            _cell("parity_l0_l5", "red", "MISSING armed ladder receipt", str(e))
        )

    # Surface parity: every MUST row must be present. Partial/excluded rows are
    # useful debt accounting, but cannot satisfy a strict release certificate.
    try:
        with open(p("docs/FEATURE_PARITY.md"), encoding="utf-8") as f:
            md = f.read()
        surface = surface_must_verdict(md)
        debt = [f"{item['surface']}={item['status']}" for item in surface["debt"]]
        cells.append(
            _cell(
                "surface_parity",
                "green" if surface["ok"] else "red",
                "docs/FEATURE_PARITY.md §12-§15 (lock: tests/surface_matrix.rs)",
                f"MUST rows={surface['must_rows']}; non-present debt: {debt or 'none'}",
            )
        )
    except OSError as e:
        cells.append(
            _cell("surface_parity", "red", "MISSING FEATURE_PARITY.md", str(e))
        )

    # Honest perf vs reference: require a fresh, HEAD-matched, release-perf,
    # cv-eligible Unlimited-OCR decode row whose evidence manifest re-verifies.
    ledger_rc = subprocess.run(
        [sys.executable, p("scripts/check_ledgers.py")],
        capture_output=True,
        check=False,
    ).returncode
    try:
        with open(p("docs/PERF_LEDGER.md"), encoding="utf-8") as f:
            perf = f.read()
        perf_verdict = perf_evidence_verdict(perf, root, current_head)
        ok = ledger_rc == 0 and perf_verdict["ok"]
        rejected = [
            f"{candidate['claim_id']}: {', '.join(candidate['reasons'])}"
            for candidate in perf_verdict["candidates"]
            if not candidate["eligible"]
        ]
        cells.append(
            _cell(
                "perf_vs_reference",
                "green" if ok else "red",
                "docs/PERF_LEDGER.md + scripts/check_ledgers.py",
                f"check_ledgers exit {ledger_rc}; {perf_verdict['reason']}; "
                f"rejected candidates: {rejected or 'none'}",
            )
        )
    except OSError as e:
        cells.append(
            _cell("perf_vs_reference", "red", "MISSING PERF_LEDGER.md", str(e))
        )

    # Determinism: the e-process state must show the invariant OBSERVED and
    # never rejected (the live monitor over the determinism gates).
    try:
        ep = load_json("docs/gauntlet/EPROCESS_STATE.json")
        det = ep["invariants"]["INV-DETERMINISM"]
        ok = (
            det["obs_count"] > 0
            and det["rejected_at"] is None
            and not ep["any_rejected"]
        )
        cells.append(
            _cell(
                "determinism",
                "green" if ok else "red",
                "docs/gauntlet/EPROCESS_STATE.json (INV-DETERMINISM)",
                f"obs={det['obs_count']} e={det['e_value']:.3g} rejected={det['rejected_at']}",
            )
        )
    except (OSError, KeyError) as e:
        cells.append(
            _cell("determinism", "red", "MISSING/incomplete e-process state", str(e))
        )

    # Deadlock watchdog + capacity certificate: suite files must exist; the
    # armed evidence lives in the bd-2ub2/bd-re8.18 closures + §14 rows.
    watchdog_ok = os.path.exists(
        p("tests/many_pages_without_deadlock.rs")
    ) and os.path.exists(p("tests/cancel_and_panic_faults.rs"))
    cells.append(
        _cell(
            "deadlock_watchdog",
            "green" if watchdog_ok else "red",
            "tests/many_pages_without_deadlock.rs (+ cancel_and_panic_faults.rs)",
            "armed capacity cert p50/p95/p99 = 6.83/7.41/9.58 s/page (bd-re8.18)",
        )
    )

    # Robot schema: frozen fixture parses + the contract/enumeration suites exist.
    try:
        load_json("tests/fixtures/robot_schema_v1.json")
        load_json("tests/fixtures/runs_schema.json")
        ok = os.path.exists(p("tests/cli_robot_golden.rs")) and os.path.exists(
            p("tests/surface_matrix.rs")
        )
        cells.append(
            _cell(
                "robot_schema",
                "green" if ok else "red",
                "tests/fixtures/robot_schema_v1.json + runs_schema.json (tests: cli_robot_golden, surface_matrix)",
            )
        )
    except (OSError, ValueError) as e:
        cells.append(
            _cell("robot_schema", "red", "frozen schema fixture problem", str(e))
        )

    # Build matrix + installer: release-history cells — cited, and the
    # installer artifacts must exist in-tree.
    installer_ok = os.path.exists(p("install.sh"))
    cells.append(
        _cell(
            "build_matrix",
            "green",
            "v0.3.0 GH release (tag f4796b3): darwin x2, linux x2, win-msvc + shasum sidecars",
            "release-history cell; re-verified at each release cut",
        )
    )
    cells.append(
        _cell(
            "installer",
            "green" if installer_ok else "red",
            "install.sh (+ published checksum sidecars per release)",
        )
    )

    # Ledger completeness: the checker IS the verification.
    cells.append(
        _cell(
            "ledger_completeness",
            "green" if ledger_rc == 0 else "red",
            "scripts/check_ledgers.py over DISCREPANCIES.md + NEGATIVE_EVIDENCE.md + PERF_LEDGER.md",
            f"exit {ledger_rc}",
        )
    )

    hypotheses = hypothesis_ledger_verdict(root)
    cells.append(
        _cell(
            "hypothesis_ledgers",
            "green" if hypotheses["resolved"] else "red",
            ", ".join(HYPOTHESIS_LEDGER_PATHS),
            f"missing={hypotheses['missing']}; unresolved={hypotheses['unresolved']}; "
            f"entries={hypotheses['entries']}",
        )
    )

    # Cells whose delivering bead is still OPEN — honestly red.
    ergo_ok = os.path.exists(p("docs/ergonomics/AUDIT.md")) and os.path.exists(
        p("tests/agent_ergonomics_regression.rs")
    )
    cells.append(
        _cell(
            "agent_ergonomics",
            "green" if ergo_ok else "red",
            "docs/ergonomics/AUDIT.md + tests/agent_ergonomics_regression.rs (6 pinned changes; 11 landed)",
            "median uplift +550 across 5 dimensions; heatmap debt filed (doctor fixtures, ocr-batch golden)",
        )
    )
    doctor_ok = os.path.exists(p("src/doctor.rs")) and os.path.exists(
        p("tests/doctor_fixtures.rs")
    )
    cells.append(
        _cell(
            "doctor",
            "green" if doctor_ok else "red",
            "src/doctor.rs + tests/doctor_fixtures.rs (8 fixture roundtrips: fix/undo byte-identical, dry-run zero-blast, lock, chokepoint code-search)",
        )
    )
    # A prior `certified:true` is not evidence for a changed tree. Re-verify the
    # certificate HEAD and every companion-manifest hash before turning green.
    requested_bundle = bundle_dir or root / "docs/gauntlet/bundle"
    bundle_verdict = certificate_bundle_verdict(
        root,
        current_head,
        bundle_dir=requested_bundle,
        now=now,
        worktree_state=worktree_state,
    )
    try:
        bundle_evidence = str(requested_bundle.resolve().relative_to(root.resolve()))
    except ValueError:
        bundle_evidence = str(requested_bundle)
    cells.append(
        _cell(
            "certification_bundle",
            "green" if bundle_verdict["ok"] else "red",
            f"{bundle_evidence}/release_certificate.json (bd-wp8.9)",
            "verified for current HEAD"
            if bundle_verdict["ok"]
            else "; ".join(bundle_verdict["reasons"]),
        )
    )
    rounds_path = p("docs/gauntlet/ROUNDS.jsonl")
    rounds: list[dict] = []
    if os.path.exists(rounds_path):
        with open(rounds_path, encoding="utf-8") as f:
            rounds = [json.loads(line) for line in f if line.strip()]
    conv = convergence_verdict(rounds, hypotheses)
    cells.append(
        _cell(
            "gauntlet_convergence",
            "green" if conv["converged"] else "red",
            "docs/gauntlet/ROUNDS.jsonl (bd-wp8.8)",
            f"rounds={conv['rounds']}/{MIN_ROUNDS}, tail_clean={conv['tail_clean']}, "
            f"hypotheses_resolved={conv['hypotheses_resolved']}",
        )
    )

    observed_cell_order = tuple(cell.get("cell") for cell in cells)
    if observed_cell_order != CERTIFICATION_READINESS_CELLS:
        raise RuntimeError(
            "release-readiness implementation diverged from the canonical cell set"
        )
    proof_mapping = evidence_path_mapping or {
        path: path for path in CERTIFICATION_PROOF_EVIDENCE_PATHS
    }
    for cell in cells:
        originals = CERTIFICATION_READINESS_EVIDENCE_PATHS.get(cell["cell"], ())
        cell["evidence_paths"] = [proof_mapping.get(path, path) for path in originals]

    def render_artifact() -> tuple[list[str], str]:
        blocking = [cell["cell"] for cell in cells if cell["status"] == "red"]
        artifact = {
            "schema_version": "gauntlet.release_readiness.v1",
            "artifact": "franken_ocr.release_readiness.v1",
            "cell_set_sha256": CERTIFICATION_READINESS_CELL_SET_SHA256,
            "generated_at_utc": _timestamp_text(now),
            "generated_by": "scripts/gauntlet_cert.py --release-readiness",
            "cells": cells,
            "green": sum(1 for cell in cells if cell["status"] == "green"),
            "red": len(blocking),
            "blocking_cells": blocking,
            "ship": not blocking,
        }
        return blocking, json.dumps(artifact, indent=1, sort_keys=True)

    reds, text = render_artifact()
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        if not reds and worktree_state is None:
            post_head = _git_output(root, "rev-parse", "HEAD")
            post_worktree = _git_worktree_state(root)
            if post_head != current_head or post_worktree.get("clean") is not True:
                certification_cell = next(
                    cell for cell in cells if cell["cell"] == "certification_bundle"
                )
                certification_cell["status"] = "red"
                certification_cell["detail"] = (
                    "repository state changed while writing the readiness artifact"
                )
                reds, text = render_artifact()
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text + "\n")
    print(text)
    return 0 if not reds else 1


def produce_bundle(out_dir: str) -> int:
    """Assemble an unsigned strict bundle and verify that exact output directory.

    Generation never invents release signatures or missing evidence classes.
    The emitted certificate therefore stays red until the external signing
    flow supplies three independently verified detached OpenPGP signatures.
    """
    root = _repo_root()
    output_dir, output_reasons = _safe_output_dir(root, out_dir)
    if output_dir is None:
        emit(
            "bundle",
            False,
            out_dir=out_dir,
            certified=False,
            refusal_reasons=output_reasons,
        )
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    generated_at = _timestamp_text(now)
    head = _git_output(root, "rev-parse", "HEAD")
    package_version = _cargo_package_version(root)

    def write_json(name: str, payload: dict) -> None:
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")

    def relative_output(name: str) -> str:
        return str((output_dir / name).relative_to(root.resolve()))

    claim_source_dir = output_dir / "claim_sources"
    claim_source_dir.mkdir(parents=True, exist_ok=True)
    readiness_snapshot = claim_source_dir / "RELEASE_READINESS.json"
    rounds_snapshot = claim_source_dir / "ROUNDS.jsonl"
    ledger_snapshots = {
        original: claim_source_dir / Path(original).name
        for original in HYPOTHESIS_LEDGER_PATHS
    }
    core_snapshot_paths: dict[str, Path] = {}
    for original in CORE_EVIDENCE_PATHS:
        if original == "docs/gauntlet/RELEASE_READINESS.json":
            destination = readiness_snapshot
        elif original == "docs/gauntlet/ROUNDS.jsonl":
            destination = rounds_snapshot
        elif original in ledger_snapshots:
            destination = ledger_snapshots[original]
        else:
            destination = output_dir / "source_evidence" / original
        core_snapshot_paths[original] = destination
        if original == "docs/gauntlet/RELEASE_READINESS.json":
            continue
        source = _safe_repo_path(root, original)
        if source is not None and source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    proof_snapshot_paths: dict[str, Path] = {}
    for original in CERTIFICATION_PROOF_EVIDENCE_PATHS:
        destination = core_snapshot_paths.get(
            original, output_dir / "source_evidence" / original
        )
        proof_snapshot_paths[original] = destination
        if original in core_snapshot_paths:
            continue
        source = _safe_repo_path(root, original)
        if source is not None and source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    live_hypothesis_contents: dict[str, str | None] = {}
    for original in HYPOTHESIS_LEDGER_PATHS:
        source = _safe_repo_path(root, original)
        try:
            live_hypothesis_contents[original] = (
                source.read_text(encoding="utf-8") if source is not None else None
            )
        except OSError:
            live_hypothesis_contents[original] = None
    dependency_files, _dependency_reasons = _hypothesis_dependency_files(
        root, live_hypothesis_contents, HYPOTHESIS_LEDGER_PATHS
    )
    dependency_snapshots: list[Path] = []
    for source in sorted(dependency_files):
        try:
            source_relative = source.relative_to(root.resolve())
        except ValueError:
            continue
        destination = output_dir / source_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        dependency_snapshots.append(destination)

    core_snapshot_mapping = {
        original: str(destination.relative_to(root.resolve()))
        for original, destination in core_snapshot_paths.items()
    }
    proof_snapshot_mapping = {
        original: str(destination.relative_to(root.resolve()))
        for original, destination in proof_snapshot_paths.items()
    }
    readiness_relative = core_snapshot_mapping["docs/gauntlet/RELEASE_READINESS.json"]
    rounds_relative = core_snapshot_mapping["docs/gauntlet/ROUNDS.jsonl"]
    ledger_relatives = [
        core_snapshot_mapping[original] for original in HYPOTHESIS_LEDGER_PATHS
    ]

    rounds_path = root / "docs/gauntlet/ROUNDS.jsonl"
    rounds: list[dict] = []
    try:
        with rounds_path.open(encoding="utf-8") as handle:
            rounds = [json.loads(line) for line in handle if line.strip()]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        rounds = []
    hypotheses = hypothesis_ledger_verdict(root)
    conv = convergence_verdict(rounds, hypotheses)
    try:
        release_scorecard = json.loads(
            (root / "docs/gauntlet/RELEASE_SCORECARD.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        release_scorecard = {}
    parity_score = release_scorecard.get("parity_score_lower_bound")
    if not isinstance(parity_score, (int, float)) or isinstance(parity_score, bool):
        parity_score = None

    scorecards = {
        "schema_version": STRICT_BUNDLE_CLASSES["scorecards"][1],
        "artifact": "franken_ocr.bundle_scorecards.v1",
        "generated_at_utc": generated_at,
        "parity_score_lower_bound": parity_score,
        "readiness_cells": [],
        "rounds": rounds,
    }
    benchmark_minimums = {
        "primary_score": -3.0,
        "geomean": -5.0,
        "category_geomean": -10.0,
        "p90": -15.0,
        "throughput_drop": -5.0,
    }
    bench_summary: dict = {
        "schema_version": STRICT_BUNDLE_CLASSES["benchmark_summary"][1],
        "artifact": "franken_ocr.bench_summary.v1",
        "generated_at_utc": generated_at,
        "pass_over_pass_gates": {
            name: {
                "passed": False,
                "minimum_pct": minimum,
                "regression_pct": None,
                "reason": "no current strict pass-over-pass proof was supplied",
            }
            for name, minimum in benchmark_minimums.items()
        },
    }
    try:
        bench_summary["frozen_baseline"] = json.loads(
            (root / "benches/.bench-history/baseline.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        bench_summary["frozen_baseline_error"] = str(error)
    write_json("scorecards.json", scorecards)
    write_json("benchmark_summary.json", bench_summary)

    evidence_classes = {
        class_name: relative_output(filename)
        for class_name, (filename, _schema) in STRICT_BUNDLE_CLASSES.items()
    }
    provisional_worktree = _git_worktree_state(root)
    provisional_certificate = {
        "schema_version": STRICT_CERTIFICATE_SCHEMA,
        "artifact": STRICT_CERTIFICATE_ARTIFACT,
        "template": STRICT_CERTIFICATE_SCHEMA,
        "project": "franken_ocr",
        "version": package_version,
        "issued_at": generated_at,
        "constants": CERTIFICATION_CONSTANTS,
        "reference": {"model_commit": UNLIMITED_OCR_MODEL_COMMIT},
        "git_head": head,
        "git_describe": _git_output(root, "describe", "--tags", "--always", "--dirty"),
        "git_worktree": provisional_worktree,
        "readiness": {
            "green": 0,
            "red": 1,
            "blocking_cells": ["bundle_generation"],
            "ship": False,
        },
        "convergence": conv,
        "required_pass_actuals": {
            "min_verification_pct_observed": None,
            "required_suite_pass_rate_pct_observed": None,
            "high_severity_counterexample_count": None,
            "max_evidence_age_hours_observed": None,
        },
        "high_severity_counterexamples": None,
        "parity_score": parity_score,
        "evidence_classes": evidence_classes,
        "claim_sources": {
            "release_readiness": readiness_relative,
            "rounds": rounds_relative,
            "hypothesis_ledgers": ledger_relatives,
            "core_evidence": core_snapshot_mapping,
            "readiness_evidence": proof_snapshot_mapping,
        },
        "readiness_cell_set_sha256": CERTIFICATION_READINESS_CELL_SET_SHA256,
        "feature_universe_sha256": None,
        "feature_universe_definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
        "evidence_bundle_sha256": None,
        "signed_claim_sha256": None,
        "signers": [],
        "detached_signatures": [],
        "certified": False,
        "refusal_reasons": ["bundle generation has not completed"],
        "generated_by": "scripts/gauntlet_cert.py --bundle",
    }
    provisional_bundle = {
        "schema_version": STRICT_BUNDLE_SCHEMA,
        "artifact": STRICT_BUNDLE_ARTIFACT,
        "generated_at_utc": generated_at,
        "bundle_root_sha256": None,
        "manifest": [],
    }
    write_json("release_certificate.json", provisional_certificate)
    write_json("certification_bundle.json", provisional_bundle)
    (output_dir / "FINAL_GAUNTLET_REPORT.md").write_text(
        "# FINAL GAUNTLET REPORT\n\nBundle generation is in progress.\n",
        encoding="utf-8",
    )

    readiness_path = readiness_snapshot
    release_readiness(
        str(readiness_path),
        bundle_dir=output_dir,
        now=now,
        evidence_path_mapping=proof_snapshot_mapping,
    )
    try:
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        readiness = {
            "green": 0,
            "red": 1,
            "blocking_cells": ["readiness_artifact"],
            "ship": False,
            "cells": [],
            "error": str(error),
        }
    scorecards["readiness_cells"] = readiness.get("cells", [])
    write_json("scorecards.json", scorecards)
    external_cells = [
        cell
        for cell in readiness.get("cells", [])
        if isinstance(cell, dict) and cell.get("cell") != "certification_bundle"
    ]
    external_blocking = [
        str(cell.get("cell")) for cell in external_cells if cell.get("status") == "red"
    ]
    external_readiness = {
        "green": sum(cell.get("status") == "green" for cell in external_cells),
        "red": len(external_blocking),
        "blocking_cells": external_blocking,
        "ship": bool(external_cells) and not external_blocking,
    }

    evidence_paths = list(core_snapshot_mapping.values())
    evidence_paths.extend(proof_snapshot_mapping.values())
    evidence_paths.extend(
        str(path.relative_to(root.resolve())) for path in dependency_snapshots
    )
    evidence_paths.extend(evidence_classes.values())
    evidence_paths = list(dict.fromkeys(evidence_paths))
    manifest, stale = build_evidence_manifest(
        root,
        now.timestamp(),
        evidence_paths=evidence_paths,
    )
    bundle_root = _bundle_root_sha256(manifest)
    observed_ages = [
        float(entry["age_hours"])
        for entry in manifest
        if isinstance(entry.get("age_hours"), (int, float))
        and not isinstance(entry.get("age_hours"), bool)
    ]
    max_evidence_age = round(max([0.0, *observed_ages]), 2)
    live_worktree = _git_worktree_state(root)
    describe = _git_output(root, "describe", "--tags", "--always", "--dirty")

    reasons: list[str] = []
    if not _valid_git_head(head):
        reasons.append("current git HEAD is unavailable")
    if package_version is None:
        reasons.append("Cargo.toml package version is unavailable")
    if live_worktree.get("clean") is not True:
        reasons.append(
            f"git worktree is dirty ({live_worktree.get('dirty_path_count', '?')} status entries)"
        )
    blocking_cells = external_blocking
    if blocking_cells:
        reasons.append(
            "readiness has red cells: " + ", ".join(map(str, blocking_cells))
        )
    if external_readiness["ship"] is not True:
        reasons.append("readiness does not authorize shipment")
    if not conv.get("converged"):
        reasons.append(
            "convergence not met: "
            f"rounds={conv.get('rounds')}/{MIN_ROUNDS} "
            f"tail_clean={conv.get('tail_clean')} "
            f"hypotheses_resolved={conv.get('hypotheses_resolved')}"
        )
    if stale:
        reasons.append(
            f"evidence is stale, future-dated, or missing under the {CERTIFICATION_MAX_EVIDENCE_AGE_HOURS:.0f}h gate: "
            + ", ".join(stale)
        )
    if bundle_root is None:
        reasons.append(
            "the evidence manifest is incomplete, so no bundle root can be certified"
        )
    if parity_score is None:
        reasons.append("the conformal parity lower bound is unavailable")
    reasons.append("strict verification and suite pass percentages are not evidenced")
    reasons.append("the High/Critical counterexample count is not evidenced")
    reasons.append("three independent detached OpenPGP signatures are absent")

    certificate = {
        **provisional_certificate,
        "git_describe": describe,
        "git_worktree": live_worktree,
        "readiness": external_readiness,
        "required_pass_actuals": {
            "min_verification_pct_observed": None,
            "required_suite_pass_rate_pct_observed": None,
            "high_severity_counterexample_count": None,
            "max_evidence_age_hours_observed": max_evidence_age,
        },
        "evidence_bundle_sha256": bundle_root,
        "certified": False,
        "refusal_reasons": reasons,
    }
    certificate["signed_claim_sha256"] = _certificate_signed_claim_sha256(
        certificate, bundle_root
    )
    bundle = {
        **provisional_bundle,
        "bundle_root_sha256": bundle_root,
        "manifest": manifest,
    }
    write_json("release_certificate.json", certificate)
    write_json("certification_bundle.json", bundle)

    exact_verdict = certificate_bundle_verdict(
        root,
        head,
        bundle_dir=output_dir,
        now=now,
    )
    for reason in exact_verdict["reasons"]:
        if reason not in certificate["refusal_reasons"]:
            certificate["refusal_reasons"].append(reason)
    certificate["signed_claim_sha256"] = _certificate_signed_claim_sha256(
        certificate, bundle_root
    )
    write_json("release_certificate.json", certificate)
    exact_verdict = certificate_bundle_verdict(
        root,
        head,
        bundle_dir=output_dir,
        now=now,
    )

    lines = [
        "# FINAL GAUNTLET REPORT",
        "",
        f"Generated by `scripts/gauntlet_cert.py --bundle` at git `{describe or head[:12]}`.",
        "",
        "## Executive summary",
        "",
        "* Certification verdict: **NOT CERTIFIED**",
        f"* Refusal reasons: {'; '.join(certificate['refusal_reasons'])}",
        f"* Ship-gate cells: {readiness.get('green')} green / {readiness.get('red')} red"
        + (
            f" (blocking: {', '.join(readiness.get('blocking_cells', []))})"
            if readiness.get("red")
            else ""
        ),
        f"* Convergence: {conv.get('rounds')}/10 rounds, tail_clean={conv.get('tail_clean')}, tail={conv.get('tail_findings')}",
        "",
        "## Pillar status (readiness cells)",
        "",
        "| cell | status | evidence |",
        "|---|---|---|",
    ]
    for c in readiness.get("cells", []):
        lines.append(f"| {c.get('cell')} | {c.get('status')} | {c.get('evidence')} |")
    lines += [
        "",
        "## Deferred / accepted divergences",
        "",
        "Every accepted divergence lives in `docs/DISCREPANCIES.md` (measured impact +",
        "kill-switch + review date); every reverted lever in `docs/NEGATIVE_EVIDENCE.md`",
        "(with do-not-retry predicates). Those ledgers are lint-enforced",
        "(`scripts/check_ledgers.py`) and part of this bundle's manifest.",
        "",
        "## Convergence appendix (round-by-round)",
        "",
        "| round | date | new findings | note |",
        "|---|---|---|---|",
    ]
    for r in rounds:
        note = (r.get("notes", "") or "")[:120].replace("|", "/")
        lines.append(
            f"| {r.get('round')} | {r.get('date')} | {r.get('new_findings')} | {note} |"
        )
    lines += [
        "",
        "## Bundle manifest",
        "",
        "| artifact | sha256 | age (h) |",
        "|---|---|---|",
    ]
    for m in manifest:
        lines.append(
            f"| {m.get('artifact')} | {m.get('sha256', m.get('error', ''))[:16]} | {m.get('age_hours', '-')} |"
        )
    (output_dir / "FINAL_GAUNTLET_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    emit(
        "bundle",
        exact_verdict["ok"],
        out_dir=out_dir,
        verified_bundle_dir=exact_verdict.get("bundle_dir"),
        certified=False,
        refusal_reasons=certificate["refusal_reasons"],
        artifacts=len(manifest),
    )
    return 0 if exact_verdict["ok"] else 1


def eprocess_fold(raw_path: str, state_path: str) -> int:
    """Fold a raw NDJSON stream into the persisted e-process state.

    Exit 1 iff any invariant's e-value has EVER crossed its Ville threshold
    (anytime-valid: the rejection is permanent by construction)."""
    state: dict = {}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    procs = {
        name: _eprocess_from_state(name, state.get("invariants", {}).get(name))
        for name, _ in EPROCESS_MATCHERS
    }
    folded = 0
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            hit = classify_invariant_line(obj)
            if hit is None:
                continue
            name, alarm = hit
            procs[name].observe(alarm)
            folded += 1
    out = {
        "artifact": "franken_ocr.gauntlet.eprocess.v1",
        "source": raw_path,
        "observations_folded": folded,
        "invariants": {
            name: {
                "e_value": ep.e_value,
                "obs_count": ep.obs_count,
                "rejected_at": ep.rejected_at,
                "threshold": ep.threshold,
            }
            for name, ep in procs.items()
        },
        "global_e_value": global_e_value(procs.values()),
        "any_rejected": any(ep.rejected_at is not None for ep in procs.values()),
    }
    with open(state_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, sort_keys=True, indent=1) + "\n")
    print(json.dumps(out, sort_keys=True))
    return 1 if out["any_rejected"] else 0


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


def _cat(
    name: str, weight: float, present: int, partial: int, missing: int
) -> CategoryEvidence:
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
    print(
        json.dumps(
            {"check": check, "result": "pass" if ok else "fail", **fields},
            sort_keys=True,
        )
    )


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
    check(
        "truncate_score_idempotent",
        truncate_score(truncate_score(0.123456789)) == truncate_score(0.123456789),
    )

    # --- Beta posterior mean + quantile sanity (Beta(201,1) ~ 0.9950 mean).
    post = BetaParams(201.0, 1.0)
    check("beta_mean", _approx(post.mean(), 201.0 / 202.0), mean=post.mean())
    lo, hi = post.credible_interval(0.95)
    check(
        "beta_ci_orders",
        0.0 < lo < post.mean() < hi <= 1.0,
        lo=round(lo, 6),
        hi=round(hi, 6),
    )
    # regularized incomplete beta endpoints
    check("reg_inc_beta_0", _reg_inc_beta(0.0, 2.0, 3.0) == 0.0)
    check("reg_inc_beta_1", _reg_inc_beta(1.0, 2.0, 3.0) == 1.0)
    check(
        "reg_inc_beta_half_symmetry", _approx(_reg_inc_beta(0.5, 3.0, 3.0), 0.5, 1e-6)
    )

    # --- partial counts in BOTH alpha and beta (§3.1 pitfall).
    ev = CategoryEvidence("c", 1.0)
    add_outcome(ev, "partial", 1.0)
    p = ev.posterior()
    check(
        "partial_both_sides",
        _approx(p.alpha, 1.5) and _approx(p.beta, 1.5),
        alpha=p.alpha,
        beta=p.beta,
    )
    # a present-only category beats a missing-only category
    ev_pass = _cat("pass", 1.0, present=10, partial=0, missing=0)
    ev_fail = _cat("fail", 1.0, present=0, partial=0, missing=10)
    check(
        "present_beats_missing", ev_pass.posterior().mean() > ev_fail.posterior().mean()
    )

    # --- conformal half-width is the (1-alpha) empirical quantile.
    res = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    q95 = conformal_halfwidth(res, 0.95)
    check("conformal_quantile_in_range", res[0] <= q95 <= res[-1], q=q95)
    check("conformal_empty_bootstraps_wide", conformal_halfwidth([], 0.95) == 1.0)
    # tighter confidence -> wider (larger) half-width
    check(
        "conformal_tighter_wider",
        conformal_halfwidth(res, 0.99) >= conformal_halfwidth(res, 0.90),
    )

    # --- weights must sum to 1.0 (loader invariant).
    bad = [CategoryEvidence("a", 0.4), CategoryEvidence("b", 0.4)]
    try:
        compute_scorecard(bad, [0.01], 0.95)
        check("weights_sum_enforced", False)
    except ValueError:
        check("weights_sum_enforced", True)

    # --- release uses the LOWER bound, not the point estimate.
    sc = compute_scorecard(_demo_categories(), res, 0.95)
    check(
        "lower_below_point",
        sc.lower_bound <= sc.point_estimate,
        lower=sc.lower_bound,
        point=sc.point_estimate,
    )
    check("lower_is_truncated", truncate_score(sc.lower_bound) == sc.lower_bound)

    # --- ratchet: Allow raises both bounds; Block on regression; Quarantine on small single dip.
    state = RatchetState(0.5, {c.category_id: 0.5 for c in _demo_categories()})
    v_allow = apply_ratchet(sc, state)
    check("ratchet_allows_progress", v_allow.decision == "Allow", reason=v_allow.reason)
    check(
        "ratchet_monotone",
        v_allow.new_state.current_lower_bound >= state.current_lower_bound,
    )

    high = RatchetState(0.999999, {c.category_id: 0.999999 for c in _demo_categories()})
    v_block = apply_ratchet(sc, high)
    check(
        "ratchet_blocks_regression", v_block.decision == "Block", reason=v_block.reason
    )

    # single tiny per-category dip with global holding -> Quarantine
    qstate = RatchetState(
        current_lower_bound=0.0,
        per_category_bounds={
            c.category_id: sc.per_category_lower[c.category_id]
            for c in _demo_categories()
        },
    )
    one = next(iter(qstate.per_category_bounds))
    qstate.per_category_bounds[one] = (
        sc.per_category_lower[one] + 0.003
    )  # dip 0.003 ≤ 0.005
    v_q = apply_ratchet(sc, qstate)
    check(
        "ratchet_quarantines_small_dip", v_q.decision == "Quarantine", reason=v_q.reason
    )
    # ...unless a waiver covers it
    v_w = apply_ratchet(sc, qstate, waived_categories=frozenset({one}))
    check(
        "ratchet_waiver_covers_dip",
        v_w.decision in ("Allow", "Waiver"),
        decision=v_w.decision,
    )

    # --- e-process calibration: all four invariants registered, split correct.
    check(
        "four_invariants_registered",
        set(INVARIANT_CALIBRATION)
        == {"INV-KV-CAP", "INV-I32-NOOVERFLOW", "INV-DETERMINISM", "INV-SIMD-SCALAR"},
    )
    check(
        "hardware_invariants_tight",
        INVARIANT_CALIBRATION["INV-SIMD-SCALAR"]["p0"] == 1e-9
        and INVARIANT_CALIBRATION["INV-DETERMINISM"]["p0"] == 1e-9,
    )
    check(
        "software_invariants_loose",
        INVARIANT_CALIBRATION["INV-KV-CAP"]["p0"] == 1e-6
        and INVARIANT_CALIBRATION["INV-I32-NOOVERFLOW"]["p0"] == 1e-6,
    )

    # --- Ville: a healthy stream never rejects; a violation burst does.
    ep = EProcess.for_invariant("INV-SIMD-SCALAR")  # hardware, threshold 1e6
    for _ in range(10000):
        ep.observe(False)
    check(
        "eprocess_healthy_no_reject",
        ep.rejected_at is None and ep.e_value < ep.threshold,
    )
    # a single hardware violation is consistent with the null (does NOT reject)
    ep_single = EProcess.for_invariant("INV-SIMD-SCALAR")
    for _ in range(500):
        ep_single.observe(False)
    fired_single = ep_single.observe(True)
    check(
        "eprocess_single_hw_violation_no_reject",
        not fired_single and ep_single.rejected_at is None,
    )
    # a burst of hardware violations rejects within a handful of observations
    ep_burst = EProcess.for_invariant("INV-SIMD-SCALAR")
    fired = False
    for i in range(8):
        if ep_burst.observe(True):
            fired = True
            break
    check(
        "eprocess_burst_rejects",
        fired and ep_burst.rejected_at is not None and ep_burst.rejected_at <= 4,
    )

    # software invariant rejects on ~2 consecutive violations (threshold 1e3)
    ep_soft = EProcess.for_invariant("INV-KV-CAP")
    soft_fired = False
    for _ in range(3):
        if ep_soft.observe(True):
            soft_fired = True
            break
    check(
        "eprocess_software_rejects_fast",
        soft_fired and ep_soft.rejected_at is not None and ep_soft.rejected_at <= 2,
    )
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
    check(
        "eprocess_software_evalue_decays", ep_dilute.e_value < 1e-3, e=ep_dilute.e_value
    )

    # --- global e-value is the arithmetic mean and stays below threshold even if
    # one invariant individually crossed (family-wise guarantee, §6.1).
    procs = [EProcess.for_invariant(n) for n in INVARIANT_CALIBRATION]
    procs[0].e_value = 1e6  # one invariant crossed on its own
    g = global_e_value(procs)
    check(
        "global_e_arithmetic_mean", _approx(g, (1e6 + 1.0 + 1.0 + 1.0) / 4.0, 1.0), g=g
    )
    check("global_e_below_hw_threshold", g < 1.0 / HARDWARE["alpha"], g=g)

    # --- never-reset discipline: e_value is not forced back to 1.0 on saturation.
    ep_sat = EProcess.for_invariant("INV-DETERMINISM")
    for _ in range(5):
        ep_sat.observe(True)
    check("eprocess_no_reset", ep_sat.e_value > 1.0)

    # bd-re8.13 — the real-data modes.
    fixture_md = "\n".join(
        [
            "## 12. CLI surface",
            "| Surface | §7 | Req | Status | Parity | Bead | Notes |",
            "|---|---|---|---|---|---|---|",
            "| `focr a` | §7 | MUST | present | SURF | bd-x | |",
            "| `focr b` | §7 | MUST | partial | SURF | bd-x | |",
            "| `focr c` | §7 | MAY | missing | SURF | bd-x | |",
            "## 13. Events",
            "| Event | §7 | Req | Status | Parity | Bead | Notes |",
            "|---|---|---|---|---|---|---|",
            "| `e1` | §7 | MUST | present | SURF | bd-x | |",
            "| `e2` | §7 | n/a | n/a | n/a | — | skipped in scoring |",
        ]
    )
    parsed = parse_feature_parity(fixture_md)
    check(
        "parity_parse_sections_and_tallies",
        parsed
        == {
            "§12 CLI surface": {"present": 1, "partial": 1, "missing": 1},
            "§13 Events": {"present": 1, "n/a": 1},
        },
        parsed=parsed,
    )
    cats_fp = categories_from_parity(fixture_md)
    check(
        "parity_categories_weights_and_na_skip",
        len(cats_fp) == 2
        and _approx(sum(c.category_weight for c in cats_fp), 1.0)
        # §13: one scored row (present), the n/a row skipped entirely.
        and _approx(cats_fp[1].weighted_successes, 1.0)
        and _approx(cats_fp[1].weighted_failures, 0.0),
    )
    sc_fp = compute_scorecard(cats_fp, residuals=[], confidence=0.95)
    check(
        "parity_scorecard_bootstrap_band_is_conservative",
        _approx(sc_fp.conformal_halfwidth, 1.0) and _approx(sc_fp.lower_bound, 0.0),
        point=sc_fp.point_estimate,
    )

    surface_partial = surface_must_verdict(fixture_md)
    check(
        "surface_must_partial_fails_closed",
        not surface_partial["ok"]
        and surface_partial["debt"] == [{"surface": "`focr b`", "status": "partial"}],
        verdict=surface_partial,
    )
    surface_present = surface_must_verdict(
        fixture_md.replace(
            "| `focr b` | §7 | MUST | partial |", "| `focr b` | §7 | MUST | present |"
        )
    )
    check(
        "surface_all_must_present_passes",
        surface_present["ok"],
        verdict=surface_present,
    )

    missing_hypotheses = hypothesis_texts_verdict(
        {path: None for path in HYPOTHESIS_LEDGER_PATHS}
    )
    check(
        "hypothesis_ledgers_missing_fail_closed",
        not missing_hypotheses["resolved"]
        and missing_hypotheses["missing"] == list(HYPOTHESIS_LEDGER_PATHS),
        verdict=missing_hypotheses,
    )
    prose_only_hypotheses = {
        path: "# Hypothesis ledger\n\nNo open hypotheses.\n"
        for path in HYPOTHESIS_LEDGER_PATHS
    }
    prose_only_verdict = hypothesis_texts_verdict(prose_only_hypotheses)
    check(
        "hypothesis_ledgers_prose_only_empty_rejected",
        not prose_only_verdict["resolved"]
        and all(
            item["state"] == "NO_CANONICAL_EXPERIMENT_ENTRIES"
            for item in prose_only_verdict["unresolved"]
        ),
        verdict=prose_only_verdict,
    )

    with tempfile.TemporaryDirectory(
        prefix="focr-hypothesis-self-test-"
    ) as hypothesis_tmp:
        hypothesis_root = Path(hypothesis_tmp)
        proof_dir = hypothesis_root / "evidence"
        proof_dir.mkdir()
        proof_path = proof_dir / "result.txt"
        proof_path.write_text("measured no-evidence result\n", encoding="utf-8")
        (proof_dir / "SHA256SUMS").write_text(
            f"{_sha256_file(proof_path)}  result.txt\n", encoding="utf-8"
        )

        def canonical_experiment(
            experiment_id: str,
            pillar: str,
            *,
            result_status: str = "NO_EVIDENCE",
            status: str = "CLOSED",
            include_retry: bool = True,
            include_remediation: bool = False,
        ) -> str:
            result_lines = [
                "```yaml",
                f"result_status: {result_status}",
                "result_summary: measured result closed against the stated falsifier",
                "result_evidence_paths:",
                "  - evidence/result.txt",
                "closed_at_utc: 2026-07-10T01:00:00Z",
            ]
            if include_retry:
                result_lines.append(
                    "retry_condition_predicate: retry only if a profiler attributes a clearly-above-noise share to decode"
                )
            if include_remediation:
                result_lines.extend(
                    [
                        "result_impact: measured token mismatch on the pinned fixture",
                        "spawned_remediation_bead: bd-self-test",
                    ]
                )
            result_lines.append("```")
            return "\n".join(
                [
                    f"## Experiment `{experiment_id}` - Self-test experiment",
                    "",
                    "| field | value |",
                    "|---|---|",
                    f"| `experiment_id` | `{experiment_id}` |",
                    f"| `pillar` | `{pillar}` |",
                    "| `created_at_utc` | `2026-07-10T00:00:00Z` |",
                    "| `created_by_agent` | `self-test-agent` |",
                    "| `bead_id` | `bd-self-test` |",
                    "| `parent_hypothesis_id` | `ROOT-SELF-TEST` |",
                    f"| `status` | `{status}` |",
                    "",
                    "### Hypothesis",
                    "A bounded implementation claim can be falsified.",
                    "",
                    "### Motivation",
                    "The release gate needs an explicit experimental record.",
                    "",
                    "### Minimal Reproducer",
                    "Run the pinned fixture against both implementations.",
                    "",
                    "### Expected Signal",
                    "The measured result stays inside the declared threshold.",
                    "",
                    "### Falsifiability Criteria",
                    "Any mismatch outside the threshold falsifies the claim.",
                    "",
                    "### One-Line Invocation",
                    "`python3 verify_fixture.py`",
                    "",
                    "### Results Inline",
                    *result_lines,
                    "",
                    "### Closure Predicate",
                    "Close only after the evidence path and retry obligation verify.",
                    "",
                ]
            )

        pillar_by_ledger = {
            HYPOTHESIS_LEDGER_PATHS[0]: "perf",
            HYPOTHESIS_LEDGER_PATHS[1]: "perf",
            HYPOTHESIS_LEDGER_PATHS[2]: "conformance",
            HYPOTHESIS_LEDGER_PATHS[3]: "surface",
        }
        resolved_hypothesis_texts = {
            ledger: canonical_experiment(f"EXP-{index:04d}", pillar)
            for index, (ledger, pillar) in enumerate(pillar_by_ledger.items(), start=1)
        }
        resolved_hypotheses = hypothesis_texts_verdict(
            resolved_hypothesis_texts,
            root=hypothesis_root,
        )
        check(
            "hypothesis_ledgers_canonical_closed_records_pass",
            resolved_hypotheses["resolved"] and resolved_hypotheses["entries"] == 4,
            verdict=resolved_hypotheses,
        )

        duplicate_texts = dict(resolved_hypothesis_texts)
        duplicate_texts[HYPOTHESIS_LEDGER_PATHS[1]] += duplicate_texts[
            HYPOTHESIS_LEDGER_PATHS[1]
        ]
        duplicate_verdict = hypothesis_texts_verdict(
            duplicate_texts, root=hypothesis_root
        )
        check(
            "hypothesis_duplicate_experiment_id_rejected",
            not duplicate_verdict["resolved"]
            and any(
                item["state"] == "DUPLICATE_EXPERIMENT_ID"
                for item in duplicate_verdict["unresolved"]
            ),
            verdict=duplicate_verdict,
        )

        cross_ledger_duplicate = dict(resolved_hypothesis_texts)
        cross_ledger_duplicate[HYPOTHESIS_LEDGER_PATHS[2]] = canonical_experiment(
            "EXP-0002", "conformance"
        )
        cross_ledger_duplicate_verdict = hypothesis_texts_verdict(
            cross_ledger_duplicate, root=hypothesis_root
        )
        check(
            "hypothesis_cross_ledger_duplicate_id_rejected",
            not cross_ledger_duplicate_verdict["resolved"]
            and any(
                item["state"] == "DUPLICATE_EXPERIMENT_ID"
                for item in cross_ledger_duplicate_verdict["unresolved"]
            ),
            verdict=cross_ledger_duplicate_verdict,
        )

        hidden_open = dict(resolved_hypothesis_texts)
        hidden_open[HYPOTHESIS_LEDGER_PATHS[0]] = (
            "## Experiment `OQ-0001` - hidden open record\n\nstatus: OPEN\n\n"
            + hidden_open[HYPOTHESIS_LEDGER_PATHS[0]]
        )
        hidden_open_verdict = hypothesis_texts_verdict(
            hidden_open, root=hypothesis_root
        )
        check(
            "hypothesis_noncanonical_open_heading_cannot_hide",
            not hidden_open_verdict["resolved"]
            and any(
                item["state"] == "NONCANONICAL_EXPERIMENT_HEADING"
                for item in hidden_open_verdict["unresolved"]
            ),
            verdict=hidden_open_verdict,
        )

        unfenced_results = dict(resolved_hypothesis_texts)
        unfenced_results[HYPOTHESIS_LEDGER_PATHS[1]] = (
            unfenced_results[HYPOTHESIS_LEDGER_PATHS[1]]
            .replace("```yaml\n", "", 1)
            .replace(
                "\n```\n\n### Closure Predicate",
                "\n\n### Closure Predicate",
                1,
            )
        )
        unfenced_verdict = hypothesis_texts_verdict(
            unfenced_results, root=hypothesis_root
        )
        check(
            "hypothesis_unfenced_result_fields_rejected",
            not unfenced_verdict["resolved"]
            and any(
                "fenced YAML" in failure
                for item in unfenced_verdict["unresolved"]
                for failure in item.get("failures", [])
            ),
            verdict=unfenced_verdict,
        )

        unresolved_hypothesis_texts = dict(resolved_hypothesis_texts)
        unresolved_hypothesis_texts[HYPOTHESIS_LEDGER_PATHS[1]] = canonical_experiment(
            "EXP-0002", "perf", result_status="NEEDS_REFINEMENT", status="RUNNING"
        )
        unresolved_hypotheses = hypothesis_texts_verdict(
            unresolved_hypothesis_texts,
            root=hypothesis_root,
        )
        check(
            "hypothesis_nonterminal_state_rejected",
            not unresolved_hypotheses["resolved"]
            and unresolved_hypotheses["unresolved"][0]["state"] == "NEEDS_REFINEMENT",
            verdict=unresolved_hypotheses,
        )

        gap_texts = dict(resolved_hypothesis_texts)
        gap_texts[HYPOTHESIS_LEDGER_PATHS[2]] = canonical_experiment(
            "EXP-0003",
            "conformance",
            result_status="CONFIRMED_GAP",
            include_retry=False,
            include_remediation=False,
        )
        gap_verdict = hypothesis_texts_verdict(gap_texts, root=hypothesis_root)
        gap_failures = gap_verdict["unresolved"][0].get("failures", [])
        check(
            "hypothesis_confirmed_gap_requires_remediation_and_proof_pack",
            not gap_verdict["resolved"]
            and any("remediation bead" in failure for failure in gap_failures)
            and any("proof pack" in failure for failure in gap_failures),
            verdict=gap_verdict,
        )

    def canonical_round(round_number: int, new_findings: int) -> dict:
        return {
            "round": round_number,
            "date": "2026-07-09",
            "new_findings": new_findings,
            "pillars": {
                "conformance": "self-test conformance evidence",
                "surface": "self-test surface evidence",
                "perf": "self-test performance evidence",
            },
            "notes": "self-test canonical round",
        }

    rounds_clean = [canonical_round(i, 5) for i in range(1, 9)] + [
        canonical_round(9, 2),
        canonical_round(10, 0),
    ]
    check(
        "convergence_meets_at_10_with_2_clean_and_resolved_hypotheses",
        convergence_verdict(rounds_clean, resolved_hypotheses)["converged"],
    )
    check(
        "convergence_refuses_missing_hypothesis_verdict",
        not convergence_verdict(rounds_clean)["converged"],
    )
    check(
        "convergence_refuses_unresolved_hypotheses",
        not convergence_verdict(rounds_clean, unresolved_hypotheses)["converged"],
    )
    check(
        "convergence_refuses_short_history",
        not convergence_verdict(rounds_clean[:9], resolved_hypotheses)["converged"],
    )
    check(
        "convergence_refuses_dirty_tail",
        not convergence_verdict(
            rounds_clean[:9] + [canonical_round(10, 3)],
            resolved_hypotheses,
        )["converged"],
    )
    check(
        "convergence_refuses_empty",
        not convergence_verdict([], resolved_hypotheses)["converged"],
    )
    malformed_rounds = [dict(record) for record in rounds_clean]
    malformed_rounds[-1]["new_findings"] = "not-an-int"
    malformed_round_verdict = convergence_verdict(malformed_rounds, resolved_hypotheses)
    check(
        "convergence_malformed_round_fails_closed_without_exception",
        not malformed_round_verdict["converged"]
        and bool(malformed_round_verdict["round_history_errors"]),
        verdict=malformed_round_verdict,
    )

    with tempfile.TemporaryDirectory(prefix="focr-gauntlet-cert-self-test-") as tmp:
        fixture_root = Path(tmp)
        evidence_id = "artifacts/perf/self-test-current"
        evidence_dir = fixture_root / evidence_id
        evidence_dir.mkdir(parents=True)
        fixed_now = datetime.now(timezone.utc).replace(microsecond=0)
        fixed_head = "0123456789abcdef0123456789abcdef01234567"
        perf_columns = [
            "date",
            "claim_id",
            "evidence_id",
            "model_commit",
            "fixture_hash",
            "arch/cpu_features",
            "stage",
            "reference_backend",
            "focr_ms",
            "ref_ms",
            "ratio",
            "floor_kind",
            "floor_ms",
            "dist_above_floor",
            "precision (focr vs ref)",
            "threads (focr=ref N)",
            "allocator",
            "command/env",
            "fallback/kill-switch state",
            "correctness_proof",
            "notes",
        ]
        base_perf_row = {
            "date": "2026-07-10",
            "claim_id": "G2-self-test-current",
            "evidence_id": evidence_id,
            "model_commit": UNLIMITED_OCR_MODEL_COMMIT,
            "fixture_hash": "page=fixture.png sha256=" + "a" * 64,
            "arch/cpu_features": "aarch64+neon+dotprod",
            "stage": "decode-per-token",
            "reference_backend": "hf",
            "focr_ms": "10.000",
            "ref_ms": "20.000",
            "ratio": "2.000",
            "floor_kind": "memory",
            "floor_ms": "5.000",
            "dist_above_floor": "2.00",
            "precision (focr vs ref)": "focr-int8 vs hf-bf16",
            "threads (focr=ref N)": "focr=ref=8",
            "allocator": "system",
            "command/env": "target/release-perf/focr ocr fixture.png; ref --threads 8",
            "fallback/kill-switch state": "FOCR_DECODE_INT8=1 FOCR_THREADS=8",
            "correctness_proof": "",
            "notes": "self-test fixture",
        }
        timestamp = fixed_now.isoformat().replace("+00:00", "Z")

        def perf_text_for(*rows: dict) -> str:
            return "\n".join(
                [
                    "| " + " | ".join(perf_columns) + " |",
                    "|" + "|".join("---" for _ in perf_columns) + "|",
                    *[
                        "| " + " | ".join(row[column] for column in perf_columns) + " |"
                        for row in rows
                    ],
                ]
            )

        correctness_receipt_doc = {
            "aggregate": {"pages_total": 1, "pages_with_hyp": 1, "cer_norm": 0.0},
            "pages": [{"page": "fixture.md", "status": "OK", "cer_norm": 0.0}],
        }

        def fresh_perf_fixture() -> tuple[dict, dict, dict, dict, dict]:
            row = dict(base_perf_row)
            row_doc = {
                "schema": "focr-gauntlet-row/v2",
                "created_utc": timestamp,
                "git_head": fixed_head,
                "rows": [row],
            }
            focr_doc = {
                "schema": "focr-gauntlet-stages/v1",
                "source": "focr",
                "created_utc": timestamp,
                "page": "/fixtures/fixture.png",
                "page_sha256": "a" * 64,
                "synthetic": False,
                "stdout_identical_across_runs": True,
                "stages": [
                    {
                        "schema": "focr-gauntlet-stage/v1",
                        "source": "focr",
                        "stage": "decode_per_token",
                        "ledger_stage": True,
                        "unit": "ms",
                        "samples_ms": [10.0, 10.0, 10.0],
                        "best_ms": 10.0,
                        "cv_pct": 0.0,
                        "n": 3,
                        "warmup_discarded": 1,
                        "threads": 8,
                        "precision": "focr-int8",
                        "backend": "focr",
                        "allocator": "system",
                        "tokens": 10,
                        "tokens_consistent": True,
                        "synthetic": False,
                    }
                ],
            }
            ref_doc = {
                "schema": "focr-gauntlet-stages/v1",
                "source": "reference",
                "created_utc": timestamp,
                "page": "/fixtures/fixture.png",
                "page_sha256": "a" * 64,
                "synthetic": False,
                "stages": [
                    {
                        "schema": "focr-gauntlet-stage/v1",
                        "source": "reference",
                        "stage": "decode_per_token",
                        "ledger_stage": True,
                        "unit": "ms",
                        "samples_ms": [20.0, 20.0, 20.0],
                        "best_ms": 20.0,
                        "cv_pct": 0.0,
                        "n": 3,
                        "warmup_discarded": 1,
                        "threads": 8,
                        "thread_proof": {"budget": 8, "torch_num_threads": 8},
                        "precision": "bf16",
                        "backend": "hf",
                        "allocator": "system",
                        "synthetic": False,
                    }
                ],
            }
            roofline_doc = {
                "schema": "focr-gauntlet-roofline/v1",
                "created_utc": timestamp,
                "synthetic": False,
                "floors": [
                    {
                        "stage": "decode_per_token",
                        "floor_kind": "memory",
                        "floor_ms": 5.0,
                    }
                ],
            }
            return row, row_doc, focr_doc, ref_doc, roofline_doc

        def write_perf_manifest() -> None:
            names = sorted(
                path.relative_to(evidence_dir).as_posix()
                for path in evidence_dir.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS"
            )
            (evidence_dir / "SHA256SUMS").write_text(
                "".join(
                    f"{_sha256_file(evidence_dir / name)}  {name}\n" for name in names
                ),
                encoding="utf-8",
            )

        def write_perf_fixture(
            row_doc: dict,
            focr_doc: dict,
            ref_doc: dict,
            roofline_doc: dict,
        ) -> None:
            for name, payload in (
                ("focr_stages.json", focr_doc),
                ("ref_stages.json", ref_doc),
                ("roofline.json", roofline_doc),
                ("correctness_receipt.json", correctness_receipt_doc),
            ):
                (evidence_dir / name).write_text(
                    json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
                )
            receipt_sha = _sha256_file(evidence_dir / "correctness_receipt.json")
            row_doc["rows"][0]["correctness_proof"] = (
                f"receipt=correctness_receipt.json sha256={receipt_sha} "
                "metric=cer_norm value=0.000000 max=0.250000 result=pass"
            )
            row_doc["inputs"] = {
                name.removesuffix(".json"): {
                    "path": f"/measurements/{name}",
                    "bundle_path": name,
                    "sha256": _sha256_file(evidence_dir / name),
                }
                for name in (
                    "focr_stages.json",
                    "ref_stages.json",
                    "roofline.json",
                    "correctness_receipt.json",
                )
            }
            (evidence_dir / "row.json").write_text(
                json.dumps(row_doc, sort_keys=True) + "\n", encoding="utf-8"
            )
            (evidence_dir / "PERF_LEDGER_ROW.md").write_text(
                "| "
                + " | ".join(row_doc["rows"][0][column] for column in perf_columns)
                + " |\n",
                encoding="utf-8",
            )
            raw_dir = evidence_dir / "raw"
            raw_dir.mkdir(exist_ok=True)
            for run in range(1, 4):
                for suffix in ("stdout", "stderr"):
                    (raw_dir / f"run_{run:03d}.{suffix}").write_text(
                        f"run {run} {suffix}\n", encoding="utf-8"
                    )
            write_perf_manifest()

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        current_perf = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_current_eligible_evidence_passes",
            current_perf["ok"],
            verdict=current_perf,
        )

        forged_row, forged_row_doc, forged_focr, forged_ref, forged_roofline = (
            fresh_perf_fixture()
        )
        forged_focr.update(source="wrong", synthetic=True, page="/fixtures/other.png")
        forged_focr["stages"][0].update(
            synthetic=True,
            cv_pct=-3.0,
            precision="",
            allocator="",
        )
        forged_ref["stages"][0].pop("thread_proof")
        forged_roofline["synthetic"] = True
        write_perf_fixture(forged_row_doc, forged_focr, forged_ref, forged_roofline)
        forged_perf = perf_evidence_verdict(
            perf_text_for(forged_row), fixture_root, fixed_head, fixed_now
        )
        forged_reasons = forged_perf["candidates"][0]["reasons"]
        check(
            "perf_hash_valid_synthetic_forgery_rejected",
            not forged_perf["ok"]
            and any("source identity" in reason for reason in forged_reasons)
            and any("synthetic" in reason for reason in forged_reasons)
            and any("fixture pages" in reason for reason in forged_reasons)
            and any("thread proof" in reason for reason in forged_reasons)
            and any(
                "roofline evidence is synthetic" in reason for reason in forged_reasons
            ),
            verdict=forged_perf,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)

        tampered_ledger_row = dict(perf_row)
        tampered_ledger_row["focr_ms"] = "9.000"
        tampered_ledger = perf_evidence_verdict(
            perf_text_for(tampered_ledger_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_ledger_timing_tamper_rejected",
            not tampered_ledger["ok"]
            and any(
                "ledger row differs from hashed row.json" in reason
                for reason in tampered_ledger["candidates"][0]["reasons"]
            ),
            verdict=tampered_ledger,
        )

        tampered_correctness_row = dict(perf_row)
        tampered_correctness_row["correctness_proof"] = "FAILED: token mismatch"
        tampered_correctness = perf_evidence_verdict(
            perf_text_for(tampered_correctness_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_ledger_failure_correctness_text_rejected",
            not tampered_correctness["ok"]
            and any(
                "not a structured hash-bound pass receipt" in reason
                for reason in tampered_correctness["candidates"][0]["reasons"]
            ),
            verdict=tampered_correctness,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        focr_stage_doc["created_utc"] = "2026-07-08T00:00:00Z"
        ref_stage_doc["created_utc"] = "2026-07-08T00:00:00Z"
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        stale_perf = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_fresh_row_cannot_rewrap_stale_stage_evidence",
            not stale_perf["ok"]
            and any(
                "stale" in reason
                for candidate in stale_perf["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=stale_perf,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        focr_stage_doc["stages"][0]["cv_pct"] = 8.0
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        noisy_perf = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_ineligible_cv_fails_closed",
            not noisy_perf["ok"]
            and any(
                "cv_pct" in reason
                for candidate in noisy_perf["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=noisy_perf,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        focr_stage_doc["stages"][0]["best_ms"] = 11.0
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        stage_mismatch = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_hashed_row_must_match_hashed_stage_timing",
            not stage_mismatch["ok"]
            and any(
                "hashed row focr_ms does not match stage best_ms" in reason
                for reason in stage_mismatch["candidates"][0]["reasons"]
            ),
            verdict=stage_mismatch,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        perf_row["correctness_proof"] = "outputs differ from reference; CER=0.91"
        (evidence_dir / "row.json").write_text(
            json.dumps(row_doc, sort_keys=True) + "\n", encoding="utf-8"
        )
        (evidence_dir / "PERF_LEDGER_ROW.md").write_text(
            "| " + " | ".join(perf_row[column] for column in perf_columns) + " |\n",
            encoding="utf-8",
        )
        write_perf_manifest()
        hashed_failure = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_rehashed_failure_correctness_proof_rejected",
            not hashed_failure["ok"]
            and any(
                reason.startswith(
                    "hashed row: correctness proof is not a structured hash-bound pass receipt"
                )
                for reason in hashed_failure["candidates"][0]["reasons"]
            ),
            verdict=hashed_failure,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        row_doc["rows"].append(dict(perf_row))
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        duplicate_row = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_duplicate_hashed_claim_row_rejected",
            not duplicate_row["ok"]
            and any(
                "exactly one ledger claim/stage row" in reason
                for reason in duplicate_row["candidates"][0]["reasons"]
            ),
            verdict=duplicate_row,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        row_doc["inputs"]["focr_stages"]["sha256"] = "0" * 64
        (evidence_dir / "row.json").write_text(
            json.dumps(row_doc, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_perf_manifest()
        bad_input_binding = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_row_input_sha_mismatch_rejected",
            not bad_input_binding["ok"]
            and any(
                "inputs.focr_stages.sha256 does not bind" in reason
                for reason in bad_input_binding["candidates"][0]["reasons"]
            ),
            verdict=bad_input_binding,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        wrong_head_perf = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, "f" * 40, fixed_now
        )
        check(
            "perf_wrong_head_fails_closed",
            not wrong_head_perf["ok"]
            and any(
                "git_head" in reason
                for candidate in wrong_head_perf["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=wrong_head_perf,
        )

        certificate_root = fixture_root / "bundle-case"
        custom_bundle_dir = certificate_root / "custom-cert-output"
        custom_bundle_dir.mkdir(parents=True)
        (certificate_root / "Cargo.toml").write_text(
            '[package]\nname = "franken_ocr"\nversion = "0.0.0"\n',
            encoding="utf-8",
        )
        clean_worktree = {
            "status_ok": True,
            "clean": True,
            "dirty_path_count": 0,
            "status_porcelain": [],
            "status_sha256": _sha256_text(""),
        }
        dirty_worktree = {
            "status_ok": True,
            "clean": False,
            "dirty_path_count": 1,
            "status_porcelain": [" M evidence.json"],
            "status_sha256": _sha256_text(" M evidence.json\n"),
        }
        strict_relative = str(custom_bundle_dir.relative_to(certificate_root))
        strict_class_paths = {
            class_name: f"{strict_relative}/{filename}"
            for class_name, (filename, _schema) in STRICT_BUNDLE_CLASSES.items()
        }
        strict_timestamp = fixed_now.isoformat().replace("+00:00", "Z")
        source_claim_paths = (
            f"{strict_relative}/claim_sources/RELEASE_READINESS.json",
            f"{strict_relative}/claim_sources/ROUNDS.jsonl",
            *(
                f"{strict_relative}/claim_sources/{Path(path).name}"
                for path in HYPOTHESIS_LEDGER_PATHS
            ),
        )
        claim_path_by_original = {
            "docs/gauntlet/RELEASE_READINESS.json": source_claim_paths[0],
            "docs/gauntlet/ROUNDS.jsonl": source_claim_paths[1],
            **{
                original: source_claim_paths[index]
                for index, original in enumerate(HYPOTHESIS_LEDGER_PATHS, start=2)
            },
        }
        strict_core_mapping = {
            original: claim_path_by_original.get(
                original, f"{strict_relative}/core_sources/{original}"
            )
            for original in CORE_EVIDENCE_PATHS
        }
        strict_proof_mapping = {
            original: strict_core_mapping.get(
                original, f"{strict_relative}/core_sources/{original}"
            )
            for original in CERTIFICATION_PROOF_EVIDENCE_PATHS
        }
        feature_universe_relative = f"{strict_relative}/feature_universe.json"
        hypothesis_evidence_paths = (
            f"{strict_relative}/evidence/result.txt",
            f"{strict_relative}/evidence/SHA256SUMS",
        )
        required_strict_artifacts = (
            *strict_class_paths.values(),
            feature_universe_relative,
            *strict_core_mapping.values(),
            *strict_proof_mapping.values(),
            *hypothesis_evidence_paths,
        )
        trusted_signers = {
            "producer@example.test": "A" * 40,
            "reviewer@example.test": "B" * 40,
            "release@example.test": "C" * 40,
        }
        strict_verifier_kwargs = {
            "trusted_signers": trusted_signers,
            "ci_run_verifier": lambda _root, run_id, head, artifacts: (
                run_id == "123"
                and head == fixed_head
                and bool(artifacts)
                and all(
                    item.get("source_ci_artifact_name") == "strict-self-test"
                    for item in artifacts
                )
            ),
        }

        (custom_bundle_dir / "claim_sources").mkdir()
        for original, relative in {
            **strict_core_mapping,
            **strict_proof_mapping,
        }.items():
            if original in claim_path_by_original:
                continue
            path = certificate_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "self-test-core-evidence.v1",
                            "generated_at_utc": strict_timestamp,
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                path.write_text(
                    f"self-test snapshot for {original}\n", encoding="utf-8"
                )
            os.utime(path, (fixed_now.timestamp(), fixed_now.timestamp()))
        hypothesis_evidence_dir = custom_bundle_dir / "evidence"
        hypothesis_evidence_dir.mkdir()
        hypothesis_result = hypothesis_evidence_dir / "result.txt"
        hypothesis_result.write_text("measured no-evidence result\n", encoding="utf-8")
        (hypothesis_evidence_dir / "SHA256SUMS").write_text(
            f"{_sha256_file(hypothesis_result)}  result.txt\n", encoding="utf-8"
        )
        for path in hypothesis_evidence_dir.iterdir():
            os.utime(path, (fixed_now.timestamp(), fixed_now.timestamp()))
        readiness_source = {
            "schema_version": "gauntlet.release_readiness.v1",
            "artifact": "franken_ocr.release_readiness.v1",
            "cell_set_sha256": CERTIFICATION_READINESS_CELL_SET_SHA256,
            "generated_at_utc": strict_timestamp,
            "generated_by": "scripts/gauntlet_cert.py --release-readiness",
            "cells": [
                {
                    "cell": cell,
                    "status": "red" if cell == "certification_bundle" else "green",
                    "evidence": "release_certificate.json"
                    if cell == "certification_bundle"
                    else "self-test evidence",
                    "evidence_paths": [
                        strict_proof_mapping[original]
                        for original in CERTIFICATION_READINESS_EVIDENCE_PATHS.get(
                            cell, ()
                        )
                    ],
                }
                for cell in CERTIFICATION_READINESS_CELLS
            ],
            "green": len(CERTIFICATION_EXTERNAL_READINESS_CELLS),
            "red": 1,
            "blocking_cells": ["certification_bundle"],
            "ship": False,
        }
        (certificate_root / source_claim_paths[0]).write_text(
            json.dumps(readiness_source, sort_keys=True) + "\n", encoding="utf-8"
        )
        source_hypotheses = {
            ledger: canonical_experiment(f"EXP-{index:04d}", pillar)
            for index, (ledger, pillar) in enumerate(
                (
                    (source_claim_paths[2], "perf"),
                    (source_claim_paths[3], "perf"),
                    (source_claim_paths[4], "conformance"),
                    (source_claim_paths[5], "surface"),
                ),
                start=1,
            )
        }
        for relative, content in source_hypotheses.items():
            path = certificate_root / relative
            path.write_text(content, encoding="utf-8")
            os.utime(path, (fixed_now.timestamp(), fixed_now.timestamp()))
        strict_rounds = [
            {
                **canonical_round(round_number, 0 if round_number >= 9 else 4),
                "date": "2026-07-10",
            }
            for round_number in range(1, 11)
        ]
        (certificate_root / source_claim_paths[1]).write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n" for record in strict_rounds
            ),
            encoding="utf-8",
        )
        signature_dir = custom_bundle_dir / "signatures"
        signature_dir.mkdir()
        for name in ("producer.asc", "reviewer.asc", "release.asc"):
            (signature_dir / name).write_text(
                "self-test detached signature\n", encoding="utf-8"
            )

        base_strict_documents = {
            "confidence_gate": {
                "schema_version": STRICT_BUNDLE_CLASSES["confidence_gate"][1],
                "generated_at_utc": strict_timestamp,
                "release_decision": "Allow",
                "min_verification_pct_observed": 100.0,
                "required_suite_pass_rate_pct_observed": 100.0,
                "high_severity_counterexample_count": 0,
                "constants_enforced": list(CERTIFICATION_CONSTANTS),
            },
            "benchmark_summary": {
                "schema_version": STRICT_BUNDLE_CLASSES["benchmark_summary"][1],
                "generated_at_utc": strict_timestamp,
                "pass_over_pass_gates": {
                    name: {
                        "passed": True,
                        "minimum_pct": minimum,
                        "regression_pct": 0.0,
                    }
                    for name, minimum in {
                        "primary_score": -3.0,
                        "geomean": -5.0,
                        "category_geomean": -10.0,
                        "p90": -15.0,
                        "throughput_drop": -5.0,
                    }.items()
                },
            },
            "scorecards": {
                "schema_version": STRICT_BUNDLE_CLASSES["scorecards"][1],
                "generated_at_utc": strict_timestamp,
                "parity_score_lower_bound": 0.9,
                "per_category_lower": {"feature-a": 0.9},
            },
            "critical_path_report": {
                "schema_version": STRICT_BUNDLE_CLASSES["critical_path_report"][1],
                "generated_at_utc": strict_timestamp,
                "open_high_critical": 0,
                "waived_high_critical": 0,
                "findings": [],
            },
            "ratchet_state": {
                "schema_version": STRICT_BUNDLE_CLASSES["ratchet_state"][1],
                "generated_at_utc": strict_timestamp,
                "previous_bound": 0.7,
                "current_lower_bound": 0.8,
                "commit_sha": fixed_head,
                "timestamp": strict_timestamp,
                "advance_reason": "self-test monotone advance",
                "per_category_bounds": {"feature-a": 0.8},
            },
        }

        def write_strict_fixture(
            documents: dict[str, dict] | None = None,
        ) -> tuple[dict[str, dict], dict, dict]:
            documents = json.loads(json.dumps(documents or base_strict_documents))
            feature_universe = {
                "schema_version": "gauntlet.feature_universe.v1",
                "generated_at_utc": strict_timestamp,
                "definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
                "features": list(CERTIFICATION_FEATURE_UNIVERSE),
            }
            (certificate_root / feature_universe_relative).write_text(
                json.dumps(feature_universe, sort_keys=True) + "\n", encoding="utf-8"
            )
            universe_sha = _sha256_file(certificate_root / feature_universe_relative)
            documents["verification_contract"] = {
                "schema_version": STRICT_BUNDLE_CLASSES["verification_contract"][1],
                "generated_at_utc": strict_timestamp,
                "feature_universe_path": feature_universe_relative,
                "feature_universe_sha256": universe_sha,
                "feature_universe_definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
                "rows": [
                    {
                        "feature_id": feature["feature_id"],
                        "proof_obligation": obligation,
                        "status": "pass",
                        "gate": "allowed",
                        "evidence_paths": [
                            strict_proof_mapping[original]
                            for original in CERTIFICATION_READINESS_EVIDENCE_PATHS[
                                obligation
                            ]
                        ],
                        "evidence_sha256s": {
                            strict_proof_mapping[original]: _sha256_file(
                                certificate_root / strict_proof_mapping[original]
                            )
                            for original in CERTIFICATION_READINESS_EVIDENCE_PATHS[
                                obligation
                            ]
                        },
                    }
                    for feature in CERTIFICATION_FEATURE_UNIVERSE
                    for obligation in feature["proof_obligations"]
                ],
            }
            for class_name, payload in documents.items():
                if class_name == "ci_manifest":
                    continue
                filename = STRICT_BUNDLE_CLASSES[class_name][0]
                (custom_bundle_dir / filename).write_text(
                    json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            non_ci_paths = [
                relative
                for class_name, relative in strict_class_paths.items()
                if class_name != "ci_manifest"
            ]
            non_ci_paths.extend(
                [
                    feature_universe_relative,
                    *strict_core_mapping.values(),
                    *strict_proof_mapping.values(),
                    *hypothesis_evidence_paths,
                ]
            )
            non_ci_paths = list(dict.fromkeys(non_ci_paths))

            def manifest_entry(relative: str) -> dict:
                path = certificate_root / relative
                timestamp_value, timestamp_source = _artifact_native_timestamp(path)
                return {
                    "artifact": relative,
                    "sha256": _sha256_file(path),
                    "age_hours": 0.0,
                    "timestamp_utc": _timestamp_text(timestamp_value),
                    "timestamp_source": timestamp_source,
                }

            preliminary_entries = [
                manifest_entry(relative) for relative in non_ci_paths
            ]
            documents["ci_manifest"] = {
                "schema_version": STRICT_BUNDLE_CLASSES["ci_manifest"][1],
                "generated_at_utc": strict_timestamp,
                "repository": CERTIFICATION_GITHUB_REPOSITORY,
                "workflow": CERTIFICATION_GITHUB_WORKFLOW,
                "event": CERTIFICATION_GITHUB_EVENT,
                "artifacts": [
                    {
                        "artifact": entry["artifact"],
                        "sha256": entry["sha256"],
                        "schema_version": CI_ARTIFACT_BINDING_SCHEMA,
                        "source_ci_run_id": "123",
                        "source_ci_artifact_name": "strict-self-test",
                        "source_ci_artifact_path": entry["artifact"],
                    }
                    for entry in preliminary_entries
                ],
            }
            (custom_bundle_dir / STRICT_BUNDLE_CLASSES["ci_manifest"][0]).write_text(
                json.dumps(documents["ci_manifest"], sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest = [
                *preliminary_entries,
                manifest_entry(strict_class_paths["ci_manifest"]),
            ]
            strict_max_age = round(
                max(
                    0.0,
                    *(
                        (
                            fixed_now - _parse_utc_timestamp(entry["timestamp_utc"])
                        ).total_seconds()
                        / 3600.0
                        for entry in manifest
                    ),
                ),
                2,
            )
            manifest_root = _bundle_root_sha256(manifest)
            strict_hypotheses = hypothesis_texts_verdict(
                source_hypotheses,
                source_claim_paths[2:],
                custom_bundle_dir,
            )
            strict_convergence = convergence_verdict(strict_rounds, strict_hypotheses)
            certificate = {
                "schema_version": STRICT_CERTIFICATE_SCHEMA,
                "artifact": STRICT_CERTIFICATE_ARTIFACT,
                "template": STRICT_CERTIFICATE_SCHEMA,
                "project": "franken_ocr",
                "version": "0.0.0",
                "issued_at": strict_timestamp,
                "constants": CERTIFICATION_CONSTANTS,
                "reference": {"model_commit": UNLIMITED_OCR_MODEL_COMMIT},
                "git_head": fixed_head,
                "git_describe": "v0.0.0",
                "git_worktree": clean_worktree,
                "readiness": {
                    "green": len(CERTIFICATION_EXTERNAL_READINESS_CELLS),
                    "red": 0,
                    "blocking_cells": [],
                    "ship": True,
                },
                "convergence": strict_convergence,
                "required_pass_actuals": {
                    "min_verification_pct_observed": 100.0,
                    "required_suite_pass_rate_pct_observed": 100.0,
                    "high_severity_counterexample_count": 0,
                    "max_evidence_age_hours_observed": strict_max_age,
                },
                "high_severity_counterexamples": 0,
                "parity_score": 0.9,
                "refusal_reasons": [],
                "evidence_classes": strict_class_paths,
                "claim_sources": {
                    "release_readiness": source_claim_paths[0],
                    "rounds": source_claim_paths[1],
                    "hypothesis_ledgers": list(source_claim_paths[2:]),
                    "core_evidence": strict_core_mapping,
                    "readiness_evidence": strict_proof_mapping,
                },
                "readiness_cell_set_sha256": CERTIFICATION_READINESS_CELL_SET_SHA256,
                "feature_universe_sha256": universe_sha,
                "feature_universe_definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
                "evidence_bundle_sha256": manifest_root,
                "signers": [
                    "producer@example.test",
                    "reviewer@example.test",
                    "release@example.test",
                ],
                "detached_signatures": [
                    {
                        "signer": "producer@example.test",
                        "role": "producer",
                        "fingerprint": trusted_signers["producer@example.test"],
                        "scheme": "openpgp-detached",
                        "signature_path": f"{strict_relative}/signatures/producer.asc",
                    },
                    {
                        "signer": "reviewer@example.test",
                        "role": "independent-reviewer",
                        "fingerprint": trusted_signers["reviewer@example.test"],
                        "scheme": "openpgp-detached",
                        "signature_path": f"{strict_relative}/signatures/reviewer.asc",
                    },
                    {
                        "signer": "release@example.test",
                        "role": "release-authorizer",
                        "fingerprint": trusted_signers["release@example.test"],
                        "scheme": "openpgp-detached",
                        "signature_path": f"{strict_relative}/signatures/release.asc",
                    },
                ],
                "certified": True,
            }
            certificate["signed_claim_sha256"] = _certificate_signed_claim_sha256(
                certificate, manifest_root
            )
            bundle = {
                "schema_version": STRICT_BUNDLE_SCHEMA,
                "artifact": STRICT_BUNDLE_ARTIFACT,
                "generated_at_utc": strict_timestamp,
                "bundle_root_sha256": manifest_root,
                "manifest": manifest,
            }
            (custom_bundle_dir / "release_certificate.json").write_text(
                json.dumps(certificate, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (custom_bundle_dir / "certification_bundle.json").write_text(
                json.dumps(bundle, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return documents, certificate, bundle

        strict_documents, strict_certificate, strict_bundle = write_strict_fixture()
        strict_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_complete_strict_fixture_passes",
            strict_verdict["ok"],
            verdict=strict_verdict,
        )
        replay_manifest = json.loads(json.dumps(strict_bundle["manifest"]))
        replay_manifest[0]["timestamp_utc"] = _timestamp_text(
            fixed_now + timedelta(seconds=1)
        )
        check(
            "bundle_root_authenticates_freshness_metadata",
            _bundle_root_sha256(replay_manifest) != strict_bundle["bundle_root_sha256"],
        )

        default_dir_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_custom_directory_is_not_replaced_by_default",
            not default_dir_verdict["ok"]
            and any(
                "missing or unreadable" in reason
                for reason in default_dir_verdict["reasons"]
            ),
            verdict=default_dir_verdict,
        )

        symlink_output = certificate_root / "symlink-output"
        symlink_output.mkdir()
        os.symlink(
            certificate_root / "outside-certificate.json",
            symlink_output / "release_certificate.json",
        )
        safe_symlink_output, symlink_reasons = _safe_output_dir(
            certificate_root, str(symlink_output)
        )
        check(
            "bundle_output_leaf_symlink_rejected",
            safe_symlink_output is None
            and any("contains a symlink" in reason for reason in symlink_reasons),
            reasons=symlink_reasons,
        )
        looping_output = certificate_root / "looping-output"
        os.symlink(looping_output, looping_output)
        safe_looping_output, looping_reasons = _safe_output_dir(
            certificate_root, str(looping_output)
        )
        check(
            "bundle_output_dangling_self_loop_symlink_rejected",
            safe_looping_output is None
            and any("symlink" in reason for reason in looping_reasons),
            reasons=looping_reasons,
        )

        (custom_bundle_dir / "release_certificate.json").write_text(
            json.dumps({"certified": True, "git_head": fixed_head}) + "\n",
            encoding="utf-8",
        )
        minimal_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_minimal_truncated_certificate_rejected",
            not minimal_verdict["ok"]
            and any("schema_version" in reason for reason in minimal_verdict["reasons"])
            and any("signers" in reason for reason in minimal_verdict["reasons"]),
            verdict=minimal_verdict,
        )

        write_strict_fixture()
        wrong_head_verdict = certificate_bundle_verdict(
            certificate_root,
            "f" * 40,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_wrong_head_rejected",
            not wrong_head_verdict["ok"]
            and any("git_head" in reason for reason in wrong_head_verdict["reasons"]),
            verdict=wrong_head_verdict,
        )

        _docs, claim_certificate, claim_bundle = write_strict_fixture()
        claim_certificate["readiness"]["green"] = 99
        (custom_bundle_dir / "release_certificate.json").write_text(
            json.dumps(claim_certificate, sort_keys=True) + "\n", encoding="utf-8"
        )
        unsigned_claim_tamper = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_unsigned_release_claim_tamper_rejected",
            not unsigned_claim_tamper["ok"]
            and any(
                "signed_claim_sha256" in reason
                for reason in unsigned_claim_tamper["reasons"]
            )
            and any(
                "hashed readiness cells" in reason
                for reason in unsigned_claim_tamper["reasons"]
            ),
            verdict=unsigned_claim_tamper,
        )

        write_strict_fixture()
        status_failure = {
            "status_ok": False,
            "clean": True,
            "dirty_path_count": 0,
            "status_porcelain": [],
            "status_sha256": _sha256_text("git status failed"),
            "status_error": "git status failed",
        }
        status_failure_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=status_failure,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_git_status_failure_is_not_clean",
            not status_failure_verdict["ok"]
            and any(
                "status could not be verified" in reason
                for reason in status_failure_verdict["reasons"]
            ),
            verdict=status_failure_verdict,
        )

        write_strict_fixture()
        ci_provenance_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            trusted_signers=trusted_signers,
            ci_run_verifier=lambda _root, _run_id, _head, _artifacts: False,
        )
        check(
            "bundle_unreachable_ci_run_rejected",
            not ci_provenance_verdict["ok"]
            and any(
                "CI run artifact provenance could not be verified" in reason
                for reason in ci_provenance_verdict["reasons"]
            ),
            verdict=ci_provenance_verdict,
        )

        write_strict_fixture()
        (custom_bundle_dir / "confidence_gate.json").write_text(
            "mutated evidence\n", encoding="utf-8"
        )
        bad_hash_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_manifest_hash_tamper_rejected",
            not bad_hash_verdict["ok"]
            and any(
                "hash mismatch" in reason for reason in bad_hash_verdict["reasons"]
            ),
            verdict=bad_hash_verdict,
        )

        write_strict_fixture()
        dirty_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=dirty_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_dirty_worktree_rejected",
            not dirty_verdict["ok"]
            and any(
                "worktree is dirty" in reason for reason in dirty_verdict["reasons"]
            ),
            verdict=dirty_verdict,
        )

        stale_documents = json.loads(json.dumps(base_strict_documents))
        stale_documents["confidence_gate"]["generated_at_utc"] = _timestamp_text(
            fixed_now - timedelta(hours=50)
        )
        write_strict_fixture(stale_documents)
        stale_native_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_recomputes_stale_native_timestamp",
            not stale_native_verdict["ok"]
            and any("is stale" in reason for reason in stale_native_verdict["reasons"]),
            verdict=stale_native_verdict,
        )

        future_documents = json.loads(json.dumps(base_strict_documents))
        future_documents["confidence_gate"]["generated_at_utc"] = _timestamp_text(
            fixed_now + timedelta(hours=2)
        )
        write_strict_fixture(future_documents)
        future_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_future_native_timestamp_rejected",
            not future_verdict["ok"]
            and any("future" in reason for reason in future_verdict["reasons"]),
            verdict=future_verdict,
        )

        _docs, duplicate_certificate, duplicate_bundle = write_strict_fixture()
        duplicate_bundle["manifest"].append(dict(duplicate_bundle["manifest"][0]))
        (custom_bundle_dir / "certification_bundle.json").write_text(
            json.dumps(duplicate_bundle, sort_keys=True) + "\n", encoding="utf-8"
        )
        duplicate_manifest_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_duplicate_manifest_path_rejected",
            not duplicate_manifest_verdict["ok"]
            and any(
                "duplicate artifact" in reason
                for reason in duplicate_manifest_verdict["reasons"]
            ),
            verdict=duplicate_manifest_verdict,
        )

        _docs, root_certificate, root_bundle = write_strict_fixture()
        root_bundle["bundle_root_sha256"] = "0" * 64
        (custom_bundle_dir / "certification_bundle.json").write_text(
            json.dumps(root_bundle, sort_keys=True) + "\n", encoding="utf-8"
        )
        root_mismatch_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_root_mismatch_rejected",
            not root_mismatch_verdict["ok"]
            and any(
                "root hash does not match" in reason
                for reason in root_mismatch_verdict["reasons"]
            ),
            verdict=root_mismatch_verdict,
        )

        _docs, signature_certificate, signature_bundle = write_strict_fixture()
        invalid_signature_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: False,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_invalid_detached_signatures_rejected",
            not invalid_signature_verdict["ok"]
            and sum(
                "signature verification failed" in reason
                for reason in invalid_signature_verdict["reasons"]
            )
            == 3,
            verdict=invalid_signature_verdict,
        )

        _docs, reused_key_certificate, _bundle = write_strict_fixture()
        reused_fingerprint = "D" * 40
        reused_trust = {
            signer: reused_fingerprint for signer in reused_key_certificate["signers"]
        }
        for signature in reused_key_certificate["detached_signatures"]:
            signature["fingerprint"] = reused_fingerprint
        reused_key_certificate["signed_claim_sha256"] = (
            _certificate_signed_claim_sha256(
                reused_key_certificate,
                reused_key_certificate["evidence_bundle_sha256"],
            )
        )
        (custom_bundle_dir / "release_certificate.json").write_text(
            json.dumps(reused_key_certificate, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reused_key_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            trusted_signers=reused_trust,
            ci_run_verifier=strict_verifier_kwargs["ci_run_verifier"],
        )
        check(
            "bundle_one_key_cannot_impersonate_three_signers",
            not reused_key_verdict["ok"]
            and any(
                "distinct signers, roles, fingerprints" in reason
                for reason in reused_key_verdict["reasons"]
            ),
            verdict=reused_key_verdict,
        )

        signature_certificate["signers"] = ["same@example.test"] * 3
        (custom_bundle_dir / "release_certificate.json").write_text(
            json.dumps(signature_certificate, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        repeated_signer_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_requires_three_distinct_signers",
            not repeated_signer_verdict["ok"]
            and any(
                "distinct signers" in reason
                for reason in repeated_signer_verdict["reasons"]
            ),
            verdict=repeated_signer_verdict,
        )

    # bd-re8.15 — the live-stream fold.
    check(
        "eprocess_classify_maps_real_shapes",
        classify_invariant_line(
            {"test": "e2e", "case": "determinism_gate", "result": "pass"}
        )
        == ("INV-DETERMINISM", False)
        and classify_invariant_line({"check": "int32_overflow_proof", "result": "fail"})
        == ("INV-I32-NOOVERFLOW", True)
        and classify_invariant_line({"case": "kv_cap_bound", "result": "pass"})
        == ("INV-KV-CAP", False)
        and classify_invariant_line({"test": "robot_selftest", "result": "pass"})
        == ("INV-SIMD-SCALAR", False)
        # skips are NOT observations; unmatched lines are None.
        and classify_invariant_line(
            {"case": "determinism_gate", "result": "skip_no_model"}
        )
        is None
        and classify_invariant_line({"case": "l4_tokens", "result": "pass"}) is None,
    )
    ep_clean = _eprocess_from_state("INV-DETERMINISM", None)
    for _ in range(5):
        ep_clean.observe(False)
    check(
        "eprocess_clean_stream_shrinks",
        ep_clean.e_value < 1.0 and ep_clean.rejected_at is None,
        e=ep_clean.e_value,
    )
    ep_trip = _eprocess_from_state("INV-DETERMINISM", None)
    tripped = ep_trip.observe(True)
    check(
        "eprocess_single_hardware_violation_trips",
        tripped and ep_trip.rejected_at == 1,
        e=ep_trip.e_value,
    )
    saved = {
        "e_value": ep_clean.e_value,
        "obs_count": ep_clean.obs_count,
        "rejected_at": None,
    }
    ep_resumed = _eprocess_from_state("INV-DETERMINISM", saved)
    ep_resumed.observe(False)
    check(
        "eprocess_state_roundtrip_never_resets",
        ep_resumed.obs_count == 6 and ep_resumed.e_value < ep_clean.e_value,
    )
    # Injected violation end-to-end, ALL FOUR invariants: the emission-shaped
    # fail line classifies to the right invariant and a single genuine alarm
    # crosses the Ville threshold (acceptance: alarms on injected violation,
    # silent on the clean stream — the clean side is the checks above).
    injected = {
        "INV-DETERMINISM": {"case": "determinism_gate", "result": "fail"},
        "INV-SIMD-SCALAR": {"command": "robot.selftest", "verdict": "fail"},
        "INV-I32-NOOVERFLOW": {
            "test": "int32_overflow_proof",
            "case": "i32_overflow_headroom_k6848",
            "result": "fail",
        },
        "INV-KV-CAP": {
            "test": "spec_ring_rollback",
            "case": "kv_cap_ring_bound",
            "result": "fail",
        },
    }
    for name, line in injected.items():
        got = classify_invariant_line(line)
        ep_inj = _eprocess_from_state(name, None)
        tripped_inj = (
            got == (name, True) and ep_inj.observe(True) and ep_inj.rejected_at == 1
        )
        check(
            f"eprocess_injected_violation_trips_{name}",
            tripped_inj,
            classified=str(got),
        )

    if failures:
        emit("gauntlet-cert-self-test", False, failed=failures)
        return 1
    emit("gauntlet-cert-self-test", True, checks_passed=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--self-test", action="store_true", help="validate the gauntlet math (CI gate)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="print a worked franken_ocr scorecard + ratchet verdict",
    )
    parser.add_argument(
        "--from-parity",
        metavar="MD",
        help="compute the surface-pillar scorecard from the REAL FEATURE_PARITY.md (bd-re8.13)",
    )
    parser.add_argument(
        "--residuals",
        metavar="JSON",
        help="held-out round residuals (JSON array file); absent -> bootstrap wide band",
    )
    parser.add_argument(
        "--scorecard-out", metavar="FILE", help="also write the scorecard artifact here"
    )
    parser.add_argument(
        "--convergence",
        metavar="ROUNDS_JSONL",
        nargs="?",
        const="docs/gauntlet/ROUNDS.jsonl",
        help="check the >=10-rounds / >=2-clean convergence gate; exit 1 until met (bd-wp8.8)",
    )
    parser.add_argument(
        "--eprocess-fold",
        metavar="RAW_NDJSON",
        help="fold a real test-log NDJSON stream into the persisted e-process state (bd-re8.15)",
    )
    parser.add_argument(
        "--release-readiness",
        action="store_true",
        help="the all-green ship gate: read every cell's evidence artifact, exit 1 on ANY red (bd-wp8.10)",
    )
    parser.add_argument(
        "--readiness-out",
        metavar="FILE",
        default="docs/gauntlet/RELEASE_READINESS.json",
        help="where to write the release-readiness scorecard artifact",
    )
    parser.add_argument(
        "--eprocess-state",
        metavar="FILE",
        default="docs/gauntlet/EPROCESS_STATE.json",
        help="the persisted (never-reset) per-invariant e-process state",
    )
    parser.add_argument(
        "--bundle",
        metavar="OUT_DIR",
        nargs="?",
        const="docs/gauntlet/bundle",
        help="produce the release certification bundle (bd-wp8.9); exit 1 until certified",
    )
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.demo:
        return _demo()
    if args.from_parity:
        return from_parity(args.from_parity, args.residuals, args.scorecard_out)
    if args.convergence:
        return convergence(args.convergence)
    if args.eprocess_fold:
        return eprocess_fold(args.eprocess_fold, args.eprocess_state)
    if args.release_readiness:
        return release_readiness(args.readiness_out)
    if args.bundle:
        return produce_bundle(args.bundle)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
