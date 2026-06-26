#!/usr/bin/env python3
"""Validate franken_ocr's artifact-graph ledgers.

The ledgers are intentionally empty while inference is unimplemented, but their
schema is load-bearing: future performance wins, negative evidence, and accepted
discrepancies must all carry truth-pack provenance and reproducible commands.
This check keeps that contract machine-enforced without needing model weights.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVIDENCE_MANIFEST_NAMES = {
    "SHA256SUMS",
    "SHA256SUMS.txt",
    "sha256sums.txt",
    "sha256.txt",
    "manifest.sha256",
    "manifest.json",
}


def emit(check: str, ok: bool, **fields: object) -> None:
    payload = {"check": check, "result": "pass" if ok else "fail", **fields}
    print(json.dumps(payload, sort_keys=True))


def strip_fenced_blocks(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def require_tokens(path: Path, text: str, tokens: list[str], failures: list[str]) -> None:
    for token in tokens:
        ok = token in text
        emit("required-token", ok, file=str(path.relative_to(ROOT)), token=token)
        if not ok:
            failures.append(f"{path}: missing required token {token!r}")


def require_any_token(path: Path, text: str, label: str, tokens: list[str], failures: list[str]) -> None:
    ok = any(token in text for token in tokens)
    emit("required-token-any", ok, file=str(path.relative_to(ROOT)), label=label, tokens=tokens)
    if not ok:
        failures.append(f"{path}: missing required token group {label!r}: {tokens!r}")


def markdown_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def is_empty_cell(value: str) -> bool:
    return value.strip() in {"", "_—_", "_-_", "_no measurements yet_"}


def kill_switch_env_vars(text: str) -> list[str]:
    return sorted(set(re.findall(r"\bFOCR_[A-Z0-9_]+\b", text)))


def undefined_kill_switches(kill_line: str, src_text: str) -> list[str]:
    return [var for var in kill_switch_env_vars(kill_line) if var not in src_text]


def has_sha256_manifest(evidence_path: Path) -> bool:
    if not evidence_path.is_dir():
        return False
    for child in evidence_path.iterdir():
        if not child.is_file():
            continue
        name = child.name.lower()
        if child.name in EVIDENCE_MANIFEST_NAMES or ("sha256" in name and "manifest" in name):
            return True
    return False


def self_test_kill_switch_validation(failures: list[str]) -> None:
    kill_line = "- Fallback / kill-switch state: FOCR_LEDGER_ONLY=1 restores reference behavior"

    missing = undefined_kill_switches(kill_line, "")
    missing_ok = missing == ["FOCR_LEDGER_ONLY"]
    emit("ledger-self-test-kill-switch-missing", missing_ok, undefined=missing)
    if not missing_ok:
        failures.append("self-test: ledger-only FOCR_* var must be reported undefined")

    source_text = 'pub const LEDGER_TEST_ENV: &str = "FOCR_LEDGER_ONLY";'
    defined = undefined_kill_switches(kill_line, source_text)
    defined_ok = defined == []
    emit("ledger-self-test-kill-switch-defined", defined_ok, undefined=defined)
    if not defined_ok:
        failures.append("self-test: source-defined FOCR_* var must be accepted")


def self_test_evidence_manifest_validation(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="focr-ledger-self-test-") as tmp:
        evidence_dir = Path(tmp) / "artifacts" / "perf" / "bd-self-test"
        evidence_dir.mkdir(parents=True)

        missing_ok = not has_sha256_manifest(evidence_dir)
        emit("ledger-self-test-evidence-manifest-missing", missing_ok)
        if not missing_ok:
            failures.append("self-test: evidence dir without a sha256 manifest must fail")

        (evidence_dir / "SHA256SUMS").write_text(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  baseline.log\n",
            encoding="utf-8",
        )
        present_ok = has_sha256_manifest(evidence_dir)
        emit("ledger-self-test-evidence-manifest-present", present_ok)
        if not present_ok:
            failures.append("self-test: evidence dir with SHA256SUMS must pass")


def require_evidence_manifest(
    ledger: Path,
    context: str,
    evidence_path: Path,
    evidence_id: str,
    failures: list[str],
) -> None:
    has_manifest = has_sha256_manifest(evidence_path)
    emit(
        "evidence-sha256-manifest",
        has_manifest,
        context=context,
        evidence_id=evidence_id,
    )
    if not has_manifest:
        failures.append(
            f"{ledger}: {context} evidence dir {evidence_id} must contain a SHA-256 manifest "
            f"({', '.join(sorted(EVIDENCE_MANIFEST_NAMES))})"
        )


def check_perf_ledger(path: Path, text: str, failures: list[str]) -> None:
    required_columns = [
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

    header = next(
        (line for line in text.splitlines() if line.startswith("| date | claim_id | evidence_id |")),
        "",
    )
    columns = markdown_cells(header) if header else []
    for column in required_columns:
        ok = column in columns
        emit("perf-ledger-column", ok, column=column)
        if not ok:
            failures.append(f"{path}: missing PERF_LEDGER column {column!r}")

    if not columns:
        return

    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("|"):
            continue
        if "------" in line or line == header:
            continue
        cells = markdown_cells(line)
        if len(cells) != len(columns):
            continue
        row = dict(zip(columns, cells, strict=True))
        if is_empty_cell(row.get("claim_id", "")):
            continue
        evidence_id = row["evidence_id"]
        evidence_ok = evidence_id.startswith("artifacts/perf/")
        emit("perf-evidence-prefix", evidence_ok, line=line_no, evidence_id=evidence_id)
        if not evidence_ok:
            failures.append(f"{path}:{line_no}: evidence_id must start with artifacts/perf/")
            continue
        evidence_path = ROOT / evidence_id.rstrip("/")
        exists = evidence_path.is_dir()
        emit("perf-evidence-dir", exists, line=line_no, evidence_id=evidence_id)
        if not exists:
            failures.append(f"{path}:{line_no}: missing evidence dir {evidence_id}")
        else:
            require_evidence_manifest(path, f"line {line_no}", evidence_path, evidence_id, failures)

        for column in (
            "command/env",
            "fallback/kill-switch state",
            "fixture_hash",
            "model_commit",
            "reference_backend",
            "correctness_proof",
        ):
            filled = not is_empty_cell(row[column])
            emit("perf-required-cell", filled, line=line_no, column=column)
            if not filled:
                failures.append(f"{path}:{line_no}: empty required cell {column!r}")


def check_negative_evidence(path: Path, text: str, failures: list[str]) -> None:
    unfenced = strip_fenced_blocks(text)
    entries = list(
        re.finditer(
            r"^(?P<date>\d{4}-\d{2}-\d{2}) \| (?P<outcome>WIN|NEGATIVE\(reverted\)) \| (?P<lever>.+)$",
            unfenced,
            flags=re.MULTILINE,
        )
    )
    emit("negative-entry-count", True, count=len(entries))

    required_fields = [
        "claim_id:",
        "evidence_id:",
        "model source commit + fixture hash:",
        "CPU feature string:",
        "exact command + env:",
        "fallback / kill-switch state:",
        "measured before -> after vs reference:",
        "bit-exact correctness proof:",
        "disposition:",
        "do-not-retry:",
        "per-lever tally:",
        "agent:",
        "evidence dir:",
    ]

    for index, entry in enumerate(entries):
        end = entries[index + 1].start() if index + 1 < len(entries) else len(unfenced)
        section = unfenced[entry.start() : end]
        title = f"{entry.group('date')} {entry.group('outcome')}"
        for field in required_fields:
            ok = field in section
            emit("negative-required-field", ok, entry=title, field=field)
            if not ok:
                failures.append(f"{path}: {title} missing {field}")

        evidence_match = re.search(r"\bevidence_id:\s*(\S+)", section)
        evidence_id = evidence_match.group(1).rstrip(",.;") if evidence_match else ""
        evidence_ok = evidence_id.startswith("artifacts/perf/")
        emit("negative-evidence-prefix", evidence_ok, entry=title, evidence_id=evidence_id)
        if evidence_match and not evidence_ok:
            failures.append(f"{path}: {title} evidence_id must start with artifacts/perf/")
        if evidence_ok:
            evidence_path = ROOT / evidence_id.rstrip("/")
            exists = evidence_path.is_dir()
            emit("negative-evidence-dir", exists, entry=title, evidence_id=evidence_id)
            if not exists:
                failures.append(f"{path}: {title} missing evidence dir {evidence_id}")
            else:
                require_evidence_manifest(path, title, evidence_path, evidence_id, failures)

        disposition_ok = bool(re.search(r"disposition:\s*(KEEP|REVERT)\b", section))
        emit("negative-disposition", disposition_ok, entry=title)
        if not disposition_ok:
            failures.append(f"{path}: {title} disposition must be KEEP or REVERT")

        tally_ok = bool(re.search(r"per-lever tally:\s*W\s+\d+\s*/\s*L\s+\d+\s*/\s*N\s+\d+", section))
        emit("negative-tally", tally_ok, entry=title)
        if not tally_ok:
            failures.append(f"{path}: {title} per-lever tally must be W n / L n / N n")


def check_discrepancies(path: Path, text: str, failures: list[str]) -> None:
    unfenced = strip_fenced_blocks(text)
    headings = list(re.finditer(r"^## DISC-\d+:", unfenced, flags=re.MULTILINE))
    emit("disc-entry-count", True, count=len(headings))
    if not headings:
        return

    required_fields = [
        "- claim_id / evidence_id:",
        "- Provenance (model commit + fixture hash):",
        "- CPU feature string:",
        "- Exact command + env:",
        "- Reference behavior:",
        "- Our impl:",
        "- Fallback / kill-switch state:",
        "- Measured impact:",
        "- Resolution:",
        "- Tests affected:",
        "- Review date:",
    ]
    src_text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.rs"))

    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(unfenced)
        section = unfenced[heading.start() : end]
        title = heading.group(0).removeprefix("## ")
        for field in required_fields:
            ok = field in section
            emit("disc-required-field", ok, entry=title, field=field)
            if not ok:
                failures.append(f"{path}: {title} missing {field}")

        kill_line = next(
            (line for line in section.splitlines() if line.startswith("- Fallback / kill-switch state:")),
            "",
        )
        undefined = set(undefined_kill_switches(kill_line, src_text))
        for var in kill_switch_env_vars(kill_line):
            defined = var not in undefined
            emit("disc-kill-switch-defined", defined, entry=title, env=var)
            if not defined:
                failures.append(f"{path}: {title} references undefined kill switch {var}")


def main() -> int:
    failures: list[str] = []
    files = {
        "negative": DOCS / "NEGATIVE_EVIDENCE.md",
        "perf": DOCS / "PERF_LEDGER.md",
        "disc": DOCS / "DISCREPANCIES.md",
    }

    for name, path in files.items():
        exists = path.is_file()
        emit("ledger-exists", exists, ledger=name, file=str(path.relative_to(ROOT)))
        if not exists:
            failures.append(f"missing ledger {path}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    self_test_kill_switch_validation(failures)
    self_test_evidence_manifest_validation(failures)

    negative = files["negative"].read_text(encoding="utf-8")
    perf = files["perf"].read_text(encoding="utf-8")
    disc = files["disc"].read_text(encoding="utf-8")

    negative_shared_tokens = [
        "claim_id",
        "evidence_id",
        "model source commit",
        "fixture hash",
        "CPU feature string",
        "exact command + env",
        "fallback / kill-switch state",
    ]
    require_tokens(files["negative"], negative, negative_shared_tokens, failures)
    for path, text in ((files["perf"], perf), (files["disc"], disc)):
        require_any_token(path, text, "claim_id", ["claim_id", "claim_id / evidence_id"], failures)
        require_any_token(path, text, "evidence_id", ["evidence_id", "claim_id / evidence_id"], failures)
        require_any_token(path, text, "model commit", ["model source commit", "model_commit", "model commit + fixture hash"], failures)
        require_any_token(path, text, "fixture hash", ["fixture hash", "fixture_hash", "model commit + fixture hash"], failures)
        require_any_token(path, text, "cpu feature", ["CPU feature string", "arch/cpu_features"], failures)
        require_any_token(path, text, "command env", ["exact command + env", "Exact command + env", "command/env"], failures)
        require_any_token(
            path,
            text,
            "fallback kill-switch",
            ["fallback / kill-switch state", "Fallback / kill-switch state", "fallback/kill-switch state"],
            failures,
        )

    require_tokens(
        files["negative"],
        negative,
        [
            "measured before -> after vs reference",
            "bit-exact correctness proof",
            "disposition: KEEP / REVERT",
            "do-not-retry",
            "per-lever tally",
            "artifacts/perf/<bead>/",
        ],
        failures,
    )
    require_tokens(
        files["disc"],
        disc,
        [
            "- Reference behavior:",
            "- Our impl:",
            "- Measured impact:",
            "- Resolution:",
            "- Tests affected:",
            "- Review date:",
        ],
        failures,
    )

    check_perf_ledger(files["perf"], perf, failures)
    check_negative_evidence(files["negative"], negative, failures)
    check_discrepancies(files["disc"], disc, failures)

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    emit("ledger-lint-summary", True, checked=sorted(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
