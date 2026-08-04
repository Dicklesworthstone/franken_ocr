#!/usr/bin/env python3
"""Merge gauntlet measurements into a PERF_LEDGER row + evidence bundle (bd-re8.17).

Inputs (all REAL measurement/derivation files; anything missing → refusal):
  * `--focr-stages`   — `scripts/gauntlet_focr.sh` output (focr-gauntlet-stages/v1)
  * `--ref-stages`    — `scripts/gauntlet_reference.py` output (same schema)
  * `--roofline`      — `scripts/gauntlet_roofline.py` output

Output:
  * `artifacts/perf/bd-re8.17/<claim_id>/` evidence bundle: the three input
    JSONs, the raw focr run logs, `row.json`, `PERF_LEDGER_ROW.md`, and a
    `SHA256SUMS` manifest (`scripts/check_ledgers.py` requires one);
  * the exact markdown row(s) for `docs/PERF_LEDGER.md`, validated by running
    the REAL `scripts/check_ledgers.py` against a shadow copy of the repo's
    ledgers with the row inserted (`--check`, the default). `--apply`
    additionally inserts the row into the real `docs/PERF_LEDGER.md` and
    re-validates in place (reverting on failure).

Fairness/honesty gates (all fail-closed, per docs/PERF_LEDGER.md):
  * refuses synthetic (self-test) inputs;
  * refuses `cv_pct > 5` on either side (noise cannot land a claim);
  * refuses a focr/reference thread-budget or allocator mismatch;
  * refuses a reference record without its thread proof or precision;
  * refuses nondeterministic focr evidence (stdout or token drift across runs);
  * `claim_id`, `fixture_hash`, `arch/cpu_features`, `correctness_proof` are
    caller-supplied and required — there are no defaults for provenance.

Usage:
  gauntlet_row.py --focr-stages F.json --ref-stages R.json --roofline RL.json \
      --claim-id G2-decode-YYYYMMDD --fixture-hash SHA256 \
      --arch-features aarch64+neon+dotprod --correctness-proof 'PROOF' \
      [--stage decode_per_token ...] [--evidence-dir DIR] [--apply] [--notes S]
  gauntlet_row.py --self-test
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
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "PERF_LEDGER.md")
CHECKER = os.path.join(ROOT, "scripts", "check_ledgers.py")

# docs/truth-pack/PINNED_SOURCES.md — the ledger legend fixes this value for
# every franken_ocr row, so defaulting it is provenance, not fabrication.
MODEL_COMMIT = "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"

MAX_CV_PCT = 5.0  # docs/PERF_LEDGER.md: cv_pct > 5% is noise, ineligible
MAX_CORRECTNESS_CER = 0.25
_CORRECTNESS_CER_RE = re.compile(r"\bCER_norm=([0-9]+(?:\.[0-9]+)?)\b")
_CORRECTNESS_RECEIPT_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])((?:/|\.{1,2}/)[^()\s;]+cer\.json)(?=$|[\s;)])"
)

LEDGER_STAGES = (
    "preprocess",
    "vision_encode",
    "prefill",
    "decode_per_token",
    "end_to_end",
)

COLUMNS = [
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


class RowError(ValueError):
    """The inputs cannot honestly produce a ledger row."""


def load_json(path: str, want_schema: str | None = None) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        raise RowError(f"{path}: {err}") from err
    if not isinstance(doc, dict):
        raise RowError(f"{path}: expected a JSON object")
    if want_schema and doc.get("schema") != want_schema:
        raise RowError(
            f"{path}: expected schema {want_schema}, got {doc.get('schema')!r}"
        )
    return doc


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_head(root: str = ROOT) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise RowError(f"cannot resolve git HEAD: {err}") from err
    head = proc.stdout.strip()
    if proc.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise RowError("cannot resolve a canonical 40-hex git HEAD")
    return head


def validate_correctness_proof(value: str) -> dict:
    cer_matches = _CORRECTNESS_CER_RE.findall(value)
    receipt_paths = _CORRECTNESS_RECEIPT_PATH_RE.findall(value)
    if len(cer_matches) != 1 or len(receipt_paths) != 1:
        raise RowError(
            "correctness_proof must name exactly one CER_norm value and one cer.json receipt path"
        )
    claimed_cer = float(cer_matches[0])
    receipt_path = os.path.abspath(receipt_paths[0])
    receipt = load_json(receipt_path)
    aggregate = receipt.get("aggregate")
    pages = receipt.get("pages")
    if not isinstance(aggregate, dict) or not isinstance(pages, list) or not pages:
        raise RowError("correctness receipt lacks aggregate/pages evidence")
    measured_cer = aggregate.get("cer_norm")
    pages_total = aggregate.get("pages_total")
    pages_with_hyp = aggregate.get("pages_with_hyp")
    if (
        not isinstance(measured_cer, (int, float))
        or isinstance(measured_cer, bool)
        or not math.isfinite(float(measured_cer))
        or not 0.0 <= float(measured_cer) <= MAX_CORRECTNESS_CER
        or not math.isclose(claimed_cer, float(measured_cer), abs_tol=5e-6)
        or pages_total != len(pages)
        or pages_with_hyp != pages_total
        or not isinstance(pages_total, int)
        or isinstance(pages_total, bool)
        or pages_total <= 0
        or any(
            not isinstance(page, dict) or page.get("status") != "OK" for page in pages
        )
    ):
        raise RowError(
            "correctness receipt does not prove an in-budget complete CER run"
        )
    digest = sha256_file(receipt_path)
    return {
        "path": receipt_path,
        "sha256": digest,
        "cer_norm": float(measured_cer),
        "canonical": (
            f"receipt=correctness_receipt.json sha256={digest} "
            f"metric=cer_norm value={float(measured_cer):.6f} "
            f"max={MAX_CORRECTNESS_CER:.6f} result=pass"
        ),
    }


def sanitize_cell(value: str) -> str:
    """Markdown table cells cannot carry '|' or newlines (the checker splits
    on '|'); substitute rather than silently breaking the row shape."""
    return " ".join(str(value).replace("|", "¦").split())


def stage_record(doc: dict, stage: str) -> dict | None:
    records = doc.get("stages")
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and record.get("stage") == stage:
            return record
    return None


def ledger_stage_name(stage: str) -> str:
    return stage.replace("_", "-")


def validate_inputs(
    focr: dict,
    ref: dict,
    stage: str,
    fixture_hash: str,
    *,
    allow_synthetic: bool,
) -> tuple[dict, dict]:
    for name, doc in (("focr", focr), ("reference", ref)):
        if doc.get("synthetic") is not False and not allow_synthetic:
            raise RowError(
                f"{name} input is synthetic or unstamped — it can never land"
            )
    if focr.get("source") != "focr" or ref.get("source") != "reference":
        raise RowError("focr/reference inputs swapped or mislabeled (`source` field)")

    # Same-input pairing (fresh-eyes fix): both measurement docs carry `page`;
    # a ratio between two DIFFERENT pages is meaningless and previously landed
    # silently if the operator passed mismatched files.
    fpage = os.path.basename(str(focr.get("page") or ""))
    rpage = os.path.basename(str(ref.get("page") or ""))
    if not fpage or not rpage:
        raise RowError("focr/reference measurement docs must both carry `page`")
    if fpage != rpage:
        raise RowError(
            f"page pairing broken: focr measured {fpage!r}, reference {rpage!r}"
        )
    fixture_match = re.search(
        r"(?:^|;\s*)page=(?P<page>[^\s;]+)\s+sha256=(?P<sha>[0-9a-f]{64})(?:;|\s|$)",
        fixture_hash,
    )
    if fixture_match is None:
        raise RowError(
            "fixture_hash must bind page=<basename> sha256=<64 lowercase hex>"
        )
    if fixture_match.group("page") != fpage:
        raise RowError("fixture_hash page does not match both measurement inputs")
    if any(doc.get("page_sha256") != fixture_match.group("sha") for doc in (focr, ref)):
        raise RowError("focr/reference page_sha256 does not match fixture_hash")

    frec = stage_record(focr, stage)
    rrec = stage_record(ref, stage)
    if frec is None:
        raise RowError(f"focr measurements carry no stage {stage!r}")
    if rrec is None:
        raise RowError(f"reference measurements carry no stage {stage!r}")

    for side, rec in (("focr", frec), ("reference", rrec)):
        expected_source = "focr" if side == "focr" else "reference"
        if (
            rec.get("schema") != "focr-gauntlet-stage/v1"
            or rec.get("source") != expected_source
            or (rec.get("synthetic") is not False and not allow_synthetic)
            or rec.get("unit") != "ms"
            or rec.get("ledger_stage") is not True
        ):
            raise RowError(f"{side} {stage}: record is not ledger-eligible")
        samples = rec.get("samples_ms")
        n = rec.get("n")
        if (
            not isinstance(samples, list)
            or len(samples) < 2
            or not isinstance(n, int)
            or isinstance(n, bool)
            or n != len(samples)
        ):
            raise RowError(f"{side} {stage}: best-of-N samples are mandatory")
        try:
            numeric_samples = [float(sample) for sample in samples]
            best_ms = float(rec["best_ms"])
            cv = float(rec["cv_pct"])
        except (KeyError, TypeError, ValueError) as error:
            raise RowError(f"{side} {stage}: invalid timing evidence") from error
        recomputed_cv = round(
            statistics.stdev(numeric_samples)
            / statistics.fmean(numeric_samples)
            * 100.0,
            3,
        )
        if (
            not all(
                math.isfinite(sample) and sample > 0.0 for sample in numeric_samples
            )
            or not math.isclose(best_ms, min(numeric_samples), rel_tol=1e-9)
            or not math.isfinite(cv)
            or not 0.0 <= cv <= MAX_CV_PCT
            or not math.isclose(cv, recomputed_cv, abs_tol=0.001)
        ):
            raise RowError(
                f"{side} {stage}: timings/CV are invalid or not sample-derived"
            )
        if rec.get("warmup_discarded", 0) < 1:
            raise RowError(f"{side} {stage}: warmup was not discarded")
        if not rec.get("precision"):
            raise RowError(f"{side} {stage}: missing precision annotation")
        if not rec.get("backend") or not rec.get("allocator"):
            raise RowError(f"{side} {stage}: missing backend/allocator identity")

    if frec.get("threads") != rrec.get("threads"):
        raise RowError(
            f"thread parity broken: focr={frec.get('threads')} ref={rrec.get('threads')}"
        )
    proof = rrec.get("thread_proof") or {}
    if proof.get("torch_num_threads") != rrec.get("threads"):
        raise RowError(
            "reference record lacks a matching torch thread proof — rejected"
        )
    if frec.get("allocator") != rrec.get("allocator"):
        raise RowError(
            f"allocator mismatch: focr={frec.get('allocator')} ref={rrec.get('allocator')}"
        )
    if focr.get("stdout_identical_across_runs") is not True:
        raise RowError(
            "focr stdout differed across runs — nondeterministic evidence rejected"
        )
    if frec.get("tokens") is not None and frec.get("tokens_consistent") is False:
        raise RowError(f"focr {stage}: token count drifted across runs — rejected")
    if stage == "decode_per_token" and frec.get("tokens_consistent") is not True:
        raise RowError("focr decode evidence lacks a positive token-consistency proof")
    return frec, rrec


def roofline_floor(roofline: dict, stage: str, *, allow_synthetic: bool) -> dict:
    if roofline.get("synthetic") and not allow_synthetic:
        raise RowError("roofline was computed over synthetic measurements — rejected")
    for floor in roofline.get("floors", []):
        if floor.get("stage") == stage:
            if not floor.get("floor_ms") or not floor.get("floor_kind"):
                raise RowError(f"roofline floor for {stage!r} is incomplete")
            return floor
    raise RowError(f"roofline carries no floor for stage {stage!r}")


def build_row(
    *,
    focr: dict,
    ref: dict,
    roofline: dict,
    stage: str,
    claim_id: str,
    evidence_id: str,
    fixture_hash: str,
    arch_features: str,
    correctness_proof: str,
    model_commit: str,
    notes: str,
    allow_synthetic: bool = False,
) -> dict:
    correctness = validate_correctness_proof(correctness_proof)
    frec, rrec = validate_inputs(
        focr,
        ref,
        stage,
        fixture_hash,
        allow_synthetic=allow_synthetic,
    )
    floor = roofline_floor(roofline, stage, allow_synthetic=allow_synthetic)

    focr_ms = float(frec["best_ms"])
    ref_ms = float(rrec["best_ms"])
    floor_ms = float(floor["floor_ms"])
    kill = focr.get("focr_env") or {}
    kill_cell = " ".join(f"{k}={v}" for k, v in sorted(kill.items()))
    if not kill_cell:
        raise RowError(
            "focr run meta carries no FOCR_* env evidence (kill-switch state)"
        )

    command_cell = (
        f"focr: {' '.join(map(str, focr.get('command') or []))} "
        f"[threads={frec['threads']} FOCR_TIMING=1]; "
        f"ref: {' '.join(map(str, ref.get('command') or []))} "
        f"[torch=={ref.get('torch_version', '?')} transformers=={ref.get('transformers_version', '?')}]"
    )
    date = (focr.get("created_utc") or "")[:10]
    if not date:
        raise RowError("focr measurement file has no created_utc date")

    cells = {
        "date": date,
        "claim_id": claim_id,
        "evidence_id": evidence_id,
        "model_commit": model_commit,
        "fixture_hash": fixture_hash,
        "arch/cpu_features": arch_features,
        "stage": ledger_stage_name(stage),
        "reference_backend": rrec["backend"],
        "focr_ms": f"{focr_ms:.3f}",
        "ref_ms": f"{ref_ms:.3f}",
        "ratio": f"{ref_ms / focr_ms:.3f}",
        "floor_kind": floor["floor_kind"],
        "floor_ms": f"{floor_ms:.3f}",
        "dist_above_floor": f"{focr_ms / floor_ms:.2f}",
        "precision (focr vs ref)": f"{frec['precision']} vs {rrec['backend']}-{rrec['precision']}",
        "threads (focr=ref N)": f"focr=ref={frec['threads']}",
        "allocator": frec["allocator"],
        "command/env": command_cell,
        "fallback/kill-switch state": kill_cell,
        "correctness_proof": correctness["canonical"],
        "notes": notes
        or f"best-of-{frec['n']} (cv {frec['cv_pct']}% / ref {rrec['cv_pct']}%)",
    }
    return {name: sanitize_cell(cells[name]) for name in COLUMNS}


def row_markdown(row: dict) -> str:
    return "| " + " | ".join(row[name] for name in COLUMNS) + " |"


# ── evidence bundle ──────────────────────────────────────────────────────────


def write_bundle(
    evidence_dir: str,
    *,
    focr_path: str,
    ref_path: str,
    roofline_path: str,
    correctness_receipt_path: str,
    focr: dict,
    rows: list[dict],
    git_head: str,
) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", git_head) is None:
        raise RowError("write_bundle requires a canonical 40-hex git_head")
    os.makedirs(evidence_dir, exist_ok=True)
    inputs = {}
    for key, src, bundle_name in (
        ("focr_stages", focr_path, "focr_stages.json"),
        ("ref_stages", ref_path, "ref_stages.json"),
        ("roofline", roofline_path, "roofline.json"),
        (
            "correctness_receipt",
            correctness_receipt_path,
            "correctness_receipt.json",
        ),
    ):
        destination = os.path.join(evidence_dir, bundle_name)
        shutil.copy2(src, destination)
        inputs[key] = {
            "path": src,
            "bundle_path": bundle_name,
            "sha256": sha256_file(destination),
        }

    run_dir = focr.get("run_dir")
    if not run_dir or not os.path.isdir(run_dir):
        raise RowError(
            f"focr raw run dir {run_dir!r} is missing — a row without its raw logs "
            "is incomplete and may not be cited"
        )
    raw_dst = os.path.join(evidence_dir, "raw")
    os.makedirs(raw_dst, exist_ok=True)
    for name in sorted(os.listdir(run_dir)):
        src = os.path.join(run_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(raw_dst, name))

    with open(os.path.join(evidence_dir, "row.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": "focr-gauntlet-row/v2",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_head": git_head,
                "inputs": inputs,
                "rows": rows,
            },
            f,
            indent=2,
        )
        f.write("\n")
    with open(
        os.path.join(evidence_dir, "PERF_LEDGER_ROW.md"), "w", encoding="utf-8"
    ) as f:
        for row in rows:
            f.write(row_markdown(row) + "\n")

    # SHA256SUMS last, over every file in the bundle (shasum -c compatible).
    entries = []
    for dirpath, _dirnames, filenames in os.walk(evidence_dir):
        for name in sorted(filenames):
            if name == "SHA256SUMS":
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, evidence_dir)
            entries.append(f"{sha256_file(path)}  {rel}\n")
    manifest = os.path.join(evidence_dir, "SHA256SUMS")
    with open(manifest, "w", encoding="utf-8") as f:
        f.writelines(sorted(entries, key=lambda line: line.split("  ", 1)[1]))
    return manifest


# ── ledger insertion + validation against the REAL checker ──────────────────


def insert_rows(ledger_text: str, rows_md: list[str]) -> str:
    lines = ledger_text.splitlines(keepends=True)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("| date | claim_id | evidence_id |"):
            header_idx = i
            break
    if (
        header_idx is None
        or header_idx + 1 >= len(lines)
        or "---" not in lines[header_idx + 1]
    ):
        raise RowError("PERF_LEDGER.md ratio-table header not found; cannot insert row")
    insert_at = header_idx + 2
    return "".join(
        lines[:insert_at] + [md + "\n" for md in rows_md] + lines[insert_at:]
    )


def shadow_check(
    rows_md: list[str], evidence_dir: str, evidence_id: str
) -> tuple[bool, str]:
    """Run the real scripts/check_ledgers.py in a shadow repo copy with the
    candidate row inserted — validation without touching the live ledger."""
    with tempfile.TemporaryDirectory(prefix="focr-gauntlet-row-") as shadow:
        os.makedirs(os.path.join(shadow, "scripts"))
        shutil.copy2(CHECKER, os.path.join(shadow, "scripts", "check_ledgers.py"))
        os.makedirs(os.path.join(shadow, "docs"))
        for name in ("NEGATIVE_EVIDENCE.md", "DISCREPANCIES.md"):
            shutil.copy2(
                os.path.join(ROOT, "docs", name), os.path.join(shadow, "docs", name)
            )
        with open(LEDGER, encoding="utf-8") as f:
            ledger_text = f.read()
        with open(
            os.path.join(shadow, "docs", "PERF_LEDGER.md"), "w", encoding="utf-8"
        ) as f:
            f.write(insert_rows(ledger_text, rows_md))
        # DISCREPANCIES kill-switch validation greps src/**/*.rs.
        for dirpath, _dirnames, filenames in os.walk(os.path.join(ROOT, "src")):
            for name in filenames:
                if not name.endswith(".rs"):
                    continue
                src = os.path.join(dirpath, name)
                rel = os.path.relpath(src, ROOT)
                dst = os.path.join(shadow, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        # Pre-existing ledger entries (NEGATIVE_EVIDENCE etc.) resolve against
        # the repo's evidence dirs; mirror them, then overlay the candidate.
        repo_perf = os.path.join(ROOT, "artifacts", "perf")
        if os.path.isdir(repo_perf):
            shutil.copytree(repo_perf, os.path.join(shadow, "artifacts", "perf"))
        shadow_evidence = os.path.join(shadow, evidence_id)
        if os.path.isdir(shadow_evidence):
            shutil.rmtree(shadow_evidence)  # shadow-only; the repo copy is untouched
        shutil.copytree(evidence_dir, shadow_evidence)
        proc = subprocess.run(
            [sys.executable, os.path.join(shadow, "scripts", "check_ledgers.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0, proc.stdout + proc.stderr


def apply_rows(rows_md: list[str]) -> tuple[bool, str]:
    with open(LEDGER, encoding="utf-8") as f:
        original = f.read()
    updated = insert_rows(original, rows_md)
    with open(LEDGER, "w", encoding="utf-8") as f:
        f.write(updated)
    proc = subprocess.run(
        [sys.executable, CHECKER], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        with open(LEDGER, "w", encoding="utf-8") as f:
            f.write(original)  # revert: a failing row never lands
        return False, proc.stdout + proc.stderr
    return True, proc.stdout


def run(args: argparse.Namespace) -> int:
    git_head = current_git_head()
    focr = load_json(args.focr_stages, "focr-gauntlet-stages/v1")
    ref = load_json(args.ref_stages, "focr-gauntlet-stages/v1")
    roofline = load_json(args.roofline, "focr-gauntlet-roofline/v1")

    stages = args.stage or [
        s
        for s in LEDGER_STAGES
        if stage_record(focr, s)
        and stage_record(ref, s)
        and any(fl.get("stage") == s for fl in roofline.get("floors", []))
    ]
    if not stages:
        raise RowError("no stage is present in all three inputs — nothing to merge")

    claim = args.claim_id.strip()
    evidence_dir = args.evidence_dir or os.path.join(
        ROOT, "artifacts", "perf", "bd-re8.17", claim
    )
    evidence_id = os.path.relpath(os.path.abspath(evidence_dir), ROOT)
    if not evidence_id.startswith(os.path.join("artifacts", "perf")):
        raise RowError(
            f"evidence dir must live under artifacts/perf/, got {evidence_id!r}"
        )

    rows = [
        build_row(
            focr=focr,
            ref=ref,
            roofline=roofline,
            stage=stage,
            claim_id=claim,
            evidence_id=evidence_id,
            fixture_hash=args.fixture_hash,
            arch_features=args.arch_features,
            correctness_proof=args.correctness_proof,
            model_commit=args.model_commit,
            notes=args.notes,
        )
        for stage in stages
    ]
    rows_md = [row_markdown(row) for row in rows]

    manifest = write_bundle(
        evidence_dir,
        focr_path=args.focr_stages,
        ref_path=args.ref_stages,
        roofline_path=args.roofline,
        correctness_receipt_path=validate_correctness_proof(args.correctness_proof)[
            "path"
        ],
        focr=focr,
        rows=rows,
        git_head=git_head,
    )

    ok, output = shadow_check(rows_md, evidence_dir, evidence_id)
    if not ok:
        print(output, file=sys.stderr)
        print(
            "ERROR: candidate row fails scripts/check_ledgers.py — not emitted",
            file=sys.stderr,
        )
        return 1

    if args.apply:
        applied, apply_output = apply_rows(rows_md)
        if not applied:
            print(apply_output, file=sys.stderr)
            print(
                "ERROR: in-place apply failed validation and was reverted",
                file=sys.stderr,
            )
            return 1

    for md in rows_md:
        print(md)
    print(
        json.dumps(
            {
                "event": "perf_ledger_row_ready",
                "stages": stages,
                "evidence_id": evidence_id,
                "manifest": manifest,
                "check": "pass",
                "applied": bool(args.apply),
            }
        ),
        file=sys.stderr,
    )
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def _synthetic_inputs(tmp: str) -> tuple[str, str, str, dict]:
    """Plumbing-only synthetic inputs (stamped synthetic; the normal path
    refuses them — the self-test exercises the internal allow_synthetic path)."""
    raw = os.path.join(tmp, "raw")
    os.makedirs(raw)
    for i in (1, 2, 3):
        with open(os.path.join(raw, f"run_{i:03d}.stderr"), "w", encoding="utf-8") as f:
            f.write("[focr-timing] decode_i8 6.00s (600 tokens, 0.010s/tok)\n")
        with open(os.path.join(raw, f"run_{i:03d}.stdout"), "w", encoding="utf-8") as f:
            f.write("identical output\n")

    def rec(stage: str, samples: list[float], source: str, **extra: object) -> dict:
        return {
            "schema": "focr-gauntlet-stage/v1",
            "source": source,
            "stage": stage,
            "ledger_stage": True,
            "unit": "ms",
            "samples_ms": samples,
            "best_ms": min(samples),
            "p50_ms": statistics.median(samples),
            "mean_ms": statistics.fmean(samples),
            "cv_pct": round(
                statistics.stdev(samples) / statistics.fmean(samples) * 100, 3
            ),
            "n": len(samples),
            "warmup_discarded": 1,
            "threads": 8,
            "precision": "focr-int8" if source == "focr" else "bf16",
            "backend": "focr" if source == "focr" else "hf",
            "allocator": "system",
            "synthetic": True,
            **extra,
        }

    focr = {
        "schema": "focr-gauntlet-stages/v1",
        "source": "focr",
        "created_utc": "2026-07-01T00:00:00Z",
        "run_dir": raw,
        "page": "page.png",
        "page_sha256": "f" * 64,
        "command": ["focr", "ocr", "page.png"],
        "focr_env": {"FOCR_TIMING": "1", "FOCR_THREADS": "8"},
        "threads": 8,
        "stdout_identical_across_runs": True,
        "stages": [
            rec(
                "decode_per_token",
                [10.0, 10.2, 10.1],
                "focr",
                tokens=600,
                tokens_consistent=True,
            )
        ],
        "synthetic": True,
    }
    ref = {
        "schema": "focr-gauntlet-stages/v1",
        "source": "reference",
        "page": "page.png",
        "page_sha256": "f" * 64,
        "command": ["python3", "gauntlet_reference.py"],
        "torch_version": "2.10.0",
        "transformers_version": "4.57.1",
        "stages": [
            rec(
                "decode_per_token",
                [25.0, 25.5, 25.2],
                "reference",
                thread_proof={"budget": 8, "torch_num_threads": 8},
            )
        ],
        "synthetic": True,
    }
    roofline = {
        "schema": "focr-gauntlet-roofline/v1",
        "arch": "unlimited-ocr",
        "precision": "int8",
        "machine_profile": {"name": "m4", "dram_gb_s": 120.0},
        "floors": [
            {"stage": "decode_per_token", "floor_kind": "memory", "floor_ms": 4.953}
        ],
        "synthetic": True,
    }
    paths = []
    for name, doc in (
        ("focr.json", focr),
        ("ref.json", ref),
        ("roofline.json", roofline),
    ):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        paths.append(path)
    return paths[0], paths[1], paths[2], focr


def _self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, **fields: object) -> None:
        print(json.dumps({"check": name, "result": "pass" if ok else "fail", **fields}))
        if not ok:
            failures.append(name)

    with tempfile.TemporaryDirectory(prefix="focr-gauntlet-row-selftest-") as tmp:
        focr_path, ref_path, roofline_path, focr = _synthetic_inputs(tmp)
        ref = load_json(ref_path)
        roofline = load_json(roofline_path)
        correctness_receipt_path = os.path.join(tmp, "cer.json")
        with open(correctness_receipt_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "aggregate": {
                        "pages_total": 1,
                        "pages_with_hyp": 1,
                        "cer_norm": 0.0,
                    },
                    "pages": [{"page": "page.md", "status": "OK", "cer_norm": 0.0}],
                },
                handle,
            )
        kwargs = dict(
            focr=focr,
            ref=ref,
            roofline=roofline,
            stage="decode_per_token",
            claim_id="selftest-claim",
            evidence_id="artifacts/perf/bd-re8.17/selftest",
            fixture_hash="page=page.png sha256=" + "f" * 64,
            arch_features="aarch64+neon+dotprod",
            correctness_proof=(
                f"CER_norm=0.000000 pinned reference comparison "
                f"({correctness_receipt_path})"
            ),
            model_commit=MODEL_COMMIT,
            notes="",
        )

        # Synthetic inputs are refused on the normal path…
        try:
            build_row(**kwargs)
            check("refuses-synthetic", False)
        except RowError:
            check("refuses-synthetic", True)

        # …and merge correctly on the explicit self-test path.
        row = build_row(**kwargs, allow_synthetic=True)
        check("row-has-all-columns", list(row) == COLUMNS)
        check("row-ratio", row["ratio"] == "2.500")
        check("row-dist-above-floor", row["dist_above_floor"] == "2.02")
        check("row-stage-hyphenated", row["stage"] == "decode-per-token")
        check("row-threads-cell", row["threads (focr=ref N)"] == "focr=ref=8")
        md = row_markdown(row)
        check("row-cell-count", md.count("|") == len(COLUMNS) + 1)

        # Refusals: noisy cv, thread mismatch, missing thread proof,
        # nondeterministic stdout, token drift.
        def mutated(patch) -> dict:
            doc = json.loads(json.dumps(ref))
            patch(doc["stages"][0])
            return doc

        for name, patch in (
            ("refuses-noisy-cv", lambda r: r.update(cv_pct=9.0)),
            ("refuses-thread-mismatch", lambda r: r.update(threads=16)),
            ("refuses-missing-thread-proof", lambda r: r.pop("thread_proof")),
            ("refuses-missing-precision", lambda r: r.update(precision="")),
        ):
            try:
                build_row(**{**kwargs, "ref": mutated(patch)}, allow_synthetic=True)
                check(name, False)
            except RowError:
                check(name, True)

        focr_bad = json.loads(json.dumps(focr))
        focr_bad["stdout_identical_across_runs"] = False
        try:
            build_row(**{**kwargs, "focr": focr_bad}, allow_synthetic=True)
            check("refuses-nondeterministic-stdout", False)
        except RowError:
            check("refuses-nondeterministic-stdout", True)

        # Bundle: manifest hashes verify against the files on disk.
        evidence_dir = os.path.join(tmp, "artifacts", "perf", "bd-re8.17", "selftest")
        manifest = write_bundle(
            evidence_dir,
            focr_path=focr_path,
            ref_path=ref_path,
            roofline_path=roofline_path,
            correctness_receipt_path=correctness_receipt_path,
            focr=focr,
            rows=[row],
            git_head="0123456789abcdef0123456789abcdef01234567",
        )
        ok = True
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                digest, rel = line.rstrip("\n").split("  ", 1)
                ok = ok and sha256_file(os.path.join(evidence_dir, rel)) == digest
        check("bundle-manifest-verifies", ok)
        check(
            "bundle-has-raw-logs",
            os.path.isfile(os.path.join(evidence_dir, "raw", "run_001.stderr")),
        )
        with open(os.path.join(evidence_dir, "row.json"), encoding="utf-8") as f:
            bundled_row = json.load(f)
        check(
            "bundle-row-schema-v2", bundled_row.get("schema") == "focr-gauntlet-row/v2"
        )
        check(
            "bundle-records-git-head",
            bundled_row.get("git_head") == "0123456789abcdef0123456789abcdef01234567",
        )
        check(
            "bundle-uses-canonical-input-names",
            all(
                os.path.isfile(os.path.join(evidence_dir, name))
                for name in (
                    "focr_stages.json",
                    "ref_stages.json",
                    "roofline.json",
                    "correctness_receipt.json",
                )
            ),
        )

        try:
            build_row(
                **{**kwargs, "correctness_proof": "FAILED: token mismatch"},
                allow_synthetic=True,
            )
            check("refuses-failing-correctness-proof", False)
        except RowError:
            check("refuses-failing-correctness-proof", True)

        for exploit in (
            "outputs differ from reference; CER=0.91",
            "token_exact=false; parity=false",
            "wrong tokens on 99 percent of pages",
            "no failures",
        ):
            try:
                build_row(
                    **{**kwargs, "correctness_proof": exploit},
                    allow_synthetic=True,
                )
                check("refuses-unstructured-correctness-claim", False, claim=exploit)
            except RowError:
                check("refuses-unstructured-correctness-claim", True, claim=exploit)

        # The REAL check_ledgers.py accepts the candidate row in a shadow repo…
        passed, output = shadow_check(
            [row_markdown(row)], evidence_dir, "artifacts/perf/bd-re8.17/selftest"
        )
        check("shadow-check-passes", passed, detail=None if passed else output[-400:])

        # …and rejects one whose correctness_proof cell is empty.
        broken = dict(row)
        broken["correctness_proof"] = ""
        broken_md = row_markdown(broken)
        rejected, _output = shadow_check(
            [broken_md], evidence_dir, "artifacts/perf/bd-re8.17/selftest"
        )
        check("shadow-check-rejects-empty-proof", not rejected)

        # …and rejects a row whose evidence dir has no SHA256SUMS manifest.
        os.remove(manifest)
        rejected, _output = shadow_check(
            [row_markdown(row)], evidence_dir, "artifacts/perf/bd-re8.17/selftest"
        )
        check("shadow-check-rejects-missing-manifest", not rejected)

    if failures:
        print(f"SELF-TEST FAILED: {failures}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "gauntlet-row-self-test", "result": "pass"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--focr-stages")
    parser.add_argument("--ref-stages")
    parser.add_argument("--roofline")
    parser.add_argument("--stage", action="append", choices=LEDGER_STAGES, default=None)
    parser.add_argument("--claim-id")
    parser.add_argument("--fixture-hash")
    parser.add_argument("--arch-features")
    parser.add_argument("--correctness-proof")
    parser.add_argument("--model-commit", default=MODEL_COMMIT)
    parser.add_argument("--notes", default="")
    parser.add_argument("--evidence-dir", default=None)
    parser.add_argument(
        "--apply", action="store_true", help="insert into docs/PERF_LEDGER.md"
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    required = (
        "focr_stages",
        "ref_stages",
        "roofline",
        "claim_id",
        "fixture_hash",
        "arch_features",
        "correctness_proof",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(
            f"missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}"
        )
    try:
        return run(args)
    except RowError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
