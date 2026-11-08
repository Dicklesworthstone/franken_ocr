#!/usr/bin/env python3
"""Prove the safety nets still catch what they exist to catch.

A net that never meets a real fault is faith. Each probe here injects one fault
and requires the net to fire, so a refactor that quietly neuters a net turns this
lane red. It runs deliberately failing tests, so it is never part of
`scripts/check.sh`, which only runs `--self-test`. Run it before a release, or
from any scheduler.

Green means every net caught its fault. Red means a net missed one: that is a
broken safety net, not an ordinary test failure.

An expected failure only counts with the net's own marker in the output. Without
that rule a compile error, a renamed test or a missing tool would all read as a
catch.

Usage:
  python3 scripts/meta_tests.py                      # every probe
  python3 scripts/meta_tests.py --deep               # plus the long property and fuzz budgets
  python3 scripts/meta_tests.py --only watchdog
  python3 scripts/meta_tests.py --disarm watchdog    # take one fault out: the lane must go red
  python3 scripts/meta_tests.py --self-test          # the verdict logic alone, no cargo

The scorecard is NDJSON in the test-log schema (`--scorecard`, default
`target/meta_tests/scorecard.ndjson`), and the lane fails if the validator rejects it.

Environment:
  FOCR_META_CARGO_TEST  how to invoke `cargo test` (default `cargo test --locked`)
  FOCR_META_FUZZ        how to invoke `cargo fuzz run` (default `cargo +nightly fuzz run`)
  FOCR_META_FUZZ_RUNS   runs per fuzz target under --deep (default 200000)
  PROPTEST_CASES        cases under --deep (default 4096)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = ROOT / "target" / "meta_tests" / "scorecard.ndjson"
BACKSTOP_S = 3600
TEST_NAME = "meta_tests"


@dataclass
class Probe:
    """One fault and how its net must answer.

    `expected` says how the net answers. `rejects` means it exits non-zero with
    `marker` in its output. `holds` means the net tests itself and passes, with
    `marker` proving that test really ran.
    """

    name: str
    injected_fault: str
    expected: str
    argv: list[str]
    marker: str
    env: dict[str, str] = field(default_factory=dict)
    rejects: bool = True
    disarmed_argv: list[str] | None = None
    disarmed_env: dict[str, str] | None = None


def caught(probe_rejects: bool, code: int | None, output: str, marker: str) -> bool:
    """Did the net answer as designed? `code` is None when the backstop killed the run."""
    if code is None or marker not in output:
        return False
    return code != 0 if probe_rejects else code == 0


def self_test() -> int:
    cases = [
        ("the net fires", True, 101, "MARK", True),
        ("a failure without the marker, such as a compile error", True, 101, "error[E0425]", False),
        ("the marker with exit 0: the net spoke but nothing failed", True, 0, "MARK", False),
        ("a clean pass: the net stayed quiet", True, 0, "ok", False),
        ("the net itself hung", True, None, "MARK", False),
        ("a self-testing net passes and ran", False, 0, "MARK", True),
        ("a self-testing net fails", False, 101, "MARK", False),
        ("a self-testing net matched nothing", False, 0, "0 passed", False),
    ]
    failed = 0
    for name, rejects, code, output, want in cases:
        got = caught(rejects, code, output, "MARK")
        print(json.dumps({"check": "meta-tests-verdict", "case": name, "result": "pass" if got == want else "fail"}))
        failed += got != want
    return 1 if failed else 0


def stage(name: str, best_ms: float) -> dict:
    return {
        "stage": name,
        "ledger_stage": True,
        "best_ms": best_ms,
        "cv_pct": 1.0,
        "threads": 8,
        "allocator": "system",
        "precision": "focr-int8",
    }


def write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def guardrail_probes(tmp: Path) -> list[Probe]:
    base = "decode_per_token"
    baseline = write_json(
        tmp / "baseline.json",
        {"schema": "focr-bench-baseline/v1", "regime": "dense_page", "stages": {base: stage(base, 100.0)}},
    )
    slow = write_json(tmp / "slow.json", {"stages": [stage(base, 150.0)]})
    level = write_json(tmp / "level.json", {"stages": [stage(base, 100.0)]})
    command = [sys.executable, "-I", "scripts/bench_guardrail.py", "--baseline", baseline, "--regime", "dense_page"]
    red_receipt = write_json(tmp / "red_receipt.json", {"all_green": False, "receipt": "meta-tests"})
    green_receipt = write_json(tmp / "green_receipt.json", {"all_green": True, "receipt": "meta-tests"})
    return [
        Probe(
            name="bench_guardrail",
            injected_fault="a stage 50% slower than its frozen baseline",
            expected="fail",
            argv=[*command, "--stages", slow, "--parity-receipt", ""],
            marker="REGRESSION",
            disarmed_argv=[*command, "--stages", level, "--parity-receipt", ""],
        ),
        Probe(
            name="parity_receipt",
            injected_fault="a parity receipt that is not all-green",
            expected="fail",
            argv=[*command, "--stages", level, "--parity-receipt", red_receipt],
            marker="NOT all-green",
            disarmed_argv=[*command, "--stages", level, "--parity-receipt", green_receipt],
        ),
    ]


def log_probe(tmp: Path) -> Probe:
    record = {
        "schema_version": 1,
        "ts": 1,
        "test": TEST_NAME,
        "case": "log_validator",
        "run_seq": 0,
        "event": "assert",
        "assertion": "a well-formed line",
        "pass": True,
        "result": "pass",
    }
    malformed = dict(record)
    del malformed["run_seq"]
    bad = tmp / "malformed.ndjson"
    bad.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    good = tmp / "wellformed.ndjson"
    good.write_text(json.dumps(record) + "\n", encoding="utf-8")
    command = [sys.executable, "-I", "scripts/check_test_logs.py"]
    return Probe(
        name="log_validator",
        injected_fault="a log line with a required field missing",
        expected="reject",
        argv=[*command, str(bad)],
        marker="run_seq",
        disarmed_argv=[*command, str(good)],
    )


def cargo_probes(tmp: Path, cargo_test: list[str]) -> list[Probe]:
    broken = tmp / "broken.focrq"
    broken.write_bytes(b"not a model artifact\n" * 64)
    gate = "recognize_real_model_when_present_else_skip_with_success"
    determinism = "parity_harness::tests::determinism_gate_passes_stable_and_fails_injected"
    return [
        Probe(
            name="watchdog",
            injected_fault="a lock held across a rayon scope whose task wants it",
            expected="timeout",
            argv=[
                *cargo_test,
                "--test",
                "many_pages_without_deadlock",
                "--",
                "--ignored",
                "--exact",
                "watchdog_catches_a_lock_held_across_rayon",
            ],
            marker="DEADLOCK SUSPECTED",
            disarmed_env={"FOCR_META_DISARM": "1"},
        ),
        Probe(
            name="model_gate",
            injected_fault="an armed model gate pointed at a broken artifact",
            expected="fail",
            argv=[*cargo_test, "--test", "e2e_recognize", "--", "--exact", gate],
            env={"FOCR_MODEL_PATH": str(broken), "FOCR_E2E_REAL_MODEL": "1"},
            marker="is a real FAIL, not a skip",
            disarmed_env={"FOCR_MODEL_PATH": "/nonexistent/meta-tests.focrq", "FOCR_E2E_REAL_MODEL": ""},
        ),
        Probe(
            name="determinism_gate",
            injected_fault="a nondeterministic function, inside the gate's own test",
            expected="fail",
            argv=[*cargo_test, "--test", "uocr2_memory", "--", "--exact", determinism],
            marker="1 passed",
            rejects=False,
            disarmed_argv=[*cargo_test, "--test", "uocr2_memory", "--", "--exact", "no_such_test"],
        ),
    ]


def deep_probes(tmp: Path, cargo_test: list[str]) -> list[Probe]:
    fuzz = shlex.split(os.environ.get("FOCR_META_FUZZ", "cargo +nightly fuzz run"))
    runs = os.environ.get("FOCR_META_FUZZ_RUNS", "200000")
    cases = os.environ.get("PROPTEST_CASES", "4096")
    probes = [
        Probe(
            name="property_budget",
            injected_fault="none: the long property budget",
            expected="pass",
            argv=[*cargo_test, "--test", "property_suite"],
            env={"PROPTEST_CASES": cases},
            marker="test result: ok",
            rejects=False,
        )
    ]
    manifest = tomllib.loads((ROOT / "fuzz" / "Cargo.toml").read_text(encoding="utf-8"))
    for target in sorted(binary["name"] for binary in manifest["bin"]):
        # libFuzzer writes new inputs to the first directory, so the committed
        # seeds go second and stay as they are.
        corpus = tmp / "corpus" / target
        corpus.mkdir(parents=True)
        seeds = ROOT / "fuzz" / "corpus" / target
        probes.append(
            Probe(
                name=f"fuzz_{target}",
                injected_fault=f"none: {runs} runs of {target}",
                expected="pass",
                argv=[*fuzz, target, str(corpus), *([str(seeds)] if seeds.is_dir() else []), "--", f"-runs={runs}"],
                # libFuzzer executes the whole corpus before it checks the limit,
                # so a corpus larger than `runs` finishes past it.
                marker=" runs in ",
                rejects=False,
            )
        )
    return probes


def run_probe(probe: Probe, disarmed: bool) -> tuple[int | None, str, int]:
    argv = (probe.disarmed_argv or probe.argv) if disarmed else probe.argv
    env = {**os.environ, **probe.env}
    if disarmed and probe.disarmed_env:
        env.update(probe.disarmed_env)
    started = time.monotonic()
    try:
        done = subprocess.run(  # ubs:ignore[python.taint.command] - test harness runs the probe binary named by its env by design; argv is not external input
            argv,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=BACKSTOP_S,
            check=False,
        )
        code: int | None = done.returncode
        output = done.stdout
    except subprocess.TimeoutExpired as hung:
        code = None
        raw = hung.stdout or b""
        output = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
    except OSError as err:
        code, output = 127, str(err)
    return code, output, int((time.monotonic() - started) * 1_000_000)


def score(seq: int, probe: Probe, disarmed: bool, code: int | None, output: str, elapsed_us: int) -> dict:
    ok = caught(probe.rejects, code, output, probe.marker)
    observed = "backstop timeout" if code is None else f"exit {code}, marker {'present' if probe.marker in output else 'absent'}"
    record = {
        "schema_version": 1,
        "ts": int(time.time() * 1000),
        "test": TEST_NAME,
        "case": probe.name,
        "run_seq": seq,
        "event": "assert",
        "assertion": "the net caught the injected fault",
        "pass": ok,
        "result": "pass" if ok else "fail",
        "meta_test": probe.name,
        "injected_fault": "disarmed: no fault" if disarmed else probe.injected_fault,
        "expected": probe.expected,
        "observed": observed,
        "disarmed": disarmed,
        "elapsed_us": elapsed_us,
    }
    if not ok:
        tail = output[-1500:]
        record["diag"] = {"error_kind": "safety_net_missed", "focr_exit_code": code if code is not None else -1, "message": tail}
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="check the verdict logic and exit")
    parser.add_argument("--deep", action="store_true", help="also run the long property and fuzz budgets")
    parser.add_argument("--only", action="append", default=[], help="run this probe; repeatable")
    parser.add_argument("--disarm", action="append", default=[], help="run this probe with its fault removed")
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    cargo_test = shlex.split(os.environ.get("FOCR_META_CARGO_TEST", "cargo test --locked"))
    with tempfile.TemporaryDirectory(prefix="focr-meta-") as name:
        tmp = Path(name)
        probes = [*guardrail_probes(tmp), log_probe(tmp), *cargo_probes(tmp, cargo_test)]
        if args.deep:
            probes.extend(deep_probes(tmp, cargo_test))
        names = {probe.name for probe in probes}
        unknown = sorted((set(args.only) | set(args.disarm)) - names)
        if unknown:
            print(f"meta_tests: unknown probe {unknown}; have {sorted(names)}", file=sys.stderr)
            return 2
        selected = [probe for probe in probes if not args.only or probe.name in args.only]

        started = time.monotonic()
        records = []
        for seq, probe in enumerate(selected):
            disarmed = probe.name in args.disarm
            code, output, elapsed_us = run_probe(probe, disarmed)
            record = score(seq, probe, disarmed, code, output, elapsed_us)
            records.append(record)
            print(json.dumps({key: record[key] for key in ("meta_test", "expected", "observed", "disarmed", "result")}))

    missed = [record["meta_test"] for record in records if not record["pass"]]
    summary = {
        "schema_version": 1,
        "ts": int(time.time() * 1000),
        "test": TEST_NAME,
        "case": "lane",
        "run_seq": len(records),
        "event": "result",
        "result": "fail" if missed else "pass",
        "elapsed_us": int((time.monotonic() - started) * 1_000_000),
        "probes": len(records),
        "missed": missed,
    }
    if missed:
        summary["diag"] = {
            "error_kind": "safety_net_missed",
            "focr_exit_code": 1,
            "message": f"nets that missed their fault: {missed}",
        }
    args.scorecard.parent.mkdir(parents=True, exist_ok=True)
    args.scorecard.write_text("".join(json.dumps(row) + "\n" for row in [*records, summary]), encoding="utf-8")

    valid = subprocess.run(
        [sys.executable, "-I", "scripts/check_test_logs.py", str(args.scorecard)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        check=False,
    )
    print(json.dumps({"check": "meta-tests", "result": summary["result"], "missed": missed, "scorecard_valid": valid.returncode == 0}))
    return 1 if missed or valid.returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
