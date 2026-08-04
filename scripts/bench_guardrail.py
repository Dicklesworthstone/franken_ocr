#!/usr/bin/env python3
"""bench_guardrail.py — the perf-regression gate vs a FROZEN baseline (bd-1a6h).

Wraps the stage records `scripts/gauntlet_focr.sh` already produces with all
the §9.3 fairness discipline (thread pins, warmup discard, best-of-N, cv%,
precision annotation) and compares them against a committed baseline:

  * a stage REGRESSES when `current_best / baseline_best > 1 + threshold`
    (default 10%, `--threshold-pct`) — the guardrail FAILS (exit 1);
  * a stage with `cv_pct > 5` on EITHER side is NOISE-INELIGIBLE: logged,
    never compared (a noisy bench must not produce a fake win/loss);
  * fairness posture (threads/allocator/precision) must MATCH the baseline
    row — a mismatch is ineligible, not a comparison;
  * the baseline only moves via `--ratchet` (an explicit, reviewed update —
    never automatic, never in CI);
  * missing baseline or stages file → SKIP-GREEN (exit 0, logged): a missing
    6.67 GB fixture must never red-flag CI.

One NDJSON row per stage:
  {"bench","regime","stage","best_us","cv_pct","baseline_us","ratio",
   "fairness":{...},"verdict":"ok|faster|REGRESSION|noise_ineligible|
   posture_mismatch|new_stage"}

The guardrail re-states the parity receipt alongside (§9.2): pass
`--parity-receipt tests/fixtures/ladder_scorecard/scorecard_armed.json` and
the run refuses to report perf at all if the receipt is not all-green — a
perf number without its correctness receipt is not a result.

Usage:
  python3 scripts/bench_guardrail.py --self-test
  python3 scripts/bench_guardrail.py --stages artifacts/.../focr_stages.json \
      [--regime dense_page] [--threshold-pct 10]
  python3 scripts/bench_guardrail.py --stages ... --ratchet   # reviewed update
"""

from __future__ import annotations

import argparse
import json
import os
import sys

BASELINE_DEFAULT = "benches/.bench-history/baseline.json"
CV_GATE_PCT = 5.0


def emit(row: dict) -> None:
    print(json.dumps(row, sort_keys=True))


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fairness_of(record: dict) -> dict:
    return {
        "threads": record.get("threads"),
        "allocator": record.get("allocator"),
        "precision": record.get("precision"),
    }


def stage_rows(stages_doc: dict) -> dict[str, dict]:
    """Ledger-stage records keyed by stage name."""
    return {
        s["stage"]: s
        for s in stages_doc.get("stages", [])
        if s.get("ledger_stage") and not s.get("synthetic", False)
    }


def compare(
    current: dict[str, dict],
    baseline: dict[str, dict],
    regime: str,
    threshold_pct: float,
) -> tuple[list[dict], bool]:
    rows: list[dict] = []
    regressed = False
    for stage, cur in sorted(current.items()):
        base = baseline.get(stage)
        row = {
            "bench": "gauntlet_focr",
            "regime": regime,
            "stage": stage,
            "best_us": round(cur["best_ms"] * 1000.0, 3),
            "cv_pct": cur.get("cv_pct"),
            "fairness": fairness_of(cur),
        }
        if base is None:
            row["verdict"] = "new_stage"
            rows.append(row)
            continue
        row["baseline_us"] = round(base["best_ms"] * 1000.0, 3)
        cur_cv = cur.get("cv_pct")
        base_cv = base.get("cv_pct")
        if (cur_cv is not None and cur_cv > CV_GATE_PCT) or (
            base_cv is not None and base_cv > CV_GATE_PCT
        ):
            row["verdict"] = "noise_ineligible"
            rows.append(row)
            continue
        if fairness_of(cur) != fairness_of(base):
            row["verdict"] = "posture_mismatch"
            rows.append(row)
            continue
        ratio = cur["best_ms"] / base["best_ms"]
        row["ratio"] = round(ratio, 4)
        if ratio > 1.0 + threshold_pct / 100.0:
            row["verdict"] = "REGRESSION"
            regressed = True
        elif ratio < 1.0:
            row["verdict"] = "faster"
        else:
            row["verdict"] = "ok"
        rows.append(row)
    return rows, regressed


def run(args) -> int:
    if not args.stages or not os.path.exists(args.stages):
        emit(
            {
                "check": "bench-guardrail",
                "result": "skip",
                "reason": f"stages file absent ({args.stages!r}) — model fixture not on this host",
            }
        )
        return 0

    # A perf number without its correctness receipt is not a result (§9.2).
    if args.parity_receipt:
        if not os.path.exists(args.parity_receipt):
            emit(
                {
                    "check": "bench-guardrail",
                    "result": "skip",
                    "reason": "parity receipt absent — refusing to report perf without correctness",
                }
            )
            return 0
        receipt = load_json(args.parity_receipt)
        if not receipt.get("all_green") or receipt.get("skipped_no_model"):
            emit(
                {
                    "check": "bench-guardrail",
                    "result": "error",
                    "reason": "parity receipt is NOT all-green — perf reporting refused (G1 > G2)",
                    "receipt": receipt.get("receipt"),
                }
            )
            return 1
        emit(
            {
                "check": "bench-guardrail-parity-receipt",
                "result": "pass",
                "receipt": receipt.get("receipt"),
            }
        )

    stages_doc = load_json(args.stages)
    current = stage_rows(stages_doc)
    if not current:
        emit({"check": "bench-guardrail", "result": "skip", "reason": "no ledger stages in file"})
        return 0

    if args.ratchet:
        os.makedirs(os.path.dirname(args.baseline), exist_ok=True)
        payload = {
            "schema": "focr-bench-baseline/v1",
            "regime": args.regime,
            "source": args.stages,
            "note": "frozen baseline — moves ONLY via --ratchet under review, never in CI",
            "stages": {k: v for k, v in current.items()},
        }
        with open(args.baseline, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        emit(
            {
                "check": "bench-guardrail-ratchet",
                "result": "pass",
                "baseline": args.baseline,
                "stages": sorted(current.keys()),
            }
        )
        return 0

    if not os.path.exists(args.baseline):
        emit(
            {
                "check": "bench-guardrail",
                "result": "skip",
                "reason": f"no frozen baseline at {args.baseline!r} — seed one with --ratchet (reviewed)",
            }
        )
        return 0
    baseline_doc = load_json(args.baseline)
    rows, regressed = compare(
        current, baseline_doc.get("stages", {}), args.regime, args.threshold_pct
    )
    for row in rows:
        emit(row)
    emit(
        {
            "check": "bench-guardrail",
            "result": "fail" if regressed else "pass",
            "regime": args.regime,
            "threshold_pct": args.threshold_pct,
            "stages_compared": len(rows),
        }
    )
    return 1 if regressed else 0


def _rec(stage: str, best_ms: float, cv: float, threads=8, alloc="system", prec="focr-int8"):
    return {
        "stage": stage,
        "ledger_stage": True,
        "best_ms": best_ms,
        "cv_pct": cv,
        "threads": threads,
        "allocator": alloc,
        "precision": prec,
    }


def self_test() -> int:
    failures = []

    def check(name: str, ok: bool):
        emit({"check": name, "result": "pass" if ok else "fail"})
        if not ok:
            failures.append(name)

    base = {r["stage"]: r for r in [_rec("decode_per_token", 50.0, 2.0), _rec("end_to_end", 2000.0, 3.0), _rec("vision_encode", 200.0, 1.0)]}
    # ok + faster + REGRESSION
    cur = {
        r["stage"]: r
        for r in [
            _rec("decode_per_token", 45.0, 2.0),  # faster
            _rec("end_to_end", 2300.0, 3.0),  # +15% -> REGRESSION at 10%
            _rec("vision_encode", 205.0, 1.0),  # +2.5% -> ok
        ]
    }
    rows, regressed = compare(cur, base, "t", 10.0)
    v = {r["stage"]: r["verdict"] for r in rows}
    check("guardrail-regression-detected", regressed and v["end_to_end"] == "REGRESSION")
    check("guardrail-faster-and-ok", v["decode_per_token"] == "faster" and v["vision_encode"] == "ok")
    # noise gate: cv > 5 on either side is ineligible
    cur_noisy = {"decode_per_token": _rec("decode_per_token", 500.0, 22.0)}
    rows, regressed = compare(cur_noisy, base, "t", 10.0)
    check("guardrail-noise-ineligible", not regressed and rows[0]["verdict"] == "noise_ineligible")
    # posture mismatch never compares
    cur_wrong = {"decode_per_token": _rec("decode_per_token", 500.0, 1.0, threads=4)}
    rows, regressed = compare(cur_wrong, base, "t", 10.0)
    check("guardrail-posture-mismatch", not regressed and rows[0]["verdict"] == "posture_mismatch")
    # a new stage is informational, never a failure
    cur_new = {"prefill": _rec("prefill", 100.0, 1.0)}
    rows, regressed = compare(cur_new, base, "t", 10.0)
    check("guardrail-new-stage", not regressed and rows[0]["verdict"] == "new_stage")

    if failures:
        emit({"check": "bench-guardrail-self-test", "result": "fail", "failed": failures})
        return 1
    emit({"check": "bench-guardrail-self-test", "result": "pass"})
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--stages", help="a fresh focr_stages.json (gauntlet_focr.sh output)")
    p.add_argument("--baseline", default=BASELINE_DEFAULT)
    p.add_argument("--regime", default="dense_page", help="output-length regime label (§9.1 axis)")
    p.add_argument("--threshold-pct", type=float, default=10.0)
    p.add_argument("--ratchet", action="store_true", help="REVIEWED baseline update (never in CI)")
    p.add_argument(
        "--parity-receipt",
        default="tests/fixtures/ladder_scorecard/scorecard_armed.json",
        help="the L0-L5 receipt; perf is refused unless all-green ('' disables)",
    )
    args = p.parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
