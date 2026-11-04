#!/usr/bin/env python3
"""Warm resident Unlimited-OCR-2 runs against inline runs, paired (bd-u77k).

The bd-fuh2 harness's method on v2: each pair runs both arms on one page, in
ABBA order so a drift in host load falls on both. The first served run spawns
the daemon and loads the model, so it is recorded as `cold` and left out of the
pairs. Output must match byte for byte, and a served client must not hold the
model. Pages are capped at `--max-page-tokens 256`, so a run is seconds, not
minutes; the saving is one model load whatever the decode length.

Three inline robot runs then time the `load` stage itself, from the stage
events' `elapsed_ms`: the part of a run a warm daemon skips.

usage: warm_bench.py FOCR MODEL PAGE STATE_DIR OUT.json
"""

import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

PAIRS = 9
LOAD_RUNS = 3
CAP = ["--max-page-tokens", "256"]
SERVED_PEAK_BYTES = 1 << 30


def run(focr, model, args, env):
    argv = ["/usr/bin/time", "-l", focr, *args, "--model", model]
    started = time.monotonic()
    done = subprocess.run(argv, env=env, capture_output=True, check=False)
    seconds = time.monotonic() - started
    if done.returncode != 0:
        raise SystemExit(f"{argv} exited {done.returncode}: {done.stderr[-400:]!r}")
    peak = re.search(rb"(\d+)\s+maximum resident set size", done.stderr)
    if peak is None:
        raise SystemExit(f"no peak memory in {done.stderr[-400:]!r}")
    return {
        "seconds": seconds,
        "peak_bytes": int(peak.group(1)),
        "stdout_sha256": hashlib.sha256(done.stdout).hexdigest(),
        "stdout": done.stdout,
    }


def load_ms(stdout):
    """The `load` stage's duration: from its event to the `forward` event."""
    stages = {}
    for line in stdout.splitlines():
        event = json.loads(line)
        if event.get("event") == "stage":
            stages[event["name"]] = event["elapsed_ms"]
    if "load" not in stages or "forward" not in stages:
        raise SystemExit(f"no load and forward stages in {stdout[-400:]!r}")
    return stages["forward"] - stages["load"]


def main():
    focr, model, page, state, out = sys.argv[1:6]
    if not Path(page).is_file():
        raise SystemExit(f"no page {page}")
    base = {k: v for k, v in os.environ.items() if not k.startswith("FOCR_")}
    inline_env = dict(base, FOCR_NO_RESIDENT="1")
    warm_env = dict(base, FOCR_RESIDENT_DIR=state, FOCR_RESIDENT_IDLE_SECS="900")
    single = ["ocr", page, "--json", *CAP]

    def kept(result):
        return {k: v for k, v in result.items() if k != "stdout"}

    report = {
        "focr": focr,
        "model": model,
        "page": page,
        "uptime_start": subprocess.check_output(["uptime"], text=True).strip(),
        "cold": kept(run(focr, model, single, warm_env)),
        "pairs": [],
        "load_ms": [],
    }

    def pair(index):
        arms = {"inline": inline_env, "warm": warm_env}
        order = ["inline", "warm"] if index % 2 == 0 else ["warm", "inline"]
        result = {"first": order[0]}
        for arm in order:
            result[arm] = kept(run(focr, model, single, arms[arm]))
        if result["inline"]["stdout_sha256"] != result["warm"]["stdout_sha256"]:
            raise SystemExit(f"output differs on pair {index}: {result}")
        if result["warm"]["peak_bytes"] >= SERVED_PEAK_BYTES:
            raise SystemExit(f"a warm client held the model: {result}")
        if result["inline"]["peak_bytes"] < SERVED_PEAK_BYTES:
            raise SystemExit(f"an inline run did not hold the model: {result}")
        return result

    try:
        for index in range(PAIRS):
            report["pairs"].append(pair(index))
    finally:
        for entry in Path(state).iterdir():
            try:
                pid = json.loads(entry.read_text()).get("pid")
            except (OSError, ValueError):
                continue
            if pid:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
    robot = ["ocr", page, "--robot", *CAP]
    for _ in range(LOAD_RUNS):
        report["load_ms"].append(load_ms(run(focr, model, robot, inline_env)["stdout"]))

    pairs = report["pairs"]
    ratios = [p["inline"]["seconds"] / p["warm"]["seconds"] for p in pairs]
    report["uptime_end"] = subprocess.check_output(["uptime"], text=True).strip()
    report["summary"] = {
        "pairs": len(pairs),
        "inline_median_s": statistics.median(p["inline"]["seconds"] for p in pairs),
        "warm_median_s": statistics.median(p["warm"]["seconds"] for p in pairs),
        "ratio_median": statistics.median(ratios),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "saved_median_s": statistics.median(
            p["inline"]["seconds"] - p["warm"]["seconds"] for p in pairs
        ),
        "load_ms_median": statistics.median(report["load_ms"]),
        "warm_peak_max_bytes": max(p["warm"]["peak_bytes"] for p in pairs),
        "inline_peak_min_bytes": min(p["inline"]["peak_bytes"] for p in pairs),
        "stdout_sha256": pairs[0]["inline"]["stdout_sha256"],
    }
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
