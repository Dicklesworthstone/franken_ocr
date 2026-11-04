#!/usr/bin/env python3
"""Warm resident runs against inline runs, paired, in one invocation (bd-fuh2).

Each pair runs both arms on the same input, in ABBA order so a drift in host
load falls on both. The first served run spawns the daemon and loads the model,
so it is recorded as `cold` and left out of the pairs. Output must match byte
for byte, and a served client must not hold the model.

usage: warm_bench.py FOCR MODEL PAGES_DIR STATE_DIR OUT.json
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

SINGLE_PAIRS = 9
BATCH_PAIRS = 3
SINGLE_PAGE = "page_0009.png"
BATCH_PAGES = ["page_0009.png", "page_0014.png", "page_0019.png"]
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
    }


def summary(pairs):
    ratios = [pair["inline"]["seconds"] / pair["warm"]["seconds"] for pair in pairs]
    saved = [pair["inline"]["seconds"] - pair["warm"]["seconds"] for pair in pairs]
    return {
        "pairs": len(pairs),
        "inline_median_s": statistics.median(p["inline"]["seconds"] for p in pairs),
        "warm_median_s": statistics.median(p["warm"]["seconds"] for p in pairs),
        "ratio_median": statistics.median(ratios),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "saved_median_s": statistics.median(saved),
        "warm_peak_max_bytes": max(p["warm"]["peak_bytes"] for p in pairs),
        "inline_peak_min_bytes": min(p["inline"]["peak_bytes"] for p in pairs),
    }


def main():
    focr, model, pages, state, out = sys.argv[1:6]
    pages = Path(pages)
    for page in {SINGLE_PAGE, *BATCH_PAGES}:
        if not (pages / page).is_file():
            raise SystemExit(f"no page {page} under {pages}")
    base = {k: v for k, v in os.environ.items() if not k.startswith("FOCR_")}
    inline_env = dict(base, FOCR_NO_RESIDENT="1")
    warm_env = dict(base, FOCR_RESIDENT_DIR=state, FOCR_RESIDENT_IDLE_SECS="900")
    single = ["ocr", str(pages / SINGLE_PAGE)]
    batch = ["ocr-batch", *(str(pages / page) for page in BATCH_PAGES)]

    report = {
        "focr": focr,
        "model": model,
        "uptime_start": subprocess.check_output(["uptime"], text=True).strip(),
        "cold": run(focr, model, single, warm_env),
        "single": [],
        "batch": [],
    }

    def pair(args, index):
        arms = {"inline": inline_env, "warm": warm_env}
        order = ["inline", "warm"] if index % 2 == 0 else ["warm", "inline"]
        result = {"first": order[0]}
        for arm in order:
            result[arm] = run(focr, model, args, arms[arm])
        if result["inline"]["stdout_sha256"] != result["warm"]["stdout_sha256"]:
            raise SystemExit(f"output differs on pair {index}: {result}")
        if result["warm"]["peak_bytes"] >= SERVED_PEAK_BYTES:
            raise SystemExit(f"a warm client held the model: {result}")
        if result["inline"]["peak_bytes"] < SERVED_PEAK_BYTES:
            raise SystemExit(f"an inline run did not hold the model: {result}")
        return result

    try:
        for index in range(SINGLE_PAIRS):
            report["single"].append(pair(single, index))
        for index in range(BATCH_PAIRS):
            report["batch"].append(pair(batch, index))
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

    report["uptime_end"] = subprocess.check_output(["uptime"], text=True).strip()
    report["single_summary"] = summary(report["single"])
    report["batch_summary"] = summary(report["batch"])
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("single_summary", "batch_summary")}, indent=1))


if __name__ == "__main__":
    main()
