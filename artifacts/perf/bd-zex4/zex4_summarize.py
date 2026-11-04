"""bd-zex4: summarize zex4/runs.tsv (from zex4_sweep.sh).

Per model and width: best-of-reps real time per page summed over the pages,
the user and sys times of those best runs, sys share of CPU, max RSS, and the
coefficient of variation of the per-rep page-set total. Per model and page:
whether the output bytes are identical across every width (the token output
must not move with width). Per model: the AF-5 sweep JSON (throughput = pages
per second of the best runs; N = threads; the `default` arm is reported, not
fitted) and the scripts/af5_usl_fit.py result for it.

Usage: python3 -I zex4_summarize.py <zex4 dir> <franken_ocr repo>
"""

import collections
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
repo = Path(sys.argv[2])
rows = list(csv.DictReader(open(out_dir / "runs.tsv", encoding="utf-8"), delimiter="\t"))
bad = [r for r in rows if r["exit"] != "0" or not r["real_s"]]
for r in bad:
    print(f"FAILED RUN: {r['model']} {r['page']} {r['arm']} r{r['rep']} exit {r['exit']}")
rows = [r for r in rows if r not in bad]

by = collections.defaultdict(list)  # (model, arm, page) -> rows
for r in rows:
    by[(r["model"], r["arm"], r["page"])].append(r)
models = sorted({r["model"] for r in rows})
pages = sorted({r["page"] for r in rows})


def arm_key(arm):
    return (0, 0) if arm == "default" else (1, int(arm))


for model in models:
    arms = sorted({r["arm"] for r in rows if r["model"] == model}, key=arm_key)
    print(f"\n## {model}\n")
    print("| FOCR_THREADS | reps | real s (best, sum of pages) | user s | sys s | sys share | max RSS GiB | cv % |")
    print("|---|---|---|---|---|---|---|---|")
    samples = []
    for arm in arms:
        best = []
        complete = True
        for page in pages:
            runs = by.get((model, arm, page), [])
            if not runs:
                complete = False
                break
            best.append(min(runs, key=lambda r: float(r["real_s"])))
        if not complete:
            print(f"| {arm} | incomplete | | | | | | |")
            continue
        real = sum(float(r["real_s"]) for r in best)
        user = sum(float(r["user_s"]) for r in best)
        sys_s = sum(float(r["sys_s"]) for r in best)
        rss = max(int(r["maxrss_b"]) for r in best) / 2**30
        reps = sorted({r["rep"] for page in pages for r in by[(model, arm, page)]})
        totals = []
        for rep in reps:
            per = [r for page in pages for r in by[(model, arm, page)] if r["rep"] == rep]
            if len(per) == len(pages):
                totals.append(sum(float(r["real_s"]) for r in per))
        cv = 100 * statistics.stdev(totals) / statistics.mean(totals) if len(totals) > 1 else None
        cv_txt = f"{cv:.1f}" if cv is not None else "—"
        print(f"| {arm} | {len(reps)} | {real:.2f} | {user:.1f} | {sys_s:.1f} | "
              f"{sys_s / (user + sys_s):.0%} | {rss:.2f} | {cv_txt} |")
        if arm != "default":
            sample = {"n": int(arm), "throughput": len(pages) / real}
            if cv is not None:
                sample["cv_pct"] = cv
            samples.append(sample)
    print("\nOutput bytes across widths:")
    for page in pages:
        shas = {r["out_sha256"] for r in rows if r["model"] == model and r["page"] == page}
        print(f"- {page}: {'identical' if len(shas) == 1 else f'DIFFER ({len(shas)} distinct)'}")
    if len(samples) >= 3:
        sweep = {"arch": "apple-m5-ultra-12p24e", "op_class": f"{model}-page-e2e",
                 "num_cpus": 36, "physical_cores": 36, "samples": samples}
        path = out_dir / f"usl_{model}.json"
        path.write_text(json.dumps(sweep, indent=1) + "\n", encoding="utf-8")
        fit = subprocess.run([sys.executable, "-I", str(repo / "scripts/af5_usl_fit.py"), "--samples", str(path)],
                             capture_output=True, text=True, check=False)
        print(f"\nAF-5 USL fit ({path.name}): {fit.stdout.strip() or fit.stderr.strip()}")
