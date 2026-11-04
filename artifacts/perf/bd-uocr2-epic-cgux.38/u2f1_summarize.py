"""U2-F1: per-run stage totals from u2f1/runs.tsv and each run's FOCR_TIMING stderr (<stem>.time).

Per run: wall/user/sys/RSS from /usr/bin/time; pages; vision = sum of vision.sam +
vision.clip + vision.bridge rows; prefill = `prefill` rows plus v2's window.recompute /
window.splice rows (seconds and rows/tokens); decode = `decode` rows (seconds, tokens).
Single pages report the best-of-3 run by wall. Output: a markdown table on stdout and
u2f1/summary.json.

usage: u2f1_summarize.py <u2f1 dir>
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = list(csv.DictReader(open(out / "runs.tsv", encoding="utf-8"), delimiter="\t"))
VISION = re.compile(r"\[focr-timing\]\s+vision\.(sam|clip|bridge) ([0-9.]+)s")
PREFILL = re.compile(r"\[focr-timing\]\s+prefill ([0-9.]+)s \(([0-9]+) tokens\)")
RECOMPUTE = re.compile(r"\[focr-timing\]\s+window\.recompute ([0-9]+) rows .* ([0-9.]+)s")
SPLICE = re.compile(r"\[focr-timing\]\s+window\.splice .* prefilled ([0-9]+) rows .* ([0-9.]+)s")
DECODE = re.compile(r"\[focr-timing\]\s+decode ([0-9.]+)s \(([0-9]+) tokens")
PAGES = {"s2": 3, "p10": 10, "p20": 20}


def stem(r):
    return out / f"{r['cell']}.{r['input']}.{r['model']}.{r['mode']}.r{r['rep']}"


def stages(r):
    text = Path(str(stem(r)) + ".time").read_text(errors="replace")
    vision = sum(float(m.group(2)) for m in VISION.finditer(text))
    pre_s = pre_n = 0.0
    for m in PREFILL.finditer(text):
        pre_s += float(m.group(1)); pre_n += int(m.group(2))
    for m in RECOMPUTE.finditer(text):
        pre_n += int(m.group(1)); pre_s += float(m.group(2))
    for m in SPLICE.finditer(text):
        pre_n += int(m.group(1)); pre_s += float(m.group(2))
    dec_s = dec_n = 0.0
    for m in DECODE.finditer(text):
        dec_s += float(m.group(1)); dec_n += int(m.group(2))
    return {"vision_s": vision, "prefill_s": pre_s, "prefill_tokens": int(pre_n),
            "decode_s": dec_s, "decode_tokens": int(dec_n),
            "s_per_tok": dec_s / dec_n if dec_n else None}


groups = collections.defaultdict(list)
for r in rows:
    if r["exit"] == "0":
        groups[(r["cell"], r["input"], r["model"], r["mode"])].append(r)
result = []
for key, runs in groups.items():
    best = min(runs, key=lambda r: float(r["real_s"]))
    pages = PAGES.get(key[1], 1)
    st = stages(best)
    result.append({"cell": key[0], "input": key[1], "model": key[2], "mode": key[3], "reps": len(runs),
                   "pages": pages, "wall_s": float(best["real_s"]), "user_s": float(best["user_s"]),
                   "sys_s": float(best["sys_s"]), "rss_gib": int(best["maxrss_b"]) / 2**30,
                   "out_chars": int(best["out_chars"]), "wall_per_page_s": float(best["real_s"]) / pages, **st})
(out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
print("| cell | input | model | mode | reps | wall s | s/page | RSS GiB | vision s | prefill s (tokens) | decode s (tokens) | s/token | chars |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in result:
    spt = f"{x['s_per_tok']:.4f}" if x["s_per_tok"] else "—"
    print(f"| {x['cell']} | {x['input']} | {x['model']} | {x['mode']} | {x['reps']} | {x['wall_s']:.2f} | "
          f"{x['wall_per_page_s']:.2f} | {x['rss_gib']:.2f} | {x['vision_s']:.2f} | {x['prefill_s']:.2f} ({x['prefill_tokens']}) | "
          f"{x['decode_s']:.2f} ({x['decode_tokens']}) | {spt} | {x['out_chars']} |")
