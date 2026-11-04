"""U2-F1 (bd-uocr2-epic-cgux.38): PERF_LEDGER rows from u2f1/runs.tsv and u2f1/summary.json.

focr_ms is v2's wall and ref_ms is v1's, from the same invocation (same binary, width and
inputs), so ratio > 1 means v2 is faster. Single pages: best of 3 per side, order
alternating per rep. Documents: one run per side; v2 document mode against v1
--multi-page and against v1 per page. Stage seconds come from u2f1_summarize.py.

usage: u2f1_ledger.py <u2f1 dir> <v1 weights sha256> <v2 weights sha256> <pages.sha256> <load text>
"""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

out_dir, v1_sha, v2_sha, pages_file, load_text = Path(sys.argv[1]), *sys.argv[2:6]
rows = [r for r in csv.DictReader(open(out_dir / "runs.tsv", encoding="utf-8"), delimiter="\t") if r["exit"] == "0"]
summary = {(x["cell"], x["input"], x["model"], x["mode"]): x for x in json.load(open(out_dir / "summary.json"))}
host = (out_dir / "host.txt").read_text()
binary_sha = next(line.split()[1] for line in host.splitlines() if line.startswith("sha256 "))
page_sha = dict(reversed(line.split()) for line in open(pages_file) if line.strip())
page_sha = {Path(k).stem: v for k, v in page_sha.items()}
DOCS = {"s2": ["page_0009", "page_0014", "page_0019"]}
DOCS["p10"] = DOCS["s2"] + ["page_0023", "page_0086", "page_0091", "page_0105", "page_0107", "page_0108", "page_0126"]
DOCS["p20"] = DOCS["p10"] + ["page_0136", "page_0179", "page_0250", "page_0358", "page_0400", "page_0437",
                             "page_0547", "page_0557", "page_0590", "page_0645"]
V2 = "baidu/Unlimited-OCR-2@5632329a32e103d5f1add270681e0f5c1233c6e4 vs v1 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"
WEIGHTS = f"weights=unlimited-ocr-2.v0.1.0.int8.focrq sha256={v2_sha}; v1 weights=unlimited-ocr.int8.focrq sha256={v1_sha}"
RUNAWAY = {("single", "page_0590", "unlimited-ocr", "base"): "v1 table-row loop: 4,033 tokens against v2's 1,644",
           ("doc", "p20", "unlimited-ocr", "multi"): "v1 counting loop \"\\( 10^{-1643} \\) to \\( 10^{-1644} \\) …\", 30,543 tokens"}
common = ["aarch64+neon+dotprod (Apple M5 Ultra, 12 Super + 24 Performance cores)"]


def stage_text(x):
    return (f"wall {x['wall_s']:.2f} s, {x['wall_per_page_s']:.2f} s/page; vision {x['vision_s']:.2f} s; prefill "
            f"{x['prefill_s']:.2f} s ({x['prefill_tokens']} tok); decode {x['decode_s']:.2f} s ({x['decode_tokens']} tok, "
            f"{x['s_per_tok']:.4f} s/tok); peak RSS {x['rss_gib']:.2f} GiB")


def shas(cell, inp, model, mode):
    files = sorted(out_dir.glob(f"{cell}.{inp}.{model}.{mode}.r*.md"))
    return {hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, len(files)


def cv(cell, inp, model, mode):
    walls = [float(r["real_s"]) for r in rows if (r["cell"], r["input"], r["model"], r["mode"]) == (cell, inp, model, mode)]
    return 100 * statistics.stdev(walls) / statistics.mean(walls) if len(walls) > 1 else None


def row(claim, inp, pages, stage, v2k, v1k, v1_label, cmd):
    v2, v1 = summary[v2k], summary[v1k]
    fixture = "; ".join(f"page={p}.png sha256={page_sha[p]}" for p in pages) + "; " + WEIGHTS
    out2, n2 = shas(*v2k)
    out1, n1 = shas(*v1k)
    proof = ("different models, so outputs differ by design; accuracy for these two artifacts is U2-C5a's "
             "(bd-uocr2-epic-cgux.33). Outputs saved as *.r1.md")
    if n2 > 1:
        proof += (f"; each side's output is byte-identical across its {n2} reps" if len(out2) == 1 and len(out1) == 1
                  else "; OUTPUT DIFFERS ACROSS REPS")
    flags = [RUNAWAY[k] for k in (v2k, v1k) if k in RUNAWAY]
    c2, c1 = cv(*v2k), cv(*v1k)
    rep_text = (f"Best of 3 per side (cv {c2:.1f}% v2, {c1:.1f}% v1). " if c2 is not None
                else "One run per side, so no cv; not eligible as a claim. ")
    notes = (rep_text + f"v2: {stage_text(v2)}. {v1_label}: {stage_text(v1)}. "
             + (f"Runaway: {'; '.join(flags)}, so this wall ratio overstates v2. " if flags else "")
             + f"Shared host: {load_text}.")
    cells = ["2026-11-01", claim, "artifacts/perf/bd-uocr2-epic-cgux.38", V2, fixture, *common, stage,
             f"none: {v1_label} (focr, same binary, width and inputs, same invocation); not a reference model",
             f"{v2['wall_s'] * 1000:.0f}", f"{v1['wall_s'] * 1000:.0f}", f"{v1['wall_s'] / v2['wall_s']:.2f}",
             "n/a", "n/a", "n/a", "focr-int8 v2 (U2-C3/C4) vs focr-int8 v1 (conservative); int8 FFN, high-precision attention on both",
             "focr=12 (FOCR_THREADS=12) on both sides", "system",
             f"u2f1_bench.sh: FOCR_THREADS=12 FOCR_NO_RESIDENT=1 FOCR_TIMING=1 /usr/bin/time -l focr {cmd}; "
             f"release-perf binary sha256 {binary_sha[:12]}…",
             "none: measurement only; no default changes", proof, notes]
    return "| " + " | ".join(cells) + " |"


out = []
for p in ("page_0009", "page_0014", "page_0590"):
    out.append(row(f"U2F1-single-{p}-20261101", p, [p], "end-to-end, single page, base, cold process",
                   ("single", p, "unlimited-ocr-2", "base"), ("single", p, "unlimited-ocr", "base"),
                   "v1 single page", "ocr <page> --model <id> -o <out>.md"))
for d, label in (("s2", "3-page"), ("p10", "10-page"), ("p20", "20-page")):
    for mode, v1_label, cmd in (("multi", "v1 --multi-page", "ocr <dir> --multi-page --model <id> -o <out>.md"),
                                ("per-page", "v1 per page", "ocr <dir> --model <id> -o <out>.md (v1 per page)")):
        tag = "v1multi" if mode == "multi" else "v1perpage"
        out.append(row(f"U2F1-doc-{d}-vs-{tag}-20261101", d, DOCS[d],
                       f"end-to-end, {label} document of corpus pages, cold process",
                       ("doc", d, "unlimited-ocr-2", "multi"), ("doc", d, "unlimited-ocr", mode), v1_label, cmd))
print("\n".join(out))
