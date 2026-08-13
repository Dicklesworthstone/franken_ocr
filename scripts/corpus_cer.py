#!/usr/bin/env python3
"""Score a corpus run against a reference decode, as normalized CER.

The project's corpus gate measures **divergence from a full-precision decode**,
not accuracy against human ground truth — `docs/DISCREPANCIES.md` says so
outright: "this is a PROXY reference … the bf16 decode is itself imperfect OCR.
No ground truth exists for the archive scans." That is the right yardstick for a
quantization experiment (it answers "did the output move?") and the wrong one for
"is the output correct". Keep the distinction when reading any number this
prints.

Two subcommands:

    run    <artifact> <pages-dir> <out-dir>     decode every page, write .md + INDEX.jsonl
    score  <ref-dir>  <cand-dir>                per-page + aggregate normalized CER

Normalization is deliberately minimal and stated here rather than buried: strip
leading/trailing whitespace per line, collapse runs of whitespace to one space,
drop empty lines. No case folding and no unicode folding — a quantization change
that alters case or punctuation IS a divergence and must not be normalized away.

Aggregate CER is total-edits / total-reference-characters, NOT the mean of
per-page rates. A mean lets a 40-character page swamp a 4,000-character one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def normalize(text: str) -> str:
    lines = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        collapsed = " ".join(raw.split())
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines)


def levenshtein(a: str, b: str) -> int:
    """Edit distance with a rolling row — the pages reach ~10k chars, so the
    full matrix would be 100M cells and pointless."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def cmd_run(args: argparse.Namespace) -> int:
    pages = sorted(Path(args.pages).glob("page_0*.png"))
    if not pages:
        print(f"no pages under {args.pages}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["FOCR_MODEL_PATH"] = str(Path(args.artifact).resolve())
    # The hard pages legitimately take thousands of tokens; the ledger used the
    # same 2-hour stage budget. A budget-kill would silently truncate a page and
    # look like a quantization regression.
    env.setdefault("FOCR_STAGE_BUDGET_FORWARD_MS", "7200000")
    if args.threads:
        env["FOCR_THREADS"] = str(args.threads)

    index = []
    for i, page in enumerate(pages, 1):
        target = out / f"{page.stem}.md"
        if target.exists() and not args.force:
            print(f"[{i}/{len(pages)}] {page.name}: cached", flush=True)
            index.append({"page": page.name, "md": target.name, "cached": True})
            continue
        started = time.time()
        proc = subprocess.run(
            [args.focr, "ocr", str(page)],
            env=env,
            capture_output=True,
            text=True,
        )
        seconds = time.time() - started
        if proc.returncode != 0:
            print(
                f"[{i}/{len(pages)}] {page.name}: FAILED rc={proc.returncode} "
                f"{proc.stderr.strip()[:160]}",
                flush=True,
            )
            index.append(
                {"page": page.name, "error": proc.stderr.strip()[:400], "seconds": seconds}
            )
            continue
        target.write_text(proc.stdout)
        print(
            f"[{i}/{len(pages)}] {page.name}: {len(proc.stdout)} chars in {seconds:.1f}s",
            flush=True,
        )
        index.append(
            {
                "page": page.name,
                "md": target.name,
                "chars": len(proc.stdout),
                "seconds": round(seconds, 2),
            }
        )
    (out / "INDEX.jsonl").write_text(
        "\n".join(json.dumps(r) for r in index) + "\n"
    )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    ref_dir, cand_dir = Path(args.reference), Path(args.candidate)
    refs = sorted(ref_dir.glob("page_0*.md"))
    if not refs:
        print(f"no reference .md under {ref_dir}", file=sys.stderr)
        return 2

    rows = []
    total_edits = total_chars = 0
    missing = []
    for ref_path in refs:
        cand_path = cand_dir / ref_path.name
        if not cand_path.exists():
            missing.append(ref_path.name)
            continue
        ref = normalize(ref_path.read_text())
        cand = normalize(cand_path.read_text())
        edits = levenshtein(ref, cand)
        rate = edits / len(ref) if ref else (0.0 if not cand else 1.0)
        rows.append(
            {
                "page": ref_path.stem,
                "ref_chars": len(ref),
                "cand_chars": len(cand),
                "edits": edits,
                "cer": rate,
            }
        )
        total_edits += edits
        total_chars += len(ref)

    rows.sort(key=lambda r: -r["cer"])
    print(f"{'page':16s} {'ref':>7s} {'cand':>7s} {'edits':>7s}   CER")
    for r in rows:
        print(
            f"{r['page']:16s} {r['ref_chars']:7d} {r['cand_chars']:7d} "
            f"{r['edits']:7d}   {r['cer']:.7f}"
        )
    aggregate = total_edits / total_chars if total_chars else 0.0
    print(
        f"\naggregate normalized CER = {aggregate:.10f}  "
        f"({total_edits} edits / {total_chars} reference chars over {len(rows)} pages)"
    )
    if missing:
        print(f"MISSING from candidate: {', '.join(missing)}")
    if args.budget is not None:
        verdict = "PASS" if aggregate <= args.budget else "FAIL"
        print(f"budget {args.budget}: {verdict}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "reference": str(ref_dir),
                    "candidate": str(cand_dir),
                    "aggregate_cer": aggregate,
                    "total_edits": total_edits,
                    "total_reference_chars": total_chars,
                    "pages": rows,
                    "missing": missing,
                },
                indent=2,
            )
            + "\n"
        )
    return 0 if not missing else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="decode every corpus page with one artifact")
    run.add_argument("artifact")
    run.add_argument("pages")
    run.add_argument("out")
    run.add_argument("--focr", default="focr")
    run.add_argument("--threads", type=int, default=0)
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="normalized CER of candidate vs reference")
    score.add_argument("reference")
    score.add_argument("candidate")
    score.add_argument("--budget", type=float, default=None)
    score.add_argument("--json-out")
    score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
