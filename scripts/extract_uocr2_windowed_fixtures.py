#!/usr/bin/env python3
"""Compact U2-D2a windowed-oracle dumps into committed fixtures (U2-D2b).

Each `<dump root>/<tag>/` holds `run.json` and per window `w<k>/window.json` plus
`rows.*.npy` (int32 per KV row). The dumps are large and live outside the tree;
this keeps what the Rust gates read, one JSON per run, at
`tests/fixtures/uocr2_windowed/<tag>.json`:

* the run's provenance, plan, forced fallbacks and stitched bodies;
* per window: its span, reuse and fallback, page base, tail-copy plan, splice
  rows, forced and generated ids, KV rows, retention, and the sha256 of each
  row array's little-endian int32 data (the dump manifest's own hash, checked
  here against the file).

Run with `python3 -I scripts/extract_uocr2_windowed_fixtures.py <dump root>
<tag>...`. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

ROW_ARRAYS = (
    "ids",
    "origin",
    "src_row",
    "img_page",
    "q_page",
    "bucket",
    "tok_pos",
    "rope_page",
    "frame_bucket",
    "frame_rope_page",
)

WINDOW_KEYS = (
    "window",
    "pages",
    "lo_hi_emit",
    "lookahead",
    "reused",
    "forced_reason",
    "page_base",
    "context_pages",
    "end_on_marker",
    "page_cap_expected",
    "tail_copy_plan",
    "splice_rows",
    "splice_src_rows",
    "seed_iso",
    "seed_coord",
    "forced_len",
    "forced_ids",
    "generated_ids",
    "generated_ids_sha256",
    "finish_reason",
    "kv_rows",
    "retention",
    "guard_steps",
)


PAGE_ID = 128827
IMAGE_ID = 128815


def image_counts(ids: list[int]) -> list[int]:
    """Image rows after each `<PAGE>` of a prompt."""
    counts: list[int] = []
    for t in ids:
        if t == PAGE_ID:
            counts.append(0)
        elif t == IMAGE_ID:
            counts[-1] += 1
    return counts


def npy_int32_data(path: Path) -> bytes:
    """The data bytes of a version-1 `.npy` holding a C-order `<i4` vector."""
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY" or raw[6] != 1:
        raise SystemExit(f"{path}: not a version-1 .npy")
    header_len = struct.unpack("<H", raw[8:10])[0]
    header = raw[10 : 10 + header_len].decode("latin-1")
    if "'descr': '<i4'" not in header or "'fortran_order': False" not in header:
        raise SystemExit(f"{path}: expected a C-order <i4 array, got {header.strip()}")
    return raw[10 + header_len :]


def window_fixture(wdir: Path) -> dict:
    w = json.loads((wdir / "window.json").read_text())
    out = {k: w.get(k) for k in WINDOW_KEYS}
    plan = out.get("tail_copy_plan")
    if plan:
        plan.pop("src_positions_note", None)
    rewritten = w.get("prompt_ids_rewritten")
    sent = w["prompt_ids_sent"]
    out["prompt_len"] = len(rewritten) if rewritten else len(sent)
    # The sent prompt is BOS, then (<PAGE> <image>...) per page, then the
    # instruction: keep the ends, which the gates cannot rebuild without the
    # tokenizer.
    last_image = max(i for i, t in enumerate(sent) if t == IMAGE_ID)
    out["prompt_bos_id"] = sent[0]
    out["prompt_instruction_ids"] = sent[last_image + 1 :]
    out["prompt_image_counts"] = image_counts(sent)
    gen = w["generated_ids"]
    digest = hashlib.sha256(json.dumps(gen, separators=(",", ":")).encode()).hexdigest()
    if digest != w["generated_ids_sha256"]:
        raise SystemExit(f"{wdir}: generated_ids do not hash to generated_ids_sha256")
    hashes = {}
    for name in ROW_ARRAYS:
        info = w["rows_manifest"][f"rows.{name}"]
        data = npy_int32_data(wdir / f"rows.{name}.npy")
        digest = hashlib.sha256(data).hexdigest()
        if digest != info["sha256"] or len(data) != 4 * info["shape"][0]:
            raise SystemExit(f"{wdir}: rows.{name} does not match its manifest")
        hashes[name] = digest
    out["rows_sha256"] = hashes
    return out


def run_fixture(tag_dir: Path) -> dict:
    run = json.loads((tag_dir / "run.json").read_text())
    windows = [window_fixture(tag_dir / f"w{k}") for k in range(len(run["windows"]))]
    prov = run["provenance"]
    return {
        "schema_version": 1,
        "source": "U2-D2a windowed SGLang oracle (tests/fixtures/PROVENANCE.md)",
        "tag": tag_dir.name,
        "provenance": {
            "script_sha256": prov.get("script_sha256"),
            "git_revision": prov.get("git_revision"),
            "wheel": run["wheel"],
        },
        "doc": run["doc"],
        "window": run["window"],
        "threads": run["threads"],
        "plan": run["plan"],
        "forced_fallbacks": {k: v["reason"] for k, v in run["forced_fallbacks"].items()},
        "document": {
            "finish": run["document"]["finish"],
            "bodies": run["document"]["bodies"],
            "text_sha256": run["document"]["text_sha256"],
        },
        "windows": windows,
    }


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    out_dir = Path(__file__).resolve().parent.parent / "tests/fixtures/uocr2_windowed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in sys.argv[2:]:
        fixture = run_fixture(root / tag)
        path = out_dir / f"{tag}.json"
        path.write_text(json.dumps(fixture, separators=(",", ":"), ensure_ascii=False) + "\n")
        print(f"{path}: {len(fixture['windows'])} windows")


if __name__ == "__main__":
    main()
