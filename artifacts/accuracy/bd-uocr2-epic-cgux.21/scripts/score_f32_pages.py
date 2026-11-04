"""Score diag_f32/v2_f32.json (v2 all-f32, no budget) with the C5a driver's _score_single and
set it beside the as-run int8 scores (scores.ndjson; reruns from rerun scores if present).

usage: score_f32_pages.py <repo> <c5a data dir> <OmniDocBench.json> <v2_f32.json>
"""
import importlib.util
import json
import sys
from pathlib import Path

repo, data, omni_json, f32_json = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
spec = importlib.util.spec_from_file_location("h2h", repo / "scripts" / "uocr2_head_to_head.py")
h2h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h2h)
omni = {p["page_info"]["image_path"]: p for p in json.load(open(omni_json, encoding="utf-8"))}
items = {i["image_path"]: i for i in json.loads((data / "items.json").read_text())["single_pages"]}
asrun = {}
for line in (data / "scores.ndjson").read_text().splitlines():
    r = json.loads(line)
    if r["cell"] in ("v1_single", "v2_single"):
        asrun.setdefault(r["item"], {})[r["cell"]] = r
out = json.loads(f32_json.read_text())
for res in out["results"]:
    name = Path(res["image"]).name
    item = items[name]
    gt = h2h.layout_views(omni[name]["layout_dets"])
    s = h2h._score_single(item, gt, res.get("markdown") if res.get("ok") else None)
    v1, v2 = (asrun.get(item["id"], {}).get(c, {}) for c in ("v1_single", "v2_single"))
    fmt = lambda r: f"{r.get('cer_text', float('nan')):.3f} ({r.get('pred_chars')} ch)"  # noqa: E731
    print(f"{name[:48]:48} f32 CER {fmt(s)} {res.get('seconds', 0):.0f}s | int8 v2 {fmt(v2)} | v1 {fmt(v1)} | gt {s['gt_chars']} ch")
