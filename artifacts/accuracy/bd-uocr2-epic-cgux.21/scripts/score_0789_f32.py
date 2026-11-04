"""Score diag_f32/0789_v2_f32_doc.json (v2 all-f32 document read of U2-C5a doc 0789) with the
driver's own doc truth and views (as counterfactual_0789.py does), beside the scored cells.
Prints per-page finish and length for f32 and for the as-run int8 v2_doc.

usage: score_0789_f32.py <repo> <c5a_data> <MarathonDocBench.json> <f32 doc json>
"""
import importlib.util
import json
import pathlib
import sys

repo, data, marathon_json, f32_path = sys.argv[1:5]
data = pathlib.Path(data)
spec = importlib.util.spec_from_file_location("h2h", pathlib.Path(repo) / "scripts/uocr2_head_to_head.py")
h2h = importlib.util.module_from_spec(spec)
sys.modules["h2h"] = h2h
spec.loader.exec_module(h2h)

items = json.loads((data / "items.json").read_text())
doc = next(d for d in items["documents"] if "0789" in d["id"])
marathon = {pathlib.Path(d["page_info"]["image_path"]).stem: d for d in json.loads(pathlib.Path(marathon_json).read_text())}
truth = h2h._doc_truth(doc, marathon, data)
gt_all = "".join(p["all"] for p in truth)
furniture = [f for p in truth for f in p["furniture"]]
scores = [json.loads(line) for line in (data / "scores.ndjson").read_text().splitlines() if line.strip()]
scored = {s["cell"]: s["ed_all"] for s in scores if s["item"] == doc["id"]}
runs = [json.loads(line) for line in (data / "runs.ndjson").read_text().splitlines() if line.strip()]
run = next(r for r in runs if r["cell"] == "v2_doc" and r["items"] == [doc["id"]])
int8 = json.loads(pathlib.Path(run["output"]).read_text())
f32 = json.loads(pathlib.Path(f32_path).read_text())
print(f"doc {doc['id']}: scored ED_all v2_doc(int8) {scored['v2_doc']:.4f}  v1_pp {scored['v1_pp']:.4f}  v1_multi {scored['v1_multi']:.4f}")
for label, out in (("int8 as run", int8), ("f32", f32)):
    ed = h2h.ned(h2h.prediction_views(out["markdown"], furniture)["all"], gt_all)
    pages = out.get("pages") or []
    desc = "; ".join(f"p{i + 1} {p.get('finish')} {len(p.get('text') or '')}ch" for i, p in enumerate(pages))
    print(f"  {label:12} ED_all {ed:.4f}  finish_reason {out.get('finish_reason')}  pages: {desc}")
for i, (a, b) in enumerate(zip(int8.get("pages") or [], f32.get("pages") or [])):
    ta, tb = a.get("text") or "", b.get("text") or ""
    pre = next((k for k, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    print(f"  page {i + 1}: " + ("identical" if ta == tb else f"differ after {pre} common chars"))
p3 = (f32.get("pages") or [{}, {}, {}])[2].get("text") or ""
print("  f32 page 3 tail:", repr(p3[-200:]))
