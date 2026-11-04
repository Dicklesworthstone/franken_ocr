"""Score the v2 f32 document sweep (diag_f32_docs.sh + diag_f32/0789_v2_f32_doc.json) exactly as the
driver's cmd_score scores document cells, and run the analysis fixed before the results:
  1. per-document ED_all, v2 f32 - v2 int8 (scored), paired bootstrap (driver's _bootstrap);
  2. page_cap pages per precision;
  3. post hoc sensitivity: v2 f32 - v1_pp and v2 f32 - v1_multi beside the scored int8 deltas.
Run with a python that has numpy (the driver's _bootstrap imports it).

usage: score_f32_docs.py <repo> <c5a_data> <MarathonDocBench.json> <diag_f32 dir>
"""
import collections
import importlib.util
import json
import pathlib
import sys

repo, data, marathon_json, diag = sys.argv[1:5]
data, diag = pathlib.Path(data), pathlib.Path(diag)
spec = importlib.util.spec_from_file_location("h2h", pathlib.Path(repo) / "scripts/uocr2_head_to_head.py")
h2h = importlib.util.module_from_spec(spec)
sys.modules["h2h"] = h2h
spec.loader.exec_module(h2h)

items = json.loads((data / "items.json").read_text())
marathon = {pathlib.Path(d["page_info"]["image_path"]).stem: d for d in json.loads(pathlib.Path(marathon_json).read_text())}
scores = collections.defaultdict(dict)
for line in (data / "scores.ndjson").read_text().splitlines():
    s = json.loads(line)
    if "ed_all" in s and s["cell"] in ("v2_doc", "v1_pp", "v1_multi"):
        scores[s["item"]][s["cell"]] = s["ed_all"]
int8_out = {}
for line in (data / "runs.ndjson").read_text().splitlines():
    r = json.loads(line)
    if r["cell"] == "v2_doc" and r["exit"] in (0, 5):
        int8_out[r["items"][0]] = h2h._load_json(pathlib.Path(r["output"]))
rc = {}
tsv = diag / "docs" / "runs.tsv"
if tsv.exists():
    for line in tsv.read_text().splitlines():
        doc_id, code, _wall = line.split("\t")
        rc[doc_id] = int(code)

rows, caps = [], collections.Counter()
for doc in items["documents"]:
    path = diag / "0789_v2_f32_doc.json" if "0789" in doc["id"] else diag / "docs" / f"{doc['id']}.json"
    code = 0 if "0789" in doc["id"] else rc.get(doc["id"])
    if code is None:
        print(f"{doc['id']}: not run yet")
        continue
    output = h2h._load_json(path) if code in (0, 5) else None
    failed = output is None or not output.get("markdown")
    truth = h2h._doc_truth(doc, marathon, data)
    gt_all = "".join(p["all"] for p in truth)
    furniture = [f for p in truth for f in p["furniture"]]
    ed = 1.0 if failed else h2h.ned(h2h.prediction_views(output["markdown"], furniture)["all"], gt_all)
    for label, out in (("f32", output), ("int8", int8_out.get(doc["id"]))):
        caps[label] += sum(1 for p in (out or {}).get("pages") or [] if p.get("finish") == "page_cap")
    s = scores[doc["id"]]
    rows.append((doc["id"], ed, s.get("v2_doc"), s.get("v1_pp"), s.get("v1_multi"), failed))
    print(f"{doc['id'][:40]:40} f32 {ed:.4f}  int8 {s.get('v2_doc', float('nan')):.4f}  "
          f"v1_pp {s.get('v1_pp', float('nan')):.4f}  v1_multi {s.get('v1_multi', float('nan')):.4f}" + ("  FAILED" if failed else ""))

paired = [r for r in rows if None not in r[1:5]]
print(f"\n{len(paired)} documents scored; page_cap pages: f32 {caps['f32']}, int8 {caps['int8']}")
if len(paired) >= 2:
    for name, idx in (("v2 f32 - v2 int8", 2), ("v2 f32 - v1_pp", 3), ("v2 f32 - v1_multi", 4)):
        b = h2h._bootstrap([r[1] - r[idx] for r in paired])
        print(f"  {name:18} mean {b['mean']:+.4f} [{b['low']:+.4f}, {b['high']:+.4f}] (n={b['n']})")
    for name, idx in (("v2 int8 - v1_pp", 3), ("v2 int8 - v1_multi", 4)):
        b = h2h._bootstrap([r[2] - r[idx] for r in paired])
        print(f"  {name:18} mean {b['mean']:+.4f} [{b['low']:+.4f}, {b['high']:+.4f}] (n={b['n']}, same documents)")
