"""Post hoc, gates nothing: doc 0789's v2 ED_all with the capped page 3 replaced by int8 v2's
read of page 3 ALONE (diag_0789_p3_r2.sh), the counterfactual_0789.py method. Also the
C5a document mean of v2_doc - v1_pp and v2_doc - v1_multi with only that one score changed.

usage: score_0789_p3_r2.py <repo> <c5a_data> <MarathonDocBench.json> <p3 single json> <f32 doc json>
"""
import importlib.util
import json
import pathlib
import sys

repo, data, marathon_json, p3_path, f32_path = sys.argv[1:6]
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

runs = [json.loads(line) for line in (data / "runs.ndjson").read_text().splitlines() if line.strip()]
run = next(r for r in runs if r["cell"] == "v2_doc" and r["items"] == [doc["id"]])
output = json.loads(pathlib.Path(run["output"]).read_text())
markdown, page3 = output["markdown"], output["pages"][2]
assert page3["finish"] == "page_cap" and markdown.count(page3["text"]) == 1, "page 3 text must occur once"
single = json.loads(pathlib.Path(p3_path).read_text())["markdown"]
f32_p3 = json.loads(pathlib.Path(f32_path).read_text())["pages"][2]["text"]


def ed(md):
    return h2h.ned(h2h.prediction_views(md, furniture)["all"], gt_all)


as_run, swapped = ed(markdown), ed(markdown.replace(page3["text"], single))
print(f"page 3: as run {len(page3['text'])} chars (page_cap); alone {len(single)} chars; f32 doc page 3 {len(f32_p3)} chars")
print(f"page 3 alone vs f32 doc page 3: NED {h2h.ned(single, f32_p3):.4f}")
print(f"doc 0789 ED_all: as run {as_run:.4f}, page 3 re-read alone {swapped:.4f}")

scores = [json.loads(line) for line in (data / "scores.ndjson").read_text().splitlines() if line.strip()]
by_doc = {}
for s in scores:
    if "ed_all" in s and s["cell"] in ("v2_doc", "v1_pp", "v1_multi"):
        by_doc.setdefault(s["item"], {})[s["cell"]] = s["ed_all"]
docs = [d for d, c in by_doc.items() if len(c) == 3]
print(f"scored ED_all for doc 0789: {by_doc[doc['id']]}")
for other in ("v1_pp", "v1_multi"):
    base = [by_doc[d]["v2_doc"] - by_doc[d][other] for d in docs]
    alt = [(swapped if d == doc["id"] else by_doc[d]["v2_doc"]) - by_doc[d][other] for d in docs]
    b0, b1 = h2h._bootstrap(base), h2h._bootstrap(alt)
    print(f"  v2_doc - {other:8} n={len(docs)}: as run {b0['mean']:+.4f} [{b0['low']:+.4f}, {b0['high']:+.4f}]"
          f"  ->  with p3 re-read {b1['mean']:+.4f} [{b1['low']:+.4f}, {b1['high']:+.4f}]")
