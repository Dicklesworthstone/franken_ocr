"""Post-hoc counterfactual (not pre-registered): doc 0789's v2 ED_all if page 3 had closed early.

Page 3 of U2-C5a doc 0789 counted "814 815 816 ..." until the 20,000-token page
cap. This rescores the as-run v2_doc output with page 3 cut where a page-level
runaway detector would have closed it, using the driver's own truth and views.
Cut points are in re-encoded v2 tokens of page 3 alone:
  as-run          the full page (must reproduce the scored ED_all, which validates the method)
  masked, floor   the digit-masked guard with the 8,192-token floor
  masked, none    the digit-masked guard without the floor
  raw guard       the guard as built (expected: never fires on a counting loop)

usage: counterfactual_0789.py <repo> <c5a_data> <MarathonDocBench.json> <v2 tokenizer.json>
"""
import importlib.util
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard_calib as g  # noqa: E402
import guard_mask as gm  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

repo, data, marathon_json, tok_path = sys.argv[1:5]
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
scores = [json.loads(line) for line in (data / "scores.ndjson").read_text().splitlines() if line.strip()]
scored = {s["cell"]: s["ed_all"] for s in scores if s["item"] == doc["id"]}
run = next(r for r in runs if r["cell"] == "v2_doc" and r["items"] == [doc["id"]])
output = json.loads(pathlib.Path(run["output"]).read_text())
markdown = output["markdown"]
page3 = output["pages"][2]
assert page3["finish"] == "page_cap" and markdown.count(page3["text"]) == 1, "page 3 text must occur once"

tok = Tokenizer.from_file(tok_path)
ids = tok.encode(page3["text"], add_special_tokens=False).ids
digits = gm.digit_ids(tok)
masked = [gm.MASK if t in digits else t for t in ids]
cuts = {
    "as-run (20,000 cap)": len(ids),
    "masked guard, 8,192 floor": gm.fires(masked, g.MIN_TOKENS),
    "masked guard, no floor": gm.fires(masked, 0),
    "raw guard, 8,192 floor": gm.fires(ids, g.MIN_TOKENS),
}
print(f"doc {doc['id']}: scored ED_all v2_doc {scored.get('v2_doc'):.4f}, v1_pp {scored.get('v1_pp'):.4f}, v1_multi {scored.get('v1_multi'):.4f}")
print(f"page 3: {len(page3['text'])} chars, {len(ids)} re-encoded tokens")
for label, cut in cuts.items():
    if cut is None:
        print(f"  {label:28} never fires")
        continue
    text = page3["text"] if cut >= len(ids) else tok.decode(ids[:cut])
    md = markdown.replace(page3["text"], text)
    ed = h2h.ned(h2h.prediction_views(md, furniture)["all"], gt_all)
    print(f"  {label:28} cut at {cut:>6} tokens, page 3 keeps {len(text):>6} chars -> ED_all {ed:.4f}")
