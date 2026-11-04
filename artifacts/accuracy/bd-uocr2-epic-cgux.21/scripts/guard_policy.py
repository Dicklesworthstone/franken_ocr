"""Post hoc, bd-gftg (not pre-registered): what the v1 RunawayGuard would do to U2-C5a's
single-page result under three abort policies, on the registered-rerun scores.

  none     today's default: the runaway text is kept (the rerun score)
  reject   the guard as built: exit 5, no output (scored as a failure, 1.0)
  prefix   a partial-output variant: keep the text before the abort point
  prefix-w keep the text before (abort point - 2,048-token window), where the loop
           the guard saw had already begun

Abort points come from the offline replay (re-encoded text, guard_calib.py); wall at
abort is the rerun wall scaled by token share, an estimate.

usage: guard_policy.py <repo> <c5a_data> <OmniDocBench.json> <v1 tokenizer.json> <guard_calib.ndjson>
"""
import importlib.util
import json
import pathlib
import sys

from tokenizers import Tokenizer

repo, data, omni_json, tok_path, calib = sys.argv[1:6]
data = pathlib.Path(data)
spec = importlib.util.spec_from_file_location("h2h", pathlib.Path(repo) / "scripts/uocr2_head_to_head.py")
h2h = importlib.util.module_from_spec(spec)
sys.modules["h2h"] = h2h
spec.loader.exec_module(h2h)

items = json.loads((data / "items.json").read_text())
single = {i["id"]: i for i in items["single_pages"]}
omni = {p["page_info"]["image_path"]: p for p in json.load(open(omni_json, encoding="utf-8"))}
scores = [json.loads(line) for line in (data / "scores.ndjson").read_text().splitlines() if line.strip()]
reruns = {r["items"][0]: r for r in (json.loads(line) for line in (data / "rerun.ndjson").read_text().splitlines() if line.strip()) if r["cell"] == "v1_single"}
aborts = {}
for line in open(calib, encoding="utf-8"):
    r = json.loads(line)
    if r["cell"] == "v1_single" and r["source"] == "rerun" and r["abort_at"]:
        aborts["omni-" + pathlib.Path(r["name"]).stem] = r
tok = Tokenizer.from_file(tok_path)

base = h2h._with_reruns(scores)
v1 = {s["item"]: s for s in base if s["cell"] == "v1_single"}
policy_rows = {p: {} for p in ("reject", "prefix", "prefix-w")}
print(f"{'page':48} {'tokens':>6} {'abort':>6} {'wall':>6} {'@abort':>6}  CER none / reject / prefix / prefix-w")
saved = total = 0.0
for item_id, a in sorted(aborts.items(), key=lambda kv: kv[1]["abort_at"]):
    run = reruns[item_id]
    item = single[item_id]
    out = json.loads(pathlib.Path(run["output"]).read_text())
    md = out["results"][0]["markdown"]
    ids = tok.encode(md, add_special_tokens=False).ids
    gt = h2h.layout_views(omni[item["image_path"]]["layout_dets"])
    got = {"none": v1[item_id]}
    got["reject"] = h2h._score_single(item, gt, None)
    got["prefix"] = h2h._score_single(item, gt, tok.decode(ids[:a["abort_at"]]))
    got["prefix-w"] = h2h._score_single(item, gt, tok.decode(ids[:max(0, a["abort_at"] - 2048)]))
    for p in policy_rows:
        policy_rows[p][item_id] = got[p]
    wall = run["wall_s"]
    at = wall * a["abort_at"] / len(ids)
    saved += wall - at
    total += wall
    cer = " / ".join(f"{got[p].get('cer_text', float('nan')):.3f}" for p in ("none", "reject", "prefix", "prefix-w"))
    print(f"{item_id[5:53]:48} {len(ids):>6} {a['abort_at']:>6} {wall:>6.0f} {at:>6.0f}  {cer}")
print(f"wall on these pages: {total:.0f} s; at abort about {total - saved:.0f} s (saves about {saved:.0f} s)")

print("\nv2 - v1 CER_text over all single pages (registered reruns substituted), paired bootstrap:")
for label in ("none", "reject", "prefix", "prefix-w"):
    rows = base if label == "none" else [dict(policy_rows[label][s["item"]], item=s["item"], cell=s["cell"])
                                         if s["cell"] == "v1_single" and s["item"] in policy_rows[label] else s for s in base]
    b = h2h._bootstrap(h2h._paired(rows, "v2_single", "v1_single", "cer_text"))
    m1 = h2h._mean([s["cer_text"] for s in rows if s["cell"] == "v1_single" and "cer_text" in s])
    print(f"  {label:9} v1 mean {m1:.4f}; delta {b['mean']:+.4f} [{b['low']:+.4f}, {b['high']:+.4f}] (n={b['n']})")
