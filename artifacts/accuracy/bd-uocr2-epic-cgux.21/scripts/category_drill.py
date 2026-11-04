"""Per-page CER_text for the single-page categories where v2 regressed (as run)."""
import json
import pathlib
import sys

D = pathlib.Path(sys.argv[1])
items = json.loads((D / "items.json").read_text())["single_pages"]
by_id = {i["id"]: i for i in items}
cer = {}
for line in (D / "scores.ndjson").read_text().splitlines():
    r = json.loads(line)
    if r["cell"] in ("v1_single", "v2_single") and r.get("cer_text") is not None:
        cer.setdefault(r["item"], {})[r["cell"]] = (r["cer_text"], r.get("pred_chars"), r.get("gt_chars"), r.get("failed"))

print("item keys:", sorted(items[0].keys()))
for key, value in (("data_source", "note"), ("language", "en_ch_mixed")):
    rows = []
    for iid, c in cer.items():
        it = by_id.get(iid, {})
        attrs = json.dumps(it, ensure_ascii=False)
        if value not in attrs or len(c) < 2:
            continue
        (v1, p1, g, f1), (v2, p2, _, f2) = c["v1_single"], c["v2_single"]
        rows.append((v2 - v1, iid, v1, v2, p1, p2, g, f1, f2))
    rows.sort(reverse=True)
    print(f"\n{key}={value}: n={len(rows)} mean delta={sum(r[0] for r in rows) / max(len(rows), 1):+.4f}")
    for r in rows[:10]:
        print(f"  d={r[0]:+.3f} v1={r[2]:.3f} v2={r[3]:.3f} chars v1/v2/gt={r[4]}/{r[5]}/{r[6]} failed={r[7]}/{r[8]} {r[1]}")
    print("  ... best for v2:", [(round(r[0], 3), r[1][:40]) for r in rows[-3:]])
