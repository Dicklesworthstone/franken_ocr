"""bd-gftg work item 2: false positives of the runaway detectors on ground truth.

Ground-truth transcriptions are legit by construction, so any fire on them is a
false positive. Streams are each page's truth in reading order (text, table HTML,
display LaTeX, furniture included, as a model would write them), re-encoded with
each model's tokenizer:
  omni      OmniDocBench's 1,651 single pages
  mpage     every MarathonDocBench page (what a v2 per-page detector sees)
  mdoc      every MarathonDocBench document as one stream (a v1 multi-page decode)
Detectors are guard_calib's predicate (15/16 period match and 4-gram novelty <= 1/4,
3 consecutive 256-token checkpoints over a 2,048-token window), on raw ids and
with all-digit tokens mapped to one id, reported at floors 2,048 (none), 2,560 and 8,192.

usage: truth_fp.py <OmniDocBench.json> <MarathonDocBench.json> <v1 tokenizer.json> <v2 tokenizer.json> <out.ndjson>
"""
import collections
import json
import sys

import numpy as np
from tokenizers import Tokenizer

WINDOW, INTERVAL, MAX_PERIOD, MIN_CYCLES, HITS = 2048, 256, 256, 8, 3
FLOORS = (2048, 2560, 8192)
MASK = 4_000_000_000


def suspicious(w):
    """guard_calib.analyze's predicate; novelty only computed when the period test passes."""
    max_period = min(MAX_PERIOD, len(w) // MIN_CYCLES)
    best = (1, 0, len(w) - 1)
    for p in range(1, max_period + 1):
        comps = len(w) - p
        matches = int((w[p:] == w[:-p]).sum())
        lhs, rhs = matches * best[2], best[1] * comps
        if lhs > rhs or (lhs == rhs and p < best[0]):
            best = (p, matches, comps)
    if best[1] * 16 < best[2] * 15:
        return False
    total = len(w) - 3
    grams = {(int(w[i]), int(w[i + 1]), int(w[i + 2]), int(w[i + 3])) for i in range(total)}
    return len(grams) * 4 <= total


def first_fires(ids):
    """{floor: first checkpoint >= floor completing HITS consecutive suspicious windows}."""
    arr = np.asarray(ids, dtype=np.int64)
    out, hits, end = {}, 0, WINDOW
    while end <= len(arr) and len(out) < len(FLOORS):
        if suspicious(arr[end - WINDOW:end]):
            hits += 1
            if hits >= HITS:
                for f in FLOORS:
                    if end >= f and f not in out:
                        out[f] = end
        else:
            hits = 0
        end += INTERVAL
    return out


def det_text(det):
    if det.get("category_type") == "table":
        return det.get("html") or det.get("text") or ""
    if det.get("category_type") == "equation_isolated":
        return det.get("latex") or det.get("text") or ""
    return det.get("text") or ""


def box_key(det):
    poly = det.get("poly") or [0, 0]
    return (min(poly[1::2] or [0]), min(poly[0::2] or [0]))


def ordered(dets):
    live = [d for d in dets if not d.get("ignore")]
    return sorted((d for d in live if d.get("order") is not None), key=lambda d: d["order"]) + \
        sorted((d for d in live if d.get("order") is None), key=box_key)


def page_text(dets):
    return "\n\n".join(t for t in (det_text(d) for d in ordered(dets)) if t)


def main():
    omni_json, marathon_json, v1_tok, v2_tok, out_path = sys.argv[1:6]
    toks = {"v1": Tokenizer.from_file(v1_tok), "v2": Tokenizer.from_file(v2_tok)}
    digits = {k: {i for s, i in t.get_vocab().items() if s.strip("Ġ▁ ").isdigit()} for k, t in toks.items()}
    streams = []
    for p in json.load(open(omni_json, encoding="utf-8")):
        streams.append(("omni", p["page_info"]["image_path"], page_text(p["layout_dets"])))
    for d in json.load(open(marathon_json, encoding="utf-8")):
        name = d["page_info"]["image_path"]
        by_page = collections.defaultdict(list)
        for det in d["layout_dets"]:
            by_page[det.get("page_id")].append(det)
        pages = [page_text(by_page[k]) for k in sorted(by_page, key=lambda k: (k is None, k or 0))]
        for k, text in zip(sorted(by_page, key=lambda k: (k is None, k or 0)), pages):
            streams.append(("mpage", f"{name}#p{k}", text))
        streams.append(("mdoc", name, "\n\n".join(pages)))
    with open(out_path, "w", encoding="utf-8") as out:
        for kind, name, text in streams:
            row = {"kind": kind, "name": name, "chars": len(text)}
            for model, tok in toks.items():
                ids = tok.encode(text, add_special_tokens=False).ids
                row[f"{model}_tokens"] = len(ids)
                if len(ids) < WINDOW:
                    continue
                masked = [MASK if t in digits[model] else t for t in ids]
                for det, seq in (("raw", ids), ("masked", masked)):
                    for f, at in first_fires(seq).items():
                        row[f"{model}_{det}_{f}"] = at
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = [json.loads(line) for line in open(out_path, encoding="utf-8")]
    for kind in ("omni", "mpage", "mdoc"):
        ks = [r for r in rows if r["kind"] == kind]
        print(f"\n{kind}: {len(ks)} streams")
        for model in ("v1", "v2"):
            long_ = {f: sum(r[f"{model}_tokens"] >= f for r in ks) for f in FLOORS}
            print(f"  {model}: streams reaching 2,048/2,560/8,192 tokens: {long_[2048]}/{long_[2560]}/{long_[8192]}")
            for det in ("raw", "masked"):
                fp = {f: [r["name"] for r in ks if f"{model}_{det}_{f}" in r] for f in FLOORS}
                print(f"    {det:6} fires (floor 2,048/2,560/8,192): {len(fp[2048])}/{len(fp[2560])}/{len(fp[8192])}"
                      + (f"  e.g. {fp[2048][:4]}" if fp[2048] else ""))


if __name__ == "__main__":
    main()
