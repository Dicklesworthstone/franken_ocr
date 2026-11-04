"""Replay truth_fp's raw and digit-masked detectors on U2-F1 outputs (re-encoded .md files).

Prints each output's token count and first fire point per detector at floors 2,048/2,560/8,192.
Then, for the Marathon truth pages whose fired window decoded to '' with special tokens
skipped, prints a repr of the truth text around the fire point.

usage: u2f1_guard_replay.py <u2f1 dir> <v1 tokenizer.json> <v2 tokenizer.json> <MarathonDocBench.json>
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I drops the script dir
import truth_fp as t  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

out_dir, v1_tok, v2_tok, marathon_json = sys.argv[1:5]
toks = {"unlimited-ocr": Tokenizer.from_file(v1_tok), "unlimited-ocr-2": Tokenizer.from_file(v2_tok)}
digits = {k: {i for s, i in tk.get_vocab().items() if s.strip("Ġ▁ ").isdigit()} for k, tk in toks.items()}
for md in sorted(pathlib.Path(out_dir).glob("*.r1.md")):
    model = md.name.split(".")[2]
    ids = toks[model].encode(md.read_text(errors="replace"), add_special_tokens=False).ids
    masked = [t.MASK if i in digits[model] else i for i in ids]
    raw_f, masked_f = (t.first_fires(s) if len(s) >= t.WINDOW else {} for s in (ids, masked))
    fmt = lambda d: "/".join(str(d.get(f, "-")) for f in t.FLOORS)  # noqa: E731
    print(f"{md.name}: {len(ids)} tok; raw {fmt(raw_f)}; masked {fmt(masked_f)}")

want = {"mp3lan_0792.pdf#p24", "mp3lan_0821.pdf#p147", "mp3lan_0791.pdf#p63", "mp3lan_0791.pdf#p54"}
tok = toks["unlimited-ocr"]
for d in json.load(open(marathon_json, encoding="utf-8")):
    name = d["page_info"]["image_path"]
    by_page = collections.defaultdict(list)
    for det in d["layout_dets"]:
        by_page[det.get("page_id")].append(det)
    for k, dets in by_page.items():
        if f"{name}#p{k}" in want:
            text = t.page_text(dets)
            enc = tok.encode(text, add_special_tokens=False)
            fire = t.first_fires(enc.ids).get(2048, len(enc.ids))
            lo = enc.offsets[fire - 2048][0]
            print(f"\n{name}#p{k}: {len(text)} chars; fired window starts at char {lo}")
            print("   ", repr(text[lo:lo + 300]))
