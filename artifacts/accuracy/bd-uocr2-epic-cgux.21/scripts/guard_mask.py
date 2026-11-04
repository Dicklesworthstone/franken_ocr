"""Would masking digit tokens let the runaway guard catch counting loops, and at what false-positive cost?

For every C5a output (v1 cells with v1's tokenizer, v2 cells with v2's), every
2,048-token window ending at 2,048 + 256k is analyzed twice with guard_calib's
`analyze`: on the raw ids, and with every all-digit token mapped to one id.
Outputs are labelled by text, not by the guard: runaway when the second half of
the text compresses to <= 0.10 of its size, legit when it is >= 0.20 and the run
finished (under 590 s for budgeted single pages). Gray-zone outputs are counted apart.

A window is "suspicious" by the guard's own predicate; an output "would fire" when
3 consecutive windows are suspicious (the guard's hysteresis), ignoring the
8,192-token floor, so the floor's protection is measured separately.

usage: guard_mask.py <c5a_data> <v1 tokenizer.json> <v2 tokenizer.json>
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I drops the script dir
import guard_calib as g  # noqa: E402
from tokenizers import Tokenizer

MASK = 4_000_000_000


def digit_ids(tok):
    vocab = tok.get_vocab()
    return {i for s, i in vocab.items() if s.strip("Ġ▁ ").isdigit()}


def fires(ids, start):
    """First window end (>= start) completing 3 consecutive suspicious windows, or None."""
    hits = 0
    end = g.WINDOW
    first = None
    while end <= len(ids):
        m = g.analyze(ids[:end])
        if m["suspicious"]:
            hits += 1
            if hits >= g.HITS and end >= start and first is None:
                first = end
        else:
            hits = 0
        end += g.INTERVAL
    return first


def main():
    data = pathlib.Path(sys.argv[1])
    toks = {"v1": Tokenizer.from_file(sys.argv[2]), "v2": Tokenizer.from_file(sys.argv[3])}
    digits = {k: digit_ids(t) for k, t in toks.items()}
    print({k: len(v) for k, v in digits.items()}, "all-digit tokens in each vocabulary")
    runs = [json.loads(line) for line in (data / "runs.ndjson").read_text().splitlines() if line.strip()]
    rerun_path = data / "rerun.ndjson"
    if rerun_path.exists():
        runs += [json.loads(line) for line in rerun_path.read_text().splitlines() if line.strip()]
    stats = {}
    shown = []
    for r in runs:
        fam = r["cell"][:2]
        budgeted = not r["run"].startswith("rerun/")
        for name, text, secs in g.pages_of(r["output"]):
            ids = toks[fam].encode(text, add_special_tokens=False).ids
            if len(ids) < g.WINDOW + 2 * g.INTERVAL:
                continue
            z = g.ratio(text)
            secs = secs if secs is not None else r.get("wall_s")
            finished = not (budgeted and r["cell"].endswith("single") and (secs or 0) >= 590)
            label = "runaway" if z <= 0.10 else ("legit" if z >= 0.20 and finished else "gray")
            masked = [MASK if t in digits[fam] else t for t in ids]
            raw_any, mask_any = fires(ids, 0), fires(masked, 0)
            key = (fam, label)
            s = stats.setdefault(key, {"n": 0, "raw": 0, "mask": 0, "mask_floor": 0})
            s["n"] += 1
            s["raw"] += raw_any is not None
            s["mask"] += mask_any is not None
            s["mask_floor"] += fires(masked, g.MIN_TOKENS) is not None
            if (label == "legit" and mask_any) or (label == "runaway" and not mask_any) or (raw_any is None and mask_any):
                shown.append((label, fam, len(ids), z, raw_any, mask_any, r["cell"], (name if name != "(document)" else r["run"])[:55]))
    print("family label: outputs >= 2,560 tokens | fire raw | fire masked | fire masked past the 8,192 floor")
    for (fam, label), s in sorted(stats.items()):
        print(f"  {fam} {label:8} {s['n']:>4} | {s['raw']:>3} | {s['mask']:>3} | {s['mask_floor']:>3}")
    print("cases: label fam tokens tail_z raw_fire_at mask_fire_at cell name")
    for x in sorted(shown):
        print(" ", x)


if __name__ == "__main__":
    main()
