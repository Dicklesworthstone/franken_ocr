"""Offline replay of the v1 RunawayGuard (src/native_engine/sampler.rs) on U2-C5a outputs.

The port mirrors `analyze_runaway_suffix` and `RunawayGuard::observe`: checkpoints
at 8,192 tokens and every 256 after, a 2,048-token suffix, best period 1..256 by
lagged-match ratio (ties to the shorter period), suspicious when matches >= 15/16
of comparisons AND unique token 4-grams <= 1/4 of all 4-grams, abort after 3
consecutive suspicious checkpoints. Self-tests replay the Rust unit-test vectors.

Caveat: outputs are post-processed text, re-encoded with v1's tokenizer, so token
counts and ids approximate what the decoder emitted.

usage: guard_calib.py <c5a_data dir> <v1 tokenizer.json>
"""
import json
import pathlib
import sys
import zlib

import numpy as np
from tokenizers import Tokenizer

MIN_TOKENS, INTERVAL, WINDOW, MAX_PERIOD, MIN_CYCLES = 8192, 256, 2048, 256, 8
MATCH_NUM, MATCH_DEN, NGRAM, NOV_NUM, NOV_DEN, HITS = 15, 16, 4, 1, 4, 3


def analyze(emitted):
    if len(emitted) < WINDOW:
        return None
    w = np.asarray(emitted[-WINDOW:], dtype=np.int64)
    max_period = min(MAX_PERIOD, len(w) // MIN_CYCLES)
    best = (1, 0, len(w) - 1)  # period, matches, comparisons
    for p in range(1, max_period + 1):
        comps = len(w) - p
        matches = int((w[p:] == w[:-p]).sum())
        lhs, rhs = matches * best[2], best[1] * comps
        if lhs > rhs or (lhs == rhs and p < best[0]):
            best = (p, matches, comps)
    grams = {tuple(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}
    total = len(w) - NGRAM + 1
    suspicious = best[1] * MATCH_DEN >= best[2] * MATCH_NUM and len(grams) * NOV_DEN <= total * NOV_NUM
    return {"period": best[0], "match": best[1] / best[2], "novelty": len(grams) / total, "suspicious": suspicious}


def replay(ids):
    """Token index at which the guard aborts (the checkpoint length), or None; plus last metrics."""
    hits, last = 0, None
    cp = MIN_TOKENS
    while cp <= len(ids):
        m = analyze(ids[:cp])
        last = m
        if m and m["suspicious"]:
            hits += 1
            if hits >= HITS:
                return cp, m
        else:
            hits = 0
        cp += INTERVAL
    return None, last


def selftest():
    periodic = lambda n, p: [i % p for i in range(n)]
    n = MIN_TOKENS + 2 * INTERVAL
    for p in (19, 23, 29):
        at, m = replay(periodic(n, p))
        assert at == n and m["period"] == p, (p, at, m)
    at, m = replay(list(range(32768)))
    assert at is None and m["novelty"] == 1.0 and not m["suspicious"]
    toks = periodic(MIN_TOKENS, 32) + [1_000_000 + i for i in range(INTERVAL)]
    assert analyze(toks[:MIN_TOKENS])["suspicious"] and not analyze(toks)["suspicious"]
    assert replay(toks)[0] is None
    print("selftest ok (periodic 19/23/29 abort at 8,704; high novelty and broken run continue)")


def ratio(text):
    b = text.encode()
    tail = b[len(b) // 2:]
    return round(len(zlib.compress(tail, 9)) / max(1, len(tail)), 3)


def pages_of(output):
    try:
        d = json.loads(pathlib.Path(output).read_text())
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(d, dict) and isinstance(d.get("results"), list):
        for x in d["results"]:
            yield pathlib.Path(x.get("image", "?")).name, x.get("markdown") or "", x.get("seconds")
    elif isinstance(d, dict):
        text = d.get("markdown") or d.get("text") or ""
        yield "(document)", text, None


def main():
    selftest()
    data, tok_path = pathlib.Path(sys.argv[1]), sys.argv[2]
    tok = Tokenizer.from_file(tok_path)
    runs = [json.loads(line) for line in (data / "runs.ndjson").read_text().splitlines() if line.strip()]
    rerun_path = data / "rerun.ndjson"
    reruns = [json.loads(line) for line in rerun_path.read_text().splitlines() if line.strip()] if rerun_path.exists() else []
    rows = []
    for r in runs + reruns:
        if not r["cell"].startswith("v1"):
            continue
        source = "rerun" if r["run"].startswith("rerun/") else "run"
        for name, text, secs in pages_of(r["output"]):
            ids = tok.encode(text, add_special_tokens=False).ids
            at, m = replay(ids)
            rows.append({
                "cell": r["cell"], "source": source, "name": name if name != "(document)" else r["run"],
                "tokens": len(ids), "chars": len(text), "seconds": secs if secs is not None else r.get("wall_s"),
                "tail_zratio": ratio(text), "abort_at": at,
                "period": m and m["period"], "match": m and round(m["match"], 3), "novelty": m and round(m["novelty"], 3),
            })
    out = data.parent / "guard_calib.ndjson"
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(len(rows), "v1 outputs ->", out)
    long_ = [x for x in rows if x["tokens"] >= MIN_TOKENS]
    print(f"{len(long_)} reach {MIN_TOKENS} tokens (the guard looks at nothing shorter)")
    print("abort_at  tokens  chars  sec  tail_zratio  period match novelty  cell/source  name")
    for x in sorted(long_, key=lambda x: (x["abort_at"] is None, x["tail_zratio"])):
        print(f"{str(x['abort_at']):>8} {x['tokens']:>7} {x['chars']:>7} {round(x['seconds'] or 0):>5} {x['tail_zratio']:>6}"
              f"  {x['period']!s:>4} {x['match']!s:>6} {x['novelty']!s:>6}  {x['cell']}/{x['source']}  {x['name'][:60]}")


if __name__ == "__main__":
    main()
