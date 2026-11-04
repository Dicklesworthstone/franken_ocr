"""bd-gftg: live FOCR_RUNAWAY_GUARD=1 runs (guard_live/<id>.json) against the offline replay
(guard_policy.out: re-encoded tokens, replay abort token, rerun wall, token-share wall estimate).

ocr-batch exits 0 and puts a per-image failure in results[].error, so the trigger is read
from the JSON, not from stderr (guard_live.sh's own log line looked only at stderr).

usage: guard_live_summary.py <scratchpad>
"""
import json
import re
import sys
from pathlib import Path

S = Path(sys.argv[1])
replay = {}
for line in (S / "guard_policy.out").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 5 and parts[1].isdigit() and parts[2].isdigit():
        replay[parts[0]] = {"tokens": int(parts[1]), "abort": int(parts[2]), "wall": int(parts[3]), "est": int(parts[4])}
TRIGGER = re.compile(r"runaway token guard triggered at (\d+) emitted tokens.*?period=(\d+) match=(\d+)/(\d+)")
print("| page | replay abort | live abort | live wall s | replay est s | rerun wall s | period |")
print("|---|---|---|---|---|---|---|")
saved = rerun_total = 0.0
for path in sorted((S / "guard_live").glob("omni-*.json")):
    name = path.stem.removeprefix("omni-")
    if not path.read_text().strip():
        print(f"| {name[:44]} | running | | | | | |")
        continue
    res = json.loads(path.read_text())["results"][0]
    m = TRIGGER.search(res.get("error") or "")
    r = replay.get(name, {})
    live_abort = m.group(1) if m else ("no trigger, ok" if res.get("ok") else f"other: {(res.get('error') or '')[:60]}")
    print(f"| {name[:44]} | {r.get('abort', 'n/a')} | {live_abort} | {res['seconds']:.0f} | {r.get('est', 'n/a')} | "
          f"{r.get('wall', 'n/a')} | {m.group(2) if m else '-'} |")
    if r:
        saved += r["wall"] - res["seconds"]
        rerun_total += r["wall"]
print(f"\nwall saved against the reruns on replayed pages: {saved:.0f} of {rerun_total:.0f} s")
