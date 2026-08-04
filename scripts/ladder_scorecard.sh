#!/bin/sh
# ladder_scorecard.sh — the L0-L5 integration runner + scorecard (bd-re8.19).
#
# Executes the parity ladder IN ORDER (single-threaded test run: the rungs are
# named l0_..l5_ so alphabetical == ladder order) and folds the rungs' own
# structured NDJSON (event:"parity" rows + event:"result" outcomes) into ONE
# scorecard artifact — the per-commit parity receipt the phase exit gates and
# the three-pillar conformance pillar consume:
#
#   {"schema":"focr-ladder-scorecard/v1","gates":[{gate,outcome,parity_rows,
#    worst:{metric,value,tolerance}}...],"all_green":bool,
#    "skipped_no_model":bool,"receipt":"..."}
#
# Short-circuit semantics: every rung still RUNS (their internal gating is
# already correct); the scorecard marks rungs above the first failure
# "not_meaningful" so a single lower-gate break reads as ONE failure, not six.
#
# Model-gated: without weights the rungs emit skip_no_model SUCCESS lines
# (with the /nonexistent native-path proof) and the scorecard says so — a
# skipped ladder is NEVER mistaken for a green one.
#
# Usage: scripts/ladder_scorecard.sh [--out FILE]   (env: FOCR_MODEL_PATH,
#        FOCR_FIXTURES_DIR arm the rungs; --self-test folds a synthetic log)
set -eu

OUT=""
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

fold() {
  # argv1: the raw NDJSON file; argv2: output path ("" = stdout only)
  python3 - "$1" "$2" <<'PY'
import json, sys

raw_path, out_path = sys.argv[1], sys.argv[2]

LADDER = ["L0", "L1", "L2", "L3", "L4", "L5"]
rows: dict = {}
outcomes: dict = {}
skipped_no_model = False
for line in open(raw_path, encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        v = json.loads(line)
    except ValueError:
        continue
    test = str(v.get("test", ""))
    rung = next((l for l in LADDER if test.upper().startswith(l)), None)
    if rung is None:
        continue
    if v.get("event") == "parity":
        rows.setdefault(rung, []).append(v)
    elif v.get("event") == "result":
        outcomes.setdefault(rung, []).append(str(v.get("result", "")))
        if v.get("result") == "skip_no_model":
            skipped_no_model = True
    elif v.get("event") == "skip" and v.get("result") == "skip_no_model":
        skipped_no_model = True
        outcomes.setdefault(rung, []).append("skip_no_model")

gates = []
first_failure = None
for rung in LADDER:
    rung_rows = rows.get(rung, [])
    rung_outcomes = outcomes.get(rung, [])
    if any(o == "fail" for o in rung_outcomes) or any(not r.get("pass", True) for r in rung_rows):
        outcome = "fail"
    elif "xfail" in rung_outcomes:
        outcome = "xfail"
    elif "skip_no_model" in rung_outcomes and not rung_rows:
        outcome = "skip_no_model"
    elif rung_rows or "pass" in rung_outcomes:
        outcome = "pass"
    else:
        outcome = "not_run"
    if first_failure is not None:
        meaningful = False
    else:
        meaningful = True
        if outcome == "fail":
            first_failure = rung
    worst = None
    real = [r for r in rung_rows if isinstance(r.get("value"), (int, float))]
    if real:
        # worst = the row with the least headroom (value closest to / past tolerance)
        def headroom(r):
            tol = r.get("tolerance") or 0.0
            return (tol - r["value"]) if tol else -abs(r["value"])
        w = min(real, key=headroom)
        worst = {"metric": w.get("metric"), "value": w.get("value"), "tolerance": w.get("tolerance")}
    gates.append({
        "gate": rung,
        "outcome": outcome if meaningful else f"{outcome}(not_meaningful:below_{first_failure})",
        "meaningful": meaningful,
        "parity_rows": len(rung_rows),
        "worst": worst,
    })

all_green = all(g["outcome"] in ("pass", "xfail") for g in gates) and not skipped_no_model
receipt_bits = []
for g in gates:
    if g["worst"]:
        receipt_bits.append(f"{g['gate']} {g['worst']['metric']}={g['worst']['value']:.6g}")
    else:
        receipt_bits.append(f"{g['gate']} {g['outcome']}")
scorecard = {
    "schema": "focr-ladder-scorecard/v1",
    "gates": gates,
    "all_green": all_green,
    "skipped_no_model": skipped_no_model,
    "receipt": "; ".join(receipt_bits),
}
text = json.dumps(scorecard, indent=1)
print(text)
if out_path:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
PY
}

if [ "$SELF_TEST" -eq 1 ]; then
  # Synthetic fold: L0/L1 pass rows, L2 fail, L3 pass (must read not_meaningful),
  # L4 skip_no_model, L5 absent.
  SYN=$(mktemp "${TMPDIR:-/tmp}/focr-ladder-selftest.XXXXXX.ndjson")
  cat > "$SYN" <<'NDJSON'
{"test":"L0_preprocess","event":"parity","gate":"sam_input","metric":"max_abs","value":0.0,"tolerance":0.0001,"pass":true}
{"test":"L0_preprocess","event":"result","result":"pass"}
{"test":"L1_per_op","event":"parity","gate":"sam","metric":"cosine","value":0.99999,"tolerance":0.9999,"pass":true}
{"test":"L2_per_layer","event":"parity","gate":"layer3","metric":"cosine","value":0.5,"tolerance":0.9999,"pass":false}
{"test":"L3_logits","event":"parity","gate":"logits","metric":"max_abs","value":0.01,"tolerance":0.05,"pass":true}
{"test":"L4_tokens","event":"result","result":"skip_no_model"}
NDJSON
  SC=$(fold "$SYN" "")
  rm -f "$SYN"
  python3 -c "
import json, sys
sc = json.loads(sys.argv[1])
gates = {g['gate']: g for g in sc['gates']}
assert gates['L0']['outcome'] == 'pass', gates['L0']
assert gates['L2']['outcome'] == 'fail', gates['L2']
assert not gates['L3']['meaningful'] and 'not_meaningful' in gates['L3']['outcome'], gates['L3']
assert sc['skipped_no_model'] is True
assert sc['all_green'] is False
assert 'L1 cosine=0.99999' in sc['receipt']
print(json.dumps({'check': 'ladder-scorecard-self-test', 'result': 'pass'}))
" "$SC"
  exit 0
fi

RAW=$(mktemp "${TMPDIR:-/tmp}/focr-ladder.XXXXXX.ndjson")
trap 'rm -f "$RAW"' EXIT
echo "ladder_scorecard: running the L0-L5 rungs (single-threaded, ordered)" >&2
( cd "$REPO_ROOT" && cargo test --release --test parity_ladder -- --test-threads=1 --nocapture ) 2> "$RAW" >&2 || {
  echo "ladder_scorecard: a rung FAILED — the scorecard below carries the break" >&2
}
# Keep the raw NDJSON beside the scorecard: the fold is a summary, the raw
# rows are the evidence (and the only diagnostic when a rung dies pre-parity).
if [ -n "$OUT" ]; then
  cp "$RAW" "${OUT%.json}.raw.ndjson"
fi
fold "$RAW" "${OUT}"
