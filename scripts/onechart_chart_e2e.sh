#!/bin/sh
# onechart_chart_e2e.sh — the D8 end-to-end chart→dict gate (bd-3jo6.4.8).
#
# Drives the REAL `focr` binary over the REAL int8 artifact:
#   1/4 GATE      zoo dir present (weights + vocab.json) else SKIP-with-SUCCESS
#   2/4 BIN       build (or take FOCR_BIN) — release profile
#   3/4 NEGATIVE  /nonexistent model => exit 3; chart-data at a got-named
#                 model => exit 2 (Usage)
#   4/4 CHART     --task chart-data on the committed chart => dict text with
#                 >= 2 of the four bar values (the measured OOD floor; the
#                 number head, certified separately, is the stable signal)
#
# Env: FOCR_ONECHART_DIR (default the USB mirror); FOCR_BIN (skips the build).
#
# Logging contract: stdout DATA-ONLY NDJSON, schema "onechart_chart_e2e/v1"
# (events gate|bin|negative|chart|result); human telemetry `OCHT `-prefixed on
# stderr. Exit 0 = PASS or gated SKIP; non-zero = a real divergence.
#
# POSIX sh; passes `sh -n`. python3 required (NDJSON emission).
set -eu

log()   { printf 'OCHT %s\n' "$*" >&2; }
step()  { printf 'OCHT ==== STEP %s ====\n' "$*" >&2; }
ok()    { printf 'OCHT   PASS  %s\n' "$*" >&2; }
skip()  { printf 'OCHT   SKIP  %s\n' "$*" >&2; }
fail()  { printf 'OCHT   FAIL  %s\n' "$*" >&2; }

ndj() {
  python3 - "$@" <<'PY'
import json, sys, time
rec = {"schema": "onechart_chart_e2e/v1", "ts_ms": int(time.time() * 1000)}
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    try:
        rec[k] = int(v)
    except ValueError:
        rec[k] = v
print(json.dumps(rec, sort_keys=True))
PY
}

now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found — required for NDJSON emission"
  exit 2
fi

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CHART="$REPO_ROOT/tests/fixtures/onechart/sample_chart.png"
ZOO="${FOCR_ONECHART_DIR:-/Volumes/USBNVME16TB/temp_agent_space/zoo/onechart}"
MODEL="$ZOO/onechart.int8.focrq"

step "1/4 GATE"
if [ ! -f "$MODEL" ] || [ ! -f "$ZOO/vocab.json" ]; then
  skip "onechart artifact/tokenizer absent under $ZOO — skipped with SUCCESS"
  ndj event=gate result=skip reason=no_weights zoo="$ZOO"
  ndj event=result result=skip
  exit 0
fi
if [ ! -f "$CHART" ]; then
  fail "committed fixture chart missing: $CHART"
  ndj event=gate result=fail reason=no_fixture_chart
  exit 1
fi
ok "artifact + tokenizer files + fixture chart present"
ndj event=gate result=pass zoo="$ZOO"

step "2/4 BIN"
if [ -n "${FOCR_BIN:-}" ]; then
  BIN="$FOCR_BIN"
else
  ( cd "$REPO_ROOT" && cargo build --release --bin focr ) >&2
  BIN="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/focr"
fi
if [ ! -x "$BIN" ]; then
  fail "focr binary not found/executable at $BIN"
  ndj event=bin result=fail path="$BIN"
  exit 1
fi
ok "binary ready: $BIN"
ndj event=bin result=pass path="$BIN"

step "3/4 NEGATIVE"
rc=0
"$BIN" ocr "$CHART" --task chart-data --model /nonexistent/onechart.focrq >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 3 ]; then
  fail "/nonexistent model must exit 3 (ModelNotFound); got $rc"
  ndj event=negative check=nonexistent_model result=fail exit_code="$rc" expect=3
  exit 1
fi
ok "/nonexistent model => exit 3"
ndj event=negative check=nonexistent_model result=pass exit_code="$rc"

rc=0
"$BIN" ocr "$CHART" --task chart-data --model got-ocr2.int8.focrq >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 2 ]; then
  fail "chart-data at a got-named model must exit 2 (Usage); got $rc"
  ndj event=negative check=wrong_family_guard result=fail exit_code="$rc" expect=2
  exit 1
fi
ok "chart-data at a got-named model => exit 2"
ndj event=negative check=wrong_family_guard result=pass exit_code="$rc"

step "4/4 CHART"
t0=$(now_ms)
OUT=$("$BIN" ocr "$CHART" --task chart-data --model "$MODEL" --max-length 256 2>/dev/null) || {
  rc=$?
  fail "chart-data run failed (exit $rc)"
  ndj event=chart result=fail exit_code="$rc"
  exit 1
}
t1=$(now_ms)
case "$OUT" in
  \{*) ok "output opens the chart dict" ;;
  *)
    fail "output does not open a dict: $OUT"
    ndj event=chart result=fail reason=no_dict
    exit 1
    ;;
esac
n_vals=0
for v in 30 45 25 10; do
  case "$OUT" in *"$v"*) n_vals=$((n_vals + 1)) ;; esac
done
if [ "$n_vals" -lt 2 ]; then
  fail "only $n_vals/4 bar values present (measured OOD floor is 2)"
  ndj event=chart result=fail reason=values_floor n_values="$n_vals"
  exit 1
fi
ok "dict text carries $n_vals/4 bar values (floor 2 on this OOD chart)"
ndj event=chart result=pass elapsed_ms=$((t1 - t0)) n_values="$n_vals"

log "ALL STEPS PASSED"
ndj event=result result=pass
exit 0
