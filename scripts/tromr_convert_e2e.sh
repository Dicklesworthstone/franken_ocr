#!/bin/sh
# tromr_convert_e2e.sh — the E2 convert gate (bd-3jo6.5.2).
#
# Drives the REAL `focr convert` over the REAL WS-folded TrOMR export:
#   1/6 GATE        zoo export present (model.safetensors + the 4 tokenizer
#                   JSONs) else SKIP-with-SUCCESS
#   2/6 BIN         build (or take FOCR_BIN) — release profile
#   3/6 NEGATIVE    convert /nonexistent => exit 3 (ModelNotFound);
#                   --model-id not-a-model => exit 2 (Usage)
#   4/6 CONVERT     model.safetensors -> tromr.focrq; JSON must report
#                   status=ok, model_id=tromr, tensors=260,
#                   tensors_quantized=0 (the default-f32 policy, spec §11)
#   5/6 DETERMINISM re-convert to a scratch path => byte-identical artifact
#   6/6 TOKENIZERS  the 4 WordLevel tables beside the artifact byte-match the
#                   COMMITTED conformance fixtures (tests/fixtures/tromr) —
#                   the E6 id-exactness anchor (bd-3jo6.5.6)
#
# Env: FOCR_TROMR_DIR (default the USB zoo); FOCR_BIN (skips the build).
#
# Logging contract: stdout DATA-ONLY NDJSON, schema "tromr_convert_e2e/v1"
# (events gate|bin|negative|convert|determinism|result); human telemetry
# `TRMR `-prefixed on stderr. Exit 0 = PASS or gated SKIP; non-zero = a real
# divergence.
#
# POSIX sh; passes `sh -n`. python3 required (NDJSON emission + JSON checks).
set -eu

log()   { printf 'TRMR %s\n' "$*" >&2; }
step()  { printf 'TRMR ==== STEP %s ====\n' "$*" >&2; }
ok()    { printf 'TRMR   PASS  %s\n' "$*" >&2; }
skip()  { printf 'TRMR   SKIP  %s\n' "$*" >&2; }
fail()  { printf 'TRMR   FAIL  %s\n' "$*" >&2; }

ndj() {
  python3 - "$@" <<'PY'
import json, sys, time
rec = {"schema": "tromr_convert_e2e/v1", "ts_ms": int(time.time() * 1000)}
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    try:
        rec[k] = int(v)
    except ValueError:
        rec[k] = v
print(json.dumps(rec, sort_keys=True))
PY
}

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found — required for NDJSON emission"
  exit 2
fi

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ZOO="${FOCR_TROMR_DIR:-/Volumes/USBNVME16TB/temp_agent_space/zoo/tromr}"
SRC="$ZOO/model.safetensors"

step "1/6 GATE"
if [ ! -f "$SRC" ] || [ ! -f "$ZOO/tokenizer_rhythm.json" ]; then
  skip "TrOMR export/tokenizers absent under $ZOO — skipped with SUCCESS"
  ndj event=gate result=skip reason=no_weights zoo="$ZOO"
  ndj event=result result=skip
  exit 0
fi
ok "WS-folded export + tokenizer tables present"
ndj event=gate result=pass zoo="$ZOO"

step "2/6 BIN"
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

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/tromr-convert-e2e.XXXXXX")
trap 'rm -r "$SCRATCH" 2>/dev/null || true' EXIT

step "3/6 NEGATIVE"
rc=0
"$BIN" convert /nonexistent/tromr/model.safetensors --output "$SCRATCH/x.focrq" \
  --model-id tromr --json >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 3 ]; then
  fail "/nonexistent input must exit 3 (ModelNotFound); got $rc"
  ndj event=negative check=nonexistent_input result=fail exit_code="$rc" expect=3
  exit 1
fi
ok "/nonexistent input => exit 3"
ndj event=negative check=nonexistent_input result=pass exit_code="$rc"

rc=0
"$BIN" convert "$SRC" --output "$SCRATCH/x.focrq" --model-id not-a-model --json >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 2 ]; then
  fail "unknown --model-id must exit 2 (Usage); got $rc"
  ndj event=negative check=unknown_model_id result=fail exit_code="$rc" expect=2
  exit 1
fi
ok "unknown --model-id => exit 2"
ndj event=negative check=unknown_model_id result=pass exit_code="$rc"

step "4/6 CONVERT"
OUT_JSON=$("$BIN" convert "$SRC" --output "$SCRATCH/tromr.focrq" --model-id tromr --json 2>/dev/null) || {
  rc=$?
  fail "convert failed (exit $rc)"
  ndj event=convert result=fail exit_code="$rc"
  exit 1
}
python3 -c '
import json, sys
d = json.loads(sys.argv[1])
assert d["status"] == "ok", d
assert d["model_id"] == "tromr", d
assert d["tensors"] == 260, d          # census 12 minus note_mask
assert d["tensors_quantized"] == 0, d  # default-f32 policy (spec 11)
' "$OUT_JSON" || { fail "convert JSON contract violated"; ndj event=convert result=fail reason=json_contract; exit 1; }
ok "convert ok: 260 tensors, 0 int8, model_id=tromr"
ndj event=convert result=pass tensors=260 tensors_quantized=0

step "5/6 DETERMINISM"
"$BIN" convert "$SRC" --output "$SCRATCH/tromr2.focrq" --model-id tromr --json >/dev/null 2>&1
if ! cmp -s "$SCRATCH/tromr.focrq" "$SCRATCH/tromr2.focrq"; then
  fail "re-convert produced different bytes — convert is nondeterministic"
  ndj event=determinism result=fail
  exit 1
fi
ok "re-convert byte-identical"
ndj event=determinism result=pass

step "6/6 TOKENIZERS"
for stream in rhythm pitch lift note; do
  if ! cmp -s "$ZOO/tokenizer_$stream.json" "$REPO_ROOT/tests/fixtures/tromr/tokenizer_$stream.json"; then
    fail "zoo tokenizer_$stream.json differs from the committed conformance fixture"
    ndj event=tokenizers result=fail stream="$stream"
    exit 1
  fi
done
ok "all 4 WordLevel tables byte-match the committed fixtures"
ndj event=tokenizers result=pass tables=4

log "ALL STEPS PASSED"
ndj event=result result=pass
exit 0
