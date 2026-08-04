#!/bin/sh
# smolvlm2_describe_e2e.sh — the C10 end-to-end describe/VQA gate
# (bd-3jo6.3.10; C8's e2e-script requirement rides along).
#
# Drives the REAL `focr` binary over the REAL int8 artifact:
#   1/5 GATE      zoo dir present (weights + tokenizer.json) else SKIP-with-SUCCESS
#   2/5 BIN       build (or take FOCR_BIN) — release profile
#   3/5 NEGATIVE  /nonexistent model => exit 3 (ModelNotFound, TL5 honesty);
#                 describe pointed at a knowably-got model => exit 2 (Usage)
#   4/5 DESCRIBE  --task describe on the committed sample photo => scene answer
#   5/5 VQA       --question "Is there a sun in the image?" => affirmative
#
# Env:
#   FOCR_SMOLVLM2_DIR   zoo dir (default the USB mirror); needs
#                       smolvlm2.int8.focrq + tokenizer.json
#   FOCR_BIN            prebuilt focr (skips the cargo build)
#
# Logging contract (AGENTS.md "Agent Ergonomics" / docs/testing/LOGGING_AND_E2E.md):
#   * stdout is DATA-ONLY: one NDJSON object per line, schema
#     "smolvlm2_describe_e2e/v1" (events: gate|bin|negative|describe|vqa|result).
#   * ALL human telemetry is `SVLM `-prefixed on stderr (grep '^SVLM ').
#   * Exit 0 = PASS or gated SKIP; non-zero = a real divergence.
#
# POSIX sh; passes `sh -n`. python3 required (NDJSON emission).
set -eu

log()   { printf 'SVLM %s\n' "$*" >&2; }
step()  { printf 'SVLM ==== STEP %s ====\n' "$*" >&2; }
info()  { printf 'SVLM   %s\n' "$*" >&2; }
ok()    { printf 'SVLM   PASS  %s\n' "$*" >&2; }
skip()  { printf 'SVLM   SKIP  %s\n' "$*" >&2; }
fail()  { printf 'SVLM   FAIL  %s\n' "$*" >&2; }

ndj() {
  python3 - "$@" <<'PY'
import json, sys, time
rec = {"schema": "smolvlm2_describe_e2e/v1", "ts_ms": int(time.time() * 1000)}
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
PHOTO="$REPO_ROOT/tests/fixtures/smolvlm2/sample_photo.png"
ZOO="${FOCR_SMOLVLM2_DIR:-/Volumes/USBNVME16TB/temp_agent_space/zoo/smolvlm2}"
MODEL="$ZOO/smolvlm2.int8.focrq"

# ── STEP 1/5: gate ───────────────────────────────────────────────────────────
step "1/5 GATE"
if [ ! -f "$MODEL" ] || [ ! -f "$ZOO/tokenizer.json" ]; then
  skip "smolvlm2 artifact/tokenizer absent under $ZOO — skipped with SUCCESS"
  ndj event=gate result=skip reason=no_weights zoo="$ZOO"
  ndj event=result result=skip
  exit 0
fi
if [ ! -f "$PHOTO" ]; then
  fail "committed fixture image missing: $PHOTO"
  ndj event=gate result=fail reason=no_fixture_photo
  exit 1
fi
ok "artifact + tokenizer + fixture photo present"
ndj event=gate result=pass zoo="$ZOO"

# ── STEP 2/5: binary ─────────────────────────────────────────────────────────
step "2/5 BIN"
if [ -n "${FOCR_BIN:-}" ]; then
  BIN="$FOCR_BIN"
  info "using FOCR_BIN=$BIN"
else
  info "cargo build --release (dominated by first-build cost; cached after)"
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

# ── STEP 3/5: negative proofs (the native path is REAL, not a fallback) ──────
step "3/5 NEGATIVE"
rc=0
"$BIN" ocr "$PHOTO" --task describe --model /nonexistent/smolvlm2.focrq >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 3 ]; then
  fail "/nonexistent model must exit 3 (ModelNotFound); got $rc — a false green (TL5)"
  ndj event=negative check=nonexistent_model result=fail exit_code="$rc" expect=3
  exit 1
fi
ok "/nonexistent model => exit 3 (ModelNotFound)"
ndj event=negative check=nonexistent_model result=pass exit_code="$rc"

rc=0
"$BIN" ocr "$PHOTO" --task describe --model got-ocr2.int8.focrq >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 2 ]; then
  fail "describe at a knowably-got model must exit 2 (Usage); got $rc"
  ndj event=negative check=wrong_family_guard result=fail exit_code="$rc" expect=2
  exit 1
fi
ok "describe at a got-named model => exit 2 (Usage guidance)"
ndj event=negative check=wrong_family_guard result=pass exit_code="$rc"

# ── STEP 4/5: describe (default caption question) ────────────────────────────
step "4/5 DESCRIBE"
t0=$(now_ms)
ANSWER=$("$BIN" ocr "$PHOTO" --task describe --model "$MODEL" 2>/dev/null) || {
  rc=$?
  fail "describe run failed (exit $rc)"
  ndj event=describe result=fail exit_code="$rc"
  exit 1
}
t1=$(now_ms)
info "answer: $ANSWER"
case "$ANSWER" in
  *[a-zA-Z]*) : ;;
  *)
    fail "describe answer is empty/non-text"
    ndj event=describe result=fail reason=empty_answer
    exit 1
    ;;
esac
case "$ANSWER" in
  *ity*|*uilding*|*sky*|*Sky*|*scene*|*Scene*) ok "caption names the scene" ;;
  *)
    fail "caption mentions none of city/building/sky/scene: $ANSWER"
    ndj event=describe result=fail reason=off_topic answer="$ANSWER"
    exit 1
    ;;
esac
ndj event=describe result=pass elapsed_ms=$((t1 - t0)) answer="$ANSWER"

# ── STEP 5/5: VQA (custom --question) ────────────────────────────────────────
step "5/5 VQA"
t0=$(now_ms)
VQA=$("$BIN" ocr "$PHOTO" --task describe --model "$MODEL" \
  --question "Is there a sun in the image?" 2>/dev/null) || {
  rc=$?
  fail "vqa run failed (exit $rc)"
  ndj event=vqa result=fail exit_code="$rc"
  exit 1
}
t1=$(now_ms)
info "answer: $VQA"
case "$VQA" in
  *[Yy]es*|*sun*|*Sun*) ok "affirmative sun answer" ;;
  *)
    fail "expected an affirmative sun answer, got: $VQA"
    ndj event=vqa result=fail reason=wrong_answer answer="$VQA"
    exit 1
    ;;
esac
ndj event=vqa result=pass elapsed_ms=$((t1 - t0)) answer="$VQA"

log "ALL STEPS PASSED"
ndj event=result result=pass
exit 0
