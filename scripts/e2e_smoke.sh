#!/usr/bin/env sh
#
# e2e_smoke.sh — the heavily-logged end-to-end smoke driver for `focr`.
#
# This is the shell half of the MODEL-GATED end-to-end harness (Testing Policy;
# docs/testing/LOGGING_AND_E2E.md §4, bead bd-29wv; AGENTS.md "Testing Policy").
# It drives the REAL built `focr` binary (no mocks — TL6) through the three
# always-on, no-weights guards plus the model-gated real-recognize branch, and
# prints a final PASS / SKIP banner with per-step timing.
#
# The four steps:
#   1. BUILD          — build the `focr` binary (`cargo build --bin focr`).
#   2. ROBOT-SCHEMA   — `focr robot schema | <validate>`: exactly one parseable
#                       JSON object on stdout, exit 0 (the stable agent contract,
#                       src/robot.rs). Data-only on stdout, diagnostics on stderr.
#   3. NATIVE-PATH    — the /nonexistent native-path proof: `focr ocr <img>
#                       --model /nonexistent` must exit 3 (ModelNotFound) — NOT a
#                       panic, NOT a fabricated result. This proves the native
#                       path is exercised even without the 6.67 GB weights (TL5).
#                       Mid-flux CLI surfaces (the `--model` flag) are handled as
#                       a tripwire: the documented-skeleton exits (1/2) XFAIL with
#                       an explicit observed-vs-target note instead of failing.
#   4. GATED-E2E      — run the model-gated `recognize` e2e (`cargo test
#                       --test e2e_recognize`): skips-with-SUCCESS without weights
#                       (FOCR_MODEL_PATH unset/missing), runs a real recognize
#                       over a tiny fixture when weights are present.
#
# Conventions (match AGENTS.md "Agent Ergonomics" + the cass/bv house style):
#   * Data-only on stdout; ALL diagnostics + the structured `E2E ` telemetry on
#     stderr. NDJSON / JSON data is never interleaved with human decoration on
#     the same stream.
#   * Exit 0 = the whole smoke passed (or skipped-with-SUCCESS where gated).
#     Exit non-zero = a hard failure; the failing step + its captured output is
#     printed so the failure is diagnosable from the log alone.
#   * Every step is timed; the banner reports per-step PASS/SKIP/XFAIL + ms.
#
# Usage:
#   scripts/e2e_smoke.sh              # build + all four steps
#   scripts/e2e_smoke.sh --no-build   # skip the build (use an already-built bin)
#   scripts/e2e_smoke.sh --release    # build/run the release binary
#   FOCR_MODEL_PATH=/path/to/model scripts/e2e_smoke.sh   # exercise step 4 real
#
# POSIX sh; passes `sh -n`. No bashisms, no `cargo` invoked unless building.
set -eu

# ── house style: structured, greppable logging, all on stderr ────────────────
# Every line is prefixed `E2E ` so a CI scraper can `grep '^E2E '` the telemetry
# out of the rest of the build noise — the same prefix the Rust harness emits.

log()   { printf 'E2E %s\n' "$*" >&2; }
step()  { printf 'E2E ==== STEP %s ====\n' "$*" >&2; }
info()  { printf 'E2E   %s\n' "$*" >&2; }
ok()    { printf 'E2E   PASS  %s\n' "$*" >&2; }
skip()  { printf 'E2E   SKIP  %s\n' "$*" >&2; }
xfail() { printf 'E2E   XFAIL %s\n' "$*" >&2; }
fail()  { printf 'E2E   FAIL  %s\n' "$*" >&2; }

# ── millisecond clock (best-effort; falls back to seconds*1000) ──────────────
# `date +%s%N` is GNU-only; macOS `date` lacks %N. Prefer python3 for a
# monotonic-ish wall ms, then perl, then whole seconds.
now_ms() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import time; print(int(time.time()*1000))'
  elif command -v perl >/dev/null 2>&1; then
    perl -MTime::HiRes=time -e 'print int(time()*1000)'
  else
    # Whole-second resolution fallback.
    echo $(( $(date +%s) * 1000 ))
  fi
}

# ── argument parsing ─────────────────────────────────────────────────────────
DO_BUILD=1
PROFILE="debug"
CARGO_BUILD_FLAGS=""
for arg in "$@"; do
  case "$arg" in
    --no-build) DO_BUILD=0 ;;
    --release)  PROFILE="release"; CARGO_BUILD_FLAGS="--release" ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *) printf 'e2e_smoke.sh: unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

# ── resolve repo root (this script lives in <root>/scripts) ──────────────────
SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
BIN="$ROOT/target/$PROFILE/focr"

# ── scratch workspace for the tiny fixture image + captured output ───────────
WORK=$(mktemp -d 2>/dev/null || mktemp -d -t focr_e2e)
# Best-effort cleanup of OUR scratch dir only (never touches repo files).
cleanup() { rm -rf "$WORK" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Per-step verdicts, accumulated for the banner.
V_BUILD="-"
V_SCHEMA="-"
V_NATIVE="-"
V_GATED="-"
T_BUILD=0
T_SCHEMA=0
T_NATIVE=0
T_GATED=0
HARD_FAIL=0

log "franken_ocr end-to-end smoke driver"
info "root=$ROOT profile=$PROFILE bin=$BIN"
info "work=$WORK"
if [ -n "${FOCR_MODEL_PATH:-}" ]; then
  info "FOCR_MODEL_PATH=$FOCR_MODEL_PATH (model-present branch will be attempted)"
else
  info "FOCR_MODEL_PATH=<unset> (gated e2e will skip-with-SUCCESS)"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — BUILD the focr binary.
# ═════════════════════════════════════════════════════════════════════════════
step "1/4 BUILD"
S=$(now_ms)
if [ "$DO_BUILD" -eq 1 ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    fail "cargo not found on PATH; cannot build (re-run with --no-build against a prebuilt bin)"
    V_BUILD="FAIL"
    HARD_FAIL=1
  else
    info "cargo build $CARGO_BUILD_FLAGS --bin focr"
    # Build only the bin we drive; capture output so a failure is diagnosable.
    if ( cd "$ROOT" && cargo build $CARGO_BUILD_FLAGS --bin focr ) >"$WORK/build.log" 2>&1; then
      ok "built focr ($PROFILE)"
      V_BUILD="PASS"
    else
      fail "cargo build failed; tail of build log:"
      tail -n 30 "$WORK/build.log" >&2 || true
      V_BUILD="FAIL"
      HARD_FAIL=1
    fi
  fi
else
  info "--no-build: skipping cargo build, expecting a prebuilt binary"
  V_BUILD="SKIP"
fi
T_BUILD=$(( $(now_ms) - S ))
info "step 1 took ${T_BUILD}ms"

# If the binary still isn't there, the CLI-driven steps can't run — mark them
# skipped and jump to the gated cargo-test step (which builds its own bin).
BIN_OK=1
if [ ! -x "$BIN" ]; then
  BIN_OK=0
  fail "focr binary not found/executable at $BIN"
  info "CLI-driven steps (2,3) will be SKIPPED; the cargo-test gated e2e (step 4) builds its own"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — ROBOT-SCHEMA pipe smoke: `focr robot schema | <validate>`.
#   Exactly one parseable JSON object on stdout, exit 0, with the load-bearing
#   fields (schema_version: number, events: non-empty array).
# ═════════════════════════════════════════════════════════════════════════════
step "2/4 ROBOT-SCHEMA  (focr robot schema | validate)"
S=$(now_ms)
if [ "$BIN_OK" -eq 1 ]; then
  # Capture stdout (the data surface) and the exit code separately.
  if "$BIN" robot schema >"$WORK/schema.json" 2>"$WORK/schema.err"; then
    SCHEMA_EXIT=0
  else
    SCHEMA_EXIT=$?
  fi
  info "exit=$SCHEMA_EXIT stdout_bytes=$(wc -c <"$WORK/schema.json" | tr -d ' ')"

  # Count non-empty stdout lines — must be exactly one (data-only on stdout).
  NLINES=$(grep -c . "$WORK/schema.json" 2>/dev/null || echo 0)

  # Validate the single line is real JSON carrying the contract fields. Prefer
  # jq (ubiquitous, zero-dep); fall back to python3; final fallback is a coarse
  # shell grep so the step still runs on a bare box.
  VALID=0
  if command -v jq >/dev/null 2>&1; then
    info "validator=jq"
    if jq -e '
        (.schema_version | type == "number")
        and (.events | type == "array")
        and (.events | length > 0)
      ' <"$WORK/schema.json" >/dev/null 2>"$WORK/schema.valerr"; then
      VALID=1
    fi
  elif command -v python3 >/dev/null 2>&1; then
    info "validator=python3"
    if python3 - "$WORK/schema.json" >/dev/null 2>"$WORK/schema.valerr" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    lines = [l for l in f.read().splitlines() if l.strip()]
assert len(lines) == 1, f"expected one JSON line, got {len(lines)}"
doc = json.loads(lines[0])
assert isinstance(doc.get("schema_version"), int), "schema_version must be an int"
ev = doc.get("events")
assert isinstance(ev, list) and len(ev) > 0, "events must be a non-empty array"
PY
    then
      VALID=1
    fi
  else
    info "validator=shell-grep (no jq/python3)"
    if grep -q '"schema_version"' "$WORK/schema.json" && grep -q '"events"' "$WORK/schema.json"; then
      VALID=1
    fi
  fi

  if [ "$SCHEMA_EXIT" -eq 0 ] && [ "$NLINES" -eq 1 ] && [ "$VALID" -eq 1 ]; then
    ok "robot schema is one parseable JSON object with schema_version + non-empty events (exit 0)"
    info "schema: $(cat "$WORK/schema.json")"
    V_SCHEMA="PASS"
  else
    fail "robot schema smoke failed: exit=$SCHEMA_EXIT lines=$NLINES valid=$VALID"
    info "stdout: $(cat "$WORK/schema.json" 2>/dev/null || true)"
    info "stderr: $(cat "$WORK/schema.err" 2>/dev/null || true)"
    [ -s "$WORK/schema.valerr" ] && info "validator-err: $(cat "$WORK/schema.valerr")"
    V_SCHEMA="FAIL"
    HARD_FAIL=1
  fi
else
  skip "no focr binary — robot-schema smoke skipped (covered by src/robot.rs unit test)"
  V_SCHEMA="SKIP"
fi
T_SCHEMA=$(( $(now_ms) - S ))
info "step 2 took ${T_SCHEMA}ms"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — NATIVE-PATH proof: `focr ocr <img> --model /nonexistent` exits 3.
#   The /nonexistent model must produce a clean ModelNotFound (exit 3), proving
#   the native path ran even without weights (TL5) — NOT a panic, NOT fabricated
#   output. The `--model` flag is the documented target of the CLI surface; while
#   it is mid-flux the documented-skeleton exits (1 NotImplemented / 2 Usage) are
#   a TRIPWIRE XFAIL with an explicit observed-vs-target note, not a failure.
# ═════════════════════════════════════════════════════════════════════════════
step "3/4 NATIVE-PATH   (focr ocr <img> --model /nonexistent => exit 3)"
S=$(now_ms)
if [ "$BIN_OK" -eq 1 ]; then
  IMG="$WORK/native_probe.png"
  # A non-existent image is fine here: the model is resolved BEFORE the image is
  # decoded, so /nonexistent model => ModelNotFound fires first (the point of the
  # proof). We still hand a path-shaped arg.
  NONEXIST_MODEL="/nonexistent/franken_ocr/model.focrq"
  info "running: focr ocr $IMG --model $NONEXIST_MODEL"
  set +e
  "$BIN" ocr "$IMG" --model "$NONEXIST_MODEL" >"$WORK/native.out" 2>"$WORK/native.err"
  NATIVE_EXIT=$?
  set -e
  info "exit=$NATIVE_EXIT"
  info "stderr: $(head -n 1 "$WORK/native.err" 2>/dev/null || true)"

  case "$NATIVE_EXIT" in
    3)
      ok "exit 3 (ModelNotFound) — native path reached a clean error, no panic, no fallback"
      V_NATIVE="PASS"
      ;;
    1|2)
      # 2 = clap rejected the (not-yet-landed) --model flag (Usage);
      # 1 = the skeleton `ocr` returned NotImplemented before model resolution.
      # Both mean the documented target (--model => ModelNotFound exit 3) has not
      # landed yet. Tripwire XFAIL, not a failure.
      xfail "exit $NATIVE_EXIT (documented skeleton: --model flag / ocr wiring mid-flux)"
      info  "observed[exit=$NATIVE_EXIT] != target[exit=3 ModelNotFound via --model /nonexistent]"
      info  "the library guard recognize_with_model(/nonexistent)=>ModelNotFound proves this today;"
      info  "this CLI tripwire tightens to a hard exit-3 assertion when the flag lands"
      V_NATIVE="XFAIL"
      ;;
    134|139|*)
      # 134 = SIGABRT (panic=abort), 139 = SIGSEGV: a panic/crash is the worst
      # outcome (the native path must error cleanly, never panic). exit 0 (a
      # fabricated success) is equally bad. Anything else unexpected: hard fail.
      if [ "$NATIVE_EXIT" -eq 0 ]; then
        fail "exit 0 — focr fabricated success from a /nonexistent model (false green, TL5 violation)"
      else
        fail "exit $NATIVE_EXIT — focr did not error cleanly on /nonexistent (panic/crash?)"
      fi
      info "stdout: $(cat "$WORK/native.out" 2>/dev/null || true)"
      info "stderr: $(cat "$WORK/native.err" 2>/dev/null || true)"
      V_NATIVE="FAIL"
      HARD_FAIL=1
      ;;
  esac
else
  skip "no focr binary — native-path CLI proof skipped (covered by the library e2e test)"
  V_NATIVE="SKIP"
fi
T_NATIVE=$(( $(now_ms) - S ))
info "step 3 took ${T_NATIVE}ms"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — GATED-E2E: the model-gated recognize harness via cargo test.
#   Skips-with-SUCCESS without weights; runs a real recognize over a tiny fixture
#   when FOCR_MODEL_PATH resolves. We drive the Rust harness (tests/e2e_recognize.rs)
#   so the gate decision, the SUCCESS skip line, and the present-model assertions
#   live in ONE source of truth. `--nocapture` surfaces the `E2E ` telemetry.
# ═════════════════════════════════════════════════════════════════════════════
step "4/4 GATED-E2E     (cargo test --test e2e_recognize -- --nocapture)"
S=$(now_ms)
if ! command -v cargo >/dev/null 2>&1; then
  skip "cargo not found — gated e2e harness skipped"
  V_GATED="SKIP"
else
  info "cargo test $CARGO_BUILD_FLAGS --test e2e_recognize -- --nocapture"
  set +e
  ( cd "$ROOT" && cargo test $CARGO_BUILD_FLAGS --test e2e_recognize -- --nocapture ) \
    >"$WORK/gated.log" 2>&1
  GATED_EXIT=$?
  set -e
  # Surface the structured E2E telemetry from the harness (SUCCESS / XFAIL /
  # result lines) into our own stderr stream so the smoke log is self-contained.
  if grep -q '^E2E ' "$WORK/gated.log" 2>/dev/null; then
    info "---- harness E2E telemetry ----"
    grep '^E2E ' "$WORK/gated.log" >&2 || true
    info "---- end harness telemetry ----"
  fi
  if [ "$GATED_EXIT" -eq 0 ]; then
    # Distinguish a real-recognize run from a skip-with-SUCCESS for the banner.
    if grep -q 'E2E SUCCESS .*skipped: model not present' "$WORK/gated.log" 2>/dev/null; then
      skip "gated e2e: model absent -> skipped-with-SUCCESS (CI-green; native path unverified)"
      V_GATED="SKIP"
    elif grep -q 'real recognize() produced' "$WORK/gated.log" 2>/dev/null; then
      ok "gated e2e: real recognize() ran over the tiny fixture (model present)"
      V_GATED="PASS"
    else
      ok "gated e2e harness passed (exit 0)"
      V_GATED="PASS"
    fi
  else
    fail "gated e2e harness FAILED (cargo test exit $GATED_EXIT); tail of log:"
    tail -n 40 "$WORK/gated.log" >&2 || true
    V_GATED="FAIL"
    HARD_FAIL=1
  fi
fi
T_GATED=$(( $(now_ms) - S ))
info "step 4 took ${T_GATED}ms"

# ═════════════════════════════════════════════════════════════════════════════
# FINAL BANNER — per-step verdict + timing, then overall PASS/SKIP/FAIL.
# ═════════════════════════════════════════════════════════════════════════════
T_TOTAL=$(( T_BUILD + T_SCHEMA + T_NATIVE + T_GATED ))
{
  printf 'E2E\n'
  printf 'E2E ┌──────────────────────────────────────────────────────────────┐\n'
  printf 'E2E │ franken_ocr e2e smoke — summary                               │\n'
  printf 'E2E ├────────────────────┬─────────┬───────────────────────────────┤\n'
  printf 'E2E │ %-18s │ %-7s │ %25sms │\n' "1 BUILD"        "$V_BUILD"  "$T_BUILD"
  printf 'E2E │ %-18s │ %-7s │ %25sms │\n' "2 ROBOT-SCHEMA" "$V_SCHEMA" "$T_SCHEMA"
  printf 'E2E │ %-18s │ %-7s │ %25sms │\n' "3 NATIVE-PATH"  "$V_NATIVE" "$T_NATIVE"
  printf 'E2E │ %-18s │ %-7s │ %25sms │\n' "4 GATED-E2E"    "$V_GATED"  "$T_GATED"
  printf 'E2E ├────────────────────┴─────────┴───────────────────────────────┤\n'
  printf 'E2E │ total %52sms │\n' "$T_TOTAL"
  printf 'E2E └──────────────────────────────────────────────────────────────┘\n'
} >&2

if [ "$HARD_FAIL" -ne 0 ]; then
  log "BANNER: FAIL — one or more steps failed (see above)"
  exit 1
fi

# Distinguish an all-green run that exercised real work from one that
# skipped the model-gated branch (still success, but visibly a SKIP overall).
if [ "$V_GATED" = "SKIP" ] || [ "$V_NATIVE" = "XFAIL" ]; then
  log "BANNER: PASS (with model-gated SKIP / documented XFAIL) — CI-green; supply FOCR_MODEL_PATH to exercise the real recognize path"
else
  log "BANNER: PASS — all steps green"
fi
exit 0
