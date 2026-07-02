#!/usr/bin/env sh
#
# spec_gate_e2e.sh — FOCR_SPEC_DECODE ON-vs-OFF byte-identity A/B gate
# (bd-1azu.36, the model-gated e2e half of the LINEAR spec-decode gate).
#
# Speculative decode (`FOCR_SPEC_DECODE`, default OFF — src/native_engine/mod.rs)
# claims to be LOSSLESS: the ON token stream is byte-for-byte the OFF stream.
# The Rust test suite cannot flip the env in-process — the flag is read ONCE into
# a process-wide OnceLock, and edition-2024 `set_var` is `unsafe` (the crate
# denies unsafe) — so this driver runs TWO `focr` processes per page, one with
# the env REMOVED and one with `FOCR_SPEC_DECODE=1`, and sha256-compares the
# stdout markdown. This is the same two-process A/B discipline used for the
# FOCR_BATCH_VISION on/off parity runs.
#
# IMPORTANT: `FOCR_SPEC_DECODE` is a PRESENCE kill-switch (`var_os(..).is_some()`)
# — even `FOCR_SPEC_DECODE=0` ARMS it. The OFF arm therefore uses `env -u` to
# REMOVE the variable (supported by both GNU and BSD/macOS env), which also
# shields the A/B from a value the caller happens to have exported.
#
# MODEL-GATED (Testing Policy: skip-with-SUCCESS without weights). Required env:
#   FOCR_MODEL_PATH       — the model artifact. An int8 `.focrq` is recommended:
#                           the spec loop lives inside the int8 decode path
#                           (`generate_cached_i8`), and a pre-quantized int8
#                           artifact self-engages it. For other artifacts export
#                           FOCR_DECODE_INT8=1 yourself — it passes through to
#                           BOTH arms equally, so composition stays controlled.
#   FOCR_SPEC_E2E_IMAGES  — a DIRECTORY of page images (*.png/*.jpg/*.jpeg,
#                           sorted) or a whitespace-separated LIST of image
#                           paths (list paths must not contain whitespace).
# Either unset/missing => SKIP with success and an explicit banner.
#
# Options / knobs:
#   --no-build              use the already-built binary (default builds focr)
#   --release               build/use the release binary
#   FOCR_BIN=/path/to/focr  binary override (skips the build)
#   SPEC_GATE_DETERMINISM=1 additionally re-run the OFF arm and require
#                           OFF==OFF, separating "spec changed the bytes" from
#                           ambient nondeterminism before blaming the lever
#
# Exit 0 = every page byte-identical ON==OFF (or the gated SKIP). Non-zero = at
# least one page diverged; outputs are PRESERVED under the printed workdir for
# diffing. RETRY-PREDICATE (bd-1azu.36): a divergence — especially page_0590 —
# means the verify kernel reordered reductions (the bd-1waa failure class):
# REVERT the spec lever and record it in docs/NEGATIVE_EVIDENCE.md; never
# tolerance it away.
#
# POSIX sh; passes `sh -n`. House logging style of scripts/e2e_smoke.sh: data
# never goes to stdout, every diagnostic line is `SPEC `-prefixed on stderr so a
# scraper can `grep '^SPEC '`.
set -eu

# ── house style: structured, greppable logging, all on stderr ────────────────
log()   { printf 'SPEC %s\n' "$*" >&2; }
step()  { printf 'SPEC ==== STEP %s ====\n' "$*" >&2; }
info()  { printf 'SPEC   %s\n' "$*" >&2; }
ok()    { printf 'SPEC   PASS  %s\n' "$*" >&2; }
skip()  { printf 'SPEC   SKIP  %s\n' "$*" >&2; }
fail()  { printf 'SPEC   FAIL  %s\n' "$*" >&2; }

# ── millisecond clock (best-effort; falls back to seconds*1000) ──────────────
now_ms() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import time; print(int(time.time()*1000))'
  elif command -v perl >/dev/null 2>&1; then
    perl -MTime::HiRes=time -e 'print int(time()*1000)'
  else
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
      sed -n '2,48p' "$0"
      exit 0
      ;;
    *) printf 'spec_gate_e2e.sh: unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

# ── resolve repo root (this script lives in <root>/scripts) ──────────────────
SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)

log "FOCR_SPEC_DECODE on/off byte-identity gate (bd-1azu.36)"

# ── model gate: skip-with-SUCCESS without weights/corpus ─────────────────────
step "1/4 GATE"
if [ -z "${FOCR_MODEL_PATH:-}" ]; then
  skip "FOCR_MODEL_PATH unset — model-gated A/B skipped with SUCCESS"
  exit 0
fi
if [ ! -f "$FOCR_MODEL_PATH" ]; then
  skip "FOCR_MODEL_PATH=$FOCR_MODEL_PATH missing on disk — skipped with SUCCESS"
  exit 0
fi
if [ -z "${FOCR_SPEC_E2E_IMAGES:-}" ]; then
  skip "FOCR_SPEC_E2E_IMAGES unset — no corpus to A/B, skipped with SUCCESS"
  exit 0
fi
info "model=$FOCR_MODEL_PATH"
info "images=$FOCR_SPEC_E2E_IMAGES"
# NON-REVIVAL (bd-1azu.36): the rejected key-batch levers (bd-1waa, the
# page_0590-runaway class — docs/NEGATIVE_EVIDENCE.md) must remain UNSET on the
# spec path. Both are PRESENCE kill-switches, so `set -but-empty` also arms
# them; refuse to certify an A/B run with either present in the environment.
if [ -n "${FOCR_ATTN_GEMM+x}" ] || [ -n "${FOCR_INT8_KV+x}" ]; then
  fail "NON-REVIVAL: FOCR_ATTN_GEMM / FOCR_INT8_KV must be UNSET on the spec path"
  fail "(rejected levers, bd-1waa / docs/NEGATIVE_EVIDENCE.md) — unset them and re-run"
  exit 2
fi
if [ -n "${FOCR_SPEC_DECODE:-}" ]; then
  info "caller has FOCR_SPEC_DECODE set — OFF arm strips it via 'env -u', ON arm forces =1"
fi
if [ -n "${FOCR_DECODE_INT8:-}" ]; then
  info "FOCR_DECODE_INT8=${FOCR_DECODE_INT8} passes through to BOTH arms (controlled composition)"
fi

# ── sha256 tool (macOS shasum / GNU sha256sum / openssl fallback) ─────────────
if command -v shasum >/dev/null 2>&1; then
  sha_of() { shasum -a 256 -- "$1" | awk '{print $1}'; }
elif command -v sha256sum >/dev/null 2>&1; then
  sha_of() { sha256sum -- "$1" | awk '{print $1}'; }
elif command -v openssl >/dev/null 2>&1; then
  sha_of() { openssl dgst -sha256 -r -- "$1" | awk '{print $1}'; }
else
  fail "no sha256 tool found (shasum/sha256sum/openssl) — cannot compare"
  exit 2
fi

# ── scratch workspace: cleaned on full PASS, preserved on any divergence ─────
WORK=$(mktemp -d 2>/dev/null || mktemp -d -t focr_spec_gate)
info "work=$WORK (preserved on failure for diffing)"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD / resolve the focr binary.
# ═════════════════════════════════════════════════════════════════════════════
step "2/4 BUILD"
S=$(now_ms)
if [ -n "${FOCR_BIN:-}" ]; then
  BIN="$FOCR_BIN"
  info "FOCR_BIN override: $BIN (skipping build)"
elif [ "$DO_BUILD" -eq 1 ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    fail "cargo not found on PATH; re-run with --no-build or set FOCR_BIN"
    exit 2
  fi
  info "cargo build $CARGO_BUILD_FLAGS --bin focr"
  if ! ( cd "$ROOT" && cargo build $CARGO_BUILD_FLAGS --bin focr ) >"$WORK/build.log" 2>&1; then
    fail "cargo build failed; tail of build log:"
    tail -n 30 "$WORK/build.log" >&2 || true
    exit 2
  fi
  BIN="$ROOT/target/$PROFILE/focr"
else
  BIN="$ROOT/target/$PROFILE/focr"
  info "--no-build: expecting a prebuilt binary at $BIN"
fi
if [ ! -x "$BIN" ]; then
  fail "focr binary not found/executable at $BIN"
  exit 2
fi
info "step 2 took $(( $(now_ms) - S ))ms; bin=$BIN"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — collect the page corpus (directory glob, or explicit list).
# ═════════════════════════════════════════════════════════════════════════════
step "3/4 CORPUS"
LIST="$WORK/pages.list"
: > "$LIST"
if [ -d "$FOCR_SPEC_E2E_IMAGES" ]; then
  # Sorted for a stable, reproducible page order.
  find "$FOCR_SPEC_E2E_IMAGES" -maxdepth 1 -type f \
    \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \
       -o -name '*.PNG' -o -name '*.JPG' -o -name '*.JPEG' \) \
    2>/dev/null | sort > "$LIST"
else
  for img in $FOCR_SPEC_E2E_IMAGES; do
    printf '%s\n' "$img" >> "$LIST"
  done
fi
N_PAGES=$(wc -l < "$LIST" | tr -d ' ')
if [ "$N_PAGES" -eq 0 ]; then
  skip "no images found under FOCR_SPEC_E2E_IMAGES — skipped with SUCCESS"
  rm -rf "$WORK" 2>/dev/null || true
  exit 0
fi
info "$N_PAGES page(s) queued"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — per-page two-process A/B: OFF (env removed) vs ON (=1), sha256.
# ═════════════════════════════════════════════════════════════════════════════
step "4/4 A/B"
PASS=0
FAIL=0
while IFS= read -r img; do
  name=$(basename -- "$img")
  if [ ! -f "$img" ]; then
    fail "$name: image missing at $img"
    FAIL=$((FAIL + 1))
    continue
  fi
  S=$(now_ms)

  off_out="$WORK/$name.off.md"
  on_out="$WORK/$name.on.md"

  # OFF arm: the variable REMOVED (presence kill-switch — see header).
  if ! env -u FOCR_SPEC_DECODE "$BIN" ocr "$img" --model "$FOCR_MODEL_PATH" \
       >"$off_out" 2>"$WORK/$name.off.log"; then
    fail "$name: OFF arm exited non-zero (log: $WORK/$name.off.log)"
    FAIL=$((FAIL + 1))
    continue
  fi

  # Optional determinism pre-gate: OFF must equal OFF before blaming the lever.
  if [ "${SPEC_GATE_DETERMINISM:-0}" = "1" ]; then
    off2_out="$WORK/$name.off2.md"
    if ! env -u FOCR_SPEC_DECODE "$BIN" ocr "$img" --model "$FOCR_MODEL_PATH" \
         >"$off2_out" 2>"$WORK/$name.off2.log"; then
      fail "$name: OFF determinism re-run exited non-zero"
      FAIL=$((FAIL + 1))
      continue
    fi
    if [ "$(sha_of "$off_out")" != "$(sha_of "$off2_out")" ]; then
      fail "$name: OFF != OFF — ambient nondeterminism, the A/B cannot attribute (investigate first)"
      FAIL=$((FAIL + 1))
      continue
    fi
  fi

  # ON arm: the spec loop armed.
  if ! env FOCR_SPEC_DECODE=1 "$BIN" ocr "$img" --model "$FOCR_MODEL_PATH" \
       >"$on_out" 2>"$WORK/$name.on.log"; then
    fail "$name: ON arm exited non-zero (log: $WORK/$name.on.log)"
    FAIL=$((FAIL + 1))
    continue
  fi

  off_sha=$(sha_of "$off_out")
  on_sha=$(sha_of "$on_out")
  ms=$(( $(now_ms) - S ))
  if [ "$off_sha" = "$on_sha" ]; then
    ok "$name: ON==OFF sha256=$off_sha (${ms}ms)"
    PASS=$((PASS + 1))
  else
    fail "$name: DIVERGED off=$off_sha on=$on_sha (${ms}ms)"
    fail "$name: diff \"$off_out\" \"$on_out\""
    FAIL=$((FAIL + 1))
  fi
done < "$LIST"

# ── banner ────────────────────────────────────────────────────────────────────
log "RESULT pages=$N_PAGES pass=$PASS fail=$FAIL"
if [ "$FAIL" -eq 0 ]; then
  log "VERDICT PASS — FOCR_SPEC_DECODE is byte-lossless on this corpus"
  rm -rf "$WORK" 2>/dev/null || true
  exit 0
fi
log "VERDICT FAIL — spec decode changed bytes; outputs preserved under $WORK"
log "RETRY-PREDICATE (bd-1azu.36): a divergence means the verify kernel reordered"
log "reductions (bd-1waa class) -> REVERT the lever + record in docs/NEGATIVE_EVIDENCE.md"
exit 1
