#!/usr/bin/env bash
# bd-1azu.14 — the batched-spine watchdog SWEEP driver.
#
# Env is process-immutable inside a Rust test (edition-2024 set_var is unsafe
# and the crate denies unsafe), so the FOCR_BATCH_SIZE sweep and the
# FOCR_BATCH_SPINE=0 sequential control run live here: each configuration is
# one fresh test process. The in-test scenario asserts, per run: completion
# within the wall-clock budget, process-wide max_concurrent_forwards <= 1
# (vision + prefill + every scheduler decode step), and no forward beginning
# while the model-cache guard is held.
#
# Usage:
#   FOCR_MODEL_PATH=/path/to/model FOCR_WATCHDOG_IMAGE=/path/to/page.png \
#     scripts/spine_watchdog_sweep.sh
#
# Requires the model + a real decodable page; refuses to "pass" without them.
set -euo pipefail

: "${FOCR_MODEL_PATH:?set FOCR_MODEL_PATH to the model artifact}"
: "${FOCR_WATCHDOG_IMAGE:?set FOCR_WATCHDOG_IMAGE to a real page image}"
[ -e "$FOCR_MODEL_PATH" ] || { echo "model not found: $FOCR_MODEL_PATH" >&2; exit 3; }
[ -f "$FOCR_WATCHDOG_IMAGE" ] || { echo "image not found: $FOCR_WATCHDOG_IMAGE" >&2; exit 4; }

CARGO=${CARGO:-cargo}
TEST_ARGS=(test --release --offline --test many_pages_without_deadlock)

echo "== spine sweep: B in {1, 4, 32} (B << pages churns retire/backfill) =="
for B in 1 4 32; do
  echo "-- FOCR_BATCH_SPINE=1 FOCR_DECODE_INT8=1 FOCR_BATCH_SIZE=$B --"
  # FOCR_DECODE_INT8: the spine engages only when int8 decode is requested
  # (recognize_batch tracks the sequential oracle's dispatch condition).
  FOCR_BATCH_SPINE=1 FOCR_DECODE_INT8=1 FOCR_BATCH_SIZE="$B" \
    "$CARGO" "${TEST_ARGS[@]}" spine_many_pages_one_live_forward_within_budget -- --nocapture
done

echo "== control: FOCR_BATCH_SPINE=0 keeps the sequential watchdog green =="
FOCR_BATCH_SPINE=0 "$CARGO" "${TEST_ARGS[@]}" -- --nocapture

echo "SPINE WATCHDOG SWEEP: ALL CONFIGURATIONS GREEN"
