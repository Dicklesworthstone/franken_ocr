#!/usr/bin/env sh
#
# token_golden_gate.sh — the model-gated token-id golden gate (bead U2-T3).
#
# Runs tests/uocr2_goldens.rs armed, in the optimized `release-perf` profile
# (a debug-profile decode takes minutes per page). scripts/check.sh calls this
# after `cargo test`, so every change re-checks v1's real outputs whenever the
# weights are on the machine.
#
#   scripts/token_golden_gate.sh                     # fast subset
#   FOCR_E2E_LONG=1 scripts/token_golden_gate.sh     # full set (decoder/loader/
#                                                    # tokenizer/sampler/PDF/CLI
#                                                    # beads, every milestone exit)
#   UPDATE_GOLDENS=1 FOCR_E2E_LONG=1 scripts/token_golden_gate.sh
#                                                    # reviewed recapture ONLY
#
# The probe below is deliberately loose: it only decides whether to pay for
# the optimized build. The test resolves the artifact itself and compares
# only on the SHA-256 the goldens pin; anything else logs skip_no_model.
# `CARGO` overrides the cargo binary (e.g. CARGO=~/.cargo/bin/cargo).
set -eu

cargo_bin="${CARGO:-cargo}"

has_artifact() {
  if [ -n "${FOCR_MODEL_PATH:-}" ] && [ -f "$FOCR_MODEL_PATH" ]; then
    return 0
  fi
  dirs="${FOCR_MODEL_DIR:-}"
  if [ -n "${HOME:-}" ]; then
    dirs="$dirs:$HOME/.cache/franken_ocr/models"
    for sub in "$HOME"/.cache/franken_ocr/models/*/; do
      [ -d "$sub" ] && dirs="$dirs:$sub"
    done
  fi
  old_ifs=$IFS
  IFS=:
  for dir in $dirs; do
    [ -n "$dir" ] || continue
    for f in "$dir"/unlimited-ocr*.focrq; do
      if [ -f "$f" ]; then
        IFS=$old_ifs
        return 0
      fi
    done
  done
  IFS=$old_ifs
  return 1
}

if ! has_artifact; then
  echo "==> token golden gate: SKIP (no unlimited-ocr*.focrq in FOCR_MODEL_PATH, FOCR_MODEL_DIR or ~/.cache/franken_ocr/models)"
  exit 0
fi

echo "==> FOCR_GOLDEN_GATE=1 $cargo_bin test --locked --profile release-perf --test uocr2_goldens"
FOCR_GOLDEN_GATE=1 "$cargo_bin" test --locked --profile release-perf --test uocr2_goldens \
  -- --nocapture --test-threads=1
