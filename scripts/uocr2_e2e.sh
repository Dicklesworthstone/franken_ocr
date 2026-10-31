#!/usr/bin/env sh
#
# uocr2_e2e.sh — the Unlimited-OCR-2 end-to-end script (bead U2-T1a). Builds or
# finds the focr binary, then runs scripts/uocr2_e2e.py, which says what each
# step does and which ones need weights, FOCR_E2E_LONG=1 or FOCR_E2E_NETWORK=1.
#
#   scripts/uocr2_e2e.sh [--only j_levers,k_cjk_streaming]
#
# `FOCR_BIN` names a built binary; `CARGO` overrides the cargo binary;
# `FOCR_E2E_DIR` is where logs and outputs are kept (default target/uocr2_e2e).
# Decoding wants the optimized profile, so with no FOCR_BIN this builds
# release-perf. Without an artifact the debug binary is enough for the steps
# that always run.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cargo_bin="${CARGO:-cargo}"
work="${FOCR_E2E_DIR:-$root/target/uocr2_e2e}"

has_artifact() {
  [ -n "${FOCR_E2E_V2_MODEL:-}" ] && return 0
  [ -n "${FOCR_E2E_V1_MODEL:-}" ] && return 0
  old_ifs=$IFS
  IFS=:
  for dir in ${FOCR_MODEL_DIR:-} "${HOME:-}/.cache/franken_ocr/models"; do
    [ -d "$dir" ] || continue
    if find "$dir" -name 'unlimited-ocr*.focrq' -print -quit 2>/dev/null | grep -q .; then
      IFS=$old_ifs
      return 0
    fi
  done
  IFS=$old_ifs
  return 1
}

if [ -n "${FOCR_BIN:-}" ]; then
  bin=$FOCR_BIN
elif has_artifact; then
  (cd "$root" && "$cargo_bin" build --locked --profile release-perf --bin focr)
  bin="$root/target/release-perf/focr"
else
  (cd "$root" && "$cargo_bin" build --locked --bin focr)
  bin="$root/target/debug/focr"
fi

exec python3 -I "$root/scripts/uocr2_e2e.py" --bin "$bin" --work "$work" "$@"
