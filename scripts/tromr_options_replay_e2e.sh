#!/usr/bin/env bash
# Build the explicit-options replay test, then execute it with no usable PATH.
# Native franken_ocr PDF decode and TrOMR inference must complete without a
# successful focr/pdftoppm/ImageMagick/Ghostscript/other helper process.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CARGO_BIN=${CARGO_BIN:-cargo}
TARGET_DIR=${CARGO_TARGET_DIR:-"$ROOT/target"}
TEST_NAME=tromr_options_replay_e2e
FOCR_TROMR_CASE=${FOCR_TROMR_CASE:-replay}
case "$FOCR_TROMR_CASE" in
  replay)
    CASE_NAME=native_spohr_pdf_has_exact_two_run_options_replay
    DEFAULT_QUANT=f32
    CASE_DESCRIPTION='native PDF + two full forwards'
    ;;
  budget-cancel)
    CASE_NAME=native_spohr_pdf_explicit_timeout_cancel_and_same_engine_reuse
    DEFAULT_QUANT=int8
    CASE_DESCRIPTION='native PDF + explicit timeout/cancellation + same-engine reuse'
    ;;
  *)
    printf 'TrOMR options replay E2E: FOCR_TROMR_CASE must be replay or budget-cancel\n' >&2
    exit 2
    ;;
esac
test_binary=

: "${FOCR_TROMR_DIR:?set FOCR_TROMR_DIR to the pinned TrOMR bundle directory}"
: "${FOCR_TROMR_PDF:?set FOCR_TROMR_PDF to louisspohrsceleb00spohuoft.pdf}"
FOCR_TROMR_QUANT=${FOCR_TROMR_QUANT:-$DEFAULT_QUANT}
if [[ "$FOCR_TROMR_QUANT" != f32 && "$FOCR_TROMR_QUANT" != int8 ]]; then
  printf 'TrOMR options replay E2E: FOCR_TROMR_QUANT must be f32 or int8\n' >&2
  exit 2
fi

while IFS= read -r message; do
  case "$message" in
    *'"target":{"kind":["test"]'*'"name":"tromr_options_replay_e2e"'*'"executable":"'*)
      executable=${message#*'"executable":"'}
      test_binary=${executable%%\"*}
      ;;
  esac
done < <(
  cd -- "$ROOT"
  CARGO_TARGET_DIR="$TARGET_DIR" "$CARGO_BIN" test --locked --test "$TEST_NAME" \
    --no-run --message-format=json
)

if [[ -z "$test_binary" || ! -x "$test_binary" ]]; then
  printf 'TrOMR options replay E2E: Cargo did not report an executable test binary\n' >&2
  exit 1
fi

printf 'TrOMR options replay E2E: %s under forbidden-process PATH\n' "$CASE_DESCRIPTION" >&2
cd -- "$ROOT"
PATH=/nonexistent/franken_ocr_process_trap \
  CARGO_TARGET_DIR="$TARGET_DIR" \
  FOCR_TROMR_CASE="$FOCR_TROMR_CASE" \
  FOCR_TROMR_QUANT="$FOCR_TROMR_QUANT" \
  "$test_binary" --ignored --exact "$CASE_NAME" --nocapture
