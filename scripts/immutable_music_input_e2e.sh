#!/usr/bin/env bash
# Build the public immutable-input E2E, then run the already-built test binary
# with no executable search path. Any attempted pdftoppm, ImageMagick, focr CLI,
# or other external conversion process therefore fails instead of being hidden.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CARGO_BIN=${CARGO_BIN:-cargo}
TARGET_DIR=${CARGO_TARGET_DIR:-"$ROOT/target"}
TEST_NAME=immutable_music_input_e2e
CASE_NAME=model_backed_immutable_music_receipt_is_complete_and_path_free
test_binary=

while IFS= read -r message; do
  case "$message" in
    *'"target":{"kind":["test"]'*'"name":"immutable_music_input_e2e"'*'"executable":"'*)
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
  printf 'immutable music E2E: Cargo did not report an executable test binary\n' >&2
  exit 1
fi

printf 'immutable music E2E: running %s with forbidden-process PATH\n' "$test_binary" >&2
PATH=/nonexistent/franken_ocr_process_trap \
  "$test_binary" --ignored --exact "$CASE_NAME" --nocapture
