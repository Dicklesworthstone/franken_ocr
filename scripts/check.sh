#!/usr/bin/env sh
#
# check.sh — the franken_ocr mandatory-check gate, in one command.
#
# Runs the same sequence AGENTS.md ("Mandatory Checks After Substantive Changes")
# requires, in order, stopping on the first failure. This is the repo's stand-in
# for a Makefile/CI `test` target until CI exists; when CI is added, wire THIS
# script as the test job rather than duplicating the commands.
#
#   scripts/check.sh            # fmt + check + clippy + test (the full gate)
#   scripts/check.sh --fast     # skip clippy (still runs fmt + check + test)
#
# `cargo test` is a HARD gate: a green bar (exit 0) is required before any change
# is handed off or a bead is closed. `cargo check --all-targets` must also be free
# of the "src/main.rs present in multiple build targets" warning — both binaries
# (`focr`, `franken_ocr`) build from their own thin shims over `cli_main()`.
set -eu

FAST=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    *) echo "check.sh: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

run() {
  echo "==> $*"
  "$@"
}

run cargo fmt --check
run cargo check --all-targets
if [ "$FAST" -eq 0 ]; then
  run cargo clippy --all-targets -- -D warnings
fi
run cargo test

echo "OK: all gates passed."
