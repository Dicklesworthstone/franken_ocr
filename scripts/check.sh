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
#
# UBS is part of the required local gate when the `ubs` binary is installed. In
# CI the tool may be unavailable; in that case the script reports the missing
# scanner explicitly instead of pretending it ran.
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

run_ubs() {
  if ! command -v ubs >/dev/null 2>&1; then
    echo "==> ubs (skipped: command not found; install UBS to enable this gate)"
    return 0
  fi

  if git rev-parse --is-inside-work-tree >/dev/null 2>&1 &&
    ! git diff --quiet --diff-filter=ACMRTUXB --; then
    echo "==> ubs --diff"
    ubs --diff
    return 0
  fi

  echo "==> ubs ."
  ubs .
}

run python3 scripts/check_ledgers.py
run python3 scripts/check_test_logs.py --self-test
run python3 scripts/check_focrq_format.py
run python3 scripts/check_oracle_provenance.py --self-test
run python3 scripts/check_oracle_provenance.py
run python3 scripts/check_oracle_provenance.py --live-replay
run python3 scripts/oracle_bridge.py --self-test
run python3 scripts/check_release_linkage.py
run python3 scripts/gauntlet_cert.py --self-test
run cargo fmt --check
run cargo check --all-targets
if [ "$FAST" -eq 0 ]; then
  run cargo clippy --all-targets -- -D warnings
fi
run cargo test
run_ubs

echo "OK: all gates passed."
