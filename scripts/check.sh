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
#   scripts/check.sh --ubs-only # run only the bounded UBS lane
#
# `cargo test` is a HARD gate: a green bar (exit 0) is required before any change
# is handed off or a bead is closed. `cargo check --all-targets` must also be free
# of the "src/main.rs present in multiple build targets" warning — both binaries
# (`focr`, `franken_ocr`) build from their own thin shims over `cli_main()`.
#
# UBS is part of the required local gate when the `ubs` binary is installed. In
# CI the tool may be unavailable; in that case the script reports the missing
# scanner explicitly instead of pretending it ran. `FOCR_UBS_TIMEOUT_S` bounds
# the UBS subprocess so a scanner hang fails loudly instead of wedging the gate.
set -eu

FAST=0
UBS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    --ubs-only) UBS_ONLY=1 ;;
    *) echo "check.sh: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

run() {
  echo "==> $*"
  "$@"
}

run_ubs_bounded() {
  timeout_s="${FOCR_UBS_TIMEOUT_S:-180}"
  echo "==> FOCR_UBS_TIMEOUT_S=$timeout_s $*"
  python3 - "$timeout_s" "$@" <<'PY'
import subprocess
import sys
import os
import signal
import time

try:
    timeout_s = float(sys.argv[1])
except ValueError:
    print(f"check.sh: FOCR_UBS_TIMEOUT_S must be numeric, got {sys.argv[1]!r}", file=sys.stderr)
    sys.exit(2)

cmd = sys.argv[2:]
if timeout_s <= 0:
    print(f"check.sh: FOCR_UBS_TIMEOUT_S must be > 0, got {timeout_s:g}", file=sys.stderr)
    sys.exit(2)

proc = subprocess.Popen(cmd, start_new_session=True)
try:
    sys.exit(proc.wait(timeout=timeout_s))
except subprocess.TimeoutExpired:
    print(
        f"check.sh: UBS timed out after {timeout_s:g}s: {' '.join(cmd)}",
        file=sys.stderr,
    )
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    time.sleep(0.05)
    sys.exit(124)
PY
}

run_ubs() {
  if ! command -v ubs >/dev/null 2>&1; then
    echo "==> ubs (skipped: command not found; install UBS to enable this gate)"
    return 0
  fi

  if git rev-parse --is-inside-work-tree >/dev/null 2>&1 &&
    ! git diff --quiet --diff-filter=ACMRTUXB --; then
    run_ubs_bounded ubs --diff
    return 0
  fi

  run_ubs_bounded ubs .
}

if [ "$UBS_ONLY" -eq 1 ]; then
  run_ubs
  echo "OK: UBS gate passed."
  exit 0
fi

run python3 scripts/check_ledgers.py
run python3 scripts/check_test_logs.py --self-test
run python3 scripts/check_focrq_format.py
run python3 scripts/check_oracle_provenance.py --self-test
run python3 scripts/check_oracle_provenance.py
run python3 scripts/check_oracle_provenance.py --live-replay
run python3 scripts/oracle_bridge.py --self-test
run python3 scripts/check_release_linkage.py
run python3 scripts/gauntlet_cert.py --self-test
# Installer end-to-end gate: drives install.sh through its gum/TTY render path
# (which only runs interactively, so nothing else exercises it) and a full
# fake-release install. Gum/pty sub-gates SKIP cleanly when gum/script are absent.
run bash tests/installer_e2e.sh
run cargo fmt --check
run cargo check --all-targets
if [ "$FAST" -eq 0 ]; then
  run cargo clippy --all-targets -- -D warnings
fi
run cargo test
run_ubs

echo "OK: all gates passed."
