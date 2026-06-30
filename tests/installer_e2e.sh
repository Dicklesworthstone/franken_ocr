#!/usr/bin/env bash
# =============================================================================
# True end-to-end installer integration test.
#
# WHY THIS EXISTS
# ---------------
# install.sh renders status with `gum` ONLY when gum is installed AND stdout is a
# TTY (`[ -t 1 ]`). Every non-interactive / CI run takes the plain-ANSI fallback,
# so the gum path was never exercised — and a gum arg-parse bug shipped: the very
# first status line `gum style --foreground 39 "-> $*"` made gum treat the leading
# `->` as an unknown flag, print its usage, and (under `set -euo pipefail`) ABORT
# the whole installer. shellcheck and `bash -n` are clean on that bug because it is
# gum-CLI semantics, not shell syntax. Only RUNNING the installer through the gum
# path catches it. This test does exactly that.
#
# GATES
#   0  static     — bash -n + shellcheck (cheap regression net)
#   1  gum render — source install.sh, force the gum branch, drive info/ok/warn/err
#                   (incl. dash-leading text) and assert gum never errors
#   2  full e2e   — run the REAL installer end-to-end under a pty (so the gum path
#                   activates) against a FAKE release served over file:// (no
#                   network), then assert it installed a working `focr`
#
# Gates 1/2 SKIP (not fail) when their prerequisites (gum / script) are absent, so
# `scripts/check.sh` stays green on minimal dev boxes; CI installs gum so the gum
# path actually runs there.
# =============================================================================
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
# FOCR_INSTALL_SH lets the test point at an alternate installer copy (used by the
# test's own regression self-check to prove these gates actually catch the bug).
INSTALL_SH="${FOCR_INSTALL_SH:-$REPO_ROOT/install.sh}"

fail=0
pass() { printf '  \033[0;32mPASS\033[0m %s\n' "$1"; }
bad()  { printf '  \033[0;31mFAIL\033[0m %s\n' "$1"; fail=1; }
skip() { printf '  \033[1;33mSKIP\033[0m %s\n' "$1"; }

# Signatures of a CLI rejecting our arguments and dumping usage (the bug class).
GUM_ERR='gum: error|Usage: gum|unknown flag|unexpected argument|unexpected token'

[ -f "$INSTALL_SH" ] || { echo "install.sh not found at $INSTALL_SH"; exit 2; }

# ---------------------------------------------------------------------------
echo "== Gate 0: static analysis =="
if bash -n "$INSTALL_SH"; then pass "bash -n install.sh"; else bad "bash -n install.sh"; fi
if command -v shellcheck >/dev/null 2>&1; then
  if shellcheck -S warning "$INSTALL_SH" >/tmp/_focr_shellcheck.$$ 2>&1; then
    pass "shellcheck -S warning"
  else
    bad "shellcheck -S warning"; sed 's/^/      /' /tmp/_focr_shellcheck.$$
  fi
  rm -f /tmp/_focr_shellcheck.$$
else
  skip "shellcheck not installed"
fi

# ---------------------------------------------------------------------------
echo "== Gate 1: gum status-helper render path =="
if command -v gum >/dev/null 2>&1; then
  # Source install.sh (its `main` is guarded off when sourced), force the gum
  # branch, and drive every status helper — including messages that START WITH a
  # dash, to prove the `--` flag-terminator guards dynamic text too.
  render_out=$(
    {
      set --                         # clear positional args before sourcing
      # shellcheck disable=SC1090
      source "$INSTALL_SH"
      set +e                          # see ALL helper output even if one errors
      HAS_GUM=1; NO_GUM=0; QUIET=0
      info "Detecting platform"
      info "-> arrow-prefixed status (the exact original bug)"
      ok   "Checksum verified (deadbeef...)"
      warn "-x dash-leading warning text"
      err  "--help-looking error text"
    } 2>&1
  )
  if printf '%s' "$render_out" | grep -Eq "$GUM_ERR"; then
    bad "status helpers tripped a gum arg-parse error:"
    printf '%s\n' "$render_out" | sed 's/^/      /'
  else
    pass "info/ok/warn/err render cleanly under gum (incl. dash-leading text)"
  fi
else
  skip "gum not installed — gum render path not exercised (CI installs gum)"
fi

# ---------------------------------------------------------------------------
echo "== Gate 2: full end-to-end install (fake release via file://, real pty) =="
if ! command -v script >/dev/null 2>&1; then
  skip "no 'script' tool — cannot allocate a pty to drive the gum path"
else
  os=$(uname -s | tr '[:upper:]' '[:lower:]')
  arch=$(uname -m)
  case "$arch" in arm64|aarch64) arch=aarch64 ;; x86_64|amd64) arch=x86_64 ;; esac
  case "$os-$arch" in
    darwin-aarch64) asset="focr-aarch64-apple-darwin-neon-sdot-i8mm" ;;
    darwin-x86_64)  asset="focr-x86_64-apple-darwin" ;;
    linux-x86_64)   asset="focr-x86_64-unknown-linux-gnu" ;;
    linux-aarch64)  asset="focr-aarch64-unknown-linux-gnu" ;;
    *) asset="" ;;
  esac
  if [ -z "$asset" ]; then
    skip "unsupported test platform ${os}-${arch}"
  else
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
    rel="$work/release"; fakehome="$work/home"
    # IMPORTANT: a NESTED, not-yet-existing install dir. This reproduces the second
    # `set -e` blocker class (check_disk_space running `df` on a path whose parent
    # does not exist yet — the default ~/.local/bin on a fresh account). The
    # installer must create it via check_write_permissions, not abort on the df.
    bindir="$work/fresh/account/.local/bin"
    mkdir -p "$rel" "$fakehome"  # NB: bindir intentionally NOT created here

    # Fake focr: a tiny stub that answers `--version` like the real CLI so the
    # installer's verify_install step succeeds. Anything else is a clean no-op.
    cat > "$rel/$asset" <<'STUB'
#!/bin/sh
case "${1:-}" in
  --version) echo "focr 0.2.0" ;;
  *) exit 0 ;;
esac
STUB
    chmod +x "$rel/$asset"

    # SHA256 sidecar in the installer's expected "<hex>  <asset>" format.
    if command -v sha256sum >/dev/null 2>&1; then
      ( cd "$rel" && sha256sum "$asset" > "$asset.sha256" )
    elif command -v shasum >/dev/null 2>&1; then
      ( cd "$rel" && shasum -a 256 "$asset" > "$asset.sha256" )
    else
      skip "no sha256 tool"; asset=""
    fi

    if [ -n "$asset" ]; then
      log="$work/transcript.txt"
      args="--version v0.2.0 --dir $bindir --no-pull --offline --force"
      # Run the REAL installer under a pty so `[ -t 1 ]` is true and the gum path
      # activates exactly as it does for a user. file:// base = no network. HOME
      # is sandboxed so nothing touches the developer's shell rc or model cache.
      if [ "$os" = "darwin" ]; then
        env HOME="$fakehome" FOCR_INSTALL_BASE_URL="file://$rel" \
          script -q /dev/null bash "$INSTALL_SH" $args >"$log" 2>&1
        rc=$?
      else
        env HOME="$fakehome" FOCR_INSTALL_BASE_URL="file://$rel" \
          script -qec "bash '$INSTALL_SH' $args" /dev/null >"$log" 2>&1
        rc=$?
      fi
      transcript=$(cat "$log" 2>/dev/null || true)

      if [ "$rc" -ne 0 ]; then
        bad "installer exited non-zero ($rc):"
        printf '%s\n' "$transcript" | tail -25 | sed 's/^/      /'
      elif printf '%s' "$transcript" | grep -Eq "$GUM_ERR"; then
        bad "installer transcript shows a gum arg-parse error:"
        printf '%s\n' "$transcript" | grep -E "$GUM_ERR" | sed 's/^/      /'
      elif [ ! -x "$bindir/focr" ]; then
        bad "focr was not installed to $bindir"
        printf '%s\n' "$transcript" | tail -25 | sed 's/^/      /'
      else
        v=$("$bindir/focr" --version 2>&1 || true)
        if printf '%s' "$v" | grep -q '0\.2\.0'; then
          pass "installer ran clean end-to-end and installed a working focr ($v)"
        else
          bad "installed focr --version did not work: '$v'"
        fi
      fi
    fi
  fi
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "installer e2e: ALL GATES PASS"
else
  echo "installer e2e: FAILURES ABOVE"
fi
exit "$fail"
