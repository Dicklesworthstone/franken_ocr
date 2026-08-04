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
#   0  static     — syntax/lint + release-matrix/installer asset contract
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
DIST_YML="$REPO_ROOT/.github/workflows/dist.yml"
CI_YML="$REPO_ROOT/.github/workflows/ci.yml"
INSTALL_PS1="$REPO_ROOT/install.ps1"
CHECK_SH="$REPO_ROOT/scripts/check.sh"
GAUNTLET_CERT="$REPO_ROOT/scripts/gauntlet_cert.py"
TOOLCHAIN_TOML="$REPO_ROOT/rust-toolchain.toml"
GITATTRIBUTES="$REPO_ROOT/.gitattributes"

STATIC_ONLY=0
case "${1:-}" in
  --static) STATIC_ONLY=1 ;;
  "") ;;
  *) echo "usage: $0 [--static]" >&2; exit 2 ;;
esac

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
  shellcheck_log="${TMPDIR:-/tmp}/_focr_shellcheck.$$"
  if shellcheck -S warning "$INSTALL_SH" >"$shellcheck_log" 2>&1; then
    pass "shellcheck -S warning"
  else
    bad "shellcheck -S warning"; sed 's/^/      /' "$shellcheck_log"
  fi
  if [ "${FOCR_PRESERVE_TMP:-0}" = "1" ]; then
    skip "preserved shellcheck log: $shellcheck_log"
  else
    rm -f "$shellcheck_log"
  fi
else
  skip "shellcheck not installed"
fi

if (
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  VERSION=""
  unset FOCR_INSTALL_BASE_URL
  curl() { return 22; }
  resolve_version
) >/dev/null 2>&1; then
  bad "latest-release resolution accepted a GitHub API failure"
else
  pass "latest-release resolution fails closed on a GitHub API failure"
fi

if (
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  VERSION=""
  FOCR_INSTALL_BASE_URL="file:///fixture"
  resolve_version
) >/dev/null 2>&1; then
  bad "custom release base accepted an implicit historical version"
else
  pass "custom release base requires an explicit fixture/release version"
fi

if (
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  VERSION="1.2.3"
  # Read indirectly by resolve_version after sourcing the installer.
  # shellcheck disable=SC2034
  FOCR_INSTALL_BASE_URL="file:///fixture"
  curl() { return 99; }
  resolve_version
  normalize_version
  [ "$VERSION" = "v1.2.3" ]
) >/dev/null 2>&1; then
  pass "explicit bare semver bypasses discovery and normalizes to a release tag"
else
  bad "explicit version behavior regressed"
fi

if (
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  run_focr_version() { return 1; }
  verify_install
) >/dev/null 2>&1; then
  bad "verify_install accepted a non-executable installed binary"
else
  pass "verify_install fails closed when the installed binary cannot run"
fi

if (
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  VERSION="v1.2.3"
  run_focr_version() { printf '%s\n' 'focr 9.9.9'; }
  verify_install
) >/dev/null 2>&1; then
  bad "verify_install accepted a binary from a different release"
else
  pass "verify_install rejects a reported semver that differs from the requested tag"
fi

if (
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  VERSION="v1.2.3"
  run_focr_version() { printf '%s\n' 'focr 1.2.3'; }
  verify_install
) >/dev/null 2>&1; then
  pass "verify_install accepts the exact requested semver"
else
  bad "verify_install rejected the exact requested semver"
fi

if (
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  export DEST=/usr/bin
  export BINARY_NAME=false
  run_selftest
) >/dev/null 2>&1; then
  bad "run_selftest accepted a divergent kernel verdict"
else
  pass "requested selftest fails closed on a non-zero verdict"
fi

atomic_work=$(mktemp -d)
mkdir -p "$atomic_work/dest" "$atomic_work/tmp"
cat > "$atomic_work/dest/focr" <<'OLD'
#!/bin/sh
echo 'focr 0.1.0'
OLD
cat > "$atomic_work/tmp/focr-test" <<'NEW'
#!/bin/sh
echo 'focr 0.2.0'
NEW
chmod +x "$atomic_work/dest/focr" "$atomic_work/tmp/focr-test"
cp "$atomic_work/dest/focr" "$atomic_work/old-focr"
if (
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  DEST="$atomic_work/dest"
  # Read indirectly by install_binary imported above.
  # shellcheck disable=SC2034
  TMP="$atomic_work/tmp"
  # shellcheck disable=SC2034
  ASSET="focr-test"
  VERSION="v0.2.0"
  # shellcheck disable=SC2034
  FOCR_INSTALL_TEST_MODE=1
  # shellcheck disable=SC2034
  FOCR_INSTALL_TEST_FAILPOINT=before-replace
  install_binary
) >/dev/null 2>&1; then
  bad "injected pre-replace failure unexpectedly succeeded"
elif ! cmp -s "$atomic_work/old-focr" "$atomic_work/dest/focr"; then
  bad "pre-replace failure changed the existing executable"
else
  pass "same-directory atomic staging preserves the old executable on failure"
fi

lock_dest="$atomic_work/lock-dest"
lock_ready="$atomic_work/lock-ready"
mkdir -p "$lock_dest"
(
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  DEST="$lock_dest"
  acquire_lock
  trap release_lock EXIT
  : > "$lock_ready"
  sleep 3
) >/dev/null 2>&1 &
lock_holder=$!
for _ in 1 2 3 4 5; do
  [ -f "$lock_ready" ] && break
  sleep 1
done
if [ ! -f "$lock_ready" ]; then
  bad "installer lock holder did not start"
elif (
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  DEST="$lock_dest"
  acquire_lock
) >/dev/null 2>&1; then
  bad "destination-scoped installer lock allowed a concurrent replacement"
else
  pass "destination-scoped installer lock rejects a concurrent replacement"
fi
wait "$lock_holder" || bad "installer lock holder exited non-zero"

crash_ready="$atomic_work/crash-ready"
bash -c '
  installer=$1
  destination=$2
  ready=$3
  set --
  source "$installer"
  DEST="$destination"
  acquire_lock
  : > "$ready"
  sleep 30
' _ "$INSTALL_SH" "$lock_dest" "$crash_ready" >/dev/null 2>&1 &
crash_holder=$!
for _ in 1 2 3 4 5; do
  [ -f "$crash_ready" ] && break
  sleep 1
done
if [ ! -f "$crash_ready" ]; then
  bad "crash-recovery lock holder did not start"
else
  kill -9 "$crash_holder" 2>/dev/null || true
  wait "$crash_holder" 2>/dev/null || true
  crash_recovered=0
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if (
      set --
      # shellcheck disable=SC1090
      source "$INSTALL_SH"
      DEST="$lock_dest"
      acquire_lock
      release_lock
    ) >/dev/null 2>&1; then
      crash_recovered=1
      break
    fi
    sleep 1
  done
  if [ "$crash_recovered" -eq 1 ]; then
    pass "kernel installer lock auto-recovers after owner process death"
  else
    bad "kernel installer lock survived owner process death"
  fi
fi

race_winners="$atomic_work/race-winners"
: > "$race_winners"
run_lock_contender() {
  (
    set --
    # shellcheck disable=SC1090
    source "$INSTALL_SH"
    DEST="$lock_dest"
    acquire_lock
    trap release_lock EXIT
    printf 'winner\n' >> "$race_winners"
    sleep 2
  ) >/dev/null 2>&1
}
run_lock_contender &
race_one=$!
run_lock_contender &
race_two=$!
race_successes=0
if wait "$race_one"; then race_successes=$((race_successes + 1)); fi
if wait "$race_two"; then race_successes=$((race_successes + 1)); fi
if [ "$race_successes" -eq 1 ] && [ "$(wc -l < "$race_winners" | tr -d ' ')" -eq 1 ]; then
  pass "simultaneous installer contenders elect exactly one lock owner"
else
  bad "simultaneous installer contenders did not serialize exclusively"
fi

fast_path_log="$atomic_work/fast-path.log"
if (
  set --
  # shellcheck disable=SC1090
  source "$INSTALL_SH"
  setup_proxy() { :; }
  print_banner() { :; }
  detect_windowsish() { :; }
  detect_platform() { :; }
  resolve_version() { VERSION=v1.2.3; }
  normalize_version() { :; }
  set_urls() { :; }
  preflight_checks() { :; }
  acquire_lock() { printf 'lock\n' >> "$fast_path_log"; }
  cleanup() { printf 'cleanup\n' >> "$fast_path_log"; }
  check_installed_version() { printf 'check\n' >> "$fast_path_log"; return 0; }
  maybe_add_path() { :; }
  verify_install() { printf 'verify\n' >> "$fast_path_log"; }
  run_selftest() { printf 'selftest\n' >> "$fast_path_log"; }
  VERIFY=1
  FORCE_INSTALL=0
  main
) >/dev/null 2>&1; then
  expected_fast_path='lock
check
verify
selftest
cleanup'
  if [ "$(cat "$fast_path_log")" = "$expected_fast_path" ]; then
    pass "exact-version fast path locks and honors explicit verification"
  else
    bad "exact-version fast path did not lock and verify in order"
  fi
else
  bad "exact-version verified fast path failed"
fi
rm -rf "$atomic_work"

if command -v pwsh >/dev/null 2>&1; then
  if powershell_parse_output="$(
    # shellcheck disable=SC2016
    INSTALL_PS1="$INSTALL_PS1" pwsh -NoLogo -NoProfile -NonInteractive -Command '
      $ErrorActionPreference = "Stop"
      $tokens = $null
      $errors = $null
      [void][System.Management.Automation.Language.Parser]::ParseFile(
        $env:INSTALL_PS1,
        [ref]$tokens,
        [ref]$errors
      )
      if ($errors.Count -gt 0) {
        $errors | ForEach-Object { [Console]::Error.WriteLine($_.ToString()) }
        exit 1
      }
    ' 2>&1
  )"; then
    pass "PowerShell installer parses"
  else
    bad "PowerShell installer has a parse error"
    printf '%s\n' "$powershell_parse_output" >&2
  fi
else
  skip "pwsh not installed — PowerShell parse check not exercised"
fi

if grep -Fq 'FallbackVersion' "$INSTALL_PS1" ||
  grep -Fq 'falls back to v0.4.0' "$INSTALL_SH" "$INSTALL_PS1"; then
  bad "installer sources still contain the silent historical-version fallback"
elif ! grep -Fq 'Confirm-Install -Exe $target -Version $version' "$INSTALL_PS1" ||
  ! grep -Fq 'if ($reported -cne $expected)' "$INSTALL_PS1" ||
  ! grep -Fq 'return ($reported -ceq $want)' "$INSTALL_PS1" ||
  ! grep -Fq '$stagedSemVer -cne $expectedVersion' "$INSTALL_PS1" ||
  ! grep -Fq '$Value -cnotmatch' "$INSTALL_PS1"; then
  bad "PowerShell post-install check is not bound to the requested semver"
else
  pass "PowerShell version verification is exact and case sensitive"
fi

if ! grep -Fq '[string]$OfflineAssetDir' "$INSTALL_PS1" ||
  ! grep -Fq '[System.IO.File]::Replace($staged, $target, $backup)' "$INSTALL_PS1" ||
  ! grep -Fq '[System.IO.FileShare]::None' "$INSTALL_PS1" ||
  ! grep -Fq 'Enter-InstallLock' "$INSTALL_PS1" ||
  ! grep -Fq "FOCR_INSTALL_TEST_FAILPOINT -eq 'replace-target-missing'" "$INSTALL_PS1" ||
  ! grep -Fq 'throw "focr installer failed with exit code $exitCode."' "$INSTALL_PS1"; then
  bad "PowerShell installer lacks offline, atomic-replace, or process-lock support"
else
  pass "PowerShell installer exposes offline replacement, recovery, and cross-session locking"
fi

matrix_contract_ok=1
for required in "$DIST_YML" "$CI_YML" "$INSTALL_PS1" "$CHECK_SH" "$TOOLCHAIN_TOML" "$GITATTRIBUTES"; do
  if [ ! -f "$required" ]; then
    bad "release contract input is missing: $required"
    matrix_contract_ok=0
  fi
done

if [ "$matrix_contract_ok" -eq 1 ]; then
  expected_assets='focr-aarch64-apple-darwin-neon-sdot-i8mm
focr-x86_64-apple-darwin
focr-x86_64-unknown-linux-gnu
focr-aarch64-unknown-linux-gnu
focr-x86_64-pc-windows-msvc.exe
focr-aarch64-pc-windows-msvc.exe'
  while IFS= read -r asset; do
    if ! grep -Fq -- "asset: $asset" "$DIST_YML"; then
      bad "dist matrix does not stage installer asset: $asset"
      matrix_contract_ok=0
    fi
  done <<EOF
$expected_assets
EOF

  if ! grep -Fq 'out="${{ matrix.asset }}"' "$DIST_YML" ||
    ! grep -Fq '$asset = "${{ matrix.asset }}"' "$DIST_YML"; then
    bad "dist staging does not consume the explicit Unix and Windows asset fields"
    matrix_contract_ok=0
  fi

  if ! grep -Fq -- 'rustflags: ""' "$DIST_YML" ||
    grep -Eq 'asset: focr-x86_64-unknown-linux-gnu$' "$DIST_YML" &&
      grep -Eq 'rustflags: "-C target-feature=' "$DIST_YML"; then
    bad "installer-served Linux x86_64 asset is not a portable baseline build"
    matrix_contract_ok=0
  fi
  if ! grep -Fq 'Smoke test focr (no weights)' "$DIST_YML"; then
    bad "Unix dist artifacts are not executed before staging"
    matrix_contract_ok=0
  fi
  if ! grep -Fq -- "--target '\${{ matrix.target }}.\${{ matrix.glibc_floor }}'" "$DIST_YML" ||
    ! grep -Fq -- "--dist-glibc-floor '\${{ matrix.glibc_floor }}'" "$DIST_YML"; then
    bad "Linux dist assets do not target and certify an explicit glibc floor"
    matrix_contract_ok=0
  fi
  if ! grep -Fq 'Test exact staged asset through offline install.ps1' "$DIST_YML" ||
    ! grep -Fq -- '-OfflineAssetDir $releaseDir' "$DIST_YML"; then
    bad "Windows dist does not exercise install.ps1 against the exact offline asset"
    matrix_contract_ok=0
  fi
  if grep -Fq '& $asset --version | Select-Object -First 1' "$DIST_YML" ||
    ! grep -Fq '$versionOutput = @(& $asset --version)' "$DIST_YML" ||
    ! grep -Fq '$versionExit = $LASTEXITCODE' "$DIST_YML"; then
    bad "Windows dist truncates the multi-line version probe or loses its exit code"
    matrix_contract_ok=0
  fi
  if ! grep -Fxq 'src/native_engine/unlimited_ocr_manifest.json text eol=lf' "$GITATTRIBUTES" ||
    ! grep -Fxq 'models/manifest-v2.json text eol=lf' "$GITATTRIBUTES" ||
    ! grep -Fq 'git config --global core.autocrlf false' "$DIST_YML" ||
    [ "$(grep -Fc '24ff1cfffe71eec6f07bfae8e8eb12b342877bccc43ad6ae4a4a4ceffb76edd3' "$DIST_YML")" -lt 2 ]; then
    bad "embedded release manifests are not protected from Windows CRLF translation"
    matrix_contract_ok=0
  fi
  if [ "$(grep -Fc -- '--dist-ref-preflight' "$DIST_YML")" -lt 2 ]; then
    bad "not every dist build job fails closed on ref/version ancestry"
    matrix_contract_ok=0
  fi
  if [ "$(grep -Fc 'ref: ${{ inputs.source_ref || github.ref }}' "$DIST_YML")" -lt 2 ] ||
    [ "$(grep -Fc 'repair workflow is not current origin/main' "$DIST_YML")" -lt 2 ] ||
    [ "$(grep -Fc 'git", "cat-file", "blob"' "$DIST_YML")" -lt 2 ] ||
    [ "$(grep -Fc 'git hash-object $evidenceScript' "$DIST_YML")" -lt 2 ] ||
    [ "$(grep -Fc 'python $evidenceScript --dist-ref-preflight' "$DIST_YML")" -ne 2 ] ||
    [ "$(grep -Fc 'authenticated dist ref preflight exited' "$DIST_YML")" -ne 2 ] ||
    [ "$(grep -Fc 'FOCR_DIST_EVIDENCE_SCRIPT=' "$DIST_YML")" -lt 2 ] ||
    ! grep -Fq 'source_ref repair release inputs differ from immutable tag' "$GAUNTLET_CERT" ||
    ! grep -Fq 'immutable-tag repair is unsupported across release-input drift' "$GAUNTLET_CERT" ||
    ! grep -Fq 'python3 "$FOCR_DIST_EVIDENCE_SCRIPT"' "$DIST_YML" ||
    ! grep -Fq 'python $env:FOCR_DIST_EVIDENCE_SCRIPT' "$DIST_YML"; then
    bad "dist immutable-tag repair is not source- and workflow-bound"
    matrix_contract_ok=0
  fi

  for asset in \
    focr-aarch64-apple-darwin-neon-sdot-i8mm \
    focr-x86_64-apple-darwin \
    focr-x86_64-unknown-linux-gnu \
    focr-aarch64-unknown-linux-gnu; do
    if ! grep -Fq -- "$asset" "$INSTALL_SH"; then
      bad "shell installer is missing dist asset: $asset"
      matrix_contract_ok=0
    fi
  done
  for asset in focr-x86_64-pc-windows-msvc.exe focr-aarch64-pc-windows-msvc.exe; do
    if ! grep -Fq -- "$asset" "$INSTALL_PS1"; then
      bad "PowerShell installer is missing dist asset: $asset"
      matrix_contract_ok=0
    fi
  done

  if ! grep -Fqx 'channel = "nightly-2026-07-09"' "$TOOLCHAIN_TOML"; then
    bad "rust-toolchain.toml is not pinned to nightly-2026-07-09"
    matrix_contract_ok=0
  fi
  for workflow in "$CI_YML" "$DIST_YML"; do
    if ! grep -Fq 'RUST_TOOLCHAIN: nightly-2026-07-09' "$workflow"; then
      bad "$(basename "$workflow") does not install the pinned repo toolchain"
      matrix_contract_ok=0
    fi
    if grep -Eq 'uses: [^#]+@(v[0-9]+|nightly)[[:space:]]*$' "$workflow"; then
      bad "$(basename "$workflow") contains a floating action ref"
      matrix_contract_ok=0
    fi
    if grep -Fq 'git clone --depth 1 https://github.com/Dicklesworthstone/' "$workflow"; then
      bad "$(basename "$workflow") contains a floating sibling checkout"
      matrix_contract_ok=0
    fi
    unlocked=$(grep -E '^[[:space:]]*run: cargo (build|check|clippy|test)' "$workflow" | grep -Fv -- '--locked' || true)
    if [ -n "$unlocked" ]; then
      bad "$(basename "$workflow") contains an unlocked Cargo command: $unlocked"
      matrix_contract_ok=0
    fi
  done
  unlocked=$(grep -E '^run cargo (check|clippy|test)' "$CHECK_SH" | grep -Fv -- '--locked' || true)
  if [ -n "$unlocked" ]; then
    bad "scripts/check.sh contains an unlocked Cargo command: $unlocked"
    matrix_contract_ok=0
  fi

  if [ "$matrix_contract_ok" -eq 1 ]; then
    pass "release matrix inputs are pinned and installer asset names match"
  fi
fi

if [ "$STATIC_ONLY" -eq 1 ]; then
  exit "$fail"
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
      # Consumed by the status helpers imported from install.sh.
      # shellcheck disable=SC2034
      HAS_GUM=1
      # shellcheck disable=SC2034
      NO_GUM=0
      # shellcheck disable=SC2034
      QUIET=0
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
    if [ "${FOCR_PRESERVE_TMP:-0}" = "1" ]; then
      trap ':' EXIT
      skip "preserving installer e2e workspace: $work"
    else
      trap 'rm -rf "$work"' EXIT
    fi
    rel="$work/release"; fakehome="$work/home"
    test_tmp="$work/tmp"
    # IMPORTANT: a NESTED, not-yet-existing install dir. This reproduces the second
    # `set -e` blocker class (check_disk_space running `df` on a path whose parent
    # does not exist yet — the default ~/.local/bin on a fresh account). The
    # installer must create it via check_write_permissions, not abort on the df.
    bindir="$work/fresh/account/.local/bin"
    mkdir -p "$rel" "$fakehome" "$test_tmp"  # NB: bindir intentionally NOT created here

    # Fake focr: compile a host-native stub so Gate 2 exercises the same kind of
    # executable as a release asset. A shebang script takes a different macOS
    # execution-policy path and can stall before its interpreter reaches main.
    stub_src="$work/focr_stub.rs"
    cat > "$stub_src" <<'STUB'
fn main() {
    if std::env::args().nth(1).as_deref() == Some("--version") {
        println!("focr 0.2.0");
    }
}
STUB
    if ! rustc --edition=2024 -C opt-level=0 -C debuginfo=0 -o "$rel/$asset" "$stub_src"; then
      bad "failed to compile the native fake-release focr"
      asset=""
    fi

    # SHA256 sidecar in the installer's expected "<hex>  <asset>" format.
    if [ -n "$asset" ] && command -v sha256sum >/dev/null 2>&1; then
      ( cd "$rel" && sha256sum "$asset" > "$asset.sha256" )
    elif [ -n "$asset" ] && command -v shasum >/dev/null 2>&1; then
      ( cd "$rel" && shasum -a 256 "$asset" > "$asset.sha256" )
    elif [ -n "$asset" ]; then
      skip "no sha256 tool"; asset=""
    fi

    if [ -n "$asset" ]; then
      log="$work/transcript.txt"
      args="--version v0.2.0 --dir $bindir --no-pull --offline --force --verify"
      # Run the REAL installer under a pty so `[ -t 1 ]` is true and the gum path
      # activates exactly as it does for a user. file:// base = no network. HOME
      # is sandboxed so nothing touches the developer's shell rc or model cache.
      if [ "$os" = "darwin" ]; then
        env HOME="$fakehome" TMPDIR="$test_tmp" FOCR_INSTALL_BASE_URL="file://$rel" \
          script -q /dev/null bash "$INSTALL_SH" $args >"$log" 2>&1
        rc=$?
      else
        env HOME="$fakehome" TMPDIR="$test_tmp" FOCR_INSTALL_BASE_URL="file://$rel" \
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
        if [ "$v" = 'focr 0.2.0' ]; then
          pass "installer ran clean end-to-end and installed a working focr ($v)"
        else
          bad "installed focr reported the wrong release version: '$v'"
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
