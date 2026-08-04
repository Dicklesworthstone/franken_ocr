#!/usr/bin/env bash
#
# focr installer (franken_ocr)
#
# focr is a pure-Rust, CPU-only OCR command-line tool. It parses document
# images into structured markdown or JSON using the Baidu Unlimited-OCR
# vision-language model. No Python, no CUDA, no GPU.
#
# One-liner install (with cache buster):
#   curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.sh?$(date +%s)" | bash
#
# Or without cache buster:
#   curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.sh | bash
#
# Options:
#   --version vX.Y.Z   Install a specific version (default: latest release)
#   --dir DIR          Install the binary into DIR (default: ~/.local/bin)
#   --easy-mode        Add the install directory to PATH in your shell rc files
#   --verify           Run "focr robot selftest" after install
#   --no-pull          Suppress the post-install model download prompt/guidance
#   --no-verify        Skip SHA256 checksum verification (not recommended)
#   --offline          Skip the network reachability preflight check
#   --quiet            Suppress non-error output
#   --no-gum           Disable gum formatting even when gum is available
#   --force            Reinstall even when the same version is present
#   --help             Show this help and exit
#
# Environment:
#   PREFIX                 Install into $PREFIX/bin instead of ~/.local/bin
#   VERSION                Same as --version
#   HTTPS_PROXY            HTTPS proxy for downloads (preferred)
#   HTTP_PROXY             HTTP proxy for downloads
#   FOCR_INSTALL_BASE_URL  Override the release base (corporate mirror / airgap
#                          cache / the e2e installer test). The asset and its
#                          .sha256 sidecar are fetched from directly under it;
#                          --version or VERSION is required with this override.
#
# WINDOWS
#   Native x86_64 and ARM64 Windows binaries use the MSVC target names consumed by
#   install.ps1. This POSIX installer cannot install them from a Git-Bash/MSYS/
#   Cygwin shell, so there it points you at the PowerShell installer:
#     irm https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1 | iex
#   Under WSL this installer proceeds as Linux.
#
# BUILD REALITY
#   franken_ocr path-depends on sibling repos that are not published to
#   crates.io (asupersync, frankentorch, frankensqlite). A fresh "cargo install"
#   or "cargo build" cannot resolve those dependencies, so this installer does
#   not offer a from-crates.io source build. Prebuilt binaries are the supported
#   path. On an unsupported platform the installer reports that honestly and
#   exits.
#
set -euo pipefail
umask 022
shopt -s lastpipe 2>/dev/null || true

# ============================================================================
# Configuration
# ============================================================================
OWNER="${OWNER:-Dicklesworthstone}"
REPO="${REPO:-franken_ocr}"
BINARY_NAME="focr"
VERSION="${VERSION:-}"

# Install directory: --dir wins, then PREFIX/bin, then ~/.local/bin.
DEST_DEFAULT="$HOME/.local/bin"
if [ -n "${PREFIX:-}" ]; then
  DEST_DEFAULT="${PREFIX%/}/bin"
fi
DEST="$DEST_DEFAULT"

EASY=0
QUIET=0
VERIFY=0
FORCE_INSTALL=0
NO_GUM=0
NO_PULL=0
NO_VERIFY=0
OFFLINE=0
# Runtime globals (initialized so set -u never trips before they are assigned).
TMP=""
LOCK_DIR=""
LOCK_HOLDER_PID=""
LOCK_READY_DIR=""
STAGED_PATH=""
PROXY_ARGS=()
OS=""
ARCH=""
ASSET=""
TARGET=""
BASE_URL=""
ASSET_URL=""
SHA_URL=""
IS_WSL=0
INSTALLED_VERSION_STR=""

# ============================================================================
# Output: gum when available, plain ANSI otherwise
# ============================================================================
HAS_GUM=0
if command -v gum >/dev/null 2>&1 && [ -t 1 ]; then
  HAS_GUM=1
fi

info() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 39 -- "-> $*"
  else
    echo -e "\033[0;34m->\033[0m $*"
  fi
}

ok() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 42 -- "ok $*"
  else
    echo -e "\033[0;32mok\033[0m $*"
  fi
}

warn() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 214 -- "warn $*"
  else
    echo -e "\033[1;33mwarn\033[0m $*"
  fi
}

# err is never silenced by --quiet; failures must always be visible.
err() {
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 196 -- "error $*"
  else
    echo -e "\033[0;31merror\033[0m $*"
  fi
}

# Spinner wrapper. gum spin can only run external commands, so when the target
# is a shell function we fall back to a plain log line and run it directly.
run_with_spinner() {
  local title="$1"
  shift
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ] && [ "$QUIET" -eq 0 ]; then
    if declare -F "$1" >/dev/null 2>&1; then
      info "$title"
      "$@"
    else
      gum spin --spinner dot --title "$title" -- "$@"
    fi
  else
    info "$title"
    "$@"
  fi
}

# Draw a box around text with automatic width calculation.
# Usage: draw_box "color" "line1" "line2" ...
draw_box() {
  local color="$1"
  shift
  local lines=("$@")
  local max_width=0
  local esc
  esc=$(printf '\033')
  local strip_ansi_sed="s/${esc}\\[[0-9;]*m//g"

  local line stripped len
  for line in "${lines[@]}"; do
    stripped=$(printf '%b' "$line" | LC_ALL=C sed "$strip_ansi_sed")
    len=${#stripped}
    if [ "$len" -gt "$max_width" ]; then
      max_width=$len
    fi
  done

  local inner_width=$((max_width + 4))
  local border=""
  local i
  for ((i = 0; i < inner_width; i++)); do
    border+="="
  done

  printf "\033[%sm+%s+\033[0m\n" "$color" "$border"
  for line in "${lines[@]}"; do
    stripped=$(printf '%b' "$line" | LC_ALL=C sed "$strip_ansi_sed")
    len=${#stripped}
    local padding=$((max_width - len))
    local pad_str=""
    for ((i = 0; i < padding; i++)); do
      pad_str+=" "
    done
    printf "\033[%sm|\033[0m  %b%s  \033[%sm|\033[0m\n" "$color" "$line" "$pad_str" "$color"
  done
  printf "\033[%sm+%s+\033[0m\n" "$color" "$border"
}

print_banner() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style \
      --border normal \
      --border-foreground 39 \
      --padding "0 1" \
      --margin "1 0" \
      "$(gum style --foreground 42 --bold 'focr installer')" \
      "$(gum style --foreground 245 'Pure-Rust CPU OCR for document images (franken_ocr)')"
  else
    echo ""
    echo -e "\033[1;32mfocr installer\033[0m"
    echo -e "\033[0;90mPure-Rust CPU OCR for document images (franken_ocr)\033[0m"
    echo ""
  fi
}

# ============================================================================
# Help
# ============================================================================
usage() {
  cat <<EOF
focr installer (franken_ocr): pure-Rust CPU OCR for document images

Usage:
  curl -fsSL https://raw.githubusercontent.com/${OWNER}/${REPO}/main/install.sh | bash
  curl -fsSL .../install.sh | bash -s -- [OPTIONS]

Options:
  --version vX.Y.Z   Install a specific version (default: latest release)
  --dir DIR          Install the binary into DIR (default: ~/.local/bin)
  --easy-mode        Add the install directory to PATH in your shell rc files
  --verify           Run "focr robot selftest" after install
  --no-pull          Suppress the post-install model download prompt/guidance
  --no-verify        Skip SHA256 checksum verification (not recommended)
  --offline          Skip the network reachability preflight check
  --quiet            Suppress non-error output
  --no-gum           Disable gum formatting even when gum is available
  --force            Reinstall even when the same version is present
  --help             Show this help and exit

Environment:
  PREFIX                 Install into \$PREFIX/bin instead of ~/.local/bin
  VERSION                Same as --version
  HTTPS_PROXY            HTTPS proxy for downloads (preferred)
  HTTP_PROXY             HTTP proxy for downloads
  FOCR_INSTALL_BASE_URL  Override the release base URL (requires --version / VERSION)

Platforms with prebuilt binaries:
  macOS Apple Silicon, macOS Intel, Linux x86-64 (glibc), Linux ARM64 (glibc)
  Windows x86-64 and ARM64: native focr.exe via the PowerShell installer
  (install.ps1), or run this script inside WSL2.

After install, download the default model once (about 4.2 GB):  focr pull
Then parse a page with:                                  focr ocr page.png
EOF
}

# ============================================================================
# Argument parsing
# ============================================================================
require_value() {
  local opt="$1"
  local val="${2:-}"
  if [ -z "$val" ] || [ "${val:0:1}" = "-" ]; then
    err "$opt requires a value."
    exit 2
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version) require_value "$1" "${2:-}"; VERSION="$2"; shift 2 ;;
    --version=*) VERSION="${1#*=}"; shift ;;
    --dir) require_value "$1" "${2:-}"; DEST="$2"; shift 2 ;;
    --dir=*) DEST="${1#*=}"; shift ;;
    --easy-mode) EASY=1; shift ;;
    --verify) VERIFY=1; shift ;;
    --no-pull) NO_PULL=1; shift ;;
    --no-verify) NO_VERIFY=1; shift ;;
    --offline) OFFLINE=1; shift ;;
    --quiet|-q) QUIET=1; shift ;;
    --no-gum) NO_GUM=1; shift ;;
    --force) FORCE_INSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) warn "Ignoring unknown option: $1"; shift ;;
  esac
done

# ============================================================================
# Proxy support: build a curl argument array honored by every download
# ============================================================================
setup_proxy() {
  PROXY_ARGS=()
  if [ -n "${HTTPS_PROXY:-}" ]; then
    PROXY_ARGS=(--proxy "$HTTPS_PROXY")
    info "Using HTTPS proxy: $HTTPS_PROXY"
  elif [ -n "${HTTP_PROXY:-}" ]; then
    PROXY_ARGS=(--proxy "$HTTP_PROXY")
    info "Using HTTP proxy: $HTTP_PROXY"
  fi
}

# Download a single URL to a destination path, honoring proxy settings.
fetch() {
  curl -fsSL ${PROXY_ARGS[@]+"${PROXY_ARGS[@]}"} \
    --connect-timeout 30 --max-time 600 "$1" -o "$2"
}

# ============================================================================
# Windows-ish detection
# ============================================================================
print_windows_note() {
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    echo ""
    gum style \
      --border normal \
      --border-foreground 214 \
      --padding "1 2" \
      --margin "1 0" \
      "$(gum style --foreground 214 --bold 'Native Windows is supported')" \
      "" \
      "$(gum style --foreground 252 'install.ps1 supports native x86_64 and ARM64 release assets.')" \
      "$(gum style --foreground 252 'This shell (Git-Bash/MSYS/Cygwin) cannot install it; use PowerShell.')" \
      "" \
      "$(gum style --foreground 42 'In a PowerShell window, run:')" \
      "$(gum style --foreground 42 'irm https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1 | iex')" \
      "" \
      "$(gum style --foreground 245 'Alternative: install and run focr inside WSL2, then re-run this installer there.')"
    echo ""
  else
    echo ""
    draw_box "1;33" \
      "Native Windows is supported" \
      "" \
      "install.ps1 supports native x86_64 and ARM64 release assets." \
      "This shell (Git-Bash/MSYS/Cygwin) cannot install it; use PowerShell." \
      "" \
      "In a PowerShell window, run:" \
      "irm https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1 | iex" \
      "" \
      "Alternative: install and run focr inside WSL2, then re-run this installer there."
    echo ""
  fi
}

detect_windowsish() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      print_windows_note
      exit 0
      ;;
  esac
  if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=1
  fi
}

# ============================================================================
# Platform detection: map os/arch to the exact published asset name
# ============================================================================
unsupported_platform() {
  err "No prebuilt focr binary is available for ${OS}/${ARCH}."
  err "Supported: macOS (arm64, x86_64) and Linux glibc (x86_64, aarch64)."
  err "A from-source build is not offered: franken_ocr path-depends on"
  err "asupersync, frankentorch, and frankensqlite, which are not on crates.io,"
  err "so 'cargo install' and a fresh 'cargo build' cannot resolve them."
  err "Questions: https://github.com/${OWNER}/${REPO}/issues"
  exit 1
}

detect_platform() {
  OS=$(uname -s | tr '[:upper:]' '[:lower:]')
  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64|amd64) ARCH="x86_64" ;;
    arm64|aarch64) ARCH="aarch64" ;;
  esac

  case "${OS}-${ARCH}" in
    darwin-aarch64)
      # The macOS Apple Silicon asset carries its ISA tier in the filename.
      ASSET="focr-aarch64-apple-darwin-neon-sdot-i8mm"
      TARGET="aarch64-apple-darwin"
      ;;
    darwin-x86_64)
      ASSET="focr-x86_64-apple-darwin"
      TARGET="x86_64-apple-darwin"
      ;;
    linux-x86_64)
      ASSET="focr-x86_64-unknown-linux-gnu"
      TARGET="x86_64-unknown-linux-gnu"
      ;;
    linux-aarch64)
      ASSET="focr-aarch64-unknown-linux-gnu"
      TARGET="aarch64-unknown-linux-gnu"
      ;;
    *)
      unsupported_platform
      ;;
  esac
}

# ============================================================================
# Version resolution and URL construction
# ============================================================================
resolve_version() {
  if [ -n "$VERSION" ]; then return 0; fi
  # A custom base has no discovery contract. Require an explicit version so a
  # mirror/airgap install cannot silently select an unrelated historical build.
  if [ -n "${FOCR_INSTALL_BASE_URL:-}" ]; then
    err "FOCR_INSTALL_BASE_URL requires --version (or VERSION) because latest-release discovery is unavailable."
    exit 1
  fi

  info "Resolving the latest release..."
  local api="https://api.github.com/repos/${OWNER}/${REPO}/releases/latest"
  local tag=""

  tag=$(curl -fsSL ${PROXY_ARGS[@]+"${PROXY_ARGS[@]}"} \
    -H "Accept: application/vnd.github.v3+json" \
    --connect-timeout 10 --max-time 30 "$api" 2>/dev/null \
    | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' | head -1) || tag=""

  if [ -n "$tag" ]; then
    VERSION="$tag"
    info "Latest release: $VERSION"
  else
    err "Could not resolve the latest release from the GitHub API."
    err "Re-run with --version vX.Y.Z to pin a known release."
    exit 1
  fi
}

# GitHub release tags are v-prefixed; accept a bare semver from --version too.
normalize_version() {
  case "$VERSION" in
    [0-9]*) VERSION="v$VERSION" ;;
  esac
  local semver_re='^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$'
  if [[ ! "$VERSION" =~ $semver_re ]]; then
    err "Invalid version '$VERSION'; expected vX.Y.Z (or a valid semver prerelease)."
    exit 2
  fi
}

set_urls() {
  # FOCR_INSTALL_BASE_URL overrides the release base (corporate mirror, airgap
  # cache, or the e2e installer test serving a fake release over file://). When
  # unset, the canonical GitHub releases URL is used. The asset + .sha256 sidecar
  # are expected directly under the base.
  BASE_URL="${FOCR_INSTALL_BASE_URL:-https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}}"
  ASSET_URL="${BASE_URL}/${ASSET}"
  SHA_URL="${ASSET_URL}.sha256"
}

# ============================================================================
# Installed-version probe (for the already-installed short-circuit)
# ============================================================================
run_focr_version_path() {
  local bin="$1"
  [ -x "$bin" ] || return 1
  local out=""
  if command -v timeout >/dev/null 2>&1; then
    out=$(timeout 30 "$bin" --version 2>/dev/null) || out=""
  elif command -v gtimeout >/dev/null 2>&1; then
    out=$(gtimeout 30 "$bin" --version 2>/dev/null) || out=""
  else
    out=$("$bin" --version 2>/dev/null) || out=""
  fi
  printf '%s\n' "${out%%$'\n'*}"
}

run_focr_version() {
  run_focr_version_path "$DEST/$BINARY_NAME"
}

reported_focr_semver() {
  local line="$1"
  local report_re='^focr[[:space:]]+([0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?)$'
  if [[ "$line" =~ $report_re ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

check_installed_version() {
  local target="$1"
  [ -x "$DEST/$BINARY_NAME" ] || return 1
  local out installed
  out=$(run_focr_version) || return 1
  installed=$(reported_focr_semver "$out") || installed=""
  [ -n "$installed" ] || return 1
  [ "${target#v}" = "${installed#v}" ]
}

# ============================================================================
# Preflight checks
# ============================================================================
check_disk_space() {
  local min_kb=20480
  local path="$DEST"
  [ -d "$path" ] || path=$(dirname "$path")
  if command -v df >/dev/null 2>&1; then
    local avail_kb
    # `df` exits non-zero when `$path` does not exist yet (e.g. the default
    # ~/.local/bin on a fresh account, before check_write_permissions mkdir's it).
    # Under `set -euo pipefail` an unguarded assignment would inherit that failure
    # and abort the whole installer with no message — so swallow it and let the
    # `-n` guard below skip the (best-effort) space check.
    avail_kb=$(df -Pk "$path" 2>/dev/null | awk 'NR==2 {print $4}') || avail_kb=""
    if [ -n "$avail_kb" ] && [ "$avail_kb" -lt "$min_kb" ]; then
      err "Not enough free space in $path (need at least 20 MB for the binary)."
      exit 1
    fi
  fi
}

check_write_permissions() {
  if [ ! -d "$DEST" ]; then
    if ! mkdir -p "$DEST" 2>/dev/null; then
      err "Cannot create install directory: $DEST"
      err "Choose a writable --dir, or set PREFIX to a writable location."
      exit 1
    fi
  fi
  if [ ! -w "$DEST" ]; then
    err "No write permission for $DEST"
    err "Choose a writable --dir, or set PREFIX to a writable location."
    exit 1
  fi
}

check_existing_install() {
  [ -x "$DEST/$BINARY_NAME" ] || return 0
  local cur
  cur=$(run_focr_version) || cur=""
  [ -n "$cur" ] && info "Existing focr detected: $cur"
  return 0
}

check_network() {
  if [ "$OFFLINE" -eq 1 ]; then
    info "Network preflight skipped (--offline)"
    return 0
  fi
  [ -n "$ASSET_URL" ] || return 0
  if ! curl -fsSL ${PROXY_ARGS[@]+"${PROXY_ARGS[@]}"} \
      --connect-timeout 5 --max-time 10 -o /dev/null -I "$ASSET_URL" 2>/dev/null; then
    warn "Could not reach $ASSET_URL during preflight."
    warn "Continuing; the download step will report a clear error if it fails."
  fi
}

preflight_checks() {
  info "Running preflight checks"
  check_disk_space
  check_write_permissions
  check_existing_install
  check_network
}

# ============================================================================
# Destination-scoped kernel locking. The helper process owns the advisory lock
# while this shell performs the install. If this shell crashes, the helper sees
# its owner PID disappear and exits, so no stale lock can survive or be raced.
# ============================================================================
release_lock() {
  if [ -n "${LOCK_READY_DIR:-}" ]; then
    : > "$LOCK_READY_DIR/release" 2>/dev/null || true
  fi
  if [ -n "${LOCK_HOLDER_PID:-}" ]; then
    wait "$LOCK_HOLDER_PID" 2>/dev/null || true
    LOCK_HOLDER_PID=""
  fi
  if [ -n "${LOCK_READY_DIR:-}" ]; then
    rm -f "$LOCK_READY_DIR/acquired" "$LOCK_READY_DIR/release"
    rmdir "$LOCK_READY_DIR" 2>/dev/null || true
    LOCK_READY_DIR=""
  fi
}

acquire_lock() {
  # The lock follows the destination, not TMPDIR, so concurrent installers with
  # different environments and path aliases serialize the same executable.
  local holder_script owner_pid platform
  LOCK_DIR="$DEST/.focr-install.lock"
  LOCK_READY_DIR=$(mktemp -d "$DEST/.focr-install.ready.XXXXXX") || {
    err "Could not create installer lock state in $DEST."
    exit 1
  }
  # Main runs in the top-level installer process. BASHPID also keeps direct
  # function-level tests correct on modern Bash; macOS Bash 3.2 falls back to $$.
  owner_pid="${BASHPID:-$$}"
  # shellcheck disable=SC2016
  holder_script='
ready=$1
owner=$2
cleanup() {
  rm -f "$ready/acquired" "$ready/release"
  rmdir "$ready" 2>/dev/null || true
}
trap cleanup EXIT
trap "exit 0" HUP INT TERM
: > "$ready/acquired" || exit 1
while kill -0 "$owner" 2>/dev/null && [ ! -f "$ready/release" ]; do
  sleep 1
done
'
  platform=$(uname -s)
  case "$platform" in
    Darwin)
      if ! command -v lockf >/dev/null 2>&1; then
        release_lock
        err "lockf is required to serialize installation on macOS."
        exit 1
      fi
      lockf -t 0 "$LOCK_DIR" sh -c "$holder_script" sh "$LOCK_READY_DIR" "$owner_pid" &
      ;;
    Linux)
      if ! command -v flock >/dev/null 2>&1; then
        release_lock
        err "flock is required to serialize installation on Linux."
        exit 1
      fi
      flock -n "$LOCK_DIR" sh -c "$holder_script" sh "$LOCK_READY_DIR" "$owner_pid" &
      ;;
    *)
      release_lock
      err "No supported process-lock backend is available for $platform."
      exit 1
      ;;
  esac
  LOCK_HOLDER_PID=$!

  for _ in {1..50}; do
    if [ -f "$LOCK_READY_DIR/acquired" ]; then
      return 0
    fi
    if ! kill -0 "$LOCK_HOLDER_PID" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done

  : > "$LOCK_READY_DIR/release" 2>/dev/null || true
  wait "$LOCK_HOLDER_PID" 2>/dev/null || true
  LOCK_HOLDER_PID=""
  rm -f "$LOCK_READY_DIR/acquired" "$LOCK_READY_DIR/release"
  rmdir "$LOCK_READY_DIR" 2>/dev/null || true
  LOCK_READY_DIR=""
  err "Another focr installer is running (lock: $LOCK_DIR)."
  exit 1
}

cleanup() {
  [ -n "${STAGED_PATH:-}" ] && rm -f "$STAGED_PATH"
  if [ "${FOCR_PRESERVE_TMP:-0}" = "1" ]; then
    warn "Preserving installer temporary state because FOCR_PRESERVE_TMP=1: ${TMP:-<none>}"
  else
    [ -n "${TMP:-}" ] && rm -rf "$TMP"
  fi
  release_lock
}

# ============================================================================
# Download, verify, install (raw binary, no archive extraction)
# ============================================================================
download_binary() {
  if ! run_with_spinner "Downloading ${ASSET} (${VERSION})..." \
      curl -fsSL ${PROXY_ARGS[@]+"${PROXY_ARGS[@]}"} \
      --connect-timeout 30 --max-time 600 "$ASSET_URL" -o "$TMP/$ASSET"; then
    err "Failed to download ${ASSET_URL}"
    err "Verify the version exists, or pass --version to pin a known release."
    exit 1
  fi
  if [ ! -s "$TMP/$ASSET" ]; then
    err "Downloaded file is empty: ${ASSET}"
    exit 1
  fi
}

is_valid_sha256() {
  [[ "${1:-}" =~ ^[[:xdigit:]]{64}$ ]]
}

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    return 1
  fi
}

verify_download() {
  if [ "$NO_VERIFY" -eq 1 ]; then
    warn "Checksum verification skipped (--no-verify)."
    return 0
  fi

  info "Fetching checksum sidecar"
  if ! curl -fsSL ${PROXY_ARGS[@]+"${PROXY_ARGS[@]}"} \
      --connect-timeout 15 --max-time 60 "$SHA_URL" -o "$TMP/$ASSET.sha256"; then
    err "Could not fetch the checksum sidecar: $SHA_URL"
    err "Re-run with --no-verify to install without verification (not recommended)."
    exit 1
  fi

  # Sidecar format is "<hex>  <asset>"; take the first field.
  local expected
  expected=$(awk '{print $1}' "$TMP/$ASSET.sha256" | head -1) || expected=""
  if ! is_valid_sha256 "$expected"; then
    err "The checksum sidecar did not contain a valid SHA256 digest."
    exit 1
  fi

  local actual=""
  # `|| actual=""` keeps a hashing-tool failure (rare: tmpfs/ENOMEM) from aborting
  # the script under `set -e`; an empty digest then falls into the mismatch check
  # below and produces a clean error instead of a silent exit.
  actual=$(file_sha256 "$TMP/$ASSET") || actual=""
  if [ -z "$actual" ]; then
    err "No SHA256 tool found (need sha256sum or shasum)."
    err "Install one, or re-run with --no-verify to skip verification."
    exit 1
  fi

  if [ "$expected" != "$actual" ]; then
    err "Checksum mismatch for ${ASSET}"
    err "  expected: $expected"
    err "  actual:   $actual"
    err "The download may be corrupt or tampered with; aborting."
    rm -f "$TMP/$ASSET"
    exit 1
  fi
  ok "Checksum verified (${actual:0:16}...)"
}

install_binary() {
  local staged_version reported expected
  STAGED_PATH=$(mktemp "$DEST/.focr.install.XXXXXX") || {
    err "Could not create a same-directory staging file in $DEST"
    exit 1
  }
  if ! install -m 0755 "$TMP/$ASSET" "$STAGED_PATH"; then
    err "Failed to stage focr in $DEST"
    exit 1
  fi
  if ! cmp -s "$TMP/$ASSET" "$STAGED_PATH"; then
    err "Same-directory staged binary does not match the verified download."
    exit 1
  fi
  staged_version=$(run_focr_version_path "$STAGED_PATH") || staged_version=""
  reported=$(reported_focr_semver "$staged_version") || reported=""
  expected="${VERSION#v}"
  if [ -z "$reported" ] || [ "$reported" != "$expected" ]; then
    err "Staged binary failed the version check before replacement (expected $expected, reported ${reported:-<none>})."
    exit 1
  fi
  if [ "${FOCR_INSTALL_TEST_MODE:-0}" = "1" ] &&
    [ "${FOCR_INSTALL_TEST_FAILPOINT:-}" = "before-replace" ]; then
    err "Injected installer failure before atomic replacement."
    exit 1
  fi
  # STAGED_PATH is on the destination filesystem. POSIX rename therefore swaps
  # the directory entry atomically; if it fails, the prior executable remains.
  if ! mv -f "$STAGED_PATH" "$DEST/$BINARY_NAME"; then
    err "Failed to atomically replace focr at $DEST/$BINARY_NAME"
    err "If focr is currently running, stop it and re-run the installer."
    exit 1
  fi
  STAGED_PATH=""
  ok "Installed focr to $DEST/$BINARY_NAME"
}

# ============================================================================
# PATH setup
# ============================================================================
maybe_add_path() {
  case ":$PATH:" in
    *:"$DEST":*) return 0 ;;
  esac

  if [ "$EASY" -eq 1 ]; then
    local updated=0
    local rc
    for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
      if [ -f "$rc" ] && [ -w "$rc" ]; then
        if ! grep -qF "$DEST" "$rc" 2>/dev/null; then
          # The literal $PATH must reach the rc file unexpanded.
          # shellcheck disable=SC2016
          printf '\n# focr installer\nexport PATH="%s:$PATH"\n' "$DEST" >> "$rc"
        fi
        updated=1
      fi
    done
    local fish_config="$HOME/.config/fish/config.fish"
    if [ -f "$fish_config" ] && [ -w "$fish_config" ]; then
      if ! grep -qF "$DEST" "$fish_config" 2>/dev/null; then
        # The literal $PATH must reach config.fish unexpanded.
        # shellcheck disable=SC2016
        printf '\n# focr installer\nset -gx PATH %s $PATH\n' "$DEST" >> "$fish_config"
      fi
      updated=1
    fi
    if [ "$updated" -eq 1 ]; then
      warn "Updated PATH in your shell rc; restart your shell or run: export PATH=\"$DEST:\$PATH\""
    else
      warn "Add $DEST to your PATH to run focr."
    fi
  else
    warn "Add $DEST to your PATH to run focr, or re-run with --easy-mode."
  fi
}

# ============================================================================
# Post-install: version check, optional self-test, optional model pull
# ============================================================================
verify_install() {
  local v reported expected
  v=$(run_focr_version) || v=""
  if [ -z "$v" ]; then
    err "Installed binary failed its mandatory execution check: $DEST/$BINARY_NAME --version"
    exit 1
  fi
  reported=$(reported_focr_semver "$v") || reported=""
  expected="${VERSION#v}"
  if [ -z "$reported" ]; then
    err "Installed binary returned an invalid version report: '$v'"
    exit 1
  fi
  if [ "$reported" != "$expected" ]; then
    err "Installed binary version mismatch: expected $expected, reported $reported"
    exit 1
  fi
  INSTALLED_VERSION_STR="$v"
  ok "focr is working: $v"
}

run_selftest() {
  info "Running focr robot selftest..."
  if "$DEST/$BINARY_NAME" robot selftest; then
    ok "Self-test passed: the int8 kernel matches the scalar oracle on this host."
  else
    err "Self-test reported a divergence (see the JSON verdict above)."
    exit 1
  fi
}

interactive_tty() {
  [ -t 0 ] && return 0
  ( : </dev/tty ) 2>/dev/null && return 0
  return 1
}

maybe_offer_pull() {
  [ "$NO_PULL" -eq 1 ] && return 0
  local bin="$DEST/$BINARY_NAME"

  # Never auto-download multi-gigabyte weights in quiet or non-interactive runs.
  if [ "$QUIET" -eq 1 ] || ! interactive_tty; then
    info "Model weights are not bundled. Download them later with: focr pull"
    return 0
  fi

  echo ""
  info "focr needs the OCR model before it can parse a page."
  info "The download is about 4.2 GB into ~/.cache/franken_ocr/models."
  local ans=""
  printf 'Download the model now with focr pull? (y/N): '
  if ( : </dev/tty ) 2>/dev/null; then
    read -r ans </dev/tty || ans=""
  else
    read -r ans || ans=""
  fi
  case "$ans" in
    y|Y|yes|Yes|YES)
      info "Running: focr pull"
      if "$bin" pull; then
        ok "Model downloaded into ~/.cache/franken_ocr/models"
      else
        warn "focr pull did not finish. Retry later with: focr pull"
      fi
      ;;
    *)
      info "Skipped. Download the model later with: focr pull"
      ;;
  esac
}

# ============================================================================
# Final summary
# ============================================================================
print_summary() {
  [ "$QUIET" -eq 1 ] && return 0

  local version_str="${INSTALLED_VERSION_STR:-$VERSION}"
  local on_path=0
  case ":$PATH:" in
    *:"$DEST":*) on_path=1 ;;
  esac

  local lines=()
  lines+=("Version:   $version_str")
  lines+=("Location:  $DEST/$BINARY_NAME")
  lines+=("")
  if [ "$on_path" -eq 0 ]; then
    lines+=("$DEST is not on your PATH yet.")
    lines+=("Add it:    export PATH=\"$DEST:\$PATH\"")
    lines+=("Or re-run this installer with --easy-mode.")
    lines+=("")
  fi
  lines+=("First steps:")
  lines+=("  focr pull                 download the default model (about 4.2 GB)")
  lines+=("  focr ocr page.png         parse an image into markdown")
  lines+=("  focr ocr page.png --json  emit structured JSON (with bounding boxes)")
  lines+=("  focr ocr page.png -o out.md    write markdown to a file")
  lines+=("  focr ocr page.png -o out.json  write JSON (markdown + boxes) to a file")
  lines+=("  focr ocr page.png -o out.md --extract-figures   save figures next to the .md")
  lines+=("  focr robot selftest       verify the int8 kernel on this host")
  lines+=("  focr --help               full command reference")
  lines+=("")
  lines+=("Model cache: ~/.cache/franken_ocr/models")
  lines+=("Uninstall:   rm $DEST/$BINARY_NAME")
  lines+=("             rm -rf ~/.cache/franken_ocr   (removes the downloaded model)")

  echo ""
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    {
      gum style --foreground 42 --bold "focr is installed."
      echo ""
      local line
      for line in "${lines[@]}"; do
        gum style --foreground 245 -- "$line"
      done
    } | gum style --border normal --border-foreground 42 --padding "1 2"
  else
    draw_box "0;32" "focr is installed." "" "${lines[@]}"
  fi
}

# ============================================================================
# Main
# ============================================================================
main() {
  setup_proxy
  print_banner

  if ! command -v curl >/dev/null 2>&1; then
    err "curl is required to download focr. Install curl and re-run."
    exit 1
  fi

  detect_windowsish
  detect_platform
  [ "$IS_WSL" -eq 1 ] && info "WSL detected; installing the Linux binary."

  resolve_version
  normalize_version
  set_urls

  info "Platform:    ${OS}/${ARCH} (${TARGET})"
  info "Asset:       ${ASSET}"
  info "Version:     ${VERSION}"
  info "Install dir: ${DEST}"

  preflight_checks

  acquire_lock
  trap cleanup EXIT
  if [ "${FOCR_INSTALL_TEST_MODE:-0}" = "1" ] &&
    [[ "${FOCR_INSTALL_TEST_HOLD_LOCK_SECONDS:-}" =~ ^[0-9]+$ ]] &&
    [ "${FOCR_INSTALL_TEST_HOLD_LOCK_SECONDS}" -gt 0 ]; then
    sleep "$FOCR_INSTALL_TEST_HOLD_LOCK_SECONDS"
  fi

  # Serialize the version decision with replacement, then honor an explicit
  # verification request even when no download is necessary.
  if [ "$FORCE_INSTALL" -eq 0 ] && check_installed_version "$VERSION"; then
    ok "focr ${VERSION} is already installed at $DEST/$BINARY_NAME"
    info "Use --force to reinstall."
    maybe_add_path
    verify_install
    if [ "$VERIFY" -eq 1 ]; then
      run_selftest
    fi
    info "Model weights are not bundled; fetch them with: focr pull"
    exit 0
  fi

  TMP=$(mktemp -d)

  download_binary
  verify_download
  install_binary
  maybe_add_path
  verify_install

  if [ "$VERIFY" -eq 1 ]; then
    run_selftest
  fi

  maybe_offer_pull
  print_summary
}

# Run main only when executed directly or via curl | bash (BASH_SOURCE empty).
# The wrapping braces let bash buffer the call, guarding against a truncated
# download in a curl | bash pipeline.
if [[ "${BASH_SOURCE[0]:-}" == "${0:-}" ]] || [[ -z "${BASH_SOURCE[0]:-}" ]]; then
  { main "$@"; }
fi
