#!/usr/bin/env sh
#
# fetch_model.sh — provision the Baidu Unlimited-OCR model files into
# $FOCR_MODEL_DIR, out-of-band, as a deliberate human-run step.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS FETCHES (from Hugging Face: baidu/Unlimited-OCR)
# ─────────────────────────────────────────────────────────────────────────────
#   model-00001-of-000001.safetensors  (~6.67 GB, single bf16 shard, 2710 tensors)
#   model.safetensors.index.json       (tensor -> shard map / total_size)
#   tokenizer.json                     (~9.98 MB, byte-level BPE, vocab 129280)
#   config.json                        (model + vision + projector config)
#
# franken_ocr NEVER downloads weights at inference time and never leaves the
# machine during inference. This script is the ONLY, explicit, opt-in provisioning
# path. After fetching, run `focr convert` to produce the quantized .focrq the
# engine actually loads.
#
# ─────────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────────
#   FOCR_MODEL_DIR=/path/to/models scripts/fetch_model.sh
#   scripts/fetch_model.sh --dest /path/to/models
#   scripts/fetch_model.sh --check-only        # validate an existing dir, no fetch
#
#   Env:
#     FOCR_MODEL_DIR        destination dir (default: $HOME/.cache/franken_ocr/model)
#     FOCR_MODEL_REVISION   HF commit/revision to pin (default: the verified commit)
#
# Files land directly in the destination dir so `focr` can resolve them by the
# canonical filenames above. Re-running is idempotent: a file already present at
# the expected size is skipped (use --force to re-download).
# ─────────────────────────────────────────────────────────────────────────────

set -eu

# ── banner / usage ───────────────────────────────────────────────────────────
usage() {
  sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# ── config ───────────────────────────────────────────────────────────────────
DEST="${FOCR_MODEL_DIR:-$HOME/.cache/franken_ocr/model}"
REPO="baidu/Unlimited-OCR"
# Verified HF commit (docs/truth-pack/PINNED_SOURCES.md). HF repos are mutable;
# pin so fixtures/perf are meaningful. Override with FOCR_MODEL_REVISION only
# after re-pinning + re-verifying the source hashes.
REVISION="${FOCR_MODEL_REVISION:-3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5}"

FORCE=0
CHECK_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --dest)
      [ $# -ge 2 ] || { echo "fetch_model: --dest needs an argument" >&2; exit 2; }
      DEST="${2%/}"; shift 2 ;;
    --revision)
      [ $# -ge 2 ] || { echo "fetch_model: --revision needs an argument" >&2; exit 2; }
      REVISION="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    *) echo "fetch_model: unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

BASE_URL="https://huggingface.co/${REPO}/resolve/${REVISION}"

# ── canonical filenames + expected byte sizes ────────────────────────────────
# Sizes are the EXACT on-disk FILE sizes at the pinned revision (HF LFS
# x-linked-size). NOTE: the safetensors FILE is larger than the index's
# `total_size` (6672212480, tensor payload only) by the 8-byte header-length
# prefix + the JSON tensor directory (~334 KB for 2710 tensors); the file size
# is 6672547120. Validating against total_size (the previous value) wrongly
# rejected a complete download. We additionally verify the shard's SHA-256
# against the pinned LFS object hash — the authoritative integrity check.
# A downloaded file whose size or hash differs is rejected — a truncated
# transfer (HTML error page, partial download) must never be trusted.
SHARD="model-00001-of-000001.safetensors"
SHARD_BYTES=6672547120          # 6.67 GB bf16 shard, full file (header + payload)
SHARD_SHA256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6
INDEX="model.safetensors.index.json"
TOKENIZER="tokenizer.json"
TOKENIZER_BYTES=9979544         # ~9.98 MB
CONFIG="config.json"

# Files to fetch (the big shard last so cheap files fail fast first).
FILES="${CONFIG} ${INDEX} ${TOKENIZER} ${SHARD}"

# Free-space safety margin: shard + tokenizer + config + index + headroom.
# ~6.67 GB shard + ~2 GB slack for the atomic temp copy / fs overhead.
REQUIRED_BYTES=9000000000       # ~9.0 GB

log() { echo "[fetch-model] $*" >&2; }

# ── portable helpers ─────────────────────────────────────────────────────────
# file size in bytes (BSD/macOS `stat -f%z` vs GNU `stat -c%s`).
file_size() {
  if stat -f%z "$1" >/dev/null 2>&1; then
    stat -f%z "$1"
  else
    stat -c%s "$1"
  fi
}

# free bytes on the filesystem holding $1 (its nearest existing ancestor).
free_bytes() {
  d="$1"
  while [ ! -d "$d" ] && [ "$d" != "/" ] && [ -n "$d" ]; do
    d="$(dirname "$d")"
  done
  # POSIX `df -k` => 1K blocks; column 4 = available. Skip the header line.
  avail_k="$(df -k "$d" 2>/dev/null | awk 'NR==2 {print $4} NR>2 && $4 ~ /^[0-9]+$/ {print $4; exit}')"
  [ -n "$avail_k" ] || { echo 0; return; }
  echo $(( avail_k * 1024 ))
}

have() { command -v "$1" >/dev/null 2>&1; }

# sha256 of $1 (BSD/macOS `shasum -a 256` vs GNU `sha256sum`); empty if neither.
file_sha256() {
  if have shasum; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif have sha256sum; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo ""
  fi
}

# Download $1 (URL) -> $2 (final path), atomically, with retries. Uses curl if
# present, else wget. Never trusts a partial file: downloads to a temp then
# size-checks before the atomic mv.
download() {
  url="$1"; out="$2"; expect_bytes="${3:-0}"; name="$(basename "$out")"
  tmp="${out}.partial.$$"

  log "downloading ${name} ..."
  if have curl; then
    # -f fail on HTTP error, -L follow redirects (HF -> CDN), -C - resume,
    # --retry transient errors. -o to the temp path.
    if ! curl -fL -C - --retry 5 --retry-delay 3 --connect-timeout 30 \
              -o "$tmp" "$url"; then
      rm -f "$tmp"
      log "ERROR: curl failed for ${name} (${url})"
      return 1
    fi
  elif have wget; then
    if ! wget -c --tries=5 --timeout=30 -O "$tmp" "$url"; then
      rm -f "$tmp"
      log "ERROR: wget failed for ${name} (${url})"
      return 1
    fi
  else
    log "ERROR: neither curl nor wget is installed."
    return 1
  fi

  got="$(file_size "$tmp")"
  if [ "$expect_bytes" -gt 0 ] && [ "$got" != "$expect_bytes" ]; then
    log "ERROR: ${name} size mismatch: got ${got} bytes, expected ${expect_bytes}."
    log "       (truncated download or wrong revision — refusing to trust it.)"
    rm -f "$tmp"
    return 1
  fi
  expect_sha="$(expected_sha_for "$name")"
  if [ -n "$expect_sha" ]; then
    got_sha="$(file_sha256 "$tmp")"
    if [ "$got_sha" != "$expect_sha" ]; then
      log "ERROR: ${name} sha256 mismatch: got ${got_sha}, expected ${expect_sha}."
      log "       (corrupt transfer or wrong revision — refusing to trust it.)"
      rm -f "$tmp"
      return 1
    fi
    log "  sha256 ok: ${name}"
  fi
  mv -f "$tmp" "$out"
  log "  ok: ${name} (${got} bytes)"
}

# expected size for a given filename (0 = unknown / no strict check).
expected_size_for() {
  case "$1" in
    "$SHARD")     echo "$SHARD_BYTES" ;;
    "$TOKENIZER") echo "$TOKENIZER_BYTES" ;;
    *)            echo 0 ;;
  esac
}

# expected sha256 for a given filename (empty = no hash check).
expected_sha_for() {
  case "$1" in
    "$SHARD") echo "$SHARD_SHA256" ;;
    *)        echo "" ;;
  esac
}

# true if $1 already exists at its expected size.
already_present() {
  f="$1"; out="${DEST}/${f}"
  [ -f "$out" ] || return 1
  exp="$(expected_size_for "$f")"
  [ "$exp" -eq 0 ] && return 0  # exists, no strict size known
  [ "$(file_size "$out")" = "$exp" ]
}

# ── --check-only: validate an existing dir and report, then exit ─────────────
if [ "$CHECK_ONLY" -eq 1 ]; then
  log "checking ${DEST} ..."
  rc=0
  for f in $FILES; do
    out="${DEST}/${f}"
    if [ ! -f "$out" ]; then
      log "  MISSING: ${f}"
      rc=1
      continue
    fi
    exp="$(expected_size_for "$f")"
    got="$(file_size "$out")"
    if [ "$exp" -gt 0 ] && [ "$got" != "$exp" ]; then
      log "  BAD SIZE: ${f} (${got} != ${exp})"
      rc=1
    else
      log "  ok: ${f} (${got} bytes)"
    fi
  done
  [ "$rc" -eq 0 ] && log "all model files present and correctly sized." \
                  || log "model dir is incomplete — run without --check-only to fetch."
  exit "$rc"
fi

# ── pre-flight: destination + free space ─────────────────────────────────────
log "destination: ${DEST}"
log "source repo: ${REPO}"
log "revision:    ${REVISION}"

mkdir -p "$DEST"

avail="$(free_bytes "$DEST")"
if [ "$avail" -gt 0 ] && [ "$avail" -lt "$REQUIRED_BYTES" ]; then
  avail_gb=$(( avail / 1000000000 ))
  req_gb=$(( REQUIRED_BYTES / 1000000000 ))
  log "ERROR: not enough free space on the filesystem holding ${DEST}."
  log "       need ~${req_gb} GB, have ~${avail_gb} GB free."
  log "       free space or pick a different --dest, then re-run."
  exit 1
fi
log "free space ok (~$(( avail / 1000000000 )) GB available)."

# ── fetch each file (skip already-present-and-correctly-sized unless --force) ─
for f in $FILES; do
  out="${DEST}/${f}"
  if [ "$FORCE" -eq 0 ] && already_present "$f"; then
    log "skip (already present, correct size): ${f}"
    continue
  fi
  download "${BASE_URL}/${f}" "$out" "$(expected_size_for "$f")"
done

# ── post-flight: re-validate sizes ───────────────────────────────────────────
rc=0
for f in $FILES; do
  if ! already_present "$f"; then
    log "ERROR: ${f} is missing or wrong size after fetch."
    rc=1
  fi
done
if [ "$rc" -ne 0 ]; then
  log "fetch incomplete — see errors above."
  exit 1
fi

log ""
log "all files fetched into ${DEST}:"
for f in $FILES; do
  log "  ${f} ($(file_size "${DEST}/${f}") bytes)"
done
log ""
log "NEXT: verify the shard sha256 against your trusted record, e.g."
log "  shasum -a 256 \"${DEST}/${SHARD}\""
log "(record the hash in docs/truth-pack/SOURCE_HASHES.md the first time)."
log ""
log "Then generate reference fixtures (CUDA host required, OQ-17):"
log "  FOCR_MODEL_DIR=\"${DEST}\" python3 scripts/gen_reference_fixtures.py \\"
log "      --corpus tests/fixtures/corpus --out tests/fixtures/native --activations"
log ""
log "Then convert to the quantized form the engine loads:"
log "  focr convert \"${DEST}/${SHARD}\" -o \"${DEST}/unlimited-ocr.focrq\" --quant int8"

exit 0
