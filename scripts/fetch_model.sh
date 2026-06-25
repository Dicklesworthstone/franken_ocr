#!/usr/bin/env sh
#
# fetch_model.sh — provision the Baidu Unlimited-OCR model files into
# $FOCR_MODEL_DIR, out-of-band, as a deliberate human-run step.
#
# ─────────────────────────────────────────────────────────────────────────────
# STATUS: STUB. This is an honest scaffold, not a finished downloader.
# The exact sha256 pins and any auth/redirect handling are TODO
# and MUST be filled in before this is wired into CI or docs as "just run it".
# ─────────────────────────────────────────────────────────────────────────────
#
# Mark executable when used:  chmod +x scripts/fetch_model.sh
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS FETCHES (from Hugging Face: baidu/Unlimited-OCR)
# ─────────────────────────────────────────────────────────────────────────────
#   model-00001-of-000001.safetensors  (~6.67 GB, single bf16 shard, 2710 tensors)
#   model.safetensors.index.json       (tensor → shard map / total_size)
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
#
#   Env:
#     FOCR_MODEL_DIR   destination dir (default: $HOME/.cache/franken_ocr/model)
#
# Files land directly in the destination dir so `focr` can resolve them by the
# canonical filenames above.
# ─────────────────────────────────────────────────────────────────────────────

set -eu

usage() {
  # Print the banner above (lines 2..NN), stripping the leading "# ".
  sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

DEST="${FOCR_MODEL_DIR:-$HOME/.cache/franken_ocr/model}"

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --dest)
      [ $# -ge 2 ] || { echo "fetch_model: --dest needs an argument" >&2; exit 2; }
      DEST="${2%/}"; shift 2 ;;
    *) echo "fetch_model: unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

REPO="baidu/Unlimited-OCR"
# Current verified HF commit as of 2026-06-25. Phase -1 must refresh this before
# kernel work and record the exact source hashes in docs/truth-pack/.
REVISION="${FOCR_MODEL_REVISION:-3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5}"
BASE_URL="https://huggingface.co/${REPO}/resolve/${REVISION}"

# The files to fetch. sha256 pins are TODO — fill these in from a trusted local
# copy via `shasum -a 256 <file>` and verify after download before trusting.
FILES="
model-00001-of-000001.safetensors
model.safetensors.index.json
tokenizer.json
config.json
"

echo "[fetch-model] STUB — not yet implemented." >&2
echo "[fetch-model] destination: ${DEST}" >&2
echo "[fetch-model] source repo:  ${REPO}" >&2
echo "[fetch-model] revision:     ${REVISION}" >&2
echo "[fetch-model]" >&2
echo "[fetch-model] TODO before this works:" >&2
echo "[fetch-model]   1. mkdir -p \"\${DEST}\"" >&2
echo "[fetch-model]   2. for each file below, download \"\${BASE_URL}/<file>\"" >&2
echo "[fetch-model]      (curl -fSL --retry 3, or hf CLI / git-lfs for the big shard)" >&2
echo "[fetch-model]   3. verify each against a pinned sha256 (pins are TODO)" >&2
echo "[fetch-model]   4. move into place atomically; be idempotent on re-run" >&2
echo "[fetch-model]" >&2
echo "[fetch-model] files to fetch from ${BASE_URL}/ :" >&2
for f in $FILES; do
  echo "[fetch-model]   - ${f}" >&2
done
echo "[fetch-model]" >&2
echo "[fetch-model] manual fallback (works today, no script needed):" >&2
echo "[fetch-model]   pip install -U huggingface_hub" >&2
echo "[fetch-model]   hf download ${REPO} --revision \"${REVISION}\" --local-dir \"${DEST}\"" >&2
echo "[fetch-model]" >&2
echo "[fetch-model] Then: focr convert \"${DEST}/model-00001-of-000001.safetensors\" \\" >&2
echo "[fetch-model]            -o \"${DEST}/unlimited-ocr.focrq\" --quant int8" >&2

# Exit non-zero so callers do not mistake the stub for a successful fetch.
exit 1
