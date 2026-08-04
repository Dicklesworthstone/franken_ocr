#!/usr/bin/env sh
#
# smolvlm2_convert_e2e.sh — the model-gated SmolVLM2-500M convert e2e
# (bd-3jo6.3.2, lane C2 of epic bd-3jo6).
#
# Runs the REAL `focr convert --model-id smolvlm2` over the REAL downloaded
# HuggingFaceTB/SmolVLM2-500M-Video-Instruct `model.safetensors` (2.03 GB F32,
# sha256 b9bfd456…) and censuses the produced `.focrq` against the C1 census
# (docs/zoo/smolvlm2-spec.md §11/§12):
#
#   * 489 tensors in, 489 tensors out (UNTIED lm_head — NOTHING omitted);
#   * exactly 224 QInt8PerChan (32 decoder layers × 7 GEMMs: q/k/v/o +
#     gate/up/down, incl. the GQA [320,960] k/v panels);
#   * exactly 265 F32 high-precision (SigLIP tower, connector, embed_tokens,
#     the UNTIED lm_head, all norms) — and NOT ONE BF16 (source is F32);
#   * `lm_head.weight` stored HIGH-PRECISION (doctrine #2 / spec §11: int8
#     lm_head only behind a measured quality kill-switch, never by default);
#   * header self-declares model_id=smolvlm2 + the Apache-2.0 notice, and
#     source_sha256 == sha256(input shard) — provenance closed end-to-end.
#
# A STALE binary (one predating the arch-aware classifier) fails this census
# loudly: it would leave every `model.text_model.*` GEMM F32 and quantize the
# lm_head — 1 int8 tensor instead of 224. That is the tripwire, not a nuisance.
#
# MODEL-GATED (Testing Policy: skip-with-SUCCESS without weights). Input:
#   SMOLVLM2_SAFETENSORS — path to the real model.safetensors. If unset, the
#                          known local mirror is probed:
#                          /Volumes/USBNVME16TB/temp_agent_space/zoo/smolvlm2/model.safetensors
#                          Neither present => SKIP with success + banner.
#
# Options / knobs:
#   --no-build              use the already-built binary (default builds focr)
#   --release               build/use the release binary
#   FOCR_BIN=/path/to/focr  binary override (skips the build)
#   SMOLVLM2_FOCRQ_OUT=…    persist the converted artifact here (default: a
#                           mktemp workdir, deleted on PASS, kept on FAIL)
#
# Logging contract (AGENTS.md "Agent Ergonomics" / docs/testing/LOGGING_AND_E2E.md):
#   * stdout is DATA-ONLY: one NDJSON object per line, schema
#     "smolvlm2_convert_e2e/v1" (events: gate|bin|convert|check|result).
#   * ALL human telemetry is `SVLM `-prefixed on stderr (grep '^SVLM ').
#   * Exit 0 = census PASS or gated SKIP; non-zero = a real divergence.
#
# POSIX sh; passes `sh -n`. python3 required (NDJSON emission + .focrq census).
set -eu

# ── house style: structured, greppable human telemetry, all on stderr ────────
log()   { printf 'SVLM %s\n' "$*" >&2; }
step()  { printf 'SVLM ==== STEP %s ====\n' "$*" >&2; }
info()  { printf 'SVLM   %s\n' "$*" >&2; }
ok()    { printf 'SVLM   PASS  %s\n' "$*" >&2; }
skip()  { printf 'SVLM   SKIP  %s\n' "$*" >&2; }
fail()  { printf 'SVLM   FAIL  %s\n' "$*" >&2; }

# ── NDJSON emitter: data-only stdout, one JSON object per line ───────────────
# usage: ndj key=value [key=value…] — ints stay ints, all else is a string.
ndj() {
  python3 - "$@" <<'PY'
import json, sys, time
rec = {"schema": "smolvlm2_convert_e2e/v1", "ts_ms": int(time.time() * 1000)}
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    try:
        rec[k] = int(v)
    except ValueError:
        rec[k] = v
print(json.dumps(rec, sort_keys=True))
PY
}

now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found — required for NDJSON emission + the .focrq census"
  exit 2
fi

# ── argument parsing ─────────────────────────────────────────────────────────
DO_BUILD=1
PROFILE="debug"
CARGO_BUILD_FLAGS=""
for arg in "$@"; do
  case "$arg" in
    --no-build) DO_BUILD=0 ;;
    --release)  PROFILE="release"; CARGO_BUILD_FLAGS="--release" ;;
    -h|--help)
      sed -n '2,46p' "$0"
      exit 0
      ;;
    *) printf 'smolvlm2_convert_e2e.sh: unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)

log "SmolVLM2-500M convert e2e census (bd-3jo6.3.2)"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — GATE: skip-with-SUCCESS without the real weights.
# ═════════════════════════════════════════════════════════════════════════════
step "1/4 GATE"
DEFAULT_SRC="/Volumes/USBNVME16TB/temp_agent_space/zoo/smolvlm2/model.safetensors"
SRC="${SMOLVLM2_SAFETENSORS:-}"
if [ -z "$SRC" ] && [ -f "$DEFAULT_SRC" ]; then
  SRC="$DEFAULT_SRC"
  info "SMOLVLM2_SAFETENSORS unset — using the local mirror $SRC"
fi
if [ -z "$SRC" ]; then
  skip "SMOLVLM2_SAFETENSORS unset and no local mirror — model-gated, skipped with SUCCESS"
  ndj event=gate result=skip reason=no_weights
  ndj event=result result=skip
  exit 0
fi
if [ ! -f "$SRC" ]; then
  skip "SMOLVLM2_SAFETENSORS=$SRC missing on disk — skipped with SUCCESS"
  ndj event=gate result=skip reason=missing_file "src=$SRC"
  ndj event=result result=skip
  exit 0
fi
SRC_BYTES=$(wc -c < "$SRC" | tr -d ' ')
info "src=$SRC (${SRC_BYTES} bytes)"
ndj event=gate result=pass "src=$SRC" "src_bytes=$SRC_BYTES"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — BIN: resolve/build the focr binary.
# ═════════════════════════════════════════════════════════════════════════════
step "2/4 BIN"
S=$(now_ms)
if [ -n "${FOCR_BIN:-}" ]; then
  BIN="$FOCR_BIN"
  info "FOCR_BIN override: $BIN (skipping build)"
elif [ "$DO_BUILD" -eq 1 ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    fail "cargo not found on PATH; re-run with --no-build or set FOCR_BIN"
    exit 2
  fi
  info "cargo build $CARGO_BUILD_FLAGS --bin focr"
  BUILD_LOG=$(mktemp 2>/dev/null || mktemp -t svlm_build)
  if ! ( cd "$ROOT" && cargo build $CARGO_BUILD_FLAGS --bin focr ) >"$BUILD_LOG" 2>&1; then
    fail "cargo build failed; tail of build log:"
    tail -n 30 "$BUILD_LOG" >&2 || true
    exit 2
  fi
  rm -f "$BUILD_LOG"
  BIN="$ROOT/target/$PROFILE/focr"
else
  BIN="$ROOT/target/$PROFILE/focr"
  info "--no-build: expecting a prebuilt binary at $BIN"
fi
if [ ! -x "$BIN" ]; then
  fail "focr binary not found/executable at $BIN"
  exit 2
fi
info "step 2 took $(( $(now_ms) - S ))ms; bin=$BIN"
ndj event=bin result=pass "bin=$BIN"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — CONVERT: the real `focr convert --model-id smolvlm2` (int8).
# ═════════════════════════════════════════════════════════════════════════════
step "3/4 CONVERT"
WORK=$(mktemp -d 2>/dev/null || mktemp -d -t svlm_convert)
if [ -n "${SMOLVLM2_FOCRQ_OUT:-}" ]; then
  OUT="$SMOLVLM2_FOCRQ_OUT"
else
  OUT="$WORK/smolvlm2.int8.focrq"
fi
info "out=$OUT (workdir $WORK preserved on failure)"
S=$(now_ms)
CONVERT_JSON="$WORK/convert.json"
if ! "$BIN" convert "$SRC" -o "$OUT" --quant int8 --model-id smolvlm2 --json \
     >"$CONVERT_JSON" 2>"$WORK/convert.log"; then
  fail "focr convert exited non-zero (log: $WORK/convert.log)"
  tail -n 15 "$WORK/convert.log" >&2 || true
  ndj event=convert result=fail "log=$WORK/convert.log"
  ndj event=result result=fail
  exit 1
fi
CONVERT_MS=$(( $(now_ms) - S ))
info "convert took ${CONVERT_MS}ms"
# Pass the machine record through to our own NDJSON stream verbatim.
cat "$CONVERT_JSON"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — CENSUS: parse the .focrq + the convert JSON, assert the C1 census.
# Every check is one NDJSON line; any failure lists loudly and exits 1.
# ═════════════════════════════════════════════════════════════════════════════
step "4/4 CENSUS"
if command -v shasum >/dev/null 2>&1; then
  SRC_SHA=$(shasum -a 256 -- "$SRC" | awk '{print $1}')
elif command -v sha256sum >/dev/null 2>&1; then
  SRC_SHA=$(sha256sum -- "$SRC" | awk '{print $1}')
else
  SRC_SHA=""
  info "no sha256 tool — the provenance check will be skipped"
fi
S=$(now_ms)
if SRC_SHA="$SRC_SHA" OUT="$OUT" CONVERT_JSON="$CONVERT_JSON" python3 - <<'PY'
import json, os, struct, sys, time
from collections import Counter

failures = []

def check(name, cond, **fields):
    rec = {
        "schema": "smolvlm2_convert_e2e/v1",
        "ts_ms": int(time.time() * 1000),
        "event": "check",
        "check": name,
        "result": "pass" if cond else "fail",
    }
    rec.update(fields)
    print(json.dumps(rec, sort_keys=True))
    if not cond:
        failures.append(name)

# ── the convert --json record ────────────────────────────────────────────────
cj = json.loads(open(os.environ["CONVERT_JSON"], encoding="utf-8").read())
check("convert_status_ok", cj.get("status") == "ok", got=str(cj.get("status")))
check("convert_model_id", cj.get("model_id") == "smolvlm2", got=str(cj.get("model_id")))
# C1 census: 489 source tensors; 224 int8 GEMMs (32 layers x 7; lm_head NOT among them).
check("convert_tensor_count", cj.get("tensors") == 489, got=cj.get("tensors"), want=489)
check("convert_quantized_count", cj.get("tensors_quantized") == 224,
      got=cj.get("tensors_quantized"), want=224)

# ── the .focrq container itself ──────────────────────────────────────────────
path = os.environ["OUT"]
with open(path, "rb") as f:
    magic = f.read(6)
    version = struct.unpack("<I", f.read(4))[0]
    arch_target = f.read(1)[0]
    sha = f.read(32).hex()
    (hlen,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(hlen))

check("focrq_magic", magic == b"FOCRQ\0", got=repr(magic))
check("focrq_model_id", header.get("model_id") == "smolvlm2", got=str(header.get("model_id")))
check("focrq_license_apache", "Apache-2.0" in header.get("license_notice", ""),
      got=header.get("license_notice", ""))

src_sha = os.environ.get("SRC_SHA", "")
if src_sha:
    check("focrq_source_sha256", sha == src_sha and header.get("source_sha256") == src_sha,
          preamble=sha, header=str(header.get("source_sha256")), input=src_sha)

tensors = header["tensors"]
dtypes = Counter(rec["dtype"] for rec in tensors.values())
# Census (spec §11/§12): 489 total = 224 QInt8PerChan + 265 F32; zero BF16
# (the source shard is F32 and high-precision copies are verbatim).
check("focrq_tensor_count", len(tensors) == 489, got=len(tensors), want=489)
check("focrq_int8_count", dtypes.get("QInt8PerChan", 0) == 224,
      got=dtypes.get("QInt8PerChan", 0), want=224)
check("focrq_f32_count", dtypes.get("F32", 0) == 265, got=dtypes.get("F32", 0), want=265)
check("focrq_no_bf16", dtypes.get("BF16", 0) == 0, got=dtypes.get("BF16", 0))

# The UNTIED lm_head: stored AND high-precision (the C2 headline).
head = tensors.get("lm_head.weight")
check("lm_head_stored", head is not None)
if head is not None:
    check("lm_head_high_precision", head["dtype"] == "F32", got=head["dtype"])
    check("lm_head_shape", head["shape"] == [49280, 960], got=str(head["shape"]))
embed = tensors.get("model.text_model.embed_tokens.weight")
check("embed_tokens_stored_f32", embed is not None and embed["dtype"] == "F32",
      got=str(embed and embed["dtype"]))

# Every decoder GEMM (7/layer x 32 layers) is QInt8PerChan; norms are F32.
gemm_suffixes = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                 "gate_proj.weight", "up_proj.weight", "down_proj.weight")
bad_gemms = [n for n, rec in tensors.items()
             if n.startswith("model.text_model.layers.") and n.endswith(gemm_suffixes)
             and rec["dtype"] != "QInt8PerChan"]
check("decoder_gemms_int8", not bad_gemms, bad=",".join(bad_gemms[:5]))
bad_norms = [n for n, rec in tensors.items()
             if n.startswith("model.text_model.") and "layernorm" in n
             and rec["dtype"] != "F32"]
check("decoder_norms_f32", not bad_norms, bad=",".join(bad_norms[:5]))

# The GQA k/v panels carried their census shape through quantization.
kv = tensors.get("model.text_model.layers.0.self_attn.k_proj.weight")
check("gqa_kv_panel_shape", kv is not None and kv["shape"] == [320, 960]
      and kv["dtype"] == "QInt8PerChan", got=str(kv and (kv["shape"], kv["dtype"])))

# The whole SigLIP tower + the connector stay high-precision F32.
bad_vision = [n for n, rec in tensors.items()
              if n.startswith("model.vision_model.") and rec["dtype"] != "F32"]
check("vision_tower_f32", not bad_vision, bad=",".join(bad_vision[:5]))
conn = tensors.get("model.connector.modality_projection.proj.weight")
check("connector_f32", conn is not None and conn["dtype"] == "F32"
      and conn["shape"] == [960, 12288], got=str(conn and (conn["shape"], conn["dtype"])))

summary = {
    "schema": "smolvlm2_convert_e2e/v1",
    "ts_ms": int(time.time() * 1000),
    "event": "census_summary",
    "tensors": len(tensors),
    "dtypes": dict(sorted(dtypes.items())),
    "focrq_bytes": os.path.getsize(path),
    "format_version": version,
    "arch_target": arch_target,
    "failed_checks": failures,
}
print(json.dumps(summary, sort_keys=True))
sys.exit(1 if failures else 0)
PY
then
  CENSUS_OK=1
else
  CENSUS_OK=0
fi
info "census took $(( $(now_ms) - S ))ms"

# ── banner ────────────────────────────────────────────────────────────────────
if [ "$CENSUS_OK" -eq 1 ]; then
  ok "smolvlm2 convert census matches docs/zoo/smolvlm2-spec.md §11/§12"
  ndj event=result result=pass "out=$OUT" "convert_ms=$CONVERT_MS"
  log "VERDICT PASS"
  if [ -z "${SMOLVLM2_FOCRQ_OUT:-}" ]; then
    rm -rf "$WORK" 2>/dev/null || true
  else
    rm -rf "$WORK" 2>/dev/null || true
    info "artifact persisted at $OUT"
  fi
  exit 0
fi
fail "census diverged from the C1 census — artifact + logs preserved under $WORK"
fail "(a STALE focr binary — pre-arch-aware classifier — shows 1 int8 tensor, not 224)"
ndj event=result result=fail "out=$OUT" "work=$WORK"
log "VERDICT FAIL"
exit 1
