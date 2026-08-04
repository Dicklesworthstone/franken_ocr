#!/usr/bin/env bash
# gauntlet_focr.sh — the focr side of the head-to-head gauntlet (bd-re8.17).
#
# Runs the release `focr` binary N times WARM with FOCR_TIMING=1, captures each
# run's stderr/stdout + a high-resolution wall clock + peak RSS, then folds the
# `[focr-timing]` lines into distribution JSON via scripts/gauntlet_timing.py.
# Single-arm output (`focr_stages.json`) feeds scripts/gauntlet_row.py. Opt-in
# A/B output (`focr_ab.json`) uses a balanced interleaved schedule and refuses
# cross-arm output drift.
#
# Honesty contract (docs/PERF_LEDGER.md §fairness):
#   - every number is a real wall-clock sample of a real run; a missing model /
#     page / binary SKIPS with exit 0 (never fabricates, never reds CI);
#   - the thread budget is pinned into the environment and recorded;
#   - warmup runs are discarded; all measured samples are kept in the record;
#   - all raw stderr/stdout/meta land beside the JSON for the evidence bundle.
#
# Usage:
#   scripts/gauntlet_focr.sh --page PAGE [--model MODEL] [--binary BIN]
#                            [--runs N] [--warmup W] [--threads T]
#                            [--allocator LABEL] [--precision P] [--out-dir DIR]
#                            [--model-sha256 HEX] [--model-size BYTES]
#                            [--quant-recipe ID] [--build-receipt RECEIPT]
#                            [--command ocr|ocr-batch] [--workload-label LABEL]
#                            [--ab-env FOCR_VAR] [--a-label LABEL] [--a-value V]
#                            [--b-label LABEL] [--b-value V]
#   scripts/gauntlet_focr.sh --self-test   # stub-binary + synthetic-model dry run
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMING_PY="$REPO_ROOT/scripts/gauntlet_timing.py"

PAGES=()
MODEL=""
BINARY=""
RUNS=5
WARMUP=1
THREADS="${FOCR_THREADS:-8}"
ALLOCATOR="system"
PRECISION=""
OUT_DIR=""
EXPECTED_MODEL_SHA256=""
EXPECTED_MODEL_SIZE=""
QUANT_RECIPE=""
BUILD_RECEIPT=""
COMMAND="ocr"
WORKLOAD_LABEL=""
AB_ENV=""
A_LABEL="control"
A_VALUE="<unset>"
B_LABEL="candidate"
B_VALUE="1"
SELF_TEST=0
SYNTHETIC=0
EXTRA_ARGS=()   # zoo lanes: e.g. --extra-arg --task --extra-arg chart-data

usage() { sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --page) PAGES+=("$2"); shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --allocator) ALLOCATOR="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --model-sha256) EXPECTED_MODEL_SHA256="$2"; shift 2 ;;
    --model-size) EXPECTED_MODEL_SIZE="$2"; shift 2 ;;
    --quant-recipe) QUANT_RECIPE="$2"; shift 2 ;;
    --build-receipt) BUILD_RECEIPT="$2"; shift 2 ;;
    --command) COMMAND="$2"; shift 2 ;;
    --workload-label) WORKLOAD_LABEL="$2"; shift 2 ;;
    --ab-env) AB_ENV="$2"; shift 2 ;;
    --a-label) A_LABEL="$2"; shift 2 ;;
    --a-value) A_VALUE="$2"; shift 2 ;;
    --b-label) B_LABEL="$2"; shift 2 ;;
    --b-value) B_VALUE="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    --synthetic) SYNTHETIC=1; shift ;;   # stamp records synthetic (stub runs)
    --extra-arg) EXTRA_ARGS+=("$2"); shift 2 ;;  # appended AFTER --model (A11 zoo tasks)
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

skip() { # reason [detail] — graceful fixture-absent skip: exit 0, one JSON line
  printf '{"event":"skip","harness":"gauntlet_focr","reason":"%s","detail":"%s"}\n' "$1" "${2:-}"
  exit 0
}

batch_spine_enabled() {
  local value
  value="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  case "$value" in
    ''|0|off|false|no) return 1 ;;
    *) return 0 ;;
  esac
}

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$WARMUP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --runs must be a positive integer, --warmup a non-negative integer" >&2
  exit 2
fi
if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || (( THREADS > 32 )); then
  # NEVER 64 — oversubscription fakes wins (docs/PERF_LEDGER.md §fairness).
  echo "ERROR: --threads must be a positive integer <= 32 (got '$THREADS')" >&2
  exit 2
fi
if [[ -n "$EXPECTED_MODEL_SHA256" && ! "$EXPECTED_MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: --model-sha256 must be 64 lowercase hexadecimal characters" >&2
  exit 2
fi
if [[ -n "$EXPECTED_MODEL_SIZE" && ! "$EXPECTED_MODEL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --model-size must be a positive integer" >&2
  exit 2
fi
if [[ "$COMMAND" != "ocr" && "$COMMAND" != "ocr-batch" ]]; then
  echo "ERROR: --command must be ocr or ocr-batch" >&2
  exit 2
fi
if [[ "$COMMAND" == "ocr" && ${#PAGES[@]} -gt 1 ]]; then
  echo "ERROR: --command ocr accepts exactly one --page" >&2
  exit 2
fi
if [[ "$COMMAND" == "ocr-batch" ]]; then
  for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--multi-page" ]]; then
      echo "ERROR: --multi-page has no sequential gauntlet evidence contract" >&2
      exit 2
    fi
  done
fi
if batch_spine_enabled "${FOCR_BATCH_SPINE:-}"; then
  echo "FATAL: gauntlet capture forbids FOCR_BATCH_SPINE until it emits per-page timing evidence" >&2
  exit 2
fi
if [[ -n "$AB_ENV" ]]; then
  if [[ ! "$AB_ENV" =~ ^FOCR_[A-Z0-9_]+$ || "$AB_ENV" == FOCR_GAUNTLET_* ]]; then
    echo "ERROR: --ab-env must name a non-harness FOCR_* variable" >&2
    exit 2
  fi
  case "$AB_ENV" in
    FOCR_TIMING|FOCR_PROFILE_DECODE|FOCR_THREADS|FOCR_DECODE_INT8|FOCR_INT8_ATTN|FOCR_INT8_LMHEAD|FOCR_ATTN_GEMM|FOCR_INT8_KV|FOCR_SPEC_DECODE|FOCR_DECODE_STATELESS|FOCR_BATCH_SPINE)
      echo "ERROR: --ab-env cannot vary timing, thread, precision, or uninstrumented batch-spine gates" >&2
      exit 2
      ;;
  esac
  [[ -n "$A_LABEL" && -n "$B_LABEL" && "$A_LABEL" != "$B_LABEL" ]] \
    || { echo "ERROR: A/B labels must be non-empty and distinct" >&2; exit 2; }
  if (( RUNS < 5 )); then
    echo "ERROR: strict A/B capture requires --runs >= 5 per arm (got '$RUNS')" >&2
    exit 2
  fi
  [[ "$A_VALUE" != "$B_VALUE" ]] \
    || { echo "ERROR: A/B values must be distinct" >&2; exit 2; }
  case "$WORKLOAD_LABEL" in
    sparse|dense)
      [[ "$COMMAND" == "ocr" && ${#PAGES[@]} -eq 1 ]] \
        || { echo "ERROR: $WORKLOAD_LABEL requires --command ocr and one --page" >&2; exit 2; }
      ;;
    10-page)
      [[ "$COMMAND" == "ocr-batch" && ${#PAGES[@]} -eq 10 ]] \
        || { echo "ERROR: 10-page requires --command ocr-batch and ten --page arguments" >&2; exit 2; }
      ;;
    20-page)
      [[ "$COMMAND" == "ocr-batch" && ${#PAGES[@]} -eq 20 ]] \
        || { echo "ERROR: 20-page requires --command ocr-batch and twenty --page arguments" >&2; exit 2; }
      ;;
    *)
      echo "ERROR: A/B --workload-label must be sparse, dense, 10-page, or 20-page" >&2
      exit 2
      ;;
  esac
fi

# ── --self-test: full dry run against a stub binary that emits synthetic
#    FOCR_TIMING lines; asserts the pipeline produces the expected stages. ────
if (( SELF_TEST )); then
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/focr-gauntlet-selftest.XXXXXX")"
  for spine_value in enabled 'f alse'; do
    SPINE_OUTPUT=""
    SPINE_RC=0
    if SPINE_OUTPUT="$(env FOCR_BATCH_SPINE="$spine_value" "$0" --page "$TMP/missing.png" 2>&1)"; then
      echo "SELF-TEST FAILED: armed FOCR_BATCH_SPINE=$spine_value was accepted" >&2
      exit 1
    else
      SPINE_RC=$?
    fi
    if [[ "$SPINE_RC" -ne 2 || "$SPINE_OUTPUT" != *"gauntlet capture forbids FOCR_BATCH_SPINE"* ]]; then
      echo "SELF-TEST FAILED: armed FOCR_BATCH_SPINE=$spine_value did not fail fast" >&2
      exit 1
    fi
  done
  STUB="$TMP/focr-stub"
  cat >"$STUB" <<'STUB_EOF'
#!/usr/bin/env bash
# Stub focr: proves the harness plumbing without weights. Refuses to pretend
# to be real: it only runs when FOCR_TIMING is set by the harness.
[[ -n "${FOCR_TIMING:-}" ]] || { echo "stub requires FOCR_TIMING" >&2; exit 9; }
if [[ "${FOCR_SELFTEST_MUTATE_MODEL:-0}" == "1" ]]; then
  printf 'changed-during-capture' >>"$4"
fi
if [[ "${FOCR_SELFTEST_MUTATE_BINARY:-0}" == "1" ]]; then
  printf '# changed-during-capture\n' >>"$0"
fi
cat >&2 <<'ERR'
[focr-timing] precision focr-full-int8
[focr-timing] preprocess 0.10s
[focr-timing]   vision.sam 0.80s
[focr-timing]   vision.clip 0.30s
[focr-timing]   vision.bridge 0.04s
[focr-timing] vision_tower 1.14s
[focr-timing] weight_cache_build_i8 0.90s
[focr-timing] prefill_i8 1.80s (289 tokens)
[focr-timing] decode_i8 6.00s (600 tokens, 0.010s/tok)
[focr-timing] decode_i8 phases (ms): lm_head 900  attn 1800  experts 3200  route 100
ERR
if [[ "${FOCR_SELFTEST_OUTPUT_DRIFT:-0}" == "1" ]]; then
  echo "stub candidate text"
else
  echo "stub page text"
fi
STUB_EOF
  chmod +x "$STUB"
  printf 'not-really-a-png' >"$TMP/page.png"
  printf 'synthetic-focrq' >"$TMP/model.focrq"
  STUB_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$STUB")"
  STUB_SIZE="$(wc -c <"$STUB" | tr -d ' ')"
  python3 - "$TMP/build_receipt.json" "$STUB" "$STUB_SHA" "$STUB_SIZE" <<'PY_EOF'
import json, sys
path, binary, digest, size = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "schema": "focr-build-receipt/v1",
            "profile": "release-perf",
            "binary": {
                "path": binary,
                "sha256": digest,
                "size": int(size),
            },
        },
        handle,
        indent=2,
    )
    handle.write("\n")
PY_EOF
  MODEL_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$TMP/model.focrq")"
  MODEL_SIZE="$(wc -c <"$TMP/model.focrq" | tr -d ' ')"
  env -u FOCR_ATTN_GEMM -u FOCR_INT8_KV -u FOCR_SPEC_DECODE \
      -u FOCR_DECODE_STATELESS \
      FOCR_DECODE_INT8=1 FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 \
      "$0" --binary "$STUB" --page "$TMP/page.png" --model "$TMP/model.focrq" \
      --model-sha256 "$MODEL_SHA" --model-size "$MODEL_SIZE" \
      --quant-recipe unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1 \
      --build-receipt "$TMP/build_receipt.json" \
      --precision focr-full-int8 --runs 3 --warmup 1 \
      --threads "$THREADS" --out-dir "$TMP/out" --synthetic
  if env -u FOCR_ATTN_GEMM -u FOCR_INT8_KV -u FOCR_SPEC_DECODE \
         -u FOCR_DECODE_STATELESS \
         FOCR_DECODE_INT8=1 FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 \
         "$0" --binary "$STUB" --page "$TMP/page.png" --model "$TMP/model.focrq" \
         --model-sha256 "$MODEL_SHA" --model-size "$MODEL_SIZE" \
         --quant-recipe unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1 \
         --build-receipt "$TMP/build_receipt.json" \
         --precision focr-mixed-ffn-int8 --runs 1 --warmup 0 \
         --threads "$THREADS" --out-dir "$TMP/mismatched" --synthetic \
         >/dev/null 2>&1; then
    echo "SELF-TEST FAILED: contradictory mixed/full gates were accepted" >&2
    exit 1
  fi
  if env FOCR_ATTN_GEMM=0 FOCR_DECODE_INT8=1 FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 \
         "$0" --binary "$STUB" --page "$TMP/page.png" --model "$TMP/model.focrq" \
         --model-sha256 "$MODEL_SHA" --model-size "$MODEL_SIZE" \
         --quant-recipe unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1 \
         --build-receipt "$TMP/build_receipt.json" \
         --precision focr-full-int8 --runs 1 --warmup 0 \
         --threads "$THREADS" --out-dir "$TMP/presence-gate" --synthetic \
         >/dev/null 2>&1; then
    echo "SELF-TEST FAILED: presence-only risky gate was accepted" >&2
    exit 1
  fi
  if env -u FOCR_ATTN_GEMM -u FOCR_INT8_KV -u FOCR_SPEC_DECODE \
         -u FOCR_DECODE_STATELESS \
         FOCR_DECODE_INT8=1 FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 \
         "$0" --binary "$STUB" --page "$TMP/page.png" --model "$TMP/model.focrq" \
         --model-sha256 "$MODEL_SHA" --model-size "$MODEL_SIZE" \
         --quant-recipe unlimited-ocr-full-int8-attn-int8-lmhead-int8-v1 \
         --build-receipt "$TMP/build_receipt.json" \
         --precision focr-full-int8 --runs 1 --warmup 0 \
         --threads "$THREADS" --out-dir "$TMP/wrong-recipe" --synthetic \
         >/dev/null 2>&1; then
    echo "SELF-TEST FAILED: contradictory artifact quant recipe was accepted" >&2
    exit 1
  fi
  cp "$TMP/model.focrq" "$TMP/mutating-model.focrq"
  MUTATING_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$TMP/mutating-model.focrq")"
  MUTATING_SIZE="$(wc -c <"$TMP/mutating-model.focrq" | tr -d ' ')"
  if env -u FOCR_ATTN_GEMM -u FOCR_INT8_KV -u FOCR_SPEC_DECODE \
         -u FOCR_DECODE_STATELESS \
         FOCR_SELFTEST_MUTATE_MODEL=1 \
         FOCR_DECODE_INT8=1 FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 \
         "$0" --binary "$STUB" --page "$TMP/page.png" --model "$TMP/mutating-model.focrq" \
         --model-sha256 "$MUTATING_SHA" --model-size "$MUTATING_SIZE" \
         --quant-recipe unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1 \
         --build-receipt "$TMP/build_receipt.json" \
         --precision focr-full-int8 --runs 1 --warmup 0 \
         --threads "$THREADS" --out-dir "$TMP/model-toctou" --synthetic \
         >/dev/null 2>&1; then
    echo "SELF-TEST FAILED: model mutation during capture was accepted" >&2
    exit 1
  fi
  cp "$STUB" "$TMP/mutating-binary"
  MUTATING_BINARY_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$TMP/mutating-binary")"
  MUTATING_BINARY_SIZE="$(wc -c <"$TMP/mutating-binary" | tr -d ' ')"
  python3 - "$TMP/mutating_build_receipt.json" "$TMP/mutating-binary" "$MUTATING_BINARY_SHA" "$MUTATING_BINARY_SIZE" <<'PY_EOF'
import json, sys
path, binary, digest, size = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "schema": "focr-build-receipt/v1",
            "profile": "release-perf",
            "binary": {
                "path": binary,
                "sha256": digest,
                "size": int(size),
            },
        },
        handle,
    )
    handle.write("\n")
PY_EOF
  chmod +x "$TMP/mutating-binary"
  if env -u FOCR_ATTN_GEMM -u FOCR_INT8_KV -u FOCR_SPEC_DECODE \
         -u FOCR_DECODE_STATELESS \
         FOCR_SELFTEST_MUTATE_BINARY=1 \
         FOCR_DECODE_INT8=1 FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 \
         "$0" --binary "$TMP/mutating-binary" --page "$TMP/page.png" --model "$TMP/model.focrq" \
         --model-sha256 "$MODEL_SHA" --model-size "$MODEL_SIZE" \
         --quant-recipe unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1 \
         --build-receipt "$TMP/mutating_build_receipt.json" \
         --precision focr-full-int8 --runs 1 --warmup 0 \
         --threads "$THREADS" --out-dir "$TMP/binary-toctou" --synthetic \
         >/dev/null 2>&1; then
    echo "SELF-TEST FAILED: evidence-local binary mutation during capture was accepted" >&2
    exit 1
  fi
  python3 - "$TMP/out/focr_stages.json" <<'PY_EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
stages = {r["stage"]: r for r in doc["stages"]}
assert doc["synthetic"] is True, "self-test output must be stamped synthetic"
assert doc["stdout_identical_across_runs"] is True
for want in ("preprocess", "vision_encode", "prefill", "decode_per_token", "end_to_end"):
    assert want in stages, f"missing stage {want}"
    assert stages[want]["n"] == 3, f"{want}: expected 3 samples"
for want, expected_ms in (
    ("decode_lm_head", 900.0),
    ("decode_attn", 1800.0),
    ("decode_experts", 3200.0),
    ("decode_route", 100.0),
):
    assert stages[want]["n"] == 3, f"{want}: expected 3 samples"
    assert stages[want]["p50_ms"] == expected_ms
    assert stages[want]["cv_pct"] == 0.0
assert stages["decode_per_token"]["tokens"] == 600
assert abs(stages["decode_per_token"]["best_ms"] - 10.0) < 1e-6
assert doc["precision"] == "focr-full-int8"
assert doc["model_kind"] == "file"
assert doc["model_sha256"] == __import__("hashlib").sha256(b"synthetic-focrq").hexdigest()
assert doc["model_size"] == len(b"synthetic-focrq")
assert doc["quant_recipe"] == "unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1"
assert doc["decode_mode"] == "full-int8"
assert doc["precision_gate_states"] == {
    "FOCR_DECODE_INT8": "1",
    "FOCR_INT8_ATTN": "1",
    "FOCR_INT8_LMHEAD": "1",
    "FOCR_ATTN_GEMM": "<unset>",
    "FOCR_INT8_KV": "<unset>",
    "FOCR_SPEC_DECODE": "<unset>",
    "FOCR_DECODE_STATELESS": "<unset>",
}
meta = json.load(open(__import__("os").path.join(doc["run_dir"], "run_001.meta.json")))
assert meta["binary"] == __import__("os").path.join(
    __import__("os").path.dirname(doc["run_dir"]), "subject", "release-perf", "focr"
)
assert meta["build_receipt"].endswith("/subject/build_receipt.json")
assert meta["build_receipt_sha256"]
print(json.dumps({"check": "gauntlet-focr-self-test", "result": "pass"}))
PY_EOF
  env -u FOCR_ATTN_GEMM -u FOCR_INT8_KV -u FOCR_SPEC_DECODE \
      -u FOCR_DECODE_STATELESS -u FOCR_RSWA_PARALLEL_ATTN \
      FOCR_DECODE_INT8=1 FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 \
      "$0" --binary "$STUB" --page "$TMP/page.png" --model "$TMP/model.focrq" \
      --model-sha256 "$MODEL_SHA" --model-size "$MODEL_SIZE" \
      --quant-recipe unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1 \
      --build-receipt "$TMP/build_receipt.json" \
      --precision focr-full-int8 --runs 5 --warmup 1 \
      --threads "$THREADS" --out-dir "$TMP/ab" --synthetic \
      --workload-label sparse --ab-env FOCR_RSWA_PARALLEL_ATTN \
      --a-label control --a-value '<unset>' --b-label parallel --b-value 1
  python3 - "$TMP/ab/focr_ab.json" <<'PY_EOF'
import json, sys

doc = json.load(open(sys.argv[1], encoding="utf-8"))
assert doc["schema"] == "focr-gauntlet-ab/v1"
assert doc["workload"] == {"label": "sparse", "command": "ocr", "page_count": 1}
assert [row["arm"] for row in doc["schedule"]] == [
    "a", "b", "b", "a", "b", "a", "a", "b", "a", "b"
]
assert doc["cross_arm_output"]["byte_identical"] is True
assert doc["arms"]["a"]["runs"] == doc["arms"]["b"]["runs"] == 5
assert doc["arms"]["a"]["performance_switch_states"]["FOCR_RSWA_PARALLEL_ATTN"] == "<unset>"
assert doc["arms"]["b"]["performance_switch_states"]["FOCR_RSWA_PARALLEL_ATTN"] == "1"
for arm in ("a", "b"):
    rss = doc["arms"][arm]["resources"]["maximum_resident_set_size"]
    assert rss["n"] == 5 and rss["p50_bytes"] > 0 and rss["p95_bytes"] > 0
    assert rss["p99_bytes"] >= rss["p95_bytes"]
    stages = {row["stage"]: row for row in doc["arms"][arm]["stages"]}
    assert stages["end_to_end"]["p50_ms"] > 0
    assert stages["end_to_end"]["p95_ms"] >= stages["end_to_end"]["p50_ms"]
    assert stages["end_to_end"]["p99_ms"] >= stages["end_to_end"]["p95_ms"]
    for phase in ("decode_lm_head", "decode_attn", "decode_experts", "decode_route"):
        assert stages[phase]["n"] == 5
        assert stages[phase]["cv_pct"] == 0.0
comparisons = {row["stage"]: row for row in doc["comparisons"]}
for phase in ("decode_lm_head", "decode_attn", "decode_experts", "decode_route"):
    assert comparisons[phase]["a_p50_ms"] == comparisons[phase]["b_p50_ms"]
print(json.dumps({"check": "gauntlet-focr-ab-self-test", "result": "pass"}))
PY_EOF
  if env -u FOCR_ATTN_GEMM -u FOCR_INT8_KV -u FOCR_SPEC_DECODE \
         -u FOCR_DECODE_STATELESS -u FOCR_SELFTEST_OUTPUT_DRIFT \
         FOCR_DECODE_INT8=1 FOCR_INT8_ATTN=1 FOCR_INT8_LMHEAD=1 \
         "$0" --binary "$STUB" --page "$TMP/page.png" --model "$TMP/model.focrq" \
         --model-sha256 "$MODEL_SHA" --model-size "$MODEL_SIZE" \
         --quant-recipe unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1 \
         --build-receipt "$TMP/build_receipt.json" \
         --precision focr-full-int8 --runs 5 --warmup 0 \
         --threads "$THREADS" --out-dir "$TMP/ab-output-drift" --synthetic \
         --workload-label sparse --ab-env FOCR_SELFTEST_OUTPUT_DRIFT \
         --a-label control --a-value '<unset>' --b-label drift --b-value 1 \
         >/dev/null 2>&1; then
    echo "SELF-TEST FAILED: cross-arm stdout drift was accepted" >&2
    exit 1
  fi
  exit 0
fi

# ── resolve inputs; absence of any fixture is a graceful skip ────────────────
CANONICAL_CAPTURE=0
if [[ "$PRECISION" == "focr-mixed-ffn-int8" || "$PRECISION" == "focr-full-int8" ]]; then
  CANONICAL_CAPTURE=1
fi
(( ${#PAGES[@]} > 0 )) || { echo "ERROR: --page is required (or --self-test)" >&2; exit 2; }
if [[ -z "$BINARY" ]]; then
  for candidate in "$REPO_ROOT/target/release-perf/focr" "$REPO_ROOT/target/release/focr"; do
    [[ -x "$candidate" ]] && BINARY="$candidate" && break
  done
  if [[ -z "$BINARY" ]]; then
    (( CANONICAL_CAPTURE == 0 )) \
      || { echo "FATAL: canonical capture has no release-perf binary" >&2; exit 2; }
    skip "no_release_binary" "build focr with the release-perf profile first"
  fi
fi
if [[ ! -f "$BINARY" || -L "$BINARY" || ! -x "$BINARY" ]]; then
  (( CANONICAL_CAPTURE == 0 )) \
    || { echo "FATAL: canonical binary must be a regular, non-symlink executable: $BINARY" >&2; exit 2; }
  skip "binary_not_executable" "$BINARY must be a regular, non-symlink executable"
fi
for page in "${PAGES[@]}"; do
  if [[ ! -f "$page" || -L "$page" ]]; then
    (( CANONICAL_CAPTURE == 0 )) \
      || { echo "FATAL: canonical page must be a regular, non-symlink file: $page" >&2; exit 2; }
    skip "page_fixture_absent" "$page must be a regular, non-symlink file"
  fi
done
if [[ -n "$MODEL" && ! -e "$MODEL" ]]; then
  (( CANONICAL_CAPTURE == 0 )) \
    || { echo "FATAL: canonical model fixture is absent: $MODEL" >&2; exit 2; }
  skip "model_fixture_absent" "$MODEL"
fi
if (( CANONICAL_CAPTURE )) && [[ -z "$BUILD_RECEIPT" || ! -f "$BUILD_RECEIPT" || -L "$BUILD_RECEIPT" ]]; then
  echo "FATAL: canonical Unlimited precision requires a regular, non-symlink --build-receipt" >&2
  exit 2
fi

stable_file_identity() {
  python3 - "$1" "$2" <<'PY_EOF'
import hashlib, json, os, stat, sys

path, maximum = sys.argv[1], int(sys.argv[2])
lexical_before = os.lstat(path)
if not stat.S_ISREG(lexical_before.st_mode):
    raise SystemExit(f"identity input is not a regular file: {path}")
if lexical_before.st_size > maximum:
    raise SystemExit(f"identity input exceeds the {maximum}-byte bound: {path}")
flags = os.O_RDONLY
flags |= getattr(os, "O_CLOEXEC", 0)
flags |= getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags)
try:
    before = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (lexical_before.st_dev, lexical_before.st_ino):
        raise SystemExit(f"identity input was replaced before open: {path}")
    digest = hashlib.sha256()
    observed = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        observed += len(chunk)
        if observed > maximum:
            raise SystemExit(f"identity input grew beyond the {maximum}-byte bound: {path}")
        digest.update(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
lexical_after = os.lstat(path)
stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
if (
    any(getattr(before, key) != getattr(after, key) for key in stable)
    or any(getattr(before, key) != getattr(lexical_after, key) for key in stable)
    or observed != before.st_size
):
    raise SystemExit(f"identity input changed while hashing: {path}")
identity = {
    "dev": before.st_dev,
    "ino": before.st_ino,
    "size": before.st_size,
    "mtime_ns": before.st_mtime_ns,
    "ctime_ns": before.st_ctime_ns,
}
print(f"{digest.hexdigest()}\t{observed}\t{json.dumps(identity, separators=(',', ':'))}")
PY_EOF
}

capture_file_identity() {
  local value digest size identity
  value="$(stable_file_identity "$1" "$2")" || return 1
  IFS=$'\t' read -r digest size identity <<<"$value"
  [[ "$digest" =~ ^[0-9a-f]{64}$ && "$size" =~ ^[0-9]+$ ]] || return 1
  printf -v "$3" '%s' "$digest"
  printf -v "$4" '%s' "$size"
  if (( $# >= 5 )); then
    [[ "$identity" == \{*\} ]] || return 1
    printf -v "$5" '%s' "$identity"
  fi
}

reject_symlink_components() {
  python3 - "$@" <<'PY_EOF'
import os, stat, sys
from pathlib import Path

for raw in sys.argv[1:]:
    requested = Path(os.path.abspath(raw))
    component = Path(requested.anchor)
    for part in requested.parts[1:]:
        component /= part
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise SystemExit(f"evidence path contains a symlink component: {component}")
PY_EOF
}

# Bind canonical Unlimited-OCR evidence to the exact subject artifact. Hash a
# regular model file once before the timed loop; directories remain usable for
# non-canonical development lanes, but the timing aggregator refuses to stamp a
# canonical Unlimited precision without a regular `.focrq` hash + size.
MODEL_KIND="none"
MODEL_SHA256=""
MODEL_SIZE=""
MODEL_IDENTITY="null"
if [[ -n "$MODEL" && -f "$MODEL" ]]; then
  MODEL_KIND="file"
  capture_file_identity "$MODEL" 17179869184 MODEL_SHA256 MODEL_SIZE MODEL_IDENTITY \
    || { echo "FATAL: cannot establish a stable model identity: $MODEL" >&2; exit 2; }
elif [[ -n "$MODEL" && -d "$MODEL" ]]; then
  MODEL_KIND="directory"
fi
if [[ -n "$EXPECTED_MODEL_SHA256" && "$MODEL_SHA256" != "$EXPECTED_MODEL_SHA256" ]]; then
  echo "FATAL: model sha256 $MODEL_SHA256 != expected $EXPECTED_MODEL_SHA256 ($MODEL)" >&2
  exit 2
fi
if [[ -n "$EXPECTED_MODEL_SIZE" && "$MODEL_SIZE" != "$EXPECTED_MODEL_SIZE" ]]; then
  echo "FATAL: model size $MODEL_SIZE != expected $EXPECTED_MODEL_SIZE ($MODEL)" >&2
  exit 2
fi
export FOCR_GAUNTLET_MODEL_KIND="$MODEL_KIND"
export FOCR_GAUNTLET_MODEL_SHA256="$MODEL_SHA256"
export FOCR_GAUNTLET_MODEL_SIZE="$MODEL_SIZE"
export FOCR_GAUNTLET_MODEL_IDENTITY="$MODEL_IDENTITY"
export FOCR_GAUNTLET_QUANT_RECIPE="$QUANT_RECIPE"
export FOCR_GAUNTLET_MODEL="$MODEL"

# Freeze every input page's identity before capture. The Python run wrapper
# re-hashes the same paths for each child, and the post-capture check below
# closes the final-run TOCTOU window.
PAGES_JSON="$(python3 - "${PAGES[@]}" <<'PY_EOF'
import hashlib, json, os, stat, sys

pages = []
seen_files = set()
for raw in sys.argv[1:]:
    path = os.path.abspath(raw)
    lexical = os.lstat(path)
    if not stat.S_ISREG(lexical.st_mode):
        raise SystemExit(f"page is not a regular file: {path}")
    identity = (lexical.st_dev, lexical.st_ino)
    if identity in seen_files:
        raise SystemExit(f"page list repeats the same file: {path}")
    seen_files.add(identity)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > 1024 * 1024 * 1024:
                raise SystemExit(f"page exceeds the 1 GiB identity bound: {path}")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    lexical_after = os.lstat(path)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        (before.st_dev, before.st_ino) != (lexical.st_dev, lexical.st_ino)
        or any(getattr(before, key) != getattr(after, key) for key in stable)
        or any(getattr(before, key) != getattr(lexical_after, key) for key in stable)
        or observed != before.st_size
    ):
        raise SystemExit(f"page changed while hashing: {path}")
    pages.append({"path": path, "sha256": digest.hexdigest(), "size": observed})
print(json.dumps(pages, separators=(",", ":")))
PY_EOF
)" || { echo "FATAL: cannot establish stable page identities" >&2; exit 2; }
CANONICAL_PAGES=()
while IFS= read -r -d '' page; do
  CANONICAL_PAGES+=("$page")
done < <(python3 - "$PAGES_JSON" <<'PY_EOF'
import json, sys

for page in json.loads(sys.argv[1]):
    sys.stdout.buffer.write(page["path"].encode("utf-8") + b"\0")
PY_EOF
)
if (( ${#CANONICAL_PAGES[@]} != ${#PAGES[@]} )); then
  echo "FATAL: canonical page argv does not match the identity manifest" >&2
  exit 2
fi
export FOCR_GAUNTLET_PAGES_JSON="$PAGES_JSON"
export FOCR_GAUNTLET_COMMAND="$COMMAND"
export FOCR_GAUNTLET_WORKLOAD="$WORKLOAD_LABEL"

truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')" in
    1|true|on|yes) return 0 ;;
    *) return 1 ;;
  esac
}

# A caller may omit --precision for a runtime-derived Unlimited label, in which
# case the Python aggregator performs this same validation after the run. When a
# canonical identity is requested up front, reject contradictions before paying
# for any model forward.
if [[ "$PRECISION" == "focr-mixed-ffn-int8" || "$PRECISION" == "focr-full-int8" ]]; then
  [[ "$MODEL_KIND" == "file" && "$MODEL" == *.focrq ]] \
    || { echo "FATAL: canonical Unlimited precision requires a regular .focrq --model" >&2; exit 2; }
  [[ ! -L "$MODEL" ]] \
    || { echo "FATAL: canonical Unlimited precision forbids a symlink --model" >&2; exit 2; }
  [[ -n "$EXPECTED_MODEL_SHA256" && -n "$EXPECTED_MODEL_SIZE" ]] \
    || { echo "FATAL: canonical Unlimited precision requires --model-sha256 and --model-size" >&2; exit 2; }
  [[ "$QUANT_RECIPE" == "unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1" ]] \
    || { echo "FATAL: canonical Unlimited precision requires the exact conservative --quant-recipe" >&2; exit 2; }
  [[ -n "$BUILD_RECEIPT" && -f "$BUILD_RECEIPT" && ! -L "$BUILD_RECEIPT" ]] \
    || { echo "FATAL: canonical Unlimited precision requires --build-receipt" >&2; exit 2; }
  if [[ -n "${FOCR_ATTN_GEMM+x}" || -n "${FOCR_INT8_KV+x}" \
        || -n "${FOCR_SPEC_DECODE+x}" || -n "${FOCR_DECODE_STATELESS+x}" ]]; then
    echo "FATAL: canonical precision forbids FOCR_ATTN_GEMM, FOCR_INT8_KV," >&2
    echo "       FOCR_SPEC_DECODE, and FOCR_DECODE_STATELESS even when set falsy" >&2
    exit 2
  fi
  master=0; attn=0; lmhead=0
  truthy "${FOCR_DECODE_INT8:-}" && master=1
  truthy "${FOCR_INT8_ATTN:-}" && attn=1
  truthy "${FOCR_INT8_LMHEAD:-}" && lmhead=1
  if [[ "$PRECISION" == "focr-mixed-ffn-int8" ]]; then
    [[ "$master$attn$lmhead" == "000" ]] \
      || { echo "FATAL: focr-mixed-ffn-int8 requires all three int8 recipe gates falsy" >&2; exit 2; }
  else
    [[ "$master$attn$lmhead" == "111" ]] \
      || { echo "FATAL: focr-full-int8 requires all three int8 recipe gates truthy" >&2; exit 2; }
  fi
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/perf/bd-re8.17/focr}"
OUT_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_DIR")"
RAW_DIR="$OUT_DIR/raw"
reject_symlink_components "$OUT_DIR" "$RAW_DIR"
if [[ -e "$OUT_DIR" && ! -d "$OUT_DIR" ]]; then
  echo "FATAL: evidence out dir exists but is not a directory: $OUT_DIR" >&2
  exit 2
fi
if [[ -d "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "FATAL: evidence out dir must be empty; use a fresh --out-dir: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$RAW_DIR"
# Refuse a dirty raw dir (fresh-eyes fix): the aggregator globs EVERY
# run_*.meta.json under it, so re-running into the same out dir would fold a
# previous session's samples into this one's best-of-N — silently mixed
# measurements. Fail closed; the operator picks a fresh dir (nothing deleted).
if compgen -G "$RAW_DIR/*run_*.meta.json" > /dev/null 2>&1; then
  echo "FATAL: $RAW_DIR already holds measured metadata from a previous session;" >&2
  echo "       re-aggregating would mix runs. Use a fresh out dir." >&2
  exit 2
fi

# Execute an evidence-local regular-file copy, not an arbitrary target-tree
# pathname. This gives every raw run one stable subject identity and prevents a
# later target rebuild from silently changing what the row appears to measure.
SUBJECT_DIR="$OUT_DIR/subject"
EVIDENCE_BINARY="$SUBJECT_DIR/release-perf/focr"
EVIDENCE_RECEIPT="$SUBJECT_DIR/build_receipt.json"
reject_symlink_components "$SUBJECT_DIR" "$EVIDENCE_BINARY" "$EVIDENCE_RECEIPT"
[[ ! -e "$EVIDENCE_BINARY" && ! -L "$EVIDENCE_BINARY" \
   && ! -e "$EVIDENCE_RECEIPT" && ! -L "$EVIDENCE_RECEIPT" ]] \
  || { echo "FATAL: $SUBJECT_DIR already contains subject evidence; use a fresh out dir" >&2; exit 2; }
mkdir -p "$(dirname "$EVIDENCE_BINARY")"
BINARY_SOURCE="$BINARY"
capture_file_identity "$BINARY_SOURCE" 1073741824 BINARY_SOURCE_SHA256 BINARY_SOURCE_SIZE \
  || { echo "FATAL: cannot establish a stable source-binary identity" >&2; exit 2; }
cp -p -- "$BINARY_SOURCE" "$EVIDENCE_BINARY"
capture_file_identity "$BINARY_SOURCE" 1073741824 BINARY_SOURCE_POST_SHA256 BINARY_SOURCE_POST_SIZE \
  || { echo "FATAL: source binary became unreadable during copy" >&2; exit 2; }
capture_file_identity "$EVIDENCE_BINARY" 1073741824 EVIDENCE_BINARY_SHA256 EVIDENCE_BINARY_SIZE \
  || { echo "FATAL: cannot establish an evidence-binary identity" >&2; exit 2; }
if [[ "$BINARY_SOURCE_POST_SHA256" != "$BINARY_SOURCE_SHA256" \
      || "$BINARY_SOURCE_POST_SIZE" != "$BINARY_SOURCE_SIZE" \
      || "$EVIDENCE_BINARY_SHA256" != "$BINARY_SOURCE_SHA256" \
      || "$EVIDENCE_BINARY_SIZE" != "$BINARY_SOURCE_SIZE" ]]; then
  echo "FATAL: source binary changed while making the evidence-local copy" >&2
  exit 2
fi
[[ -x "$EVIDENCE_BINARY" ]] \
  || { echo "FATAL: copied subject binary is not executable: $EVIDENCE_BINARY" >&2; exit 2; }

BUILD_RECEIPT_SHA256=""
if [[ -n "$BUILD_RECEIPT" ]]; then
  capture_file_identity "$BUILD_RECEIPT" 1048576 BUILD_RECEIPT_SHA256 BUILD_RECEIPT_SIZE \
    || { echo "FATAL: cannot establish a stable build-receipt identity" >&2; exit 2; }
  python3 - "$BUILD_RECEIPT" "$BINARY_SOURCE" "$BINARY_SOURCE_SHA256" "$BINARY_SOURCE_SIZE" <<'PY_EOF'
import json, os, stat, sys

receipt_path, binary_path, expected_sha256, expected_size = sys.argv[1:]
metadata = os.lstat(receipt_path)
if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
    raise SystemExit("build receipt must be a regular file no larger than 1 MiB")
with open(receipt_path, encoding="utf-8") as handle:
    receipt = json.load(handle)
binary = receipt.get("binary") if isinstance(receipt, dict) else None
if (
    receipt.get("schema") != "focr-build-receipt/v1"
    or receipt.get("profile") != "release-perf"
    or not isinstance(binary, dict)
    or os.path.realpath(str(binary.get("path", ""))) != os.path.realpath(binary_path)
    or binary.get("sha256") != expected_sha256
    or binary.get("size") != int(expected_size)
):
    raise SystemExit("build receipt does not bind the requested release-perf binary")
PY_EOF
  cp -p -- "$BUILD_RECEIPT" "$EVIDENCE_RECEIPT"
  capture_file_identity "$BUILD_RECEIPT" 1048576 BUILD_RECEIPT_POST_SHA256 BUILD_RECEIPT_POST_SIZE \
    || { echo "FATAL: source build receipt became unreadable during copy" >&2; exit 2; }
  capture_file_identity "$EVIDENCE_RECEIPT" 1048576 EVIDENCE_RECEIPT_SHA256 EVIDENCE_RECEIPT_SIZE \
    || { echo "FATAL: cannot establish an evidence-receipt identity" >&2; exit 2; }
  if [[ "$BUILD_RECEIPT_POST_SHA256" != "$BUILD_RECEIPT_SHA256" \
        || "$BUILD_RECEIPT_POST_SIZE" != "$BUILD_RECEIPT_SIZE" \
        || "$EVIDENCE_RECEIPT_SHA256" != "$BUILD_RECEIPT_SHA256" \
        || "$EVIDENCE_RECEIPT_SIZE" != "$BUILD_RECEIPT_SIZE" ]]; then
    echo "FATAL: build receipt changed while making the evidence-local copy" >&2
    exit 2
  fi
fi

BINARY="$EVIDENCE_BINARY"
export FOCR_GAUNTLET_BINARY_ORIGIN="$BINARY_SOURCE"
export FOCR_GAUNTLET_BINARY_SIZE="$EVIDENCE_BINARY_SIZE"
export FOCR_GAUNTLET_BUILD_RECEIPT="${BUILD_RECEIPT:+$EVIDENCE_RECEIPT}"
export FOCR_GAUNTLET_BUILD_RECEIPT_SHA256="$BUILD_RECEIPT_SHA256"

CMD=("$BINARY" "$COMMAND" "${CANONICAL_PAGES[@]}")
[[ -n "$MODEL" ]] && CMD+=("--model" "$MODEL")
# Extra command arguments are recorded verbatim in every run receipt.
(( ${#EXTRA_ARGS[@]} )) && CMD+=("${EXTRA_ARGS[@]}")

# Thread parity pins: focr's own budget plus the pool knobs a helper could
# read. Recorded verbatim into every run's meta (the ledger `command/env`).
export FOCR_TIMING=1
export FOCR_PROFILE_DECODE=1
export FOCR_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export RAYON_NUM_THREADS="$THREADS"

capture_run() {
  local tag="$1"
  local arm="${2:-}"
  local arm_label="${3:-}"
  local arm_value="${4:-}"
  local schedule_index="${5:-0}"
  local arm_run="${6:-0}"
  local env_switch=()
  if [[ -n "$AB_ENV" ]]; then
    if [[ "$arm_value" == "<unset>" ]]; then
      env_switch=(-u "$AB_ENV")
    else
      env_switch=("$AB_ENV=$arm_value")
    fi
  fi
  # The python wrapper provides the high-resolution wall clock (BSD `date` has
  # no %N), invokes the platform `time` binary for peak RSS, and writes the run
  # metadata consumed by the strict aggregator.
  env "${env_switch[@]}" \
    FOCR_GAUNTLET_TAG="$tag" \
    FOCR_GAUNTLET_RAW_DIR="$RAW_DIR" \
    FOCR_GAUNTLET_WARMUP="$WARMUP" \
    FOCR_GAUNTLET_AB_ENV="$AB_ENV" \
    FOCR_GAUNTLET_AB_ARM="$arm" \
    FOCR_GAUNTLET_AB_LABEL="$arm_label" \
    FOCR_GAUNTLET_AB_VALUE="$arm_value" \
    FOCR_GAUNTLET_AB_SCHEDULE_INDEX="$schedule_index" \
    FOCR_GAUNTLET_AB_ARM_RUN="$arm_run" \
    python3 - "${CMD[@]}" <<'PY_EOF'
import hashlib, json, os, stat, subprocess, sys, time

cmd = sys.argv[1:]
tag = os.environ["FOCR_GAUNTLET_TAG"]
raw = os.environ["FOCR_GAUNTLET_RAW_DIR"]
stdout_path = os.path.join(raw, f"{tag}.stdout")
stderr_path = os.path.join(raw, f"{tag}.stderr")

if not os.path.isfile("/usr/bin/time"):
    raise SystemExit("/usr/bin/time is required for gauntlet RSS evidence")
time_args = ["-lp"] if sys.platform == "darwin" else ["-v"]
timed_cmd = ["/usr/bin/time", *time_args, *cmd]

def file_identity(path):
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"timed model is no longer a regular file: {path}")
    return {
        "dev": metadata.st_dev,
        "ino": metadata.st_ino,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }

model_kind = os.environ["FOCR_GAUNTLET_MODEL_KIND"]
expected_model_identity = json.loads(os.environ["FOCR_GAUNTLET_MODEL_IDENTITY"])
model_path = os.environ.get("FOCR_GAUNTLET_MODEL", "")
model_identity = None
if model_kind == "file":
    model_identity = file_identity(model_path)
    if model_identity != expected_model_identity:
        raise SystemExit(f"model identity changed before timed child: {model_path}")
elif expected_model_identity is not None:
    raise SystemExit("non-file model carried a file identity receipt")

t0 = time.perf_counter()
with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
    rc = subprocess.run(timed_cmd, stdout=out, stderr=err).returncode
wall_ms = (time.perf_counter() - t0) * 1000.0
if model_kind == "file" and file_identity(model_path) != model_identity:
    raise SystemExit(f"model identity changed during timed child: {model_path}")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

pins = (
    "FOCR_TIMING",
    "FOCR_PROFILE_DECODE",
    "FOCR_THREADS",
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
precision_gates = (
    "FOCR_DECODE_INT8",
    "FOCR_INT8_ATTN",
    "FOCR_INT8_LMHEAD",
    "FOCR_ATTN_GEMM",
    "FOCR_INT8_KV",
    "FOCR_SPEC_DECODE",
    "FOCR_DECODE_STATELESS",
)
performance_switches = list(dict.fromkeys((
    "FOCR_RSWA_PARALLEL_ATTN",
    "FOCR_MMAP",
    "FOCR_SOFTMAX_COPY",
    "FOCR_UNLIMITED_VISION_CACHE",
    "FOCR_INT8_AUTOVEC",
    "FOCR_BATCHED_MOE",
    "FOCR_BATCH_PACK",
    "FOCR_BATCH_SIZE",
    "FOCR_BATCH_SPINE",
    "FOCR_BATCH_VISION",
    "FOCR_MAX_NEW_TOKENS",
    os.environ.get("FOCR_GAUNTLET_AB_ENV", ""),
)))
performance_switches = [name for name in performance_switches if name]
expected_pages = json.loads(os.environ["FOCR_GAUNTLET_PAGES_JSON"])
pages = []
for page in expected_pages:
    path = page["path"]
    pages.append(
        {
            "path": path,
            "sha256": sha256(path),
            "size": os.stat(path, follow_symlinks=False).st_size,
        }
    )
ab_env = os.environ.get("FOCR_GAUNTLET_AB_ENV", "")
ab = None
if ab_env:
    ab = {
        "env": ab_env,
        "arm": os.environ["FOCR_GAUNTLET_AB_ARM"],
        "label": os.environ["FOCR_GAUNTLET_AB_LABEL"],
        "value": os.environ["FOCR_GAUNTLET_AB_VALUE"],
        "schedule_index": int(os.environ["FOCR_GAUNTLET_AB_SCHEDULE_INDEX"]),
        "arm_run": int(os.environ["FOCR_GAUNTLET_AB_ARM_RUN"]),
    }
meta = {
    "tag": tag,
    "command": cmd,
    "exit_code": rc,
    "wall_ms": round(wall_ms, 3),
    "stdout": os.path.basename(stdout_path),
    "stderr": os.path.basename(stderr_path),
    "binary": cmd[0],
    "binary_sha256": sha256(cmd[0]),
    "binary_size": int(os.environ["FOCR_GAUNTLET_BINARY_SIZE"]),
    "binary_origin": os.environ["FOCR_GAUNTLET_BINARY_ORIGIN"],
    "build_receipt": os.environ["FOCR_GAUNTLET_BUILD_RECEIPT"] or None,
    "build_receipt_sha256": os.environ["FOCR_GAUNTLET_BUILD_RECEIPT_SHA256"] or None,
    "page": pages[0]["path"] if len(pages) == 1 else None,
    "page_sha256": pages[0]["sha256"] if len(pages) == 1 else None,
    "pages": pages,
    "workload": {
        "label": os.environ.get("FOCR_GAUNTLET_WORKLOAD") or None,
        "command": os.environ["FOCR_GAUNTLET_COMMAND"],
        "page_count": len(pages),
    },
    "model": os.environ.get("FOCR_GAUNTLET_MODEL") or None,
    "model_kind": os.environ["FOCR_GAUNTLET_MODEL_KIND"],
    "model_sha256": os.environ["FOCR_GAUNTLET_MODEL_SHA256"] or None,
    "model_size": int(os.environ["FOCR_GAUNTLET_MODEL_SIZE"])
                  if os.environ["FOCR_GAUNTLET_MODEL_SIZE"] else None,
    "model_identity": model_identity,
    "quant_recipe": os.environ["FOCR_GAUNTLET_QUANT_RECIPE"] or None,
    "threads": int(os.environ["FOCR_THREADS"]),
    "warmup": int(os.environ["FOCR_GAUNTLET_WARMUP"]),
    "env_pins": {k: os.environ[k] for k in pins},
    # The full FOCR_* surface = the row's fallback/kill-switch state evidence.
    "focr_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("FOCR_")
                 and not k.startswith("FOCR_GAUNTLET_")},
    # Presence-only gates must distinguish unset from explicitly falsy. The
    # ordinary focr_env map cannot express an absent variable, so carry the
    # complete precision surface separately.
    "precision_gate_states": {k: os.environ.get(k, "<unset>") for k in precision_gates},
    "performance_switch_states": {
        k: os.environ.get(k, "<unset>") for k in performance_switches
    },
    "ab": ab,
}
with open(os.path.join(raw, f"{tag}.meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
if rc != 0:
    sys.stderr.write(open(stderr_path, errors="replace").read())
    sys.exit(rc)
arm_text = f" [{ab['arm']}:{ab['label']}]" if ab else ""
print(f"  {tag}{arm_text}: {wall_ms:.0f} ms", file=sys.stderr)
PY_EOF
}

if [[ -z "$AB_ENV" ]]; then
  TOTAL=$(( WARMUP + RUNS ))
  echo "gauntlet_focr: $TOTAL runs ($WARMUP warmup + $RUNS measured) of: ${CMD[*]}" >&2
  for i in $(seq 1 "$TOTAL"); do
    if (( i <= WARMUP )); then
      TAG="warmup_$i"
    else
      TAG="run_$(printf '%03d' $(( i - WARMUP )))"
    fi
    capture_run "$TAG"
  done
else
  echo "gauntlet_focr: balanced A/B, $WARMUP warmup + $RUNS measured per arm" >&2
  echo "  workload=$WORKLOAD_LABEL switch=$AB_ENV A=$A_LABEL:$A_VALUE B=$B_LABEL:$B_VALUE" >&2
  for i in $(seq 1 "$WARMUP"); do
    if (( i % 2 == 1 )); then
      capture_run "a_warmup_$i" a "$A_LABEL" "$A_VALUE" 0 0
      capture_run "b_warmup_$i" b "$B_LABEL" "$B_VALUE" 0 0
    else
      capture_run "b_warmup_$i" b "$B_LABEL" "$B_VALUE" 0 0
      capture_run "a_warmup_$i" a "$A_LABEL" "$A_VALUE" 0 0
    fi
  done

  A_RUN=0
  B_RUN=0
  SCHEDULE_INDEX=0
  BLOCK=0
  while (( A_RUN < RUNS || B_RUN < RUNS )); do
    if (( BLOCK % 2 == 0 )); then
      PATTERN=(a b b a)
    else
      PATTERN=(b a a b)
    fi
    for ARM in "${PATTERN[@]}"; do
      if [[ "$ARM" == a ]]; then
        (( A_RUN >= RUNS )) && continue
        A_RUN=$(( A_RUN + 1 ))
        TAG="a_run_$(printf '%03d' "$A_RUN")"
        LABEL="$A_LABEL"
        VALUE="$A_VALUE"
        ARM_RUN="$A_RUN"
      else
        (( B_RUN >= RUNS )) && continue
        B_RUN=$(( B_RUN + 1 ))
        TAG="b_run_$(printf '%03d' "$B_RUN")"
        LABEL="$B_LABEL"
        VALUE="$B_VALUE"
        ARM_RUN="$B_RUN"
      fi
      SCHEDULE_INDEX=$(( SCHEDULE_INDEX + 1 ))
      capture_run "$TAG" "$ARM" "$LABEL" "$VALUE" "$SCHEDULE_INDEX" "$ARM_RUN"
    done
    BLOCK=$(( BLOCK + 1 ))
  done
fi

# The executable actually invoked is immutable evidence for the whole capture.
# Hash after every run has exited and reject even a same-size replacement.
capture_file_identity "$EVIDENCE_BINARY" 1073741824 POST_BINARY_SHA256 POST_BINARY_SIZE \
  || { echo "FATAL: evidence-local binary became unreadable during capture" >&2; exit 2; }
if [[ "$POST_BINARY_SHA256" != "$EVIDENCE_BINARY_SHA256" \
      || "$POST_BINARY_SIZE" != "$EVIDENCE_BINARY_SIZE" ]]; then
  echo "FATAL: evidence-local binary changed during capture: $EVIDENCE_BINARY" >&2
  exit 2
fi
if [[ -n "$BUILD_RECEIPT_SHA256" ]]; then
  capture_file_identity "$EVIDENCE_RECEIPT" 1048576 POST_RECEIPT_SHA256 POST_RECEIPT_SIZE \
    || { echo "FATAL: evidence-local receipt became unreadable during capture" >&2; exit 2; }
  [[ "$POST_RECEIPT_SHA256" == "$BUILD_RECEIPT_SHA256" \
     && "$POST_RECEIPT_SIZE" == "$BUILD_RECEIPT_SIZE" ]] \
    || { echo "FATAL: evidence-local build receipt changed during capture" >&2; exit 2; }
fi

# Bind the timed samples to one immutable subject image. A model replacement or
# in-place rewrite during the loop invalidates the whole capture before the
# aggregator can turn it into evidence.
if [[ "$MODEL_KIND" == "file" ]]; then
  [[ -f "$MODEL" ]] || { echo "FATAL: model artifact disappeared during capture: $MODEL" >&2; exit 2; }
  capture_file_identity "$MODEL" 17179869184 POST_MODEL_SHA256 POST_MODEL_SIZE \
    || { echo "FATAL: model became unreadable during capture" >&2; exit 2; }
  if [[ "$POST_MODEL_SHA256" != "$MODEL_SHA256" || "$POST_MODEL_SIZE" != "$MODEL_SIZE" ]]; then
    echo "FATAL: model artifact changed during capture: $MODEL" >&2
    echo "       before sha256=$MODEL_SHA256 size=$MODEL_SIZE" >&2
    echo "       after  sha256=$POST_MODEL_SHA256 size=$POST_MODEL_SIZE" >&2
    exit 2
  fi
fi

python3 - "$PAGES_JSON" <<'PY_EOF' \
  || { echo "FATAL: an input page changed during capture" >&2; exit 2; }
import hashlib, json, os, stat, sys

expected = json.loads(sys.argv[1])
for page in expected:
    path = page["path"]
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != page["size"]:
        raise SystemExit(f"page identity changed: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != page["sha256"]:
        raise SystemExit(f"page identity changed: {path}")
PY_EOF

if [[ -z "$AB_ENV" ]]; then
  AGG_ARGS=(aggregate --run-dir "$RAW_DIR" --out "$OUT_DIR/focr_stages.json"
            --threads "$THREADS" --allocator "$ALLOCATOR")
else
  AGG_ARGS=(aggregate-ab --run-dir "$RAW_DIR" --out "$OUT_DIR/focr_ab.json"
            --threads "$THREADS" --allocator "$ALLOCATOR"
            --ab-env "$AB_ENV" --a-label "$A_LABEL" --a-value "$A_VALUE"
            --b-label "$B_LABEL" --b-value "$B_VALUE")
fi
[[ -n "$PRECISION" ]] && AGG_ARGS+=(--precision "$PRECISION")
(( SYNTHETIC )) && AGG_ARGS+=(--synthetic)
python3 "$TIMING_PY" "${AGG_ARGS[@]}"
