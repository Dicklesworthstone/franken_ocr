#!/usr/bin/env python3
"""Merge gauntlet measurements into a PERF_LEDGER row + evidence bundle (bd-re8.17).

Inputs (all REAL measurement/derivation files; anything missing → refusal):
  * `--focr-stages`   — `scripts/gauntlet_focr.sh` output (focr-gauntlet-stages/v1)
  * `--ref-stages`    — `scripts/gauntlet_reference.py` output (same schema)
  * `--roofline`      — `scripts/gauntlet_roofline.py` output

Output:
  * `artifacts/perf/bd-re8.17/<claim_id>/` evidence bundle: the three input
    JSONs, the raw focr run logs, `row.json`, `PERF_LEDGER_ROW.md`, and a
    `SHA256SUMS` manifest (`scripts/check_ledgers.py` requires one), with a v3
    row binding the source commit/root, evidence-producer root, and sole path
    that a later evidence-only descendant commit may change;
  * the exact markdown row(s) for `docs/PERF_LEDGER.md`, validated by running
    the REAL `scripts/check_ledgers.py` against a shadow copy of the repo's
    ledgers with the row inserted (`--check`, the default). `--apply`
    additionally inserts the row into the real `docs/PERF_LEDGER.md` and
    re-validates in place (reverting on failure).

Fairness/honesty gates (all fail-closed, per docs/PERF_LEDGER.md):
  * refuses synthetic (self-test) inputs;
  * refuses `cv_pct > 5` on either side (noise cannot land a claim);
  * refuses a focr/reference thread-budget or allocator mismatch;
  * refuses a reference record without its thread proof or precision;
  * refuses nondeterministic focr evidence (stdout or token drift across runs);
  * `claim_id`, `fixture_hash`, `arch/cpu_features`, `correctness_proof` are
    caller-supplied and required — there are no defaults for provenance.

Usage:
  gauntlet_row.py --focr-stages F.json --ref-stages R.json --roofline RL.json \
      --claim-id G2-decode-YYYYMMDD --fixture-hash SHA256 \
      --arch-features aarch64+neon+dotprod --correctness-proof 'PROOF' \
      [--stage decode_per_token ...] [--evidence-dir DIR] [--apply] [--notes S]
  gauntlet_row.py --self-test
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "PERF_LEDGER.md")
CHECKER = os.path.join(ROOT, "scripts", "check_ledgers.py")
PERF_ROOT = Path(ROOT) / "artifacts" / "perf"

# docs/truth-pack/PINNED_SOURCES.md — the ledger legend fixes this value for
# every franken_ocr row, so defaulting it is provenance, not fabrication.
MODEL_COMMIT = "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"
REFERENCE_CONTRACT_SCHEMA = "focr-reference-contract/v1"
REFERENCE_CONTRACTS = {
    "gauntlet_ref_unlimited:run_stage": (
        "unlimited-ocr-hf-v1",
        "2.10.0",
        "4.57.1",
    ),
    "gauntlet_ref_zoo:run_smolvlm2": (
        "smolvlm2-500m-hf-v1",
        "2.10.0",
        "4.57.1",
    ),
    "gauntlet_ref_zoo:run_got": ("got-ocr2-hf-v1", "2.12.1", "4.45.2"),
    "gauntlet_ref_zoo:run_onechart": ("onechart-hf-v1", "2.12.1", "4.45.2"),
    "gauntlet_ref_zoo:run_tromr": ("tromr-hf-v1", "2.12.1", "4.45.2"),
}

# The current Unlimited-OCR release claim is deliberately narrower than the
# generic gauntlet schemas. Historical rows and zoo lanes remain parseable, but
# only this exact conservative artifact/recipe may produce a new Unlimited
# release row. Change these constants only as part of an explicit release-model
# rollover with fresh parity and performance evidence.
CURRENT_UNLIMITED_REFERENCE_ENTRY = "gauntlet_ref_unlimited:run_stage"
CURRENT_UNLIMITED_PRECISION = "focr-mixed-ffn-int8"
CURRENT_UNLIMITED_DECODE_MODE = "mixed-ffn-int8"
CURRENT_UNLIMITED_QUANT_RECIPE = (
    "unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1"
)
CURRENT_UNLIMITED_MODEL_SHA256 = (
    "573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592"
)
CURRENT_UNLIMITED_MODEL_SIZE = 4_157_448_783
PRECISION_GATE_VARS = (
    "FOCR_DECODE_INT8",
    "FOCR_INT8_ATTN",
    "FOCR_INT8_LMHEAD",
    "FOCR_ATTN_GEMM",
    "FOCR_INT8_KV",
    "FOCR_SPEC_DECODE",
    "FOCR_DECODE_STATELESS",
)
OQ14_FALSE_OR_UNSET_VARS = (
    "FOCR_DECODE_INT8",
    "FOCR_INT8_ATTN",
    "FOCR_INT8_LMHEAD",
)
PRESENCE_REJECTED_VARS = (
    "FOCR_ATTN_GEMM",
    "FOCR_INT8_KV",
    "FOCR_SPEC_DECODE",
    "FOCR_DECODE_STATELESS",
)
RAW_TIMING_SCHEMA = "focr-gauntlet-raw-timing/v1"
MAX_RAW_TIMING_RUNS = 256
MAX_RAW_TIMING_STAGES = 64
MAX_RAW_TIMING_FILES = 1024
MAX_RAW_TIMING_FILE_BYTES = 64 * 1024 * 1024
MAX_RAW_TIMING_TOTAL_BYTES = 512 * 1024 * 1024
BUILD_RECEIPT_SCHEMA = "focr-build-receipt/v1"
SOURCE_MANIFEST_SCHEMA = "focr-source-input-manifest/v1"
SOURCE_PACK_SCHEMA = "focr-source-input-pack/v1"
SOURCE_ROOT_DOMAIN = b"focr-source-input-root/v1\0"
SOURCE_PACK_DOMAIN = b"focr-source-input-pack/v1\0"
SOURCE_PACK_RECORD_DOMAIN = b"focr-source-input-pack-record/v1\0"
SOURCE_PACK_TRAILER_DOMAIN = b"focr-source-input-pack-trailer/v1\0"
PRODUCER_ROOT_DOMAIN = b"focr-gauntlet-producer-root/v1\0"
SOURCE_ROOT_ALGORITHM = (
    "sha256(domain='focr-source-input-root/v1\\0'; "
    "sorted(repository,path); repository\\0path\\0size\\0sha256\\n)"
)
PRODUCER_ROOT_ALGORITHM = (
    "sha256(domain='focr-gauntlet-producer-root/v1\\0'; "
    "sorted(path); path\\0size\\0sha256\\n)"
)
PRODUCER_PATHS = (
    "docs/PERF_LEDGER.md",
    "docs/truth-pack/PINNED_SOURCES.md",
    "docs/truth-pack/SOURCE_HASHES.md",
    "scripts/baseline/compare_ocr.py",
    "scripts/baseline/run_baidu_reference.py",
    "scripts/check_ledgers.py",
    "scripts/gauntlet_cert.py",
    "scripts/gauntlet_focr.sh",
    "scripts/gauntlet_ref_unlimited.py",
    "scripts/gauntlet_reference.py",
    "scripts/gauntlet_roofline.py",
    "scripts/gauntlet_row.py",
    "scripts/gauntlet_runbook.sh",
    "scripts/gauntlet_timing.py",
)
REFERENCE_MODEL_MANIFEST_SCHEMA = "focr-reference-model-manifest/v1"
REFERENCE_INFERENCE_BINDING_SCHEMA = "focr-reference-inference-binding/v1"
REFERENCE_MODEL_ROOT_DOMAIN = b"focr-reference-model-root/v1\0"
REFERENCE_INFERENCE_BINDING_DOMAIN = b"focr-reference-inference-binding/v1\0"
MAX_SOURCE_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_BUILD_RECEIPT_BYTES = 1024 * 1024
MAX_SOURCE_ENTRIES = 50_000
MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_LOGICAL_PATH_BYTES = 4096
MAX_SOURCE_PACK_HEADER_BYTES = 32 * 1024
MAX_SOURCE_PACK_BYTES = (
    MAX_SOURCE_TOTAL_BYTES
    + MAX_SOURCE_ENTRIES * (2 * MAX_LOGICAL_PATH_BYTES + 1024)
    + MAX_SOURCE_PACK_HEADER_BYTES
)
MAX_SUBJECT_BINARY_BYTES = 1024 * 1024 * 1024
UNLIMITED_MODEL_FILES = (
    ("config.json", 2881, "27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9"),
    (
        "configuration_deepseek_v2.py",
        10720,
        "b8470dd616ba8745fce6e27b093aef73a098863cc891b2477dcf9326a36000f7",
    ),
    (
        "conversation.py",
        9253,
        "ec7b6ce89bcda643de1f43269ffa66a7b2e65dc3ed30e427958f776546b4ba03",
    ),
    (
        "deepencoder.py",
        38008,
        "0ae2fb6d1e5ae8cf100fc32f854830acd08c821a0a1f23a94a76588c222ddcf2",
    ),
    (
        "model-00001-of-000001.safetensors",
        6_672_547_120,
        "2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6",
    ),
    (
        "model.safetensors.index.json",
        257611,
        "354be1f2dcfb72ebb385e25465522ce5413a77c36f3b35fec088a3162a11af99",
    ),
    (
        "modeling_deepseekv2.py",
        90162,
        "74e36e6bd0ba7bc565ef76464a99baa8e6bccb710ae9c1007b54ac30b855fa4c",
    ),
    (
        "modeling_unlimitedocr.py",
        53431,
        "268bdcbe12cf37bf5a2debb53faf542e56570958a5d9f3314aab3cab2cf6cb48",
    ),
    (
        "processor_config.json",
        466,
        "92588cffb1d7032ec83d0a06c3a5171b41df5cbf432d68765441139a57899328",
    ),
    (
        "special_tokens_map.json",
        801,
        "ab4bd57ce17d62e39e0a39e739de1e407484f090f0b2c7e391312bca7a5b061a",
    ),
    (
        "tokenizer.json",
        9979544,
        "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4",
    ),
    (
        "tokenizer_config.json",
        165938,
        "a0cbe8464049da1f891b7a12676de06af4cb54c130995d42f71adc1c30c6e9f3",
    ),
)
UNLIMITED_MODEL_INDEX = {
    "path": "model.safetensors.index.json",
    "total_size": 6_672_212_480,
    "weight_count": 2710,
    "shards": ["model-00001-of-000001.safetensors"],
}
CORRECTNESS_RECEIPT_SCHEMA = "focr-ocr-comparison/v1"
CORRECTNESS_INPUTS_SCHEMA = "focr-gauntlet-correctness-inputs/v1"
CORRECTNESS_NORMALIZATION = "collapse-unicode-whitespace-v1"
CORRECTNESS_REFERENCE_TRANSFORM = "strip-unlimited-det-spans-v1"
CORRECTNESS_IDENTITY_TRANSFORM = "identity-v1"
CORRECTNESS_METRIC_FORMULAS = {
    "cer_raw": "raw_edit_distance / max(1, raw_reference_chars)",
    "cer_norm": "normalized_edit_distance / max(1, normalized_reference_chars)",
}
CORRECTNESS_TOP_FIELDS = {
    "schema",
    "created_utc",
    "normalization",
    "metric_formulas",
    "aggregate",
    "pages",
}
CORRECTNESS_AGGREGATE_FIELDS = {
    "pages_total",
    "pages_with_hyp",
    "exact_raw",
    "exact_norm",
    "raw_edit_distance",
    "raw_reference_chars",
    "normalized_edit_distance",
    "normalized_reference_chars",
    "reference_source_bytes",
    "reference_source_chars",
    "hypothesis_source_bytes",
    "hypothesis_source_chars",
    "cer_raw",
    "cer_norm",
}
CORRECTNESS_PAGE_FIELDS = {
    "page",
    "status",
    "reference",
    "hypothesis",
    "raw_edit_distance",
    "normalized_edit_distance",
    "ref_chars",
    "hyp_chars",
    "cer_raw",
    "cer_norm",
    "exact",
    "exact_norm",
}
CORRECTNESS_SOURCE_BINDING_FIELDS = {
    "basename",
    "sha256",
    "bytes",
    "chars",
    "normalized_sha256",
    "normalized_bytes",
    "normalized_chars",
}
CORRECTNESS_SCORED_BINDING_FIELDS = CORRECTNESS_SOURCE_BINDING_FIELDS - {"basename"}
MAX_CORRECTNESS_TEXT_CHARS = 131_072
_CORRECTNESS_DET_SPAN_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
_CORRECTNESS_DET_TOKEN_RE = re.compile(r"<\|/?det\|>")

MAX_CV_PCT = 5.0  # docs/PERF_LEDGER.md: cv_pct > 5% is noise, ineligible
MAX_CORRECTNESS_CER = 0.25
_CORRECTNESS_CER_RE = re.compile(r"\bCER_norm=([0-9]+(?:\.[0-9]+)?)\b")
_CORRECTNESS_RECEIPT_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])((?:/|\.{1,2}/)[^()\s;]+cer\.json)(?=$|[\s;)])"
)

LEDGER_STAGES = (
    "preprocess",
    "vision_encode",
    "prefill",
    "decode_per_token",
    "end_to_end",
)

COLUMNS = [
    "date",
    "claim_id",
    "evidence_id",
    "model_commit",
    "fixture_hash",
    "arch/cpu_features",
    "stage",
    "reference_backend",
    "focr_ms",
    "ref_ms",
    "ratio",
    "floor_kind",
    "floor_ms",
    "dist_above_floor",
    "precision (focr vs ref)",
    "threads (focr=ref N)",
    "allocator",
    "command/env",
    "fallback/kill-switch state",
    "correctness_proof",
    "notes",
]


class RowError(ValueError):
    """The inputs cannot honestly produce a ledger row."""


def _command_option(command: object, flag: str, *, required: bool) -> str | None:
    if not isinstance(command, list) or not all(
        isinstance(arg, str) for arg in command
    ):
        raise RowError("reference command must be an argv string list")
    values: list[str] = []
    for index, arg in enumerate(command):
        if arg == flag:
            if index + 1 >= len(command) or not command[index + 1].strip():
                raise RowError(f"reference command has no value for {flag}")
            values.append(command[index + 1])
        elif arg.startswith(flag + "="):
            value = arg.partition("=")[2]
            if not value:
                raise RowError(f"reference command has no value for {flag}")
            values.append(value)
    if len(values) > 1:
        raise RowError(f"reference command repeats {flag}")
    if not values:
        if required:
            raise RowError(f"reference command lacks required {flag}")
        return None
    return values[0]


def _contract_for_entry(entry: str) -> dict:
    spec = REFERENCE_CONTRACTS.get(entry)
    if spec is None:
        raise RowError(
            f"reference command uses unregistered entry {entry!r}; expected one of "
            f"{sorted(REFERENCE_CONTRACTS)}"
        )
    contract_id, torch_version, transformers_version = spec
    return {
        "schema": REFERENCE_CONTRACT_SCHEMA,
        "id": contract_id,
        "entry": entry,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
    }


def _strictly_falsy_or_unset(value: object) -> bool:
    if value == "<unset>":
        return True
    return isinstance(value, str) and value.strip().lower() in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _argv_option(command: object, flag: str) -> str | None:
    """Return one exact argv option value; reject ambiguity."""
    if not isinstance(command, list) or not all(
        isinstance(arg, str) for arg in command
    ):
        return None
    values: list[str] = []
    for index, arg in enumerate(command):
        if arg == flag:
            if index + 1 >= len(command):
                return None
            values.append(command[index + 1])
        elif arg.startswith(flag + "="):
            values.append(arg.partition("=")[2])
    if len(values) != 1 or not values[0]:
        return None
    return values[0]


def _canonical_kill_switch_evidence(focr: dict) -> dict[str, object]:
    evidence = dict(focr.get("focr_env") or {})
    gates = focr.get("precision_gate_states")
    if isinstance(gates, dict):
        evidence.update(gates)
    return evidence


def validate_current_unlimited_release_contract(
    focr: dict,
    roofline: dict,
    focr_stage: dict,
    *,
    focr_stages_sha256: str,
) -> None:
    """Fail closed unless evidence names the current conservative release."""
    if focr.get("precision") != CURRENT_UNLIMITED_PRECISION:
        raise RowError(
            "current Unlimited release rows require runtime precision "
            f"{CURRENT_UNLIMITED_PRECISION!r}"
        )
    if focr_stage.get("precision") != CURRENT_UNLIMITED_PRECISION:
        raise RowError("Unlimited stage precision disagrees with the runtime marker")
    if focr.get("decode_mode") != CURRENT_UNLIMITED_DECODE_MODE:
        raise RowError(
            "current Unlimited release rows require execution-derived decode_mode "
            f"{CURRENT_UNLIMITED_DECODE_MODE!r}"
        )
    if focr.get("quant_recipe") != CURRENT_UNLIMITED_QUANT_RECIPE:
        raise RowError(
            "current Unlimited release rows require quant_recipe "
            f"{CURRENT_UNLIMITED_QUANT_RECIPE!r}"
        )

    model = focr.get("model")
    if (
        not isinstance(model, str)
        or not model.endswith(".focrq")
        or focr.get("model_kind") != "file"
        or focr.get("model_sha256") != CURRENT_UNLIMITED_MODEL_SHA256
        or focr.get("model_size") != CURRENT_UNLIMITED_MODEL_SIZE
    ):
        raise RowError(
            "current Unlimited release rows require the exact hashed conservative "
            "model artifact and byte size"
        )
    if _argv_option(focr.get("command"), "--model") != model:
        raise RowError("focr command does not bind the measured model artifact")

    gates = focr.get("precision_gate_states")
    if not isinstance(gates, dict) or set(gates) != set(PRECISION_GATE_VARS):
        raise RowError("Unlimited precision gate evidence is missing or incomplete")
    for name in OQ14_FALSE_OR_UNSET_VARS:
        if not _strictly_falsy_or_unset(gates.get(name)):
            raise RowError(f"current Unlimited release requires {name} falsy or unset")
    for name in PRESENCE_REJECTED_VARS:
        if gates.get(name) != "<unset>":
            raise RowError(
                f"current Unlimited release forbids presence-only switch {name}, "
                "even when its value looks falsy"
            )

    focr_env = focr.get("focr_env")
    if not isinstance(focr_env, dict):
        raise RowError("focr run lacks structured FOCR_* environment evidence")
    for name in PRECISION_GATE_VARS:
        state = gates[name]
        if state == "<unset>":
            if name in focr_env:
                raise RowError(f"{name} is marked unset but appears in focr_env")
        elif focr_env.get(name) != state:
            raise RowError(f"{name} precision state disagrees with focr_env")

    if not re.fullmatch(r"[0-9a-f]{64}", focr_stages_sha256):
        raise RowError("focr stages input lacks a canonical SHA-256 binding")
    expected_roofline = {
        "arch": "unlimited-ocr",
        "precision": CURRENT_UNLIMITED_DECODE_MODE,
        "timing_precision": CURRENT_UNLIMITED_PRECISION,
        "decode_mode": CURRENT_UNLIMITED_DECODE_MODE,
        "quant_recipe": CURRENT_UNLIMITED_QUANT_RECIPE,
        "model_sha256": CURRENT_UNLIMITED_MODEL_SHA256,
        "model_size": CURRENT_UNLIMITED_MODEL_SIZE,
        "precision_gate_states": gates,
        "stages_json_sha256": focr_stages_sha256,
    }
    mismatches = [
        name
        for name, expected in expected_roofline.items()
        if roofline.get(name) != expected
    ]
    if mismatches:
        raise RowError(
            "Unlimited roofline does not bind the measured precision/model/stages: "
            + ", ".join(mismatches)
        )


def validate_reference_contract(ref: dict, record: dict | None = None) -> dict:
    """Infer the lane contract from argv and validate every supplied stamp."""
    command = ref.get("command")
    entry = _command_option(command, "--entry", required=True)
    if entry is None:
        raise RowError("reference command lacks a usable --entry value")
    contract = _contract_for_entry(entry)

    def base_version(value: object) -> str:
        return value.split("+", 1)[0] if isinstance(value, str) else ""

    for field in ("torch_version", "transformers_version"):
        if base_version(ref.get(field)) != contract[field]:
            raise RowError(
                f"reference {field}={ref.get(field)!r} violates contract "
                f"{contract['id']}: expected {contract[field]}"
            )
    for flag, field in (
        ("--pin-torch", "torch_version"),
        ("--pin-transformers", "transformers_version"),
    ):
        asserted = _command_option(command, flag, required=False)
        if asserted is not None and asserted != contract[field]:
            raise RowError(
                f"reference command {flag}={asserted!r} contradicts contract "
                f"{contract['id']}: expected {contract[field]}"
            )

    stamp = ref.get("reference_contract")
    if stamp is not None and stamp != contract:
        raise RowError("reference document contract stamp disagrees with its entry")
    if record is not None:
        record_stamp = record.get("reference_contract")
        if stamp is not None and record_stamp != contract:
            raise RowError("stamped reference document has an unstamped/mixed stage")
        if record_stamp is not None and record_stamp != contract:
            raise RowError("reference stage contract stamp disagrees with its entry")
    return contract


def _read_bounded_file(
    path: str, max_bytes: int = MAX_RAW_TIMING_FILE_BYTES
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RowError(f"input is not a regular file: {path}")
        if before.st_size > max_bytes:
            raise RowError(f"input exceeds {max_bytes} bytes: {path}")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > max_bytes:
                raise RowError(f"input exceeds {max_bytes} bytes while reading: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise RowError(f"input changed while being read: {path}")
        content = b"".join(chunks)
        if len(content) != before.st_size:
            raise RowError(f"input changed length while being read: {path}")
        return content
    finally:
        os.close(descriptor)


def load_json(path: str, want_schema: str | None = None) -> dict:
    try:
        doc = json.loads(_read_bounded_file(path).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RowError) as err:
        raise RowError(f"{path}: {err}") from err
    if not isinstance(doc, dict):
        raise RowError(f"{path}: expected a JSON object")
    if want_schema and doc.get("schema") != want_schema:
        raise RowError(
            f"{path}: expected schema {want_schema}, got {doc.get('schema')!r}"
        )
    return doc


def sha256_file(path: str, max_bytes: int = MAX_RAW_TIMING_FILE_BYTES) -> str:
    return _stable_file_identity(path, max_bytes)["sha256"]


def current_git_head(root: str = ROOT) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise RowError(f"cannot resolve git HEAD: {err}") from err
    head = proc.stdout.strip()
    if proc.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise RowError("cannot resolve a canonical 40-hex git HEAD")
    return head


def _git_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_INDEX_FILE",
        "GIT_CONFIG",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_NAMESPACE",
        "GIT_CEILING_DIRECTORIES",
    ):
        env.pop(key, None)
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    return env


def _git_blob(root: str, git_head: str, relative: str) -> bytes:
    if re.fullmatch(r"[0-9a-f]{40}", git_head) is None:
        raise RowError("producer root requires a canonical source git HEAD")
    try:
        result = subprocess.run(
            ["git", "show", f"{git_head}:{relative}"],
            cwd=root,
            env=_git_environment(),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RowError(f"cannot read producer input {relative} from source HEAD: {error}") from error
    if result.returncode != 0:
        raise RowError(f"producer input is absent from source HEAD: {relative}")
    if len(result.stdout) > MAX_SOURCE_FILE_BYTES:
        raise RowError(f"producer input exceeds {MAX_SOURCE_FILE_BYTES} bytes: {relative}")
    return result.stdout


def gauntlet_producer_root(
    root: str, git_head: str, *, verify_live: bool = True
) -> str:
    """Hash the exact committed scripts/config that produced and validate a row."""
    digest = hashlib.sha256(PRODUCER_ROOT_DOMAIN)
    for relative in PRODUCER_PATHS:
        content = _git_blob(root, git_head, relative)
        identity = {
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        if verify_live:
            live = _stable_file_identity(
                os.path.join(root, *Path(relative).parts), MAX_SOURCE_FILE_BYTES
            )
            if live != identity:
                raise RowError(
                    f"live evidence producer/config differs from source HEAD: {relative}"
                )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(identity["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(identity["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_allowed_evidence_path(value: object) -> str:
    relative = _canonical_logical_path(value, "allowed evidence path")
    path = Path(relative)
    if len(path.parts) < 3 or path.parts[:2] != ("artifacts", "perf"):
        raise RowError(
            "allowed evidence path must be a child of artifacts/perf"
        )
    return relative


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict:
    value: dict = {}
    for key, item in pairs:
        if key in value:
            raise RowError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _strict_json_bytes(content: bytes, label: str) -> dict:
    try:
        value = json.loads(
            content.decode("utf-8"), object_pairs_hook=_strict_json_object
        )
    except RowError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise RowError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise RowError(f"{label} is not a JSON object")
    return value


def _canonical_utc_timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ) is None:
        raise RowError(f"{label} is not canonical UTC seconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise RowError(f"{label} is not a valid UTC timestamp") from error
    return value


def _reject_symlink_components(path: str) -> str:
    requested = Path(os.path.abspath(path))
    component = Path(requested.anchor)
    for part in requested.parts[1:]:
        component /= part
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise RowError(f"evidence path contains a symlink component: {component}")
    return os.fspath(requested)


def _stable_file_identity(path: str, maximum: int) -> dict:
    path = _reject_symlink_components(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RowError(f"cannot open bound file {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RowError(f"bound input is not a regular file: {path}")
        if before.st_size > maximum:
            raise RowError(f"bound input exceeds {maximum} bytes: {path}")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum:
                raise RowError(f"bound input grew beyond {maximum} bytes: {path}")
            digest.update(chunk)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise RowError(f"bound input changed while hashing: {path}")
        if observed != before.st_size:
            raise RowError(f"bound input changed length while hashing: {path}")
        return {"sha256": digest.hexdigest(), "size": observed}
    finally:
        os.close(descriptor)


def _canonical_logical_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise RowError(f"{label} is not a nonempty logical path")
    if len(value.encode("utf-8")) > MAX_LOGICAL_PATH_BYTES:
        raise RowError(f"{label} exceeds the logical path bound")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RowError(f"{label} is noncanonical: {value!r}")
    if path.as_posix() != value:
        raise RowError(f"{label} is not canonical POSIX spelling: {value!r}")
    return value


def _canonical_source_root(entries: list[dict]) -> str:
    digest = hashlib.sha256(SOURCE_ROOT_DOMAIN)
    seen: set[tuple[str, str]] = set()
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "repository",
            "path",
            "size",
            "sha256",
        }:
            raise RowError("source manifest entry fields are noncanonical")
        repository = entry.get("repository")
        path = entry.get("path")
        size = entry.get("size")
        sha256 = entry.get("sha256")
        if (
            not isinstance(repository, str)
            or not repository
            or "\0" in repository
            or len(repository.encode("utf-8")) > MAX_LOGICAL_PATH_BYTES
        ):
            raise RowError("source manifest repository id is noncanonical")
        path = _canonical_logical_path(path, "source manifest path")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or not 0 <= size <= MAX_SOURCE_FILE_BYTES
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise RowError("source manifest size/hash is noncanonical")
        identity = (repository, path)
        if identity in seen:
            raise RowError(f"source manifest contains duplicate entry: {repository}/{path}")
        seen.add(identity)
        total += size
        if total > MAX_SOURCE_TOTAL_BYTES:
            raise RowError("source manifest exceeds its total-byte bound")
        digest.update(repository.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_head_at(path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RowError(f"cannot resolve repository HEAD for {path}: {error}") from error
    value = result.stdout.strip()
    if result.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RowError(f"cannot resolve canonical repository HEAD for {path}")
    return value


def _active_build_toolchain(workspace_root: str) -> tuple[dict, str]:
    outputs: dict[str, str] = {}
    for name, command in (
        ("rustc_verbose_version", ["rustc", "-vV"]),
        ("cargo_version", ["cargo", "--version"]),
        ("rch_version", ["rch", "--version"]),
    ):
        try:
            result = subprocess.run(
                command,
                cwd=workspace_root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RowError(f"cannot identify active build toolchain: {error}") from error
        if result.returncode != 0 or not result.stdout.strip():
            raise RowError(f"cannot identify active build toolchain component {name}")
        outputs[name] = result.stdout if name == "rustc_verbose_version" else result.stdout.strip()
    host = next(
        (
            line.split(": ", 1)[1]
            for line in outputs["rustc_verbose_version"].splitlines()
            if line.startswith("host: ")
        ),
        "",
    )
    if not host or any(character.isspace() for character in host):
        raise RowError("active rustc output has no canonical host triple")
    return outputs, host


def _validate_source_manifest(
    manifest: dict,
    *,
    workspace_root: str,
    expected_head: str,
    verify_live: bool,
    verify_repository_heads: bool | None = None,
) -> dict:
    if verify_repository_heads is None:
        verify_repository_heads = verify_live
    required_fields = {
        "schema",
        "created_utc",
        "root_hash_algorithm",
        "root_sha256",
        "entry_count",
        "repositories",
        "cargo_config_files",
        "entries",
    }
    if set(manifest) != required_fields or manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise RowError("source input manifest fields/schema are noncanonical")
    _canonical_utc_timestamp(manifest.get("created_utc"), "source manifest created_utc")
    entries = manifest.get("entries")
    if (
        not isinstance(entries, list)
        or not entries
        or len(entries) > MAX_SOURCE_ENTRIES
        or manifest.get("entry_count") != len(entries)
    ):
        raise RowError("source input manifest count is invalid")
    identities = [
        (entry.get("repository"), entry.get("path"))
        if isinstance(entry, dict)
        else (None, None)
        for entry in entries
    ]
    if identities != sorted(identities):
        raise RowError("source input manifest entries are not stably sorted")
    root_sha256 = _canonical_source_root(entries)
    if (
        manifest.get("root_hash_algorithm") != SOURCE_ROOT_ALGORITHM
        or manifest.get("root_sha256") != root_sha256
    ):
        raise RowError("source input manifest root is invalid")

    repositories = manifest.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise RowError("source input manifest has no repository census")
    repository_paths: dict[str, str] = {}
    for record in repositories:
        if not isinstance(record, dict) or set(record) != {
            "id",
            "path",
            "git_head",
            "packages",
            "selectors",
        }:
            raise RowError("source repository record fields are noncanonical")
        identifier = record.get("id")
        physical = record.get("path")
        git_head = record.get("git_head")
        packages = record.get("packages")
        selectors = record.get("selectors")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in repository_paths
            or not isinstance(physical, str)
            or not os.path.isabs(physical)
            or os.path.realpath(physical) != os.path.normpath(physical)
            or re.fullmatch(r"[0-9a-f]{40}", str(git_head)) is None
            or not isinstance(packages, list)
            or packages != sorted(set(packages))
            or not all(isinstance(item, str) and item for item in packages)
            or not isinstance(selectors, list)
            or selectors != sorted(set(selectors))
            or not all(isinstance(item, str) and item for item in selectors)
        ):
            raise RowError("source repository record identity is noncanonical")
        physical = _reject_symlink_components(physical)
        if verify_repository_heads and _git_head_at(physical) != git_head:
            raise RowError(f"source repository HEAD drifted: {identifier}")
        repository_paths[identifier] = physical
    workspace = repository_paths.get("workspace")
    if workspace is None or os.path.realpath(workspace) != os.path.realpath(workspace_root):
        raise RowError("source manifest workspace repository path is not current workspace")
    workspace_record = next(record for record in repositories if record["id"] == "workspace")
    if workspace_record.get("git_head") != expected_head:
        raise RowError("source manifest workspace HEAD does not match current HEAD")

    configs = manifest.get("cargo_config_files")
    if not isinstance(configs, list):
        raise RowError("source input manifest cargo config census is invalid")
    config_paths: dict[str, tuple[str, dict]] = {}
    for record in configs:
        if not isinstance(record, dict) or set(record) != {
            "logical_path",
            "physical_path",
            "sha256",
            "size",
        }:
            raise RowError("cargo config record fields are noncanonical")
        logical = _canonical_logical_path(record.get("logical_path"), "cargo config logical path")
        physical = record.get("physical_path")
        if (
            logical in config_paths
            or not isinstance(physical, str)
            or not os.path.isabs(physical)
            or os.path.realpath(physical) != os.path.normpath(physical)
        ):
            raise RowError("cargo config record identity is noncanonical")
        config_paths[logical] = (_reject_symlink_components(physical), record)

    physical_entries: list[dict] = []
    for entry in entries:
        repository = entry["repository"]
        logical = entry["path"]
        if repository == "cargo-config":
            config = config_paths.get(logical)
            if config is None:
                raise RowError(f"source manifest cargo config is unregistered: {logical}")
            physical, record = config
            if {"sha256": record["sha256"], "size": record["size"]} != {
                "sha256": entry["sha256"],
                "size": entry["size"],
            }:
                raise RowError(f"cargo config binding disagrees with source entry: {logical}")
        else:
            repository_root = repository_paths.get(repository)
            if repository_root is None:
                raise RowError(f"source manifest uses unknown repository: {repository}")
            physical = os.path.abspath(os.path.join(repository_root, logical))
            try:
                if os.path.commonpath((repository_root, physical)) != repository_root:
                    raise RowError(f"source input escapes repository: {repository}/{logical}")
            except ValueError as error:
                raise RowError(f"source input path is not comparable: {logical}") from error
        if verify_live and _stable_file_identity(physical, MAX_SOURCE_FILE_BYTES) != {
            "sha256": entry["sha256"],
            "size": entry["size"],
        }:
            raise RowError(f"source input drifted after build: {repository}/{logical}")
        physical_entries.append(
            {"entry": dict(entry), "physical_path": physical}
        )
    if {entry["path"] for entry in entries if entry["repository"] == "cargo-config"} != set(config_paths):
        raise RowError("cargo config census is not exhaustive")
    return {
        "root_sha256": root_sha256,
        "entry_count": len(entries),
        "physical_entries": physical_entries,
    }


def _workspace_manifest_entry(manifest: dict, name: str) -> dict:
    matches = [
        entry
        for entry in manifest["entries"]
        if entry.get("repository") == "workspace" and entry.get("path") == name
    ]
    if len(matches) != 1:
        raise RowError(f"source manifest must bind exactly one workspace/{name}")
    return {"sha256": matches[0]["sha256"], "size": matches[0]["size"]}


def build_receipt_document(
    *,
    created_utc: str,
    git_head: str,
    target_triple: str,
    cargo_target_dir: str,
    toolchain: dict,
    build_environment: dict,
    source_manifest_path: str,
    source_manifest_identity: dict,
    source_manifest: dict,
    binary_path: str,
    binary_identity: dict,
) -> dict:
    """Construct the exact receipt document written by the runbook producer."""
    return {
        "schema": BUILD_RECEIPT_SCHEMA,
        "created_utc": created_utc,
        "git_head": git_head,
        "profile": "release-perf",
        "target_triple": target_triple,
        "build": {
            "runner": "rch",
            "command": [
                "rch",
                "exec",
                "--",
                "cargo",
                "build",
                "--locked",
                "--profile",
                "release-perf",
                "--bin",
                "focr",
                "--target",
                target_triple,
            ],
            "cargo_target_dir": os.path.realpath(cargo_target_dir),
        },
        "toolchain": dict(toolchain),
        "build_environment": copy.deepcopy(build_environment),
        "inputs": {
            name: _workspace_manifest_entry(source_manifest, name)
            for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")
        },
        "source_manifest": {
            "path": os.path.realpath(source_manifest_path),
            **dict(source_manifest_identity),
            "schema": SOURCE_MANIFEST_SCHEMA,
            "root_sha256": source_manifest["root_sha256"],
            "entry_count": source_manifest["entry_count"],
            "root_hash_algorithm": source_manifest["root_hash_algorithm"],
        },
        "binary": {"path": os.path.realpath(binary_path), **dict(binary_identity)},
    }


def validate_build_provenance(
    focr: dict,
    *,
    current_head: str,
    workspace_root: str = ROOT,
    verify_live: bool = True,
    expected_toolchain: dict | None = None,
    expected_host: str | None = None,
) -> dict:
    aggregate_fields = (
        "binary",
        "binary_sha256",
        "binary_size",
        "binary_origin",
        "build_receipt",
        "build_receipt_sha256",
    )
    if re.fullmatch(r"[0-9a-f]{40}", current_head) is None:
        raise RowError("build provenance requires a canonical current HEAD")
    if any(field not in focr for field in aggregate_fields):
        raise RowError("focr aggregate lacks trusted build-receipt identity")
    run_dir = focr.get("run_dir")
    if (
        not isinstance(run_dir, str)
        or not os.path.isabs(run_dir)
        or os.path.normpath(run_dir) != run_dir
        or os.path.basename(run_dir) != "raw"
    ):
        raise RowError("focr run directory is not an absolute capture path")
    capture_root = os.path.dirname(os.path.normpath(run_dir))
    expected_binary_path = os.path.join(capture_root, "subject", "release-perf", "focr")
    expected_receipt_path = os.path.join(capture_root, "subject", "build_receipt.json")
    if focr.get("binary") != expected_binary_path:
        raise RowError("focr aggregate binary is not the evidence-local release-perf subject")
    if focr.get("build_receipt") != expected_receipt_path:
        raise RowError("focr aggregate build receipt escapes the capture subject directory")

    binary_identity = _stable_file_identity(expected_binary_path, MAX_SUBJECT_BINARY_BYTES)
    if binary_identity != {
        "sha256": focr.get("binary_sha256"),
        "size": focr.get("binary_size"),
    }:
        raise RowError("focr aggregate binary identity is not physical")
    receipt_bytes = _read_bounded_file(expected_receipt_path, MAX_BUILD_RECEIPT_BYTES)
    receipt_identity = {
        "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "size": len(receipt_bytes),
    }
    if receipt_identity["sha256"] != focr.get("build_receipt_sha256"):
        raise RowError("focr aggregate build receipt hash is not physical")
    receipt = _strict_json_bytes(receipt_bytes, "build receipt")

    receipt_fields = {
        "schema",
        "created_utc",
        "git_head",
        "profile",
        "target_triple",
        "build",
        "toolchain",
        "build_environment",
        "inputs",
        "source_manifest",
        "binary",
    }
    if set(receipt) != receipt_fields or receipt.get("schema") != BUILD_RECEIPT_SCHEMA:
        raise RowError("build receipt fields/schema are noncanonical")
    _canonical_utc_timestamp(receipt.get("created_utc"), "build receipt created_utc")
    target = receipt.get("target_triple")
    expected_command = [
        "rch",
        "exec",
        "--",
        "cargo",
        "build",
        "--locked",
        "--profile",
        "release-perf",
        "--bin",
        "focr",
        "--target",
        target,
    ]
    build = receipt.get("build")
    if (
        receipt.get("git_head") != current_head
        or receipt.get("profile") != "release-perf"
        or not isinstance(target, str)
        or not target
        or any(character.isspace() for character in target)
        or not isinstance(build, dict)
        or set(build) != {"runner", "command", "cargo_target_dir"}
        or build.get("runner") != "rch"
        or build.get("command") != expected_command
        or not isinstance(build.get("cargo_target_dir"), str)
        or not build["cargo_target_dir"]
        or not os.path.isabs(build["cargo_target_dir"])
        or os.path.normpath(build["cargo_target_dir"]) != build["cargo_target_dir"]
    ):
        raise RowError("build receipt HEAD/profile/target/command is invalid")
    toolchain = receipt.get("toolchain")
    if (
        not isinstance(toolchain, dict)
        or set(toolchain)
        != {"rustc_verbose_version", "cargo_version", "rch_version"}
        or any(not isinstance(toolchain.get(key), str) or not toolchain[key].strip() for key in toolchain)
    ):
        raise RowError("build receipt toolchain identity is invalid")
    if expected_toolchain is None or expected_host is None:
        expected_toolchain, expected_host = _active_build_toolchain(workspace_root)
    if verify_live and (toolchain != expected_toolchain or target != expected_host):
        raise RowError("active target/toolchain drifted from the trusted build receipt")

    environment = receipt.get("build_environment")
    expected_environment_fields = {
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_RUSTFLAGS",
        "CARGO_HOME",
        "target_rustflags_env_name",
        "target_rustflags_env_value",
        "rustc_overrides",
        "release_perf_profile_overrides",
        "cargo_config_build_rustflags",
        "cargo_config_target",
    }
    if not isinstance(environment, dict) or set(environment) != expected_environment_fields:
        raise RowError("build receipt environment fields are noncanonical")
    scalar_environment = expected_environment_fields - {
        "rustc_overrides",
        "release_perf_profile_overrides",
        "cargo_config_build_rustflags",
        "cargo_config_target",
    }
    if any(environment[key] is not None and not isinstance(environment[key], str) for key in scalar_environment):
        raise RowError("build receipt Rust flag environment is malformed")
    expected_target_flags = "CARGO_TARGET_" + target.upper().replace("-", "_") + "_RUSTFLAGS"
    overrides = environment.get("rustc_overrides")
    profile_overrides = environment.get("release_perf_profile_overrides")
    if (
        environment.get("target_rustflags_env_name") != expected_target_flags
        or not isinstance(overrides, dict)
        or set(overrides) != {"RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"}
        or any(value is not None and not isinstance(value, str) for value in overrides.values())
        or not isinstance(profile_overrides, dict)
        or any(
            not isinstance(key, str)
            or not key.startswith("CARGO_PROFILE_RELEASE_PERF_")
            or not isinstance(value, str)
            for key, value in profile_overrides.items()
        )
    ):
        raise RowError("build receipt compiler/profile environment is malformed")
    for key in ("cargo_config_build_rustflags", "cargo_config_target"):
        value = environment.get(key)
        if (
            not isinstance(value, dict)
            or set(value) != {"status", "value"}
            or value.get("status") not in {"set", "unset"}
            or (value.get("status") == "unset" and value.get("value") is not None)
            or (value.get("status") == "set" and not isinstance(value.get("value"), str))
        ):
            raise RowError(f"build receipt {key} binding is malformed")

    source_binding = receipt.get("source_manifest")
    if not isinstance(source_binding, dict) or set(source_binding) != {
        "path",
        "sha256",
        "size",
        "schema",
        "root_sha256",
        "entry_count",
        "root_hash_algorithm",
    }:
        raise RowError("build receipt source-manifest binding is noncanonical")
    source_path = source_binding.get("path")
    if (
        not isinstance(source_path, str)
        or not os.path.isabs(source_path)
        or os.path.normpath(source_path) != source_path
    ):
        raise RowError("build receipt source-manifest path is not absolute")
    manifest_bytes = _read_bounded_file(source_path, MAX_SOURCE_MANIFEST_BYTES)
    manifest_identity = {
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "size": len(manifest_bytes),
    }
    if manifest_identity != {
        "sha256": source_binding.get("sha256"),
        "size": source_binding.get("size"),
    }:
        raise RowError("build receipt does not bind its physical source manifest")
    manifest = _strict_json_bytes(manifest_bytes, "source input manifest")
    manifest_summary = _validate_source_manifest(
        manifest,
        workspace_root=workspace_root,
        expected_head=current_head,
        verify_live=verify_live,
    )
    if (
        source_binding.get("schema") != SOURCE_MANIFEST_SCHEMA
        or source_binding.get("root_hash_algorithm") != SOURCE_ROOT_ALGORITHM
        or source_binding.get("root_sha256") != manifest_summary["root_sha256"]
        or source_binding.get("entry_count") != manifest_summary["entry_count"]
    ):
        raise RowError("build receipt source root disagrees with its manifest")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
    }:
        raise RowError("build receipt required input bindings are noncanonical")
    for name in inputs:
        if inputs[name] != _workspace_manifest_entry(manifest, name):
            raise RowError(f"build receipt {name} binding is invalid")

    receipt_binary = receipt.get("binary")
    if (
        not isinstance(receipt_binary, dict)
        or set(receipt_binary) != {"path", "sha256", "size"}
        or not isinstance(receipt_binary.get("path"), str)
        or not os.path.isabs(receipt_binary["path"])
        or os.path.normpath(receipt_binary["path"]) != receipt_binary["path"]
        or receipt_binary.get("path") != focr.get("binary_origin")
        or receipt_binary.get("sha256") != binary_identity["sha256"]
        or receipt_binary.get("size") != binary_identity["size"]
    ):
        raise RowError("build receipt does not bind the captured subject binary/origin")

    raw = focr.get("raw_timing")
    records = raw.get("records") if isinstance(raw, dict) else None
    if not isinstance(records, list) or not records:
        raise RowError("focr aggregate has no raw records for build receipt replay")
    for index, record in enumerate(records, 1):
        binding = record.get("raw_files", {}).get("meta") if isinstance(record, dict) else None
        expected_id = f"run_{index:03d}"
        expected_name = expected_id + ".meta.json"
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or record.get("run_id") != expected_id
            or binding.get("path") != expected_name
            or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", ""))) is None
        ):
            raise RowError("focr raw record lacks build-bound metadata")
        meta_path = os.path.join(run_dir, expected_name)
        meta_bytes = _read_bounded_file(meta_path)
        if hashlib.sha256(meta_bytes).hexdigest() != binding.get("sha256"):
            raise RowError("focr raw metadata hash does not verify")
        meta = _strict_json_bytes(meta_bytes, "focr raw metadata")
        drift = [field for field in aggregate_fields if meta.get(field) != focr.get(field)]
        if drift:
            raise RowError("focr raw build identity drifted: " + ", ".join(drift))
    return {
        "receipt": receipt,
        "receipt_path": expected_receipt_path,
        "receipt_bytes": receipt_bytes,
        "receipt_identity": receipt_identity,
        "manifest": manifest,
        "manifest_path": source_path,
        "manifest_bytes": manifest_bytes,
        "manifest_identity": manifest_identity,
        "physical_entries": manifest_summary["physical_entries"],
        "binary_path": expected_binary_path,
        "binary_identity": binary_identity,
    }


def _reference_model_root(files: list[dict]) -> str:
    digest = hashlib.sha256(REFERENCE_MODEL_ROOT_DOMAIN)
    for item in files:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_reference_model_manifest(manifest: object) -> dict:
    fields = {
        "schema",
        "model_id",
        "model_commit",
        "synthetic",
        "citable",
        "file_count",
        "root_hash_domain",
        "root_sha256",
        "index",
        "files",
    }
    expected_files = [
        {"path": path, "bytes": size, "sha256": sha256}
        for path, size, sha256 in UNLIMITED_MODEL_FILES
    ]
    if (
        not isinstance(manifest, dict)
        or set(manifest) != fields
        or manifest.get("schema") != REFERENCE_MODEL_MANIFEST_SCHEMA
        or manifest.get("model_id") != "baidu/Unlimited-OCR"
        or manifest.get("model_commit") != MODEL_COMMIT
        or manifest.get("synthetic") is not False
        or manifest.get("citable") is not True
        or manifest.get("file_count") != len(expected_files)
        or manifest.get("root_hash_domain") != REFERENCE_MODEL_ROOT_DOMAIN.decode("ascii")
        or manifest.get("index") != UNLIMITED_MODEL_INDEX
        or manifest.get("files") != expected_files
    ):
        raise RowError("reference model manifest does not match the exact 12-file truth pack")
    expected_root = _reference_model_root(expected_files)
    if manifest.get("root_sha256") != expected_root:
        raise RowError("reference model manifest root is invalid")
    return manifest


def _binding_option(argv: list[str], name: str, *, required: bool = True) -> str | None:
    value = _command_option(argv, name, required=required)
    return value


def _validate_reference_inference_binding(
    binding: object,
    *,
    ref: dict,
    manifest: dict,
    workspace_root: str,
    verify_sources: bool,
) -> dict:
    fields = {
        "schema",
        "model_root_sha256",
        "model_commit",
        "entry",
        "setup",
        "stage",
        "page",
        "page_sha256",
        "model_dir",
        "max_length",
        "text_dir",
        "backend",
        "precision",
        "threads",
        "runs",
        "warmup",
        "allocator",
        "argv",
        "env_pins",
        "ambient_env",
        "torch_version",
        "transformers_version",
        "reference_contract",
        "hf_modules_cache",
        "sources",
        "binding_hash_domain",
        "binding_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != fields:
        raise RowError("reference inference binding fields are noncanonical")
    argv = binding.get("argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise RowError("reference inference binding argv is malformed")
    entry = CURRENT_UNLIMITED_REFERENCE_ENTRY
    setup = "gauntlet_ref_unlimited:setup"
    if (
        binding.get("schema") != REFERENCE_INFERENCE_BINDING_SCHEMA
        or binding.get("model_root_sha256") != manifest["root_sha256"]
        or binding.get("model_commit") != MODEL_COMMIT
        or binding.get("entry") != entry
        or binding.get("setup") != setup
        or binding.get("page") != ref.get("page")
        or binding.get("page_sha256") != ref.get("page_sha256")
        or binding.get("model_dir") != ref.get("model")
        or binding.get("max_length") != ref.get("max_length")
        or binding.get("text_dir") != ref.get("text_dir")
        or binding.get("backend") != ref.get("backend")
        or binding.get("precision") != ref.get("precision")
        or binding.get("threads") != ref.get("threads")
        or binding.get("runs") != ref.get("runs")
        or binding.get("warmup") != ref.get("warmup")
        or binding.get("allocator") != ref.get("allocator")
        or argv != ref.get("command")
        or binding.get("env_pins") != ref.get("env_pins")
        or binding.get("ambient_env") != ref.get("ambient_env")
        or binding.get("torch_version") != ref.get("torch_version")
        or binding.get("transformers_version") != ref.get("transformers_version")
        or binding.get("reference_contract") != ref.get("reference_contract")
        or _binding_option(argv, "--entry") != entry
        or _binding_option(argv, "--setup") != setup
        or _binding_option(argv, "--page") != ref.get("page")
        or _binding_option(argv, "--model-dir") != ref.get("model")
        or _binding_option(argv, "--backend") != ref.get("backend")
        or _binding_option(argv, "--precision") != ref.get("precision")
        or _binding_option(argv, "--max-length") != str(ref.get("max_length"))
        or _binding_option(argv, "--text-dir") != ref.get("text_dir")
        or _binding_option(argv, "--threads") != str(ref.get("threads"))
        or _binding_option(argv, "--runs") != str(ref.get("runs"))
        or _binding_option(argv, "--warmup") != str(ref.get("warmup"))
    ):
        raise RowError("reference inference binding disagrees with measured runtime/argv")
    stage = binding.get("stage")
    command_stage = _binding_option(argv, "--stage", required=False)
    if not isinstance(stage, str) or stage not in {*LEDGER_STAGES, "all"} or (
        command_stage is not None and command_stage != stage
    ):
        raise RowError("reference inference binding stage is invalid")
    output = _binding_option(argv, "--out")
    expected_argv = [
        argv[0] if argv else "",
        "--stage",
        stage,
        "--page",
        str(ref.get("page")),
        "--model-dir",
        str(ref.get("model")),
        "--backend",
        str(ref.get("backend")),
        "--precision",
        str(ref.get("precision")),
        "--max-length",
        str(ref.get("max_length")),
        "--text-dir",
        str(ref.get("text_dir")),
        "--entry",
        entry,
        "--setup",
        setup,
        "--runs",
        str(ref.get("runs")),
        "--warmup",
        str(ref.get("warmup")),
        "--threads",
        str(ref.get("threads")),
        "--out",
        str(output),
    ]
    if (
        not argv
        or os.path.basename(argv[0]) != "gauntlet_reference.py"
        or argv != expected_argv
    ):
        raise RowError("reference inference argv is not the canonical runbook invocation")
    if (
        ref.get("max_length") != 8192
        or not isinstance(ref.get("text_dir"), str)
        or not os.path.isabs(ref["text_dir"])
        or binding.get("ambient_env")
        != {"FOCR_REF_MAX_LENGTH": "<unset>", "FOCR_REF_TEXT_DIR": "<unset>"}
    ):
        raise RowError("reference max-length/text output ambient contract is invalid")
    pins = binding.get("env_pins")
    expected_pins = {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "FOCR_THREADS",
    }
    if not isinstance(pins, dict) or set(pins) != expected_pins or any(
        value != str(ref.get("threads")) for value in pins.values()
    ):
        raise RowError("reference inference binding thread pins are incomplete")
    cache = binding.get("hf_modules_cache")
    expected_cache_name = os.path.basename(str(output)) + ".hf_modules_cache"
    expected_cache_parent = os.path.dirname(os.path.abspath(str(output)))
    if (
        not isinstance(cache, dict)
        or set(cache) != {"evidence_dir", "path", "effective_path", "fresh"}
        or cache.get("evidence_dir") != expected_cache_parent
        or cache.get("path") != expected_cache_name
        or cache.get("effective_path") != os.path.join(expected_cache_parent, expected_cache_name)
        or cache.get("fresh") is not True
    ):
        raise RowError("reference inference binding HF module cache is not evidence-local/fresh")

    expected_sources = (
        ("harness", "gauntlet_reference:main", "scripts/gauntlet_reference.py"),
        ("entry", entry, "scripts/gauntlet_ref_unlimited.py"),
        ("setup", setup, "scripts/gauntlet_ref_unlimited.py"),
    )
    sources = binding.get("sources")
    if not isinstance(sources, list) or len(sources) != len(expected_sources):
        raise RowError("reference inference binding source census is incomplete")
    for source, (role, callable_name, relative) in zip(sources, expected_sources, strict=True):
        if (
            not isinstance(source, dict)
            or set(source) != {"role", "callable", "path", "bytes", "sha256"}
            or source.get("role") != role
            or source.get("callable") != callable_name
            or source.get("path") != relative
            or not isinstance(source.get("bytes"), int)
            or isinstance(source.get("bytes"), bool)
            or source["bytes"] <= 0
            or re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))) is None
        ):
            raise RowError("reference inference source binding is noncanonical")
        if verify_sources and _stable_file_identity(
            os.path.join(workspace_root, relative), MAX_SOURCE_FILE_BYTES
        ) != {"sha256": source["sha256"], "size": source["bytes"]}:
            raise RowError(f"reference inference source drifted: {relative}")
    unsigned = dict(binding)
    unsigned.pop("binding_hash_domain")
    unsigned.pop("binding_sha256")
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    expected_hash = hashlib.sha256(REFERENCE_INFERENCE_BINDING_DOMAIN + canonical).hexdigest()
    if (
        binding.get("binding_hash_domain")
        != REFERENCE_INFERENCE_BINDING_DOMAIN.decode("ascii")
        or binding.get("binding_sha256") != expected_hash
    ):
        raise RowError("reference inference binding hash is invalid")
    return binding


def validate_reference_provenance(
    ref: dict, *, workspace_root: str = ROOT, verify_sources: bool = True
) -> dict:
    manifest = _validate_reference_model_manifest(ref.get("reference_model_manifest"))
    binding = _validate_reference_inference_binding(
        ref.get("reference_inference_binding"),
        ref=ref,
        manifest=manifest,
        workspace_root=workspace_root,
        verify_sources=verify_sources,
    )
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    binding_bytes = (
        json.dumps(binding, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return {
        "manifest": manifest,
        "manifest_bytes": manifest_bytes,
        "manifest_identity": {
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "size": len(manifest_bytes),
        },
        "binding": binding,
        "binding_bytes": binding_bytes,
        "binding_identity": {
            "sha256": hashlib.sha256(binding_bytes).hexdigest(),
            "size": len(binding_bytes),
        },
    }


def _correctness_exact_fields(value: object, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise RowError(f"{label} fields are noncanonical")
    return value


def _correctness_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RowError(f"{label} must be a nonnegative integer")
    return value


def _correctness_float(value: object, expected: float, label: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise RowError(f"{label} is not derived from the authoritative integers")


def _correctness_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _correctness_binding(raw: bytes, *, basename: str | None = None) -> dict:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RowError(f"correctness text is not UTF-8: {error}") from error
    normalized = _correctness_normalize(text).encode("utf-8")
    binding = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "chars": len(text),
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "normalized_bytes": len(normalized),
        "normalized_chars": len(normalized.decode("utf-8")),
    }
    return {"basename": basename, **binding} if basename is not None else binding


def _correctness_validate_binding_types(binding: dict, label: str) -> None:
    for field in ("sha256", "normalized_sha256"):
        if (
            not isinstance(binding.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", binding[field]) is None
        ):
            raise RowError(f"{label} {field} is not a canonical SHA-256")
    for field in (
        "bytes",
        "chars",
        "normalized_bytes",
        "normalized_chars",
    ):
        _correctness_int(binding.get(field), f"{label} {field}")


def _correctness_transform_reference(text: str) -> tuple[str, int]:
    transformed, matches = _CORRECTNESS_DET_SPAN_RE.subn("", text)
    if _CORRECTNESS_DET_TOKEN_RE.search(transformed):
        raise RowError("correctness reference has an unbalanced or nested det span")
    return transformed, matches


def _correctness_levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)
    width = len(right)
    mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    char_masks: dict[str, int] = {}
    for index, char in enumerate(right):
        char_masks[char] = char_masks.get(char, 0) | (1 << index)
    positive = mask
    negative = 0
    distance = width
    for char in left:
        equal = char_masks.get(char, 0)
        vertical = equal | negative
        horizontal = ((((equal & positive) + positive) ^ positive) | equal) & mask
        positive_horizontal = (negative | ~(horizontal | positive)) & mask
        negative_horizontal = positive & horizontal
        if positive_horizontal & high_bit:
            distance += 1
        elif negative_horizontal & high_bit:
            distance -= 1
        positive_horizontal = ((positive_horizontal << 1) | 1) & mask
        negative_horizontal = (negative_horizontal << 1) & mask
        positive = (negative_horizontal | ~(vertical | positive_horizontal)) & mask
        negative = (positive_horizontal & vertical) & mask
    return distance


def _correctness_safe_basename(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or os.path.basename(value) != value
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None
    ):
        raise RowError(f"{label} basename is unsafe")
    return value


def _validate_correctness_receipt_payload(
    payload: object,
    *,
    reference_bytes: bytes,
    hypothesis_runs: dict[str, bytes],
    focr: dict,
    ref: dict,
) -> dict:
    if not isinstance(reference_bytes, bytes) or not isinstance(hypothesis_runs, dict):
        raise RowError("correctness physical inputs are malformed")
    if not all(
        isinstance(run_id, str) and isinstance(content, bytes)
        for run_id, content in hypothesis_runs.items()
    ):
        raise RowError("correctness hypothesis inputs are malformed")
    receipt = _correctness_exact_fields(
        payload, CORRECTNESS_TOP_FIELDS, "correctness receipt"
    )
    if receipt.get("schema") != CORRECTNESS_RECEIPT_SCHEMA:
        raise RowError("correctness receipt schema is unsupported")
    if receipt.get("normalization") != CORRECTNESS_NORMALIZATION:
        raise RowError("correctness receipt normalization is unsupported")
    if receipt.get("metric_formulas") != CORRECTNESS_METRIC_FORMULAS:
        raise RowError("correctness receipt metric formulas are noncanonical")
    created = receipt.get("created_utc")
    try:
        parsed_created = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
    except ValueError as error:
        raise RowError("correctness receipt created_utc is invalid") from error
    if (
        not isinstance(created, str)
        or not created.endswith("Z")
        or parsed_created.tzinfo is None
        or parsed_created.utcoffset() != timezone.utc.utcoffset(parsed_created)
    ):
        raise RowError("correctness receipt created_utc must be UTC")

    pages = receipt.get("pages")
    if not isinstance(pages, list) or len(pages) != 1:
        raise RowError("correctness receipt must contain exactly one page")
    page = _correctness_exact_fields(pages[0], CORRECTNESS_PAGE_FIELDS, "correctness page")
    if page.get("status") != "OK":
        raise RowError("correctness receipt page is not complete")
    reference = _correctness_exact_fields(
        page.get("reference"), {"source", "transform", "scored"}, "reference"
    )
    hypothesis = _correctness_exact_fields(
        page.get("hypothesis"), {"source", "transform", "scored"}, "hypothesis"
    )
    reference_source = _correctness_exact_fields(
        reference.get("source"),
        CORRECTNESS_SOURCE_BINDING_FIELDS,
        "reference source binding",
    )
    hypothesis_source = _correctness_exact_fields(
        hypothesis.get("source"),
        CORRECTNESS_SOURCE_BINDING_FIELDS,
        "hypothesis source binding",
    )
    _correctness_validate_binding_types(reference_source, "reference source binding")
    _correctness_validate_binding_types(hypothesis_source, "hypothesis source binding")
    reference_basename = _correctness_safe_basename(
        reference_source.get("basename"), "reference source"
    )
    hypothesis_basename = _correctness_safe_basename(
        hypothesis_source.get("basename"), "hypothesis source"
    )
    if page.get("page") != reference_basename:
        raise RowError("correctness page identity does not match the reference basename")
    reference_page = os.path.basename(str(ref.get("page") or ""))
    expected_reference_basename = os.path.splitext(reference_page)[0] + ".md"
    if not reference_page or reference_basename != expected_reference_basename:
        raise RowError("correctness reference basename does not match the timed page")

    reference_expected = _correctness_binding(
        reference_bytes, basename=reference_basename
    )
    if reference_source != reference_expected:
        raise RowError("correctness reference source binding does not match physical bytes")
    if not hypothesis_runs:
        raise RowError("correctness evidence has no timed hypothesis outputs")
    if len(hypothesis_runs) > MAX_RAW_TIMING_RUNS:
        raise RowError("correctness evidence has too many timed hypothesis outputs")
    if len(reference_bytes) + sum(map(len, hypothesis_runs.values())) > MAX_RAW_TIMING_TOTAL_BYTES:
        raise RowError("correctness source bytes exceed the total evidence bound")
    expected_run_ids = [f"run_{index:03d}" for index in range(1, len(hypothesis_runs) + 1)]
    if list(hypothesis_runs) != expected_run_ids:
        raise RowError("correctness hypothesis run ids are noncanonical")
    if hypothesis_basename != "run_001.stdout":
        raise RowError("correctness hypothesis must bind the first measured stdout")
    hypothesis_bytes = hypothesis_runs["run_001"]
    if any(content != hypothesis_bytes for content in hypothesis_runs.values()):
        raise RowError("correctness hypothesis outputs drift across timed runs")
    hypothesis_expected = _correctness_binding(
        hypothesis_bytes, basename=hypothesis_basename
    )
    if hypothesis_source != hypothesis_expected:
        raise RowError("correctness hypothesis source binding does not match physical bytes")

    focr_raw_timing = focr.get("raw_timing")
    ref_raw_timing = ref.get("raw_timing")
    focr_records = (
        focr_raw_timing.get("records") if isinstance(focr_raw_timing, dict) else None
    )
    ref_records = (
        ref_raw_timing.get("records") if isinstance(ref_raw_timing, dict) else None
    )
    if (
        not isinstance(focr_records, list)
        or len(focr_records) != len(hypothesis_runs)
        or not isinstance(ref_records, list)
        or len(ref_records) != len(hypothesis_runs)
    ):
        raise RowError("correctness receipt run count does not match timing evidence")
    reference_sha = reference_expected["sha256"]
    for run_id, content, focr_record, ref_record in zip(
        expected_run_ids,
        hypothesis_runs.values(),
        focr_records,
        ref_records,
        strict=True,
    ):
        if not isinstance(focr_record, dict) or not isinstance(ref_record, dict):
            raise RowError("correctness timing record is malformed")
        raw_files = focr_record.get("raw_files")
        stdout = raw_files.get("stdout") if isinstance(raw_files, dict) else None
        if (
            focr_record.get("run_id") != run_id
            or not isinstance(stdout, dict)
            or stdout.get("path") != f"{run_id}.stdout"
            or stdout.get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise RowError("correctness hypothesis is not bound by focr raw timing")
        if (
            ref_record.get("run_id") != run_id
            or ref_record.get("text_sha256") != reference_sha
        ):
            raise RowError("correctness reference is not bound by per-run timing hashes")
    if (
        ref.get("text_sha256") != reference_sha
        or ref.get("text_identical_across_runs") is not True
        or focr.get("stdout_identical_across_runs") is not True
    ):
        raise RowError("correctness source determinism is not top-level bound")

    reference_transform = _correctness_exact_fields(
        reference.get("transform"), {"name", "matches"}, "reference transform"
    )
    hypothesis_transform = _correctness_exact_fields(
        hypothesis.get("transform"), {"name", "matches"}, "hypothesis transform"
    )
    _correctness_int(reference_transform.get("matches"), "reference transform matches")
    _correctness_int(hypothesis_transform.get("matches"), "hypothesis transform matches")
    reference_text = reference_bytes.decode("utf-8")
    hypothesis_text = hypothesis_bytes.decode("utf-8")
    if (
        len(reference_text) > MAX_CORRECTNESS_TEXT_CHARS
        or len(hypothesis_text) > MAX_CORRECTNESS_TEXT_CHARS
    ):
        raise RowError("correctness text exceeds the exact-distance character bound")
    scored_reference, matches = _correctness_transform_reference(reference_text)
    if reference_transform != {
        "name": CORRECTNESS_REFERENCE_TRANSFORM,
        "matches": matches,
    }:
        raise RowError("correctness reference transform is not exact")
    if hypothesis_transform != {
        "name": CORRECTNESS_IDENTITY_TRANSFORM,
        "matches": 0,
    }:
        raise RowError("correctness hypothesis transform is not identity")
    if not scored_reference or not _correctness_normalize(scored_reference):
        raise RowError("correctness reference is empty after transformation")
    reference_scored = _correctness_exact_fields(
        reference.get("scored"),
        CORRECTNESS_SCORED_BINDING_FIELDS,
        "scored reference binding",
    )
    hypothesis_scored = _correctness_exact_fields(
        hypothesis.get("scored"),
        CORRECTNESS_SCORED_BINDING_FIELDS,
        "scored hypothesis binding",
    )
    _correctness_validate_binding_types(reference_scored, "scored reference binding")
    _correctness_validate_binding_types(hypothesis_scored, "scored hypothesis binding")
    if reference_scored != _correctness_binding(scored_reference.encode("utf-8")):
        raise RowError("correctness scored reference binding is invalid")
    if hypothesis_scored != _correctness_binding(hypothesis_bytes):
        raise RowError("correctness scored hypothesis binding is invalid")

    normalized_reference = _correctness_normalize(scored_reference)
    normalized_hypothesis = _correctness_normalize(hypothesis_text)
    raw_distance = _correctness_levenshtein(scored_reference, hypothesis_text)
    normalized_distance = _correctness_levenshtein(
        normalized_reference, normalized_hypothesis
    )
    raw_chars = len(scored_reference)
    normalized_chars = len(normalized_reference)
    expected_page = {
        "raw_edit_distance": raw_distance,
        "normalized_edit_distance": normalized_distance,
        "ref_chars": raw_chars,
        "hyp_chars": len(hypothesis_text),
        "exact": scored_reference == hypothesis_text,
        "exact_norm": normalized_reference == normalized_hypothesis,
    }
    for field in (
        "raw_edit_distance",
        "normalized_edit_distance",
        "ref_chars",
        "hyp_chars",
    ):
        if _correctness_int(page.get(field), f"correctness page {field}") != expected_page[field]:
            raise RowError(f"correctness page {field} is not source-derived")
    for field in ("exact", "exact_norm"):
        if not isinstance(page.get(field), bool) or page[field] is not expected_page[field]:
            raise RowError(f"correctness page {field} is not source-derived")
    raw_cer = raw_distance / max(1, raw_chars)
    normalized_cer = normalized_distance / max(1, normalized_chars)
    _correctness_float(page.get("cer_raw"), raw_cer, "correctness page cer_raw")
    _correctness_float(page.get("cer_norm"), normalized_cer, "correctness page cer_norm")

    aggregate = _correctness_exact_fields(
        receipt.get("aggregate"),
        CORRECTNESS_AGGREGATE_FIELDS,
        "correctness aggregate",
    )
    expected_aggregate = {
        "pages_total": 1,
        "pages_with_hyp": 1,
        "exact_raw": int(scored_reference == hypothesis_text),
        "exact_norm": int(normalized_reference == normalized_hypothesis),
        "raw_edit_distance": raw_distance,
        "raw_reference_chars": raw_chars,
        "normalized_edit_distance": normalized_distance,
        "normalized_reference_chars": normalized_chars,
        "reference_source_bytes": len(reference_bytes),
        "reference_source_chars": len(reference_text),
        "hypothesis_source_bytes": len(hypothesis_bytes),
        "hypothesis_source_chars": len(hypothesis_text),
    }
    for field, expected in expected_aggregate.items():
        if _correctness_int(aggregate.get(field), f"correctness aggregate {field}") != expected:
            raise RowError(f"correctness aggregate {field} is not page-derived")
    _correctness_float(aggregate.get("cer_raw"), raw_cer, "correctness aggregate cer_raw")
    _correctness_float(
        aggregate.get("cer_norm"), normalized_cer, "correctness aggregate cer_norm"
    )
    return {
        "payload": receipt,
        "reference_basename": reference_basename,
        "reference_bytes": reference_bytes,
        "hypothesis_runs": hypothesis_runs,
        "cer_norm": normalized_cer,
    }


def validate_correctness_proof(
    value: str, *, focr: dict, ref: dict, ref_stages_path: str
) -> dict:
    cer_matches = _CORRECTNESS_CER_RE.findall(value)
    receipt_paths = _CORRECTNESS_RECEIPT_PATH_RE.findall(value)
    if len(cer_matches) != 1 or len(receipt_paths) != 1:
        raise RowError(
            "correctness_proof must name exactly one CER_norm value and one cer.json receipt path"
        )
    claimed_cer = float(cer_matches[0])
    receipt_path = os.path.abspath(receipt_paths[0])
    receipt_bytes = _read_bounded_file(receipt_path)
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise RowError(f"correctness receipt is unreadable: {error}") from error
    try:
        reference_source = receipt["pages"][0]["reference"]["source"]
        reference_basename = _correctness_safe_basename(
            reference_source["basename"], "reference source"
        )
    except (KeyError, IndexError, TypeError) as error:
        raise RowError("correctness receipt lacks a reference source binding") from error
    reference_path = os.path.join(
        os.path.dirname(os.path.abspath(ref_stages_path)), "text", reference_basename
    )
    reference_bytes = _read_bounded_file(reference_path)
    run_dir = focr.get("run_dir")
    raw_timing = focr.get("raw_timing")
    records = raw_timing.get("records") if isinstance(raw_timing, dict) else None
    if not isinstance(run_dir, str) or not isinstance(records, list):
        raise RowError("focr timing evidence cannot locate hypothesis outputs")
    hypothesis_runs: dict[str, bytes] = {}
    for index, record in enumerate(records, 1):
        run_id = f"run_{index:03d}"
        if not isinstance(record, dict) or record.get("run_id") != run_id:
            raise RowError("focr timing run ids are noncanonical")
        hypothesis_runs[run_id] = _read_bounded_file(
            os.path.join(run_dir, f"{run_id}.stdout")
        )
    verified = _validate_correctness_receipt_payload(
        receipt,
        reference_bytes=reference_bytes,
        hypothesis_runs=hypothesis_runs,
        focr=focr,
        ref=ref,
    )
    measured_cer = verified["cer_norm"]
    if (
        not 0.0 <= measured_cer <= MAX_CORRECTNESS_CER
        or not math.isclose(claimed_cer, measured_cer, abs_tol=5e-6)
    ):
        raise RowError("correctness receipt does not prove an in-budget CER run")
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    return {
        **verified,
        "source_proof": value,
        "path": receipt_path,
        "receipt_bytes": receipt_bytes,
        "sha256": digest,
        "canonical": (
            f"receipt=correctness_receipt.json sha256={digest} "
            f"metric=cer_norm value={measured_cer:.6f} "
            f"max={MAX_CORRECTNESS_CER:.6f} result=pass"
        ),
    }


def sanitize_cell(value: str) -> str:
    """Markdown table cells cannot carry '|' or newlines (the checker splits
    on '|'); substitute rather than silently breaking the row shape."""
    return " ".join(str(value).replace("|", "¦").split())


def stage_record(doc: dict, stage: str) -> dict | None:
    records = doc.get("stages")
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and record.get("stage") == stage:
            return record
    return None


def ledger_stage_name(stage: str) -> str:
    return stage.replace("_", "-")


def _recomputed_stats(samples: list[float]) -> dict[str, float | int | list[float]]:
    mean = statistics.fmean(samples)
    cv = statistics.stdev(samples) / mean * 100.0
    return {
        "samples_ms": [round(sample, 6) for sample in samples],
        "best_ms": round(min(samples), 6),
        "p50_ms": round(statistics.median(samples), 6),
        "mean_ms": round(mean, 6),
        "cv_pct": round(cv, 3),
        "n": len(samples),
    }


def validate_raw_timing(doc: dict, record: dict, *, source: str) -> None:
    """Recompute one aggregate record from its bounded per-invocation inputs."""
    raw = doc.get("raw_timing")
    if not isinstance(raw, dict):
        raise RowError(f"{source} document lacks versioned raw timing observations")
    if set(raw) != {"schema", "source", "unit", "measured_runs", "records"}:
        raise RowError(f"{source} raw timing has noncanonical fields")
    records = raw.get("records")
    measured_runs = raw.get("measured_runs")
    if (
        raw.get("schema") != RAW_TIMING_SCHEMA
        or raw.get("source") != source
        or raw.get("unit") != "ms"
        or not isinstance(records, list)
        or not isinstance(measured_runs, int)
        or isinstance(measured_runs, bool)
        or not 2 <= measured_runs <= MAX_RAW_TIMING_RUNS
        or len(records) != measured_runs
        or doc.get("runs") != measured_runs
    ):
        raise RowError(f"{source} raw timing identity/count is invalid")

    stage = record.get("stage")
    expected_ids = [f"run_{index:03d}" for index in range(1, measured_runs + 1)]
    samples: list[float] = []
    tokens: list[int | None] = []
    text_hashes: list[str] = []
    for expected_id, raw_record in zip(expected_ids, records, strict=True):
        if not isinstance(raw_record, dict):
            raise RowError(f"{source} raw timing record is not an object")
        expected_record_fields = (
            {"run_id", "stages", "raw_files"}
            if source == "focr"
            else {"run_id", "stages", "text_sha256"}
        )
        if set(raw_record) != expected_record_fields or raw_record.get(
            "run_id"
        ) != expected_id:
            raise RowError(f"{source} raw timing run ids/fields are noncanonical")
        stages = raw_record.get("stages")
        if (
            not isinstance(stages, dict)
            or not stages
            or len(stages) > MAX_RAW_TIMING_STAGES
            or not all(isinstance(name, str) and name for name in stages)
        ):
            raise RowError(f"{source} raw timing stage map is invalid")
        sample = stages.get(stage)
        if not isinstance(sample, dict) or set(sample) not in (
            {"ms"},
            {"ms", "tokens"},
        ):
            raise RowError(f"{source} raw timing lacks stage {stage!r}")
        value = sample.get("ms")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise RowError(f"{source} raw timing sample is invalid")
        token_count = sample.get("tokens")
        if token_count is not None and (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
        ):
            raise RowError(f"{source} raw timing token count is invalid")
        samples.append(float(value))
        tokens.append(token_count)

        if source == "reference":
            text_sha = raw_record.get("text_sha256")
            if (
                not isinstance(text_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", text_sha) is None
            ):
                raise RowError("reference raw timing lacks a valid per-run text hash")
            text_hashes.append(text_sha)

        if source == "focr":
            raw_files = raw_record.get("raw_files")
            if not isinstance(raw_files, dict) or set(raw_files) != {
                "meta",
                "stderr",
                "stdout",
            }:
                raise RowError("focr raw timing lacks exact file bindings")
            for kind, binding in raw_files.items():
                expected_suffix = ".meta.json" if kind == "meta" else f".{kind}"
                if (
                    not isinstance(binding, dict)
                    or set(binding) != {"path", "sha256"}
                    or binding.get("path") != expected_id + expected_suffix
                    or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", "")))
                    is None
                ):
                    raise RowError("focr raw timing has a malformed file binding")

    try:
        recomputed = _recomputed_stats(samples)
    except (OverflowError, ValueError, statistics.StatisticsError) as error:
        raise RowError(f"{source} raw timing cannot be summarized") from error
    for field, expected in recomputed.items():
        actual = record.get(field)
        if field == "samples_ms":
            if not isinstance(actual, list) or len(actual) != len(expected):
                raise RowError(f"{source} aggregate samples are not raw-derived")
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isclose(float(value), expected_value, abs_tol=5e-7)
                for value, expected_value in zip(actual, expected, strict=True)
            ):
                raise RowError(f"{source} aggregate samples are not raw-derived")
        elif field == "n":
            if actual != expected:
                raise RowError(f"{source} aggregate n is not raw-derived")
        elif (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isclose(float(actual), float(expected), abs_tol=5e-7)
        ):
            raise RowError(f"{source} aggregate {field} is not raw-derived")

    if any(token is not None for token in tokens):
        if any(token is None for token in tokens) or len(set(tokens)) != 1:
            raise RowError(f"{source} raw timing token counts drift across runs")
        if record.get("tokens") != tokens[0]:
            raise RowError(f"{source} aggregate token count is not raw-derived")
    if source == "reference" and (
        len(text_hashes) != measured_runs
        or len(set(text_hashes)) != 1
        or doc.get("text_sha256") != text_hashes[0]
        or doc.get("text_identical_across_runs") is not True
    ):
        raise RowError(
            "reference text hashes are missing, invalid, drifting, or not top-level bound"
        )


def validate_inputs(
    focr: dict,
    ref: dict,
    stage: str,
    fixture_hash: str,
    *,
    allow_synthetic: bool,
) -> tuple[dict, dict]:
    for name, doc in (("focr", focr), ("reference", ref)):
        if doc.get("synthetic") is not False and not allow_synthetic:
            raise RowError(
                f"{name} input is synthetic or unstamped — it can never land"
            )
    if focr.get("source") != "focr" or ref.get("source") != "reference":
        raise RowError("focr/reference inputs swapped or mislabeled (`source` field)")

    # Same-input pairing (fresh-eyes fix): both measurement docs carry `page`;
    # a ratio between two DIFFERENT pages is meaningless and previously landed
    # silently if the operator passed mismatched files.
    fpage = os.path.basename(str(focr.get("page") or ""))
    rpage = os.path.basename(str(ref.get("page") or ""))
    if not fpage or not rpage:
        raise RowError("focr/reference measurement docs must both carry `page`")
    if fpage != rpage:
        raise RowError(
            f"page pairing broken: focr measured {fpage!r}, reference {rpage!r}"
        )
    fixture_match = re.search(
        r"(?:^|;\s*)page=(?P<page>[^\s;]+)\s+sha256=(?P<sha>[0-9a-f]{64})(?:;|\s|$)",
        fixture_hash,
    )
    if fixture_match is None:
        raise RowError(
            "fixture_hash must bind page=<basename> sha256=<64 lowercase hex>"
        )
    if fixture_match.group("page") != fpage:
        raise RowError("fixture_hash page does not match both measurement inputs")
    if any(doc.get("page_sha256") != fixture_match.group("sha") for doc in (focr, ref)):
        raise RowError("focr/reference page_sha256 does not match fixture_hash")

    frec = stage_record(focr, stage)
    rrec = stage_record(ref, stage)
    if frec is None:
        raise RowError(f"focr measurements carry no stage {stage!r}")
    if rrec is None:
        raise RowError(f"reference measurements carry no stage {stage!r}")
    validate_reference_contract(ref, rrec)

    for side, doc, rec in (("focr", focr, frec), ("reference", ref, rrec)):
        expected_source = "focr" if side == "focr" else "reference"
        if (
            rec.get("schema") != "focr-gauntlet-stage/v1"
            or rec.get("source") != expected_source
            or (rec.get("synthetic") is not False and not allow_synthetic)
            or rec.get("unit") != "ms"
            or rec.get("ledger_stage") is not True
        ):
            raise RowError(f"{side} {stage}: record is not ledger-eligible")
        samples = rec.get("samples_ms")
        n = rec.get("n")
        if (
            not isinstance(samples, list)
            or len(samples) < 2
            or not isinstance(n, int)
            or isinstance(n, bool)
            or n != len(samples)
        ):
            raise RowError(f"{side} {stage}: best-of-N samples are mandatory")
        try:
            numeric_samples = [float(sample) for sample in samples]
            best_ms = float(rec["best_ms"])
            cv = float(rec["cv_pct"])
        except (KeyError, TypeError, ValueError) as error:
            raise RowError(f"{side} {stage}: invalid timing evidence") from error
        recomputed_cv = round(
            statistics.stdev(numeric_samples)
            / statistics.fmean(numeric_samples)
            * 100.0,
            3,
        )
        if (
            not all(
                math.isfinite(sample) and sample > 0.0 for sample in numeric_samples
            )
            or not math.isclose(best_ms, min(numeric_samples), rel_tol=1e-9)
            or not math.isfinite(cv)
            or not 0.0 <= cv <= MAX_CV_PCT
            or not math.isclose(cv, recomputed_cv, abs_tol=0.001)
        ):
            raise RowError(
                f"{side} {stage}: timings/CV are invalid or not sample-derived"
            )
        if rec.get("warmup_discarded", 0) < 1:
            raise RowError(f"{side} {stage}: warmup was not discarded")
        if not rec.get("precision"):
            raise RowError(f"{side} {stage}: missing precision annotation")
        if not rec.get("backend") or not rec.get("allocator"):
            raise RowError(f"{side} {stage}: missing backend/allocator identity")
        validate_raw_timing(doc, rec, source=expected_source)

    if frec.get("threads") != rrec.get("threads"):
        raise RowError(
            f"thread parity broken: focr={frec.get('threads')} ref={rrec.get('threads')}"
        )
    proof = rrec.get("thread_proof") or {}
    if proof.get("torch_num_threads") != rrec.get("threads"):
        raise RowError(
            "reference record lacks a matching torch thread proof — rejected"
        )
    if frec.get("allocator") != rrec.get("allocator"):
        raise RowError(
            f"allocator mismatch: focr={frec.get('allocator')} ref={rrec.get('allocator')}"
        )
    if focr.get("stdout_identical_across_runs") is not True:
        raise RowError(
            "focr stdout differed across runs — nondeterministic evidence rejected"
        )
    if frec.get("tokens") is not None and frec.get("tokens_consistent") is False:
        raise RowError(f"focr {stage}: token count drifted across runs — rejected")
    if stage == "decode_per_token" and frec.get("tokens_consistent") is not True:
        raise RowError("focr decode evidence lacks a positive token-consistency proof")
    return frec, rrec


def roofline_floor(roofline: dict, stage: str, *, allow_synthetic: bool) -> dict:
    if roofline.get("synthetic") is not False and not allow_synthetic:
        raise RowError("roofline is synthetic or unstamped - rejected")
    for floor in roofline.get("floors", []):
        if floor.get("stage") == stage:
            if not floor.get("floor_ms") or not floor.get("floor_kind"):
                raise RowError(f"roofline floor for {stage!r} is incomplete")
            return floor
    raise RowError(f"roofline carries no floor for stage {stage!r}")


def build_row(
    *,
    focr: dict,
    ref: dict,
    roofline: dict,
    stage: str,
    claim_id: str,
    evidence_id: str,
    fixture_hash: str,
    arch_features: str,
    correctness_proof: str,
    model_commit: str,
    notes: str,
    focr_stages_sha256: str,
    correctness: dict,
    allow_synthetic: bool = False,
) -> dict:
    if (
        correctness.get("path") is None
        or correctness.get("canonical") is None
        or correctness.get("source_proof") != correctness_proof
        or not isinstance(correctness.get("receipt_bytes"), bytes)
    ):
        raise RowError("build_row requires a fully validated correctness receipt")
    frec, rrec = validate_inputs(
        focr,
        ref,
        stage,
        fixture_hash,
        allow_synthetic=allow_synthetic,
    )
    reference_entry = _command_option(ref.get("command"), "--entry", required=True)
    if reference_entry == CURRENT_UNLIMITED_REFERENCE_ENTRY:
        validate_current_unlimited_release_contract(
            focr,
            roofline,
            frec,
            focr_stages_sha256=focr_stages_sha256,
        )
    floor = roofline_floor(roofline, stage, allow_synthetic=allow_synthetic)

    focr_ms = float(frec["best_ms"])
    ref_ms = float(rrec["best_ms"])
    floor_ms = float(floor["floor_ms"])
    kill = (
        _canonical_kill_switch_evidence(focr)
        if reference_entry == CURRENT_UNLIMITED_REFERENCE_ENTRY
        else (focr.get("focr_env") or {})
    )
    kill_cell = " ".join(f"{k}={v}" for k, v in sorted(kill.items()))
    if not kill_cell:
        raise RowError(
            "focr run meta carries no FOCR_* env evidence (kill-switch state)"
        )

    command_cell = (
        f"focr: {' '.join(map(str, focr.get('command') or []))} "
        f"[threads={frec['threads']} FOCR_TIMING=1]; "
        f"ref: {' '.join(map(str, ref.get('command') or []))} "
        f"[torch=={ref.get('torch_version', '?')} transformers=={ref.get('transformers_version', '?')}]"
    )
    if reference_entry == CURRENT_UNLIMITED_REFERENCE_ENTRY:
        command_cell += (
            f" [decode_mode={CURRENT_UNLIMITED_DECODE_MODE} "
            f"quant_recipe={CURRENT_UNLIMITED_QUANT_RECIPE} "
            f"model_sha256={CURRENT_UNLIMITED_MODEL_SHA256} "
            f"model_size={CURRENT_UNLIMITED_MODEL_SIZE}]"
        )
    date = (focr.get("created_utc") or "")[:10]
    if not date:
        raise RowError("focr measurement file has no created_utc date")

    cells = {
        "date": date,
        "claim_id": claim_id,
        "evidence_id": evidence_id,
        "model_commit": model_commit,
        "fixture_hash": fixture_hash,
        "arch/cpu_features": arch_features,
        "stage": ledger_stage_name(stage),
        "reference_backend": rrec["backend"],
        "focr_ms": f"{focr_ms:.3f}",
        "ref_ms": f"{ref_ms:.3f}",
        "ratio": f"{ref_ms / focr_ms:.3f}",
        "floor_kind": floor["floor_kind"],
        "floor_ms": f"{floor_ms:.3f}",
        "dist_above_floor": f"{focr_ms / floor_ms:.2f}",
        "precision (focr vs ref)": f"{frec['precision']} vs {rrec['backend']}-{rrec['precision']}",
        "threads (focr=ref N)": f"focr=ref={frec['threads']}",
        "allocator": frec["allocator"],
        "command/env": command_cell,
        "fallback/kill-switch state": kill_cell,
        "correctness_proof": correctness["canonical"],
        "notes": notes
        or f"best-of-{frec['n']} (cv {frec['cv_pct']}% / ref {rrec['cv_pct']}%)",
    }
    return {name: sanitize_cell(cells[name]) for name in COLUMNS}


def row_markdown(row: dict) -> str:
    return "| " + " | ".join(row[name] for name in COLUMNS) + " |"


# ── evidence bundle ──────────────────────────────────────────────────────────


def validate_evidence_destination(
    evidence_dir: str | os.PathLike[str], *, perf_root: Path = PERF_ROOT
) -> Path:
    """Resolve and validate an unused bundle destination under artifacts/perf."""
    lexical_root = Path(os.path.abspath(perf_root))
    requested = Path(evidence_dir).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    requested = Path(os.path.abspath(requested))
    try:
        resolved_root = lexical_root.resolve()
        resolved = requested.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise RowError(
            f"cannot resolve evidence destination {requested}: {error}"
        ) from error

    try:
        relative = resolved.relative_to(resolved_root)
        lexical_relative = requested.relative_to(lexical_root)
    except ValueError as error:
        raise RowError(
            f"evidence dir must resolve under {resolved_root}, got {resolved}"
        ) from error
    if not relative.parts or not lexical_relative.parts:
        raise RowError("evidence dir must be a child of artifacts/perf, not the root")

    component = Path(requested.anchor)
    for part in requested.parts[1:]:
        component /= part
        if component.is_symlink():
            raise RowError(f"evidence dir contains symlink component: {component}")

    if resolved.exists():
        if not resolved.is_dir():
            raise RowError(f"evidence destination is not a directory: {resolved}")
        try:
            nonempty = next(resolved.iterdir(), None) is not None
        except OSError as error:
            raise RowError(
                f"cannot inspect evidence destination {resolved}: {error}"
            ) from error
        if nonempty:
            raise RowError(f"evidence destination must be empty: {resolved}")
    return resolved


def _copy_bound_binary(source: str, destination: str, expected: dict) -> None:
    source = _reject_symlink_components(source)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_SUBJECT_BINARY_BYTES:
            raise RowError("subject binary is not a bounded regular file")
        digest = hashlib.sha256()
        observed = 0
        with open(destination, "xb") as output:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
                if observed > MAX_SUBJECT_BINARY_BYTES:
                    raise RowError("subject binary grew beyond the copy bound")
                digest.update(chunk)
                output.write(chunk)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise RowError("subject binary changed while being bundled")
        actual = {"sha256": digest.hexdigest(), "size": observed}
        if actual != expected or _stable_file_identity(destination, MAX_SUBJECT_BINARY_BYTES) != expected:
            raise RowError("bundled subject binary does not match its build receipt")
    finally:
        os.close(descriptor)


def _canonical_pack_json(value: dict, label: str) -> bytes:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise RowError(f"cannot canonicalize {label}: {error}") from error
    if not encoded or len(encoded) > MAX_SOURCE_PACK_HEADER_BYTES:
        raise RowError(f"{label} exceeds the source-pack header bound")
    return encoded


def _write_source_pack(destination: str, build_provenance: dict) -> dict:
    manifest = build_provenance.get("manifest")
    physical_entries = build_provenance.get("physical_entries")
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if (
        not isinstance(entries, list)
        or not isinstance(physical_entries, list)
        or len(entries) != len(physical_entries)
        or not entries
        or len(entries) > MAX_SOURCE_ENTRIES
    ):
        raise RowError("source pack requires the complete validated manifest closure")
    total_content_bytes = sum(entry["size"] for entry in entries)
    if total_content_bytes > MAX_SOURCE_TOTAL_BYTES:
        raise RowError("source pack content exceeds the total-byte bound")
    header = _canonical_pack_json(
        {
            "schema": SOURCE_PACK_SCHEMA,
            "root_sha256": manifest["root_sha256"],
            "entry_count": len(entries),
            "total_content_bytes": total_content_bytes,
        },
        "source-pack header",
    )
    with open(destination, "xb") as output:
        output.write(SOURCE_PACK_DOMAIN)
        output.write(len(header).to_bytes(8, "big"))
        output.write(header)
        for expected, physical_record in zip(entries, physical_entries, strict=True):
            if (
                not isinstance(physical_record, dict)
                or set(physical_record) != {"entry", "physical_path"}
                or physical_record.get("entry") != expected
                or not isinstance(physical_record.get("physical_path"), str)
            ):
                raise RowError("source pack physical closure is out of manifest order")
            metadata = _canonical_pack_json(expected, "source-pack record")
            output.write(SOURCE_PACK_RECORD_DOMAIN)
            output.write(len(metadata).to_bytes(8, "big"))
            output.write(metadata)
            output.write(expected["size"].to_bytes(8, "big"))

            source = _reject_symlink_components(physical_record["physical_path"])
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(source, flags)
            except OSError as error:
                raise RowError(
                    f"cannot open source-pack input "
                    f"{expected['repository']}/{expected['path']}: {error}"
                ) from error
            try:
                try:
                    before = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or before.st_size != expected["size"]
                        or before.st_size > MAX_SOURCE_FILE_BYTES
                    ):
                        raise RowError(
                            f"source-pack input size/type drifted: "
                            f"{expected['repository']}/{expected['path']}"
                        )
                    digest = hashlib.sha256()
                    remaining = expected["size"]
                    while remaining:
                        chunk = os.read(descriptor, min(1024 * 1024, remaining))
                        if not chunk:
                            raise RowError(
                                f"source-pack input truncated: "
                                f"{expected['repository']}/{expected['path']}"
                            )
                        output.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                    if os.read(descriptor, 1):
                        raise RowError(
                            f"source-pack input grew: "
                            f"{expected['repository']}/{expected['path']}"
                        )
                    after = os.fstat(descriptor)
                    stable = (
                        "st_dev",
                        "st_ino",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                    if any(
                        getattr(before, field) != getattr(after, field)
                        for field in stable
                    ):
                        raise RowError(
                            f"source-pack input changed while reading: "
                            f"{expected['repository']}/{expected['path']}"
                        )
                    if digest.hexdigest() != expected["sha256"]:
                        raise RowError(
                            f"source-pack input hash drifted: "
                            f"{expected['repository']}/{expected['path']}"
                        )
                except OSError as error:
                    raise RowError(
                        f"cannot read source-pack input "
                        f"{expected['repository']}/{expected['path']}: {error}"
                    ) from error
            finally:
                os.close(descriptor)
        output.write(SOURCE_PACK_TRAILER_DOMAIN)
        output.write(len(entries).to_bytes(8, "big"))
        output.write(total_content_bytes.to_bytes(8, "big"))
        output.write(bytes.fromhex(manifest["root_sha256"]))
    identity = _stable_file_identity(destination, MAX_SOURCE_PACK_BYTES)
    if identity["size"] > MAX_SOURCE_PACK_BYTES:
        raise RowError("source pack exceeds its final size bound")
    return identity


def write_bundle(
    evidence_dir: str,
    *,
    focr_path: str,
    ref_path: str,
    roofline_path: str,
    correctness: dict,
    focr: dict,
    ref: dict,
    build_provenance: dict,
    reference_provenance: dict,
    rows: list[dict],
    source_git_head: str,
    source_root: str,
    producer_root: str,
    allowed_evidence_path: str,
    perf_root: Path = PERF_ROOT,
) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", source_git_head) is None:
        raise RowError("write_bundle requires a canonical 40-hex source_git_head")
    if re.fullmatch(r"[0-9a-f]{64}", source_root) is None:
        raise RowError("write_bundle requires a canonical source_root")
    if re.fullmatch(r"[0-9a-f]{64}", producer_root) is None:
        raise RowError("write_bundle requires a canonical producer_root")
    allowed_evidence_path = _canonical_allowed_evidence_path(allowed_evidence_path)
    evidence_dir = os.fspath(
        validate_evidence_destination(evidence_dir, perf_root=perf_root)
    )
    os.makedirs(evidence_dir, exist_ok=True)
    if build_provenance["receipt"].get("git_head") != source_git_head:
        raise RowError("build receipt HEAD changed before bundling")
    if build_provenance["manifest"].get("root_sha256") != source_root:
        raise RowError("source manifest root changed before bundling")
    inputs = {}
    for key, src, bundle_name in (
        ("focr_stages", focr_path, "focr_stages.json"),
        ("ref_stages", ref_path, "ref_stages.json"),
        ("roofline", roofline_path, "roofline.json"),
        (
            "correctness_receipt",
            correctness["path"],
            "correctness_receipt.json",
        ),
    ):
        destination = os.path.join(evidence_dir, bundle_name)
        if key == "correctness_receipt":
            with open(destination, "xb") as handle:
                handle.write(correctness["receipt_bytes"])
        else:
            content = _read_bounded_file(src)
            with open(destination, "xb") as handle:
                handle.write(content)
        inputs[key] = {
            "bundle_path": bundle_name,
            "sha256": sha256_file(destination),
            "size": os.path.getsize(destination),
        }
    if inputs["correctness_receipt"]["sha256"] != correctness["sha256"]:
        raise RowError("correctness receipt changed between validation and bundling")

    provenance_files = (
        (
            "build_receipt",
            "subject/build_receipt.json",
            build_provenance["receipt_bytes"],
            build_provenance["receipt_identity"],
        ),
        (
            "source_input_manifest",
            "subject/source_input_manifest.json",
            build_provenance["manifest_bytes"],
            build_provenance["manifest_identity"],
        ),
        (
            "reference_model_manifest",
            "reference_model_manifest.json",
            reference_provenance["manifest_bytes"],
            reference_provenance["manifest_identity"],
        ),
        (
            "reference_inference_binding",
            "reference_inference_binding.json",
            reference_provenance["binding_bytes"],
            reference_provenance["binding_identity"],
        ),
    )
    for key, bundle_name, content, expected_identity in provenance_files:
        destination = os.path.join(evidence_dir, bundle_name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "xb") as handle:
            handle.write(content)
        identity = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        if identity != expected_identity:
            raise RowError(f"{key} changed between validation and bundling")
        inputs[key] = {"bundle_path": bundle_name, **identity}
    binary_bundle_path = "subject/release-perf/focr"
    binary_destination = os.path.join(evidence_dir, binary_bundle_path)
    os.makedirs(os.path.dirname(binary_destination), exist_ok=True)
    _copy_bound_binary(
        build_provenance["binary_path"],
        binary_destination,
        build_provenance["binary_identity"],
    )
    inputs["subject_binary"] = {
        "bundle_path": binary_bundle_path,
        **build_provenance["binary_identity"],
    }
    source_pack_bundle_path = "subject/source_input_pack.bin"
    source_pack_destination = os.path.join(evidence_dir, source_pack_bundle_path)
    source_pack_identity = _write_source_pack(
        source_pack_destination, build_provenance
    )
    inputs["source_input_pack"] = {
        "bundle_path": source_pack_bundle_path,
        **source_pack_identity,
    }

    run_dir = focr.get("run_dir")
    if not run_dir or not os.path.isdir(run_dir):
        raise RowError(
            f"focr raw run dir {run_dir!r} is missing — a row without its raw logs "
            "is incomplete and may not be cited"
        )
    raw_dst = os.path.join(evidence_dir, "raw")
    os.makedirs(raw_dst, exist_ok=True)
    raw_names = [name for name in sorted(os.listdir(run_dir)) if not name.startswith("._")]
    if len(raw_names) > MAX_RAW_TIMING_FILES:
        raise RowError(
            f"raw timing file count exceeds {MAX_RAW_TIMING_FILES}: {len(raw_names)}"
        )
    raw_total = 0
    for name in raw_names:
        if os.path.basename(name) != name:
            raise RowError(f"noncanonical raw timing filename: {name!r}")
        src = os.path.join(run_dir, name)
        try:
            metadata = os.lstat(src)
        except OSError as error:
            raise RowError(f"cannot inspect raw timing file {src}: {error}") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise RowError(f"raw timing input is not a regular file: {src}")
        if metadata.st_size > MAX_RAW_TIMING_FILE_BYTES:
            raise RowError(f"raw timing file exceeds the size bound: {src}")
        raw_total += metadata.st_size
        if raw_total > MAX_RAW_TIMING_TOTAL_BYTES:
            raise RowError("raw timing inputs exceed the total byte bound")
        content = _read_bounded_file(src)
        with open(os.path.join(raw_dst, name), "wb") as destination:
            destination.write(content)

    timing_inputs = {}
    for source, document in (("focr", focr), ("reference", ref)):
        raw_timing = document.get("raw_timing")
        bundle_path = f"raw/{source}_timing.json"
        destination = os.path.join(evidence_dir, bundle_path)
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(raw_timing, handle, indent=2, sort_keys=True)
            handle.write("\n")
        timing_inputs[source] = {
            "bundle_path": bundle_path,
            "sha256": sha256_file(destination),
        }

    for raw_record in focr["raw_timing"]["records"]:
        for binding in raw_record["raw_files"].values():
            bundled = os.path.join(raw_dst, binding["path"])
            if not os.path.isfile(bundled) or sha256_file(bundled) != binding["sha256"]:
                raise RowError(
                    f"focr raw timing file binding does not verify: {binding['path']}"
                )

    correctness_root = os.path.join(evidence_dir, "correctness")
    reference_root = os.path.join(correctness_root, "reference")
    hypothesis_root = os.path.join(correctness_root, "hypothesis")
    os.makedirs(reference_root, exist_ok=True)
    os.makedirs(hypothesis_root, exist_ok=True)
    reference_name = correctness["reference_basename"]
    reference_relative = f"correctness/reference/{reference_name}"
    reference_destination = os.path.join(evidence_dir, reference_relative)
    with open(reference_destination, "xb") as handle:
        handle.write(correctness["reference_bytes"])
    correctness_hypotheses = []
    for run_id, content in correctness["hypothesis_runs"].items():
        name = f"{run_id}.stdout"
        relative = f"correctness/hypothesis/{name}"
        with open(os.path.join(evidence_dir, relative), "xb") as handle:
            handle.write(content)
        correctness_hypotheses.append(
            {
                "run_id": run_id,
                "bundle_path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    correctness_inputs = {
        "schema": CORRECTNESS_INPUTS_SCHEMA,
        "reference": {
            "bundle_path": reference_relative,
            "sha256": hashlib.sha256(correctness["reference_bytes"]).hexdigest(),
            "bytes": len(correctness["reference_bytes"]),
        },
        "hypotheses": correctness_hypotheses,
    }

    with open(os.path.join(evidence_dir, "row.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": "focr-gauntlet-row/v3",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_git_head": source_git_head,
                "source_root": source_root,
                "producer_root": producer_root,
                "producer_root_algorithm": PRODUCER_ROOT_ALGORITHM,
                "allowed_evidence_path": allowed_evidence_path,
                "inputs": inputs,
                "timing_inputs": timing_inputs,
                "correctness_inputs": correctness_inputs,
                "rows": rows,
            },
            f,
            indent=2,
        )
        f.write("\n")
    with open(
        os.path.join(evidence_dir, "PERF_LEDGER_ROW.md"), "w", encoding="utf-8"
    ) as f:
        for row in rows:
            f.write(row_markdown(row) + "\n")

    # SHA256SUMS last, over every file in the bundle (shasum -c compatible).
    entries = []
    for dirpath, _dirnames, filenames in os.walk(evidence_dir):
        for name in sorted(filenames):
            if name == "SHA256SUMS":
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, evidence_dir)
            maximum = (
                MAX_SUBJECT_BINARY_BYTES
                if rel == "subject/release-perf/focr"
                else MAX_SOURCE_PACK_BYTES
                if rel == "subject/source_input_pack.bin"
                else MAX_RAW_TIMING_FILE_BYTES
            )
            entries.append(f"{sha256_file(path, maximum)}  {rel}\n")
    manifest = os.path.join(evidence_dir, "SHA256SUMS")
    with open(manifest, "w", encoding="utf-8") as f:
        f.writelines(sorted(entries, key=lambda line: line.split("  ", 1)[1]))
    return manifest


# ── ledger insertion + validation against the REAL checker ──────────────────


def insert_rows(ledger_text: str, rows_md: list[str]) -> str:
    lines = ledger_text.splitlines(keepends=True)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("| date | claim_id | evidence_id |"):
            header_idx = i
            break
    if (
        header_idx is None
        or header_idx + 1 >= len(lines)
        or "---" not in lines[header_idx + 1]
    ):
        raise RowError("PERF_LEDGER.md ratio-table header not found; cannot insert row")
    insert_at = header_idx + 2
    return "".join(
        lines[:insert_at] + [md + "\n" for md in rows_md] + lines[insert_at:]
    )


def shadow_check(
    rows_md: list[str], evidence_dir: str, evidence_id: str
) -> tuple[bool, str]:
    """Run the real scripts/check_ledgers.py in a shadow repo copy with the
    candidate row inserted — validation without touching the live ledger."""
    with tempfile.TemporaryDirectory(prefix="focr-gauntlet-row-") as shadow:
        os.makedirs(os.path.join(shadow, "scripts"))
        shutil.copy2(CHECKER, os.path.join(shadow, "scripts", "check_ledgers.py"))
        os.makedirs(os.path.join(shadow, "docs"))
        for name in ("NEGATIVE_EVIDENCE.md", "DISCREPANCIES.md"):
            shutil.copy2(
                os.path.join(ROOT, "docs", name), os.path.join(shadow, "docs", name)
            )
        with open(LEDGER, encoding="utf-8") as f:
            ledger_text = f.read()
        with open(
            os.path.join(shadow, "docs", "PERF_LEDGER.md"), "w", encoding="utf-8"
        ) as f:
            f.write(insert_rows(ledger_text, rows_md))
        # DISCREPANCIES kill-switch validation greps src/**/*.rs.
        for dirpath, _dirnames, filenames in os.walk(os.path.join(ROOT, "src")):
            for name in filenames:
                if not name.endswith(".rs"):
                    continue
                src = os.path.join(dirpath, name)
                rel = os.path.relpath(src, ROOT)
                dst = os.path.join(shadow, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        # Pre-existing ledger entries (NEGATIVE_EVIDENCE etc.) resolve against
        # the repo's evidence dirs; mirror them, then overlay the candidate.
        repo_perf = os.path.join(ROOT, "artifacts", "perf")
        if os.path.isdir(repo_perf):
            shutil.copytree(repo_perf, os.path.join(shadow, "artifacts", "perf"))
        shadow_evidence = os.path.join(shadow, evidence_id)
        if os.path.isdir(shadow_evidence):
            shutil.rmtree(shadow_evidence)  # shadow-only; the repo copy is untouched
        shutil.copytree(evidence_dir, shadow_evidence)
        proc = subprocess.run(
            [sys.executable, os.path.join(shadow, "scripts", "check_ledgers.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0, proc.stdout + proc.stderr


def apply_rows(rows_md: list[str]) -> tuple[bool, str]:
    with open(LEDGER, encoding="utf-8") as f:
        original = f.read()
    updated = insert_rows(original, rows_md)
    with open(LEDGER, "w", encoding="utf-8") as f:
        f.write(updated)
    proc = subprocess.run(
        [sys.executable, CHECKER], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        with open(LEDGER, "w", encoding="utf-8") as f:
            f.write(original)  # revert: a failing row never lands
        return False, proc.stdout + proc.stderr
    return True, proc.stdout


def run(args: argparse.Namespace) -> int:
    source_git_head = current_git_head()
    focr = load_json(args.focr_stages, "focr-gauntlet-stages/v1")
    ref = load_json(args.ref_stages, "focr-gauntlet-stages/v1")
    roofline = load_json(args.roofline, "focr-gauntlet-roofline/v1")
    build_provenance = validate_build_provenance(focr, current_head=source_git_head)
    reference_provenance = validate_reference_provenance(ref)
    if args.model_commit != reference_provenance["manifest"]["model_commit"]:
        raise RowError("ledger model_commit disagrees with reference model provenance")
    focr_stages_sha256 = sha256_file(args.focr_stages)
    correctness = validate_correctness_proof(
        args.correctness_proof,
        focr=focr,
        ref=ref,
        ref_stages_path=args.ref_stages,
    )

    stages = args.stage or [
        s
        for s in LEDGER_STAGES
        if stage_record(focr, s)
        and stage_record(ref, s)
        and any(fl.get("stage") == s for fl in roofline.get("floors", []))
    ]
    if not stages:
        raise RowError("no stage is present in all three inputs — nothing to merge")

    claim = args.claim_id.strip()
    requested_evidence_dir = args.evidence_dir or os.path.join(
        ROOT, "artifacts", "perf", "bd-re8.17", claim
    )
    evidence_dir = os.fspath(validate_evidence_destination(requested_evidence_dir))
    evidence_id = os.fspath(Path(evidence_dir).relative_to(Path(ROOT).resolve()))
    allowed_evidence_path = _canonical_allowed_evidence_path(evidence_id)
    source_root = str(build_provenance["manifest"].get("root_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", source_root) is None:
        raise RowError("validated build provenance has no canonical source root")
    producer_root = gauntlet_producer_root(
        ROOT, source_git_head, verify_live=True
    )

    rows = [
        build_row(
            focr=focr,
            ref=ref,
            roofline=roofline,
            stage=stage,
            claim_id=claim,
            evidence_id=evidence_id,
            fixture_hash=args.fixture_hash,
            arch_features=args.arch_features,
            correctness_proof=args.correctness_proof,
            model_commit=args.model_commit,
            notes=args.notes,
            focr_stages_sha256=focr_stages_sha256,
            correctness=correctness,
        )
        for stage in stages
    ]
    rows_md = [row_markdown(row) for row in rows]

    manifest = write_bundle(
        evidence_dir,
        focr_path=args.focr_stages,
        ref_path=args.ref_stages,
        roofline_path=args.roofline,
        correctness=correctness,
        focr=focr,
        ref=ref,
        build_provenance=build_provenance,
        reference_provenance=reference_provenance,
        rows=rows,
        source_git_head=source_git_head,
        source_root=source_root,
        producer_root=producer_root,
        allowed_evidence_path=allowed_evidence_path,
    )

    ok, output = shadow_check(rows_md, evidence_dir, evidence_id)
    if not ok:
        print(output, file=sys.stderr)
        print(
            "ERROR: candidate row fails scripts/check_ledgers.py — not emitted",
            file=sys.stderr,
        )
        return 1

    if args.apply:
        applied, apply_output = apply_rows(rows_md)
        if not applied:
            print(apply_output, file=sys.stderr)
            print(
                "ERROR: in-place apply failed validation and was reverted",
                file=sys.stderr,
            )
            return 1

    for md in rows_md:
        print(md)
    print(
        json.dumps(
            {
                "event": "perf_ledger_row_ready",
                "stages": stages,
                "evidence_id": evidence_id,
                "manifest": manifest,
                "check": "pass",
                "applied": bool(args.apply),
            }
        ),
        file=sys.stderr,
    )
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def _synthetic_inputs(tmp: str) -> tuple[str, str, str, dict]:
    """Plumbing-only synthetic inputs (stamped synthetic; the normal path
    refuses them — the self-test exercises the internal allow_synthetic path)."""
    raw = os.path.join(tmp, "raw")
    os.makedirs(raw)
    hypothesis_bytes = b"identical output\n"
    reference_bytes = b"identical <|det|>box<|/det|>output\n"
    reference_sha = hashlib.sha256(reference_bytes).hexdigest()
    text_dir = os.path.join(tmp, "text")
    os.makedirs(text_dir)
    with open(os.path.join(text_dir, "page.md"), "wb") as handle:
        handle.write(reference_bytes)
    focr_samples = [10.0, 10.2, 10.1]
    for i, sample_ms in enumerate(focr_samples, start=1):
        tag = f"run_{i:03d}"
        with open(os.path.join(raw, f"run_{i:03d}.stderr"), "w", encoding="utf-8") as f:
            f.write(
                "[focr-timing] precision focr-mixed-ffn-int8\n"
                f"[focr-timing] decode {sample_ms * 600.0 / 1000.0:.3f}s "
                "(600 tokens, 0.010s/tok)\n"
            )
        with open(os.path.join(raw, f"run_{i:03d}.stdout"), "w", encoding="utf-8") as f:
            f.write(hypothesis_bytes.decode("utf-8"))
        with open(os.path.join(raw, f"{tag}.meta.json"), "w", encoding="utf-8") as f:
            json.dump({"tag": tag}, f)
            f.write("\n")

    def rec(stage: str, samples: list[float], source: str, **extra: object) -> dict:
        return {
            "schema": "focr-gauntlet-stage/v1",
            "source": source,
            "stage": stage,
            "ledger_stage": True,
            "unit": "ms",
            "samples_ms": samples,
            "best_ms": min(samples),
            "p50_ms": statistics.median(samples),
            "mean_ms": statistics.fmean(samples),
            "cv_pct": round(
                statistics.stdev(samples) / statistics.fmean(samples) * 100, 3
            ),
            "n": len(samples),
            "warmup_discarded": 1,
            "threads": 8,
            "precision": CURRENT_UNLIMITED_PRECISION if source == "focr" else "bf16",
            "backend": "focr" if source == "focr" else "hf",
            "allocator": "system",
            "synthetic": True,
            **extra,
        }

    def raw_timing(samples: list[float], source: str, *, tokens: int | None) -> dict:
        records = []
        for index, sample in enumerate(samples, start=1):
            run_id = f"run_{index:03d}"
            stage_sample = {"ms": round(sample, 6)}
            if tokens is not None:
                stage_sample["tokens"] = tokens
            record = {
                "run_id": run_id,
                "stages": {"decode_per_token": stage_sample},
            }
            if source == "focr":
                record["raw_files"] = {
                    kind: {
                        "path": run_id + suffix,
                        "sha256": sha256_file(os.path.join(raw, run_id + suffix)),
                    }
                    for kind, suffix in (
                        ("meta", ".meta.json"),
                        ("stderr", ".stderr"),
                        ("stdout", ".stdout"),
                    )
                }
            else:
                record["text_sha256"] = reference_sha
            records.append(record)
        return {
            "schema": RAW_TIMING_SCHEMA,
            "source": source,
            "unit": "ms",
            "measured_runs": len(records),
            "records": records,
        }

    focr = {
        "schema": "focr-gauntlet-stages/v1",
        "source": "focr",
        "created_utc": "2026-07-01T00:00:00Z",
        "run_dir": raw,
        "page": "page.png",
        "page_sha256": "f" * 64,
        "command": [
            "target/release-perf/focr",
            "ocr",
            "page.png",
            "--model",
            "/models/unlimited-ocr.focrq",
        ],
        "focr_env": {"FOCR_TIMING": "1", "FOCR_THREADS": "8"},
        "precision_gate_states": {name: "<unset>" for name in PRECISION_GATE_VARS},
        "precision": CURRENT_UNLIMITED_PRECISION,
        "decode_mode": CURRENT_UNLIMITED_DECODE_MODE,
        "quant_recipe": CURRENT_UNLIMITED_QUANT_RECIPE,
        "model": "/models/unlimited-ocr.focrq",
        "model_kind": "file",
        "model_sha256": CURRENT_UNLIMITED_MODEL_SHA256,
        "model_size": CURRENT_UNLIMITED_MODEL_SIZE,
        "threads": 8,
        "runs": 3,
        "stdout_identical_across_runs": True,
        "raw_timing": raw_timing(focr_samples, "focr", tokens=600),
        "stages": [
            rec(
                "decode_per_token",
                focr_samples,
                "focr",
                tokens=600,
                tokens_consistent=True,
            )
        ],
        "synthetic": True,
    }
    reference_entry = "gauntlet_ref_unlimited:run_stage"
    reference_contract = _contract_for_entry(reference_entry)
    ref = {
        "schema": "focr-gauntlet-stages/v1",
        "source": "reference",
        "page": "page.png",
        "page_sha256": "f" * 64,
        "command": [
            "python3",
            "gauntlet_reference.py",
            "--entry",
            reference_entry,
        ],
        "torch_version": "2.10.0",
        "transformers_version": "4.57.1",
        "reference_contract": reference_contract,
        "runs": 3,
        "text_sha256": reference_sha,
        "text_identical_across_runs": True,
        "raw_timing": raw_timing(
            [25.0, 25.5, 25.2], "reference", tokens=None
        ),
        "stages": [
            rec(
                "decode_per_token",
                [25.0, 25.5, 25.2],
                "reference",
                thread_proof={"budget": 8, "torch_num_threads": 8},
                reference_contract=reference_contract,
            )
        ],
        "synthetic": True,
    }
    roofline = {
        "schema": "focr-gauntlet-roofline/v1",
        "arch": "unlimited-ocr",
        "precision": CURRENT_UNLIMITED_DECODE_MODE,
        "timing_precision": CURRENT_UNLIMITED_PRECISION,
        "decode_mode": CURRENT_UNLIMITED_DECODE_MODE,
        "quant_recipe": CURRENT_UNLIMITED_QUANT_RECIPE,
        "model_sha256": CURRENT_UNLIMITED_MODEL_SHA256,
        "model_size": CURRENT_UNLIMITED_MODEL_SIZE,
        "precision_gate_states": focr["precision_gate_states"],
        "machine_profile": {"name": "m4", "dram_gb_s": 120.0},
        "floors": [
            {"stage": "decode_per_token", "floor_kind": "memory", "floor_ms": 4.953}
        ],
        "synthetic": True,
    }
    paths: list[str] = []
    for name, doc in (("focr.json", focr), ("ref.json", ref)):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        paths.append(path)
    roofline["stages_json_sha256"] = sha256_file(paths[0])
    roofline_path = os.path.join(tmp, "roofline.json")
    with open(roofline_path, "w", encoding="utf-8") as f:
        json.dump(roofline, f, indent=2)
    paths.append(roofline_path)
    return paths[0], paths[1], paths[2], focr


def _synthetic_correctness_receipt(
    reference_bytes: bytes, hypothesis_bytes: bytes
) -> dict:
    reference_text = reference_bytes.decode("utf-8")
    hypothesis_text = hypothesis_bytes.decode("utf-8")
    scored_reference, matches = _correctness_transform_reference(reference_text)
    normalized_reference = _correctness_normalize(scored_reference)
    normalized_hypothesis = _correctness_normalize(hypothesis_text)
    raw_distance = _correctness_levenshtein(scored_reference, hypothesis_text)
    normalized_distance = _correctness_levenshtein(
        normalized_reference, normalized_hypothesis
    )
    raw_cer = raw_distance / max(1, len(scored_reference))
    normalized_cer = normalized_distance / max(1, len(normalized_reference))
    page = {
        "page": "page.md",
        "status": "OK",
        "reference": {
            "source": _correctness_binding(reference_bytes, basename="page.md"),
            "transform": {
                "name": CORRECTNESS_REFERENCE_TRANSFORM,
                "matches": matches,
            },
            "scored": _correctness_binding(scored_reference.encode("utf-8")),
        },
        "hypothesis": {
            "source": _correctness_binding(
                hypothesis_bytes, basename="run_001.stdout"
            ),
            "transform": {"name": CORRECTNESS_IDENTITY_TRANSFORM, "matches": 0},
            "scored": _correctness_binding(hypothesis_bytes),
        },
        "raw_edit_distance": raw_distance,
        "normalized_edit_distance": normalized_distance,
        "ref_chars": len(scored_reference),
        "hyp_chars": len(hypothesis_text),
        "cer_raw": raw_cer,
        "cer_norm": normalized_cer,
        "exact": scored_reference == hypothesis_text,
        "exact_norm": normalized_reference == normalized_hypothesis,
    }
    return {
        "schema": CORRECTNESS_RECEIPT_SCHEMA,
        "created_utc": "2026-07-01T00:00:00Z",
        "normalization": CORRECTNESS_NORMALIZATION,
        "metric_formulas": dict(CORRECTNESS_METRIC_FORMULAS),
        "aggregate": {
            "pages_total": 1,
            "pages_with_hyp": 1,
            "exact_raw": int(page["exact"]),
            "exact_norm": int(page["exact_norm"]),
            "raw_edit_distance": raw_distance,
            "raw_reference_chars": len(scored_reference),
            "normalized_edit_distance": normalized_distance,
            "normalized_reference_chars": len(normalized_reference),
            "reference_source_bytes": len(reference_bytes),
            "reference_source_chars": len(reference_text),
            "hypothesis_source_bytes": len(hypothesis_bytes),
            "hypothesis_source_chars": len(hypothesis_text),
            "cer_raw": raw_cer,
            "cer_norm": normalized_cer,
        },
        "pages": [page],
    }


def _attach_synthetic_provenance(
    tmp: str,
    *,
    focr_path: str,
    ref_path: str,
    roofline_path: str,
    current_head: str,
) -> tuple[dict, dict, dict, dict, str]:
    focr = load_json(focr_path)
    ref = load_json(ref_path)
    workspace = Path(tmp) / "build-workspace"
    (workspace / "src").mkdir(parents=True)
    source_contents = {
        "Cargo.toml": b"[package]\nname='selftest'\nversion='0.0.0'\n",
        "Cargo.lock": b"version = 4\n",
        "rust-toolchain.toml": b"[toolchain]\nchannel='nightly'\n",
        "src/lib.rs": b"pub fn selftest() {}\n",
    }
    entries = []
    for name, content in source_contents.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(
            {
                "repository": "workspace",
                "path": name,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    entries.sort(key=lambda item: (item["repository"], item["path"]))
    manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "created_utc": "2026-07-01T00:00:00Z",
        "root_hash_algorithm": SOURCE_ROOT_ALGORITHM,
        "root_sha256": _canonical_source_root(entries),
        "entry_count": len(entries),
        "repositories": [
            {
                "id": "workspace",
                "path": os.fspath(workspace.resolve()),
                "git_head": current_head,
                "packages": ["selftest"],
                "selectors": sorted(source_contents),
            }
        ],
        "cargo_config_files": [],
        "entries": entries,
    }
    origin = Path(tmp) / "build-origin"
    origin.mkdir()
    manifest_path = origin / "source_input_manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_identity = {
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "size": len(manifest_bytes),
    }
    target = "aarch64-apple-darwin"
    origin_binary = origin / "target" / target / "release-perf" / "focr"
    origin_binary.parent.mkdir(parents=True)
    binary_bytes = b"#!/bin/sh\nexit 0\n"
    origin_binary.write_bytes(binary_bytes)
    subject_binary = Path(tmp) / "subject" / "release-perf" / "focr"
    subject_binary.parent.mkdir(parents=True)
    subject_binary.write_bytes(binary_bytes)
    binary_identity = {
        "sha256": hashlib.sha256(binary_bytes).hexdigest(),
        "size": len(binary_bytes),
    }
    synthetic_toolchain = {
        "rustc_verbose_version": "rustc self-test\nhost: aarch64-apple-darwin\n",
        "cargo_version": "cargo self-test",
        "rch_version": "rch self-test",
    }
    receipt = build_receipt_document(
        created_utc="2026-07-01T00:00:00Z",
        git_head=current_head,
        target_triple=target,
        cargo_target_dir=os.fspath(origin / "target"),
        toolchain=synthetic_toolchain,
        build_environment={
            "RUSTFLAGS": None,
            "CARGO_ENCODED_RUSTFLAGS": None,
            "CARGO_BUILD_RUSTFLAGS": None,
            "CARGO_HOME": None,
            "target_rustflags_env_name": "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS",
            "target_rustflags_env_value": None,
            "rustc_overrides": {
                "RUSTC": None,
                "RUSTC_WRAPPER": None,
                "RUSTC_WORKSPACE_WRAPPER": None,
            },
            "release_perf_profile_overrides": {},
            "cargo_config_build_rustflags": {"status": "unset", "value": None},
            "cargo_config_target": {"status": "unset", "value": None},
        },
        source_manifest_path=os.fspath(manifest_path.resolve()),
        source_manifest_identity=manifest_identity,
        source_manifest=manifest,
        binary_path=os.fspath(origin_binary.resolve()),
        binary_identity=binary_identity,
    )
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    receipt_path = Path(tmp) / "subject" / "build_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    focr.update(
        binary=os.fspath(subject_binary.resolve()),
        binary_sha256=binary_identity["sha256"],
        binary_size=binary_identity["size"],
        binary_origin=os.fspath(origin_binary.resolve()),
        build_receipt=os.fspath(receipt_path.resolve()),
        build_receipt_sha256=receipt_sha,
    )
    focr["command"][0] = focr["binary"]
    for record in focr["raw_timing"]["records"]:
        meta_path = Path(focr["run_dir"]) / record["raw_files"]["meta"]["path"]
        meta = {"tag": record["run_id"]}
        meta.update(
            {field: focr[field] for field in (
                "binary",
                "binary_sha256",
                "binary_size",
                "binary_origin",
                "build_receipt",
                "build_receipt_sha256",
            )}
        )
        meta_bytes = (json.dumps(meta, sort_keys=True) + "\n").encode()
        meta_path.write_bytes(meta_bytes)
        record["raw_files"]["meta"]["sha256"] = hashlib.sha256(meta_bytes).hexdigest()
    Path(focr_path).write_text(json.dumps(focr, indent=2) + "\n", encoding="utf-8")

    expected_files = [
        {"path": path, "bytes": size, "sha256": sha256}
        for path, size, sha256 in UNLIMITED_MODEL_FILES
    ]
    model_manifest = {
        "schema": REFERENCE_MODEL_MANIFEST_SCHEMA,
        "model_id": "baidu/Unlimited-OCR",
        "model_commit": MODEL_COMMIT,
        "synthetic": False,
        "citable": True,
        "file_count": len(expected_files),
        "root_hash_domain": REFERENCE_MODEL_ROOT_DOMAIN.decode("ascii"),
        "root_sha256": _reference_model_root(expected_files),
        "index": dict(UNLIMITED_MODEL_INDEX),
        "files": expected_files,
    }
    reference_entry = CURRENT_UNLIMITED_REFERENCE_ENTRY
    setup = "gauntlet_ref_unlimited:setup"
    model_dir = "/models/unlimited-ocr"
    reference_text_dir = os.path.join(tmp, "text")
    ref_command = [
        "scripts/gauntlet_reference.py",
        "--stage",
        "all",
        "--page",
        ref["page"],
        "--model-dir",
        model_dir,
        "--backend",
        "hf",
        "--precision",
        "bf16",
        "--max-length",
        "8192",
        "--text-dir",
        reference_text_dir,
        "--entry",
        reference_entry,
        "--setup",
        setup,
        "--runs",
        "3",
        "--warmup",
        "1",
        "--threads",
        "8",
        "--out",
        ref_path,
    ]
    pins = {
        name: "8"
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "FOCR_THREADS",
        )
    }
    ref.update(
        created_utc="2026-07-01T00:00:00Z",
        command=ref_command,
        env_pins=pins,
        model=model_dir,
        max_length=8192,
        text_dir=reference_text_dir,
        threads=8,
        precision="bf16",
        backend="hf",
        allocator="system",
        warmup=1,
        ambient_env={
            "FOCR_REF_MAX_LENGTH": "<unset>",
            "FOCR_REF_TEXT_DIR": "<unset>",
        },
        reference_model_manifest=model_manifest,
    )
    source_bindings = []
    for role, callable_name, relative in (
        ("harness", "gauntlet_reference:main", "scripts/gauntlet_reference.py"),
        ("entry", reference_entry, "scripts/gauntlet_ref_unlimited.py"),
        ("setup", setup, "scripts/gauntlet_ref_unlimited.py"),
    ):
        identity = _stable_file_identity(os.path.join(ROOT, relative), MAX_SOURCE_FILE_BYTES)
        source_bindings.append(
            {
                "role": role,
                "callable": callable_name,
                "path": relative,
                "bytes": identity["size"],
                "sha256": identity["sha256"],
            }
        )
    cache_name = os.path.basename(ref_path) + ".hf_modules_cache"
    binding = {
        "schema": REFERENCE_INFERENCE_BINDING_SCHEMA,
        "model_root_sha256": model_manifest["root_sha256"],
        "model_commit": MODEL_COMMIT,
        "entry": reference_entry,
        "setup": setup,
        "stage": "all",
        "page": ref["page"],
        "page_sha256": ref["page_sha256"],
        "model_dir": model_dir,
        "max_length": 8192,
        "text_dir": reference_text_dir,
        "backend": "hf",
        "precision": "bf16",
        "threads": 8,
        "runs": 3,
        "warmup": 1,
        "allocator": "system",
        "argv": ref_command,
        "env_pins": pins,
        "ambient_env": ref["ambient_env"],
        "torch_version": ref["torch_version"],
        "transformers_version": ref["transformers_version"],
        "reference_contract": ref["reference_contract"],
        "hf_modules_cache": {
            "evidence_dir": os.path.dirname(ref_path),
            "path": cache_name,
            "effective_path": os.path.join(os.path.dirname(ref_path), cache_name),
            "fresh": True,
        },
        "sources": source_bindings,
    }
    canonical = json.dumps(
        binding, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    binding["binding_hash_domain"] = REFERENCE_INFERENCE_BINDING_DOMAIN.decode("ascii")
    binding["binding_sha256"] = hashlib.sha256(
        REFERENCE_INFERENCE_BINDING_DOMAIN + canonical
    ).hexdigest()
    ref["reference_inference_binding"] = binding
    Path(ref_path).write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    roofline = load_json(roofline_path)
    roofline["stages_json_sha256"] = sha256_file(focr_path)
    Path(roofline_path).write_text(json.dumps(roofline, indent=2) + "\n", encoding="utf-8")
    return focr, ref, receipt, manifest, target


def _self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, **fields: object) -> None:
        print(json.dumps({"check": name, "result": "pass" if ok else "fail", **fields}))
        if not ok:
            failures.append(name)

    with tempfile.TemporaryDirectory(prefix="focr-gauntlet-row-selftest-") as tmp:
        focr_path, ref_path, roofline_path, focr = _synthetic_inputs(tmp)
        synthetic_head = "0123456789abcdef0123456789abcdef01234567"
        focr, ref, _receipt, _manifest, synthetic_host = _attach_synthetic_provenance(
            tmp,
            focr_path=focr_path,
            ref_path=ref_path,
            roofline_path=roofline_path,
            current_head=synthetic_head,
        )
        roofline = load_json(roofline_path)
        synthetic_toolchain = _receipt["toolchain"]
        build_provenance = validate_build_provenance(
            focr,
            current_head=synthetic_head,
            workspace_root=_manifest["repositories"][0]["path"],
            verify_live=False,
            expected_toolchain=synthetic_toolchain,
            expected_host=synthetic_host,
        )
        check(
            "build-producer-document-consumed-by-row-validator",
            build_provenance["receipt"] == _receipt,
        )
        reference_provenance = validate_reference_provenance(ref)

        import gauntlet_reference as reference_writer

        emitted_reference_binding = ref["reference_inference_binding"]
        writer_args = argparse.Namespace(
            entry=emitted_reference_binding["entry"],
            setup=emitted_reference_binding["setup"],
            page=ref["page"],
            model_dir=ref["model"],
            max_length=ref["max_length"],
            text_dir=ref["text_dir"],
            backend=ref["backend"],
            precision=ref["precision"],
            runs=ref["runs"],
            warmup=ref["warmup"],
            allocator=ref["allocator"],
        )
        writer_environment_names = tuple(ref["env_pins"]) + (
            "FOCR_REF_MAX_LENGTH",
            "FOCR_REF_TEXT_DIR",
        )
        saved_writer_environment = {
            name: os.environ.get(name) for name in writer_environment_names
        }
        saved_argv = list(sys.argv)
        try:
            sys.argv = list(ref["command"])
            for name, value in ref["env_pins"].items():
                os.environ[name] = value
            os.environ.pop("FOCR_REF_MAX_LENGTH", None)
            os.environ.pop("FOCR_REF_TEXT_DIR", None)
            produced_binding = reference_writer.build_inference_binding(
                writer_args,
                emitted_reference_binding["stage"],
                ref["threads"],
                ref["page_sha256"],
                ref["reference_contract"],
                ref["reference_model_manifest"],
                emitted_reference_binding["hf_modules_cache"],
                emitted_reference_binding["sources"],
                ref["torch_version"],
                ref["transformers_version"],
            )
        finally:
            sys.argv = saved_argv
            for name, value in saved_writer_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        writer_ref = copy.deepcopy(ref)
        writer_ref["reference_inference_binding"] = produced_binding
        try:
            validate_reference_provenance(writer_ref)
            check("reference-producer-binding-consumed-by-row-validator", True)
        except RowError as error:
            check(
                "reference-producer-binding-consumed-by-row-validator",
                False,
                error=str(error),
            )

        def provenance_refused(name: str, operation) -> None:
            try:
                operation()
                check(name, False)
            except RowError:
                check(name, True)

        def manifest_refused(name: str, mutate, *, verify_files: bool = False) -> None:
            candidate = copy.deepcopy(_manifest)
            mutate(candidate)
            provenance_refused(
                name,
                lambda: _validate_source_manifest(
                    candidate,
                    workspace_root=_manifest["repositories"][0]["path"],
                    expected_head=synthetic_head,
                    verify_live=verify_files,
                    verify_repository_heads=False,
                ),
            )

        manifest_refused("build-refuses-missing-manifest-field", lambda value: value.pop("schema"))
        manifest_refused("build-refuses-extra-manifest-field", lambda value: value.update(extra=True))
        manifest_refused(
            "build-refuses-duplicate-source-entry",
            lambda value: value["entries"].append(copy.deepcopy(value["entries"][-1])),
        )
        manifest_refused(
            "build-refuses-uppercase-source-hash",
            lambda value: value["entries"][0].update(
                sha256=value["entries"][0]["sha256"].upper()
            ),
        )
        manifest_refused(
            "build-refuses-bad-source-size",
            lambda value: value["entries"][0].update(size=True),
        )
        manifest_refused(
            "build-refuses-source-root-drift",
            lambda value: value.update(root_sha256="0" * 64),
        )
        manifest_refused(
            "build-refuses-source-head-drift",
            lambda value: value["repositories"][0].update(git_head="f" * 40),
        )

        def mutate_rehashed_source(value: dict) -> None:
            entry = next(item for item in value["entries"] if item["path"] == "src/lib.rs")
            entry["sha256"] = "0" * 64
            value["root_sha256"] = _canonical_source_root(value["entries"])

        manifest_refused(
            "build-refuses-rehashed-live-source-entry",
            mutate_rehashed_source,
            verify_files=True,
        )

        provenance_refused(
            "build-refuses-receipt-head-drift",
            lambda: validate_build_provenance(
                {**focr, "build_receipt_sha256": "0" * 64},
                current_head=synthetic_head,
                workspace_root=_manifest["repositories"][0]["path"],
                verify_live=False,
                expected_toolchain=synthetic_toolchain,
                expected_host=synthetic_host,
            ),
        )
        provenance_refused(
            "build-refuses-receipt-path-escape",
            lambda: validate_build_provenance(
                {**focr, "build_receipt": os.path.join(tmp, "outside-receipt.json")},
                current_head=synthetic_head,
                workspace_root=_manifest["repositories"][0]["path"],
                verify_live=False,
                expected_toolchain=synthetic_toolchain,
                expected_host=synthetic_host,
            ),
        )
        provenance_refused(
            "build-refuses-binary-size-drift",
            lambda: validate_build_provenance(
                {**focr, "binary_size": focr["binary_size"] + 1},
                current_head=synthetic_head,
                workspace_root=_manifest["repositories"][0]["path"],
                verify_live=False,
                expected_toolchain=synthetic_toolchain,
                expected_host=synthetic_host,
            ),
        )

        def reference_refused(name: str, mutate) -> None:
            candidate = copy.deepcopy(ref)
            mutate(candidate)
            provenance_refused(name, lambda: validate_reference_provenance(candidate))

        reference_refused(
            "reference-refuses-missing-model-manifest",
            lambda value: value.pop("reference_model_manifest"),
        )
        reference_refused(
            "reference-refuses-extra-model-file",
            lambda value: value["reference_model_manifest"]["files"].append(
                {"path": "extra", "bytes": 1, "sha256": "0" * 64}
            ),
        )
        reference_refused(
            "reference-refuses-duplicate-model-file",
            lambda value: value["reference_model_manifest"]["files"].append(
                copy.deepcopy(value["reference_model_manifest"]["files"][0])
            ),
        )
        reference_refused(
            "reference-refuses-uppercase-model-hash",
            lambda value: value["reference_model_manifest"]["files"][0].update(
                sha256=value["reference_model_manifest"]["files"][0]["sha256"].upper()
            ),
        )
        reference_refused(
            "reference-refuses-model-commit-drift",
            lambda value: value["reference_model_manifest"].update(model_commit="f" * 40),
        )
        reference_refused(
            "reference-refuses-model-root-drift",
            lambda value: value["reference_model_manifest"].update(root_sha256="0" * 64),
        )
        reference_refused(
            "reference-refuses-model-shard-drift",
            lambda value: value["reference_model_manifest"]["files"][4].update(
                sha256="0" * 64
            ),
        )
        reference_refused(
            "reference-refuses-model-index-census-drift",
            lambda value: value["reference_model_manifest"]["index"].update(
                weight_count=2709
            ),
        )
        reference_refused(
            "reference-refuses-inference-source-drift",
            lambda value: value["reference_inference_binding"]["sources"][0].update(
                sha256="0" * 64
            ),
        )
        reference_refused(
            "reference-refuses-inference-argv-drift",
            lambda value: value["reference_inference_binding"]["argv"].append("--bogus"),
        )

        def refresh_test_reference_binding(value: dict) -> None:
            binding = value["reference_inference_binding"]
            unsigned = dict(binding)
            unsigned.pop("binding_hash_domain", None)
            unsigned.pop("binding_sha256", None)
            canonical = json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
            binding["binding_hash_domain"] = (
                REFERENCE_INFERENCE_BINDING_DOMAIN.decode("ascii")
            )
            binding["binding_sha256"] = hashlib.sha256(
                REFERENCE_INFERENCE_BINDING_DOMAIN + canonical
            ).hexdigest()

        def drift_reference_max_length(value: dict) -> None:
            value["max_length"] = 4096
            binding = value["reference_inference_binding"]
            binding["max_length"] = 4096
            option = binding["argv"].index("--max-length") + 1
            binding["argv"][option] = "4096"
            value["command"] = list(binding["argv"])
            refresh_test_reference_binding(value)

        reference_refused(
            "reference-refuses-explicit-max-length-drift",
            drift_reference_max_length,
        )

        def arm_reference_ambient(value: dict, name: str, setting: str) -> None:
            value["ambient_env"][name] = setting
            value["reference_inference_binding"]["ambient_env"][name] = setting
            refresh_test_reference_binding(value)

        reference_refused(
            "reference-refuses-ambient-max-length-override",
            lambda value: arm_reference_ambient(
                value, "FOCR_REF_MAX_LENGTH", "4096"
            ),
        )
        reference_refused(
            "reference-refuses-ambient-text-dir-override",
            lambda value: arm_reference_ambient(
                value, "FOCR_REF_TEXT_DIR", "/tmp/ambient-text"
            ),
        )
        correctness_receipt_path = os.path.join(tmp, "cer.json")
        reference_bytes = _read_bounded_file(os.path.join(tmp, "text", "page.md"))
        hypothesis_bytes = _read_bounded_file(
            os.path.join(focr["run_dir"], "run_001.stdout")
        )
        correctness_receipt = _synthetic_correctness_receipt(
            reference_bytes, hypothesis_bytes
        )
        with open(correctness_receipt_path, "w", encoding="utf-8") as handle:
            json.dump(correctness_receipt, handle)
        correctness = validate_correctness_proof(
            f"CER_norm=0.000000 pinned reference comparison ({correctness_receipt_path})",
            focr=focr,
            ref=ref,
            ref_stages_path=ref_path,
        )
        kwargs = dict(
            focr=focr,
            ref=ref,
            roofline=roofline,
            stage="decode_per_token",
            claim_id="selftest-claim",
            evidence_id="artifacts/perf/bd-re8.17/selftest",
            fixture_hash="page=page.png sha256=" + "f" * 64,
            arch_features="aarch64+neon+dotprod",
            correctness_proof=(
                f"CER_norm=0.000000 pinned reference comparison "
                f"({correctness_receipt_path})"
            ),
            model_commit=MODEL_COMMIT,
            notes="",
            focr_stages_sha256=sha256_file(focr_path),
            correctness=correctness,
        )

        hypothesis_runs = {
            f"run_{index:03d}": _read_bounded_file(
                os.path.join(focr["run_dir"], f"run_{index:03d}.stdout")
            )
            for index in range(1, 4)
        }

        def receipt_rejected(name: str, patch) -> None:
            candidate = copy.deepcopy(correctness_receipt)
            patch(candidate)
            try:
                _validate_correctness_receipt_payload(
                    candidate,
                    reference_bytes=reference_bytes,
                    hypothesis_runs=hypothesis_runs,
                    focr=focr,
                    ref=ref,
                )
                check(name, False)
            except RowError:
                check(name, True)

        for name, patch in (
            ("correctness-refuses-legacy-receipt", lambda d: d.pop("schema")),
            ("correctness-refuses-extra-top-field", lambda d: d.update(extra=True)),
            ("correctness-refuses-created-utc-drift", lambda d: d.update(created_utc="today")),
            ("correctness-refuses-normalization-drift", lambda d: d.update(normalization="unknown")),
            (
                "correctness-refuses-metric-formula-drift",
                lambda d: d["metric_formulas"].update(cer_norm="self asserted"),
            ),
            (
                "correctness-refuses-reference-basename-drift",
                lambda d: d["pages"][0]["reference"]["source"].update(
                    basename="../page.md"
                ),
            ),
            (
                "correctness-refuses-source-sha-drift",
                lambda d: d["pages"][0]["reference"]["source"].update(
                    sha256="0" * 64
                ),
            ),
            (
                "correctness-refuses-source-byte-count-drift",
                lambda d: d["pages"][0]["reference"]["source"].update(bytes=1),
            ),
            (
                "correctness-refuses-boolean-source-count",
                lambda d: d["pages"][0]["reference"]["source"].update(bytes=True),
            ),
            (
                "correctness-refuses-source-char-count-drift",
                lambda d: d["pages"][0]["reference"]["source"].update(chars=1),
            ),
            (
                "correctness-refuses-normalized-sha-drift",
                lambda d: d["pages"][0]["reference"]["source"].update(
                    normalized_sha256="0" * 64
                ),
            ),
            (
                "correctness-refuses-normalized-count-drift",
                lambda d: d["pages"][0]["reference"]["source"].update(
                    normalized_chars=1
                ),
            ),
            (
                "correctness-refuses-reference-transform-drift",
                lambda d: d["pages"][0]["reference"]["transform"].update(
                    name="identity-v1"
                ),
            ),
            (
                "correctness-refuses-transform-count-drift",
                lambda d: d["pages"][0]["reference"]["transform"].update(matches=99),
            ),
            (
                "correctness-refuses-boolean-transform-count",
                lambda d: d["pages"][0]["reference"]["transform"].update(matches=True),
            ),
            (
                "correctness-refuses-hypothesis-transform-drift",
                lambda d: d["pages"][0]["hypothesis"]["transform"].update(matches=1),
            ),
            (
                "correctness-refuses-scored-binding-drift",
                lambda d: d["pages"][0]["reference"]["scored"].update(
                    sha256="0" * 64
                ),
            ),
            (
                "correctness-refuses-raw-distance-drift",
                lambda d: d["pages"][0].update(raw_edit_distance=1),
            ),
            (
                "correctness-refuses-boolean-page-distance",
                lambda d: d["pages"][0].update(raw_edit_distance=False),
            ),
            (
                "correctness-refuses-normalized-distance-drift",
                lambda d: d["pages"][0].update(normalized_edit_distance=1),
            ),
            (
                "correctness-refuses-page-cer-drift",
                lambda d: d["pages"][0].update(cer_norm=0.01),
            ),
            (
                "correctness-refuses-exactness-drift",
                lambda d: d["pages"][0].update(exact=False),
            ),
            (
                "correctness-refuses-aggregate-integer-drift",
                lambda d: d["aggregate"].update(normalized_reference_chars=1),
            ),
            (
                "correctness-refuses-boolean-aggregate-integer",
                lambda d: d["aggregate"].update(exact_raw=True),
            ),
            (
                "correctness-refuses-aggregate-cer-drift",
                lambda d: d["aggregate"].update(cer_norm=0.01),
            ),
        ):
            receipt_rejected(name, patch)

        drifted_hypotheses = dict(hypothesis_runs)
        drifted_hypotheses["run_003"] = b"different output\n"
        try:
            _validate_correctness_receipt_payload(
                correctness_receipt,
                reference_bytes=reference_bytes,
                hypothesis_runs=drifted_hypotheses,
                focr=focr,
                ref=ref,
            )
            check("correctness-refuses-physical-hypothesis-drift", False)
        except RowError:
            check("correctness-refuses-physical-hypothesis-drift", True)

        # Synthetic inputs are refused on the normal path…
        try:
            build_row(**kwargs)
            check("refuses-synthetic", False)
        except RowError:
            check("refuses-synthetic", True)

        # …and merge correctly on the explicit self-test path.
        row = build_row(**kwargs, allow_synthetic=True)
        check("row-has-all-columns", list(row) == COLUMNS)
        check("row-ratio", row["ratio"] == "2.500")
        check("row-dist-above-floor", row["dist_above_floor"] == "2.02")
        check("row-stage-hyphenated", row["stage"] == "decode-per-token")
        check("row-threads-cell", row["threads (focr=ref N)"] == "focr=ref=8")
        check(
            "row-current-unlimited-precision",
            row["precision (focr vs ref)"]
            == f"{CURRENT_UNLIMITED_PRECISION} vs hf-bf16",
        )
        check(
            "row-binds-current-unlimited-model",
            f"model_sha256={CURRENT_UNLIMITED_MODEL_SHA256}" in row["command/env"]
            and f"model_size={CURRENT_UNLIMITED_MODEL_SIZE}" in row["command/env"],
        )
        md = row_markdown(row)
        check("row-cell-count", md.count("|") == len(COLUMNS) + 1)

        aggregate_tamper = json.loads(json.dumps(focr))
        aggregate_tamper["stages"][0].update(
            samples_ms=[9.0, 9.0, 9.0],
            best_ms=9.0,
            p50_ms=9.0,
            mean_ms=9.0,
            cv_pct=0.0,
        )
        try:
            build_row(
                **{**kwargs, "focr": aggregate_tamper}, allow_synthetic=True
            )
            check("refuses-aggregate-contradicting-raw-timing", False)
        except RowError as error:
            check(
                "refuses-aggregate-contradicting-raw-timing",
                "raw-derived" in str(error),
                error=str(error),
            )

        raw_tamper = json.loads(json.dumps(ref))
        raw_tamper["raw_timing"]["records"][0]["stages"][
            "decode_per_token"
        ]["ms"] = 99.0
        try:
            build_row(**{**kwargs, "ref": raw_tamper}, allow_synthetic=True)
            check("refuses-raw-timing-contradicting-aggregate", False)
        except RowError as error:
            check(
                "refuses-raw-timing-contradicting-aggregate",
                "raw-derived" in str(error),
                error=str(error),
            )

        for name, mutate in (
            (
                "refuses-missing-reference-per-run-text-hash",
                lambda doc: doc["raw_timing"]["records"][0].pop("text_sha256"),
            ),
            (
                "refuses-invalid-reference-per-run-text-hash",
                lambda doc: doc["raw_timing"]["records"][0].update(
                    text_sha256="invalid"
                ),
            ),
            (
                "refuses-drifting-reference-per-run-text-hash",
                lambda doc: doc["raw_timing"]["records"][0].update(
                    text_sha256="cd" * 32
                ),
            ),
            (
                "refuses-list-valued-reference-top-level-text-hash",
                lambda doc: doc.update(text_sha256=["ab" * 32]),
            ),
        ):
            bad_ref = json.loads(json.dumps(ref))
            mutate(bad_ref)
            try:
                build_row(**{**kwargs, "ref": bad_ref}, allow_synthetic=True)
                check(name, False)
            except RowError:
                check(name, True)

        def refuses_current_contract(
            name: str,
            *,
            mutate_focr=None,
            mutate_roofline=None,
        ) -> None:
            bad_focr = json.loads(json.dumps(focr))
            bad_roofline = json.loads(json.dumps(roofline))
            if mutate_focr is not None:
                mutate_focr(bad_focr)
            if mutate_roofline is not None:
                mutate_roofline(bad_roofline)
            try:
                build_row(
                    **{**kwargs, "focr": bad_focr, "roofline": bad_roofline},
                    allow_synthetic=True,
                )
                check(name, False)
            except RowError:
                check(name, True)

        refuses_current_contract(
            "refuses-historical-focr-int8-label",
            mutate_focr=lambda doc: (
                doc.update(precision="focr-int8"),
                doc["stages"][0].update(precision="focr-int8"),
            ),
        )
        refuses_current_contract(
            "refuses-full-int8-release-label",
            mutate_focr=lambda doc: (
                doc.update(precision="focr-full-int8"),
                doc["stages"][0].update(precision="focr-full-int8"),
            ),
        )
        refuses_current_contract(
            "refuses-wrong-decode-mode",
            mutate_focr=lambda doc: doc.update(decode_mode="full-int8"),
        )
        refuses_current_contract(
            "refuses-wrong-quant-recipe",
            mutate_focr=lambda doc: doc.update(quant_recipe="decoder-ffn-int8-v1"),
        )
        refuses_current_contract(
            "refuses-wrong-current-model-hash",
            mutate_focr=lambda doc: doc.update(model_sha256="0" * 64),
        )
        refuses_current_contract(
            "refuses-wrong-current-model-size",
            mutate_focr=lambda doc: doc.update(model_size=CURRENT_UNLIMITED_MODEL_SIZE + 1),
        )
        refuses_current_contract(
            "refuses-ambiguous-falsy-gate",
            mutate_focr=lambda doc: doc["precision_gate_states"].update(
                FOCR_INT8_ATTN="banana"
            ),
        )
        refuses_current_contract(
            "refuses-armed-oq14-gate",
            mutate_focr=lambda doc: doc["precision_gate_states"].update(
                FOCR_INT8_ATTN="1"
            ),
        )
        refuses_current_contract(
            "refuses-present-falsy-kv-switch",
            mutate_focr=lambda doc: doc["precision_gate_states"].update(
                FOCR_INT8_KV="0"
            ),
        )
        for field, value in (
            ("precision", "int8"),
            ("timing_precision", "focr-full-int8"),
            ("decode_mode", "full-int8"),
            ("quant_recipe", "decoder-ffn-int8-v1"),
            ("model_sha256", "0" * 64),
            ("model_size", CURRENT_UNLIMITED_MODEL_SIZE + 1),
            (
                "precision_gate_states",
                {
                    **focr["precision_gate_states"],
                    "FOCR_INT8_ATTN": "1",
                },
            ),
            ("stages_json_sha256", "0" * 64),
        ):
            refuses_current_contract(
                f"refuses-wrong-roofline-{field.replace('_', '-')}",
                mutate_roofline=lambda doc, field=field, value=value: doc.update(
                    {field: value}
                ),
            )

        def contract_ref(entry: str) -> dict:
            _contract_id, torch_pin, transformers_pin = REFERENCE_CONTRACTS[entry]
            contract = _contract_for_entry(entry)
            doc = json.loads(json.dumps(ref))
            doc["command"] = [
                "python3",
                "gauntlet_reference.py",
                "--entry",
                entry,
                "--pin-torch",
                torch_pin,
                "--pin-transformers",
                transformers_pin,
            ]
            doc["torch_version"] = torch_pin
            doc["transformers_version"] = transformers_pin
            doc["reference_contract"] = contract
            doc["stages"][0]["reference_contract"] = contract
            return doc

        for entry, (contract_id, _torch, _transformers) in REFERENCE_CONTRACTS.items():
            doc = contract_ref(entry)
            check(
                f"reference-contract-map-{contract_id}",
                validate_reference_contract(doc, doc["stages"][0])
                == _contract_for_entry(entry),
            )

        zoo_focr = json.loads(json.dumps(focr))
        zoo_focr.update(precision="focr-int8")
        zoo_focr["stages"][0].update(precision="focr-int8")
        zoo_focr.pop("decode_mode")
        zoo_focr.pop("quant_recipe")
        zoo_roofline = json.loads(json.dumps(roofline))
        zoo_roofline.update(precision="int8")
        for field in (
            "quant_recipe",
            "model_sha256",
            "model_size",
            "timing_precision",
            "decode_mode",
            "precision_gate_states",
            "stages_json_sha256",
        ):
            zoo_roofline.pop(field)
        try:
            build_row(
                **{
                    **kwargs,
                    "focr": zoo_focr,
                    "ref": contract_ref("gauntlet_ref_zoo:run_got"),
                    "roofline": zoo_roofline,
                },
                allow_synthetic=True,
            )
            check("zoo-row-does-not-require-unlimited-release-contract", True)
        except RowError as error:
            check(
                "zoo-row-does-not-require-unlimited-release-contract",
                False,
                error=str(error),
            )

        def refuses_contract(name: str, doc: dict) -> None:
            try:
                validate_reference_contract(doc, doc["stages"][0])
                check(name, False)
            except RowError:
                check(name, True)

        unknown_entry = json.loads(json.dumps(ref))
        unknown_entry["command"] = [
            "gauntlet_reference.py",
            "--entry",
            "gauntlet_ref_zoo:run_unknown",
        ]
        refuses_contract("refuses-unknown-reference-entry", unknown_entry)

        got_mismatched_stamp = contract_ref("gauntlet_ref_zoo:run_got")
        got_mismatched_stamp["reference_contract"] = _contract_for_entry(
            "gauntlet_ref_unlimited:run_stage"
        )
        refuses_contract("refuses-entry-contract-stamp-mismatch", got_mismatched_stamp)

        got_mismatched_pin = contract_ref("gauntlet_ref_zoo:run_got")
        pin_index = got_mismatched_pin["command"].index("--pin-torch") + 1
        got_mismatched_pin["command"][pin_index] = "2.10.0"
        refuses_contract("refuses-entry-command-pin-mismatch", got_mismatched_pin)

        got_mismatched_stage = contract_ref("gauntlet_ref_zoo:run_got")
        got_mismatched_stage["stages"][0]["reference_contract"] = _contract_for_entry(
            "gauntlet_ref_zoo:run_smolvlm2"
        )
        refuses_contract("refuses-entry-stage-stamp-mismatch", got_mismatched_stage)

        got_unstamped_stage = contract_ref("gauntlet_ref_zoo:run_got")
        got_unstamped_stage["stages"][0].pop("reference_contract")
        refuses_contract("refuses-mixed-stamped-and-legacy-stages", got_unstamped_stage)

        # Refusals: noisy cv, thread mismatch, missing thread proof,
        # nondeterministic stdout, token drift.
        def mutated(patch) -> dict:
            doc = json.loads(json.dumps(ref))
            patch(doc["stages"][0])
            return doc

        for name, patch in (
            ("refuses-noisy-cv", lambda r: r.update(cv_pct=9.0)),
            ("refuses-thread-mismatch", lambda r: r.update(threads=16)),
            ("refuses-missing-thread-proof", lambda r: r.pop("thread_proof")),
            ("refuses-missing-precision", lambda r: r.update(precision="")),
        ):
            try:
                build_row(**{**kwargs, "ref": mutated(patch)}, allow_synthetic=True)
                check(name, False)
            except RowError:
                check(name, True)

        for name, field, version in (
            ("refuses-bogus-torch-version", "torch_version", "2.12.1"),
            (
                "refuses-bogus-transformers-version",
                "transformers_version",
                "4.45.2",
            ),
        ):
            bad_ref = json.loads(json.dumps(ref))
            bad_ref[field] = version
            try:
                build_row(**{**kwargs, "ref": bad_ref}, allow_synthetic=True)
                check(name, False)
            except RowError:
                check(name, True)

        focr_bad = json.loads(json.dumps(focr))
        focr_bad["stdout_identical_across_runs"] = False
        try:
            build_row(**{**kwargs, "focr": focr_bad}, allow_synthetic=True)
            check("refuses-nondeterministic-stdout", False)
        except RowError:
            check("refuses-nondeterministic-stdout", True)

        # Bundle: manifest hashes verify against the files on disk.
        test_perf_root = Path(tmp) / "artifacts" / "perf"
        evidence_dir = os.path.join(test_perf_root, "bd-re8.17", "selftest")
        manifest = write_bundle(
            evidence_dir,
            focr_path=focr_path,
            ref_path=ref_path,
            roofline_path=roofline_path,
            correctness=correctness,
            focr=focr,
            ref=ref,
            build_provenance=build_provenance,
            reference_provenance=reference_provenance,
            rows=[row],
            source_git_head="0123456789abcdef0123456789abcdef01234567",
            source_root=build_provenance["manifest"]["root_sha256"],
            producer_root="1" * 64,
            allowed_evidence_path="artifacts/perf/bd-re8.17/selftest",
            perf_root=test_perf_root,
        )
        ok = True
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                digest, rel = line.rstrip("\n").split("  ", 1)
                ok = ok and sha256_file(os.path.join(evidence_dir, rel)) == digest
        check("bundle-manifest-verifies", ok)
        check(
            "bundle-has-raw-logs",
            os.path.isfile(os.path.join(evidence_dir, "raw", "run_001.stderr")),
        )
        check(
            "bundle-has-both-raw-timing-inputs",
            all(
                os.path.isfile(os.path.join(evidence_dir, "raw", name))
                for name in ("focr_timing.json", "reference_timing.json")
            ),
        )
        with open(os.path.join(evidence_dir, "row.json"), encoding="utf-8") as f:
            bundled_row = json.load(f)
        check(
            "bundle-row-schema-v3", bundled_row.get("schema") == "focr-gauntlet-row/v3"
        )
        check(
            "bundle-binds-source-and-producer-roots",
            bundled_row.get("source_git_head")
            == "0123456789abcdef0123456789abcdef01234567"
            and bundled_row.get("source_root")
            == build_provenance["manifest"]["root_sha256"]
            and bundled_row.get("producer_root") == "1" * 64
            and bundled_row.get("producer_root_algorithm") == PRODUCER_ROOT_ALGORITHM
            and bundled_row.get("allowed_evidence_path")
            == "artifacts/perf/bd-re8.17/selftest",
        )
        check(
            "bundle-row-binds-both-raw-timing-inputs",
            set(bundled_row.get("timing_inputs", {})) == {"focr", "reference"},
        )
        correctness_inputs = bundled_row.get("correctness_inputs", {})
        check(
            "bundle-row-binds-correctness-source-inputs",
            correctness_inputs.get("schema") == CORRECTNESS_INPUTS_SCHEMA
            and correctness_inputs.get("reference", {}).get("bundle_path")
            == "correctness/reference/page.md"
            and len(correctness_inputs.get("hypotheses", [])) == 3,
        )
        check(
            "bundle-has-physical-correctness-sources",
            os.path.isfile(
                os.path.join(evidence_dir, "correctness", "reference", "page.md")
            )
            and all(
                os.path.isfile(
                    os.path.join(
                        evidence_dir,
                        "correctness",
                        "hypothesis",
                        f"run_{index:03d}.stdout",
                    )
                )
                for index in range(1, 4)
            ),
        )
        check(
            "bundle-uses-canonical-input-names",
            all(
                os.path.isfile(os.path.join(evidence_dir, name))
                for name in (
                    "focr_stages.json",
                    "ref_stages.json",
                    "roofline.json",
                    "correctness_receipt.json",
                )
            ),
        )

        containment_root = Path(tmp) / "containment" / "artifacts" / "perf"
        containment_root.mkdir(parents=True)
        prefix_sibling = containment_root.parent / "performance" / "claim"
        try:
            validate_evidence_destination(prefix_sibling, perf_root=containment_root)
            check("refuses-evidence-prefix-sibling", False)
        except RowError:
            check("refuses-evidence-prefix-sibling", True)

        producer_repo = Path(tmp) / "producer-root"
        producer_repo.mkdir()
        producer_contents: dict[str, bytes] = {}
        for index, relative in enumerate(PRODUCER_PATHS, start=1):
            content = f"producer input {index}: {relative}\n".encode("utf-8")
            destination = producer_repo.joinpath(*Path(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            producer_contents[relative] = content
        for command in (
            ["git", "init", "-q"],
            ["git", "config", "user.name", "Gauntlet Self Test"],
            ["git", "config", "user.email", "gauntlet@example.invalid"],
            ["git", "add", "--", *PRODUCER_PATHS],
            ["git", "commit", "-qm", "producer fixture"],
        ):
            result = subprocess.run(
                command,
                cwd=producer_repo,
                env=_git_environment(),
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RowError(
                    f"cannot construct producer-root self-test repository: {command!r}"
                )
        producer_head = current_git_head(os.fspath(producer_repo))
        producer_digest = gauntlet_producer_root(
            os.fspath(producer_repo), producer_head, verify_live=True
        )
        check(
            "producer-root-valid-committed-closure-accepted",
            re.fullmatch(r"[0-9a-f]{64}", producer_digest) is not None,
        )
        (producer_repo / "scripts/gauntlet_row.py").write_text(
            "validator tamper\n", encoding="utf-8"
        )
        try:
            gauntlet_producer_root(
                os.fspath(producer_repo), producer_head, verify_live=True
            )
            check("producer-root-validator-tamper-rejected", False)
        except RowError:
            check("producer-root-validator-tamper-rejected", True)
        (producer_repo / "scripts/gauntlet_row.py").write_bytes(
            producer_contents["scripts/gauntlet_row.py"]
        )
        (producer_repo / "scripts/gauntlet_runbook.sh").write_text(
            "config tamper\n", encoding="utf-8"
        )
        try:
            gauntlet_producer_root(
                os.fspath(producer_repo), producer_head, verify_live=True
            )
            check("producer-root-config-tamper-rejected", False)
        except RowError:
            check("producer-root-config-tamper-rejected", True)
        try:
            _canonical_allowed_evidence_path("docs/PERF_LEDGER.md")
            check("allowed-evidence-path-refuses-ledger-exception", False)
        except RowError:
            check("allowed-evidence-path-refuses-ledger-exception", True)

        escape_target = Path(tmp) / "outside"
        escape_target.mkdir()
        symlink_component = containment_root / "escape"
        symlink_component.symlink_to(escape_target, target_is_directory=True)
        try:
            validate_evidence_destination(
                symlink_component / "claim", perf_root=containment_root
            )
            check("refuses-evidence-symlink-escape", False)
        except RowError:
            check("refuses-evidence-symlink-escape", True)

        nonempty_destination = containment_root / "nonempty"
        nonempty_destination.mkdir()
        (nonempty_destination / "existing.json").write_text("{}\n", encoding="utf-8")
        try:
            write_bundle(
                os.fspath(nonempty_destination),
                focr_path="unread-focr.json",
                ref_path="unread-ref.json",
                roofline_path="unread-roofline.json",
                correctness={},
                focr={},
                ref={},
                build_provenance={},
                reference_provenance={},
                rows=[],
                source_git_head="0123456789abcdef0123456789abcdef01234567",
                source_root="2" * 64,
                producer_root="3" * 64,
                allowed_evidence_path="artifacts/perf/nonempty",
                perf_root=containment_root,
            )
            check("refuses-nonempty-evidence-destination", False)
        except RowError:
            check(
                "refuses-nonempty-evidence-destination",
                (nonempty_destination / "existing.json").read_text(encoding="utf-8")
                == "{}\n",
            )

        try:
            build_row(
                **{**kwargs, "correctness_proof": "FAILED: token mismatch"},
                allow_synthetic=True,
            )
            check("refuses-failing-correctness-proof", False)
        except RowError:
            check("refuses-failing-correctness-proof", True)

        for exploit in (
            "outputs differ from reference; CER=0.91",
            "token_exact=false; parity=false",
            "wrong tokens on 99 percent of pages",
            "no failures",
        ):
            try:
                build_row(
                    **{**kwargs, "correctness_proof": exploit},
                    allow_synthetic=True,
                )
                check("refuses-unstructured-correctness-claim", False, claim=exploit)
            except RowError:
                check("refuses-unstructured-correctness-claim", True, claim=exploit)

        # The REAL check_ledgers.py accepts the candidate row in a shadow repo…
        passed, output = shadow_check(
            [row_markdown(row)], evidence_dir, "artifacts/perf/bd-re8.17/selftest"
        )
        check("shadow-check-passes", passed, detail=None if passed else output[-400:])

        # …and rejects one whose correctness_proof cell is empty.
        broken = dict(row)
        broken["correctness_proof"] = ""
        broken_md = row_markdown(broken)
        rejected, _output = shadow_check(
            [broken_md], evidence_dir, "artifacts/perf/bd-re8.17/selftest"
        )
        check("shadow-check-rejects-empty-proof", not rejected)

        # …and rejects a row whose evidence dir has no SHA256SUMS manifest.
        os.remove(manifest)
        rejected, _output = shadow_check(
            [row_markdown(row)], evidence_dir, "artifacts/perf/bd-re8.17/selftest"
        )
        check("shadow-check-rejects-missing-manifest", not rejected)

    if failures:
        print(f"SELF-TEST FAILED: {failures}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "gauntlet-row-self-test", "result": "pass"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--focr-stages")
    parser.add_argument("--ref-stages")
    parser.add_argument("--roofline")
    parser.add_argument("--stage", action="append", choices=LEDGER_STAGES, default=None)
    parser.add_argument("--claim-id")
    parser.add_argument("--fixture-hash")
    parser.add_argument("--arch-features")
    parser.add_argument("--correctness-proof")
    parser.add_argument("--model-commit", default=MODEL_COMMIT)
    parser.add_argument("--notes", default="")
    parser.add_argument("--evidence-dir", default=None)
    parser.add_argument(
        "--apply", action="store_true", help="insert into docs/PERF_LEDGER.md"
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    required = (
        "focr_stages",
        "ref_stages",
        "roofline",
        "claim_id",
        "fixture_hash",
        "arch_features",
        "correctness_proof",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(
            f"missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}"
        )
    try:
        return run(args)
    except RowError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
