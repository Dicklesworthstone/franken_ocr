#!/usr/bin/env python3
"""Reference implementation of the franken_ocr three-pillar release gauntlet math.

This is the executable backing for ``docs/gauntlet/METHODOLOGY.md`` (beads
VERIFY-three-pillar-cert / VERIFY-conformal-ratchet / VERIFY-eprocess-invariants,
= bd-re8.13/14/15). It is stdlib-only (no numpy, no torch) so it runs in CI
without model weights, mirroring scripts/check_ledgers.py.

It implements, exactly as the methodology specifies:

  (1) The conformance parity score: a per-category Beta posterior, a
      distribution-free conformal band, ``truncate_score`` to 6 dp, and the
      release-decision rule that uses the LOWER bound (never the point estimate).

  (2) The ratchet state machine: Allow / Block / Quarantine / Waiver against a
      persisted, monotone, per-category high-water mark.

  (3) The Ville e-process monitor for the four load-bearing invariants
      (KV-cap L*(m+128), i32 no-overflow at K=6848, same-input determinism,
      SIMD==scalar bit-identical), with the hardware/software p0/lambda/alpha
      calibration split and the arithmetic-mean global e-value.

Nothing here ships in the ``focr`` binary; like the PyO3 oracle bridge this is a
verification-only artifact (G3's no-FFI runtime claim is preserved).

Usage:
    python3 scripts/gauntlet_cert.py --self-test     # validate the math (CI gate)
    python3 scripts/gauntlet_cert.py --demo          # print a worked franken_ocr cert
    # bd-re8.13 real-data modes:
    python3 scripts/gauntlet_cert.py --from-parity docs/FEATURE_PARITY.md \
        --scorecard-out docs/gauntlet/RELEASE_SCORECARD.json   # score the REAL scoreboard
    python3 scripts/gauntlet_cert.py --convergence   # >=10 rounds / >=2 clean gate (exit 1 until met)
"""

from __future__ import annotations

import argparse
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
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    from gauntlet_row import (
        CORRECTNESS_INPUTS_SCHEMA,
        RowError as CorrectnessValidationError,
        _synthetic_correctness_receipt,
        _validate_correctness_receipt_payload,
        build_receipt_document,
    )
except ModuleNotFoundError:  # imported as scripts.gauntlet_cert
    from scripts.gauntlet_row import (  # type: ignore[no-redef]
        CORRECTNESS_INPUTS_SCHEMA,
        RowError as CorrectnessValidationError,
        _synthetic_correctness_receipt,
        _validate_correctness_receipt_payload,
        build_receipt_document,
    )

# --------------------------------------------------------------------------- #
# K-5 / §3.4 — truncate_score: cross-platform LSB determinism.
# x86 vs ARM vs WASM differ at the LSB of IEEE-754 double; truncate (NOT round)
# to 6 dp at every release boundary so the byte-wise ratchet diff never flickers.
# --------------------------------------------------------------------------- #

SCORE_DECIMALS = 6
_SCORE_SCALE = 10.0**SCORE_DECIMALS

CERTIFICATION_MAX_EVIDENCE_AGE_HOURS = 24.0
UNLIMITED_OCR_MODEL_COMMIT = "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"
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
STRICT_CERTIFICATE_SCHEMA = "strict-conformant-release.v1"
STRICT_CERTIFICATE_ARTIFACT = "franken_ocr.release_certificate.v1"
STRICT_BUNDLE_SCHEMA = "gauntlet.certification_bundle_manifest.v1"
STRICT_BUNDLE_ARTIFACT = "franken_ocr.certification_bundle.v1"
CERTIFICATION_REQUIRED_SIGNERS = 3
CERTIFICATION_MAX_BUNDLE_ARTIFACTS = 256
CERTIFICATION_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
# The citable source pack may reach ~1.43 GiB at its declared content/metadata
# bounds. Leave another 512 MiB for the remaining, individually bounded bundle
# artifacts while keeping total verification work finite.
CERTIFICATION_MAX_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024
CERTIFICATION_MAX_JSON_DEPTH = 128
RAW_TIMING_SCHEMA = "focr-gauntlet-raw-timing/v1"
PERF_MAX_RAW_TIMING_RUNS = 256
PERF_MAX_RAW_TIMING_STAGES = 64
PERF_MAX_RAW_TIMING_FILES = 1024
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
PERF_MAX_SOURCE_ENTRIES = 50_000
PERF_MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
PERF_MAX_SOURCE_TOTAL_BYTES = 1024 * 1024 * 1024
PERF_MAX_SOURCE_PACK_HEADER_BYTES = 32 * 1024
PERF_MAX_LOGICAL_PATH_BYTES = 4096
PERF_MAX_SOURCE_PACK_BYTES = (
    PERF_MAX_SOURCE_TOTAL_BYTES
    + PERF_MAX_SOURCE_ENTRIES * (2 * PERF_MAX_LOGICAL_PATH_BYTES + 1024)
    + PERF_MAX_SOURCE_PACK_HEADER_BYTES
)
PERF_MAX_SUBJECT_BINARY_BYTES = 1024 * 1024 * 1024
PERF_INPUT_BINDINGS = {
    "focr_stages": "focr_stages.json",
    "ref_stages": "ref_stages.json",
    "roofline": "roofline.json",
    "correctness_receipt": "correctness_receipt.json",
    "build_receipt": "subject/build_receipt.json",
    "source_input_manifest": "subject/source_input_manifest.json",
    "source_input_pack": "subject/source_input_pack.bin",
    "subject_binary": "subject/release-perf/focr",
    "reference_model_manifest": "reference_model_manifest.json",
    "reference_inference_binding": "reference_inference_binding.json",
}
UNLIMITED_REFERENCE_MODEL_FILES = (
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
UNLIMITED_REFERENCE_MODEL_INDEX = {
    "path": "model.safetensors.index.json",
    "total_size": 6_672_212_480,
    "weight_count": 2710,
    "shards": ["model-00001-of-000001.safetensors"],
}
CERTIFICATION_MAX_EVIDENCE_MANIFEST_ENTRIES = 4096
CERTIFICATION_REQUIRED_SIGNATURE_ROLES = {
    "producer",
    "independent-reviewer",
    "release-authorizer",
}
STRICT_CERTIFICATE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "template",
        "project",
        "version",
        "issued_at",
        "constants",
        "reference",
        "git_head",
        "evidence_git_head",
        "git_branch",
        "git_describe",
        "git_worktree",
        "readiness",
        "convergence",
        "required_pass_actuals",
        "high_severity_counterexamples",
        "parity_score",
        "evidence_classes",
        "claim_sources",
        "readiness_cell_set_sha256",
        "feature_universe_sha256",
        "feature_universe_definition_sha256",
        "evidence_bundle_sha256",
        "signed_claim_sha256",
        "signers",
        "detached_signatures",
        "certified",
        "refusal_reasons",
        "generated_by",
    }
)
MAX_CORRECTNESS_CER = 0.25
CI_ARTIFACT_BINDING_SCHEMA = "gauntlet.ci_artifact_file.v1"
CERTIFICATION_GITHUB_REPOSITORY = "Dicklesworthstone/franken_ocr"
CERTIFICATION_GITHUB_WORKFLOW = "CI"
CERTIFICATION_GITHUB_EVENT = "push"
CERTIFICATION_DIST_WORKFLOW = "dist"
DIST_RELEASE_INPUT_NAMES = (
    "RUST_TOOLCHAIN",
    "FRANKENTORCH_REV",
    "FRANKENSQLITE_REV",
    "ASUPERSYNC_REV",
)
CERTIFICATION_MODEL_PARITY_WORKFLOW = "Model Parity"
CERTIFICATION_MODEL_PARITY_EVENT = "workflow_dispatch"
CERTIFICATION_MODEL_PARITY_REQUIRED_JOBS = {"weighted-model-parity"}
CERTIFICATION_PERFORMANCE_WORKFLOW = "Performance Gauntlet"
CERTIFICATION_PERFORMANCE_EVENT = "workflow_dispatch"
CERTIFICATION_PERFORMANCE_REQUIRED_JOBS = {"weighted-performance-gauntlet"}
CERTIFICATION_MODEL_PARITY_MIN_ROWS = {
    "L0": 2,
    "L1": 6,
    "L2": 4,
    "L3": 2,
    "L4": 2,
    "L5": 4,
}
CERTIFICATION_MODEL_PARITY_CASES = {
    "L0": ("page_0009:sam_input", "page_0014:sam_input"),
    "L1": tuple(
        f"{page}:{seam}"
        for page in ("page_0009", "page_0014")
        for seam in ("sam_output", "clip_output", "projector_output")
    ),
    "L2": tuple(
        f"{page}:{seam}"
        for page in ("page_0009", "page_0014")
        for seam in ("projector_output", "inputs_embeds")
    ),
    "L3": ("page_0009:lm_head_logits", "page_0014:lm_head_logits"),
    "L4": ("page_0009:token_stream", "page_0014:token_stream"),
    "L5": (
        "page_0009:decoded_text",
        "page_0014:decoded_text",
        "p10:multi_page",
        "p9_p14:multi_page",
    ),
}
CERTIFICATION_MODEL_PARITY_MIN_PAYLOAD_ITEMS = {
    "L0": 1_024,
    "L1": 1_280,
    "L2": 1_280,
    "L3": 129_280,
    "L4": 32,
    "L5": 20,
}
CERTIFICATION_WORKFLOW_PATHS = {
    CERTIFICATION_GITHUB_WORKFLOW: ".github/workflows/ci.yml",
    CERTIFICATION_DIST_WORKFLOW: ".github/workflows/dist.yml",
    CERTIFICATION_MODEL_PARITY_WORKFLOW: ".github/workflows/model-parity.yml",
    CERTIFICATION_PERFORMANCE_WORKFLOW: ".github/workflows/performance-gauntlet.yml",
}
CERTIFICATION_CI_REQUIRED_JOBS = (
    "gate (macos-15)",
    "gate (ubuntu-latest)",
)
CERTIFICATION_DIST_TARGETS = (
    "aarch64-apple-darwin (neon+sdot+i8mm)",
    "x86_64-apple-darwin (baseline)",
    "aarch64-linux (baseline)",
    "x86_64-linux (baseline + runtime dispatch)",
    "x86_64-pc-windows-msvc (baseline)",
    "aarch64-pc-windows-msvc (baseline)",
)
CERTIFICATION_DIST_ASSETS = {
    "aarch64-apple-darwin (neon+sdot+i8mm)": "focr-aarch64-apple-darwin-neon-sdot-i8mm",
    "x86_64-apple-darwin (baseline)": "focr-x86_64-apple-darwin",
    "aarch64-linux (baseline)": "focr-aarch64-unknown-linux-gnu",
    "x86_64-linux (baseline + runtime dispatch)": "focr-x86_64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc (baseline)": "focr-x86_64-pc-windows-msvc.exe",
    "aarch64-pc-windows-msvc (baseline)": "focr-aarch64-pc-windows-msvc.exe",
}
CERTIFICATION_DIST_TRIPLES = {
    "aarch64-apple-darwin (neon+sdot+i8mm)": "aarch64-apple-darwin",
    "x86_64-apple-darwin (baseline)": "x86_64-apple-darwin",
    "aarch64-linux (baseline)": "aarch64-unknown-linux-gnu",
    "x86_64-linux (baseline + runtime dispatch)": "x86_64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc (baseline)": "x86_64-pc-windows-msvc",
    "aarch64-pc-windows-msvc (baseline)": "aarch64-pc-windows-msvc",
}
CERTIFICATION_LINUX_GLIBC_FLOOR = "2.17"
CERTIFICATION_LINUX_DIST_TARGETS = {
    "aarch64-linux (baseline)",
    "x86_64-linux (baseline + runtime dispatch)",
}
CERTIFICATION_WINDOWS_DIST_TARGETS = {
    "x86_64-pc-windows-msvc (baseline)",
    "aarch64-pc-windows-msvc (baseline)",
}
CERTIFICATION_AUDIT_DOMAINS = (
    "security",
    "correctness",
    "concurrency",
    "numerics",
    "release",
)
CERTIFICATION_AUDIT_TOOLS = {
    "security": {
        "ubs-python": "ubs --only=python --diff .",
        "bandit": "uvx bandit -r scripts",
    },
    "correctness": {
        "ledger-check": "python3 scripts/check_ledgers.py",
        "model-parity-ladder": "scripts/ladder_scorecard.sh --out scorecard.json",
    },
    "concurrency": {
        "many-pages-watchdog": "cargo test --test many_pages_without_deadlock",
        "cancel-panic-faults": "cargo test --test cancel_and_panic_faults",
    },
    "numerics": {
        "gauntlet-cert-self-test": "python3 scripts/gauntlet_cert.py --self-test",
        "int8-overflow-proof": "cargo test int8_i32_accumulation_worst_case",
    },
    "release": {
        "full-check": "scripts/check.sh",
        "dist-matrix": "gh workflow run dist.yml",
    },
}
CERTIFICATION_BENCHMARK_THRESHOLDS = {
    "primary_score": -3.0,
    "geomean": -5.0,
    "category_geomean": -10.0,
    "p90": -15.0,
    "throughput_drop": -5.0,
}
TRUSTED_SIGNERS_PATH = "docs/gauntlet/TRUSTED_SIGNERS.json"
TRUSTED_KEYRING_PATH = "docs/gauntlet/signing/trusted-release-keys.gpg"
HYPOTHESIS_LEDGER_PATHS = (
    "docs/gauntlet/GAUNTLET_EXPERIMENT_DESIGNS.md",
    "docs/gauntlet/PERF_HYPOTHESIS_LEDGER.md",
    "docs/gauntlet/CONFORMANCE_HYPOTHESIS_LEDGER.md",
    "docs/gauntlet/SURFACE_PARITY_HYPOTHESIS_LEDGER.md",
)
CERTIFICATION_READINESS_CELLS = (
    "parity_l0_l5",
    "surface_parity",
    "perf_vs_reference",
    "determinism",
    "deadlock_watchdog",
    "robot_schema",
    "build_matrix",
    "installer",
    "ledger_completeness",
    "hypothesis_ledgers",
    "agent_ergonomics",
    "doctor",
    "certification_bundle",
    "gauntlet_convergence",
)
CERTIFICATION_EXTERNAL_READINESS_CELLS = tuple(
    cell for cell in CERTIFICATION_READINESS_CELLS if cell != "certification_bundle"
)
CERTIFICATION_READINESS_EVIDENCE_PATHS = {
    "parity_l0_l5": ("tests/fixtures/ladder_scorecard/scorecard_armed.json",),
    "surface_parity": ("docs/FEATURE_PARITY.md",),
    "perf_vs_reference": ("docs/PERF_LEDGER.md", "scripts/check_ledgers.py"),
    "determinism": ("docs/gauntlet/EPROCESS_STATE.json",),
    "deadlock_watchdog": (
        "tests/many_pages_without_deadlock.rs",
        "tests/cancel_and_panic_faults.rs",
    ),
    "robot_schema": (
        "tests/fixtures/robot_schema_v1.json",
        "tests/fixtures/runs_schema.json",
        "tests/cli_robot_golden.rs",
        "tests/surface_matrix.rs",
    ),
    "build_matrix": tuple(CERTIFICATION_WORKFLOW_PATHS.values()),
    "installer": ("install.sh", "install.ps1", "tests/installer_e2e.sh"),
    "ledger_completeness": (
        "scripts/check_ledgers.py",
        "docs/DISCREPANCIES.md",
        "docs/NEGATIVE_EVIDENCE.md",
        "docs/PERF_LEDGER.md",
    ),
    "hypothesis_ledgers": HYPOTHESIS_LEDGER_PATHS,
    "agent_ergonomics": (
        "docs/ergonomics/AUDIT.md",
        "tests/agent_ergonomics_regression.rs",
    ),
    "doctor": ("src/doctor.rs", "tests/doctor_fixtures.rs"),
    "gauntlet_convergence": ("docs/gauntlet/ROUNDS.jsonl", *HYPOTHESIS_LEDGER_PATHS),
}
CERTIFICATION_PROOF_EVIDENCE_PATHS = tuple(
    dict.fromkeys(
        path
        for cell in CERTIFICATION_EXTERNAL_READINESS_CELLS
        for path in CERTIFICATION_READINESS_EVIDENCE_PATHS[cell]
    )
)
CERTIFICATION_FEATURE_UNIVERSE = (
    {
        "feature_id": "conformance",
        "required": True,
        "proof_obligations": ["parity_l0_l5", "surface_parity"],
    },
    {
        "feature_id": "performance",
        "required": True,
        "proof_obligations": ["perf_vs_reference"],
    },
    {
        "feature_id": "runtime-invariants",
        "required": True,
        "proof_obligations": ["determinism", "deadlock_watchdog"],
    },
    {
        "feature_id": "agent-contract",
        "required": True,
        "proof_obligations": ["robot_schema", "agent_ergonomics", "doctor"],
    },
    {
        "feature_id": "distribution",
        "required": True,
        "proof_obligations": ["build_matrix", "installer"],
    },
    {
        "feature_id": "evidence-integrity",
        "required": True,
        "proof_obligations": [
            "ledger_completeness",
            "hypothesis_ledgers",
            "gauntlet_convergence",
        ],
    },
)
CERTIFICATION_READINESS_CELL_SET_SHA256 = hashlib.sha256(
    json.dumps(
        CERTIFICATION_READINESS_CELLS,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()
CERTIFICATION_FEATURE_UNIVERSE_SHA256 = hashlib.sha256(
    json.dumps(
        CERTIFICATION_FEATURE_UNIVERSE,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()
CERTIFICATION_CONSTANTS = {
    "CERTIFICATION_MIN_VERIFICATION_PCT": 100.0,
    "CERTIFICATION_REQUIRED_SUITE_PASS_RATE_PCT": 100.0,
    "CERTIFICATION_MAX_HIGH_SEVERITY_COUNTEREXAMPLES": 0,
    "CERTIFICATION_MAX_EVIDENCE_AGE_HOURS": CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
}
CORE_EVIDENCE_PATHS = (
    "docs/gauntlet/RELEASE_READINESS.json",
    "docs/gauntlet/ROUNDS.jsonl",
    "docs/gauntlet/EPROCESS_STATE.json",
    "docs/gauntlet/RELEASE_SCORECARD.json",
    "tests/fixtures/ladder_scorecard/scorecard_armed.json",
    "benches/.bench-history/baseline.json",
    "docs/DISCREPANCIES.md",
    "docs/NEGATIVE_EVIDENCE.md",
    "docs/PERF_LEDGER.md",
    "docs/FEATURE_PARITY.md",
    "scripts/gauntlet_cert.py",
    "scripts/gauntlet_row.py",
    "scripts/gauntlet_reference.py",
    *HYPOTHESIS_LEDGER_PATHS,
)
EVIDENCE_MANIFEST_NAMES = (
    "SHA256SUMS",
    "SHA256SUMS.txt",
    "sha256sums.txt",
    "sha256.txt",
    "manifest.sha256",
    "manifest.json",
)
STRICT_BUNDLE_CLASSES = {
    "confidence_gate": ("confidence_gate.json", "confidence_gate.v1"),
    "verification_contract": (
        "verification_contract.json",
        "gauntlet.verification_contract.v1",
    ),
    "ci_manifest": ("ci_manifest.json", "gauntlet.ci_artifact_manifest.v1"),
    "ci_gate_receipt": ("ci_gate_receipt.json", "gauntlet.ci_gate_receipt.v1"),
    "model_parity_receipt": (
        "model_parity_receipt.json",
        "gauntlet.model_parity_receipt.v1",
    ),
    "dist_matrix_receipt": (
        "dist_matrix_receipt.json",
        "gauntlet.dist_matrix_receipt.v1",
    ),
    "benchmark_summary": ("benchmark_summary.json", "gauntlet.benchmark_summary.v1"),
    "scorecards": ("scorecards.json", "gauntlet.scorecards.v1"),
    "critical_path_report": (
        "critical_path_report.json",
        "gauntlet.critical_path_report.v1",
    ),
    "critical_path_inventory": (
        "critical_path_inventory.json",
        "gauntlet.critical_path_inventory.v1",
    ),
    "ratchet_state": ("ratchet_state.json", "ratchet_state.v1"),
}
_RETRY_CONDITION_PREFIXES = (
    "retry only if a profiler attributes a clearly-above-noise share to ",
    "reconsider only inside the broader ",
    "worth reconsidering when ",
    "not worth retrying as a standalone patch",
    "do not retry from a cold read; use comprehensive-bench attribution instead",
    "retry condition not applicable",
    "retry only if this workload class exhibits measurable ",
    "blocked until ",
)
_CORRECTNESS_CLAIM_RE = re.compile(
    r"receipt=correctness_receipt\.json "
    r"sha256=(?P<sha256>[0-9a-f]{64}) "
    r"metric=cer_norm value=(?P<value>[0-9]+\.[0-9]{6}) "
    r"max=(?P<maximum>[0-9]+\.[0-9]{6}) result=pass"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_env() -> dict[str, str]:
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


def _git_output(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            env=_git_env(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_blob_sha256(root: Path, git_head: str, relative: str) -> str | None:
    if not _valid_git_head(git_head):
        return None
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or relative_path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        return None
    try:
        result = subprocess.run(
            ["git", "show", f"{git_head}:{relative}"],
            cwd=root,
            env=_git_env(),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def _git_blob_identity(root: Path, git_head: str, relative: str) -> dict | None:
    if not _valid_git_head(git_head):
        return None
    try:
        result = subprocess.run(
            ["git", "show", f"{git_head}:{relative}"],
            cwd=root,
            env=_git_env(),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or len(result.stdout) > PERF_MAX_SOURCE_FILE_BYTES:
        return None
    return {
        "size": len(result.stdout),
        "sha256": hashlib.sha256(result.stdout).hexdigest(),
    }


def _gauntlet_producer_root(root: Path, source_git_head: str) -> str | None:
    digest = hashlib.sha256(PRODUCER_ROOT_DOMAIN)
    for relative in PRODUCER_PATHS:
        identity = _git_blob_identity(root, source_git_head, relative)
        if identity is None:
            return None
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(identity["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(identity["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_allowed_evidence_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\0" in value:
        return None
    path = Path(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) < 3
        or path.parts[:2] != ("artifacts", "perf")
    ):
        return None
    return value


def _evidence_descendant_reasons(
    root: Path,
    source_git_head: str,
    evidence_git_head: str,
    allowed_evidence_path: object,
) -> list[str]:
    """Require every commit after source to touch only one declared evidence tree."""
    allowed = _canonical_allowed_evidence_path(allowed_evidence_path)
    if allowed is None:
        return ["row allowed_evidence_path is missing or noncanonical"]
    if not _valid_git_head(source_git_head) or not _valid_git_head(evidence_git_head):
        return ["source/evidence git HEAD binding is noncanonical"]
    if source_git_head == evidence_git_head:
        return []
    try:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_git_head, evidence_git_head],
            cwd=root,
            env=_git_env(),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [f"cannot verify source/evidence ancestry: {error}"]
    if ancestor.returncode != 0:
        return ["row source_git_head is not an ancestor of the evidence git HEAD"]
    try:
        commits_result = subprocess.run(
            ["git", "rev-list", "--reverse", f"{source_git_head}..{evidence_git_head}"],
            cwd=root,
            env=_git_env(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [f"cannot enumerate source/evidence descendants: {error}"]
    commits = commits_result.stdout.splitlines() if commits_result.returncode == 0 else []
    if not commits or any(not _valid_git_head(commit) for commit in commits):
        return ["source/evidence descendant commit range is unavailable"]
    touched: set[str] = set()
    for commit in commits:
        try:
            changed = subprocess.run(
                [
                    "git",
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "--no-renames",
                    "-r",
                    "-m",
                    "-z",
                    commit,
                ],
                cwd=root,
                env=_git_env(),
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return [f"cannot inspect evidence descendant {commit}: {error}"]
        if changed.returncode != 0:
            return [f"cannot inspect evidence descendant {commit}"]
        try:
            paths = [
                item.decode("utf-8")
                for item in changed.stdout.split(b"\0")
                if item
            ]
        except UnicodeDecodeError:
            return [f"evidence descendant {commit} contains a non-UTF-8 path"]
        touched.update(paths)
    outside = sorted(
        relative
        for relative in touched
        if relative != allowed and not relative.startswith(allowed + "/")
    )
    if outside:
        return [
            "source/evidence descendants changed paths outside "
            f"{allowed}: " + ", ".join(outside)
        ]
    if not touched:
        return ["source/evidence descendant range contains no evidence changes"]
    return []


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_read_bounded_file(path)).hexdigest()


def _readonly_binary_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _read_bounded_file(
    path: Path, max_bytes: int = CERTIFICATION_MAX_ARTIFACT_BYTES
) -> bytes:
    """Read one stable regular-file snapshot through a single bounded fd."""
    if max_bytes < 0:
        raise ValueError("file size limit must be nonnegative")
    descriptor = os.open(path, _readonly_binary_flags())
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"input is not a regular file: {path}")
        if before.st_size > max_bytes:
            raise ValueError(f"input exceeds the {max_bytes}-byte size limit: {path}")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > max_bytes:
                raise ValueError(
                    f"input exceeds the {max_bytes}-byte size limit while reading: {path}"
                )
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"input changed while being read: {path}")
        content = b"".join(chunks)
        if len(content) != before.st_size:
            raise ValueError(f"input length changed while being read: {path}")
        return content
    finally:
        os.close(descriptor)


def _stream_file_identity(
    path: Path,
    max_bytes: int,
    *,
    snapshot_path: Path | None = None,
) -> dict:
    """Hash and optionally snapshot one stable regular file without buffering it."""
    if max_bytes < 0:
        raise ValueError("file size limit must be nonnegative")
    descriptor = os.open(path, _readonly_binary_flags())
    snapshot_descriptor: int | None = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"input is not a regular file: {path}")
        if before.st_size > max_bytes:
            raise ValueError(f"input exceeds the {max_bytes}-byte size limit: {path}")
        if snapshot_path is not None:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            snapshot_descriptor = os.open(snapshot_path, snapshot_flags, 0o600)

        digest = hashlib.sha256()
        observed = 0
        while observed < before.st_size:
            chunk = os.read(
                descriptor, min(1024 * 1024, before.st_size - observed)
            )
            if not chunk:
                raise ValueError(f"input length changed while streaming: {path}")
            digest.update(chunk)
            observed += len(chunk)
            if snapshot_descriptor is not None:
                pending = memoryview(chunk)
                while pending:
                    written = os.write(snapshot_descriptor, pending)
                    if written <= 0:
                        raise OSError("snapshot write made no progress")
                    pending = pending[written:]
        if os.read(descriptor, 1):
            raise ValueError(f"input grew while streaming: {path}")
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"input changed while streaming: {path}")
        if observed != before.st_size:
            raise ValueError(f"input length changed while streaming: {path}")
        return {
            "sha256": digest.hexdigest(),
            "size": observed,
            "mtime": before.st_mtime,
        }
    finally:
        if snapshot_descriptor is not None:
            os.close(snapshot_descriptor)
        os.close(descriptor)


def _streamed_binary_timestamp(
    identity: dict,
    certification_time: datetime,
    *,
    static_head_source: bool,
) -> tuple[datetime, str]:
    if static_head_source:
        return certification_time, "source:git-head+fresh-ci"
    mtime = identity.get("mtime")
    if not isinstance(mtime, (int, float)) or isinstance(mtime, bool):
        raise ValueError("streamed artifact has no valid descriptor timestamp")
    return datetime.fromtimestamp(float(mtime), timezone.utc), "filesystem:mtime"


def _read_bounded_text(
    path: Path, max_bytes: int = CERTIFICATION_MAX_ARTIFACT_BYTES
) -> str:
    try:
        return _read_bounded_file(path, max_bytes).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"input is not valid UTF-8: {path}: {error}") from error


def _json_depth(value: object, max_depth: int = CERTIFICATION_MAX_JSON_DEPTH) -> int:
    """Return JSON container depth while refusing recursive/pathological values."""
    stack: list[tuple[object, int]] = [(value, 0)]
    observed = 0
    while stack:
        current, depth = stack.pop()
        observed = max(observed, depth)
        if observed > max_depth:
            raise ValueError(f"JSON nesting exceeds the depth limit of {max_depth}")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return observed


def _parse_json_bytes(content: bytes, *, label: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            content.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
        _json_depth(payload)
        return payload
    except (UnicodeDecodeError, ValueError, TypeError, RecursionError) as error:
        raise ValueError(f"{label} is malformed or too deeply nested: {error}") from error


def _parse_strict_json_object(content: bytes, *, label: str) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            content.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
        _json_depth(payload)
    except (UnicodeDecodeError, ValueError, TypeError, RecursionError) as error:
        raise ValueError(f"{label} is not strict bounded JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def _is_canonical_utc_seconds(value: object) -> bool:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _load_bounded_json(
    path: Path, max_bytes: int = CERTIFICATION_MAX_ARTIFACT_BYTES
) -> object:
    return _parse_json_bytes(_read_bounded_file(path, max_bytes), label=str(path))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_exact(left: object, right: object) -> bool:
    try:
        return json.dumps(
            left, allow_nan=False, separators=(",", ":"), sort_keys=True
        ) == json.dumps(right, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError, RecursionError):
        return False


def _finite_number(
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return (
        math.isfinite(number)
        and (minimum is None or number >= minimum)
        and (maximum is None or number <= maximum)
    )


def _levenshtein_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _derived_parity_metrics(
    rung: str, oracle_payload: object, subject_payload: object
) -> tuple[dict | None, bool]:
    if rung == "L0":
        metrics = {"mismatch_count": 0 if oracle_payload == subject_payload else 1}
        return metrics, metrics["mismatch_count"] == 0
    if rung in {"L1", "L2", "L3"}:
        if (
            not isinstance(oracle_payload, list)
            or not isinstance(subject_payload, list)
            or len(oracle_payload) != len(subject_payload)
            or not oracle_payload
            or len(oracle_payload) > 1_000_000
            or any(
                not _finite_number(value, minimum=-1e6, maximum=1e6)
                for value in oracle_payload
            )
            or any(
                not _finite_number(value, minimum=-1e6, maximum=1e6)
                for value in subject_payload
            )
        ):
            return None, False
        try:
            dot = math.fsum(
                float(left) * float(right)
                for left, right in zip(oracle_payload, subject_payload, strict=True)
            )
            oracle_norm = math.sqrt(
                math.fsum(float(value) ** 2 for value in oracle_payload)
            )
            subject_norm = math.sqrt(
                math.fsum(float(value) ** 2 for value in subject_payload)
            )
            cosine = dot / (oracle_norm * subject_norm)
        except (OverflowError, ValueError, ZeroDivisionError):
            return None, False
        max_abs_diff = max(
            abs(float(left) - float(right))
            for left, right in zip(oracle_payload, subject_payload, strict=True)
        )
        metrics = {
            "cosine_min": truncate_score(cosine),
            "max_abs_diff": truncate_score(max_abs_diff),
        }
        passed = _finite_number(cosine, minimum=0.9999, maximum=1.000001)
        if rung == "L3":
            oracle_argmax = max(
                range(len(oracle_payload)), key=oracle_payload.__getitem__
            )
            subject_argmax = max(
                range(len(subject_payload)), key=subject_payload.__getitem__
            )
            metrics["argmax_exact"] = oracle_argmax == subject_argmax
            passed = passed and metrics["argmax_exact"]
        return metrics, passed
    if rung == "L4":
        if (
            not isinstance(oracle_payload, list)
            or not isinstance(subject_payload, list)
            or len(oracle_payload) > 1_000_000
            or len(subject_payload) > 1_000_000
        ):
            return None, False
        metrics = {
            "token_exact": oracle_payload == subject_payload,
            "tokens_compared": len(oracle_payload),
        }
        return metrics, metrics["token_exact"] and metrics["tokens_compared"] > 0
    if rung == "L5":
        if (
            not isinstance(oracle_payload, str)
            or not isinstance(subject_payload, str)
            or len(oracle_payload) > 100_000
            or len(subject_payload) > 100_000
            or len(oracle_payload) * len(subject_payload) > 10_000_000
        ):
            return None, False
        denominator = max(1, len(oracle_payload))
        cer = _levenshtein_distance(oracle_payload, subject_payload) / denominator
        metrics = {"cer_norm": truncate_score(cer), "characters": len(oracle_payload)}
        return metrics, cer <= 0.01 and len(oracle_payload) > 0
    return None, False


def _benchmark_metrics(payload: object) -> dict[str, float] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), dict):
        return None
    stages = payload["stages"]
    required_stages = {"vision_encode", "decode_per_token", "end_to_end"}
    if not required_stages <= set(stages):
        return None
    stage_best: dict[str, float] = {}
    stage_means: dict[str, float] = {}
    stage_samples: dict[str, list[float]] = {}
    for stage_name, record in stages.items():
        if not isinstance(stage_name, str) or not isinstance(record, dict):
            return None
        samples = record.get("samples_ms")
        if (
            not isinstance(samples, list)
            or len(samples) < 3
            or len(samples) > 10_000
            or any(
                not _finite_number(sample, minimum=1e-12, maximum=1e12)
                for sample in samples
            )
        ):
            return None
        numeric = [float(sample) for sample in samples]
        try:
            mean = statistics.fmean(numeric)
            best = min(numeric)
        except (OverflowError, ValueError, statistics.StatisticsError):
            return None
        if (
            not _finite_number(mean, minimum=1e-12, maximum=1e12)
            or not _finite_number(best, minimum=1e-12, maximum=1e12)
            or not _finite_number(record.get("best_ms"), minimum=1e-12, maximum=1e12)
            or not math.isclose(float(record["best_ms"]), best, rel_tol=1e-9)
        ):
            return None
        stage_best[stage_name] = best
        stage_means[stage_name] = mean
        stage_samples[stage_name] = numeric
    try:
        all_best = list(stage_best.values())
        category_means = [stage_means[name] for name in sorted(required_stages)]
        end_to_end = sorted(stage_samples["end_to_end"])
        p90_index = max(0, math.ceil(0.9 * len(end_to_end)) - 1)
        metrics = {
            "primary_score": stage_means["end_to_end"],
            "throughput_drop": stage_means["decode_per_token"],
            "geomean": math.exp(
                statistics.fmean(math.log(value) for value in all_best)
            ),
            "category_geomean": math.exp(
                statistics.fmean(math.log(value) for value in category_means)
            ),
            "p90": end_to_end[p90_index],
        }
    except (
        OverflowError,
        ValueError,
        ZeroDivisionError,
        statistics.StatisticsError,
    ):
        return None
    return (
        metrics
        if all(
            _finite_number(value, minimum=1e-12, maximum=1e12)
            for value in metrics.values()
        )
        else None
    )


def _safe_repo_path(root: Path, relative: str) -> Path | None:
    try:
        candidate = (root / relative).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _safe_output_dir(root: Path, out_dir: str) -> tuple[Path | None, list[str]]:
    reasons: list[str] = []
    raw = Path(out_dir)
    candidate = raw if raw.is_absolute() else root / raw
    current = candidate
    while current != root and current != current.parent:
        if current.is_symlink():
            reasons.append(f"bundle output path traverses a symlink: {current}")
            break
        current = current.parent
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        reasons.append(f"bundle output path cannot be resolved: {error}")
        return None, reasons
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        reasons.append("bundle output path escapes the repository")
    git_dir = (root / ".git").resolve()
    if resolved == root.resolve() or resolved == git_dir or git_dir in resolved.parents:
        reasons.append("bundle output path overlaps the repository root or .git")
    if resolved.exists() and not resolved.is_dir():
        reasons.append("bundle output path collides with a non-directory")
    elif resolved.is_dir():
        try:
            for child in resolved.rglob("*"):
                if child.is_symlink():
                    reasons.append(f"bundle output contains a symlink: {child}")
        except OSError as error:
            reasons.append(f"bundle output directory cannot be inspected: {error}")
    for relative in CORE_EVIDENCE_PATHS:
        evidence = _safe_repo_path(root, relative)
        if evidence is not None and (
            resolved == evidence or resolved in evidence.parents
        ):
            reasons.append(
                f"bundle output path would overwrite input evidence: {relative}"
            )
    return (None if reasons else resolved), reasons


def _safe_generated_output_dir(
    root: Path, out_dir: str
) -> tuple[Path | None, list[str]]:
    output, reasons = _safe_output_dir(root, out_dir)
    if output is None:
        return None, reasons
    ignored_root = (root / ".gauntlet-output").resolve()
    try:
        descendant = output.relative_to(ignored_root)
    except ValueError:
        reasons.append("generated certificate output must be under .gauntlet-output")
        return None, reasons
    if not descendant.parts:
        reasons.append("generated certificate output must be a child of .gauntlet-output")
        return None, reasons
    probe = (Path(".gauntlet-output") / descendant / "release_certificate.json").as_posix()
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", "--", probe],
            cwd=root,
            env=_git_env(),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        reasons.append(f"cannot verify .gauntlet-output ignore contract: {error}")
        return None, reasons
    if ignored.returncode != 0:
        reasons.append("generated certificate output is not git-ignored")
    return (None if reasons else output), reasons


def _git_worktree_state(root: Path) -> dict:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            env=_git_env(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        message = f"git status failed: {error}"
        return {
            "status_ok": False,
            "clean": False,
            "dirty_path_count": None,
            "status_porcelain": [],
            "status_sha256": _sha256_text(message),
            "status_error": message,
        }
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "git status failed").strip()
        return {
            "status_ok": False,
            "clean": False,
            "dirty_path_count": None,
            "status_porcelain": [],
            "status_sha256": _sha256_text(message),
            "status_error": message,
        }
    lines = result.stdout.splitlines()
    return {
        "status_ok": True,
        "clean": not lines,
        "dirty_path_count": len(lines),
        "status_porcelain": lines,
        "status_sha256": _sha256_text("\n".join(lines) + ("\n" if lines else "")),
    }


def _cargo_package_version(root: Path) -> str | None:
    try:
        with (root / "Cargo.toml").open("rb") as handle:
            package = tomllib.load(handle).get("package")
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError):
        return None
    if not isinstance(package, dict):
        return None
    version = package.get("version")
    return version if isinstance(version, str) and version else None


def _valid_git_head(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _timestamp_text(parsed: datetime) -> str:
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _artifact_native_timestamp(
    path: Path, content: bytes | None = None
) -> tuple[datetime, str]:
    """Return a native structured timestamp, using mtime only for plain files."""
    timestamp_keys = (
        "issued_at",
        "generated_at_utc",
        "generated_utc",
        "created_utc",
        "timestamp",
        "date",
    )
    if path.suffix == ".json":
        try:
            payload = _parse_json_bytes(
                content if content is not None else _read_bounded_file(path),
                label=str(path),
            )
        except (OSError, ValueError, TypeError, RecursionError) as error:
            raise ValueError(f"JSON artifact is unreadable: {error}") from error
        if isinstance(payload, dict):
            native_timestamps: list[tuple[str, datetime]] = []
            for key in timestamp_keys:
                if key not in payload:
                    continue
                parsed = _parse_utc_timestamp(payload.get(key))
                if parsed is None:
                    raise ValueError(
                        f"JSON artifact has invalid native timestamp: {key}"
                    )
                native_timestamps.append((key, parsed))
            if native_timestamps:
                distinct = {parsed for _key, parsed in native_timestamps}
                if len(distinct) != 1:
                    raise ValueError("JSON artifact has conflicting native timestamps")
                keys = "+".join(key for key, _parsed in native_timestamps)
                return native_timestamps[0][1], f"json:{keys}"
        raise ValueError("JSON artifact has no valid native timestamp")
    if path.suffix == ".jsonl":
        try:
            native: list[datetime] = []
            text = (
                content.decode("utf-8")
                if content is not None
                else _read_bounded_text(path)
            )
            for line in text.splitlines():
                if not line.strip():
                    continue
                payload = _parse_json_bytes(line.encode("utf-8"), label=str(path))
                if not isinstance(payload, dict):
                    raise ValueError("JSONL row is not an object")
                row_timestamps: list[tuple[str, datetime]] = []
                for key in timestamp_keys:
                    if key not in payload:
                        continue
                    parsed = _parse_utc_timestamp(payload.get(key))
                    if parsed is None:
                        raise ValueError(
                            f"JSONL row has invalid native timestamp: {key}"
                        )
                    row_timestamps.append((key, parsed))
                if not row_timestamps:
                    raise ValueError("JSONL row has no valid native timestamp")
                if len({parsed for _key, parsed in row_timestamps}) != 1:
                    raise ValueError("JSONL row has conflicting native timestamps")
                native.append(row_timestamps[0][1])
            if native:
                return max(native), "jsonl:native"
        except (OSError, ValueError, TypeError, RecursionError) as error:
            raise ValueError(f"JSONL artifact is invalid: {error}") from error
        raise ValueError("JSONL artifact contains no timestamped rows")
    return datetime.fromtimestamp(
        path.stat().st_mtime, timezone.utc
    ), "filesystem:mtime"


def _manifest_timestamp(
    path: Path,
    content: bytes,
    certification_time: datetime,
    *,
    static_head_source: bool,
) -> tuple[datetime, str]:
    if static_head_source:
        return certification_time, "source:git-head+fresh-ci"
    if path.suffix not in {".json", ".jsonl"}:
        return certification_time, "source:fresh-ci-artifact"
    return _artifact_native_timestamp(path, content)


def _age_reason(
    timestamp: datetime | None,
    now: datetime,
    label: str,
    max_age_hours: float = CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
) -> str | None:
    if timestamp is None:
        return f"{label} has no valid timestamp"
    age_hours = (now - timestamp).total_seconds() / 3600.0
    if age_hours < -5.0 / 60.0:
        return f"{label} timestamp is in the future"
    if age_hours > max_age_hours:
        return f"{label} is stale ({age_hours:.1f}h > {max_age_hours:.0f}h)"
    return None


def _correctness_claim(value: object) -> tuple[dict | None, str | None]:
    match = _CORRECTNESS_CLAIM_RE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return None, "correctness proof is not a structured hash-bound pass receipt"
    measured = float(match.group("value"))
    maximum = float(match.group("maximum"))
    if (
        not math.isfinite(measured)
        or not math.isfinite(maximum)
        or maximum != MAX_CORRECTNESS_CER
        or not 0.0 <= measured <= maximum
    ):
        return None, "correctness receipt claims an invalid or out-of-budget CER"
    return {
        "sha256": match.group("sha256"),
        "cer_norm": measured,
        "maximum": maximum,
    }, None


def truncate_score(x: float) -> float:
    """Truncate to 6 decimal places (associative across the ULP; round-mode-free)."""
    if not _finite_number(x, minimum=-1e12, maximum=1e12):
        raise ValueError("score must be finite and bounded")
    return math.floor(x * _SCORE_SCALE) / _SCORE_SCALE


# --------------------------------------------------------------------------- #
# §3.1 — Beta posterior per category (Jeffreys-like uniform prior).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BetaParams:
    """A Beta(alpha, beta) posterior over a per-category pass rate."""

    alpha: float
    beta: float

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1.0))

    def quantile(self, p: float) -> float:
        """Quantile of Beta(alpha, beta) via bisection on the regularized
        incomplete beta function (stdlib-only; no scipy)."""
        if not 0.0 < p < 1.0:
            return 0.0 if p <= 0.0 else 1.0
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _reg_inc_beta(mid, self.alpha, self.beta) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def credible_interval(self, confidence: float) -> tuple[float, float]:
        tail = (1.0 - confidence) / 2.0
        return (self.quantile(tail), self.quantile(1.0 - tail))


def _reg_inc_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) via the Lentz continued fraction.

    Standard Numerical-Recipes ``betai`` translated to pure Python; accurate to
    well past the 6-dp truncation floor we round to anyway.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    # Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) for fast CF convergence.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


# --------------------------------------------------------------------------- #
# §3.1 — per-category evidence (present/partial/missing/excluded weighting).
# --------------------------------------------------------------------------- #

PRIOR = BetaParams(
    alpha=1.0, beta=1.0
)  # uniform on [0,1]; declared in the score contract


@dataclass
class CategoryEvidence:
    """Weighted outcome tallies for one FeatureUniverse category."""

    category_id: str
    category_weight: float  # release weight; categories sum to 1.0 (loader-enforced)
    weighted_successes: float = 0.0  # Σ 1.0*w over present rows
    weighted_partials: float = (
        0.0  # Σ 0.5*w over partial rows (counts in BOTH alpha and beta)
    )
    weighted_failures: float = 0.0  # Σ 1.0*w over missing rows
    # excluded rows are skipped here but their weight stays in the denominator for
    # a strict-100% claim (see §2.1.2); tracked separately for coverage debt.
    weighted_excluded: float = 0.0

    def posterior(self) -> BetaParams:
        return BetaParams(
            alpha=PRIOR.alpha + self.weighted_successes + self.weighted_partials,
            beta=PRIOR.beta + self.weighted_failures + self.weighted_partials,
        )


def add_outcome(ev: CategoryEvidence, status: str, weight: float) -> None:
    """Apply one feature outcome to a category's evidence (§3.1 scoring rule)."""
    if status == "present":
        ev.weighted_successes += 1.0 * weight
    elif status == "partial":
        ev.weighted_partials += 0.5 * weight
    elif status == "missing":
        ev.weighted_failures += 1.0 * weight
    elif status == "excluded":
        ev.weighted_excluded += 1.0 * weight
    else:
        raise ValueError(f"unknown status {status!r}")


# --------------------------------------------------------------------------- #
# §3.2/§3.3 — distribution-free conformal band; release uses the LOWER bound.
# --------------------------------------------------------------------------- #


def conformal_halfwidth(residuals: Iterable[float], confidence: float) -> float:
    """(1 - alpha) empirical quantile of held-out nonconformity residuals
    (Vovk-Gammerman-Shafer 2005). Distribution-free finite-sample coverage."""
    rs = sorted(residuals)
    if not rs:
        # Bootstrap with a wide band before residuals exist (§3.2 pitfall guard).
        return 1.0
    n = len(rs)
    k = (
        math.ceil(confidence * (n + 1)) - 1
    )  # 0-indexed (1-α)-quantile, α = 1-confidence
    k = max(0, min(k, n - 1))
    return rs[k]


@dataclass
class Scorecard:
    """The output of the conformance parity score for one gauntlet round."""

    point_estimate: float  # S_mean (dashboards only)
    lower_bound: float  # S_lower = truncate_score(S_mean - q); THE release number
    conformal_halfwidth: float
    per_category_lower: dict[str, float]  # truncate_score'd; ratchet compares these
    per_category_mean: dict[str, float]


def compute_scorecard(
    categories: list[CategoryEvidence],
    residuals: Iterable[float],
    confidence: float = 0.95,
) -> Scorecard:
    """Beta posterior per category -> weighted mean -> conformal band -> lower bound."""
    total_w = sum(c.category_weight for c in categories)
    if abs(total_w - 1.0) > 1e-9:
        raise ValueError(
            f"category weights must sum to 1.0 (loader invariant); got {total_w!r}"
        )
    q = conformal_halfwidth(residuals, confidence)
    per_mean: dict[str, float] = {}
    per_lower: dict[str, float] = {}
    s_mean = 0.0
    for c in categories:
        post = c.posterior()
        mean_c = post.mean()
        per_mean[c.category_id] = truncate_score(mean_c)
        per_lower[c.category_id] = truncate_score(max(0.0, mean_c - q))
        s_mean += c.category_weight * mean_c
    s_lower = truncate_score(max(0.0, s_mean - q))
    return Scorecard(
        point_estimate=truncate_score(s_mean),
        lower_bound=s_lower,
        conformal_halfwidth=q,
        per_category_lower=per_lower,
        per_category_mean=per_mean,
    )


# --------------------------------------------------------------------------- #
# §3.5 — the ratchet state machine (Allow | Block | Quarantine | Waiver).
# --------------------------------------------------------------------------- #

QUARANTINE_DELTA = 0.005  # a single per-category dip ≤ this -> Quarantine, not Block


@dataclass
class RatchetState:
    current_lower_bound: float
    per_category_bounds: dict[str, float]


@dataclass
class RatchetVerdict:
    decision: str  # "Allow" | "Block" | "Quarantine" | "Waiver"
    reason: str
    new_state: RatchetState | None = None


def apply_ratchet(
    scorecard: Scorecard,
    state: RatchetState,
    waived_categories: frozenset[str] = frozenset(),
) -> RatchetVerdict:
    """Decide whether this round may land, per §3.5. Release uses the LOWER bound."""
    dipped: list[str] = []
    worst_dip = 0.0
    for cat, floor in state.per_category_bounds.items():
        cur = scorecard.per_category_lower.get(cat, 0.0)
        if cur < floor and cat not in waived_categories:
            dipped.append(cat)
            worst_dip = max(worst_dip, floor - cur)

    global_ok = scorecard.lower_bound >= state.current_lower_bound

    if waived_categories and dipped == []:
        # all dips covered by waivers (none remain after filtering)
        pass

    if not dipped and global_ok:
        new = RatchetState(
            current_lower_bound=max(scorecard.lower_bound, state.current_lower_bound),
            per_category_bounds={
                cat: max(scorecard.per_category_lower.get(cat, 0.0), floor)
                for cat, floor in state.per_category_bounds.items()
            },
        )
        return RatchetVerdict(
            "Allow", "raises lower bound; no per-category regression", new
        )

    if dipped and all(c in waived_categories for c in dipped):
        return RatchetVerdict(
            "Waiver", f"regression in {dipped} covered by active waiver", None
        )

    if global_ok and len(dipped) == 1 and worst_dip <= QUARANTINE_DELTA:
        return RatchetVerdict(
            "Quarantine",
            f"single per-category dip {dipped[0]} by {worst_dip:.6f} ≤ {QUARANTINE_DELTA}",
            None,
        )

    return RatchetVerdict(
        "Block",
        f"lower bound or per-category bound regressed (global_ok={global_ok}, dipped={dipped})",
        None,
    )


# --------------------------------------------------------------------------- #
# §6 — Ville e-processes for the four load-bearing invariants.
# Howard-Ramdas-McAuliffe-Sekhon 2021; Ville's inequality gives
# P_{H_0}(exists t: E_t >= 1/alpha) <= alpha  (anytime-valid; no Bonferroni).
# --------------------------------------------------------------------------- #

# Calibration split (§6.2): hardware-enforced (CPU integer semantics / determinism
# flag; a violation is a CPU/logic bug, tight prior) vs software-enforced (code
# path; rare but plausible, looser prior).
HARDWARE = dict(p0=1e-9, lam=0.999, alpha=1e-6)  # threshold 1/alpha = 1_000_000
SOFTWARE = dict(p0=1e-6, lam=0.9, alpha=1e-3)  # threshold 1/alpha = 1_000

# The four franken_ocr invariants, line-backed to the truth pack / plan §5.4.
INVARIANT_CALIBRATION = {
    "INV-KV-CAP": SOFTWARE,  # KV never exceeds L*(m+128); m_max=32896, W=128, L=12 (CENSUS d)
    "INV-I32-NOOVERFLOW": SOFTWARE,  # i32 acc never overflows at K_max=6848 (plan §5.4, ≥9x headroom)
    "INV-DETERMINISM": HARDWARE,  # same input twice -> byte-identical output
    "INV-SIMD-SCALAR": HARDWARE,  # SDOT/SMMLA/VNNI == scalar bit-identical (integer add associative)
}


@dataclass
class EProcess:
    """An anytime-valid e-process over one invariant's observation stream."""

    p0: float
    lam: float
    alpha: float
    e_value: float = 1.0
    obs_count: int = 0
    rejected_at: int | None = None

    @classmethod
    def for_invariant(cls, name: str) -> "EProcess":
        cal = INVARIANT_CALIBRATION[name]
        return cls(p0=cal["p0"], lam=cal["lam"], alpha=cal["alpha"])

    def observe(self, alarm: bool) -> bool:
        """Feed one observation (alarm == invariant violated). Returns True the
        first time the Ville threshold is crossed."""
        self.obs_count += 1
        if alarm:
            factor = (1.0 - self.lam) + self.lam * 1.0 / self.p0
        else:
            factor = 1.0 - self.lam  # (1-lam) + lam*0/p0
        self.e_value *= factor
        # Saturate to keep f64 noise away from the threshold logic; do NOT reset
        # to 1.0 (that breaks the supermartingale property / Ville's bound).
        self.e_value = min(max(self.e_value, 5e-324), sys.float_info.max / 2.0)
        if self.e_value >= 1.0 / self.alpha and self.rejected_at is None:
            self.rejected_at = self.obs_count
            return True
        return False

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha


def global_e_value(processes: Iterable[EProcess]) -> float:
    """Arithmetic mean of per-invariant e-values -- itself an e-process under the
    global null REGARDLESS of dependence (§6.1). Never max, never product."""
    ps = list(processes)
    if not ps:
        return 1.0
    return sum(p.e_value for p in ps) / len(ps)


# --------------------------------------------------------------------------- #
# bd-re8.13 — the REAL-data modes: score FEATURE_PARITY.md, track convergence.
# --------------------------------------------------------------------------- #

STATUS_TOKENS = ("present", "partial", "missing", "n/a", "excluded")


def parse_feature_parity(md_text: str) -> dict[str, dict[str, int]]:
    """Parse FEATURE_PARITY.md into per-section status tallies.

    A row is status-bearing when one of its pipe-cells is EXACTLY a status
    token — the same rule `tests/surface_matrix.rs` uses, so the two parsers
    cannot diverge on what counts as a scoreboard row. Returns
    {section_title: {status: count}} for every `## N.` section with rows.
    """
    import re

    sections: dict[str, dict[str, int]] = {}
    current = None
    for line in md_text.splitlines():
        m = re.match(r"^## (\d+)\.\s*(.*)", line)
        if m:
            current = f"§{m.group(1)} {m.group(2).split('(')[0].split('—')[0].strip()}"
            continue
        if current is None or not line.startswith("|") or line.startswith("|-"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 6:
            continue
        for c in cells:
            if c.lower() in STATUS_TOKENS:
                sections.setdefault(current, {})
                sections[current][c.lower()] = sections[current].get(c.lower(), 0) + 1
                break
    return {k: v for k, v in sections.items() if v}


def categories_from_parity(md_text: str) -> list[CategoryEvidence]:
    """One CategoryEvidence per scoreboard section, equal category weights
    (normalized to the loader's sum-to-1.0 invariant), equal row weights
    within a section. `n/a` rows are skipped entirely; `excluded` rows keep
    their weight in the coverage-debt tally per §2.1.2."""
    sections = parse_feature_parity(md_text)
    if not sections:
        raise ValueError("no scoreboard sections parsed from FEATURE_PARITY.md")
    cat_w = 1.0 / len(sections)
    cats: list[CategoryEvidence] = []
    for title, tallies in sections.items():
        scored = {s: n for s, n in tallies.items() if s != "n/a"}
        n_rows = sum(scored.values())
        row_w = 1.0 / n_rows if n_rows else 0.0
        ev = CategoryEvidence(title, cat_w)
        for status, count in scored.items():
            for _ in range(count):
                add_outcome(ev, status, row_w)
        cats.append(ev)
    return cats


def surface_must_verdict(md_text: str) -> dict:
    """Require every release-surface MUST row (§12-§15) to be present.

    `partial`, `missing`, `excluded`, `n/a`, and unrecognized statuses are all
    release debt. The methodology explicitly forbids rounding partial upward,
    and a strict certificate cannot silently waive excluded MUST rows.
    """
    section = 0
    must_rows = 0
    debt: list[dict[str, str]] = []
    for line in md_text.splitlines():
        heading = re.match(r"^## (\d+)\.", line)
        if heading:
            section = int(heading.group(1))
            continue
        if not (12 <= section <= 15) or not line.startswith("|"):
            continue
        cells = _markdown_cells(line)
        if "MUST" not in cells:
            continue
        must_rows += 1
        status = next(
            (cell.lower() for cell in cells if cell.lower() in STATUS_TOKENS),
            "unrecognized",
        )
        if status != "present":
            debt.append(
                {
                    "surface": cells[0][:80] if cells else "<unnamed>",
                    "status": status,
                }
            )
    return {
        "ok": must_rows > 0 and not debt,
        "must_rows": must_rows,
        "debt": debt,
    }


_ANY_EXPERIMENT_HEADING_RE = re.compile(r"^##\s+Experiment\b.*$", re.MULTILINE)
_HYPOTHESIS_ENTRY_RE = re.compile(
    r"^##\s+Experiment\s+`(EXP-[0-9]{4})`(?:\s+[-\N{EM DASH}]\s+.+)?\s*$",
    re.MULTILINE,
)
_HYPOTHESIS_TERMINAL_STATES = {"CONFIRMED_GAP", "NO_EVIDENCE"}


def _clean_experiment_value(value: str) -> str:
    return value.strip().strip("`").strip().strip('"').strip("'")


def _experiment_field_values(section: str, field_name: str) -> list[str]:
    escaped = re.escape(field_name)
    patterns = (
        rf"^\|\s*`{escaped}`\s*\|\s*(.*?)\s*\|\s*$",
        rf"^\s*{escaped}:\s*(.*?)\s*$",
        rf"^\s*-\s*\*\*{escaped}:\*\*\s*(.*?)\s*$",
        rf"^\s*-\s*\*\*{escaped}\*\*:\s*(.*?)\s*$",
    )
    values: list[str] = []
    for pattern in patterns:
        values.extend(re.findall(pattern, section, re.IGNORECASE | re.MULTILINE))
    return [_clean_experiment_value(value) for value in values]


def _experiment_table_field_values(section: str, field_name: str) -> list[str]:
    values = re.findall(
        rf"^\|\s*`{re.escape(field_name)}`\s*\|\s*(.*?)\s*\|\s*$",
        section,
        re.IGNORECASE | re.MULTILINE,
    )
    return [_clean_experiment_value(value) for value in values]


def _experiment_list_field(section: str, field_name: str) -> list[str]:
    match = re.search(
        rf"^\s*{re.escape(field_name)}:\s*$\n(?P<body>(?:^[ \t]+-\s+.*(?:\n|$))*)",
        section,
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return []
    return [
        _clean_experiment_value(value)
        for value in re.findall(
            r"^[ \t]+-\s+(.+?)\s*$", match.group("body"), re.MULTILINE
        )
    ]


def _experiment_section_body(section: str, heading: str) -> str | None:
    match = re.search(
        rf"^###\s+{re.escape(heading)}\s*$\n(?P<body>.*?)(?=^###\s+|\Z)",
        section,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not match:
        return None
    body = match.group("body").strip()
    return body if body and "<" not in body else None


def _experiment_section_present(section: str, heading: str) -> bool:
    return _experiment_section_body(section, heading) is not None


def _experiment_results_fence(section: str) -> str | None:
    body = _experiment_section_body(section, "Results Inline")
    if body is None:
        return None
    fences = re.findall(
        r"^```(?:yaml|yml)?\s*$\n(?P<fields>.*?)^```\s*$",
        body,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return fences[0] if len(fences) == 1 else None


def _retry_condition_valid(value: str) -> bool:
    normalized = _clean_experiment_value(value).lower().rstrip(".")
    return any(normalized.startswith(prefix) for prefix in _RETRY_CONDITION_PREFIXES)


def _evidence_path_verdict(root: Path | None, relative: str) -> str | None:
    if not relative or "<" in relative or relative.upper() in {"N/A", "NONE", "[]"}:
        return "evidence path is empty or a placeholder"
    if root is None:
        return "evidence root is unavailable"
    path = _safe_repo_path(root, relative)
    if path is None or not path.exists():
        return f"evidence path is missing or unsafe: {relative}"
    if path.is_dir():
        manifest_ok, reasons, _covered = _verify_sha256_manifest(path)
        if not manifest_ok:
            return f"evidence directory is not hash-verified: {relative}: {', '.join(reasons)}"
        return None
    manifest_ok, reasons, covered = _verify_sha256_manifest(path.parent)
    if not manifest_ok or path.name not in covered:
        detail = ", ".join(reasons) if reasons else "manifest omits file"
        return f"evidence file is not hash-verified: {relative}: {detail}"
    return None


def _proof_pack_path(
    root: Path | None, section: str, evidence_paths: Sequence[str]
) -> str | None:
    explicit = _experiment_field_values(section, "proof_pack_path")
    candidates = explicit + [path for path in evidence_paths if "proof_pack" in path]
    for candidate in candidates:
        if root is None:
            continue
        path = _safe_repo_path(root, candidate)
        if path is not None and path.is_dir() and any(path.iterdir()):
            manifest_ok, _reasons, _covered = _verify_sha256_manifest(path)
            if manifest_ok:
                return candidate
    return None


def _remediation_bead_failure(root: Path | None, bead_id: str) -> str | None:
    if root is None:
        return "remediation bead registry is unavailable"
    issues_path = root / ".beads/issues.jsonl"
    try:
        issues = {
            issue["id"]: issue
            for line_number, line in enumerate(
                _read_bounded_text(issues_path).splitlines(), start=1
            )
            if line.strip()
            for issue in [
                _parse_json_bytes(
                    line.encode("utf-8"), label=f"{issues_path}:{line_number}"
                )
            ]
            if isinstance(issue, dict) and isinstance(issue.get("id"), str)
        }
    except (OSError, ValueError, TypeError, RecursionError) as error:
        return f"remediation bead registry is unreadable: {error}"
    issue = issues.get(bead_id)
    if issue is None:
        return f"remediation bead does not exist: {bead_id}"
    labels = issue.get("labels")
    if not isinstance(labels, list) or "kind:remediation" not in labels:
        return f"remediation bead lacks kind:remediation label: {bead_id}"
    dependencies = issue.get("dependencies")
    if not isinstance(dependencies, list):
        return f"remediation bead has no dependency graph: {bead_id}"
    dependency_texts: list[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        dependency_id = dependency.get("depends_on_id") or dependency.get("id")
        dependency_issue = issues.get(dependency_id)
        if not isinstance(dependency_issue, dict):
            continue
        dependency_texts.append(
            " ".join(
                [
                    str(dependency_issue.get("id", "")),
                    str(dependency_issue.get("title", "")),
                    str(dependency_issue.get("issue_type", "")),
                    " ".join(map(str, dependency_issue.get("labels", []))),
                ]
            ).lower()
        )
    required_kinds = {
        "test": re.compile(
            r"\b(?:test|oracle|metamorphic|sanitizer|fuzz|property|golden)\b"
        ),
        "bench": re.compile(
            r"\b(?:bench|benchmark|criterion|hyperfine|comprehensive-bench)\b"
        ),
        "doc": re.compile(r"\b(?:doc|docs|documentation|safety|contract|agents\.md)\b"),
    }
    missing = [
        kind
        for kind, pattern in required_kinds.items()
        if not any(pattern.search(text) for text in dependency_texts)
    ]
    if missing:
        return f"remediation bead lacks test+bench+doc dependencies ({', '.join(missing)}): {bead_id}"
    return None


def _proof_pack_failure(root: Path | None, bead_id: str, pillar: str) -> str | None:
    if root is None:
        return "proof-pack root is unavailable"
    relative = f"artifacts/{bead_id}/proof_pack"
    path = _safe_repo_path(root, relative)
    if path is None or not path.is_dir():
        return f"required proof pack is missing: {relative}"
    manifest_ok, reasons, covered = _verify_sha256_manifest(path)
    if not manifest_ok:
        return f"proof pack is not exhaustively hash-verified: {', '.join(reasons)}"
    required = {"README.md", "delta_summary.json", "rerun.sh", "rollback.md"}
    if pillar == "perf":
        required.update(
            {
                "baseline_profile.flame.svg",
                "baseline_profile.samply.json",
                "selections_byte_identical.txt",
                "concurrent_mode_default_guard.txt",
                "cv_pct.json",
            }
        )
    elif pillar == "conformance":
        required.update(
            {"selections_byte_identical.txt", "concurrent_mode_default_guard.txt"}
        )
    missing = sorted(required - covered)
    return f"proof pack omits required files: {', '.join(missing)}" if missing else None


def hypothesis_texts_verdict(
    contents: dict[str, str | None],
    ledger_paths: Sequence[str] = HYPOTHESIS_LEDGER_PATHS,
    root: Path | None = None,
) -> dict:
    """Validate canonical experiment closure and proof obligations."""
    missing: list[str] = []
    unresolved: list[dict[str, str]] = []
    entries = 0
    seen: set[str] = set()
    for ledger in ledger_paths:
        text = contents.get(ledger)
        if text is None:
            missing.append(ledger)
            continue
        matches = list(_HYPOTHESIS_ENTRY_RE.finditer(text))
        all_headings = list(_ANY_EXPERIMENT_HEADING_RE.finditer(text))
        recognized_headings = {match.group(0).strip() for match in matches}
        for heading in all_headings:
            if heading.group(0).strip() not in recognized_headings:
                unresolved.append(
                    {
                        "ledger": ledger,
                        "hypothesis": heading.group(0)[:120],
                        "state": "NONCANONICAL_EXPERIMENT_HEADING",
                    }
                )
        if not matches:
            unresolved.append(
                {
                    "ledger": ledger,
                    "hypothesis": "<ledger>",
                    "state": "NO_CANONICAL_EXPERIMENT_ENTRIES",
                }
            )
            continue
        for index, match in enumerate(matches):
            hypothesis_id = match.group(1)
            if hypothesis_id in seen:
                unresolved.append(
                    {
                        "ledger": ledger,
                        "hypothesis": hypothesis_id,
                        "state": "DUPLICATE_EXPERIMENT_ID",
                    }
                )
                continue
            seen.add(hypothesis_id)
            entries += 1
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section = text[match.start() : end]
            failures: list[str] = []
            required_fields = (
                "experiment_id",
                "pillar",
                "created_at_utc",
                "created_by_agent",
                "bead_id",
                "parent_hypothesis_id",
                "status",
            )
            fields: dict[str, str] = {}
            for field_name in required_fields:
                values = _experiment_table_field_values(section, field_name)
                if len(values) != 1:
                    failures.append(f"{field_name} must occur exactly once")
                else:
                    fields[field_name] = values[0]
            if fields.get("experiment_id", "").strip("<>") != hypothesis_id:
                failures.append("experiment_id does not match heading")
            if fields.get("status", "").strip("`").upper() != "CLOSED":
                failures.append("status is not CLOSED")
            if _parse_utc_timestamp(fields.get("created_at_utc")) is None:
                failures.append("created_at_utc is invalid")
            for name in ("created_by_agent", "bead_id", "parent_hypothesis_id"):
                if not fields.get(name) or "<" in fields.get(name, ""):
                    failures.append(f"{name} is empty or a placeholder")
            expected_pillar = {
                "PERF_HYPOTHESIS_LEDGER.md": "perf",
                "CONFORMANCE_HYPOTHESIS_LEDGER.md": "conformance",
                "SURFACE_PARITY_HYPOTHESIS_LEDGER.md": "surface",
            }.get(Path(ledger).name)
            pillar = fields.get("pillar", "").strip("`").lower()
            if pillar not in {"perf", "conformance", "surface"}:
                failures.append("pillar is invalid")
            elif expected_pillar is not None and pillar != expected_pillar:
                failures.append(f"pillar {pillar!r} does not match ledger")

            for heading in (
                "Hypothesis",
                "Motivation",
                "Minimal Reproducer",
                "Expected Signal",
                "Falsifiability Criteria",
                "One-Line Invocation",
                "Results Inline",
                "Closure Predicate",
            ):
                if not _experiment_section_present(section, heading):
                    failures.append(f"missing or placeholder section: {heading}")

            results_fence = _experiment_results_fence(section)
            if results_fence is None:
                failures.append(
                    "Results Inline must contain exactly one fenced YAML field block"
                )
                results_fence = ""
            result_values = _experiment_field_values(results_fence, "result_status")
            state = (
                result_values[0].upper()
                if len(result_values) == 1
                else "RESULT_MISSING_OR_DUPLICATE"
            )
            if state not in _HYPOTHESIS_TERMINAL_STATES:
                failures.append(f"result_status is not terminal: {state}")
            summary_values = _experiment_field_values(results_fence, "result_summary")
            if (
                len(summary_values) != 1
                or not summary_values[0]
                or "<" in summary_values[0]
            ):
                failures.append("result_summary is missing or a placeholder")
            closed_values = _experiment_field_values(results_fence, "closed_at_utc")
            closed_at = (
                _parse_utc_timestamp(closed_values[0])
                if len(closed_values) == 1
                else None
            )
            if closed_at is None:
                failures.append("closed_at_utc is missing or invalid")
            created_at = _parse_utc_timestamp(fields.get("created_at_utc"))
            if (
                closed_at is not None
                and created_at is not None
                and closed_at < created_at
            ):
                failures.append("closed_at_utc predates created_at_utc")

            evidence_paths = _experiment_list_field(
                results_fence, "result_evidence_paths"
            )
            if not evidence_paths:
                failures.append("result_evidence_paths is empty")
            elif len(set(evidence_paths)) != len(evidence_paths):
                failures.append("result_evidence_paths contains duplicates")
            for evidence_path in evidence_paths:
                evidence_reason = _evidence_path_verdict(root, evidence_path)
                if evidence_reason:
                    failures.append(evidence_reason)

            if state == "CONFIRMED_GAP":
                bead_values = _experiment_field_values(
                    results_fence, "spawned_remediation_bead"
                )
                remediation_id = bead_values[0] if len(bead_values) == 1 else ""
                if (
                    re.fullmatch(r"bd-[A-Za-z0-9][A-Za-z0-9.-]*", remediation_id)
                    is None
                ):
                    failures.append("CONFIRMED_GAP has no remediation bead")
                    failures.append("CONFIRMED_GAP has no hash-verified proof pack")
                else:
                    bead_failure = _remediation_bead_failure(root, remediation_id)
                    if bead_failure:
                        failures.append(bead_failure)
                    expected_pack = f"artifacts/{remediation_id}/proof_pack"
                    declared_pack = _proof_pack_path(
                        root, results_fence, evidence_paths
                    )
                    if declared_pack != expected_pack:
                        failures.append(
                            f"CONFIRMED_GAP must declare proof_pack_path: {expected_pack}"
                        )
                    pack_failure = _proof_pack_failure(root, remediation_id, pillar)
                    if pack_failure:
                        failures.append(pack_failure)
                impact_values = _experiment_field_values(results_fence, "result_impact")
                if (
                    len(impact_values) != 1
                    or not impact_values[0]
                    or "<" in impact_values[0]
                ):
                    failures.append("CONFIRMED_GAP has no measured result_impact")
            elif state == "NO_EVIDENCE":
                retry_values = _experiment_field_values(
                    results_fence, "retry_condition_predicate"
                )
                if len(retry_values) != 1 or not _retry_condition_valid(
                    retry_values[0]
                ):
                    failures.append(
                        "NO_EVIDENCE has no allowed retry_condition_predicate"
                    )

            if failures:
                unresolved.append(
                    {
                        "ledger": ledger,
                        "hypothesis": hypothesis_id,
                        "state": state,
                        "failures": failures,
                    }
                )
    resolved = not missing and not unresolved
    return {
        "resolved": resolved,
        "missing": missing,
        "unresolved": unresolved,
        "entries": entries,
        "required_ledgers": list(ledger_paths),
    }


def hypothesis_ledger_verdict(
    root: Path, ledger_paths: Sequence[str] = HYPOTHESIS_LEDGER_PATHS
) -> dict:
    contents: dict[str, str | None] = {}
    for ledger in ledger_paths:
        path = _safe_repo_path(root, ledger)
        try:
            contents[ledger] = _read_bounded_text(path) if path else None
        except (OSError, ValueError):
            contents[ledger] = None
    return hypothesis_texts_verdict(contents, ledger_paths, root)


def _markdown_cells(line: str) -> list[str]:
    return [
        cell.strip().replace(r"\|", "|")
        for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))
    ]


def _perf_ledger_rows(md_text: str) -> list[dict[str, str]]:
    header = next(
        (
            line
            for line in md_text.splitlines()
            if line.startswith("| date | claim_id | evidence_id |")
        ),
        "",
    )
    if not header:
        return []
    columns = _markdown_cells(header)
    rows: list[dict[str, str]] = []
    for line in md_text.splitlines():
        if not line.startswith("|") or line == header or "------" in line:
            continue
        cells = _markdown_cells(line)
        if len(cells) == len(columns):
            rows.append(dict(zip(columns, cells, strict=True)))
    return rows


def _verify_sha256_manifest(evidence_dir: Path) -> tuple[bool, list[str], set[str]]:
    manifest = next(
        (
            evidence_dir / name
            for name in EVIDENCE_MANIFEST_NAMES
            if (evidence_dir / name).is_file()
        ),
        None,
    )
    if manifest is None:
        return False, ["missing SHA-256 manifest"], set()
    if manifest.is_symlink():
        return False, ["SHA-256 manifest must not be a symlink"], set()
    if manifest.suffix == ".json":
        return (
            False,
            ["JSON evidence manifests are not yet a supported strict format"],
            set(),
        )
    reasons: list[str] = []
    covered: set[str] = set()
    try:
        lines = _read_bounded_text(manifest).splitlines()
    except (OSError, ValueError) as error:
        return False, [f"unreadable manifest: {error}"], set()
    if len(lines) > CERTIFICATION_MAX_EVIDENCE_MANIFEST_ENTRIES:
        return (
            False,
            ["evidence manifest exceeds the verifier entry-count limit"],
            set(),
        )
    for line in lines:
        if not line.strip():
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+[* ]?(.+?)\s*$", line)
        if not match:
            reasons.append(f"malformed manifest line: {line[:80]}")
            continue
        expected, relative = match.group(1).lower(), match.group(2)
        relative_path = Path(relative)
        canonical = relative_path.as_posix()
        if (
            relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or canonical != relative
        ):
            reasons.append(f"noncanonical manifest target: {relative}")
            continue
        if relative in covered:
            reasons.append(f"duplicate manifest target: {relative}")
            continue
        target = _safe_repo_path(evidence_dir, relative)
        if (
            target is None
            or not target.is_file()
            or any(
                evidence_dir.joinpath(*relative_path.parts[:index]).is_symlink()
                for index in range(1, len(relative_path.parts) + 1)
            )
        ):
            reasons.append(
                f"manifest target missing, symlinked, or escapes evidence dir: {relative}"
            )
            continue
        covered.add(relative)
        maximum = (
            PERF_MAX_SUBJECT_BINARY_BYTES
            if relative == PERF_INPUT_BINDINGS["subject_binary"]
            else PERF_MAX_SOURCE_PACK_BYTES
            if relative == PERF_INPUT_BINDINGS["source_input_pack"]
            else CERTIFICATION_MAX_ARTIFACT_BYTES
        )
        try:
            actual = _bounded_file_identity(target, maximum)["sha256"]
        except (OSError, ValueError) as error:
            reasons.append(f"manifest target is unreadable: {relative}: {error}")
            continue
        if actual != expected:
            reasons.append(f"manifest hash mismatch: {relative}")
    if not covered:
        reasons.append("manifest covers no evidence files")
    actual_files: set[str] = set()
    symlinks: list[str] = []
    observed_paths = 0
    for path in evidence_dir.rglob("*"):
        observed_paths += 1
        if observed_paths > CERTIFICATION_MAX_EVIDENCE_MANIFEST_ENTRIES:
            reasons.append("evidence directory exceeds the verifier path-count limit")
            break
        relative = path.relative_to(evidence_dir).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
        elif path.is_file() and path != manifest and not path.name.startswith("._"):
            actual_files.add(relative)
    symlinks.sort()
    if symlinks:
        reasons.append("evidence directory contains symlinks: " + ", ".join(symlinks))
    uncovered = sorted(actual_files - covered)
    if uncovered:
        reasons.append("manifest omits evidence files: " + ", ".join(uncovered))
    return not reasons, reasons, covered


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _stage_record(payload: dict, stage: str) -> dict | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), list):
        return None
    wanted = stage.replace("-", "_")
    for record in payload.get("stages", []):
        if (
            isinstance(record, dict)
            and str(record.get("stage", "")).replace("-", "_") == wanted
        ):
            return record
    return None


def _fixture_identity(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = re.search(
        r"(?:^|;\s*)page=(?P<page>[^\s;]+)\s+sha256=(?P<sha>[0-9a-f]{64})(?:;|\s|$)",
        value,
    )
    return (match.group("page"), match.group("sha")) if match else None


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


def _canonical_kill_switch_cell(focr_payload: dict) -> str | None:
    focr_env = focr_payload.get("focr_env")
    gates = focr_payload.get("precision_gate_states")
    if not isinstance(focr_env, dict) or not isinstance(gates, dict):
        return None
    evidence = dict(focr_env)
    evidence.update(gates)
    raw = " ".join(f"{name}={value}" for name, value in sorted(evidence.items()))
    return " ".join(raw.replace("|", "¦").split())


def _current_unlimited_measurement_reasons(
    focr_payload: dict,
    focr_stage: dict,
    row: dict,
) -> list[str]:
    """Validate the one current Unlimited release precision/artifact contract."""
    reasons: list[str] = []
    if focr_payload.get("precision") != CURRENT_UNLIMITED_PRECISION:
        reasons.append(
            "historical or ambiguous Unlimited precision is ineligible for the current release"
        )
    if focr_stage.get("precision") != CURRENT_UNLIMITED_PRECISION:
        reasons.append("focr stage does not carry the current conservative precision")
    if focr_payload.get("decode_mode") != CURRENT_UNLIMITED_DECODE_MODE:
        reasons.append("focr decode_mode is not execution-derived conservative mode")
    if focr_payload.get("quant_recipe") != CURRENT_UNLIMITED_QUANT_RECIPE:
        reasons.append("focr quant_recipe is not the current conservative recipe")

    model = focr_payload.get("model")
    if (
        not isinstance(model, str)
        or not model.endswith(".focrq")
        or focr_payload.get("model_kind") != "file"
        or focr_payload.get("model_sha256") != CURRENT_UNLIMITED_MODEL_SHA256
        or focr_payload.get("model_size") != CURRENT_UNLIMITED_MODEL_SIZE
    ):
        reasons.append(
            "focr evidence does not identify the exact current conservative model hash/size"
        )
    if _argv_option(focr_payload.get("command"), "--model") != model:
        reasons.append("focr command does not bind the measured model artifact")

    gates = focr_payload.get("precision_gate_states")
    focr_env = focr_payload.get("focr_env")
    if not isinstance(gates, dict) or set(gates) != set(PRECISION_GATE_VARS):
        reasons.append("focr precision gate evidence is missing or incomplete")
    else:
        for name in OQ14_FALSE_OR_UNSET_VARS:
            if not _strictly_falsy_or_unset(gates.get(name)):
                reasons.append(f"current release requires {name} falsy or unset")
        for name in PRESENCE_REJECTED_VARS:
            if gates.get(name) != "<unset>":
                reasons.append(
                    f"current release forbids presence-only switch {name}, even when falsy"
                )
        if not isinstance(focr_env, dict):
            reasons.append("focr run lacks structured FOCR_* environment evidence")
        else:
            for name in PRECISION_GATE_VARS:
                state = gates[name]
                if state == "<unset>":
                    if name in focr_env:
                        reasons.append(f"{name} is marked unset but appears in focr_env")
                elif focr_env.get(name) != state:
                    reasons.append(f"{name} precision state disagrees with focr_env")

    expected_kill_cell = _canonical_kill_switch_cell(focr_payload)
    if (
        expected_kill_cell is None
        or row.get("fallback/kill-switch state") != expected_kill_cell
    ):
        reasons.append("ledger kill-switch cell does not bind the complete precision gate state")
    command_cell = row.get("command/env", "")
    for token in (
        f"decode_mode={CURRENT_UNLIMITED_DECODE_MODE}",
        f"quant_recipe={CURRENT_UNLIMITED_QUANT_RECIPE}",
        f"model_sha256={CURRENT_UNLIMITED_MODEL_SHA256}",
        f"model_size={CURRENT_UNLIMITED_MODEL_SIZE}",
    ):
        if token not in command_cell:
            reasons.append(f"ledger command/env omits current subject binding: {token}")
    return reasons


def _current_unlimited_roofline_reasons(
    roofline_payload: dict,
    focr_payload: dict,
    *,
    focr_stages_sha256: str,
) -> list[str]:
    expected = {
        "arch": "unlimited-ocr",
        "precision": CURRENT_UNLIMITED_DECODE_MODE,
        "timing_precision": CURRENT_UNLIMITED_PRECISION,
        "decode_mode": CURRENT_UNLIMITED_DECODE_MODE,
        "quant_recipe": CURRENT_UNLIMITED_QUANT_RECIPE,
        "model_sha256": CURRENT_UNLIMITED_MODEL_SHA256,
        "model_size": CURRENT_UNLIMITED_MODEL_SIZE,
        "precision_gate_states": focr_payload.get("precision_gate_states"),
        "stages_json_sha256": focr_stages_sha256,
    }
    reasons = [
        f"roofline {field} does not match current measured evidence"
        for field, value in expected.items()
        if roofline_payload.get(field) != value
    ]
    if not isinstance(roofline_payload.get("stages_json"), str) or not roofline_payload[
        "stages_json"
    ]:
        reasons.append("roofline lacks its source stages-json identity")
    return reasons


def _runtime_decode_marker_reasons(
    evidence_dir: Path,
    raw_stderr: set[str],
) -> list[str]:
    """Independently re-read raw timing markers instead of trusting labels."""
    reasons: list[str] = []
    if len(raw_stderr) > PERF_MAX_RAW_TIMING_FILES:
        return ["raw timing stderr count exceeds the verifier bound"]
    precision_pattern = re.compile(
        r"^\[focr-timing\] precision ([^\r\n]+)\r?$", re.MULTILINE
    )
    mode_patterns = {
        "weight_cache_build": re.compile(
            r"^\[focr-timing\] weight_cache_build(?P<suffix>_i8)?\s+\d",
            re.MULTILINE,
        ),
        "prefill": re.compile(
            r"^\[focr-timing\] prefill(?P<suffix>_i8)?\s+\d", re.MULTILINE
        ),
        "decode": re.compile(
            r"^\[focr-timing\] decode(?P<suffix>_i8)?\s+\d", re.MULTILINE
        ),
    }
    for relative in sorted(raw_stderr):
        path = _safe_repo_path(evidence_dir, relative)
        try:
            text = _read_bounded_text(path) if path is not None else ""
        except (OSError, ValueError) as error:
            reasons.append(f"raw timing log is unreadable ({relative}): {error}")
            continue
        markers = precision_pattern.findall(text)
        if markers != [CURRENT_UNLIMITED_PRECISION]:
            reasons.append(
                f"raw timing log {relative} lacks exactly one current runtime precision marker"
            )
        for name, pattern in mode_patterns.items():
            matches = list(pattern.finditer(text))
            if len(matches) != 1:
                reasons.append(
                    f"raw timing log {relative} lacks exactly one {name} decode-mode marker"
                )
            elif matches[0].group("suffix") is not None:
                reasons.append(
                    f"raw timing log {relative} executed the full-int8 {name} path"
                )
    return reasons


def _raw_run_metadata_reasons(
    evidence_dir: Path,
    raw_meta: set[str],
    raw_stdout: set[str],
    raw_stderr: set[str],
    focr_payload: dict,
    focr_stage: dict,
) -> list[str]:
    """Rebind the aggregate to every captured process-run identity."""
    reasons: list[str] = []
    if any(
        len(paths) > PERF_MAX_RAW_TIMING_FILES
        for paths in (raw_meta, raw_stdout, raw_stderr)
    ):
        return ["raw timing file count exceeds the verifier bound"]
    measured_meta = {
        relative
        for relative in raw_meta
        if re.fullmatch(r"raw/run_\d+\.meta\.json", relative)
    }
    expected_runs = focr_stage.get("n")
    if (
        not isinstance(expected_runs, int)
        or isinstance(expected_runs, bool)
        or expected_runs <= 0
        or len(measured_meta) != expected_runs
        or focr_payload.get("runs") != expected_runs
    ):
        reasons.append("raw run metadata count does not match the aggregate")

    invariant_fields = (
        "command",
        "env_pins",
        "focr_env",
        "precision_gate_states",
        "binary",
        "binary_sha256",
        "binary_size",
        "binary_origin",
        "build_receipt",
        "build_receipt_sha256",
        "page",
        "page_sha256",
        "model",
        "model_kind",
        "model_sha256",
        "model_size",
        "quant_recipe",
        "threads",
        "warmup",
    )
    for relative in sorted(raw_meta):
        path = _safe_repo_path(evidence_dir, relative)
        try:
            payload = _load_bounded_json(path) if path is not None else None
        except (OSError, ValueError, TypeError, RecursionError) as error:
            reasons.append(f"raw run metadata is unreadable ({relative}): {error}")
            continue
        if not isinstance(payload, dict):
            reasons.append(f"raw run metadata is not a JSON object: {relative}")
            continue
        drift = [
            field
            for field in invariant_fields
            if payload.get(field) != focr_payload.get(field)
        ]
        if drift:
            reasons.append(
                f"raw run metadata {relative} disagrees with aggregate: "
                + ", ".join(drift)
            )
        wall_ms = payload.get("wall_ms")
        if (
            payload.get("exit_code") != 0
            or not _finite_number(wall_ms, minimum=0.0)
            or float(wall_ms) <= 0.0
        ):
            reasons.append(f"raw run metadata {relative} is not a successful timed run")
        tag = payload.get("tag")
        expected_name = f"raw/{tag}.meta.json" if isinstance(tag, str) else ""
        stdout = payload.get("stdout")
        stderr = payload.get("stderr")
        if expected_name != relative:
            reasons.append(f"raw run metadata tag/path mismatch: {relative}")
        if (
            not isinstance(stdout, str)
            or f"raw/{stdout}" not in raw_stdout
            or not isinstance(stderr, str)
            or f"raw/{stderr}" not in raw_stderr
        ):
            reasons.append(f"raw run metadata log paths are unbound: {relative}")
    return reasons


def _raw_timing_reasons(
    payload: dict,
    aggregate: dict,
    *,
    source: str,
) -> tuple[list[str], list[float]]:
    """Recompute an aggregate from bounded, versioned run observations."""
    reasons: list[str] = []
    raw = payload.get("raw_timing")
    if not isinstance(raw, dict):
        return [f"{source} document lacks versioned raw timing observations"], []
    if set(raw) != {"schema", "source", "unit", "measured_runs", "records"}:
        reasons.append(f"{source} raw timing has noncanonical fields")
    records = raw.get("records")
    measured_runs = raw.get("measured_runs")
    if (
        raw.get("schema") != RAW_TIMING_SCHEMA
        or raw.get("source") != source
        or raw.get("unit") != "ms"
        or not isinstance(records, list)
        or not isinstance(measured_runs, int)
        or isinstance(measured_runs, bool)
        or not 2 <= measured_runs <= PERF_MAX_RAW_TIMING_RUNS
        or len(records) != measured_runs
        or payload.get("runs") != measured_runs
    ):
        reasons.append(f"{source} raw timing identity/count is invalid")
        return reasons, []

    stage = aggregate.get("stage")
    expected_ids = [f"run_{index:03d}" for index in range(1, measured_runs + 1)]
    samples: list[float] = []
    tokens: list[int | None] = []
    text_hashes: list[str] = []
    for expected_id, raw_record in zip(expected_ids, records, strict=True):
        if not isinstance(raw_record, dict):
            reasons.append(f"{source} raw timing record is not an object")
            continue
        expected_fields = (
            {"run_id", "stages", "raw_files"}
            if source == "focr"
            else {"run_id", "stages", "text_sha256"}
        )
        if set(raw_record) != expected_fields or raw_record.get("run_id") != expected_id:
            reasons.append(f"{source} raw timing run ids/fields are noncanonical")
            continue
        stages = raw_record.get("stages")
        if (
            not isinstance(stages, dict)
            or not stages
            or len(stages) > PERF_MAX_RAW_TIMING_STAGES
            or not all(isinstance(name, str) and name for name in stages)
        ):
            reasons.append(f"{source} raw timing stage map is invalid")
            continue
        sample = stages.get(stage)
        if not isinstance(sample, dict) or set(sample) not in (
            {"ms"},
            {"ms", "tokens"},
        ):
            reasons.append(f"{source} raw timing lacks stage {stage!r}")
            continue
        value = sample.get("ms")
        if not _finite_number(value, minimum=1e-12, maximum=1e12):
            reasons.append(f"{source} raw timing sample is invalid")
            continue
        token_count = sample.get("tokens")
        if token_count is not None and (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
        ):
            reasons.append(f"{source} raw timing token count is invalid")
            continue
        samples.append(float(value))
        tokens.append(token_count)

        if source == "reference":
            text_sha = raw_record.get("text_sha256")
            if (
                not isinstance(text_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", text_sha) is None
            ):
                reasons.append(
                    "reference raw timing lacks a valid per-run text hash"
                )
            else:
                text_hashes.append(text_sha)

        if source == "focr":
            raw_files = raw_record.get("raw_files")
            if not isinstance(raw_files, dict) or set(raw_files) != {
                "meta",
                "stderr",
                "stdout",
            }:
                reasons.append("focr raw timing lacks exact file bindings")
                continue
            for kind, binding in raw_files.items():
                suffix = ".meta.json" if kind == "meta" else f".{kind}"
                if (
                    not isinstance(binding, dict)
                    or set(binding) != {"path", "sha256"}
                    or binding.get("path") != expected_id + suffix
                    or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", "")))
                    is None
                ):
                    reasons.append("focr raw timing has a malformed file binding")

    if len(samples) != measured_runs or reasons:
        return reasons, samples
    try:
        mean = statistics.fmean(samples)
        recomputed = {
            "samples_ms": [round(sample, 6) for sample in samples],
            "best_ms": round(min(samples), 6),
            "p50_ms": round(statistics.median(samples), 6),
            "mean_ms": round(mean, 6),
            "cv_pct": round(statistics.stdev(samples) / mean * 100.0, 3),
            "n": len(samples),
        }
    except (OverflowError, ValueError, ZeroDivisionError, statistics.StatisticsError):
        return [f"{source} raw timing cannot be summarized"], samples

    for field, expected in recomputed.items():
        actual = aggregate.get(field)
        if field == "samples_ms":
            if (
                not isinstance(actual, list)
                or len(actual) != len(expected)
                or any(
                    not _finite_number(value, minimum=1e-12, maximum=1e12)
                    or not math.isclose(
                        float(value), expected_value, abs_tol=5e-7
                    )
                    for value, expected_value in zip(actual, expected, strict=True)
                )
            ):
                reasons.append(f"{source} aggregate samples are not raw-derived")
        elif field == "n":
            if actual != expected:
                reasons.append(f"{source} aggregate n is not raw-derived")
        elif (
            not _finite_number(actual, minimum=0.0, maximum=1e12)
            or not math.isclose(float(actual), float(expected), abs_tol=5e-7)
        ):
            reasons.append(f"{source} aggregate {field} is not raw-derived")

    if any(token is not None for token in tokens):
        if (
            any(token is None for token in tokens)
            or len(set(tokens)) != 1
            or aggregate.get("tokens") != tokens[0]
        ):
            reasons.append(f"{source} aggregate token count is not raw-derived")
    if source == "reference" and (
        len(text_hashes) != measured_runs
        or len(set(text_hashes)) != 1
        or payload.get("text_sha256") != text_hashes[0]
        or payload.get("text_identical_across_runs") is not True
    ):
        reasons.append(
            "reference text hashes are missing, invalid, drifting, or not top-level bound"
        )
    return reasons, samples


def _physical_focr_timing_reasons(
    evidence_dir: Path,
    raw_meta: set[str],
    raw_stdout: set[str],
    raw_stderr: set[str],
    focr_payload: dict,
    focr_stage: dict,
) -> list[str]:
    """Bind structured focr observations to physical run logs and durations."""
    reasons: list[str] = []
    raw = focr_payload.get("raw_timing")
    records = raw.get("records") if isinstance(raw, dict) else None
    if not isinstance(records, list) or len(records) > PERF_MAX_RAW_TIMING_RUNS:
        return ["focr physical timing has an invalid raw record count"]
    if any(
        len(paths) > PERF_MAX_RAW_TIMING_FILES
        for paths in (raw_meta, raw_stdout, raw_stderr)
    ):
        return ["focr physical timing file count exceeds the verifier bound"]

    decode_pattern = re.compile(
        r"^\[focr-timing\]\s+decode (?P<seconds>\d+(?:\.\d+)?)s "
        r"\((?P<tokens>\d+) tokens, \d+(?:\.\d+)?s/tok\)\r?$",
        re.MULTILINE,
    )
    physical_samples: list[float] = []
    for raw_record in records:
        if not isinstance(raw_record, dict) or not isinstance(
            raw_record.get("raw_files"), dict
        ):
            reasons.append("focr physical timing lacks raw file bindings")
            continue
        bindings = raw_record["raw_files"]
        loaded: dict[str, bytes] = {}
        for kind, expected_set in (
            ("meta", raw_meta),
            ("stderr", raw_stderr),
            ("stdout", raw_stdout),
        ):
            binding = bindings.get(kind)
            relative = (
                f"raw/{binding.get('path')}" if isinstance(binding, dict) else ""
            )
            path = _safe_repo_path(evidence_dir, relative) if relative else None
            try:
                content = _read_bounded_file(path) if path is not None else b""
            except (OSError, ValueError) as error:
                reasons.append(f"focr raw {kind} is unreadable ({relative}): {error}")
                continue
            if (
                relative not in expected_set
                or not isinstance(binding, dict)
                or hashlib.sha256(content).hexdigest() != binding.get("sha256")
            ):
                reasons.append(f"focr raw {kind} binding does not verify: {relative}")
                continue
            loaded[kind] = content
        if "meta" not in loaded or "stderr" not in loaded:
            continue
        try:
            meta = _parse_json_bytes(loaded["meta"], label="focr raw metadata")
            stderr_text = loaded["stderr"].decode("utf-8")
        except (UnicodeDecodeError, ValueError, TypeError, RecursionError) as error:
            reasons.append(f"focr physical timing input is malformed: {error}")
            continue
        if not isinstance(meta, dict) or meta.get("tag") != raw_record.get("run_id"):
            reasons.append("focr physical timing metadata/run id mismatch")
            continue
        matches = list(decode_pattern.finditer(stderr_text))
        if len(matches) != 1:
            reasons.append("focr physical timing lacks exactly one decode duration")
            continue
        tokens = int(matches[0].group("tokens"))
        seconds = float(matches[0].group("seconds"))
        if tokens <= 0 or not math.isfinite(seconds) or seconds <= 0.0:
            reasons.append("focr physical decode duration is invalid")
            continue
        sample = round(seconds * 1000.0 / tokens, 6)
        structured = raw_record.get("stages", {}).get("decode_per_token", {})
        if (
            structured.get("tokens") != tokens
            or not _finite_number(structured.get("ms"), minimum=1e-12)
            or not math.isclose(float(structured["ms"]), sample, abs_tol=5e-7)
        ):
            reasons.append("focr structured timing contradicts physical stderr")
        physical_samples.append(sample)
    aggregate_samples = focr_stage.get("samples_ms")
    if (
        len(physical_samples) != len(records)
        or not isinstance(aggregate_samples, list)
        or len(aggregate_samples) != len(physical_samples)
        or any(
            not _finite_number(value, minimum=1e-12)
            or not math.isclose(float(value), sample, abs_tol=5e-7)
            for value, sample in zip(
                aggregate_samples, physical_samples, strict=True
            )
        )
    ):
        reasons.append("focr aggregate timing contradicts physical stderr")
    return reasons


def _measurement_contract(
    focr_payload: object,
    ref_payload: object,
    row: dict,
    threads: int | None,
) -> tuple[list[str], dict | None, dict | None]:
    reasons: list[str] = []
    if not isinstance(focr_payload, dict) or not isinstance(ref_payload, dict):
        return ["stage payloads must both be JSON objects"], None, None
    for side, payload, source in (
        ("focr", focr_payload, "focr"),
        ("reference", ref_payload, "reference"),
    ):
        if payload.get("schema") != "focr-gauntlet-stages/v1":
            reasons.append(f"{side} stage payload has an unsupported schema")
        if payload.get("source") != source:
            reasons.append(f"{side} stage payload has the wrong source identity")
        if payload.get("synthetic") is not False:
            reasons.append(f"{side} stage payload is synthetic or unstamped")
    if focr_payload.get("stdout_identical_across_runs") is not True:
        reasons.append("focr stdout is not deterministic across runs")

    fixture = _fixture_identity(row.get("fixture_hash"))
    if fixture is None:
        reasons.append("ledger fixture_hash lacks a page SHA-256 binding")
        fixture_page, fixture_sha = "", ""
    else:
        fixture_page, fixture_sha = fixture
    focr_page = Path(str(focr_payload.get("page", ""))).name
    ref_page = Path(str(ref_payload.get("page", ""))).name
    if not focr_page or focr_page != ref_page or focr_page != fixture_page:
        reasons.append("focr/reference/ledger fixture pages do not match")
    if focr_payload.get("page_sha256") != fixture_sha:
        reasons.append("focr page_sha256 does not match the ledger fixture hash")
    ref_page_sha = ref_payload.get("page_sha256")
    if ref_page_sha != fixture_sha:
        reasons.append("reference page_sha256 does not match the ledger fixture hash")

    focr_stage = _stage_record(focr_payload, "decode-per-token")
    ref_stage = _stage_record(ref_payload, "decode-per-token")
    if focr_stage is None or ref_stage is None:
        reasons.append("decode-per-token stage evidence is missing")
        return reasons, focr_stage, ref_stage

    for side, payload, record, source in (
        ("focr", focr_payload, focr_stage, "focr"),
        ("reference", ref_payload, ref_stage, "reference"),
    ):
        if record.get("schema") != "focr-gauntlet-stage/v1":
            reasons.append(f"{side} stage record has an unsupported schema")
        if record.get("source") != source:
            reasons.append(f"{side} stage record has the wrong source identity")
        if str(record.get("stage", "")).replace("-", "_") != "decode_per_token":
            reasons.append(f"{side} stage record has the wrong stage")
        if record.get("ledger_stage") is not True:
            reasons.append(f"{side} stage record is not ledger-eligible")
        if record.get("synthetic") is not False:
            reasons.append(f"{side} stage record is synthetic or unstamped")
        if record.get("unit") != "ms":
            reasons.append(f"{side} stage record unit is not ms")
        samples = record.get("samples_ms")
        n = record.get("n")
        if (
            not isinstance(samples, list)
            or len(samples) < 2
            or not isinstance(n, int)
            or isinstance(n, bool)
            or n != len(samples)
        ):
            reasons.append(f"{side} stage samples/n are incomplete")
            continue
        try:
            numeric_samples = [float(sample) for sample in samples]
            if not all(
                math.isfinite(sample) and sample > 0.0 for sample in numeric_samples
            ):
                raise ValueError("non-positive sample")
            best_ms = float(record["best_ms"])
            cv_pct = float(record["cv_pct"])
            recomputed_cv = round(
                statistics.stdev(numeric_samples)
                / statistics.fmean(numeric_samples)
                * 100.0,
                3,
            )
            if not math.isclose(best_ms, min(numeric_samples), rel_tol=1e-9):
                reasons.append(f"{side} best_ms is not the best measured sample")
            if (
                not math.isfinite(cv_pct)
                or not 0.0 <= cv_pct <= 5.0
                or not math.isclose(cv_pct, recomputed_cv, abs_tol=0.001)
            ):
                reasons.append(f"{side} cv_pct is invalid or not sample-derived")
        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            statistics.StatisticsError,
        ):
            reasons.append(f"{side} stage timing samples are invalid")
        if record.get("warmup_discarded", 0) < 1:
            reasons.append(f"{side} stage did not discard a warmup")
        if not isinstance(record.get("precision"), str) or not record.get("precision"):
            reasons.append(f"{side} stage lacks precision identity")
        if not isinstance(record.get("backend"), str) or not record.get("backend"):
            reasons.append(f"{side} stage lacks backend identity")
        if not isinstance(record.get("allocator"), str) or not record.get("allocator"):
            reasons.append(f"{side} stage lacks allocator identity")
        if threads is None or record.get("threads") != threads:
            reasons.append(f"{side} stage does not preserve thread parity")
        raw_reasons, _raw_samples = _raw_timing_reasons(
            payload, record, source=source
        )
        reasons.extend(raw_reasons)

    if focr_stage.get("backend") != "focr":
        reasons.append("focr stage backend identity is not focr")
    if focr_stage.get("tokens_consistent") is not True:
        reasons.append("focr decode token count is not deterministic")
    proof = ref_stage.get("thread_proof")
    if not isinstance(proof, dict) or (
        proof.get("budget") != threads or proof.get("torch_num_threads") != threads
    ):
        reasons.append("reference stage lacks a matching thread proof")
    if ref_stage.get("backend") != row.get("reference_backend"):
        reasons.append("reference backend does not match the ledger row")
    expected_precision = (
        f"{focr_stage.get('precision')} vs "
        f"{ref_stage.get('backend')}-{ref_stage.get('precision')}"
    )
    if row.get("precision (focr vs ref)") != expected_precision:
        reasons.append("stage precision identities do not match the ledger row")
    if focr_stage.get("allocator") != ref_stage.get("allocator") or focr_stage.get(
        "allocator"
    ) != row.get("allocator"):
        reasons.append("stage allocator identities do not match the ledger row")
    reasons.extend(_current_unlimited_measurement_reasons(focr_payload, focr_stage, row))
    return reasons, focr_stage, ref_stage


def _correctness_receipt_reasons(
    path: Path,
    claim: dict | None,
    *,
    evidence_dir: Path,
    row_doc: dict,
    focr_payload: object,
    ref_payload: object,
    covered: set[str],
) -> list[str]:
    if claim is None:
        return ["structured correctness claim is unavailable"]
    reasons: list[str] = []
    try:
        receipt_bytes = _read_bounded_file(path)
        payload = _parse_json_bytes(receipt_bytes, label=str(path))
    except (OSError, ValueError, TypeError, RecursionError) as error:
        return [f"correctness receipt is unreadable: {error}"]
    if not isinstance(payload, dict):
        return ["correctness receipt is not a JSON object"]
    if hashlib.sha256(receipt_bytes).hexdigest() != claim.get("sha256"):
        reasons.append("correctness receipt bytes do not match the structured claim")

    inputs = row_doc.get("correctness_inputs")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"schema", "reference", "hypotheses"}
        or inputs.get("schema") != CORRECTNESS_INPUTS_SCHEMA
    ):
        return reasons + ["row.json correctness_inputs contract is missing or noncanonical"]
    reference_binding = inputs.get("reference")
    hypotheses = inputs.get("hypotheses")
    if (
        not isinstance(reference_binding, dict)
        or set(reference_binding) != {"bundle_path", "sha256", "bytes"}
        or not isinstance(hypotheses, list)
        or not hypotheses
        or len(hypotheses) > PERF_MAX_RAW_TIMING_RUNS
    ):
        return reasons + ["row.json correctness source bindings are malformed"]

    reference_relative = reference_binding.get("bundle_path")
    reference_match = (
        re.fullmatch(r"correctness/reference/([^/]+)", reference_relative)
        if isinstance(reference_relative, str)
        else None
    )
    if (
        reference_match is None
        or reference_match.group(1) in {".", ".."}
        or reference_relative not in covered
        or not isinstance(reference_binding.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", reference_binding["sha256"]) is None
        or not isinstance(reference_binding.get("bytes"), int)
        or isinstance(reference_binding.get("bytes"), bool)
        or reference_binding["bytes"] < 0
    ):
        return reasons + ["correctness reference bundle path is noncanonical or unmanifested"]
    reference_path = _safe_repo_path(evidence_dir, reference_relative)
    try:
        reference_bytes = (
            _read_bounded_file(reference_path) if reference_path is not None else b""
        )
    except (OSError, ValueError) as error:
        return reasons + [f"correctness reference source is unreadable: {error}"]
    if (
        not reference_bytes
        or reference_binding.get("bytes") != len(reference_bytes)
        or reference_binding.get("sha256")
        != hashlib.sha256(reference_bytes).hexdigest()
    ):
        reasons.append("correctness reference source binding does not match physical bytes")

    hypothesis_runs: dict[str, bytes] = {}
    expected_ids = [f"run_{index:03d}" for index in range(1, len(hypotheses) + 1)]
    for expected_id, binding in zip(expected_ids, hypotheses, strict=True):
        expected_relative = f"correctness/hypothesis/{expected_id}.stdout"
        if (
            not isinstance(binding, dict)
            or set(binding) != {"run_id", "bundle_path", "sha256", "bytes"}
            or binding.get("run_id") != expected_id
            or binding.get("bundle_path") != expected_relative
            or expected_relative not in covered
            or not isinstance(binding.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]) is None
            or not isinstance(binding.get("bytes"), int)
            or isinstance(binding.get("bytes"), bool)
            or binding["bytes"] < 0
        ):
            reasons.append("correctness hypothesis bundle binding is noncanonical")
            continue
        hypothesis_path = _safe_repo_path(evidence_dir, expected_relative)
        try:
            content = (
                _read_bounded_file(hypothesis_path) if hypothesis_path is not None else b""
            )
        except (OSError, ValueError) as error:
            reasons.append(f"correctness hypothesis source is unreadable: {error}")
            continue
        if (
            binding.get("bytes") != len(content)
            or binding.get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            reasons.append("correctness hypothesis source binding does not match physical bytes")
        hypothesis_runs[expected_id] = content

    expected_correctness_paths = {reference_relative} | {
        f"correctness/hypothesis/{run_id}.stdout" for run_id in expected_ids
    }
    manifested_correctness_paths = {
        relative for relative in covered if relative.startswith("correctness/")
    }
    if manifested_correctness_paths != expected_correctness_paths:
        reasons.append("manifested correctness source paths are not canonical and exhaustive")

    if not isinstance(focr_payload, dict) or not isinstance(ref_payload, dict):
        return reasons + ["correctness timing documents are unavailable"]
    try:
        verified = _validate_correctness_receipt_payload(
            payload,
            reference_bytes=reference_bytes,
            hypothesis_runs=hypothesis_runs,
            focr=focr_payload,
            ref=ref_payload,
        )
    except CorrectnessValidationError as error:
        reasons.append(f"correctness receipt strict replay failed: {error}")
        return reasons
    measured = verified["cer_norm"]
    if (
        not 0.0 <= measured <= MAX_CORRECTNESS_CER
        or not math.isclose(measured, claim.get("cer_norm", math.nan), abs_tol=5e-7)
    ):
        reasons.append("correctness receipt does not match the structured in-budget claim")
    return reasons


def _bounded_file_identity(path: Path, maximum: int) -> dict:
    descriptor = os.open(path, _readonly_binary_flags())
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(f"input is not a bounded regular file: {path}")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum:
                raise ValueError(f"input grew beyond its size bound: {path}")
            digest.update(chunk)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise ValueError(f"input changed while hashing: {path}")
        if observed != before.st_size:
            raise ValueError(f"input changed length while hashing: {path}")
        return {"sha256": digest.hexdigest(), "size": observed}
    finally:
        os.close(descriptor)


def _canonical_cert_source_root(
    manifest: object, current_head: str, workspace_root: Path
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    fields = {
        "schema",
        "created_utc",
        "root_hash_algorithm",
        "root_sha256",
        "entry_count",
        "repositories",
        "cargo_config_files",
        "entries",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        return "", ["source input manifest fields are noncanonical"]
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        reasons.append("source input manifest schema is not v1")
    if not _is_canonical_utc_seconds(manifest.get("created_utc")):
        reasons.append("source input manifest created_utc is noncanonical")
    entries = manifest.get("entries")
    if (
        not isinstance(entries, list)
        or not entries
        or len(entries) > PERF_MAX_SOURCE_ENTRIES
        or manifest.get("entry_count") != len(entries)
    ):
        return "", reasons + ["source input manifest count is invalid"]
    identities: list[tuple[str, str]] = []
    seen_identities: set[tuple[str, str]] = set()
    digest = hashlib.sha256(SOURCE_ROOT_DOMAIN)
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "repository",
            "path",
            "size",
            "sha256",
        }:
            reasons.append("source input manifest entry fields are noncanonical")
            continue
        repository = entry.get("repository")
        logical = entry.get("path")
        size = entry.get("size")
        sha256 = entry.get("sha256")
        logical_path = Path(logical) if isinstance(logical, str) else Path(".")
        if (
            not isinstance(repository, str)
            or not repository
            or "\0" in repository
            or not isinstance(logical, str)
            or not logical
            or "\0" in logical
            or logical_path.is_absolute()
            or logical_path.as_posix() != logical
            or any(part in {"", ".", ".."} for part in logical_path.parts)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not 0 <= size <= PERF_MAX_SOURCE_FILE_BYTES
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            reasons.append("source input manifest entry identity is noncanonical")
            continue
        identity = (repository, logical)
        if identity in seen_identities:
            reasons.append(f"source input manifest has duplicate entry: {repository}/{logical}")
            continue
        seen_identities.add(identity)
        identities.append(identity)
        total += size
        if total > PERF_MAX_SOURCE_TOTAL_BYTES:
            reasons.append("source input manifest exceeds its total-byte bound")
            continue
        digest.update(repository.encode("utf-8"))
        digest.update(b"\0")
        digest.update(logical.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    if identities != sorted(identities):
        reasons.append("source input manifest entries are not stably sorted")
    repositories = manifest.get("repositories")
    repository_ids: set[str] = set()
    workspace_heads: list[str] = []
    if not isinstance(repositories, list) or not repositories:
        reasons.append("source input manifest repository census is missing")
    else:
        for repository in repositories:
            if not isinstance(repository, dict) or set(repository) != {
                "id",
                "path",
                "git_head",
                "packages",
                "selectors",
            }:
                reasons.append("source repository record fields are noncanonical")
                continue
            identifier = repository.get("id")
            packages = repository.get("packages")
            selectors = repository.get("selectors")
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in repository_ids
                or not isinstance(repository.get("path"), str)
                or not os.path.isabs(repository["path"])
                or os.path.normpath(repository["path"]) != repository["path"]
                or re.fullmatch(r"[0-9a-f]{40}", str(repository.get("git_head"))) is None
                or not isinstance(packages, list)
                or packages != sorted(set(packages))
                or not all(isinstance(item, str) and item for item in packages)
                or not isinstance(selectors, list)
                or selectors != sorted(set(selectors))
                or not all(isinstance(item, str) and item for item in selectors)
            ):
                reasons.append("source repository record identity is noncanonical")
                continue
            repository_ids.add(identifier)
            if identifier == "workspace":
                workspace_heads.append(repository["git_head"])
                if os.path.realpath(repository["path"]) != os.path.realpath(
                    workspace_root
                ):
                    reasons.append(
                        "source input manifest workspace path is not current workspace"
                    )
    if workspace_heads != [current_head]:
        reasons.append("source input manifest workspace HEAD does not match current HEAD")
    configs = manifest.get("cargo_config_files")
    config_ids: set[str] = set()
    if not isinstance(configs, list):
        reasons.append("source input manifest cargo config census is malformed")
        configs = []
    for config in configs:
        if not isinstance(config, dict) or set(config) != {
            "logical_path",
            "physical_path",
            "sha256",
            "size",
        }:
            reasons.append("source cargo config fields are noncanonical")
            continue
        logical = config.get("logical_path")
        if (
            not isinstance(logical, str)
            or not logical
            or logical in config_ids
            or not isinstance(config.get("physical_path"), str)
            or not os.path.isabs(config["physical_path"])
            or os.path.normpath(config["physical_path"])
            != config["physical_path"]
            or re.fullmatch(r"[0-9a-f]{64}", str(config.get("sha256"))) is None
            or not isinstance(config.get("size"), int)
            or isinstance(config.get("size"), bool)
            or not 0 <= config["size"] <= PERF_MAX_SOURCE_FILE_BYTES
        ):
            reasons.append("source cargo config identity is noncanonical")
            continue
        config_ids.add(logical)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        repository = entry.get("repository")
        logical = entry.get("path")
        if repository == "cargo-config":
            matches = [
                item
                for item in configs
                if isinstance(item, dict) and item.get("logical_path") == logical
            ]
            if len(matches) != 1 or {
                "sha256": entry.get("sha256"),
                "size": entry.get("size"),
            } != {"sha256": matches[0].get("sha256"), "size": matches[0].get("size")}:
                reasons.append("source cargo config entry is not census-bound")
        elif repository not in repository_ids:
            reasons.append(f"source manifest uses unknown repository: {repository}")
        if repository == "workspace" and isinstance(logical, str):
            physical = _safe_repo_path(workspace_root, logical)
            try:
                identity = (
                    _bounded_file_identity(physical, PERF_MAX_SOURCE_FILE_BYTES)
                    if physical is not None
                    else None
                )
            except (OSError, ValueError):
                identity = None
            if identity != {"sha256": entry.get("sha256"), "size": entry.get("size")}:
                reasons.append(f"current workspace source entry drifted: {logical}")
    if {
        entry.get("path")
        for entry in entries
        if isinstance(entry, dict) and entry.get("repository") == "cargo-config"
    } != config_ids:
        reasons.append("source cargo config census is not exhaustive")
    root_sha256 = digest.hexdigest()
    if (
        manifest.get("root_hash_algorithm") != SOURCE_ROOT_ALGORITHM
        or manifest.get("root_sha256") != root_sha256
    ):
        reasons.append("source input manifest root is invalid")
    return root_sha256, reasons


def _workspace_entry_binding(manifest: dict, name: str) -> dict | None:
    matches = [
        entry
        for entry in manifest.get("entries", [])
        if isinstance(entry, dict)
        and entry.get("repository") == "workspace"
        and entry.get("path") == name
    ]
    if len(matches) != 1:
        return None
    return {"sha256": matches[0].get("sha256"), "size": matches[0].get("size")}


def _read_exact_fd(descriptor: int, length: int, label: str) -> bytes:
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError(f"{label} length is invalid")
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise ValueError(f"source pack truncated while reading {label}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _source_pack_reasons(path: Path, manifest: dict) -> list[str]:
    reasons: list[str] = []
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return ["source pack cannot replay a malformed manifest entry list"]
    try:
        descriptor = os.open(path, _readonly_binary_flags())
    except OSError as error:
        return [f"source input pack is unreadable: {error}"]
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > PERF_MAX_SOURCE_PACK_BYTES
        ):
            return ["source input pack is not a bounded regular file"]
        try:
            if _read_exact_fd(
                descriptor, len(SOURCE_PACK_DOMAIN), "pack domain"
            ) != SOURCE_PACK_DOMAIN:
                raise ValueError("source pack domain is invalid")
            header_length = int.from_bytes(
                _read_exact_fd(descriptor, 8, "pack header length"), "big"
            )
            if not 0 < header_length <= PERF_MAX_SOURCE_PACK_HEADER_BYTES:
                raise ValueError("source pack header length exceeds its bound")
            header = _parse_strict_json_object(
                _read_exact_fd(descriptor, header_length, "pack header"),
                label="source-pack header",
            )
            total_expected = sum(
                entry.get("size", 0) if isinstance(entry, dict) else 0
                for entry in entries
            )
            if (
                set(header)
                != {
                    "schema",
                    "root_sha256",
                    "entry_count",
                    "total_content_bytes",
                }
                or header.get("schema") != SOURCE_PACK_SCHEMA
                or header.get("root_sha256") != manifest.get("root_sha256")
                or not isinstance(header.get("entry_count"), int)
                or isinstance(header.get("entry_count"), bool)
                or header.get("entry_count") != len(entries)
                or not isinstance(header.get("total_content_bytes"), int)
                or isinstance(header.get("total_content_bytes"), bool)
                or header.get("total_content_bytes") != total_expected
                or not 0 <= total_expected <= PERF_MAX_SOURCE_TOTAL_BYTES
            ):
                raise ValueError("source pack header is not manifest-derived")

            total_observed = 0
            for ordinal, expected in enumerate(entries):
                if _read_exact_fd(
                    descriptor,
                    len(SOURCE_PACK_RECORD_DOMAIN),
                    f"record {ordinal} domain",
                ) != SOURCE_PACK_RECORD_DOMAIN:
                    raise ValueError(f"source pack record {ordinal} domain is invalid")
                metadata_length = int.from_bytes(
                    _read_exact_fd(descriptor, 8, f"record {ordinal} metadata length"),
                    "big",
                )
                if not 0 < metadata_length <= PERF_MAX_SOURCE_PACK_HEADER_BYTES:
                    raise ValueError(
                        f"source pack record {ordinal} metadata length exceeds its bound"
                    )
                metadata_bytes = _read_exact_fd(
                    descriptor, metadata_length, f"record {ordinal} metadata"
                )
                _parse_strict_json_object(
                    metadata_bytes,
                    label=f"source-pack record {ordinal}",
                )
                expected_metadata_bytes = json.dumps(
                    expected,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
                if metadata_bytes != expected_metadata_bytes:
                    raise ValueError(
                        f"source pack record {ordinal} is missing, duplicated, or reordered"
                    )
                content_length = int.from_bytes(
                    _read_exact_fd(descriptor, 8, f"record {ordinal} content length"),
                    "big",
                )
                expected_size = expected.get("size") if isinstance(expected, dict) else None
                if (
                    not isinstance(expected_size, int)
                    or isinstance(expected_size, bool)
                    or content_length != expected_size
                    or not 0 <= content_length <= PERF_MAX_SOURCE_FILE_BYTES
                    or total_observed + content_length > PERF_MAX_SOURCE_TOTAL_BYTES
                ):
                    raise ValueError(
                        f"source pack record {ordinal} content length is invalid"
                    )
                digest = hashlib.sha256()
                remaining = content_length
                while remaining:
                    chunk = os.read(descriptor, min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(
                            f"source pack record {ordinal} content is truncated"
                        )
                    digest.update(chunk)
                    remaining -= len(chunk)
                if digest.hexdigest() != expected.get("sha256"):
                    raise ValueError(
                        f"source pack record {ordinal} content hash does not match manifest"
                    )
                total_observed += content_length

            if _read_exact_fd(
                descriptor, len(SOURCE_PACK_TRAILER_DOMAIN), "pack trailer domain"
            ) != SOURCE_PACK_TRAILER_DOMAIN:
                raise ValueError("source pack trailer domain is invalid")
            trailer_count = int.from_bytes(
                _read_exact_fd(descriptor, 8, "pack trailer count"), "big"
            )
            trailer_total = int.from_bytes(
                _read_exact_fd(descriptor, 8, "pack trailer total"), "big"
            )
            trailer_root = _read_exact_fd(descriptor, 32, "pack trailer root").hex()
            if (
                trailer_count != len(entries)
                or trailer_total != total_observed
                or trailer_total != total_expected
                or trailer_root != manifest.get("root_sha256")
            ):
                raise ValueError("source pack trailer is not manifest-derived")
            if os.read(descriptor, 1):
                raise ValueError("source pack contains trailing bytes")
        except (OSError, ValueError, TypeError, OverflowError) as error:
            reasons.append(str(error))
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            reasons.append("source input pack changed while being replayed")
    finally:
        os.close(descriptor)
    return reasons


def _build_receipt_reasons(
    *,
    root: Path,
    evidence_dir: Path,
    current_head: str,
    focr_payload: object,
    raw_meta: set[str],
    covered: set[str],
    input_paths: dict[str, Path],
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(focr_payload, dict):
        return ["focr payload is unavailable for build receipt replay"]
    try:
        receipt_bytes = _read_bounded_file(input_paths["build_receipt"], 1024 * 1024)
        manifest_bytes = _read_bounded_file(input_paths["source_input_manifest"], 32 * 1024 * 1024)
        receipt = _parse_strict_json_object(receipt_bytes, label="bundled build receipt")
        manifest = _parse_strict_json_object(manifest_bytes, label="bundled source manifest")
        binary_identity = _bounded_file_identity(
            input_paths["subject_binary"], PERF_MAX_SUBJECT_BINARY_BYTES
        )
    except (OSError, ValueError, KeyError) as error:
        return [f"bundled build provenance is unreadable: {error}"]
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
        reasons.append("build receipt fields/schema are noncanonical")
    if not _is_canonical_utc_seconds(receipt.get("created_utc")):
        reasons.append("build receipt created_utc is noncanonical")
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
        reasons.append("build receipt HEAD/profile/target/command is invalid")
    toolchain = receipt.get("toolchain")
    if (
        not isinstance(toolchain, dict)
        or set(toolchain) != {"rustc_verbose_version", "cargo_version", "rch_version"}
        or any(not isinstance(value, str) or not value.strip() for value in toolchain.values())
    ):
        reasons.append("build receipt toolchain identity is invalid")
    environment = receipt.get("build_environment")
    environment_fields = {
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
    if not isinstance(environment, dict) or set(environment) != environment_fields:
        reasons.append("build receipt environment fields are noncanonical")
    else:
        scalar_names = environment_fields - {
            "rustc_overrides",
            "release_perf_profile_overrides",
            "cargo_config_build_rustflags",
            "cargo_config_target",
        }
        if any(
            environment[name] is not None
            and not isinstance(environment[name], str)
            for name in scalar_names
        ):
            reasons.append("build receipt Rust flag environment is malformed")
        expected_target_env = "CARGO_TARGET_" + str(target).upper().replace("-", "_") + "_RUSTFLAGS"
        overrides = environment.get("rustc_overrides")
        profile_overrides = environment.get("release_perf_profile_overrides")
        if (
            environment.get("target_rustflags_env_name") != expected_target_env
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
            reasons.append("build receipt compiler/profile environment is malformed")
        for name in ("cargo_config_build_rustflags", "cargo_config_target"):
            value = environment.get(name)
            if (
                not isinstance(value, dict)
                or set(value) != {"status", "value"}
                or value.get("status") not in {"set", "unset"}
                or (value.get("status") == "unset" and value.get("value") is not None)
                or (value.get("status") == "set" and not isinstance(value.get("value"), str))
            ):
                reasons.append(f"build receipt {name} binding is malformed")

    root_sha256, manifest_reasons = _canonical_cert_source_root(
        manifest, current_head, root
    )
    reasons.extend(manifest_reasons)
    try:
        reasons.extend(
            _source_pack_reasons(input_paths["source_input_pack"], manifest)
        )
    except KeyError:
        reasons.append("bundled source input pack is missing")
    source = receipt.get("source_manifest")
    manifest_identity = {
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "size": len(manifest_bytes),
    }
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "path",
            "sha256",
            "size",
            "schema",
            "root_sha256",
            "entry_count",
            "root_hash_algorithm",
        }
        or not isinstance(source.get("path"), str)
        or not os.path.isabs(source["path"])
        or os.path.normpath(source["path"]) != source["path"]
        or {"sha256": source.get("sha256"), "size": source.get("size")}
        != manifest_identity
        or source.get("schema") != SOURCE_MANIFEST_SCHEMA
        or source.get("root_sha256") != root_sha256
        or source.get("entry_count") != manifest.get("entry_count")
        or source.get("root_hash_algorithm") != SOURCE_ROOT_ALGORITHM
    ):
        reasons.append("build receipt source-manifest binding is invalid")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
    }:
        reasons.append("build receipt required input bindings are noncanonical")
    else:
        for name in inputs:
            if inputs[name] != _workspace_entry_binding(manifest, name):
                reasons.append(f"build receipt {name} binding is invalid")
    receipt_binary = receipt.get("binary")
    if (
        not isinstance(receipt_binary, dict)
        or set(receipt_binary) != {"path", "sha256", "size"}
        or not isinstance(receipt_binary.get("path"), str)
        or not os.path.isabs(receipt_binary["path"])
        or os.path.normpath(receipt_binary["path"]) != receipt_binary["path"]
        or receipt_binary.get("sha256") != binary_identity["sha256"]
        or receipt_binary.get("size") != binary_identity["size"]
        or receipt_binary.get("path") != focr_payload.get("binary_origin")
    ):
        reasons.append("build receipt does not bind bundled binary and original origin")
    aggregate_fields = (
        "binary",
        "binary_sha256",
        "binary_size",
        "binary_origin",
        "build_receipt",
        "build_receipt_sha256",
    )
    capture_run_dir = focr_payload.get("run_dir")
    capture_root = (
        os.path.dirname(os.path.normpath(capture_run_dir))
        if isinstance(capture_run_dir, str)
        and os.path.isabs(capture_run_dir)
        and os.path.normpath(capture_run_dir) == capture_run_dir
        and os.path.basename(capture_run_dir) == "raw"
        else ""
    )
    expected_capture_binary = os.path.join(capture_root, "subject", "release-perf", "focr")
    expected_capture_receipt = os.path.join(capture_root, "subject", "build_receipt.json")
    if (
        not capture_root
        or focr_payload.get("binary") != expected_capture_binary
        or focr_payload.get("build_receipt") != expected_capture_receipt
        or focr_payload.get("binary_sha256") != binary_identity["sha256"]
        or focr_payload.get("binary_size") != binary_identity["size"]
        or focr_payload.get("build_receipt_sha256")
        != hashlib.sha256(receipt_bytes).hexdigest()
    ):
        reasons.append("focr aggregate does not bind canonical captured subject/receipt")
    expected_meta = {
        relative for relative in raw_meta if re.fullmatch(r"raw/run_\d{3}\.meta\.json", relative)
    }
    if len(expected_meta) != focr_payload.get("runs"):
        reasons.append("build receipt raw metadata count is not aggregate-bound")
    for relative in sorted(expected_meta):
        try:
            meta_path = _safe_repo_path(evidence_dir, relative)
            meta = _parse_strict_json_object(
                _read_bounded_file(meta_path) if meta_path is not None else b"",
                label=f"raw build metadata {relative}",
            )
        except (OSError, ValueError) as error:
            reasons.append(f"raw build metadata is unreadable ({relative}): {error}")
            continue
        drift = [field for field in aggregate_fields if meta.get(field) != focr_payload.get(field)]
        if drift:
            reasons.append(f"raw build identity drifted ({relative}): " + ", ".join(drift))
    required_paths = {
        PERF_INPUT_BINDINGS[name]
        for name in (
            "build_receipt",
            "source_input_manifest",
            "source_input_pack",
            "subject_binary",
        )
    }
    if not required_paths.issubset(covered):
        reasons.append("build provenance files are not exhaustively manifested")
    return reasons


def _cert_reference_model_root(files: list[dict]) -> str:
    digest = hashlib.sha256(REFERENCE_MODEL_ROOT_DOMAIN)
    for item in files:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _reference_provenance_reasons(
    *,
    root: Path,
    row: dict,
    ref_payload: object,
    input_paths: dict[str, Path],
    covered: set[str],
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(ref_payload, dict):
        return ["reference payload is unavailable for provenance replay"]
    try:
        manifest_bytes = _read_bounded_file(input_paths["reference_model_manifest"])
        binding_bytes = _read_bounded_file(input_paths["reference_inference_binding"])
        manifest = _parse_strict_json_object(manifest_bytes, label="reference model manifest")
        binding = _parse_strict_json_object(binding_bytes, label="reference inference binding")
    except (OSError, ValueError, KeyError) as error:
        return [f"bundled reference provenance is unreadable: {error}"]
    expected_files = [
        {"path": path, "bytes": size, "sha256": sha256}
        for path, size, sha256 in UNLIMITED_REFERENCE_MODEL_FILES
    ]
    manifest_fields = {
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
    expected_model_root = _cert_reference_model_root(expected_files)
    if (
        set(manifest) != manifest_fields
        or manifest.get("schema") != REFERENCE_MODEL_MANIFEST_SCHEMA
        or manifest.get("model_id") != "baidu/Unlimited-OCR"
        or manifest.get("model_commit") != UNLIMITED_OCR_MODEL_COMMIT
        or manifest.get("synthetic") is not False
        or manifest.get("citable") is not True
        or manifest.get("file_count") != 12
        or manifest.get("root_hash_domain") != REFERENCE_MODEL_ROOT_DOMAIN.decode("ascii")
        or manifest.get("root_sha256") != expected_model_root
        or manifest.get("index") != UNLIMITED_REFERENCE_MODEL_INDEX
        or manifest.get("files") != expected_files
    ):
        reasons.append("reference model manifest is not the exact pinned 12-file truth pack")
    if not _json_exact(ref_payload.get("reference_model_manifest"), manifest):
        reasons.append("embedded and bundled reference model manifests differ")
    if row.get("model_commit") != manifest.get("model_commit"):
        reasons.append("ledger model_commit is not reference-manifest bound")

    binding_fields = {
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
    if set(binding) != binding_fields:
        reasons.append("reference inference binding fields are noncanonical")
        return reasons
    argv = binding.get("argv")
    entry = "gauntlet_ref_unlimited:run_stage"
    setup = "gauntlet_ref_unlimited:setup"
    fixture = _fixture_identity(row.get("fixture_hash"))
    if (
        binding.get("schema") != REFERENCE_INFERENCE_BINDING_SCHEMA
        or binding.get("model_root_sha256") != expected_model_root
        or binding.get("model_commit") != UNLIMITED_OCR_MODEL_COMMIT
        or binding.get("entry") != entry
        or binding.get("setup") != setup
        or binding.get("page") != ref_payload.get("page")
        or binding.get("page_sha256") != ref_payload.get("page_sha256")
        or fixture is None
        or Path(str(binding.get("page"))).name != fixture[0]
        or binding.get("page_sha256") != fixture[1]
        or binding.get("model_dir") != ref_payload.get("model")
        or binding.get("max_length") != ref_payload.get("max_length")
        or binding.get("text_dir") != ref_payload.get("text_dir")
        or binding.get("backend") != ref_payload.get("backend")
        or binding.get("precision") != ref_payload.get("precision")
        or binding.get("threads") != ref_payload.get("threads")
        or binding.get("runs") != ref_payload.get("runs")
        or binding.get("warmup") != ref_payload.get("warmup")
        or binding.get("allocator") != ref_payload.get("allocator")
        or argv != ref_payload.get("command")
        or binding.get("env_pins") != ref_payload.get("env_pins")
        or binding.get("ambient_env") != ref_payload.get("ambient_env")
        or binding.get("torch_version") != ref_payload.get("torch_version")
        or binding.get("transformers_version") != ref_payload.get("transformers_version")
        or binding.get("reference_contract") != ref_payload.get("reference_contract")
        or not isinstance(argv, list)
        or _argv_option(argv, "--entry") != entry
        or _argv_option(argv, "--setup") != setup
        or _argv_option(argv, "--page") != ref_payload.get("page")
        or _argv_option(argv, "--model-dir") != ref_payload.get("model")
        or _argv_option(argv, "--backend") != ref_payload.get("backend")
        or _argv_option(argv, "--precision") != ref_payload.get("precision")
        or _argv_option(argv, "--max-length")
        != str(ref_payload.get("max_length"))
        or _argv_option(argv, "--text-dir") != ref_payload.get("text_dir")
        or _argv_option(argv, "--threads") != str(ref_payload.get("threads"))
        or _argv_option(argv, "--runs") != str(ref_payload.get("runs"))
        or _argv_option(argv, "--warmup") != str(ref_payload.get("warmup"))
    ):
        reasons.append("reference inference binding disagrees with ledger/runtime/argv")
    if binding.get("stage") not in {
        "all",
        "preprocess",
        "vision_encode",
        "prefill",
        "decode_per_token",
        "end_to_end",
    }:
        reasons.append("reference inference binding stage is invalid")
    output = _argv_option(argv, "--out") if isinstance(argv, list) else None
    expected_argv = [
        argv[0] if isinstance(argv, list) and argv else "",
        "--stage",
        str(binding.get("stage")),
        "--page",
        str(ref_payload.get("page")),
        "--model-dir",
        str(ref_payload.get("model")),
        "--backend",
        str(ref_payload.get("backend")),
        "--precision",
        str(ref_payload.get("precision")),
        "--max-length",
        str(ref_payload.get("max_length")),
        "--text-dir",
        str(ref_payload.get("text_dir")),
        "--entry",
        entry,
        "--setup",
        setup,
        "--runs",
        str(ref_payload.get("runs")),
        "--warmup",
        str(ref_payload.get("warmup")),
        "--threads",
        str(ref_payload.get("threads")),
        "--out",
        str(output),
    ]
    if (
        not isinstance(argv, list)
        or not argv
        or os.path.basename(argv[0]) != "gauntlet_reference.py"
        or argv != expected_argv
    ):
        reasons.append("reference inference argv is not the canonical runbook invocation")
    if (
        ref_payload.get("max_length") != 8192
        or not isinstance(ref_payload.get("text_dir"), str)
        or not os.path.isabs(ref_payload["text_dir"])
        or binding.get("ambient_env")
        != {"FOCR_REF_MAX_LENGTH": "<unset>", "FOCR_REF_TEXT_DIR": "<unset>"}
    ):
        reasons.append("reference max-length/text output ambient contract is invalid")
    pins = binding.get("env_pins")
    expected_pin_names = {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "FOCR_THREADS",
    }
    if not isinstance(pins, dict) or set(pins) != expected_pin_names or any(
        value != str(binding.get("threads")) for value in pins.values()
    ):
        reasons.append("reference inference thread pins are incomplete")
    cache = binding.get("hf_modules_cache")
    expected_parent = os.path.dirname(os.path.abspath(str(output)))
    expected_cache = os.path.basename(str(output)) + ".hf_modules_cache"
    if (
        not isinstance(cache, dict)
        or set(cache) != {"evidence_dir", "path", "effective_path", "fresh"}
        or cache.get("evidence_dir") != expected_parent
        or cache.get("path") != expected_cache
        or cache.get("effective_path") != os.path.join(expected_parent, expected_cache)
        or cache.get("fresh") is not True
    ):
        reasons.append("reference inference HF module cache is not evidence-local/fresh")
    expected_sources = (
        ("harness", "gauntlet_reference:main", "scripts/gauntlet_reference.py"),
        ("entry", entry, "scripts/gauntlet_ref_unlimited.py"),
        ("setup", setup, "scripts/gauntlet_ref_unlimited.py"),
    )
    sources = binding.get("sources")
    if not isinstance(sources, list) or len(sources) != len(expected_sources):
        reasons.append("reference inference source census is incomplete")
    else:
        for source, (role, callable_name, relative) in zip(sources, expected_sources, strict=True):
            if (
                not isinstance(source, dict)
                or set(source) != {"role", "callable", "path", "bytes", "sha256"}
                or source.get("role") != role
                or source.get("callable") != callable_name
                or source.get("path") != relative
            ):
                reasons.append("reference inference source identity is noncanonical")
                continue
            path = _safe_repo_path(root, relative)
            try:
                identity = _bounded_file_identity(path, PERF_MAX_SOURCE_FILE_BYTES) if path else None
            except (OSError, ValueError):
                identity = None
            if identity != {"sha256": source.get("sha256"), "size": source.get("bytes")}:
                reasons.append(f"reference inference source hash drifted: {relative}")
    unsigned = dict(binding)
    unsigned.pop("binding_hash_domain", None)
    unsigned.pop("binding_sha256", None)
    try:
        canonical = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError):
        canonical = b""
    expected_binding_hash = hashlib.sha256(
        REFERENCE_INFERENCE_BINDING_DOMAIN + canonical
    ).hexdigest()
    if (
        binding.get("binding_hash_domain")
        != REFERENCE_INFERENCE_BINDING_DOMAIN.decode("ascii")
        or binding.get("binding_sha256") != expected_binding_hash
    ):
        reasons.append("reference inference binding hash is invalid")
    if not _json_exact(ref_payload.get("reference_inference_binding"), binding):
        reasons.append("embedded and bundled reference inference bindings differ")
    required = {
        PERF_INPUT_BINDINGS["reference_model_manifest"],
        PERF_INPUT_BINDINGS["reference_inference_binding"],
    }
    if not required.issubset(covered):
        reasons.append("reference provenance files are not exhaustively manifested")
    return reasons


def perf_evidence_verdict(
    perf_text: str,
    root: Path,
    current_head: str,
    now: datetime | None = None,
    max_age_hours: float = CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
    *,
    lineage_root: Path | None = None,
) -> dict:
    """Find a current, eligible Unlimited-OCR decode-per-token proof row."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    lineage_root = lineage_root or root
    candidates: list[dict[str, object]] = []
    for row in _perf_ledger_rows(perf_text):
        stage = row.get("stage", "").replace("_", "-")
        if (
            row.get("model_commit") != UNLIMITED_OCR_MODEL_COMMIT
            or stage != "decode-per-token"
        ):
            continue
        reasons: list[str] = []
        claim_id = row.get("claim_id", "<missing-claim>")
        try:
            focr_ms = float(row.get("focr_ms", "nan"))
            ref_ms = float(row.get("ref_ms", "nan"))
            ratio = float(row.get("ratio", "nan"))
            expected_ratio = ref_ms / focr_ms
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (focr_ms, ref_ms, ratio)
            ):
                reasons.append("non-positive or non-finite timing/ratio")
            elif ratio <= 1.0:
                reasons.append(f"decode ratio does not beat reference: {ratio:.3f}")
            elif abs(ratio - expected_ratio) / expected_ratio > 0.01:
                reasons.append("ratio does not match ref_ms / focr_ms")
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            reasons.append("invalid timing/ratio cells")

        command = row.get("command/env", "")
        if "release-perf" not in command:
            reasons.append("subject was not measured from the release-perf profile")
        precision = row.get("precision (focr vs ref)", "")
        if not precision.startswith(f"{CURRENT_UNLIMITED_PRECISION} vs "):
            reasons.append(
                "ledger precision is historical, ambiguous, or outside the current "
                "conservative Unlimited release contract"
            )
        thread_match = re.fullmatch(
            r"focr=ref=(\d+)", row.get("threads (focr=ref N)", "")
        )
        if (
            not thread_match
            or int(thread_match.group(1)) <= 0
            or int(thread_match.group(1)) == 64
        ):
            reasons.append("invalid or unfair thread-parity proof")
        correctness_claim, correctness_reason = _correctness_claim(
            row.get("correctness_proof")
        )
        if correctness_reason:
            reasons.append(correctness_reason)
        if row.get("reference_backend", "").lower() in {"", "unknown", "n/a"}:
            reasons.append("missing reference backend identity")

        evidence_id = row.get("evidence_id", "").rstrip("/")
        evidence_dir = _safe_repo_path(root, evidence_id)
        if evidence_dir is None or not evidence_dir.is_dir():
            reasons.append(f"missing or unsafe evidence directory: {evidence_id}")
        else:
            manifest_ok, manifest_reasons, covered = _verify_sha256_manifest(
                evidence_dir
            )
            if not manifest_ok:
                reasons.extend(manifest_reasons)
            try:
                row_doc = _load_bounded_json(evidence_dir / "row.json")
                if not isinstance(row_doc, dict):
                    raise TypeError("row.json is not a JSON object")
                row_v3_fields = {
                    "schema",
                    "created_utc",
                    "source_git_head",
                    "source_root",
                    "producer_root",
                    "producer_root_algorithm",
                    "allowed_evidence_path",
                    "inputs",
                    "timing_inputs",
                    "correctness_inputs",
                    "rows",
                }
                if (
                    row_doc.get("schema") != "focr-gauntlet-row/v3"
                    or set(row_doc) != row_v3_fields
                ):
                    reasons.append("row.json schema/fields are not focr-gauntlet-row/v3")
                age_reason = _age_reason(
                    _parse_utc_timestamp(row_doc.get("created_utc")),
                    now,
                    "row.json",
                    max_age_hours,
                )
                if age_reason:
                    reasons.append(age_reason)
                source_git_head = str(row_doc.get("source_git_head", ""))
                allowed_evidence_path = row_doc.get("allowed_evidence_path")
                if allowed_evidence_path != evidence_id:
                    reasons.append(
                        "row allowed_evidence_path does not equal the ledger evidence_id"
                    )
                reasons.extend(
                    _evidence_descendant_reasons(
                        lineage_root,
                        source_git_head,
                        current_head,
                        allowed_evidence_path,
                    )
                )
                if row_doc.get("producer_root_algorithm") != PRODUCER_ROOT_ALGORITHM:
                    reasons.append("row producer_root_algorithm is unsupported")
                expected_producer_root = _gauntlet_producer_root(
                    lineage_root, source_git_head
                )
                if (
                    expected_producer_root is None
                    or row_doc.get("producer_root") != expected_producer_root
                ):
                    reasons.append(
                        "row producer_root does not match source-HEAD validator/config blobs"
                    )
                if re.fullmatch(
                    r"[0-9a-f]{64}", str(row_doc.get("source_root", ""))
                ) is None:
                    reasons.append("row source_root is noncanonical")
                local_rows = row_doc.get("rows")
                if not isinstance(local_rows, list):
                    local_rows = []
                    reasons.append("row.json rows is not a list")
                matching_rows = [
                    item
                    for item in local_rows
                    if isinstance(item, dict)
                    and item.get("claim_id") == claim_id
                    and str(item.get("stage", "")).replace("_", "-")
                    == "decode-per-token"
                ]
                bound_correctness_claim: dict | None = None
                if len(matching_rows) != 1:
                    reasons.append(
                        f"row.json must contain exactly one ledger claim/stage row; found {len(matching_rows)}"
                    )
                    bound_row: dict = {}
                else:
                    bound_row = matching_rows[0]
                    mismatched_columns = [
                        column
                        for column, value in row.items()
                        if str(bound_row.get(column, "")) != value
                    ]
                    if mismatched_columns:
                        reasons.append(
                            "ledger row differs from hashed row.json columns: "
                            + ", ".join(mismatched_columns)
                        )
                    bound_correctness_claim, bound_correctness_reason = (
                        _correctness_claim(bound_row.get("correctness_proof"))
                    )
                    if bound_correctness_reason:
                        reasons.append("hashed row: " + bound_correctness_reason)

                inputs = row_doc.get("inputs")
                if not isinstance(inputs, dict):
                    inputs = {}
                    reasons.append("row.json inputs is not an object")
                expected_inputs = set(PERF_INPUT_BINDINGS)
                if set(inputs) != expected_inputs:
                    reasons.append(
                        "row.json inputs must bind exactly: "
                        + ", ".join(sorted(expected_inputs))
                    )

                def bound_input(name: str, fallback: str) -> tuple[Path, dict]:
                    record = inputs.get(name)
                    if not isinstance(record, dict) or set(record) != {
                        "bundle_path",
                        "sha256",
                        "size",
                    }:
                        reasons.append(f"row.json inputs.{name} is missing")
                        record = {}
                    bundle_path = record.get("bundle_path")
                    if bundle_path != fallback:
                        reasons.append(
                            f"row.json inputs.{name}.bundle_path is not canonical: {bundle_path!r}"
                        )
                        bundle_path = fallback
                    if bundle_path not in covered:
                        reasons.append(f"manifest omits bound input: {bundle_path}")
                    path = _safe_repo_path(evidence_dir, bundle_path)
                    if path is None or not path.is_file():
                        reasons.append(
                            f"bound input is missing or unsafe: {bundle_path}"
                        )
                        path = evidence_dir / "__missing__"
                    expected_sha = record.get("sha256")
                    expected_size = record.get("size")
                    if path.is_file():
                        try:
                            maximum = (
                                PERF_MAX_SUBJECT_BINARY_BYTES
                                if name == "subject_binary"
                                else PERF_MAX_SOURCE_PACK_BYTES
                                if name == "source_input_pack"
                                else CERTIFICATION_MAX_ARTIFACT_BYTES
                            )
                            identity = _bounded_file_identity(path, maximum)
                        except (OSError, ValueError) as error:
                            identity = None
                            reasons.append(
                                f"row.json inputs.{name} is unreadable: {error}"
                            )
                        if identity != {"sha256": expected_sha, "size": expected_size}:
                            reasons.append(
                                f"row.json inputs.{name} does not bind {bundle_path}"
                            )
                    return path, record

                if "row.json" not in covered:
                    reasons.append("manifest omits: row.json")
                focr_path, _focr_input = bound_input("focr_stages", "focr_stages.json")
                ref_path, _ref_input = bound_input("ref_stages", "ref_stages.json")
                roofline_path, _roofline_input = bound_input(
                    "roofline", "roofline.json"
                )
                correctness_path, _correctness_input = bound_input(
                    "correctness_receipt", "correctness_receipt.json"
                )
                input_paths = {
                    "focr_stages": focr_path,
                    "ref_stages": ref_path,
                    "roofline": roofline_path,
                    "correctness_receipt": correctness_path,
                }
                for name in (
                    "build_receipt",
                    "source_input_manifest",
                    "source_input_pack",
                    "subject_binary",
                    "reference_model_manifest",
                    "reference_inference_binding",
                ):
                    path, _record = bound_input(name, PERF_INPUT_BINDINGS[name])
                    input_paths[name] = path
                source_manifest_payload = _load_bounded_json(
                    input_paths["source_input_manifest"]
                )
                if (
                    not isinstance(source_manifest_payload, dict)
                    or source_manifest_payload.get("root_sha256")
                    != row_doc.get("source_root")
                ):
                    reasons.append(
                        "row source_root does not match the bundled source manifest"
                    )
                required_bundle_files = {"PERF_LEDGER_ROW.md", "row.json"}
                for required_file in required_bundle_files:
                    if required_file not in covered:
                        reasons.append(
                            f"manifest omits required evidence: {required_file}"
                        )
                raw_stdout = {
                    path
                    for path in covered
                    if path.startswith("raw/")
                    and path.endswith(".stdout")
                    and not Path(path).name.startswith("._")
                }
                raw_stderr = {
                    path
                    for path in covered
                    if path.startswith("raw/")
                    and path.endswith(".stderr")
                    and not Path(path).name.startswith("._")
                }
                raw_meta = {
                    path
                    for path in covered
                    if path.startswith("raw/")
                    and path.endswith(".meta.json")
                    and not Path(path).name.startswith("._")
                }
                if any(
                    len(paths) > PERF_MAX_RAW_TIMING_FILES
                    for paths in (raw_stdout, raw_stderr, raw_meta)
                ):
                    reasons.append("raw timing file count exceeds the verifier bound")
                if not raw_stdout or not raw_stderr or not raw_meta:
                    reasons.append(
                        "manifest lacks mandatory raw stdout/stderr/run metadata"
                    )
                elif raw_stderr:
                    reasons.extend(
                        _runtime_decode_marker_reasons(evidence_dir, raw_stderr)
                    )

                timing_inputs = row_doc.get("timing_inputs")
                if not isinstance(timing_inputs, dict) or set(timing_inputs) != {
                    "focr",
                    "reference",
                }:
                    reasons.append(
                        "row.json timing_inputs must bind focr and reference raw observations"
                    )
                    timing_inputs = {}
                bundled_raw_timing: dict[str, object] = {}
                for source in ("focr", "reference"):
                    binding = timing_inputs.get(source)
                    expected_relative = f"raw/{source}_timing.json"
                    if (
                        not isinstance(binding, dict)
                        or set(binding) != {"bundle_path", "sha256"}
                        or binding.get("bundle_path") != expected_relative
                    ):
                        reasons.append(
                            f"row.json timing_inputs.{source} is not canonical"
                        )
                        continue
                    path = _safe_repo_path(evidence_dir, expected_relative)
                    try:
                        content = _read_bounded_file(path) if path is not None else b""
                        parsed = _parse_json_bytes(
                            content, label=f"{source} bundled raw timing"
                        )
                    except (OSError, ValueError, TypeError, RecursionError) as error:
                        reasons.append(
                            f"{source} bundled raw timing is unreadable: {error}"
                        )
                        continue
                    if (
                        expected_relative not in covered
                        or hashlib.sha256(content).hexdigest() != binding.get("sha256")
                        or not isinstance(parsed, dict)
                    ):
                        reasons.append(
                            f"row.json timing_inputs.{source} does not bind bundled raw timing"
                        )
                        continue
                    bundled_raw_timing[source] = parsed
                if bound_row:
                    expected_row_line = (
                        "| "
                        + " | ".join(str(bound_row.get(column, "")) for column in row)
                        + " |"
                    )
                    try:
                        emitted_rows = (
                            _read_bounded_text(evidence_dir / "PERF_LEDGER_ROW.md")
                            .splitlines()
                        )
                    except (OSError, ValueError) as error:
                        emitted_rows = []
                        reasons.append(f"PERF_LEDGER_ROW.md is unreadable: {error}")
                    if emitted_rows.count(expected_row_line) != 1:
                        reasons.append(
                            "PERF_LEDGER_ROW.md does not contain exactly one bound claim row"
                        )
                focr_payload = _load_bounded_json(focr_path)
                ref_payload = _load_bounded_json(ref_path)
                for source, payload in (
                    ("focr", focr_payload),
                    ("reference", ref_payload),
                ):
                    if (
                        not isinstance(payload, dict)
                        or not _json_exact(
                            bundled_raw_timing.get(source), payload.get("raw_timing")
                        )
                    ):
                        reasons.append(
                            f"{source} stage document raw timing disagrees with its bundled input"
                        )
                for side, payload in (
                    ("focr", focr_payload),
                    ("reference", ref_payload),
                ):
                    if isinstance(payload, dict):
                        stage_age_reason = _age_reason(
                            _parse_utc_timestamp(payload.get("created_utc")),
                            now,
                            f"{side} stage measurement",
                            max_age_hours,
                        )
                        if stage_age_reason:
                            reasons.append(stage_age_reason)
                reasons.extend(
                    _build_receipt_reasons(
                        root=root,
                        evidence_dir=evidence_dir,
                        current_head=source_git_head,
                        focr_payload=focr_payload,
                        raw_meta=raw_meta,
                        covered=covered,
                        input_paths=input_paths,
                    )
                )
                reasons.extend(
                    _reference_provenance_reasons(
                        root=root,
                        row=bound_row,
                        ref_payload=ref_payload,
                        input_paths=input_paths,
                        covered=covered,
                    )
                )
                contract_reasons, focr_stage, ref_stage = _measurement_contract(
                    focr_payload,
                    ref_payload,
                    bound_row,
                    int(thread_match.group(1)) if thread_match else None,
                )
                reasons.extend(contract_reasons)
                if isinstance(focr_payload, dict) and focr_stage is not None:
                    reasons.extend(
                        _raw_run_metadata_reasons(
                            evidence_dir,
                            raw_meta,
                            raw_stdout,
                            raw_stderr,
                            focr_payload,
                            focr_stage,
                        )
                    )
                    reasons.extend(
                        _physical_focr_timing_reasons(
                            evidence_dir,
                            raw_meta,
                            raw_stdout,
                            raw_stderr,
                            focr_payload,
                            focr_stage,
                        )
                    )
                if focr_stage is not None and ref_stage is not None:
                    try:
                        stage_focr_ms = float(focr_stage["best_ms"])
                        stage_ref_ms = float(ref_stage["best_ms"])
                        expected_focr = f"{stage_focr_ms:.3f}"
                        expected_ref = f"{stage_ref_ms:.3f}"
                        expected_ratio_text = f"{stage_ref_ms / stage_focr_ms:.3f}"
                        if bound_row.get("focr_ms") != expected_focr:
                            reasons.append(
                                "hashed row focr_ms does not match stage best_ms"
                            )
                        if bound_row.get("ref_ms") != expected_ref:
                            reasons.append(
                                "hashed row ref_ms does not match stage best_ms"
                            )
                        if bound_row.get("ratio") != expected_ratio_text:
                            reasons.append(
                                "hashed row ratio does not match stage-derived ratio"
                            )
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        OverflowError,
                        ZeroDivisionError,
                    ):
                        reasons.append("stage timing evidence is missing or invalid")
                    required_raw_count = max(
                        int(focr_stage.get("n", 0)), int(ref_stage.get("n", 0))
                    )
                    if (
                        len(raw_stdout) < required_raw_count
                        or len(raw_stderr) < required_raw_count
                    ):
                        reasons.append("raw logs do not cover every measured run")

                    try:
                        roofline_payload = _load_bounded_json(roofline_path)
                    except (
                        OSError,
                        ValueError,
                        TypeError,
                        RecursionError,
                    ) as error:
                        roofline_payload = {}
                        reasons.append(f"roofline evidence is unreadable: {error}")
                    if not isinstance(roofline_payload, dict):
                        roofline_payload = {}
                        reasons.append("roofline evidence is not a JSON object")
                    if roofline_payload.get("schema") != "focr-gauntlet-roofline/v1":
                        reasons.append("roofline evidence has an unsupported schema")
                    if roofline_payload.get("synthetic") is not False:
                        reasons.append("roofline evidence is synthetic or unstamped")
                    focr_stages_sha256 = hashlib.sha256(
                        _read_bounded_file(focr_path)
                    ).hexdigest()
                    reasons.extend(
                        _current_unlimited_roofline_reasons(
                            roofline_payload,
                            focr_payload,
                            focr_stages_sha256=focr_stages_sha256,
                        )
                    )
                    floors = roofline_payload.get("floors")
                    matching_floors = (
                        [
                            floor
                            for floor in floors
                            if isinstance(floor, dict)
                            and str(floor.get("stage", "")).replace("-", "_")
                            == "decode_per_token"
                        ]
                        if isinstance(floors, list)
                        else []
                    )
                    if len(matching_floors) != 1:
                        reasons.append(
                            "roofline must contain exactly one decode-per-token floor"
                        )
                    else:
                        floor = matching_floors[0]
                        try:
                            floor_ms = float(floor["floor_ms"])
                            if not math.isfinite(floor_ms) or floor_ms <= 0.0:
                                raise ValueError("invalid floor")
                            if bound_row.get("floor_kind") != floor.get("floor_kind"):
                                reasons.append(
                                    "hashed row floor_kind does not match roofline"
                                )
                            if bound_row.get("floor_ms") != f"{floor_ms:.3f}":
                                reasons.append(
                                    "hashed row floor_ms does not match roofline"
                                )
                            if (
                                bound_row.get("dist_above_floor")
                                != f"{stage_focr_ms / floor_ms:.2f}"
                            ):
                                reasons.append(
                                    "hashed row dist_above_floor does not match roofline"
                                )
                        except (
                            KeyError,
                            TypeError,
                            ValueError,
                            OverflowError,
                            ZeroDivisionError,
                        ):
                            reasons.append("roofline floor is missing or invalid")

                reasons.extend(
                    _correctness_receipt_reasons(
                        correctness_path,
                        bound_correctness_claim or correctness_claim,
                        evidence_dir=evidence_dir,
                        row_doc=row_doc,
                        focr_payload=focr_payload,
                        ref_payload=ref_payload,
                        covered=covered,
                    )
                )
            except (
                AttributeError,
                OSError,
                ValueError,
                TypeError,
                OverflowError,
                RecursionError,
            ) as error:
                reasons.append(f"unreadable structured evidence: {error}")
        candidates.append(
            {"claim_id": claim_id, "eligible": not reasons, "reasons": reasons}
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    return {
        "ok": bool(eligible),
        "eligible_claims": [candidate["claim_id"] for candidate in eligible],
        "candidates": candidates,
        "reason": (
            "eligible current Unlimited-OCR decode evidence found"
            if eligible
            else "no eligible current Unlimited-OCR decode-per-token evidence"
        ),
    }


def _bundle_root_sha256(manifest: Sequence[dict]) -> str | None:
    paths: set[str] = set()
    bindings: list[tuple[str, str, str, str]] = []
    for entry in manifest:
        if not isinstance(entry, dict) or "error" in entry:
            return None
        relative = entry.get("artifact")
        digest = entry.get("sha256")
        timestamp = entry.get("timestamp_utc")
        timestamp_source = entry.get("timestamp_source")
        if (
            not isinstance(relative, str)
            or relative in paths
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or _parse_utc_timestamp(timestamp) is None
            or not isinstance(timestamp_source, str)
            or not timestamp_source
        ):
            return None
        paths.add(relative)
        bindings.append((relative, digest, str(timestamp), timestamp_source))
    if not bindings:
        return None
    canonical = "".join(
        f"{relative}\0{digest}\0{timestamp}\0{source}\n"
        for relative, digest, timestamp, source in sorted(bindings)
    )
    return _sha256_text(canonical)


def _trusted_signer_fingerprints(
    root: Path, content: bytes | None = None
) -> dict[str, str]:
    path = _safe_repo_path(root, TRUSTED_SIGNERS_PATH)
    if path is None or not path.is_file():
        return {}
    try:
        payload = _parse_json_bytes(
            content if content is not None else _read_bounded_file(path),
            label=str(path),
        )
    except (OSError, ValueError, TypeError, RecursionError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "gauntlet.trusted_signers.v1"
        or not isinstance(payload.get("signers"), list)
    ):
        return {}
    trusted: dict[str, str] = {}
    for signer in payload["signers"]:
        if not isinstance(signer, dict) or signer.get("active") is not True:
            continue
        identity = signer.get("identity")
        fingerprint = str(signer.get("fingerprint", "")).upper()
        if (
            isinstance(identity, str)
            and identity
            and re.fullmatch(r"[0-9A-F]{40,64}", fingerprint)
        ):
            trusted[identity] = fingerprint
    return trusted


def _certificate_signed_claim_sha256(
    certificate: dict, bundle_root: str | None
) -> str | None:
    if (
        not isinstance(bundle_root, str)
        or re.fullmatch(r"[0-9a-f]{64}", bundle_root) is None
    ):
        return None
    signatures = certificate.get("detached_signatures")
    if not isinstance(signatures, list):
        return None
    declarations: list[dict] = []
    signature_keys = {
        "signer",
        "role",
        "fingerprint",
        "scheme",
        "signature_path",
    }
    for signature in signatures:
        if not isinstance(signature, dict) or set(signature) != signature_keys:
            return None
        declarations.append(
            {
                "signer": signature.get("signer"),
                "role": signature.get("role"),
                "fingerprint": signature.get("fingerprint"),
                "scheme": signature.get("scheme"),
                "signature_path": signature.get("signature_path"),
            }
        )
    claim = {
        key: value
        for key, value in certificate.items()
        if key not in {"detached_signatures", "signed_claim_sha256"}
    }
    claim["detached_signature_declarations"] = declarations
    claim["evidence_bundle_sha256"] = bundle_root
    try:
        canonical = json.dumps(
            claim,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError):
        return None
    return _sha256_text(canonical)


def _final_report_text(certificate: dict, bundle: dict) -> str:
    certified = certificate.get("certified") is True
    refusal_reasons = certificate.get("refusal_reasons")
    reasons = refusal_reasons if isinstance(refusal_reasons, list) else []
    signers = certificate.get("signers")
    signer_names = signers if isinstance(signers, list) else []
    lines = [
        "# FINAL GAUNTLET REPORT",
        "",
        f"* Certification verdict: **{'CERTIFIED' if certified else 'NOT CERTIFIED'}**",
        f"* Project: `{certificate.get('project', '')}`",
        f"* Version: `{certificate.get('version', '')}`",
        f"* Git HEAD: `{certificate.get('git_head', '')}`",
        f"* Final evidence HEAD: `{certificate.get('evidence_git_head', '')}`",
        f"* Bundle root: `{bundle.get('bundle_root_sha256') or 'unavailable'}`",
        f"* Signed claim: `{certificate.get('signed_claim_sha256') or 'unavailable'}`",
        f"* Signers: {', '.join(map(str, signer_names)) if signer_names else 'none'}",
        "",
        "## Refusal Reasons",
        "",
    ]
    lines.extend(
        [f"{index}. {reason}" for index, reason in enumerate(reasons, start=1)]
        or ["None."]
    )
    return "\n".join(lines) + "\n"


def _default_signature_verifier(
    root: Path,
    signature: dict,
    signed_claim_sha256: str,
    *,
    signature_bytes: bytes | None = None,
    keyring_bytes: bytes | None = None,
) -> bool:
    verifier = shutil.which("gpgv")
    signature_path = _safe_repo_path(root, str(signature.get("signature_path", "")))
    keyring_path = _safe_repo_path(root, TRUSTED_KEYRING_PATH)
    expected_fingerprint = str(signature.get("fingerprint", "")).upper()
    if (
        verifier is None
        or signature_path is None
        or keyring_path is None
        or not signature_path.is_file()
        or not keyring_path.is_file()
        or re.fullmatch(r"[0-9A-F]{40,64}", expected_fingerprint) is None
    ):
        return False
    try:
        detached = (
            signature_bytes
            if signature_bytes is not None
            else _read_bounded_file(signature_path)
        )
        trusted_keyring = (
            keyring_bytes
            if keyring_bytes is not None
            else _read_bounded_file(keyring_path)
        )
        with tempfile.TemporaryDirectory(prefix="focr-cert-signature-") as tmp:
            payload = Path(tmp) / "bundle-root.txt"
            signature_snapshot = Path(tmp) / "signature.asc"
            keyring_snapshot = Path(tmp) / "trusted-release-keys.gpg"
            payload.write_text(signed_claim_sha256 + "\n", encoding="utf-8")
            signature_snapshot.write_bytes(detached)
            keyring_snapshot.write_bytes(trusted_keyring)
            result = subprocess.run(
                [
                    verifier,
                    "--batch",
                    "--no-default-keyring",
                    "--homedir",
                    tmp,
                    "--status-fd=1",
                    "--keyring",
                    str(keyring_snapshot),
                    str(signature_snapshot),
                    str(payload),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    valid_fingerprints = {
        match.group(1).upper()
        for match in re.finditer(
            r"^\[GNUPG:\] VALIDSIG ([0-9A-F]+)\b", result.stdout, re.MULTILINE
        )
    }
    return result.returncode == 0 and valid_fingerprints == {expected_fingerprint}


def _default_ci_run_verifier(
    root: Path,
    run_id: str,
    current_head: str,
    artifacts: Sequence[dict],
) -> bool:
    gh = shutil.which("gh")
    if gh is None or re.fullmatch(r"[1-9][0-9]*", run_id) is None:
        return False
    expected_repositories = {
        item.get("source_ci_repository") for item in artifacts if isinstance(item, dict)
    }
    expected_workflows = {
        item.get("source_ci_workflow") for item in artifacts if isinstance(item, dict)
    }
    expected_events = {
        item.get("source_ci_event") for item in artifacts if isinstance(item, dict)
    }
    if (
        expected_repositories != {CERTIFICATION_GITHUB_REPOSITORY}
        or len(expected_workflows) != 1
        or not expected_workflows
        <= {
            CERTIFICATION_GITHUB_WORKFLOW,
            CERTIFICATION_DIST_WORKFLOW,
            CERTIFICATION_MODEL_PARITY_WORKFLOW,
            CERTIFICATION_PERFORMANCE_WORKFLOW,
        }
    ):
        return False
    expected_workflow = next(iter(expected_workflows))
    expected_event = {
        CERTIFICATION_MODEL_PARITY_WORKFLOW: CERTIFICATION_MODEL_PARITY_EVENT,
        CERTIFICATION_PERFORMANCE_WORKFLOW: CERTIFICATION_PERFORMANCE_EVENT,
    }.get(expected_workflow, CERTIFICATION_GITHUB_EVENT)
    if expected_events != {expected_event}:
        return False
    try:
        result = subprocess.run(
            [
                gh,
                "run",
                "view",
                run_id,
                "--repo",
                CERTIFICATION_GITHUB_REPOSITORY,
                "--json",
                "databaseId,headSha,headBranch,status,conclusion,workflowName,event,createdAt,updatedAt,jobs",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        payload = (
            _parse_json_bytes(result.stdout.encode("utf-8"), label="gh run view")
            if result.returncode == 0
            else {}
        )
        api_result = subprocess.run(
            [
                gh,
                "api",
                f"repos/{CERTIFICATION_GITHUB_REPOSITORY}/actions/runs/{run_id}",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        api_payload = (
            _parse_json_bytes(api_result.stdout.encode("utf-8"), label="gh run api")
            if api_result.returncode == 0
            else {}
        )
    except (
        OSError,
        ValueError,
        TypeError,
        RecursionError,
        subprocess.SubprocessError,
    ):
        return False
    now = datetime.now(timezone.utc)
    created_at = (
        _parse_utc_timestamp(payload.get("createdAt"))
        if isinstance(payload, dict)
        else None
    )
    updated_at = (
        _parse_utc_timestamp(payload.get("updatedAt"))
        if isinstance(payload, dict)
        else None
    )
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    successful_jobs = (
        {
            job.get("name")
            for job in jobs
            if isinstance(job, dict)
            and job.get("status") == "completed"
            and job.get("conclusion") == "success"
            and isinstance(job.get("name"), str)
        }
        if isinstance(jobs, list)
        else set()
    )
    if expected_workflow == CERTIFICATION_GITHUB_WORKFLOW:
        required_jobs_ok = set(CERTIFICATION_CI_REQUIRED_JOBS) <= successful_jobs
    elif expected_workflow == CERTIFICATION_DIST_WORKFLOW:
        required_jobs_ok = successful_jobs == set(CERTIFICATION_DIST_TARGETS)
    elif expected_workflow == CERTIFICATION_MODEL_PARITY_WORKFLOW:
        required_jobs_ok = successful_jobs == CERTIFICATION_MODEL_PARITY_REQUIRED_JOBS
    else:
        required_jobs_ok = successful_jobs == CERTIFICATION_PERFORMANCE_REQUIRED_JOBS
    run_ok = (
        isinstance(payload, dict)
        and str(payload.get("databaseId")) == run_id
        and payload.get("headSha") == current_head
        and payload.get("headBranch") == "main"
        and payload.get("status") == "completed"
        and payload.get("conclusion") == "success"
        and payload.get("workflowName") == expected_workflow
        and payload.get("event") == next(iter(expected_events))
        and _age_reason(created_at, now, "CI run creation") is None
        and _age_reason(updated_at, now, "CI run completion") is None
        and required_jobs_ok
        and isinstance(api_payload, dict)
        and api_payload.get("head_sha") == current_head
        and api_payload.get("head_branch") == "main"
        and str(api_payload.get("path", "")).split("@", 1)[0]
        == CERTIFICATION_WORKFLOW_PATHS[expected_workflow]
    )
    if not run_ok or not artifacts:
        return False
    grouped: dict[str, list[dict]] = {}
    for item in artifacts:
        artifact_name = item.get("source_ci_artifact_name")
        member = item.get("source_ci_artifact_path")
        digest = item.get("sha256")
        member_path = Path(member) if isinstance(member, str) else None
        if (
            not isinstance(artifact_name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", artifact_name) is None
            or member_path is None
            or member_path.is_absolute()
            or member_path.as_posix() != member
            or any(part in {"", ".", ".."} for part in member_path.parts)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            return False
        grouped.setdefault(artifact_name, []).append(item)
    try:
        with tempfile.TemporaryDirectory(prefix="focr-cert-ci-artifacts-") as tmp:
            download_root = Path(tmp)
            for index, (artifact_name, bound_files) in enumerate(
                sorted(grouped.items())
            ):
                destination = download_root / f"artifact-{index}"
                result = subprocess.run(
                    [
                        gh,
                        "run",
                        "download",
                        run_id,
                        "--repo",
                        CERTIFICATION_GITHUB_REPOSITORY,
                        "--name",
                        artifact_name,
                        "--dir",
                        str(destination),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode != 0:
                    return False
                for item in bound_files:
                    member = str(item["source_ci_artifact_path"])
                    member_path = Path(member)
                    candidate = _safe_repo_path(destination, member)
                    if (
                        candidate is None
                        or not candidate.is_file()
                        or any(
                            destination.joinpath(
                                *member_path.parts[:part_count]
                            ).is_symlink()
                            for part_count in range(1, len(member_path.parts) + 1)
                        )
                        or _sha256_file(candidate) != item["sha256"]
                    ):
                        return False
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _model_parity_evidence_reasons(
    root: Path,
    certificate: dict,
    entries: dict[str, dict],
    receipt: dict,
) -> list[str]:
    """Replay the real weighted-ladder scorecard, NDJSON, and oracle bindings."""
    reasons: list[str] = []
    expected_fields = {
        "schema_version",
        "generated_at_utc",
        "git_head",
        "source_ci_run_id",
        "model_commit",
        "weighted_model_loaded",
        "skipped_no_model",
        "rungs",
        "scorecard_path",
        "scorecard_sha256",
        "raw_log_path",
        "raw_log_sha256",
        "oracle_fixture_bindings",
        "raw_evidence_paths",
        "raw_evidence_sha256s",
    }
    if set(receipt) != expected_fields:
        return ["model parity receipt has a noncanonical field set"]
    scorecard_relative = receipt.get("scorecard_path")
    raw_relative = receipt.get("raw_log_path")
    bindings = receipt.get("oracle_fixture_bindings")
    raw_paths = receipt.get("raw_evidence_paths")
    raw_hashes = receipt.get("raw_evidence_sha256s")
    if (
        receipt.get("git_head") != certificate.get("git_head")
        or receipt.get("model_commit") != UNLIMITED_OCR_MODEL_COMMIT
        or receipt.get("weighted_model_loaded") is not True
        or receipt.get("skipped_no_model") is not False
        or not isinstance(scorecard_relative, str)
        or not isinstance(raw_relative, str)
        or not isinstance(bindings, list)
        or not isinstance(raw_paths, list)
        or not isinstance(raw_hashes, dict)
    ):
        return ["model parity receipt does not bind a weighted current-HEAD run"]

    expected_raw_hashes = {
        relative: entries[relative]["sha256"]
        for relative in raw_paths
        if isinstance(relative, str)
        and isinstance(entries.get(relative), dict)
        and isinstance(entries[relative].get("sha256"), str)
    }
    if (
        not raw_paths
        or any(not isinstance(relative, str) for relative in raw_paths)
        or len(set(raw_paths)) != len(raw_paths)
        or raw_hashes != expected_raw_hashes
        or len(expected_raw_hashes) != len(raw_paths)
        or receipt.get("scorecard_sha256")
        != expected_raw_hashes.get(scorecard_relative)
        or receipt.get("raw_log_sha256") != expected_raw_hashes.get(raw_relative)
    ):
        reasons.append("model parity raw evidence is not exhaustively manifest-bound")

    scorecard_path = _safe_repo_path(root, scorecard_relative)
    raw_path = _safe_repo_path(root, raw_relative)
    try:
        scorecard = (
            _load_bounded_json(scorecard_path)
            if scorecard_path is not None and scorecard_path.is_file()
            else None
        )
        raw_text = (
            _read_bounded_text(raw_path)
            if raw_path is not None and raw_path.is_file()
            else ""
        )
    except (OSError, ValueError, TypeError, RecursionError):
        scorecard = None
        raw_text = ""
    scorecard_gates = scorecard.get("gates") if isinstance(scorecard, dict) else None
    gate_rows = (
        {
            gate.get("gate"): gate
            for gate in scorecard_gates
            if isinstance(gate, dict) and isinstance(gate.get("gate"), str)
        }
        if isinstance(scorecard_gates, list)
        else {}
    )
    if (
        not isinstance(scorecard, dict)
        or scorecard.get("schema") != "focr-ladder-scorecard/v1"
        or scorecard.get("all_green") is not True
        or scorecard.get("skipped_no_model") is not False
        or len(scorecard_gates or []) != 6
        or set(gate_rows) != set(CERTIFICATION_MODEL_PARITY_MIN_ROWS)
        or any(
            gate_rows[rung].get("outcome") != "pass"
            or gate_rows[rung].get("meaningful") is not True
            or gate_rows[rung].get("parity_rows") != minimum
            for rung, minimum in CERTIFICATION_MODEL_PARITY_MIN_ROWS.items()
        )
    ):
        reasons.append("model parity scorecard is not an exact armed L0-L5 pass")

    events: list[dict] = []
    malformed_json_line = False
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            event = _parse_json_bytes(stripped.encode("utf-8"), label=str(raw_path))
        except (ValueError, TypeError, RecursionError):
            malformed_json_line = True
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            malformed_json_line = True
    if malformed_json_line or "skip_no_model" in raw_text:
        reasons.append("model parity raw log is malformed or contains a model-less skip")

    observed_cases: dict[tuple[str, str], dict] = {}
    passing_results: set[str] = set()
    invalid_ladder_event = False
    for event in events:
        test = str(event.get("test", ""))
        rung_match = re.match(r"^(L[0-5])(?:_|$)", test, re.IGNORECASE)
        if rung_match is None:
            continue
        rung = rung_match.group(1).upper()
        if event.get("event") == "result":
            if event.get("result") == "pass":
                passing_results.add(rung)
            elif event.get("result") in {"fail", "skip_no_model"}:
                invalid_ladder_event = True
            continue
        if event.get("event") == "skip":
            invalid_ladder_event = True
            continue
        if event.get("event") != "parity":
            continue
        envelope = event.get("nondeterminism_envelope")
        doc = envelope.get("doc") if isinstance(envelope, dict) else None
        fixture = event.get("oracle_fixture")
        seam = (
            str(fixture).split(".", 1)[0]
            if isinstance(fixture, str) and fixture
            else ""
        )
        multi_page = test.lower().startswith("l5_multi_page")
        case = (
            f"{event.get('case')}:multi_page"
            if multi_page
            else f"{doc}:decoded_text"
            if rung == "L5"
            else f"{doc}:{seam}"
        )
        identity_valid = (
            envelope.get("subject") == "f32 greedy"
            and envelope.get("oracle") == "bf16-cpu greedy (deterministic)"
            if multi_page and isinstance(envelope, dict)
            else isinstance(envelope, dict)
            and envelope.get("subject") == "franken_ocr"
            and envelope.get("oracle") == "unlimited-ocr-oracle"
        )
        if (
            event.get("schema_version") != 1
            or event.get("result") != "pass"
            or event.get("pass") is not True
            or not _finite_number(event.get("value"), minimum=-1e12, maximum=1e12)
            or not _finite_number(
                event.get("tolerance"), minimum=-1e12, maximum=1e12
            )
            or not isinstance(envelope, dict)
            or not identity_valid
            or case not in CERTIFICATION_MODEL_PARITY_CASES[rung]
            or (rung, case) in observed_cases
            or re.fullmatch(r"[0-9a-f]{64}", str(event.get("oracle_sha256")))
            is None
        ):
            invalid_ladder_event = True
            continue
        observed_cases[(rung, case)] = event
    expected_cases = {
        (rung, case)
        for rung, cases in CERTIFICATION_MODEL_PARITY_CASES.items()
        for case in cases
    }
    rungs = receipt.get("rungs")
    if (
        invalid_ladder_event
        or passing_results != set(CERTIFICATION_MODEL_PARITY_MIN_ROWS)
        or set(observed_cases) != expected_cases
        or not isinstance(rungs, dict)
        or set(rungs) != set(CERTIFICATION_MODEL_PARITY_MIN_ROWS)
        or any(value != "pass" for value in rungs.values())
    ):
        reasons.append("model parity NDJSON does not prove every exact L0-L5 case")

    observed_bindings: dict[tuple[str, str], dict] = {}
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {
            "case",
            "rung",
            "path",
            "file_sha256",
            "oracle_sha256",
        }:
            invalid_ladder_event = True
            continue
        case = binding.get("case")
        rung = binding.get("rung")
        relative = binding.get("path")
        pair = (rung, case)
        entry = entries.get(relative) if isinstance(relative, str) else None
        if (
            not isinstance(case, str)
            or not isinstance(rung, str)
            or pair in observed_bindings
            or pair not in expected_cases
            or not isinstance(entry, dict)
            or binding.get("file_sha256") != entry.get("sha256")
            or relative not in raw_paths
            or binding.get("oracle_sha256")
            != observed_cases.get(pair, {}).get("oracle_sha256")
        ):
            invalid_ladder_event = True
            continue
        observed_bindings[pair] = binding
    if invalid_ladder_event or set(observed_bindings) != expected_cases:
        reasons.append("model parity oracle outputs are not physically hash-bound")
    return reasons


def _strict_evidence_reasons(
    root: Path,
    bundle_dir: Path,
    certificate: dict,
    entries: dict[str, dict],
    *,
    provenance_root: Path | None = None,
    ci_run_verifier: Callable[[Path, str, str, Sequence[dict]], bool] | None = None,
) -> list[str]:
    bundle_dir = bundle_dir.resolve()
    reasons: list[str] = []
    evidence_classes = certificate.get("evidence_classes")
    if not isinstance(evidence_classes, dict):
        return ["certificate evidence_classes is missing or not an object"]
    documents: dict[str, dict] = {}
    for class_name, (expected_name, expected_schema) in STRICT_BUNDLE_CLASSES.items():
        relative = evidence_classes.get(class_name)
        if not isinstance(relative, str) or not relative:
            reasons.append(f"certificate omits evidence class: {class_name}")
            continue
        if Path(relative).name != expected_name:
            reasons.append(
                f"evidence class {class_name} has noncanonical filename: {relative}"
            )
        if relative not in entries:
            reasons.append(
                f"bundle manifest omits evidence class: {class_name} ({relative})"
            )
            continue
        path = _safe_repo_path(root, relative)
        if path is None or not path.is_file():
            reasons.append(
                f"evidence class {class_name} is missing or unsafe: {relative}"
            )
            continue
        try:
            path.relative_to(bundle_dir)
        except ValueError:
            reasons.append(
                f"evidence class {class_name} is outside the selected bundle: {relative}"
            )
            continue
        try:
            payload = _load_bounded_json(path)
        except (OSError, ValueError, TypeError, RecursionError) as error:
            reasons.append(f"evidence class {class_name} is unreadable JSON: {error}")
            continue
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != expected_schema
        ):
            reasons.append(
                f"evidence class {class_name} schema is not {expected_schema}"
            )
            continue
        if _parse_utc_timestamp(payload.get("generated_at_utc")) is None:
            reasons.append(f"evidence class {class_name} has no valid generated_at_utc")
            continue
        documents[class_name] = payload

    confidence = documents.get("confidence_gate", {})
    if confidence:
        if confidence.get("release_decision") != "Allow":
            reasons.append("confidence gate release_decision is not Allow")
        if confidence.get("min_verification_pct_observed") != 100.0:
            reasons.append("confidence gate verification percentage is not 100.0")
        if confidence.get("required_suite_pass_rate_pct_observed") != 100.0:
            reasons.append("confidence gate suite pass rate is not 100.0")
        confidence_high_count = confidence.get("high_severity_counterexample_count")
        if (
            not isinstance(confidence_high_count, int)
            or isinstance(confidence_high_count, bool)
            or confidence_high_count != 0
        ):
            reasons.append("confidence gate has high-severity counterexamples")
        if confidence.get("constants_enforced") != list(CERTIFICATION_CONSTANTS):
            reasons.append(
                "confidence gate does not enumerate the exact strict constants"
            )
        actuals = certificate.get("required_pass_actuals")
        expected_actuals = {
            "min_verification_pct_observed": confidence.get(
                "min_verification_pct_observed"
            ),
            "required_suite_pass_rate_pct_observed": confidence.get(
                "required_suite_pass_rate_pct_observed"
            ),
            "high_severity_counterexample_count": confidence.get(
                "high_severity_counterexample_count"
            ),
            "max_evidence_age_hours_observed": (
                actuals.get("max_evidence_age_hours_observed")
                if isinstance(actuals, dict)
                else None
            ),
        }
        if actuals != expected_actuals:
            reasons.append(
                "certificate required-pass actuals do not bind the confidence gate"
            )
        if certificate.get("high_severity_counterexamples") != confidence.get(
            "high_severity_counterexample_count"
        ):
            reasons.append(
                "certificate counterexample count does not bind the confidence gate"
            )

    ci_gate_receipt = documents.get("ci_gate_receipt", {})
    if ci_gate_receipt:
        jobs = ci_gate_receipt.get("jobs")
        suite_rate = ci_gate_receipt.get("suite_pass_rate_pct")
        if (
            ci_gate_receipt.get("git_head") != certificate.get("git_head")
            or not isinstance(jobs, dict)
            or set(jobs) != set(CERTIFICATION_CI_REQUIRED_JOBS)
            or any(status != "pass" for status in jobs.values())
            or not _finite_number(suite_rate, minimum=100.0, maximum=100.0)
            or (
                confidence
                and confidence.get("required_suite_pass_rate_pct_observed")
                != suite_rate
            )
        ):
            reasons.append("CI gate receipt does not prove the complete current suite")

    parity_receipt = documents.get("model_parity_receipt", {})
    if parity_receipt and "oracle_fixture_bindings" in parity_receipt:
        reasons.extend(
            _model_parity_evidence_reasons(root, certificate, entries, parity_receipt)
        )
    elif parity_receipt:
        rungs = parity_receipt.get("rungs")
        raw_paths = parity_receipt.get("raw_evidence_paths")
        raw_hashes = parity_receipt.get("raw_evidence_sha256s")
        expected_raw_hashes = (
            {
                relative: entries[relative]["sha256"]
                for relative in raw_paths
                if isinstance(raw_paths, list)
                and isinstance(relative, str)
                and isinstance(entries.get(relative), dict)
                and isinstance(entries[relative].get("sha256"), str)
            }
            if isinstance(raw_paths, list)
            else {}
        )
        expected_raw_basenames = {
            "scorecard_armed.json",
            *(f"L{index}.json" for index in range(6)),
            *(f"L{index}.log" for index in range(6)),
            *(f"L{index}_oracle.json" for index in range(6)),
            *(f"L{index}_subject.json" for index in range(6)),
        }
        rung_records_valid = True
        observed_rungs: set[str] = set()
        observed_fixture_paths: set[str] = set()
        claim_sources = certificate.get("claim_sources")
        core_sources = (
            claim_sources.get("core_evidence")
            if isinstance(claim_sources, dict)
            and isinstance(claim_sources.get("core_evidence"), dict)
            else {}
        )
        canonical_fixture_relative = core_sources.get(
            "tests/fixtures/ladder_scorecard/scorecard_armed.json"
        )
        if isinstance(raw_paths, list):
            for relative in raw_paths:
                if (
                    not isinstance(relative, str)
                    or re.fullmatch(r"L[0-5]\.json", Path(relative).name) is None
                ):
                    continue
                path = _safe_repo_path(root, relative)
                try:
                    record = (
                        _load_bounded_json(path)
                        if path is not None and path.is_file()
                        else None
                    )
                except (OSError, ValueError, TypeError, RecursionError):
                    record = None
                rung = record.get("rung") if isinstance(record, dict) else None
                log_relative = (
                    record.get("raw_log_path") if isinstance(record, dict) else None
                )
                fixture_relative = (
                    record.get("fixture_path") if isinstance(record, dict) else None
                )
                oracle_relative = (
                    record.get("oracle_output_path")
                    if isinstance(record, dict)
                    else None
                )
                subject_relative = (
                    record.get("subject_output_path")
                    if isinstance(record, dict)
                    else None
                )
                log_entry = (
                    entries.get(log_relative) if isinstance(log_relative, str) else None
                )
                fixture_entry = (
                    entries.get(fixture_relative)
                    if isinstance(fixture_relative, str)
                    else None
                )
                oracle_entry = (
                    entries.get(oracle_relative)
                    if isinstance(oracle_relative, str)
                    else None
                )
                subject_entry = (
                    entries.get(subject_relative)
                    if isinstance(subject_relative, str)
                    else None
                )
                oracle_path = (
                    _safe_repo_path(root, oracle_relative)
                    if isinstance(oracle_relative, str)
                    else None
                )
                subject_path = (
                    _safe_repo_path(root, subject_relative)
                    if isinstance(subject_relative, str)
                    else None
                )
                try:
                    oracle_output = (
                        _load_bounded_json(oracle_path)
                        if oracle_path is not None and oracle_path.is_file()
                        else None
                    )
                    subject_output = (
                        _load_bounded_json(subject_path)
                        if subject_path is not None and subject_path.is_file()
                        else None
                    )
                except (OSError, ValueError, TypeError, RecursionError):
                    oracle_output = None
                    subject_output = None
                fixture_sha = (
                    fixture_entry.get("sha256")
                    if isinstance(fixture_entry, dict)
                    else None
                )
                fixture_path = (
                    _safe_repo_path(root, fixture_relative)
                    if isinstance(fixture_relative, str)
                    else None
                )
                try:
                    fixture = (
                        _load_bounded_json(fixture_path)
                        if fixture_path is not None and fixture_path.is_file()
                        else None
                    )
                except (OSError, ValueError, TypeError, RecursionError):
                    fixture = None
                fixture_gates = (
                    fixture.get("gates") if isinstance(fixture, dict) else None
                )
                fixture_row_counts = (
                    {
                        gate.get("gate"): gate.get("parity_rows")
                        for gate in fixture_gates
                        if isinstance(gate, dict)
                        and gate.get("outcome") == "pass"
                        and gate.get("meaningful") is True
                        and isinstance(gate.get("parity_rows"), int)
                        and not isinstance(gate.get("parity_rows"), bool)
                    }
                    if isinstance(fixture_gates, list)
                    else {}
                )
                fixture_contract_valid = (
                    isinstance(fixture, dict)
                    and fixture.get("schema") == "focr-ladder-scorecard/v1"
                    and fixture.get("all_green") is True
                    and fixture.get("skipped_no_model") is False
                    and isinstance(fixture_gates, list)
                    and len(fixture_gates) == 6
                    and set(fixture_row_counts)
                    == set(CERTIFICATION_MODEL_PARITY_MIN_ROWS)
                    and all(
                        fixture_row_counts[rung_name] >= minimum_rows
                        for rung_name, minimum_rows in CERTIFICATION_MODEL_PARITY_MIN_ROWS.items()
                    )
                )
                output_contract_valid = all(
                    isinstance(output, dict)
                    and output.get("schema_version")
                    == "gauntlet.model_parity_output.v1"
                    and output.get("git_head") == certificate.get("git_head")
                    and output.get("model_commit") == UNLIMITED_OCR_MODEL_COMMIT
                    and output.get("fixture_sha256") == fixture_sha
                    and output.get("rung") == rung
                    and output.get("kind") == kind
                    for output, kind in (
                        (oracle_output, "oracle"),
                        (subject_output, "subject"),
                    )
                )
                derived_metrics, derived_pass = (
                    _derived_parity_metrics(
                        str(rung),
                        oracle_output.get("payload"),
                        subject_output.get("payload"),
                    )
                    if output_contract_valid
                    and isinstance(oracle_output, dict)
                    and isinstance(subject_output, dict)
                    else (None, False)
                )
                if (
                    not isinstance(record, dict)
                    or record.get("schema_version") != "gauntlet.model_parity_rung.v1"
                    or record.get("git_head") != certificate.get("git_head")
                    or record.get("model_commit") != UNLIMITED_OCR_MODEL_COMMIT
                    or rung not in {f"L{index}" for index in range(6)}
                    or Path(relative).name != f"{rung}.json"
                    or rung in observed_rungs
                    or record.get("weighted_model_loaded") is not True
                    or record.get("oracle_backend")
                    != "torch-2.10.0-transformers-4.57.1"
                    or not isinstance(fixture_entry, dict)
                    or fixture_relative != canonical_fixture_relative
                    or Path(str(fixture_relative)).name != "scorecard_armed.json"
                    or not fixture_contract_valid
                    or record.get("fixture_sha256") != fixture_sha
                    or not output_contract_valid
                    or not isinstance(oracle_entry, dict)
                    or Path(str(oracle_relative)).name != f"{rung}_oracle.json"
                    or record.get("oracle_output_sha256") != oracle_entry.get("sha256")
                    or not isinstance(subject_entry, dict)
                    or Path(str(subject_relative)).name != f"{rung}_subject.json"
                    or record.get("subject_output_sha256")
                    != subject_entry.get("sha256")
                    or derived_metrics is None
                    or not _json_exact(record.get("metrics"), derived_metrics)
                    or record.get("result") != ("pass" if derived_pass else "fail")
                    or not derived_pass
                    or not isinstance(log_entry, dict)
                    or Path(str(log_relative)).name != f"{rung}.log"
                    or record.get("raw_log_sha256") != log_entry.get("sha256")
                ):
                    rung_records_valid = False
                elif isinstance(rung, str):
                    observed_rungs.add(rung)
                    observed_fixture_paths.add(str(fixture_relative))
        if (
            parity_receipt.get("git_head") != certificate.get("git_head")
            or parity_receipt.get("model_commit") != UNLIMITED_OCR_MODEL_COMMIT
            or parity_receipt.get("skipped_no_model") is not False
            or not isinstance(rungs, dict)
            or set(rungs) != {f"L{index}" for index in range(6)}
            or any(status != "pass" for status in rungs.values())
            or not isinstance(raw_paths, list)
            or not raw_paths
            or any(not isinstance(relative, str) for relative in raw_paths)
            or len(set(relative for relative in raw_paths if isinstance(relative, str)))
            != len(raw_paths)
            or {
                Path(relative).name
                for relative in raw_paths
                if isinstance(relative, str)
            }
            != expected_raw_basenames
            or raw_hashes != expected_raw_hashes
            or len(expected_raw_hashes) != len(raw_paths)
            or not rung_records_valid
            or observed_rungs != {f"L{index}" for index in range(6)}
            or len(observed_fixture_paths) != 1
        ):
            reasons.append(
                "model parity receipt does not prove fresh current-head L0-L5 execution"
            )

    dist_receipt = documents.get("dist_matrix_receipt", {})
    if dist_receipt:
        targets = dist_receipt.get("targets")
        certificate_sources = certificate.get("claim_sources")
        core_sources = (
            certificate_sources.get("core_evidence")
            if isinstance(certificate_sources, dict)
            and isinstance(certificate_sources.get("core_evidence"), dict)
            else {}
        )
        readiness_sources = (
            certificate_sources.get("readiness_evidence")
            if isinstance(certificate_sources, dict)
            and isinstance(certificate_sources.get("readiness_evidence"), dict)
            else {}
        )
        installer_relative = core_sources.get("install.ps1") or readiness_sources.get(
            "install.ps1"
        )
        installer_entry = (
            entries.get(installer_relative)
            if isinstance(installer_relative, str)
            else None
        )
        expected_installer_sha256 = (
            installer_entry.get("sha256")
            if isinstance(installer_entry, dict)
            else None
        )
        portability_failures = [
            reason
            for target, result in (targets.items() if isinstance(targets, dict) else [])
            if isinstance(result, dict)
            for reason in _dist_portability_reasons(
                root,
                target,
                result.get("portability"),
                entries,
                expected_installer_sha256,
                certificate.get("version"),
            )
        ]
        if (
            dist_receipt.get("git_head") != certificate.get("git_head")
            or not isinstance(targets, dict)
            or set(targets) != set(CERTIFICATION_DIST_TARGETS)
            or any(
                not isinstance(result, dict)
                or set(result)
                != {
                    "status",
                    "built",
                    "checksum_sidecar",
                    "smoke_test",
                    "portability",
                }
                or result.get("status") != "pass"
                or result.get("built") is not True
                or result.get("checksum_sidecar") is not True
                or result.get("smoke_test") != "pass"
                for target, result in targets.items()
            )
            or portability_failures
        ):
            reasons.append(
                "dist matrix receipt does not prove every release target"
                + (f": {'; '.join(portability_failures)}" if portability_failures else "")
            )

    verification = documents.get("verification_contract", {})
    if verification:
        rows = verification.get("rows")
        universe_relative = verification.get("feature_universe_path")
        universe_entry = (
            entries.get(universe_relative)
            if isinstance(universe_relative, str)
            else None
        )
        universe_path = (
            _safe_repo_path(root, universe_relative)
            if isinstance(universe_relative, str)
            else None
        )
        try:
            universe_inside_bundle = (
                universe_path is not None
                and universe_path.is_file()
                and universe_path.relative_to(bundle_dir) is not None
            )
        except ValueError:
            universe_inside_bundle = False
        try:
            universe = (
                _load_bounded_json(universe_path)
                if universe_inside_bundle
                else None
            )
        except (OSError, ValueError, TypeError, RecursionError):
            universe = None
        if (
            not isinstance(universe_entry, dict)
            or not isinstance(universe, dict)
            or universe.get("schema_version") != "gauntlet.feature_universe.v1"
            or _parse_utc_timestamp(universe.get("generated_at_utc")) is None
            or universe.get("definition_sha256")
            != CERTIFICATION_FEATURE_UNIVERSE_SHA256
            or universe.get("features") != list(CERTIFICATION_FEATURE_UNIVERSE)
            or verification.get("feature_universe_sha256")
            != universe_entry.get("sha256")
            or verification.get("feature_universe_definition_sha256")
            != CERTIFICATION_FEATURE_UNIVERSE_SHA256
            or certificate.get("feature_universe_sha256")
            != universe_entry.get("sha256")
            or certificate.get("feature_universe_definition_sha256")
            != CERTIFICATION_FEATURE_UNIVERSE_SHA256
        ):
            reasons.append(
                "verification contract lacks the canonical hash-bound in-bundle feature universe"
            )
            expected_pairs: set[tuple[str, str]] = set()
        else:
            features = universe.get("features")
            expected_pairs = set()
            if not isinstance(features, list) or not features:
                reasons.append("feature universe has no required features")
            else:
                for feature in features:
                    if (
                        not isinstance(feature, dict)
                        or not isinstance(feature.get("feature_id"), str)
                        or feature.get("required") is not True
                        or not isinstance(feature.get("proof_obligations"), list)
                        or not feature.get("proof_obligations")
                    ):
                        reasons.append("feature universe contains a malformed feature")
                        continue
                    expected_pairs.update(
                        (feature["feature_id"], obligation)
                        for obligation in feature["proof_obligations"]
                        if isinstance(obligation, str) and obligation
                    )
        if not isinstance(rows, list) or not rows:
            reasons.append("verification contract has no proof-obligation rows")
        else:
            observed_pairs: set[tuple[str, str]] = set()
            claim_sources = certificate.get("claim_sources")
            readiness_evidence = (
                claim_sources.get("readiness_evidence")
                if isinstance(claim_sources, dict)
                and isinstance(claim_sources.get("readiness_evidence"), dict)
                else {}
            )
            for row in rows:
                if not isinstance(row, dict):
                    reasons.append("verification contract contains a malformed row")
                    continue
                pair = (row.get("feature_id"), row.get("proof_obligation"))
                if (
                    not all(isinstance(value, str) and value for value in pair)
                    or pair in observed_pairs
                    or row.get("status") != "pass"
                    or row.get("gate") != "allowed"
                ):
                    reasons.append(
                        "verification contract contains a duplicate or non-pass row"
                    )
                    continue
                observed_pairs.add(pair)
                obligation = str(row["proof_obligation"])
                expected_evidence_paths = [
                    readiness_evidence.get(original)
                    for original in CERTIFICATION_READINESS_EVIDENCE_PATHS.get(
                        obligation, ()
                    )
                ]
                evidence_paths = row.get("evidence_paths")
                evidence_sha256s = row.get("evidence_sha256s")
                expected_hashes = {
                    relative: entries[relative]["sha256"]
                    for relative in expected_evidence_paths
                    if isinstance(relative, str)
                    and isinstance(entries.get(relative), dict)
                    and isinstance(entries[relative].get("sha256"), str)
                }
                if (
                    not expected_evidence_paths
                    or any(
                        not isinstance(relative, str) or not relative
                        for relative in expected_evidence_paths
                    )
                    or evidence_paths != expected_evidence_paths
                    or evidence_sha256s != expected_hashes
                    or len(expected_hashes) != len(expected_evidence_paths)
                ):
                    reasons.append(
                        f"verification row {pair!r} lacks hash-bound manifest evidence"
                    )
            if observed_pairs != expected_pairs:
                reasons.append(
                    "verification contract does not exhaust the feature universe"
                )

    ci_manifest = documents.get("ci_manifest", {})
    if ci_manifest:
        if ci_manifest.get(
            "repository"
        ) != CERTIFICATION_GITHUB_REPOSITORY or ci_manifest.get(
            "required_workflows"
        ) != {
            CERTIFICATION_GITHUB_WORKFLOW: CERTIFICATION_GITHUB_EVENT,
            CERTIFICATION_DIST_WORKFLOW: CERTIFICATION_GITHUB_EVENT,
            CERTIFICATION_MODEL_PARITY_WORKFLOW: CERTIFICATION_MODEL_PARITY_EVENT,
            CERTIFICATION_PERFORMANCE_WORKFLOW: CERTIFICATION_PERFORMANCE_EVENT,
        }:
            reasons.append("CI manifest does not pin the canonical repository workflow")
        artifacts = ci_manifest.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or len(artifacts) > CERTIFICATION_MAX_BUNDLE_ARTIFACTS
        ):
            reasons.append("CI manifest has no reconstructable artifacts")
        else:
            ci_receipt_relative = evidence_classes.get("ci_gate_receipt")
            dist_receipt_relative = evidence_classes.get("dist_matrix_receipt")
            parity_receipt_relative = evidence_classes.get("model_parity_receipt")
            ci_raw_paths = ci_gate_receipt.get("raw_evidence_paths")
            dist_raw_paths = dist_receipt.get("raw_evidence_paths")
            parity_raw_paths = parity_receipt.get("raw_evidence_paths")
            benchmark_document = documents.get("benchmark_summary", {})
            benchmark_paths = {
                gate.get("current_path")
                for gate in (
                    benchmark_document.get("pass_over_pass_gates", {}).values()
                    if isinstance(benchmark_document.get("pass_over_pass_gates"), dict)
                    else []
                )
                if isinstance(gate, dict) and isinstance(gate.get("current_path"), str)
            }
            expected_ci_paths = {
                relative
                for candidates in (
                    ci_raw_paths
                    if isinstance(ci_raw_paths, list)
                    else [ci_receipt_relative],
                    dist_raw_paths
                    if isinstance(dist_raw_paths, list)
                    else [dist_receipt_relative],
                    parity_raw_paths
                    if isinstance(parity_raw_paths, list)
                    else [parity_receipt_relative],
                    benchmark_paths,
                )
                for relative in candidates
                if isinstance(relative, str)
            }
            expected_ci_paths.discard(None)
            observed_ci_paths: set[str] = set()
            run_artifacts: dict[str, list[dict]] = {}
            workflow_by_path: dict[str, str] = {}
            run_by_path: dict[str, str] = {}
            for item in artifacts:
                if not isinstance(item, dict):
                    reasons.append("CI manifest contains a malformed artifact")
                    continue
                relative = item.get("artifact")
                run_id = str(item.get("source_ci_run_id", ""))
                entry = entries.get(relative) if isinstance(relative, str) else None
                if (
                    not isinstance(relative, str)
                    or relative in observed_ci_paths
                    or not isinstance(entry, dict)
                    or item.get("sha256") != entry.get("sha256")
                    or re.fullmatch(r"[1-9][0-9]*", run_id) is None
                    or item.get("schema_version") != CI_ARTIFACT_BINDING_SCHEMA
                    or re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
                        str(item.get("source_ci_artifact_name", "")),
                    )
                    is None
                    or not isinstance(item.get("source_ci_artifact_path"), str)
                    or not item.get("source_ci_artifact_path")
                    or item.get("source_ci_repository")
                    != CERTIFICATION_GITHUB_REPOSITORY
                    or item.get("source_ci_workflow")
                    not in {
                        CERTIFICATION_GITHUB_WORKFLOW,
                        CERTIFICATION_DIST_WORKFLOW,
                        CERTIFICATION_MODEL_PARITY_WORKFLOW,
                        CERTIFICATION_PERFORMANCE_WORKFLOW,
                    }
                    or item.get("source_ci_event")
                    != (
                        {
                            CERTIFICATION_MODEL_PARITY_WORKFLOW: CERTIFICATION_MODEL_PARITY_EVENT,
                            CERTIFICATION_PERFORMANCE_WORKFLOW: CERTIFICATION_PERFORMANCE_EVENT,
                        }.get(
                            item.get("source_ci_workflow"),
                            CERTIFICATION_GITHUB_EVENT,
                        )
                    )
                    or item.get("source_ci_workflow_path")
                    != CERTIFICATION_WORKFLOW_PATHS.get(item.get("source_ci_workflow"))
                ):
                    reasons.append("CI manifest contains an untraceable artifact")
                    continue
                observed_ci_paths.add(relative)
                run_artifacts.setdefault(run_id, []).append(item)
                workflow_by_path[relative] = str(item["source_ci_workflow"])
                run_by_path[relative] = run_id
            if not expected_ci_paths or not expected_ci_paths <= observed_ci_paths:
                reasons.append(
                    "CI manifest omits required workflow-produced evidence"
                )
            observed_workflows = {
                item.get("source_ci_workflow")
                for items in run_artifacts.values()
                for item in items
            }
            if observed_workflows != {
                CERTIFICATION_GITHUB_WORKFLOW,
                CERTIFICATION_DIST_WORKFLOW,
                CERTIFICATION_MODEL_PARITY_WORKFLOW,
                CERTIFICATION_PERFORMANCE_WORKFLOW,
            }:
                reasons.append(
                    "CI manifest does not prove CI, dist, model-parity, and performance workflows"
                )
            runs_by_workflow: dict[str, set[str]] = {}
            for relative, workflow in workflow_by_path.items():
                runs_by_workflow.setdefault(workflow, set()).add(run_by_path[relative])
            canonical_run_shape = (
                len(run_artifacts) == len(CERTIFICATION_WORKFLOW_PATHS)
                and set(runs_by_workflow) == set(CERTIFICATION_WORKFLOW_PATHS)
                and all(len(run_ids) == 1 for run_ids in runs_by_workflow.values())
            )
            if not canonical_run_shape:
                reasons.append(
                    "CI manifest must bind exactly one run per canonical workflow"
                )
            ci_anchor = (
                next(
                    (relative for relative in ci_raw_paths if isinstance(relative, str)),
                    ci_receipt_relative,
                )
                if isinstance(ci_raw_paths, list)
                else ci_receipt_relative
            )
            dist_anchor = (
                next(
                    (
                        relative
                        for relative in dist_raw_paths
                        if isinstance(relative, str)
                    ),
                    dist_receipt_relative,
                )
                if isinstance(dist_raw_paths, list)
                else dist_receipt_relative
            )
            if (
                workflow_by_path.get(ci_anchor)
                != CERTIFICATION_GITHUB_WORKFLOW
                or workflow_by_path.get(dist_anchor)
                != CERTIFICATION_DIST_WORKFLOW
                or run_by_path.get(ci_anchor) == run_by_path.get(dist_anchor)
            ):
                reasons.append(
                    "CI and dist receipts do not come from distinct canonical workflow runs"
                )
            if str(ci_gate_receipt.get("source_ci_run_id", "")) != run_by_path.get(
                ci_anchor
            ) or str(dist_receipt.get("source_ci_run_id", "")) != run_by_path.get(
                dist_anchor
            ):
                reasons.append("workflow receipts do not bind their CI run ids")
            parity_anchor = (
                next(
                    (
                        relative
                        for relative in parity_raw_paths
                        if isinstance(relative, str)
                    ),
                    parity_receipt_relative,
                )
                if isinstance(parity_raw_paths, list)
                else parity_receipt_relative
            )
            parity_run_id = run_by_path.get(parity_anchor)
            if (
                workflow_by_path.get(parity_anchor)
                != CERTIFICATION_MODEL_PARITY_WORKFLOW
                or str(parity_receipt.get("source_ci_run_id", "")) != parity_run_id
                or not isinstance(parity_raw_paths, list)
                or any(not isinstance(relative, str) for relative in parity_raw_paths)
                or any(
                    workflow_by_path.get(relative)
                    != CERTIFICATION_MODEL_PARITY_WORKFLOW
                    or run_by_path.get(relative) != parity_run_id
                    for relative in parity_raw_paths
                )
            ):
                reasons.append(
                    "model parity receipt/raw evidence is not bound to one canonical model run"
                )
            performance_anchor = next(iter(benchmark_paths), None)
            performance_run_id = run_by_path.get(performance_anchor)
            if (
                workflow_by_path.get(performance_anchor)
                != CERTIFICATION_PERFORMANCE_WORKFLOW
                or len(benchmark_paths) != 1
                or any(
                    workflow_by_path.get(relative) != CERTIFICATION_PERFORMANCE_WORKFLOW
                    or run_by_path.get(relative) != performance_run_id
                    for relative in benchmark_paths
                )
            ):
                reasons.append(
                    "benchmark summary/current samples are not bound to one canonical performance run"
                )
            verify_ci_run = ci_run_verifier or _default_ci_run_verifier
            if canonical_run_shape:
                for run_id, bound_artifacts in sorted(run_artifacts.items()):
                    if not verify_ci_run(
                        provenance_root or root,
                        run_id,
                        str(certificate.get("git_head", "")),
                        bound_artifacts,
                    ):
                        reasons.append(
                            f"CI run artifact provenance could not be verified: {run_id}"
                        )

    benchmark = documents.get("benchmark_summary", {})
    if benchmark:
        gates = benchmark.get("pass_over_pass_gates")
        claim_sources = certificate.get("claim_sources")
        core_sources = (
            claim_sources.get("core_evidence")
            if isinstance(claim_sources, dict)
            and isinstance(claim_sources.get("core_evidence"), dict)
            else {}
        )
        canonical_baseline_relative = core_sources.get(
            "benches/.bench-history/baseline.json"
        )
        if not isinstance(gates, dict) or set(gates) != set(
            CERTIFICATION_BENCHMARK_THRESHOLDS
        ):
            reasons.append(
                "benchmark summary does not carry all five pass-over-pass gates"
            )
        else:
            for gate_name, minimum in CERTIFICATION_BENCHMARK_THRESHOLDS.items():
                gate = gates[gate_name]
                observed = (
                    gate.get("regression_pct") if isinstance(gate, dict) else None
                )
                baseline_relative = (
                    gate.get("baseline_path") if isinstance(gate, dict) else None
                )
                current_relative = (
                    gate.get("current_path") if isinstance(gate, dict) else None
                )
                raw_documents: list[dict | None] = []
                for relative in (baseline_relative, current_relative):
                    entry = entries.get(relative) if isinstance(relative, str) else None
                    path = (
                        _safe_repo_path(root, relative)
                        if isinstance(relative, str)
                        else None
                    )
                    try:
                        raw = (
                            _load_bounded_json(path)
                            if path is not None and path.is_file()
                            else None
                        )
                    except (OSError, ValueError, TypeError, RecursionError):
                        raw = None
                    if not isinstance(entry, dict) or not isinstance(raw, dict):
                        raw_documents.append(None)
                    else:
                        raw_documents.append(raw)
                baseline_entry = (
                    entries.get(baseline_relative)
                    if isinstance(baseline_relative, str)
                    else None
                )
                current_entry = (
                    entries.get(current_relative)
                    if isinstance(current_relative, str)
                    else None
                )
                baseline_document, current_document = raw_documents
                baseline_metrics = _benchmark_metrics(baseline_document)
                current_metrics = _benchmark_metrics(current_document)
                documents_valid = (
                    baseline_relative == canonical_baseline_relative
                    and isinstance(baseline_entry, dict)
                    and isinstance(current_entry, dict)
                    and isinstance(baseline_document, dict)
                    and baseline_document.get("schema") == "focr-bench-baseline/v1"
                    and isinstance(current_document, dict)
                    and current_document.get("schema_version")
                    == "gauntlet.current_benchmark.v1"
                    and current_document.get("git_head") == certificate.get("git_head")
                    and _parse_utc_timestamp(current_document.get("generated_at_utc"))
                    is not None
                    and current_document.get("baseline_sha256")
                    == baseline_entry.get("sha256")
                    and gate.get("baseline_sha256") == baseline_entry.get("sha256")
                    and gate.get("current_sha256") == current_entry.get("sha256")
                    and baseline_metrics is not None
                    and current_metrics is not None
                )
                derived = None
                if documents_valid:
                    try:
                        baseline_mean = baseline_metrics[gate_name]
                        current_mean = current_metrics[gate_name]
                        candidate = (
                            (baseline_mean - current_mean) / baseline_mean * 100.0
                        )
                        if _finite_number(candidate, minimum=-1e6, maximum=1e6):
                            derived = truncate_score(candidate)
                    except (
                        KeyError,
                        OverflowError,
                        statistics.StatisticsError,
                        ZeroDivisionError,
                    ):
                        derived = None
                if (
                    not isinstance(gate, dict)
                    or set(gate)
                    != {
                        "passed",
                        "minimum_pct",
                        "regression_pct",
                        "baseline_path",
                        "baseline_sha256",
                        "current_path",
                        "current_sha256",
                    }
                    or gate.get("minimum_pct") != minimum
                    or derived is None
                    or observed != derived
                    or gate.get("passed") is not (derived >= minimum)
                    or derived < minimum
                ):
                    reasons.append(
                        f"benchmark pass-over-pass gate failed or is unbound: {gate_name}"
                    )

    scorecards = documents.get("scorecards", {})
    ratchet = documents.get("ratchet_state", {})
    if scorecards and ratchet:
        lower_bound = scorecards.get("parity_score_lower_bound")
        current_bound = ratchet.get("current_lower_bound")
        if (
            not _finite_number(lower_bound, minimum=0.0, maximum=1.0)
            or not _finite_number(current_bound, minimum=0.0, maximum=1.0)
            or truncate_score(float(lower_bound)) < float(current_bound)
        ):
            reasons.append("scorecard lower bound does not satisfy the ratchet")
        if certificate.get("parity_score") != lower_bound:
            reasons.append(
                "certificate parity_score does not equal the scorecard lower bound"
            )
        previous = ratchet.get("previous_bound")
        if (
            not _finite_number(previous, minimum=0.0, maximum=1.0)
            or not _finite_number(current_bound, minimum=0.0, maximum=1.0)
            or float(previous) > float(current_bound)
        ):
            reasons.append("ratchet state is not monotone")
        if ratchet.get("commit_sha") != certificate.get("git_head"):
            reasons.append("ratchet state commit_sha does not match certificate HEAD")
        per_category = ratchet.get("per_category_bounds")
        scorecard_categories = scorecards.get("per_category_lower")
        if (
            not isinstance(per_category, dict)
            or not per_category
            or not isinstance(scorecard_categories, dict)
            or set(per_category) != set(scorecard_categories)
            or any(
                not _finite_number(bound, minimum=0.0, maximum=1.0)
                or not _finite_number(
                    scorecard_categories.get(category), minimum=0.0, maximum=1.0
                )
                or float(scorecard_categories[category]) < float(bound)
                for category, bound in per_category.items()
            )
            or _parse_utc_timestamp(ratchet.get("timestamp")) is None
            or not isinstance(ratchet.get("advance_reason"), str)
            or not ratchet.get("advance_reason")
        ):
            reasons.append("ratchet state lacks valid per-category history invariants")

    critical_inventory = documents.get("critical_path_inventory", {})
    derived_critical_findings: list[dict] = []
    inventory_valid = bool(critical_inventory)
    if critical_inventory:
        audits = critical_inventory.get("audits")
        if not isinstance(audits, dict) or set(audits) != set(
            CERTIFICATION_AUDIT_DOMAINS
        ):
            inventory_valid = False
        else:
            for domain, audit_binding in audits.items():
                if not isinstance(audit_binding, dict):
                    inventory_valid = False
                    continue
                relative = audit_binding.get("evidence_path")
                entry = entries.get(relative) if isinstance(relative, str) else None
                path = (
                    _safe_repo_path(root, relative)
                    if isinstance(relative, str)
                    else None
                )
                try:
                    receipt = (
                        _load_bounded_json(path)
                        if path is not None and path.is_file()
                        else None
                    )
                except (OSError, ValueError, TypeError, RecursionError):
                    receipt = None
                findings = (
                    receipt.get("findings") if isinstance(receipt, dict) else None
                )
                tools = receipt.get("tools") if isinstance(receipt, dict) else None
                tool_contract_valid = isinstance(tools, list) and len(tools) == len(
                    CERTIFICATION_AUDIT_TOOLS[domain]
                )
                observed_tool_ids: set[str] = set()
                observed_tool_outputs: set[str] = set()
                if isinstance(tools, list):
                    for tool in tools:
                        tool_id = tool.get("id") if isinstance(tool, dict) else None
                        output_relative = (
                            tool.get("output_path") if isinstance(tool, dict) else None
                        )
                        output_entry = (
                            entries.get(output_relative)
                            if isinstance(output_relative, str)
                            else None
                        )
                        output_path = (
                            _safe_repo_path(root, output_relative)
                            if isinstance(output_relative, str)
                            else None
                        )
                        try:
                            output = (
                                _load_bounded_json(output_path)
                                if output_path is not None and output_path.is_file()
                                else None
                            )
                        except (OSError, ValueError, TypeError, RecursionError):
                            output = None
                        output_contract_valid = (
                            isinstance(output, dict)
                            and output.get("schema_version")
                            == "gauntlet.audit_tool_output.v1"
                            and _parse_utc_timestamp(output.get("generated_at_utc"))
                            is not None
                            and output.get("git_head") == certificate.get("git_head")
                            and output.get("domain") == domain
                            and output.get("tool_id") == tool_id
                            and output.get("command")
                            == CERTIFICATION_AUDIT_TOOLS[domain].get(tool_id)
                            and output.get("exit_code") == 0
                            and output.get("result") == "pass"
                        )
                        if (
                            not isinstance(tool, dict)
                            or set(tool)
                            != {
                                "id",
                                "version",
                                "command",
                                "output_path",
                                "output_sha256",
                                "result",
                            }
                            or not isinstance(tool_id, str)
                            or tool_id in observed_tool_ids
                            or tool_id not in CERTIFICATION_AUDIT_TOOLS[domain]
                            or tool.get("command")
                            != CERTIFICATION_AUDIT_TOOLS[domain].get(tool_id)
                            or not isinstance(tool.get("version"), str)
                            or not tool.get("version")
                            or not isinstance(output_relative, str)
                            or output_relative in observed_tool_outputs
                            or not isinstance(output_entry, dict)
                            or tool.get("output_sha256") != output_entry.get("sha256")
                            or tool.get("result") != "pass"
                            or not output_contract_valid
                        ):
                            tool_contract_valid = False
                            continue
                        observed_tool_ids.add(tool_id)
                        observed_tool_outputs.add(output_relative)
                tool_contract_valid = tool_contract_valid and observed_tool_ids == set(
                    CERTIFICATION_AUDIT_TOOLS[domain]
                )
                if (
                    not isinstance(relative, str)
                    or Path(relative).name != f"{domain}_audit_receipt.json"
                    or not isinstance(entry, dict)
                    or audit_binding.get("evidence_sha256") != entry.get("sha256")
                    or not isinstance(receipt, dict)
                    or receipt.get("schema_version") != "gauntlet.audit_receipt.v1"
                    or _parse_utc_timestamp(receipt.get("generated_at_utc")) is None
                    or receipt.get("git_head") != certificate.get("git_head")
                    or receipt.get("domain") != domain
                    or receipt.get("scope_complete") is not True
                    or not tool_contract_valid
                    or not isinstance(findings, list)
                ):
                    inventory_valid = False
                    continue
                for finding in findings:
                    if (
                        not isinstance(finding, dict)
                        or set(finding)
                        != {
                            "id",
                            "severity",
                            "status",
                            "evidence_path",
                            "evidence_sha256",
                        }
                        or not isinstance(finding.get("id"), str)
                        or finding.get("severity") not in {"High", "Critical"}
                        or finding.get("status") not in {"open", "resolved", "waived"}
                        or not isinstance(finding.get("evidence_path"), str)
                        or not isinstance(
                            entries.get(finding.get("evidence_path")), dict
                        )
                        or finding.get("evidence_sha256")
                        != entries[finding["evidence_path"]].get("sha256")
                    ):
                        inventory_valid = False
                        continue
                    derived_critical_findings.append(finding)
    if not inventory_valid:
        reasons.append("critical-path inventory is missing or not exhaustively audited")

    critical = documents.get("critical_path_report", {})
    if critical:
        findings = critical.get("findings")
        open_count = critical.get("open_high_critical")
        waived_count = critical.get("waived_high_critical")
        derived_open = sum(
            finding.get("status") == "open" for finding in derived_critical_findings
        )
        derived_waived = sum(
            finding.get("status") == "waived" for finding in derived_critical_findings
        )
        derived_resolved = [
            finding
            for finding in derived_critical_findings
            if finding.get("status") == "resolved"
        ]
        if (
            not isinstance(open_count, int)
            or isinstance(open_count, bool)
            or open_count != 0
            or not isinstance(waived_count, int)
            or isinstance(waived_count, bool)
            or waived_count != 0
            or not isinstance(findings, list)
            or open_count != derived_open
            or waived_count != derived_waived
            or findings != derived_resolved
        ):
            reasons.append(
                "critical-path report has open, waived, or unproven High/Critical findings"
            )
    return reasons


def _hypothesis_dependency_files(
    bundle_dir: Path,
    contents: dict[str, str | None],
    ledger_paths: Sequence[str],
) -> tuple[set[Path], list[str]]:
    """Return every file whose bytes affect hypothesis closure."""
    dependencies: set[Path] = set()
    reasons: list[str] = []

    def include_tree(path: Path, declared: str) -> None:
        scope = path if path.is_dir() else path.parent
        if not scope.is_dir():
            reasons.append(f"hypothesis dependency is missing: {declared}")
            return
        for candidate in scope.rglob("*"):
            if candidate.is_file() and not candidate.name.startswith("._"):
                dependencies.add(candidate.resolve())

    for ledger in ledger_paths:
        text = contents.get(ledger)
        if not isinstance(text, str):
            continue
        matches = list(_HYPOTHESIS_ENTRY_RE.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section = text[match.start() : end]
            results_fence = _experiment_results_fence(section) or ""
            for relative in _experiment_list_field(
                results_fence, "result_evidence_paths"
            ):
                path = _safe_repo_path(bundle_dir, relative)
                if path is None or not path.exists():
                    reasons.append(
                        f"hypothesis dependency is missing or unsafe: {relative}"
                    )
                    continue
                include_tree(path, relative)
            result_status = _experiment_field_values(results_fence, "result_status")
            if len(result_status) != 1 or result_status[0].upper() != "CONFIRMED_GAP":
                continue
            remediation = _experiment_field_values(
                results_fence, "spawned_remediation_bead"
            )
            if len(remediation) != 1:
                continue
            issues = _safe_repo_path(bundle_dir, ".beads/issues.jsonl")
            if issues is None or not issues.is_file():
                reasons.append(
                    "confirmed-gap hypothesis dependency is missing: "
                    ".beads/issues.jsonl"
                )
            else:
                dependencies.add(issues.resolve())
            proof_relative = f"artifacts/{remediation[0]}/proof_pack"
            proof_pack = _safe_repo_path(bundle_dir, proof_relative)
            if proof_pack is None or not proof_pack.is_dir():
                reasons.append(
                    f"confirmed-gap hypothesis dependency is missing: {proof_relative}"
                )
            else:
                include_tree(proof_pack, proof_relative)
    return dependencies, reasons


def _bundled_perf_evidence_ok(
    ledger_text: str, bundle_dir: Path, git_head: str, lineage_root: Path
) -> bool:
    return (
        perf_evidence_verdict(
            ledger_text,
            bundle_dir,
            git_head,
            lineage_root=lineage_root,
        )["ok"]
        is True
    )


def _certificate_source_claim_reasons(
    root: Path,
    bundle_dir: Path,
    certificate: dict,
    entries: dict[str, dict],
    *,
    source_root: Path | None = None,
    source_digest_resolver: Callable[[Path, str, str], str | None] | None = None,
    perf_evidence_checker: Callable[[str, Path, str], bool] | None = None,
) -> list[str]:
    bundle_dir = bundle_dir.resolve()
    reasons: list[str] = []
    claim_sources = certificate.get("claim_sources")
    readiness_relative = (
        claim_sources.get("release_readiness")
        if isinstance(claim_sources, dict)
        else None
    )
    rounds_relative = (
        claim_sources.get("rounds") if isinstance(claim_sources, dict) else None
    )
    ledger_paths = (
        claim_sources.get("hypothesis_ledgers")
        if isinstance(claim_sources, dict)
        else None
    )
    core_evidence = (
        claim_sources.get("core_evidence") if isinstance(claim_sources, dict) else None
    )
    readiness_evidence = (
        claim_sources.get("readiness_evidence")
        if isinstance(claim_sources, dict)
        else None
    )
    if (
        not isinstance(readiness_relative, str)
        or not isinstance(rounds_relative, str)
        or not isinstance(ledger_paths, list)
        or len(ledger_paths) != len(HYPOTHESIS_LEDGER_PATHS)
        or any(not isinstance(path, str) or not path for path in ledger_paths)
        or {Path(path).name for path in ledger_paths}
        != {Path(path).name for path in HYPOTHESIS_LEDGER_PATHS}
        or not isinstance(core_evidence, dict)
        or set(core_evidence) != set(CORE_EVIDENCE_PATHS)
        or any(
            not isinstance(relative, str) or not relative
            for relative in core_evidence.values()
        )
        or any(
            Path(relative).name != Path(original).name
            for original, relative in core_evidence.items()
        )
        or len(set(core_evidence.values())) != len(core_evidence)
        or not isinstance(readiness_evidence, dict)
        or set(readiness_evidence) != set(CERTIFICATION_PROOF_EVIDENCE_PATHS)
        or any(
            not isinstance(relative, str) or not relative
            for relative in readiness_evidence.values()
        )
        or any(
            Path(relative).name != Path(original).name
            for original, relative in readiness_evidence.items()
        )
        or len(set(readiness_evidence.values())) != len(readiness_evidence)
    ):
        return ["certificate claim_sources are missing or noncanonical"]
    expected_core_claims = {
        "docs/gauntlet/RELEASE_READINESS.json": readiness_relative,
        "docs/gauntlet/ROUNDS.jsonl": rounds_relative,
        **{
            original: next(
                path for path in ledger_paths if Path(path).name == Path(original).name
            )
            for original in HYPOTHESIS_LEDGER_PATHS
        },
    }
    if any(
        core_evidence.get(original) != relative
        for original, relative in expected_core_claims.items()
    ):
        reasons.append(
            "canonical claim sources do not match the core-evidence snapshot mapping"
        )
    for original in set(core_evidence) & set(readiness_evidence):
        if core_evidence[original] != readiness_evidence[original]:
            reasons.append(
                "core/readiness evidence mappings disagree for canonical source: "
                f"{original}"
            )
    resolve_source_digest = source_digest_resolver or _git_blob_sha256
    live_source_root = source_root or root
    git_head = str(certificate.get("git_head", ""))
    canonical_mappings = {**core_evidence, **readiness_evidence}
    for original, relative in canonical_mappings.items():
        if original == "docs/gauntlet/RELEASE_READINESS.json":
            continue
        entry = entries.get(relative)
        expected_digest = entry.get("sha256") if isinstance(entry, dict) else None
        if (
            resolve_source_digest(live_source_root, git_head, original)
            != expected_digest
        ):
            reasons.append(
                "bundle snapshot does not match canonical source at certificate HEAD: "
                f"{original}"
            )
    source_paths = tuple(
        dict.fromkeys((*core_evidence.values(), *readiness_evidence.values()))
    )
    for relative in source_paths:
        if relative not in entries:
            reasons.append(f"bundle manifest omits signed claim source: {relative}")
        path = _safe_repo_path(root, relative)
        try:
            inside_bundle = (
                path is not None
                and path.is_file()
                and path.relative_to(bundle_dir) is not None
            )
        except ValueError:
            inside_bundle = False
        if not inside_bundle:
            reasons.append(
                f"signed claim source is outside the selected bundle: {relative}"
            )
    readiness_path = _safe_repo_path(root, readiness_relative)
    try:
        readiness = (
            _load_bounded_json(readiness_path)
            if readiness_path
            else None
        )
    except (OSError, ValueError, TypeError, RecursionError) as error:
        readiness = None
        reasons.append(f"release-readiness claim source is unreadable: {error}")
    if not isinstance(readiness, dict):
        reasons.append("release-readiness claim source is not a JSON object")
    else:
        cells = readiness.get("cells")
        canonical_cells = (
            isinstance(cells, list)
            and all(isinstance(cell, dict) for cell in cells)
            and tuple(cell.get("cell") for cell in cells)
            == CERTIFICATION_READINESS_CELLS
            and all(cell.get("status") in {"green", "red"} for cell in cells)
            and readiness.get("schema_version") == "gauntlet.release_readiness.v1"
            and readiness.get("artifact") == "franken_ocr.release_readiness.v1"
            and _parse_utc_timestamp(readiness.get("generated_at_utc")) is not None
            and readiness.get("generated_by")
            == "scripts/gauntlet_cert.py --release-readiness"
            and readiness.get("cell_set_sha256")
            == CERTIFICATION_READINESS_CELL_SET_SHA256
            and certificate.get("readiness_cell_set_sha256")
            == CERTIFICATION_READINESS_CELL_SET_SHA256
        )
        if not canonical_cells:
            reasons.append(
                "release-readiness does not contain the exact canonical cell set"
            )
        else:
            external_cells = [
                cell for cell in cells if cell["cell"] != "certification_bundle"
            ]
            for cell in external_cells:
                expected_evidence = [
                    readiness_evidence[original]
                    for original in CERTIFICATION_READINESS_EVIDENCE_PATHS[cell["cell"]]
                ]
                if cell.get("evidence_paths") != expected_evidence:
                    reasons.append(
                        "release-readiness cell lacks canonical manifest evidence: "
                        f"{cell['cell']}"
                    )
            all_blocking = [cell["cell"] for cell in cells if cell["status"] == "red"]
            if (
                not isinstance(readiness.get("green"), int)
                or isinstance(readiness.get("green"), bool)
                or not isinstance(readiness.get("red"), int)
                or isinstance(readiness.get("red"), bool)
                or readiness.get("green")
                != sum(cell["status"] == "green" for cell in cells)
                or readiness.get("red") != len(all_blocking)
                or readiness.get("blocking_cells") != all_blocking
                or readiness.get("ship") != (not all_blocking)
            ):
                reasons.append(
                    "release-readiness aggregate fields do not match canonical cells"
                )
            blocking = [
                cell["cell"] for cell in external_cells if cell["status"] == "red"
            ]
            derived_readiness = {
                "green": sum(cell["status"] == "green" for cell in external_cells),
                "red": len(blocking),
                "blocking_cells": blocking,
                "ship": not blocking,
            }
            if certificate.get("readiness") != derived_readiness:
                reasons.append(
                    "certificate readiness does not match the hashed readiness cells"
                )

    perf_snapshot_relative = readiness_evidence.get("docs/PERF_LEDGER.md")
    perf_snapshot_path = (
        _safe_repo_path(root, perf_snapshot_relative)
        if isinstance(perf_snapshot_relative, str)
        else None
    )
    try:
        perf_snapshot_text = (
            _read_bounded_text(perf_snapshot_path)
            if perf_snapshot_path is not None
            else ""
        )
    except (OSError, ValueError):
        perf_snapshot_text = ""
    perf_ok = (
        perf_evidence_checker(perf_snapshot_text, bundle_dir, git_head)
        if perf_evidence_checker is not None
        else _bundled_perf_evidence_ok(
            perf_snapshot_text, bundle_dir, git_head, live_source_root
        )
    )
    if not perf_ok:
        reasons.append(
            "bundled performance ledger has no self-contained eligible evidence"
        )

    rounds_path = _safe_repo_path(root, rounds_relative)
    try:
        rounds = (
            [
                _parse_json_bytes(
                    line.encode("utf-8"), label=f"{rounds_path}:{line_number}"
                )
                for line_number, line in enumerate(
                    _read_bounded_text(rounds_path).splitlines(), start=1
                )
                if line.strip()
            ]
            if rounds_path
            else []
        )
    except (OSError, ValueError, TypeError, RecursionError) as error:
        rounds = []
        reasons.append(f"round-history claim source is unreadable: {error}")
    for round_reason in _round_history_reasons(rounds):
        reasons.append(f"round-history claim source is invalid: {round_reason}")
    contents: dict[str, str | None] = {}
    for relative in ledger_paths:
        path = _safe_repo_path(root, relative)
        try:
            contents[relative] = _read_bounded_text(path) if path else None
        except (OSError, ValueError):
            contents[relative] = None
    hypotheses = hypothesis_texts_verdict(contents, ledger_paths, bundle_dir)
    dependency_files, dependency_reasons = _hypothesis_dependency_files(
        bundle_dir, contents, ledger_paths
    )
    reasons.extend(dependency_reasons)
    repository_root = root.resolve()
    for dependency in sorted(dependency_files):
        try:
            relative = dependency.relative_to(repository_root).as_posix()
        except ValueError:
            reasons.append(
                f"hypothesis dependency escapes the repository: {dependency}"
            )
            continue
        if relative not in entries:
            reasons.append(f"bundle manifest omits hypothesis dependency: {relative}")
    derived_convergence = convergence_verdict(rounds, hypotheses)
    if certificate.get("convergence") != derived_convergence:
        reasons.append(
            "certificate convergence does not match hashed rounds and hypothesis ledgers"
        )
    return reasons


def _is_canonical_source_pack_artifact(relative: str) -> bool:
    relative_path = Path(relative)
    parts = relative_path.parts
    return (
        not relative_path.is_absolute()
        and relative_path.as_posix() == relative
        and len(parts) >= 5
        and parts[-2:] == ("subject", "source_input_pack.bin")
        and any(
            parts[index : index + 2] == ("artifacts", "perf")
            for index in range(len(parts) - 1)
        )
    )


def _nofollow_regular_file_size(path: Path, maximum: int) -> int:
    descriptor = os.open(path, _readonly_binary_flags())
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"input is not a regular file: {path}")
        if not 0 <= metadata.st_size <= maximum:
            raise ValueError(f"input exceeds the {maximum}-byte size limit: {path}")
        return metadata.st_size
    finally:
        os.close(descriptor)


def _bound_source_pack_artifacts(
    root: Path, bundle_dir: Path, entries: dict[str, dict]
) -> set[str]:
    """Return only source packs named by canonical colocated PERF row bindings."""
    bundle_dir = bundle_dir.resolve()
    bound: set[str] = set()
    for relative, entry in entries.items():
        if "error" in entry or Path(relative).name != "row.json":
            continue
        row_path = _safe_repo_path(root, relative)
        try:
            if row_path is None:
                continue
            row_path.relative_to(bundle_dir)
            row = _parse_strict_json_object(
                _read_bounded_file(row_path), label=f"PERF row binding {relative}"
            )
        except (OSError, ValueError, TypeError, RecursionError):
            continue
        inputs = row.get("inputs")
        binding = inputs.get("source_input_pack") if isinstance(inputs, dict) else None
        if (
            row.get("schema") != "focr-gauntlet-row/v3"
            or not isinstance(inputs, dict)
            or set(inputs) != set(PERF_INPUT_BINDINGS)
            or not isinstance(binding, dict)
            or set(binding) != {"bundle_path", "sha256", "size"}
            or binding.get("bundle_path")
            != PERF_INPUT_BINDINGS["source_input_pack"]
            or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256"))) is None
            or not isinstance(binding.get("size"), int)
            or isinstance(binding.get("size"), bool)
            or not 0 <= binding["size"] <= PERF_MAX_SOURCE_PACK_BYTES
        ):
            continue
        pack_relative = (
            Path(relative).parent / PERF_INPUT_BINDINGS["source_input_pack"]
        ).as_posix()
        pack_entry = entries.get(pack_relative)
        pack_path = _safe_repo_path(root, pack_relative)
        try:
            physical_size = (
                _nofollow_regular_file_size(pack_path, PERF_MAX_SOURCE_PACK_BYTES)
                if pack_path is not None
                else None
            )
        except (OSError, ValueError):
            physical_size = None
        if (
            isinstance(pack_entry, dict)
            and "error" not in pack_entry
            and _is_canonical_source_pack_artifact(pack_relative)
            and binding["sha256"] == pack_entry.get("sha256")
            and binding["size"] == physical_size
        ):
            bound.add(pack_relative)
    return bound


def _snapshot_bundle_artifacts(
    *,
    root: Path,
    bundle_dir: Path,
    entries: dict[str, dict],
    snapshot_root: Path,
    now: datetime,
    issued_at: datetime,
    static_source_paths: set[str],
    source_pack_paths: set[str],
    streamed_artifact_paths: set[str] | None = None,
    max_bundle_bytes: int = CERTIFICATION_MAX_BUNDLE_BYTES,
) -> tuple[list[str], float, int, set[str]]:
    """Verify artifacts and materialize an immutable replay snapshot."""
    bundle_dir = bundle_dir.resolve()
    reasons: list[str] = []
    max_age = 0.0
    total_artifact_bytes = 0
    streamed_paths: set[str] = set()
    total_limit_reported = False
    for relative, entry in entries.items():
        if "error" in entry:
            reasons.append(f"bundle artifact error for {relative}: {entry['error']}")
            continue
        path = _safe_repo_path(root, relative)
        expected = entry.get("sha256")
        if path is None or not path.is_file():
            reasons.append(f"bundle artifact is missing or unsafe: {relative}")
            continue
        try:
            path.relative_to(bundle_dir)
        except ValueError:
            reasons.append(
                f"bundle artifact is outside the selected bundle: {relative}"
            )
            continue
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            reasons.append(f"bundle artifact has invalid SHA-256: {relative}")
            continue
        snapshot_path = _safe_repo_path(snapshot_root, relative)
        if snapshot_path is None:
            reasons.append(f"bundle artifact cannot be snapshotted: {relative}")
            continue
        identity: dict | None = None
        artifact_bytes: bytes | None = None
        try:
            if relative in source_pack_paths or relative in (streamed_artifact_paths or set()):
                identity = _stream_file_identity(
                    path,
                    PERF_MAX_SOURCE_PACK_BYTES
                    if relative in source_pack_paths
                    else PERF_MAX_SUBJECT_BINARY_BYTES,
                    snapshot_path=snapshot_path,
                )
                artifact_size = identity["size"]
                actual_sha256 = identity["sha256"]
                streamed_paths.add(relative)
            else:
                artifact_bytes = _read_bounded_file(path)
                artifact_size = len(artifact_bytes)
                actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_path.write_bytes(artifact_bytes)
        except (OSError, ValueError) as error:
            reasons.append(f"bundle artifact is unreadable: {relative}: {error}")
            continue
        total_artifact_bytes += artifact_size
        if total_artifact_bytes > max_bundle_bytes and not total_limit_reported:
            reasons.append("bundle artifacts exceed the total size limit")
            total_limit_reported = True
        if actual_sha256 != expected:
            reasons.append(f"bundle artifact hash mismatch: {relative}")
        try:
            if identity is not None:
                timestamp, timestamp_source = _streamed_binary_timestamp(
                    identity,
                    issued_at,
                    static_head_source=relative in static_source_paths,
                )
            elif artifact_bytes is not None:
                timestamp, timestamp_source = _manifest_timestamp(
                    path,
                    artifact_bytes,
                    issued_at,
                    static_head_source=relative in static_source_paths,
                )
            else:
                raise ValueError("artifact snapshot has no timestamp source")
        except (OSError, ValueError, TypeError, RecursionError) as error:
            reasons.append(
                f"bundle artifact timestamp unreadable for {relative}: {error}"
            )
            continue
        recorded_timestamp = _parse_utc_timestamp(entry.get("timestamp_utc"))
        if (
            recorded_timestamp != timestamp
            or entry.get("timestamp_source") != timestamp_source
        ):
            reasons.append(f"bundle artifact timestamp binding mismatch: {relative}")
        age_hours = (now - timestamp).total_seconds() / 3600.0
        recorded_age = entry.get("age_hours")
        if (
            not _finite_number(
                recorded_age,
                minimum=-5.0 / 60.0,
                maximum=CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
            )
            or abs(float(recorded_age) - round(age_hours, 2)) > 0.001
        ):
            reasons.append(f"bundle artifact age binding mismatch: {relative}")
        max_age = max(max_age, age_hours)
        age_reason = _age_reason(timestamp, now, f"bundle artifact {relative}")
        if age_reason:
            reasons.append(age_reason)
    return reasons, max_age, total_artifact_bytes, streamed_paths


def certificate_bundle_verdict(
    root: Path,
    current_head: str,
    required_artifacts: Sequence[str] = (),
    *,
    bundle_dir: Path | None = None,
    now: datetime | None = None,
    worktree_state: dict | None = None,
    signature_verifier: Callable[[Path, dict, str], bool] | None = None,
    trusted_signers: dict[str, str] | None = None,
    ci_run_verifier: Callable[[Path, str, str, Sequence[dict]], bool] | None = None,
    source_digest_resolver: Callable[[Path, str, str], str | None] | None = None,
    perf_evidence_checker: Callable[[str, Path, str], bool] | None = None,
    control_documents: tuple[bytes, bytes, bytes] | None = None,
) -> dict:
    """Verify the complete strict certificate against live source and evidence."""
    reasons: list[str] = []
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    requested_bundle = bundle_dir or root / ".gauntlet-output/bundle"
    bundle_dir, output_reasons = _safe_output_dir(root, str(requested_bundle))
    if bundle_dir is None:
        return {"ok": False, "reasons": output_reasons}
    certificate_path = bundle_dir / "release_certificate.json"
    bundle_path = bundle_dir / "certification_bundle.json"
    report_path = bundle_dir / "FINAL_GAUNTLET_REPORT.md"
    try:
        if control_documents is None:
            certificate_bytes = _read_bounded_file(certificate_path)
            bundle_bytes = _read_bounded_file(bundle_path)
            report_bytes = _read_bounded_file(report_path)
        else:
            certificate_bytes, bundle_bytes, report_bytes = control_documents
        certificate = _parse_json_bytes(certificate_bytes, label=str(certificate_path))
        bundle = _parse_json_bytes(bundle_bytes, label=str(bundle_path))
        control_file_hashes = (
            {
                certificate_path: hashlib.sha256(certificate_bytes).hexdigest(),
                bundle_path: hashlib.sha256(bundle_bytes).hexdigest(),
                report_path: hashlib.sha256(report_bytes).hexdigest(),
            }
            if control_documents is None
            else {}
        )
    except (OSError, ValueError, TypeError, RecursionError) as error:
        return {
            "ok": False,
            "reasons": [f"missing or unreadable certificate bundle: {error}"],
        }
    if not isinstance(certificate, dict) or not isinstance(bundle, dict):
        return {
            "ok": False,
            "reasons": ["certificate and bundle must both be JSON objects"],
        }
    if set(certificate) != STRICT_CERTIFICATE_FIELDS:
        reasons.append("certificate object has a noncanonical field set")
    if report_bytes != _final_report_text(certificate, bundle).encode("utf-8"):
        reasons.append(
            "FINAL_GAUNTLET_REPORT.md is not the canonical signed projection"
        )

    if certificate.get("schema_version") != STRICT_CERTIFICATE_SCHEMA:
        reasons.append(f"certificate schema_version is not {STRICT_CERTIFICATE_SCHEMA}")
    if certificate.get("artifact") != STRICT_CERTIFICATE_ARTIFACT:
        reasons.append(f"certificate artifact is not {STRICT_CERTIFICATE_ARTIFACT}")
    if certificate.get("template") != STRICT_CERTIFICATE_SCHEMA:
        reasons.append(f"certificate template is not {STRICT_CERTIFICATE_SCHEMA}")
    if certificate.get("certified") is not True:
        reasons.append("certificate verdict is not certified")
    if not _json_exact(certificate.get("constants"), CERTIFICATION_CONSTANTS):
        reasons.append("certificate required-pass constants are missing or altered")
    if certificate.get("project") != "franken_ocr":
        reasons.append("certificate project identity is missing or incorrect")
    package_version = _cargo_package_version(root)
    if package_version is None or certificate.get("version") != package_version:
        reasons.append("certificate version does not match Cargo.toml")
    if worktree_state is None:
        cargo_path = root / "Cargo.toml"
        try:
            live_cargo_sha256 = _sha256_file(cargo_path)
        except OSError:
            live_cargo_sha256 = None
        if _git_blob_sha256(root, current_head, "Cargo.toml") != live_cargo_sha256:
            reasons.append("Cargo.toml does not match the certificate HEAD")
    if not _valid_git_head(current_head):
        reasons.append("current git HEAD is unavailable")
    elif certificate.get("git_head") != current_head:
        reasons.append(
            f"certificate git_head {certificate.get('git_head')!r} != current HEAD {current_head}"
        )
    if certificate.get("evidence_git_head") != current_head:
        reasons.append(
            "certificate evidence_git_head does not bind the final evidence HEAD"
        )
    if certificate.get("evidence_git_head") != certificate.get("git_head"):
        reasons.append("certificate git_head/evidence_git_head disagree")
    if certificate.get("git_branch") != "main":
        reasons.append("certificate is not bound to the main branch")
    if (
        worktree_state is None
        and _git_output(root, "symbolic-ref", "--short", "HEAD") != "main"
    ):
        reasons.append("live checkout is not on the main branch")
    if _git_output(root, "replace", "-l"):
        reasons.append("git replacement refs are present")
    reference = certificate.get("reference")
    if (
        not isinstance(reference, dict)
        or reference.get("model_commit") != UNLIMITED_OCR_MODEL_COMMIT
    ):
        reasons.append("certificate reference identity is missing or unpinned")
    issued_at = _parse_utc_timestamp(certificate.get("issued_at"))
    issued_reason = _age_reason(
        issued_at, now, "certificate", CERTIFICATION_MAX_EVIDENCE_AGE_HOURS
    )
    if issued_reason:
        reasons.append(issued_reason)

    live_worktree = worktree_state or _git_worktree_state(root)
    live_describe = _git_output(root, "describe", "--tags", "--always", "--dirty")
    if live_worktree.get("status_ok") is not True:
        reasons.append("git worktree status could not be verified")
    if live_worktree.get("clean") is not True:
        reasons.append(
            f"git worktree is dirty ({live_worktree.get('dirty_path_count', '?')} status entries)"
        )
    recorded_worktree = certificate.get("git_worktree")
    if not isinstance(recorded_worktree, dict) or not _json_exact(
        recorded_worktree, live_worktree
    ):
        reasons.append(
            "certificate git_worktree state does not match the live worktree"
        )
    if worktree_state is None and certificate.get("git_describe") != live_describe:
        reasons.append(
            "certificate git_describe does not match the live dirty-state description"
        )

    readiness = certificate.get("readiness")
    readiness_green = readiness.get("green") if isinstance(readiness, dict) else None
    if not isinstance(readiness, dict) or not (
        readiness.get("ship") is True
        and isinstance(readiness.get("red"), int)
        and not isinstance(readiness.get("red"), bool)
        and readiness.get("red") == 0
        and readiness.get("blocking_cells") == []
        and isinstance(readiness_green, int)
        and not isinstance(readiness_green, bool)
        and readiness_green > 0
    ):
        reasons.append("certificate readiness is not an all-green ship decision")
    convergence = certificate.get("convergence")
    convergence_rounds = (
        convergence.get("rounds") if isinstance(convergence, dict) else None
    )
    if not isinstance(convergence, dict) or not (
        convergence.get("converged") is True
        and convergence.get("result") == "pass"
        and isinstance(convergence_rounds, int)
        and not isinstance(convergence_rounds, bool)
        and convergence_rounds >= MIN_ROUNDS
        and convergence.get("tail_clean") is True
        and convergence.get("hypotheses_resolved") is True
    ):
        reasons.append("certificate convergence claim is incomplete or false")
    actuals = certificate.get("required_pass_actuals")
    if not isinstance(actuals, dict) or not (
        isinstance(actuals.get("min_verification_pct_observed"), (int, float))
        and not isinstance(actuals.get("min_verification_pct_observed"), bool)
        and actuals.get("min_verification_pct_observed") == 100.0
        and isinstance(
            actuals.get("required_suite_pass_rate_pct_observed"), (int, float)
        )
        and not isinstance(actuals.get("required_suite_pass_rate_pct_observed"), bool)
        and actuals.get("required_suite_pass_rate_pct_observed") == 100.0
        and isinstance(actuals.get("high_severity_counterexample_count"), int)
        and not isinstance(actuals.get("high_severity_counterexample_count"), bool)
        and actuals.get("high_severity_counterexample_count") == 0
        and _finite_number(
            actuals.get("max_evidence_age_hours_observed"),
            minimum=0.0,
            maximum=CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
        )
    ):
        reasons.append(
            "certificate observed required-pass values do not satisfy the strict constants"
        )
    counterexample_count = certificate.get("high_severity_counterexamples")
    if (
        not isinstance(counterexample_count, int)
        or isinstance(counterexample_count, bool)
        or counterexample_count != 0
    ):
        reasons.append(
            "certificate has a nonzero or unknown high-severity counterexample count"
        )
    parity_score = certificate.get("parity_score")
    if (
        not _finite_number(parity_score, minimum=0.0, maximum=1.0)
        or truncate_score(float(parity_score)) != parity_score
    ):
        reasons.append(
            "certificate parity_score is not a truncated conformal lower bound"
        )
    if certificate.get("refusal_reasons") != []:
        reasons.append("a certified certificate must have no refusal reasons")

    if bundle.get("schema_version") != STRICT_BUNDLE_SCHEMA:
        reasons.append(f"bundle schema_version is not {STRICT_BUNDLE_SCHEMA}")
    if bundle.get("artifact") != STRICT_BUNDLE_ARTIFACT:
        reasons.append(f"bundle artifact is not {STRICT_BUNDLE_ARTIFACT}")
    if set(bundle) != {
        "schema_version",
        "artifact",
        "generated_at_utc",
        "bundle_root_sha256",
        "manifest",
    }:
        reasons.append("bundle object has a noncanonical field set")
    if _parse_utc_timestamp(bundle.get("generated_at_utc")) != issued_at:
        reasons.append(
            "bundle generation timestamp does not match certificate issuance"
        )
    manifest = bundle.get("manifest")
    if not isinstance(manifest, list):
        reasons.append("bundle manifest is missing or not a list")
        manifest = []
    elif len(manifest) > CERTIFICATION_MAX_BUNDLE_ARTIFACTS:
        reasons.append("bundle manifest exceeds the verifier artifact-count limit")
        manifest = manifest[:CERTIFICATION_MAX_BUNDLE_ARTIFACTS]
    entries: dict[str, dict] = {}
    max_age = 0.0
    certificate_claim_sources = certificate.get("claim_sources")
    static_source_paths: set[str] = set()
    if isinstance(certificate_claim_sources, dict):
        for mapping_name in ("core_evidence", "readiness_evidence"):
            mapping = certificate_claim_sources.get(mapping_name)
            if isinstance(mapping, dict):
                static_source_paths.update(
                    relative
                    for original, relative in mapping.items()
                    if isinstance(original, str)
                    and original
                    not in {
                        "docs/gauntlet/RELEASE_READINESS.json",
                        "docs/gauntlet/ROUNDS.jsonl",
                    }
                    and isinstance(relative, str)
                )
    for entry in manifest:
        if not isinstance(entry, dict) or not isinstance(entry.get("artifact"), str):
            reasons.append("bundle manifest contains a malformed entry")
            continue
        relative = entry["artifact"]
        expected_entry_keys = (
            {"artifact", "error"}
            if "error" in entry
            else {
                "artifact",
                "sha256",
                "age_hours",
                "timestamp_utc",
                "timestamp_source",
            }
        )
        if set(entry) != expected_entry_keys:
            reasons.append(f"bundle manifest entry has noncanonical fields: {relative}")
            continue
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            reasons.append(
                f"bundle manifest contains a noncanonical artifact: {relative}"
            )
            continue
        if relative in entries:
            reasons.append(f"bundle manifest contains duplicate artifact: {relative}")
            continue
        entries[relative] = entry
    declared_signature_paths = (
        {
            signature.get("signature_path")
            for signature in certificate.get("detached_signatures", [])
            if isinstance(signature, dict)
            and isinstance(signature.get("signature_path"), str)
        }
        if isinstance(certificate.get("detached_signatures"), list)
        else set()
    )
    allowed_unmanifested = {
        str(certificate_path.relative_to(root.resolve())),
        str(bundle_path.relative_to(root.resolve())),
        str((bundle_dir / "FINAL_GAUNTLET_REPORT.md").relative_to(root.resolve())),
        *declared_signature_paths,
    }
    actual_bundle_files = {
        str(path.relative_to(root.resolve()))
        for path in bundle_dir.rglob("*")
        if path.is_file() and not path.name.startswith("._")
    }
    unmanifested = sorted(actual_bundle_files - set(entries) - allowed_unmanifested)
    if unmanifested:
        reasons.append(
            "selected bundle contains unmanifested evidence files: "
            + ", ".join(unmanifested)
        )
    for required in required_artifacts:
        if required not in entries:
            reasons.append(f"bundle manifest omits required artifact: {required}")
    source_pack_paths = _bound_source_pack_artifacts(root, bundle_dir, entries)
    streamed_artifact_paths: set[str] = set()
    evidence_classes = certificate.get("evidence_classes")
    parity_relative = (
        evidence_classes.get("model_parity_receipt")
        if isinstance(evidence_classes, dict)
        else None
    )
    parity_path = (
        _safe_repo_path(root, parity_relative)
        if isinstance(parity_relative, str)
        else None
    )
    try:
        parity_document = (
            _load_bounded_json(parity_path)
            if parity_path is not None and parity_path.is_file()
            else {}
        )
    except (OSError, ValueError, TypeError, RecursionError):
        parity_document = {}
    if isinstance(parity_document, dict) and isinstance(
        parity_document.get("oracle_fixture_bindings"), list
    ):
        streamed_artifact_paths = {
            binding.get("path")
            for binding in parity_document["oracle_fixture_bindings"]
            if isinstance(binding, dict) and isinstance(binding.get("path"), str)
        }

    root_hash = _bundle_root_sha256(manifest)
    if root_hash is None:
        reasons.append("bundle root cannot be computed from the manifest")
    else:
        if bundle.get("bundle_root_sha256") != root_hash:
            reasons.append("bundle root hash does not match the manifest")
        if certificate.get("evidence_bundle_sha256") != root_hash:
            reasons.append(
                "certificate evidence_bundle_sha256 does not match the manifest"
            )
    bundle_relative = bundle_dir.relative_to(root.resolve())
    with tempfile.TemporaryDirectory(prefix="focr-cert-evidence-snapshot-") as tmp:
        snapshot_root = Path(tmp)
        (
            artifact_reasons,
            artifact_max_age,
            _total_artifact_bytes,
            _streamed_artifacts,
        ) = _snapshot_bundle_artifacts(
            root=root,
            bundle_dir=bundle_dir,
            entries=entries,
            snapshot_root=snapshot_root,
            now=now,
            issued_at=issued_at or now,
            static_source_paths=static_source_paths,
            source_pack_paths=source_pack_paths,
            streamed_artifact_paths=streamed_artifact_paths,
        )
        reasons.extend(artifact_reasons)
        max_age = max(max_age, artifact_max_age)
        snapshot_bundle_dir = snapshot_root / bundle_relative
        reasons.extend(
            _strict_evidence_reasons(
                snapshot_root,
                snapshot_bundle_dir,
                certificate,
                entries,
                provenance_root=root,
                ci_run_verifier=ci_run_verifier,
            )
        )
        reasons.extend(
            _certificate_source_claim_reasons(
                snapshot_root,
                snapshot_bundle_dir,
                certificate,
                entries,
                source_root=root,
                source_digest_resolver=source_digest_resolver,
                perf_evidence_checker=perf_evidence_checker,
            )
        )

    signers = certificate.get("signers")
    signatures = certificate.get("detached_signatures")
    if (
        not isinstance(signers, list)
        or any(not isinstance(signer, str) or not signer for signer in signers)
        or len(signers) != CERTIFICATION_REQUIRED_SIGNERS
        or len(set(signers)) != CERTIFICATION_REQUIRED_SIGNERS
    ):
        reasons.append("certificate must declare exactly three distinct signers")
        signers = []
    if (
        not isinstance(signatures, list)
        or len(signatures) != CERTIFICATION_REQUIRED_SIGNERS
    ):
        reasons.append("certificate must declare exactly three detached signatures")
        signatures = []
    signer_roles: set[str] = set()
    signature_signers: set[str] = set()
    signature_fingerprints: set[str] = set()
    signature_paths: set[str] = set()
    signature_file_hashes: dict[Path, str] = {}
    trust_file_hashes: dict[Path, str] = {}
    trusted_registry_bytes: bytes | None = None
    keyring_bytes: bytes | None = None
    if trusted_signers is None:
        for relative in (TRUSTED_SIGNERS_PATH, TRUSTED_KEYRING_PATH):
            path = _safe_repo_path(root, relative)
            try:
                content = _read_bounded_file(path) if path is not None else None
            except (OSError, ValueError):
                content = None
            if path is None or content is None:
                reasons.append(f"release trust artifact is missing: {relative}")
                continue
            trust_file_hashes[path] = hashlib.sha256(content).hexdigest()
            if (
                _git_blob_sha256(root, current_head, relative)
                != trust_file_hashes[path]
            ):
                reasons.append(
                    f"release trust artifact does not match certificate HEAD: {relative}"
                )
            if relative == TRUSTED_SIGNERS_PATH:
                trusted_registry_bytes = content
            else:
                keyring_bytes = content
    trusted = {
        identity: fingerprint.upper()
        for identity, fingerprint in (
            trusted_signers
            if trusted_signers is not None
            else _trusted_signer_fingerprints(root, trusted_registry_bytes)
        ).items()
        if isinstance(identity, str)
        and isinstance(fingerprint, str)
        and re.fullmatch(r"[0-9A-Fa-f]{40,64}", fingerprint)
    }
    signed_claim_sha256 = _certificate_signed_claim_sha256(certificate, root_hash)
    if signed_claim_sha256 is None:
        reasons.append("certificate signed claim cannot be canonicalized")
    elif certificate.get("signed_claim_sha256") != signed_claim_sha256:
        reasons.append(
            "certificate signed_claim_sha256 does not authenticate its claims"
        )
    for signature in signatures:
        if not isinstance(signature, dict) or set(signature) != {
            "signer",
            "role",
            "fingerprint",
            "scheme",
            "signature_path",
        }:
            reasons.append("certificate contains a malformed detached signature")
            continue
        signer = signature.get("signer")
        role = signature.get("role")
        fingerprint = str(signature.get("fingerprint", "")).upper()
        signature_relative = signature.get("signature_path")
        if signer not in signers or not isinstance(role, str) or not role:
            reasons.append("detached signature signer/role is not declared")
            continue
        if trusted.get(signer) != fingerprint:
            reasons.append(
                f"detached signature signer is not pinned to a trusted fingerprint: {signer}"
            )
            continue
        if not isinstance(signature_relative, str) or not signature_relative:
            reasons.append(f"detached signature path is missing for {signer}")
            continue
        signature_relative_path = Path(signature_relative)
        if (
            signature_relative_path.is_absolute()
            or signature_relative_path.as_posix() != signature_relative
            or any(part in {"", ".", ".."} for part in signature_relative_path.parts)
        ):
            reasons.append(f"detached signature path is noncanonical for {signer}")
            continue
        if (
            signer in signature_signers
            or role in signer_roles
            or fingerprint in signature_fingerprints
            or signature_relative in signature_paths
        ):
            reasons.append(
                "detached signatures do not have distinct signers, roles, fingerprints, and files"
            )
            continue
        signature_path = (
            _safe_repo_path(root, signature_relative)
            if isinstance(signature_relative, str)
            else None
        )
        try:
            signature_inside_bundle = (
                signature_path is not None
                and signature_path.is_file()
                and signature_path.relative_to(bundle_dir) is not None
            )
        except ValueError:
            signature_inside_bundle = False
        if not signature_inside_bundle:
            reasons.append(
                f"detached signature file is missing or outside the selected bundle: {signer}"
            )
            continue
        try:
            signature_bytes = _read_bounded_file(signature_path)
        except (OSError, ValueError):
            reasons.append(f"detached signature file is unreadable for {signer}")
            continue
        signature_file_hashes[signature_path] = hashlib.sha256(
            signature_bytes
        ).hexdigest()
        signature_signers.add(signer)
        signer_roles.add(role)
        signature_fingerprints.add(fingerprint)
        signature_paths.add(str(signature_relative))
        if signature.get("scheme") != "openpgp-detached":
            reasons.append(f"detached signature for {signer} has the wrong scheme")
        elif signed_claim_sha256 is None:
            reasons.append(f"detached signature verification failed for {signer}")
        elif signature_verifier is not None:
            if not signature_verifier(root, signature, signed_claim_sha256):
                reasons.append(f"detached signature verification failed for {signer}")
        elif not _default_signature_verifier(
            root,
            signature,
            signed_claim_sha256,
            signature_bytes=signature_bytes,
            keyring_bytes=keyring_bytes,
        ):
            reasons.append(f"detached signature verification failed for {signer}")
    if len(signature_signers) != CERTIFICATION_REQUIRED_SIGNERS:
        reasons.append("fewer than three independent detached signatures verified")
    if signer_roles != CERTIFICATION_REQUIRED_SIGNATURE_ROLES:
        reasons.append("detached signatures do not cover the exact required roles")
    if len(signature_fingerprints) != CERTIFICATION_REQUIRED_SIGNERS:
        reasons.append("detached signatures do not cover three trusted fingerprints")
    if set(signers) != signature_signers:
        reasons.append("certificate signer list does not exactly match signatures")

    recorded_max_age = (
        actuals.get("max_evidence_age_hours_observed")
        if isinstance(actuals, dict)
        else None
    )
    manifest_max_age = max(
        [
            0.0,
            *(
                float(entry["age_hours"])
                for entry in manifest
                if isinstance(entry, dict)
                and _finite_number(
                    entry.get("age_hours"),
                    minimum=-5.0 / 60.0,
                    maximum=CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
                )
            ),
        ]
    )
    if _finite_number(
        recorded_max_age,
        minimum=0.0,
        maximum=CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
    ):
        if abs(float(recorded_max_age) - manifest_max_age) > 0.001:
            reasons.append(
                "certificate max evidence age does not match its signed manifest"
            )
    for control_path, initial_digest in control_file_hashes.items():
        try:
            control_changed = (
                not control_path.is_file()
                or hashlib.sha256(_read_bounded_file(control_path)).hexdigest()
                != initial_digest
            )
        except (OSError, ValueError):
            control_changed = True
        if control_changed:
            reasons.append(
                f"bundle control file changed during verification: {control_path.name}"
            )
    for protected_path, initial_digest in {
        **trust_file_hashes,
        **signature_file_hashes,
    }.items():
        try:
            protected_changed = (
                not protected_path.is_file()
                or hashlib.sha256(_read_bounded_file(protected_path)).hexdigest()
                != initial_digest
            )
        except (OSError, ValueError):
            protected_changed = True
        if protected_changed:
            reasons.append(
                "release trust/signature file changed during verification: "
                f"{protected_path}"
            )
    for relative, entry in entries.items():
        if "error" in entry:
            continue
        path = _safe_repo_path(root, relative)
        try:
            if path is None or not path.is_file():
                raise ValueError("artifact is missing")
            if relative in source_pack_paths or relative in streamed_artifact_paths:
                final_identity = _stream_file_identity(
                    path,
                    PERF_MAX_SOURCE_PACK_BYTES
                    if relative in source_pack_paths
                    else PERF_MAX_SUBJECT_BINARY_BYTES,
                )
                final_sha256 = final_identity["sha256"]
                final_timestamp, final_timestamp_source = (
                    _streamed_binary_timestamp(
                        final_identity,
                        issued_at or now,
                        static_head_source=relative in static_source_paths,
                    )
                )
            else:
                final_bytes = _read_bounded_file(path)
                final_sha256 = hashlib.sha256(final_bytes).hexdigest()
                final_timestamp, final_timestamp_source = _manifest_timestamp(
                    path,
                    final_bytes,
                    issued_at or now,
                    static_head_source=relative in static_source_paths,
                )
        except (OSError, ValueError, TypeError, RecursionError):
            reasons.append(f"bundle artifact changed during verification: {relative}")
            continue
        if (
            final_sha256 != entry.get("sha256")
            or _parse_utc_timestamp(entry.get("timestamp_utc")) != final_timestamp
            or entry.get("timestamp_source") != final_timestamp_source
        ):
            reasons.append(f"bundle artifact changed during verification: {relative}")
    _, final_output_reasons = _safe_output_dir(root, str(bundle_dir))
    reasons.extend(
        f"bundle output changed during verification: {reason}"
        for reason in final_output_reasons
    )
    final_bundle_files = {
        str(path.relative_to(root.resolve()))
        for path in bundle_dir.rglob("*")
        if path.is_file() and not path.name.startswith("._")
    }
    if final_bundle_files != actual_bundle_files:
        reasons.append("selected bundle file set changed during verification")
    if worktree_state is None:
        final_head = _git_output(root, "rev-parse", "HEAD")
        final_worktree = _git_worktree_state(root)
        final_describe = _git_output(root, "describe", "--tags", "--always", "--dirty")
        if final_head != current_head:
            reasons.append("git HEAD changed during certificate verification")
        if final_worktree != live_worktree:
            reasons.append("git worktree changed during certificate verification")
        if final_describe != live_describe:
            reasons.append("git describe changed during certificate verification")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "bundle_dir": str(bundle_dir),
        "max_evidence_age_hours": round(max_age, 2),
    }


def build_evidence_manifest(
    root: Path,
    now_timestamp: float,
    evidence_paths: Sequence[str] = CORE_EVIDENCE_PATHS,
    max_age_hours: float = CERTIFICATION_MAX_EVIDENCE_AGE_HOURS,
    static_source_paths: set[str] | None = None,
    source_pack_paths: set[str] | None = None,
    streamed_artifact_paths: set[str] | None = None,
    max_bundle_bytes: int = CERTIFICATION_MAX_BUNDLE_BYTES,
) -> tuple[list[dict], list[str]]:
    manifest: list[dict] = []
    stale: list[str] = []
    total_artifact_bytes = 0
    total_limit_reported = False
    for relative in evidence_paths:
        path = _safe_repo_path(root, relative)
        if path is None:
            manifest.append({"artifact": relative, "error": "path escapes repository"})
            stale.append(relative)
            continue
        try:
            now = datetime.fromtimestamp(now_timestamp, timezone.utc)
            if (
                relative in (source_pack_paths or set())
                and _is_canonical_source_pack_artifact(relative)
            ) or relative in (streamed_artifact_paths or set()):
                identity = _stream_file_identity(
                    path,
                    PERF_MAX_SOURCE_PACK_BYTES
                    if relative in (source_pack_paths or set())
                    else PERF_MAX_SUBJECT_BINARY_BYTES,
                )
                artifact_size = identity["size"]
                digest = identity["sha256"]
                timestamp, timestamp_source = _streamed_binary_timestamp(
                    identity,
                    now,
                    static_head_source=relative in (static_source_paths or set()),
                )
            else:
                content = _read_bounded_file(path)
                artifact_size = len(content)
                digest = hashlib.sha256(content).hexdigest()
                timestamp, timestamp_source = _manifest_timestamp(
                    path,
                    content,
                    now,
                    static_head_source=relative in (static_source_paths or set()),
                )
            total_artifact_bytes += artifact_size
            if total_artifact_bytes > max_bundle_bytes:
                total_limit_reported = True
                raise ValueError("bundle artifacts exceed the total size limit")
            age_hours = (now - timestamp).total_seconds() / 3600.0
            entry = {
                "artifact": relative,
                "sha256": digest,
                "age_hours": round(age_hours, 2),
                "timestamp_utc": _timestamp_text(timestamp),
                "timestamp_source": timestamp_source,
            }
            manifest.append(entry)
            if age_hours > max_age_hours or age_hours < -5.0 / 60.0:
                stale.append(relative)
        except (OSError, ValueError, TypeError, RecursionError) as error:
            manifest.append({"artifact": relative, "error": str(error)})
            stale.append(relative)
    if total_limit_reported:
        stale.extend(
            relative for relative in evidence_paths if relative not in stale
        )
    return manifest, stale


def from_parity(md_path: str, residuals_path: str | None, out_path: str | None) -> int:
    """Emit the surface-pillar scorecard computed from the REAL scoreboard.

    With no residual history the conformal band bootstraps WIDE (halfwidth
    1.0 -> lower bound 0.0) per the §3.2 pitfall guard — the artifact says so
    explicitly rather than inventing confidence it does not have. Residual
    history accrues one entry per gauntlet round (|observed - Beta-mean|).
    """
    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()
    cats = categories_from_parity(md_text)
    residuals: list[float] = []
    if residuals_path and os.path.exists(residuals_path):
        with open(residuals_path, encoding="utf-8") as f:
            residuals = json.load(f)
    sc = compute_scorecard(cats, residuals, confidence=0.95)
    debt = {
        c.category_id: round(c.weighted_excluded, 6)
        for c in cats
        if c.weighted_excluded > 0
    }
    artifact = {
        "artifact": "franken_ocr.gauntlet.scorecard.v1",
        "source": md_path,
        "pillar": "surface",
        "point_estimate": sc.point_estimate,
        "parity_score_lower_bound": sc.lower_bound,
        "conformal_halfwidth": round(sc.conformal_halfwidth, 6),
        "residual_history_n": len(residuals),
        "band_note": (
            "no residual history yet: bootstrap band 1.0 -> lower bound 0 (maximally conservative, §3.2)"
            if not residuals
            else "band from held-out round residuals"
        ),
        "per_category_mean": sc.per_category_mean,
        "per_category_lower": sc.per_category_lower,
        "excluded_weight_debt": debt,
    }
    text = json.dumps(artifact, sort_keys=True, indent=1)
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


# Convergence (METHODOLOGY §7): >=10 full rounds, the last >=2 consecutive
# rounds each produced <3 new genuine findings, AND every hypothesis ledger is
# present with no unresolved entry. Exits nonzero until all three conditions
# hold — this is the gate bd-wp8.8 runs behind.
MIN_ROUNDS = 10
CLEAN_ROUNDS = 2
CLEAN_FINDING_CEILING = 3


def _round_history_reasons(rounds: object) -> list[str]:
    if not isinstance(rounds, list):
        return ["round history is not a list"]
    reasons: list[str] = []
    required_keys = {"round", "date", "new_findings", "pillars", "notes"}
    required_pillars = {"conformance", "surface", "perf"}
    for expected_round, record in enumerate(rounds, start=1):
        if not isinstance(record, dict) or set(record) != required_keys:
            reasons.append(f"round {expected_round} has a noncanonical schema")
            continue
        round_number = record.get("round")
        findings = record.get("new_findings")
        pillars = record.get("pillars")
        if (
            not isinstance(round_number, int)
            or isinstance(round_number, bool)
            or round_number != expected_round
        ):
            reasons.append(f"round {expected_round} has a nonsequential round id")
        if not isinstance(findings, int) or isinstance(findings, bool) or findings < 0:
            reasons.append(f"round {expected_round} has invalid new_findings")
        if (
            not isinstance(record.get("date"), str)
            or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", record["date"]) is None
            or _parse_utc_timestamp(record["date"]) is None
        ):
            reasons.append(f"round {expected_round} has an invalid date")
        if (
            not isinstance(pillars, dict)
            or set(pillars) != required_pillars
            or any(
                not isinstance(value, str) or not value for value in pillars.values()
            )
        ):
            reasons.append(f"round {expected_round} has invalid pillar evidence")
        if not isinstance(record.get("notes"), str) or not record["notes"]:
            reasons.append(f"round {expected_round} has invalid notes")
    return reasons


def convergence_verdict(rounds: list[dict], hypotheses: dict | None = None) -> dict:
    round_errors = _round_history_reasons(rounds)
    n = len(rounds) if isinstance(rounds, list) else 0
    tail = (
        rounds[-CLEAN_ROUNDS:]
        if isinstance(rounds, list) and not round_errors and n >= CLEAN_ROUNDS
        else []
    )
    tail_clean = len(tail) == CLEAN_ROUNDS and all(
        r["new_findings"] < CLEAN_FINDING_CEILING for r in tail
    )
    hypotheses_resolved = bool(hypotheses and hypotheses.get("resolved") is True)
    converged = (
        not round_errors and n >= MIN_ROUNDS and tail_clean and hypotheses_resolved
    )
    return {
        "check": "gauntlet-convergence",
        "rounds": n,
        "min_rounds": MIN_ROUNDS,
        "tail_clean": tail_clean,
        "tail_findings": [int(r.get("new_findings", -1)) for r in tail],
        "clean_ceiling": CLEAN_FINDING_CEILING,
        "round_history_errors": round_errors,
        "hypotheses_resolved": hypotheses_resolved,
        "hypothesis_ledgers": hypotheses
        or {
            "resolved": False,
            "missing": list(HYPOTHESIS_LEDGER_PATHS),
            "unresolved": [],
            "entries": 0,
        },
        "converged": converged,
        "result": "pass" if converged else "fail",
    }


def convergence(rounds_path: str) -> int:
    root = _repo_root()
    rounds: list[dict] = []
    path = Path(rounds_path)
    if path.exists():
        try:
            rounds = [
                parsed
                for line_number, line in enumerate(
                    _read_bounded_text(path).splitlines(), start=1
                )
                if line.strip()
                for parsed in [
                    _parse_json_bytes(
                        line.encode("utf-8"), label=f"{path}:{line_number}"
                    )
                ]
                if isinstance(parsed, dict)
            ]
        except (OSError, TypeError, ValueError, RecursionError):
            rounds = []
    verdict = convergence_verdict(rounds, hypothesis_ledger_verdict(root))
    verdict["rounds_file"] = rounds_path
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["converged"] else 1


# --------------------------------------------------------------------------- #
# bd-re8.15 — wiring the e-processes to the LIVE invariant streams.
#
# The four load-bearing invariants already EMIT structured NDJSON wherever
# they are exercised (the determinism gates, `robot selftest`, the i32
# overflow proofs, the R-SWA/KV bound gates). The fold below maps those
# real lines onto e-process observations and persists the per-invariant
# state ACROSS runs — the e-value only ever multiplies (Ville's inequality
# holds over the whole history; resetting would forge the guarantee).
# --------------------------------------------------------------------------- #

# invariant -> lowercase substrings matched against the line's identifying
# fields. The mapping is deliberately explicit and versioned here: adding an
# invariant means adding its emission-site vocabulary alongside.
EPROCESS_MATCHERS: list[tuple[str, tuple[str, ...]]] = [
    ("INV-DETERMINISM", ("determin",)),
    ("INV-SIMD-SCALAR", ("selftest", "simd")),
    ("INV-I32-NOOVERFLOW", ("overflow",)),
    ("INV-KV-CAP", ("kv_cap", "kv_bound", "kvcache_bound", "rswa_bound")),
]

_EPROCESS_ID_FIELDS = (
    "test",
    "case",
    "check",
    "gate",
    "suite",
    "relation",
    "assertion",
    "invariant",
    "command",
)


def classify_invariant_line(obj: dict) -> tuple[str, bool] | None:
    """Map one structured log line to (invariant, alarm) or None.

    Only lines carrying an explicit pass/fail verdict count as observations —
    skips (`skip_no_model`) are NOT evidence in either direction."""
    result = str(obj.get("result", obj.get("verdict", "")))
    if result not in ("pass", "fail"):
        return None
    hay = " ".join(str(obj.get(f, "")) for f in _EPROCESS_ID_FIELDS).lower()
    for invariant, needles in EPROCESS_MATCHERS:
        if any(n in hay for n in needles):
            return (invariant, result == "fail")
    return None


def _eprocess_from_state(name: str, saved: dict | None) -> EProcess:
    ep = EProcess.for_invariant(name)
    if saved:
        ep.e_value = float(saved["e_value"])
        ep.obs_count = int(saved["obs_count"])
        ep.rejected_at = saved.get("rejected_at")
    return ep


# --------------------------------------------------------------------------- #
# bd-wp8.10 — the release-readiness scorecard (the all-green ship gate).
#
# Reads each cell's EVIDENCE ARTIFACT and asserts green; a missing or stale
# artifact is a RED cell, and ANY red cell exits nonzero — the gate is hard,
# not advisory (G7). Cells whose delivering bead is still open are honestly
# RED with the owning bead named; the gate goes green when the work exists,
# never before.
# --------------------------------------------------------------------------- #


def _cell(name: str, status: str, evidence: str, detail: str = "") -> dict:
    out = {"cell": name, "status": status, "evidence": evidence}
    if detail:
        out["detail"] = detail
    return out


def _readiness_audit_receipt(
    root: Path,
    bundle_dir: Path,
    current_head: str,
    now: datetime,
    domain: str,
    required_tools: set[str],
) -> tuple[bool, str, str]:
    """Verify current-HEAD tool receipts used by standalone readiness cells."""
    receipt_path = bundle_dir / "audit_receipts" / f"{domain}_audit_receipt.json"
    evidence = str(receipt_path)
    try:
        receipt = _load_bounded_json(receipt_path)
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema_version",
            "generated_at_utc",
            "git_head",
            "domain",
            "scope_complete",
            "tools",
            "findings",
        }:
            raise ValueError("audit receipt has a noncanonical top-level contract")
        if (
            receipt.get("schema_version") != "gauntlet.audit_receipt.v1"
            or receipt.get("git_head") != current_head
            or receipt.get("domain") != domain
            or receipt.get("scope_complete") is not True
            or receipt.get("findings") != []
        ):
            raise ValueError("audit receipt is incomplete, stale by HEAD, or has findings")
        receipt_time = _parse_utc_timestamp(receipt.get("generated_at_utc"))
        age_reason = _age_reason(receipt_time, now, f"{domain} audit receipt")
        if age_reason:
            raise ValueError(age_reason)
        tools = receipt.get("tools")
        if not isinstance(tools, list):
            raise ValueError("audit receipt tools are missing or malformed")
        observed: set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict) or set(tool) != {
                "id",
                "version",
                "command",
                "output_path",
                "output_sha256",
                "result",
            }:
                raise ValueError("audit receipt contains a malformed tool binding")
            tool_id = tool.get("id")
            if tool_id not in required_tools:
                continue
            expected_command = CERTIFICATION_AUDIT_TOOLS.get(domain, {}).get(tool_id)
            if (
                not isinstance(tool_id, str)
                or tool_id in observed
                or not isinstance(tool.get("version"), str)
                or not tool.get("version")
                or tool.get("command") != expected_command
                or tool.get("result") != "pass"
            ):
                raise ValueError(f"audit tool binding is not a pass: {tool_id!r}")
            output_relative = tool.get("output_path")
            output_path = (
                _safe_repo_path(root, output_relative)
                if isinstance(output_relative, str)
                else None
            )
            try:
                inside_bundle = (
                    output_path is not None
                    and output_path.relative_to(bundle_dir.resolve()) is not None
                )
            except ValueError:
                inside_bundle = False
            if not inside_bundle:
                raise ValueError(f"audit tool output is outside the bundle: {tool_id}")
            output_bytes = _read_bounded_file(output_path)
            if hashlib.sha256(output_bytes).hexdigest() != tool.get("output_sha256"):
                raise ValueError(f"audit tool output hash mismatch: {tool_id}")
            output = _parse_json_bytes(output_bytes, label=str(output_path))
            if not isinstance(output, dict) or set(output) != {
                "schema_version",
                "generated_at_utc",
                "git_head",
                "domain",
                "tool_id",
                "command",
                "exit_code",
                "result",
            }:
                raise ValueError(f"audit tool output is noncanonical: {tool_id}")
            output_time = _parse_utc_timestamp(output.get("generated_at_utc"))
            output_age_reason = _age_reason(
                output_time, now, f"{domain}/{tool_id} audit output"
            )
            if (
                output_age_reason
                or output.get("schema_version") != "gauntlet.audit_tool_output.v1"
                or output.get("git_head") != current_head
                or output.get("domain") != domain
                or output.get("tool_id") != tool_id
                or output.get("command") != expected_command
                or output.get("exit_code") != 0
                or output.get("result") != "pass"
            ):
                raise ValueError(
                    output_age_reason or f"audit tool output is not a pass: {tool_id}"
                )
            observed.add(tool_id)
        missing = required_tools - observed
        if missing:
            raise ValueError(f"audit receipt omits required passing tools: {sorted(missing)}")
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        return False, evidence, str(error)
    return True, evidence, f"current-HEAD passing tools: {sorted(observed)}"


def _readiness_dist_receipt(
    root: Path, bundle_dir: Path, current_head: str, now: datetime
) -> tuple[bool, str, str]:
    receipt_path = bundle_dir / STRICT_BUNDLE_CLASSES["dist_matrix_receipt"][0]
    evidence = str(receipt_path)
    try:
        receipt = _load_bounded_json(receipt_path)
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema_version",
            "generated_at_utc",
            "git_head",
            "source_ci_run_id",
            "targets",
            "raw_evidence_paths",
            "raw_evidence_sha256s",
        }:
            raise ValueError("dist receipt has a noncanonical top-level contract")
        timestamp = _parse_utc_timestamp(receipt.get("generated_at_utc"))
        age_reason = _age_reason(timestamp, now, "dist matrix receipt")
        targets = receipt.get("targets")
        source_run = receipt.get("source_ci_run_id")
        raw_paths = receipt.get("raw_evidence_paths")
        raw_hashes = receipt.get("raw_evidence_sha256s")
        valid_raw_evidence = (
            isinstance(raw_paths, list)
            and raw_paths
            and len(raw_paths) == len(set(raw_paths))
            and isinstance(raw_hashes, dict)
            and set(raw_hashes) == set(raw_paths)
            and all(
                isinstance(relative, str)
                and (candidate := _safe_repo_path(root, relative)) is not None
                and candidate.is_file()
                and bundle_dir.resolve() in candidate.parents
                and re.fullmatch(r"[0-9a-f]{64}", str(raw_hashes.get(relative)))
                and _sha256_file(candidate) == raw_hashes[relative]
                for relative in raw_paths
            )
        )
        valid_targets = (
            isinstance(targets, dict)
            and set(targets) == set(CERTIFICATION_DIST_TARGETS)
            and all(
                isinstance(result, dict)
                and set(result)
                == {
                    "status",
                    "built",
                    "checksum_sidecar",
                    "smoke_test",
                    "portability",
                }
                and result.get("status") == "pass"
                and result.get("built") is True
                and result.get("checksum_sidecar") is True
                and result.get("smoke_test") == "pass"
                and not _dist_portability_reasons(
                    root, target, result.get("portability")
                )
                for target, result in targets.items()
            )
        )
        if (
            receipt.get("schema_version")
            != STRICT_BUNDLE_CLASSES["dist_matrix_receipt"][1]
            or receipt.get("git_head") != current_head
            or not isinstance(source_run, str)
            or not source_run
            or age_reason
            or not valid_targets
            or not valid_raw_evidence
        ):
            raise ValueError(
                age_reason or "dist receipt does not prove every current-HEAD target"
            )
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        return False, evidence, str(error)
    return True, evidence, f"current-HEAD dist run {source_run} passed every target"


def _copy_evidence_file(source: Path, destination: Path, maximum: int) -> str:
    """Copy one stable regular file without replacing different evidence."""
    source_identity = _stream_file_identity(source, maximum)
    if destination.exists():
        destination_identity = _stream_file_identity(destination, maximum)
        if destination_identity["sha256"] != source_identity["sha256"]:
            raise ValueError(f"refusing to replace different evidence: {destination}")
        return str(source_identity["sha256"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied = _stream_file_identity(destination, maximum)
    if copied["sha256"] != source_identity["sha256"]:
        raise ValueError(f"evidence copy hash mismatch: {destination}")
    return str(copied["sha256"])


def _write_workflow_evidence_manifest(
    root: Path,
    path: Path,
    *,
    workflow: str,
    workflow_path: str,
    artifact_name: str,
    run_id: str,
    files: list[dict],
) -> None:
    if re.fullmatch(r"[1-9][0-9]*", run_id) is None:
        raise ValueError("workflow evidence requires a numeric GitHub run id")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", artifact_name) is None:
        raise ValueError("workflow evidence artifact name is noncanonical")
    head = _git_output(root, "rev-parse", "HEAD")
    payload = {
        "schema_version": "gauntlet.workflow_evidence.v1",
        "generated_at_utc": _timestamp_text(datetime.now(timezone.utc)),
        "repository": CERTIFICATION_GITHUB_REPOSITORY,
        "workflow": workflow,
        "workflow_path": workflow_path,
        "event": {
            CERTIFICATION_MODEL_PARITY_WORKFLOW: CERTIFICATION_MODEL_PARITY_EVENT,
            CERTIFICATION_PERFORMANCE_WORKFLOW: CERTIFICATION_PERFORMANCE_EVENT,
        }.get(workflow, CERTIFICATION_GITHUB_EVENT),
        "run_id": run_id,
        "git_head": head,
        "artifact_name": artifact_name,
        "files": files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"workflow evidence manifest already exists: {path}")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def produce_model_parity_evidence(
    scorecard_name: str,
    raw_name: str,
    fixtures_name: str,
    artifact_name: str,
) -> int:
    """Package the physical evidence emitted by one real weighted ladder run."""
    root = _repo_root()
    head = _git_output(root, "rev-parse", "HEAD")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    scorecard_path = Path(scorecard_name).resolve()
    raw_path = Path(raw_name).resolve()
    fixtures_root = Path(fixtures_name).resolve()
    bundle_dir = (root / ".gauntlet-output/bundle").resolve()
    model_path = Path(os.environ.get("FOCR_MODEL_PATH", "")).resolve()
    try:
        scorecard_relative = scorecard_path.relative_to(root)
        raw_relative = raw_path.relative_to(root)
        scorecard_path.relative_to(bundle_dir)
        raw_path.relative_to(bundle_dir)
        scorecard = _load_bounded_json(scorecard_path)
        raw_text = _read_bounded_text(raw_path)
        model_identity = _stream_file_identity(model_path, PERF_MAX_SUBJECT_BINARY_BYTES)
        if (
            model_identity.get("sha256") != CURRENT_UNLIMITED_MODEL_SHA256
            or model_identity.get("size") != CURRENT_UNLIMITED_MODEL_SIZE
        ):
            raise ValueError("weighted ladder model does not match the release candidate")
        gates = scorecard.get("gates") if isinstance(scorecard, dict) else None
        gate_map = (
            {gate.get("gate"): gate for gate in gates if isinstance(gate, dict)}
            if isinstance(gates, list)
            else {}
        )
        if (
            not isinstance(scorecard, dict)
            or scorecard.get("schema") != "focr-ladder-scorecard/v1"
            or scorecard.get("all_green") is not True
            or scorecard.get("skipped_no_model") is not False
            or set(gate_map) != set(CERTIFICATION_MODEL_PARITY_MIN_ROWS)
            or any(
                gate_map[rung].get("outcome") != "pass"
                or gate_map[rung].get("meaningful") is not True
                or gate_map[rung].get("parity_rows") != rows
                for rung, rows in CERTIFICATION_MODEL_PARITY_MIN_ROWS.items()
            )
        ):
            raise ValueError("weighted ladder scorecard is not an exact armed L0-L5 pass")

        parity_events: dict[tuple[str, str], dict] = {}
        passing_results: set[str] = set()
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            event = _parse_json_bytes(stripped.encode("utf-8"), label=str(raw_path))
            if not isinstance(event, dict):
                raise ValueError("weighted ladder log contains a non-object JSON line")
            test = str(event.get("test", ""))
            match = re.match(r"^(L[0-5])(?:_|$)", test, re.IGNORECASE)
            if match is None:
                continue
            rung = match.group(1).upper()
            if event.get("event") in {"skip", "result"} and event.get("result") != "pass":
                raise ValueError("weighted ladder log contains a failed or skipped rung")
            if event.get("event") == "result":
                passing_results.add(rung)
                continue
            if event.get("event") != "parity":
                continue
            envelope = event.get("nondeterminism_envelope")
            doc = envelope.get("doc") if isinstance(envelope, dict) else None
            fixture = event.get("oracle_fixture")
            seam = str(fixture).split(".", 1)[0] if isinstance(fixture, str) else ""
            multi_page = test.lower().startswith("l5_multi_page")
            case = (
                f"{event.get('case')}:multi_page"
                if multi_page
                else f"{doc}:decoded_text"
                if rung == "L5"
                else f"{doc}:{seam}"
            )
            pair = (rung, case)
            identity_valid = (
                envelope.get("subject") == "f32 greedy"
                and envelope.get("oracle") == "bf16-cpu greedy (deterministic)"
                if multi_page and isinstance(envelope, dict)
                else isinstance(envelope, dict)
                and envelope.get("subject") == "franken_ocr"
                and envelope.get("oracle") == "unlimited-ocr-oracle"
            )
            if (
                event.get("schema_version") != 1
                or event.get("result") != "pass"
                or event.get("pass") is not True
                or not isinstance(envelope, dict)
                or not identity_valid
                or case not in CERTIFICATION_MODEL_PARITY_CASES[rung]
                or pair in parity_events
                or re.fullmatch(r"[0-9a-f]{64}", str(event.get("oracle_sha256")))
                is None
            ):
                raise ValueError(f"weighted ladder contains a malformed parity event: {pair}")
            parity_events[pair] = event
        expected_pairs = {
            (rung, case)
            for rung, cases in CERTIFICATION_MODEL_PARITY_CASES.items()
            for case in cases
        }
        if passing_results != set(CERTIFICATION_MODEL_PARITY_MIN_ROWS):
            raise ValueError("weighted ladder does not contain every passing terminal rung")
        if set(parity_events) != expected_pairs or "skip_no_model" in raw_text:
            raise ValueError("weighted ladder does not contain every exact L0-L5 case")

        copied_sources: dict[Path, tuple[str, str]] = {}
        oracle_bindings: list[dict] = []
        raw_paths = [scorecard_relative.as_posix(), raw_relative.as_posix()]
        for (rung, case), event in sorted(parity_events.items()):
            doc, seam = case.split(":", 1)
            if seam == "multi_page":
                source = root / str(event.get("oracle_fixture", ""))
                logical_sha = _stream_file_identity(
                    source, CERTIFICATION_MAX_ARTIFACT_BYTES
                )["sha256"]
                destination = bundle_dir / "model_parity/oracle/multi_page" / source.name
                if logical_sha != event.get("oracle_sha256"):
                    raise ValueError(f"multi-page oracle hash mismatch: {source}")
                file_sha = _copy_evidence_file(
                    source, destination, CERTIFICATION_MAX_ARTIFACT_BYTES
                )
                destination_relative = destination.relative_to(root).as_posix()
                if source not in copied_sources:
                    copied_sources[source] = (destination_relative, file_sha)
                    raw_paths.append(destination_relative)
                oracle_bindings.append(
                    {
                        "rung": rung,
                        "case": case,
                        "path": destination_relative,
                        "file_sha256": file_sha,
                        "oracle_sha256": logical_sha,
                    }
                )
                continue
            golden_source = fixtures_root / f"{doc}_reference.json"
            golden = _load_bounded_json(golden_source)
            provenance = golden.get("provenance") if isinstance(golden, dict) else None
            if (
                not isinstance(golden, dict)
                or not isinstance(provenance, dict)
                or provenance.get("hf_commit") != UNLIMITED_OCR_MODEL_COMMIT
                or provenance.get("pinned_torch") != "2.10.0"
                or provenance.get("pinned_transformers") != "4.57.1"
            ):
                raise ValueError(f"oracle golden has unpinned provenance: {golden_source}")
            if seam == "token_stream":
                token_stream = golden.get("token_stream")
                logical_sha = (
                    token_stream.get("generated_ids_sha256")
                    if isinstance(token_stream, dict)
                    else None
                )
                source = golden_source
            elif seam == "decoded_text":
                logical_sha = golden.get("decoded_text_sha256")
                source = golden_source
            else:
                activations = golden.get("activations")
                activation = (
                    activations.get(seam) if isinstance(activations, dict) else None
                )
                if not isinstance(activation, dict):
                    raise ValueError(f"oracle golden omits activation {doc}:{seam}")
                logical_sha = activation.get("sha256")
                source = fixtures_root / "activations" / doc / str(activation.get("file", ""))
                declared_file_sha = activation.get("file_sha256")
                actual_file_sha = _stream_file_identity(
                    source, PERF_MAX_SUBJECT_BINARY_BYTES
                )["sha256"]
                if declared_file_sha != actual_file_sha:
                    raise ValueError(f"oracle activation file hash mismatch: {source}")
            if logical_sha != event.get("oracle_sha256"):
                raise ValueError(f"oracle logical hash mismatch: {rung}/{case}")
            destination = bundle_dir / "model_parity/oracle" / doc / source.name
            file_sha = _copy_evidence_file(
                source, destination, PERF_MAX_SUBJECT_BINARY_BYTES
            )
            destination_relative = destination.relative_to(root).as_posix()
            previous = copied_sources.get(source)
            if previous is None:
                copied_sources[source] = (destination_relative, file_sha)
                raw_paths.append(destination_relative)
            elif previous != (destination_relative, file_sha):
                raise ValueError(f"oracle fixture destination collision: {source}")
            oracle_bindings.append(
                {
                    "rung": rung,
                    "case": case,
                    "path": destination_relative,
                    "file_sha256": file_sha,
                    "oracle_sha256": logical_sha,
                }
            )

        raw_paths = list(dict.fromkeys(raw_paths))
        raw_hashes = {
            relative: _stream_file_identity(
                root / relative, PERF_MAX_SUBJECT_BINARY_BYTES
            )["sha256"]
            for relative in raw_paths
        }
        generated_at = _timestamp_text(datetime.now(timezone.utc))
        receipt_path = bundle_dir / "model_parity_receipt.json"
        receipt = {
            "schema_version": STRICT_BUNDLE_CLASSES["model_parity_receipt"][1],
            "generated_at_utc": generated_at,
            "git_head": head,
            "source_ci_run_id": run_id,
            "model_commit": UNLIMITED_OCR_MODEL_COMMIT,
            "weighted_model_loaded": True,
            "skipped_no_model": False,
            "rungs": {rung: "pass" for rung in CERTIFICATION_MODEL_PARITY_MIN_ROWS},
            "scorecard_path": scorecard_relative.as_posix(),
            "scorecard_sha256": raw_hashes[scorecard_relative.as_posix()],
            "raw_log_path": raw_relative.as_posix(),
            "raw_log_sha256": raw_hashes[raw_relative.as_posix()],
            "oracle_fixture_bindings": oracle_bindings,
            "raw_evidence_paths": raw_paths,
            "raw_evidence_sha256s": raw_hashes,
        }
        if receipt_path.exists():
            raise ValueError(f"model parity receipt already exists: {receipt_path}")
        receipt_path.write_text(
            json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt_relative = receipt_path.relative_to(root).as_posix()
        workflow_files = [
            {
                "bundle_path": relative,
                "source_ci_artifact_path": relative.removeprefix(".gauntlet-output/"),
                "sha256": _stream_file_identity(
                    root / relative, PERF_MAX_SUBJECT_BINARY_BYTES
                )["sha256"],
            }
            for relative in [receipt_relative, *raw_paths]
        ]
        _write_workflow_evidence_manifest(
            root,
            root / ".gauntlet-output/model-parity-workflow-evidence.json",
            workflow=CERTIFICATION_MODEL_PARITY_WORKFLOW,
            workflow_path=CERTIFICATION_WORKFLOW_PATHS[
                CERTIFICATION_MODEL_PARITY_WORKFLOW
            ],
            artifact_name=artifact_name,
            run_id=run_id,
            files=workflow_files,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        emit("model-parity-evidence", False, reason=str(error))
        return 1
    emit("model-parity-evidence", True, receipt=str(receipt_path), files=len(raw_paths))
    return 0


def produce_performance_evidence(out_name: str, artifact_name: str) -> int:
    """Derive benchmark inputs from one complete real performance runbook tree."""
    root = _repo_root()
    head = _git_output(root, "rev-parse", "HEAD")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    source_root = Path(out_name).resolve()
    bundle_dir = (root / ".gauntlet-output/bundle").resolve()
    try:
        source_relative = source_root.relative_to(root).as_posix()
        if not source_relative.startswith("artifacts/perf/") or not source_root.is_dir():
            raise ValueError("performance evidence must be a real artifacts/perf subtree")
        candidates = sorted(source_root.glob("focr_*/focr_stages.json"))
        if not candidates:
            raise ValueError("performance run has no focr stage document")
        stage_document = _load_bounded_json(candidates[0])
        records = stage_document.get("stages") if isinstance(stage_document, dict) else None
        if not isinstance(records, list):
            raise ValueError("performance stage document has no stage records")
        by_name = {
            record.get("stage"): record
            for record in records
            if isinstance(record, dict) and isinstance(record.get("stage"), str)
        }
        required = {"vision_encode", "decode_per_token", "end_to_end"}
        if not required <= set(by_name):
            raise ValueError("performance stage document omits a strict benchmark stage")
        stages = {name: by_name[name] for name in sorted(required)}
        baseline_path = root / "benches/.bench-history/baseline.json"
        current = {
            "schema_version": "gauntlet.current_benchmark.v1",
            "generated_at_utc": _timestamp_text(datetime.now(timezone.utc)),
            "git_head": head,
            "baseline_sha256": _sha256_file(baseline_path),
            "stages": stages,
        }
        if _benchmark_metrics(current) is None:
            raise ValueError("performance stages cannot derive all strict benchmark metrics")
        current_path = bundle_dir / "benchmark_inputs/current_benchmark_samples.json"
        current_path.parent.mkdir(parents=True, exist_ok=True)
        if current_path.exists():
            raise ValueError(f"current benchmark evidence already exists: {current_path}")
        current_path.write_text(
            json.dumps(current, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        current_relative = current_path.relative_to(root).as_posix()
        files = [
            {
                "bundle_path": current_relative,
                "source_ci_artifact_path": current_relative,
                "sha256": _sha256_file(current_path),
            }
        ]
        _write_workflow_evidence_manifest(
            root,
            root / ".gauntlet-output/performance-workflow-evidence.json",
            workflow=CERTIFICATION_PERFORMANCE_WORKFLOW,
            workflow_path=CERTIFICATION_WORKFLOW_PATHS[
                CERTIFICATION_PERFORMANCE_WORKFLOW
            ],
            artifact_name=artifact_name,
            run_id=run_id,
            files=files,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        emit("performance-evidence", False, reason=str(error))
        return 1
    emit("performance-evidence", True, current=current_relative, files=len(files))
    return 0


def produce_ci_gate_evidence(job_name: str, log_name: str, artifact_name: str) -> int:
    """Bind one successful scripts/check.sh matrix job to its physical log."""
    root = _repo_root()
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    head = _git_output(root, "rev-parse", "HEAD")
    log_source = Path(log_name).resolve()
    safe_job = re.sub(r"[^A-Za-z0-9._-]+", "-", job_name).strip("-")
    try:
        if job_name not in CERTIFICATION_CI_REQUIRED_JOBS or not safe_job:
            raise ValueError(f"CI gate job is not canonical: {job_name!r}")
        log_destination = (
            root / ".gauntlet-output/bundle/ci" / f"{safe_job}.log"
        ).resolve()
        log_sha = _copy_evidence_file(
            log_source, log_destination, CERTIFICATION_MAX_ARTIFACT_BYTES
        )
        log_relative = log_destination.relative_to(root).as_posix()
        output_path = (
            root / ".gauntlet-output/bundle/ci" / f"{safe_job}.json"
        ).resolve()
        output = {
            "schema_version": "gauntlet.ci_job_output.v1",
            "generated_at_utc": _timestamp_text(datetime.now(timezone.utc)),
            "git_head": head,
            "source_ci_run_id": run_id,
            "job_name": job_name,
            "command": "scripts/check.sh",
            "exit_code": 0,
            "result": "pass",
            "log_path": log_relative,
            "log_sha256": log_sha,
        }
        if output_path.exists():
            raise ValueError(f"CI gate output already exists: {output_path}")
        output_path.write_text(
            json.dumps(output, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_relative = output_path.relative_to(root).as_posix()
        files = [
            {
                "bundle_path": relative,
                "source_ci_artifact_path": relative,
                "sha256": _sha256_file(root / relative),
            }
            for relative in (output_relative, log_relative)
        ]
        _write_workflow_evidence_manifest(
            root,
            root / ".gauntlet-output" / f"ci-{safe_job}-workflow-evidence.json",
            workflow=CERTIFICATION_GITHUB_WORKFLOW,
            workflow_path=CERTIFICATION_WORKFLOW_PATHS[CERTIFICATION_GITHUB_WORKFLOW],
            artifact_name=artifact_name,
            run_id=run_id,
            files=files,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        emit("ci-gate-evidence", False, reason=str(error))
        return 1
    emit("ci-gate-evidence", True, job=job_name, output=output_relative)
    return 0


def _numeric_version(value: str) -> tuple[int, ...] | None:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value) is None:
        return None
    return tuple(int(part) for part in value.split("."))


def _reported_focr_version(stdout: str) -> str | None:
    lines = stdout.splitlines()
    if not lines:
        return None
    match = re.fullmatch(
        r"focr ([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)",
        lines[0],
    )
    return match.group(1) if match is not None else None


def _dist_build_command(target_name: str) -> str | None:
    target = CERTIFICATION_DIST_TRIPLES.get(target_name)
    if target is None:
        return None
    if target_name in CERTIFICATION_LINUX_DIST_TARGETS:
        return (
            "cargo zigbuild --locked --release --bin focr --target "
            f"{target}.{CERTIFICATION_LINUX_GLIBC_FLOOR}"
        )
    return f"cargo build --locked --release --bin focr --target {target}"


def _dist_portability_reasons(
    root: Path,
    target_name: str,
    portability: object,
    entries: dict[str, dict] | None = None,
    expected_installer_sha256: str | None = None,
    expected_version: str | None = None,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(portability, dict):
        return [f"{target_name} has no portability proof"]
    if target_name in CERTIFICATION_LINUX_DIST_TARGETS:
        expected_fields = {
            "kind",
            "result",
            "supported_floor",
            "maximum_required",
            "required_versions",
            "raw_evidence_path",
            "raw_evidence_sha256",
        }
        versions = portability.get("required_versions")
        maximum = portability.get("maximum_required")
        raw_relative = portability.get("raw_evidence_path")
        raw_path = (
            _safe_repo_path(root, raw_relative)
            if isinstance(raw_relative, str)
            else None
        )
        try:
            raw_text = _read_bounded_text(raw_path) if raw_path is not None else ""
        except (OSError, ValueError):
            raw_text = ""
        observed = sorted(
            {
                match
                for match in re.findall(r"\bGLIBC_([0-9]+(?:\.[0-9]+)+)\b", raw_text)
                if _numeric_version(match) is not None
            },
            key=lambda value: _numeric_version(value) or (),
        )
        observed_maximum = observed[-1] if observed else None
        floor_tuple = _numeric_version(CERTIFICATION_LINUX_GLIBC_FLOOR)
        maximum_tuple = _numeric_version(str(maximum))
        if (
            set(portability) != expected_fields
            or portability.get("kind") != "glibc-symbol-floor"
            or portability.get("result") != "pass"
            or portability.get("supported_floor")
            != CERTIFICATION_LINUX_GLIBC_FLOOR
            or not isinstance(versions, list)
            or not versions
            or versions != observed
            or maximum != observed_maximum
            or maximum_tuple is None
            or floor_tuple is None
            or maximum_tuple > floor_tuple
            or raw_path is None
            or not raw_path.is_file()
            or _sha256_file(raw_path) != portability.get("raw_evidence_sha256")
            or (
                entries is not None
                and (
                    not isinstance(entries.get(raw_relative), dict)
                    or entries[raw_relative].get("sha256")
                    != portability.get("raw_evidence_sha256")
                )
            )
        ):
            reasons.append(f"{target_name} does not prove the glibc 2.17 ABI floor")
    elif target_name in CERTIFICATION_WINDOWS_DIST_TARGETS:
        expected_fields = {
            "kind",
            "result",
            "offline",
            "asset_sha256",
            "installed_sha256",
            "reported_version",
            "installer_sha256",
            "transcript_path",
            "transcript_sha256",
        }
        transcript_relative = portability.get("transcript_path")
        transcript_path = (
            _safe_repo_path(root, transcript_relative)
            if isinstance(transcript_relative, str)
            else None
        )
        if expected_installer_sha256 is None and (root / "install.ps1").is_file():
            expected_installer_sha256 = _sha256_file(root / "install.ps1")
        if expected_version is None and (root / "Cargo.toml").is_file():
            expected_version = _cargo_package_version(root)
        if (
            set(portability) != expected_fields
            or portability.get("kind") != "native-offline-install.ps1"
            or portability.get("result") != "pass"
            or portability.get("offline") is not True
            or re.fullmatch(r"[0-9a-f]{64}", str(portability.get("asset_sha256")))
            is None
            or portability.get("installed_sha256")
            != portability.get("asset_sha256")
            or re.fullmatch(
                r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?",
                str(portability.get("reported_version")),
            )
            is None
            or re.fullmatch(r"[0-9a-f]{64}", str(portability.get("installer_sha256")))
            is None
            or portability.get("installer_sha256") != expected_installer_sha256
            or portability.get("reported_version") != expected_version
            or transcript_path is None
            or not transcript_path.is_file()
            or _sha256_file(transcript_path) != portability.get("transcript_sha256")
            or (
                entries is not None
                and (
                    not isinstance(entries.get(transcript_relative), dict)
                    or entries[transcript_relative].get("sha256")
                    != portability.get("transcript_sha256")
                )
            )
        ):
            reasons.append(
                f"{target_name} does not prove native offline install.ps1 replacement"
            )
    elif portability != {"kind": "native-smoke", "result": "pass"}:
        reasons.append(f"{target_name} does not prove a native portable smoke run")
    return reasons


def _parse_tag_owned_dist_release_inputs(
    workflow_text: str, toolchain_channel: str
) -> dict[str, str]:
    """Read the immutable build-input pins recorded in a release tag's workflow."""
    inputs: dict[str, str] = {}
    for name in DIST_RELEASE_INPUT_NAMES:
        matches = re.findall(
            rf"(?m)^  {re.escape(name)}: ([^\s#]+)[ \t]*$", workflow_text
        )
        if len(matches) != 1:
            raise ValueError(
                "tag-owned dist workflow must define exactly one "
                f"top-level {name} release input"
            )
        inputs[name] = matches[0]
    if re.fullmatch(r"nightly-[0-9]{4}-[0-9]{2}-[0-9]{2}", inputs["RUST_TOOLCHAIN"]) is None:
        raise ValueError("tag-owned RUST_TOOLCHAIN is not an exact dated nightly")
    for name in DIST_RELEASE_INPUT_NAMES[1:]:
        if not _valid_git_head(inputs[name]):
            raise ValueError(f"tag-owned {name} is not a lowercase 40-hex revision")
    if toolchain_channel != inputs["RUST_TOOLCHAIN"]:
        raise ValueError(
            "tag-owned RUST_TOOLCHAIN disagrees with rust-toolchain.toml: "
            f"workflow={inputs['RUST_TOOLCHAIN']} toolchain={toolchain_channel}"
        )
    return inputs


def _tag_owned_dist_release_inputs(root: Path) -> dict[str, str]:
    workflow_path = root / CERTIFICATION_WORKFLOW_PATHS[CERTIFICATION_DIST_WORKFLOW]
    workflow_text = _read_bounded_text(workflow_path)
    try:
        with (root / "rust-toolchain.toml").open("rb") as handle:
            toolchain = tomllib.load(handle).get("toolchain")
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot read tag-owned rust-toolchain.toml: {error}") from error
    if not isinstance(toolchain, dict) or not isinstance(toolchain.get("channel"), str):
        raise ValueError("tag-owned rust-toolchain.toml has no string toolchain.channel")
    return _parse_tag_owned_dist_release_inputs(workflow_text, toolchain["channel"])


def _dist_release_input_drift(
    tag_inputs: dict[str, str], workflow_inputs: dict[str, str]
) -> list[str]:
    drift: list[str] = []
    for name in DIST_RELEASE_INPUT_NAMES:
        tag_value = tag_inputs[name]
        workflow_value = workflow_inputs.get(name) or "<unset>"
        if tag_value != workflow_value:
            drift.append(f"{name} tag={tag_value} workflow={workflow_value}")
    return drift


def dist_ref_preflight() -> int:
    """Reject release builds whose ref, version, or main ancestry is ambiguous."""
    root = _repo_root()
    try:
        event = os.environ.get("GITHUB_EVENT_NAME", "")
        ref_type = os.environ.get("GITHUB_REF_TYPE", "")
        ref_name = os.environ.get("GITHUB_REF_NAME", "")
        head = _git_output(root, "rev-parse", "HEAD")
        if event not in {"push", "workflow_dispatch"}:
            raise ValueError(f"dist cannot run for event {event!r}")
        fetch_args = ["git", "fetch"]
        if _git_output(root, "rev-parse", "--is-shallow-repository") == "true":
            fetch_args.append("--unshallow")
        fetch_args.extend(
            [
                "--no-tags",
                "origin",
                "+refs/heads/main:refs/remotes/origin/main",
            ]
        )
        fetch = subprocess.run(
            fetch_args,
            cwd=root,
            env=_git_env(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if fetch.returncode != 0:
            raise ValueError(f"cannot refresh origin/main: {fetch.stderr.strip()}")
        origin_main = _git_output(root, "rev-parse", "refs/remotes/origin/main")
        if ref_type == "branch" and ref_name == "main":
            if head != origin_main:
                raise ValueError("main dist build is not the current origin/main HEAD")
        elif ref_type == "tag":
            expected_tag = f"v{_cargo_package_version(root)}"
            if ref_name != expected_tag:
                raise ValueError(
                    f"release tag/version mismatch: expected {expected_tag}, got {ref_name}"
                )
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", head, origin_main],
                cwd=root,
                env=_git_env(),
                capture_output=True,
                timeout=30,
                check=False,
            )
            if ancestor.returncode != 0:
                raise ValueError("tagged dist commit is not reachable from origin/main")
            if event == "workflow_dispatch":
                tag_inputs = _tag_owned_dist_release_inputs(root)
                workflow_inputs = {
                    name: os.environ.get(name, "") for name in DIST_RELEASE_INPUT_NAMES
                }
                drift = _dist_release_input_drift(tag_inputs, workflow_inputs)
                if drift:
                    raise ValueError(
                        "source_ref repair release inputs differ from immutable tag: "
                        + "; ".join(drift)
                        + "; immutable-tag repair is unsupported across release-input drift"
                    )
        else:
            raise ValueError(
                "dist is restricted to refs/heads/main or an exact Cargo-version tag"
            )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        emit("dist-ref-preflight", False, reason=str(error))
        return 1
    emit(
        "dist-ref-preflight",
        True,
        event=event,
        ref_type=ref_type,
        ref_name=ref_name,
        git_head=head,
    )
    return 0


def produce_dist_target_evidence(
    target_name: str,
    asset_name: str,
    artifact_name: str,
    glibc_floor: str | None,
    installed_asset_name: str | None,
    installer_log_name: str | None,
) -> int:
    """Bind one successful portable dist job to its binary and checksum."""
    root = _repo_root()
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    head = _git_output(root, "rev-parse", "HEAD")
    expected_asset = CERTIFICATION_DIST_ASSETS.get(target_name)
    asset_source = Path(asset_name).resolve()
    checksum_source = asset_source.with_name(asset_source.name + ".sha256")
    safe_target = re.sub(r"[^A-Za-z0-9._-]+", "-", target_name).strip("-")
    try:
        if expected_asset is None or asset_source.name != expected_asset or not safe_target:
            raise ValueError(f"dist target/asset is not canonical: {target_name!r}")
        asset_identity = _stream_file_identity(
            asset_source, CERTIFICATION_MAX_ARTIFACT_BYTES
        )
        checksum_text = _read_bounded_text(checksum_source)
        if checksum_text != f"{asset_identity['sha256']}  {expected_asset}\n":
            raise ValueError(f"dist checksum sidecar does not bind {expected_asset}")
        asset_destination = (
            root / ".gauntlet-output/bundle/dist/assets" / expected_asset
        ).resolve()
        checksum_destination = asset_destination.with_name(expected_asset + ".sha256")
        asset_sha = _copy_evidence_file(
            asset_source, asset_destination, CERTIFICATION_MAX_ARTIFACT_BYTES
        )
        checksum_sha = _copy_evidence_file(
            checksum_source, checksum_destination, CERTIFICATION_MAX_ARTIFACT_BYTES
        )
        asset_relative = asset_destination.relative_to(root).as_posix()
        checksum_relative = checksum_destination.relative_to(root).as_posix()
        extra_files: list[str] = []
        if target_name in CERTIFICATION_LINUX_DIST_TARGETS:
            if glibc_floor != CERTIFICATION_LINUX_GLIBC_FLOOR:
                raise ValueError(
                    f"Linux dist target must be linked for glibc {CERTIFICATION_LINUX_GLIBC_FLOOR}"
                )
            readelf = shutil.which("readelf")
            if readelf is None:
                raise ValueError("readelf is required to certify a Linux dist asset")
            audit = subprocess.run(
                [readelf, "-W", "--version-info", "--dyn-syms", str(asset_source)],
                cwd=root,
                env={**os.environ, "LANG": "C", "LC_ALL": "C"},
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if audit.returncode != 0:
                raise ValueError(f"readelf failed for {expected_asset}: {audit.stderr.strip()}")
            versions = sorted(
                set(
                    re.findall(
                        r"\bGLIBC_([0-9]+(?:\.[0-9]+)+)\b",
                        audit.stdout,
                    )
                ),
                key=lambda value: _numeric_version(value) or (),
            )
            floor_tuple = _numeric_version(glibc_floor)
            maximum_tuple = _numeric_version(versions[-1]) if versions else None
            if (
                not versions
                or floor_tuple is None
                or maximum_tuple is None
                or maximum_tuple > floor_tuple
            ):
                raise ValueError(
                    f"{expected_asset} requires GLIBC_{versions[-1] if versions else '<none>'}, "
                    f"above the supported {glibc_floor} floor"
                )
            raw_path = (
                root / ".gauntlet-output/bundle/dist/portability" / f"{expected_asset}.readelf.txt"
            ).resolve()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if raw_path.exists():
                raise ValueError(f"glibc audit output already exists: {raw_path}")
            raw_path.write_text(audit.stdout, encoding="utf-8")
            raw_relative = raw_path.relative_to(root).as_posix()
            extra_files.append(raw_relative)
            portability = {
                "kind": "glibc-symbol-floor",
                "result": "pass",
                "supported_floor": glibc_floor,
                "maximum_required": versions[-1],
                "required_versions": versions,
                "raw_evidence_path": raw_relative,
                "raw_evidence_sha256": _sha256_file(raw_path),
            }
        elif target_name in CERTIFICATION_WINDOWS_DIST_TARGETS:
            if not installed_asset_name or not installer_log_name:
                raise ValueError(
                    "Windows dist target requires installed asset and offline installer transcript"
                )
            installed_path = Path(installed_asset_name).resolve()
            installer_log = Path(installer_log_name).resolve()
            installed_identity = _stream_file_identity(
                installed_path, CERTIFICATION_MAX_ARTIFACT_BYTES
            )
            if installed_identity["sha256"] != asset_sha:
                raise ValueError("offline install.ps1 output differs from the staged asset")
            transcript_destination = (
                root
                / ".gauntlet-output/bundle/dist/portability"
                / f"{expected_asset}.install.ps1.log"
            ).resolve()
            transcript_sha = _copy_evidence_file(
                installer_log,
                transcript_destination,
                CERTIFICATION_MAX_ARTIFACT_BYTES,
            )
            transcript_relative = transcript_destination.relative_to(root).as_posix()
            extra_files.append(transcript_relative)
            version_line = subprocess.run(
                [str(installed_path), "--version"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            reported_version = _reported_focr_version(version_line.stdout)
            if version_line.returncode != 0 or reported_version is None:
                raise ValueError("offline-installed Windows asset has no canonical version")
            portability = {
                "kind": "native-offline-install.ps1",
                "result": "pass",
                "offline": True,
                "asset_sha256": asset_sha,
                "installed_sha256": installed_identity["sha256"],
                "reported_version": reported_version,
                "installer_sha256": _sha256_file(root / "install.ps1"),
                "transcript_path": transcript_relative,
                "transcript_sha256": transcript_sha,
            }
        else:
            if glibc_floor or installed_asset_name or installer_log_name:
                raise ValueError("macOS dist target received inapplicable portability inputs")
            portability = {"kind": "native-smoke", "result": "pass"}
        portability_reasons = _dist_portability_reasons(root, target_name, portability)
        if portability_reasons:
            raise ValueError("; ".join(portability_reasons))
        output_path = (
            root / ".gauntlet-output/bundle/dist/raw" / f"{safe_target}.json"
        ).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "schema_version": "gauntlet.dist_target_output.v2",
            "generated_at_utc": _timestamp_text(datetime.now(timezone.utc)),
            "git_head": head,
            "source_ci_run_id": run_id,
            "target": target_name,
            "build_command": _dist_build_command(target_name),
            "built": True,
            "smoke_test": "pass",
            "asset_path": asset_relative,
            "asset_sha256": asset_sha,
            "checksum_path": checksum_relative,
            "checksum_sha256": checksum_sha,
            "portability": portability,
            "result": "pass",
        }
        if output_path.exists():
            raise ValueError(f"dist target output already exists: {output_path}")
        output_path.write_text(
            json.dumps(output, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_relative = output_path.relative_to(root).as_posix()
        files = [
            {
                "bundle_path": relative,
                "source_ci_artifact_path": relative,
                "sha256": _sha256_file(root / relative),
            }
            for relative in (
                output_relative,
                asset_relative,
                checksum_relative,
                *extra_files,
            )
        ]
        manifest_path = (
            root / ".gauntlet-output" / f"dist-{safe_target}-workflow-evidence.json"
        )
        _write_workflow_evidence_manifest(
            root,
            manifest_path,
            workflow=CERTIFICATION_DIST_WORKFLOW,
            workflow_path=CERTIFICATION_WORKFLOW_PATHS[CERTIFICATION_DIST_WORKFLOW],
            artifact_name=artifact_name,
            run_id=run_id,
            files=files,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        emit("dist-target-evidence", False, reason=str(error))
        return 1
    emit("dist-target-evidence", True, target=target_name, output=output_relative)
    return 0


def _github_run_summary(root: Path, run_id: str) -> dict:
    gh = shutil.which("gh")
    if gh is None or re.fullmatch(r"[1-9][0-9]*", run_id) is None:
        raise ValueError("GitHub run verification requires gh and a numeric run id")
    result = subprocess.run(
        [
            gh,
            "run",
            "view",
            run_id,
            "--repo",
            CERTIFICATION_GITHUB_REPOSITORY,
            "--json",
            "databaseId,headSha,headBranch,status,conclusion,workflowName,event,createdAt,updatedAt,jobs",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot inspect GitHub run {run_id}: {result.stderr.strip()}")
    payload = _parse_json_bytes(result.stdout.encode("utf-8"), label=f"run {run_id}")
    if not isinstance(payload, dict):
        raise ValueError(f"GitHub run {run_id} is not a JSON object")
    return payload


def _workflow_evidence_inputs(
    root: Path, manifest_names: Sequence[str], bundle_dir: Path, current_head: str
) -> tuple[list[dict], dict[str, str]]:
    """Verify downloaded workflow artifacts live, then copy their exact bytes."""
    bindings: list[dict] = []
    runs_by_workflow: dict[str, set[str]] = {}
    local_sources: dict[str, Path] = {}
    for manifest_name in manifest_names:
        manifest_path = Path(manifest_name).resolve()
        manifest = _load_bounded_json(manifest_path)
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "generated_at_utc",
            "repository",
            "workflow",
            "workflow_path",
            "event",
            "run_id",
            "git_head",
            "artifact_name",
            "files",
        }:
            raise ValueError(f"workflow evidence manifest is noncanonical: {manifest_path}")
        workflow = manifest.get("workflow")
        expected_event = {
            CERTIFICATION_MODEL_PARITY_WORKFLOW: CERTIFICATION_MODEL_PARITY_EVENT,
            CERTIFICATION_PERFORMANCE_WORKFLOW: CERTIFICATION_PERFORMANCE_EVENT,
        }.get(workflow, CERTIFICATION_GITHUB_EVENT)
        run_id = str(manifest.get("run_id", ""))
        files = manifest.get("files")
        if (
            manifest.get("schema_version") != "gauntlet.workflow_evidence.v1"
            or _age_reason(
                _parse_utc_timestamp(manifest.get("generated_at_utc")),
                datetime.now(timezone.utc),
                f"workflow evidence {manifest_path.name}",
            )
            or manifest.get("repository") != CERTIFICATION_GITHUB_REPOSITORY
            or workflow not in CERTIFICATION_WORKFLOW_PATHS
            or manifest.get("workflow_path") != CERTIFICATION_WORKFLOW_PATHS[workflow]
            or manifest.get("event") != expected_event
            or manifest.get("git_head") != current_head
            or re.fullmatch(r"[1-9][0-9]*", run_id) is None
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
                str(manifest.get("artifact_name", "")),
            )
            is None
            or not isinstance(files, list)
            or not files
        ):
            raise ValueError(f"workflow evidence manifest is stale or unpinned: {manifest_path}")
        source_paths: list[str] = []
        for item in files:
            if not isinstance(item, dict) or set(item) != {
                "bundle_path",
                "source_ci_artifact_path",
                "sha256",
            }:
                raise ValueError(f"workflow evidence file binding is malformed: {manifest_path}")
            source_relative = item.get("source_ci_artifact_path")
            bundle_relative = item.get("bundle_path")
            if (
                not isinstance(source_relative, str)
                or not isinstance(bundle_relative, str)
                or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256"))) is None
            ):
                raise ValueError(f"workflow evidence file binding is noncanonical: {item!r}")
            for relative in (source_relative, bundle_relative):
                relative_path = Path(relative)
                if (
                    relative_path.is_absolute()
                    or relative_path.as_posix() != relative
                    or any(part in {"", ".", ".."} for part in relative_path.parts)
                ):
                    raise ValueError(f"workflow evidence path is unsafe: {relative!r}")
            destination = _safe_repo_path(root, bundle_relative)
            try:
                inside_bundle = (
                    destination is not None
                    and destination.relative_to(bundle_dir) is not None
                )
            except ValueError:
                inside_bundle = False
            if (
                not inside_bundle
                or Path(bundle_relative).name
                in {
                    "release_certificate.json",
                    "certification_bundle.json",
                    "FINAL_GAUNTLET_REPORT.md",
                }
                or "signatures" in Path(bundle_relative).parts
            ):
                raise ValueError(f"workflow evidence destination is forbidden: {bundle_relative}")
            source_paths.append(source_relative)
        extraction_root = None
        for candidate in (manifest_path.parent, *manifest_path.parents):
            if all((candidate / relative).is_file() for relative in source_paths):
                extraction_root = candidate.resolve()
                break
        if extraction_root is None:
            raise ValueError(f"cannot locate downloaded artifact root for {manifest_path}")
        for item in files:
            source = (extraction_root / item["source_ci_artifact_path"]).resolve()
            try:
                source.relative_to(extraction_root)
            except ValueError as error:
                raise ValueError(f"downloaded workflow evidence escapes its artifact: {source}") from error
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"downloaded workflow evidence is not a regular file: {source}")
            identity = _stream_file_identity(source, PERF_MAX_SOURCE_PACK_BYTES)
            if identity["sha256"] != item["sha256"]:
                raise ValueError(f"downloaded workflow evidence hash mismatch: {source}")
            bundle_relative = item["bundle_path"]
            previous = local_sources.get(bundle_relative)
            if previous is not None and previous != source:
                raise ValueError(f"workflow evidence destination is duplicated: {bundle_relative}")
            local_sources[bundle_relative] = source
            bindings.append(
                {
                    "artifact": bundle_relative,
                    "sha256": item["sha256"],
                    "schema_version": CI_ARTIFACT_BINDING_SCHEMA,
                    "source_ci_run_id": run_id,
                    "source_ci_artifact_name": manifest["artifact_name"],
                    "source_ci_artifact_path": item["source_ci_artifact_path"],
                    "source_ci_repository": CERTIFICATION_GITHUB_REPOSITORY,
                    "source_ci_workflow": workflow,
                    "source_ci_workflow_path": CERTIFICATION_WORKFLOW_PATHS[workflow],
                    "source_ci_event": expected_event,
                }
            )
        runs_by_workflow.setdefault(str(workflow), set()).add(run_id)
    if set(runs_by_workflow) != set(CERTIFICATION_WORKFLOW_PATHS) or any(
        len(run_ids) != 1 for run_ids in runs_by_workflow.values()
    ):
        raise ValueError("workflow evidence must name exactly one run per canonical workflow")
    if len({item["artifact"] for item in bindings}) != len(bindings):
        raise ValueError("workflow evidence contains duplicate bundle destinations")
    if len(bindings) > CERTIFICATION_MAX_BUNDLE_ARTIFACTS:
        raise ValueError("workflow evidence exceeds the bundle artifact-count limit")
    for run_id in sorted({item["source_ci_run_id"] for item in bindings}):
        run_bindings = [
            item for item in bindings if item["source_ci_run_id"] == run_id
        ]
        if not _default_ci_run_verifier(root, run_id, current_head, run_bindings):
            raise ValueError(f"downloaded workflow evidence failed live replay: {run_id}")
    for bundle_relative, source in sorted(local_sources.items()):
        destination = _safe_repo_path(root, bundle_relative)
        if destination is None:
            raise ValueError(f"workflow destination became unsafe: {bundle_relative}")
        _copy_evidence_file(source, destination, PERF_MAX_SOURCE_PACK_BYTES)
    return bindings, {
        workflow: next(iter(run_ids)) for workflow, run_ids in runs_by_workflow.items()
    }


def _write_generated_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def finalize_bundle(
    out_name: str, manifest_names: Sequence[str], signer_specs: Sequence[str]
) -> int:
    """Derive, sign, and verify a strict certificate from physical CI evidence."""
    root = _repo_root()
    output_dir, output_reasons = _safe_generated_output_dir(root, out_name)
    if output_dir is None:
        emit("finalize-bundle", False, refusal_reasons=output_reasons)
        return 1
    certificate_path = output_dir / "release_certificate.json"
    bundle_path = output_dir / "certification_bundle.json"
    report_path = output_dir / "FINAL_GAUNTLET_REPORT.md"
    head = _git_output(root, "rev-parse", "HEAD")
    now = datetime.now(timezone.utc)
    try:
        if _git_output(root, "symbolic-ref", "--short", "HEAD") != "main":
            raise ValueError("finalization requires the main branch")
        if _git_worktree_state(root).get("clean") is not True:
            raise ValueError("finalization requires a clean worktree")
        provisional = _load_bounded_json(certificate_path)
        if not isinstance(provisional, dict) or set(provisional) != STRICT_CERTIFICATE_FIELDS:
            raise ValueError("run --bundle first to create a canonical provisional bundle")
        claim_sources = provisional.get("claim_sources")
        evidence_classes = provisional.get("evidence_classes")
        if not isinstance(claim_sources, dict) or not isinstance(evidence_classes, dict):
            raise ValueError("provisional bundle lacks canonical source mappings")
        bindings, run_ids = _workflow_evidence_inputs(
            root, manifest_names, output_dir, head
        )

        by_workflow: dict[str, list[dict]] = {}
        for binding in bindings:
            by_workflow.setdefault(binding["source_ci_workflow"], []).append(binding)

        ci_outputs: dict[str, tuple[str, dict]] = {}
        for binding in by_workflow[CERTIFICATION_GITHUB_WORKFLOW]:
            path = _safe_repo_path(root, binding["artifact"])
            try:
                payload = _load_bounded_json(path) if path is not None else None
            except (OSError, ValueError, TypeError, RecursionError):
                payload = None
            if not isinstance(payload, dict) or payload.get("schema_version") != "gauntlet.ci_job_output.v1":
                continue
            job_name = payload.get("job_name")
            log_relative = payload.get("log_path")
            log_entry = next(
                (
                    item
                    for item in by_workflow[CERTIFICATION_GITHUB_WORKFLOW]
                    if item["artifact"] == log_relative
                ),
                None,
            )
            if (
                job_name not in CERTIFICATION_CI_REQUIRED_JOBS
                or payload.get("git_head") != head
                or str(payload.get("source_ci_run_id", ""))
                != run_ids[CERTIFICATION_GITHUB_WORKFLOW]
                or payload.get("command") != "scripts/check.sh"
                or payload.get("exit_code") != 0
                or payload.get("result") != "pass"
                or not isinstance(log_entry, dict)
                or payload.get("log_sha256") != log_entry.get("sha256")
            ):
                raise ValueError(f"CI job output is not a physical pass: {binding['artifact']}")
            ci_outputs[str(job_name)] = (binding["artifact"], payload)
        if set(ci_outputs) != set(CERTIFICATION_CI_REQUIRED_JOBS):
            raise ValueError("CI evidence does not cover both required gate jobs")
        ci_raw_paths = list(
            dict.fromkeys(
                relative
                for relative, payload in ci_outputs.values()
                for relative in (relative, payload["log_path"])
            )
        )
        ci_raw_hashes = {
            relative: _sha256_file(root / relative) for relative in ci_raw_paths
        }
        ci_receipt = {
            "schema_version": STRICT_BUNDLE_CLASSES["ci_gate_receipt"][1],
            "generated_at_utc": _timestamp_text(now),
            "git_head": head,
            "source_ci_run_id": run_ids[CERTIFICATION_GITHUB_WORKFLOW],
            "suite_pass_rate_pct": 100.0,
            "jobs": {job: "pass" for job in CERTIFICATION_CI_REQUIRED_JOBS},
            "raw_evidence_paths": ci_raw_paths,
            "raw_evidence_sha256s": ci_raw_hashes,
        }
        _write_generated_json(output_dir / "ci_gate_receipt.json", ci_receipt)

        dist_outputs: dict[str, tuple[str, dict]] = {}
        dist_raw_paths: list[str] = []
        for binding in by_workflow[CERTIFICATION_DIST_WORKFLOW]:
            path = _safe_repo_path(root, binding["artifact"])
            try:
                payload = _load_bounded_json(path) if path is not None else None
            except (OSError, ValueError, TypeError, RecursionError):
                payload = None
            if (
                not isinstance(payload, dict)
                or set(payload)
                != {
                    "schema_version",
                    "generated_at_utc",
                    "git_head",
                    "source_ci_run_id",
                    "target",
                    "build_command",
                    "built",
                    "smoke_test",
                    "asset_path",
                    "asset_sha256",
                    "checksum_path",
                    "checksum_sha256",
                    "portability",
                    "result",
                }
                or payload.get("schema_version")
                != "gauntlet.dist_target_output.v2"
            ):
                continue
            target = payload.get("target")
            asset_relative = payload.get("asset_path")
            checksum_relative = payload.get("checksum_path")
            asset_binding = next(
                (item for item in by_workflow[CERTIFICATION_DIST_WORKFLOW] if item["artifact"] == asset_relative),
                None,
            )
            checksum_binding = next(
                (item for item in by_workflow[CERTIFICATION_DIST_WORKFLOW] if item["artifact"] == checksum_relative),
                None,
            )
            if (
                target not in CERTIFICATION_DIST_TARGETS
                or payload.get("git_head") != head
                or str(payload.get("source_ci_run_id", ""))
                != run_ids[CERTIFICATION_DIST_WORKFLOW]
                or payload.get("built") is not True
                or payload.get("build_command") != _dist_build_command(str(target))
                or payload.get("smoke_test") != "pass"
                or payload.get("result") != "pass"
                or not isinstance(asset_binding, dict)
                or not isinstance(checksum_binding, dict)
                or payload.get("asset_sha256") != asset_binding.get("sha256")
                or payload.get("checksum_sha256") != checksum_binding.get("sha256")
                or Path(str(asset_relative)).name != CERTIFICATION_DIST_ASSETS.get(target)
                or _dist_portability_reasons(
                    root, str(target), payload.get("portability")
                )
                or (
                    target in CERTIFICATION_WINDOWS_DIST_TARGETS
                    and payload.get("portability", {}).get("asset_sha256")
                    != payload.get("asset_sha256")
                )
            ):
                raise ValueError(f"dist target output is not a physical pass: {binding['artifact']}")
            checksum_text = _read_bounded_text(root / str(checksum_relative))
            if checksum_text != f"{payload['asset_sha256']}  {Path(str(asset_relative)).name}\n":
                raise ValueError(f"dist checksum content is invalid: {checksum_relative}")
            dist_outputs[str(target)] = (binding["artifact"], payload)
            dist_raw_paths.extend(
                [binding["artifact"], str(asset_relative), str(checksum_relative)]
            )
            portability = payload.get("portability", {})
            for key in ("raw_evidence_path", "transcript_path"):
                relative = portability.get(key)
                if isinstance(relative, str):
                    dist_raw_paths.append(relative)
        if set(dist_outputs) != set(CERTIFICATION_DIST_TARGETS):
            raise ValueError("dist evidence does not cover all six portable targets")
        dist_raw_paths = list(dict.fromkeys(dist_raw_paths))
        dist_receipt = {
            "schema_version": STRICT_BUNDLE_CLASSES["dist_matrix_receipt"][1],
            "generated_at_utc": _timestamp_text(now),
            "git_head": head,
            "source_ci_run_id": run_ids[CERTIFICATION_DIST_WORKFLOW],
            "targets": {
                target: {
                    "status": "pass",
                    "built": True,
                    "checksum_sidecar": True,
                    "smoke_test": "pass",
                    "portability": dist_outputs[target][1]["portability"],
                }
                for target in CERTIFICATION_DIST_TARGETS
            },
            "raw_evidence_paths": dist_raw_paths,
            "raw_evidence_sha256s": {
                relative: _sha256_file(root / relative)
                for relative in dist_raw_paths
            },
        }
        _write_generated_json(output_dir / "dist_matrix_receipt.json", dist_receipt)

        parity_path = output_dir / "model_parity_receipt.json"
        parity_receipt = _load_bounded_json(parity_path)
        if (
            not isinstance(parity_receipt, dict)
            or parity_receipt.get("source_ci_run_id")
            != run_ids[CERTIFICATION_MODEL_PARITY_WORKFLOW]
            or parity_receipt.get("git_head") != head
        ):
            raise ValueError("weighted model-parity receipt is missing or run-mismatched")

        current_candidates = [
            binding["artifact"]
            for binding in by_workflow[CERTIFICATION_PERFORMANCE_WORKFLOW]
            if Path(binding["artifact"]).name == "current_benchmark_samples.json"
        ]
        if len(current_candidates) != 1:
            raise ValueError("performance evidence must contain one current benchmark sample")
        current_relative = current_candidates[0]
        current_document = _load_bounded_json(root / current_relative)
        baseline_relative = claim_sources["core_evidence"][
            "benches/.bench-history/baseline.json"
        ]
        baseline_document = _load_bounded_json(root / baseline_relative)
        baseline_metrics = _benchmark_metrics(baseline_document)
        current_metrics = _benchmark_metrics(current_document)
        if baseline_metrics is None or current_metrics is None:
            raise ValueError("benchmark inputs cannot derive all five strict metrics")
        benchmark_summary = {
            "schema_version": STRICT_BUNDLE_CLASSES["benchmark_summary"][1],
            "generated_at_utc": _timestamp_text(now),
            "pass_over_pass_gates": {},
        }
        for name, minimum in CERTIFICATION_BENCHMARK_THRESHOLDS.items():
            regression = truncate_score(
                (baseline_metrics[name] - current_metrics[name])
                / baseline_metrics[name]
                * 100.0
            )
            benchmark_summary["pass_over_pass_gates"][name] = {
                "passed": regression >= minimum,
                "minimum_pct": minimum,
                "regression_pct": regression,
                "baseline_path": baseline_relative,
                "baseline_sha256": _sha256_file(root / baseline_relative),
                "current_path": current_relative,
                "current_sha256": _sha256_file(root / current_relative),
            }
        if not all(
            gate["passed"]
            for gate in benchmark_summary["pass_over_pass_gates"].values()
        ):
            raise ValueError("current performance evidence fails a pass-over-pass gate")
        _write_generated_json(output_dir / "benchmark_summary.json", benchmark_summary)

        audit_paths: dict[str, str] = {}
        audit_findings: list[dict] = []
        for domain in CERTIFICATION_AUDIT_DOMAINS:
            path = output_dir / "audit_receipts" / f"{domain}_audit_receipt.json"
            receipt = _load_bounded_json(path)
            if not isinstance(receipt, dict):
                raise ValueError(f"audit receipt is missing: {domain}")
            relative = path.relative_to(root).as_posix()
            audit_paths[domain] = relative
            findings = receipt.get("findings")
            if not isinstance(findings, list):
                raise ValueError(f"audit findings are malformed: {domain}")
            audit_findings.extend(findings)
        critical_inventory = {
            "schema_version": STRICT_BUNDLE_CLASSES["critical_path_inventory"][1],
            "generated_at_utc": _timestamp_text(now),
            "audits": {
                domain: {
                    "evidence_path": relative,
                    "evidence_sha256": _sha256_file(root / relative),
                }
                for domain, relative in audit_paths.items()
            },
        }
        _write_generated_json(
            output_dir / "critical_path_inventory.json", critical_inventory
        )
        open_count = sum(
            finding.get("status") == "open"
            for finding in audit_findings
            if isinstance(finding, dict)
        )
        waived_count = sum(
            finding.get("status") == "waived"
            for finding in audit_findings
            if isinstance(finding, dict)
        )
        critical_report = {
            "schema_version": STRICT_BUNDLE_CLASSES["critical_path_report"][1],
            "generated_at_utc": _timestamp_text(now),
            "open_high_critical": open_count,
            "waived_high_critical": waived_count,
            "findings": [
                finding
                for finding in audit_findings
                if isinstance(finding, dict) and finding.get("status") == "resolved"
            ],
        }
        _write_generated_json(output_dir / "critical_path_report.json", critical_report)
        if open_count or waived_count:
            raise ValueError("High/Critical audit findings remain open or waived")

        readiness_path = _safe_repo_path(root, claim_sources["release_readiness"])
        if readiness_path is None:
            raise ValueError("provisional readiness snapshot path is unsafe")
        release_readiness(
            str(readiness_path),
            bundle_dir=output_dir,
            now=now,
            evidence_path_mapping=claim_sources["readiness_evidence"],
        )
        readiness_document = _load_bounded_json(readiness_path)
        cells = readiness_document.get("cells") if isinstance(readiness_document, dict) else None
        if not isinstance(cells, list):
            raise ValueError("release-readiness snapshot is malformed")
        external_cells = [
            cell
            for cell in cells
            if isinstance(cell, dict) and cell.get("cell") != "certification_bundle"
        ]
        blocking = [cell["cell"] for cell in external_cells if cell.get("status") != "green"]
        if blocking:
            raise ValueError("external release-readiness cells are red: " + ", ".join(blocking))
        readiness_claim = {
            "green": len(external_cells),
            "red": 0,
            "blocking_cells": [],
            "ship": True,
        }

        release_scorecard = _load_bounded_json(
            root / claim_sources["core_evidence"]["docs/gauntlet/RELEASE_SCORECARD.json"]
        )
        parity_score = release_scorecard.get("parity_score_lower_bound")
        category_bounds = release_scorecard.get("per_category_lower")
        if (
            not _finite_number(parity_score, minimum=0.0, maximum=1.0)
            or not isinstance(category_bounds, dict)
            or not category_bounds
            or any(
                not _finite_number(value, minimum=0.0, maximum=1.0)
                for value in category_bounds.values()
            )
        ):
            raise ValueError("release scorecard lacks derived conformal lower bounds")
        scorecards = {
            "schema_version": STRICT_BUNDLE_CLASSES["scorecards"][1],
            "generated_at_utc": _timestamp_text(now),
            "parity_score_lower_bound": parity_score,
            "per_category_lower": category_bounds,
            "readiness_cells": cells,
        }
        _write_generated_json(output_dir / "scorecards.json", scorecards)
        ratchet = {
            "schema_version": STRICT_BUNDLE_CLASSES["ratchet_state"][1],
            "generated_at_utc": _timestamp_text(now),
            "previous_bound": parity_score,
            "current_lower_bound": parity_score,
            "commit_sha": head,
            "timestamp": _timestamp_text(now),
            "advance_reason": "final-HEAD conformal scorecard certification baseline",
            "per_category_bounds": category_bounds,
        }
        _write_generated_json(output_dir / "ratchet_state.json", ratchet)

        feature_path = output_dir / "feature_universe.json"
        feature_universe = {
            "schema_version": "gauntlet.feature_universe.v1",
            "generated_at_utc": _timestamp_text(now),
            "definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
            "features": list(CERTIFICATION_FEATURE_UNIVERSE),
        }
        _write_generated_json(feature_path, feature_universe)
        feature_relative = feature_path.relative_to(root).as_posix()
        verification_rows: list[dict] = []
        for feature in CERTIFICATION_FEATURE_UNIVERSE:
            for obligation in feature["proof_obligations"]:
                evidence_paths = [
                    claim_sources["readiness_evidence"][original]
                    for original in CERTIFICATION_READINESS_EVIDENCE_PATHS[obligation]
                ]
                verification_rows.append(
                    {
                        "feature_id": feature["feature_id"],
                        "proof_obligation": obligation,
                        "status": "pass",
                        "gate": "allowed",
                        "evidence_paths": evidence_paths,
                        "evidence_sha256s": {
                            relative: _sha256_file(root / relative)
                            for relative in evidence_paths
                        },
                    }
                )
        verification_contract = {
            "schema_version": STRICT_BUNDLE_CLASSES["verification_contract"][1],
            "generated_at_utc": _timestamp_text(now),
            "feature_universe_path": feature_relative,
            "feature_universe_sha256": _sha256_file(feature_path),
            "feature_universe_definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
            "rows": verification_rows,
        }
        _write_generated_json(output_dir / "verification_contract.json", verification_contract)
        confidence = {
            "schema_version": STRICT_BUNDLE_CLASSES["confidence_gate"][1],
            "generated_at_utc": _timestamp_text(now),
            "release_decision": "Allow",
            "min_verification_pct_observed": 100.0,
            "required_suite_pass_rate_pct_observed": 100.0,
            "high_severity_counterexample_count": 0,
            "constants_enforced": list(CERTIFICATION_CONSTANTS),
        }
        _write_generated_json(output_dir / "confidence_gate.json", confidence)
        ci_manifest = {
            "schema_version": STRICT_BUNDLE_CLASSES["ci_manifest"][1],
            "generated_at_utc": _timestamp_text(now),
            "repository": CERTIFICATION_GITHUB_REPOSITORY,
            "required_workflows": {
                CERTIFICATION_GITHUB_WORKFLOW: CERTIFICATION_GITHUB_EVENT,
                CERTIFICATION_DIST_WORKFLOW: CERTIFICATION_GITHUB_EVENT,
                CERTIFICATION_MODEL_PARITY_WORKFLOW: CERTIFICATION_MODEL_PARITY_EVENT,
                CERTIFICATION_PERFORMANCE_WORKFLOW: CERTIFICATION_PERFORMANCE_EVENT,
            },
            "artifacts": bindings,
        }
        _write_generated_json(output_dir / "ci_manifest.json", ci_manifest)

        excluded = {
            certificate_path.resolve(),
            bundle_path.resolve(),
            report_path.resolve(),
        }
        evidence_paths = sorted(
            path.relative_to(root).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
            and not path.name.startswith("._")
            and path.resolve() not in excluded
            and "signatures" not in path.relative_to(output_dir).parts
        )
        if len(evidence_paths) > CERTIFICATION_MAX_BUNDLE_ARTIFACTS:
            raise ValueError("final evidence exceeds the bundle artifact-count limit")
        static_sources = {
            relative
            for original, relative in {
                **claim_sources["core_evidence"],
                **claim_sources["readiness_evidence"],
            }.items()
            if original
            not in {
                "docs/gauntlet/RELEASE_READINESS.json",
                "docs/gauntlet/ROUNDS.jsonl",
            }
        }
        source_pack_paths = {
            relative
            for relative in evidence_paths
            if _is_canonical_source_pack_artifact(relative)
        }
        streamed_paths = {
            relative
            for relative in evidence_paths
            if (root / relative).stat().st_size > CERTIFICATION_MAX_ARTIFACT_BYTES
        }
        manifest, stale = build_evidence_manifest(
            root,
            now.timestamp(),
            evidence_paths=evidence_paths,
            static_source_paths=static_sources,
            source_pack_paths=source_pack_paths,
            streamed_artifact_paths=streamed_paths,
        )
        if stale or any("error" in entry for entry in manifest):
            raise ValueError("final evidence is stale or unreadable: " + ", ".join(stale))
        bundle_root = _bundle_root_sha256(manifest)
        if bundle_root is None:
            raise ValueError("final evidence manifest has no canonical root")
        max_age = max(
            [0.0, *(float(entry["age_hours"]) for entry in manifest if "age_hours" in entry)]
        )

        trusted = _trusted_signer_fingerprints(root)
        signer_records: list[tuple[str, str, str]] = []
        for spec in signer_specs:
            try:
                role, identity, fingerprint = spec.split(":", 2)
            except ValueError as error:
                raise ValueError(
                    "signers must use ROLE:IDENTITY:FINGERPRINT"
                ) from error
            fingerprint = fingerprint.upper()
            if trusted.get(identity) != fingerprint:
                raise ValueError(f"signer is not active and fingerprint-pinned: {identity}")
            signer_records.append((role, identity, fingerprint))
        if (
            len(signer_records) != CERTIFICATION_REQUIRED_SIGNERS
            or {role for role, _identity, _fingerprint in signer_records}
            != CERTIFICATION_REQUIRED_SIGNATURE_ROLES
            or len({identity for _role, identity, _fingerprint in signer_records})
            != CERTIFICATION_REQUIRED_SIGNERS
            or len({fingerprint for _role, _identity, fingerprint in signer_records})
            != CERTIFICATION_REQUIRED_SIGNERS
        ):
            raise ValueError("finalization requires three distinct trusted signer roles")
        signature_dir = output_dir / "signatures"
        if signature_dir.exists() and any(signature_dir.iterdir()):
            raise ValueError("signature output directory is not empty")
        signature_dir.mkdir(parents=True, exist_ok=True)
        signature_declarations = [
            {
                "signer": identity,
                "role": role,
                "fingerprint": fingerprint,
                "scheme": "openpgp-detached",
                "signature_path": (
                    signature_dir / f"{role}.asc"
                ).relative_to(root).as_posix(),
            }
            for role, identity, fingerprint in signer_records
        ]
        rounds_path = _safe_repo_path(root, claim_sources["rounds"])
        rounds = [
            _parse_json_bytes(line.encode("utf-8"), label=str(rounds_path))
            for line in _read_bounded_text(rounds_path).splitlines()
            if line.strip()
        ]
        ledger_contents = {
            relative: _read_bounded_text(root / relative)
            for relative in claim_sources["hypothesis_ledgers"]
        }
        hypotheses = hypothesis_texts_verdict(
            ledger_contents, claim_sources["hypothesis_ledgers"], output_dir
        )
        convergence_claim = convergence_verdict(rounds, hypotheses)
        certificate = {
            **provisional,
            "version": _cargo_package_version(root),
            "issued_at": _timestamp_text(now),
            "git_head": head,
            "evidence_git_head": head,
            "git_branch": "main",
            "git_describe": _git_output(root, "describe", "--tags", "--always", "--dirty"),
            "git_worktree": _git_worktree_state(root),
            "readiness": readiness_claim,
            "convergence": convergence_claim,
            "required_pass_actuals": {
                "min_verification_pct_observed": 100.0,
                "required_suite_pass_rate_pct_observed": 100.0,
                "high_severity_counterexample_count": 0,
                "max_evidence_age_hours_observed": max_age,
            },
            "high_severity_counterexamples": 0,
            "parity_score": parity_score,
            "feature_universe_sha256": _sha256_file(feature_path),
            "feature_universe_definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
            "evidence_bundle_sha256": bundle_root,
            "signers": [identity for _role, identity, _fingerprint in signer_records],
            "detached_signatures": signature_declarations,
            "certified": True,
            "refusal_reasons": [],
            "generated_by": "scripts/gauntlet_cert.py --finalize-bundle",
        }
        certificate["signed_claim_sha256"] = _certificate_signed_claim_sha256(
            certificate, bundle_root
        )
        signed_claim = certificate["signed_claim_sha256"]
        if not isinstance(signed_claim, str):
            raise ValueError("final signed claim cannot be canonicalized")
        gpg = shutil.which("gpg")
        if gpg is None:
            raise ValueError("gpg is required to use existing trusted signer keys")
        with tempfile.TemporaryDirectory(prefix="focr-final-signatures-") as tmp:
            payload_path = Path(tmp) / "signed-claim.txt"
            payload_path.write_text(signed_claim + "\n", encoding="utf-8")
            generated_signatures: list[tuple[Path, Path]] = []
            for declaration in signature_declarations:
                temporary = Path(tmp) / f"{declaration['role']}.asc"
                result = subprocess.run(
                    [
                        gpg,
                        "--batch",
                        "--yes",
                        "--armor",
                        "--detach-sign",
                        "--local-user",
                        declaration["fingerprint"],
                        "--output",
                        str(temporary),
                        str(payload_path),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode != 0:
                    raise ValueError(
                        f"trusted signer failed: {declaration['signer']}: {result.stderr.strip()}"
                    )
                destination = root / declaration["signature_path"]
                generated_signatures.append((temporary, destination))
            for temporary, destination in generated_signatures:
                _copy_evidence_file(
                    temporary, destination, CERTIFICATION_MAX_ARTIFACT_BYTES
                )
        bundle = {
            "schema_version": STRICT_BUNDLE_SCHEMA,
            "artifact": STRICT_BUNDLE_ARTIFACT,
            "generated_at_utc": certificate["issued_at"],
            "bundle_root_sha256": bundle_root,
            "manifest": manifest,
        }
        certificate_bytes = (json.dumps(certificate, indent=1, sort_keys=True) + "\n").encode("utf-8")
        bundle_bytes = (json.dumps(bundle, indent=1, sort_keys=True) + "\n").encode("utf-8")
        report_bytes = _final_report_text(certificate, bundle).encode("utf-8")
        candidate_verdict = certificate_bundle_verdict(
            root,
            head,
            bundle_dir=output_dir,
            now=now,
            control_documents=(certificate_bytes, bundle_bytes, report_bytes),
        )
        if not candidate_verdict["ok"]:
            raise ValueError(
                "candidate certificate failed strict verification: "
                + "; ".join(candidate_verdict["reasons"])
            )
        certificate_path.write_bytes(certificate_bytes)
        bundle_path.write_bytes(bundle_bytes)
        report_path.write_bytes(report_bytes)
        final_verdict = certificate_bundle_verdict(
            root, head, bundle_dir=output_dir, now=now
        )
        if not final_verdict["ok"]:
            refused = {
                **certificate,
                "certified": False,
                "refusal_reasons": final_verdict["reasons"],
            }
            refused["signed_claim_sha256"] = _certificate_signed_claim_sha256(
                refused, bundle_root
            )
            _write_generated_json(certificate_path, refused)
            report_path.write_text(_final_report_text(refused, bundle), encoding="utf-8")
            raise ValueError(
                "persisted certificate failed strict re-verification: "
                + "; ".join(final_verdict["reasons"])
            )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        RecursionError,
        subprocess.SubprocessError,
    ) as error:
        emit("finalize-bundle", False, refusal_reasons=[str(error)])
        return 1
    emit(
        "finalize-bundle",
        True,
        certified=True,
        out_dir=out_name,
        bundle_root_sha256=bundle_root,
        signed_claim_sha256=signed_claim,
    )
    return 0


def release_readiness(
    out_path: str | None,
    *,
    bundle_dir: Path | None = None,
    now: datetime | None = None,
    worktree_state: dict | None = None,
    evidence_path_mapping: dict[str, str] | None = None,
) -> int:
    root = _repo_root()
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    current_head = _git_output(root, "rev-parse", "HEAD")
    requested_bundle = bundle_dir or root / ".gauntlet-output/bundle"

    def p(rel: str) -> str:
        return str(root / rel)

    def load_json(rel: str):
        return _load_bounded_json(Path(p(rel)))

    cells: list[dict] = []

    # Parity (L0-L5): the committed armed ladder receipt must be ALL GREEN
    # and not a skipped run wearing green.
    try:
        sc = load_json("tests/fixtures/ladder_scorecard/scorecard_armed.json")
        if not isinstance(sc, dict):
            raise ValueError("armed ladder receipt is not a JSON object")
        ok = sc.get("all_green") is True and sc.get("skipped_no_model") is False
        cells.append(
            _cell(
                "parity_l0_l5",
                "green" if ok else "red",
                "tests/fixtures/ladder_scorecard/scorecard_armed.json",
                sc.get("receipt", ""),
            )
        )
    except (OSError, TypeError, ValueError, RecursionError) as e:
        cells.append(
            _cell("parity_l0_l5", "red", "MISSING armed ladder receipt", str(e))
        )

    # Surface parity: every MUST row must be present. Partial/excluded rows are
    # useful debt accounting, but cannot satisfy a strict release certificate.
    try:
        md = _read_bounded_text(Path(p("docs/FEATURE_PARITY.md")))
        surface = surface_must_verdict(md)
        debt = [f"{item['surface']}={item['status']}" for item in surface["debt"]]
        cells.append(
            _cell(
                "surface_parity",
                "green" if surface["ok"] else "red",
                "docs/FEATURE_PARITY.md §12-§15 (lock: tests/surface_matrix.rs)",
                f"MUST rows={surface['must_rows']}; non-present debt: {debt or 'none'}",
            )
        )
    except (OSError, TypeError, ValueError, RecursionError) as e:
        cells.append(
            _cell("surface_parity", "red", "MISSING FEATURE_PARITY.md", str(e))
        )

    # Honest perf vs reference: require a fresh, HEAD-matched, release-perf,
    # cv-eligible Unlimited-OCR decode row whose evidence manifest re-verifies.
    ledger_rc = subprocess.run(
        [sys.executable, p("scripts/check_ledgers.py")],
        capture_output=True,
        check=False,
    ).returncode
    try:
        perf = _read_bounded_text(Path(p("docs/PERF_LEDGER.md")))
        perf_verdict = perf_evidence_verdict(perf, root, current_head)
        ok = ledger_rc == 0 and perf_verdict["ok"]
        rejected = [
            f"{candidate['claim_id']}: {', '.join(candidate['reasons'])}"
            for candidate in perf_verdict["candidates"]
            if not candidate["eligible"]
        ]
        cells.append(
            _cell(
                "perf_vs_reference",
                "green" if ok else "red",
                "docs/PERF_LEDGER.md + scripts/check_ledgers.py",
                f"check_ledgers exit {ledger_rc}; {perf_verdict['reason']}; "
                f"rejected candidates: {rejected or 'none'}",
            )
        )
    except (OSError, TypeError, ValueError, RecursionError) as e:
        cells.append(
            _cell("perf_vs_reference", "red", "MISSING PERF_LEDGER.md", str(e))
        )

    # Determinism: the e-process state must show the invariant OBSERVED and
    # never rejected (the live monitor over the determinism gates).
    try:
        ep = load_json("docs/gauntlet/EPROCESS_STATE.json")
        if not isinstance(ep, dict) or not isinstance(ep.get("invariants"), dict):
            raise ValueError("e-process state is not a canonical JSON object")
        det = ep["invariants"]["INV-DETERMINISM"]
        if not isinstance(det, dict):
            raise ValueError("determinism invariant is not a JSON object")
        ok = (
            det["obs_count"] > 0
            and det["rejected_at"] is None
            and not ep["any_rejected"]
        )
        cells.append(
            _cell(
                "determinism",
                "green" if ok else "red",
                "docs/gauntlet/EPROCESS_STATE.json (INV-DETERMINISM)",
                f"obs={det['obs_count']} e={det['e_value']:.3g} rejected={det['rejected_at']}",
            )
        )
    except (OSError, KeyError, TypeError, ValueError, RecursionError) as e:
        cells.append(
            _cell("determinism", "red", "MISSING/incomplete e-process state", str(e))
        )

    # Current-HEAD structured tool outputs, hash-bound by their audit receipt.
    watchdog_ok, watchdog_evidence, watchdog_detail = _readiness_audit_receipt(
        root,
        requested_bundle,
        current_head,
        now,
        "concurrency",
        {"many-pages-watchdog", "cancel-panic-faults"},
    )
    cells.append(
        _cell(
            "deadlock_watchdog",
            "green" if watchdog_ok else "red",
            watchdog_evidence,
            watchdog_detail,
        )
    )

    full_suite_ok, full_suite_evidence, full_suite_detail = _readiness_audit_receipt(
        root,
        requested_bundle,
        current_head,
        now,
        "release",
        {"full-check"},
    )
    cells.append(
        _cell(
            "robot_schema",
            "green" if full_suite_ok else "red",
            full_suite_evidence,
            full_suite_detail,
        )
    )

    # Distribution claims require a fresh current-HEAD matrix receipt. Historical
    # release prose cannot certify the tree being evaluated.
    build_ok, build_evidence, build_detail = _readiness_dist_receipt(
        root, requested_bundle, current_head, now
    )
    installer_ok = all(
        os.path.exists(p(relative))
        for relative in CERTIFICATION_READINESS_EVIDENCE_PATHS["installer"]
    )
    cells.append(
        _cell(
            "build_matrix",
            "green" if build_ok else "red",
            build_evidence,
            build_detail,
        )
    )
    cells.append(
        _cell(
            "installer",
            "green" if installer_ok else "red",
            "install.sh + install.ps1 + tests/installer_e2e.sh (+ published checksums)",
        )
    )

    # Ledger completeness: the checker IS the verification.
    cells.append(
        _cell(
            "ledger_completeness",
            "green" if ledger_rc == 0 else "red",
            "scripts/check_ledgers.py over DISCREPANCIES.md + NEGATIVE_EVIDENCE.md + PERF_LEDGER.md",
            f"exit {ledger_rc}",
        )
    )

    hypotheses = hypothesis_ledger_verdict(root)
    cells.append(
        _cell(
            "hypothesis_ledgers",
            "green" if hypotheses["resolved"] else "red",
            ", ".join(HYPOTHESIS_LEDGER_PATHS),
            f"missing={hypotheses['missing']}; unresolved={hypotheses['unresolved']}; "
            f"entries={hypotheses['entries']}",
        )
    )

    cells.append(
        _cell(
            "agent_ergonomics",
            "green" if full_suite_ok else "red",
            full_suite_evidence,
            full_suite_detail,
        )
    )
    cells.append(
        _cell(
            "doctor",
            "green" if full_suite_ok else "red",
            full_suite_evidence,
            full_suite_detail,
        )
    )
    # A prior `certified:true` is not evidence for a changed tree. Re-verify the
    # certificate HEAD and every companion-manifest hash before turning green.
    bundle_verdict = certificate_bundle_verdict(
        root,
        current_head,
        bundle_dir=requested_bundle,
        now=now,
        worktree_state=worktree_state,
    )
    try:
        bundle_evidence = str(requested_bundle.resolve().relative_to(root.resolve()))
    except ValueError:
        bundle_evidence = str(requested_bundle)
    cells.append(
        _cell(
            "certification_bundle",
            "green" if bundle_verdict["ok"] else "red",
            f"{bundle_evidence}/release_certificate.json (bd-wp8.9)",
            "verified for current HEAD"
            if bundle_verdict["ok"]
            else "; ".join(bundle_verdict["reasons"]),
        )
    )
    rounds_path = Path(p("docs/gauntlet/ROUNDS.jsonl"))
    rounds: list[dict] = []
    round_error = ""
    if rounds_path.exists():
        try:
            rounds = [
                parsed
                for line_number, line in enumerate(
                    _read_bounded_text(rounds_path).splitlines(), start=1
                )
                if line.strip()
                for parsed in [
                    _parse_json_bytes(
                        line.encode("utf-8"),
                        label=f"{rounds_path}:{line_number}",
                    )
                ]
            ]
            if any(not isinstance(record, dict) for record in rounds):
                raise ValueError("round history contains a non-object row")
        except (OSError, TypeError, ValueError, RecursionError) as error:
            rounds = []
            round_error = str(error)
    conv = convergence_verdict(rounds, hypotheses)
    cells.append(
        _cell(
            "gauntlet_convergence",
            "green" if conv["converged"] else "red",
            "docs/gauntlet/ROUNDS.jsonl (bd-wp8.8)",
            f"rounds={conv['rounds']}/{MIN_ROUNDS}, tail_clean={conv['tail_clean']}, "
            f"hypotheses_resolved={conv['hypotheses_resolved']}"
            + (f"; input_error={round_error}" if round_error else ""),
        )
    )

    observed_cell_order = tuple(cell.get("cell") for cell in cells)
    if observed_cell_order != CERTIFICATION_READINESS_CELLS:
        raise RuntimeError(
            "release-readiness implementation diverged from the canonical cell set"
        )
    proof_mapping = evidence_path_mapping or {
        path: path for path in CERTIFICATION_PROOF_EVIDENCE_PATHS
    }
    for cell in cells:
        originals = CERTIFICATION_READINESS_EVIDENCE_PATHS.get(cell["cell"], ())
        cell["evidence_paths"] = [proof_mapping.get(path, path) for path in originals]

    def render_artifact() -> tuple[list[str], str]:
        blocking = [cell["cell"] for cell in cells if cell["status"] == "red"]
        artifact = {
            "schema_version": "gauntlet.release_readiness.v1",
            "artifact": "franken_ocr.release_readiness.v1",
            "cell_set_sha256": CERTIFICATION_READINESS_CELL_SET_SHA256,
            "generated_at_utc": _timestamp_text(now),
            "generated_by": "scripts/gauntlet_cert.py --release-readiness",
            "cells": cells,
            "green": sum(1 for cell in cells if cell["status"] == "green"),
            "red": len(blocking),
            "blocking_cells": blocking,
            "ship": not blocking,
        }
        return blocking, json.dumps(artifact, indent=1, sort_keys=True)

    reds, text = render_artifact()
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        if not reds and worktree_state is None:
            post_head = _git_output(root, "rev-parse", "HEAD")
            post_worktree = _git_worktree_state(root)
            if post_head != current_head or post_worktree.get("clean") is not True:
                certification_cell = next(
                    cell for cell in cells if cell["cell"] == "certification_bundle"
                )
                certification_cell["status"] = "red"
                certification_cell["detail"] = (
                    "repository state changed while writing the readiness artifact"
                )
                reds, text = render_artifact()
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text + "\n")
    print(text)
    return 0 if not reds else 1


def produce_bundle(out_dir: str) -> int:
    """Assemble an unsigned strict bundle and verify that exact output directory.

    Generation never invents release signatures or missing evidence classes.
    The emitted certificate therefore stays red until the external signing
    flow supplies three independently verified detached OpenPGP signatures.
    """
    root = _repo_root()
    output_dir, output_reasons = _safe_generated_output_dir(root, out_dir)
    if output_dir is None:
        emit(
            "bundle",
            False,
            out_dir=out_dir,
            certified=False,
            refusal_reasons=output_reasons,
        )
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    generated_at = _timestamp_text(now)
    head = _git_output(root, "rev-parse", "HEAD")
    package_version = _cargo_package_version(root)

    def write_json(name: str, payload: dict) -> None:
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")

    def relative_output(name: str) -> str:
        return str((output_dir / name).relative_to(root.resolve()))

    claim_source_dir = output_dir / "claim_sources"
    claim_source_dir.mkdir(parents=True, exist_ok=True)
    readiness_snapshot = claim_source_dir / "RELEASE_READINESS.json"
    rounds_snapshot = claim_source_dir / "ROUNDS.jsonl"
    ledger_snapshots = {
        original: claim_source_dir / Path(original).name
        for original in HYPOTHESIS_LEDGER_PATHS
    }
    core_snapshot_paths: dict[str, Path] = {}
    for original in CORE_EVIDENCE_PATHS:
        if original == "docs/gauntlet/RELEASE_READINESS.json":
            destination = readiness_snapshot
        elif original == "docs/gauntlet/ROUNDS.jsonl":
            destination = rounds_snapshot
        elif original in ledger_snapshots:
            destination = ledger_snapshots[original]
        else:
            destination = output_dir / "source_evidence" / original
        core_snapshot_paths[original] = destination
        if original == "docs/gauntlet/RELEASE_READINESS.json":
            continue
        source = _safe_repo_path(root, original)
        if source is not None and source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    proof_snapshot_paths: dict[str, Path] = {}
    for original in CERTIFICATION_PROOF_EVIDENCE_PATHS:
        destination = core_snapshot_paths.get(
            original, output_dir / "source_evidence" / original
        )
        proof_snapshot_paths[original] = destination
        if original in core_snapshot_paths:
            continue
        source = _safe_repo_path(root, original)
        if source is not None and source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    live_hypothesis_contents: dict[str, str | None] = {}
    for original in HYPOTHESIS_LEDGER_PATHS:
        source = _safe_repo_path(root, original)
        try:
            live_hypothesis_contents[original] = (
                _read_bounded_text(source) if source is not None else None
            )
        except (OSError, ValueError):
            live_hypothesis_contents[original] = None
    dependency_files, _dependency_reasons = _hypothesis_dependency_files(
        root, live_hypothesis_contents, HYPOTHESIS_LEDGER_PATHS
    )
    dependency_snapshots: list[Path] = []
    for source in sorted(dependency_files):
        try:
            source_relative = source.relative_to(root.resolve())
        except ValueError:
            continue
        destination = output_dir / source_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        dependency_snapshots.append(destination)

    perf_dependency_snapshots: list[Path] = []
    perf_source_pack_snapshots: set[Path] = set()
    live_perf_path = _safe_repo_path(root, "docs/PERF_LEDGER.md")
    try:
        live_perf_text = (
            _read_bounded_text(live_perf_path)
            if live_perf_path is not None
            else ""
        )
    except (OSError, ValueError):
        live_perf_text = ""
    live_perf_verdict = perf_evidence_verdict(live_perf_text, root, head, now)
    eligible_claims = set(live_perf_verdict.get("eligible_claims", []))
    for row in _perf_ledger_rows(live_perf_text):
        if row.get("claim_id") not in eligible_claims:
            continue
        evidence_relative = row.get("evidence_id", "").rstrip("/")
        evidence_dir = _safe_repo_path(root, evidence_relative)
        if evidence_dir is None or not evidence_dir.is_dir():
            continue
        for source in evidence_dir.rglob("*"):
            if not source.is_file() or source.name.startswith("._"):
                continue
            source_relative = source.resolve().relative_to(root.resolve())
            destination = output_dir / source_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            perf_dependency_snapshots.append(destination)
            if (
                source.relative_to(evidence_dir).as_posix()
                == PERF_INPUT_BINDINGS["source_input_pack"]
            ):
                perf_source_pack_snapshots.add(destination)

    core_snapshot_mapping = {
        original: str(destination.relative_to(root.resolve()))
        for original, destination in core_snapshot_paths.items()
    }
    proof_snapshot_mapping = {
        original: str(destination.relative_to(root.resolve()))
        for original, destination in proof_snapshot_paths.items()
    }
    readiness_relative = core_snapshot_mapping["docs/gauntlet/RELEASE_READINESS.json"]
    rounds_relative = core_snapshot_mapping["docs/gauntlet/ROUNDS.jsonl"]
    ledger_relatives = [
        core_snapshot_mapping[original] for original in HYPOTHESIS_LEDGER_PATHS
    ]

    rounds_path = root / "docs/gauntlet/ROUNDS.jsonl"
    rounds: list[dict] = []
    try:
        rounds = [
            parsed
            for line_number, line in enumerate(
                _read_bounded_text(rounds_path).splitlines(), start=1
            )
            if line.strip()
            for parsed in [
                _parse_json_bytes(
                    line.encode("utf-8"), label=f"{rounds_path}:{line_number}"
                )
            ]
            if isinstance(parsed, dict)
        ]
    except (OSError, ValueError, TypeError, RecursionError):
        rounds = []
    hypotheses = hypothesis_ledger_verdict(root)
    conv = convergence_verdict(rounds, hypotheses)
    try:
        release_scorecard = _load_bounded_json(
            root / "docs/gauntlet/RELEASE_SCORECARD.json"
        )
    except (OSError, ValueError, TypeError, RecursionError):
        release_scorecard = {}
    parity_score = release_scorecard.get("parity_score_lower_bound")
    if not isinstance(parity_score, (int, float)) or isinstance(parity_score, bool):
        parity_score = None

    scorecards = {
        "schema_version": STRICT_BUNDLE_CLASSES["scorecards"][1],
        "artifact": "franken_ocr.bundle_scorecards.v1",
        "generated_at_utc": generated_at,
        "parity_score_lower_bound": parity_score,
        "readiness_cells": [],
        "rounds": rounds,
    }
    benchmark_minimums = {
        "primary_score": -3.0,
        "geomean": -5.0,
        "category_geomean": -10.0,
        "p90": -15.0,
        "throughput_drop": -5.0,
    }
    bench_summary: dict = {
        "schema_version": STRICT_BUNDLE_CLASSES["benchmark_summary"][1],
        "artifact": "franken_ocr.bench_summary.v1",
        "generated_at_utc": generated_at,
        "pass_over_pass_gates": {
            name: {
                "passed": False,
                "minimum_pct": minimum,
                "regression_pct": None,
                "reason": "no current strict pass-over-pass proof was supplied",
            }
            for name, minimum in benchmark_minimums.items()
        },
    }
    try:
        bench_summary["frozen_baseline"] = _load_bounded_json(
            root / "benches/.bench-history/baseline.json"
        )
    except (OSError, ValueError, TypeError, RecursionError) as error:
        bench_summary["frozen_baseline_error"] = str(error)
    write_json("scorecards.json", scorecards)
    write_json("benchmark_summary.json", bench_summary)

    evidence_classes = {
        class_name: relative_output(filename)
        for class_name, (filename, _schema) in STRICT_BUNDLE_CLASSES.items()
    }
    provisional_worktree = _git_worktree_state(root)
    provisional_certificate = {
        "schema_version": STRICT_CERTIFICATE_SCHEMA,
        "artifact": STRICT_CERTIFICATE_ARTIFACT,
        "template": STRICT_CERTIFICATE_SCHEMA,
        "project": "franken_ocr",
        "version": package_version,
        "issued_at": generated_at,
        "constants": CERTIFICATION_CONSTANTS,
        "reference": {"model_commit": UNLIMITED_OCR_MODEL_COMMIT},
        "git_head": head,
        "evidence_git_head": head,
        "git_branch": _git_output(root, "symbolic-ref", "--short", "HEAD"),
        "git_describe": _git_output(root, "describe", "--tags", "--always", "--dirty"),
        "git_worktree": provisional_worktree,
        "readiness": {
            "green": 0,
            "red": 1,
            "blocking_cells": ["bundle_generation"],
            "ship": False,
        },
        "convergence": conv,
        "required_pass_actuals": {
            "min_verification_pct_observed": None,
            "required_suite_pass_rate_pct_observed": None,
            "high_severity_counterexample_count": None,
            "max_evidence_age_hours_observed": None,
        },
        "high_severity_counterexamples": None,
        "parity_score": parity_score,
        "evidence_classes": evidence_classes,
        "claim_sources": {
            "release_readiness": readiness_relative,
            "rounds": rounds_relative,
            "hypothesis_ledgers": ledger_relatives,
            "core_evidence": core_snapshot_mapping,
            "readiness_evidence": proof_snapshot_mapping,
        },
        "readiness_cell_set_sha256": CERTIFICATION_READINESS_CELL_SET_SHA256,
        "feature_universe_sha256": None,
        "feature_universe_definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
        "evidence_bundle_sha256": None,
        "signed_claim_sha256": None,
        "signers": [],
        "detached_signatures": [],
        "certified": False,
        "refusal_reasons": ["bundle generation has not completed"],
        "generated_by": "scripts/gauntlet_cert.py --bundle",
    }
    provisional_bundle = {
        "schema_version": STRICT_BUNDLE_SCHEMA,
        "artifact": STRICT_BUNDLE_ARTIFACT,
        "generated_at_utc": generated_at,
        "bundle_root_sha256": None,
        "manifest": [],
    }
    write_json("release_certificate.json", provisional_certificate)
    write_json("certification_bundle.json", provisional_bundle)
    (output_dir / "FINAL_GAUNTLET_REPORT.md").write_text(
        "# FINAL GAUNTLET REPORT\n\nBundle generation is in progress.\n",
        encoding="utf-8",
    )

    readiness_path = readiness_snapshot
    release_readiness(
        str(readiness_path),
        bundle_dir=output_dir,
        now=now,
        evidence_path_mapping=proof_snapshot_mapping,
    )
    try:
        readiness = _load_bounded_json(readiness_path)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        readiness = {
            "green": 0,
            "red": 1,
            "blocking_cells": ["readiness_artifact"],
            "ship": False,
            "cells": [],
            "error": str(error),
        }
    scorecards["readiness_cells"] = readiness.get("cells", [])
    write_json("scorecards.json", scorecards)
    external_cells = [
        cell
        for cell in readiness.get("cells", [])
        if isinstance(cell, dict) and cell.get("cell") != "certification_bundle"
    ]
    external_blocking = [
        str(cell.get("cell")) for cell in external_cells if cell.get("status") == "red"
    ]
    external_readiness = {
        "green": sum(cell.get("status") == "green" for cell in external_cells),
        "red": len(external_blocking),
        "blocking_cells": external_blocking,
        "ship": bool(external_cells) and not external_blocking,
    }

    evidence_paths = list(core_snapshot_mapping.values())
    evidence_paths.extend(proof_snapshot_mapping.values())
    evidence_paths.extend(
        str(path.relative_to(root.resolve())) for path in dependency_snapshots
    )
    evidence_paths.extend(
        str(path.relative_to(root.resolve())) for path in perf_dependency_snapshots
    )
    evidence_paths.extend(evidence_classes.values())
    evidence_paths = list(dict.fromkeys(evidence_paths))
    manifest, stale = build_evidence_manifest(
        root,
        now.timestamp(),
        evidence_paths=evidence_paths,
        static_source_paths={
            relative
            for original, relative in {
                **core_snapshot_mapping,
                **proof_snapshot_mapping,
            }.items()
            if original
            not in {
                "docs/gauntlet/RELEASE_READINESS.json",
                "docs/gauntlet/ROUNDS.jsonl",
            }
        },
        source_pack_paths={
            str(path.relative_to(root.resolve()))
            for path in perf_source_pack_snapshots
        },
    )
    bundle_root = _bundle_root_sha256(manifest)
    observed_ages = [
        float(entry["age_hours"])
        for entry in manifest
        if isinstance(entry.get("age_hours"), (int, float))
        and not isinstance(entry.get("age_hours"), bool)
    ]
    max_evidence_age = round(max([0.0, *observed_ages]), 2)
    live_worktree = _git_worktree_state(root)
    describe = _git_output(root, "describe", "--tags", "--always", "--dirty")

    reasons: list[str] = []
    if not _valid_git_head(head):
        reasons.append("current git HEAD is unavailable")
    if package_version is None:
        reasons.append("Cargo.toml package version is unavailable")
    if live_worktree.get("clean") is not True:
        reasons.append(
            f"git worktree is dirty ({live_worktree.get('dirty_path_count', '?')} status entries)"
        )
    blocking_cells = external_blocking
    if blocking_cells:
        reasons.append(
            "readiness has red cells: " + ", ".join(map(str, blocking_cells))
        )
    if external_readiness["ship"] is not True:
        reasons.append("readiness does not authorize shipment")
    if not conv.get("converged"):
        reasons.append(
            "convergence not met: "
            f"rounds={conv.get('rounds')}/{MIN_ROUNDS} "
            f"tail_clean={conv.get('tail_clean')} "
            f"hypotheses_resolved={conv.get('hypotheses_resolved')}"
        )
    if stale:
        reasons.append(
            f"evidence is stale, future-dated, or missing under the {CERTIFICATION_MAX_EVIDENCE_AGE_HOURS:.0f}h gate: "
            + ", ".join(stale)
        )
    if bundle_root is None:
        reasons.append(
            "the evidence manifest is incomplete, so no bundle root can be certified"
        )
    if parity_score is None:
        reasons.append("the conformal parity lower bound is unavailable")
    reasons.append("strict verification and suite pass percentages are not evidenced")
    reasons.append("the High/Critical counterexample count is not evidenced")
    reasons.append("three independent detached OpenPGP signatures are absent")

    certificate = {
        **provisional_certificate,
        "git_describe": describe,
        "git_worktree": live_worktree,
        "readiness": external_readiness,
        "required_pass_actuals": {
            "min_verification_pct_observed": None,
            "required_suite_pass_rate_pct_observed": None,
            "high_severity_counterexample_count": None,
            "max_evidence_age_hours_observed": max_evidence_age,
        },
        "evidence_bundle_sha256": bundle_root,
        "certified": False,
        "refusal_reasons": reasons,
    }
    certificate["signed_claim_sha256"] = _certificate_signed_claim_sha256(
        certificate, bundle_root
    )
    bundle = {
        **provisional_bundle,
        "bundle_root_sha256": bundle_root,
        "manifest": manifest,
    }
    write_json("release_certificate.json", certificate)
    write_json("certification_bundle.json", bundle)
    (output_dir / "FINAL_GAUNTLET_REPORT.md").write_text(
        _final_report_text(certificate, bundle), encoding="utf-8"
    )

    exact_verdict = certificate_bundle_verdict(
        root,
        head,
        bundle_dir=output_dir,
        now=now,
    )
    for reason in exact_verdict["reasons"]:
        if reason not in certificate["refusal_reasons"]:
            certificate["refusal_reasons"].append(reason)
    certificate["signed_claim_sha256"] = _certificate_signed_claim_sha256(
        certificate, bundle_root
    )
    write_json("release_certificate.json", certificate)
    (output_dir / "FINAL_GAUNTLET_REPORT.md").write_text(
        _final_report_text(certificate, bundle), encoding="utf-8"
    )
    exact_verdict = certificate_bundle_verdict(
        root,
        head,
        bundle_dir=output_dir,
        now=now,
    )

    emit(
        "bundle",
        exact_verdict["ok"],
        out_dir=out_dir,
        verified_bundle_dir=exact_verdict.get("bundle_dir"),
        certified=False,
        refusal_reasons=certificate["refusal_reasons"],
        artifacts=len(manifest),
    )
    return 0 if exact_verdict["ok"] else 1


def eprocess_fold(raw_path: str, state_path: str) -> int:
    """Fold a raw NDJSON stream into the persisted e-process state.

    Exit 1 iff any invariant's e-value has EVER crossed its Ville threshold
    (anytime-valid: the rejection is permanent by construction)."""
    state: dict = {}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    procs = {
        name: _eprocess_from_state(name, state.get("invariants", {}).get(name))
        for name, _ in EPROCESS_MATCHERS
    }
    folded = 0
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            hit = classify_invariant_line(obj)
            if hit is None:
                continue
            name, alarm = hit
            procs[name].observe(alarm)
            folded += 1
    out = {
        "artifact": "franken_ocr.gauntlet.eprocess.v1",
        "source": raw_path,
        "observations_folded": folded,
        "invariants": {
            name: {
                "e_value": ep.e_value,
                "obs_count": ep.obs_count,
                "rejected_at": ep.rejected_at,
                "threshold": ep.threshold,
            }
            for name, ep in procs.items()
        },
        "global_e_value": global_e_value(procs.values()),
        "any_rejected": any(ep.rejected_at is not None for ep in procs.values()),
    }
    with open(state_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, sort_keys=True, indent=1) + "\n")
    print(json.dumps(out, sort_keys=True))
    return 1 if out["any_rejected"] else 0


# --------------------------------------------------------------------------- #
# A worked franken_ocr scorecard (illustrative; matches METHODOLOGY §3.6).
# --------------------------------------------------------------------------- #


def _demo_categories() -> list[CategoryEvidence]:
    """A plausible mid-maturity round: most categories strong, R-SWA mid-climb.

    Weights mirror METHODOLOGY §2.2. Tallies are illustrative (the real numbers
    come from FEATURE_PARITY.md once kernels land), but they demonstrate the
    full Beta + conformal + ratchet pipeline end to end.
    """
    return [
        _cat("Preprocess", 0.12, present=17, partial=0, missing=1),
        _cat("Tokenizer", 0.10, present=4, partial=0, missing=0),
        _cat("VisionSAM", 0.10, present=9, partial=1, missing=0),
        _cat("VisionCLIP", 0.08, present=8, partial=0, missing=1),
        _cat("Connector", 0.08, present=7, partial=0, missing=0),
        _cat("DecoderMoE", 0.16, present=13, partial=1, missing=0),
        _cat("RSWA", 0.14, present=7, partial=1, missing=1),
        _cat("SamplerPost", 0.10, present=8, partial=0, missing=1),
        _cat("OpMapFacade", 0.06, present=19, partial=1, missing=0),
        _cat("QuantRecipe", 0.06, present=8, partial=0, missing=1),
    ]


def _cat(
    name: str, weight: float, present: int, partial: int, missing: int
) -> CategoryEvidence:
    # equal-weight within the category (default rubric) so per-row weight = 1/n_rows
    n = present + partial + missing
    w = 1.0 / n if n else 0.0
    ev = CategoryEvidence(name, weight)
    for _ in range(present):
        add_outcome(ev, "present", w)
    for _ in range(partial):
        add_outcome(ev, "partial", w)
    for _ in range(missing):
        add_outcome(ev, "missing", w)
    return ev


def _demo() -> int:
    cats = _demo_categories()
    # Held-out residuals from prior cycles (|observed - Beta-mean|); illustrative.
    residuals = [0.012, 0.018, 0.021, 0.027, 0.031, 0.038, 0.041, 0.047, 0.052, 0.061]
    sc = compute_scorecard(cats, residuals, confidence=0.95)
    # A prior round's persisted high-water mark below this round's bounds, so this
    # round is an Allow that advances the ratchet (the instructive default case).
    state = RatchetState(
        current_lower_bound=0.540000,
        per_category_bounds={c.category_id: 0.540000 for c in cats},
    )
    verdict = apply_ratchet(sc, state)
    print(
        json.dumps(
            {
                "artifact": "franken_ocr.gauntlet.scorecard.v1",
                "point_estimate": sc.point_estimate,
                "parity_score_lower_bound": sc.lower_bound,
                "conformal_halfwidth": round(sc.conformal_halfwidth, 6),
                "per_category_lower": sc.per_category_lower,
                "ratchet_decision": verdict.decision,
                "ratchet_reason": verdict.reason,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


# --------------------------------------------------------------------------- #
# Self-test (CI gate; mirrors scripts/check_test_logs.py --self-test style).
# --------------------------------------------------------------------------- #


def emit(check: str, ok: bool, **fields: object) -> None:
    print(
        json.dumps(
            {"check": check, "result": "pass" if ok else "fail", **fields},
            sort_keys=True,
        )
    )


def _approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, **fields: object) -> None:
        emit(name, cond, **fields)
        if not cond:
            failures.append(name)

    # --- truncate_score: truncates, never rounds; idempotent; cross-arch stable.
    check("truncate_score_truncates", _approx(truncate_score(0.8472919), 0.847291))
    check("truncate_score_no_round_up", _approx(truncate_score(0.9999999), 0.999999))
    check(
        "truncate_score_idempotent",
        truncate_score(truncate_score(0.123456789)) == truncate_score(0.123456789),
    )
    check(
        "glibc_floor_numeric_order_accepts_2_17",
        _numeric_version("2.17") <= _numeric_version(CERTIFICATION_LINUX_GLIBC_FLOOR),
    )
    check(
        "glibc_floor_numeric_order_rejects_2_18",
        _numeric_version("2.18") > _numeric_version(CERTIFICATION_LINUX_GLIBC_FLOOR),
    )
    check(
        "focr_version_accepts_canonical_multiline_report",
        _reported_focr_version(
            "focr 0.7.0\n"
            "source_license: franken_ocr - MIT License\n"
            "model_license: Baidu Unlimited-OCR - MIT License\n"
        )
        == "0.7.0"
        and _reported_focr_version("focr 0.7.0-rc.1+build.9\r\nsource_license: MIT\r\n")
        == "0.7.0-rc.1+build.9",
    )
    check(
        "focr_version_rejects_noncanonical_first_line",
        _reported_focr_version("source_license: MIT\nfocr 0.7.0\n") is None
        and _reported_focr_version("") is None,
    )

    legacy_tag_inputs = {
        "RUST_TOOLCHAIN": "nightly-2026-07-09",
        "FRANKENTORCH_REV": "f00c3ce79737ffd0e7c7e2f2e81540907ce26562",
        "FRANKENSQLITE_REV": "14c5781680833c248ed18d84914312e233d145df",
        "ASUPERSYNC_REV": "e7e96829632d89815248a0809499332ab0beabcb",
    }
    legacy_tag_workflow = "env:\n" + "".join(
        f"  {name}: {value}\n" for name, value in legacy_tag_inputs.items()
    )
    parsed_legacy_inputs = _parse_tag_owned_dist_release_inputs(
        legacy_tag_workflow, "nightly-2026-07-09"
    )
    check(
        "dist_release_inputs_parse_exact_legacy_tag",
        parsed_legacy_inputs == legacy_tag_inputs,
    )
    check(
        "dist_release_inputs_exact_match_has_no_drift",
        _dist_release_input_drift(parsed_legacy_inputs, dict(legacy_tag_inputs)) == [],
    )
    advanced_inputs = dict(legacy_tag_inputs)
    advanced_inputs["ASUPERSYNC_REV"] = "a9b99549609b39bc1dd8b128591ef324d6ef2d71"
    check(
        "dist_release_inputs_reject_v0_7_asupersync_drift",
        _dist_release_input_drift(parsed_legacy_inputs, advanced_inputs)
        == [
            "ASUPERSYNC_REV "
            "tag=e7e96829632d89815248a0809499332ab0beabcb "
            "workflow=a9b99549609b39bc1dd8b128591ef324d6ef2d71"
        ],
    )

    def release_input_parse_rejected(workflow_text: str, channel: str) -> bool:
        try:
            _parse_tag_owned_dist_release_inputs(workflow_text, channel)
        except ValueError:
            return True
        return False

    check(
        "dist_release_inputs_reject_missing_pin",
        release_input_parse_rejected(
            legacy_tag_workflow.replace(
                f"  ASUPERSYNC_REV: {legacy_tag_inputs['ASUPERSYNC_REV']}\n", ""
            ),
            "nightly-2026-07-09",
        ),
    )
    check(
        "dist_release_inputs_reject_duplicate_pin",
        release_input_parse_rejected(
            legacy_tag_workflow
            + f"  ASUPERSYNC_REV: {legacy_tag_inputs['ASUPERSYNC_REV']}\n",
            "nightly-2026-07-09",
        ),
    )
    check(
        "dist_release_inputs_reject_malformed_revision",
        release_input_parse_rejected(
            legacy_tag_workflow.replace(
                legacy_tag_inputs["ASUPERSYNC_REV"], "not-a-revision"
            ),
            "nightly-2026-07-09",
        ),
    )
    check(
        "dist_release_inputs_reject_toolchain_disagreement",
        release_input_parse_rejected(legacy_tag_workflow, "nightly-2026-07-10"),
    )

    # --- Beta posterior mean + quantile sanity (Beta(201,1) ~ 0.9950 mean).
    post = BetaParams(201.0, 1.0)
    check("beta_mean", _approx(post.mean(), 201.0 / 202.0), mean=post.mean())
    lo, hi = post.credible_interval(0.95)
    check(
        "beta_ci_orders",
        0.0 < lo < post.mean() < hi <= 1.0,
        lo=round(lo, 6),
        hi=round(hi, 6),
    )
    # regularized incomplete beta endpoints
    check("reg_inc_beta_0", _reg_inc_beta(0.0, 2.0, 3.0) == 0.0)
    check("reg_inc_beta_1", _reg_inc_beta(1.0, 2.0, 3.0) == 1.0)
    check(
        "reg_inc_beta_half_symmetry", _approx(_reg_inc_beta(0.5, 3.0, 3.0), 0.5, 1e-6)
    )

    # --- partial counts in BOTH alpha and beta (§3.1 pitfall).
    ev = CategoryEvidence("c", 1.0)
    add_outcome(ev, "partial", 1.0)
    p = ev.posterior()
    check(
        "partial_both_sides",
        _approx(p.alpha, 1.5) and _approx(p.beta, 1.5),
        alpha=p.alpha,
        beta=p.beta,
    )
    # a present-only category beats a missing-only category
    ev_pass = _cat("pass", 1.0, present=10, partial=0, missing=0)
    ev_fail = _cat("fail", 1.0, present=0, partial=0, missing=10)
    check(
        "present_beats_missing", ev_pass.posterior().mean() > ev_fail.posterior().mean()
    )

    # --- conformal half-width is the (1-alpha) empirical quantile.
    res = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    q95 = conformal_halfwidth(res, 0.95)
    check("conformal_quantile_in_range", res[0] <= q95 <= res[-1], q=q95)
    check("conformal_empty_bootstraps_wide", conformal_halfwidth([], 0.95) == 1.0)
    # tighter confidence -> wider (larger) half-width
    check(
        "conformal_tighter_wider",
        conformal_halfwidth(res, 0.99) >= conformal_halfwidth(res, 0.90),
    )

    # --- weights must sum to 1.0 (loader invariant).
    bad = [CategoryEvidence("a", 0.4), CategoryEvidence("b", 0.4)]
    try:
        compute_scorecard(bad, [0.01], 0.95)
        check("weights_sum_enforced", False)
    except ValueError:
        check("weights_sum_enforced", True)

    # --- release uses the LOWER bound, not the point estimate.
    sc = compute_scorecard(_demo_categories(), res, 0.95)
    check(
        "lower_below_point",
        sc.lower_bound <= sc.point_estimate,
        lower=sc.lower_bound,
        point=sc.point_estimate,
    )
    check("lower_is_truncated", truncate_score(sc.lower_bound) == sc.lower_bound)

    # --- ratchet: Allow raises both bounds; Block on regression; Quarantine on small single dip.
    state = RatchetState(0.5, {c.category_id: 0.5 for c in _demo_categories()})
    v_allow = apply_ratchet(sc, state)
    check("ratchet_allows_progress", v_allow.decision == "Allow", reason=v_allow.reason)
    check(
        "ratchet_monotone",
        v_allow.new_state.current_lower_bound >= state.current_lower_bound,
    )

    high = RatchetState(0.999999, {c.category_id: 0.999999 for c in _demo_categories()})
    v_block = apply_ratchet(sc, high)
    check(
        "ratchet_blocks_regression", v_block.decision == "Block", reason=v_block.reason
    )

    # single tiny per-category dip with global holding -> Quarantine
    qstate = RatchetState(
        current_lower_bound=0.0,
        per_category_bounds={
            c.category_id: sc.per_category_lower[c.category_id]
            for c in _demo_categories()
        },
    )
    one = next(iter(qstate.per_category_bounds))
    qstate.per_category_bounds[one] = (
        sc.per_category_lower[one] + 0.003
    )  # dip 0.003 ≤ 0.005
    v_q = apply_ratchet(sc, qstate)
    check(
        "ratchet_quarantines_small_dip", v_q.decision == "Quarantine", reason=v_q.reason
    )
    # ...unless a waiver covers it
    v_w = apply_ratchet(sc, qstate, waived_categories=frozenset({one}))
    check(
        "ratchet_waiver_covers_dip",
        v_w.decision in ("Allow", "Waiver"),
        decision=v_w.decision,
    )

    # --- e-process calibration: all four invariants registered, split correct.
    check(
        "four_invariants_registered",
        set(INVARIANT_CALIBRATION)
        == {"INV-KV-CAP", "INV-I32-NOOVERFLOW", "INV-DETERMINISM", "INV-SIMD-SCALAR"},
    )
    check(
        "hardware_invariants_tight",
        INVARIANT_CALIBRATION["INV-SIMD-SCALAR"]["p0"] == 1e-9
        and INVARIANT_CALIBRATION["INV-DETERMINISM"]["p0"] == 1e-9,
    )
    check(
        "software_invariants_loose",
        INVARIANT_CALIBRATION["INV-KV-CAP"]["p0"] == 1e-6
        and INVARIANT_CALIBRATION["INV-I32-NOOVERFLOW"]["p0"] == 1e-6,
    )

    # --- Ville: a healthy stream never rejects; a violation burst does.
    ep = EProcess.for_invariant("INV-SIMD-SCALAR")  # hardware, threshold 1e6
    for _ in range(10000):
        ep.observe(False)
    check(
        "eprocess_healthy_no_reject",
        ep.rejected_at is None and ep.e_value < ep.threshold,
    )
    # a single hardware violation is consistent with the null (does NOT reject)
    ep_single = EProcess.for_invariant("INV-SIMD-SCALAR")
    for _ in range(500):
        ep_single.observe(False)
    fired_single = ep_single.observe(True)
    check(
        "eprocess_single_hw_violation_no_reject",
        not fired_single and ep_single.rejected_at is None,
    )
    # a burst of hardware violations rejects within a handful of observations
    ep_burst = EProcess.for_invariant("INV-SIMD-SCALAR")
    fired = False
    for i in range(8):
        if ep_burst.observe(True):
            fired = True
            break
    check(
        "eprocess_burst_rejects",
        fired and ep_burst.rejected_at is not None and ep_burst.rejected_at <= 4,
    )

    # software invariant rejects on ~2 consecutive violations (threshold 1e3)
    ep_soft = EProcess.for_invariant("INV-KV-CAP")
    soft_fired = False
    for _ in range(3):
        if ep_soft.observe(True):
            soft_fired = True
            break
    check(
        "eprocess_software_rejects_fast",
        soft_fired and ep_soft.rejected_at is not None and ep_soft.rejected_at <= 2,
    )
    # ...and the e-VALUE decays back toward 0 under sustained health: 2 sporadic
    # violations diluted by 18 healthy observations leave E_t far below 1.0 (the
    # §4 worked figure ~8.1e-7), even though rejected_at latches the first crossing.
    # (A single software alarm DOES cross 1e3 -- 9e5 > 1e3 -- so the latch fires;
    # the anytime-valid guarantee is about the e-VALUE trajectory, which decays.)
    ep_dilute = EProcess.for_invariant("INV-KV-CAP")
    ep_dilute.observe(True)
    for _ in range(18):
        ep_dilute.observe(False)
    ep_dilute.observe(True)
    check(
        "eprocess_software_evalue_decays", ep_dilute.e_value < 1e-3, e=ep_dilute.e_value
    )

    # --- global e-value is the arithmetic mean and stays below threshold even if
    # one invariant individually crossed (family-wise guarantee, §6.1).
    procs = [EProcess.for_invariant(n) for n in INVARIANT_CALIBRATION]
    procs[0].e_value = 1e6  # one invariant crossed on its own
    g = global_e_value(procs)
    check(
        "global_e_arithmetic_mean", _approx(g, (1e6 + 1.0 + 1.0 + 1.0) / 4.0, 1.0), g=g
    )
    check("global_e_below_hw_threshold", g < 1.0 / HARDWARE["alpha"], g=g)

    # --- never-reset discipline: e_value is not forced back to 1.0 on saturation.
    ep_sat = EProcess.for_invariant("INV-DETERMINISM")
    for _ in range(5):
        ep_sat.observe(True)
    check("eprocess_no_reset", ep_sat.e_value > 1.0)

    # bd-re8.13 — the real-data modes.
    fixture_md = "\n".join(
        [
            "## 12. CLI surface",
            "| Surface | §7 | Req | Status | Parity | Bead | Notes |",
            "|---|---|---|---|---|---|---|",
            "| `focr a` | §7 | MUST | present | SURF | bd-x | |",
            "| `focr b` | §7 | MUST | partial | SURF | bd-x | |",
            "| `focr c` | §7 | MAY | missing | SURF | bd-x | |",
            "## 13. Events",
            "| Event | §7 | Req | Status | Parity | Bead | Notes |",
            "|---|---|---|---|---|---|---|",
            "| `e1` | §7 | MUST | present | SURF | bd-x | |",
            "| `e2` | §7 | n/a | n/a | n/a | — | skipped in scoring |",
        ]
    )
    parsed = parse_feature_parity(fixture_md)
    check(
        "parity_parse_sections_and_tallies",
        parsed
        == {
            "§12 CLI surface": {"present": 1, "partial": 1, "missing": 1},
            "§13 Events": {"present": 1, "n/a": 1},
        },
        parsed=parsed,
    )
    cats_fp = categories_from_parity(fixture_md)
    check(
        "parity_categories_weights_and_na_skip",
        len(cats_fp) == 2
        and _approx(sum(c.category_weight for c in cats_fp), 1.0)
        # §13: one scored row (present), the n/a row skipped entirely.
        and _approx(cats_fp[1].weighted_successes, 1.0)
        and _approx(cats_fp[1].weighted_failures, 0.0),
    )
    sc_fp = compute_scorecard(cats_fp, residuals=[], confidence=0.95)
    check(
        "parity_scorecard_bootstrap_band_is_conservative",
        _approx(sc_fp.conformal_halfwidth, 1.0) and _approx(sc_fp.lower_bound, 0.0),
        point=sc_fp.point_estimate,
    )

    surface_partial = surface_must_verdict(fixture_md)
    check(
        "surface_must_partial_fails_closed",
        not surface_partial["ok"]
        and surface_partial["debt"] == [{"surface": "`focr b`", "status": "partial"}],
        verdict=surface_partial,
    )
    surface_present = surface_must_verdict(
        fixture_md.replace(
            "| `focr b` | §7 | MUST | partial |", "| `focr b` | §7 | MUST | present |"
        )
    )
    check(
        "surface_all_must_present_passes",
        surface_present["ok"],
        verdict=surface_present,
    )

    missing_hypotheses = hypothesis_texts_verdict(
        {path: None for path in HYPOTHESIS_LEDGER_PATHS}
    )
    check(
        "hypothesis_ledgers_missing_fail_closed",
        not missing_hypotheses["resolved"]
        and missing_hypotheses["missing"] == list(HYPOTHESIS_LEDGER_PATHS),
        verdict=missing_hypotheses,
    )
    prose_only_hypotheses = {
        path: "# Hypothesis ledger\n\nNo open hypotheses.\n"
        for path in HYPOTHESIS_LEDGER_PATHS
    }
    prose_only_verdict = hypothesis_texts_verdict(prose_only_hypotheses)
    check(
        "hypothesis_ledgers_prose_only_empty_rejected",
        not prose_only_verdict["resolved"]
        and all(
            item["state"] == "NO_CANONICAL_EXPERIMENT_ENTRIES"
            for item in prose_only_verdict["unresolved"]
        ),
        verdict=prose_only_verdict,
    )

    with tempfile.TemporaryDirectory(
        prefix="focr-hypothesis-self-test-"
    ) as hypothesis_tmp:
        hypothesis_root = Path(hypothesis_tmp)
        proof_dir = hypothesis_root / "evidence"
        proof_dir.mkdir()
        proof_path = proof_dir / "result.txt"
        proof_path.write_text("measured no-evidence result\n", encoding="utf-8")
        (proof_dir / "SHA256SUMS").write_text(
            f"{_sha256_file(proof_path)}  result.txt\n", encoding="utf-8"
        )

        def canonical_experiment(
            experiment_id: str,
            pillar: str,
            *,
            result_status: str = "NO_EVIDENCE",
            status: str = "CLOSED",
            include_retry: bool = True,
            include_remediation: bool = False,
        ) -> str:
            result_lines = [
                "```yaml",
                f"result_status: {result_status}",
                "result_summary: measured result closed against the stated falsifier",
                "result_evidence_paths:",
                "  - evidence/result.txt",
                "closed_at_utc: 2026-07-10T01:00:00Z",
            ]
            if include_retry:
                result_lines.append(
                    "retry_condition_predicate: retry only if a profiler attributes a clearly-above-noise share to decode"
                )
            if include_remediation:
                result_lines.extend(
                    [
                        "result_impact: measured token mismatch on the pinned fixture",
                        "spawned_remediation_bead: bd-self-test",
                    ]
                )
            result_lines.append("```")
            return "\n".join(
                [
                    f"## Experiment `{experiment_id}` - Self-test experiment",
                    "",
                    "| field | value |",
                    "|---|---|",
                    f"| `experiment_id` | `{experiment_id}` |",
                    f"| `pillar` | `{pillar}` |",
                    "| `created_at_utc` | `2026-07-10T00:00:00Z` |",
                    "| `created_by_agent` | `self-test-agent` |",
                    "| `bead_id` | `bd-self-test` |",
                    "| `parent_hypothesis_id` | `ROOT-SELF-TEST` |",
                    f"| `status` | `{status}` |",
                    "",
                    "### Hypothesis",
                    "A bounded implementation claim can be falsified.",
                    "",
                    "### Motivation",
                    "The release gate needs an explicit experimental record.",
                    "",
                    "### Minimal Reproducer",
                    "Run the pinned fixture against both implementations.",
                    "",
                    "### Expected Signal",
                    "The measured result stays inside the declared threshold.",
                    "",
                    "### Falsifiability Criteria",
                    "Any mismatch outside the threshold falsifies the claim.",
                    "",
                    "### One-Line Invocation",
                    "`python3 verify_fixture.py`",
                    "",
                    "### Results Inline",
                    *result_lines,
                    "",
                    "### Closure Predicate",
                    "Close only after the evidence path and retry obligation verify.",
                    "",
                ]
            )

        pillar_by_ledger = {
            HYPOTHESIS_LEDGER_PATHS[0]: "perf",
            HYPOTHESIS_LEDGER_PATHS[1]: "perf",
            HYPOTHESIS_LEDGER_PATHS[2]: "conformance",
            HYPOTHESIS_LEDGER_PATHS[3]: "surface",
        }
        resolved_hypothesis_texts = {
            ledger: canonical_experiment(f"EXP-{index:04d}", pillar)
            for index, (ledger, pillar) in enumerate(pillar_by_ledger.items(), start=1)
        }
        resolved_hypotheses = hypothesis_texts_verdict(
            resolved_hypothesis_texts,
            root=hypothesis_root,
        )
        check(
            "hypothesis_ledgers_canonical_closed_records_pass",
            resolved_hypotheses["resolved"] and resolved_hypotheses["entries"] == 4,
            verdict=resolved_hypotheses,
        )

        duplicate_texts = dict(resolved_hypothesis_texts)
        duplicate_texts[HYPOTHESIS_LEDGER_PATHS[1]] += duplicate_texts[
            HYPOTHESIS_LEDGER_PATHS[1]
        ]
        duplicate_verdict = hypothesis_texts_verdict(
            duplicate_texts, root=hypothesis_root
        )
        check(
            "hypothesis_duplicate_experiment_id_rejected",
            not duplicate_verdict["resolved"]
            and any(
                item["state"] == "DUPLICATE_EXPERIMENT_ID"
                for item in duplicate_verdict["unresolved"]
            ),
            verdict=duplicate_verdict,
        )

        cross_ledger_duplicate = dict(resolved_hypothesis_texts)
        cross_ledger_duplicate[HYPOTHESIS_LEDGER_PATHS[2]] = canonical_experiment(
            "EXP-0002", "conformance"
        )
        cross_ledger_duplicate_verdict = hypothesis_texts_verdict(
            cross_ledger_duplicate, root=hypothesis_root
        )
        check(
            "hypothesis_cross_ledger_duplicate_id_rejected",
            not cross_ledger_duplicate_verdict["resolved"]
            and any(
                item["state"] == "DUPLICATE_EXPERIMENT_ID"
                for item in cross_ledger_duplicate_verdict["unresolved"]
            ),
            verdict=cross_ledger_duplicate_verdict,
        )

        hidden_open = dict(resolved_hypothesis_texts)
        hidden_open[HYPOTHESIS_LEDGER_PATHS[0]] = (
            "## Experiment `OQ-0001` - hidden open record\n\nstatus: OPEN\n\n"
            + hidden_open[HYPOTHESIS_LEDGER_PATHS[0]]
        )
        hidden_open_verdict = hypothesis_texts_verdict(
            hidden_open, root=hypothesis_root
        )
        check(
            "hypothesis_noncanonical_open_heading_cannot_hide",
            not hidden_open_verdict["resolved"]
            and any(
                item["state"] == "NONCANONICAL_EXPERIMENT_HEADING"
                for item in hidden_open_verdict["unresolved"]
            ),
            verdict=hidden_open_verdict,
        )

        unfenced_results = dict(resolved_hypothesis_texts)
        unfenced_results[HYPOTHESIS_LEDGER_PATHS[1]] = (
            unfenced_results[HYPOTHESIS_LEDGER_PATHS[1]]
            .replace("```yaml\n", "", 1)
            .replace(
                "\n```\n\n### Closure Predicate",
                "\n\n### Closure Predicate",
                1,
            )
        )
        unfenced_verdict = hypothesis_texts_verdict(
            unfenced_results, root=hypothesis_root
        )
        check(
            "hypothesis_unfenced_result_fields_rejected",
            not unfenced_verdict["resolved"]
            and any(
                "fenced YAML" in failure
                for item in unfenced_verdict["unresolved"]
                for failure in item.get("failures", [])
            ),
            verdict=unfenced_verdict,
        )

        unresolved_hypothesis_texts = dict(resolved_hypothesis_texts)
        unresolved_hypothesis_texts[HYPOTHESIS_LEDGER_PATHS[1]] = canonical_experiment(
            "EXP-0002", "perf", result_status="NEEDS_REFINEMENT", status="RUNNING"
        )
        unresolved_hypotheses = hypothesis_texts_verdict(
            unresolved_hypothesis_texts,
            root=hypothesis_root,
        )
        check(
            "hypothesis_nonterminal_state_rejected",
            not unresolved_hypotheses["resolved"]
            and unresolved_hypotheses["unresolved"][0]["state"] == "NEEDS_REFINEMENT",
            verdict=unresolved_hypotheses,
        )

        gap_texts = dict(resolved_hypothesis_texts)
        gap_texts[HYPOTHESIS_LEDGER_PATHS[2]] = canonical_experiment(
            "EXP-0003",
            "conformance",
            result_status="CONFIRMED_GAP",
            include_retry=False,
            include_remediation=False,
        )
        gap_verdict = hypothesis_texts_verdict(gap_texts, root=hypothesis_root)
        gap_failures = gap_verdict["unresolved"][0].get("failures", [])
        check(
            "hypothesis_confirmed_gap_requires_remediation_and_proof_pack",
            not gap_verdict["resolved"]
            and any("remediation bead" in failure for failure in gap_failures)
            and any("proof pack" in failure for failure in gap_failures),
            verdict=gap_verdict,
        )

    def canonical_round(round_number: int, new_findings: int) -> dict:
        return {
            "round": round_number,
            "date": "2026-07-09",
            "new_findings": new_findings,
            "pillars": {
                "conformance": "self-test conformance evidence",
                "surface": "self-test surface evidence",
                "perf": "self-test performance evidence",
            },
            "notes": "self-test canonical round",
        }

    rounds_clean = [canonical_round(i, 5) for i in range(1, 9)] + [
        canonical_round(9, 2),
        canonical_round(10, 0),
    ]
    check(
        "convergence_meets_at_10_with_2_clean_and_resolved_hypotheses",
        convergence_verdict(rounds_clean, resolved_hypotheses)["converged"],
    )
    check(
        "convergence_refuses_missing_hypothesis_verdict",
        not convergence_verdict(rounds_clean)["converged"],
    )
    check(
        "convergence_refuses_unresolved_hypotheses",
        not convergence_verdict(rounds_clean, unresolved_hypotheses)["converged"],
    )
    check(
        "convergence_refuses_short_history",
        not convergence_verdict(rounds_clean[:9], resolved_hypotheses)["converged"],
    )
    check(
        "convergence_refuses_dirty_tail",
        not convergence_verdict(
            rounds_clean[:9] + [canonical_round(10, 3)],
            resolved_hypotheses,
        )["converged"],
    )
    check(
        "convergence_refuses_empty",
        not convergence_verdict([], resolved_hypotheses)["converged"],
    )
    malformed_rounds = [dict(record) for record in rounds_clean]
    malformed_rounds[-1]["new_findings"] = "not-an-int"
    malformed_round_verdict = convergence_verdict(malformed_rounds, resolved_hypotheses)
    check(
        "convergence_malformed_round_fails_closed_without_exception",
        not malformed_round_verdict["converged"]
        and bool(malformed_round_verdict["round_history_errors"]),
        verdict=malformed_round_verdict,
    )

    with tempfile.TemporaryDirectory(prefix="focr-gauntlet-cert-self-test-") as tmp:
        fixture_root = Path(tmp)
        bounded_input = fixture_root / "bounded-input.bin"
        bounded_input.write_bytes(b"12345")
        bounded_rejected = False
        try:
            _read_bounded_file(bounded_input, 4)
        except ValueError:
            bounded_rejected = True
        check("bounded_reader_rejects_oversize_input", bounded_rejected)
        binary_payload = b"PE\x00\r\n\x1a\x00payload\r\n"
        binary_input = fixture_root / "binary-input.bin"
        binary_snapshot = fixture_root / "binary-snapshot.bin"
        binary_input.write_bytes(binary_payload)
        check(
            "bounded_reader_preserves_windows_binary_bytes",
            _read_bounded_file(binary_input) == binary_payload,
        )
        bounded_binary_identity = _bounded_file_identity(
            binary_input, len(binary_payload)
        )
        check(
            "bounded_identity_preserves_windows_binary_bytes",
            bounded_binary_identity
            == {
                "sha256": hashlib.sha256(binary_payload).hexdigest(),
                "size": len(binary_payload),
            },
        )
        binary_identity = _stream_file_identity(
            binary_input,
            len(binary_payload),
            snapshot_path=binary_snapshot,
        )
        check(
            "streamed_identity_preserves_windows_binary_bytes",
            binary_identity["size"] == len(binary_payload)
            and binary_identity["sha256"] == hashlib.sha256(binary_payload).hexdigest()
            and binary_snapshot.read_bytes() == binary_payload,
        )
        deep_json_rejected = False
        try:
            _parse_json_bytes(
                ("[" * 2_000 + "0" + "]" * 2_000).encode("ascii"),
                label="self-test-deep-json",
            )
        except ValueError:
            deep_json_rejected = True
        check("bounded_json_rejects_deep_input_without_exception", deep_json_rejected)
        duplicate_json_rejected = False
        try:
            _parse_json_bytes(
                b'{"certified":false,"certified":true}',
                label="self-test-duplicate-json",
            )
        except ValueError:
            duplicate_json_rejected = True
        check("bounded_json_rejects_duplicate_keys", duplicate_json_rejected)
        evidence_id = "artifacts/perf/self-test-current"
        evidence_dir = fixture_root / evidence_id
        evidence_dir.mkdir(parents=True)
        fixed_now = datetime.now(timezone.utc).replace(microsecond=0)
        fixed_head = ""
        perf_columns = [
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
        base_perf_row = {
            "date": "2026-07-10",
            "claim_id": "G2-self-test-current",
            "evidence_id": evidence_id,
            "model_commit": UNLIMITED_OCR_MODEL_COMMIT,
            "fixture_hash": "page=fixture.png sha256=" + "a" * 64,
            "arch/cpu_features": "aarch64+neon+dotprod",
            "stage": "decode-per-token",
            "reference_backend": "hf",
            "focr_ms": "10.000",
            "ref_ms": "20.000",
            "ratio": "2.000",
            "floor_kind": "memory",
            "floor_ms": "5.000",
            "dist_above_floor": "2.00",
            "precision (focr vs ref)": (
                f"{CURRENT_UNLIMITED_PRECISION} vs hf-bf16"
            ),
            "threads (focr=ref N)": "focr=ref=8",
            "allocator": "system",
            "command/env": (
                "focr: target/release-perf/focr ocr fixture.png --model "
                "/models/unlimited-ocr.focrq [threads=8 FOCR_TIMING=1]; "
                "ref: gauntlet_reference.py --threads 8 "
                f"[decode_mode={CURRENT_UNLIMITED_DECODE_MODE} "
                f"quant_recipe={CURRENT_UNLIMITED_QUANT_RECIPE} "
                f"model_sha256={CURRENT_UNLIMITED_MODEL_SHA256} "
                f"model_size={CURRENT_UNLIMITED_MODEL_SIZE}]"
            ),
            "fallback/kill-switch state": (
                "FOCR_ATTN_GEMM=<unset> FOCR_DECODE_INT8=<unset> "
                "FOCR_DECODE_STATELESS=<unset> FOCR_INT8_ATTN=<unset> "
                "FOCR_INT8_KV=<unset> FOCR_INT8_LMHEAD=<unset> "
                "FOCR_SPEC_DECODE=<unset> FOCR_THREADS=8 FOCR_TIMING=1"
            ),
            "correctness_proof": "",
            "notes": "self-test fixture",
        }
        timestamp = fixed_now.isoformat().replace("+00:00", "Z")

        def perf_text_for(*rows: dict) -> str:
            return "\n".join(
                [
                    "| " + " | ".join(perf_columns) + " |",
                    "|" + "|".join("---" for _ in perf_columns) + "|",
                    *[
                        "| " + " | ".join(row[column] for column in perf_columns) + " |"
                        for row in rows
                    ],
                ]
            )

        correctness_reference_bytes = b"identical <|det|>box<|/det|>output\n"
        correctness_hypothesis_bytes = b"identical output\n"
        correctness_reference_sha = hashlib.sha256(
            correctness_reference_bytes
        ).hexdigest()
        correctness_receipt_doc = _synthetic_correctness_receipt(
            correctness_reference_bytes, correctness_hypothesis_bytes
        )
        correctness_receipt_doc["pages"][0]["page"] = "fixture.md"
        correctness_receipt_doc["pages"][0]["reference"]["source"][
            "basename"
        ] = "fixture.md"

        def raw_timing_doc(
            source: str, samples: list[float], *, tokens: int | None
        ) -> dict:
            records = []
            for index, sample in enumerate(samples, start=1):
                run_id = f"run_{index:03d}"
                stage_sample = {"ms": sample}
                if tokens is not None:
                    stage_sample["tokens"] = tokens
                record = {
                    "run_id": run_id,
                    "stages": {"decode_per_token": stage_sample},
                }
                if source == "focr":
                    record["raw_files"] = {}
                else:
                    record["text_sha256"] = correctness_reference_sha
                records.append(record)
            return {
                "schema": RAW_TIMING_SCHEMA,
                "source": source,
                "unit": "ms",
                "measured_runs": len(records),
                "records": records,
            }

        live_root = Path(__file__).resolve().parent.parent
        for relative in PRODUCER_PATHS:
            source = live_root.joinpath(*Path(relative).parts)
            destination = fixture_root.joinpath(*Path(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        build_source_contents = {
            "Cargo.toml": b"[package]\nname='selftest'\n",
            "Cargo.lock": b"version = 4\n",
            "rust-toolchain.toml": b"[toolchain]\nchannel='nightly'\n",
            "src/lib.rs": b"pub fn selftest() {}\n",
        }
        for relative, content in build_source_contents.items():
            destination = fixture_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        (fixture_root / ".gitignore").write_text(
            "/.gauntlet-output/\n", encoding="utf-8"
        )
        for command in (
            ["git", "init", "-q"],
            ["git", "config", "user.name", "Gauntlet Self Test"],
            ["git", "config", "user.email", "gauntlet@example.invalid"],
            [
                "git",
                "add",
                "--",
                ".gitignore",
                *PRODUCER_PATHS,
                *build_source_contents,
            ],
            ["git", "commit", "-qm", "source fixture"],
        ):
            result = subprocess.run(
                command,
                cwd=fixture_root,
                env=_git_env(),
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"cannot construct gauntlet provenance self-test repo: {command!r}"
                )
        fixed_head = _git_output(fixture_root, "rev-parse", "HEAD")
        fixture_producer_root = _gauntlet_producer_root(fixture_root, fixed_head)
        if fixture_producer_root is None:
            raise RuntimeError("cannot compute self-test producer root")
        build_entries = [
            {
                "repository": "workspace",
                "path": name,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in sorted(build_source_contents.items())
        ]
        build_root_digest = hashlib.sha256(SOURCE_ROOT_DOMAIN)
        for item in build_entries:
            build_root_digest.update(item["repository"].encode())
            build_root_digest.update(b"\0")
            build_root_digest.update(item["path"].encode())
            build_root_digest.update(b"\0")
            build_root_digest.update(str(item["size"]).encode())
            build_root_digest.update(b"\0")
            build_root_digest.update(item["sha256"].encode())
            build_root_digest.update(b"\n")
        build_manifest_doc = {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "created_utc": timestamp,
            "root_hash_algorithm": SOURCE_ROOT_ALGORITHM,
            "root_sha256": build_root_digest.hexdigest(),
            "entry_count": len(build_entries),
            "repositories": [
                {
                    "id": "workspace",
                    "path": str(fixture_root.resolve()),
                    "git_head": fixed_head,
                    "packages": ["franken_ocr"],
                    "selectors": ["Cargo.lock", "Cargo.toml", "rust-toolchain.toml", "src"],
                }
            ],
            "cargo_config_files": [],
            "entries": build_entries,
        }
        build_manifest_bytes = (
            json.dumps(build_manifest_doc, indent=2, sort_keys=True) + "\n"
        ).encode()
        build_manifest_identity = {
            "sha256": hashlib.sha256(build_manifest_bytes).hexdigest(),
            "size": len(build_manifest_bytes),
        }
        subject_binary_bytes = b"#!/bin/sh\nexit 0\n"
        subject_binary_identity = {
            "sha256": hashlib.sha256(subject_binary_bytes).hexdigest(),
            "size": len(subject_binary_bytes),
        }
        build_target = "aarch64-apple-darwin"
        build_receipt_doc = build_receipt_document(
            created_utc=timestamp,
            git_head=fixed_head,
            target_triple=build_target,
            cargo_target_dir="/capture/build/target",
            toolchain={
                "rustc_verbose_version": "rustc self-test\nhost: aarch64-apple-darwin\n",
                "cargo_version": "cargo self-test",
                "rch_version": "rch self-test",
            },
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
            source_manifest_path="/capture/build/source_input_manifest.json",
            source_manifest_identity=build_manifest_identity,
            source_manifest=build_manifest_doc,
            binary_path="/capture/build/target/aarch64-apple-darwin/release-perf/focr",
            binary_identity=subject_binary_identity,
        )
        build_receipt_bytes = (
            json.dumps(build_receipt_doc, indent=2, sort_keys=True) + "\n"
        ).encode()
        build_receipt_identity = {
            "sha256": hashlib.sha256(build_receipt_bytes).hexdigest(),
            "size": len(build_receipt_bytes),
        }

        def write_test_source_pack(
            path: Path,
            manifest: dict,
            contents: dict[tuple[str, str], bytes],
            *,
            ordered_entries: list[dict] | None = None,
            header_patch: dict | None = None,
            content_overrides: dict[tuple[str, str], bytes] | None = None,
            metadata_overrides: dict[tuple[str, str], dict] | None = None,
            trailing: bytes = b"",
        ) -> None:
            entries = ordered_entries or manifest["entries"]
            total = sum(entry["size"] for entry in manifest["entries"])
            header_doc = {
                "schema": SOURCE_PACK_SCHEMA,
                "root_sha256": manifest["root_sha256"],
                "entry_count": len(manifest["entries"]),
                "total_content_bytes": total,
            }
            if header_patch is not None:
                header_doc.update(header_patch)
            header = json.dumps(
                header_doc,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            with path.open("wb") as output:
                output.write(SOURCE_PACK_DOMAIN)
                output.write(len(header).to_bytes(8, "big"))
                output.write(header)
                for entry in entries:
                    identity = (entry["repository"], entry["path"])
                    metadata_entry = (
                        metadata_overrides.get(identity, entry)
                        if metadata_overrides is not None
                        else entry
                    )
                    metadata = json.dumps(
                        metadata_entry, sort_keys=True, separators=(",", ":")
                    ).encode("ascii")
                    content = (
                        content_overrides.get(identity, contents[identity])
                        if content_overrides is not None
                        else contents[identity]
                    )
                    output.write(SOURCE_PACK_RECORD_DOMAIN)
                    output.write(len(metadata).to_bytes(8, "big"))
                    output.write(metadata)
                    output.write(len(content).to_bytes(8, "big"))
                    output.write(content)
                output.write(SOURCE_PACK_TRAILER_DOMAIN)
                output.write(len(manifest["entries"]).to_bytes(8, "big"))
                output.write(total.to_bytes(8, "big"))
                output.write(bytes.fromhex(manifest["root_sha256"]))
                output.write(trailing)

        canonical_source_contents = {
            ("workspace", relative): content
            for relative, content in build_source_contents.items()
        }
        expected_reference_files = [
            {"path": path, "bytes": size, "sha256": sha256}
            for path, size, sha256 in UNLIMITED_REFERENCE_MODEL_FILES
        ]
        reference_model_doc = {
            "schema": REFERENCE_MODEL_MANIFEST_SCHEMA,
            "model_id": "baidu/Unlimited-OCR",
            "model_commit": UNLIMITED_OCR_MODEL_COMMIT,
            "synthetic": False,
            "citable": True,
            "file_count": 12,
            "root_hash_domain": REFERENCE_MODEL_ROOT_DOMAIN.decode("ascii"),
            "root_sha256": _cert_reference_model_root(expected_reference_files),
            "index": dict(UNLIMITED_REFERENCE_MODEL_INDEX),
            "files": expected_reference_files,
        }

        def make_reference_binding(ref_doc: dict) -> dict:
            entry = "gauntlet_ref_unlimited:run_stage"
            setup = "gauntlet_ref_unlimited:setup"
            model_dir = "/models/unlimited-ocr"
            output = "/capture/reference/ref_stages.json"
            text_dir = "/capture/reference/text"
            argv = [
                "scripts/gauntlet_reference.py",
                "--stage",
                "all",
                "--page",
                ref_doc["page"],
                "--model-dir",
                model_dir,
                "--backend",
                "hf",
                "--precision",
                "bf16",
                "--max-length",
                "8192",
                "--text-dir",
                text_dir,
                "--entry",
                entry,
                "--setup",
                setup,
                "--runs",
                "3",
                "--warmup",
                "1",
                "--threads",
                "8",
                "--out",
                output,
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
            ref_doc.update(
                command=argv,
                env_pins=pins,
                model=model_dir,
                max_length=8192,
                text_dir=text_dir,
                threads=8,
                precision="bf16",
                backend="hf",
                allocator="system",
                warmup=1,
                ambient_env={
                    "FOCR_REF_MAX_LENGTH": "<unset>",
                    "FOCR_REF_TEXT_DIR": "<unset>",
                },
                torch_version="2.10.0",
                transformers_version="4.57.1",
                reference_contract={
                    "schema": "focr-reference-contract/v1",
                    "id": "unlimited-ocr-hf-v1",
                    "entry": entry,
                    "torch_version": "2.10.0",
                    "transformers_version": "4.57.1",
                },
                reference_model_manifest=reference_model_doc,
            )
            source_bindings = []
            for role, callable_name, relative in (
                ("harness", "gauntlet_reference:main", "scripts/gauntlet_reference.py"),
                ("entry", entry, "scripts/gauntlet_ref_unlimited.py"),
                ("setup", setup, "scripts/gauntlet_ref_unlimited.py"),
            ):
                identity = _bounded_file_identity(
                    fixture_root / relative, PERF_MAX_SOURCE_FILE_BYTES
                )
                source_bindings.append(
                    {
                        "role": role,
                        "callable": callable_name,
                        "path": relative,
                        "bytes": identity["size"],
                        "sha256": identity["sha256"],
                    }
                )
            import gauntlet_reference as reference_writer

            writer_args = argparse.Namespace(
                entry=entry,
                setup=setup,
                page=ref_doc["page"],
                model_dir=model_dir,
                max_length=8192,
                text_dir=text_dir,
                backend="hf",
                precision="bf16",
                runs=3,
                warmup=1,
                allocator="system",
            )
            modules_cache = {
                "evidence_dir": "/capture/reference",
                "path": "ref_stages.json.hf_modules_cache",
                "effective_path": "/capture/reference/ref_stages.json.hf_modules_cache",
                "fresh": True,
            }
            writer_environment_names = tuple(pins) + (
                "FOCR_REF_MAX_LENGTH",
                "FOCR_REF_TEXT_DIR",
            )
            saved_writer_environment = {
                name: os.environ.get(name) for name in writer_environment_names
            }
            saved_argv = list(sys.argv)
            try:
                sys.argv = list(argv)
                for name, value in pins.items():
                    os.environ[name] = value
                os.environ.pop("FOCR_REF_MAX_LENGTH", None)
                os.environ.pop("FOCR_REF_TEXT_DIR", None)
                binding = reference_writer.build_inference_binding(
                    writer_args,
                    "all",
                    8,
                    ref_doc["page_sha256"],
                    ref_doc["reference_contract"],
                    reference_model_doc,
                    modules_cache,
                    source_bindings,
                    "2.10.0",
                    "4.57.1",
                )
            finally:
                sys.argv = saved_argv
                for name, value in saved_writer_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            ref_doc["reference_inference_binding"] = binding
            return binding

        def fresh_perf_fixture() -> tuple[dict, dict, dict, dict, dict]:
            row = dict(base_perf_row)
            row_doc = {
                "schema": "focr-gauntlet-row/v3",
                "created_utc": timestamp,
                "source_git_head": fixed_head,
                "source_root": build_manifest_doc["root_sha256"],
                "producer_root": fixture_producer_root,
                "producer_root_algorithm": PRODUCER_ROOT_ALGORITHM,
                "allowed_evidence_path": evidence_id,
                "rows": [row],
            }
            focr_doc = {
                "schema": "focr-gauntlet-stages/v1",
                "source": "focr",
                "created_utc": timestamp,
                "run_dir": "/capture/raw",
                "binary": "/capture/subject/release-perf/focr",
                "binary_sha256": subject_binary_identity["sha256"],
                "binary_size": subject_binary_identity["size"],
                "binary_origin": build_receipt_doc["binary"]["path"],
                "build_receipt": "/capture/subject/build_receipt.json",
                "build_receipt_sha256": build_receipt_identity["sha256"],
                "page": "/fixtures/fixture.png",
                "page_sha256": "a" * 64,
                "command": [
                    "/capture/subject/release-perf/focr",
                    "ocr",
                    "/fixtures/fixture.png",
                    "--model",
                    "/models/unlimited-ocr.focrq",
                ],
                "env_pins": {
                    "FOCR_TIMING": "1",
                    "FOCR_THREADS": "8",
                    "OMP_NUM_THREADS": "8",
                    "RAYON_NUM_THREADS": "8",
                },
                "focr_env": {"FOCR_THREADS": "8", "FOCR_TIMING": "1"},
                "precision_gate_states": {
                    name: "<unset>" for name in PRECISION_GATE_VARS
                },
                "precision": CURRENT_UNLIMITED_PRECISION,
                "decode_mode": CURRENT_UNLIMITED_DECODE_MODE,
                "quant_recipe": CURRENT_UNLIMITED_QUANT_RECIPE,
                "model": "/models/unlimited-ocr.focrq",
                "model_kind": "file",
                "model_sha256": CURRENT_UNLIMITED_MODEL_SHA256,
                "model_size": CURRENT_UNLIMITED_MODEL_SIZE,
                "threads": 8,
                "runs": 3,
                "warmup": 1,
                "synthetic": False,
                "stdout_identical_across_runs": True,
                "raw_timing": raw_timing_doc("focr", [10.0, 10.0, 10.0], tokens=10),
                "stages": [
                    {
                        "schema": "focr-gauntlet-stage/v1",
                        "source": "focr",
                        "stage": "decode_per_token",
                        "ledger_stage": True,
                        "unit": "ms",
                        "samples_ms": [10.0, 10.0, 10.0],
                        "best_ms": 10.0,
                        "p50_ms": 10.0,
                        "mean_ms": 10.0,
                        "cv_pct": 0.0,
                        "n": 3,
                        "warmup_discarded": 1,
                        "threads": 8,
                        "precision": CURRENT_UNLIMITED_PRECISION,
                        "backend": "focr",
                        "allocator": "system",
                        "tokens": 10,
                        "tokens_consistent": True,
                        "synthetic": False,
                    }
                ],
            }
            ref_doc = {
                "schema": "focr-gauntlet-stages/v1",
                "source": "reference",
                "created_utc": timestamp,
                "page": "/fixtures/fixture.png",
                "page_sha256": "a" * 64,
                "runs": 3,
                "text_sha256": correctness_reference_sha,
                "text_identical_across_runs": True,
                "synthetic": False,
                "raw_timing": raw_timing_doc(
                    "reference", [20.0, 20.0, 20.0], tokens=None
                ),
                "stages": [
                    {
                        "schema": "focr-gauntlet-stage/v1",
                        "source": "reference",
                        "stage": "decode_per_token",
                        "ledger_stage": True,
                        "unit": "ms",
                        "samples_ms": [20.0, 20.0, 20.0],
                        "best_ms": 20.0,
                        "p50_ms": 20.0,
                        "mean_ms": 20.0,
                        "cv_pct": 0.0,
                        "n": 3,
                        "warmup_discarded": 1,
                        "threads": 8,
                        "thread_proof": {"budget": 8, "torch_num_threads": 8},
                        "precision": "bf16",
                        "backend": "hf",
                        "allocator": "system",
                        "synthetic": False,
                    }
                ],
            }
            make_reference_binding(ref_doc)
            roofline_doc = {
                "schema": "focr-gauntlet-roofline/v1",
                "created_utc": timestamp,
                "arch": "unlimited-ocr",
                "precision": CURRENT_UNLIMITED_DECODE_MODE,
                "timing_precision": CURRENT_UNLIMITED_PRECISION,
                "decode_mode": CURRENT_UNLIMITED_DECODE_MODE,
                "quant_recipe": CURRENT_UNLIMITED_QUANT_RECIPE,
                "model_sha256": CURRENT_UNLIMITED_MODEL_SHA256,
                "model_size": CURRENT_UNLIMITED_MODEL_SIZE,
                "precision_gate_states": {
                    name: "<unset>" for name in PRECISION_GATE_VARS
                },
                "stages_json": "/measurements/focr_stages.json",
                "synthetic": False,
                "floors": [
                    {
                        "stage": "decode_per_token",
                        "floor_kind": "memory",
                        "floor_ms": 5.0,
                    }
                ],
            }
            return row, row_doc, focr_doc, ref_doc, roofline_doc

        def write_perf_manifest() -> None:
            names = sorted(
                path.relative_to(evidence_dir).as_posix()
                for path in evidence_dir.rglob("*")
                if path.is_file()
                and path.name != "SHA256SUMS"
                and not path.name.startswith("._")
            )
            (evidence_dir / "SHA256SUMS").write_text(
                "".join(
                    f"{_sha256_file(evidence_dir / name)}  {name}\n" for name in names
                ),
                encoding="utf-8",
            )

        def write_perf_fixture(
            row_doc: dict,
            focr_doc: dict,
            ref_doc: dict,
            roofline_doc: dict,
        ) -> None:
            subject_dir = evidence_dir / "subject"
            (subject_dir / "release-perf").mkdir(parents=True, exist_ok=True)
            (subject_dir / "build_receipt.json").write_bytes(build_receipt_bytes)
            (subject_dir / "source_input_manifest.json").write_bytes(build_manifest_bytes)
            write_test_source_pack(
                subject_dir / "source_input_pack.bin",
                build_manifest_doc,
                canonical_source_contents,
            )
            (subject_dir / "release-perf" / "focr").write_bytes(subject_binary_bytes)
            (evidence_dir / "reference_model_manifest.json").write_text(
                json.dumps(ref_doc["reference_model_manifest"], indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            (evidence_dir / "reference_inference_binding.json").write_text(
                json.dumps(ref_doc["reference_inference_binding"], indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            raw_dir = evidence_dir / "raw"
            raw_dir.mkdir(exist_ok=True)
            for run in range(1, 4):
                tag = f"run_{run:03d}"
                stdout_path = raw_dir / f"{tag}.stdout"
                stderr_path = raw_dir / f"{tag}.stderr"
                meta_path = raw_dir / f"{tag}.meta.json"
                stdout_path.write_bytes(correctness_hypothesis_bytes)
                stderr_path.write_text(
                    "[focr-timing] precision focr-mixed-ffn-int8\n"
                    "[focr-timing] weight_cache_build 1.00s\n"
                    "[focr-timing] prefill 1.00s (10 tokens)\n"
                    "[focr-timing] decode 0.10s (10 tokens, 0.010s/tok)\n",
                    encoding="utf-8",
                )
                run_meta = {
                    field: focr_doc[field]
                    for field in (
                        "command",
                        "env_pins",
                        "focr_env",
                        "precision_gate_states",
                        "binary",
                        "binary_sha256",
                        "binary_size",
                        "binary_origin",
                        "build_receipt",
                        "build_receipt_sha256",
                        "page",
                        "page_sha256",
                        "model",
                        "model_kind",
                        "model_sha256",
                        "model_size",
                        "quant_recipe",
                        "threads",
                        "warmup",
                    )
                }
                run_meta.update(
                    tag=tag,
                    exit_code=0,
                    wall_ms=1000.0 + run,
                    stdout=f"{tag}.stdout",
                    stderr=f"{tag}.stderr",
                )
                meta_path.write_text(
                    json.dumps(run_meta, sort_keys=True) + "\n", encoding="utf-8"
                )
                focr_doc["raw_timing"]["records"][run - 1]["raw_files"] = {
                    kind: {
                        "path": path.name,
                        "sha256": _sha256_file(path),
                    }
                    for kind, path in (
                        ("meta", meta_path),
                        ("stderr", stderr_path),
                        ("stdout", stdout_path),
                    )
                }

            for name, payload in (
                ("focr_stages.json", focr_doc),
                ("ref_stages.json", ref_doc),
                ("correctness_receipt.json", correctness_receipt_doc),
            ):
                (evidence_dir / name).write_text(
                    json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
                )
            correctness_reference_dir = evidence_dir / "correctness" / "reference"
            correctness_hypothesis_dir = evidence_dir / "correctness" / "hypothesis"
            correctness_reference_dir.mkdir(parents=True, exist_ok=True)
            correctness_hypothesis_dir.mkdir(parents=True, exist_ok=True)
            correctness_reference_path = correctness_reference_dir / "fixture.md"
            correctness_reference_path.write_bytes(correctness_reference_bytes)
            correctness_hypothesis_paths = []
            for run in range(1, 4):
                run_id = f"run_{run:03d}"
                hypothesis_path = correctness_hypothesis_dir / f"{run_id}.stdout"
                hypothesis_path.write_bytes(correctness_hypothesis_bytes)
                correctness_hypothesis_paths.append((run_id, hypothesis_path))
            for source, raw_timing in (
                ("focr", focr_doc["raw_timing"]),
                ("reference", ref_doc["raw_timing"]),
            ):
                (raw_dir / f"{source}_timing.json").write_text(
                    json.dumps(raw_timing, sort_keys=True) + "\n", encoding="utf-8"
                )
            roofline_doc["stages_json_sha256"] = _sha256_file(
                evidence_dir / "focr_stages.json"
            )
            (evidence_dir / "roofline.json").write_text(
                json.dumps(roofline_doc, sort_keys=True) + "\n", encoding="utf-8"
            )
            receipt_sha = _sha256_file(evidence_dir / "correctness_receipt.json")
            row_doc["rows"][0]["correctness_proof"] = (
                f"receipt=correctness_receipt.json sha256={receipt_sha} "
                "metric=cer_norm value=0.000000 max=0.250000 result=pass"
            )
            row_doc["inputs"] = {}
            for key, relative in PERF_INPUT_BINDINGS.items():
                path = evidence_dir / relative
                identity = _bounded_file_identity(
                    path,
                    PERF_MAX_SUBJECT_BINARY_BYTES
                    if key == "subject_binary"
                    else PERF_MAX_SOURCE_PACK_BYTES
                    if key == "source_input_pack"
                    else CERTIFICATION_MAX_ARTIFACT_BYTES,
                )
                row_doc["inputs"][key] = {"bundle_path": relative, **identity}
            row_doc["timing_inputs"] = {
                source: {
                    "bundle_path": f"raw/{source}_timing.json",
                    "sha256": _sha256_file(raw_dir / f"{source}_timing.json"),
                }
                for source in ("focr", "reference")
            }
            row_doc["correctness_inputs"] = {
                "schema": CORRECTNESS_INPUTS_SCHEMA,
                "reference": {
                    "bundle_path": "correctness/reference/fixture.md",
                    "sha256": _sha256_file(correctness_reference_path),
                    "bytes": len(correctness_reference_bytes),
                },
                "hypotheses": [
                    {
                        "run_id": run_id,
                        "bundle_path": (
                            f"correctness/hypothesis/{run_id}.stdout"
                        ),
                        "sha256": _sha256_file(hypothesis_path),
                        "bytes": len(correctness_hypothesis_bytes),
                    }
                    for run_id, hypothesis_path in correctness_hypothesis_paths
                ],
            }
            (evidence_dir / "row.json").write_text(
                json.dumps(row_doc, sort_keys=True) + "\n", encoding="utf-8"
            )
            (evidence_dir / "PERF_LEDGER_ROW.md").write_text(
                "| "
                + " | ".join(row_doc["rows"][0][column] for column in perf_columns)
                + " |\n",
                encoding="utf-8",
            )
            write_perf_manifest()

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        current_perf = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_current_eligible_evidence_passes",
            current_perf["ok"],
            verdict=current_perf,
        )

        def commit_fixture_paths(message: str, *relative_paths: str) -> str:
            for command in (
                ["git", "add", "--", *relative_paths],
                ["git", "commit", "-qm", message],
            ):
                result = subprocess.run(
                    command,
                    cwd=fixture_root,
                    env=_git_env(),
                    capture_output=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"cannot commit gauntlet lineage self-test fixture: {command!r}"
                    )
            return _git_output(fixture_root, "rev-parse", "HEAD")

        evidence_head = commit_fixture_paths("evidence only", evidence_id)
        descendant_perf = perf_evidence_verdict(
            perf_text_for(perf_row),
            fixture_root,
            evidence_head,
            fixed_now,
        )
        check(
            "perf-valid-evidence-only-descendant-accepted",
            descendant_perf["ok"],
            verdict=descendant_perf,
        )

        (fixture_root / "scripts/gauntlet_cert.py").write_text(
            "validator descendant tamper\n", encoding="utf-8"
        )
        validator_head = commit_fixture_paths(
            "validator tamper", "scripts/gauntlet_cert.py"
        )
        validator_tamper = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, validator_head, fixed_now
        )
        check(
            "perf-validator-descendant-tamper-rejected",
            not validator_tamper["ok"]
            and any(
                "outside" in reason
                for candidate in validator_tamper["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=validator_tamper,
        )

        (fixture_root / "scripts/gauntlet_runbook.sh").write_text(
            "config descendant tamper\n", encoding="utf-8"
        )
        config_head = commit_fixture_paths(
            "config tamper", "scripts/gauntlet_runbook.sh"
        )
        config_tamper = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, config_head, fixed_now
        )
        check(
            "perf-config-descendant-tamper-rejected",
            not config_tamper["ok"]
            and any(
                "outside" in reason
                for candidate in config_tamper["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=config_tamper,
        )

        (fixture_root / "src/lib.rs").write_text(
            "pub fn source_tamper() {}\n", encoding="utf-8"
        )
        source_tamper_head = commit_fixture_paths("source tamper", "src/lib.rs")
        source_tamper = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, source_tamper_head, fixed_now
        )
        check(
            "perf-source-descendant-tamper-rejected",
            not source_tamper["ok"]
            and any(
                "outside" in reason
                for candidate in source_tamper["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=source_tamper,
        )
        (fixture_root / "src/lib.rs").write_bytes(
            build_source_contents["src/lib.rs"]
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        row_doc["source_root"] = "f" * 64
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        source_root_tamper = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf-row-source-root-tamper-rejected",
            not source_root_tamper["ok"]
            and any(
                "source_root" in reason
                for candidate in source_root_tamper["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=source_root_tamper,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        row_doc["producer_root"] = "f" * 64
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        producer_root_tamper = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf-row-producer-root-tamper-rejected",
            not producer_root_tamper["ok"]
            and any(
                "producer_root" in reason
                for candidate in producer_root_tamper["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=producer_root_tamper,
        )

        def direct_paths() -> tuple[dict[str, Path], set[str], set[str]]:
            paths = {
                name: evidence_dir / relative
                for name, relative in PERF_INPUT_BINDINGS.items()
            }
            covered = {
                path.relative_to(evidence_dir).as_posix()
                for path in evidence_dir.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS"
            }
            raw_meta = {
                relative
                for relative in covered
                if re.fullmatch(r"raw/run_\d{3}\.meta\.json", relative)
            }
            return paths, covered, raw_meta

        producer_paths, producer_covered, producer_raw_meta = direct_paths()
        producer_build_reasons = _build_receipt_reasons(
            root=fixture_root,
            evidence_dir=evidence_dir,
            current_head=fixed_head,
            focr_payload=focr_stage_doc,
            raw_meta=producer_raw_meta,
            covered=producer_covered,
            input_paths=producer_paths,
        )
        check(
            "build-producer-document-consumed-by-cert-validator",
            not producer_build_reasons,
            reasons=producer_build_reasons,
        )

        def source_root_for_test(entries: list[dict]) -> str:
            digest = hashlib.sha256(SOURCE_ROOT_DOMAIN)
            for entry in sorted(entries, key=lambda item: (item["repository"], item["path"])):
                digest.update(str(entry["repository"]).encode())
                digest.update(b"\0")
                digest.update(str(entry["path"]).encode())
                digest.update(b"\0")
                digest.update(str(entry["size"]).encode())
                digest.update(b"\0")
                digest.update(str(entry["sha256"]).encode())
                digest.update(b"\n")
            return digest.hexdigest()

        pack_contents = {
            ("cargo-config", "cwd-ancestor-0/config.toml"): b"x=1\n",
            ("local-dep/core", "src/lib.rs"): b"same",
            ("workspace", "src/lib.rs"): b"same",
        }
        pack_entries = [
            {
                "repository": repository,
                "path": logical,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for (repository, logical), content in sorted(pack_contents.items())
        ]
        pack_manifest = {
            "entries": pack_entries,
            "root_sha256": source_root_for_test(pack_entries),
        }
        pack_contract_path = fixture_root / "source-pack-contract.bin"

        def source_pack_case(
            name: str,
            *,
            ordered_entries: list[dict] | None = None,
            header_patch: dict | None = None,
            content_overrides: dict[tuple[str, str], bytes] | None = None,
            metadata_overrides: dict[tuple[str, str], dict] | None = None,
            trailing: bytes = b"",
            mutate_bytes=None,
            expect_ok: bool = False,
        ) -> None:
            write_test_source_pack(
                pack_contract_path,
                pack_manifest,
                pack_contents,
                ordered_entries=ordered_entries,
                header_patch=header_patch,
                content_overrides=content_overrides,
                metadata_overrides=metadata_overrides,
                trailing=trailing,
            )
            if mutate_bytes is not None:
                raw = bytearray(pack_contract_path.read_bytes())
                mutate_bytes(raw)
                pack_contract_path.write_bytes(raw)
            reasons = _source_pack_reasons(pack_contract_path, pack_manifest)
            check(name, not reasons if expect_ok else bool(reasons), reasons=reasons)

        source_pack_case("cert-source-pack-valid-closure-accepted", expect_ok=True)
        source_pack_case(
            "cert-source-pack-missing-local-dep-record-rejected",
            ordered_entries=[pack_entries[0], pack_entries[2]],
        )
        source_pack_case(
            "cert-source-pack-extra-record-rejected",
            ordered_entries=[*pack_entries, pack_entries[0]],
        )
        source_pack_case(
            "cert-source-pack-duplicate-record-rejected",
            ordered_entries=[pack_entries[0], pack_entries[0], pack_entries[2]],
        )
        source_pack_case(
            "cert-source-pack-identical-byte-record-reorder-rejected",
            ordered_entries=[pack_entries[0], pack_entries[2], pack_entries[1]],
        )
        source_pack_case(
            "cert-source-pack-local-dep-byte-mutation-rejected",
            content_overrides={("local-dep/core", "src/lib.rs"): b"evil"},
        )
        source_pack_case(
            "cert-source-pack-cargo-config-byte-mutation-rejected",
            content_overrides={
                ("cargo-config", "cwd-ancestor-0/config.toml"): b"y=2\n"
            },
        )
        source_pack_case(
            "cert-source-pack-truncation-rejected",
            mutate_bytes=lambda raw: raw.__delitem__(slice(len(raw) - 1, len(raw))),
        )
        source_pack_case(
            "cert-source-pack-trailing-bytes-rejected",
            trailing=b"trailing",
        )
        source_pack_case(
            "cert-source-pack-boolean-entry-count-rejected",
            header_patch={"entry_count": True},
        )
        source_pack_case(
            "cert-source-pack-boolean-total-length-rejected",
            header_patch={"total_content_bytes": True},
        )
        first_pack_identity = (
            pack_entries[0]["repository"],
            pack_entries[0]["path"],
        )
        source_pack_case(
            "cert-source-pack-boolean-record-size-rejected",
            metadata_overrides={
                first_pack_identity: {**pack_entries[0], "size": True}
            },
        )
        source_pack_case(
            "cert-source-pack-float-record-size-rejected",
            metadata_overrides={
                first_pack_identity: {**pack_entries[0], "size": 4.0}
            },
        )

        def oversize_header(raw: bytearray) -> None:
            offset = len(SOURCE_PACK_DOMAIN)
            raw[offset : offset + 8] = (
                PERF_MAX_SOURCE_PACK_HEADER_BYTES + 1
            ).to_bytes(8, "big")

        source_pack_case(
            "cert-source-pack-oversized-header-length-rejected",
            mutate_bytes=oversize_header,
        )

        def oversize_record_metadata(raw: bytearray) -> None:
            header_offset = len(SOURCE_PACK_DOMAIN)
            header_length = int.from_bytes(raw[header_offset : header_offset + 8], "big")
            record_offset = header_offset + 8 + header_length
            length_offset = record_offset + len(SOURCE_PACK_RECORD_DOMAIN)
            raw[length_offset : length_offset + 8] = (
                PERF_MAX_SOURCE_PACK_HEADER_BYTES + 1
            ).to_bytes(8, "big")

        source_pack_case(
            "cert-source-pack-oversized-record-metadata-length-rejected",
            mutate_bytes=oversize_record_metadata,
        )

        def oversize_record_content(raw: bytearray) -> None:
            header_offset = len(SOURCE_PACK_DOMAIN)
            header_length = int.from_bytes(raw[header_offset : header_offset + 8], "big")
            record_offset = header_offset + 8 + header_length
            metadata_length_offset = record_offset + len(SOURCE_PACK_RECORD_DOMAIN)
            metadata_length = int.from_bytes(
                raw[metadata_length_offset : metadata_length_offset + 8], "big"
            )
            content_length_offset = metadata_length_offset + 8 + metadata_length
            raw[content_length_offset : content_length_offset + 8] = (
                PERF_MAX_SOURCE_FILE_BYTES + 1
            ).to_bytes(8, "big")

        source_pack_case(
            "cert-source-pack-oversized-record-content-length-rejected",
            mutate_bytes=oversize_record_content,
        )

        large_bundle_dir = fixture_root / "large-stream-bundle"
        large_perf_dir = large_bundle_dir / "artifacts/perf/selftest"
        large_pack_path = large_perf_dir / PERF_INPUT_BINDINGS["source_input_pack"]
        large_pack_path.parent.mkdir(parents=True, exist_ok=True)
        large_pack_size = CERTIFICATION_MAX_ARTIFACT_BYTES + 4096
        with large_pack_path.open("wb") as sparse_pack:
            sparse_pack.seek(large_pack_size - 1)
            sparse_pack.write(b"\0")
        os.utime(
            large_pack_path,
            (fixed_now.timestamp(), fixed_now.timestamp()),
        )
        large_physical_identity = _stream_file_identity(
            large_pack_path, PERF_MAX_SOURCE_PACK_BYTES
        )
        large_pack_relative = large_pack_path.relative_to(fixture_root).as_posix()
        large_row_path = large_perf_dir / "row.json"
        large_row_relative = large_row_path.relative_to(fixture_root).as_posix()
        def write_large_row_binding(pack_sha256: str, pack_size: int) -> None:
            large_row_inputs = {
                name: {
                    "bundle_path": relative,
                    "sha256": "0" * 64,
                    "size": 0,
                }
                for name, relative in PERF_INPUT_BINDINGS.items()
            }
            large_row_inputs["source_input_pack"].update(
                sha256=pack_sha256,
                size=pack_size,
            )
            large_row_path.write_text(
                json.dumps(
                    {
                        "schema": "focr-gauntlet-row/v3",
                        "created_utc": _timestamp_text(fixed_now),
                        "inputs": large_row_inputs,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        write_large_row_binding(
            large_physical_identity["sha256"], large_physical_identity["size"]
        )
        large_manifest, large_stale = build_evidence_manifest(
            fixture_root,
            fixed_now.timestamp(),
            evidence_paths=[large_row_relative, large_pack_relative],
            source_pack_paths={large_pack_relative},
        )
        large_entries = {
            entry["artifact"]: entry
            for entry in large_manifest
            if isinstance(entry, dict) and isinstance(entry.get("artifact"), str)
        }
        bound_large_packs = _bound_source_pack_artifacts(
            fixture_root, large_bundle_dir, large_entries
        )
        write_large_row_binding("f" * 64, large_physical_identity["size"])
        mismatched_hash_bound_packs = _bound_source_pack_artifacts(
            fixture_root, large_bundle_dir, large_entries
        )
        write_large_row_binding(
            large_physical_identity["sha256"], large_physical_identity["size"] + 1
        )
        mismatched_size_bound_packs = _bound_source_pack_artifacts(
            fixture_root, large_bundle_dir, large_entries
        )
        write_large_row_binding(
            large_physical_identity["sha256"], large_physical_identity["size"]
        )
        check(
            "bundle-large-pack-classification-requires-row-manifest-hash-binding",
            mismatched_hash_bound_packs == set(),
        )
        check(
            "bundle-large-pack-classification-requires-row-physical-size-binding",
            mismatched_size_bound_packs == set(),
        )
        with tempfile.TemporaryDirectory(
            prefix="focr-large-pack-snapshot-self-test-"
        ) as large_snapshot_tmp:
            (
                large_snapshot_reasons,
                _large_max_age,
                large_snapshot_total,
                large_streamed_paths,
            ) = _snapshot_bundle_artifacts(
                root=fixture_root,
                bundle_dir=large_bundle_dir,
                entries=large_entries,
                snapshot_root=Path(large_snapshot_tmp),
                now=fixed_now,
                issued_at=fixed_now,
                static_source_paths=set(),
                source_pack_paths=bound_large_packs,
            )
            snapshotted_large_pack = (
                Path(large_snapshot_tmp) / large_pack_relative
            )
            snapshotted_identity: dict | None = None
            if not large_snapshot_reasons:
                if snapshotted_large_pack.is_file():
                    try:
                        snapshotted_identity = _stream_file_identity(
                            snapshotted_large_pack, PERF_MAX_SOURCE_PACK_BYTES
                        )
                    except (OSError, ValueError) as error:
                        large_snapshot_reasons.append(
                            f"large source-pack snapshot is unreadable: {error}"
                        )
                else:
                    large_snapshot_reasons.append(
                        "large source-pack snapshot is missing"
                    )
        check(
            "bundle-large-bound-source-pack-streams-through-manifest-and-snapshot",
            not large_stale
            and all("error" not in entry for entry in large_manifest)
            and bound_large_packs == {large_pack_relative}
            and not large_snapshot_reasons
            and large_streamed_paths == {large_pack_relative}
            and large_snapshot_total > CERTIFICATION_MAX_ARTIFACT_BYTES
            and snapshotted_identity is not None
            and snapshotted_identity["sha256"]
            == large_entries[large_pack_relative]["sha256"]
            and snapshotted_identity["size"] == large_pack_size,
            reasons=large_snapshot_reasons,
        )
        unbound_manifest, unbound_stale = build_evidence_manifest(
            fixture_root,
            fixed_now.timestamp(),
            evidence_paths=[large_pack_relative],
        )
        check(
            "bundle-large-unbound-source-pack-keeps-generic-artifact-limit",
            _bound_source_pack_artifacts(
                fixture_root,
                large_bundle_dir,
                {large_pack_relative: large_entries[large_pack_relative]},
            )
            == set()
            and unbound_stale == [large_pack_relative]
            and len(unbound_manifest) == 1
            and str(CERTIFICATION_MAX_ARTIFACT_BYTES)
            in str(unbound_manifest[0].get("error")),
        )
        limited_manifest, limited_stale = build_evidence_manifest(
            fixture_root,
            fixed_now.timestamp(),
            evidence_paths=[large_pack_relative],
            source_pack_paths={large_pack_relative},
            max_bundle_bytes=large_pack_size - 1,
        )
        check(
            "bundle-large-source-pack-total-size-limit-fails-closed",
            limited_stale == [large_pack_relative]
            and limited_manifest
            == [
                {
                    "artifact": large_pack_relative,
                    "error": "bundle artifacts exceed the total size limit",
                }
            ],
        )

        def build_contract_case(
            name: str,
            *,
            mutate_receipt=None,
            mutate_manifest=None,
            mutate_focr=None,
            mutate_meta=None,
            binary_bytes: bytes | None = None,
            duplicate_receipt: bool = False,
            rebind_manifest: bool = False,
            required_reason: str | None = None,
        ) -> None:
            perf_row, row_doc, focr_doc, ref_doc, roofline_doc = fresh_perf_fixture()
            write_perf_fixture(row_doc, focr_doc, ref_doc, roofline_doc)
            receipt = json.loads(json.dumps(build_receipt_doc))
            manifest = json.loads(json.dumps(build_manifest_doc))
            if mutate_manifest is not None:
                mutate_manifest(manifest)
            if mutate_receipt is not None:
                mutate_receipt(receipt)
            manifest_content = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode()
            if rebind_manifest:
                receipt["source_manifest"].update(
                    sha256=hashlib.sha256(manifest_content).hexdigest(),
                    size=len(manifest_content),
                    root_sha256=manifest.get("root_sha256"),
                    entry_count=manifest.get("entry_count"),
                )
            paths, covered, raw_meta = direct_paths()
            paths["source_input_manifest"].write_bytes(manifest_content)
            if duplicate_receipt:
                paths["build_receipt"].write_text(
                    '{"schema":"focr-build-receipt/v1",'
                    '"schema":"focr-build-receipt/v1"}\n',
                    encoding="utf-8",
                )
            else:
                paths["build_receipt"].write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if binary_bytes is not None:
                paths["subject_binary"].write_bytes(binary_bytes)
            if mutate_focr is not None:
                mutate_focr(focr_doc)
            if mutate_meta is not None:
                first_meta = evidence_dir / "raw" / "run_001.meta.json"
                meta = json.loads(first_meta.read_text(encoding="utf-8"))
                mutate_meta(meta)
                first_meta.write_text(json.dumps(meta) + "\n", encoding="utf-8")
            reasons = _build_receipt_reasons(
                root=fixture_root,
                evidence_dir=evidence_dir,
                current_head=fixed_head,
                focr_payload=focr_doc,
                raw_meta=raw_meta,
                covered=covered,
                input_paths=paths,
            )
            check(
                name,
                bool(reasons)
                and (
                    required_reason is None
                    or any(required_reason in reason for reason in reasons)
                ),
                reasons=reasons,
            )

        build_contract_case(
            "cert-build-missing-field-rejected",
            mutate_receipt=lambda value: value.pop("profile"),
        )
        build_contract_case(
            "cert-build-legacy-schema-rejected",
            mutate_receipt=lambda value: value.update(schema="focr-build-receipt/v0"),
        )
        build_contract_case(
            "cert-build-extra-field-rejected",
            mutate_receipt=lambda value: value.update(extra=True),
        )
        build_contract_case(
            "cert-build-duplicate-key-rejected",
            duplicate_receipt=True,
        )
        build_contract_case(
            "cert-build-uppercase-source-hash-rejected",
            mutate_manifest=lambda value: value["entries"][0].update(
                sha256=value["entries"][0]["sha256"].upper()
            ),
        )
        build_contract_case(
            "cert-build-bad-source-size-rejected",
            mutate_manifest=lambda value: value["entries"][0].update(size=True),
        )
        build_contract_case(
            "cert-build-root-drift-rejected",
            mutate_manifest=lambda value: value.update(root_sha256="0" * 64),
            rebind_manifest=True,
        )

        def forge_head(receipt: dict) -> None:
            receipt["git_head"] = "f" * 40

        def forge_manifest_head(manifest: dict) -> None:
            manifest["repositories"][0]["git_head"] = "f" * 40

        build_contract_case(
            "cert-build-head-drift-rejected",
            mutate_receipt=forge_head,
            mutate_manifest=forge_manifest_head,
            rebind_manifest=True,
        )
        build_contract_case(
            "cert-build-binary-drift-rejected",
            binary_bytes=b"forged binary\n",
        )

        def forge_source_entry(manifest: dict) -> None:
            source_entry = next(
                entry for entry in manifest["entries"] if entry["path"] == "src/lib.rs"
            )
            source_entry["sha256"] = "0" * 64
            manifest["root_sha256"] = source_root_for_test(manifest["entries"])

        build_contract_case(
            "cert-build-fully-rehashed-source-forgery-rejected",
            mutate_manifest=forge_source_entry,
            rebind_manifest=True,
        )

        def forge_workspace_path(manifest: dict) -> None:
            source_entry = next(
                entry for entry in manifest["entries"] if entry["path"] == "src/lib.rs"
            )
            source_entry["path"] = "src/nonexistent.rs"
            source_entry["sha256"] = hashlib.sha256(b"forged").hexdigest()
            source_entry["size"] = len(b"forged")
            manifest["entries"].sort(
                key=lambda item: (item["repository"], item["path"])
            )
            manifest["root_sha256"] = source_root_for_test(manifest["entries"])

        build_contract_case(
            "cert-build-fully-rehashed-workspace-path-forgery-rejected",
            mutate_manifest=forge_workspace_path,
            rebind_manifest=True,
            required_reason="current workspace source entry drifted",
        )

        def inject_local_dep_without_pack_record(manifest: dict) -> None:
            content = b"evil local dependency\n"
            manifest["repositories"].append(
                {
                    "id": "local-dep/evil",
                    "path": str((fixture_root / "nonexistent-evil-dep").resolve()),
                    "git_head": "e" * 40,
                    "packages": ["evil"],
                    "selectors": ["src"],
                }
            )
            manifest["entries"].append(
                {
                    "repository": "local-dep/evil",
                    "path": "src/lib.rs",
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
            manifest["entries"].sort(
                key=lambda item: (item["repository"], item["path"])
            )
            manifest["entry_count"] = len(manifest["entries"])
            manifest["root_sha256"] = source_root_for_test(manifest["entries"])

        build_contract_case(
            "cert-build-rehashed-local-dep-injection-without-pack-record-rejected",
            mutate_manifest=inject_local_dep_without_pack_record,
            rebind_manifest=True,
            required_reason="source pack",
        )

        def inject_cargo_config_without_pack_record(manifest: dict) -> None:
            content = b"[build]\nrustflags=[]\n"
            logical = "cargo-home/config.toml"
            manifest["cargo_config_files"].append(
                {
                    "logical_path": logical,
                    "physical_path": str(
                        (fixture_root / "nonexistent-cargo-home" / "config.toml").resolve()
                    ),
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
            manifest["entries"].append(
                {
                    "repository": "cargo-config",
                    "path": logical,
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
            manifest["entries"].sort(
                key=lambda item: (item["repository"], item["path"])
            )
            manifest["entry_count"] = len(manifest["entries"])
            manifest["root_sha256"] = source_root_for_test(manifest["entries"])

        build_contract_case(
            "cert-build-rehashed-cargo-config-injection-without-pack-record-rejected",
            mutate_manifest=inject_cargo_config_without_pack_record,
            rebind_manifest=True,
            required_reason="source pack",
        )
        build_contract_case(
            "cert-build-raw-meta-drift-rejected",
            mutate_meta=lambda value: value.update(binary_size=value["binary_size"] + 1),
        )
        build_contract_case(
            "cert-build-receipt-reuse-rejected",
            mutate_focr=lambda value: value.update(
                build_receipt="/other-capture/subject/build_receipt.json"
            ),
        )
        build_contract_case(
            "cert-build-subject-path-escape-rejected",
            mutate_focr=lambda value: value.update(binary="/capture/../escape/focr"),
        )

        def refresh_inference_hash(binding: dict) -> None:
            unsigned = dict(binding)
            unsigned.pop("binding_hash_domain", None)
            unsigned.pop("binding_sha256", None)
            canonical = json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
            binding["binding_hash_domain"] = REFERENCE_INFERENCE_BINDING_DOMAIN.decode("ascii")
            binding["binding_sha256"] = hashlib.sha256(
                REFERENCE_INFERENCE_BINDING_DOMAIN + canonical
            ).hexdigest()

        def reference_contract_case(
            name: str,
            *,
            mutate_manifest=None,
            mutate_binding=None,
            embed_manifest: bool = True,
            embed_binding: bool = True,
            rehash_binding: bool = False,
        ) -> None:
            perf_row, row_doc, focr_doc, ref_doc, roofline_doc = fresh_perf_fixture()
            write_perf_fixture(row_doc, focr_doc, ref_doc, roofline_doc)
            manifest = json.loads(json.dumps(reference_model_doc))
            binding = json.loads(json.dumps(ref_doc["reference_inference_binding"]))
            if mutate_manifest is not None:
                mutate_manifest(manifest)
            if mutate_binding is not None:
                mutate_binding(binding, ref_doc)
            if rehash_binding:
                refresh_inference_hash(binding)
            if embed_manifest:
                ref_doc["reference_model_manifest"] = json.loads(json.dumps(manifest))
            if embed_binding:
                ref_doc["reference_inference_binding"] = json.loads(json.dumps(binding))
            paths, covered, _raw_meta = direct_paths()
            paths["reference_model_manifest"].write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            paths["reference_inference_binding"].write_text(
                json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            reasons = _reference_provenance_reasons(
                root=fixture_root,
                row=perf_row,
                ref_payload=ref_doc,
                input_paths=paths,
                covered=covered,
            )
            check(name, bool(reasons), reasons=reasons)

        reference_contract_case(
            "cert-reference-commit-drift-rejected",
            mutate_manifest=lambda value: value.update(model_commit="f" * 40),
        )
        reference_contract_case(
            "cert-reference-root-drift-rejected",
            mutate_manifest=lambda value: value.update(root_sha256="0" * 64),
        )

        def forge_reference_shard(manifest: dict) -> None:
            shard = next(
                item
                for item in manifest["files"]
                if item["path"] == "model-00001-of-000001.safetensors"
            )
            shard["sha256"] = "0" * 64
            manifest["root_sha256"] = _cert_reference_model_root(manifest["files"])

        reference_contract_case(
            "cert-reference-fully-rehashed-shard-forgery-rejected",
            mutate_manifest=forge_reference_shard,
        )
        reference_contract_case(
            "cert-reference-index-census-drift-rejected",
            mutate_manifest=lambda value: value["index"].update(weight_count=2709),
        )

        def forge_inference_source(binding: dict, _ref: dict) -> None:
            binding["sources"][0]["sha256"] = "0" * 64

        reference_contract_case(
            "cert-reference-inference-source-drift-rejected",
            mutate_binding=forge_inference_source,
            rehash_binding=True,
        )

        def forge_inference_argv(binding: dict, ref_doc: dict) -> None:
            binding["argv"].append("--bogus")
            ref_doc["command"] = list(binding["argv"])

        reference_contract_case(
            "cert-reference-inference-argv-drift-rejected",
            mutate_binding=forge_inference_argv,
            rehash_binding=True,
        )

        def forge_reference_max_length(binding: dict, ref_doc: dict) -> None:
            binding["max_length"] = 4096
            option = binding["argv"].index("--max-length") + 1
            binding["argv"][option] = "4096"
            ref_doc["max_length"] = 4096
            ref_doc["command"] = list(binding["argv"])

        reference_contract_case(
            "cert-reference-explicit-max-length-drift-rejected",
            mutate_binding=forge_reference_max_length,
            rehash_binding=True,
        )

        def forge_reference_ambient(
            binding: dict, ref_doc: dict, name: str, value: str
        ) -> None:
            binding["ambient_env"][name] = value
            ref_doc["ambient_env"][name] = value

        reference_contract_case(
            "cert-reference-ambient-max-length-override-rejected",
            mutate_binding=lambda binding, ref_doc: forge_reference_ambient(
                binding, ref_doc, "FOCR_REF_MAX_LENGTH", "4096"
            ),
            rehash_binding=True,
        )
        reference_contract_case(
            "cert-reference-ambient-text-dir-override-rejected",
            mutate_binding=lambda binding, ref_doc: forge_reference_ambient(
                binding, ref_doc, "FOCR_REF_TEXT_DIR", "/tmp/ambient-text"
            ),
            rehash_binding=True,
        )
        reference_contract_case(
            "cert-reference-embedded-vs-bundled-drift-rejected",
            mutate_manifest=lambda value: value.update(root_sha256="0" * 64),
            embed_manifest=False,
        )

        correctness_hypothesis_runs = {
            f"run_{index:03d}": correctness_hypothesis_bytes
            for index in range(1, 4)
        }

        def cert_correctness_rejected(name: str, patch) -> None:
            candidate = json.loads(json.dumps(correctness_receipt_doc))
            patch(candidate)
            try:
                _validate_correctness_receipt_payload(
                    candidate,
                    reference_bytes=correctness_reference_bytes,
                    hypothesis_runs=correctness_hypothesis_runs,
                    focr=focr_stage_doc,
                    ref=ref_stage_doc,
                )
                check(name, False)
            except CorrectnessValidationError:
                check(name, True)

        for name, patch in (
            ("cert_correctness_legacy_schema_rejected", lambda d: d.pop("schema")),
            ("cert_correctness_extra_field_rejected", lambda d: d.update(extra=True)),
            (
                "cert_correctness_metric_formula_rejected",
                lambda d: d["metric_formulas"].update(cer_norm="claimed"),
            ),
            (
                "cert_correctness_source_hash_rejected",
                lambda d: d["pages"][0]["reference"]["source"].update(
                    sha256="0" * 64
                ),
            ),
            (
                "cert_correctness_source_counts_rejected",
                lambda d: d["pages"][0]["reference"]["source"].update(bytes=1),
            ),
            (
                "cert_correctness_normalized_binding_rejected",
                lambda d: d["pages"][0]["reference"]["source"].update(
                    normalized_sha256="0" * 64
                ),
            ),
            (
                "cert_correctness_transform_rejected",
                lambda d: d["pages"][0]["reference"]["transform"].update(
                    name="identity-v1"
                ),
            ),
            (
                "cert_correctness_scored_binding_rejected",
                lambda d: d["pages"][0]["reference"]["scored"].update(chars=1),
            ),
            (
                "cert_correctness_edit_distance_rejected",
                lambda d: d["pages"][0].update(normalized_edit_distance=1),
            ),
            (
                "cert_correctness_page_cer_rejected",
                lambda d: d["pages"][0].update(cer_norm=0.01),
            ),
            (
                "cert_correctness_aggregate_integer_rejected",
                lambda d: d["aggregate"].update(normalized_reference_chars=1),
            ),
            (
                "cert_correctness_aggregate_cer_rejected",
                lambda d: d["aggregate"].update(cer_norm=0.01),
            ),
        ):
            cert_correctness_rejected(name, patch)

        (evidence_dir / "correctness" / "reference" / "fixture.md").write_bytes(
            b"tampered reference\n"
        )
        write_perf_manifest()
        physical_correctness_tamper = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "cert_correctness_physical_source_tamper_rejected",
            not physical_correctness_tamper["ok"]
            and any(
                "correctness reference source binding" in reason
                for reason in physical_correctness_tamper["candidates"][0]["reasons"]
            ),
            verdict=physical_correctness_tamper,
        )

        (
            aggregate_row,
            aggregate_row_doc,
            aggregate_focr,
            aggregate_ref,
            aggregate_roofline,
        ) = fresh_perf_fixture()
        aggregate_focr["stages"][0].update(
            samples_ms=[9.0, 9.0, 9.0],
            best_ms=9.0,
            p50_ms=9.0,
            mean_ms=9.0,
            cv_pct=0.0,
        )
        aggregate_row.update(focr_ms="9.000", ratio="2.222", dist_above_floor="1.80")
        write_perf_fixture(
            aggregate_row_doc,
            aggregate_focr,
            aggregate_ref,
            aggregate_roofline,
        )
        aggregate_contradiction = perf_evidence_verdict(
            perf_text_for(aggregate_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_aggregate_contradicting_raw_focr_timing_rejected",
            not aggregate_contradiction["ok"]
            and any(
                "aggregate samples are not raw-derived" in reason
                or "aggregate timing contradicts physical stderr" in reason
                for reason in aggregate_contradiction["candidates"][0]["reasons"]
            ),
            verdict=aggregate_contradiction,
        )

        raw_row, raw_row_doc, raw_focr, raw_ref, raw_roofline = fresh_perf_fixture()
        raw_ref["raw_timing"]["records"][0]["stages"]["decode_per_token"][
            "ms"
        ] = 99.0
        write_perf_fixture(raw_row_doc, raw_focr, raw_ref, raw_roofline)
        raw_contradiction = perf_evidence_verdict(
            perf_text_for(raw_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_raw_reference_timing_contradicting_aggregate_rejected",
            not raw_contradiction["ok"]
            and any(
                "reference aggregate samples are not raw-derived" in reason
                for reason in raw_contradiction["candidates"][0]["reasons"]
            ),
            verdict=raw_contradiction,
        )

        for name, mutate in (
            (
                "perf_missing_reference_per_run_text_hash_rejected",
                lambda doc: doc["raw_timing"]["records"][0].pop("text_sha256"),
            ),
            (
                "perf_invalid_reference_per_run_text_hash_rejected",
                lambda doc: doc["raw_timing"]["records"][0].update(
                    text_sha256="invalid"
                ),
            ),
            (
                "perf_drifting_reference_per_run_text_hash_rejected",
                lambda doc: doc["raw_timing"]["records"][0].update(
                    text_sha256="cd" * 32
                ),
            ),
            (
                "perf_list_valued_reference_top_level_text_hash_rejected",
                lambda doc: doc.update(text_sha256=["ab" * 32]),
            ),
        ):
            _row, _row_doc, _focr, bad_ref, _roofline = fresh_perf_fixture()
            mutate(bad_ref)
            text_reasons, _samples = _raw_timing_reasons(
                bad_ref, bad_ref["stages"][0], source="reference"
            )
            check(name, bool(text_reasons), reasons=text_reasons)

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)

        direct_contract = _current_unlimited_measurement_reasons(
            focr_stage_doc,
            focr_stage_doc["stages"][0],
            perf_row,
        )
        check(
            "perf_current_precision_contract_passes_directly",
            not direct_contract,
            reasons=direct_contract,
        )
        direct_roofline = _current_unlimited_roofline_reasons(
            roofline_doc,
            focr_stage_doc,
            focr_stages_sha256=roofline_doc["stages_json_sha256"],
        )
        check(
            "perf_current_roofline_contract_passes_directly",
            not direct_roofline,
            reasons=direct_roofline,
        )

        explicitly_falsy = json.loads(json.dumps(focr_stage_doc))
        explicitly_falsy["precision_gate_states"].update(
            FOCR_DECODE_INT8="0",
            FOCR_INT8_ATTN="false",
            FOCR_INT8_LMHEAD="off",
        )
        explicitly_falsy["focr_env"].update(
            FOCR_DECODE_INT8="0",
            FOCR_INT8_ATTN="false",
            FOCR_INT8_LMHEAD="off",
        )
        explicitly_falsy_row = dict(perf_row)
        explicitly_falsy_row["fallback/kill-switch state"] = str(
            _canonical_kill_switch_cell(explicitly_falsy)
        )
        explicitly_falsy_reasons = _current_unlimited_measurement_reasons(
            explicitly_falsy,
            explicitly_falsy["stages"][0],
            explicitly_falsy_row,
        )
        check(
            "perf_explicit_falsy_oq14_gates_are_unambiguous",
            not explicitly_falsy_reasons,
            reasons=explicitly_falsy_reasons,
        )

        def current_measurement_mutation(
            mutate,
        ) -> list[str]:
            bad = json.loads(json.dumps(focr_stage_doc))
            bad_row = dict(perf_row)
            mutate(bad, bad_row)
            return _current_unlimited_measurement_reasons(
                bad,
                bad["stages"][0],
                bad_row,
            )

        measurement_mutations = (
            (
                "perf_rejects_historical_focr_int8_precision",
                lambda doc, _row: (
                    doc.update(precision="focr-int8"),
                    doc["stages"][0].update(precision="focr-int8"),
                ),
                "historical or ambiguous",
            ),
            (
                "perf_rejects_full_int8_precision",
                lambda doc, _row: (
                    doc.update(precision="focr-full-int8"),
                    doc["stages"][0].update(precision="focr-full-int8"),
                ),
                "historical or ambiguous",
            ),
            (
                "perf_rejects_wrong_decode_mode",
                lambda doc, _row: doc.update(decode_mode="full-int8"),
                "decode_mode",
            ),
            (
                "perf_rejects_wrong_quant_recipe",
                lambda doc, _row: doc.update(quant_recipe="decoder-ffn-int8-v1"),
                "quant_recipe",
            ),
            (
                "perf_rejects_wrong_model_hash",
                lambda doc, _row: doc.update(model_sha256="0" * 64),
                "model hash/size",
            ),
            (
                "perf_rejects_wrong_model_size",
                lambda doc, _row: doc.update(
                    model_size=CURRENT_UNLIMITED_MODEL_SIZE + 1
                ),
                "model hash/size",
            ),
            (
                "perf_rejects_ambiguous_falsy_gate",
                lambda doc, _row: doc["precision_gate_states"].update(
                    FOCR_INT8_ATTN="banana"
                ),
                "falsy or unset",
            ),
            (
                "perf_rejects_present_falsy_kv_switch",
                lambda doc, _row: (
                    doc["precision_gate_states"].update(FOCR_INT8_KV="0"),
                    doc["focr_env"].update(FOCR_INT8_KV="0"),
                ),
                "presence-only switch",
            ),
        )
        for name, mutate, expected_reason in measurement_mutations:
            mutation_reasons = current_measurement_mutation(mutate)
            check(
                name,
                any(expected_reason in reason for reason in mutation_reasons),
                reasons=mutation_reasons,
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
                    **focr_stage_doc["precision_gate_states"],
                    "FOCR_INT8_ATTN": "1",
                },
            ),
            ("stages_json_sha256", "0" * 64),
        ):
            bad_roofline = json.loads(json.dumps(roofline_doc))
            bad_roofline[field] = value
            roofline_reasons = _current_unlimited_roofline_reasons(
                bad_roofline,
                focr_stage_doc,
                focr_stages_sha256=roofline_doc["stages_json_sha256"],
            )
            check(
                f"perf_rejects_wrong_roofline_{field}",
                any(field in reason for reason in roofline_reasons),
                reasons=roofline_reasons,
            )

        raw_stderr = {f"raw/run_{run:03d}.stderr" for run in range(1, 4)}
        raw_marker_reasons = _runtime_decode_marker_reasons(
            evidence_dir, raw_stderr
        )
        check(
            "perf_raw_runtime_mode_markers_pass",
            not raw_marker_reasons,
            reasons=raw_marker_reasons,
        )
        raw_meta = {f"raw/run_{run:03d}.meta.json" for run in range(1, 4)}
        raw_stdout = {f"raw/run_{run:03d}.stdout" for run in range(1, 4)}
        raw_meta_reasons = _raw_run_metadata_reasons(
            evidence_dir,
            raw_meta,
            raw_stdout,
            raw_stderr,
            focr_stage_doc,
            focr_stage_doc["stages"][0],
        )
        check(
            "perf_raw_run_metadata_passes",
            not raw_meta_reasons,
            reasons=raw_meta_reasons,
        )
        first_meta_path = evidence_dir / "raw" / "run_001.meta.json"
        first_meta = json.loads(first_meta_path.read_text(encoding="utf-8"))
        first_meta["model_sha256"] = "0" * 64
        first_meta_path.write_text(
            json.dumps(first_meta, sort_keys=True) + "\n", encoding="utf-8"
        )
        forged_meta_reasons = _raw_run_metadata_reasons(
            evidence_dir,
            raw_meta,
            raw_stdout,
            raw_stderr,
            focr_stage_doc,
            focr_stage_doc["stages"][0],
        )
        check(
            "perf_raw_run_metadata_subject_mutation_rejected",
            any(
                "model_sha256" in reason and "disagrees with aggregate" in reason
                for reason in forged_meta_reasons
            ),
            reasons=forged_meta_reasons,
        )
        first_meta["model_sha256"] = focr_stage_doc["model_sha256"]
        first_meta_path.write_text(
            json.dumps(first_meta, sort_keys=True) + "\n", encoding="utf-8"
        )
        (evidence_dir / "raw" / "run_001.stderr").write_text(
            "[focr-timing] precision focr-full-int8\n"
            "[focr-timing] weight_cache_build_i8 1.00s\n"
            "[focr-timing] prefill_i8 1.00s (10 tokens)\n"
            "[focr-timing] decode_i8 1.00s (10 tokens, 0.100s/tok)\n",
            encoding="utf-8",
        )
        forged_runtime_reasons = _runtime_decode_marker_reasons(
            evidence_dir, raw_stderr
        )
        check(
            "perf_raw_full_int8_runtime_markers_rejected",
            any("runtime precision marker" in reason for reason in forged_runtime_reasons)
            and any("full-int8" in reason for reason in forged_runtime_reasons),
            reasons=forged_runtime_reasons,
        )

        (
            historical_row,
            historical_row_doc,
            historical_focr,
            historical_ref,
            historical_roofline,
        ) = fresh_perf_fixture()
        historical_row["precision (focr vs ref)"] = "focr-int8 vs hf-bf16"
        historical_focr.update(precision="focr-int8")
        historical_focr["stages"][0].update(precision="focr-int8")
        write_perf_fixture(
            historical_row_doc,
            historical_focr,
            historical_ref,
            historical_roofline,
        )
        historical_perf = perf_evidence_verdict(
            perf_text_for(historical_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_historical_rows_parse_but_cannot_certify_current_release",
            len(historical_perf["candidates"]) == 1
            and not historical_perf["ok"]
            and any(
                "historical or ambiguous" in reason
                for reason in historical_perf["candidates"][0]["reasons"]
            ),
            verdict=historical_perf,
        )

        forged_row, forged_row_doc, forged_focr, forged_ref, forged_roofline = (
            fresh_perf_fixture()
        )
        forged_focr.update(source="wrong", synthetic=True, page="/fixtures/other.png")
        forged_focr["stages"][0].update(
            synthetic=True,
            cv_pct=-3.0,
            precision="",
            allocator="",
        )
        forged_ref["stages"][0].pop("thread_proof")
        forged_roofline["synthetic"] = True
        write_perf_fixture(forged_row_doc, forged_focr, forged_ref, forged_roofline)
        forged_perf = perf_evidence_verdict(
            perf_text_for(forged_row), fixture_root, fixed_head, fixed_now
        )
        forged_reasons = forged_perf["candidates"][0]["reasons"]
        check(
            "perf_hash_valid_synthetic_forgery_rejected",
            not forged_perf["ok"]
            and any("source identity" in reason for reason in forged_reasons)
            and any("synthetic" in reason for reason in forged_reasons)
            and any("fixture pages" in reason for reason in forged_reasons)
            and any("thread proof" in reason for reason in forged_reasons)
            and any(
                "roofline evidence is synthetic" in reason for reason in forged_reasons
            ),
            verdict=forged_perf,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)

        tampered_ledger_row = dict(perf_row)
        tampered_ledger_row["focr_ms"] = "9.000"
        tampered_ledger = perf_evidence_verdict(
            perf_text_for(tampered_ledger_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_ledger_timing_tamper_rejected",
            not tampered_ledger["ok"]
            and any(
                "ledger row differs from hashed row.json" in reason
                for reason in tampered_ledger["candidates"][0]["reasons"]
            ),
            verdict=tampered_ledger,
        )

        tampered_correctness_row = dict(perf_row)
        tampered_correctness_row["correctness_proof"] = "FAILED: token mismatch"
        tampered_correctness = perf_evidence_verdict(
            perf_text_for(tampered_correctness_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_ledger_failure_correctness_text_rejected",
            not tampered_correctness["ok"]
            and any(
                "not a structured hash-bound pass receipt" in reason
                for reason in tampered_correctness["candidates"][0]["reasons"]
            ),
            verdict=tampered_correctness,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        focr_stage_doc["created_utc"] = "2026-07-08T00:00:00Z"
        ref_stage_doc["created_utc"] = "2026-07-08T00:00:00Z"
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        stale_perf = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_fresh_row_cannot_rewrap_stale_stage_evidence",
            not stale_perf["ok"]
            and any(
                "stale" in reason
                for candidate in stale_perf["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=stale_perf,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        focr_stage_doc["stages"][0]["cv_pct"] = 8.0
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        noisy_perf = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_ineligible_cv_fails_closed",
            not noisy_perf["ok"]
            and any(
                "cv_pct" in reason
                for candidate in noisy_perf["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=noisy_perf,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        focr_stage_doc["stages"][0]["best_ms"] = 11.0
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        stage_mismatch = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_hashed_row_must_match_hashed_stage_timing",
            not stage_mismatch["ok"]
            and any(
                "hashed row focr_ms does not match stage best_ms" in reason
                for reason in stage_mismatch["candidates"][0]["reasons"]
            ),
            verdict=stage_mismatch,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        perf_row["correctness_proof"] = "outputs differ from reference; CER=0.91"
        (evidence_dir / "row.json").write_text(
            json.dumps(row_doc, sort_keys=True) + "\n", encoding="utf-8"
        )
        (evidence_dir / "PERF_LEDGER_ROW.md").write_text(
            "| " + " | ".join(perf_row[column] for column in perf_columns) + " |\n",
            encoding="utf-8",
        )
        write_perf_manifest()
        hashed_failure = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_rehashed_failure_correctness_proof_rejected",
            not hashed_failure["ok"]
            and any(
                reason.startswith(
                    "hashed row: correctness proof is not a structured hash-bound pass receipt"
                )
                for reason in hashed_failure["candidates"][0]["reasons"]
            ),
            verdict=hashed_failure,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        row_doc["rows"].append(dict(perf_row))
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        duplicate_row = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_duplicate_hashed_claim_row_rejected",
            not duplicate_row["ok"]
            and any(
                "exactly one ledger claim/stage row" in reason
                for reason in duplicate_row["candidates"][0]["reasons"]
            ),
            verdict=duplicate_row,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        row_doc["inputs"]["focr_stages"]["sha256"] = "0" * 64
        (evidence_dir / "row.json").write_text(
            json.dumps(row_doc, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_perf_manifest()
        bad_input_binding = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, fixed_head, fixed_now
        )
        check(
            "perf_row_input_sha_mismatch_rejected",
            not bad_input_binding["ok"]
            and any(
                "inputs.focr_stages does not bind" in reason
                for reason in bad_input_binding["candidates"][0]["reasons"]
            ),
            verdict=bad_input_binding,
        )

        perf_row, row_doc, focr_stage_doc, ref_stage_doc, roofline_doc = (
            fresh_perf_fixture()
        )
        write_perf_fixture(row_doc, focr_stage_doc, ref_stage_doc, roofline_doc)
        wrong_head_perf = perf_evidence_verdict(
            perf_text_for(perf_row), fixture_root, "f" * 40, fixed_now
        )
        check(
            "perf_wrong_head_fails_closed",
            not wrong_head_perf["ok"]
            and any(
                "git_head" in reason
                for candidate in wrong_head_perf["candidates"]
                for reason in candidate["reasons"]
            ),
            verdict=wrong_head_perf,
        )

        certificate_root = fixture_root / "bundle-case"
        custom_bundle_dir = certificate_root / "custom-cert-output"
        custom_bundle_dir.mkdir(parents=True)
        (certificate_root / "Cargo.toml").write_text(
            '[package]\nname = "franken_ocr"\nversion = "0.0.0"\n',
            encoding="utf-8",
        )
        (certificate_root / "install.ps1").write_text(
            "self-test snapshot for install.ps1\n", encoding="utf-8"
        )
        clean_worktree = {
            "status_ok": True,
            "clean": True,
            "dirty_path_count": 0,
            "status_porcelain": [],
            "status_sha256": _sha256_text(""),
        }
        dirty_worktree = {
            "status_ok": True,
            "clean": False,
            "dirty_path_count": 1,
            "status_porcelain": [" M evidence.json"],
            "status_sha256": _sha256_text(" M evidence.json\n"),
        }
        strict_relative = str(custom_bundle_dir.relative_to(certificate_root))
        strict_class_paths = {
            class_name: f"{strict_relative}/{filename}"
            for class_name, (filename, _schema) in STRICT_BUNDLE_CLASSES.items()
        }
        strict_timestamp = fixed_now.isoformat().replace("+00:00", "Z")
        source_claim_paths = (
            f"{strict_relative}/claim_sources/RELEASE_READINESS.json",
            f"{strict_relative}/claim_sources/ROUNDS.jsonl",
            *(
                f"{strict_relative}/claim_sources/{Path(path).name}"
                for path in HYPOTHESIS_LEDGER_PATHS
            ),
        )
        claim_path_by_original = {
            "docs/gauntlet/RELEASE_READINESS.json": source_claim_paths[0],
            "docs/gauntlet/ROUNDS.jsonl": source_claim_paths[1],
            **{
                original: source_claim_paths[index]
                for index, original in enumerate(HYPOTHESIS_LEDGER_PATHS, start=2)
            },
        }
        strict_core_mapping = {
            original: claim_path_by_original.get(
                original, f"{strict_relative}/core_sources/{original}"
            )
            for original in CORE_EVIDENCE_PATHS
        }
        strict_proof_mapping = {
            original: strict_core_mapping.get(
                original, f"{strict_relative}/core_sources/{original}"
            )
            for original in CERTIFICATION_PROOF_EVIDENCE_PATHS
        }
        feature_universe_relative = f"{strict_relative}/feature_universe.json"
        hypothesis_evidence_paths = (
            f"{strict_relative}/evidence/result.txt",
            f"{strict_relative}/evidence/SHA256SUMS",
        )
        audit_receipt_paths = {
            domain: f"{strict_relative}/audit_receipts/{domain}_audit_receipt.json"
            for domain in CERTIFICATION_AUDIT_DOMAINS
        }
        audit_tool_output_paths = {
            (domain, tool_id): (
                f"{strict_relative}/audit_receipts/raw/{domain}/{tool_id}.json"
            )
            for domain, tools in CERTIFICATION_AUDIT_TOOLS.items()
            for tool_id in tools
        }
        benchmark_baseline_path = strict_core_mapping[
            "benches/.bench-history/baseline.json"
        ]
        benchmark_current_path = (
            f"{strict_relative}/benchmark_inputs/current_benchmark_samples.json"
        )
        dist_evidence_path = f"{strict_relative}/dist/raw/self-test-output.json"
        dist_portability_paths = {
            target: f"{strict_relative}/dist/portability/{index}.txt"
            for index, target in enumerate(CERTIFICATION_DIST_TARGETS)
            if target
            in CERTIFICATION_LINUX_DIST_TARGETS | CERTIFICATION_WINDOWS_DIST_TARGETS
        }
        benchmark_input_paths = {
            (gate, "baseline"): benchmark_baseline_path
            for gate in CERTIFICATION_BENCHMARK_THRESHOLDS
        } | {
            (gate, "current"): benchmark_current_path
            for gate in CERTIFICATION_BENCHMARK_THRESHOLDS
        }
        model_rung_paths = {
            f"L{index}": f"{strict_relative}/model_parity/L{index}.json"
            for index in range(6)
        }
        model_rung_log_paths = {
            f"L{index}": f"{strict_relative}/model_parity/L{index}.log"
            for index in range(6)
        }
        model_fixture_path = strict_core_mapping[
            "tests/fixtures/ladder_scorecard/scorecard_armed.json"
        ]
        model_oracle_paths = {
            f"L{index}": f"{strict_relative}/model_parity/L{index}_oracle.json"
            for index in range(6)
        }
        model_subject_paths = {
            f"L{index}": f"{strict_relative}/model_parity/L{index}_subject.json"
            for index in range(6)
        }
        required_strict_artifacts = (
            *strict_class_paths.values(),
            feature_universe_relative,
            *strict_core_mapping.values(),
            *strict_proof_mapping.values(),
            *hypothesis_evidence_paths,
            *audit_receipt_paths.values(),
            *audit_tool_output_paths.values(),
            *benchmark_input_paths.values(),
            dist_evidence_path,
            *dist_portability_paths.values(),
            *model_rung_paths.values(),
            *model_rung_log_paths.values(),
            model_fixture_path,
            *model_oracle_paths.values(),
            *model_subject_paths.values(),
        )
        trusted_signers = {
            "producer@example.test": "A" * 40,
            "reviewer@example.test": "B" * 40,
            "release@example.test": "C" * 40,
        }
        strict_verifier_kwargs = {
            "trusted_signers": trusted_signers,
            "perf_evidence_checker": lambda _text, _bundle, head: head == fixed_head,
            "source_digest_resolver": lambda _root, head, original: (
                _sha256_file(
                    certificate_root
                    / ({**strict_core_mapping, **strict_proof_mapping})[original]
                )
                if head == fixed_head
                else None
            ),
            "ci_run_verifier": lambda _root, run_id, head, artifacts: (
                head == fixed_head
                and bool(artifacts)
                and (
                    (
                        run_id == "123"
                        and all(
                            item.get("source_ci_workflow")
                            == CERTIFICATION_GITHUB_WORKFLOW
                            for item in artifacts
                        )
                    )
                    or (
                        run_id == "124"
                        and all(
                            item.get("source_ci_workflow")
                            == CERTIFICATION_DIST_WORKFLOW
                            for item in artifacts
                        )
                    )
                    or (
                        run_id == "125"
                        and all(
                            item.get("source_ci_workflow")
                            == CERTIFICATION_MODEL_PARITY_WORKFLOW
                            for item in artifacts
                        )
                    )
                    or (
                        run_id == "126"
                        and all(
                            item.get("source_ci_workflow")
                            == CERTIFICATION_PERFORMANCE_WORKFLOW
                            for item in artifacts
                        )
                    )
                )
            ),
        }

        (custom_bundle_dir / "claim_sources").mkdir()
        for original, relative in {
            **strict_core_mapping,
            **strict_proof_mapping,
        }.items():
            if original in claim_path_by_original:
                continue
            path = certificate_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "self-test-core-evidence.v1",
                            "generated_at_utc": strict_timestamp,
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                path.write_text(
                    f"self-test snapshot for {original}\n", encoding="utf-8"
                )
            os.utime(path, (fixed_now.timestamp(), fixed_now.timestamp()))
        hypothesis_evidence_dir = custom_bundle_dir / "evidence"
        hypothesis_evidence_dir.mkdir()
        hypothesis_result = hypothesis_evidence_dir / "result.txt"
        hypothesis_result.write_text("measured no-evidence result\n", encoding="utf-8")
        (hypothesis_evidence_dir / "SHA256SUMS").write_text(
            f"{_sha256_file(hypothesis_result)}  result.txt\n", encoding="utf-8"
        )
        for path in hypothesis_evidence_dir.iterdir():
            os.utime(path, (fixed_now.timestamp(), fixed_now.timestamp()))
        for (domain, tool_id), relative in audit_tool_output_paths.items():
            path = certificate_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "gauntlet.audit_tool_output.v1",
                        "generated_at_utc": strict_timestamp,
                        "git_head": fixed_head,
                        "domain": domain,
                        "tool_id": tool_id,
                        "command": CERTIFICATION_AUDIT_TOOLS[domain][tool_id],
                        "exit_code": 0,
                        "result": "pass",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        for domain, relative in audit_receipt_paths.items():
            path = certificate_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "gauntlet.audit_receipt.v1",
                        "generated_at_utc": strict_timestamp,
                        "git_head": fixed_head,
                        "domain": domain,
                        "scope_complete": True,
                        "tools": [
                            {
                                "id": tool_id,
                                "version": "self-test-v1",
                                "command": command,
                                "output_path": audit_tool_output_paths[
                                    (domain, tool_id)
                                ],
                                "output_sha256": _sha256_file(
                                    certificate_root
                                    / audit_tool_output_paths[(domain, tool_id)]
                                ),
                                "result": "pass",
                            }
                            for tool_id, command in CERTIFICATION_AUDIT_TOOLS[
                                domain
                            ].items()
                        ],
                        "findings": [],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        benchmark_stages = {
            stage: {
                "schema": "focr-gauntlet-stage/v1",
                "stage": stage,
                "samples_ms": [100.0, 100.0, 100.0],
                "best_ms": 100.0,
            }
            for stage in ("vision_encode", "decode_per_token", "end_to_end")
        }
        benchmark_baseline = certificate_root / benchmark_baseline_path
        benchmark_baseline.parent.mkdir(parents=True, exist_ok=True)
        benchmark_baseline.write_text(
            json.dumps(
                {
                    "schema": "focr-bench-baseline/v1",
                    "note": "self-test frozen baseline",
                    "stages": benchmark_stages,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        benchmark_current = certificate_root / benchmark_current_path
        benchmark_current.parent.mkdir(parents=True, exist_ok=True)
        benchmark_current.write_text(
            json.dumps(
                {
                    "schema_version": "gauntlet.current_benchmark.v1",
                    "generated_at_utc": strict_timestamp,
                    "git_head": fixed_head,
                    "baseline_sha256": _sha256_file(benchmark_baseline),
                    "stages": benchmark_stages,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        dist_evidence = certificate_root / dist_evidence_path
        dist_evidence.parent.mkdir(parents=True, exist_ok=True)
        dist_evidence.write_text(
            json.dumps(
                {
                    "schema_version": "gauntlet.dist_target_output.v1",
                    "generated_at_utc": strict_timestamp,
                    "git_head": fixed_head,
                    "result": "pass",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for target, relative in dist_portability_paths.items():
            path = certificate_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "GLIBC_2.2.5 GLIBC_2.17\n"
                if target in CERTIFICATION_LINUX_DIST_TARGETS
                else "offline install.ps1 self-test passed\n",
                encoding="utf-8",
            )
        model_fixture = certificate_root / model_fixture_path
        model_fixture.parent.mkdir(parents=True, exist_ok=True)
        model_fixture.write_text(
            json.dumps(
                {
                    "schema": "focr-ladder-scorecard/v1",
                    "gates": [
                        {
                            "gate": rung,
                            "outcome": "pass",
                            "meaningful": True,
                            "parity_rows": rows,
                            "worst": {"metric": "self-test", "value": 1.0},
                        }
                        for rung, rows in CERTIFICATION_MODEL_PARITY_MIN_ROWS.items()
                    ],
                    "all_green": True,
                    "skipped_no_model": False,
                    "receipt": "self-test canonical weighted ladder",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        model_fixture_sha256 = _sha256_file(model_fixture)
        model_payloads: dict[str, object] = {
            "L0": {"pixels": [1, 2], "geometry": [1024, 1024]},
            "L1": [1.0, 2.0, 3.0],
            "L2": [1.0, 2.0, 3.0],
            "L3": [1.0, 3.0, 2.0],
            "L4": [1, 2, 3],
            "L5": "hello world",
        }
        for rung, payload in model_payloads.items():
            for kind, relative in (
                ("oracle", model_oracle_paths[rung]),
                ("subject", model_subject_paths[rung]),
            ):
                path = certificate_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "gauntlet.model_parity_output.v1",
                            "generated_at_utc": strict_timestamp,
                            "git_head": fixed_head,
                            "model_commit": UNLIMITED_OCR_MODEL_COMMIT,
                            "fixture_sha256": model_fixture_sha256,
                            "rung": rung,
                            "kind": kind,
                            "payload": payload,
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        for rung, relative in model_rung_log_paths.items():
            path = certificate_root / relative
            path.write_text(f"{rung} weighted parity passed\n", encoding="utf-8")
        for rung, relative in model_rung_paths.items():
            path = certificate_root / relative
            log_relative = model_rung_log_paths[rung]
            oracle_relative = model_oracle_paths[rung]
            subject_relative = model_subject_paths[rung]
            derived_metrics, derived_pass = _derived_parity_metrics(
                rung, model_payloads[rung], model_payloads[rung]
            )
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "gauntlet.model_parity_rung.v1",
                        "generated_at_utc": strict_timestamp,
                        "git_head": fixed_head,
                        "model_commit": UNLIMITED_OCR_MODEL_COMMIT,
                        "rung": rung,
                        "weighted_model_loaded": True,
                        "oracle_backend": "torch-2.10.0-transformers-4.57.1",
                        "fixture_path": model_fixture_path,
                        "fixture_sha256": model_fixture_sha256,
                        "oracle_output_path": oracle_relative,
                        "oracle_output_sha256": _sha256_file(
                            certificate_root / oracle_relative
                        ),
                        "subject_output_path": subject_relative,
                        "subject_output_sha256": _sha256_file(
                            certificate_root / subject_relative
                        ),
                        "metrics": derived_metrics,
                        "result": "pass" if derived_pass else "fail",
                        "raw_log_path": log_relative,
                        "raw_log_sha256": _sha256_file(certificate_root / log_relative),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        readiness_source = {
            "schema_version": "gauntlet.release_readiness.v1",
            "artifact": "franken_ocr.release_readiness.v1",
            "cell_set_sha256": CERTIFICATION_READINESS_CELL_SET_SHA256,
            "generated_at_utc": strict_timestamp,
            "generated_by": "scripts/gauntlet_cert.py --release-readiness",
            "cells": [
                {
                    "cell": cell,
                    "status": "red" if cell == "certification_bundle" else "green",
                    "evidence": "release_certificate.json"
                    if cell == "certification_bundle"
                    else "self-test evidence",
                    "evidence_paths": [
                        strict_proof_mapping[original]
                        for original in CERTIFICATION_READINESS_EVIDENCE_PATHS.get(
                            cell, ()
                        )
                    ],
                }
                for cell in CERTIFICATION_READINESS_CELLS
            ],
            "green": len(CERTIFICATION_EXTERNAL_READINESS_CELLS),
            "red": 1,
            "blocking_cells": ["certification_bundle"],
            "ship": False,
        }
        (certificate_root / source_claim_paths[0]).write_text(
            json.dumps(readiness_source, sort_keys=True) + "\n", encoding="utf-8"
        )
        source_hypotheses = {
            ledger: canonical_experiment(f"EXP-{index:04d}", pillar)
            for index, (ledger, pillar) in enumerate(
                (
                    (source_claim_paths[2], "perf"),
                    (source_claim_paths[3], "perf"),
                    (source_claim_paths[4], "conformance"),
                    (source_claim_paths[5], "surface"),
                ),
                start=1,
            )
        }
        for relative, content in source_hypotheses.items():
            path = certificate_root / relative
            path.write_text(content, encoding="utf-8")
            os.utime(path, (fixed_now.timestamp(), fixed_now.timestamp()))
        strict_rounds = [
            {
                **canonical_round(round_number, 0 if round_number >= 9 else 4),
                "date": fixed_now.date().isoformat(),
            }
            for round_number in range(1, 11)
        ]
        (certificate_root / source_claim_paths[1]).write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n" for record in strict_rounds
            ),
            encoding="utf-8",
        )
        signature_dir = custom_bundle_dir / "signatures"
        signature_dir.mkdir()
        for name in ("producer.asc", "reviewer.asc", "release.asc"):
            (signature_dir / name).write_text(
                "self-test detached signature\n", encoding="utf-8"
            )

        base_strict_documents = {
            "confidence_gate": {
                "schema_version": STRICT_BUNDLE_CLASSES["confidence_gate"][1],
                "generated_at_utc": strict_timestamp,
                "release_decision": "Allow",
                "min_verification_pct_observed": 100.0,
                "required_suite_pass_rate_pct_observed": 100.0,
                "high_severity_counterexample_count": 0,
                "constants_enforced": list(CERTIFICATION_CONSTANTS),
            },
            "ci_gate_receipt": {
                "schema_version": STRICT_BUNDLE_CLASSES["ci_gate_receipt"][1],
                "generated_at_utc": strict_timestamp,
                "git_head": fixed_head,
                "source_ci_run_id": "123",
                "suite_pass_rate_pct": 100.0,
                "jobs": {job: "pass" for job in CERTIFICATION_CI_REQUIRED_JOBS},
            },
            "model_parity_receipt": {
                "schema_version": STRICT_BUNDLE_CLASSES["model_parity_receipt"][1],
                "generated_at_utc": strict_timestamp,
                "git_head": fixed_head,
                "source_ci_run_id": "125",
                "model_commit": UNLIMITED_OCR_MODEL_COMMIT,
                "skipped_no_model": False,
                "rungs": {f"L{index}": "pass" for index in range(6)},
                "raw_evidence_paths": [
                    *model_rung_paths.values(),
                    *model_rung_log_paths.values(),
                    model_fixture_path,
                    *model_oracle_paths.values(),
                    *model_subject_paths.values(),
                ],
                "raw_evidence_sha256s": {
                    relative: _sha256_file(certificate_root / relative)
                    for relative in (
                        *model_rung_paths.values(),
                        *model_rung_log_paths.values(),
                        model_fixture_path,
                        *model_oracle_paths.values(),
                        *model_subject_paths.values(),
                    )
                },
            },
            "dist_matrix_receipt": {
                "schema_version": STRICT_BUNDLE_CLASSES["dist_matrix_receipt"][1],
                "generated_at_utc": strict_timestamp,
                "git_head": fixed_head,
                "source_ci_run_id": "124",
                "targets": {
                    target: {
                        "status": "pass",
                        "built": True,
                        "checksum_sidecar": True,
                        "smoke_test": "pass",
                        "portability": (
                            {
                                "kind": "glibc-symbol-floor",
                                "result": "pass",
                                "supported_floor": CERTIFICATION_LINUX_GLIBC_FLOOR,
                                "maximum_required": "2.17",
                                "required_versions": ["2.2.5", "2.17"],
                                "raw_evidence_path": dist_portability_paths[target],
                                "raw_evidence_sha256": _sha256_file(
                                    certificate_root / dist_portability_paths[target]
                                ),
                            }
                            if target in CERTIFICATION_LINUX_DIST_TARGETS
                            else {
                                "kind": "native-offline-install.ps1",
                                "result": "pass",
                                "offline": True,
                                "asset_sha256": "a" * 64,
                                "installed_sha256": "a" * 64,
                                "reported_version": "0.0.0",
                                "installer_sha256": _sha256_file(
                                    certificate_root / "install.ps1"
                                ),
                                "transcript_path": dist_portability_paths[target],
                                "transcript_sha256": _sha256_file(
                                    certificate_root / dist_portability_paths[target]
                                ),
                            }
                            if target in CERTIFICATION_WINDOWS_DIST_TARGETS
                            else {"kind": "native-smoke", "result": "pass"}
                        ),
                    }
                    for target in CERTIFICATION_DIST_TARGETS
                },
                "raw_evidence_paths": [
                    dist_evidence_path,
                    *dist_portability_paths.values(),
                ],
                "raw_evidence_sha256s": {
                    relative: _sha256_file(certificate_root / relative)
                    for relative in (
                        dist_evidence_path,
                        *dist_portability_paths.values(),
                    )
                },
            },
            "benchmark_summary": {
                "schema_version": STRICT_BUNDLE_CLASSES["benchmark_summary"][1],
                "generated_at_utc": strict_timestamp,
                "pass_over_pass_gates": {
                    name: {
                        "passed": True,
                        "minimum_pct": minimum,
                        "regression_pct": 0.0,
                        "baseline_path": benchmark_input_paths[(name, "baseline")],
                        "baseline_sha256": _sha256_file(
                            certificate_root / benchmark_input_paths[(name, "baseline")]
                        ),
                        "current_path": benchmark_input_paths[(name, "current")],
                        "current_sha256": _sha256_file(
                            certificate_root / benchmark_input_paths[(name, "current")]
                        ),
                    }
                    for name, minimum in CERTIFICATION_BENCHMARK_THRESHOLDS.items()
                },
            },
            "scorecards": {
                "schema_version": STRICT_BUNDLE_CLASSES["scorecards"][1],
                "generated_at_utc": strict_timestamp,
                "parity_score_lower_bound": 0.9,
                "per_category_lower": {"feature-a": 0.9},
            },
            "critical_path_report": {
                "schema_version": STRICT_BUNDLE_CLASSES["critical_path_report"][1],
                "generated_at_utc": strict_timestamp,
                "open_high_critical": 0,
                "waived_high_critical": 0,
                "findings": [],
            },
            "critical_path_inventory": {
                "schema_version": STRICT_BUNDLE_CLASSES["critical_path_inventory"][1],
                "generated_at_utc": strict_timestamp,
                "audits": {
                    domain: {
                        "evidence_path": relative,
                        "evidence_sha256": _sha256_file(certificate_root / relative),
                    }
                    for domain, relative in audit_receipt_paths.items()
                },
            },
            "ratchet_state": {
                "schema_version": STRICT_BUNDLE_CLASSES["ratchet_state"][1],
                "generated_at_utc": strict_timestamp,
                "previous_bound": 0.7,
                "current_lower_bound": 0.8,
                "commit_sha": fixed_head,
                "timestamp": strict_timestamp,
                "advance_reason": "self-test monotone advance",
                "per_category_bounds": {"feature-a": 0.8},
            },
        }

        def write_strict_fixture(
            documents: dict[str, dict] | None = None,
            *,
            ci_artifact_paths: set[str] | None = None,
        ) -> tuple[dict[str, dict], dict, dict]:
            documents = json.loads(json.dumps(documents or base_strict_documents))
            feature_universe = {
                "schema_version": "gauntlet.feature_universe.v1",
                "generated_at_utc": strict_timestamp,
                "definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
                "features": list(CERTIFICATION_FEATURE_UNIVERSE),
            }
            (certificate_root / feature_universe_relative).write_text(
                json.dumps(feature_universe, sort_keys=True) + "\n", encoding="utf-8"
            )
            universe_sha = _sha256_file(certificate_root / feature_universe_relative)
            documents["verification_contract"] = {
                "schema_version": STRICT_BUNDLE_CLASSES["verification_contract"][1],
                "generated_at_utc": strict_timestamp,
                "feature_universe_path": feature_universe_relative,
                "feature_universe_sha256": universe_sha,
                "feature_universe_definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
                "rows": [
                    {
                        "feature_id": feature["feature_id"],
                        "proof_obligation": obligation,
                        "status": "pass",
                        "gate": "allowed",
                        "evidence_paths": [
                            strict_proof_mapping[original]
                            for original in CERTIFICATION_READINESS_EVIDENCE_PATHS[
                                obligation
                            ]
                        ],
                        "evidence_sha256s": {
                            strict_proof_mapping[original]: _sha256_file(
                                certificate_root / strict_proof_mapping[original]
                            )
                            for original in CERTIFICATION_READINESS_EVIDENCE_PATHS[
                                obligation
                            ]
                        },
                    }
                    for feature in CERTIFICATION_FEATURE_UNIVERSE
                    for obligation in feature["proof_obligations"]
                ],
            }
            for class_name, payload in documents.items():
                if class_name == "ci_manifest":
                    continue
                filename = STRICT_BUNDLE_CLASSES[class_name][0]
                (custom_bundle_dir / filename).write_text(
                    json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            non_ci_paths = [
                relative
                for class_name, relative in strict_class_paths.items()
                if class_name != "ci_manifest"
            ]
            non_ci_paths.extend(
                [
                    feature_universe_relative,
                    *strict_core_mapping.values(),
                    *strict_proof_mapping.values(),
                    *hypothesis_evidence_paths,
                    *audit_receipt_paths.values(),
                    *audit_tool_output_paths.values(),
                    *benchmark_input_paths.values(),
                    dist_evidence_path,
                    *dist_portability_paths.values(),
                    *model_rung_paths.values(),
                    *model_rung_log_paths.values(),
                    model_fixture_path,
                    *model_oracle_paths.values(),
                    *model_subject_paths.values(),
                ]
            )
            non_ci_paths = list(dict.fromkeys(non_ci_paths))
            strict_static_paths = {
                relative
                for original, relative in {
                    **strict_core_mapping,
                    **strict_proof_mapping,
                }.items()
                if original
                not in {
                    "docs/gauntlet/RELEASE_READINESS.json",
                    "docs/gauntlet/ROUNDS.jsonl",
                }
            }

            def manifest_entry(relative: str) -> dict:
                path = certificate_root / relative
                content = _read_bounded_file(path)
                timestamp_value, timestamp_source = _manifest_timestamp(
                    path,
                    content,
                    fixed_now,
                    static_head_source=relative in strict_static_paths,
                )
                age_hours = (fixed_now - timestamp_value).total_seconds() / 3600.0
                return {
                    "artifact": relative,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "age_hours": round(age_hours, 2),
                    "timestamp_utc": _timestamp_text(timestamp_value),
                    "timestamp_source": timestamp_source,
                }

            preliminary_entries = [
                manifest_entry(relative) for relative in non_ci_paths
            ]
            model_ci_paths = {
                strict_class_paths["model_parity_receipt"],
                *model_rung_paths.values(),
                *model_rung_log_paths.values(),
                model_fixture_path,
                *model_oracle_paths.values(),
                *model_subject_paths.values(),
            }
            performance_ci_paths = {
                strict_class_paths["benchmark_summary"],
                benchmark_current_path,
            }

            def fixture_ci_workflow(relative: str) -> str:
                if relative in {
                    strict_class_paths["dist_matrix_receipt"],
                    dist_evidence_path,
                    *dist_portability_paths.values(),
                }:
                    return CERTIFICATION_DIST_WORKFLOW
                if relative in model_ci_paths:
                    return CERTIFICATION_MODEL_PARITY_WORKFLOW
                if relative in performance_ci_paths:
                    return CERTIFICATION_PERFORMANCE_WORKFLOW
                return CERTIFICATION_GITHUB_WORKFLOW

            def fixture_ci_run_id(relative: str) -> str:
                return {
                    CERTIFICATION_GITHUB_WORKFLOW: "123",
                    CERTIFICATION_DIST_WORKFLOW: "124",
                    CERTIFICATION_MODEL_PARITY_WORKFLOW: "125",
                    CERTIFICATION_PERFORMANCE_WORKFLOW: "126",
                }[fixture_ci_workflow(relative)]

            documents["ci_manifest"] = {
                "schema_version": STRICT_BUNDLE_CLASSES["ci_manifest"][1],
                "generated_at_utc": strict_timestamp,
                "repository": CERTIFICATION_GITHUB_REPOSITORY,
                "required_workflows": {
                    CERTIFICATION_GITHUB_WORKFLOW: CERTIFICATION_GITHUB_EVENT,
                    CERTIFICATION_DIST_WORKFLOW: CERTIFICATION_GITHUB_EVENT,
                    CERTIFICATION_MODEL_PARITY_WORKFLOW: CERTIFICATION_MODEL_PARITY_EVENT,
                    CERTIFICATION_PERFORMANCE_WORKFLOW: CERTIFICATION_PERFORMANCE_EVENT,
                },
                "artifacts": [
                    {
                        "artifact": entry["artifact"],
                        "sha256": entry["sha256"],
                        "schema_version": CI_ARTIFACT_BINDING_SCHEMA,
                        "source_ci_run_id": fixture_ci_run_id(entry["artifact"]),
                        "source_ci_artifact_name": "strict-"
                        + fixture_ci_workflow(entry["artifact"])
                        .lower()
                        .replace(" ", "-")
                        + "-self-test",
                        "source_ci_artifact_path": entry["artifact"],
                        "source_ci_repository": CERTIFICATION_GITHUB_REPOSITORY,
                        "source_ci_workflow": fixture_ci_workflow(entry["artifact"]),
                        "source_ci_workflow_path": CERTIFICATION_WORKFLOW_PATHS[
                            fixture_ci_workflow(entry["artifact"])
                        ],
                        "source_ci_event": {
                            CERTIFICATION_MODEL_PARITY_WORKFLOW: CERTIFICATION_MODEL_PARITY_EVENT,
                            CERTIFICATION_PERFORMANCE_WORKFLOW: CERTIFICATION_PERFORMANCE_EVENT,
                        }.get(
                            fixture_ci_workflow(entry["artifact"]),
                            CERTIFICATION_GITHUB_EVENT,
                        ),
                    }
                    for entry in preliminary_entries
                    if ci_artifact_paths is None
                    or entry["artifact"] in ci_artifact_paths
                ],
            }
            (custom_bundle_dir / STRICT_BUNDLE_CLASSES["ci_manifest"][0]).write_text(
                json.dumps(documents["ci_manifest"], sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest = [
                *preliminary_entries,
                manifest_entry(strict_class_paths["ci_manifest"]),
            ]
            strict_max_age = round(
                max(
                    0.0,
                    *(
                        (
                            fixed_now - _parse_utc_timestamp(entry["timestamp_utc"])
                        ).total_seconds()
                        / 3600.0
                        for entry in manifest
                    ),
                ),
                2,
            )
            manifest_root = _bundle_root_sha256(manifest)
            strict_hypotheses = hypothesis_texts_verdict(
                source_hypotheses,
                source_claim_paths[2:],
                custom_bundle_dir,
            )
            strict_convergence = convergence_verdict(strict_rounds, strict_hypotheses)
            certificate = {
                "schema_version": STRICT_CERTIFICATE_SCHEMA,
                "artifact": STRICT_CERTIFICATE_ARTIFACT,
                "template": STRICT_CERTIFICATE_SCHEMA,
                "project": "franken_ocr",
                "version": "0.0.0",
                "issued_at": strict_timestamp,
                "constants": CERTIFICATION_CONSTANTS,
                "reference": {"model_commit": UNLIMITED_OCR_MODEL_COMMIT},
                "git_head": fixed_head,
                "evidence_git_head": fixed_head,
                "git_branch": "main",
                "git_describe": "v0.0.0",
                "git_worktree": clean_worktree,
                "readiness": {
                    "green": len(CERTIFICATION_EXTERNAL_READINESS_CELLS),
                    "red": 0,
                    "blocking_cells": [],
                    "ship": True,
                },
                "convergence": strict_convergence,
                "required_pass_actuals": {
                    "min_verification_pct_observed": 100.0,
                    "required_suite_pass_rate_pct_observed": 100.0,
                    "high_severity_counterexample_count": 0,
                    "max_evidence_age_hours_observed": strict_max_age,
                },
                "high_severity_counterexamples": 0,
                "parity_score": 0.9,
                "refusal_reasons": [],
                "evidence_classes": strict_class_paths,
                "claim_sources": {
                    "release_readiness": source_claim_paths[0],
                    "rounds": source_claim_paths[1],
                    "hypothesis_ledgers": list(source_claim_paths[2:]),
                    "core_evidence": strict_core_mapping,
                    "readiness_evidence": strict_proof_mapping,
                },
                "readiness_cell_set_sha256": CERTIFICATION_READINESS_CELL_SET_SHA256,
                "feature_universe_sha256": universe_sha,
                "feature_universe_definition_sha256": CERTIFICATION_FEATURE_UNIVERSE_SHA256,
                "evidence_bundle_sha256": manifest_root,
                "signers": [
                    "producer@example.test",
                    "reviewer@example.test",
                    "release@example.test",
                ],
                "detached_signatures": [
                    {
                        "signer": "producer@example.test",
                        "role": "producer",
                        "fingerprint": trusted_signers["producer@example.test"],
                        "scheme": "openpgp-detached",
                        "signature_path": f"{strict_relative}/signatures/producer.asc",
                    },
                    {
                        "signer": "reviewer@example.test",
                        "role": "independent-reviewer",
                        "fingerprint": trusted_signers["reviewer@example.test"],
                        "scheme": "openpgp-detached",
                        "signature_path": f"{strict_relative}/signatures/reviewer.asc",
                    },
                    {
                        "signer": "release@example.test",
                        "role": "release-authorizer",
                        "fingerprint": trusted_signers["release@example.test"],
                        "scheme": "openpgp-detached",
                        "signature_path": f"{strict_relative}/signatures/release.asc",
                    },
                ],
                "certified": True,
                "generated_by": "scripts/gauntlet_cert.py --bundle",
            }
            certificate["signed_claim_sha256"] = _certificate_signed_claim_sha256(
                certificate, manifest_root
            )
            bundle = {
                "schema_version": STRICT_BUNDLE_SCHEMA,
                "artifact": STRICT_BUNDLE_ARTIFACT,
                "generated_at_utc": strict_timestamp,
                "bundle_root_sha256": manifest_root,
                "manifest": manifest,
            }
            (custom_bundle_dir / "release_certificate.json").write_text(
                json.dumps(certificate, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (custom_bundle_dir / "certification_bundle.json").write_text(
                json.dumps(bundle, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (custom_bundle_dir / "FINAL_GAUNTLET_REPORT.md").write_text(
                _final_report_text(certificate, bundle), encoding="utf-8"
            )
            return documents, certificate, bundle

        strict_documents, strict_certificate, strict_bundle = write_strict_fixture()
        strict_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_complete_strict_fixture_passes",
            strict_verdict["ok"],
            verdict=strict_verdict,
        )
        control_paths = (
            custom_bundle_dir / "release_certificate.json",
            custom_bundle_dir / "certification_bundle.json",
            custom_bundle_dir / "FINAL_GAUNTLET_REPORT.md",
        )
        control_before = tuple(_sha256_file(path) for path in control_paths)
        in_memory_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            control_documents=tuple(_read_bounded_file(path) for path in control_paths),
            **strict_verifier_kwargs,
        )
        check(
            "bundle_in_memory_candidate_verifies_without_control_mutation",
            in_memory_verdict["ok"]
            and control_before == tuple(_sha256_file(path) for path in control_paths),
            verdict=in_memory_verdict,
        )
        minimal_ci_paths = {
            strict_class_paths["ci_gate_receipt"],
            *base_strict_documents["dist_matrix_receipt"]["raw_evidence_paths"],
            *base_strict_documents["model_parity_receipt"]["raw_evidence_paths"],
            benchmark_current_path,
        }
        write_strict_fixture(ci_artifact_paths=minimal_ci_paths)
        minimal_ci_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_ci_manifest_requires_only_ci_produced_evidence",
            minimal_ci_verdict["ok"],
            verdict=minimal_ci_verdict,
        )
        strict_documents, strict_certificate, strict_bundle = write_strict_fixture()
        audit_ready = _readiness_audit_receipt(
            certificate_root,
            custom_bundle_dir,
            fixed_head,
            fixed_now,
            "concurrency",
            {"many-pages-watchdog", "cancel-panic-faults"},
        )
        check(
            "readiness_deadlock_requires_current_hashed_tool_receipts",
            audit_ready[0],
            verdict=audit_ready,
        )
        dist_ready = _readiness_dist_receipt(
            certificate_root, custom_bundle_dir, fixed_head, fixed_now
        )
        check(
            "readiness_build_matrix_requires_current_complete_dist_receipt",
            dist_ready[0],
            verdict=dist_ready,
        )
        glibc_drift_documents = json.loads(json.dumps(base_strict_documents))
        linux_target = next(iter(sorted(CERTIFICATION_LINUX_DIST_TARGETS)))
        glibc_drift_documents["dist_matrix_receipt"]["targets"][linux_target][
            "portability"
        ]["supported_floor"] = "2.18"
        write_strict_fixture(glibc_drift_documents)
        glibc_drift_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_linux_glibc_floor_drift_rejected",
            not glibc_drift_verdict["ok"]
            and any("glibc 2.17" in reason for reason in glibc_drift_verdict["reasons"]),
            verdict=glibc_drift_verdict,
        )
        windows_drift_documents = json.loads(json.dumps(base_strict_documents))
        windows_target = next(iter(sorted(CERTIFICATION_WINDOWS_DIST_TARGETS)))
        windows_drift_documents["dist_matrix_receipt"]["targets"][windows_target][
            "portability"
        ]["installer_sha256"] = "f" * 64
        write_strict_fixture(windows_drift_documents)
        windows_drift_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_windows_installer_identity_drift_rejected",
            not windows_drift_verdict["ok"]
            and any(
                "offline install.ps1" in reason
                for reason in windows_drift_verdict["reasons"]
            ),
            verdict=windows_drift_verdict,
        )
        strict_documents, strict_certificate, strict_bundle = write_strict_fixture()
        replay_manifest = json.loads(json.dumps(strict_bundle["manifest"]))
        replay_manifest[0]["timestamp_utc"] = _timestamp_text(
            fixed_now + timedelta(seconds=1)
        )
        check(
            "bundle_root_authenticates_freshness_metadata",
            _bundle_root_sha256(replay_manifest) != strict_bundle["bundle_root_sha256"],
        )

        extra_field_certificate = json.loads(json.dumps(strict_certificate))
        extra_field_certificate["unsigned_note"] = "must be rejected"
        (custom_bundle_dir / "release_certificate.json").write_text(
            json.dumps(extra_field_certificate, sort_keys=True) + "\n", encoding="utf-8"
        )
        (custom_bundle_dir / "FINAL_GAUNTLET_REPORT.md").write_text(
            _final_report_text(extra_field_certificate, strict_bundle), encoding="utf-8"
        )
        extra_field_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_certificate_extra_top_level_field_rejected",
            not extra_field_verdict["ok"]
            and "certificate object has a noncanonical field set"
            in extra_field_verdict["reasons"],
            verdict=extra_field_verdict,
        )

        write_strict_fixture()
        (custom_bundle_dir / "release_certificate.json").write_text(
            "[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8"
        )
        deep_control_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_deep_control_json_fails_closed_without_exception",
            not deep_control_verdict["ok"]
            and any(
                "missing or unreadable certificate bundle" in reason
                for reason in deep_control_verdict["reasons"]
            ),
            verdict=deep_control_verdict,
        )

        write_strict_fixture()
        concurrency_receipt_path = (
            custom_bundle_dir / "audit_receipts/concurrency_audit_receipt.json"
        )
        original_concurrency_receipt = _read_bounded_text(concurrency_receipt_path)
        concurrency_receipt_path.write_text(
            "[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8"
        )
        malformed_audit_ready = _readiness_audit_receipt(
            certificate_root,
            custom_bundle_dir,
            fixed_head,
            fixed_now,
            "concurrency",
            {"many-pages-watchdog", "cancel-panic-faults"},
        )
        check(
            "readiness_deep_audit_receipt_fails_closed_without_exception",
            not malformed_audit_ready[0],
            verdict=malformed_audit_ready,
        )
        concurrency_receipt_path.write_text(
            original_concurrency_receipt, encoding="utf-8"
        )
        write_strict_fixture()

        malformed_ci_documents = json.loads(json.dumps(base_strict_documents))
        malformed_ci_documents["ci_gate_receipt"]["suite_pass_rate_pct"] = 10**1000
        write_strict_fixture(malformed_ci_documents)
        malformed_ci_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_huge_ci_number_fails_closed_without_exception",
            not malformed_ci_verdict["ok"]
            and any(
                "CI gate receipt" in reason
                for reason in malformed_ci_verdict["reasons"]
            ),
            verdict=malformed_ci_verdict,
        )

        malformed_paths_documents = json.loads(json.dumps(base_strict_documents))
        malformed_paths_documents["model_parity_receipt"]["raw_evidence_paths"] = [{}]
        write_strict_fixture(malformed_paths_documents)
        malformed_paths_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_unhashable_parity_path_fails_closed_without_exception",
            not malformed_paths_verdict["ok"]
            and any(
                "model parity receipt" in reason
                for reason in malformed_paths_verdict["reasons"]
            ),
            verdict=malformed_paths_verdict,
        )

        l3_record_path = certificate_root / model_rung_paths["L3"]
        original_l3_record = _read_bounded_text(l3_record_path)
        tampered_l3_record = json.loads(original_l3_record)
        tampered_l3_record["metrics"] = {"cosine_min": 1.0}
        l3_record_path.write_text(
            json.dumps(tampered_l3_record, sort_keys=True) + "\n", encoding="utf-8"
        )
        tampered_parity_documents = json.loads(json.dumps(base_strict_documents))
        tampered_parity_documents["model_parity_receipt"]["raw_evidence_sha256s"][
            model_rung_paths["L3"]
        ] = _sha256_file(l3_record_path)
        write_strict_fixture(tampered_parity_documents)
        tampered_parity_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_self_asserted_parity_metric_is_recomputed",
            not tampered_parity_verdict["ok"]
            and any(
                "model parity receipt" in reason
                for reason in tampered_parity_verdict["reasons"]
            ),
            verdict=tampered_parity_verdict,
        )
        l3_record_path.write_text(original_l3_record, encoding="utf-8")

        security_receipt_path = certificate_root / audit_receipt_paths["security"]
        original_security_receipt = _read_bounded_text(security_receipt_path)
        malformed_security_receipt = json.loads(original_security_receipt)
        malformed_security_receipt["findings"] = [
            {
                "id": "SELF-TEST-MALFORMED",
                "severity": "High",
                "status": "resolved",
                "evidence_path": [],
                "evidence_sha256": "0" * 64,
            }
        ]
        security_receipt_path.write_text(
            json.dumps(malformed_security_receipt, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        malformed_audit_documents = json.loads(json.dumps(base_strict_documents))
        malformed_audit_documents["critical_path_inventory"]["audits"]["security"][
            "evidence_sha256"
        ] = _sha256_file(security_receipt_path)
        write_strict_fixture(malformed_audit_documents)
        malformed_audit_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_unhashable_audit_path_fails_closed_without_exception",
            not malformed_audit_verdict["ok"]
            and any(
                "critical-path inventory" in reason
                for reason in malformed_audit_verdict["reasons"]
            ),
            verdict=malformed_audit_verdict,
        )
        security_receipt_path.write_text(original_security_receipt, encoding="utf-8")

        default_dir_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_custom_directory_is_not_replaced_by_default",
            not default_dir_verdict["ok"]
            and any(
                "missing or unreadable" in reason
                for reason in default_dir_verdict["reasons"]
            ),
            verdict=default_dir_verdict,
        )

        generated_output, generated_output_reasons = _safe_generated_output_dir(
            fixture_root, ".gauntlet-output/self-test-bundle"
        )
        check(
            "bundle-generated-output-requires-ignored-gauntlet-tree",
            generated_output
            == (fixture_root / ".gauntlet-output/self-test-bundle").resolve()
            and not generated_output_reasons,
            reasons=generated_output_reasons,
        )
        outside_generated, outside_generated_reasons = _safe_generated_output_dir(
            fixture_root, "generated-cert-outside-ignore"
        )
        check(
            "bundle-generated-output-outside-ignore-rejected",
            outside_generated is None
            and any(
                "under .gauntlet-output" in reason
                for reason in outside_generated_reasons
            ),
            reasons=outside_generated_reasons,
        )

        symlink_output = certificate_root / "symlink-output"
        symlink_output.mkdir()
        os.symlink(
            certificate_root / "outside-certificate.json",
            symlink_output / "release_certificate.json",
        )
        safe_symlink_output, symlink_reasons = _safe_output_dir(
            certificate_root, str(symlink_output)
        )
        check(
            "bundle_output_leaf_symlink_rejected",
            safe_symlink_output is None
            and any("contains a symlink" in reason for reason in symlink_reasons),
            reasons=symlink_reasons,
        )
        looping_output = certificate_root / "looping-output"
        os.symlink(looping_output, looping_output)
        safe_looping_output, looping_reasons = _safe_output_dir(
            certificate_root, str(looping_output)
        )
        check(
            "bundle_output_dangling_self_loop_symlink_rejected",
            safe_looping_output is None
            and any("symlink" in reason for reason in looping_reasons),
            reasons=looping_reasons,
        )

        (custom_bundle_dir / "release_certificate.json").write_text(
            json.dumps({"certified": True, "git_head": fixed_head}) + "\n",
            encoding="utf-8",
        )
        minimal_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_minimal_truncated_certificate_rejected",
            not minimal_verdict["ok"]
            and any("schema_version" in reason for reason in minimal_verdict["reasons"])
            and any("signers" in reason for reason in minimal_verdict["reasons"]),
            verdict=minimal_verdict,
        )

        write_strict_fixture()
        wrong_head_verdict = certificate_bundle_verdict(
            certificate_root,
            "f" * 40,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_wrong_head_rejected",
            not wrong_head_verdict["ok"]
            and any("git_head" in reason for reason in wrong_head_verdict["reasons"]),
            verdict=wrong_head_verdict,
        )

        _docs, claim_certificate, claim_bundle = write_strict_fixture()
        claim_certificate["readiness"]["green"] = 99
        (custom_bundle_dir / "release_certificate.json").write_text(
            json.dumps(claim_certificate, sort_keys=True) + "\n", encoding="utf-8"
        )
        unsigned_claim_tamper = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_unsigned_release_claim_tamper_rejected",
            not unsigned_claim_tamper["ok"]
            and any(
                "signed_claim_sha256" in reason
                for reason in unsigned_claim_tamper["reasons"]
            )
            and any(
                "hashed readiness cells" in reason
                for reason in unsigned_claim_tamper["reasons"]
            ),
            verdict=unsigned_claim_tamper,
        )

        write_strict_fixture()
        status_failure = {
            "status_ok": False,
            "clean": True,
            "dirty_path_count": 0,
            "status_porcelain": [],
            "status_sha256": _sha256_text("git status failed"),
            "status_error": "git status failed",
        }
        status_failure_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=status_failure,
            signature_verifier=lambda _root, _signature, _claim: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_git_status_failure_is_not_clean",
            not status_failure_verdict["ok"]
            and any(
                "status could not be verified" in reason
                for reason in status_failure_verdict["reasons"]
            ),
            verdict=status_failure_verdict,
        )

        write_strict_fixture()
        ci_provenance_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            trusted_signers=trusted_signers,
            ci_run_verifier=lambda _root, _run_id, _head, _artifacts: False,
        )
        check(
            "bundle_unreachable_ci_run_rejected",
            not ci_provenance_verdict["ok"]
            and any(
                "CI run artifact provenance could not be verified" in reason
                for reason in ci_provenance_verdict["reasons"]
            ),
            verdict=ci_provenance_verdict,
        )

        write_strict_fixture()
        (custom_bundle_dir / "confidence_gate.json").write_text(
            "mutated evidence\n", encoding="utf-8"
        )
        bad_hash_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_manifest_hash_tamper_rejected",
            not bad_hash_verdict["ok"]
            and any(
                "hash mismatch" in reason for reason in bad_hash_verdict["reasons"]
            ),
            verdict=bad_hash_verdict,
        )

        write_strict_fixture()
        dirty_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=dirty_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_dirty_worktree_rejected",
            not dirty_verdict["ok"]
            and any(
                "worktree is dirty" in reason for reason in dirty_verdict["reasons"]
            ),
            verdict=dirty_verdict,
        )

        stale_documents = json.loads(json.dumps(base_strict_documents))
        stale_documents["confidence_gate"]["generated_at_utc"] = _timestamp_text(
            fixed_now - timedelta(hours=50)
        )
        write_strict_fixture(stale_documents)
        stale_native_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_recomputes_stale_native_timestamp",
            not stale_native_verdict["ok"]
            and any("is stale" in reason for reason in stale_native_verdict["reasons"]),
            verdict=stale_native_verdict,
        )

        future_documents = json.loads(json.dumps(base_strict_documents))
        future_documents["confidence_gate"]["generated_at_utc"] = _timestamp_text(
            fixed_now + timedelta(hours=2)
        )
        write_strict_fixture(future_documents)
        future_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_future_native_timestamp_rejected",
            not future_verdict["ok"]
            and any("future" in reason for reason in future_verdict["reasons"]),
            verdict=future_verdict,
        )

        _docs, duplicate_certificate, duplicate_bundle = write_strict_fixture()
        duplicate_bundle["manifest"].append(dict(duplicate_bundle["manifest"][0]))
        (custom_bundle_dir / "certification_bundle.json").write_text(
            json.dumps(duplicate_bundle, sort_keys=True) + "\n", encoding="utf-8"
        )
        duplicate_manifest_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_duplicate_manifest_path_rejected",
            not duplicate_manifest_verdict["ok"]
            and any(
                "duplicate artifact" in reason
                for reason in duplicate_manifest_verdict["reasons"]
            ),
            verdict=duplicate_manifest_verdict,
        )

        _docs, duplicate_key_certificate, _bundle = write_strict_fixture()
        duplicate_key_text = json.dumps(duplicate_key_certificate, sort_keys=True)
        duplicate_key_text = duplicate_key_text.replace(
            '"certified": true',
            '"certified": false, "certified": true',
            1,
        )
        (custom_bundle_dir / "release_certificate.json").write_text(
            duplicate_key_text + "\n", encoding="utf-8"
        )
        duplicate_key_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_duplicate_json_key_rejected",
            not duplicate_key_verdict["ok"]
            and any(
                "missing or unreadable certificate bundle" in reason
                and "duplicate JSON key" in reason
                for reason in duplicate_key_verdict["reasons"]
            ),
            verdict=duplicate_key_verdict,
        )

        _docs, root_certificate, root_bundle = write_strict_fixture()
        root_bundle["bundle_root_sha256"] = "0" * 64
        (custom_bundle_dir / "certification_bundle.json").write_text(
            json.dumps(root_bundle, sort_keys=True) + "\n", encoding="utf-8"
        )
        root_mismatch_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_root_mismatch_rejected",
            not root_mismatch_verdict["ok"]
            and any(
                "root hash does not match" in reason
                for reason in root_mismatch_verdict["reasons"]
            ),
            verdict=root_mismatch_verdict,
        )

        _docs, signature_certificate, signature_bundle = write_strict_fixture()
        invalid_signature_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: False,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_invalid_detached_signatures_rejected",
            not invalid_signature_verdict["ok"]
            and sum(
                "signature verification failed" in reason
                for reason in invalid_signature_verdict["reasons"]
            )
            == 3,
            verdict=invalid_signature_verdict,
        )

        _docs, reused_key_certificate, _bundle = write_strict_fixture()
        reused_fingerprint = "D" * 40
        reused_trust = {
            signer: reused_fingerprint for signer in reused_key_certificate["signers"]
        }
        for signature in reused_key_certificate["detached_signatures"]:
            signature["fingerprint"] = reused_fingerprint
        reused_key_certificate["signed_claim_sha256"] = (
            _certificate_signed_claim_sha256(
                reused_key_certificate,
                reused_key_certificate["evidence_bundle_sha256"],
            )
        )
        (custom_bundle_dir / "release_certificate.json").write_text(
            json.dumps(reused_key_certificate, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reused_key_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _claim: True,
            trusted_signers=reused_trust,
            ci_run_verifier=strict_verifier_kwargs["ci_run_verifier"],
        )
        check(
            "bundle_one_key_cannot_impersonate_three_signers",
            not reused_key_verdict["ok"]
            and any(
                "distinct signers, roles, fingerprints" in reason
                for reason in reused_key_verdict["reasons"]
            ),
            verdict=reused_key_verdict,
        )

        signature_certificate["signers"] = ["same@example.test"] * 3
        (custom_bundle_dir / "release_certificate.json").write_text(
            json.dumps(signature_certificate, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        repeated_signer_verdict = certificate_bundle_verdict(
            certificate_root,
            fixed_head,
            required_artifacts=required_strict_artifacts,
            bundle_dir=custom_bundle_dir,
            now=fixed_now,
            worktree_state=clean_worktree,
            signature_verifier=lambda _root, _signature, _bundle_root: True,
            **strict_verifier_kwargs,
        )
        check(
            "bundle_requires_three_distinct_signers",
            not repeated_signer_verdict["ok"]
            and any(
                "distinct signers" in reason
                for reason in repeated_signer_verdict["reasons"]
            ),
            verdict=repeated_signer_verdict,
        )

    # bd-re8.15 — the live-stream fold.
    check(
        "eprocess_classify_maps_real_shapes",
        classify_invariant_line(
            {"test": "e2e", "case": "determinism_gate", "result": "pass"}
        )
        == ("INV-DETERMINISM", False)
        and classify_invariant_line({"check": "int32_overflow_proof", "result": "fail"})
        == ("INV-I32-NOOVERFLOW", True)
        and classify_invariant_line({"case": "kv_cap_bound", "result": "pass"})
        == ("INV-KV-CAP", False)
        and classify_invariant_line({"test": "robot_selftest", "result": "pass"})
        == ("INV-SIMD-SCALAR", False)
        # skips are NOT observations; unmatched lines are None.
        and classify_invariant_line(
            {"case": "determinism_gate", "result": "skip_no_model"}
        )
        is None
        and classify_invariant_line({"case": "l4_tokens", "result": "pass"}) is None,
    )
    ep_clean = _eprocess_from_state("INV-DETERMINISM", None)
    for _ in range(5):
        ep_clean.observe(False)
    check(
        "eprocess_clean_stream_shrinks",
        ep_clean.e_value < 1.0 and ep_clean.rejected_at is None,
        e=ep_clean.e_value,
    )
    ep_trip = _eprocess_from_state("INV-DETERMINISM", None)
    tripped = ep_trip.observe(True)
    check(
        "eprocess_single_hardware_violation_trips",
        tripped and ep_trip.rejected_at == 1,
        e=ep_trip.e_value,
    )
    saved = {
        "e_value": ep_clean.e_value,
        "obs_count": ep_clean.obs_count,
        "rejected_at": None,
    }
    ep_resumed = _eprocess_from_state("INV-DETERMINISM", saved)
    ep_resumed.observe(False)
    check(
        "eprocess_state_roundtrip_never_resets",
        ep_resumed.obs_count == 6 and ep_resumed.e_value < ep_clean.e_value,
    )
    # Injected violation end-to-end, ALL FOUR invariants: the emission-shaped
    # fail line classifies to the right invariant and a single genuine alarm
    # crosses the Ville threshold (acceptance: alarms on injected violation,
    # silent on the clean stream — the clean side is the checks above).
    injected = {
        "INV-DETERMINISM": {"case": "determinism_gate", "result": "fail"},
        "INV-SIMD-SCALAR": {"command": "robot.selftest", "verdict": "fail"},
        "INV-I32-NOOVERFLOW": {
            "test": "int32_overflow_proof",
            "case": "i32_overflow_headroom_k6848",
            "result": "fail",
        },
        "INV-KV-CAP": {
            "test": "spec_ring_rollback",
            "case": "kv_cap_ring_bound",
            "result": "fail",
        },
    }
    for name, line in injected.items():
        got = classify_invariant_line(line)
        ep_inj = _eprocess_from_state(name, None)
        tripped_inj = (
            got == (name, True) and ep_inj.observe(True) and ep_inj.rejected_at == 1
        )
        check(
            f"eprocess_injected_violation_trips_{name}",
            tripped_inj,
            classified=str(got),
        )

    if failures:
        emit("gauntlet-cert-self-test", False, failed=failures)
        return 1
    emit("gauntlet-cert-self-test", True, checks_passed=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--self-test", action="store_true", help="validate the gauntlet math (CI gate)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="print a worked franken_ocr scorecard + ratchet verdict",
    )
    parser.add_argument(
        "--from-parity",
        metavar="MD",
        help="compute the surface-pillar scorecard from the REAL FEATURE_PARITY.md (bd-re8.13)",
    )
    parser.add_argument(
        "--residuals",
        metavar="JSON",
        help="held-out round residuals (JSON array file); absent -> bootstrap wide band",
    )
    parser.add_argument(
        "--scorecard-out", metavar="FILE", help="also write the scorecard artifact here"
    )
    parser.add_argument(
        "--convergence",
        metavar="ROUNDS_JSONL",
        nargs="?",
        const="docs/gauntlet/ROUNDS.jsonl",
        help="check the >=10-rounds / >=2-clean convergence gate; exit 1 until met (bd-wp8.8)",
    )
    parser.add_argument(
        "--eprocess-fold",
        metavar="RAW_NDJSON",
        help="fold a real test-log NDJSON stream into the persisted e-process state (bd-re8.15)",
    )
    parser.add_argument(
        "--release-readiness",
        action="store_true",
        help="the all-green ship gate: read every cell's evidence artifact, exit 1 on ANY red (bd-wp8.10)",
    )
    parser.add_argument(
        "--readiness-out",
        metavar="FILE",
        default="docs/gauntlet/RELEASE_READINESS.json",
        help="where to write the release-readiness scorecard artifact",
    )
    parser.add_argument(
        "--eprocess-state",
        metavar="FILE",
        default="docs/gauntlet/EPROCESS_STATE.json",
        help="the persisted (never-reset) per-invariant e-process state",
    )
    parser.add_argument(
        "--bundle",
        metavar="OUT_DIR",
        nargs="?",
        const=".gauntlet-output/bundle",
        help="produce the release certification bundle (bd-wp8.9); exit 1 until certified",
    )
    parser.add_argument(
        "--finalize-bundle",
        metavar="OUT_DIR",
        help="derive, sign, and verify a provisional bundle from downloaded CI evidence",
    )
    parser.add_argument(
        "--workflow-evidence",
        metavar="MANIFEST",
        action="append",
        default=[],
        help="downloaded gauntlet.workflow_evidence.v1 manifest (repeat per artifact)",
    )
    parser.add_argument(
        "--trusted-signer",
        metavar="ROLE:IDENTITY:FINGERPRINT",
        action="append",
        default=[],
        help="active trusted OpenPGP signer input (repeat for all three roles)",
    )
    parser.add_argument(
        "--model-parity-evidence",
        metavar="SCORECARD",
        help="package one real armed weighted-ladder scorecard for CI reconstruction",
    )
    parser.add_argument(
        "--model-parity-raw",
        metavar="NDJSON",
        help="structured raw NDJSON emitted beside --model-parity-evidence",
    )
    parser.add_argument(
        "--model-parity-fixtures",
        metavar="DIR",
        help="pinned physical oracle-fixture root for --model-parity-evidence",
    )
    parser.add_argument(
        "--performance-evidence",
        metavar="DIR",
        help="package one complete real gauntlet_runbook performance tree",
    )
    parser.add_argument(
        "--workflow-artifact-name",
        metavar="NAME",
        help="GitHub Actions artifact name used by a workflow evidence producer",
    )
    parser.add_argument(
        "--ci-gate-evidence",
        metavar="JOB",
        help="package one successful CI scripts/check.sh job and its physical log",
    )
    parser.add_argument(
        "--ci-gate-log",
        metavar="FILE",
        help="physical scripts/check.sh log for --ci-gate-evidence",
    )
    parser.add_argument(
        "--dist-target-evidence",
        metavar="TARGET",
        help="package one successful portable dist target",
    )
    parser.add_argument(
        "--dist-ref-preflight",
        action="store_true",
        help="fail unless the Actions ref, Cargo version, and origin/main agree",
    )
    parser.add_argument(
        "--dist-asset",
        metavar="FILE",
        help="built binary whose adjacent .sha256 is bound by --dist-target-evidence",
    )
    parser.add_argument(
        "--dist-glibc-floor",
        metavar="VERSION",
        help="glibc floor used to link and audit a Linux dist asset",
    )
    parser.add_argument(
        "--dist-installed-asset",
        metavar="FILE",
        help="native Windows asset installed offline by install.ps1",
    )
    parser.add_argument(
        "--dist-installer-e2e-log",
        metavar="FILE",
        help="native offline install.ps1 transcript for a Windows dist asset",
    )
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.demo:
        return _demo()
    if args.from_parity:
        return from_parity(args.from_parity, args.residuals, args.scorecard_out)
    if args.convergence:
        return convergence(args.convergence)
    if args.eprocess_fold:
        return eprocess_fold(args.eprocess_fold, args.eprocess_state)
    if args.release_readiness:
        return release_readiness(args.readiness_out)
    if args.bundle:
        return produce_bundle(args.bundle)
    if args.dist_ref_preflight:
        return dist_ref_preflight()
    if args.finalize_bundle:
        if not args.workflow_evidence:
            parser.error("--finalize-bundle requires --workflow-evidence")
        if len(args.trusted_signer) != CERTIFICATION_REQUIRED_SIGNERS:
            parser.error(
                f"--finalize-bundle requires exactly {CERTIFICATION_REQUIRED_SIGNERS} "
                "--trusted-signer inputs"
            )
        return finalize_bundle(
            args.finalize_bundle, args.workflow_evidence, args.trusted_signer
        )
    if args.model_parity_evidence:
        if not (
            args.model_parity_raw
            and args.model_parity_fixtures
            and args.workflow_artifact_name
        ):
            parser.error(
                "--model-parity-evidence requires --model-parity-raw, "
                "--model-parity-fixtures, and --workflow-artifact-name"
            )
        return produce_model_parity_evidence(
            args.model_parity_evidence,
            args.model_parity_raw,
            args.model_parity_fixtures,
            args.workflow_artifact_name,
        )
    if args.performance_evidence:
        if not args.workflow_artifact_name:
            parser.error(
                "--performance-evidence requires --workflow-artifact-name"
            )
        return produce_performance_evidence(
            args.performance_evidence, args.workflow_artifact_name
        )
    if args.ci_gate_evidence:
        if not (args.ci_gate_log and args.workflow_artifact_name):
            parser.error(
                "--ci-gate-evidence requires --ci-gate-log and "
                "--workflow-artifact-name"
            )
        return produce_ci_gate_evidence(
            args.ci_gate_evidence, args.ci_gate_log, args.workflow_artifact_name
        )
    if args.dist_target_evidence:
        if not (args.dist_asset and args.workflow_artifact_name):
            parser.error(
                "--dist-target-evidence requires --dist-asset and "
                "--workflow-artifact-name"
            )
        return produce_dist_target_evidence(
            args.dist_target_evidence,
            args.dist_asset,
            args.workflow_artifact_name,
            args.dist_glibc_floor,
            args.dist_installed_asset,
            args.dist_installer_e2e_log,
        )
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
