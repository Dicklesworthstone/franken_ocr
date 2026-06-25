#!/usr/bin/env python3
"""Validate the docs/focrq-format.md contract.

This is a docs-lint for bead bd-1es.1. It does not validate real .focrq files;
it keeps the format specification itself machine-checkable until the reader and
writer land.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "focrq-format.md"

REQUIRED_TOKENS = [
    'b"FOCRQ\\0"',
    "format_version",
    "arch_target",
    "source_sha256",
    "header_len",
    "header_sha256",
    "payload_len",
    "little-endian",
    "license_notice",
    "Copyright (c) 2026 Baidu",
    "MIT License",
    "model_config",
    "tensor_directory",
    "QInt8PerChan",
    "QInt4PerGroup",
    "scales_offset",
    "scales_len",
    "group_size",
    "BF16",
    "FormatMismatch",
    "round_ties_to_even",
    "Aarch64Smmla",
    "X86Vnni",
    "X86Amx",
]

PREFIX_FIELDS = [
    "magic",
    "format_version",
    "arch_target",
    "source_sha256",
    "header_len",
    "header_sha256",
    "payload_len",
]

ARCH_TARGETS = {
    "Generic": 0,
    "Aarch64Smmla": 1,
    "X86Vnni": 2,
    "X86Amx": 3,
}

DTYPES = {"F32", "F16", "BF16", "QInt8PerChan", "QInt4PerGroup"}
PACKINGS = {"RowMajor", "Aarch64Smmla2x8", "Aarch64Sdot4x16", "X86VnniU8S8", "X86AmxTile16x16"}
TIERS = {"BF16", "F32", "Int8", "Int4G16", "Int4G32"}

REQUIRED_MODEL_CONFIG = [
    "model_type",
    "torch_dtype",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "v_head_dim",
    "intermediate_size",
    "moe_intermediate_size",
    "n_routed_experts",
    "num_experts_per_tok",
    "n_shared_experts",
    "vocab_size",
    "max_position_embeddings",
    "sliding_window",
    "use_mla",
    "vision_config",
    "projector_config",
    "source_hashes.config_json_sha256",
    "source_hashes.model_index_sha256",
    "source_hashes.tokenizer_json_sha256",
]


def emit(check: str, ok: bool, **fields: object) -> None:
    payload = {"check": check, "result": "pass" if ok else "fail", **fields}
    print(json.dumps(payload, sort_keys=True))


def fail(failures: list[str], message: str, check: str, **fields: object) -> None:
    emit(check, False, error=message, **fields)
    failures.append(message)


def extract_json_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in re.finditer(r"```json\n(.*?)\n```", text, flags=re.DOTALL):
        raw = match.group(1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            blocks.append(value)
    return blocks


def extract_table_values(text: str, heading: str) -> set[str]:
    marker = f"### `{heading}`"
    start = text.find(marker)
    if start == -1:
        return set()
    next_heading = text.find("\n### ", start + len(marker))
    section = text[start:] if next_heading == -1 else text[start:next_heading]
    values: set[str] = set()
    for line in section.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0] and cells[0] not in {"Value", "JSON string"}:
            if cells[0].isdigit() and len(cells) >= 2:
                values.add(cells[1])
            elif not set(cells[0]) <= {"-", ":"}:
                values.add(cells[0])
    return values


def main() -> int:
    failures: list[str] = []
    exists = SPEC.is_file()
    emit("focrq-spec-exists", exists, file=str(SPEC.relative_to(ROOT)))
    if not exists:
        return 1

    text = SPEC.read_text(encoding="utf-8")

    for token in REQUIRED_TOKENS:
        ok = token in text
        emit("focrq-required-token", ok, token=token)
        if not ok:
            failures.append(f"missing required token {token!r}")

    for field in PREFIX_FIELDS:
        pattern = rf"\|\s*`[^`]+`\s*\|\s*\d+\s*\|\s*`{re.escape(field)}`\s*\|"
        ok = bool(re.search(pattern, text))
        emit("focrq-prefix-field", ok, field=field)
        if not ok:
            failures.append(f"fixed prefix table missing {field}")

    arch_values = extract_table_values(text, "arch_target")
    dtype_values = extract_table_values(text, "dtype")
    packing_values = extract_table_values(text, "packing")
    for name, expected, actual in (
        ("arch_target", set(ARCH_TARGETS), arch_values),
        ("dtype", DTYPES, dtype_values),
        ("packing", PACKINGS, packing_values),
    ):
        ok = expected.issubset(actual)
        emit("focrq-enum-table", ok, enum=name, expected=sorted(expected), actual=sorted(actual))
        if not ok:
            failures.append(f"{name} enum missing {sorted(expected - actual)}")

    for tier in sorted(TIERS):
        ok = f"- `{tier}`" in text
        emit("focrq-tier-value", ok, tier=tier)
        if not ok:
            failures.append(f"tier enum missing {tier}")

    for field in REQUIRED_MODEL_CONFIG:
        ok = f"`{field}`" in text
        emit("focrq-model-config-field", ok, field=field)
        if not ok:
            failures.append(f"model_config list missing {field}")

    headers = [
        block
        for block in extract_json_blocks(text)
        if {"format_version", "arch_target", "license_notice", "model_config", "tensor_directory"}.issubset(block)
    ]
    ok = bool(headers)
    emit("focrq-header-example-present", ok, count=len(headers))
    if not ok:
        failures.append("no parseable header JSON example found")
    else:
        header = headers[-1]
        checks = {
            "format_version": header.get("format_version") == 1,
            "arch_target": header.get("arch_target") in ARCH_TARGETS,
            "license_notice": "Copyright (c) 2026 Baidu" in header.get("license_notice", "")
            and "MIT License" in header.get("license_notice", ""),
            "provenance": isinstance(header.get("provenance"), dict)
            and "hf_commit" in header["provenance"]
            and "github_commit" in header["provenance"]
            and "source_sha256_hex" in header["provenance"],
            "model_config": isinstance(header.get("model_config"), dict),
            "packing_manifest": isinstance(header.get("packing_manifest"), dict),
            "tensor_directory": isinstance(header.get("tensor_directory"), list),
        }
        for name, passed in checks.items():
            emit("focrq-header-example-field", passed, field=name)
            if not passed:
                failures.append(f"header example failed {name}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    emit("focrq-spec-summary", True, file=str(SPEC.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
