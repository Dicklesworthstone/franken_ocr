#!/usr/bin/env python3
"""Validate franken_ocr structured test-log NDJSON.

With --self-test this validates the schema fixture and a small corpus of good and
bad log lines. With file arguments it validates each supplied NDJSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "tests" / "fixtures" / "test_log_schema.json"
HEX64 = set("0123456789abcdef")


def emit(check: str, ok: bool, **fields: object) -> None:
    payload = {"check": check, "result": "pass" if ok else "fail", **fields}
    print(json.dumps(payload, sort_keys=True))


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def require_type(value: Any, expected: type | tuple[type, ...], field: str, errors: list[str]) -> None:
    if not isinstance(value, expected):
        if expected is int and isinstance(value, bool):
            errors.append(f"{field}: expected int, got bool")
            return
        errors.append(f"{field}: expected {expected}, got {type(value).__name__}")


def validate_log_record(record: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for field in schema["required_common"]:
        if field not in record:
            errors.append(f"missing common field {field}")

    if errors:
        return errors

    if record["schema_version"] != schema["schema_version"]:
        errors.append(
            f"schema_version mismatch: expected {schema['schema_version']}, got {record['schema_version']}"
        )

    require_type(record["ts"], (int, float), "ts", errors)
    require_type(record["test"], str, "test", errors)
    require_type(record["case"], str, "case", errors)
    require_type(record["run_seq"], int, "run_seq", errors)
    if isinstance(record["run_seq"], bool):
        errors.append("run_seq: expected int, got bool")

    event = record["event"]
    result = record["result"]
    if event not in schema["enums"]["event"]:
        errors.append(f"unknown event {event!r}")
        return errors
    if result not in schema["enums"]["result"]:
        errors.append(f"unknown result {result!r}")

    for field in schema["required_by_event"][event]:
        if field not in record:
            errors.append(f"event {event!r} missing field {field}")

    if event == "stage":
        if record.get("stage") not in schema["enums"]["stage"]:
            errors.append(f"unknown stage {record.get('stage')!r}")
        if record.get("dtype") not in schema["enums"]["dtype"]:
            errors.append(f"unknown dtype {record.get('dtype')!r}")
        if record.get("simd_tier") not in schema["enums"]["simd_tier"]:
            errors.append(f"unknown simd_tier {record.get('simd_tier')!r}")
        require_type(record.get("inputs"), dict, "inputs", errors)
        require_type(record.get("shapes"), dict, "shapes", errors)
        require_type(record.get("elapsed_us"), int, "elapsed_us", errors)
        require_type(record.get("seed"), int, "seed", errors)
        if "layer_idx" in record:
            require_type(record["layer_idx"], int, "layer_idx", errors)

    if event == "parity":
        if record.get("gate") not in schema["enums"]["gate"]:
            errors.append(f"unknown gate {record.get('gate')!r}")
        if record.get("metric") not in schema["enums"]["metric"]:
            errors.append(f"unknown metric {record.get('metric')!r}")
        require_type(record.get("value"), (int, float, bool), "value", errors)
        require_type(record.get("tolerance"), (int, float, bool), "tolerance", errors)
        require_type(record.get("oracle_fixture"), str, "oracle_fixture", errors)
        sha = record.get("oracle_sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(ch not in HEX64 for ch in sha.lower()):
            errors.append("oracle_sha256: expected 64 hex characters")
        require_type(record.get("nondeterminism_envelope"), dict, "nondeterminism_envelope", errors)
        require_type(record.get("pass"), bool, "pass", errors)
        if record.get("simd_tier") == "avx2" and "avx2_exception" not in record:
            errors.append("avx2 parity line requires avx2_exception")

    if event == "assert":
        require_type(record.get("assertion"), str, "assertion", errors)
        require_type(record.get("pass"), bool, "pass", errors)

    if event == "skip":
        require_type(record.get("reason"), str, "reason", errors)

    if event == "result":
        require_type(record.get("elapsed_us"), int, "elapsed_us", errors)

    diag = record.get("diag")
    if event == "error" or result == "fail":
        require_type(diag, dict, "diag", errors)
        if isinstance(diag, dict):
            for field in schema["required_diag_fields"]:
                if field not in diag:
                    errors.append(f"diag missing field {field}")
            if "focr_exit_code" in diag:
                require_type(diag["focr_exit_code"], int, "diag.focr_exit_code", errors)

    native = schema["native_path_proof"]
    if record.get(native["field"]) is True and record.get(native["fallback_field"]) != native["required_fallback"]:
        errors.append(
            f"{native['field']}=true requires {native['fallback_field']}={native['required_fallback']!r}"
        )

    return errors


def validate_line(raw: str, schema: dict[str, Any], source: str, line_no: int) -> bool:
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        emit("test-log-line", False, source=source, line=line_no, error=f"invalid json: {exc}")
        return False
    if not isinstance(record, dict):
        emit("test-log-line", False, source=source, line=line_no, error="line is not a JSON object")
        return False
    errors = validate_log_record(record, schema)
    emit("test-log-line", not errors, source=source, line=line_no, errors=errors)
    return not errors


def validate_file(path: Path, schema: dict[str, Any]) -> bool:
    ok = True
    seen = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        seen += 1
        ok = validate_line(raw, schema, str(path), line_no) and ok
    emit("test-log-file", ok and seen > 0, file=str(path), lines=seen)
    return ok and seen > 0


def good_records() -> list[dict[str, Any]]:
    base = {
        "schema_version": 1,
        "ts": 1.0,
        "test": "schema_self_test",
        "case": "case0",
        "run_seq": 0,
        "result": "pass",
    }
    sha = "a" * 64
    return [
        {**base, "event": "setup", "seed": 1234},
        {
            **base,
            "event": "stage",
            "stage": "rswa_attn",
            "inputs": {"tokens": 273},
            "shapes": {"hidden": [273, 1280]},
            "dtype": "f32",
            "elapsed_us": 10,
            "simd_tier": "scalar",
            "seed": 1234,
            "layer_idx": 0,
        },
        {
            **base,
            "event": "parity",
            "gate": "L0",
            "metric": "max_abs_diff",
            "value": 0.0,
            "tolerance": 0.0,
            "oracle_fixture": "tests/fixtures/native/base_001/preprocess.npy",
            "oracle_sha256": sha,
            "nondeterminism_envelope": {"source": "oracle_nondeterminism_envelope.json"},
            "pass": True,
        },
        {**base, "event": "assert", "assertion": "schema accepts assert", "pass": True},
        {**base, "event": "skip", "result": "skip_no_model", "reason": "model unavailable"},
        {**base, "event": "result", "elapsed_us": 100},
        {
            **base,
            "event": "error",
            "result": "fail",
            "diag": {"error_kind": "NotImplemented", "focr_exit_code": 1, "message": "self-test"},
        },
    ]


def self_test(schema: dict[str, Any]) -> bool:
    ok = True
    for idx, record in enumerate(good_records(), start=1):
        errors = validate_log_record(record, schema)
        emit("self-test-good-record", not errors, case=idx, errors=errors)
        ok = ok and not errors

    bad_records = [
        {"schema_version": 1, "event": "stage", "result": "pass"},
        {**good_records()[0], "event": "not_an_event"},
        {**good_records()[1], "stage": "typo_stage"},
        {**good_records()[2], "simd_tier": "avx2"},
        {**good_records()[5], "result": "fail"},
        {**good_records()[4], "native_path_ran": True, "fallback_target": "real_model"},
    ]
    for idx, record in enumerate(bad_records, start=1):
        errors = validate_log_record(record, schema)
        emit("self-test-bad-record", bool(errors), case=idx, errors=errors)
        ok = ok and bool(errors)

    emit("test-log-schema-summary", ok, schema=str(SCHEMA_PATH))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path, help="NDJSON log files to validate")
    parser.add_argument("--self-test", action="store_true", help="validate the schema and embedded good/bad examples")
    args = parser.parse_args()

    schema = load_schema()
    ok = True
    if args.self_test or not args.logs:
        ok = self_test(schema) and ok
    for path in args.logs:
        ok = validate_file(path, schema) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
