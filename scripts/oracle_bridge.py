#!/usr/bin/env python3
"""Test-only subprocess bridge to the pinned Unlimited-OCR oracle.

The shipping Rust binary must never link Python or torch. This script is only a
test harness helper: it launches a separate Python process for the pinned
reference stack when available, forces deterministic torch settings, and emits
structured JSON for parity checks. Its self-test is safe on machines without the
6.67 GB model or torch; missing/unpinned oracle dependencies are reported as a
skip-with-success, not as a false parity pass.
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SCHEMA_VERSION = 1
PINNED_TORCH = "2.10.0"
PINNED_TRANSFORMERS = "4.57.1"
DEFAULT_SEED = 1337
DEFAULT_THREADS = 1
ULP_TOLERANCE_BY_OP = {
    "matmul_f32": 4,
    "rmsnorm_f32": 2,
    "elementwise_f32": 2,
}


class EngineIdentity(str, enum.Enum):
    SUBJECT = "subject"
    ORACLE = "oracle"


def emit(check: str, ok: bool, **fields: object) -> None:
    payload = {"check": check, "result": "pass" if ok else "fail", **fields}
    print(json.dumps(payload, sort_keys=True))


def json_response(result: str, **fields: object) -> dict[str, object]:
    return {"schema_version": BRIDGE_SCHEMA_VERSION, "result": result, **fields}


def assert_distinct_identities() -> bool:
    return EngineIdentity.SUBJECT.value != EngineIdentity.ORACLE.value


def reference_env(seed: int = DEFAULT_SEED, threads: int = DEFAULT_THREADS) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": str(threads),
            "TORCH_NUM_THREADS": str(threads),
            "FOCR_ORACLE_SEED": str(seed),
        }
    )
    return env


def f32_ordered_bits(value: float) -> int:
    bits = struct.unpack(">i", struct.pack(">f", float(value)))[0]
    return bits if bits >= 0 else 0x80000000 - bits


def ulp_distance_f32(lhs: float, rhs: float) -> int:
    if math.isnan(lhs) or math.isnan(rhs):
        return 0 if math.isnan(lhs) and math.isnan(rhs) else 2**31
    if lhs == rhs:
        return 0
    return abs(f32_ordered_bits(lhs) - f32_ordered_bits(rhs))


def compare_vectors(lhs: list[float], rhs: list[float], max_ulp: int) -> dict[str, object]:
    if len(lhs) != len(rhs):
        return {"within_tolerance": False, "max_ulp": None, "error": "length mismatch"}
    distances = [ulp_distance_f32(a, b) for a, b in zip(lhs, rhs, strict=True)]
    observed = max(distances, default=0)
    return {"within_tolerance": observed <= max_ulp, "max_ulp": observed}


def subject_rmsnorm(values: list[float], weight: list[float], eps: float) -> list[float]:
    if len(values) != len(weight):
        raise ValueError("rmsnorm values and weight lengths differ")
    mean_square = sum(v * v for v in values) / len(values)
    inv_rms = 1.0 / math.sqrt(mean_square + eps)
    return [float(v * inv_rms * w) for v, w in zip(values, weight, strict=True)]


def run_worker(payload: dict[str, Any]) -> dict[str, object]:
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:
        return json_response("skip_no_oracle", reason=f"missing oracle dependency: {exc.name}")

    torch_version = torch.__version__.split("+", 1)[0]
    transformers_version = transformers.__version__
    if torch_version != PINNED_TORCH or transformers_version != PINNED_TRANSFORMERS:
        return json_response(
            "skip_unpinned_oracle",
            torch_version=torch.__version__,
            transformers_version=transformers_version,
            required_torch=PINNED_TORCH,
            required_transformers=PINNED_TRANSFORMERS,
        )

    seed = int(payload.get("seed", DEFAULT_SEED))
    threads = int(payload.get("threads", DEFAULT_THREADS))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)

    op = payload.get("op")
    if op != "rmsnorm_f32":
        return json_response("fail", error=f"unsupported oracle op {op!r}")

    values = torch.tensor(payload["values"], dtype=torch.float32)
    weight = torch.tensor(payload["weight"], dtype=torch.float32)
    eps = float(payload["eps"])
    out = values * torch.rsqrt(torch.mean(values * values) + eps) * weight
    return json_response(
        "pass",
        identity=EngineIdentity.ORACLE.value,
        op=op,
        seed=seed,
        deterministic_algorithms=True,
        torch_threads=torch.get_num_threads(),
        output=[float(v) for v in out.tolist()],
    )


def call_oracle(payload: dict[str, Any], python: str = sys.executable, timeout_s: float = 10.0) -> dict[str, object]:
    try:
        proc = subprocess.run(
            [python, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(payload, sort_keys=True),
            text=True,
            capture_output=True,
            env=reference_env(int(payload.get("seed", DEFAULT_SEED)), int(payload.get("threads", DEFAULT_THREADS))),
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json_response("fail", error="oracle worker timed out", timeout_s=timeout_s)
    if proc.returncode != 0:
        return json_response("fail", error="oracle worker failed", returncode=proc.returncode, stderr=proc.stderr)
    try:
        decoded = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return json_response("fail", error=f"oracle worker emitted invalid JSON: {exc}", stdout=proc.stdout)
    if not isinstance(decoded, dict):
        return json_response("fail", error="oracle worker did not return a JSON object")
    return decoded


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, **fields: object) -> None:
        emit(name, cond, **fields)
        if not cond:
            failures.append(name)

    check("engine-identities-distinct", assert_distinct_identities())
    check("ulp-tolerance-matmul", ULP_TOLERANCE_BY_OP["matmul_f32"] == 4)
    check("ulp-tolerance-elementwise", ULP_TOLERANCE_BY_OP["elementwise_f32"] == 2)
    check("ulp-tolerance-rmsnorm", ULP_TOLERANCE_BY_OP["rmsnorm_f32"] == 2)
    env = reference_env(seed=7, threads=2)
    check(
        "deterministic-reference-env",
        env["PYTHONHASHSEED"] == "7" and env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and env["OMP_NUM_THREADS"] == "2",
    )

    request = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "op": "rmsnorm_f32",
        "identity": EngineIdentity.SUBJECT.value,
        "values": [1.0, -2.0, 3.5, -4.25],
        "weight": [1.0, 0.5, 1.25, 0.75],
        "eps": 1e-6,
        "seed": DEFAULT_SEED,
        "threads": DEFAULT_THREADS,
    }
    subject = subject_rmsnorm(request["values"], request["weight"], request["eps"])
    local_cmp = compare_vectors(subject, subject, ULP_TOLERANCE_BY_OP["rmsnorm_f32"])
    check("subject-rmsnorm-self-compare", bool(local_cmp["within_tolerance"]), max_ulp=local_cmp["max_ulp"])

    oracle = call_oracle(request)
    if oracle.get("result") in {"skip_no_oracle", "skip_unpinned_oracle"}:
        emit("oracle-subprocess-smoke", True, skipped=True, reason=oracle.get("reason"), detail=oracle)
    else:
        check("oracle-subprocess-result", oracle.get("result") == "pass", detail=oracle)
        if oracle.get("result") == "pass":
            cmp = compare_vectors(subject, list(oracle["output"]), ULP_TOLERANCE_BY_OP["rmsnorm_f32"])
            check(
                "oracle-rmsnorm-within-ulp",
                bool(cmp["within_tolerance"]),
                max_ulp=cmp["max_ulp"],
                tolerance=ULP_TOLERANCE_BY_OP["rmsnorm_f32"],
            )
            check("oracle-identity", oracle.get("identity") == EngineIdentity.ORACLE.value)
            check("oracle-deterministic-flag", oracle.get("deterministic_algorithms") is True)

    if failures:
        emit("oracle-bridge-self-test", False, failed=failures)
        return 1
    emit("oracle-bridge-self-test", True, checks_passed=True)
    return 0


def worker_main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(json.dumps(json_response("fail", error=f"invalid request JSON: {exc}"), sort_keys=True))
        return 0
    if not isinstance(payload, dict):
        print(json.dumps(json_response("fail", error="request must be a JSON object"), sort_keys=True))
        return 0
    print(json.dumps(run_worker(payload), sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run stdlib-safe bridge self-tests")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return worker_main()
    if args.self_test:
        return self_test()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
