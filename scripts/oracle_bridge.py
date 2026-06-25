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


BRIDGE_SCHEMA_VERSION = 1
PINNED_TORCH = "2.10.0"
PINNED_TRANSFORMERS = "4.57.1"
DEFAULT_SEED = 1337
DEFAULT_THREADS = 1
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
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
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            "OMP_NUM_THREADS": str(threads),
            "TORCH_NUM_THREADS": str(threads),
            "FOCR_ORACLE_SEED": str(seed),
        }
    )
    return env


def parse_seed_threads(payload: dict[str, Any]) -> tuple[int, int]:
    try:
        seed = int(payload.get("seed", DEFAULT_SEED))
        threads = int(payload.get("threads", DEFAULT_THREADS))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid seed/threads: {exc}") from exc
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    return seed, threads


def f32_ordered_bits(value: float) -> int:
    bits = struct.unpack(">i", struct.pack(">f", float(value)))[0]
    return bits if bits >= 0 else 0x80000000 - bits


def round_f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


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
    if not values:
        raise ValueError("rmsnorm input must be non-empty")
    square_sum = round_f32(0.0)
    for value in values:
        value_f32 = round_f32(value)
        square_sum = round_f32(square_sum + round_f32(value_f32 * value_f32))
    mean_square = round_f32(square_sum / round_f32(float(len(values))))
    inv_rms = round_f32(1.0 / math.sqrt(round_f32(mean_square + round_f32(eps))))
    return [
        round_f32(round_f32(round_f32(value) * inv_rms) * round_f32(scale))
        for value, scale in zip(values, weight, strict=True)
    ]


def parse_rmsnorm_payload(payload: dict[str, Any]) -> tuple[list[float], list[float], float, int, int]:
    seed, threads = parse_seed_threads(payload)
    values_raw = payload["values"]
    weight_raw = payload["weight"]
    if not isinstance(values_raw, list) or not isinstance(weight_raw, list):
        raise ValueError("values and weight must be JSON arrays")
    values = [float(value) for value in values_raw]
    weight = [float(value) for value in weight_raw]
    if len(values) != len(weight):
        raise ValueError("rmsnorm values and weight lengths differ")
    if not values:
        raise ValueError("rmsnorm input must be non-empty")
    return values, weight, float(payload["eps"]), seed, threads


def apply_reference_determinism(torch: Any, transformers: Any, seed: int, threads: int) -> dict[str, object]:
    record: dict[str, object] = {
        "seed": seed,
        "requested_threads": threads,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "transformers_set_seed": False,
        "torch_manual_seed": False,
        "torch_cuda_manual_seed_all": False,
        "torch_deterministic_algorithms": False,
        "torch_threads": None,
    }

    transformers.set_seed(seed)
    record["transformers_set_seed"] = True

    torch.manual_seed(seed)
    record["torch_manual_seed"] = True

    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        record["torch_cuda_manual_seed_all"] = True

    torch.use_deterministic_algorithms(True)
    record["torch_deterministic_algorithms"] = True

    torch.set_num_threads(threads)
    record["torch_threads"] = torch.get_num_threads()
    return record


def run_worker(payload: dict[str, Any]) -> dict[str, object]:
    op = payload.get("op")
    if op != "rmsnorm_f32":
        return json_response("fail", error=f"unsupported oracle op {op!r}")
    try:
        values_list, weight_list, eps, seed, threads = parse_rmsnorm_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return json_response("fail", error=f"invalid rmsnorm_f32 request: {exc}")

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
            reason="oracle dependency versions are not pinned",
            torch_version=torch.__version__,
            transformers_version=transformers_version,
            required_torch=PINNED_TORCH,
            required_transformers=PINNED_TRANSFORMERS,
        )

    determinism = apply_reference_determinism(torch, transformers, seed, threads)

    values = torch.tensor(values_list, dtype=torch.float32)
    weight = torch.tensor(weight_list, dtype=torch.float32)
    out = values * torch.rsqrt(torch.mean(values * values) + eps) * weight
    return json_response(
        "pass",
        identity=EngineIdentity.ORACLE.value,
        op=op,
        seed=seed,
        determinism=determinism,
        deterministic_algorithms=determinism["torch_deterministic_algorithms"],
        torch_threads=determinism["torch_threads"],
        output=[float(v) for v in out.tolist()],
    )


def call_oracle(payload: dict[str, Any], python: str = sys.executable, timeout_s: float = 10.0) -> dict[str, object]:
    try:
        seed, threads = parse_seed_threads(payload)
    except ValueError as exc:
        return json_response("fail", error=str(exc))
    try:
        proc = subprocess.run(
            [python, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(payload, sort_keys=True),
            text=True,
            capture_output=True,
            env=reference_env(seed, threads),
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
        env["PYTHONHASHSEED"] == "7"
        and env["CUBLAS_WORKSPACE_CONFIG"] == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        and env["OMP_NUM_THREADS"] == "2"
        and env["TORCH_NUM_THREADS"] == "2"
        and env["FOCR_ORACLE_SEED"] == "7",
    )
    try:
        reference_env(seed=-1)
    except ValueError as exc:
        check("reference-env-rejects-negative-seed", "seed must be >= 0" in str(exc))
    else:
        check("reference-env-rejects-negative-seed", False)

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

        def __init__(self) -> None:
            self.seed: int | None = None
            self.threads = 0
            self.deterministic = False

        def manual_seed(self, seed: int) -> None:
            self.seed = seed

        def use_deterministic_algorithms(self, enabled: bool) -> None:
            self.deterministic = enabled

        def set_num_threads(self, threads: int) -> None:
            self.threads = threads

        def get_num_threads(self) -> int:
            return self.threads

    class FakeTransformers:
        def __init__(self) -> None:
            self.seed: int | None = None

        def set_seed(self, seed: int) -> None:
            self.seed = seed

    old_hashseed = os.environ.get("PYTHONHASHSEED")
    old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ["PYTHONHASHSEED"] = "11"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        fake_torch = FakeTorch()
        fake_transformers = FakeTransformers()
        record = apply_reference_determinism(fake_torch, fake_transformers, seed=11, threads=3)
    finally:
        if old_hashseed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = old_hashseed
        if old_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_cublas
    check(
        "determinism-record-self-test",
        record["seed"] == 11
        and record["requested_threads"] == 3
        and record["pythonhashseed"] == "11"
        and record["cublas_workspace_config"] == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        and record["transformers_set_seed"] is True
        and record["torch_manual_seed"] is True
        and record["torch_deterministic_algorithms"] is True
        and record["torch_threads"] == 3,
        detail=record,
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

    negative_seed_request = dict(request)
    negative_seed_request["seed"] = -1
    negative_seed = call_oracle(negative_seed_request)
    check(
        "oracle-call-rejects-negative-seed",
        negative_seed.get("result") == "fail" and "seed must be >= 0" in str(negative_seed.get("error")),
        detail=negative_seed,
    )

    oracle = call_oracle(request)
    if oracle.get("result") in {"skip_no_oracle", "skip_unpinned_oracle"}:
        emit("oracle-subprocess-smoke", True, skipped=True, reason=oracle.get("reason"), detail=oracle)
    else:
        check("oracle-subprocess-result", oracle.get("result") == "pass", detail=oracle)
        if oracle.get("result") == "pass":
            output = oracle.get("output")
            check("oracle-output-vector", isinstance(output, list), detail=oracle)
            if isinstance(output, list):
                try:
                    output_values = [float(value) for value in output]
                except (TypeError, ValueError) as exc:
                    check("oracle-output-values-numeric", False, error=str(exc), detail=oracle)
                else:
                    cmp = compare_vectors(subject, output_values, ULP_TOLERANCE_BY_OP["rmsnorm_f32"])
                    check(
                        "oracle-rmsnorm-within-ulp",
                        bool(cmp["within_tolerance"]),
                        max_ulp=cmp["max_ulp"],
                        tolerance=ULP_TOLERANCE_BY_OP["rmsnorm_f32"],
                    )
            check("oracle-identity", oracle.get("identity") == EngineIdentity.ORACLE.value)
            check("oracle-deterministic-flag", oracle.get("deterministic_algorithms") is True)
            determinism = oracle.get("determinism")
            check("oracle-determinism-record", isinstance(determinism, dict), detail=oracle)
            if isinstance(determinism, dict):
                check("oracle-determinism-seed", determinism.get("seed") == DEFAULT_SEED, detail=determinism)
                check("oracle-determinism-threads", determinism.get("torch_threads") == DEFAULT_THREADS, detail=determinism)
                check(
                    "oracle-determinism-cublas",
                    determinism.get("cublas_workspace_config") == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
                    detail=determinism,
                )
                check("oracle-determinism-manual-seed", determinism.get("torch_manual_seed") is True, detail=determinism)

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
