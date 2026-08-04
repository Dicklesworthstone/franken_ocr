#!/usr/bin/env python3
"""Generate or verify the pinned Torch-2.10 CPU MoE primitive fixture.

This tool is deliberately model-free. It freezes the two CPU primitives whose
bit order affects Unlimited-OCR MoE inference: ``topk(sorted=False)`` at
``[64] -> [6]`` and ``sum(dim=1)`` at ``[N, 6, 1280]``. The canonical oracle is
the pinned macOS arm64 Torch wheel; other hosts fail closed instead of emitting
a platform-dependent replacement fixture.

``--check`` only reads the committed fixture and exits nonzero on drift.
``--write`` is the explicit human-reviewed golden update path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TORCH_VERSION = "2.10.0"
TORCH_COMMIT = "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
TOPK_SOURCE_SHA256 = "dddf5fc982e7d25f9ba38fe1ebd6645fb8851485f25fd5e975822ae90828635a"
NTH_ELEMENT_SHA256 = "e0152c1647c275c112fea5fb477b6859a2b2f213d905d28183159731d77ffdd9"
LIBCXX_SORT_SHA256 = "33f739d139c79d5467aa69083e696b157c3d73e72711ff7552e6cd66e87d3f36"
MODEL_SOURCE_SHA256 = "74e36e6bd0ba7bc565ef76464a99baa8e6bccb710ae9c1007b54ac30b855fa4c"
TOPK_CASES = 2048
EXPERTS = 64
TOP_K = 6
REDUCTION_CASES = 256
HIDDEN = 1280
THREAD_COUNTS = (1, 2, 4, 8)
MASK64 = (1 << 64) - 1
TOPK_SEED = 0x6A09E667F3BCC909
REDUCTION_SEED = 0xBB67AE8584CAA73B
CHECK_COMMAND = (
    "uv run --python 3.12 --with torch==2.10.0 python "
    "scripts/gen_moe_torch_2_10_cpu_fixture.py --check "
    "tests/fixtures/moe_torch_2_10_cpu.json"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = REPO_ROOT / "docs" / "truth-pack" / "snapshots" / "modeling_deepseekv2.py"


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) & MASK64


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def tensor_u32(torch: Any, tensor: Any) -> list[int]:
    return [int(value) & 0xFFFFFFFF for value in tensor.contiguous().view(torch.int32).reshape(-1).tolist()]


def tensor_bytes(torch: Any, tensor: Any) -> bytes:
    values = tensor.contiguous().view(torch.uint8).reshape(-1).tolist()
    return bytes(int(value) for value in values)


def assert_runtime(torch: Any) -> dict[str, Any]:
    version = str(torch.__version__).split("+", 1)[0]
    if version != TORCH_VERSION:
        raise SystemExit(f"expected torch=={TORCH_VERSION}, got {torch.__version__}")
    git_commit = str(getattr(torch.version, "git_version", ""))
    if git_commit != TORCH_COMMIT:
        raise SystemExit(f"expected torch commit {TORCH_COMMIT}, got {git_commit or '<missing>'}")
    if sys.byteorder != "little":
        raise SystemExit(f"canonical fixture requires little-endian bytes, got {sys.byteorder}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit(
            "canonical fixture requires the pinned macOS arm64 Torch CPU wheel; "
            f"got {platform.system()} {platform.machine()}"
        )

    config = torch.__config__.show()
    required = ("clang 15.0.0", "C++ Version: 201703", "BUILD_TYPE=Release", "-DNDEBUG")
    missing = [needle for needle in required if needle not in config]
    if missing:
        raise SystemExit(f"Torch compiler contract missing {missing!r}\n{config}")
    model_source_hash = sha256_file(MODEL_SOURCE)
    if model_source_hash != MODEL_SOURCE_SHA256:
        raise SystemExit(
            f"reference model source hash drifted: {model_source_hash} != {MODEL_SOURCE_SHA256}"
        )

    macos_version = platform.mac_ver()[0]
    if not macos_version:
        raise SystemExit("cannot identify the canonical macOS runtime version")
    return {
        "torch": TORCH_VERSION,
        "torch_git_tag": f"v{TORCH_VERSION}",
        "torch_git_commit": TORCH_COMMIT,
        "platform": f"macOS-{macos_version}-arm64",
        "torch_build_compiler": "clang 15.0.0, C++17, NDEBUG",
        "topk_source": "aten/src/ATen/native/TopKImpl.h:43-93",
        "topk_source_sha256": TOPK_SOURCE_SHA256,
        "libcxx_nth_element_source": (
            "llvm-project llvmorg-15.0.7 libcxx/include/__algorithm/nth_element.h"
        ),
        "libcxx_nth_element_sha256": NTH_ELEMENT_SHA256,
        "libcxx_sort_source": "llvm-project llvmorg-15.0.7 libcxx/include/__algorithm/sort.h",
        "libcxx_sort_sha256": LIBCXX_SORT_SHA256,
        "libcxx_provenance_scope": (
            "transcription source; torch wheel metadata does not expose the Apple SDK header "
            "commit, so the corpus SHA-256 is the behavioral identity proof"
        ),
        "reference_model_source_sha256": MODEL_SOURCE_SHA256,
        "reference_model_source_lines": "modeling_deepseekv2.py:449-453,631-703",
    }


def manual_cases(torch: Any) -> list[dict[str, Any]]:
    torch.manual_seed(1)
    unique = torch.rand(EXPERTS, dtype=torch.float32)
    unsorted = torch.topk(unique, TOP_K, sorted=False)
    sorted_result = torch.topk(unique, TOP_K, sorted=True)
    tied = torch.tensor([(index * 17) % 7 for index in range(EXPERTS)], dtype=torch.float32)
    tied_unsorted = torch.topk(tied, TOP_K, sorted=False)
    return [
        {
            "name": "unique_torch_manual_seed_1",
            "scores_f32_bits": tensor_u32(torch, unique),
            "torch_unsorted_indices": [int(value) for value in unsorted.indices.tolist()],
            "torch_unsorted_value_bits": tensor_u32(torch, unsorted.values),
            "torch_sorted_indices": [int(value) for value in sorted_result.indices.tolist()],
            "torch_sorted_value_bits": tensor_u32(torch, sorted_result.values),
        },
        {
            "name": "exact_ties_i_times_17_mod_7",
            "score_formula": "(i * 17) % 7 for i in 0..64",
            "torch_unsorted_indices": [int(value) for value in tied_unsorted.indices.tolist()],
            "torch_unsorted_value_bits": tensor_u32(torch, tied_unsorted.values),
        },
    ]


def topk_score(case_index: int, expert: int, random: int) -> float:
    mode = case_index % 8
    if mode == 0:
        return f32_from_bits(0x3F000000 | (random & 0x007FFFFF))
    if mode == 1:
        return float(random % 2)
    if mode == 2:
        return float(random % 3)
    if mode == 3:
        return float(random % 7)
    if mode == 4:
        selector = random % 11
        if selector == 0:
            return float("inf")
        if selector == 1:
            return float("-inf")
        if selector == 2:
            return -0.0
        return float((random % 19) - 9)
    if mode == 5:
        return f32_from_bits(0x7FC00000) if random % 13 == 0 else float(random % 17)
    if mode == 6:
        return float((expert * 17 + case_index * 13) % 64)
    return 100.0 if (expert + case_index) % 9 == 0 else float((random % 5) - 2)


def topk_corpus(torch: Any) -> dict[str, Any]:
    rng = SplitMix64(TOPK_SEED)
    rows = [
        [topk_score(case_index, expert, rng.next()) for expert in range(EXPERTS)]
        for case_index in range(TOPK_CASES)
    ]
    scores = torch.tensor(rows, dtype=torch.float32)
    indices = torch.topk(scores, TOP_K, dim=-1, sorted=False).indices
    output = bytes(int(value) for value in indices.reshape(-1).tolist())
    spot_cases = (0, 1, 2, 3, 4, 5, 6, 7, 255, 1023, 2047)
    return {
        "case_count": TOPK_CASES,
        "expert_count": EXPERTS,
        "top_k": TOP_K,
        "seed_hex": f"{TOPK_SEED:016x}",
        "prng": (
            "SplitMix64: state += 0x9e3779b97f4a7c15; "
            "z=(state^(state>>30))*0xbf58476d1ce4e5b9; "
            "z=(z^(z>>27))*0x94d049bb133111eb; output=z^(z>>31), all u64 wrapping"
        ),
        "generation_order": (
            "case 0..2048, then expert 0..64; consume exactly one PRNG output r per score; "
            "mode=case%8"
        ),
        "modes": [
            "0: f32::from_bits(0x3f000000 | (r & 0x007fffff))",
            "1: f32(r % 2)",
            "2: f32(r % 3)",
            "3: f32(r % 7)",
            "4: r%11 == 0:+inf, 1:-inf, 2:-0.0, otherwise f32(i64(r%19)-9)",
            "5: r%13 == 0:canonical NaN, otherwise f32(r%17)",
            "6: f32((expert*17 + case*13) % 64), while still consuming r",
            "7: if (expert+case)%9 == 0 then 100.0 else f32(i64(r%5)-2)",
        ],
        "output_encoding": "concatenate the six returned expert indices as u8 in case order",
        "torch_output_bytes": len(output),
        "torch_output_sha256": sha256_bytes(output),
        "spot_indices": {
            str(case): [int(value) for value in indices[case].tolist()] for case in spot_cases
        },
    }


def reduction_tensor(torch: Any) -> Any:
    storage = bytearray(REDUCTION_CASES * TOP_K * HIDDEN * 4)
    rng = SplitMix64(REDUCTION_SEED)
    cancellation = (16_777_216.0, 1.0, -16_777_216.0, 1.0, 1.0, 1.0)

    def put(case: int, slot: int, channel: int, bits: int) -> None:
        offset = ((case * TOP_K + slot) * HIDDEN + channel) * 4
        struct.pack_into("<I", storage, offset, bits & 0xFFFFFFFF)

    for case in range(REDUCTION_CASES):
        mode = case % 4
        for channel in range(HIDDEN):
            if mode == 1:
                rotation = rng.next() % TOP_K
                for slot in range(TOP_K):
                    put(case, slot, channel, f32_bits(cancellation[(slot + rotation) % TOP_K]))
                continue
            for slot in range(TOP_K):
                random = rng.next()
                if mode == 0:
                    bits = (
                        ((random >> 63) << 31)
                        | ((125 + ((random >> 60) & 3)) << 23)
                        | (random & 0x007FFFFF)
                    )
                elif mode == 2:
                    bits = (
                        ((random >> 63) << 31)
                        | ((1 + ((random >> 32) % 253)) << 23)
                        | (random & 0x007FFFFF)
                    )
                else:
                    bits = f32_bits(float((random % 17) - 8))
                put(case, slot, channel, bits)
    return torch.frombuffer(storage, dtype=torch.float32).reshape(REDUCTION_CASES, TOP_K, HIDDEN).clone()


def reduction_corpus(torch: Any) -> dict[str, Any]:
    values = reduction_tensor(torch)
    reference = None
    reference_bytes = b""
    reference_hash = ""
    for threads in THREAD_COUNTS:
        torch.set_num_threads(threads)
        result = values.sum(dim=1)
        output = tensor_bytes(torch, result)
        digest = sha256_bytes(output)
        if reference is None:
            reference = result
            reference_bytes = output
            reference_hash = digest
        elif digest != reference_hash:
            raise SystemExit(
                f"reduction output changed at {threads} threads: {digest} != {reference_hash}"
            )

    sequential = torch.zeros((REDUCTION_CASES, HIDDEN), dtype=torch.float32)
    for slot in range(TOP_K):
        sequential.add_(values[:, slot, :])
    mismatches = int((sequential.view(torch.int32) != reference.view(torch.int32)).sum().item())
    return {
        "case_count": REDUCTION_CASES,
        "top_k": TOP_K,
        "hidden": HIDDEN,
        "seed_hex": f"{REDUCTION_SEED:016x}",
        "prng": "the same SplitMix64 transition as topk_corpus",
        "generation_order": (
            "case 0..256, channel 0..1280, then slot 0..6 where the mode consumes "
            "per-slot values; mode=case%4"
        ),
        "modes": [
            "0: for each slot consume r; bits=(r>>63)<<31 | (125+((r>>60)&3))<<23 | (r&0x007fffff)",
            "1: consume one r per channel; rotation=r%6; slot value=[16777216,1,-16777216,1,1,1][(slot+rotation)%6]",
            "2: for each slot consume r; bits=(r>>63)<<31 | (1+((r>>32)%253))<<23 | (r&0x007fffff)",
            "3: for each slot consume r; value=f32(i64(r%17)-8)",
        ],
        "torch_expression": "x.view(256,6,1280).sum(dim=1)",
        "output_encoding": (
            "all 256*1280 f32 results as little-endian IEEE-754 bytes in row-major order"
        ),
        "torch_output_bytes": len(reference_bytes),
        "torch_output_sha256": reference_hash,
        "sequential_slot_fold_bit_mismatches": mismatches,
        "thread_counts_checked": list(THREAD_COUNTS),
    }


def left_fold(torch: Any, values: list[float], order: list[int]) -> tuple[int, float]:
    result = torch.tensor(0.0, dtype=torch.float32)
    for slot in order:
        result = result + torch.tensor(values[slot], dtype=torch.float32)
    return tensor_u32(torch, result)[0], float(result.item())


def weighted_combine(torch: Any) -> dict[str, Any]:
    indices = [30, 60, 19, 49, 8, 38]
    values = [16_777_216.0, 1.0, -16_777_216.0, 1.0, 1.0, 1.0]
    production_input = (
        torch.tensor(values, dtype=torch.float32)
        .reshape(1, TOP_K, 1)
        .expand(1, TOP_K, HIDDEN)
        .contiguous()
    )
    production = production_input.sum(dim=1)
    scalar = torch.tensor(values, dtype=torch.float32).reshape(1, TOP_K, 1).sum(dim=1)
    slot_bits, slot_value = left_fold(torch, values, list(range(TOP_K)))
    ascending_order = sorted(range(TOP_K), key=lambda slot: indices[slot])
    ascending_bits, ascending_value = left_fold(torch, values, ascending_order)
    return {
        "slot_expert_indices": indices,
        "contribution_f32_bits": [f32_bits(value) for value in values],
        "weight_f32_bits": [f32_bits(1.0)] * TOP_K,
        "rust_ascending_left_fold_f32_bits": ascending_bits,
        "rust_ascending_left_fold": ascending_value,
        "slot_order_left_fold_f32_bits": slot_bits,
        "slot_order_left_fold": slot_value,
        "torch_scalar_shape_1x6x1_sum_dim_1_f32_bits": tensor_u32(torch, scalar)[0],
        "torch_scalar_shape_1x6x1_sum_dim_1": float(scalar.item()),
        "torch_production_shape_1x6x1280_sum_dim_1_f32_bits": tensor_u32(
            torch, production[0, 0]
        )[0],
        "torch_production_shape_1x6x1280_sum_dim_1": float(production[0, 0].item()),
    }


def generated_fields(torch: Any) -> dict[str, Any]:
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    script_path = Path(__file__).resolve()
    return {
        "runtime": assert_runtime(torch),
        "oracle_command": CHECK_COMMAND,
        "generator": {
            "path": "scripts/gen_moe_torch_2_10_cpu_fixture.py",
            "sha256": sha256_file(script_path),
            "check_command": CHECK_COMMAND,
        },
        "embedded_generator_test": (
            "src/native_engine/moe.rs::tests::{torch_2_10_topk_matches_2048_case_oracle_corpus,"
            "production_six_term_combine_matches_256_case_torch_oracle,"
            "moe_policy_env_subprocess_matrix}"
        ),
        "cases": manual_cases(torch),
        "topk_corpus": topk_corpus(torch),
        "reduction_corpus": reduction_corpus(torch),
        "weighted_combine": weighted_combine(torch),
    }


def check_fixture(path: Path, expected: dict[str, Any]) -> None:
    before = sha256_file(path)
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual.get("schema") != "focr-moe-topk-oracle/v1":
        raise SystemExit(f"{path}: unexpected schema {actual.get('schema')!r}")
    mismatches = [field for field, value in expected.items() if actual.get(field) != value]
    if mismatches:
        for field in mismatches:
            print(f"fixture mismatch: {field}", file=sys.stderr)
            print("expected:", json.dumps(expected[field], sort_keys=True), file=sys.stderr)
            print("actual:  ", json.dumps(actual.get(field), sort_keys=True), file=sys.stderr)
        raise SystemExit(f"{path}: {len(mismatches)} semantic field(s) drifted")
    after = sha256_file(path)
    if after != before:
        raise SystemExit(f"--check mutated {path}: {before} -> {after}")
    print(
        json.dumps(
            {
                "status": "ok",
                "fixture": str(path),
                "fixture_sha256": before,
                "generator_sha256": expected["generator"]["sha256"],
                "topk_sha256": expected["topk_corpus"]["torch_output_sha256"],
                "reduction_sha256": expected["reduction_corpus"]["torch_output_sha256"],
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", type=Path, help="verify a fixture without modifying it")
    mode.add_argument("--write", type=Path, help="explicitly rewrite the canonical fixture")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    fields = generated_fields(torch)
    if args.check is not None:
        check_fixture(args.check, fields)
        return 0

    payload = {
        "schema": "focr-moe-topk-oracle/v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        **fields,
    }
    args.write.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.write} sha256={sha256_file(args.write)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
