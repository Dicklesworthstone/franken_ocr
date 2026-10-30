#!/usr/bin/env python3
"""Validate oracle fixture provenance and artifact hashes.

This is the bd-re8.1.1 guard. It is strict once tests/fixtures/native contains
oracle fixtures, but it skips successfully while the CUDA-only oracle corpus has
not been generated on this machine.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess  # nosec B404 - live replay runs the repo-pinned generator with shell=False.
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "tests" / "fixtures" / "native"

PIN_TORCH = "2.10.0"
PIN_TRANSFORMERS = "4.57.1"
HF_COMMIT = "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"
GITHUB_COMMIT = "7e98affeacba24e95562fbaa234ddb89b856874a"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPLAY_SCHEMA_VERSION = 1
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# Unlimited-OCR-2 (U2-A6). The v2 oracle has its own stack and pin, never the v1 ones
# above: HF f32 on CPU, committed twin in tests/fixtures/uocr2_oracle, dumps off-tree.
UOCR2_ORACLE = ROOT / "tests" / "fixtures" / "uocr2_oracle"
UOCR2_GENERATOR = ROOT / "scripts" / "gen_reference_fixtures_uocr2.py"
UOCR2_SOURCE_HASHES = ROOT / "docs" / "truth-pack" / "SOURCE_HASHES.md"
UOCR2_PIN_PYTHON = "3.10"
UOCR2_PIN_TORCH = "2.13.0"
UOCR2_PIN_TRANSFORMERS = "4.52.1"
UOCR2_PIN_TOKENIZERS = "0.21.4"
UOCR2_PIN_HF_HUB = "0.36.0"
UOCR2_HF_COMMIT = "5632329a32e103d5f1add270681e0f5c1233c6e4"
UOCR2_NGRAM_SIZE = 35
UOCR2_NGRAM_WINDOW_SINGLE = 128
UOCR2_NGRAM_WINDOW_MULTI = 1024
UOCR2_MAX_PAGE_TOKENS = 20000
UOCR2_IDS_HEADER = re.compile(r"^# .+ generated token ids: (\S+) \((\d+) ids\)$")


def emit(check: str, ok: bool, **fields: object) -> None:
    payload = {"check": check, "result": "pass" if ok else "fail", **fields}
    print(json.dumps(payload, sort_keys=True))


def fail(failures: list[str], message: str, check: str, **fields: object) -> None:
    emit(check, False, error=message, **fields)
    failures.append(message)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_json_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_non_empty_string_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(arg, str) and arg for arg in value)


def replay_script_args(command_argv: object) -> list[str] | None:
    if not is_non_empty_string_list(command_argv):
        return None
    argv = list(command_argv)
    for index, arg in enumerate(argv):
        if Path(arg).name == "gen_reference_fixtures.py":
            return argv[index + 1 :]
    return None


def option_value(args: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, arg in enumerate(args):
        if arg == option and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def replace_option(args: list[str], option: str, value: str) -> list[str]:
    replaced = False
    out: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == option:
            out.extend([option, value])
            replaced = True
            skip_next = True
        elif arg.startswith(f"{option}="):
            out.append(f"{option}={value}")
            replaced = True
        else:
            out.append(arg)
    if not replaced:
        out.extend([option, value])
    return out


def load_json(path: Path, failures: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(failures, f"{path}: invalid json: {exc}", "oracle-json-parse", file=str(path.relative_to(ROOT)))
        return None
    if not isinstance(value, dict):
        fail(failures, f"{path}: expected JSON object", "oracle-json-object", file=str(path.relative_to(ROOT)))
        return None
    emit("oracle-json-parse", True, file=str(path.relative_to(ROOT)))
    return value


def require_eq(
    failures: list[str],
    source: Path,
    payload: dict[str, Any],
    field: str,
    expected: object,
    *,
    normalize_version: bool = False,
) -> None:
    actual = payload.get(field)
    comparable = actual.split("+", 1)[0] if normalize_version and isinstance(actual, str) else actual
    ok = comparable == expected
    emit("oracle-provenance-field", ok, file=str(source.relative_to(ROOT)), field=field, expected=expected, actual=actual)
    if not ok:
        failures.append(f"{source}: {field}={actual!r}, expected {expected!r}")


def require_hex(
    failures: list[str],
    source: Path,
    payload: dict[str, Any],
    field: str,
    pattern: re.Pattern[str],
) -> None:
    actual = payload.get(field)
    ok = isinstance(actual, str) and bool(pattern.fullmatch(actual))
    emit("oracle-provenance-hex", ok, file=str(source.relative_to(ROOT)), field=field, actual=actual)
    if not ok:
        failures.append(f"{source}: {field} must match {pattern.pattern}")


def validate_provenance(path: Path, value: dict[str, Any], failures: list[str]) -> None:
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        fail(failures, f"{path}: missing provenance object", "oracle-provenance-object", file=str(path.relative_to(ROOT)))
        return
    emit("oracle-provenance-object", True, file=str(path.relative_to(ROOT)))

    require_eq(failures, path, provenance, "pinned_torch", PIN_TORCH)
    require_eq(failures, path, provenance, "pinned_transformers", PIN_TRANSFORMERS)
    require_eq(failures, path, provenance, "torch_version", PIN_TORCH, normalize_version=True)
    require_eq(failures, path, provenance, "transformers_version", PIN_TRANSFORMERS)
    require_eq(failures, path, provenance, "hf_commit", HF_COMMIT)
    require_eq(failures, path, provenance, "github_commit", GITHUB_COMMIT)
    require_eq(failures, path, provenance, "oracle_is_correctness_golden", True)
    require_hex(failures, path, provenance, "hf_commit", HEX40)
    require_hex(failures, path, provenance, "github_commit", HEX40)
    require_hex(failures, path, provenance, "model_weights_sha256", HEX64)

    command_argv = provenance.get("command_argv")
    exact_command = provenance.get("exact_command")
    ok_argv = is_non_empty_string_list(command_argv)
    emit("oracle-command-argv", ok_argv, file=str(path.relative_to(ROOT)), argc=len(command_argv) if isinstance(command_argv, list) else None)
    if not ok_argv:
        failures.append(f"{path}: command_argv must be a non-empty list of strings")
    ok_command = isinstance(exact_command, str) and "gen_reference_fixtures.py" in exact_command
    emit("oracle-exact-command", ok_command, file=str(path.relative_to(ROOT)), exact_command=exact_command)
    if not ok_command:
        failures.append(f"{path}: exact_command must name gen_reference_fixtures.py")

    model_bytes = provenance.get("model_weights_bytes")
    ok_bytes = is_json_int(model_bytes) and model_bytes > 0
    emit("oracle-model-bytes", ok_bytes, file=str(path.relative_to(ROOT)), bytes=model_bytes)
    if not ok_bytes:
        failures.append(f"{path}: model_weights_bytes must be a positive integer")

    determinism = provenance.get("determinism")
    if not isinstance(determinism, dict):
        fail(failures, f"{path}: missing determinism object", "oracle-determinism-object", file=str(path.relative_to(ROOT)))
    else:
        seed = determinism.get("seed")
        ok_seed = is_json_int(seed) and seed >= 0
        emit("oracle-determinism-seed", ok_seed, file=str(path.relative_to(ROOT)), seed=seed)
        if not ok_seed:
            failures.append(f"{path}: determinism.seed must be a non-negative integer")
        for field in ("torch_manual_seed", "torch_deterministic_algorithms"):
            ok_flag = determinism.get(field) is True
            emit("oracle-determinism-flag", ok_flag, file=str(path.relative_to(ROOT)), field=field, actual=determinism.get(field))
            if not ok_flag:
                failures.append(f"{path}: determinism.{field} must be true")
        cublas_ok = determinism.get("cublas_workspace_config") == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        emit(
            "oracle-determinism-cublas-workspace",
            cublas_ok,
            file=str(path.relative_to(ROOT)),
            expected=DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            actual=determinism.get("cublas_workspace_config"),
        )
        if not cublas_ok:
            failures.append(f"{path}: determinism.cublas_workspace_config must be {DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG}")

    generation = provenance.get("generation_config")
    if not isinstance(generation, dict):
        fail(failures, f"{path}: missing generation_config object", "oracle-generation-config", file=str(path.relative_to(ROOT)))
    else:
        deterministic_generation = generation.get("temperature") == 0.0 and generation.get("do_sample") is False
        emit(
            "oracle-generation-deterministic",
            deterministic_generation,
            file=str(path.relative_to(ROOT)),
            temperature=generation.get("temperature"),
            do_sample=generation.get("do_sample"),
        )
        if not deterministic_generation:
            failures.append(f"{path}: generation_config must be greedy deterministic")


def validate_deterministic_replay(
    path: Path,
    value: dict[str, Any],
    decoded: object,
    expected_decoded_sha: object,
    failures: list[str],
) -> None:
    replay = value.get("deterministic_replay")
    if not isinstance(replay, dict):
        fail(failures, f"{path}: missing deterministic_replay object", "oracle-replay-object", file=str(path.relative_to(ROOT)))
        return
    emit("oracle-replay-object", True, file=str(path.relative_to(ROOT)))

    schema_version = replay.get("schema_version")
    schema_ok = is_json_int(schema_version) and schema_version == REPLAY_SCHEMA_VERSION
    emit("oracle-replay-schema", schema_ok, file=str(path.relative_to(ROOT)), schema_version=replay.get("schema_version"))
    if not schema_ok:
        failures.append(f"{path}: deterministic_replay.schema_version must be {REPLAY_SCHEMA_VERSION}")

    seed = replay.get("rng_seed")
    seed_ok = is_json_int(seed) and seed >= 0
    emit("oracle-replay-seed", seed_ok, file=str(path.relative_to(ROOT)), seed=seed)
    if not seed_ok:
        failures.append(f"{path}: deterministic_replay.rng_seed must be a non-negative integer")

    requires_cuda_ok = replay.get("requires_cuda") is True
    emit("oracle-replay-requires-cuda", requires_cuda_ok, file=str(path.relative_to(ROOT)), requires_cuda=replay.get("requires_cuda"))
    if not requires_cuda_ok:
        failures.append(f"{path}: correctness-golden replay must require CUDA")

    kind_ok = replay.get("expected_prefix_kind") == "full_decoded_text"
    emit("oracle-replay-prefix-kind", kind_ok, file=str(path.relative_to(ROOT)), kind=replay.get("expected_prefix_kind"))
    if not kind_ok:
        failures.append(f"{path}: deterministic_replay.expected_prefix_kind must be full_decoded_text")

    prefix_chars = replay.get("expected_prefix_chars")
    expected_chars = len(decoded) if isinstance(decoded, str) else None
    chars_ok = is_json_int(prefix_chars) and prefix_chars == expected_chars
    emit("oracle-replay-prefix-chars", chars_ok, file=str(path.relative_to(ROOT)), expected=expected_chars, actual=prefix_chars)
    if not chars_ok:
        failures.append(f"{path}: deterministic_replay.expected_prefix_chars mismatch")

    for field in ("expected_prefix_sha256", "expected_decoded_text_sha256"):
        actual = replay.get(field)
        ok = isinstance(actual, str) and actual == expected_decoded_sha
        emit("oracle-replay-sha", ok, file=str(path.relative_to(ROOT)), field=field, expected=expected_decoded_sha, actual=actual)
        if not ok:
            failures.append(f"{path}: deterministic_replay.{field} mismatch")

    replay_argv = replay.get("replay_command_argv")
    argv_ok = is_non_empty_string_list(replay_argv) and any("gen_reference_fixtures.py" in arg for arg in replay_argv)
    emit("oracle-replay-command", argv_ok, file=str(path.relative_to(ROOT)), argc=len(replay_argv) if isinstance(replay_argv, list) else None)
    if not argv_ok:
        failures.append(f"{path}: deterministic_replay.replay_command_argv must be a non-empty list of strings naming gen_reference_fixtures.py")

    provenance = value.get("provenance")
    if isinstance(provenance, dict):
        determinism = provenance.get("determinism")
        if isinstance(determinism, dict):
            seed_match = determinism.get("seed") == seed
            emit(
                "oracle-replay-seed-matches-provenance",
                seed_match,
                file=str(path.relative_to(ROOT)),
                replay_seed=seed,
                provenance_seed=determinism.get("seed"),
            )
            if not seed_match:
                failures.append(f"{path}: replay seed does not match provenance determinism seed")


def validate_reference_payload(path: Path, value: dict[str, Any], failures: list[str]) -> set[Path]:
    covered_npys: set[Path] = set()
    schema_version = value.get("schema_version")
    schema_ok = is_json_int(schema_version) and schema_version == 1
    emit("oracle-reference-schema", schema_ok, file=str(path.relative_to(ROOT)), schema_version=schema_version)
    if not schema_ok:
        failures.append(f"{path}: schema_version must be 1")

    validate_provenance(path, value, failures)

    decoded = value.get("decoded_text")
    expected = value.get("decoded_text_sha256")
    if decoded is None:
        ok = expected is None
        emit("oracle-decoded-text-sha", ok, file=str(path.relative_to(ROOT)), expected=expected, actual=None)
        if not ok:
            failures.append(f"{path}: decoded_text_sha256 must be null when decoded_text is null")
    elif isinstance(decoded, str) and isinstance(expected, str):
        actual = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
        ok = actual == expected
        emit("oracle-decoded-text-sha", ok, file=str(path.relative_to(ROOT)), expected=expected, actual=actual)
        if not ok:
            failures.append(f"{path}: decoded_text_sha256 mismatch")
    else:
        fail(failures, f"{path}: decoded_text/decoded_text_sha256 have invalid types", "oracle-decoded-text-sha", file=str(path.relative_to(ROOT)))

    validate_deterministic_replay(path, value, decoded, expected, failures)

    activations = value.get("activations", {})
    if not isinstance(activations, dict):
        fail(failures, f"{path}: activations must be an object", "oracle-activations-object", file=str(path.relative_to(ROOT)))
        return covered_npys
    stem = path.name.removesuffix("_reference.json")
    for stage, record in sorted(activations.items()):
        if not isinstance(record, dict):
            fail(failures, f"{path}: activation {stage} must be an object", "oracle-activation-record", file=str(path.relative_to(ROOT)), stage=stage)
            continue
        file_name = record.get("file")
        expected_file_sha = record.get("file_sha256")
        npy_path = NATIVE / "activations" / stem / str(file_name)
        exists = isinstance(file_name, str) and npy_path.is_file()
        emit("oracle-activation-file", exists, file=str(npy_path.relative_to(ROOT)) if isinstance(file_name, str) else None, stage=stage)
        if not exists:
            failures.append(f"{path}: missing activation file for {stage}: {file_name!r}")
            continue
        covered_npys.add(npy_path)
        actual_file_sha = sha256_file(npy_path)
        ok = isinstance(expected_file_sha, str) and actual_file_sha == expected_file_sha
        emit("oracle-activation-file-sha", ok, file=str(npy_path.relative_to(ROOT)), stage=stage, expected=expected_file_sha, actual=actual_file_sha)
        if not ok:
            failures.append(f"{npy_path}: file_sha256 mismatch")
    return covered_npys


def validate_reference_json(path: Path, failures: list[str]) -> set[Path]:
    value = load_json(path, failures)
    if value is None:
        return set()
    return validate_reference_payload(path, value, failures)


def validate_manifest(path: Path, failures: list[str]) -> None:
    value = load_json(path, failures)
    if value is None:
        return
    schema_version = value.get("schema_version")
    schema_ok = is_json_int(schema_version) and schema_version == 1
    emit("oracle-manifest-schema", schema_ok, file=str(path.relative_to(ROOT)), schema_version=value.get("schema_version"))
    if not schema_ok:
        failures.append(f"{path}: schema_version must be 1")
    validate_provenance(path, value, failures)

    documents = value.get("documents")
    n_documents = value.get("n_documents")
    ok_docs = isinstance(documents, list) and is_json_int(n_documents) and n_documents == len(documents)
    emit("oracle-manifest-documents", ok_docs, file=str(path.relative_to(ROOT)), n_documents=n_documents, actual=len(documents) if isinstance(documents, list) else None)
    if not ok_docs:
        failures.append(f"{path}: documents must be a list and n_documents must match")
        return
    for doc in documents:
        if not isinstance(doc, dict):
            failures.append(f"{path}: document entry must be an object")
            continue
        golden = doc.get("golden")
        golden_path = NATIVE / str(golden)
        exists = isinstance(golden, str) and golden_path.is_file()
        emit("oracle-manifest-golden", exists, file=str(golden_path.relative_to(ROOT)) if isinstance(golden, str) else None)
        if not exists:
            failures.append(f"{path}: missing golden reference {golden!r}")

    md_path = path.with_suffix(".md")
    md_exists = md_path.is_file()
    emit("oracle-manifest-markdown", md_exists, file=str(md_path.relative_to(ROOT)))
    if not md_exists:
        failures.append(f"{path}: missing sibling {md_path.name}")
    else:
        text = md_path.read_text(encoding="utf-8")
        for token in (f"torch=={PIN_TORCH}", f"transformers=={PIN_TRANSFORMERS}", HF_COMMIT, GITHUB_COMMIT, "Exact command"):
            ok = token in text
            emit("oracle-manifest-markdown-token", ok, file=str(md_path.relative_to(ROOT)), token=token)
            if not ok:
                failures.append(f"{md_path}: missing token {token!r}")


def cuda_replay_available() -> tuple[bool, str]:
    try:
        import torch  # type: ignore[import-not-found]  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001 - optional offline oracle dependency
        return False, f"torch unavailable: {exc}"
    try:
        if not bool(torch.cuda.is_available()):
            return False, "CUDA unavailable"
    except Exception as exc:  # noqa: BLE001 - backend probing can fail on partial installs
        return False, f"CUDA probe failed: {exc}"
    return True, "CUDA available"


def replay_prereq_skip(args: list[str]) -> str | None:
    ok_cuda, cuda_reason = cuda_replay_available()
    if not ok_cuda:
        return cuda_reason

    model_dir = option_value(args, "--model-dir") or os.environ.get("FOCR_MODEL_DIR")
    if not model_dir:
        return "no --model-dir and FOCR_MODEL_DIR unset"
    if not Path(model_dir).expanduser().is_dir():
        return f"model dir not found: {model_dir}"

    corpus = option_value(args, "--corpus") or "tests/fixtures/corpus"
    if not Path(corpus).expanduser().is_dir():
        return f"corpus dir not found: {corpus}"
    return None


def load_replay_expectations(reference_paths: list[Path], failures: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    grouped: dict[tuple[str, ...], dict[str, str]] = {}
    for path in reference_paths:
        value = load_json(path, failures)
        if value is None:
            continue
        replay = value.get("deterministic_replay")
        replay_args = replay_script_args(replay.get("replay_command_argv") if isinstance(replay, dict) else None)
        if replay_args is None:
            failures.append(f"{path}: deterministic_replay.replay_command_argv cannot be replayed")
            emit("oracle-live-replay-argv", False, file=str(path.relative_to(ROOT)))
            continue
        expected_sha = value.get("decoded_text_sha256")
        if not isinstance(expected_sha, str):
            failures.append(f"{path}: decoded_text_sha256 must be a string for live replay")
            emit("oracle-live-replay-expected-sha", False, file=str(path.relative_to(ROOT)), expected=expected_sha)
            continue
        grouped.setdefault(tuple(replay_args), {})[path.name] = expected_sha
        emit("oracle-live-replay-argv", True, file=str(path.relative_to(ROOT)), argc=len(replay_args))
    return grouped


def validate_live_replay(reference_paths: list[Path], failures: list[str]) -> None:
    if not reference_paths:
        emit("oracle-live-replay-summary", True, references=0, skipped=True, reason="no native references")
        return

    grouped = load_replay_expectations(reference_paths, failures)
    if not grouped:
        return

    for replay_args_tuple, expected_by_file in sorted(grouped.items()):
        replay_args = list(replay_args_tuple)
        skip_reason = replay_prereq_skip(replay_args)
        if skip_reason is not None:
            emit(
                "oracle-live-replay-summary",
                True,
                references=len(expected_by_file),
                skipped=True,
                reason=skip_reason,
            )
            continue

        with tempfile.TemporaryDirectory(prefix="focr-oracle-replay-") as tmp:
            replay_args = replace_option(replay_args, "--out", tmp)
            cmd = [sys.executable, str(ROOT / "scripts" / "gen_reference_fixtures.py"), *replay_args]
            emit("oracle-live-replay-command", True, argc=len(cmd), out=tmp)
            proc = subprocess.run(  # nosec B603 - argv is sanitized to an anchored script path.
                cmd,
                cwd=ROOT,
                env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            ok = proc.returncode == 0
            emit(
                "oracle-live-replay-exit",
                ok,
                returncode=proc.returncode,
                stdout_tail=proc.stdout[-1000:],
                stderr_tail=proc.stderr[-1000:],
            )
            if not ok:
                failures.append(f"live oracle replay failed with exit {proc.returncode}")
                continue

            for file_name, expected_sha in sorted(expected_by_file.items()):
                replayed = Path(tmp) / file_name
                if not replayed.is_file():
                    fail(
                        failures,
                        f"live oracle replay did not regenerate {file_name}",
                        "oracle-live-replay-file",
                        file=file_name,
                    )
                    continue
                value = load_json(replayed, failures)
                actual_sha = value.get("decoded_text_sha256") if value is not None else None
                ok_sha = actual_sha == expected_sha
                emit(
                    "oracle-live-replay-sha",
                    ok_sha,
                    file=file_name,
                    expected=expected_sha,
                    actual=actual_sha,
                )
                if not ok_sha:
                    failures.append(f"{file_name}: live replay decoded_text_sha256 mismatch")


def self_test_reference_record() -> dict[str, Any]:
    decoded = "oracle provenance self-test"
    decoded_sha = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
    command_argv = [
        sys.executable,
        "scripts/gen_reference_fixtures.py",
        "--model-dir",
        "/nonexistent/model",
        "--corpus",
        "/nonexistent/corpus",
        "--seed",
        "7",
    ]
    return {
        "schema_version": 1,
        "decoded_text": decoded,
        "decoded_text_sha256": decoded_sha,
        "deterministic_replay": {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "rng_seed": 7,
            "requires_cuda": True,
            "expected_prefix_kind": "full_decoded_text",
            "expected_prefix_chars": len(decoded),
            "expected_prefix_sha256": decoded_sha,
            "expected_decoded_text_sha256": decoded_sha,
            "replay_command_argv": command_argv,
        },
        "provenance": {
            "pinned_torch": PIN_TORCH,
            "pinned_transformers": PIN_TRANSFORMERS,
            "torch_version": f"{PIN_TORCH}+cu128",
            "transformers_version": PIN_TRANSFORMERS,
            "hf_commit": HF_COMMIT,
            "github_commit": GITHUB_COMMIT,
            "oracle_is_correctness_golden": True,
            "model_weights_sha256": "a" * 64,
            "model_weights_bytes": 1,
            "command_argv": command_argv,
            "exact_command": "python3 scripts/gen_reference_fixtures.py --seed 7",
            "determinism": {
                "seed": 7,
                "torch_manual_seed": True,
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            },
            "generation_config": {
                "temperature": 0.0,
                "do_sample": False,
            },
        },
    }


def clone_record(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record))


def reference_validation_failures(record: dict[str, Any], *, quiet: bool = False) -> list[str]:
    path = ROOT / "tests" / "fixtures" / "native" / "self_test_reference.json"
    failures: list[str] = []
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            validate_reference_payload(path, record, failures)
    else:
        validate_reference_payload(path, record, failures)
    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Unlimited-OCR-2 oracle (U2-A6)
# ─────────────────────────────────────────────────────────────────────────────
def uocr2_source_pins(text: str) -> dict[str, str]:
    """`{file: sha256}` from the Unlimited-OCR-2 tables of SOURCE_HASHES.md."""
    start = text.find("# Unlimited-OCR-2")
    if start < 0:
        return {}
    end = text.find("**Upstream license text**", start)
    section = text[start : end if end >= 0 else len(text)]
    return {f: h for h, f in re.findall(r"^\| `([0-9a-f]{64})` \| `([^`]+)` \|", section, re.M)}


def uocr2_ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def uocr2_parse_ids(fixture: str, text: str) -> list[int] | str:
    """The ids of a committed `<fixture>.ids.txt`, or why it is malformed."""
    lines = text.splitlines()
    head = UOCR2_IDS_HEADER.match(lines[0]) if lines else None
    if head is None or head.group(1) != fixture:
        return "bad header"
    try:
        ids = [int(t) for line in lines[1:] for t in line.split()]
    except ValueError:
        return "non-integer id"
    if len(ids) != int(head.group(2)):
        return f"header says {head.group(2)} ids, body has {len(ids)}"
    return ids


def uocr2_record_failures(
    doc: dict[str, Any],
    files: dict[str, bytes],
    source_pins: dict[str, str],
    corpus: Path | None,
) -> list[str]:
    """Every problem with the committed v2 oracle twin. `files` maps the committed
    file names to their bytes; `corpus`, when present, holds the input pages."""
    problems: list[str] = []
    prov = doc.get("provenance")
    if not isinstance(prov, dict):
        return ["ORACLE.json: missing provenance"]
    pinned = {"python": UOCR2_PIN_PYTHON, "torch": UOCR2_PIN_TORCH, "transformers": UOCR2_PIN_TRANSFORMERS, "tokenizers": UOCR2_PIN_TOKENIZERS, "huggingface_hub": UOCR2_PIN_HF_HUB}
    if prov.get("pinned") != pinned:
        problems.append(f"provenance.pinned {prov.get('pinned')} != {pinned}")
    stack = prov.get("stack") if isinstance(prov.get("stack"), dict) else {}
    for key, want in pinned.items():
        if stack.get(key) != want:
            problems.append(f"provenance.stack.{key} {stack.get(key)!r} != {want!r}")
    if prov.get("hf_commit") != UOCR2_HF_COMMIT:
        problems.append(f"provenance.hf_commit {prov.get('hf_commit')!r} != {UOCR2_HF_COMMIT}")
    for key, want in (("device", "cpu"), ("dtype", "f32"), ("compute_dtype", "float32"), ("shims", ["cpu-patch", "f32"]), ("smoke", False)):
        if prov.get(key) != want:
            problems.append(f"provenance.{key} {prov.get(key)!r} != {want!r}")
    det = prov.get("determinism") if isinstance(prov.get("determinism"), dict) else {}
    if not (is_json_int(det.get("seed")) and det.get("torch_manual_seed") is True and det.get("torch_deterministic_algorithms") is True):
        problems.append(f"provenance.determinism not seeded and deterministic: {det}")
    if not (isinstance(prov.get("exact_command"), str) and "gen_reference_fixtures_uocr2.py" in prov["exact_command"]):
        problems.append("provenance.exact_command must name gen_reference_fixtures_uocr2.py")
    if not is_non_empty_string_list(prov.get("command_argv")):
        problems.append("provenance.command_argv must be a non-empty list of strings")
    if not (isinstance(prov.get("script_sha256"), str) and HEX64.match(prov["script_sha256"])):
        problems.append("provenance.script_sha256 must be a sha256")
    model_files = prov.get("model_files_sha256")
    if not isinstance(model_files, dict) or not model_files:
        problems.append("provenance.model_files_sha256 missing")
    else:
        for name, got in sorted(model_files.items()):
            if source_pins.get(name) != got:
                problems.append(f"model file {name}: {got} is not the SOURCE_HASHES.md pin {source_pins.get(name)}")
        if not {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "modeling_unlimitedocr.py"} <= set(model_files):
            problems.append("provenance.model_files_sha256 must cover both shards and the remote code")

    fixtures = doc.get("fixtures")
    if not isinstance(fixtures, dict) or not fixtures:
        return problems + ["ORACLE.json: no fixtures"]
    for fixture, entry in sorted(fixtures.items()):
        where = f"{fixture}:"
        pages = entry.get("pages")
        if not isinstance(pages, list) or not pages:
            problems.append(f"{where} no pages")
            continue
        for page in pages:
            if not (isinstance(page, dict) and isinstance(page.get("sha256"), str) and HEX64.match(page["sha256"])):
                problems.append(f"{where} page without a sha256: {page}")
            elif corpus is not None and (corpus / page["name"]).is_file() and sha256_file(corpus / page["name"]) != page["sha256"]:
                problems.append(f"{where} input {page['name']} no longer matches its recorded sha256")
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        window = UOCR2_NGRAM_WINDOW_SINGLE if len(pages) == 1 else UOCR2_NGRAM_WINDOW_MULTI
        want_params = {
            "no_repeat_ngram_size": UOCR2_NGRAM_SIZE,
            "ngram_window": window,
            "max_page_tokens": UOCR2_MAX_PAGE_TOKENS,
            "page_count_guard": True,
            "temperature": 0.0,
            "crop_mode": entry.get("mode") == "gundam",
            "max_new_tokens": None,
        }
        for key, want in want_params.items():
            if params.get(key, "absent") != want:
                problems.append(f"{where} params.{key} {params.get(key, 'absent')!r} != {want!r}")
        counters = entry.get("counters") if isinstance(entry.get("counters"), dict) else {}
        hits = counters.get("disc_u2_2_all_neg_inf_steps")
        if not is_json_int(hits) or hits != 0:
            problems.append(f"{where} DISC-U2-2 all-(-inf) steps {hits!r} (must be recorded, and 0)")
        routing = entry.get("routing") if isinstance(entry.get("routing"), dict) else {}
        for key in ("route_replica_mismatches", "kbar_replica_mismatches"):
            if routing.get(key) != 0:
                problems.append(f"{where} routing.{key} {routing.get(key)!r} != 0")
        if routing.get("stage0") != ["head_cls"]:
            problems.append(f"{where} routing.stage0 {routing.get('stage0')!r} != ['head_cls']")
        if entry.get("finish_reason") == "stop" and not (
            is_json_int(entry.get("num_pages_out")) and is_json_int(entry.get("expected_pages")) and entry["num_pages_out"] >= entry["expected_pages"] == len(pages)
        ):
            problems.append(f"{where} stopped with {entry.get('num_pages_out')} of {entry.get('expected_pages')} pages")
        ids_name, md_name = f"{fixture}.ids.txt", f"{fixture}.md"
        if ids_name not in files:
            problems.append(f"{where} {ids_name} missing")
        else:
            ids = uocr2_parse_ids(fixture, files[ids_name].decode("utf-8", errors="replace"))
            if isinstance(ids, str):
                problems.append(f"{where} {ids_name}: {ids}")
            elif len(ids) != entry.get("token_count") or uocr2_ids_sha256(ids) != entry.get("token_ids_sha256"):
                problems.append(f"{where} {ids_name} does not match token_count/token_ids_sha256")
        if md_name not in files:
            problems.append(f"{where} {md_name} missing")
        elif hashlib.sha256(files[md_name]).hexdigest() != entry.get("decoded_text_sha256"):
            problems.append(f"{where} {md_name} does not match decoded_text_sha256")
    return problems


def uocr2_floor_failures(floor: dict[str, Any], fixtures: set[str]) -> list[str]:
    groups = floor.get("groups")
    if not isinstance(groups, dict) or "f32" not in groups:
        return ["FLOOR.json: no f32 group"]
    problems = []
    for name, group in sorted(groups.items()):
        runs = group.get("run_dirs")
        if not (isinstance(runs, list) and len(runs) >= 2):
            problems.append(f"FLOOR.json group {name}: fewer than two runs")
        if name == "f32" and set(group.get("fixtures", {})) != fixtures:
            problems.append(f"FLOOR.json group f32 covers {sorted(group.get('fixtures', {}))}, ORACLE.json has {sorted(fixtures)}")
        if not isinstance(group.get("summary"), dict):
            problems.append(f"FLOOR.json group {name}: no summary")
    return problems


def uocr2_dump_failures(dumps: Path, doc: dict[str, Any]) -> list[str]:
    """The off-tree dumps of the committed run agree with ORACLE.json, file by file."""
    problems = []
    for fixture, entry in sorted(doc.get("fixtures", {}).items()):
        run = entry.get("run", {})
        ref_path = dumps / str(run.get("dtype")) / str(run.get("tag")) / f"{fixture}_reference.json"
        if not ref_path.is_file():
            problems.append(f"{fixture}: dump {ref_path} missing")
            continue
        ref = json.loads(ref_path.read_text())
        for key in ("token_ids_sha256", "decoded_text_sha256"):
            if ref.get(key) != entry.get(key):
                problems.append(f"{fixture}: dump {key} differs from ORACLE.json")
        for stage, act in sorted(ref.get("activations", {}).items()):
            path = ref_path.parent / "activations" / fixture / act["file"]
            if not path.is_file() or sha256_file(path) != act.get("file_sha256"):
                problems.append(f"{fixture}: activation {stage} missing or changed")
    return problems


def uocr2_generator_pins() -> dict[str, Any] | None:
    """The generator's own pin constants, read without importing torch."""
    import importlib.util  # noqa: WPS433

    spec = importlib.util.spec_from_file_location("gen_reference_fixtures_uocr2", UOCR2_GENERATOR)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return {
        "python": module.PIN_PYTHON,
        "torch": module.PIN_TORCH,
        "transformers": module.PIN_TRANSFORMERS,
        "tokenizers": module.PIN_TOKENIZERS,
        "huggingface_hub": module.PIN_HF_HUB,
        "hf_commit": module.HF_COMMIT,
    }


def validate_uocr2(failures: list[str], dumps: Path | None) -> None:
    pins = {"python": UOCR2_PIN_PYTHON, "torch": UOCR2_PIN_TORCH, "transformers": UOCR2_PIN_TRANSFORMERS, "tokenizers": UOCR2_PIN_TOKENIZERS, "huggingface_hub": UOCR2_PIN_HF_HUB, "hf_commit": UOCR2_HF_COMMIT}
    generator = uocr2_generator_pins()
    same = generator == pins
    emit("uocr2-oracle-pins-match-generator", same, checker=pins, generator=generator)
    if not same:
        failures.append("the v2 pins in check_oracle_provenance.py and gen_reference_fixtures_uocr2.py differ")
    oracle_json = UOCR2_ORACLE / "ORACLE.json"
    if not oracle_json.is_file():
        emit("uocr2-oracle-provenance", True, skipped=True, reason="tests/fixtures/uocr2_oracle/ORACLE.json absent")
        return
    doc = json.loads(oracle_json.read_text())
    files = {p.name: p.read_bytes() for p in UOCR2_ORACLE.iterdir() if p.is_file()}
    corpus_env = os.environ.get("FOCR_CORPUS_DIR")
    corpus = Path(corpus_env) if corpus_env else Path.home() / ".cache/franken_ocr/corpus/pages"
    corpus = corpus if corpus.is_dir() else None
    problems = uocr2_record_failures(doc, files, uocr2_source_pins(UOCR2_SOURCE_HASHES.read_text()), corpus)
    floor_path = UOCR2_ORACLE / "FLOOR.json"
    if floor_path.is_file():
        problems += uocr2_floor_failures(json.loads(floor_path.read_text()), set(doc.get("fixtures", {})))
    else:
        problems.append("tests/fixtures/uocr2_oracle/FLOOR.json missing (the nondeterminism floor)")
    if dumps is not None:
        problems += uocr2_dump_failures(dumps, doc)
    for problem in problems:
        fail(failures, f"uocr2_oracle: {problem}", "uocr2-oracle-provenance-problem")
    emit(
        "uocr2-oracle-provenance",
        not problems,
        fixtures=len(doc.get("fixtures", {})),
        inputs_verified=corpus is not None,
        dumps_verified=dumps is not None,
    )


def uocr2_self_test_doc() -> tuple[dict[str, Any], dict[str, bytes], dict[str, str]]:
    ids = [128827, 128828, 2067]
    text = "<PAGE><BLOCK>x"
    pins = {"model-00001-of-00002.safetensors": "1" * 64, "model-00002-of-00002.safetensors": "2" * 64, "modeling_unlimitedocr.py": "3" * 64}
    stack = {"python": UOCR2_PIN_PYTHON, "torch": UOCR2_PIN_TORCH, "transformers": UOCR2_PIN_TRANSFORMERS, "tokenizers": UOCR2_PIN_TOKENIZERS, "huggingface_hub": UOCR2_PIN_HF_HUB}
    doc = {
        "schema_version": 1,
        "provenance": {
            "pinned": dict(stack),
            "stack": dict(stack, python_full="3.10.19"),
            "hf_commit": UOCR2_HF_COMMIT,
            "device": "cpu",
            "dtype": "f32",
            "compute_dtype": "float32",
            "shims": ["cpu-patch", "f32"],
            "smoke": False,
            "determinism": {"seed": 1337, "torch_manual_seed": True, "torch_deterministic_algorithms": True},
            "exact_command": "python -I scripts/gen_reference_fixtures_uocr2.py --sets S1",
            "command_argv": ["python", "scripts/gen_reference_fixtures_uocr2.py"],
            "script_sha256": "4" * 64,
            "model_files_sha256": dict(pins),
        },
        "fixtures": {
            "S1-page_0009": {
                "mode": "base",
                "pages": [{"name": "page_0009.png", "sha256": "5" * 64}],
                "params": {"no_repeat_ngram_size": 35, "ngram_window": 128, "max_page_tokens": 20000, "page_count_guard": True, "temperature": 0.0, "crop_mode": False, "max_new_tokens": None},
                "token_count": len(ids),
                "token_ids_sha256": uocr2_ids_sha256(ids),
                "decoded_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "finish_reason": "stop",
                "num_pages_out": 1,
                "expected_pages": 1,
                "counters": {"disc_u2_2_all_neg_inf_steps": 0},
                "routing": {"route_replica_mismatches": 0, "kbar_replica_mismatches": 0, "stage0": ["head_cls"]},
            }
        },
    }
    files = {
        "S1-page_0009.ids.txt": ("# m generated token ids: S1-page_0009 (3 ids)\n" + " ".join(map(str, ids)) + "\n").encode(),
        "S1-page_0009.md": text.encode(),
    }
    return doc, files, pins


def uocr2_self_test() -> bool:
    import copy  # noqa: WPS433

    doc, files, pins = uocr2_self_test_doc()
    good = uocr2_record_failures(doc, files, pins, None)
    emit("uocr2-oracle-self-test-good", not good, failures=good)
    ok = not good
    entry = ("fixtures", "S1-page_0009")
    bad_cases = [
        ("v1-torch-pin", lambda d, f: d["provenance"]["stack"].update(torch="2.10.0")),
        ("wrong-hf-commit", lambda d, f: d["provenance"].update(hf_commit="3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5")),
        ("unpinned-shard", lambda d, f: d["provenance"]["model_files_sha256"].update({"model-00001-of-00002.safetensors": "9" * 64})),
        ("bf16-run-committed", lambda d, f: d["provenance"].update(dtype="bf16")),
        ("smoke-run-committed", lambda d, f: d["provenance"].update(smoke=True)),
        ("nondeterministic", lambda d, f: d["provenance"]["determinism"].update(torch_deterministic_algorithms=False)),
        ("disc-u2-2-hit", lambda d, f: d[entry[0]][entry[1]]["counters"].update(disc_u2_2_all_neg_inf_steps=1)),
        ("disc-u2-2-unrecorded", lambda d, f: d[entry[0]][entry[1]].update(counters={})),
        ("route-replica-drift", lambda d, f: d[entry[0]][entry[1]]["routing"].update(route_replica_mismatches=2)),
        ("multi-page-window-on-single", lambda d, f: d[entry[0]][entry[1]]["params"].update(ngram_window=1024)),
        ("capped-run", lambda d, f: d[entry[0]][entry[1]]["params"].update(max_new_tokens=80)),
        ("stopped-short-of-its-pages", lambda d, f: d[entry[0]][entry[1]].update(num_pages_out=0)),
        ("ids-file-edited", lambda d, f: f.update({"S1-page_0009.ids.txt": b"# m generated token ids: S1-page_0009 (3 ids)\n128827 128828 2068\n"})),
        ("ids-file-miscounted", lambda d, f: f.update({"S1-page_0009.ids.txt": b"# m generated token ids: S1-page_0009 (4 ids)\n128827 128828 2067\n"})),
        ("markdown-edited", lambda d, f: f.update({"S1-page_0009.md": b"<PAGE>"})),
        ("markdown-missing", lambda d, f: f.pop("S1-page_0009.md")),
        ("page-hash-missing", lambda d, f: d[entry[0]][entry[1]]["pages"][0].pop("sha256")),
    ]
    for case, mutate in bad_cases:
        d, f = copy.deepcopy(doc), dict(files)
        mutate(d, f)
        rejected = bool(uocr2_record_failures(d, f, pins, None))
        emit("uocr2-oracle-self-test-bad", rejected, case=case)
        ok = ok and rejected
    floor_ok = not uocr2_floor_failures({"groups": {"f32": {"run_dirs": ["a", "b"], "summary": {}, "fixtures": {"S1-page_0009": {}}}}}, {"S1-page_0009"})
    floor_bad = bool(uocr2_floor_failures({"groups": {"f32": {"run_dirs": ["a"], "summary": {}, "fixtures": {}}}}, {"S1-page_0009"}))
    emit("uocr2-oracle-self-test-floor", floor_ok and floor_bad)
    text = "# Unlimited-OCR (v1)\n| `" + "a" * 64 + "` | `config.json` | 1 | v1 |\n# Unlimited-OCR-2 (v2)\n| `" + "b" * 64 + "` | `config.json` | 1 | v2 |\n"
    pins_ok = uocr2_source_pins(text) == {"config.json": "b" * 64}
    emit("uocr2-oracle-self-test-source-pins", pins_ok)
    return ok and floor_ok and floor_bad and pins_ok


def self_test() -> int:
    ok = uocr2_self_test()
    good_record = self_test_reference_record()
    good_failures = reference_validation_failures(good_record, quiet=True)
    emit("oracle-provenance-self-test-good", not good_failures, failures=good_failures)
    ok = ok and not good_failures

    helper_args = ["--model-dir", "/m", "--corpus=/c", "--out", "/old"]
    helper_ok = (
        replay_script_args(["python3", "scripts/gen_reference_fixtures.py", *helper_args]) == helper_args
        and option_value(helper_args, "--model-dir") == "/m"
        and option_value(helper_args, "--corpus") == "/c"
        and replace_option(helper_args, "--out", "replay-out")
        == ["--model-dir", "/m", "--corpus=/c", "--out", "replay-out"]
        and replace_option(["--model-dir=/m"], "--out", "replay-out")
        == ["--model-dir=/m", "--out", "replay-out"]
    )
    emit("oracle-live-replay-helper-self-test", helper_ok)
    ok = ok and helper_ok

    def set_bool_prefix_for_one_char(record: dict[str, Any]) -> None:
        decoded = "x"
        decoded_sha = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
        record.update({"decoded_text": decoded, "decoded_text_sha256": decoded_sha})
        record["deterministic_replay"].update(
            {
                "expected_prefix_chars": True,
                "expected_prefix_sha256": decoded_sha,
                "expected_decoded_text_sha256": decoded_sha,
            }
        )

    bad_cases = [
        (
            "bool-reference-schema",
            lambda record: record.update({"schema_version": True}),
        ),
        (
            "bool-model-bytes",
            lambda record: record["provenance"].update({"model_weights_bytes": True}),
        ),
        (
            "bool-provenance-seed",
            lambda record: record["provenance"]["determinism"].update({"seed": True}),
        ),
        (
            "empty-provenance-command-argv",
            lambda record: record["provenance"].update({"command_argv": []}),
        ),
        (
            "bool-replay-schema",
            lambda record: record["deterministic_replay"].update({"schema_version": True}),
        ),
        (
            "bool-replay-seed",
            lambda record: record["deterministic_replay"].update({"rng_seed": True}),
        ),
        (
            "bool-prefix-chars",
            set_bool_prefix_for_one_char,
        ),
        (
            "negative-provenance-seed",
            lambda record: record["provenance"]["determinism"].update({"seed": -1}),
        ),
        (
            "nondeterministic-generation",
            lambda record: record["provenance"]["generation_config"].update({"do_sample": True}),
        ),
        (
            "replay-seed-mismatch",
            lambda record: record["deterministic_replay"].update({"rng_seed": 8}),
        ),
        (
            "replay-sha-mismatch",
            lambda record: record["deterministic_replay"].update(
                {"expected_prefix_sha256": "b" * 64}
            ),
        ),
        (
            "replay-without-cuda",
            lambda record: record["deterministic_replay"].update({"requires_cuda": False}),
        ),
        (
            "replay-command-missing-generator",
            lambda record: record["deterministic_replay"].update(
                {"replay_command_argv": ["python3", "other.py"]}
            ),
        ),
        (
            "empty-replay-command-argv",
            lambda record: record["deterministic_replay"].update({"replay_command_argv": []}),
        ),
        (
            "non-string-replay-command-argv",
            lambda record: record["deterministic_replay"].update(
                {"replay_command_argv": [sys.executable, {"script": "gen_reference_fixtures.py"}]}
            ),
        ),
        (
            "invalid-activations-container",
            lambda record: record.update({"activations": []}),
        ),
    ]
    for case, mutate in bad_cases:
        record = clone_record(good_record)
        mutate(record)
        failures = reference_validation_failures(record, quiet=True)
        rejected = bool(failures)
        emit("oracle-provenance-self-test-bad", rejected, case=case, failures=failures)
        ok = ok and rejected

    emit("oracle-provenance-self-test-summary", ok)
    return 0 if ok else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run in-memory validator checks")
    parser.add_argument(
        "--live-replay",
        action="store_true",
        help="model-gated: rerun recorded generator commands into a temp dir and compare decoded hashes",
    )
    parser.add_argument(
        "--uocr2-dumps",
        type=Path,
        default=None,
        help="also verify the off-tree Unlimited-OCR-2 oracle dumps (<FOCR_FIXTURES_DIR>/uocr2_oracle) file by file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    failures: list[str] = []
    validate_uocr2(failures, args.uocr2_dumps)

    if not NATIVE.exists():
        emit("oracle-native-fixtures-root", True, file=str(NATIVE.relative_to(ROOT)), skipped=True)
        if args.live_replay:
            emit(
                "oracle-live-replay-summary",
                True,
                references=0,
                skipped=True,
                reason="native fixtures root absent",
            )
        if failures:
            for failure in failures:
                print(f"ERROR: {failure}", file=sys.stderr)
            return 1
        emit("oracle-provenance-summary", True, manifests=0, references=0, skipped=True)
        return 0

    manifests = sorted(NATIVE.glob("PROVENANCE*.json"))
    references = sorted(NATIVE.glob("*_reference.json"))
    npys = sorted(NATIVE.rglob("*.npy"))
    has_artifacts = bool(references or npys)
    emit("oracle-native-fixtures-root", True, file=str(NATIVE.relative_to(ROOT)), manifests=len(manifests), references=len(references), npy=len(npys))
    if has_artifacts and not manifests:
        fail(failures, "native fixtures exist without PROVENANCE*.json", "oracle-manifest-present", file=str(NATIVE.relative_to(ROOT)))

    for manifest in manifests:
        validate_manifest(manifest, failures)
    covered_npys: set[Path] = set()
    for reference in references:
        covered_npys.update(validate_reference_json(reference, failures))
    for npy in npys:
        covered = npy in covered_npys
        emit("oracle-npy-covered-by-manifest", covered, file=str(npy.relative_to(ROOT)))
        if not covered:
            failures.append(f"{npy}: .npy is not referenced by a fixture manifest")

    if args.live_replay:
        validate_live_replay(references, failures)

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    emit("oracle-provenance-summary", True, manifests=len(manifests), references=len(references), npy=len(npys), skipped=not has_artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
