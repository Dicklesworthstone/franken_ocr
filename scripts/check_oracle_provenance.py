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


def self_test() -> int:
    ok = True
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    failures: list[str] = []

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
