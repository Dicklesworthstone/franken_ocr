#!/usr/bin/env python3
"""Torch-side reference timer for the head-to-head gauntlet (bd-re8.17).

Times the pinned HF CPU reference per stage and emits the SAME
`focr-gauntlet-stage/v1` JSON records as `scripts/gauntlet_timing.py`, plus (as
the last stdout line, one per measured stage) the timing envelope
`benches/gauntlet_harness.rs` / docs/gauntlet/BENCH_HARNESS.md §5 expects from
`FOCR_REFERENCE_CMD`:

    {"stage":"decode_per_token","result":"pass","p50_ms":14.5,
     "precision":"bf16","threads":8,"reference_backend":"hf", ...}

FAIRNESS IS MANDATORY AND FAIL-CLOSED (docs/PERF_LEDGER.md §9.3; the hardened
frankentorch lesson — NEVER benchmark torch at @64):

  * a positive thread budget must be pinned (FOCR_THREADS or --threads),
    budget > 32 is refused outright;
  * the BLAS/OMP pool env vars must already equal the budget BEFORE torch is
    imported (OMP reads them at import) — a missing or mismatched pin REFUSES
    to emit any timing record (`result:"error"`, non-zero exit);
  * after `torch.set_num_threads(N)`, `torch.get_num_threads()` must equal the
    budget or the run refuses;
  * the exact entry selects an immutable truth-pack runtime contract: Unlimited
    and SmolVLM2 use torch==2.10.0 / transformers==4.57.1; GOT, OneChart, and
    TrOMR use torch==2.12.1 / transformers==4.45.2. Unknown citable entries and
    drifted stacks refuse; legacy `--pin-*` flags can only assert that mapping.
  * citable Unlimited captures hash and validate the exact 12-file truth-pack
    runtime model before and after inference, and use a fresh evidence-local
    HF_MODULES_CACHE; any provenance or source drift refuses the timing record.

The model-specific measurement is injected via `--entry module:function`
(loading/setup belongs in `--setup module:function`, run once outside the
clock). The REAL wired entry is `scripts/gauntlet_ref_unlimited.py` — the
truth-pack CPU-patched HF baseline (`scripts/baseline/run_baidu_reference.py`
flow) with per-stage instance-level forward timers. Entry protocols:

  * `None`                  — the harness's outer wall clock times the call
                              (single-stage legacy);
  * `{"tokens": n}`         — outer wall; `decode_per_token` divides by n;
  * `{"stages": {...}}`     — the entry measured each stage internally:
                              `{"ms": float, "tokens": int?}` per stage; ONE
                              entry call yields EVERY stage's sample, so
                              `--stage all` costs one inference per run.

Without an entry the run emits `result:"skip"` — it NEVER invents a number.

`--smoke` proves the plumbing end-to-end (single run, warmup 0) and stamps
every output SYNTHETIC/non-citable (`gauntlet_row.py` refuses it); envelopes
carry `result:"smoke"`. Timings from a smoke run are NOT evidence.

Usage:
  gauntlet_reference.py --stage all --page PAGE --model-dir DIR \
      --backend hf --precision bf16 --max-length 8192 --text-dir ABS_DIR \
      --entry gauntlet_ref_unlimited:run_stage \
      --setup gauntlet_ref_unlimited:setup [--runs 5] [--warmup 1] [--out FILE]
  FOCR_GAUNTLET_STAGE=prefill FOCR_THREADS=8 gauntlet_reference.py ...  # envelope mode
  gauntlet_reference.py --self-test
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import re
import stat
import statistics
import sys
import time

SCHEMA_STAGE = "focr-gauntlet-stage/v1"
SCHEMA_DOC = "focr-gauntlet-stages/v1"
SCHEMA_RAW_TIMING = "focr-gauntlet-raw-timing/v1"
SCHEMA_MODEL_MANIFEST = "focr-reference-model-manifest/v1"
SCHEMA_INFERENCE_BINDING = "focr-reference-inference-binding/v1"

STAGES = ("preprocess", "vision_encode", "prefill", "decode_per_token", "end_to_end")
ALL = "all"  # measure every stage the entry exposes, one entry call per run


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Pool pins that must equal the budget BEFORE `import torch` (OMP/MKL read the
# environment at import time; setting them afterwards silently does nothing).
# Mirrors the benches/gauntlet_harness.rs reference env list.
PRE_IMPORT_PINS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Entry-specific truth-pack runtime contracts. The entry, never an operator
# flag, selects the only stack that can produce citable evidence for that lane.
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

MAX_THREAD_BUDGET = 32  # NEVER @64 — oversubscription inflates fake wins
UNLIMITED_MAX_LENGTH = 8192
UNLIMITED_AMBIENT_ENV = ("FOCR_REF_MAX_LENGTH", "FOCR_REF_TEXT_DIR")
MAX_MEASURED_RUNS = 256
MAX_WARMUP_RUNS = 64
TEXT_SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
MODEL_ROOT_DOMAIN = b"focr-reference-model-root/v1\0"
INFERENCE_BINDING_DOMAIN = b"focr-reference-inference-binding/v1\0"
MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024
MAX_INDEX_FILE_BYTES = 2 * 1024 * 1024
READ_CHUNK_BYTES = 8 * 1024 * 1024

UNLIMITED_MODEL_COMMIT = "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"
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
        6672547120,
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
UNLIMITED_MODEL_SPEC = {
    "model_id": "baidu/Unlimited-OCR",
    "model_commit": UNLIMITED_MODEL_COMMIT,
    "files": UNLIMITED_MODEL_FILES,
    "file_count": 12,
    "index_path": "model.safetensors.index.json",
    "index_total_size": 6672212480,
    "index_weight_count": 2710,
    "shard_path": "model-00001-of-000001.safetensors",
}


class FairnessError(RuntimeError):
    """A mandatory fairness control is not satisfied; no record may be emitted."""


def _stable_stat_fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_directory_nofollow(path: str) -> int:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise FairnessError(f"cannot inspect directory {path!r}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise FairnessError(f"directory is a symlink or non-directory: {path!r}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FairnessError(f"cannot open directory {path!r}: {error}") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise FairnessError(f"directory changed while opening: {path!r}")
    return descriptor


def _read_registered_file(
    directory_fd: int,
    name: str,
    expected_size: int,
    *,
    retain_contents: bool,
) -> tuple[dict, bytes | None]:
    try:
        before_link = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise FairnessError(f"registered model file is missing: {name}: {error}") from error
    if stat.S_ISLNK(before_link.st_mode) or not stat.S_ISREG(before_link.st_mode):
        raise FairnessError(f"registered model file is symlink/special: {name}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise FairnessError(f"cannot open registered model file {name}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before_link.st_dev, before_link.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise FairnessError(f"registered model file changed while opening: {name}")
        if before.st_size != expected_size:
            raise FairnessError(
                f"registered model file size mismatch for {name}: "
                f"got {before.st_size}, expected {expected_size}"
            )

        digest = hashlib.sha256()
        retained = bytearray() if retain_contents else None
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise FairnessError(f"registered model file truncated while reading: {name}")
            digest.update(chunk)
            if retained is not None:
                retained.extend(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise FairnessError(f"registered model file grew while reading: {name}")
        after = os.fstat(descriptor)
        if _stable_stat_fields(before) != _stable_stat_fields(after):
            raise FairnessError(f"registered model file drifted while hashing: {name}")
        evidence = {
            "path": name,
            "bytes": before.st_size,
            "sha256": digest.hexdigest(),
            "kind": "regular",
        }
        return evidence, bytes(retained) if retained is not None else None
    finally:
        os.close(descriptor)


def _validate_model_spec(spec: dict) -> tuple[dict[str, tuple[int, str]], str]:
    model_id = spec.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise FairnessError("reference model registration has an invalid model id")
    commit = spec.get("model_commit")
    if not isinstance(commit, str) or COMMIT_SHA_RE.fullmatch(commit) is None:
        raise FairnessError("reference model registration has an invalid commit hash")
    registered = spec.get("files")
    if not isinstance(registered, (tuple, list)) or not registered:
        raise FairnessError("reference model registration has no files")

    files: dict[str, tuple[int, str]] = {}
    names: list[str] = []
    for item in registered:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise FairnessError("reference model registration entry is malformed")
        name, size, expected_sha = item
        if (
            not isinstance(name, str)
            or not name
            or name != os.path.basename(name)
            or name in {".", ".."}
        ):
            raise FairnessError(f"unsafe registered model path: {name!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise FairnessError(f"invalid registered size for {name!r}")
        if (
            not isinstance(expected_sha, str)
            or TEXT_SHA256_RE.fullmatch(expected_sha) is None
        ):
            raise FairnessError(f"invalid registered lowercase SHA-256 for {name!r}")
        if name in files:
            raise FairnessError(f"duplicate registered model path: {name}")
        files[name] = (size, expected_sha)
        names.append(name)
    if names != sorted(names):
        raise FairnessError("reference model registration is not stably sorted")
    if spec.get("file_count") != len(files):
        raise FairnessError("reference model registration file count disagrees")

    index_path = spec.get("index_path")
    shard_path = spec.get("shard_path")
    if index_path not in files or shard_path not in files:
        raise FairnessError("reference model registration omits index or shard")
    if files[index_path][0] > MAX_INDEX_FILE_BYTES:
        raise FairnessError("reference model index exceeds the bounded read limit")
    return files, index_path


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise FairnessError(f"duplicate JSON key in model index: {key!r}")
        result[key] = value
    return result


def _validate_model_index(index_bytes: bytes, spec: dict) -> dict:
    try:
        parsed = json.loads(
            index_bytes.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except FairnessError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FairnessError(f"model index is not strict UTF-8 JSON: {error}") from error
    if not isinstance(parsed, dict) or set(parsed) != {"metadata", "weight_map"}:
        raise FairnessError("model index must contain exactly metadata and weight_map")
    metadata = parsed["metadata"]
    weight_map = parsed["weight_map"]
    if not isinstance(metadata, dict) or set(metadata) != {"total_size"}:
        raise FairnessError("model index metadata is malformed")
    total_size = metadata["total_size"]
    if (
        not isinstance(total_size, int)
        or isinstance(total_size, bool)
        or total_size != spec.get("index_total_size")
    ):
        raise FairnessError(
            f"model index total_size mismatch: got {total_size!r}, "
            f"expected {spec.get('index_total_size')!r}"
        )
    expected_count = spec.get("index_weight_count")
    if not isinstance(weight_map, dict) or len(weight_map) != expected_count:
        raise FairnessError(
            f"model index weight count mismatch: got "
            f"{len(weight_map) if isinstance(weight_map, dict) else 'non-object'}, "
            f"expected {expected_count!r}"
        )
    if any(not isinstance(key, str) or not key for key in weight_map):
        raise FairnessError("model index contains an invalid tensor name")
    shard_path = spec["shard_path"]
    if any(not isinstance(value, str) for value in weight_map.values()):
        raise FairnessError("model index contains a non-string shard path")
    shards = set(weight_map.values())
    if shards != {shard_path}:
        raise FairnessError(
            f"model index references unregistered shards: {sorted(map(str, shards))}"
        )
    return {
        "path": spec["index_path"],
        "total_size": total_size,
        "weight_count": len(weight_map),
        "shards": [shard_path],
    }


def _model_root_sha(files: list[dict]) -> str:
    digest = hashlib.sha256(MODEL_ROOT_DOMAIN)
    for item in sorted(files, key=lambda value: value["path"]):
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_reference_model_manifest(
    observations: dict[str, dict], index_bytes: bytes, spec: dict
) -> dict:
    registered, _ = _validate_model_spec(spec)
    if set(observations) != set(registered):
        missing = sorted(set(registered) - set(observations))
        extra = sorted(set(observations) - set(registered))
        raise FairnessError(
            f"reference model observation set mismatch: missing={missing}, extra={extra}"
        )

    files: list[dict] = []
    for name in sorted(registered):
        expected_size, expected_sha = registered[name]
        item = observations[name]
        if not isinstance(item, dict) or item.get("kind") != "regular":
            raise FairnessError(f"reference model entry is symlink/special: {name}")
        observed_sha = item.get("sha256")
        if (
            not isinstance(observed_sha, str)
            or TEXT_SHA256_RE.fullmatch(observed_sha) is None
        ):
            raise FairnessError(f"reference model entry has non-lowercase hash: {name}")
        if item.get("bytes") != expected_size or observed_sha != expected_sha:
            raise FairnessError(
                f"reference model entry does not match truth pack: {name}"
            )
        files.append(
            {"path": name, "bytes": expected_size, "sha256": observed_sha}
        )

    index_evidence = _validate_model_index(index_bytes, spec)
    index_observation = observations[spec["index_path"]]
    if hashlib.sha256(index_bytes).hexdigest() != index_observation["sha256"]:
        raise FairnessError("retained model index bytes disagree with hashed bytes")
    root_sha = _model_root_sha(files)
    return {
        "schema": SCHEMA_MODEL_MANIFEST,
        "model_id": spec["model_id"],
        "model_commit": spec["model_commit"],
        "synthetic": False,
        "citable": True,
        "file_count": len(files),
        "root_hash_domain": MODEL_ROOT_DOMAIN.decode("ascii"),
        "root_sha256": root_sha,
        "index": index_evidence,
        "files": files,
    }


def capture_reference_model_manifest(model_dir: str, spec: dict) -> dict:
    if not isinstance(model_dir, str) or not model_dir:
        raise FairnessError("citable Unlimited-OCR capture requires --model-dir")
    registered, index_path = _validate_model_spec(spec)
    directory_fd = _open_directory_nofollow(model_dir)
    try:
        observations: dict[str, dict] = {}
        index_bytes: bytes | None = None
        for name in sorted(registered):
            expected_size, _ = registered[name]
            observation, contents = _read_registered_file(
                directory_fd,
                name,
                expected_size,
                retain_contents=name == index_path,
            )
            observations[name] = observation
            if name == index_path:
                index_bytes = contents
        if index_bytes is None:
            raise FairnessError("registered model index was not retained")
        return build_reference_model_manifest(observations, index_bytes, spec)
    finally:
        os.close(directory_fd)


def assert_capture_unchanged(label: str, before: object, after: object) -> None:
    if before != after:
        raise FairnessError(f"{label} drifted during reference capture")


def create_fresh_hf_modules_cache(output_path: str | None) -> dict:
    if not isinstance(output_path, str) or not output_path:
        raise FairnessError(
            "citable Unlimited-OCR capture requires --out for an evidence-local "
            "HF_MODULES_CACHE"
        )
    output_abs = os.path.abspath(output_path)
    parent = os.path.dirname(output_abs)
    output_name = os.path.basename(output_abs)
    if not output_name or output_name in {".", ".."}:
        raise FairnessError("citable --out has an unsafe basename")
    cache_name = output_name + ".hf_modules_cache"
    parent_fd = _open_directory_nofollow(parent)
    try:
        try:
            os.mkdir(cache_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise FairnessError(
                f"evidence-local HF_MODULES_CACHE is not fresh: {cache_name}"
            ) from error
        except OSError as error:
            raise FairnessError(
                f"cannot create evidence-local HF_MODULES_CACHE: {error}"
            ) from error
        created = os.stat(cache_name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(created.st_mode):
            raise FairnessError("fresh HF_MODULES_CACHE is not a directory")
    finally:
        os.close(parent_fd)
    effective_path = os.path.join(parent, cache_name)
    verification_fd = _open_directory_nofollow(effective_path)
    try:
        resolved = os.fstat(verification_fd)
        if (created.st_dev, created.st_ino) != (resolved.st_dev, resolved.st_ino):
            raise FairnessError(
                "evidence-local HF_MODULES_CACHE path changed while creating it"
            )
    finally:
        os.close(verification_fd)
    os.environ["HF_MODULES_CACHE"] = effective_path
    return {
        "evidence_dir": parent,
        "path": cache_name,
        "effective_path": effective_path,
        "fresh": True,
    }


def _hash_regular_source(path: str) -> dict:
    try:
        before_link = os.lstat(path)
    except OSError as error:
        raise FairnessError(f"cannot inspect inference source {path!r}: {error}") from error
    if stat.S_ISLNK(before_link.st_mode) or not stat.S_ISREG(before_link.st_mode):
        raise FairnessError(f"inference source is symlink/special: {path!r}")
    if before_link.st_size <= 0 or before_link.st_size > MAX_SOURCE_FILE_BYTES:
        raise FairnessError(f"inference source exceeds bounded size: {path!r}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FairnessError(f"cannot open inference source {path!r}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before_link.st_dev, before_link.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise FairnessError(f"inference source changed while opening: {path!r}")
        remaining = before.st_size
        digest = hashlib.sha256()
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise FairnessError(f"inference source truncated while reading: {path!r}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise FairnessError(f"inference source grew while reading: {path!r}")
        after = os.fstat(descriptor)
        if _stable_stat_fields(before) != _stable_stat_fields(after):
            raise FairnessError(f"inference source drifted while hashing: {path!r}")
    finally:
        os.close(descriptor)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, repo_root)
    except ValueError:
        relative = absolute
    display_path = absolute if relative == ".." or relative.startswith("../") else relative
    return {
        "path": display_path,
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _callable_source(role: str, callable_spec: str, function: object) -> dict:
    source_path = inspect.getsourcefile(function)
    if not isinstance(source_path, str) or not source_path.endswith(".py"):
        raise FairnessError(f"{role} callable has no bindable Python source")
    return {"role": role, "callable": callable_spec} | _hash_regular_source(
        source_path
    )


def capture_inference_sources(
    entry_spec: str,
    entry: object,
    setup_spec: str | None,
    setup: object | None,
) -> list[dict]:
    bindings = [
        {"role": "harness", "callable": "gauntlet_reference:main"}
        | _hash_regular_source(os.path.abspath(__file__)),
        _callable_source("entry", entry_spec, entry),
    ]
    if setup_spec is not None and setup is not None:
        bindings.append(_callable_source("setup", setup_spec, setup))
    return bindings


def build_inference_binding(
    args: argparse.Namespace,
    stage: str,
    budget: int,
    page_sha256: str,
    contract: dict,
    model_manifest: dict,
    modules_cache: dict,
    sources: list[dict],
    torch_version: str,
    transformers_version: str,
) -> dict:
    binding = {
        "schema": SCHEMA_INFERENCE_BINDING,
        "model_root_sha256": model_manifest["root_sha256"],
        "model_commit": model_manifest["model_commit"],
        "entry": args.entry,
        "setup": args.setup,
        "stage": stage,
        "page": args.page,
        "page_sha256": page_sha256,
        "model_dir": args.model_dir,
        "max_length": args.max_length,
        "text_dir": args.text_dir,
        "backend": args.backend,
        "precision": args.precision,
        "threads": budget,
        "runs": args.runs,
        "warmup": args.warmup,
        "allocator": args.allocator,
        "argv": list(sys.argv),
        "env_pins": {
            key: os.environ.get(key, "")
            for key in PRE_IMPORT_PINS + ("FOCR_THREADS",)
        },
        "ambient_env": {
            key: os.environ.get(key, "<unset>") for key in UNLIMITED_AMBIENT_ENV
        },
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "reference_contract": dict(contract),
        "hf_modules_cache": dict(modules_cache),
        "sources": sources,
    }
    canonical = json.dumps(
        binding, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    binding["binding_hash_domain"] = INFERENCE_BINDING_DOMAIN.decode("ascii")
    binding["binding_sha256"] = hashlib.sha256(
        INFERENCE_BINDING_DOMAIN + canonical
    ).hexdigest()
    return binding


def resolve_budget(arg_threads: int | None, env: dict[str, str]) -> int:
    """The pinned thread budget, or a refusal. There is NO default here — an
    unpinned reference run must never silently measure at the machine width."""
    raw = str(arg_threads) if arg_threads is not None else env.get("FOCR_THREADS", "")
    if not raw.strip():
        raise FairnessError(
            "thread budget unpinned: set FOCR_THREADS or pass --threads"
        )
    try:
        budget = int(raw)
    except ValueError as err:
        raise FairnessError(f"thread budget {raw!r} is not an integer") from err
    if budget <= 0:
        raise FairnessError(f"thread budget must be positive, got {budget}")
    if budget > MAX_THREAD_BUDGET:
        raise FairnessError(
            f"thread budget {budget} > {MAX_THREAD_BUDGET} — oversubscribed torch "
            "runs are rejected (measure at @8/@32, NEVER @64)"
        )
    return budget


def verify_env_pins(budget: int, env: dict[str, str]) -> None:
    """Every pool pin must be present and equal to the budget pre-import."""
    for key in PRE_IMPORT_PINS:
        value = env.get(key, "")
        if value.strip() != str(budget):
            raise FairnessError(
                f"{key}={value!r} does not pin the budget {budget}; export "
                f"{key}={budget} before running (torch/BLAS read it at import)"
            )


def verify_unlimited_citable_args(
    entry: str | None,
    max_length: int | None,
    text_dir: str | None,
    env: dict[str, str],
) -> None:
    if entry != "gauntlet_ref_unlimited:run_stage":
        return
    inherited = [name for name in UNLIMITED_AMBIENT_ENV if name in env]
    if inherited:
        raise FairnessError(
            "ambient Unlimited reference overrides are forbidden: "
            + ", ".join(inherited)
        )
    if max_length != UNLIMITED_MAX_LENGTH:
        raise FairnessError(
            f"Unlimited reference requires --max-length {UNLIMITED_MAX_LENGTH}"
        )
    if not isinstance(text_dir, str) or not os.path.isabs(text_dir):
        raise FairnessError("Unlimited reference requires an absolute --text-dir")


def verify_torch_pinned(budget: int, torch_threads: int) -> None:
    """Post-`set_num_threads` proof that torch actually honors the budget."""
    if torch_threads != budget:
        raise FairnessError(
            f"torch.get_num_threads()={torch_threads} != pinned budget {budget}"
        )


def infer_reference_contract(entry: str | None) -> dict | None:
    spec = REFERENCE_CONTRACTS.get(entry or "")
    if spec is None:
        return None
    contract_id, torch_version, transformers_version = spec
    return {
        "schema": REFERENCE_CONTRACT_SCHEMA,
        "id": contract_id,
        "entry": entry,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
    }


def require_reference_contract(entry: str | None) -> dict:
    contract = infer_reference_contract(entry)
    if contract is None:
        raise FairnessError(
            f"unregistered reference entry {entry!r}; citable runs require one of "
            f"{sorted(REFERENCE_CONTRACTS)}"
        )
    return contract


def unregistered_smoke_contract(
    entry: str, torch_version: str, transformers_version: str
) -> dict:
    return {
        "schema": REFERENCE_CONTRACT_SCHEMA,
        "id": "unregistered-smoke",
        "entry": entry,
        "torch_version": torch_version.split("+", 1)[0],
        "transformers_version": transformers_version.split("+", 1)[0],
        "citable": False,
    }


def verify_stack_pins(
    contract: dict,
    torch_version: str,
    transformers_version: str,
) -> None:
    """Refuse a runtime stack that drifts from the entry's fixed contract."""

    def base(v: str) -> str:
        return v.split("+", 1)[0]

    if (
        base(torch_version) != contract["torch_version"]
        or base(transformers_version) != contract["transformers_version"]
    ):
        raise FairnessError(
            f"unpinned reference stack: torch=={torch_version}, "
            f"transformers=={transformers_version} ({contract['entry']} requires "
            f"torch=={contract['torch_version']}, "
            f"transformers=={contract['transformers_version']}); "
            "a ratio against a drifted stack is not comparable"
        )


def verify_requested_stack_pins(
    contract: dict, pin_torch: str | None, pin_transformers: str | None
) -> None:
    """Treat legacy CLI pins as assertions, never contract definitions."""
    for flag, requested, field in (
        ("--pin-torch", pin_torch, "torch_version"),
        ("--pin-transformers", pin_transformers, "transformers_version"),
    ):
        if requested is not None and requested != contract[field]:
            raise FairnessError(
                f"{flag}={requested!r} contradicts immutable contract "
                f"{contract['id']}: expected {contract[field]}"
            )


def stats(samples_ms: list[float]) -> dict:
    mean = statistics.fmean(samples_ms)
    cv = (
        statistics.stdev(samples_ms) / mean * 100.0
        if len(samples_ms) > 1 and mean > 0
        else None
    )
    return {
        "samples_ms": [round(v, 6) for v in samples_ms],
        "best_ms": round(min(samples_ms), 6),
        "p50_ms": round(statistics.median(samples_ms), 6),
        "mean_ms": round(mean, 6),
        "cv_pct": None if cv is None else round(cv, 3),
        "n": len(samples_ms),
    }


def build_record(
    stage: str,
    samples_ms: list[float],
    *,
    budget: int,
    torch_threads: int,
    precision: str,
    backend: str,
    allocator: str,
    warmup_discarded: int,
    reference_contract: dict,
    tokens: int | None = None,
    synthetic: bool = False,
) -> dict:
    if not samples_ms:
        raise FairnessError("no measured samples — a record cannot be built")
    record = {
        "schema": SCHEMA_STAGE,
        "source": "reference",
        "stage": stage,
        "ledger_stage": stage in STAGES,
        "unit": "ms",
        **stats(samples_ms),
        "warmup_discarded": warmup_discarded,
        "threads": budget,
        "thread_proof": {"budget": budget, "torch_num_threads": torch_threads},
        "precision": precision,
        "backend": backend,
        "allocator": allocator,
        "reference_contract": dict(reference_contract),
        "synthetic": synthetic,
    }
    if tokens is not None:
        record["tokens"] = tokens
    return record


def build_raw_timing(records: list[dict], *, source: str) -> dict:
    """Build the non-aggregated timing input consumed by row/cert verifiers."""
    if source not in {"focr", "reference"}:
        raise FairnessError(f"unsupported raw timing source: {source!r}")
    if not records or len(records) > MAX_MEASURED_RUNS:
        raise FairnessError("raw timing record count is outside the supported bound")
    expected_ids = [f"run_{index:03d}" for index in range(1, len(records) + 1)]
    if [record.get("run_id") for record in records] != expected_ids:
        raise FairnessError("raw timing run ids are not canonical and contiguous")
    return {
        "schema": SCHEMA_RAW_TIMING,
        "source": source,
        "unit": "ms",
        "measured_runs": len(records),
        "records": records,
    }


def require_fresh_output(path: str | None, *, smoke: bool) -> None:
    """Citable output is one capture session, never a stale cross-session merge."""
    if path and not smoke and os.path.lexists(path):
        raise FairnessError(
            f"citable --out must be a fresh path, but it already exists: {path}"
        )


def invocation_text_sha(result: object, index: int, *, citable: bool) -> str | None:
    value = result.get("text_sha256") if isinstance(result, dict) else None
    if value is not None and (
        not isinstance(value, str) or TEXT_SHA256_RE.fullmatch(value) is None
    ):
        raise FairnessError(
            f"invocation {index} emitted invalid text_sha256 {value!r}"
        )
    if citable and value is None:
        raise FairnessError(
            f"citable invocation {index} emitted no text_sha256 determinism proof"
        )
    return value


def deterministic_text_sha(
    hashes: list[str], total_invocations: int, *, citable: bool
) -> str | None:
    if citable and (
        len(hashes) != total_invocations or len(set(hashes)) != 1
    ):
        raise FairnessError(
            "citable reference text hashes are missing or drift across invocations"
        )
    return hashes[0] if hashes and len(set(hashes)) == 1 else None


def merge_raw_timing(existing: object, current: dict, replaced: list[str]) -> dict:
    """Merge separately measured stages without losing their raw observations."""
    if not isinstance(existing, dict):
        raise FairnessError("existing stage document lacks versioned raw timing")
    if (
        existing.get("schema") != SCHEMA_RAW_TIMING
        or existing.get("source") != "reference"
        or existing.get("unit") != "ms"
        or existing.get("measured_runs") != current.get("measured_runs")
    ):
        raise FairnessError("existing reference raw timing is incompatible")
    old_records = existing.get("records")
    new_records = current.get("records")
    if not isinstance(old_records, list) or not isinstance(new_records, list):
        raise FairnessError("reference raw timing records are malformed")
    if len(old_records) != len(new_records) or len(old_records) > MAX_MEASURED_RUNS:
        raise FairnessError("reference raw timing run counts disagree")

    merged_records: list[dict] = []
    for old, new in zip(old_records, new_records, strict=True):
        if (
            not isinstance(old, dict)
            or not isinstance(new, dict)
            or old.get("run_id") != new.get("run_id")
            or not isinstance(old.get("stages"), dict)
            or not isinstance(new.get("stages"), dict)
            or old.get("text_sha256") != new.get("text_sha256")
        ):
            raise FairnessError("reference raw timing run identities disagree")
        stages = {
            name: sample
            for name, sample in old["stages"].items()
            if name not in replaced
        }
        stages.update(new["stages"])
        merged = {"run_id": old["run_id"], "stages": stages}
        if old.get("text_sha256") is not None:
            merged["text_sha256"] = old["text_sha256"]
        merged_records.append(merged)
    return build_raw_timing(merged_records, source="reference")


def stage_sample(
    stage: str, result: object, outer_ms: float
) -> tuple[float, int | None]:
    """One measured `(ms, tokens|None)` for `stage` from one entry invocation.

    Protocols (module docstring): `None` → the harness's outer wall (legacy
    single-stage); `{"tokens": n}` → outer wall + token count;
    `{"stages": {...}}` → the entry measured the stage internally (`ms`
    mandatory and positive, `tokens` optional and positive)."""
    if result is None:
        return outer_ms, None
    if not isinstance(result, dict):
        raise FairnessError(
            f"entry returned {type(result).__name__}; want dict or None"
        )
    if "stages" in result:
        stages = result["stages"]
        if not isinstance(stages, dict):
            raise FairnessError("entry result 'stages' is not a dict")
        rec = stages.get(stage)
        if rec is None:
            raise FairnessError(
                f"entry measured no stage {stage!r} (measured: {sorted(stages)})"
            )
        ms = rec.get("ms")
        if not isinstance(ms, (int, float)) or isinstance(ms, bool) or ms <= 0:
            raise FairnessError(f"entry stage {stage!r} has no positive 'ms': {ms!r}")
        tokens = rec.get("tokens")
        if tokens is not None:
            tokens = int(tokens)
            if tokens <= 0:
                raise FairnessError(f"entry stage {stage!r} has non-positive tokens")
        return float(ms), tokens
    tokens = result.get("tokens")
    if tokens is None:
        return outer_ms, None
    tokens = int(tokens)
    if tokens <= 0:
        raise FairnessError("entry returned non-positive tokens")
    return outer_ms, tokens


def requested_from_result(stage_req: str, result: object) -> list[str]:
    """The stage list one entry call satisfies. `--stage all` requires the
    multi-stage protocol and keeps only ledger-vocabulary stages, in order."""
    if stage_req != ALL:
        return [stage_req]
    if not (isinstance(result, dict) and isinstance(result.get("stages"), dict)):
        raise FairnessError(
            "--stage all requires an entry returning {'stages': {...}} — the "
            "legacy outer-wall protocol times exactly one stage"
        )
    requested = [s for s in STAGES if s in result["stages"]]
    if not requested:
        raise FairnessError(
            f"entry measured no ledger-vocabulary stage (measured: {sorted(result['stages'])})"
        )
    return requested


def envelope_from_record(record: dict, result: str = "pass") -> dict:
    """The benches/gauntlet_harness.rs timing envelope (BENCH_HARNESS.md §5):
    requires `result`, a duration key, a thread proof, and a precision."""
    envelope = {
        "stage": record["stage"],
        "result": result,
        "p50_ms": record["p50_ms"],
        "min_ms": record["best_ms"],
        "cv_pct": record["cv_pct"],
        "precision": record["precision"],
        "reference_precision": record["precision"],
        "threads": record["threads"],
        "reference_threads": record["threads"],
        "torch_threads": record["thread_proof"]["torch_num_threads"],
        "reference_backend": record["backend"],
        "n": record["n"],
    }
    contract = record.get("reference_contract")
    if isinstance(contract, dict):
        envelope.update(
            {
                "reference_contract_id": contract.get("id"),
                "reference_torch_version": contract.get("torch_version"),
                "reference_transformers_version": contract.get(
                    "transformers_version"
                ),
            }
        )
    return envelope


def _emit(obj: dict) -> None:
    print(json.dumps(obj, sort_keys=False))


def _load_callable(spec: str):
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise FairnessError(f"--entry/--setup must be module:function, got {spec!r}")
    # Resolve entry modules from the scripts dir (where gauntlet_ref_unlimited
    # lives) regardless of the caller's cwd, then the cwd for ad-hoc entries.
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def run_stage(
    args: argparse.Namespace,
    stage_req: str,
    budget: int,
    contract: dict | None,
) -> int:
    """Measure the requested stage(s) with the injected entry, or refuse/skip
    honestly. One entry call per run; with the multi-stage protocol that one
    call samples EVERY stage (so `--stage all` costs one inference per run)."""
    require_fresh_output(args.out, smoke=args.smoke)
    if args.entry is None:
        _emit(
            {
                "stage": stage_req,
                "result": "skip",
                "reason": "stage_entry_not_wired",
                "detail": "pass --entry module:function (the real wiring is "
                "--entry gauntlet_ref_unlimited:run_stage --setup "
                "gauntlet_ref_unlimited:setup); the harness never invents a number",
            }
        )
        return 0

    unlimited_citable = bool(
        not args.smoke
        and isinstance(contract, dict)
        and contract.get("id") == "unlimited-ocr-hf-v1"
    )
    model_manifest_before: dict | None = None
    modules_cache: dict | None = None
    page_sha256_before: str | None = None
    if unlimited_citable:
        model_manifest_before = capture_reference_model_manifest(
            args.model_dir, UNLIMITED_MODEL_SPEC
        )
        page_sha256_before = sha256_file(args.page)
        modules_cache = create_fresh_hf_modules_cache(args.out)

    # Import torch only AFTER the env pins are proven (they are read at import).
    import torch  # noqa: PLC0415 — deliberate post-gate import
    import transformers  # noqa: PLC0415

    if contract is None:
        if not args.smoke:
            raise FairnessError("unregistered reference contract in citable run")
        contract = unregistered_smoke_contract(
            args.entry, torch.__version__, transformers.__version__
        )
    else:
        verify_stack_pins(contract, torch.__version__, transformers.__version__)
    torch.set_num_threads(budget)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized by a prior op; intra-op pin below still gates
    verify_torch_pinned(budget, torch.get_num_threads())

    entry = _load_callable(args.entry)
    setup_callable = _load_callable(args.setup) if args.setup is not None else None
    inference_sources_before: list[dict] | None = None
    if unlimited_citable:
        inference_sources_before = capture_inference_sources(
            args.entry, entry, args.setup, setup_callable
        )
    setup_state = None
    if setup_callable is not None:
        # stdout is reserved for the envelope lines; anything the model code
        # prints is redirected to stderr (visible in logs, never parsed).
        with contextlib.redirect_stdout(sys.stderr):
            if args.entry == "gauntlet_ref_unlimited:run_stage":
                setup_state = setup_callable(
                    stage_req,
                    args.page,
                    args.model_dir,
                    max_length=args.max_length,
                    text_dir=args.text_dir,
                )
            else:
                setup_state = setup_callable(stage_req, args.page, args.model_dir)

    requested: list[str] | None = None
    samples_ms: dict[str, list[float]] = {}
    tokens_by_stage: dict[str, int] = {}
    text_shas: list[str] = []
    raw_records: list[dict] = []
    for i in range(args.warmup + args.runs):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            result = entry(stage_req, args.page, args.model_dir, setup_state)
        outer_ms = (time.perf_counter() - t0) * 1000.0

        if requested is None:
            requested = requested_from_result(stage_req, result)
        text_sha = invocation_text_sha(result, i + 1, citable=not args.smoke)
        if isinstance(text_sha, str):
            text_shas.append(text_sha)
        raw_stages: dict[str, dict] = {}
        for stage in requested:
            elapsed_ms, tokens = stage_sample(stage, result, outer_ms)
            if tokens is not None:
                prev = tokens_by_stage.get(stage)
                if prev is not None and prev != tokens:
                    raise FairnessError(
                        f"{stage}: token count drifted across runs ({prev} -> {tokens}); "
                        "a nondeterministic reference cannot land a ratio"
                    )
                tokens_by_stage[stage] = tokens
                if stage == "decode_per_token":
                    elapsed_ms /= tokens
            if i >= args.warmup:
                rounded_ms = round(elapsed_ms, 6)
                samples_ms.setdefault(stage, []).append(rounded_ms)
                raw_sample = {"ms": rounded_ms}
                if tokens is not None:
                    raw_sample["tokens"] = tokens
                raw_stages[stage] = raw_sample
        if i >= args.warmup:
            raw_record = {
                "run_id": f"run_{i - args.warmup + 1:03d}",
                "stages": raw_stages,
            }
            if isinstance(text_sha, str):
                raw_record["text_sha256"] = text_sha
            raw_records.append(raw_record)

    if requested is None:
        raise FairnessError("reference entry produced no measured stage")
    total_invocations = args.warmup + args.runs
    canonical_text_sha = deterministic_text_sha(
        text_shas, total_invocations, citable=not args.smoke
    )
    page_sha256 = sha256_file(args.page)
    reference_model_manifest: dict | None = None
    reference_inference_binding: dict | None = None
    if unlimited_citable:
        if (
            model_manifest_before is None
            or modules_cache is None
            or inference_sources_before is None
        ):
            raise FairnessError(
                "Unlimited-OCR provenance state was not initialized"
            )
        assert_capture_unchanged(
            "reference page", page_sha256_before, page_sha256
        )
        model_manifest_after = capture_reference_model_manifest(
            args.model_dir, UNLIMITED_MODEL_SPEC
        )
        assert_capture_unchanged(
            "reference model manifest",
            model_manifest_before,
            model_manifest_after,
        )
        inference_sources_after = capture_inference_sources(
            args.entry, entry, args.setup, setup_callable
        )
        assert_capture_unchanged(
            "reference inference sources",
            inference_sources_before,
            inference_sources_after,
        )
        reference_model_manifest = model_manifest_before
        reference_inference_binding = build_inference_binding(
            args,
            stage_req,
            budget,
            page_sha256,
            contract,
            reference_model_manifest,
            modules_cache,
            inference_sources_before,
            torch.__version__,
            transformers.__version__,
        )
    records = [
        build_record(
            stage,
            samples_ms[stage],
            budget=budget,
            torch_threads=torch.get_num_threads(),
            precision=args.precision,
            backend=args.backend,
            allocator=args.allocator,
            warmup_discarded=args.warmup,
            reference_contract=contract,
            tokens=tokens_by_stage.get(stage),
            synthetic=args.smoke,
        )
        for stage in requested
    ]
    raw_timing = build_raw_timing(raw_records, source="reference")
    doc = {
        "schema": SCHEMA_DOC,
        "source": "reference",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "env_pins": {
            k: os.environ.get(k, "") for k in PRE_IMPORT_PINS + ("FOCR_THREADS",)
        },
        "ambient_env": {
            key: os.environ.get(key, "<unset>") for key in UNLIMITED_AMBIENT_ENV
        },
        "page": args.page,
        "page_sha256": page_sha256,
        "model": args.model_dir,
        "max_length": args.max_length,
        "text_dir": args.text_dir,
        "threads": budget,
        "precision": args.precision,
        "backend": args.backend,
        "allocator": args.allocator,
        "runs": args.runs,
        "warmup": args.warmup,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "reference_contract": contract,
        "reference_model_manifest": reference_model_manifest,
        "reference_inference_binding": reference_inference_binding,
        "raw_timing": raw_timing,
        "stages": records,
        "synthetic": args.smoke,
        "smoke": args.smoke,
    }
    if canonical_text_sha is not None:
        doc["text_sha256"] = canonical_text_sha
        doc["text_identical_across_runs"] = (
            len(text_shas) == total_invocations and len(set(text_shas)) == 1
        )
    if args.out:
        existing = None
        if args.smoke and os.path.exists(args.out):
            with open(args.out, encoding="utf-8") as f:
                existing = json.load(f)
        if existing is not None and existing.get("schema") == SCHEMA_DOC:
            invariant_fields = (
                "source",
                "page_sha256",
                "model",
                "threads",
                "precision",
                "backend",
                "allocator",
                "runs",
                "warmup",
                "torch_version",
                "transformers_version",
                "reference_contract",
                "reference_model_manifest",
                "reference_inference_binding",
                "max_length",
                "text_dir",
                "ambient_env",
                "synthetic",
                "smoke",
            )
            drift = [
                field for field in invariant_fields if existing.get(field) != doc.get(field)
            ]
            existing_stages = existing.get("stages")
            if drift or not isinstance(existing_stages, list) or any(
                not isinstance(record, dict) for record in existing_stages
            ):
                raise FairnessError(
                    "existing reference stage document is incompatible: "
                    + (", ".join(drift) if drift else "malformed stages")
                )
            remaining_stages = [
                r for r in existing_stages if r.get("stage") not in requested
            ]
            if remaining_stages:
                raw_timing = merge_raw_timing(
                    existing.get("raw_timing"), raw_timing, requested
                )
            existing["stages"] = [
                r for r in existing_stages if r.get("stage") not in requested
            ] + records
            for key in (
                "page_sha256",
                "command",
                "torch_version",
                "transformers_version",
                "reference_contract",
                "reference_model_manifest",
                "reference_inference_binding",
                "max_length",
                "text_dir",
                "ambient_env",
                "text_sha256",
                "text_identical_across_runs",
                "smoke",
                "synthetic",
            ):
                if key in doc:
                    existing[key] = doc[key]
            existing["raw_timing"] = raw_timing
            doc = existing
        mode = "w" if args.smoke else "x"
        try:
            with open(args.out, mode, encoding="utf-8") as output:
                json.dump(doc, output, indent=2)
                output.write("\n")
        except FileExistsError as error:
            raise FairnessError(
                f"citable --out appeared during capture: {args.out}"
            ) from error
    # The envelope(s) MUST be the last stdout lines (the bench harness parses
    # the final line; single-stage behavior is unchanged).
    for record in records:
        _emit(envelope_from_record(record, result="smoke" if args.smoke else "pass"))
    return 0


# ── self-test (no torch required; the gate logic is pure) ───────────────────


def _self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool) -> None:
        print(json.dumps({"check": name, "result": "pass" if ok else "fail"}))
        if not ok:
            failures.append(name)

    def refuses(name: str, fn) -> None:
        try:
            fn()
            check(name, False)
        except FairnessError:
            check(name, True)

    pinned8 = {k: "8" for k in PRE_IMPORT_PINS} | {"FOCR_THREADS": "8"}

    def tiny_model_fixture(index_bytes: bytes | None = None) -> tuple[dict, dict]:
        if index_bytes is None:
            index_bytes = json.dumps(
                {
                    "metadata": {"total_size": 6},
                    "weight_map": {
                        "tensor.a": "weights.bin",
                        "tensor.b": "weights.bin",
                        "tensor.c": "weights.bin",
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        blobs = {
            "config.json": b"{}",
            "index.json": index_bytes,
            "weights.bin": b"abcdef",
        }
        registered = tuple(
            (
                name,
                len(contents),
                hashlib.sha256(contents).hexdigest(),
            )
            for name, contents in sorted(blobs.items())
        )
        spec = {
            "model_id": "test/tiny-model",
            "model_commit": "1" * 40,
            "files": registered,
            "file_count": 3,
            "index_path": "index.json",
            "index_total_size": 6,
            "index_weight_count": 3,
            "shard_path": "weights.bin",
        }
        observations = {
            name: {
                "path": name,
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
                "kind": "regular",
            }
            for name, contents in blobs.items()
        }
        return spec, observations

    check(
        "unlimited-registration-exact-12",
        UNLIMITED_MODEL_SPEC["file_count"] == 12
        and len(UNLIMITED_MODEL_FILES) == 12
        and [item[0] for item in UNLIMITED_MODEL_FILES]
        == sorted(item[0] for item in UNLIMITED_MODEL_FILES)
        and UNLIMITED_MODEL_SPEC["model_commit"] == UNLIMITED_MODEL_COMMIT,
    )
    tiny_spec, tiny_observations = tiny_model_fixture()
    tiny_index = json.dumps(
        {
            "metadata": {"total_size": 6},
            "weight_map": {
                "tensor.a": "weights.bin",
                "tensor.b": "weights.bin",
                "tensor.c": "weights.bin",
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tiny_manifest = build_reference_model_manifest(
        tiny_observations, tiny_index, tiny_spec
    )
    manual_root = hashlib.sha256(MODEL_ROOT_DOMAIN)
    for item in tiny_manifest["files"]:
        manual_root.update(item["path"].encode("utf-8") + b"\0")
        manual_root.update(str(item["bytes"]).encode("ascii") + b"\0")
        manual_root.update(item["sha256"].encode("ascii") + b"\0")
    check(
        "model-manifest-domain-sorted-and-structural",
        tiny_manifest["schema"] == SCHEMA_MODEL_MANIFEST
        and tiny_manifest["root_hash_domain"] == MODEL_ROOT_DOMAIN.decode("ascii")
        and tiny_manifest["root_sha256"] == manual_root.hexdigest()
        and tiny_manifest["index"]
        == {
            "path": "index.json",
            "total_size": 6,
            "weight_count": 3,
            "shards": ["weights.bin"],
        }
        and [item["path"] for item in tiny_manifest["files"]]
        == ["config.json", "index.json", "weights.bin"],
    )
    reversed_observations = dict(reversed(list(tiny_observations.items())))
    check(
        "model-root-independent-of-observation-order",
        build_reference_model_manifest(
            reversed_observations, tiny_index, tiny_spec
        )["root_sha256"]
        == tiny_manifest["root_sha256"],
    )

    missing_observations = dict(tiny_observations)
    missing_observations.pop("config.json")
    refuses(
        "model-manifest-missing-file-refused",
        lambda: build_reference_model_manifest(
            missing_observations, tiny_index, tiny_spec
        ),
    )
    extra_observations = dict(tiny_observations)
    extra_observations["extra.txt"] = {
        "path": "extra.txt",
        "bytes": 1,
        "sha256": "0" * 64,
        "kind": "regular",
    }
    refuses(
        "model-manifest-extra-observation-refused",
        lambda: build_reference_model_manifest(
            extra_observations, tiny_index, tiny_spec
        ),
    )
    for file_kind in ("symlink", "special"):
        bad_kind = {name: dict(value) for name, value in tiny_observations.items()}
        bad_kind["config.json"]["kind"] = file_kind
        refuses(
            f"model-manifest-{file_kind}-refused",
            lambda observations=bad_kind: build_reference_model_manifest(
                observations, tiny_index, tiny_spec
            ),
        )
    uppercase_observation = {
        name: dict(value) for name, value in tiny_observations.items()
    }
    uppercase_observation["config.json"]["sha256"] = uppercase_observation[
        "config.json"
    ]["sha256"].upper()
    refuses(
        "model-manifest-uppercase-observation-refused",
        lambda: build_reference_model_manifest(
            uppercase_observation, tiny_index, tiny_spec
        ),
    )
    wrong_hash = {name: dict(value) for name, value in tiny_observations.items()}
    wrong_hash["config.json"]["sha256"] = "0" * 64
    refuses(
        "model-manifest-hash-mismatch-refused",
        lambda: build_reference_model_manifest(wrong_hash, tiny_index, tiny_spec),
    )
    uppercase_spec = dict(tiny_spec)
    uppercase_files = list(tiny_spec["files"])
    name, size, digest = uppercase_files[0]
    uppercase_files[0] = (name, size, digest.upper())
    uppercase_spec["files"] = tuple(uppercase_files)
    refuses(
        "model-registration-uppercase-hash-refused",
        lambda: build_reference_model_manifest(
            tiny_observations, tiny_index, uppercase_spec
        ),
    )
    wrong_total_spec = dict(tiny_spec) | {"index_total_size": 7}
    refuses(
        "model-index-total-size-refused",
        lambda: build_reference_model_manifest(
            tiny_observations, tiny_index, wrong_total_spec
        ),
    )
    wrong_count_spec = dict(tiny_spec) | {"index_weight_count": 4}
    refuses(
        "model-index-weight-count-refused",
        lambda: build_reference_model_manifest(
            tiny_observations, tiny_index, wrong_count_spec
        ),
    )
    alternate_index = json.dumps(
        {
            "metadata": {"total_size": 6},
            "weight_map": {
                "tensor.a": "other.bin",
                "tensor.b": "other.bin",
                "tensor.c": "other.bin",
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    alternate_spec, alternate_observations = tiny_model_fixture(alternate_index)
    refuses(
        "model-index-alternate-shard-refused",
        lambda: build_reference_model_manifest(
            alternate_observations, alternate_index, alternate_spec
        ),
    )
    duplicate_index = (
        b'{"metadata":{"total_size":6,"total_size":6},'
        b'"weight_map":{"tensor.a":"weights.bin",'
        b'"tensor.b":"weights.bin","tensor.c":"weights.bin"}}'
    )
    duplicate_spec, duplicate_observations = tiny_model_fixture(duplicate_index)
    refuses(
        "model-index-duplicate-key-refused",
        lambda: build_reference_model_manifest(
            duplicate_observations, duplicate_index, duplicate_spec
        ),
    )
    check(
        "capture-drift-equal-pass",
        assert_capture_unchanged("tiny manifest", tiny_manifest, tiny_manifest) is None,
    )
    drifted_manifest = dict(tiny_manifest) | {"root_sha256": "0" * 64}
    refuses(
        "capture-drift-refused",
        lambda: assert_capture_unchanged(
            "tiny manifest", tiny_manifest, drifted_manifest
        ),
    )

    binding_args = argparse.Namespace(
        entry="gauntlet_ref_unlimited:run_stage",
        setup="gauntlet_ref_unlimited:setup",
        page="page.png",
        model_dir="model",
        max_length=UNLIMITED_MAX_LENGTH,
        text_dir="/evidence/text",
        backend="hf",
        precision="bf16",
        runs=3,
        warmup=1,
        allocator="system",
    )
    binding_contract = require_reference_contract(binding_args.entry)
    tiny_binding = build_inference_binding(
        binding_args,
        "all",
        8,
        "2" * 64,
        binding_contract,
        tiny_manifest,
        {
            "evidence_dir": "/evidence",
            "path": "ref.json.hf_modules_cache",
            "effective_path": "/evidence/ref.json.hf_modules_cache",
            "fresh": True,
        },
        [
            {
                "role": "harness",
                "callable": "gauntlet_reference:main",
                "path": "scripts/gauntlet_reference.py",
                "bytes": 1,
                "sha256": "3" * 64,
            }
        ],
        "2.10.0+cpu",
        "4.57.1",
    )
    check(
        "inference-binding-covers-effective-args-and-source",
        tiny_binding["schema"] == SCHEMA_INFERENCE_BINDING
        and tiny_binding["model_root_sha256"] == tiny_manifest["root_sha256"]
        and tiny_binding["threads"] == 8
        and tiny_binding["max_length"] == UNLIMITED_MAX_LENGTH
        and tiny_binding["text_dir"] == "/evidence/text"
        and tiny_binding["hf_modules_cache"]["fresh"]
        and tiny_binding["sources"][0]["sha256"] == "3" * 64
        and TEXT_SHA256_RE.fullmatch(tiny_binding["binding_sha256"]) is not None,
    )

    # Budget resolution: pinned accepted; unset/zero/64 refused.
    check("budget-env", resolve_budget(None, pinned8) == 8)
    check("budget-arg-overrides", resolve_budget(4, pinned8) == 4)
    refuses("budget-unpinned-refused", lambda: resolve_budget(None, {}))
    refuses("budget-zero-refused", lambda: resolve_budget(0, pinned8))
    refuses("budget-64-refused", lambda: resolve_budget(64, pinned8))
    refuses(
        "budget-garbage-refused",
        lambda: resolve_budget(None, {"FOCR_THREADS": "eight"}),
    )

    # Env pins: all present+equal passes; missing or drifted refuses.
    check("pins-ok", verify_env_pins(8, pinned8) is None)
    refuses(
        "pins-missing-refused", lambda: verify_env_pins(8, {"OMP_NUM_THREADS": "8"})
    )
    refuses(
        "pins-drifted-refused",
        lambda: verify_env_pins(8, pinned8 | {"MKL_NUM_THREADS": "64"}),
    )
    check(
        "unlimited-explicit-inference-args-pass",
        verify_unlimited_citable_args(
            "gauntlet_ref_unlimited:run_stage",
            UNLIMITED_MAX_LENGTH,
            "/evidence/text",
            {},
        )
        is None,
    )
    refuses(
        "unlimited-ambient-max-length-refused",
        lambda: verify_unlimited_citable_args(
            "gauntlet_ref_unlimited:run_stage",
            UNLIMITED_MAX_LENGTH,
            "/evidence/text",
            {"FOCR_REF_MAX_LENGTH": "4096"},
        ),
    )
    refuses(
        "unlimited-ambient-text-dir-refused",
        lambda: verify_unlimited_citable_args(
            "gauntlet_ref_unlimited:run_stage",
            UNLIMITED_MAX_LENGTH,
            "/evidence/text",
            {"FOCR_REF_TEXT_DIR": "/ambient"},
        ),
    )
    refuses(
        "unlimited-max-length-drift-refused",
        lambda: verify_unlimited_citable_args(
            "gauntlet_ref_unlimited:run_stage", 4096, "/evidence/text", {}
        ),
    )

    # torch thread proof and immutable entry-specific stack contracts.
    check("torch-proof-ok", verify_torch_pinned(8, 8) is None)
    refuses("torch-proof-64-refused", lambda: verify_torch_pinned(8, 64))
    for entry, (
        contract_id,
        torch_pin,
        transformers_pin,
    ) in REFERENCE_CONTRACTS.items():
        contract = require_reference_contract(entry)
        check(
            f"contract-map-{contract_id}",
            contract
            == {
                "schema": REFERENCE_CONTRACT_SCHEMA,
                "id": contract_id,
                "entry": entry,
                "torch_version": torch_pin,
                "transformers_version": transformers_pin,
            },
        )
        check(
            f"contract-stack-{contract_id}",
            verify_stack_pins(contract, torch_pin + "+cpu", transformers_pin) is None,
        )
        check(
            f"contract-assertions-{contract_id}",
            verify_requested_stack_pins(contract, torch_pin, transformers_pin) is None,
        )
        check(
            f"contract-optional-assertions-{contract_id}",
            verify_requested_stack_pins(contract, None, None) is None,
        )
        refuses(
            f"contract-torch-mismatch-{contract_id}",
            lambda c=contract: verify_requested_stack_pins(c, "0.0.0", None),
        )
        refuses(
            f"contract-transformers-mismatch-{contract_id}",
            lambda c=contract: verify_requested_stack_pins(c, None, "0.0.0"),
        )
    refuses(
        "contract-unknown-entry-refused",
        lambda: require_reference_contract("gauntlet_ref_zoo:run_unknown"),
    )
    unlimited_contract = require_reference_contract(
        "gauntlet_ref_unlimited:run_stage"
    )
    refuses(
        "contract-runtime-torch-drift-refused",
        lambda: verify_stack_pins(unlimited_contract, "2.11.0", "4.57.1"),
    )
    refuses(
        "contract-runtime-transformers-drift-refused",
        lambda: verify_stack_pins(unlimited_contract, "2.10.0", "4.58.0"),
    )

    # Records refuse to exist without samples; the envelope carries every
    # mandatory field of the BENCH_HARNESS.md contract.
    refuses(
        "empty-samples-refused",
        lambda: build_record(
            "prefill",
            [],
            budget=8,
            torch_threads=8,
            precision="bf16",
            backend="hf",
            allocator="system",
            warmup_discarded=1,
            reference_contract=unlimited_contract,
        ),
    )
    record = build_record(
        "decode_per_token",
        [14.5, 15.0, 14.7],
        budget=8,
        torch_threads=8,
        precision="bf16",
        backend="hf",
        allocator="system",
        warmup_discarded=1,
        reference_contract=unlimited_contract,
        tokens=600,
        synthetic=True,
    )
    check("record-best", record["best_ms"] == 14.5)
    check("record-thread-proof", record["thread_proof"]["torch_num_threads"] == 8)
    check("record-contract-stamp", record["reference_contract"] == unlimited_contract)
    envelope = envelope_from_record(record)
    for field in (
        "stage",
        "result",
        "p50_ms",
        "precision",
        "threads",
        "reference_backend",
    ):
        check(
            f"envelope-has-{field}",
            field in envelope and envelope[field] not in ("", None),
        )
    check("envelope-result-pass", envelope["result"] == "pass")
    check(
        "envelope-contract-stamp",
        envelope["reference_contract_id"] == unlimited_contract["id"]
        and envelope["reference_torch_version"]
        == unlimited_contract["torch_version"]
        and envelope["reference_transformers_version"]
        == unlimited_contract["transformers_version"],
    )
    check(
        "envelope-result-smoke",
        envelope_from_record(record, result="smoke")["result"] == "smoke",
    )

    raw = build_raw_timing(
        [
            {
                "run_id": f"run_{index:03d}",
                "stages": {"decode_per_token": {"ms": sample, "tokens": 600}},
                "text_sha256": "ab" * 32,
            }
            for index, sample in enumerate((14.5, 15.0, 14.7), start=1)
        ],
        source="reference",
    )
    check(
        "raw-timing-versioned-and-unaggregated",
        raw["schema"] == SCHEMA_RAW_TIMING
        and raw["measured_runs"] == 3
        and [
            item["stages"]["decode_per_token"]["ms"]
            for item in raw["records"]
        ]
        == record["samples_ms"],
    )
    merged = merge_raw_timing(
        raw,
        build_raw_timing(
            [
                {
                    "run_id": f"run_{index:03d}",
                    "stages": {"prefill": {"ms": sample}},
                    "text_sha256": "ab" * 32,
                }
                for index, sample in enumerate((1.0, 2.0, 3.0), start=1)
            ],
            source="reference",
        ),
        ["prefill"],
    )
    check(
        "raw-timing-stage-merge-preserves-observations",
        all(
            set(item["stages"]) == {"decode_per_token", "prefill"}
            for item in merged["records"]
        ),
    )
    refuses(
        "raw-timing-noncanonical-run-id-refused",
        lambda: build_raw_timing(
            [{"run_id": "arbitrary", "stages": {"prefill": {"ms": 1.0}}}],
            source="reference",
        ),
    )
    check(
        "citable-text-sha-valid",
        invocation_text_sha(
            {"text_sha256": "ab" * 32}, 1, citable=True
        )
        == "ab" * 32
        and deterministic_text_sha(["ab" * 32] * 3, 3, citable=True)
        == "ab" * 32,
    )
    refuses(
        "citable-text-sha-missing-refused",
        lambda: invocation_text_sha({}, 1, citable=True),
    )
    refuses(
        "citable-text-sha-invalid-refused",
        lambda: invocation_text_sha({"text_sha256": "not-a-hash"}, 1, citable=True),
    )
    refuses(
        "citable-text-sha-partial-sequence-refused",
        lambda: deterministic_text_sha(["ab" * 32] * 2, 3, citable=True),
    )
    refuses(
        "citable-text-sha-drift-refused",
        lambda: deterministic_text_sha(
            ["ab" * 32, "cd" * 32, "ab" * 32], 3, citable=True
        ),
    )
    check(
        "smoke-output-may-reuse-noncitable-path",
        require_fresh_output(__file__, smoke=True) is None,
    )
    refuses(
        "citable-output-must-be-fresh",
        lambda: require_fresh_output(__file__, smoke=False),
    )

    # Entry protocols (stage_sample): outer wall, legacy tokens, multi-stage.
    check("sample-none-outer", stage_sample("prefill", None, 123.0) == (123.0, None))
    check(
        "sample-legacy-tokens",
        stage_sample("decode_per_token", {"tokens": 600}, 90.0) == (90.0, 600),
    )
    multi = {
        "stages": {
            "preprocess": {"ms": 120.0},
            "prefill": {"ms": 600.0, "tokens": 290},
            "decode_per_token": {"ms": 750.0, "tokens": 3},
            "end_to_end": {"ms": 3000.0},
        },
        "text_sha256": "ab" * 32,
    }
    check("sample-multi-ms", stage_sample("prefill", multi, 9e9) == (600.0, 290))
    check(
        "sample-multi-decode",
        stage_sample("decode_per_token", multi, 9e9) == (750.0, 3),
    )
    refuses(
        "sample-missing-stage-refused",
        lambda: stage_sample("vision_encode", multi, 1.0),
    )
    refuses(
        "sample-nonpositive-ms-refused",
        lambda: stage_sample("prefill", {"stages": {"prefill": {"ms": 0}}}, 1.0),
    )
    refuses(
        "sample-nonpositive-tokens-refused",
        lambda: stage_sample(
            "prefill", {"stages": {"prefill": {"ms": 1.0, "tokens": 0}}}, 1.0
        ),
    )
    refuses("sample-bad-type-refused", lambda: stage_sample("prefill", "nope", 1.0))

    # `--stage all` resolution: ledger order, multi-stage protocol required.
    check(
        "all-resolves-ledger-order",
        requested_from_result(ALL, multi)
        == ["preprocess", "prefill", "decode_per_token", "end_to_end"],
    )
    check(
        "single-stage-passthrough",
        requested_from_result("prefill", None) == ["prefill"],
    )
    refuses("all-requires-stages-refused", lambda: requested_from_result(ALL, None))
    refuses(
        "all-empty-vocab-refused",
        lambda: requested_from_result(ALL, {"stages": {"warpdrive": {"ms": 1.0}}}),
    )

    if failures:
        print(f"SELF-TEST FAILED: {failures}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "gauntlet-reference-self-test", "result": "pass"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--stage", choices=STAGES + (ALL,), default=None)
    parser.add_argument("--page", default=None)
    parser.add_argument("--model-dir", default=os.environ.get("FOCR_MODEL_DIR"))
    parser.add_argument("--backend", default=os.environ.get("FOCR_REFERENCE_BACKEND"))
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--text-dir", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--allocator", default="system")
    parser.add_argument("--entry", default=None, help="module:function timed per stage")
    parser.add_argument(
        "--setup", default=None, help="module:function run once, unclocked"
    )
    parser.add_argument(
        "--out", default=None, help="stage-record JSON (merged per stage)"
    )
    parser.add_argument(
        "--pin-torch",
        default=None,
        help="optional legacy assertion; must equal the entry's inferred contract",
    )
    parser.add_argument(
        "--pin-transformers",
        default=None,
        help="optional legacy assertion; must equal the entry's inferred contract",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="single untimed plumbing run: forces runs=1 warmup=0 and stamps every "
        "output synthetic/non-citable (gauntlet_row.py refuses it)",
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if args.smoke:
        args.runs, args.warmup = 1, 0

    stage = (
        args.stage
        or os.environ.get("FOCR_GAUNTLET_STAGE")
        or os.environ.get("FOCR_STAGE")
    )
    if stage not in STAGES + (ALL,):
        _emit({"result": "error", "reason": "no_stage", "detail": f"stage={stage!r}"})
        return 2
    if not args.backend or not str(args.backend).strip():
        _emit({"stage": stage, "result": "error", "reason": "no_reference_backend"})
        return 2
    if (
        args.runs < 1
        or args.runs > MAX_MEASURED_RUNS
        or args.warmup < 0
        or args.warmup > MAX_WARMUP_RUNS
    ):
        _emit({"stage": stage, "result": "error", "reason": "bad_run_counts"})
        return 2

    try:
        if args.entry is None:
            contract = None
        elif args.smoke:
            contract = infer_reference_contract(args.entry)
        else:
            contract = require_reference_contract(args.entry)
        if contract is not None:
            verify_requested_stack_pins(
                contract, args.pin_torch, args.pin_transformers
            )
        elif args.pin_torch is not None or args.pin_transformers is not None:
            raise FairnessError(
                "legacy --pin-* assertions require a registered reference entry"
            )
        budget = resolve_budget(args.threads, dict(os.environ))
        verify_env_pins(budget, dict(os.environ))
        verify_unlimited_citable_args(
            args.entry, args.max_length, args.text_dir, dict(os.environ)
        )
        return run_stage(args, stage, budget, contract)
    except FairnessError as err:
        # Fail-closed: an unfair run emits an error envelope and NO timing row.
        _emit(
            {
                "stage": stage,
                "result": "error",
                "reason": "fairness",
                "detail": str(err),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
