#!/usr/bin/env bash
# gauntlet_runbook.sh — the EXACT serial command sequence for the bd-re8.17
# quiet-host head-to-head measurement (focr vs the pinned CPU HF baseline).
#
# EVERYTHING is embedded: truth-pack provenance (model_commit, fixture hashes),
# fairness pins (N=8 threads on BOTH sides), page/model/venv paths, claim ids.
# On the quiet host you run ONE command and read the printed ledger row:
#
#   bash scripts/gauntlet_runbook.sh all            # build -> preflight -> ... -> row
#
# or step-by-step (each step is idempotent and strictly serial):
#
#   bash scripts/gauntlet_runbook.sh build          # clean-tree RCH release-perf build
#   bash scripts/gauntlet_runbook.sh preflight      # receipt + fixtures + self-tests
#   bash scripts/gauntlet_runbook.sh focr           # TIMED: focr best-of-5 warm @8
#   bash scripts/gauntlet_runbook.sh reference      # TIMED: HF baseline best-of-5 warm @8
#   bash scripts/gauntlet_runbook.sh roofline       # untimed: §9.1 floors
#   bash scripts/gauntlet_runbook.sh cer            # untimed: correctness proof (CER)
#   bash scripts/gauntlet_runbook.sh row            # untimed: PERF_LEDGER row draft + validation
#   bash scripts/gauntlet_runbook.sh --self-test     # pure receipt-binding checks
#
# QUIET-HOST DOCTRINE: the two TIMED steps poison their cv% on a contended
# host. preflight refuses when 1-min loadavg >= LOAD_MAX (default 2.0);
# FORCE=1 overrides (recorded). Close editors/indexers/other agents first.
#
# Knobs (all optional):
#   APPLY=1          REFUSED here: strict evidence descendants may change only
#                    the row-declared artifacts/perf subtree, never the ledger
#   VERIFY_SHARD=1   re-hash the 6.7GB weights shard in preflight (~40s)
#   FORCE=1          bypass the loadavg gate (the row notes must say why)
#   PAGES="page_0009.png page_0014.png"   override the measured page set
#   FOCR_GAUNTLET_SCRATCH_ROOT=/path       fresh Cargo target parent for `build`
#   FOCR_GAUNTLET_HF_HOME=/path            offline Hugging Face cache directory
#   FOCR_GAUNTLET_TMPDIR=/path             reference-side temporary directory
# Required subject path:
#   FOCR_GAUNTLET_FOCR_MODEL=/absolute/path/to/conservative-recipe.focrq
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── EMBEDDED PROVENANCE (docs/truth-pack/PINNED_SOURCES.md + SOURCE_HASHES.md
#    + the verified baseline workspace) — resolve nothing at measurement time ──
MODEL_COMMIT="3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"    # HF baidu/Unlimited-OCR pin
PINNED_TORCH="2.10.0"
PINNED_TRANSFORMERS="4.57.1"
# model-00001-of-000001.safetensors (6672547120 bytes) — HF LFS etag, VERIFIED 2026-06-26
SHARD_SHA256="2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6"
# Perf fixture pages (200 DPI renders, seed 20260626; hashed 2026-07-02).
# (A function, not `declare -A`: /bin/bash on macOS is 3.2.)
page_sha256() {
  case "$1" in
    page_0009.png) echo "62526eb02efd588005ba70eba171d0e4bf64f4a0f0d258f6e72e8a14830b74da" ;;
    page_0014.png) echo "f1e35a58f673036b16302c86676f1ec6a89218fb4e666c477dbb4ae21b904df2" ;;
    *) echo "" ;;
  esac
}

# ── paths (the proven baseline workspace; override via env if it moved) ─────
WORK="${FOCR_GAUNTLET_WORK:-/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work}"
REFERENCE_MODEL_DIR="${FOCR_GAUNTLET_REFERENCE_MODEL_DIR:-$WORK/model}"
SUBJECT_MODEL="${FOCR_GAUNTLET_FOCR_MODEL:-}"
PAGES_DIR="${FOCR_GAUNTLET_PAGES_DIR:-$WORK/pages}"
VENV_PY="${FOCR_GAUNTLET_VENV_PY:-/Volumes/focrvenv/venv/bin/python}"
REFERENCE_HF_HOME="${FOCR_GAUNTLET_HF_HOME:-$WORK/hf_home}"
REFERENCE_TMPDIR="${FOCR_GAUNTLET_TMPDIR:-$WORK/tmp}"
FOCR_BIN="${FOCR_BIN:-}"

# ── measurement contract (docs/PERF_LEDGER.md §9.3) ─────────────────────────
THREADS=8                      # ONE budget, BOTH sides; NEVER 64
RUNS=5                         # best-of-5 ...
WARMUP=1                       # ... after 1 discarded warm run
LOAD_MAX="${LOAD_MAX:-2.0}"
PAGES="${PAGES:-page_0009.png page_0014.png}"
STAMP="$(date -u +%Y%m%d)"
# OUT_DIR overrides the evidence home (default = the original bd-re8.17 run;
# a RE-RUN must use a fresh dir — the aggregator refuses mixed sessions, and
# committed evidence is immutable).
OUT="${OUT_DIR:-$REPO_ROOT/artifacts/perf/bd-re8.17}"
ARCH_JSON="$OUT/arch.json"
BUILD_DIR="$OUT/build"
BUILD_RECEIPT="${FOCR_GAUNTLET_BUILD_RECEIPT:-$BUILD_DIR/build_receipt.json}"
SOURCE_MANIFEST="${FOCR_GAUNTLET_SOURCE_MANIFEST:-$BUILD_DIR/source_input_manifest.json}"
FOCR_BIN="${FOCR_BIN:-$BUILD_DIR/focr}"

# Native subject contract. The hash/size are the measured conservative artifact
# and are intentionally independent of the pinned raw HF reference shard.
SUBJECT_MODEL_SHA256="573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592"
SUBJECT_MODEL_SIZE="4157448783"
SUBJECT_QUANT_RECIPE="unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1"
SUBJECT_PRECISION="focr-mixed-ffn-int8"
SUBJECT_DECODE_MODE="mixed-ffn-int8"

# Reference-side pool pins — MUST be in the environment BEFORE python starts
# (gauntlet_reference.py refuses otherwise; torch/OMP read them at import).
ref_env() {
  env -u FOCR_REF_MAX_LENGTH -u FOCR_REF_TEXT_DIR \
      FOCR_THREADS="$THREADS" \
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      OPENBLAS_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS" \
      NUMEXPR_NUM_THREADS="$THREADS" \
      HF_HOME="$REFERENCE_HF_HOME" TMPDIR="$REFERENCE_TMPDIR" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "$@"
}

die() { echo "RUNBOOK-FATAL: $*" >&2; exit 1; }
note() { echo "runbook: $*" >&2; }

stem() { local p="$1"; p="${p%.png}"; echo "$p"; }

# Build receipts bind the measured executable to a clean, reproducible local
# source closure. The source root is SHA-256 over entries sorted by
# (repository, UTF-8 path), each encoded as:
#
#   repository NUL path NUL decimal-size NUL lowercase-sha256 LF
#
# The version-domain prefix `focr-source-input-root/v1 NUL` is hashed first.
# Reachable local path dependencies are included because Cargo.lock does not
# bind their source bytes.
build_receipt_tool() {
  python3 - "$REPO_ROOT" "$@" <<'PY'
import datetime
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from gauntlet_row import build_receipt_document

MANIFEST_SCHEMA = "focr-source-input-manifest/v1"
RECEIPT_SCHEMA = "focr-build-receipt/v1"
ROOT_DOMAIN = b"focr-source-input-root/v1\0"
MAX_SOURCE_ENTRIES = 50_000
MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_SOURCE_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_BUILD_RECEIPT_BYTES = 1024 * 1024
MAX_BINARY_BYTES = 1024 * 1024 * 1024
MAX_LOGICAL_PATH_BYTES = 4096
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class ReceiptError(RuntimeError):
    pass


def run(command, *, cwd, timeout=120, text=True, check=True):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=text,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReceiptError(f"cannot run {command!r}: {error}") from error
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode("utf-8", "replace").strip()
        raise ReceiptError(f"{command!r} failed: {stderr}")
    return result


def stable_file(path, maximum=MAX_SOURCE_FILE_BYTES):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ReceiptError(f"build input is not a regular file: {path}")
    if before.st_size > maximum:
        raise ReceiptError(f"build input exceeds the {maximum}-byte bound: {path}")
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            observed += len(chunk)
            digest.update(chunk)
    after = path.lstat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ReceiptError(f"build input changed while hashing: {path}")
    if observed != before.st_size:
        raise ReceiptError(f"build input changed length while hashing: {path}")
    return {"sha256": digest.hexdigest(), "size": observed}


def canonical_root(entries):
    if len(entries) > MAX_SOURCE_ENTRIES:
        raise ReceiptError("source manifest exceeds the entry-count bound")
    ordered = sorted(entries, key=lambda item: (item["repository"], item["path"]))
    digest = hashlib.sha256(ROOT_DOMAIN)
    identities = set()
    total_bytes = 0
    for entry in ordered:
        repository = entry["repository"]
        path = entry["path"]
        size = entry["size"]
        sha256 = entry["sha256"]
        if (
            not isinstance(repository, str)
            or not isinstance(path, str)
            or "\0" in repository
            or "\0" in path
            or len(repository.encode("utf-8")) > MAX_LOGICAL_PATH_BYTES
            or len(path.encode("utf-8")) > MAX_LOGICAL_PATH_BYTES
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(sha256, str)
            or HEX64.fullmatch(sha256) is None
        ):
            raise ReceiptError("source manifest contains a noncanonical entry")
        identity = (repository, path)
        if identity in identities:
            raise ReceiptError(f"source manifest contains duplicate entry: {repository}/{path}")
        identities.add(identity)
        if size > MAX_SOURCE_FILE_BYTES:
            raise ReceiptError(f"source manifest file exceeds its size bound: {repository}/{path}")
        total_bytes += size
        if total_bytes > MAX_SOURCE_TOTAL_BYTES:
            raise ReceiptError("source manifest exceeds the total-byte bound")
        digest.update(repository.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), ordered


def git_root(path):
    output = run(
        ["git", "rev-parse", "--show-toplevel"], cwd=path
    ).stdout.strip()
    return Path(output).resolve()


def git_head(path):
    output = run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=path
    ).stdout.strip()
    if HEX40.fullmatch(output) is None:
        raise ReceiptError(f"repository has no canonical HEAD: {path}")
    return output


def reachable_local_packages(root):
    metadata_result = run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        cwd=root,
        timeout=300,
    )
    try:
        metadata = json.loads(metadata_result.stdout)
        packages = {package["id"]: package for package in metadata["packages"]}
        nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
        pending = [metadata["resolve"]["root"]]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"cargo metadata is malformed: {error}") from error
    reachable = set()
    while pending:
        package_id = pending.pop()
        if package_id in reachable:
            continue
        reachable.add(package_id)
        node = nodes.get(package_id)
        if node is None:
            raise ReceiptError(f"cargo metadata omits resolve node {package_id}")
        pending.extend(dependency["pkg"] for dependency in node.get("deps", []))
    return [
        packages[package_id]
        for package_id in sorted(reachable)
        if packages[package_id].get("source") is None
    ]


def selected_repositories(root):
    root = Path(root).resolve()
    root_git = git_root(root)
    if root_git != root:
        raise ReceiptError(f"repository root mismatch: {root_git} != {root}")
    repositories = {}
    for package in reachable_local_packages(root):
        manifest = Path(package["manifest_path"]).resolve()
        package_root = manifest.parent
        repository = git_root(package_root)
        record = repositories.setdefault(
            repository,
            {"packages": set(), "selectors": set()},
        )
        record["packages"].add(package["name"])
        if repository == root:
            record["selectors"].update(
                {
                    "Cargo.toml",
                    "Cargo.lock",
                    "rust-toolchain.toml",
                    "build.rs",
                    ".cargo/config",
                    ".cargo/config.toml",
                    "models/manifest.json",
                    "src",
                }
            )
        else:
            # Workspace-level resources are commonly reached by compile-time
            # include_str!/include_bytes! from a nested package (for example,
            # fsqlite-observability includes a root docs file). They are local
            # build inputs even though they sit outside the crate directory.
            record["selectors"].update(
                {"assets", "artifacts", "docs", "models", "templates"}
            )
        if repository != root and package_root == repository:
            # A root package can compile include_str!/include_bytes! resources
            # anywhere below CARGO_MANIFEST_DIR. Select the whole dependency
            # repository so those bytes cannot escape the clean-tree gate.
            record["selectors"].update(
                {
                    ".",
                    ":(exclude).beads",
                    ":(exclude).claude/worktrees",
                    ":(exclude).stash_janitor_workspace",
                }
            )
        elif repository != root:
            record["selectors"].add(package_root.relative_to(repository).as_posix())

        ancestor = package_root
        while True:
            candidate = ancestor / "Cargo.toml"
            if candidate.is_file():
                record["selectors"].add(candidate.relative_to(repository).as_posix())
            if ancestor == repository:
                break
            if repository not in ancestor.parents:
                raise ReceiptError(f"package escaped its git repository: {manifest}")
            ancestor = ancestor.parent

        for target in package.get("targets", []):
            kinds = set(target.get("kind", []))
            if not (kinds.intersection({"lib", "rlib", "proc-macro", "custom-build"})):
                continue
            source = Path(target["src_path"]).resolve()
            try:
                relative = source.relative_to(repository).as_posix()
            except ValueError as error:
                raise ReceiptError(f"target source escaped its git repository: {source}") from error
            record["selectors"].add(relative)

    names = {}
    for repository in repositories:
        identifier = "workspace" if repository == root else f"local-dep/{repository.name}"
        if identifier in names:
            raise ReceiptError(f"ambiguous local dependency repository id: {identifier}")
        names[identifier] = repository
        repositories[repository]["id"] = identifier
    return repositories


def decode_git_paths(raw):
    try:
        return [part.decode("utf-8") for part in raw.split(b"\0") if part]
    except UnicodeDecodeError as error:
        raise ReceiptError("git build-input path is not UTF-8") from error


def cargo_config_paths(root):
    root = Path(root).resolve()
    candidates = []
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")).resolve()
    for name in ("config", "config.toml"):
        candidates.append((f"cargo-home/{name}", cargo_home / name))
    current = root
    depth = 0
    while True:
        for name in ("config", "config.toml"):
            candidates.append((f"cwd-ancestor-{depth}/{name}", current / ".cargo" / name))
        if current.parent == current:
            break
        current = current.parent
        depth += 1
    unique = {}
    for logical, physical in candidates:
        try:
            metadata = physical.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ReceiptError(f"Cargo config is not a regular file: {physical}")
        unique.setdefault(physical.resolve(), logical)
    return [(logical, physical) for physical, logical in unique.items()]


def collect_manifest(root):
    root = Path(root).resolve()
    selected = selected_repositories(root)
    entries = []
    repository_docs = []
    physical_seen = set()
    total_bytes = 0
    for repository, record in sorted(selected.items(), key=lambda item: item[1]["id"]):
        selectors = sorted(record["selectors"])
        status = run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
                *selectors,
            ],
            cwd=repository,
            text=False,
        ).stdout
        if status:
            dirty = decode_git_paths(status)
            detail = "; ".join(repr(item) for item in dirty[:20])
            raise ReceiptError(
                f"dirty tracked/untracked build inputs in {record['id']}: {detail}"
            )
        files = decode_git_paths(
            run(
                ["git", "ls-files", "-z", "--", *selectors],
                cwd=repository,
                text=False,
            ).stdout
        )
        if not files:
            raise ReceiptError(f"no tracked build inputs selected in {record['id']}")
        for relative in sorted(set(files)):
            lexical = repository / relative
            try:
                lexical_metadata = lexical.lstat()
            except FileNotFoundError as error:
                raise ReceiptError(f"tracked build input is missing: {lexical}") from error
            if not stat.S_ISREG(lexical_metadata.st_mode):
                raise ReceiptError(f"tracked build input is not a regular file: {lexical}")
            physical = lexical.resolve()
            try:
                physical.relative_to(repository)
            except ValueError as error:
                raise ReceiptError(f"build input escapes repository: {relative}") from error
            if physical in physical_seen:
                continue
            physical_seen.add(physical)
            identity = stable_file(lexical)
            total_bytes += identity["size"]
            if len(entries) >= MAX_SOURCE_ENTRIES:
                raise ReceiptError("source closure exceeds the entry-count bound")
            if total_bytes > MAX_SOURCE_TOTAL_BYTES:
                raise ReceiptError("source closure exceeds the total-byte bound")
            entries.append(
                {
                    "repository": record["id"],
                    "path": relative,
                    **identity,
                }
            )
        repository_docs.append(
            {
                "id": record["id"],
                "path": str(repository),
                "git_head": git_head(repository),
                "packages": sorted(record["packages"]),
                "selectors": selectors,
            }
        )

    cargo_configs = []
    for logical, physical in cargo_config_paths(root):
        if physical in physical_seen:
            continue
        physical_seen.add(physical)
        identity = stable_file(physical)
        total_bytes += identity["size"]
        if len(entries) >= MAX_SOURCE_ENTRIES:
            raise ReceiptError("source closure exceeds the entry-count bound")
        if total_bytes > MAX_SOURCE_TOTAL_BYTES:
            raise ReceiptError("source closure exceeds the total-byte bound")
        entries.append(
            {
                "repository": "cargo-config",
                "path": logical,
                **identity,
            }
        )
        cargo_configs.append(
            {"logical_path": logical, "physical_path": str(physical), **identity}
        )

    root_sha256, entries = canonical_root(entries)
    return {
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "root_hash_algorithm": (
            "sha256(domain='focr-source-input-root/v1\\0'; "
            "sorted(repository,path); repository\\0path\\0size\\0sha256\\n)"
        ),
        "root_sha256": root_sha256,
        "entry_count": len(entries),
        "repositories": repository_docs,
        "cargo_config_files": cargo_configs,
        "entries": entries,
    }


def load_json(path, maximum=MAX_SOURCE_MANIFEST_BYTES):
    metadata = Path(path).lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
        raise ReceiptError(f"JSON evidence is not a bounded regular file: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReceiptError(f"JSON evidence is not an object: {path}")
    return value


def ensure_no_symlink_components(path):
    requested = Path(os.path.abspath(path))
    component = Path(requested.anchor)
    for part in requested.parts[1:]:
        component /= part
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ReceiptError(f"evidence path contains a symlink component: {component}")
    return requested


def write_exclusive(path, value, maximum):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(data) > maximum:
        raise ReceiptError(f"JSON evidence exceeds the {maximum}-byte bound: {path}")
    path = ensure_no_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ReceiptError(f"refusing to overwrite evidence: {path}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def comparable_manifest(manifest):
    return {
        key: manifest.get(key)
        for key in (
            "schema",
            "root_hash_algorithm",
            "root_sha256",
            "entry_count",
            "repositories",
            "cargo_config_files",
            "entries",
        )
    }


def verify_snapshot(root, path):
    expected = load_json(path)
    if expected.get("schema") != MANIFEST_SCHEMA:
        raise ReceiptError("source manifest schema is not v1")
    actual = collect_manifest(root)
    if comparable_manifest(actual) != comparable_manifest(expected):
        raise ReceiptError("source inputs changed after the trusted snapshot")
    return expected


def workspace_entry(manifest, path):
    matches = [
        entry
        for entry in manifest.get("entries", [])
        if entry.get("repository") == "workspace" and entry.get("path") == path
    ]
    if len(matches) != 1:
        raise ReceiptError(f"source manifest must contain exactly one workspace/{path}")
    return {"sha256": matches[0]["sha256"], "size": matches[0]["size"]}


def config_get(root, key):
    result = run(
        ["cargo", "-Z", "unstable-options", "config", "get", key],
        cwd=root,
        check=False,
    )
    if result.returncode == 0:
        return {"status": "set", "value": result.stdout.strip()}
    if "is not set" in result.stderr:
        return {"status": "unset", "value": None}
    raise ReceiptError(f"cannot resolve effective Cargo config {key}: {result.stderr.strip()}")


def validate_receipt_shape(receipt, *, head, binary_path, binary_identity, manifest_identity):
    binary = receipt.get("binary") if isinstance(receipt, dict) else None
    source = receipt.get("source_manifest") if isinstance(receipt, dict) else None
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ReceiptError("build receipt schema is not v1")
    if receipt.get("git_head") != head:
        raise ReceiptError("build receipt git_head does not match current HEAD")
    if receipt.get("profile") != "release-perf":
        raise ReceiptError("build receipt profile is not release-perf")
    triple = receipt.get("target_triple")
    if not isinstance(triple, str) or not triple or any(character.isspace() for character in triple):
        raise ReceiptError("build receipt target triple is invalid")
    if not isinstance(binary, dict) or (
        os.path.realpath(str(binary.get("path", ""))) != os.path.realpath(binary_path)
        or binary.get("sha256") != binary_identity["sha256"]
        or binary.get("size") != binary_identity["size"]
    ):
        raise ReceiptError("build receipt does not bind FOCR_BIN")
    if not isinstance(source, dict) or (
        source.get("sha256") != manifest_identity["sha256"]
        or source.get("size") != manifest_identity["size"]
    ):
        raise ReceiptError("build receipt does not bind the source manifest file")
    build = receipt.get("build")
    if (
        not isinstance(build, dict)
        or build.get("runner") != "rch"
        or not isinstance(build.get("cargo_target_dir"), str)
        or not build["cargo_target_dir"]
    ):
        raise ReceiptError("build receipt runner/target-dir identity is incomplete")
    command = build.get("command")
    expected = [
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
        triple,
    ]
    if command != expected:
        raise ReceiptError("build receipt command is not the canonical RCH release-perf build")
    environment = receipt.get("build_environment")
    required_environment = (
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_RUSTFLAGS",
        "CARGO_HOME",
        "target_rustflags_env_name",
        "target_rustflags_env_value",
    )
    if not isinstance(environment, dict) or any(
        key not in environment or environment[key] is not None and not isinstance(environment[key], str)
        for key in required_environment
    ):
        raise ReceiptError("build receipt does not explicitly bind the Rust flag environment")
    expected_target_flags = "CARGO_TARGET_" + triple.upper().replace("-", "_") + "_RUSTFLAGS"
    rustc_overrides = environment.get("rustc_overrides")
    profile_overrides = environment.get("release_perf_profile_overrides")
    if (
        environment.get("target_rustflags_env_name") != expected_target_flags
        or not isinstance(rustc_overrides, dict)
        or set(rustc_overrides) != {"RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"}
        or any(value is not None and not isinstance(value, str) for value in rustc_overrides.values())
        or not isinstance(profile_overrides, dict)
        or any(
            not isinstance(key, str)
            or not key.startswith("CARGO_PROFILE_RELEASE_PERF_")
            or not isinstance(value, str)
            for key, value in profile_overrides.items()
        )
    ):
        raise ReceiptError("build receipt does not bind compiler/profile environment overrides")
    toolchain = receipt.get("toolchain")
    if not isinstance(toolchain, dict) or any(
        not isinstance(toolchain.get(key), str) or not toolchain[key].strip()
        for key in ("rustc_verbose_version", "cargo_version", "rch_version")
    ):
        raise ReceiptError("build receipt toolchain identity is incomplete")


def mode_snapshot(root, path):
    write_exclusive(path, collect_manifest(root), MAX_SOURCE_MANIFEST_BYTES)


def mode_validate_destinations(paths):
    seen = set()
    for path in paths:
        canonical = ensure_no_symlink_components(path).resolve(strict=False)
        if canonical in seen:
            raise ReceiptError(f"release evidence destinations alias each other: {canonical}")
        seen.add(canonical)


def mode_write_receipt(root, manifest_path, receipt_path, binary_path, target, target_dir):
    manifest = verify_snapshot(root, manifest_path)
    binary_identity = stable_file(binary_path, MAX_BINARY_BYTES)
    manifest_identity = stable_file(manifest_path, MAX_SOURCE_MANIFEST_BYTES)
    root_repository = [
        repository
        for repository in manifest["repositories"]
        if repository.get("id") == "workspace"
    ]
    if len(root_repository) != 1:
        raise ReceiptError("source manifest lacks the workspace repository")
    receipt = build_receipt_document(
        created_utc=datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        git_head=root_repository[0]["git_head"],
        target_triple=target,
        cargo_target_dir=target_dir,
        toolchain={
            "rustc_verbose_version": run(["rustc", "-vV"], cwd=root).stdout,
            "cargo_version": run(["cargo", "--version"], cwd=root).stdout.strip(),
            "rch_version": run(["rch", "--version"], cwd=root).stdout.strip(),
        },
        build_environment={
            "RUSTFLAGS": os.environ.get("RUSTFLAGS"),
            "CARGO_ENCODED_RUSTFLAGS": os.environ.get("CARGO_ENCODED_RUSTFLAGS"),
            "CARGO_BUILD_RUSTFLAGS": os.environ.get("CARGO_BUILD_RUSTFLAGS"),
            "CARGO_HOME": os.environ.get("CARGO_HOME"),
            "target_rustflags_env_name": (
                "CARGO_TARGET_" + target.upper().replace("-", "_") + "_RUSTFLAGS"
            ),
            "target_rustflags_env_value": os.environ.get(
                "CARGO_TARGET_" + target.upper().replace("-", "_") + "_RUSTFLAGS"
            ),
            "rustc_overrides": {
                name: os.environ.get(name)
                for name in ("RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")
            },
            "release_perf_profile_overrides": {
                name: value
                for name, value in sorted(os.environ.items())
                if name.startswith("CARGO_PROFILE_RELEASE_PERF_")
            },
            "cargo_config_build_rustflags": config_get(root, "build.rustflags"),
            "cargo_config_target": config_get(root, "target"),
        },
        source_manifest_path=manifest_path,
        source_manifest_identity=manifest_identity,
        source_manifest=manifest,
        binary_path=binary_path,
        binary_identity=binary_identity,
    )
    validate_receipt_shape(
        receipt,
        head=root_repository[0]["git_head"],
        binary_path=binary_path,
        binary_identity=binary_identity,
        manifest_identity=manifest_identity,
    )
    write_exclusive(receipt_path, receipt, MAX_BUILD_RECEIPT_BYTES)


def mode_verify_receipt(root, receipt_path, manifest_path, binary_path):
    manifest = verify_snapshot(root, manifest_path)
    receipt = load_json(receipt_path, maximum=MAX_BUILD_RECEIPT_BYTES)
    manifest_identity = stable_file(manifest_path, MAX_SOURCE_MANIFEST_BYTES)
    binary_identity = stable_file(binary_path, MAX_BINARY_BYTES)
    current_head = git_head(root)
    validate_receipt_shape(
        receipt,
        head=current_head,
        binary_path=binary_path,
        binary_identity=binary_identity,
        manifest_identity=manifest_identity,
    )
    source = receipt["source_manifest"]
    if (
        source.get("root_sha256") != manifest.get("root_sha256")
        or source.get("entry_count") != manifest.get("entry_count")
        or source.get("root_hash_algorithm") != manifest.get("root_hash_algorithm")
    ):
        raise ReceiptError("build receipt source root disagrees with its manifest")
    for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"):
        if receipt.get("inputs", {}).get(name) != workspace_entry(manifest, name):
            raise ReceiptError(f"build receipt {name} binding is invalid")
    host = None
    for line in run(["rustc", "-vV"], cwd=root).stdout.splitlines():
        if line.startswith("host: "):
            host = line.split(": ", 1)[1]
    if receipt.get("target_triple") != host:
        raise ReceiptError("build receipt target triple does not match the active Rust host")
    current_toolchain = {
        "rustc_verbose_version": run(["rustc", "-vV"], cwd=root).stdout,
        "cargo_version": run(["cargo", "--version"], cwd=root).stdout.strip(),
        "rch_version": run(["rch", "--version"], cwd=root).stdout.strip(),
    }
    if receipt.get("toolchain") != current_toolchain:
        raise ReceiptError("active Rust/RCH toolchain drifted from the build receipt")


def mode_self_test():
    entries = [
        {"repository": "workspace", "path": "src/lib.rs", "size": 3, "sha256": "a" * 64},
        {"repository": "local-dep/core", "path": "src/lib.rs", "size": 7, "sha256": "b" * 64},
    ]
    forward, ordered = canonical_root(entries)
    reverse, _ = canonical_root(list(reversed(entries)))
    if forward != reverse or ordered[0]["repository"] != "local-dep/core":
        raise ReceiptError("canonical source root is not order-independent")
    changed = json.loads(json.dumps(entries))
    changed[0]["size"] += 1
    mutated, _ = canonical_root(changed)
    if mutated == forward:
        raise ReceiptError("canonical source root ignored an input mutation")
    try:
        canonical_root([{"repository": "bad\0id", "path": "x", "size": 0, "sha256": "c" * 64}])
    except ReceiptError:
        pass
    else:
        raise ReceiptError("canonical source root accepted an ambiguous separator")
    for invalid_entries, label in (
        (entries + [dict(entries[0])], "duplicate identity"),
        ([{"repository": "workspace", "path": "huge", "size": MAX_SOURCE_FILE_BYTES + 1, "sha256": "c" * 64}], "oversized input"),
    ):
        try:
            canonical_root(invalid_entries)
        except ReceiptError:
            pass
        else:
            raise ReceiptError(f"canonical source root accepted {label}")
    binary_identity = {"sha256": "d" * 64, "size": 11}
    manifest_identity = {"sha256": "e" * 64, "size": 19}
    head = "f" * 40
    binary_path = "/evidence/release-perf/focr"
    triple = "aarch64-apple-darwin"
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "git_head": head,
        "profile": "release-perf",
        "target_triple": triple,
        "build": {
            "runner": "rch",
            "cargo_target_dir": "/scratch/target",
            "command": [
                "rch", "exec", "--", "cargo", "build", "--locked",
                "--profile", "release-perf", "--bin", "focr", "--target", triple,
            ],
        },
        "toolchain": {
            "rustc_verbose_version": "rustc test",
            "cargo_version": "cargo test",
            "rch_version": "rch test",
        },
        "build_environment": {
            "RUSTFLAGS": None,
            "CARGO_ENCODED_RUSTFLAGS": "-C\x1ftarget-cpu=native",
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
        },
        "source_manifest": {"sha256": manifest_identity["sha256"], "size": 19},
        "binary": {"path": binary_path, **binary_identity},
    }
    validate_receipt_shape(
        receipt,
        head=head,
        binary_path=binary_path,
        binary_identity=binary_identity,
        manifest_identity=manifest_identity,
    )
    for field, replacement in (
        ("git_head", "0" * 40),
        ("profile", "release"),
        ("target_triple", ""),
    ):
        invalid = json.loads(json.dumps(receipt))
        invalid[field] = replacement
        try:
            validate_receipt_shape(
                invalid,
                head=head,
                binary_path=binary_path,
                binary_identity=binary_identity,
                manifest_identity=manifest_identity,
            )
        except ReceiptError:
            pass
        else:
            raise ReceiptError(f"build receipt self-test accepted invalid {field}")
    mutations = []
    wrong_binary = json.loads(json.dumps(receipt))
    wrong_binary["binary"]["sha256"] = "0" * 64
    mutations.append((wrong_binary, "binary hash"))
    wrong_manifest = json.loads(json.dumps(receipt))
    wrong_manifest["source_manifest"]["sha256"] = "1" * 64
    mutations.append((wrong_manifest, "source-manifest hash"))
    wrong_runner = json.loads(json.dumps(receipt))
    wrong_runner["build"]["runner"] = "cargo"
    mutations.append((wrong_runner, "non-RCH runner"))
    missing_flags = json.loads(json.dumps(receipt))
    missing_flags["build_environment"].pop("RUSTFLAGS")
    mutations.append((missing_flags, "missing Rust flags"))
    for invalid, label in mutations:
        try:
            validate_receipt_shape(
                invalid,
                head=head,
                binary_path=binary_path,
                binary_identity=binary_identity,
                manifest_identity=manifest_identity,
            )
        except ReceiptError:
            pass
        else:
            raise ReceiptError(f"build receipt self-test accepted {label}")
    try:
        mode_validate_destinations(["/evidence/subject", "/evidence/./subject"])
    except ReceiptError:
        pass
    else:
        raise ReceiptError("destination validator accepted aliased output paths")
    print(json.dumps({"check": "gauntlet-build-receipt-self-test", "result": "pass"}))


root = Path(sys.argv[1]).resolve()
arguments = sys.argv[2:]
try:
    if arguments == ["self-test"]:
        mode_self_test()
    elif len(arguments) == 2 and arguments[0] == "snapshot":
        mode_snapshot(root, arguments[1])
    elif len(arguments) == 2 and arguments[0] == "verify-snapshot":
        verify_snapshot(root, arguments[1])
    elif len(arguments) >= 2 and arguments[0] == "validate-destinations":
        mode_validate_destinations(arguments[1:])
    elif len(arguments) == 6 and arguments[0] == "write-receipt":
        mode_write_receipt(root, *arguments[1:])
    elif len(arguments) == 4 and arguments[0] == "verify-receipt":
        mode_verify_receipt(root, *arguments[1:])
    else:
        raise ReceiptError("invalid internal build receipt command")
except (ReceiptError, OSError, ValueError, TypeError, KeyError) as error:
    print(f"build receipt validation failed: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
}

SUBJECT_VERIFIED=0

reject_ambient_precision_switches() {
  local name
  for name in FOCR_DECODE_INT8 FOCR_INT8_ATTN FOCR_INT8_LMHEAD \
              FOCR_ATTN_GEMM FOCR_INT8_KV FOCR_SPEC_DECODE FOCR_DECODE_STATELESS; do
    if printenv "$name" >/dev/null 2>&1; then
      die "ambient $name is set; conservative release evidence requires a clean precision environment"
    fi
  done
}

verify_subject_model() {
  (( SUBJECT_VERIFIED == 0 )) || return 0
  [[ -n "$SUBJECT_MODEL" ]] \
    || die "FOCR_GAUNTLET_FOCR_MODEL must name the conservative subject .focrq"
  [[ -f "$SUBJECT_MODEL" && "$SUBJECT_MODEL" == *.focrq ]] \
    || die "native subject must be a regular .focrq file: $SUBJECT_MODEL"
  local have_sha have_size
  have_sha="$(shasum -a 256 "$SUBJECT_MODEL" | awk '{print $1}')"
  have_size="$(wc -c <"$SUBJECT_MODEL" | tr -d '[:space:]')"
  [[ "$have_sha" == "$SUBJECT_MODEL_SHA256" ]] \
    || die "subject sha256 drifted: $have_sha != $SUBJECT_MODEL_SHA256"
  [[ "$have_size" == "$SUBJECT_MODEL_SIZE" ]] \
    || die "subject size drifted: $have_size != $SUBJECT_MODEL_SIZE"
  SUBJECT_VERIFIED=1
  note "subject .focrq sha256/size verified ($SUBJECT_QUANT_RECIPE)"
}

conservative_env() {
  env -u FOCR_ATTN_GEMM -u FOCR_INT8_KV -u FOCR_SPEC_DECODE \
      -u FOCR_DECODE_STATELESS \
      FOCR_DECODE_INT8=0 FOCR_INT8_ATTN=0 FOCR_INT8_LMHEAD=0 \
      FOCR_RSWA_PARALLEL_ATTN=1 FOCR_MODEL_PATH="$SUBJECT_MODEL" \
      "$@"
}

# Validate that the scorer inputs are the immutable outputs of the measured
# invocations, then print "<hypothesis-sha>\t<reference-sha>". The focr side
# preserves every stdout and is re-hashed here. The reference harness writes the
# final measured text and records every warm/measured text hash; a singleton hash
# plus the explicit deterministic flag proves that final file represents them all.
verify_timed_text_pair() {
  python3 - "$@" <<'PY'
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_TEXT_BYTES = 64 * 1024 * 1024


class BindingError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(focr, reference, read_focr, reference_bytes, page, runs):
    if focr.get("source") != "focr" or focr.get("runs") != runs:
        raise BindingError("focr stages identity/run count is invalid")
    if os.path.basename(str(focr.get("page") or "")) != page:
        raise BindingError("focr stages page does not match the requested fixture")
    if focr.get("stdout_identical_across_runs") is not True:
        raise BindingError("focr stdout is missing or nondeterministic across timed runs")
    raw = focr.get("raw_timing")
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != "focr-gauntlet-raw-timing/v1"
        or raw.get("source") != "focr"
        or raw.get("measured_runs") != runs
    ):
        raise BindingError("focr raw timing receipt is missing or has the wrong run count")
    records = raw.get("records")
    if not isinstance(records, list) or len(records) != runs:
        raise BindingError("focr raw timing records are missing or incomplete")

    hypothesis_hashes = set()
    for index, record in enumerate(records, 1):
        run_id = f"run_{index:03d}"
        binding = (
            record.get("raw_files", {}).get("stdout")
            if isinstance(record, dict)
            else None
        )
        expected_name = f"{run_id}.stdout"
        if (
            record.get("run_id") != run_id
            or not isinstance(binding, dict)
            or binding.get("path") != expected_name
        ):
            raise BindingError(f"focr {run_id} stdout binding is malformed")
        data = read_focr(expected_name)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BindingError(f"focr {run_id} stdout is not UTF-8: {error}") from error
        actual = digest(data)
        if binding.get("sha256") != actual:
            raise BindingError(f"focr {run_id} stdout hash does not match raw timing")
        hypothesis_hashes.add(actual)
    if len(hypothesis_hashes) != 1:
        raise BindingError("focr stdout bytes are nondeterministic across timed runs")

    if reference.get("source") != "reference" or reference.get("runs") != runs:
        raise BindingError("reference stages identity/run count is invalid")
    if os.path.basename(str(reference.get("page") or "")) != page:
        raise BindingError("reference stages page does not match the requested fixture")
    if reference.get("text_identical_across_runs") is not True:
        raise BindingError("reference text is missing or nondeterministic across timed runs")
    if not reference_bytes:
        raise BindingError("reference text is empty")
    try:
        reference_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BindingError(f"reference text is not UTF-8: {error}") from error
    reference_hash = digest(reference_bytes)
    if reference.get("text_sha256") != reference_hash:
        raise BindingError("reference text hash does not match its timed-run receipt")
    return next(iter(hypothesis_hashes)), reference_hash


def bounded_bytes(path, maximum):
    size = os.stat(path).st_size
    if size > maximum:
        raise BindingError(f"{path} exceeds the {maximum}-byte evidence limit")
    return Path(path).read_bytes()


def self_test():
    hypothesis = b"timed hypothesis\n"
    reference_text = b"timed <|det|>box<|/det|> reference\n"
    records = []
    raw = {}
    for index in range(1, 4):
        name = f"run_{index:03d}.stdout"
        raw[name] = hypothesis
        records.append({
            "run_id": f"run_{index:03d}",
            "raw_files": {"stdout": {"path": name, "sha256": digest(hypothesis)}},
        })
    focr = {
        "source": "focr",
        "runs": 3,
        "page": "/fixtures/page_0014.png",
        "stdout_identical_across_runs": True,
        "raw_timing": {
            "schema": "focr-gauntlet-raw-timing/v1",
            "source": "focr",
            "measured_runs": 3,
            "records": records,
        },
    }
    reference = {
        "source": "reference",
        "runs": 3,
        "page": "/fixtures/page_0014.png",
        "text_identical_across_runs": True,
        "text_sha256": digest(reference_text),
    }
    got = validate(focr, reference, raw.__getitem__, reference_text, "page_0014.png", 3)
    if got != (digest(hypothesis), digest(reference_text)):
        raise BindingError("valid timed bindings did not round-trip")

    def refuses(changed_focr, changed_reference, changed_raw, changed_text):
        try:
            validate(
                changed_focr,
                changed_reference,
                changed_raw.__getitem__,
                changed_text,
                "page_0014.png",
                3,
            )
        except (BindingError, KeyError):
            return
        raise BindingError("invalid timed binding was accepted")

    nondeterministic = copy.deepcopy(reference)
    nondeterministic["text_identical_across_runs"] = False
    refuses(focr, nondeterministic, raw, reference_text)
    refuses(focr, reference, raw, b"")
    missing = dict(raw)
    missing.pop("run_002.stdout")
    refuses(focr, reference, missing, reference_text)
    drifted = dict(raw)
    drifted["run_003.stdout"] = b"different\n"
    refuses(focr, reference, drifted, reference_text)
    print(json.dumps({"check": "runbook-timed-text-binding", "result": "pass"}))


try:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        if len(sys.argv) != 7:
            raise BindingError("internal validator requires six arguments")
        focr_path, reference_path, raw_dir, text_path, page, runs_text = sys.argv[1:]
        focr = json.loads(bounded_bytes(focr_path, MAX_JSON_BYTES))
        reference = json.loads(bounded_bytes(reference_path, MAX_JSON_BYTES))
        raw_root = Path(raw_dir)

        def read_focr(name):
            return bounded_bytes(raw_root / name, MAX_TEXT_BYTES)

        hypothesis_hash, reference_hash = validate(
            focr,
            reference,
            read_focr,
            bounded_bytes(text_path, MAX_TEXT_BYTES),
            page,
            int(runs_text),
        )
        print(f"{hypothesis_hash}\t{reference_hash}")
except (BindingError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
    print(f"timed text validation failed: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
}

step_self_test() {
  local ref_paths
  ref_paths="$(ref_env sh -c 'printf "%s\n%s\n" "$HF_HOME" "$TMPDIR"')"
  [[ "$ref_paths" == "$REFERENCE_HF_HOME"$'\n'"$REFERENCE_TMPDIR" ]] \
    || die "reference cache/temp environment override wiring failed"
  python3 scripts/baseline/compare_ocr.py --self-test >/dev/null
  verify_timed_text_pair --self-test >/dev/null
  build_receipt_tool self-test >/dev/null
  note "build/CER receipt and timed-text binding self-tests: pass"
}

active_target_triple() {
  rustc -vV | awk -F': ' '$1 == "host" {print $2}'
}

verify_build_receipt() {
  [[ -f "$SOURCE_MANIFEST" ]] \
    || die "trusted source manifest missing: $SOURCE_MANIFEST (run 'build' first)"
  [[ -f "$BUILD_RECEIPT" ]] \
    || die "trusted build receipt missing: $BUILD_RECEIPT (run 'build' first)"
  [[ -x "$FOCR_BIN" ]] \
    || die "receipted focr binary missing or not executable: $FOCR_BIN"
  build_receipt_tool verify-receipt "$BUILD_RECEIPT" "$SOURCE_MANIFEST" "$FOCR_BIN" \
    || die "trusted build receipt does not match the current clean source tree and FOCR_BIN"
  note "trusted release-perf build receipt verified"
}

# ── build — clean-source RCH build + immutable producer receipt ─────────────
step_build() {
  note "build: clean local source closure + RCH release-perf receipt"
  build_receipt_tool validate-destinations \
    "$OUT" "$SOURCE_MANIFEST" "$BUILD_RECEIPT" "$FOCR_BIN" \
    || die "release evidence destination contains a symlink component"
  if [[ -e "$OUT" && ! -d "$OUT" ]]; then
    die "OUT_DIR exists but is not a directory: $OUT"
  fi
  if [[ -d "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "release build evidence requires a fresh OUT_DIR: $OUT"
  fi
  [[ ! -e "$SOURCE_MANIFEST" ]] \
    || die "source manifest already exists; use a fresh OUT_DIR: $SOURCE_MANIFEST"
  [[ ! -e "$BUILD_RECEIPT" ]] \
    || die "build receipt already exists; use a fresh OUT_DIR: $BUILD_RECEIPT"
  [[ ! -e "$FOCR_BIN" ]] \
    || die "receipted focr binary already exists; use a fresh OUT_DIR: $FOCR_BIN"

  # Snapshot first: a dirty current tree must fail before a target directory or
  # release artifact is created. The post-build check rejects source/config
  # changes that occurred while RCH was compiling.
  build_receipt_tool snapshot "$SOURCE_MANIFEST" \
    || die "release evidence build refused the dirty or unreadable source closure"

  local target scratch target_dir built pre_sha pre_size post_sha post_size
  target="$(active_target_triple)"
  [[ -n "$target" ]] || die "cannot resolve rustc host target triple"
  scratch="${FOCR_GAUNTLET_SCRATCH_ROOT:-/Volumes/USBNVME16TB/temp_agent_space}"
  [[ -d "$scratch" ]] || die "gauntlet scratch root is missing: $scratch"
  target_dir="$(mktemp -d "$scratch/focr-gauntlet-target.XXXXXX")"
  note "RCH build target: $target_dir ($target)"
  CARGO_TARGET_DIR="$target_dir" \
    rch exec -- cargo build --locked --profile release-perf --bin focr --target "$target"
  built="$target_dir/$target/release-perf/focr"
  [[ -x "$built" ]] || die "RCH build did not produce the expected focr binary: $built"

  build_receipt_tool verify-snapshot "$SOURCE_MANIFEST" \
    || die "source closure changed during the RCH build"
  pre_sha="$(shasum -a 256 "$built" | awk '{print $1}')"
  pre_size="$(wc -c <"$built" | tr -d '[:space:]')"
  mkdir -p "$(dirname "$FOCR_BIN")"
  cp -p -- "$built" "$FOCR_BIN"
  post_sha="$(shasum -a 256 "$built" | awk '{print $1}')"
  post_size="$(wc -c <"$built" | tr -d '[:space:]')"
  [[ "$pre_sha" == "$post_sha" && "$pre_size" == "$post_size" ]] \
    || die "RCH output changed while copying the receipted binary"
  [[ "$(shasum -a 256 "$FOCR_BIN" | awk '{print $1}')" == "$pre_sha" ]] \
    || die "evidence binary copy hash does not match the RCH output"

  build_receipt_tool write-receipt \
    "$SOURCE_MANIFEST" "$BUILD_RECEIPT" "$FOCR_BIN" "$target" "$target_dir" \
    || die "could not write the trusted build receipt"
  verify_build_receipt
}

# ── preflight — untimed gates; nothing here touches a stopwatch ─────────────
step_preflight() {
  note "preflight: quiet-host + fixtures + pins + harness self-tests"
  reject_ambient_precision_switches
  verify_build_receipt
  local load
  load="$(sysctl -n vm.loadavg | awk '{print $2}')"
  if awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{exit !(l>=m)}'; then
    [[ "${FORCE:-0}" == "1" ]] \
      || die "1-min loadavg $load >= $LOAD_MAX — host is NOT quiet (FORCE=1 to override; a contended host poisons cv%)"
    note "WARNING: loadavg $load >= $LOAD_MAX but FORCE=1 — timings from this host are suspect"
  else
    note "loadavg $load OK (< $LOAD_MAX)"
  fi

  [[ -d "$REFERENCE_MODEL_DIR" ]] || die "reference model dir missing: $REFERENCE_MODEL_DIR"
  [[ -x "$VENV_PY" ]] || die "reference venv python missing: $VENV_PY"
  mkdir -p "$REFERENCE_HF_HOME" "$REFERENCE_TMPDIR" \
    || die "cannot create reference cache/temp directories"
  [[ -d "$REFERENCE_HF_HOME" && -w "$REFERENCE_HF_HOME" ]] \
    || die "reference Hugging Face cache is not a writable directory: $REFERENCE_HF_HOME"
  [[ -d "$REFERENCE_TMPDIR" && -w "$REFERENCE_TMPDIR" ]] \
    || die "reference TMPDIR is not a writable directory: $REFERENCE_TMPDIR"
  verify_subject_model

  # Fixture hashes must match the embedded truth-pack values (moved page = STOP).
  local page have want
  for page in $PAGES; do
    [[ -f "$PAGES_DIR/$page" ]] || die "page fixture missing: $PAGES_DIR/$page"
    want="$(page_sha256 "$page")"
    [[ -n "$want" ]] || die "no embedded sha256 for $page — add it to page_sha256() first"
    have="$(shasum -a 256 "$PAGES_DIR/$page" | awk '{print $1}')"
    [[ "$have" == "$want" ]] || die "$page sha256 drifted: $have != $want"
    note "$page sha256 verified"
  done
  if [[ "${VERIFY_SHARD:-0}" == "1" ]]; then
    have="$(shasum -a 256 "$REFERENCE_MODEL_DIR/model-00001-of-000001.safetensors" | awk '{print $1}')"
    [[ "$have" == "$SHARD_SHA256" ]] || die "weights shard sha256 drifted: $have"
    note "weights shard sha256 verified"
  fi

  # Truth-pack runtime pins (the reference harness re-verifies fail-closed).
  "$VENV_PY" - <<PY
import sys, torch, transformers
ok = (torch.__version__.split("+")[0] == "$PINNED_TORCH"
      and transformers.__version__ == "$PINNED_TRANSFORMERS")
print(f"runbook: venv torch={torch.__version__} transformers={transformers.__version__}",
      file=sys.stderr)
sys.exit(0 if ok else 1)
PY

  # Harness self-tests (all untimed, no model needed).
  python3 scripts/gauntlet_timing.py --self-test >/dev/null
  python3 scripts/gauntlet_reference.py --self-test >/dev/null
  python3 scripts/gauntlet_ref_unlimited.py --self-test >/dev/null
  python3 scripts/gauntlet_roofline.py --self-test >/dev/null
  python3 scripts/gauntlet_row.py --self-test >/dev/null
  python3 scripts/baseline/compare_ocr.py --self-test >/dev/null
  verify_timed_text_pair --self-test >/dev/null
  build_receipt_tool self-test >/dev/null
  bash scripts/gauntlet_focr.sh --self-test >/dev/null
  note "harness self-tests: all pass"

  # Dispatched SIMD tier -> the ledger arch/cpu_features cell (recorded once).
  mkdir -p "$OUT"
  "$FOCR_BIN" robot selftest >"$ARCH_JSON"
  python3 - "$ARCH_JSON" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
assert doc["all_ok"] is True, "focr robot selftest FAILED — kernels unproven"
print(f"runbook: arch/cpu_features = {doc['selected_feature']} (selftest 24/24)",
      file=sys.stderr)
PY
}

arch_features() {
  [[ -f "$ARCH_JSON" ]] || die "run preflight first ($ARCH_JSON missing)"
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['selected_feature'])" "$ARCH_JSON"
}

# ── focr side — TIMED (best-of-5 warm, N=8, conservative mixed decode) ─────
step_focr() {
  local page s
  reject_ambient_precision_switches
  verify_build_receipt
  verify_subject_model
  for page in $PAGES; do
    s="$(stem "$page")"
    note "focr side: $page (runs=$RUNS warmup=$WARMUP threads=$THREADS $SUBJECT_DECODE_MODE)"
    conservative_env bash scripts/gauntlet_focr.sh \
        --binary "$FOCR_BIN" \
        --page "$PAGES_DIR/$page" \
        --model "$SUBJECT_MODEL" \
        --model-sha256 "$SUBJECT_MODEL_SHA256" \
        --model-size "$SUBJECT_MODEL_SIZE" \
        --quant-recipe "$SUBJECT_QUANT_RECIPE" \
        --build-receipt "$BUILD_RECEIPT" \
        --precision "$SUBJECT_PRECISION" \
        --runs "$RUNS" --warmup "$WARMUP" --threads "$THREADS" \
        --out-dir "$OUT/focr_$s"
  done
}

# ── reference side — TIMED (same N, pinned stack, instrumented stages) ──────
step_reference() {
  local page s
  for page in $PAGES; do
    s="$(stem "$page")"
    note "reference side: $page (runs=$RUNS warmup=$WARMUP threads=$THREADS bf16)"
    [[ ! -e "$OUT/ref_$s/ref_stages.json" && ! -e "$OUT/ref_$s/text" ]] \
      || die "reference evidence already exists for $page; use a fresh OUT_DIR"
    mkdir -p "$OUT/ref_$s"
    ref_env "$VENV_PY" scripts/gauntlet_reference.py \
        --stage all \
        --page "$PAGES_DIR/$page" \
        --model-dir "$REFERENCE_MODEL_DIR" \
        --backend hf --precision bf16 \
        --max-length 8192 --text-dir "$OUT/ref_$s/text" \
        --entry gauntlet_ref_unlimited:run_stage \
        --setup gauntlet_ref_unlimited:setup \
        --runs "$RUNS" --warmup "$WARMUP" --threads "$THREADS" \
        --out "$OUT/ref_$s/ref_stages.json"
  done
}

# ── roofline — untimed derivation from the focr measurement ─────────────────
step_roofline() {
  local page s
  for page in $PAGES; do
    s="$(stem "$page")"
    python3 scripts/gauntlet_roofline.py \
      --arch unlimited-ocr --precision "$SUBJECT_DECODE_MODE" --profile m4 \
      --stages-json "$OUT/focr_$s/focr_stages.json" \
      --out "$OUT/roofline_$s.json"
  done
}

# ── correctness proof — CER of focr text vs the reference text from the SAME
#    timed runs. The HF reference emits <|det|>…<|/det|> grounding spans that
#    focr's plain-text mode never produces (smoke-proven 2026-07-02; the raw
#    diff is ~0.61 CER purely from the markers) — strip them from the ref side
#    before scoring, and say so in the proof string. ──────────────────────────
step_cer() {
  local page s focr_text reference_text receipt bindings hypothesis_sha reference_sha
  for page in $PAGES; do
    s="$(stem "$page")"
    focr_text="$OUT/focr_$s/raw/run_001.stdout"
    reference_text="$OUT/ref_$s/text/$s.md"
    receipt="$OUT/cer_$s/cer.json"
    [[ -f "$focr_text" ]] || die "focr timed stdout missing for $page — run 'focr' first"
    [[ -f "$reference_text" ]] || die "reference timed text missing for $page — run 'reference' first"
    [[ ! -e "$receipt" ]] || die "CER receipt already exists for $page; use a fresh OUT_DIR"
    bindings="$(verify_timed_text_pair \
      "$OUT/focr_$s/focr_stages.json" \
      "$OUT/ref_$s/ref_stages.json" \
      "$OUT/focr_$s/raw" "$reference_text" "$page" "$RUNS")"
    IFS=$'\t' read -r hypothesis_sha reference_sha <<<"$bindings"
    [[ -n "$hypothesis_sha" && -n "$reference_sha" ]] \
      || die "timed text validator returned incomplete bindings for $page"
    mkdir -p "$OUT/cer_$s"
    python3 scripts/baseline/compare_ocr.py \
      --ref "$reference_text" --hyp "$focr_text" \
      --reference-transform strip-unlimited-det-spans-v1 \
      --expected-reference-sha256 "$reference_sha" \
      --expected-hypothesis-sha256 "$hypothesis_sha" \
      --require-complete --json "$receipt"
  done
}

# ── row — merge, bundle, validate (shadow check_ledgers), optionally apply ──
step_row() {
  local page s claim fixture proof cer arch
  [[ "${APPLY:-0}" != "1" ]] \
    || die "APPLY=1 is not certifiable: commit ledger/config first, then capture from that source HEAD"
  verify_build_receipt
  verify_subject_model
  arch="$(arch_features)"
  for page in $PAGES; do
    s="$(stem "$page")"
    claim="G2-unlimited-mixed-ffn-int8-${s#page_}-$STAMP"
    fixture="page=$page sha256=$(page_sha256 "$page"); subject=$(basename "$SUBJECT_MODEL") sha256=$SUBJECT_MODEL_SHA256 size=$SUBJECT_MODEL_SIZE quant_recipe=$SUBJECT_QUANT_RECIPE; reference=model-00001-of-000001.safetensors sha256=$SHARD_SHA256"
    cer="$(python3 -c "import json,sys; print(f\"{float(json.load(open(sys.argv[1]))['aggregate']['cer_norm']):.6f}\")" "$OUT/cer_$s/cer.json")"
    proof="CER_norm=$cer $SUBJECT_PRECISION exact timed text vs pinned HF bf16 exact timed text on $page; reference transform=strip-unlimited-det-spans-v1 (scripts/baseline/compare_ocr.py; $OUT/cer_$s/cer.json)"
    note "row: $page claim_id=$claim"
    python3 scripts/gauntlet_row.py \
      --focr-stages "$OUT/focr_$s/focr_stages.json" \
      --ref-stages "$OUT/ref_$s/ref_stages.json" \
      --roofline "$OUT/roofline_$s.json" \
      --claim-id "$claim" \
      --model-commit "$MODEL_COMMIT" \
      --fixture-hash "$fixture" \
      --arch-features "$arch" \
      --correctness-proof "$proof" \
      --notes "quiet-host runbook (scripts/gauntlet_runbook.sh); $SUBJECT_DECODE_MODE; quant_recipe=$SUBJECT_QUANT_RECIPE; best-of-$RUNS warm, N=$THREADS both sides${FORCE:+; FORCE=1 loadavg-gate bypassed}"
  done
  python3 scripts/check_ledgers.py >/dev/null && note "check_ledgers: pass"
}

case "${1:-}" in
  self-test|--self-test) step_self_test ;;
  build) step_build ;;
  preflight) step_preflight ;;
  focr) step_focr ;;
  reference) step_reference ;;
  roofline) step_roofline ;;
  cer) step_cer ;;
  row) step_row ;;
  all)
    step_build
    step_preflight
    step_focr
    step_reference
    step_roofline
    step_cer
    step_row
    ;;
  *) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
