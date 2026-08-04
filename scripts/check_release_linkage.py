#!/usr/bin/env python3
"""Check release binaries do not link or embed Python/torch artifacts.

This guard belongs to bd-re8.3: the oracle bridge is test-only, so the shipped
`focr` and `franken_ocr` binaries must not depend on Python, torch, libtorch,
or related runtime libraries. In the normal dev gate this skips successfully
when release binaries have not been built yet. Release CI should run it with
`--require-binary` after `cargo build --release`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess  # nosec B404 - invokes fixed platform dependency listers for local release binaries.
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RELEASE_NAMES = ("focr", "franken_ocr")
FORBIDDEN_DYNAMIC_TOKENS = (
    "libpython",
    "python",
    "torch",
    "pytorch",
    "libtorch",
    "torch_cpu",
    "torch_python",
    "c10",
)
C10_ABI_TOKEN = "c" + "10"
# The BYTE scan looks for LINKAGE indicators only. Bare "python"/"torch" are
# legitimately embedded in a correct binary: OneChart's actual task prompt IS
# "Convert the key information of the chart to a python dict:" (plus the CLI
# help describing it), and panic-location paths of the ft-kernel-cpu path dep
# embed ".../frankentorch/...". Linkage-relevant byte patterns (libpython,
# libtorch, torch_cpu, the CPython C-ABI) still fail the scan; the dynamic-
# deps check (otool -L / ldd) keeps the bare tokens for real link entries.
FORBIDDEN_BYTE_TOKENS = (
    "libpython",
    "pytorch",
    "libtorch",
    "torch_cpu",
    "torch_python",
    "_PyObject",
    "Py_Initialize",
)


def emit(check: str, ok: bool, **fields: object) -> None:
    payload = {"check": check, "result": "pass" if ok else "fail", **fields}
    print(json.dumps(payload, sort_keys=True))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def unique(items: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for item in items:
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(item)
    return out


def configured_target_root() -> Path:
    value = os.environ.get("CARGO_TARGET_DIR")
    if not value:
        return ROOT / "target"
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def target_roots() -> list[Path]:
    return unique([configured_target_root(), ROOT / "target"])


def candidate_binaries() -> list[Path]:
    names = set(RELEASE_NAMES) | {f"{name}.exe" for name in RELEASE_NAMES}
    direct = [target / "release" / name for target in target_roots() for name in names]
    triplet_builds = [
        path
        for target in target_roots()
        for path in (target.glob("*/release/*") if target.is_dir() else [])
    ]
    candidates = [path for path in direct + triplet_builds if path.name in names and path.is_file()]
    return unique(sorted(candidates))


def find_forbidden_text(text: str) -> list[str]:
    lower = text.lower()
    return [token for token in FORBIDDEN_DYNAMIC_TOKENS if token in lower]


def find_forbidden_bytes(chunks: Iterable[bytes]) -> list[str]:
    tokens = {token: token.encode("ascii") for token in FORBIDDEN_BYTE_TOKENS}
    max_token = max(len(token) for token in tokens.values())
    found: set[str] = set()
    tail = b""
    for chunk in chunks:
        data = tail + chunk.lower()
        for token, needle in tokens.items():
            if needle in data:
                found.add(token)
        tail = data[-(max_token - 1) :]
    return [token for token in FORBIDDEN_BYTE_TOKENS if token in found]


def scan_binary_bytes(path: Path, chunk_size: int = 1 << 20) -> list[str]:
    def chunks() -> Iterable[bytes]:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    return
                yield chunk

    return find_forbidden_bytes(chunks())


def dependency_command() -> list[str] | None:
    system = platform.system().lower()
    if system == "darwin" and shutil.which("otool"):
        return ["otool", "-L"]
    if system == "linux":
        if shutil.which("readelf"):
            return ["readelf", "-d"]
        if shutil.which("ldd"):
            return ["ldd"]
    if system == "windows":
        if shutil.which("dumpbin"):
            return ["dumpbin", "/DEPENDENTS"]
        if shutil.which("objdump"):
            return ["objdump", "-p"]
    return None


def dependency_output(path: Path) -> tuple[str, int, str] | None:
    command = dependency_command()
    if command is None:
        return None
    proc = subprocess.run(  # nosec B603 - command is a fixed dependency lister; path is a discovered binary.
        [*command, str(path)],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return " ".join(command), proc.returncode, proc.stdout + proc.stderr


def check_binary(path: Path, failures: list[str]) -> None:
    size = path.stat().st_size
    emit("release-linkage-binary-present", True, file=rel(path), bytes=size)
    deps = dependency_output(path)
    if deps is None:
        emit("release-linkage-dependency-tool", True, file=rel(path), skipped=True, reason="no dependency listing tool")
    else:
        command, returncode, output = deps
        found = find_forbidden_text(output)
        ok = not found
        emit(
            "release-linkage-dynamic-deps",
            ok,
            file=rel(path),
            command=command,
            returncode=returncode,
            forbidden_tokens=found,
        )
        if not ok:
            failures.append(f"{path}: dynamic dependencies mention {', '.join(found)}")

    found_bytes = scan_binary_bytes(path)
    ok_bytes = not found_bytes
    emit("release-linkage-byte-scan", ok_bytes, file=rel(path), forbidden_tokens=found_bytes)
    if not ok_bytes:
        failures.append(f"{path}: binary bytes mention {', '.join(found_bytes)}")


def run_linkage_checks(require_binary: bool) -> int:
    failures: list[str] = []
    binaries = candidate_binaries()
    if not binaries:
        ok = not require_binary
        emit(
            "release-linkage-binary-present",
            ok,
            skipped=ok,
            reason="no release binary found",
            expected=[rel(target / "release" / name) for target in target_roots() for name in RELEASE_NAMES],
        )
        if not ok:
            failures.append("no release binary found")
    for binary in binaries:
        check_binary(binary, failures)
    emit("release-linkage-summary", not failures, binaries_checked=len(binaries), failed=failures)
    return 0 if not failures else 1


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, **fields: object) -> None:
        emit(name, cond, **fields)
        if not cond:
            failures.append(name)

    check("release-linkage-selftest-clean-text", find_forbidden_text("/usr/lib/libSystem.B.dylib") == [])
    check("release-linkage-selftest-dirty-text", "libpython" in find_forbidden_text("/opt/lib/libpython3.13.dylib"))
    check("release-linkage-selftest-clean-bytes", find_forbidden_bytes([b"ELF", b"libSystem"]) == [])
    check(
        "release-linkage-selftest-dirty-bytes",
        "torch" in find_forbidden_bytes([b"prefix-libt", b"orch_cpu.so-suffix"]),
    )
    old_target_dir = os.environ.get("CARGO_TARGET_DIR")
    try:
        os.environ["CARGO_TARGET_DIR"] = "custom-target"
        check(
            "release-linkage-selftest-relative-cargo-target-dir",
            target_roots()[0] == ROOT / "custom-target",
            target_roots=[rel(path) for path in target_roots()],
        )
        absolute_target = (ROOT / "selftest-absolute-target").resolve()
        os.environ["CARGO_TARGET_DIR"] = str(absolute_target)
        check(
            "release-linkage-selftest-absolute-cargo-target-dir",
            target_roots()[0] == absolute_target,
            target_roots=[rel(path) for path in target_roots()],
        )
    finally:
        if old_target_dir is None:
            os.environ.pop("CARGO_TARGET_DIR", None)
        else:
            os.environ["CARGO_TARGET_DIR"] = old_target_dir
    emit("release-linkage-selftest-summary", not failures, failed=failures)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-binary", action="store_true", help="fail if no release binary is present")
    parser.add_argument("--self-test", action="store_true", help="run in-memory helper checks")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    return run_linkage_checks(require_binary=args.require_binary)


if __name__ == "__main__":
    raise SystemExit(main())
