#!/usr/bin/env python3
"""bd-2pgf: validate tests/fixtures/MANIFEST.toml against the fixture tree.

Bidirectional: every top-level entry under tests/fixtures/ must be declared
in the manifest, and every declared entry must exist on disk. Kinds are
closed (`committed` | `regenerated-committed`); `regenerated-committed`
entries must name an existing generator script. Wired into scripts/check.sh.

Exit 0 = consistent; 1 = a violation (each printed as a JSON line);
`--self-test` proves the checker fails on an undeclared/missing entry.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
MANIFEST = os.path.join(FIXTURES, "MANIFEST.toml")
KINDS = {"committed", "regenerated-committed"}


def check(fixtures_dir: str, manifest_path: str) -> list[str]:
    failures: list[str] = []
    with open(manifest_path, "rb") as f:
        manifest = tomllib.load(f)
    entries: dict = manifest.get("entries", {})

    on_disk = {
        name
        for name in os.listdir(fixtures_dir)
        # `.actual` files are the golden review loop's OWN diff aids — a
        # failing golden writes `<fixture>.actual` beside the frozen file for
        # review (GOLDEN.md §4). They are ephemeral local debris by design,
        # never committed, so the manifest contract does not govern them.
        if not name.startswith(".") and not name.endswith(".actual")
    }
    declared = set(entries)

    for name in sorted(on_disk - declared):
        failures.append(f"undeclared fixture entry: tests/fixtures/{name} (add it to MANIFEST.toml)")
    for name in sorted(declared - on_disk):
        failures.append(f"declared but missing: tests/fixtures/{name}")
    for name, spec in sorted(entries.items()):
        kind = spec.get("kind")
        if kind not in KINDS:
            failures.append(f"{name}: kind {kind!r} not in {sorted(KINDS)}")
        if kind == "regenerated-committed":
            script = spec.get("script")
            if not script or not os.path.isfile(os.path.join(ROOT, script)):
                failures.append(f"{name}: generator script {script!r} missing")
    return failures


def self_test() -> int:
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "declared_dir"))
        os.makedirs(os.path.join(tmp, "undeclared_dir"))
        m = os.path.join(tmp, "MANIFEST.toml")
        with open(m, "w", encoding="utf-8") as f:
            f.write(
                '[entries]\n'
                '"declared_dir" = { kind = "committed" }\n'
                '"ghost_dir" = { kind = "committed" }\n'
                '"MANIFEST.toml" = { kind = "committed" }\n'
            )
        failures = check(tmp, m)
        has_undeclared = any("undeclared" in f and "undeclared_dir" in f for f in failures)
        has_missing = any("missing: " in f and "ghost_dir" in f for f in failures)
        for name, cond in [("catches-undeclared", has_undeclared), ("catches-missing", has_missing)]:
            print(json.dumps({"check": name, "result": "pass" if cond else "fail"}))
            ok = ok and cond
    # The REAL manifest must be clean.
    real = check(FIXTURES, MANIFEST)
    print(json.dumps({"check": "real-manifest-clean", "result": "pass" if not real else "fail",
                      "failures": real}))
    ok = ok and not real
    print(json.dumps({"check": "fixture-manifest-self-test", "result": "pass" if ok else "fail"}))
    return 0 if ok else 1


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    failures = check(FIXTURES, MANIFEST)
    for f in failures:
        print(json.dumps({"check": "fixture-manifest", "result": "fail", "detail": f}))
    if failures:
        return 1
    print(json.dumps({"check": "fixture-manifest", "result": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
