#!/usr/bin/env python3
"""Validate the focr skill folder."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REQUIRED_REFS = [
    "CLI.md",
    "LIBRARY.md",
    "ROBOT.md",
    "ARTIFACTS-AND-ENV.md",
    "DEVELOPMENT.md",
    "BEADS-REALITY.md",
    "TROUBLESHOOTING.md",
    "OPERATORS.md",
    "RESEARCH.md",
]

FORBIDDEN_PATTERNS = [
    (r"focr convert\s+--source", "convert uses positional input, not --source"),
    (r"focr convert[^\n]*--tokenizer", "convert has no --tokenizer flag"),
    (r"focr convert[^\n]*--out\b", "convert uses -o/--output, not --out"),
    (r"focr pull[^\n]*--cache-dir", "pull has no --cache-dir flag"),
    (r"focr robot run[^\n]*--json", "robot run is already NDJSON and has no --json flag"),
    (r"ocr[^\n]*--timeout", "ocr has no --timeout flag; use stage-budget env vars"),
]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"missing {path}")


def validate_skill(root: Path) -> None:
    skill = root / "SKILL.md"
    body = read(skill)

    if not body.startswith("---\n"):
        fail("SKILL.md must start with YAML frontmatter")

    if "\ndescription: >-\n" not in body:
        fail("frontmatter must use folded description")

    frontmatter = body.split("---", 2)[1]
    if "Use when" not in frontmatter:
        fail("description must include an explicit Use when trigger")
    desc_lines = []
    capture = False
    for line in frontmatter.splitlines():
        if line == "description: >-":
            capture = True
            continue
        if capture and line.startswith("  "):
            desc_lines.append(line.strip())
            continue
        if capture:
            break
    description = " ".join(desc_lines)
    if len(description) > 200:
        fail(f"description is too long for reliable triggering: {len(description)} chars")

    if "focr" not in frontmatter or "OcrEngine" not in body:
        fail("skill must mention both focr and OcrEngine")

    line_count = len(body.splitlines())
    if line_count > 220:
        fail(f"SKILL.md is too long for an entrypoint: {line_count} lines")

    refs_dir = root / "references"
    all_text = body
    for name in REQUIRED_REFS:
        ref = refs_dir / name
        text = read(ref)
        all_text += "\n" + text
        rel = f"references/{name}"
        if rel not in body:
            fail(f"SKILL.md does not link {rel}")
        if len(text.splitlines()) > 100 and "Table of Contents" not in text:
            fail(f"{name} is over 100 lines and needs a Table of Contents")

        nested_ref_links = re.findall(r"\]\((?:references/)?[A-Za-z0-9-]+\.md(?:#[^)]+)?\)", text)
        if nested_ref_links:
            fail(f"{name} links to another reference file: {nested_ref_links[0]}")

    for pattern, message in FORBIDDEN_PATTERNS:
        if re.search(pattern, all_text):
            fail(f"stale command form found: {message}")

    script = root / "scripts" / "validate.py"
    script_text = read(script)
    if not script_text.startswith("#!/usr/bin/env python3"):
        fail("validator must keep its shebang")


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1]
    validate_skill(root)
    print("OK: focr skill validates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
