# Research Notes for the focr Skill

## Table of Contents

- [Sources Inspected](#sources-inspected)
- [Skill Design Decisions](#skill-design-decisions)
- [Ground Truth Findings](#ground-truth-findings)
- [CASS Findings](#cass-findings)
- [Known Caveats](#known-caveats)
- [Update Discipline](#update-discipline)

## Sources Inspected

Skill models:

- `ntm`
- `cass`
- `beads-br`
- `beads-bv`
- `sc`
- `sw`
- `operationalizing-expertise`

Project sources:

- `/Users/jemanuel/projects/franken_ocr/AGENTS.md`
- `/Users/jemanuel/projects/franken_ocr/README.md`
- `/Users/jemanuel/projects/franken_ocr/Cargo.toml`
- `/Users/jemanuel/projects/franken_ocr/src/cli.rs`
- `/Users/jemanuel/projects/franken_ocr/src/lib.rs`
- `/Users/jemanuel/projects/franken_ocr/src/error.rs`
- `/Users/jemanuel/projects/franken_ocr/src/robot.rs`
- `/Users/jemanuel/projects/franken_ocr/src/dist.rs`
- `/Users/jemanuel/projects/franken_ocr/docs/focrq-format.md`
- `/Users/jemanuel/projects/franken_ocr/tests/cli_robot_golden.rs`

Live project tools:

- `br list --json`
- `bv --robot-triage`
- targeted `br show` for closed and open issues
- `cass status --json`
- targeted `cass search`

## Skill Design Decisions

From `sw` and `sc`:

- Keep `SKILL.md` as a concise entrypoint.
- Put depth in first-level `references/`.
- Include validation.
- Make trigger language explicit in frontmatter.

From `cass`:

- Include health/staleness handling instead of blindly requiring a rebuild.
- Treat historical search as useful but secondary to live source.

From `beads-br` and `beads-bv`:

- Use JSON/robot-safe command forms.
- Never run bare `bv`.
- Explicitly sync Beads after tracker mutation.

From `ntm`:

- Prefer action cards and recovery operators that an agent can execute under
  pressure.

From `operationalizing-expertise`:

- Encode expert moves as named operators.
- Require evidence ledgers for adaptive/lossy decisions.
- Separate deterministic fallback from experimental policy.

## Ground Truth Findings

Current source showed:

- two binaries, `focr` and `franken_ocr`, both thin shims over `cli_main()`,
- synchronous/blocking `OcrEngine`,
- model cache and runtime ownership inside the library,
- CLI surfaces for `ocr`, `ocr-batch`, `convert`, `pull`, and `robot`,
- robot schema version 1 event names,
- stable user-facing exit codes,
- `.focrq` format version 1 and `FOCRQ\0` magic,
- int8 conversion implemented and int4 still phase-gated.

Beads showed enough closed evidence to document int8 conversion and batch work,
but enough open parity/pipeline issues to keep claims conservative.

## CASS Findings

CASS was available but did not produce useful `focr` or `OcrEngine` session
hits during skill creation. The skill therefore treats CASS as a fallback
history source and uses source, tests, README, and Beads as authority.

## Known Caveats

- The project is moving quickly; re-run source probes before strong claims.
- Installed or target-dir binaries can be stale.
- Some commands may exist as scaffolds before full implementation.
- Windows support claims need separation between OCR runtime and model pull.
- Experimental env vars are not production advice.

## Update Discipline

When updating this skill:

1. Re-read current `src/cli.rs`, `src/lib.rs`, `src/error.rs`, and `src/robot.rs`.
2. Re-run `br`/`bv` robot-safe probes.
3. Check whether CASS now has useful focr sessions.
4. Update references first, then tighten `SKILL.md`.
5. Run the local validator.
