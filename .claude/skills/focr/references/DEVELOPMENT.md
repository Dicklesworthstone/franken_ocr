# franken_ocr Development Workflow

## Table of Contents

- [Read First](#read-first)
- [Repo Discipline](#repo-discipline)
- [Tracker Workflow](#tracker-workflow)
- [Source Truth Probes](#source-truth-probes)
- [Quality Gates](#quality-gates)
- [Kernel and Model Gates](#kernel-and-model-gates)
- [Robot and CLI Changes](#robot-and-cli-changes)
- [Commit Discipline](#commit-discipline)

## Read First

In `/Users/jemanuel/projects/franken_ocr`, read:

1. `AGENTS.md`
2. `README.md`
3. `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md` for model/kernel/architecture work
4. Relevant source and tests

The plan is the source of project intent. The source is the source of current
truth.

## Repo Discipline

Hard local rules:

- Do not delete files without explicit written permission.
- Do not run destructive git/filesystem commands.
- Do not stash, revert, or overwrite other agents' dirty work.
- Do not use script-based mass edits.
- Use manual, scoped edits.
- Keep `.beads/` sync commits separate when appropriate.

If the worktree is dirty, classify changes before editing. Other agents may be
working concurrently.

## Tracker Workflow

Use `br` and `bv` in agent-safe modes:

```bash
br ready --json
bv --robot-triage
br show <id> --json
br update <id> --status in_progress
```

After tracker changes:

```bash
br sync --flush-only
git add .beads/
```

Never run bare `bv`; it launches a blocking TUI. Never assume `br` runs git.

## Source Truth Probes

Useful fast probes:

```bash
rg -n "enum Commands|enum RobotCommands" src/cli.rs
rg -n "pub struct OcrEngine|impl OcrEngine|pub fn recognize" src/lib.rs
rg -n "enum FocrError|exit_code" src/error.rs
rg -n "schema_version|run_start|run_error" src/robot.rs tests
rg -n "FOCR_[A-Z0-9_]+" src README.md docs
```

Use these before updating the skill, README examples, or downstream
integrations.

## Quality Gates

For substantive code changes:

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
ubs $(git diff --name-only)
```

The repo also has `scripts/check.sh` to run the main cargo sequence. If heavy
builds are a concern, use RCH explicitly when appropriate:

```bash
rch exec -- cargo check --all-targets
```

If a gate cannot run, record the exact blocker and do not claim it passed.

## Kernel and Model Gates

Non-negotiables:

- Correctness outranks speed.
- No kernel ships against unresolved `[OPEN]`/OQ dependencies.
- Keep quantization within validated surfaces unless evidence expands it.
- Record accepted numeric divergence in `docs/DISCREPANCIES.md`.
- Record rejected optimizations in `docs/NEGATIVE_EVIDENCE.md`.
- Do not hand-roll wide SIMD over scalar loops without proof; prior evidence
  says this regressed badly.
- One live forward at a time; kernel internals own core fanout.

For lossy/adaptive work, define state space, actions, loss matrix,
calibration/posterior terms, deterministic fallback, and evidence ledger.

## Robot and CLI Changes

When adding or changing CLI/robot behavior:

1. Update `src/cli.rs` and tests together.
2. Keep both binaries as thin one-line shims to `cli_main()`.
3. Preserve stable exit-code semantics.
4. Keep robot stdout pure JSON/NDJSON.
5. Update `robot schema` and golden tests before telling downstream agents the
   contract changed.

Run at least:

```bash
cargo run --bin focr -- --help
cargo run --bin focr -- robot schema | jq .
cargo test cli_robot
```

Use exact test names from current source.

## Commit Discipline

Before finishing:

1. File or update Beads for remaining work.
2. Run quality gates or document exact blockers.
3. Close/update issues as appropriate.
4. `br sync --flush-only` after tracker mutation.
5. Commit only intended paths.
6. Push.

In shared dirty checkouts, use path-scoped `git add` and inspect `git diff
--cached` before committing.
