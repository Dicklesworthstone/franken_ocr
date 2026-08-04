# Beads and BV Reality for franken_ocr

## Table of Contents

- [Purpose](#purpose)
- [Refresh Commands](#refresh-commands)
- [Current Signals to Recheck](#current-signals-to-recheck)
- [Closed Evidence Patterns](#closed-evidence-patterns)
- [Open Caution Patterns](#open-caution-patterns)
- [How to Use This Evidence](#how-to-use-this-evidence)
- [When to Update This File](#when-to-update-this-file)

## Purpose

This file captures the kind of project-state evidence the skill should look for.
It is not a substitute for live `br` and `bv` output. The franken_ocr tracker
moves quickly, so re-run commands before making current claims.

## Refresh Commands

```bash
cd /Users/jemanuel/projects/franken_ocr
br list --json | jq '{total: length, by_status: group_by(.status)|map({status:.[0].status,count:length})}'
bv --robot-triage
br show <id> --json
```

Use only JSON or robot-safe output. Never run bare `bv`.

## Current Signals to Recheck

Signals observed while creating this skill:

- Many issues remain open, and many are blocked by parity/model gates.
- `bv --robot-triage` reported no dependency cycles at the time checked.
- Actionable issues existed, but that does not mean they are safe to claim
  without reading the issue and current worktree.
- Some high-level OCR pipeline and parity tasks were still open.

Because these are live project facts, verify them again before final answers.

## Closed Evidence Patterns

Closed Beads showed these implemented/proven areas at the time checked:

- `focr convert --quant int8` had been proven on the Baidu shard and produced a
  smaller `.focrq` artifact.
- `ocr-batch` had been wired through a continuous-batch scheduler.
- Batch spine work had byte-identical corpus-output evidence.
- Several speculative/batch/perf subtasks had close reasons with proof notes.

Use these as examples of the evidence level expected: exact command, artifact,
hash or output, and a Beads close reason.

## Open Caution Patterns

Open or cautionary areas observed while creating this skill included:

- full native pipeline assembly,
- prompt ids / `images_seq_mask` source-truth completion,
- stale `NotImplemented` helper stubs,
- L5 end-to-end parity gates,
- combined lossy-stack CER gates,
- speculative decode bit-identical gates,
- Windows packaging/pull gaps.

Do not claim any of these are solved without current source and tracker proof.

## How to Use This Evidence

When answering "can focr do X?":

1. Check source for the command/API.
2. Check tests for exercised behavior.
3. Search Beads for the feature and read close reasons.
4. If CASS has hits, use them for historical context only.
5. State uncertainty explicitly when a capability is scaffolded or phase-gated.

Command snippets:

```bash
br search "ocr-batch" --json
br search "int8 convert" --json
br search "L5 parity" --json
br search "Windows pull" --json
```

If `br search` output shape differs, adapt with `jq` after inspecting it. Do
not silently drop failed tracker queries.

## When to Update This File

Update this reference when:

- a major phase gate closes,
- int4 or additional lossy paths become validated defaults,
- robot schema version changes,
- artifact distribution changes,
- Windows/macOS/Linux support claims change,
- source API signatures in `src/lib.rs` change.

Every update should cite source or Beads evidence in the commit message or
adjacent research notes.
