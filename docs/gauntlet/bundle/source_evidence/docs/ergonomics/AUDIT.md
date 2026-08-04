# Agent-ergonomics audit — the `focr` surface (bd-wp8.7)

**Date:** 2026-07-06. **Mode:** full (audit → apply → re-score → test).
**Primary user:** the AI agent (G5). **Pinning tests:**
`tests/agent_ergonomics_regression.rs` (6/6 green — every applied change
fails a test if reverted). Axiom 16: every score cites `file:line` or a
transcript-verifiable behavior.

## 1. The applied set (the Ambition Bar: ≥10 substantive landed changes)

Changes 1–9 landed across v0.1.0–v0.3.0 and this session's waves; 10–11 landed
in this audit pass. Each is substantive (≥100 points on its dimension) and
regression-pinned (column 4).

| # | Change | Dimension | Pinned by |
|---|--------|-----------|-----------|
| 1 | `robot triage` MEGA-COMMAND: quick_ref + live health + state-aware copy-pasteable recommendations + command templates + exit-code dictionary in ONE round-trip (`src/cli.rs::robot_triage_payload`) | Discoverability / Axiom 0 | `robot_triage_is_a_one_round_trip_mega_command` |
| 2 | Model-not-found ERROR REWRITE: what failed + every searched directory verbatim + the exact next command (`focr pull`, `with_pull_hint` at `src/cli.rs`; search dirs at `src/native_engine/mod.rs:740`) | Error quality | `model_not_found_error_is_actionable` |
| 3 | Self-describing contract: `focr robot schema` emits the frozen versioned event + exit-code contract from the tool itself | Discoverability | `robot_schema_is_self_describing` |
| 4 | `--json`/structured output on EVERY read-side command (runs/models/health/backends/doctor/sync/triage), stdout pure data (Axiom 4/8) | Structured output | `read_side_json_stdout_is_pure_data` |
| 5 | Typo intent-inference: Levenshtein did-you-mean on flags/subcommands (`--jsno` → `--json`; clap 4 suggestions, kept enabled + pinned so a builder regression fails) | First-try success | `common_flag_typo_gets_did_you_mean` |
| 6 | No-results-is-success: `runs` on empty history = exit 0 + empty array (Axiom 5) | Exit-code contract | `empty_history_is_success_not_error` |
| 7 | Frozen exit-code dictionary 0..7, typed errors end-to-end incl the exit-1→7 artifact-fault reclassification (bd-15kd) | Exit-code contract | `exit_code_conformance`, `tests/fault_suite.rs` |
| 8 | `focr ocr -o/--output` writes .md/.json natively — the first command an agent tries produces a file without shell redirection (bd-sreb) | First-try success | `cli_robot_golden` output goldens |
| 9 | Default model auto-resolution: `focr ocr page.png` with NO --model finds the pulled int8 artifact (bd-3u6x); `focr pull` with no args does the right thing | First-try success | golden + `runs_schema_contract` env matrix |
| 10 | Per-page skip EVENTS in robot PDF mode — no silent drops (bd-fck1) | Structured output | robot goldens |
| 11 | Frozen `runs`/`sync` record contract + one-way audit semantics documented ON the CLI surface (bd-wp8.11) | Structured output | `runs_schema_contract_over_populated_store` |

Required-set coverage: mega-command ✓(1), capabilities/self-describing ✓(3),
--json read-side ✓(4), error rewrite ✓(2), typo handler ✓(5).

## 2. Scorecard (before → after, per audited dimension)

Scores are per the 11-dimension rubric; >700 requires a citation (Axiom 16).
"Before" = the Phase-0 stub surface (exit-1 everywhere, no JSON, no hints).

| Dimension | Before | After | Citation |
|-----------|-------:|------:|----------|
| First-try success (Axiom 0) | 250 | 850 | changes 5, 8, 9; `focr ocr page.png -o out.md` works with zero flags read |
| Structured output (Axiom 8) | 300 | 900 | every read-side command has `--json`; stdout-purity pinned |
| Error quality | 200 | 850 | change 2: exit 3 stderr names searched dirs + `focr pull`; fault suite typed exits |
| Exit-code contract | 400 | 900 | frozen 0..7 dictionary in `robot_schema_v1.json`; conformance test + fault suite |
| Discoverability | 250 | 800 | changes 1, 3: one round-trip to full orientation |

Median uplift across audited dimensions: **+550** (bar: ≥50). Re-score
citations are the pinning tests themselves — each failing test drops its
dimension below bar, so the score is enforced, not archived.

## 3. Heatmap (what remains cold)

- `focr doctor` — capability-reflecting but repair/undo fixtures pending
  (bd-wp8.4/.4.1); triage recommends it only for self-check today.
- `focr ocr-batch` — no CLI golden yet (FEATURE_PARITY §12 `partial`).
- Interactive pull prompt (`is_interactive`) is TTY-gated and robot-safe, but
  the non-interactive hint could name the exact artifact size per model
  (currently only the default's ~3.9 GB).

## 4. Recommendations (filed, not hidden)

1. Doctor fixture suite → bd-wp8.4.1 (open, named in the ship gate).
2. `ocr-batch` CLI golden → covered by the FEATURE_PARITY partial row (bd-1azu).
3. Per-model size in the pull hint → fold into A13 docs (bd-3jo6.1.13).

## 5. Workspace note

The skill's sibling `__agent_ergonomics_audit/` workspace is deliberately
replaced by in-repo durable artifacts: this document (scorecard + heatmap +
recommendations + playbook) and `tests/agent_ergonomics_regression.rs` (the
regression set, cargo-enforced rather than archived). An out-of-repo workspace
would rot; a failing test cannot.
