# PROVENANCE — CLI golden artifacts (`tests/fixtures/golden/`)

These are the **frozen, known-good** CLI / robot surface outputs that
`tests/cli_robot_golden.rs` diffs against forever (golden-artifact pattern B/D +
the Agent-Ergonomics contract test). A mismatch is a deliberate contract change
that a human reviews and re-blesses with `UPDATE_GOLDENS=1` (see "Regeneration").

## What produced them

| Field | Value |
|---|---|
| Binary | `focr` (built from this repo via `env!("CARGO_BIN_EXE_focr")`) |
| Surface | `src/cli.rs` (clap-derive `Cli`), `src/robot.rs` (`robot_schema`, `ROBOT_SCHEMA_VERSION`, `EVENT_KINDS`), `src/error.rs` (stable exit codes) |
| Package version | `Cargo.toml [package] version` (scrubbed to `[version]` in goldens — see below) |
| clap | `4.5` (`features = ["derive", "env"]`) |
| serde_json | `1` (`preserve_order` ON transitively via the dep graph; goldens are compared **after canonicalization** so the suite does not depend on that transitive feature staying on) |
| Capture method | `std::process::Command::new(env!("CARGO_BIN_EXE_focr"))` — the STABLE committed surface (AGENTS.md "Agent Ergonomics", plan §7.2/§7.4) |
| Generated/last-reviewed | 2026-06-25 (initial freeze, derived from the committed `src/cli.rs` + `src/robot.rs` source) |

## Provenance of the schema fixture (the contract)

`tests/fixtures/robot_schema_v1.json` is the **frozen robot-schema contract**
(`bd-zc1o`). It is the **canonical** (sorted-key, 2-space-pretty) form of
`robot::robot_schema()`'s output. The contract test parses `focr robot schema`'s
NDJSON line and asserts it canonicalizes byte-for-byte to this fixture, plus that
`ROBOT_SCHEMA_VERSION` and **every** `EVENT_KIND` is present. Bumping
`ROBOT_SCHEMA_VERSION` or changing the event set MUST update this fixture (a new
`robot_schema_v<N>.json` on a major bump) through the reviewed regeneration path.

## Canonicalization / scrubbing (why these goldens are cross-platform stable)

The goldens are compared **after** a scrubber (`scrub()` in the test) that, per
`docs/conformance/GOLDEN.md` §2E, normalizes non-determinism so one golden passes
on all 5 release targets:

- line endings `\r\n` / `\r` → `\n`;
- the package **version** string (e.g. `0.0.0`) → `[version]` (so a `Cargo.toml`
  version bump does not flap `--help` / `--version`);
- `logical_cpus` value in `robot backends` → `[cpus]` (host core count);
- the SIMD-tier `available`/`selected`/`status` host-specific fields are matched
  structurally, not byte-exact, until `FOCR_FORCE_ARCH` pins them (plan §6.2);
- absolute paths → `[path]`.

`*.golden` files here are the committed baseline; `*.actual` are the transient
observed outputs written **only on mismatch** for the human diff and are
gitignored (`docs/conformance/GOLDEN.md` §5).

## Regeneration (human-in-the-loop — `docs/conformance/GOLDEN.md` §4)

```bash
# 1. A change alters a surface. The golden test FAILS, writing *.actual + the diff.
cargo test --test cli_robot_golden            # -> FAIL, shows before/after

# 2. A HUMAN inspects each *.actual vs its *.golden. Intended and correct?
git diff --no-index tests/fixtures/golden/<name>.golden tests/fixtures/golden/<name>.actual

# 3. ONLY a deliberate, reviewed update regenerates the goldens:
UPDATE_GOLDENS=1 cargo test --test cli_robot_golden   # rewrites the committed *.golden

# 4. Commit the regenerated goldens WITH the git diff as the review artifact, and
#    re-stamp the "Generated/last-reviewed" date in this file.
```

CI NEVER sets `UPDATE_GOLDENS`; a mismatch in CI is a build failure, not an
auto-fix. The suite contains a guard test asserting `UPDATE_GOLDENS` is unset in
the live env when it runs in compare mode is the only sanctioned writer.
