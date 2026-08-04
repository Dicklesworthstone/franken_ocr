# RELEASE CERTIFICATION TEMPLATE — strict-conformant-release.v1 (bd-wp8.9)

The contract enforced by `scripts/gauntlet_cert.py --bundle` and
`--finalize-bundle`. A release is CERTIFIED only when every predicate below
holds at finalization time. `--bundle` only writes an honestly red provisional
bundle and exits 1; no manual JSON edit or detached signature copy can promote
it.

## Required-pass constants

| constant | value | meaning |
|---|---|---|
| `CERTIFICATION_MIN_VERIFICATION_PCT` | 100.0 | every readiness cell green — no partial credit |
| `CERTIFICATION_REQUIRED_SUITE_PASS_RATE_PCT` | 100.0 | the full gate (`scripts/check.sh`) exits 0 |
| `CERTIFICATION_MAX_HIGH_SEVERITY_COUNTEREXAMPLES` | 0 | no open high-severity finding anywhere in the ledgers |
| `CERTIFICATION_MAX_EVIDENCE_AGE_HOURS` | 24 | every core evidence artifact regenerated within a day of the bundle |

## Evidence-bundle classes

1. **Parity receipts** — `tests/fixtures/ladder_scorecard/scorecard_armed.json`
   (L0–L5), the armed multi-page rung logs, `focr robot selftest` per-model
   verdicts.
2. **Convergence record** — `docs/gauntlet/ROUNDS.jsonl` (≥10 rounds, last 2
   with <3 new genuine findings each; every finding a real find-fix cycle).
3. **Statistical state** — `docs/gauntlet/EPROCESS_STATE.json` (never-reset
   Ville e-processes; no invariant rejected), the conformal ratchet state in
   `docs/gauntlet/RELEASE_SCORECARD.json`.
4. **Ledgers** — `docs/DISCREPANCIES.md` (accepted divergences: measured
   impact + kill-switch + review date), `docs/NEGATIVE_EVIDENCE.md` (reverted
   levers + do-not-retry predicates), `docs/PERF_LEDGER.md` (fairness-pinned
   measured rows) — all lint-enforced by `scripts/check_ledgers.py`.
5. **Perf baseline** — `benches/.bench-history/baseline.json` (the frozen
   guardrail baseline; ratchet-only advances).

## Gate / ratchet spec

* The ship gate is `scripts/gauntlet_cert.py --release-readiness`: every cell
  reads its evidence artifact live; ANY red exits 1.
* The conformal ratchet (`--from-parity`) uses the Jeffreys×Hoeffding LOWER
  bound, truncated to 6 dp; the persisted baseline may only move up
  (`--ratchet`), never down, and `MIN_CALIBRATION_N = 20`.
* Bench regressions gate at +10% vs the frozen baseline with cv% > 5 runs
  ineligible as evidence (`scripts/bench_guardrail.py`).

## Certification flow

```
bash scripts/check.sh
# Commit and push the final source/evidence inputs. Work from clean main.
python3 scripts/gauntlet_cert.py --bundle .gauntlet-output/bundle

# Produce one fresh exact-HEAD run of each canonical workflow:
#   CI (both gate jobs), dist (all six portable jobs), Model Parity, and
#   Performance Gauntlet. Download all ten workflow artifacts locally.

python3 scripts/gauntlet_cert.py \
  --finalize-bundle .gauntlet-output/bundle \
  --workflow-evidence /evidence/ci-macos/.gauntlet-output/ci-gate-macos-15-workflow-evidence.json \
  --workflow-evidence /evidence/ci-linux/.gauntlet-output/ci-gate-ubuntu-latest-workflow-evidence.json \
  --workflow-evidence /evidence/dist-apple-arm/.gauntlet-output/dist-aarch64-apple-darwin-neon-sdot-i8mm-workflow-evidence.json \
  --workflow-evidence /evidence/dist-apple-x86/.gauntlet-output/dist-x86_64-apple-darwin-baseline-workflow-evidence.json \
  --workflow-evidence /evidence/dist-linux-arm/.gauntlet-output/dist-aarch64-linux-baseline-workflow-evidence.json \
  --workflow-evidence /evidence/dist-linux-x86/.gauntlet-output/dist-x86_64-linux-baseline-runtime-dispatch-workflow-evidence.json \
  --workflow-evidence /evidence/dist-win-x86/.gauntlet-output/dist-x86_64-pc-windows-msvc-baseline-workflow-evidence.json \
  --workflow-evidence /evidence/dist-win-arm/.gauntlet-output/dist-aarch64-pc-windows-msvc-baseline-workflow-evidence.json \
  --workflow-evidence /evidence/model-parity/.gauntlet-output/model-parity-workflow-evidence.json \
  --workflow-evidence /evidence/performance/.gauntlet-output/performance-workflow-evidence.json \
  --trusted-signer producer:PRODUCER_ID:FULL_OPENPGP_FINGERPRINT \
  --trusted-signer independent-reviewer:REVIEWER_ID:FULL_OPENPGP_FINGERPRINT \
  --trusted-signer release-authorizer:AUTHORIZER_ID:FULL_OPENPGP_FINGERPRINT

python3 scripts/gauntlet_cert.py --release-readiness \
  --readiness-out .gauntlet-output/release_readiness.json
```

The manifest paths above are examples; use the paths emitted by the downloaded
artifacts. Every manifest is hash-checked locally and replayed against the live
GitHub run. All files must come from exactly one run per workflow, all runs must
name the same final HEAD, and no run may be skipped or stale. The three signer
identities and full fingerprints must already be active in
`docs/gauntlet/TRUSTED_SIGNERS.json`, their public keys must be in the tracked
keyring, and their existing secret keys must be available to `gpg`. The
finalizer never generates a key or invents a receipt.

The provisional bundle must also contain fresh production-generated
`audit_receipts/{security,correctness,concurrency,numerics,release}_audit_receipt.json`
files and every raw tool output they name. The finalizer replays their exact
commands, hashes, findings, HEAD, and timestamps; a generic green CI log is not
a substitute for an omitted named tool. There is currently no production audit
receipt producer in this repository, so current `main` remains blocked until
one runs and packages those receipts. Do not hand-author them.

The six dist jobs are four portable runtime-dispatch binaries plus native
Windows x86-64 and ARM64. Linux is linked for glibc 2.17 and independently
audited with `readelf`; Windows runs `install.ps1` offline against the exact
staged asset, including a failed-replacement preservation check. Tag builds
also fail before compilation unless `GITHUB_REF_NAME` is exactly
`v$(Cargo.toml package.version)` and the commit is reachable from `origin/main`.

The finalizer derives the strict documents from the downloaded physical
evidence, creates signatures with the three existing trusted keys, verifies the
candidate entirely in memory, and writes `certified:true` only after
`certificate_bundle_verdict` succeeds. The final readiness command then checks
the persisted bytes without dirtying the worktree.

The signed record is `.gauntlet-output/bundle/release_certificate.json` (git
head + describe + constants + verdicts); the human-readable twin is
`FINAL_GAUNTLET_REPORT.md` beside it. The tracked `docs/gauntlet/bundle/` tree is
a historical snapshot and is not a valid output directory for current
certification.
