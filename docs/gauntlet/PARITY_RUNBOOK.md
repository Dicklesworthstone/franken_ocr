# PARITY RUNBOOK — reproducing the franken_ocr parity verification (bd-wp8.9)

How a future agent/maintainer re-verifies every parity claim in the
certification bundle, from a fresh checkout. Every step is a committed script
with its own receipt; nothing here depends on session state.

## 0. Prerequisites

* The real Unlimited-OCR checkpoint (see `docs/FEATURE_PARITY.md` §weights +
  the baseline workspace notes): point `FOCR_GAUNTLET_REFERENCE_MODEL_DIR` at
  the raw BF16 safetensors directory for the Torch reference leg. A native
  all-high-precision diagnostic must additionally set
  `FOCR_DECODE_STATELESS=1`; raw storage alone still builds the conservative
  cached int8 FFN/expert route. Do
  **not** use the committed default `focr pull`: its 3,914,093,440-byte artifact
  declares the incompatible legacy recipe
  `unlimited-ocr-full-int8-attn-int8-lmhead-int8-v1` and the runtime rejects it
  before downloading any artifact part.
* A locally converted candidate may be tested only when its recipe is exactly
  `unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1`. Record its sha256 in the
  receipt, pass it as `FOCR_GAUNTLET_FOCR_MODEL`, and keep it distinct from the
  BF16 oracle. The current candidate emits EOS on `page0590` and passes the
  complete local 20-page CER budget, but it is **not certified**: the compatible
  artifact is unpublished and its clean-tree bundle, signatures, platform
  distribution, and remaining readiness cells are still pending.
* The torch oracle env for reference regeneration (pinned
  `torch==2.10.0 transformers==4.57.1`, CPU) — only needed to REGENERATE
  frozen fixtures or produce a new citable performance capture. Verification
  of an already assembled evidence bundle needs neither that torch/transformers
  environment nor the external weight directory.
* A citable Unlimited-OCR reference capture requires the exact truth-pack
  12-file model set: `config.json`, `configuration_deepseek_v2.py`,
  `conversation.py`, `deepencoder.py`,
  `model-00001-of-000001.safetensors`, `model.safetensors.index.json`,
  `modeling_deepseekv2.py`, `modeling_unlimitedocr.py`,
  `processor_config.json`, `special_tokens_map.json`, `tokenizer.json`, and
  `tokenizer_config.json`. The harness verifies every exact size/SHA-256 plus
  the 2,710-weight, one-shard index contract; a partial or merely same-named
  directory refuses.

## 1. The L0–L5 ladder receipt (the core parity proof)

```
bash scripts/ladder_scorecard.sh          # writes tests/fixtures/ladder_scorecard/scorecard_armed.json
```

All-green armed receipt = L0 preprocess / L1 vision / L2 hidden / L3 logits /
L4 token-exact / L5 CER within the documented budgets, against the frozen
torch-oracle fixtures. `scorecard_armed.json` carries the per-rung numbers the
readiness gate reads.

## 2. Multi-page cross-page parity (infer_multi)

```
FOCR_MODEL_PATH=<model dir> FOCR_CORPUS_DIR=<pages dir> \
  cargo test --release --test parity_ladder l5_multi -- --nocapture
```

Two armed rungs: the 2-page leg (CER ≤ 0.25, plate byte-exact) and the
10-page long-horizon leg (CER ≤ 0.50, subject capped at 7600 tokens — a true
prefix). Oracle fixtures: `tests/fixtures/multi_page/` (regenerate via
`scripts/baseline/run_baidu_reference_multi.py`). Accepted divergence class:
DISC-004.

## 3. Kernel parity on the host silicon

```
focr robot selftest                       # 44/44 per-model int8-kernel parity vs the scalar oracle
```

Machine-readable per-model verdicts (`models[]`), including each registered
decoder's worst-case-K overflow row. `FOCR_FORCE_ARCH=<tier>` sweeps tiers.

## 4. Property + fuzz hardening

```
cargo test --test property_suite          # PROPTEST_CASES=2048 for the deep lane
PATH=$CARGO_HOME/bin:$PATH cargo fuzz run <target> -- -max_total_time=300
```

Targets: `focrq_parse`, `safetensors_parse`, `image_decode`, `pretok_split`
(committed seed corpus under `fuzz/corpus/`).

## 5. The perf pillar (quiet host REQUIRED)

```
bash scripts/gauntlet_runbook.sh all      # build -> preflight -> timed legs -> row
```

Serial: preflight → focr timing → torch reference (pinned env) → roofline →
CER → the PERF_LEDGER row draft. Fairness discipline per
`docs/gauntlet/METHODOLOGY.md` §9.3 (thread pins, warmup discard, cv% bound,
within-regime pairs only).

For an inspected step-by-step run, use a fresh `OUT_DIR` throughout:

```
bash scripts/gauntlet_runbook.sh build
bash scripts/gauntlet_runbook.sh preflight
bash scripts/gauntlet_runbook.sh focr
bash scripts/gauntlet_runbook.sh reference
bash scripts/gauntlet_runbook.sh roofline
bash scripts/gauntlet_runbook.sh cer
bash scripts/gauntlet_runbook.sh row
```

### 5.1 Producer-side provenance gates

`build` first writes `focr-source-input-manifest/v1` over the clean workspace,
all reachable local path-dependency source inputs, and effective Cargo config.
It then performs the locked RCH `release-perf` build, verifies the source
snapshot again, copies the output into the fresh evidence directory, and emits
`focr-build-receipt/v1` binding the command, target/toolchain/flags, source root,
and binary hash/size. Dirty selected source, a changed build input, a reused
output directory, or a receipt/binary mismatch refuses before timing.

The final row is `focr-gauntlet-row/v3`. It separately binds the source commit
(`source_git_head`), Cargo/local-dependency closure (`source_root`), exact
gauntlet producer/validator/config blobs (`producer_root`), and the one
`artifacts/perf/...` subtree allowed to change in a later evidence commit. The
certificate accepts a later HEAD only when Git proves the source commit is its
ancestor and every path touched by every intervening commit stays within that
declared subtree. This is deliberately stricter than an endpoint diff: a
validator edit followed by a revert is still rejected.

The focr timing leg copies that receipted executable again under its own
evidence directory and executes the copy, not a mutable `target/` path. It
checks source/copy/receipt/model identities before and after capture. The
reference leg hashes the exact 12-file model before and after inference, binds
the page and the Python harness/entry/setup sources in
`focr-reference-inference-binding/v1`, and creates a fresh evidence-local
`HF_MODULES_CACHE`. Model, page, source, cache, stack, or invocation drift
refuses the capture.

### 5.2 Replay from physical evidence

Both timing legs emit `focr-gauntlet-raw-timing/v1`. The row and certificate
reconstruct sample vectors and recompute count, best, median, mean, CV, and
decode-per-token values; aggregate JSON is never trusted by itself. Each focr
run binds its raw meta/stdout/stderr, while each reference invocation binds the
deterministic OCR-text hash.

The CER step compares the exact first measured focr stdout with the exact timed
reference text and emits `focr-ocr-comparison/v1`. The row bundles the reference
text plus every measured focr stdout, requires all timed outputs to agree, then
replays the named transform, whitespace normalization, integer Levenshtein
distances, denominators, and aggregate CER. A prose claim, edited receipt, or
receipt whose physical inputs are missing or drifted cannot land.

`--smoke` and synthetic self-tests are non-citable by contract. They exercise
plumbing only; the normal row path refuses them even when every smoke check
passes.

Do not set `APPLY=1` in the strict runbook. Updating `docs/PERF_LEDGER.md` is not
an evidence-only change and there is no ledger exception. Any required ledger,
validator, or configuration update belongs in a new clean source commit before
the citable build and capture.

## 6. The full gate + the bundle

```
bash scripts/check.sh                     # every validator + fmt/check/clippy/test/ubs
# Writes an uncertified provisional bundle and exits 1 by design.
python3 scripts/gauntlet_cert.py --bundle .gauntlet-output/bundle
# Run/download exact-HEAD CI, six-job dist, weighted Model Parity, and weighted
# Performance Gauntlet evidence, then use --finalize-bundle with every emitted
# workflow manifest and three registry-pinned signer inputs. See
# RELEASE_CERTIFICATION_TEMPLATE.md for the complete command.
python3 scripts/gauntlet_cert.py --release-readiness \
  --readiness-out .gauntlet-output/release_readiness.json
```

Commit and push the final source/evidence inputs before the bundle step;
certification refuses a dirty worktree. Provisional generation never emits a
green claim. The production finalizer requires exactly one fresh run per
canonical workflow at that final HEAD, reconstructs every strict receipt from
the downloaded bytes, and requires three distinct active OpenPGP signer roles.
It proves the candidate certificate in memory before persisting
`certified:true`. The final readiness command then verifies the persisted
certificate. Its explicit `.gauntlet-output` destination keeps the verification
output ignored and the worktree clean. Regenerate steps 1–5 first when evidence
is stale. The five domain audit receipts and their named raw tool outputs are
also mandatory; the repository does not yet have a production producer for
them, so they remain an explicit certification blocker rather than being
inferred from a generic green CI log.

The verifier operates on the bundled raw timings, physical OCR texts,
source/build receipts, captured subject identity, and reference model/inference
bindings. The original 6.67 GB model directory is not needed to validate a
completed bundle; it is still mandatory to produce new reference evidence.
Certification also refuses a dirty live worktree or dirty build closure. The
certificate's signed claims bind `evidence_git_head` to the final evidence
commit. Bundle generation refuses every output outside the git-ignored
`.gauntlet-output/` tree, including the tracked historical
`docs/gauntlet/bundle/` directory.

Distribution proof is six portable jobs, never globally ISA-specialized
binaries. Both Linux assets are linked for glibc 2.17 and independently checked
with `readelf`. Both Windows assets run the PowerShell installer offline against
the exact staged `.exe`; an injected pre-replace failure must preserve the old
working executable before the successful atomic install is accepted. Main/tag
ancestry and exact `v$(Cargo.toml version)` tag identity are pre-build gates.

The `v0.6.0` 13/13 receipt is historical and cannot authorize a release from
current `main`. Even if a stale committed scorecard still says `ship: true`, it
does not cover the current candidate. The replacement exact-recipe artifact now
passes the local `page0590` termination and 20-page corpus budget; a new release
still requires publication plus fresh steps 1–6, a newly generated bundle,
signatures, platform distribution, and every remaining readiness cell.

## 7. Known residual limits

The pre/post identity checks detect ordinary drift, replacement, and partial
writes. They do not prove resistance to a hostile mutate-use-restore actor that
restores the exact bytes before the post-check. Likewise, the RCH receipt binds
the requested command, local source closure, configuration/toolchain, and
returned binary, but it is not a cryptographic attestation from the remote
worker. Do not report either property as solved.

The deterministic runaway guard remains opt-in and unpromoted. `EXP-1403` closed
as `NO_EVIDENCE` because exact Torch routing removed the no-EOS failure without
needing the controller; the accepted receipt still records a high `page0590`
CER tail. Evidence plumbing can be internally valid while release readiness
remains red: no current artifact is certified until publication, distribution,
signing, clean-tree bundle generation, and every other readiness cell pass.
