# focr Expert Operators

Use these as reusable moves when ordinary command lookup is not enough.

## Table of Contents

- [LC: Live Contract Probe](#lc-live-contract-probe)
- [MD: Model Discovery](#md-model-discovery)
- [MA: Model Artifact Setup](#ma-model-artifact-setup)
- [ZM: Zoo Manifest Pullability](#zm-zoo-manifest-pullability)
- [OF: Output File Contract](#of-output-file-contract)
- [FX: Figure Extraction Contract](#fx-figure-extraction-contract)
- [OE: One Engine](#oe-one-engine)
- [PQ: Parity First](#pq-parity-first)
- [PDF: PDF Boundary](#pdf-pdf-boundary)
- [PS: PDF Page Selection and Spread Splitting](#ps-pdf-page-selection-and-spread-splitting)
- [RP: Robot Purity](#rp-robot-purity)
- [RT: Robot Triage and Agent Ergonomics](#rt-robot-triage-and-agent-ergonomics)
- [BG: Backend and SIMD Claim Guard](#bg-backend-and-simd-claim-guard)
- [AP: Arch-Specific Prepack Boundary](#op-ap-arch-specific-prepack-boundary)
- [RS: Run Store and Sync](#rs-run-store-and-sync)
- [DR: Doctor Repair Contract](#dr-doctor-repair-contract)
- [VG: Verification Gate Reality](#vg-verification-gate-reality)
- [DG: Determinism and Fixture Governance](#dg-determinism-and-fixture-governance)
- [CM: Conformance Matrix and XFAIL Discipline](#cm-conformance-matrix-and-xfail-discipline)
- [DF: Differential Oracle Comparator](#df-differential-oracle-comparator)
- [MR: Metamorphic Relations](#mr-metamorphic-relations)
- [MP: Multi-Page Cross-Page Parsing](#mp-multi-page-cross-page-parsing)
- [GA: Golden Artifact Discipline](#ga-golden-artifact-discipline)
- [LS: Ladder Scorecard Runner](#ls-ladder-scorecard-runner)
- [SG: Release Ship Gate](#sg-release-ship-gate)
- [SQ: Stale Binary Quarantine](#sq-stale-binary-quarantine)
- [LQ: Lossy Lever Quarantine](#lq-lossy-lever-quarantine)
- [TI: Tracker-Informed Claim](#ti-tracker-informed-claim)
- [TR: Task Routing](#tr-task-routing)
- [RR: Reference Resampler](#rr-reference-resampler)
- [BS: Batch Spine Proof](#bs-batch-spine-proof)
- [PP: Preprocess Proof](#pp-preprocess-proof)
- [SD: Spec Decode Gate](#sd-spec-decode-gate)
- [SC: SmolVLM2 Conversion and Decoder Census](#sc-smolvlm2-conversion-and-decoder-census)
- [ST: SmolVLM2 Tokenizer Conformance](#st-smolvlm2-tokenizer-conformance)
- [SV: SmolVLM2 Vision and Connector Seams](#sv-smolvlm2-vision-and-connector-seams)
- [SP: SmolVLM2 Preprocess and Prompt/IO](#sp-smolvlm2-preprocess-and-promptio)
- [VQ: SmolVLM2 VQA Quality Guard](#vq-smolvlm2-vqa-quality-guard)
- [OC: OneChart Chart-Data Route and Distribution Boundary](#oc-onechart-chart-data-route-and-distribution-boundary)
- [TM: TrOMR Local Runtime, Distribution, and Quality Boundary](#tm-tromr-local-runtime-distribution-and-quality-boundary)
- [GB: Gauntlet Baseline](#gb-gauntlet-baseline)
- [Operationalized Operator Cards](#operationalized-operator-cards)

## LC: Live Contract Probe

Purpose: resolve source/help/schema disagreement.

Steps:

1. `git status --short --branch`
2. `rg -n "enum Commands|enum RobotCommands" src/cli.rs`
3. `rg -n "schema_version|run_start|run_error" src/robot.rs tests`
4. Run exact binary help if feasible.
5. Classify: current, stale binary, scaffolded, or unimplemented.

Output format:

```text
contract: current|stale-binary|scaffolded|unimplemented
evidence: <source/test/help commands>
next: <rebuild/run-source/file-bead/update-doc>
```

## MD: Model Discovery

Purpose: avoid treating model-zoo descriptors as runnable artifacts.

Steps:

1. Run `focr models --json | jq .` or inspect `model_arch::registry()`.
2. Check `implemented` / `status`; `ready` rows are normal user support.
3. Inspect `focr models --json` `pull.in_manifest` and `pull.quants`, then the
   resolved manifest; `focr pull <id>` works only for ids present in that
   manifest.
4. For a non-default artifact, inspect `.focrq` `model_id` and license metadata.
5. For GOT-OCR2, prefer `focr pull got-ocr2`; for self-converted artifacts,
   verify `qwen.tiktoken` sits beside the `.focrq`.
6. For SmolVLM2, OneChart, and TrOMR, prefer the committed `bd-av64.7`
   packaged pulls when the binary embeds that manifest; older binaries may still
   need supplied/converted artifacts or an explicit manifest override.
7. For TrOMR, report the selected quant exactly: default pull is
   `tromr.int8.focrq` after `efccce9`, while `focr pull tromr --quant f32`
   selects the bit-exact `tromr.focrq` reference. Do not turn storage-int8 into
   an int8 compute or perf claim.
8. For GOT-OCR2, classify prompt mode: plain OCR (`format=false`) or current
   structured `.mmd` mode (`--format`, `FOCR_GOT_FORMAT`, `OCR with format: `).
   Use it for specialized structured-output needs, not fast general OCR.
9. Refuse to invent task subcommands (`focr music`, `focr chart`,
   `focr describe`) before help/tests; use current `focr ocr --task` only on
   the OCR command.

Output:

```text
model: <id>
status: ready|planned|stale-binary
manifest_entry: packaged|absent|custom|unknown
pull_quants: <list-or-unknown>
artifact: present|missing|unknown
tokenizer: present|not-needed|missing
mode: plain|format-mmd|describe-vqa|not-applicable
next: <pull|convert|copy-tokenizer|wait-for-release>
```

## MA: Model Artifact Setup

Purpose: prepare inference once so runtime OCR stays offline.

Steps:

1. Prefer packaged pulls: `focr pull`, `focr pull got-ocr2`,
   `focr pull smolvlm2`, `focr pull onechart`, or `focr pull tromr`.
2. If using a custom distribution, pass `focr pull --manifest <path-or-url>` or
   set `FOCR_MANIFEST_URL`; record the manifest source and entries.
3. For self-converted weights, run `focr convert ... --quant int8 --model-id <id>`
   and record source SHA256, output hash, model id, converter revision, and command transcript.
4. Verify hashes/artifact metadata.
5. Set `FOCR_MODEL_PATH` or `FOCR_MODEL_DIR` in runtime environment.
6. Run `focr robot schema`, `focr robot health`, `focr robot backends`, and
   `focr robot selftest`.
7. Disable network in an inference smoke test when possible.

Reject a deployment plan that downloads the model on first request.

## ZM: Zoo Manifest Pullability

Purpose: answer whether a named model is pullable by this binary without
confusing registry readiness, manifest entries, and quant availability.

Steps:

1. Run `focr models --json | jq '.models[] | {id, implemented, pull}'`.
2. Inspect `models/manifest.json` or `src/dist.rs` for
   `BUILTIN_MANIFEST_JSON`, `ModelEntry.sidecars`, and `select_quant` when
   source truth is needed.
3. For `got-ocr2`, expect `int8` plus `qwen.tiktoken`.
4. For `smolvlm2`, expect `int8` plus `tokenizer.json`.
5. For `onechart`, expect `int8` plus `vocab.json`, `merges.txt`, and
   `added_tokens.json`.
6. For `tromr`, expect default `int8` plus f32 reference quant, each with
   `tokenizer_rhythm.json`, `tokenizer_pitch.json`, `tokenizer_lift.json`, and
   `tokenizer_note.json`; `focr pull tromr` selects `tromr.int8.focrq`, while
   `focr pull tromr --quant f32` selects `tromr.focrq`.
7. If binary/source disagree, classify stale binary or release lag before
   changing docs.

Output:

```text
model: <id>
pull_in_manifest: true|false|unknown
pull_quants: <list>
sidecars: <list>
cache_layout: flat|models/<id>|unknown
binary_boundary: current|stale|release-lag|unknown
next: <pull|rebuild|use-explicit-manifest|convert-local|do-not-claim>
```

## OF: Output File Contract

Purpose: integrate `focr ocr -o/--output FILE` without corrupting pipes.

Rules:

1. Use `-o out.md` for markdown and `-o out.json` for JSON-with-boxes.
2. Treat `--json` as an override that forces JSON regardless of extension.
3. Expect stdout to stay empty in human mode when `-o` is present; a short
   confirmation may go to stderr.
4. For JSON, bind to `schema_version`, `markdown`, and either top-level
   `layout` or PDF `pages`.
5. On failure, assert the output file was not created or left partial.

## FX: Figure Extraction Contract

Purpose: use `--extract-figures` without inventing files or overasserting output.

Rules:

1. Prefer `focr ocr page.png -o page.md --extract-figures`.
2. Use `--figures-dir DIR` for stdout or a custom directory.
3. Treat no destination as a usage error, not a model error.
4. Bind JSON to `figures: [{label,page,bbox,path}]` only when present.
5. Do not assert at least one figure unless the corpus is known to ground one.
6. On failure, assert no output file and no derived figures directory.

## OE: One Engine

Purpose: keep Rust integration aligned with runtime doctrine.

Steps:

1. Create one `OcrEngine`.
2. Store it in application state.
3. Route requests through a blocking boundary if host is async.
4. Use batch APIs for multi-image work.
5. Avoid outer parallel page loops unless upstream exposes a safe policy.
6. In current `bd-223.2` source, keep `request_shutdown`, `cancel_checkpoint`,
   `thread_budget`, `FOCR_THREADS`, and `stream_pages` scoped to the one-engine /
   one-live-forward doctrine.

Review smell: `OcrEngine::new()` appears inside a hot request handler.

## PQ: Parity First

Purpose: evaluate any optimization or quantization claim.

Steps:

1. Identify exact behavior surface.
2. Establish baseline output and metrics.
3. Change one lever.
4. Run parity/golden/CER evidence.
5. Keep only if the gate passes; otherwise revert or keep behind a kill switch.
6. Record accepted divergence or negative evidence in the project docs.

Never justify an OCR regression with throughput alone.

## PDF: PDF Boundary

Purpose: use the current scanned-PDF fast path without overstating it.

Current CLI `ocr` and `robot run` accept document images and scanned PDFs. When
a user gives a PDF:

1. Use `focr ocr file.pdf` or `focr robot run file.pdf` for the CLI path.
2. For library integrations, use `franken_ocr::pdf::PdfPages` plus
   `OcrEngine::recognize_dynamic`.
3. If `InputDecode` names `JPXDecode`, `JBIG2Decode`, unsupported color spaces,
   or a born-digital/vector page, rasterize out of band and retry with images.
4. Preserve the rasterization command and DPI in evidence, because pixel
   differences affect OCR parity.

## PS: PDF Page Selection and Spread Splitting

Purpose: use `--pages` and `--split-spreads` without overstating PDF proof.

Steps:

1. Confirm the input is a native scanned PDF path through `focr ocr` or
   `focr robot run`; these flags do not apply to `ocr-batch`.
2. Parse page specs as 1-based comma/range selectors such as `3,5-9`.
3. Treat invalid ranges as usage errors, not model failures.
4. For `--split-spreads`, inspect `src/pdf.rs::split_spread` and tests when
   exact behavior matters.
5. Preserve original page and half metadata in JSON/robot consumers.
6. Remember current source applies page `/Rotate` plus axis-aligned
   content-stream image-placement rotation through `content_rotation`; stale
   binaries may still miss that fix.
7. Do not combine `--split-spreads` with `--extract-figures`; current source
   cleanly refuses that pair.

Output:

```text
pdf_pages: all|subset:<spec>
split_spreads: enabled|disabled
rotation_normalization: present|absent|unknown
figures_combo: refused|not-used|unknown
logical_pages: <count-or-unknown>
half_metadata: present|absent|not-checked
boundary: extraction-heuristic|accepted-contract|stale-binary
```

## RP: Robot Purity

Purpose: protect automation consumers.

Checks:

```bash
focr robot schema | jq .
focr robot run page.png | while IFS= read -r line; do
  printf '%s\n' "$line" | jq -e . >/dev/null || exit 1
done
```

Rules:

- stdout is JSON/NDJSON only,
- schema is versioned,
- exit code is checked,
- current `bd-223.2` `threads` fields are host-dependent and should be scrubbed in
  goldens,
- human messages go to stderr or human commands.

## RT: Robot Triage and Agent Ergonomics

Purpose: orient agents with one JSON call instead of several brittle help probes.

Steps:

1. Run `focr robot triage | jq .`.
2. Inspect `quick_ref` for cache/model/setup state.
3. Use `recommendations[0]` before proposing commands.
4. Copy from `commands` rather than retyping human examples.
5. Use `exit_codes` to keep caller handling precise.
6. If missing, classify the binary as pre-`bd-wp8.7` or stale.

Output:

```text
triage: available|missing|invalid-json
top_recommendation: <summary>
next_command: <command-or-none>
stdout_purity: pass|fail|not-checked
```

## BG: Backend and SIMD Claim Guard

Purpose: keep backend facts, parity facts, and speed facts separate.

Steps:

1. Run or inspect `focr robot backends` for selected tier, available tiers,
   `override_env`, `logical_cpus`, and current `threads`.
2. Use `FOCR_FORCE_ARCH=scalar|sdot|smmla|avx2|avxvnni|avx512vnni` only for a
   tier that exists on the host; unsupported forced tiers should fail or fall
   back explicitly, not silently prove support.
3. Run `focr robot selftest` for selected-kernel parity against the scalar
   oracle; do not cite it as throughput evidence.
4. For performance, use OP-GB and `docs/PERF_LEDGER.md`; record host, threads,
   model, corpus/page, correctness, and raw timings.
5. Keep architecture claims exact: Apple Silicon prefers SDOT over SMMLA,
   non-Apple aarch64 may prefer SMMLA, x86 tiers are AVX2 / AVX-VNNI /
   AVX-512-VNNI, and AMX is not current until `robot backends` advertises it.

Output:

```text
backend_claim: capability|parity|performance
selected_tier: <tier|unknown>
available_tiers: <list|unknown>
force_arch: <unset|tier|unsupported>
selftest: pass|fail|not-run
perf_evidence: ledger|fresh-gauntlet|missing
```

## VG: Verification Gate Reality

Purpose: use recent test-infrastructure closures without inflating their scope.

Steps:

1. Confirm the relevant closure in live Beads before citing it:
   `bd-zc1o`, `bd-n68o`, `bd-29wv`, or `bd-re8.7`. Also check
   `bd-wp8.2.2` before claiming current robot-schema goldens are green.
2. Bind the claim to the exact artifact: frozen robot schema, structured test
   log schema, model-gated e2e runner, or L5 OCR fixture gate.
3. For model-gated tests, distinguish skip-with-SUCCESS from success-path proof
   and require `native_path_ran=true` plus the fallback-target proof when the
   native path is claimed.
4. Do not treat harness closure as a universal quality/performance result; name
   the model, fixture/corpus, metric, and remaining gates.
5. If older worktrees or sessions disagree with current `main`, classify them
   as stale history unless source/tests/Beads on `main` confirm the same claim.
6. Treat `bd-zc1o` as the historical frozen-schema closure, not a permanent
   all-green claim. If `staff` appears in `robot::EVENT_KINDS`, verify whether
   the checkout includes `adb4ee6`: at/after that source, the frozen schema
   fixture and advertised-events assertion include `staff`. If `bd-wp8.2.2`
   still appears open, treat it as possible stale tracker state until focused
   schema tests prove otherwise.

Output:

```text
gate: robot-schema|test-log|model-gated-e2e|l5-parity|other
bead: <id-status>
artifact: <file-or-script>
skip_status: skipped|native-path-ran|not-applicable|unknown
metric: <cer|teds|schema|jsonl|none>
remaining_scope: <corpus|model|perf|none|unknown>
```

## DG: Determinism and Fixture Governance

Purpose: use `bd-3kge` / `bd-2pgf` evidence without confusing infrastructure
determinism with oracle-quality proof.

Steps:

1. Check committed `HEAD` for `tests/support/parity_harness.rs`,
   `tests/e2e_recognize.rs`, `tests/fixtures/PROVENANCE.md`,
   `tests/fixtures/MANIFEST.toml`, and `scripts/check_fixture_manifest.py`.
2. For same-input determinism, use `assert_deterministic` or
   `assert_outputs_deterministic`; require `parity` / `token_exact` output and
   byte-identical payloads. A divergence is a real engine bug, not tolerance
   noise.
3. Keep this separate from oracle nondeterminism-floor gates. `bd-3kge` proves
   our greedy path is stable; it does not prove the output is correct relative
   to the Python/HF oracle.
4. For fixtures, every top-level `tests/fixtures/` entry must be declared in
   `MANIFEST.toml`; regenerated-committed entries need a generator script, and
   `PROVENANCE.md` must explain the committed vs off-tree policy.
5. Run `python3 scripts/check_fixture_manifest.py` or `scripts/check.sh` before
   accepting fixture-policy changes.

Output:

```text
determinism_gate: present|absent|not-checked
adoption: e2e|unit|none|unknown
fixture_manifest: clean|missing|stale|not-run
claim_boundary: determinism-only|fixture-policy-only|oracle-quality|unknown
```

## CM: Conformance Matrix and XFAIL Discipline

Purpose: use `bd-re8.12` evidence as release/conformance accounting without
rounding it up to universal model quality or a complete release certificate.

Steps:

1. Check committed `HEAD` for `src/conformance.rs` and
   `tests/conformance_matrix.rs`; verify `br show bd-re8.12 --json` is closed.
2. Confirm `ConformanceTest` exposes `name`, `category`, `requirement_level`,
   `clauses`, and `run`; every registry row must run a real in-process
   representative check, not a no-op.
3. Treat the matrix as spec-side accounting: it parses
   `docs/truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md`, classifies clauses as
   MUST/SHOULD/MAY, scans `src/**` and `tests/**` for `[SPEC-NNN]` references,
   and gates MUST coverage >= 0.95.
4. Preserve XFAIL discipline: an xfail emission must be attached to a real
   `DISC-NNN` or an explicit phase-gap reason. Missing model/fixture can be an
   honest skip-with-SUCCESS; accepted divergence is XFAIL and counted.
5. Keep proof families separate. `bd-re8.9`, `bd-re8.10`, and `bd-re8.11` are
   now closed-current, but they prove differential, metamorphic, and
   golden-artifact contracts respectively; they are not replacements for the
   conformance matrix. `bd-re8.13` three-pillar release certification,
   `bd-re8.14` conformal ratchet, and `bd-re8.15` e-process invariants still
   need their own evidence if open.

Output:

```text
conformance_trait: present|absent|not-checked
must_coverage_gate: pass|fail|not-run
xfail_discipline: pass|fail|not-run
registry_run: pass|fail|not-run
claim_boundary: accounting-only|release-cert|quality-proof|unknown
```

## DF: Differential Oracle Comparator

Purpose: use `bd-re8.9` evidence to compare focr against the bf16 reference
without accidentally comparing the oracle to itself or hiding divergences as
skips.

Steps:

1. Check `tests/parity_ladder.rs` for `differential_per_op_vs_bf16_oracle`,
   `differential_row`, and the `EngineIdentity` guard.
2. Confirm each row names the `scope`, `oracle`, `module`, `max_diff`,
   `within_tol`, `xfail`, and `disc` boundary.
3. Use ULP tables and the existing L3-L5 ladder tolerances; do not invent a
   text-only comparator for tensor/op parity.
4. Treat accepted intentional drift as `DISC-NNN` XFAIL, never as SKIP.
5. When model artifacts are absent, report model-gated skip-with-SUCCESS as
   unarmed evidence. Only cite real e2e parity when the native path ran.

Output:

```text
differential_gate: present|absent|not-checked
subject_identity: not-oracle|oracle-vs-oracle-risk|unknown
rows: present|missing|not-run
xfail_discipline: pass|fail|not-run
model_path: native-ran|skipped-missing-artifact|not-applicable|unknown
claim_boundary: per-op|l3-l5|e2e|not-proven
```

## MR: Metamorphic Relations

Purpose: use `bd-re8.10` evidence for oracle-free self-consistency claims
without adding false relations.

Steps:

1. Inspect `tests/metamorphic.rs` and `docs/conformance/METAMORPHIC.md`.
2. Verify the always-on strict relations: MR-1 identity resize, MR-3a
   mean-gray padding through `preprocess::PAD_FILL`, and MR-4 determinism
   across repeat runs plus `FOCR_THREADS=1` vs `4`.
3. For MR-2, separate always-on coordinate math from live grounding-box proof;
   the live leg is model/page gated.
4. For MR-3b, treat white padding as a logged SHOULD/existential observation,
   not a strict assertion.
5. For MR-5, keep cross-page dependence gated on the documented open question.
   Do not assert that multi-page output equals the concatenation or sum of
   single-page parses; R-SWA makes multi-page context intentionally dependent.

Output:

```text
metamorphic_gate: present|absent|not-checked
strict_relations: pass|fail|not-run
observational_relations: logged|missing|not-run
gated_relations: honest|misreported|not-checked
false_relation_guard: pass|fail|not-checked
claim_boundary: oracle-free-self-consistency|quality-proof|not-proven
```

## MP: Multi-Page Cross-Page Parsing

Purpose: use the current Unlimited-OCR `infer_multi` surface without
flattening it into independent pages or overclaiming streaming/page-layout
support.

Steps:

1. Decide the input class. Use `focr ocr-batch page1.png page2.png --multi-page`
   for page image lists, and `focr ocr doc.pdf --multi-page [--pages ...]` for
   native scanned PDFs.
2. Verify the running binary/source exposes the flag; older binaries can lack
   it even when current source has it.
3. Keep model scope exact: current multi-page parsing is the Unlimited-OCR
   contract. Non-default zoo models parse pages independently unless source
   adds and proves a model-specific multi-page route.
4. Preserve composition rules. PDF `--multi-page` composes with `--pages`, but
   refuses `--split-spreads` and `--extract-figures`; image-list multi-page is
   on `ocr-batch`, not native PDF routing.
5. Bind output shape to the route. Current result is one markdown document with
   `<PAGE>` separators. Batch JSON uses `command: "batch.multi_page"`,
   `pages`, `seconds`, and `markdown`. PDF multi-page JSON has empty
   per-page layout metadata because boxes are not produced for this route.
6. Preserve the 32K context guard. Large documents must be split; do not ask
   agents to bypass `MAX_POSITION_EMBEDDINGS`.
7. For robot/streaming claims, treat `bd-2z0y` as closed-current: PDF
   `--multi-page` emits additive schema-v1 `page` decoded progress events at
   `<PAGE>` boundaries in robot mode. These events expose raw decoded page
   text, not per-page layout boxes, figures, or split-spread results.

Failure Modes:

- calling `--multi-page` a loop over single-page recognition,
- asserting a multi-page concat/sum equality relation,
- promising per-page layout boxes, figures, split-spread support, or layout-like
  semantics from the decoded streaming `page` events,
- using it with GOT-OCR2, SmolVLM2, OneChart, or TrOMR without model-specific
  source/test proof,
- ignoring the 32K prompt guard and telling users to process arbitrarily large
  PDFs in one pass.

Evidence anchors:

- `src/native_engine/mod.rs` `recognize_multi_page`
- `src/native_engine/mod.rs` `build_prompt_multi`
- `src/native_engine/mod.rs` `recognize_multi_page_dynamic_streaming`
- `src/native_engine/postprocess.rs` `PageStream`
- `src/preprocess/mod.rs` `preprocess_dynamic_squash`,
  `multi_page_base_640_placeholder_is_111`
- `src/robot.rs` `page_decoded_event`
- `src/lib.rs` `OcrEngine::recognize_multi_page`
- `src/lib.rs` `recognize_multi_page_dynamic_with_model`
- `src/cli.rs` `ocr-batch --multi-page`
- `src/cli.rs` `recognize_pdf_multi_page`
- `tests/e2e_recognize.rs`
  `recognize_multi_page_real_model_when_present_else_skip_with_success`
- `tests/e2e_recognize.rs`
  `multi_page_streaming_matches_terminal_assembly_when_armed`
- `tests/parity_ladder.rs` `l5_multi_page_matches_infer_multi_oracle`
- `tests/cli_robot_golden.rs`
  `batch_multi_page_flag_routes_to_the_cross_page_pass`
- `tests/metamorphic.rs` false multi-page concat guard
- `README.md` multi-page examples and caveats
- `br show bd-1gv.25 --json`
- `br show bd-2z0y --json`
- `br show bd-1gv.26 --json`
- `br show bd-1465 --json`
- `4afcaca`, `f115403`, `b9cc16c`, `a2dd1c9`, `750a69a`, `828ea4c`,
  `727701b`, `6e297f6`, `c6ab897`

Output:

```text
multi_page_surface: ocr-batch-images|ocr-pdf|library|absent|stale-binary
model_scope: unlimited-ocr-only|model-specific-proof|unknown
input_count: <n-or-unknown>
preprocess: squash-640-pil-bicubic-111-placeholders|unknown
context_guard: pass|too-large|not-checked
output_shape: markdown-page-separators|batch-json|pdf-json-empty-layout|unknown
streaming_page_events: decoded-progress|absent|not-checked
claim_boundary: <one sentence>
```

## GA: Golden Artifact Discipline

Purpose: use `bd-re8.11` evidence to preserve frozen CLI/robot/schema/numeric
surfaces without blindly blessing changes.

Steps:

1. Classify the artifact before comparing it:
   exact CLI/help/schema JSON, fuzzy numeric tensor/logit artifact, scrubbed
   robot NDJSON, canonicalized cross-platform JSON/text, or reference-output
   canonicalized exact.
2. Inspect `tests/cli_robot_golden.rs`, `tests/fixtures/golden/PROVENANCE.md`,
   and `docs/conformance/GOLDEN.md` for the comparison and scrubber path.
3. On mismatch, expect a transient `.actual` or `.snap.new` review artifact;
   do not commit those files.
4. Use `UPDATE_GOLDENS=1` only as a deliberate human-reviewed refresh step.
   CI must not set it or silently auto-update goldens.
5. Restamp provenance whenever committed goldens change, and explain why the
   surface change is intended rather than merely convenient.

Output:

```text
golden_gate: present|absent|not-checked
artifact_pattern: exact|fuzzy|scrubbed|canonicalized|reference-output|unknown
diff_review: clean|needs-human-review|not-run
update_mode: manual|ci-forbidden|unsafe-auto-update|unknown
provenance: current|stale|missing|not-checked
claim_boundary: surface-freeze|numeric-envelope|quality-proof|not-proven
```

## LS: Ladder Scorecard Runner

Purpose: use `scripts/ladder_scorecard.sh` without confusing a summarized
receipt, skipped model gates, and Beads close status.

Steps:

1. Check committed source for `scripts/ladder_scorecard.sh` and record the
   source revision. `1b84428` adds the initial runner for `bd-re8.19`;
   `1112cf8` plus close evidence make the current snapshot closed-current.
2. Check live tracker state with `br show bd-re8.19 --json`; source-present and
   tracker-closed are different facts.
3. For a fast parser/self-check, run or cite `scripts/ladder_scorecard.sh
   --self-test`; it validates fold logic without model weights.
4. For a real receipt, run `scripts/ladder_scorecard.sh [--out FILE]` only when
   the build/model budget is acceptable. It runs `parity_ladder` serially with
   `--test-threads=1` so rung names preserve L0-L5 order.
5. Interpret `skipped_no_model=true` as visible missing-artifact evidence, not
   as an all-green ladder. Interpret `not_meaningful` gates as downstream of
   the first hard failure.

Output:

```text
scorecard_runner: present|absent|not-checked
source_revision: <sha-or-unknown>
tracker_status: open|closed|unknown
self_test: pass|fail|not-run
scorecard_schema: focr-ladder-scorecard/v1|unknown
all_green: true|false|not-run
skipped_no_model: true|false|not-run
first_failure_boundary: <gate-or-none-or-unknown>
```

## SG: Release Ship Gate

Purpose: answer release-ready questions without collapsing many proof families
into a vague "green" claim.

Steps:

1. Run or inspect `python3 scripts/gauntlet_cert.py --release-readiness`.
2. Read `docs/gauntlet/RELEASE_READINESS.json`.
3. Check `br show bd-wp8.10 --json` for the capstone status.
4. For bundle questions, run or inspect `python3 scripts/gauntlet_cert.py
   --bundle` and read `docs/gauntlet/bundle/release_certificate.json`.
5. Check `br show bd-wp8.9 --json` for the bundle machinery status.
6. If any cell is red or `ship=false`, report not shippable and name the red
   cells.
7. Keep convergence (`bd-wp8.8`) and certification bundle state separate from
   the scorecard script existing.
8. After `9bc715e`, do not call `certification_bundle` hard-coded red or
   self-referential: it reads the live `release_certificate.json`, and
   `--bundle` excludes that cell from its own certification predicate. Current
   `c29a78b` / `7c7bd00` / `29516b9` evidence has this cell green with
   `certified:true`.

Output:

```text
release_scorecard: present|absent|not-run
artifact_schema: franken_ocr.release_readiness.v1|unknown
ship: true|false|unknown
red_cells: <list>
capstone_status: open|closed|unknown
bundle_status: open|closed|unknown
bundle_certified: true|false|unknown
next_gate: certification_bundle|gauntlet_convergence|none|unknown
```

## RS: Run Store and Sync

Purpose: classify and use the `bd-223.4` run-history/audit surface while
separating closed-current source from stale or release-lagged binaries.

Steps:

1. Probe source/help for `RunStore`, `FOCR_RUN_STORE`, `focr runs`, and
   `focr sync export-jsonl|import-jsonl`.
2. Check `br show bd-223.4 --json`; if it is closed and source/help/tests
   agree, label the feature closed-current. If the tracker is open/in_progress,
   label it WIP even when code exists.
3. For tests, set `FOCR_RUN_STORE` to a temp database path before invoking
   `focr ocr`, `focr runs`, or `focr sync`.
4. Bind automation to JSON/NDJSON records, not human plain output.
5. Verify sync writes canonical JSONL and leaves no `.jsonl.tmp` or stale
   `.jsonl.lock` on success.

Output:

```text
run_store: absent|scaffolded|live-wip|closed-current|stale-binary
store_path: <FOCR_RUN_STORE|default|unknown>
schema_version: <n|unknown>
records_shape: json|ndjson|plain|unknown
sync: export|import|both|absent|unknown
tracker: <bd-223.4 status>
```

## DR: Doctor Repair Contract

Purpose: diagnose and repair local focr state without bypassing the doctor
ledger or mutating user caches by accident.

Steps:

1. Start with `focr doctor --json` unless the user explicitly requests repair.
2. For repair planning, use `focr doctor --dry-run --fix --json`.
3. For real repair, use `focr doctor --fix --json` and record the run id.
4. If repair must be reverted, use `focr doctor undo <run-id> --json`.
5. Use `focr doctor capabilities --json` or `focr doctor robot-docs` for
   integration docs.
6. Treat doctor exit code 5 as doctor-specific lock contention, not global OCR
   cancellation.

Output:

```text
doctor: implemented|stale-scaffold|missing|unknown
mode: detect|dry-run|fix|undo|capabilities|robot-docs
findings: <count-or-unknown>
mutations: none|planned|applied|rolled-back|unknown
run_id: <id-or-none>
exit_code: <code-or-unknown>
```

## SQ: Stale Binary Quarantine

Purpose: avoid false docs from old builds.

Trigger:

- help output conflicts with source,
- command missing from installed binary,
- behavior contradicts tests.

Action:

1. Mark binary stale in notes/final answer.
2. Use source and tests for truth.
3. Rebuild or run from source only if needed and feasible.
4. Do not edit docs to match stale output.

## LQ: Lossy Lever Quarantine

Purpose: contain experimental env vars and quantization changes.

Before enabling a lossy path, require:

- model artifact hash,
- corpus/image list,
- metric and allowed budget,
- deterministic fallback,
- env var list,
- Beads issue or evidence ledger.

If any item is missing, leave the lever off.

## TI: Tracker-Informed Claim

Purpose: answer capability questions honestly.

Steps:

1. Search source for the surface.
2. Search tests for proof.
3. Search `br` for the feature.
4. Use `bv --robot-triage` only for graph-level context.
5. If CASS has prior sessions, treat them as history, not live truth.

Final phrasing should distinguish:

- implemented and tested,
- implemented but not fully proven,
- scaffolded,
- planned,
- blocked.

## TR: Task Routing

Purpose: select the right model/mode without inventing CLI or library surfaces.

Steps:

1. For plain document OCR, use the default `unlimited-ocr`.
2. For formulas, tables, charts, molecular structures, geometry, or sheet
   music, use GOT-OCR2 and `--task <task>` or `--format`.
3. Ensure `focr pull got-ocr2` has installed `got-ocr2.int8.focrq` and
   `qwen.tiktoken`.
4. For photo description or VQA, use `--task describe` with a supplied or
   converted `smolvlm2.int8.focrq`; add `--question` for VQA.
5. Do not claim `focr music`, `focr chart`, or `focr describe` subcommands.
6. In Rust integrations, remember that `--task` is CLI sugar; library callers
   need explicit model path plus a process-level GOT format or SmolVLM2 question
   policy.

## RR: Reference Resampler

Purpose: diagnose L0/preprocess mismatch without changing product defaults.

Steps:

1. Reproduce the issue on the default CatmullRom path.
2. Rerun with `FOCR_RESAMPLE=pil-bicubic`.
3. Compare L0 tensors/output hashes and record whether the difference is only
   the DISC-001 resampler divergence.
4. If the model is SmolVLM2, branch to SP instead: C7 uses fixed
   Pillow-exact LANCZOS via `resize_lanczos`, not the `FOCR_RESAMPLE`
   BICUBIC investigation knob.
5. Do not recommend `pil-bicubic` as a quality fix unless an e2e A/B proves it
   on the target corpus.

## BS: Batch Spine Proof

Purpose: use `ocr-batch` throughput controls without violating one-forward
doctrine or silently changing output.

Steps:

1. Establish a sequential batch baseline with `FOCR_BATCH_SPINE` unset.
2. Arm `FOCR_BATCH_SPINE=1` and record `FOCR_BATCH_SIZE`; to disable, unset it
   or use `0`/`off`/`false`/`no`.
3. Record the model id/arch. Default Unlimited-OCR int8 spine evidence and the
   closed dense zoo evidence are separate. Current dense zoo proof covers
   GOT-OCR2, SmolVLM2, and OneChart; do not carry it to future zoo artifacts
   without source/test proof. For dense zoo runs, record
   `OcrModel::recognize_batch_dense`, `recognize_batch_dense_got`,
   `smolvlm2::recognize_batch`, `onechart::recognize_batch`,
   `generate_greedy_batched` per-stream `caps: &[usize]`,
   `PageStream::with_max_emit`, `FOCR_BATCH_PACK`, and the 128/256 default/cap.
4. Keep `FOCR_BATCH_VISION` default-on unless proving the vision kill-switch;
   set it to `0` only for parity/diagnosis.
5. If `FOCR_BATCH_PACK` is armed, prove it is a pure admission-order change:
   same stream set, no duplicated/lost pages, output restored to input order,
   and byte-identical per-stream text.
6. Compare per-image markdown/output hashes in input order.
7. Cite `bd-1azu.10` and `bd-1azu.14` for current source evidence, but rerun on
   the target corpus before making a performance claim.

## PP: Preprocess Proof

Purpose: determine whether `--base-size`, `--image-size`, and `--crop-mode`
are live, and whether a Gundam claim is merely wired or actually proven.

Steps:

1. Inspect `src/cli.rs` for `preprocess_overrides_from` and
   `native_engine::PreprocessOverrides`.
2. Establish the default `base` output/hash.
3. Run an explicit size override or `--crop-mode gundam`.
4. Compare output, view count, and CER/golden results as the task requires.
5. Cite `bd-1e9n` only for first e2e evidence; rerun target-corpus proof before
   changing defaults or parity claims.

## SD: Spec Decode Gate

Purpose: handle `FOCR_SPEC_DECODE` without silently changing generated tokens.

Steps:

1. Remember that `FOCR_SPEC_DECODE` is presence-armed; OFF means env removed.
2. Run `scripts/spec_gate_e2e.sh` with `FOCR_MODEL_PATH` and
   `FOCR_SPEC_E2E_IMAGES`.
3. Confirm OFF==OFF determinism if available.
4. Compare ON and OFF outputs by hash and inspect preserved workdir on failure.
5. Refuse any certification with `FOCR_ATTN_GEMM` or `FOCR_INT8_KV` present.

## SC: SmolVLM2 Conversion and Decoder Census

Purpose: prove or troubleshoot SmolVLM2 source-weight conversion and C5 decoder
seam evidence without confusing conversion with the separate C8/C10 route and
quality/e2e evidence.

Steps:

1. Inspect `src/native_engine/model_arch.rs` and live help. Current source marks
   SmolVLM2 `implemented=true` after C7/C9 route wiring; older binaries can
   still report planned/NotImplemented.
2. Run `scripts/smolvlm2_convert_e2e.sh` with `SMOLVLM2_SAFETENSORS`, or run
   `focr convert --model-id smolvlm2 --json` and then census the `.focrq`.
3. Require 489 tensors, 224 int8 decoder GEMMs, 265 F32 high-precision tensors,
   `model_id=smolvlm2`, the Apache-2.0 notice, and an untied high-precision
   `lm_head`.
4. For decoder seam claims, require `FOCR_SMOLVLM2_MODEL`,
   `FOCR_SMOLVLM2_ORACLE_HIDDEN0`, and `FOCR_SMOLVLM2_ORACLE_LOGITS` evidence;
   cite C5 f32 cos 1.000000, int8 cos 0.998301, and DISC-002.
5. Treat a one-int8-tensor or int8-lm_head result as a stale classifier/binary.
6. Route `--task describe` only with `--model smolvlm2.int8.focrq`; attach
   DISC-003 and C8/C10/A11 evidence before making quality/perf claims.

## ST: SmolVLM2 Tokenizer Conformance

Purpose: prove or troubleshoot C6 SmolLM2 tokenizer behavior without falling
back to the Baidu pretokenizer.

Steps:

1. Inspect `src/tokenizer/mod.rs` for `PretokScheme` classification and
   `PretokScheme::SmolLm2` accessors.
2. Set `FOCR_SMOLVLM2_TOKENIZER_JSON` or `FOCR_SMOLVLM2_DIR` to the pinned real
   tokenizer JSON before running the real-tokenizer gate.
3. Require the GPT-2-style path selected by `Digits(individual_digits=true)` and
   `ByteLevel(use_regex=true)`, not the DeepSeek four-stage split.
4. Confirm special ids: bos 1, eos 49279, pad 2, image 49190.
5. Require the C6 corpus result: 128/128 token-id-exact and decode-exact against
   HF tokenizers over sha prefix `5ece781d`.
6. If ids change, regenerate with `scripts/gen_smolvlm2_token_id_fixtures.py`
   only after documenting the new tokenizer source and corpus.

## SV: SmolVLM2 Vision and Connector Seams

Purpose: use the closed C3/C4 SigLIP/pixel-shuffle seam evidence without
claiming more than the relevant layer proves.

Steps:

1. Check live Beads first: current source observed `bd-3jo6.3.3`,
   `bd-3jo6.3.4`, `bd-3jo6.1.8`, and `bd-3jo6.1.9` closed; C8/C10 are now
   closed too, but seam evidence still does not by itself prove downstream
   quality or performance.
2. Inspect current source for `src/native_engine/vision_siglip.rs`,
   `src/native_engine/token_compress.rs`, and the `pub mod` entries in
   `src/native_engine/mod.rs`; if current source includes `f1ac972` or later,
   also inspect `smolvlm2::vision_rows`,
   `vision_siglip::forward_frames_batched`, and `FOCR_SIGLIP_SEQ`.
3. For SigLIP, verify the contract: 512 input, 1024 patch tokens, hidden 768,
   12 layers/heads, bidirectional SDPA, NaViT bucketized learned 1-D positions
   (`[0,0,1,...,30]`, not identity), final post-layernorm, and
   `nn::gelu_tanh`.
4. For connector, verify `pixel_shuffle` scale 4 maps `[1024,768]` to
   `[64,12288]`, then modality projection maps `12288 -> 960`.
5. Use `FOCR_SMOLVLM2_DIR` plus
   `scripts/gen_reference_fixtures_smolvlm2_vision.py` artifacts for seams:
   `smolvlm2_pixel_values.bin`, `smolvlm2_vision_post_ln.bin`,
   `smolvlm2_pixel_shuffle_out.bin`, `smolvlm2_connector_out.bin`, and
   `smolvlm2_vision_tensors.npz`.
6. Require C3/C4 evidence before making seam claims: C3 worst cos 1.00000000
   and maxabs 4.4e-4 over 13 real frames; C4 `pixel_shuffle` bit-exact and
   connector projection cos 1.00000000, maxabs 2.59e-4 within the measured
   1.1e-3 budget.
7. Keep the conclusion narrow: seam-certified plus implemented route; use the
   C8/C10/A11 evidence for generated-output quality and performance.

## SP: SmolVLM2 Preprocess and Prompt/IO

Purpose: use the active C7/C9 SmolVLM2 preprocess, prompt/IO, and source-route
evidence without mistaking it for C8 image-forward certification.

Steps:

1. Check live Beads first: `bd-3jo6.3.7` is closed for C7
   preprocess/prompt-IO, `bd-3jo6.3.9` is closed for C9 route support,
   `bd-3jo6.3.8` is closed for C8, `bd-3jo6.3.10` is closed for C10, and
   `bd-3jo6.3` is closed for sub-epic C.
2. Inspect `src/preprocess/mod.rs` for `preprocess_smolvlm2` and
   `preprocess_smolvlm2_path`.
3. Inspect `src/preprocess/pil_resample.rs` for `resize_lanczos`; require
   Pillow LANCZOS (`resample: 1`), not CatmullRom or BICUBIC.
4. Verify the shape contract: longest side 2048, dimensions ceiled to 512
   multiples, row-major local 512x512 frames, plus one global 512 frame.
5. Inspect `src/native_engine/smolvlm2.rs` when present: prompt ids, 64
   `<image>` slots per frame, global-last order, `build_inputs_embeds`,
   `--question`, and `FOCR_SMOLVLM2_QUESTION`.
6. For L0 proof, use the Pillow 12.3.0 LANCZOS goldens and, when available,
   `FOCR_SMOLVLM2_DIR` pixel-value oracle artifacts.
7. For prompt/IO claims, pair this with ST so token ids, image-token expansion,
   and special ids are exact.
8. End with implemented route status, DISC-003 near-tie ledger, exact C8/C10
   quality/e2e evidence, and any remaining manifest/A11/perf boundaries.

## VQ: SmolVLM2 VQA Quality Guard

Purpose: run or interpret the C8 informational VQA guard without turning it
into a public benchmark or a human-label score.

Steps:

1. Check live Beads first: `bd-3jo6.3.8`, `bd-3jo6.3.10`, and `bd-3jo6.3` are
   now closed in current source, but only cite that after current `br show`
   confirms the same state.
2. Ensure `tests/fixtures/smolvlm2/vqa_fixtures.json` exists. If it must be
   regenerated, use `scripts/gen_smolvlm2_vqa_fixtures.py` against the pinned
   `FOCR_SMOLVLM2_DIR` and preserve the generator transcript.
3. Arm `FOCR_SMOLVLM2_DIR` with `tokenizer.json` plus `model.safetensors`,
   `smolvlm2.int8.focrq`, or both. Missing artifacts skip their weight leg;
   present-but-broken artifacts must fail.
4. Run
   `FOCR_SMOLVLM2_DIR=/path/to/smolvlm2 cargo test vqa_quality_matches_oracle_l5 -- --nocapture`.
5. Interpret the guard narrowly: each answer is compared to the fixture oracle's
   own greedy text by normalized exact match or symmetric content-word
   containment >= 0.5; f32 needs >=70% and int8 needs >=50% when those artifacts
   are present.
6. For CLI e2e, if `scripts/smolvlm2_describe_e2e.sh` exists, run it with `sh`
   and the same `FOCR_SMOLVLM2_DIR`. It should emit
   `smolvlm2_describe_e2e/v1` NDJSON, prove missing-model and wrong-family
   negative paths, then check describe and VQA through the real int8 artifact.
7. A pass is C8/C10 evidence when armed and live Beads/source agree, but it
   still does not replace DISC-003, public benchmark evaluation, or A11
   fairness-controlled performance rows.

## OC: OneChart Chart-Data Route and Distribution Boundary

Purpose: handle OneChart D1-D9 work now that chart-data extraction is callable
with a supplied artifact, while keeping distribution, subcommand, and quality
claims scoped.

Steps:

1. Check live Beads first: `bd-3jo6.4.1`, `bd-3jo6.4.2`,
   `bd-3jo6.4.3`, `bd-3jo6.4.5`, and `bd-3jo6.4.9` are closed in current source;
   `20ac599` committed D4 prefill half 1, `2c77d21` committed D4 cached decode
   support, `2769d21` closed `bd-3jo6.4.4`, `0145419` added D5 recognize
   assembly, `2a56c96` closed `bd-3jo6.4.5`, and `e926c46` closed
   D6/D7/D8 plus `bd-3jo6.4`.
2. Inspect `src/native_engine/model_arch.rs` for `ONECHART`: it should be
   `Decoder::OptDense`, `TokenizerKind::Gpt2Bpe`, tasks `[Task::Chart]`,
   tied embeddings/head, `model.vision_tower`, `model.decoder.layers.`, and
   `implemented=true`.
3. For conversion, run the complete shape, or inspect the equivalent source/test
   path: `focr convert /path/to/onechart/model.safetensors -o onechart.int8.focrq
   --quant int8 --model-id onechart --arch generic --json`. Require OPT GEMM
   classification only under `model.decoder.layers.*`:
   `self_attn.{q,k,v,out}_proj` and `fc1`/`fc2`.
4. Require tied-head behavior: `lm_head.weight` must byte-match
   `model.decoder.embed_tokens.weight` and be omitted/deduped in the output.
   An untied OneChart checkpoint should fail closed with `FormatMismatch`.
5. Require current D2 census when citing the real checkpoint: 384 source records
   -> 383 `.focrq` records, 72 int8 GEMMs, 346 MB artifact,
   `model_id=onechart`, correct OneChart Apache-2.0 license, and overflow
   proofs for K=768/K=3072.
6. For tokenization, arm `FOCR_ONECHART_DIR` with `vocab.json`, `merges.txt`,
   and `added_tokens.json`; require `PretokScheme::Gpt2`, no Digits stage, and
   29/29 token-id exact against the slow HF `GPT2Tokenizer` fixtures.
7. Cite special-token pins exactly: `<imgpad>` 50265, `<img>` 50266,
   `</img>` 50267, `<Number>` 50268, bos=eos 2, pad 1.
8. For D3 vision, check `preprocess::onechart_view_tensor`,
   `src/native_engine/onechart.rs::vision_features`,
   `scripts/gen_reference_fixtures_onechart.py`, and
   `tests/fixtures/onechart/oracle_fixtures.json`. Require squash-resized
   1024x1024 raw `[0,1]` RGB input, no CLIP constants,
   `model.vision_tower`, `model.mm_projector` `Linear(1024->768,bias)`,
   `[256,768]` output rows, `onechart_preproc.bin`,
   `onechart_proj_out.bin`, `proj_out cos 1.00000000`, and maxabs `6.5e-4`
   when citing the armed proof.
9. For D4-prefill half 1, check `src/native_engine/decoder_qwen2.rs`,
   `src/native_engine/nn.rs`, `src/native_engine/onechart.rs`, and the
   `opt_prefill_matches_torch_oracle` test. Require `DecoderFamily::Opt`,
   `DecoderConfig::onechart()`, learned absolute positions offset 2, no RoPE,
   OPT pre-LN `LayerNorm` with bias, biased q/k/v/out/fc1/fc2 linears, ReLU
   MLP via `nn::relu`, tied head, q pre-scaling inside the shared attention
   kernel, and `build_inputs_embeds` splicing 256 vision rows into `<imgpad>`
   50265 slots. The armed proof uses `onechart_proj_out.bin` rows plus
   `onechart_final_logits.bin`, prompt length 308, last-position argmax 50268
   (`<Number>`), cos `1.00000000`, and maxabs `6.1e-5`.
10. For D4 cached decode, check `generate_greedy_kvcache` in
   `src/native_engine/decoder_qwen2.rs` and
   `opt_kvcache_matches_greedy_and_oracle` in
   `src/native_engine/onechart.rs`. Require the OPT family path to carry
   family-specific decode weights through `GotDecodeWeights` and `MlpW::ReluFc`,
   use
   `family_norm`, learned absolute positions, no RoPE, OPT output-proj bias,
   final norm bias, OPT ReLU `fc1`/`fc2`, centralized `lm_head`, and the same
   `build_inputs_embeds` splice as D4 prefill. The armed D4-decode proof uses
   oracle projector rows, `FOCR_ONECHART_DIR`, tokenizer files, and weights;
   it compares a 24-token KV-cache greedy stream to O(n^2) re-prefill greedy,
   prefers `onechart.int8.focrq` when present for same-quantization int8
   checking, records a measured 13-step exact prefix at about 320 positions,
   gates prefix >=12, asserts first id 50268 (`<Number>`), and requires
   dict-open decoded output.
   If `onechart.rs` module-level prose still says decoder assembly lands later,
   treat that as stale commentary when committed code/tests and Beads disagree.
11. For D5 native recognition assembly, check `src/native_engine/onechart.rs`
   and the `bd-3jo6.4.5` close. Require `ChartResult`, `recognize`, fixed
   308-id `chart_prompt_ids`, `complete_json_string`, `<Number>` tap through
   `prefill_final_hidden`, `number_head`, `reliable_distance`,
   `recognize_reads_the_committed_chart`, `reliable_check_matches_upstream_goldens`,
   `number_head_matches_golden`, and `chart_prompt_ids_match_oracle_l0c`.
   Treat this as native-module proof, not public `OcrEngine`/CLI support.
12. Treat the `onechart.int8.focrq` preference as committed from `2769d21`
   onward, and as the local artifact name to pass to `--task chart-data` once
   D7 is closed.
13. For D6/D7/D8, check `src/cli.rs`, `src/native_engine/mod.rs`,
   `scripts/onechart_chart_e2e.sh`, `bd-3jo6.4.6`, `.4.7`, `.4.8`, and
   `e926c46`. Require `OcrTask::ChartData`,
   `model_spec_is_knowably_not_onechart`, `forward_onechart`,
   `model_arch implemented=true`, `onechart_chart_e2e/v1`, missing-model
   exit 3, wrong-family exit 2, and a real chart-data run when armed.
14. For scoped quality, check `bd-2lje` / `corpus_quality_scrm_proxy`: six
   in-distribution charts, head fires 6/6, mean distance about 0.015 int8 /
   0.014 f32, byte-identical f32-vs-int8 decoded text, and valid JSON 1/6 in
   both precisions.
15. End with the boundary: `focr ocr --task chart-data --model
   onechart.int8.focrq` is current with a pulled or supplied artifact;
   `focr pull onechart` is current after `bd-av64.7`, but there is no separate
   `focr chart` and no broad quality claim unless corpus/A11 evidence supports
   it.

## TM: TrOMR Local Runtime, Distribution, and Quality Boundary

Purpose: handle TrOMR work without confusing local runtime support, packaged
int8-storage distribution, f32 reference artifacts, and still-open
dewarp/perf/export work with the closed E5/E8/E10 v1 evidence.

Steps:

1. Check live Beads first: `bd-3jo6.5.2` is closed for E2 conversion,
   `bd-3jo6.5.6` is closed for E6 tokenizer, `bd-3jo6.5.3` is closed through
   `45da3a3` E3 encoder proof, `bd-3jo6.5.4` is closed through `3472c1b` E4
   decoder proof, `bd-3jo6.5.7` is closed through `79d715c` E7 merge/MusicXML,
   `bd-3jo6.5.9` is closed through `78a2de3` E9 CLI/runtime,
   `bd-3jo6.5.8` is closed through `2cbded9` for the single-staff ladder,
   `bd-3jo6.5.5` is closed through `752f3cd` for E5 full-page staff detection,
   and `bd-3jo6.5.10` plus sub-epic `bd-3jo6.5` are closed through `ab0bae0`.
2. Inspect `src/native_engine/model_arch.rs`: `TROMR` should be
   `Decoder::Seq2SeqDense`, `TokenizerKind::MusicVocab`, `tasks:
   &[Task::Music]`, `default_artifact_basename: "tromr.focrq"`, and
   `implemented=true`.
3. For the E2 reference artifact, require `scripts/gen_tromr_safetensors.py`
   provenance, WS-folded convs, `model_id=tromr`, 260 tensors, `0 int8`, and
   byte-exact roundtrip against the WS-folded export. For current
   self-conversion, remember `focr convert --model-id tromr` has no f32 mode;
   it is the `--quant int8` storage path after `bd-av64.12`.
4. For tokenizer work, require `src/tokenizer/music.rs`, four decode-only
   WordLevel tables, dense duplicate-free id validation, sizes 260/71/7/2,
   rhythm-only specials, kept pitch/lift/note low ids, out-of-range `""`, and
   `tests/fixtures/tromr/detokenize_goldens.json`.
5. For E3, require both layers of evidence: shared NN leaves
   (`tf_same_pad_amounts`, `tf_same_pad`, `max_pool2d`, `group_norm`, optional
   `fuse_relu`) and the committed encoder (`TromrEncoderW`, WS-prefolded
   ResNetV2 stages `[2,3,7]`, crop-indexed learned positions, four pre-LN ViT
   blocks, `tromr_encoder_matches_torch_oracle`, `tromr_oracle_fixtures.json`,
   `encoder_out cos 1.00000000`, maxabs `3.8e-6`, oracle floor 0.0).
6. For E4, require `TromrDecoderW`, `decoder_forward`, `generate`,
   `MusicStreams`, `FOCR_TROMR_SAMPLE`, `tromr_decoder_matches_argmax_oracle`,
   all four step-0 heads at cos `1.00000000`, maxabs <= `7.6e-6`, and
   42-step x 3-stream token-exact argmax generation. Treat this as decoder
   conformance, not output assembly by itself.
7. For E7, require `merge_semantic`, fail-loud aligned-stream rules,
   `semantic_to_musicxml`, partwise MusicXML 4.0, divisions 64, clef/key/time,
   chord, dotted-duration, rest/multirest, and accidental handling. `**kern` is a
   follow-up unless source shows an emitted surface.
8. For E9, require `preprocess::tromr_staff_tensor`, `tromr::recognize`,
   `MusicResult`, `forward_tromr`, `model_spec_is_knowably_not_tromr`,
   `model_arch implemented=true`, `scripts/tromr_music_e2e.sh`, and
   `tromr_music_e2e/v1` NDJSON. Cite the negative legs: missing model exits 3,
   wrong-family exits 2, real staff emits MusicXML.
9. For E8, require `2cbded9`, `bd-3jo6.5.8`, L0b-L5 ladder evidence, mean SER
   0.211, per-example max 0.375, DISC-004, and e2e 4/4. For broader page
   claims, add E5 source evidence and keep the scope to the v1
   stacked/printed-scanned page cert, not camera-dewarped arbitrary photos. For
   performance, cite `bd-2sez` / `5430e2c` as the f32 baseline row: exact
   token-stream agreement, but focr f32 is slower than pinned upstream torch.
10. For E5, require `fc9d88a`, `src/preprocess/staff_detect.rs`,
    `preprocess::staff_detect`, and `detect_staves` for committed module
    evidence: DISC-004 ink plane, Otsu thresholding, projection-profile deskew,
    five-line grouping, ordered crops, and synthetic tests. Require `752f3cd`,
    `recognize_page`, `staves_to_musicxml`, full-page `forward_tromr` behavior,
    and `tromr_page_detects_and_reads_stacked_examples` before making the
    committed full-page claim; cite detector-lossless SER 0.125 / 0.040 and
    the v1 scope (printed/scanned pages, global deskew only).
11. For E10/sub-epic closure, require `9127676`
    `tromr_alpha_ink_path_fires_only_when_alpha_varies` plus `ab0bae0` close
    evidence: about 25 unit tests, 6 armed certs, 2 NDJSON e2e scripts, and the
    full 891-test `scripts/check.sh` gate.
12. End with the boundary: `focr pull tromr` is current after `bd-av64.7`, and
    `efccce9` / closed `bd-av64.12` makes the default artifact
    `tromr.int8.focrq` storage plus tokenizers while
    `focr pull tromr --quant f32` selects the `tromr.focrq` reference.
    `bd-2sez` supplies the f32 PERF_LEDGER baseline row. There is still no
    standalone `focr music`, no TrOMR int8 compute, no
    camera dewarp/default-barline-quality claim, no TrOMR perf win or int8 perf
    row, and no `**kern` export unless a current source surface emits it. Treat
    `FOCR_TROMR_SPLIT=1` as separate experimental rescue, not as a
    default-quality statement. Current
    `--task music` is dual-lane: TrOMR native MusicXML with
    `--model tromr.int8.focrq` after default pull or `--model tromr.focrq`
    after `focr pull tromr --quant f32`, and GOT format mode with
    `--model got-ocr2.int8.focrq`.

## GB: Gauntlet Baseline

Purpose: keep head-to-head performance evidence honest.

Steps:

1. Use `scripts/gauntlet_runbook.sh` for the quiet-host flow.
2. Run `preflight` before any timed step; respect loadavg and fixture hash
   failures.
3. Keep the same thread budget on focr and the HF/CPU reference.
4. Record `artifacts/perf/bd-re8.17/arch.json` from `robot selftest`.
5. Current source has first closed `bd-re8.17` rows; still cite each stage
   exactly and never summarize it as a universal stage win because `page_0009`
   preprocess is 0.916.

## Operationalized Operator Cards

These cards are the agent-facing form of the operators above. Use them when the
task is ambiguous, high-risk, or likely to cross source/docs/release boundaries.
They deliberately include triggers, failure modes, reusable prompt modules, and
evidence anchors so an agent can select an operator without rereading every
reference file.

### OP-LC: Live Contract Probe

Canonical tag: `OP-LC`

When-to-Use Triggers:

- user reports a command/error that conflicts with README or skill text,
- installed `focr` help lacks a feature that source seems to contain,
- robot schema, CLI help, tests, and docs disagree,
- a claim depends on whether a feature is release-current or source-current.

Failure Modes:

- matching docs to a stale installed binary,
- trusting README prose over `src/cli.rs`/tests,
- treating scaffolded commands as implemented,
- omitting exact source/help/test evidence in the final answer.

Prompt Module:

```text
Use OP-LC. Resolve the live focr contract by checking git status, source command
definitions, robot schema/tests, and binary help if feasible. Classify the
surface as current, release-lagged, stale-binary, scaffolded, or unimplemented.
Report the evidence and the next action.
```

Evidence anchors:

- `src/cli.rs`
- `src/robot.rs`
- `tests/cli_robot_golden.rs`
- `tests/e2e_recognize.rs`
- installed `focr --version` and `focr <cmd> --help`

Exit artifact:

```text
contract: current|release-lagged|stale-binary|scaffolded|unimplemented
evidence: <source/test/help lines or commands>
next: <rebuild|run-source|open-bead|update-skill|tell-user>
```

### OP-MD: Model Discovery

Canonical tag: `OP-MD`

When-to-Use Triggers:

- user asks which model to use,
- a non-default `.focrq` artifact fails to load,
- a task-specific OCR request mentions math, tables, music, charts, molecules,
  or layout-style output,
- model registry status might differ from artifact availability.

Failure Modes:

- treating planned registry rows as ready,
- treating ready registry rows as proof of pullability without checking
  `pull.in_manifest` / `pull.quants`,
- treating a stale binary's embedded manifest as current source truth,
- treating TrOMR's default int8-storage pull as int8 compute or speed proof,
- treating `--manifest` or `FOCR_MANIFEST_URL` as magic distribution support
  instead of a concrete manifest source whose entries must be inspected,
- inventing task subcommands like `focr music`, `focr chart`, or
  `focr describe`,
- forgetting `qwen.tiktoken` for GOT-OCR2,
- treating `--format` as proof of task aliases or broad specialized accuracy.

Prompt Module:

```text
Use OP-MD. Inspect focr models JSON or model_arch registry, then inspect the
artifact model_id/tokenizer needs. Distinguish ready/implemented models from
planned descriptors, manifest-packaged pull artifacts from supplied/converted
artifacts, current GOT `--format` support, SmolVLM2 `describe` routing, OneChart
`chart-data`, TrOMR `music`, and still-missing task subcommands. Use
`pull.in_manifest` and `pull.quants`; if `--manifest` or `FOCR_MANIFEST_URL` is
in play, inspect that exact manifest before saying what `pull` can fetch.
```

Evidence anchors:

- `focr models --json`
- `src/native_engine/model_arch.rs`
- `models/manifest.json`
- `src/dist.rs`
- `docs/zoo/got-ocr2-spec.md`
- `docs/zoo/smolvlm2-spec.md`
- `tests/fixtures/tokenizer_got/expected.json`

Exit artifact:

```text
model: <id>
status: ready|planned|source-only|unknown
manifest_entry: packaged|absent|custom|unknown
pull_quants: <list-or-unknown>
artifact: present|missing|wrong-model-id|unknown
tokenizer: present|not-needed|missing
mode: plain-public|format-mmd|describe-vqa|chart-data|musicxml|not-applicable
next: <pull|convert|copy-tokenizer|wait-for-release|update-integration>
```

### OP-MA: Model Artifact Setup

Canonical tag: `OP-MA`

When-to-Use Triggers:

- deployment or CI needs real OCR instead of schema-only checks,
- `ModelNotFound` or exit code 3 appears,
- an integration is about to download weights inside request handling,
- model setup must work offline after installation.

Failure Modes:

- making first OCR request download the model,
- converting GOT weights but omitting tokenizer placement,
- pulling named zoo models but copying only the `.focrq` and losing sidecars,
- treating TrOMR as int8 because the request default was `--quant int8`,
- failing to record artifact hashes,
- assuming registry readiness means manifest distribution,
- using cache heuristics when explicit `FOCR_MODEL_PATH` is needed.

Prompt Module:

```text
Use OP-MA. Prefer focr pull for packaged artifacts, recording the exact manifest
source (`--manifest`, FOCR_MANIFEST_URL, or built-in), selected quant, cache
subdirectory, and sidecars. Record hashes and source revision for self-converted
artifacts, configure explicit model paths for runtime, then prove inference with
network disabled where possible.
```

Evidence anchors:

- `focr pull`
- `focr pull got-ocr2`
- `focr pull smolvlm2`
- `focr pull onechart`
- `focr pull tromr`
- `focr pull --manifest`
- `src/dist.rs`
- `models/manifest.json`
- `FOCR_MANIFEST_URL`, `FOCR_MODEL_PATH`, `FOCR_MODEL_DIR`, `FOCR_QUANT`

Exit artifact:

```text
artifact: <path>
model_id: <id>
manifest_source: <builtin|env|explicit:path-or-url|not-used>
manifest_entry: <present|absent|custom|unknown>
selected_quant: <int8|f32|unknown>
hash: <sha256-or-not-recorded>
sidecars: <paths|not-needed|missing>
runtime_env: <FOCR_MODEL_PATH-or-FOCR_MODEL_DIR>
offline_smoke: passed|skipped|failed
```

### OP-ZM: Zoo Manifest Pullability

Canonical tag: `OP-ZM`

When-to-Use Triggers:

- user asks whether `focr pull smolvlm2`, `focr pull onechart`, or
  `focr pull tromr` works,
- `focr models` says a model is ready but pull fails,
- docs/source disagree about named model distribution,
- cache sidecars are missing after a named pull,
- a TrOMR answer mentions int8.

Failure Modes:

- using registry readiness as a substitute for manifest pullability,
- ignoring `pull.in_manifest` / `pull.quants`,
- copying a named model out of its cache subdirectory without sidecars,
- reporting the requested quant instead of the selected quant,
- assuming `bd-av64.7` manifest presence by itself proves public release bytes,
- assuming `bd-av64.8` / `bd-av64.9` GitHub clean-cache proof also proves HF
  mirror availability,
- forgetting the one-command pull-e2e script is still deferred.

Prompt Module:

```text
Use OP-ZM. Check the exact binary/source boundary, then inspect focr models
JSON pull.in_manifest/pull.quants and the resolved manifest. For named models,
record artifact name, selected quant, sidecars, and cache subdirectory. For
TrOMR, report the selected quant precisely: default `tromr.int8.focrq` after
efccce9 / closed bd-av64.12, or `tromr.focrq` when
`focr pull tromr --quant f32` is requested.
If making a distribution claim, distinguish committed manifest (`bd-av64.7`),
GitHub release publication (`bd-av64.8`), clean-cache pull plus real inference
(`bd-av64.9`), bd-av64.12 storage-int8 publication, and HF mirror status
(known 401 gap unless live evidence supersedes it).
```

Evidence anchors:

- `focr models --json`
- `models/manifest.json`
- `src/dist.rs` `BUILTIN_MANIFEST_JSON`, `ModelEntry.sidecars`, `select_quant`
- `br show bd-av64.7 --json`
- `br show bd-av64.8 --json`
- `br show bd-av64.9 --json`
- `ece14f9`
- `models-smolvlm2-v1`
- `models-onechart-v1`
- `models-tromr-v1`

Exit artifact:

```text
model: <got-ocr2|smolvlm2|onechart|tromr|other>
pull_in_manifest: true|false|unknown
pull_quants: <list>
selected_quant: <quant-or-not-tested>
sidecars: <list>
cache_dir: <path-or-unknown>
boundary: current|stale-binary|custom-manifest|unknown
github_release: verified|missing|not-checked
clean_cache_pull: pass|fail|not-run
hf_mirror: verified|401-gap|not-checked
```

### OP-OF: Output File Contract

Canonical tag: `OP-OF`

When-to-Use Triggers:

- user asks for markdown/JSON files from `focr ocr`,
- an integration pipes stdout while also requesting `-o`,
- tests need to prove no partial output on failure,
- layout JSON shape matters.

Failure Modes:

- assuming stdout always carries result data when `-o` is present,
- treating extension inference and `--json` as independent outputs,
- writing or asserting partial files after model-load failure,
- binding downstream consumers to unstated JSON fields.

Prompt Module:

```text
Use OP-OF. Exercise or document the -o/--output contract: extension selects
markdown vs JSON, --json forces JSON, stdout stays automation-safe, and failures
must not leave partial destination files.
```

Evidence anchors:

- `src/cli.rs` output plan code
- `tests/e2e_recognize.rs`
- `RecognizedDocument`
- `LayoutSpan`

Exit artifact:

```text
command: <focr ocr ... -o ...>
format: markdown|json
stdout_contract: empty|data|diagnostics-only
failure_atomicity: proven|not-proven
schema_fields: <fields-used>
```

### OP-FX: Figure Extraction Contract

Canonical tag: `OP-FX`

When-to-Use Triggers:

- user asks for extracted figures/regions alongside OCR,
- downstream wants markdown references to saved image crops,
- `--extract-figures` fails without an output path,
- tests need to distinguish zero figures from failure.

Failure Modes:

- asserting every document has at least one figure,
- allowing figure output without an explicit destination,
- writing crops before recognition has succeeded,
- forgetting that JSON figure metadata appears only when figures are present.

Prompt Module:

```text
Use OP-FX. Require -o or --figures-dir, run extraction only after recognition
succeeds, treat zero figures as valid unless the corpus grounds figures, and
verify markdown/JSON references point to valid crop files.
```

Evidence anchors:

- `src/cli.rs` figure plan
- `tests/e2e_recognize.rs`
- `ExtractedFigure`
- `recognize_with_figures`
- `recognize_dynamic_with_figures`

Exit artifact:

```text
destination: output-derived|figures-dir|missing
figures_count: <n|unknown>
references: valid|not-checked|not-present
failure_cleanup: proven|not-proven
```

### OP-PS: PDF Page Selection and Spread Splitting

Canonical tag: `OP-PS`

When-to-Use Triggers:

- user asks to OCR only certain PDF pages,
- a scanned book spread should become left/right logical pages,
- JSON or robot consumers need page provenance,
- a scanned PDF is rendered sideways despite looking upright in a viewer,
- `--split-spreads` interacts with `--extract-figures`.

Failure Modes:

- applying `--pages` or `--split-spreads` to `ocr-batch`,
- treating split-spread as an OCR quality guarantee,
- losing original page and half metadata in downstream JSON/robot handling,
- forgetting that committed source now applies page `/Rotate` plus axis-aligned
  content-stream image-placement rotation before OCR,
- combining `--split-spreads` with `--extract-figures`; current source refuses
  that combination because figure naming across split halves has no contract yet.

Prompt Module:

```text
Use OP-PS. Verify the exact binary/source supports PDF --pages,
--split-spreads, and content-stream rotation normalization. Treat page specs as
1-based PDF selectors and spread splitting as a heuristic extraction step.
Preserve original page and left/right half metadata. Treat bd-av64.11 as
closed-current when source/help/tests agree, while still quarantining stale
binaries that predate 11f60ea/9546571/5679268/b3f74b6.
```

Evidence anchors:

- `src/cli.rs` `OcrRequestArgs.pages` / `split_spreads`
- `src/pdf.rs` `split_spread`, `content_rotation`, `apply_rotation`
- `README.md` PDF examples
- `br show bd-av64.11 --json`
- robot/page JSON tests when present

Exit artifact:

```text
pages_spec: <spec|all|invalid>
split_spreads: true|false
source_support: present|absent|stale-binary
tracker_status: closed|open|in_progress|unknown
rotation_normalization: page-rotate|content-ctm|both|absent|unknown
metadata: original-page-and-half|flat|not-checked
boundary: extraction-heuristic|accepted-contract|unknown
```

### OP-OE: One Engine

Canonical tag: `OP-OE`

When-to-Use Triggers:

- integrating `franken_ocr` into another Rust service,
- a review finds `OcrEngine::new()` inside request handlers,
- async host code wants concurrent OCR calls,
- multi-page or multi-image work needs throughput.

Failure Modes:

- building one engine per request,
- nesting runtime ownership inside an existing async task without a blocking
  boundary,
- adding outer page parallelism over a model whose kernels already own fanout,
- treating the current process-global shutdown flag as per-request cancellation for
  concurrent tenants,
- sizing outer concurrency from `logical_cpus` instead of `FOCR_THREADS` /
  physical-core `thread_budget`,
- bypassing batch APIs.

Prompt Module:

```text
Use OP-OE. Keep one long-lived OcrEngine in application state, call it through a
blocking boundary from async hosts, use model-path explicit APIs for deployment,
use batch APIs rather than ad hoc outer page parallelism, and in current
bd-223.2 source, keep request_shutdown/cancel_checkpoint/thread_budget scoped to
process-global cooperative shutdown and host-wide capacity.
```

Evidence anchors:

- `src/lib.rs`
- `OcrEngine::recognize_with_model`
- `OcrEngine::recognize_batch`
- `request_shutdown`
- `cancel_checkpoint`
- `thread_budget`
- `FOCR_THREADS`
- `stream_pages`
- `FOCR_BATCH_SPINE`
- `FOCR_BATCH_SIZE`

Exit artifact:

```text
engine_lifetime: singleton|per-request|unknown
async_boundary: blocking|unsafe-direct|not-applicable
model_path: explicit|env|implicit-cache
batch_strategy: batch-api|manual-parallel|sequential
cancellation: none|process-global|per-request|unknown
thread_budget: focr_threads|physical-cores|logical-cpus|unknown
```

### OP-PQ: Parity First

Canonical tag: `OP-PQ`

When-to-Use Triggers:

- any optimization, quantization, SIMD, batch, or decode lever is proposed,
- output changes but timing improves,
- a feature is justified by expected speed rather than proof,
- a lossy path might be enabled by env var.

Failure Modes:

- benchmarking before freezing baseline behavior,
- averaging away a single catastrophic page,
- accepting throughput gains without token/CER proof,
- failing to record rejected levers.

Prompt Module:

```text
Use OP-PQ. Establish baseline behavior and metrics, change one lever, run the
right parity/CER/golden gate, keep only passing changes, and record divergence
or negative evidence with fallback state.
```

Evidence anchors:

- `docs/DISCREPANCIES.md`
- `docs/NEGATIVE_EVIDENCE.md`
- `docs/PERF_LEDGER.md`
- `tests/parity_ladder.rs`
- `scripts/got_cer.py`

Exit artifact:

```text
lever: <name>
baseline: <output/metric>
after: <output/metric>
gate: passed|failed|skipped
fallback: <env-or-code-path>
ledger: <path-or-missing>
```

### OP-PDF: PDF Boundary

Canonical tag: `OP-PDF`

When-to-Use Triggers:

- user passes a PDF to CLI or library integration,
- `InputDecode` mentions unsupported filters or vector pages,
- downstream wants a PDF support claim,
- OCR parity depends on rasterization.

Failure Modes:

- saying all PDFs are supported,
- rasterizing out of band without recording DPI/tool,
- treating born-digital text/vector pages as OCR-ready scanned pages,
- losing page-order or image-resolution evidence.

Prompt Module:

```text
Use OP-PDF. For supported scanned PDFs, use focr's native path. For unsupported
codecs, vector pages, or born-digital pages, rasterize out of band, record the
tool/DPI, and retry on images while preserving page order.
```

Evidence anchors:

- `src/pdf.rs`
- `franken_ocr::pdf::PdfPages`
- `OcrEngine::recognize_dynamic`
- `JPXDecode`
- `JBIG2Decode`

Exit artifact:

```text
pdf_kind: scanned-supported|unsupported-codec|vector-or-born-digital|unknown
page_count: <n|unknown>
rasterization: native|external|needed
evidence: <error-or-command>
```

### OP-RP: Robot Purity

Canonical tag: `OP-RP`

When-to-Use Triggers:

- shell automation consumes `focr robot`,
- stdout contains human decoration,
- schema/event names are being documented,
- a regression affects exit codes or NDJSON.

Failure Modes:

- mixing human messages into robot stdout,
- forgetting to validate every NDJSON line,
- scraping human mode instead of robot schema,
- ignoring non-zero exit code semantics,
- treating `threads` as a stable golden integer instead of a host-dependent
  value to scrub.

Prompt Module:

```text
Use OP-RP. Validate robot schema and every robot run line as JSON/NDJSON, keep
stdout data-only, route diagnostics to stderr, and bind consumers to versioned
schema fields rather than human prose. In current bd-223.2 source, assert
`threads` in robot health/backends and scrub it in goldens.
```

Evidence anchors:

- `src/robot.rs`
- `ROBOT_SCHEMA_VERSION`
- `tests/fixtures/robot_schema_v1.json`
- `tests/cli_robot_golden.rs`
- `src/error.rs`
- `thread_budget`
- `tests/fixtures/golden/robot_backends.golden`

Exit artifact:

```text
schema_version: <n>
stdout: json-only|mixed|not-checked
events_checked: yes|no
exit_code: <code|not-checked>
consumer_contract: schema|human-prose|unknown
threads_field: present|absent|not-applicable
```

### OP-RT: Robot Triage and Agent Ergonomics

Canonical tag: `OP-RT`

When-to-Use Triggers:

- an agent needs a one-command orientation to local focr state,
- user asks what command to run next,
- automation should choose between pulling models and running OCR,
- a CLI ergonomics review mentions stdout purity, actionable errors, or
  did-you-mean behavior.

Failure Modes:

- using several human help probes when `focr robot triage` answers the state,
- ignoring `recommendations[0]`,
- recommending OCR before model setup on an empty cache,
- treating empty history as failure,
- mixing stderr decoration into JSON stdout.

Prompt Module:

```text
Use OP-RT. Run focr robot triage and parse its single JSON object. Report
quick_ref, top recommendation, relevant commands, and exit-code guidance. If
the command is absent, classify the binary as pre-bd-wp8.7 or stale rather than
inventing a replacement schema.
```

Evidence anchors:

- `src/cli.rs` `robot_triage_payload`
- `tests/agent_ergonomics_regression.rs`
- `docs/ergonomics/AUDIT.md`
- `br show bd-wp8.7 --json`
- `055e513`

Exit artifact:

```text
robot_triage: present|absent|invalid
quick_ref: <summary-or-missing>
top_recommendation: <summary-or-none>
commands: <count-or-unknown>
exit_codes: present|missing|not-checked
stdout_purity: pass|fail|not-checked
```

### OP-BG: Backend and SIMD Claim Guard

Canonical tag: `OP-BG`

When-to-Use Triggers:

- user asks which CPU/backend/SIMD tier focr is using,
- `FOCR_FORCE_ARCH`, `robot backends`, or `robot selftest` appears,
- `robot selftest.models`, `got-ocr2:overflow_k2816`,
  `smolvlm2:overflow_k2560`, or `onechart:overflow_k3072` appears,
- a README/support answer might imply AMX, SMMLA, AVX-VNNI, or speedup,
- a perf claim lacks `docs/PERF_LEDGER.md` or gauntlet evidence.

Failure Modes:

- using `robot selftest` as benchmark evidence,
- treating `robot selftest.models` as OCR model-quality, throughput, or TrOMR
  int8 evidence,
- advertising AMX before current `robot backends` reports it,
- claiming SMMLA is always faster than SDOT on Apple Silicon,
- treating an unsupported forced tier as proof of hardware support,
- recording `logical_cpus` but omitting current `threads`.

Prompt Module:

```text
Use OP-BG. First classify the claim as capability, parity, or performance.
For capability, inspect focr robot backends and report selected/available tiers,
override_env=FOCR_FORCE_ARCH, logical_cpus, and threads. For parity, run focr
robot selftest, optionally forcing scalar/sdot/smmla/avx2/avxvnni/avx512vnni
when that tier is available. At/after ad3ad20 and adb4ee6, inspect the selftest
models rollup too: it summarizes per-decoder int8 parity for unlimited-ocr,
got-ocr2, smolvlm2, and onechart from underlying case rows, including
model-specific overflow rows. Closed bd-3jo6.1.12 evidence reports 44/44 cases
green across scalar, sdot, and smmla. TrOMR is intentionally absent because its
published int8 artifact is storage-only and runtime dequants through f32
accessors rather than using an int8 decoder kernel. If the claim is about focr
convert --arch or offline SMMLA panels, switch to OP-AP. For performance,
switch to OP-GB and require gauntlet/PERF_LEDGER evidence. Keep Apple
SDOT-vs-SMMLA, x86 VNNI, and AMX boundaries exact.
```

Evidence anchors:

- `src/simd/dispatch.rs`
- `src/simd/dispatch.rs`
  `selftest_reports_a_per_model_verdict_for_every_registered_decoder`
- `tests/cli_robot_golden.rs`
  `robot_selftest_proves_per_model_kernel_parity_e2e`
- `src/simd/arm.rs`
- `tests/batched_igemm_parity.rs`
- `src/cli.rs` `robot_backends_payload`
- `src/cli.rs` `run_robot_selftest`
- `docs/PERF_LEDGER.md`
- `README.md` CPU backend section
- `br show bd-3jo6.1.12 --json`
- `adb4ee6`

Exit artifact:

```text
claim_type: capability|parity|performance
selected_tier: <tier|unknown>
available_tiers: <tiers|unknown>
force_arch: unset|scalar|sdot|smmla|avx2|avxvnni|avx512vnni|unsupported
threads: <n|unknown>
selftest: pass|fail|not-run
model_rollup: present|absent|not-checked
model_rollup_ids: <ids|unknown>
perf_basis: none|perf-ledger|fresh-gauntlet
claim_boundary: <one sentence>
```

### OP-AP: Arch-Specific Prepack Boundary

Canonical tag: `OP-AP`

When-to-Use Triggers:

- `focr convert --arch`, `aarch64-smmla`, `arch_target`, `WeightLayout`,
  `SmmlaPanels`, `src/simd/pack.rs`, or offline packing appears,
- someone asks whether a `.focrq` is optimized for a CPU tier,
- a support answer wants to claim SMMLA, VNNI, AMX, zero-shuffle, or a speedup,
- `bd-2mo.3`, `bd-2mo.3.1`, `fdbaaec`, `9989bc0`, or `5c64547` appears.

Failure Modes:

- treating `--arch` as a quantization policy change rather than a storage layout
  target,
- claiming packed-consuming x86 kernels exist for VNNI/AMX because their tags
  are accepted,
- treating offline SMMLA panels as an M-series speed win; Apple Silicon still
  usually selects SDOT at runtime,
- failing to mention that non-SMMLA hosts un-permute with a warning/fallback,
- mixing row-major bytes and SMMLA panel bytes in a parity proof,
- calling closed `bd-2mo.3` proof that the parent P3 performance epic is done.

Prompt Module:

```text
Use OP-AP. Inspect the converter, format, loader, and dispatch layers before
making an arch-pack claim. The current closed boundary is: focr convert --arch
aarch64-smmla emits real offline [2x8] SMMLA panels through src/simd/pack.rs;
the .focrq preamble/header records arch_target; QInt8 weights can load as
WeightLayout::SmmlaPanels; SMMLA dispatch consumes the panels without a runtime
shuffle; non-SMMLA routes un-permute loudly and keep correctness. VNNI/AMX are
tag-only until a packed-consuming x86 kernel exists. Row-major remains the AVX2
zero-shuffle layout. For speed, switch to OP-GB and require PERF_LEDGER or a
fresh gauntlet row; do not infer throughput from packing correctness.
```

Evidence anchors:

- `src/quant/convert.rs` `--arch aarch64-smmla` path
- `src/quant/focrq.rs` `arch_target`
- `src/simd/pack.rs`
- `src/native_engine/tensor.rs` `WeightLayout::SmmlaPanels`
- `src/native_engine/weights.rs` loader fallback/warning behavior
- `src/native_engine/decoder.rs`
  `smmla_panel_layout_is_byte_identical_through_the_gemv_paths`
- `src/native_engine/decoder.rs`
  `fuse_qkv_concatenates_smmla_panels_losslessly`
- `docs/FEATURE_PARITY.md`
- `br show bd-2mo.3 --json`
- `br show bd-2mo.3.1 --json`
- `fdbaaec`, `9989bc0`, `5c64547`

Exit artifact:

```text
arch_target: generic|aarch64-smmla|x86-vnni|x86-amx|unknown
packed_layout: row-major|smmla-panels|tag-only|unknown
loader_route: consume-packed|unpermute-warning|row-major|not-checked
parity_basis: pack-unpack|gemv-bit-identical|decoder-layout|not-run
x86_packed_kernel: present|absent|not-checked
perf_basis: none|perf-ledger|fresh-gauntlet
claim_boundary: <one sentence>
```

### OP-RS: Run Store and Sync

Canonical tag: `OP-RS`

When-to-Use Triggers:

- user asks about `focr runs`, `focr sync`, run history, telemetry, or audit
  export/import,
- tests or scripts must avoid writing to the user's real run database,
- a binary returns `NotImplemented` for `runs` / `sync` while source contains
  `src/storage.rs`,
- a support answer needs to explain `FOCR_RUN_STORE`, `_meta.schema_version`,
  or JSONL sync failure behavior.

Failure Modes:

- treating old installed-binary `runs` / `sync` stubs as proof that current
  source lacks the closed `bd-223.4` feature,
- lumping `runs` / `sync` and `doctor` into one phase bucket instead of using
  OP-RS for storage and OP-DR for repair,
- writing tests against `~/.cache/franken_ocr/runs.db` instead of a temp
  `FOCR_RUN_STORE`,
- using human plain output as the machine contract instead of JSON/NDJSON,
- treating store-write failure as OCR failure when source records telemetry
  best-effort,
- ignoring stale `.jsonl.lock` / partial `.jsonl.tmp` evidence when debugging
  sync.

Prompt Module:

```text
Use OP-RS. Probe source/help/tests for RunStore, FOCR_RUN_STORE, focr runs, and
focr sync export-jsonl/import-jsonl; then check br show bd-223.4 --json. If the
tracker is closed and source/help/tests agree, classify it as closed-current.
If a binary still returns NotImplemented or omits flags, classify that binary as
stale/release-lagged before changing docs. For automation, set FOCR_RUN_STORE
to a temp runs.db, prefer --format json or ndjson, and verify export/import
JSONL records plus lock/temp-file behavior. Keep doctor separate: it may still
has its own implemented repair contract and exit-code sub-contract.
```

Evidence anchors:

- `src/storage.rs`
- `src/cli.rs::run_runs`
- `src/cli.rs::run_sync`
- `tests/cli_robot_golden.rs::runs_and_sync_args_obey_exit_categories`
- `FOCR_RUN_STORE`
- `RunStore`
- `RunRecord`
- `_meta`
- `SCHEMA_VERSION`
- `focr runs --format json`
- `focr sync export-jsonl --file ... --json`
- `focr sync import-jsonl --file ... --json`
- `bd-223.4`

Exit artifact:

```text
run_store: absent|scaffolded|live-wip|closed-current|stale-binary
tracker: open|in_progress|closed|unknown
store_path: temp|default|custom|unknown
schema_version: <n|unknown>
record_fields: complete|partial|not-checked
sync_contract: export|import|both|absent|not-checked
telemetry_failure: best-effort|run-fatal|unknown
```

### OP-DR: Doctor Repair Contract

Canonical tag: `OP-DR`

When-to-Use Triggers:

- user asks to diagnose or repair focr cache/model/install state,
- a support flow mentions `focr doctor`,
- automation needs machine-readable doctor capabilities,
- a previous repair must be undone.

Failure Modes:

- calling current doctor scaffolded because an old binary is installed,
- running `--fix` when the user only asked for diagnosis,
- bypassing the doctor mutation ledger with manual cache edits,
- ignoring `.doctor/lock` / exit code 5,
- treating doctor exit codes as the global OCR exit-code table.

Prompt Module:

```text
Use OP-DR. Start with focr doctor --json or capabilities --json. Use
--dry-run --fix before mutation, use --fix only with user intent, record the
run id/actions ledger, and use doctor undo for rollback. Classify scaffolded
doctor output as stale binary/source if current bd-wp8.4 source is present.
```

Evidence anchors:

- `src/doctor.rs`
- `src/cli.rs::run_doctor`
- `tests/doctor_fixtures.rs`
- `br show bd-wp8.4 --json`
- `br show bd-wp8.4.1 --json`
- `25eadc5`

Exit artifact:

```text
doctor_contract: current|stale-scaffold|missing|unknown
mode: detect|dry-run|fix|undo|capabilities|robot-docs
findings: <count-or-unknown>
mutations: none|planned|applied|rolled-back|unknown
run_id: <id-or-none>
exit_code: <0|1|2|3|4|5|6|unknown>
```

### OP-VG: Verification Gate Reality

Canonical tag: `OP-VG`

When-to-Use Triggers:

- user asks whether a model/test/robot contract is "proven",
- support text cites `bd-re8.7`, `bd-zc1o`, `bd-n68o`, or `bd-29wv`,
- a model-gated test skipped but someone wants to call the feature green,
- docs need to distinguish harness infrastructure from corpus/perf completion.

Failure Modes:

- treating skip-with-SUCCESS as success-path OCR proof,
- omitting `native_path_ran` and `fallback_target` from e2e evidence,
- calling a frozen schema fixture a generated-output quality proof,
- assuming L5 fixture CER covers every model, document type, or performance row,
- trusting stale worktree/session evidence over current `main`.

Prompt Module:

```text
Use OP-VG. Recheck live Beads/source for the relevant gate, then bind the claim
to the exact artifact: tests/fixtures/robot_schema_v1.json for bd-zc1o,
docs/TEST_LOGGING.md plus tests/fixtures/test_log_schema.json for bd-n68o,
tests/common/model_gate.rs and native_path_ran/fallback_target discipline for
bd-29wv, or parity_ladder L5 fixture/CER evidence for bd-re8.7. If robot
EVENT_KINDS advertises staff, verify whether the checkout is at/after adb4ee6:
that source refresh includes staff in the frozen schema fixture and
advertised-events assertion. If bd-wp8.2.2 still appears open, treat it as a
possible stale tracker mismatch until focused schema tests prove otherwise.
State what the gate proves and what remains outside scope.
```

Evidence anchors:

- `tests/cli_robot_golden.rs`
- `tests/fixtures/robot_schema_v1.json`
- `docs/TEST_LOGGING.md`
- `tests/fixtures/test_log_schema.json`
- `docs/testing/LOGGING_AND_E2E.md`
- `tests/e2e_recognize.rs`
- `tests/parity_ladder.rs`
- `br show bd-zc1o --json`
- `br show bd-wp8.2.2 --json`
- `br show bd-n68o --json`
- `br show bd-29wv --json`
- `br show bd-re8.7 --json`

Exit artifact:

```text
gate: robot-schema|test-log|model-gated-e2e|l5-parity
bead_status: closed|open|unknown
artifact: <path>
native_path: ran|skipped|not-applicable|unknown
metric: <schema|jsonl|cer|none>
scope_boundary: <one sentence>
```

### OP-DG: Determinism and Fixture Governance

Canonical tag: `OP-DG`

When-to-Use Triggers:

- user asks whether focr output is deterministic,
- a test or review mentions `bd-3kge`, `bd-2pgf`, `assert_deterministic`,
  fixture provenance, `MANIFEST.toml`, or `UPDATE_GOLDENS`,
- a fixture family is added, regenerated, or moved off-tree,
- someone treats same-input byte identity as an OCR quality claim.

Failure Modes:

- conflating our-engine determinism with the oracle nondeterminism envelope,
- treating byte-identical output as correctness against the oracle,
- adding top-level fixtures without `tests/fixtures/MANIFEST.toml`,
- forgetting that `regenerated-committed` entries need an existing generator
  script,
- letting CI auto-update goldens instead of requiring reviewed diffs.

Prompt Module:

```text
Use OP-DG. For determinism, verify tests/support/parity_harness.rs exposes
assert_deterministic/assert_outputs_deterministic and that the target path emits
parity/token_exact evidence; cite e2e adoption only when recognize() or the CLI
path is run twice. For fixture work, update tests/fixtures/PROVENANCE.md and
tests/fixtures/MANIFEST.toml together, then run scripts/check_fixture_manifest.py
or scripts/check.sh. State explicitly that determinism and fixture policy are
infrastructure gates, not standalone OCR-quality proof.
```

Evidence anchors:

- `tests/support/parity_harness.rs`
- `tests/e2e_recognize.rs`
- `tests/fixtures/PROVENANCE.md`
- `tests/fixtures/MANIFEST.toml`
- `scripts/check_fixture_manifest.py`
- `scripts/check.sh`
- `br show bd-3kge --json`
- `br show bd-2pgf --json`

Exit artifact:

```text
determinism_helper: present|absent|not-checked
determinism_adoption: e2e-recognize|cli|unit|none|unknown
fixture_manifest: pass|fail|not-run
provenance_policy: committed|regenerated-committed|off-tree|unknown
claim_boundary: <one sentence>
```

### OP-CM: Conformance Matrix and XFAIL Discipline

Canonical tag: `OP-CM`

When-to-Use Triggers:

- user asks whether focr is conformant, release-ready, or spec-covered,
- a test/review mentions `bd-re8.12`, `ConformanceTest`,
  `conformance_registry`, `tests/conformance_matrix.rs`, `SPEC-NNN`,
  MUST coverage, XFAIL, or `DISC-NNN`,
- someone cites a coverage ratio as if it were OCR quality, benchmark, or
  release certification,
- new source/test code needs a spec-accounting citation.

Failure Modes:

- computing coverage from the test list instead of from the spec,
- treating a `SPEC-NNN` citation as proof that a model output is correct,
- letting partial coverage round up to present/conformant,
- allowing bare XFAIL emissions that hide unledgered debt,
- conflating this closed accounting bead with differential, metamorphic,
  conformal, e-process, or three-pillar release-certification beads that have
  their own live statuses.

Prompt Module:

```text
Use OP-CM. Verify src/conformance.rs defines ConformanceTest with
name/category/requirement_level/clauses/run and that
tests/conformance_matrix.rs enumerates SPEC-NNN clauses from
docs/truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md, not from the test list.
Run or cite the matrix only as accounting evidence: MUST coverage >= 0.95,
per-clause NDJSON, no bare XFAIL without DISC-NNN or a stated phase gap, and
conformance_registry entries that run green in-process. State explicitly which
release/quality beads remain separate.
```

Evidence anchors:

- `src/conformance.rs`
- `tests/conformance_matrix.rs`
- `docs/truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md`
- `docs/DISCREPANCIES.md`
- `docs/conformance/PARITY_LADDER.md`
- `docs/conformance/LADDER_HARNESS.md`
- `docs/FEATURE_PARITY.md`
- `br show bd-re8.12 --json`
- `br show bd-re8.9 --json`
- `br show bd-re8.10 --json`
- `br show bd-re8.13 --json`

Exit artifact:

```text
bd_re8_12: closed|open|unknown
spec_clause_source: spec|test-list|unknown
must_coverage: <ratio|pass|fail|not-run>
xfail_sites: ledgered|violations|not-run
registry_entries: pass|fail|not-run
outside_scope: <open beads or remaining proof>
```

### OP-DF: Differential Oracle Comparator

Canonical tag: `OP-DF`

When-to-Use Triggers:

- user asks whether focr matches the bf16/HF/Python oracle,
- a review mentions `bd-re8.9`, `differential_per_op_vs_bf16_oracle`,
  `differential_row`, L3/L4/L5 parity, ULP tables, `EngineIdentity`, or
  `DISC-NNN`,
- someone wants to accept a numeric or token drift as "close enough",
- an e2e result is model-gated and the distinction between skipped and native
  path ran matters.

Failure Modes:

- accidentally comparing the oracle to itself and producing a false green,
- replacing ULP/tolerance tables with a prose/text-only comparator,
- treating an accepted intentional divergence as a skip instead of an XFAIL
  with a `DISC-NNN` ledger entry,
- citing a model-gated skip-with-SUCCESS as if real native inference ran,
- collapsing per-op differential evidence into a broad product-quality claim.

Prompt Module:

```text
Use OP-DF. Inspect tests/parity_ladder.rs for
differential_per_op_vs_bf16_oracle, differential_row, EngineIdentity, ULP table
use, and L3-L5 tolerance boundaries. Require subject != oracle identity, row
fields scope/oracle/module/max_diff/within_tol/xfail/disc, and DISC-NNN for any
intentional drift. If artifacts are missing, report skip-with-SUCCESS as
unarmed rather than green e2e evidence.
```

Evidence anchors:

- `tests/parity_ladder.rs`
- `tests/parity_ladder.rs::differential_per_op_vs_bf16_oracle`
- `tests/parity_ladder.rs::differential_row`
- `tests/support/parity_harness.rs`
- `docs/conformance/PARITY_LADDER.md`
- `docs/conformance/LADDER_HARNESS.md`
- `docs/DISCREPANCIES.md`
- `br show bd-re8.9 --json`

Exit artifact:

```text
bd_re8_9: closed|open|unknown
oracle_identity: subject-not-oracle|oracle-vs-oracle-risk|unknown
row_contract: complete|missing-fields|not-checked
tolerance_source: ulp-table|ladder|ad-hoc|unknown
xfail_discipline: ledgered|violations|not-run
native_path: ran|skipped-missing-artifact|not-applicable|unknown
claim_boundary: per-op|l3-l5|e2e|not-proven
```

### OP-MR: Metamorphic Relations

Canonical tag: `OP-MR`

When-to-Use Triggers:

- user asks for oracle-free consistency, robustness, transform invariance, or
  relation-based test evidence,
- a review mentions `bd-re8.10`, `tests/metamorphic.rs`, MR-1 through MR-5,
  identity resize, rotation bbox mapping, padding, thread determinism, or
  multi-page behavior,
- someone proposes a new invariant without checking whether model semantics
  actually require it.

Failure Modes:

- adding a false concat/sum relation for multi-page output; R-SWA makes
  multi-page parsing cross-page dependent,
- hard-coding a different pad color instead of using `preprocess::PAD_FILL`,
- turning white-pad SHOULD evidence into a strict must-pass assertion,
- treating model-gated MR-2-live or MR-5 legs as absent or green,
- omitting the `FOCR_THREADS=1` vs `4` axis when claiming determinism.

Prompt Module:

```text
Use OP-MR. Inspect tests/metamorphic.rs and docs/conformance/METAMORPHIC.md.
Report strict relations separately from SHOULD and gated relations: MR-1
identity resize strict text, MR-2 rotation coordinate math plus model-gated live
bbox proof, MR-3a mean-gray PAD_FILL strict text, MR-4 same-thread and
FOCR_THREADS=1 vs 4 determinism, MR-3b white-pad observed/logged, and MR-5
cross-page dependence gated. Explicitly state that the concat/sum page relation
is invalid for R-SWA multi-page output.
```

Evidence anchors:

- `tests/metamorphic.rs`
- `docs/conformance/METAMORPHIC.md`
- `src/preprocess/mod.rs`
- `tests/support/parity_harness.rs`
- `docs/DISCREPANCIES.md`
- `br show bd-re8.10 --json`

Exit artifact:

```text
bd_re8_10: closed|open|unknown
strict_relations: pass|fail|not-run
should_relations: logged|missing|not-run
gated_relations: honest|misreported|not-checked
thread_axis: covered|missing|not-run
false_relation_guard: pass|fail|not-checked
claim_boundary: self-consistency|oracle-quality|not-proven
```

### OP-MP: Multi-Page Cross-Page Parsing

Canonical tag: `OP-MP`

When-to-Use Triggers:

- user mentions `--multi-page`, `infer_multi`, `recognize_multi_page`,
  `recognize_multi_page_dynamic`, `<PAGE>`, cross-page parsing, page context,
  `bd-1gv.25`, `bd-2z0y`, or multi-page PDFs,
- docs or code compare multi-page output with individual page parses,
- a robot/streaming parser wants page events for a multi-page document,
- someone tries to combine PDF `--multi-page` with `--split-spreads`,
  `--extract-figures`, or a non-Unlimited model.

Failure Modes:

- treating `--multi-page` as a batch loop over independent pages,
- asserting equality with concatenated single-page outputs,
- promising per-page layout boxes, figure extraction, split-spread behavior, or
  streaming `page` events from the current multi-page route,
- forgetting that image lists use `ocr-batch --multi-page` and PDFs use
  `ocr --multi-page`,
- ignoring the 32K context guard and advising one pass over an arbitrarily large
  book,
- applying the Unlimited-OCR multi-page contract to GOT-OCR2, SmolVLM2,
  OneChart, or TrOMR without fresh model-specific proof.

Prompt Module:

```text
Use OP-MP. First classify the input: image list, PDF, or library call. For image
lists, cite focr ocr-batch page1.png page2.png --multi-page and the batch JSON
shape command=batch.multi_page/pages/seconds/markdown. For PDFs, cite focr ocr
doc.pdf --multi-page, note that it composes with --pages but refuses
--split-spreads and --extract-figures, and note that PDF JSON has no per-page
layout boxes for this route. For library use, cite OcrEngine::recognize_multi_page
or recognize_multi_page_dynamic(_with_model), and check whether the streaming
route `recognize_multi_page_dynamic_streaming(_with_model)` is in play. Then
verify source/tests: recognize_multi_page uses `preprocess_dynamic_squash` to
produce one 640x640 PIL-bicubic-squashed page tensor per page, 111 placeholders
per page, one cross-page prompt/decode, ngram_window=1024,
MAX_POSITION_EMBEDDINGS guard, and finalize_multi/PageStream <PAGE> separators.
End with the bd-2z0y/bd-1gv.26/bd-1465 boundary: PDF multi-page and robot
decoded page progress events are shipped, the 2-page L5 oracle rung is shipped,
the 10-page long-horizon rung is shipped with CER 0.4045 <= 0.50, plate exact,
markers 8-vs-9, and a 7600-token true-prefix cap, and the 20-page frozen oracle
fixture shows reference-model collapse. Do not infer a meaningful 40-page CER
gate or arbitrary-long-document quality claim from that evidence.
```

Evidence anchors:

- `src/native_engine/mod.rs` `recognize_multi_page`, `build_prompt_multi`,
  `recognize_multi_page_dynamic_streaming`
- `src/native_engine/postprocess.rs` `PageStream`
- `src/preprocess/mod.rs` `preprocess_dynamic_squash`,
  `multi_page_base_640_placeholder_is_111`
- `src/robot.rs` `page_decoded_event`
- `src/lib.rs` `OcrEngine::recognize_multi_page`
- `src/lib.rs` `recognize_multi_page_dynamic_with_model`
- `src/cli.rs` `ocr-batch --multi-page`
- `src/cli.rs` `recognize_pdf_multi_page`
- `tests/e2e_recognize.rs`
  `recognize_multi_page_real_model_when_present_else_skip_with_success`
- `tests/e2e_recognize.rs`
  `multi_page_streaming_matches_terminal_assembly_when_armed`
- `tests/parity_ladder.rs` `l5_multi_page_matches_infer_multi_oracle`
- `tests/parity_ladder.rs` `l5_multi_page_10p_long_horizon`
- `tests/fixtures/multi_page/p10_raw.md`
- `tests/fixtures/multi_page/p20_raw.md`
- `tests/cli_robot_golden.rs`
  `batch_multi_page_flag_routes_to_the_cross_page_pass`
- `tests/metamorphic.rs`
- `README.md`
- `br show bd-1gv.25 --json`
- `br show bd-2z0y --json`
- `br show bd-1gv.26 --json`
- `br show bd-1465 --json`
- `4afcaca`, `f115403`, `b9cc16c`, `a2dd1c9`, `750a69a`, `828ea4c`,
  `727701b`, `6e297f6`, `c6ab897`, `3201e8c`, `e1332a7`

Exit artifact:

```text
multi_page_surface: ocr-batch-images|ocr-pdf|library|absent|stale-binary
model_scope: unlimited-ocr-only|model-specific-proof|unknown
input_count: <n-or-unknown>
preprocess: squash-640-pil-bicubic-111-placeholders|unknown
context_guard: pass|too-large|not-checked
output_shape: markdown-page-separators|batch-json|pdf-json-empty-layout|unknown
streaming_page_events: decoded-progress|absent|not-checked
claim_boundary: <one sentence>
```

### OP-GA: Golden Artifact Discipline

Canonical tag: `OP-GA`

When-to-Use Triggers:

- user changes or reviews CLI help, robot NDJSON, schema JSON, reference
  outputs, numeric activation/logit artifacts, or committed golden fixtures,
- a test prints a golden mismatch, writes `.actual`, or mentions
  `UPDATE_GOLDENS=1`,
- a cross-platform output needs scrubbing or canonicalization,
- someone wants to refresh snapshots because the diff is inconvenient.

Failure Modes:

- auto-blessing goldens, especially in CI,
- comparing volatile robot NDJSON exactly without scrubbing version, CPU/SIMD,
  path, or host-count fields,
- using fuzzy comparison for CLI/help/schema surfaces that should be exact,
- committing `.actual` or `.snap.new` review artifacts,
- changing goldens without restamping provenance or naming the intentional
  surface change.

Prompt Module:

```text
Use OP-GA. Classify each artifact as exact, fuzzy ULP, scrubbed robot NDJSON,
canonicalized cross-platform output, or reference-output canonicalized exact.
Inspect tests/cli_robot_golden.rs, tests/fixtures/golden/PROVENANCE.md, and
docs/conformance/GOLDEN.md. On mismatch, require human review of the .actual
diff before UPDATE_GOLDENS=1, keep CI update flags forbidden, keep transient
files gitignored, and restamp provenance when committed goldens change.
```

Evidence anchors:

- `tests/cli_robot_golden.rs`
- `tests/fixtures/golden/PROVENANCE.md`
- `tests/fixtures/golden/`
- `tests/fixtures/robot_schema_v1.json`
- `tests/support/parity_harness.rs`
- `docs/conformance/GOLDEN.md`
- `br show bd-re8.11 --json`

Exit artifact:

```text
bd_re8_11: closed|open|unknown
artifact_pattern: exact|fuzzy|scrubbed|canonicalized|reference-output|unknown
golden_diff: clean|needs-human-review|not-run
update_guard: ci-forbidden|manual-only|unsafe|unknown
transient_outputs: ignored|committed-risk|not-checked
provenance: current|stale|missing|not-checked
claim_boundary: surface-freeze|numeric-envelope|quality-proof|not-proven
```

### OP-LS: Ladder Scorecard Runner

Canonical tag: `OP-LS`

When-to-Use Triggers:

- user asks whether the L0-L5 ladder is all green or wants a per-commit parity
  receipt,
- a review mentions `bd-re8.19`, `scripts/ladder_scorecard.sh`,
  `focr-ladder-scorecard/v1`, `not_meaningful`, `all_green`, or
  `skipped_no_model`,
- a perf/release/conformance step needs one ordered artifact instead of six
  independent rung logs,
- a lower-rung failure should prevent noisy higher-rung interpretation.

Failure Modes:

- treating a scorecard summary as a replacement for the underlying parity rows,
- treating `skipped_no_model=true` as green ladder evidence,
- ignoring `not_meaningful` annotations after the first hard failure,
- treating `bd-re8.19` closure as proof that every future ladder run is green,
- running the heavy ladder without naming model/fixture paths and build cost.

Prompt Module:

```text
Use OP-LS. Verify scripts/ladder_scorecard.sh exists in committed source
(1b84428 initial runner; 1112cf8 plus bd-re8.19 close evidence make the bead
closed-current). For parser confidence use scripts/ladder_scorecard.sh
--self-test. For a real receipt, run scripts/ladder_scorecard.sh [--out FILE]
only when cargo/model budget allows; it runs parity_ladder serially and folds
event=parity plus event=result rows into focr-ladder-scorecard/v1. Report
all_green, skipped_no_model, first hard failure, and not_meaningful downstream
gates. Do not treat unarmed skipped_no_model=true as a green ladder.
```

Evidence anchors:

- `scripts/ladder_scorecard.sh`
- `tests/parity_ladder.rs`
- `docs/conformance/PARITY_LADDER.md`
- `docs/conformance/LADDER_HARNESS.md`
- `tests/support/parity_harness.rs`
- `br show bd-re8.19 --json`
- `git show 1b84428 -- scripts/ladder_scorecard.sh`
- `git show 1112cf8 -- tests/fixtures/ladder_scorecard`

Exit artifact:

```text
bd_re8_19: open|closed|unknown
runner_source: present|absent|not-checked
self_test: pass|fail|not-run
scorecard_schema: focr-ladder-scorecard/v1|unknown
armed_close: all-six-green|not-checked|not-applicable
all_green: true|false|not-run
skipped_no_model: true|false|not-run
not_meaningful_boundary: <gate-or-none-or-unknown>
claim_boundary: ordered-receipt|underlying-parity-proof|not-proven
```

### OP-SG: Release Ship Gate

Canonical tag: `OP-SG`

When-to-Use Triggers:

- user asks whether franken_ocr/focr is release-ready or shippable,
- a PR/release note says "all green",
- `docs/gauntlet/RELEASE_READINESS.json` or `bd-wp8.10` appears,
- a clean gauntlet round is being treated as complete convergence.

Failure Modes:

- treating old scorecard machinery as ship approval before checking the current
  artifact and Beads,
- treating the conformal ratchet, Ville e-process monitors, or capacity
  certificate as ship approval by themselves,
- treating an old `ship:false` artifact as current after the `c29a78b` /
  `7c7bd00` / `29516b9` certification,
- collapsing `bd-wp8.8` convergence, certification bundle, and capstone status,
- using the L0-L5 ladder scorecard as the whole release gate,
- claiming release capstone closure without live Beads proof.

Prompt Module:

```text
Use OP-SG. Run or inspect scripts/gauntlet_cert.py --release-readiness, read
docs/gauntlet/RELEASE_READINESS.json, and check br show bd-wp8.10 --json. If
ship is false or any cell is red, say not shippable and name the blocking cells.
For bundle questions, run or inspect scripts/gauntlet_cert.py --bundle, read
docs/gauntlet/bundle/release_certificate.json, and check br show bd-wp8.9
--json. Current `c29a78b` / `7c7bd00` / `29516b9` evidence is certified and
published as `v0.6.0`: readiness reports ship:true, green:13, red:0;
release_certificate.json reports certified:true at v0.5.2-8-gc4c1684 because
the bundle was generated at `c4c1684`; convergence is rounds=11/10 with
tail_clean=True; and bd-wp8.8, bd-wp8.9, and bd-wp8.10 are closed. `beaed7c`
adds CI/dist supplement notes, `db02421` refreshes README evidence, and
`5df6395` commits post-certification fuzz corpus growth. `592426c` refreshes
README public `v0.6.0` release identity/asset-size/backend prose. Keep
convergence and
certification bundle evidence separate from the mere existence of the
scorecard/bundle scripts, and keep the ship gate separate from installed-binary
version and unrelated open epics. For the supporting
instruments, inspect
docs/conformance/RATCHET.md for the conformal ratchet, use
scripts/gauntlet_cert.py --eprocess-fold ... --eprocess-state ... for Ville
e-process invariant folding, and use
cargo test --test many_pages_without_deadlock
capacity_certificate_bounded_stream_soak -- --nocapture for the bounded-stream
capacity certificate. If FOCR_FIXTURES_DIR or another armed fixture root is
missing, report unarmed evidence rather than green release proof.
```

Evidence anchors:

- `scripts/gauntlet_cert.py --release-readiness`
- `scripts/gauntlet_cert.py --bundle`
- `scripts/gauntlet_cert.py --eprocess-fold`
- `docs/gauntlet/RELEASE_READINESS.json`
- `docs/gauntlet/bundle/release_certificate.json`
- `docs/conformance/RATCHET.md`
- `capacity_certificate_bounded_stream_soak`
- `FOCR_FIXTURES_DIR`
- `br show bd-wp8.10 --json`
- `br show bd-wp8.9 --json`
- `br show bd-wp8.8 --json`
- `2bdccc5`
- `c29a78b`
- `7c7bd00`

Exit artifact:

```text
release_artifact: present|absent|not-run
schema: franken_ocr.release_readiness.v1|unknown
ship: true|false|unknown
red_cells: <list>
capstone: open|closed|unknown
claim_boundary: ship-gate-green|machinery-present-not-shippable|not-proven
supporting_instruments: ratchet|eprocess|capacity-certificate|mixed|none|unknown
```

### OP-SQ: Stale Binary Quarantine

Canonical tag: `OP-SQ`

When-to-Use Triggers:

- installed `focr` reports older version or missing help,
- installer fallback may have landed an older release,
- target-dir binary differs from source,
- user reproduces a bug that source tests already cover.

Failure Modes:

- editing docs to match stale binary behavior,
- failing to name the exact binary path/version,
- assuming raw `main` installer implies installed release capabilities,
- not distinguishing source-current from release-current.

Prompt Module:

```text
Use OP-SQ. Identify the exact focr binary and version, compare it to current
source/help/tests, classify release lag or stale local build, and only then
decide whether to rebuild, install, patch docs, or patch source.
```

Evidence anchors:

- `focr --version`
- `which focr`
- `target/release/focr --version`
- `CHANGELOG.md`
- `install.sh`
- `install.ps1`

Exit artifact:

```text
binary: <path>
version: <version-or-unknown>
source_head: <sha-or-unknown>
classification: current|release-lag|stale-local|unknown
next: <reinstall|rebuild|patch-source|patch-docs|tell-user>
```

### OP-LQ: Lossy Lever Quarantine

Canonical tag: `OP-LQ`

When-to-Use Triggers:

- env vars like `FOCR_INT8_ATTN`, `FOCR_INT8_KV`, `FOCR_ATTN_GEMM`,
  `FOCR_INT8_LMHEAD`, `FOCR_LMHEAD_INT4`, `FOCR_SPEC_DECODE`, or
  `FOCR_SPEC_VERIFY` enter a proposed workflow,
  including measured-negative levers such as `FOCR_FUSE_NGRAM_LMHEAD`,
- a speedup affects attention, KV, lm_head, sampler, or quantization,
- behavior looks visually close but is not exact,
- a fallback/kill-switch is missing.

Failure Modes:

- calling a lossy lever production-safe from microbenchmarks,
- omitting model hash or corpus,
- failing to test the hardest pages,
- forgetting to document the fallback env or revert path.
- reviving a ledgered negative result as a "maybe free" optimization without
  satisfying its do-not-retry predicate.

Prompt Module:

```text
Use OP-LQ. Treat lossy or speculative levers as quarantined until the evidence
packet includes model hash, corpus, metric/budget, exact env vars, fallback,
command transcript, and ledger/Beads anchor. Leave the lever off if any field is
missing.
```

Evidence anchors:

- `docs/NEGATIVE_EVIDENCE.md`
- `docs/DISCREPANCIES.md`
- `artifacts/perf/`
- `FOCR_GOT_INT8_LMHEAD`
- `FOCR_GOT_SEQ_ATTN`
- `FOCR_FUSE_NGRAM_LMHEAD`
- `FOCR_SPEC_DECODE`
- `FOCR_SPEC_VERIFY`
- `bd-2mo.24`
- `fused_ngram_lmhead_is_byte_identical_to_separate_mask`

Exit artifact:

```text
lever: <env-or-code-path>
category: lossless|lossy|speculative|unknown
evidence_packet: complete|incomplete
fallback: <env/code-path-or-missing>
recommendation: enable|keep-off|revert|needs-bead
```

### OP-TI: Tracker-Informed Claim

Canonical tag: `OP-TI`

When-to-Use Triggers:

- user asks if a capability is done,
- Beads mention a feature not obvious in source,
- source has code but tests/proofs may be missing,
- a release note or README claim feels ahead of implementation.

Failure Modes:

- replacing source inspection with tracker optimism,
- running bare `bv` and blocking in a TUI,
- forgetting closed beads can still leave follow-ups,
- omitting open blockers from the answer.

Prompt Module:

```text
Use OP-TI. Search source, tests, br JSON, bv robot output, and CASS if useful.
Then classify the capability as implemented-and-tested, implemented-not-proven,
scaffolded, planned, blocked, or stale-binary, with exact evidence.
```

Evidence anchors:

- `br list --json`
- `br show <id>`
- `bv --robot-triage`
- `cass search ... --robot`
- source/tests/docs paths relevant to the surface

Exit artifact:

```text
claim: <capability>
classification: implemented-tested|implemented-not-proven|scaffolded|planned|blocked|stale-binary
source_evidence: <paths>
tracker_evidence: <beads-or-none>
remaining_risk: <gap-or-none>
```

### OP-TR: Task Routing

Canonical tag: `OP-TR`

When-to-Use Triggers:

- user asks for formulas, tables, charts, molecular output, geometry, or music,
- user asks for photo description or VQA through SmolVLM2,
- a caller tries to use GOT-OCR2 but gets plain OCR,
- a docs/API task says `focr music`, `focr chart`, `focr describe`, or `--task`,
- library integration wants task selection.

Failure Modes:

- treating task-specific subcommands as shipped,
- using a specialized task without a GOT model,
- using `--task describe` without a SmolVLM2 artifact,
- assuming CLI `--task` is a Rust library request enum,
- presenting smoke-tested format mode as fully budgeted accuracy proof.

Prompt Module:

```text
Use OP-TR. Classify the requested output as plain OCR or specialized GOT output.
For specialized output, require got-ocr2.int8.focrq plus qwen.tiktoken and use
focr ocr --task <task> --model got-ocr2.int8.focrq or --format. In Rust, use an
explicit GOT model path and a format-mode process policy; do not invent subcommands
or per-call task APIs. For photo description/VQA, require a supplied or
pulled/converted smolvlm2.int8.focrq artifact and use focr ocr --task describe
--model smolvlm2.int8.focrq [--question "..."], with DISC-003/C8/C10 caveats.
For chart-data, use pulled/converted onechart.int8.focrq and keep quality
caveats. For TrOMR music, default pull gives `tromr.int8.focrq` storage with
f32 dequant-on-access; use `focr pull tromr --quant f32` or local `tromr.focrq`
for byte-exact reference debugging.
```

Evidence anchors:

- `src/cli.rs` `OcrTask`
- `src/native_engine/model_arch.rs`
- `bd-3jo6.1.5`
- `bd-3kix`
- `focr models --json`

Exit artifact:

```text
task: <ocr|formula|tables|chart|molecular|geometry|music|describe|chart-data>
model_required: <unlimited-ocr|got-ocr2|smolvlm2|onechart|tromr>
mode: plain|format-mmd|describe-vqa|chart-data|musicxml|not-implemented
command_or_api: <exact surface>
accuracy_proof: smoke|budgeted|missing
```

### OP-RR: Reference Resampler

Canonical tag: `OP-RR`

When-to-Use Triggers:

- L0/preprocess parity drifts against a Pillow/torch oracle,
- screenshots differ only after resize/pad,
- a user asks whether `FOCR_RESAMPLE=pil-bicubic` should be enabled,
- a model-zoo port reuses PIL bicubic or LANCZOS preprocessing.

Failure Modes:

- mistaking the reference-comparison mode for a production quality default,
- comparing CatmullRom output to a PIL oracle without DISC-001 context,
- failing to record the env var polarity,
- applying `FOCR_RESAMPLE=pil-bicubic` to SmolVLM2 even though C7 uses fixed
  LANCZOS (`resample: 1`) through `resize_lanczos`.

Prompt Module:

```text
Use OP-RR. Run the default path and FOCR_RESAMPLE=pil-bicubic, then classify the
delta as DISC-001 resampler divergence, another preprocessing bug, or downstream
decode drift. Keep CatmullRom as default unless target-corpus e2e numbers justify
changing policy. If the model is SmolVLM2, switch to OP-SP because its
preprocessor is Pillow LANCZOS, not the Baidu/GOT BICUBIC reference mode.
```

Evidence anchors:

- `bd-30me`
- `bd-3jo6.3.7`
- `docs/DISCREPANCIES.md` DISC-001
- `src/preprocess/pil_resample.rs`
- `scripts/gen_pil_bicubic_goldens.py`

Exit artifact:

```text
resampler: default-catmullrom|pil-bicubic|smolvlm2-lanczos
comparison: l0-exact|l0-drift|e2e-drift
classification: disc-001|bug|unproven
recommendation: keep-default|use-for-oracle|needs-bead
```

### OP-BS: Batch Spine Proof

Canonical tag: `OP-BS`

When-to-Use Triggers:

- user asks for batch throughput or `FOCR_BATCH_SPINE`,
- output differs between sequential batch and spine batch,
- deadlock/oversubscription concerns appear,
- a batch run toggles `FOCR_BATCH_VISION`.

Failure Modes:

- citing throughput without recording `FOCR_BATCH_SPINE` and `FOCR_BATCH_SIZE`,
- running outer parallelism around the engine,
- forgetting `FOCR_BATCH_VISION` is only read inside the armed spine,
- treating batch-wide setup errors as per-image OCR failures,
- treating old "GOT falls back to sequential under the spine env" evidence as
  current after `cf0b037`,
- treating `bd-3jo6.1.7.5` dense batched decode closure as broad batched
  `lm_head`, a future-zoo route, or a fairness-controlled `ocr-batch`
  throughput proof,
- confusing `FOCR_BATCH_SIZE` with live `FOCR_THREADS`: one is batch in-flight
  width, the other is the host-wide physical-core budget.

Prompt Module:

```text
Use OP-BS. Establish sequential ocr-batch output hashes, arm FOCR_BATCH_SPINE=1
with an explicit FOCR_BATCH_SIZE, keep or deliberately kill-switch FOCR_BATCH_VISION,
and compare per-image outputs in input order. Confirm FOCR_BATCH_SPINE=0 really
keeps the sequential path if testing controls. Cite bd-1azu.10/.14 and
CONTROL_CORRECTION.md for default Unlimited-OCR spine evidence, but rerun on
the target corpus before performance claims. In current bd-223.2 source, record
FOCR_THREADS / thread_budget separately from FOCR_BATCH_SIZE and logical_cpus.
If the task mentions dense batched decode, classify bd-3jo6.1.7.5 separately:
after 8497080 it has gemm_i8_bias_prequant_batched, BatchedQwen2KvCache,
qwen2_batched_decode_step, DenseDecoderBatchStep, and generate_greedy_batched;
after cf0b037 GOT ocr-batch routes through recognize_batch_dense_got and
got::recognize_batch under FOCR_BATCH_SPINE; after 4ca1577
OcrModel::recognize_batch_dense routes got-ocr2|smolvlm2|onechart with
smolvlm2::recognize_batch, onechart::recognize_batch,
generate_greedy_batched taking caps: &[usize], PageStream::with_max_emit,
DEFAULT_BATCH_SIZE 128 / MAX_BATCH_SIZE 256, and FOCR_BATCH_PACK admission docs;
fdd1d64 closes the bead for v0.4.0. That proves lossless dense zoo batching, not
fully batched lm_head or a final fairness-controlled throughput claim. Cite the
scoped speed rows only as scoped rows: SmolVLM2 1.32x, OneChart 1.27x, GOT about
+3% to +16% on vision-dominated fixtures. For new performance claims, still run
the target corpus and record env, model artifact, page set, CPU tier, and raw
timings.
```

Evidence anchors:

- `bd-1azu.10`
- `bd-1azu.14`
- `src/native_engine/batch_scheduler.rs`
- `tests/many_pages_without_deadlock.rs`
- `scripts/spine_watchdog_sweep.sh`
- `artifacts/perf/bd-1azu.14/CONTROL_CORRECTION.md`
- `bd-3jo6.1.7.5`
- `8497080`
- `cf0b037`
- `4ca1577`
- `fdd1d64`
- `53f6581`
- `gemm_i8_bias_prequant_batched`
- `BatchedQwen2KvCache`
- `qwen2_batched_decode_step`
- `DenseDecoderBatchStep`
- `generate_greedy_batched`
- `PageStream::with_max_emit`
- `recognize_batch_dense_got`
- `recognize_batch_dense`
- `smolvlm2::recognize_batch`
- `onechart::recognize_batch`
- `max_emit`
- `DEFAULT_BATCH_SIZE`
- `MAX_BATCH_SIZE`
- `FOCR_BATCH_PACK`
- `got::recognize_batch`

Exit artifact:

```text
batch_spine: off|on
batch_size: <n>
threads: <n|unknown>
model_id: <id|unknown>
batch_vision: default-on|off
batch_pack: off|on|unknown
arch_route: spine|sequential-arch-dispatch
parity: byte-identical|drift|not-run
watchdog: green|not-run|failed
batched_decode_spine: absent|source-present-in-progress|dirty-wip|closed|unknown
batched_decode_proof: bit-identical-per-stream|throughput|not-checked
per_stream_caps: absent|present|unknown
```

### OP-PP: Preprocess Proof

Canonical tag: `OP-PP`

When-to-Use Triggers:

- a user asks whether `--base-size`, `--image-size`, or `--crop-mode` works,
- README/source/Beads disagree about Gundam certification,
- OCR output changes after a preprocess flag,
- a port wants to make Gundam the default or cite parity for it.

Failure Modes:

- repeating stale "parsed but ignored" wording after `bd-1e9n`,
- treating first Gundam e2e as a full corpus L0-L5 pass,
- changing the default away from `base` without parity evidence,
- ignoring view count or CER when comparing tiled and base runs.

Prompt Module:

```text
Use OP-PP. Inspect current source for PreprocessOverrides, run or cite bd-1e9n
evidence for flag liveness, then separate default-base certification from
Gundam first-e2e evidence. Do not make target-corpus parity claims without a
fresh run.
```

Evidence anchors:

- `src/cli.rs` `preprocess_overrides_from`
- `src/native_engine/mod.rs` `PreprocessOverrides`
- `artifacts/perf/bd-1e9n/validation_summary.txt`
- `br show bd-1e9n --json`
- `docs/truth-pack/oq/preprocess-infer.md`

Exit artifact:

```text
preprocess_surface: base-size|image-size|crop-mode|gundam
source_wired: yes|no|unknown
e2e_evidence: bd-1e9n|target-run|missing
default_changed: yes|no
claim_level: flag-live|first-e2e|full-parity|unproven
```

### OP-SD: Spec Decode Gate

Canonical tag: `OP-SD`

When-to-Use Triggers:

- `FOCR_SPEC_DECODE` or `FOCR_SPEC_VERIFY` appears in a workflow,
- a speedup claim depends on speculative decoding,
- ON and OFF outputs differ,
- the environment might contain rejected attention/KV levers.

Failure Modes:

- setting `FOCR_SPEC_DECODE=0` and assuming the lever is off,
- certifying a run while `FOCR_ATTN_GEMM` or `FOCR_INT8_KV` is present,
- treating unit proof as model-gated e2e proof,
- tolerancing token drift instead of reverting or gating the lever.

Prompt Module:

```text
Use OP-SD. Run or cite the two-process FOCR_SPEC_DECODE ON/OFF gate, remove the
env for OFF, verify output hashes and determinism, and reject any composition
with known rejected key-batch levers.
```

Evidence anchors:

- `scripts/spec_gate_e2e.sh`
- `artifacts/perf/bd-1azu.36/`
- `br show bd-1azu.36 --json`
- `src/native_engine/spec.rs`
- `src/native_engine/mod.rs` `SPEC_DECODE_ENV`

Exit artifact:

```text
spec_decode: off|on|ab-tested
off_arm: env-removed|unclear
on_off_hashes: identical|different|not-run
determinism: pass|not-run|failed
blocked_levers_present: yes|no
recommendation: keep-off|allow-opt-in|revert|needs-investigation
```

### OP-SP: SmolVLM2 Preprocess and Prompt/IO

Canonical tag: `OP-SP`

When-to-Use Triggers:

- a task mentions SmolVLM2 C7/C9, prompt/IO, `preprocess_smolvlm2`,
  `preprocess_smolvlm2_path`, `resize_lanczos`, `SmolVLMImageProcessor`,
  `src/native_engine/smolvlm2.rs`, `--question`, `FOCR_SMOLVLM2_QUESTION`,
  LANCZOS, `resample: 1`, frame order, or image-slot counts,
- an L0/input-prep drift appears before `vision_siglip.rs`,
- a docs update might imply C7/C9 route proof is sufficient for C8/C10
  quality/perf.

Failure Modes:

- using BICUBIC or CatmullRom for SmolVLM2 when the oracle says LANCZOS,
- assuming `FOCR_RESAMPLE=pil-bicubic` controls SmolVLM2 preprocessing,
- omitting the global 512 frame or changing row-major local frame order,
- using C7 preprocessing proof as if it were the C8/C10 quality/perf
  certificate,
- treating `model_arch implemented=true` as a corpus-quality/perf certificate,
- changing prompt/image expansion without the ST tokenizer proof.

Prompt Module:

```text
Use OP-SP. Classify SmolVLM2 C7/C9 from live Beads, help, and source, then
verify preprocess_smolvlm2, Pillow-exact resize_lanczos LANCZOS resample: 1,
longest-side 2048, 512-frame local tiles plus global frame, tokenizer-backed
prompt/image-slot metadata, and smolvlm2.rs route/question handling. End with
implemented describe/VQA route, DISC-003 C8 L4 near-tie ledger, C8/C10 close
evidence, and any remaining manifest/A11/perf boundaries.
```

Evidence anchors:

- `src/preprocess/mod.rs` `preprocess_smolvlm2`
- `src/preprocess/pil_resample.rs` `resize_lanczos`
- `src/native_engine/smolvlm2.rs`
- `src/cli.rs` `--question`
- `FOCR_SMOLVLM2_QUESTION`
- `FOCR_SMOLVLM2_DIR`
- `tests/fixtures/smolvlm2/sample_photo.png`
- `br show bd-3jo6.3.7 --json`
- `br show bd-3jo6.3.8 --json`
- `Pillow 12.3.0` LANCZOS goldens, seed 301466

Exit artifact:

```text
bead_status: C7=<status> C8=<status> C9=<status> C10=<status>
resampler: LANCZOS|BICUBIC|CatmullRom|unknown
resize_proof: pillow-golden-pass|oracle-pass|skip|fail|not-run
frames: local=<n-or-unknown> global=<0-or-1-or-unknown>
prompt_slots: <n_image_slots-or-unknown>
forward_status: implemented|stale-binary|preprocess-current|unknown
near_tie_ledger: DISC-003|missing|not-checked
recommendation: keep-narrow|fix-preprocess|rerun-tokenizer|cite-or-rerun-c8-c10
```

### OP-VQ: SmolVLM2 VQA Quality Guard

Canonical tag: `OP-VQ`

When-to-Use Triggers:

- a task mentions SmolVLM2 C8, OQ-6, L5 VQA quality, oracle-answer fixtures,
  `vqa_quality_matches_oracle_l5`, or `scripts/gen_smolvlm2_vqa_fixtures.py`,
- a task mentions C10 e2e, `scripts/smolvlm2_describe_e2e.sh`, or
  `smolvlm2_describe_e2e/v1` NDJSON,
- `tests/fixtures/smolvlm2/vqa_fixtures.json` changes,
- `src/native_engine/smolvlm2.rs` generation or answer scoring changes,
- a docs or release note might overstate the VQA guard as full certification.

Failure Modes:

- treating oracle-answer parity as a public VQA benchmark or human-label score,
- interpreting skipped missing artifacts as a proof,
- lowering the 70% f32 or 50% int8 floor to make a regression pass,
- omitting one weight leg even though the corresponding artifact is present,
- treating the e2e script's skip-with-success as a real model proof,
- mixing the Rust oracle-answer guard with the CLI e2e script and losing which
  one supplied evidence,
- forgetting that the fixture answers are greedy text over one committed sample
  photo, not a broad captioning corpus.

Prompt Module:

```text
Use OP-VQ. Confirm the live C8/C10 Beads state, inspect the VQA fixture and
generator, arm FOCR_SMOLVLM2_DIR with tokenizer plus f32 and/or int8 artifacts,
run vqa_quality_matches_oracle_l5, and if present run
sh scripts/smolvlm2_describe_e2e.sh for CLI e2e. Report the f32/int8 scores,
NDJSON script result, and negative-path results separately. Do not call either a
public benchmark or human-label score; cite C8/C10 closure only from live Beads.
```

Evidence anchors:

- `scripts/gen_smolvlm2_vqa_fixtures.py`
- `scripts/smolvlm2_describe_e2e.sh`
- `smolvlm2_describe_e2e/v1`
- `tests/fixtures/smolvlm2/vqa_fixtures.json`
- `src/native_engine/smolvlm2.rs` `vqa_quality_matches_oracle_l5`
- `tests/fixtures/smolvlm2/sample_photo.png`
- `FOCR_SMOLVLM2_DIR`
- `model.safetensors`
- `smolvlm2.int8.focrq`
- `br show bd-3jo6.3.8 --json`
- `OQ-6`
- normalized exact match
- symmetric content-word containment >= 0.5
- negative path: `/nonexistent` model exit 3
- negative path: wrong-family model exit 2

Exit artifact:

```text
c8_status: closed|open|unknown
c10_status: closed|open|unknown
fixture: present|missing|regenerated
artifacts: f32|int8|both|missing
f32_score: <n>/<total>|not-run
int8_score: <n>/<total>|not-run
e2e_script: pass|skip|fail|not-present|not-run
negative_paths: pass|fail|not-run
interpretation: scoped-pass|regression|unarmed|needs-beads-check|needs-a11
```

### OP-OC: OneChart Chart-Data Route and Distribution Boundary

Canonical tag: `OP-OC`

When-to-Use Triggers:

- a task mentions OneChart, `onechart`, D2, D3, D4, D5, D6, D7, D8, D9, OPT, chart-data
  extraction, `focr convert --model-id onechart`, `FOCR_ONECHART_DIR`,
  `vocab.json`, `merges.txt`, `added_tokens.json`, `PretokScheme::Gpt2`,
  `onechart_view_tensor`, `vision_features`, `model.vision_tower`,
  `mm_projector`, `DecoderFamily::Opt`, `build_inputs_embeds`,
  `opt_prefill_matches_torch_oracle`, `onechart_final_logits.bin`,
  `generate_greedy_kvcache`, `opt_kvcache_matches_greedy_and_oracle`, or
  `tests/fixtures/tokenizer_onechart/expected.json`,
- a task mentions `bd-3jo6.4.5`, `2a56c96`, `0145419`, `ChartResult`,
  `complete_json_string`, `prefill_final_hidden`,
  `recognize_reads_the_committed_chart`,
  `reliable_check_matches_upstream_goldens`, `number_head_matches_golden`, or
  `chart_prompt_ids_match_oracle_l0c`,
- a task asks about `--task chart-data`, `OcrTask::ChartData`,
  `forward_onechart`, or `onechart implemented=true`,
- a `.focrq` says `model_id=onechart`,
- a docs or release note might imply OneChart is ready because conversion or
  tokenizer conformance is ready,
- a live checkout contains `scripts/gen_reference_fixtures_onechart.py`,
  `tests/fixtures/onechart/*`, `DecoderConfig::onechart`, or OPT prefill code.

Failure Modes:

- treating the unsuffixed OneChart artifact name as preferred after the int8
  artifact convention landed; prefer `onechart.int8.focrq`,
- treating D3 vision/projector seam proof as decoder, number-head, JSON/CSV, or
  CLI proof,
- treating committed D4-prefill half 1 as number-head, structured chart
  JSON/CSV, CLI, e2e, parity, or product proof,
- treating closed D4 cached decode support as D5 native assembly, D6 parity/perf,
  D7 public routing, D8 e2e/logging, or product proof,
- treating route/pull support as `focr chart` support,
- treating route support or `bd-2lje` as a broad chart-quality benchmark,
- copying a pulled OneChart `.focrq` without `vocab.json`, `merges.txt`, and
  `added_tokens.json`,
- using Qwen tiktoken, SentencePiece, or SmolLM2 Digits instead of OPT GPT-2
  byte-level BPE,
- using CLIP mean/std normalization for OneChart instead of raw `[0,1]` pixels,
- trusting older 309-token prose over the live `prompt_n` fixture value 308,
- quantizing `num_decoder.*`, projector, vision tower, norms, biases, or the
  tied high-precision head,
- failing to reject an untied OneChart checkpoint,
- citing untracked or modified future decoder/runtime WIP as committed source
  truth.

Prompt Module:

```text
Use OP-OC. Confirm live Beads for bd-3jo6.4.1/.4.2/.4.3/.4.4/.4.5/.4.6/.4.7/.4.8/.4.9
and bd-3jo6.4. Confirm current committed source, not just README prose:
model_arch ONECHART is implemented=true, src/cli.rs has OcrTask::ChartData and
model_spec_is_knowably_not_onechart, and native_engine/mod.rs has
forward_onechart. Validate a full conversion shape such as `focr convert
/path/to/onechart/model.safetensors -o onechart.int8.focrq --quant int8
--model-id onechart --arch generic --json` with the 72-GEMM OPT census and
tied-head dedup.
Validate the D9 GPT-2 tokenizer over vocab.json/merges.txt/added_tokens.json
with 29/29 token-id exact. Validate D3 as a vision/projector seam:
onechart_view_tensor raw [0,1], model.vision_tower, mm_projector
Linear(1024->768,bias), onechart_proj_out.bin, proj_out cos 1.00000000,
maxabs 6.5e-4. Validate D4-prefill as prefill only: DecoderFamily::Opt,
DecoderConfig::onechart, nn::relu, build_inputs_embeds, onechart_final_logits.bin,
argmax 50268, cos 1.00000000, maxabs 6.1e-5, prompt length 308. Validate D4
cached decode as source/test proof only: generate_greedy_kvcache,
opt_kvcache_matches_greedy_and_oracle, GotDecodeWeights, family_norm, learned
positions, no RoPE, output-proj/final-norm bias, 24-token KV-cache vs re-prefill
greedy, same-int8 onechart.int8.focrq preference, measured 13-step exact prefix,
gate >=12, first id 50268, and dict-open decoded output. Validate D5 as
native-module assembly: ChartResult, recognize, complete_json_string,
prefill_final_hidden, number_head, reliable_distance,
chart_prompt_ids_match_oracle_l0c, recognize_reads_the_committed_chart,
reliable_check_matches_upstream_goldens, and number_head_matches_golden.
Validate D6-D8 as public route proof: onechart_chart_e2e/v1, missing-model exit
3, wrong-family exit 2, real chart-data run, and model_arch implemented=true.
Validate bd-2lje as scoped corpus evidence, not a broad benchmark. Validate
bd-av64.7/ece14f9 for distribution: focr pull onechart installs
onechart.int8.focrq plus vocab.json, merges.txt, and added_tokens.json. End by
saying whether the evidence is conversion, tokenizer, D3 vision seam, D4
decoder closure, D5 native assembly, D6-D8 public route, scoped quality corpus,
or distribution; if the claim is broad chart-quality, say that remains a
separate corpus/evidence question. Current runtime chart-data is callable with a
pulled or supplied artifact, while focr chart is not.
```

Evidence anchors:

- `src/native_engine/model_arch.rs` `ONECHART`
- `src/preprocess/mod.rs` `onechart_view_tensor`
- `src/native_engine/onechart.rs` `vision_features`, `build_inputs_embeds`,
  `ChartResult`, `recognize`, `complete_json_string`, `number_head`,
  `reliable_distance`, `opt_prefill_matches_torch_oracle`,
  `opt_kvcache_matches_greedy_and_oracle`, `recognize_reads_the_committed_chart`,
  `reliable_check_matches_upstream_goldens`, `number_head_matches_golden`,
  `chart_prompt_ids_match_oracle_l0c`
- `src/native_engine/decoder_qwen2.rs` `DecoderFamily::Opt`,
  `DecoderConfig::onechart`, `generate_greedy_kvcache`, `prefill_final_hidden`,
  `GotDecodeWeights`, `family_norm`
- `src/cli.rs` absence/presence of `OcrTask::ChartData`
- `src/native_engine/mod.rs` absence/presence of `forward_onechart`
- `src/native_engine/nn.rs` `relu`
- `src/quant/convert.rs` `onechart_convert_dedups_tied_head_and_tags_arch`
- `src/tokenizer/mod.rs` `Tokenizer::from_opt_dir`
- `src/tokenizer/pretok.rs` `pretokenize_gpt2`
- `scripts/gen_reference_fixtures_onechart.py`
- `docs/zoo/onechart-spec.md`
- `tests/fixtures/tokenizer_onechart/expected.json`
- `tests/fixtures/onechart/oracle_fixtures.json`
- `FOCR_ONECHART_DIR`
- `model.safetensors`
- `onechart.int8.focrq`
- `onechart_preproc.bin`
- `onechart_proj_out.bin`
- `onechart_final_logits.bin`
- `vocab.json`
- `merges.txt`
- `added_tokens.json`
- `br show bd-3jo6.4.2 --json`
- `br show bd-3jo6.4.3 --json`
- `br show bd-3jo6.4.4 --json`
- `br show bd-3jo6.4.5 --json`
- `br show bd-3jo6.4.6 --json`
- `br show bd-3jo6.4.7 --json`
- `br show bd-3jo6.4.8 --json`
- `br show bd-3jo6.4.9 --json`
- `br show bd-3jo6.4 --json`
- `br show bd-2lje --json`
- `br show bd-av64.7 --json`
- `models/manifest.json`

Exit artifact:

```text
onechart_subepic: open|closed|unknown
model_arch: implemented=false|implemented=true|unknown
conversion: pass|fail|skip|not-run
convert_census: records=<n-or-unknown> int8_gemms=<n-or-unknown> tied_head=<dedup|kept|unknown>
tokenizer: pass|fail|skip|not-run
tokenizer_cases: <passed>/<total-or-unknown>
d3_vision_projector: pass|fail|skip|not-run
d3_artifacts: preproc=<present|missing|unknown> proj_out=<present|missing|unknown> logits=<present|missing|unknown>
d3_metrics: proj_cos=<value-or-unknown> maxabs=<value-or-unknown> prompt_n=<value-or-unknown>
d4_prefill: absent|committed-certified|in-progress|closed|unknown
d4_prefill_metrics: argmax=<value-or-unknown> cos=<value-or-unknown> maxabs=<value-or-unknown> prompt_n=<value-or-unknown>
d4_cached_decode: absent|committed-closed|closed|unknown
d4_decode_metrics: kv_vs_refill_prefix=<n-or-unknown>/24 first_id=<value-or-unknown> dict_open=<true|false|unknown>
d5_native_assembly: absent|committed-closed|closed|unknown
d5_structured_assembly: chart_result=<present|missing|unknown> json_repair=<present|missing|unknown> number_head=<present|missing|unknown> reliable_check=<present|missing|unknown>
d5_metrics: prompt_n=<value-or-unknown> tests=<pass|fail|skip|not-run>
d6_status: open|closed|unknown
d7_status: open|closed|unknown
d8_status: open|closed|unknown
public_route_status: absent|committed|unknown
corpus_quality: pass|fail|skip|not-run
corpus_quality_scope: number-head|text-json|both|unknown
artifact_distribution: manifest-present|stale-binary-missing|unknown
runtime_status: pullable-and-runnable|local-artifact-runnable|unknown
next_gate: quality-corpus|chart-subcommand|none
```

### OP-TM: TrOMR Local Runtime, Distribution, and Quality Boundary

Canonical tag: `OP-TM`

When-to-Use Triggers:

- a task mentions TrOMR, Polyphonic-TrOMR, `tromr`, optical music recognition,
  OMR, `MusicVocab`, `MusicTokenizer`, `WordLevel`, `tromr.focrq`,
  `tromr.int8.focrq`, `--quant f32`, `storage-int8`, `DISC-005`,
  `efccce9`, `69039c3`,
  `bd-3jo6.5.2`, `bd-3jo6.5.3`, `bd-3jo6.5.6`, `group_norm`, `tf_same_pad`,
  `max_pool2d`, `TromrEncoderW`, `TromrDecoderW`, `decoder_forward`,
  `tromr_encoder_matches_torch_oracle`, `tromr_decoder_matches_argmax_oracle`,
  `merge_semantic`, `semantic_to_musicxml`, `MusicResult`, `forward_tromr`,
  `tromr_music_e2e/v1`, `staff_detect`, `detect_staves`, `recognize_page`,
  `staves_to_musicxml`, `PageRecognition`, `StaffSkip`, stale bead wording
  `staff_detection` / `staff_result`, `MusicPageMeta`,
  `take_music_page_meta`, `OcrEngine::take_music_page_meta`,
  `OcrModel::take_music_meta`, `robot::staff_event`, `music_meta_to_json`,
  robot event kind `staff`, `bd-av64.2`, `bd-av64.6`,
  `tests/fixtures/realscan_music`, `scripts/realscan_music_gate.sh`,
  `realscan_music/v1`, `bd-av64.14`, `eb0c70e`, `40ee875`, `91d552f`,
  `fit-first`, `min_recognized`, `ink-extent`, `extend-to-fit`,
  `neighbor-bounded`, `fitting_bands_keep_the_classic_full_width_geometry`,
  `trim_cuts_page_margins_but_keeps_ink`,
  `wide_staff_with_room_fits_the_positional_budget`, or `fuse_relu`,
  `bd-av64.5`, `sanity_warnings`, `music_warning`, `warnings`,
  `focr-sanity`, `overfull_bar`, `underfull_bar`, `impossible_duration`,
  `key_mismatch`, `bd-av64.13`, `refine_band_skew`, residual skew, or
  `FOCR_TROMR_TTA`, `bd-av64.12`, `QInt8PerChan`, `dequant_qint8`,
  `Decoder::Seq2SeqDense`, `bd-10sb.1`, `proptest`, `tests/property_suite.rs`,
  `fuzz/`, `focrq_parse`, `safetensors_parse`, `image_decode`,
  `pretok_split`, or `decompression_bomb_png_is_rejected_before_allocation`,
- a task asks whether `--task music` uses TrOMR,
- a `.focrq` says `model_id=tromr`.

Failure Modes:

- treating E2 conversion as runtime inference,
- treating E6 decode-only tokenization as an encoder/decoder forward,
- treating E3 helper kernels alone as the full encoder,
- treating E3/E4 as runtime without E7/E9 evidence,
- pretending the TrOMR storage-int8 pull artifact is int8 compute or a speed
  win,
- treating the E5 v1 detector as camera-dewarp/default-barline-quality support,
- treating the old `staff_detection` / `staff_result` acceptance wording as
  current robot event names,
- treating current `bd-av64.2` resilience/observability as crop-geometry proof,
- treating closed `bd-av64.14` fit-first geometry/p169 evidence as camera
  dewarp, default/lossless barline quality, TrOMR int8, perf, or broad
  note-level SER proof,
- treating closed `bd-av64.4` / `FOCR_TROMR_SPLIT=1` as default behavior,
  lossless/broad barline quality, camera dewarp, broad note-level SER, int8, or
  performance evidence,
- treating closed `bd-av64.5` musical-sanity warnings as automatic correction,
  MusicXML structural validation, broad quality proof, or a reason to hide the
  underlying recognition error,
- requiring a schema bump for additive schema-v1 `music_warning` events,
- treating `bd-av64.13` as successful quality work because it is closed; the
  close ledger says residual-skew stayed safe, but `FOCR_TROMR_TTA=3` and
  single-staff refined-crop routing measured negative and were reverted,
- claiming residual-skew lever 1 fixed the double-dotted no21 XFAIL; close
  evidence says it did not flip and future work needs held-out corpus plus a
  presence-first scorer,
- treating the closed `bd-av64.12` storage artifact as a decoder-kernel or
  perf-winning capability; it is a published quantized-storage artifact with
  f32 dequant-on-access and no int8 decoder kernels,
- treating `bd-10sb.1` property/fuzz plumbing as a user-facing CLI feature,
  exhaustive fuzz-coverage claim, or release-wide CI proof instead of
  verification infrastructure,
- missing the `40ee875` fit-first guard and saying every detected staff crop is
  trimmed even when the historic full-width crop already fits,
- describing p055/p100 as page-level XFAILs after their `min_recognized` truth
  floors were promoted,
- demanding a robot schema bump for additive schema-v1 `staff` events,
- assuming PDF+music multi-page JSON carries every page's staves instead of the
  current last-page side-channel caveat,
- treating E8/E5/E10 v1 proof as unconstrained full-page quality/perf proof,
- turning the closed `bd-av64.6` real-scan corpus v1 measuring device into a
  full-SER, full expansion, GOT cross-reference, or aggregate-score closure
  claim,
- misclassifying frozen `goldens/*.musicxml` regression anchors as verifier
  truth,
- treating a real-scan XFAIL as a silent skip instead of a promoted-on-XPASS
  expectation,
- extending TrOMR quantization beyond the measured 40 decoder GEMM suffixes,
- recommending a standalone `focr music` subcommand,
- dropping the dual-lane distinction for `--task music`.

Prompt Module:

```text
Use OP-TM. Confirm bd-3jo6.5.2, .5.6, .5.3, .5.4, .5.7, .5.8, .5.9, .5.5,
.5.10, and .5 close state. Inspect
model_arch TROMR: Decoder::Seq2SeqDense, TokenizerKind::MusicVocab, tasks
&[Task::Music], implemented=true. For E2,
validate scripts/gen_tromr_safetensors.py provenance, published/reference
tromr.focrq, model_id=tromr, 260 tensors, 0 int8, and byte-exact roundtrip for
the f32 reference; do not use `focr convert --quant f32`, because current
convert has no f32 mode. For closed bd-av64.12, validate efccce9,
tromr.int8.focrq, exactly 40 quantized Seq2SeqDense decoder GEMM suffixes,
high-precision encoder/embeddings/norms/heads, QInt8PerChan dequant-on-access,
DISC-005, clean-cache pull byte-exactness, and no int8 compute/perf claim. For
E6, validate src/tokenizer/music.rs, four WordLevel tables,
dense sizes 260/71/7/2, rhythm-only specials, detokenize_goldens, and clean
missing/malformed error paths. For E3, validate tf_same_pad, max_pool2d,
group_norm, fuse_relu, TromrEncoderW, tromr_encoder_matches_torch_oracle,
tromr_oracle_fixtures.json, encoder_out cos 1.00000000, maxabs 3.8e-6, and
oracle floor 0.0. For E4, validate TromrDecoderW, decoder_forward, generate,
tromr_decoder_matches_argmax_oracle, step-0 head cos 1.00000000, maxabs <=
7.6e-6, and 42-step x 3-stream token-exact argmax generation. For E7, validate
merge_semantic, semantic_to_musicxml, fail-loud alignment, and partwise
MusicXML. For E9, validate tromr_staff_tensor, tromr::recognize, MusicResult,
forward_tromr, model_spec_is_knowably_not_tromr, scripts/tromr_music_e2e.sh,
tromr_music_e2e/v1, exit 3/2 negative legs, and real MusicXML output. For E8,
validate 2cbded9, mean SER 0.211, max SER 0.375, DISC-004, and e2e 4/4. For
E5, validate fc9d88a staff_detect/detect_staves module evidence plus 752f3cd
recognize_page/staves_to_musicxml/full-page forward_tromr, and cite
tromr_page_detects_and_reads_stacked_examples with detector-lossless SER 0.125
/ 0.040. For E10, validate 9127676 alpha-routing test and ab0bae0 E10/sub-epic
closure. For distribution, validate bd-av64.7/ece14f9, efccce9, and
models/manifest.json: default focr pull tromr installs tromr.int8.focrq plus
four tokenizer tables, while `focr pull tromr --quant f32` installs
tromr.focrq. For resilience and observability, validate closed bd-av64.2:
3da9dac resilience core, 8af3887 observability half, and 4e881d7 bead close.
Runtime proof: PageRecognition { staves, skips }, StaffSkip { index, bbox,
reason }, skip failed staves when any staff recognizes, aggregate every staff
reason on all-fail pages, human stderr skip notes, and FOCR_TIMING per-staff
dims/outcome. Robot/JSON proof: MusicPageMeta, OcrModel::take_music_meta,
OcrEngine::take_music_page_meta(), robot::staff_event, robot event kind staff,
music_meta_to_json, schema-v1 additive staff events
{staff,total,bbox,status,reason?}, and detection-ordered --json/-o .json staves
for music runs. Cite staff_event_shapes_ok_and_skipped,
music_meta_json_interleaves_in_detection_order, schema_advertises_all_events,
and the scripts/tromr_music_e2e.sh robot staff-event arm. Do not call the event
staff_detection or staff_result, do not invent a schema bump, and do not claim
crop geometry from skip warnings alone. For crop geometry, validate eb0c70e,
40ee875, src/preprocess/staff_detect.rs, fit-first preservation of already
fitting full-width crops, over-budget ink-extent trim with line-spacing pad,
neighbor-bounded extend-to-fit toward the 1280-column position budget, and the
focused tests fitting_bands_keep_the_classic_full_width_geometry,
trim_cuts_page_margins_but_keeps_ink,
wide_staff_with_room_fits_the_positional_budget,
packed_staves_stop_at_the_midline, unpressured_band_keeps_the_generous_margins,
tromr_page_skips_overwide_staff_and_keeps_the_rest, and
tromr_page_all_staves_failing_is_a_named_error. Say this closes bd-av64.14 for
the fit-first geometry lane and p169 5/5 recognized-staff acceptance, not for
camera dewarp, default/lossless barline quality, TrOMR int8, perf, or broad
note-level SER. For barline rescue, validate 64edce3 / closed bd-av64.4,
src/preprocess/staff_detect.rs barline_columns, src/native_engine/tromr.rs
recognize_split/recognize_page, and `FOCR_TROMR_SPLIT=1`. Treat it as an
experimental off-by-default over-budget-band recognition-count rescue: measured
evidence includes p055 5/7 -> 7/7 recognized when armed, but isolated segments
are OOD, continuations lose pitch registration, rhythm agreement is around
0.2, and pixel clef prepend made the target case worse. Do not turn that into
default quality, camera dewarp, broad SER, int8, or perf proof.
For musical-sanity telemetry, validate d51d7d9 / closed bd-av64.5:
src/native_engine/tromr.rs sanity_warnings and MusicWarning, XML
`<!--focr-sanity: ...-->` comments that strip cleanly without changing musical
content, robot music_warning events with {kind, part, measure, detail}, JSON
warnings arrays for music runs, stderr count summaries, FOCR_TIMING per-warning
detail, and machine-stable warning kinds overfull_bar, underfull_bar,
impossible_duration, and key_mismatch. Treat the pass as annotate-only and the
deterministic fallback for later correction work; do not describe it as
auto-correction, structural MusicXML validation, or broad quality proof.
For residual-skew refinement, validate 39651e6 and bd-av64.13 comment 78:
src/preprocess/staff_detect.rs refine_band_skew, per-band +/-1.5 degree /
0.1-degree sweep, >=0.2-degree engage threshold, line-center re-derivation, and
abandon-on-failure. Straight bands must stay byte-identical, corpus gate stayed
green, and the no21 double-dotted XFAIL did not flip. Say bd-av64.13 is closed
after 69039c3 with negative/reverted follow-ups: FOCR_TROMR_TTA=3 regressed
no17_sys at about 2.8x and did not flip the target, while single-staff
refined-crop routing broke the committed no17_top golden. Do not retry those
levers without held-out corpus plus a presence-first scorer.
For the TrOMR int8 storage artifact, validate closed bd-av64.12 / efccce9:
`src/quant/convert.rs` `is_decoder_int8_tensor_for` selects exactly 40
`Decoder::Seq2SeqDense` decoder GEMMs, and `src/native_engine/weights.rs`
dequants `QInt8PerChan` through `Weights::mat()`, `Weights::vec()`, and
`dequant_qint8()`. Evidence includes byte-identical committed golden output,
5/6 truth-tier MusicXML matches, real-scan gate delta 0 with same
verdicts/counts, DISC-005 for the sole no-truth p100 fork, clean-cache pull
byte-exactness, pulled-artifact inference matching the committed golden, and a
954-test library pass. Say storage-int8 is current; do not say int8 compute,
speed win, selftest model, or broad quality proof.
For property/fuzz plumbing, validate closed `bd-10sb.1` / `f9f4c49` / `2dda846`:
`proptest`, `tests/property_suite.rs`, `tests/support/proptest_support.rs`,
`PROPTEST_CASES`, shrink seeds, `fuzz/` cargo-fuzz workspace, and targets
`focrq_parse`, `safetensors_parse`, `image_decode`, and `pretok_split`. Classify
it as verification infrastructure for parser/kernel/preprocess robustness, not a
product feature, release quality proof, exhaustive fuzz campaign, or CI health
claim. Current safety evidence includes public `tokenizer::pretok` for fuzzing,
the mutated `.focrq` parser totality lane, SIMD/scalar and accumulator
properties, and a `decode_reader` decompression-bomb guard with
`decompression_bomb_png_is_rejected_before_allocation` after `image_decode`
found a tiny PNG declaring a huge allocation. For CI/gate health, validate
closed `bd-4yks`: `e80360b` repaired provisioning, `cc79d70` fixed
fixture-manifest/linkage false positives, `c960b77` skipped golden `.actual`
  diff aids, `29aa40a` added `gate-log` artifacts and advisory
  `bench-guardrail`, `7777e34` fixed the aarch64 SMMLA layout-aware comparison,
  and `2e5801b` / `18712cc` record full macOS+Ubuntu gate, dist, and
  advisory-matrix closure. `3f3d9d0` then records round-8 deep verification:
  4.7M fuzz runs, `PROPTEST_CASES=2048`, and 6/6 advisory matrix, with
  `ab6fa6c` committing the earlier grown fuzz corpus. Round 11 records 3.65M
  zero-crash fuzz runs and `5df6395` commits 3,271 post-certification seed
  files. `592426c` then refreshes README public `v0.6.0` release identity and
  asset-size/backend prose. Treat `c960b77` through `592426c`
  as post-tag current-main evidence unless a later release asset proves them. Do
  not convert that into a claim of `TEST_LOG_DIR` capture, a scheduled
  full-model self-hosted runner, ARM64 Windows completion (`bd-3u97` remains),
  or Phase-5 release approval by itself. Native Windows x86_64 is supported.
  Current release approval is a separate OP-SG fact from `c29a78b` / `7c7bd00`
  / `29516b9`.
Note the current caveat: PDF+music multi-page runs expose only the last page's
staves through this side channel; single-image music is the documented path.
For real-scan quality, validate closed bd-av64.6, af13d3e, 40ee875, 91d552f,
tests/fixtures/realscan_music/README.md, truth/attributes.json,
goldens/*.musicxml, scripts/realscan_music_gate.sh, and schema
realscan_music/v1. Classify tier 1 attributes as truth, tier 2 MusicXML goldens
as frozen model-output anchors, and tier 3 page floors as robustness via robot
staff events. Source after 91d552f reads full-page floors from
truth/attributes.json; p055 floor is 5 and p100 floor is 1. Keep
XFAIL-never-SKIP for remaining XFAILs: XPASS fails so fixtures get promoted.
Do not inflate bd-av64.6's scoped corpus-v1 closure into a 10-20 item expansion,
GOT cross-reference output, ladder-scorecard row wiring, or aggregate SER.
For GOT cross-reference, use full systems because narrow staff strips can be
classified as SMILES/molecules. Treat missing --json music semantic as a
doc/code gap unless source has added it. For performance, validate
bd-2sez/5430e2c and docs/PERF_LEDGER:
the f32 baseline row has exact token-stream agreement but loses to pinned
upstream torch on vision_encode, decode-per-token, and end-to-end; bd-av64.12
is storage publication, not the perf gate. End by naming the exact supported
evidence layer: conversion, tokenizer, encoder, decoder, assembly, runtime,
single-staff quality, v1 full-page support, distribution, storage-int8, or f32
baseline perf. Current TrOMR runtime is MusicXML for single-staff and v1
printed/scanned full-page inputs with pulled or local artifacts; focr music,
int8 decoder kernels/compute, camera dewarp, default/lossless barline quality,
broad note-level SER, perf wins/int8 perf rows, and **kern export remain
separate gated claims. `FOCR_TROMR_SPLIT=1` is a separate experimental rescue
switch, not a default-quality statement.
```

Evidence anchors:

- `src/native_engine/model_arch.rs` `TROMR`
- `src/quant/convert.rs` `tromr_real_artifact_roundtrips_byte_exact`
- `scripts/gen_tromr_safetensors.py`
- `scripts/tromr_convert_e2e.sh`
- `src/tokenizer/music.rs`
- `tests/fixtures/tromr/tokenizer_rhythm.json`
- `tests/fixtures/tromr/tokenizer_pitch.json`
- `tests/fixtures/tromr/tokenizer_lift.json`
- `tests/fixtures/tromr/tokenizer_note.json`
- `tests/fixtures/tromr/detokenize_goldens.json`
- `src/native_engine/nn.rs` `tf_same_pad`, `max_pool2d`, `group_norm`
- `src/native_engine/tromr.rs` `TromrEncoderW`, `encode`,
  `tromr_encoder_matches_torch_oracle`
- `scripts/gen_reference_fixtures_tromr.py`
- `tromr_oracle_fixtures.json` / `tromr_seam_encoder_out.bin` under
  `FOCR_TROMR_DIR`
- `src/native_engine/tromr.rs` `TromrDecoderW`, `decoder_forward`, `generate`,
  `tromr_decoder_matches_argmax_oracle`, `FOCR_TROMR_SAMPLE`
- `src/native_engine/tromr.rs` `merge_semantic`, `semantic_to_musicxml`,
  `MusicResult`, `recognize`
- `src/preprocess/mod.rs` `tromr_staff_tensor`
- `src/preprocess/staff_detect.rs` `detect_staves`
- `src/native_engine/mod.rs` `forward_tromr`
- `src/native_engine/tromr.rs` `recognize_page`, `staves_to_musicxml`
- `tests/fixtures/realscan_music/README.md`
- `tests/fixtures/realscan_music/truth/attributes.json`
- `tests/fixtures/realscan_music/goldens/`
- `scripts/realscan_music_gate.sh`
- `br show bd-av64.6 --json`
- `eb0c70e`
- `40ee875`
- `91d552f`
- `src/preprocess/staff_detect.rs`
  `fitting_bands_keep_the_classic_full_width_geometry`,
  `trim_cuts_page_margins_but_keeps_ink`,
  `wide_staff_with_room_fits_the_positional_budget`,
  `packed_staves_stop_at_the_midline`,
  `unpressured_band_keeps_the_generous_margins`
- `src/native_engine/tromr.rs` `PageRecognition`, `StaffSkip`
- `src/native_engine/mod.rs` `MusicPageMeta`, `OcrModel::take_music_meta`
- `src/lib.rs` `OcrEngine::take_music_page_meta`
- `src/robot.rs` `robot::staff_event`, event kind `staff`,
  `staff_event_shapes_ok_and_skipped`, `schema_advertises_all_events`
- `src/cli.rs` `music_meta_to_json`,
  `music_meta_json_interleaves_in_detection_order`
- `8af3887`, `4e881d7` / closed `bd-av64.2`
- `src/cli.rs` `OcrTask::Music`, `model_spec_is_knowably_not_tromr`
- `scripts/tromr_music_e2e.sh` / `tromr_music_e2e/v1`
- `src/preprocess/mod.rs` `tromr_alpha_ink_path_fires_only_when_alpha_varies`
- `src/native_engine/tromr.rs` `tromr_page_detects_and_reads_stacked_examples`
- `src/native_engine/tromr.rs`
  `tromr_page_skips_overwide_staff_and_keeps_the_rest`,
  `tromr_page_all_staves_failing_is_a_named_error`
- `docs/DISCREPANCIES.md` `DISC-004`
- `2cbded9` / `bd-3jo6.5.8`
- `fc9d88a` and `752f3cd` / `bd-3jo6.5.5`
- `9127676`, `ab0bae0` / `bd-3jo6.5.10`, `bd-3jo6.5`
- `br show bd-3jo6.5.2 --json`
- `br show bd-3jo6.5.3 --json`
- `br show bd-3jo6.5.5 --json`
- `br show bd-3jo6.5.6 --json`
- `br show bd-3jo6.5.7 --json`
- `br show bd-3jo6.5.8 --json`
- `br show bd-3jo6.5.9 --json`
- `br show bd-3jo6.5.10 --json`
- `br show bd-av64.7 --json`
- `models/manifest.json`
- `bd-2sez` / `5430e2c`
- `docs/PERF_LEDGER.md` `G2-tromr-f32-staff1-20260706`
- `br show bd-av64.2 --json`
- `br show bd-av64.14 --json`
- `br show bd-av64.12 --json`

Exit artifact:

```text
tromr_subepic: open|closed|unknown
model_arch: implemented=false|implemented=true|unknown
conversion: pass|fail|skip|not-run
convert_census: tensors=<n-or-unknown> int8=<n-or-unknown> roundtrip=<pass|fail|unknown>
tokenizer: pass|fail|skip|not-run
tokenizer_tables: rhythm=<ok|bad|missing> pitch=<ok|bad|missing> lift=<ok|bad|missing> note=<ok|bad|missing>
e3_helpers: pass|fail|partial|not-run
e3_encoder: pass|fail|skip|not-run
e4_decoder: absent|committed|unknown
e7_assembly: absent|in-progress|closed|unknown
e9_runtime: absent|committed|unknown
e8_single_staff_quality: open|closed|unknown
e5_detector_module: absent|committed|unknown
full_page_runtime_wiring: absent|committed|unknown
page_resilience: absent|skip-staff-current|robot-observability-current|unknown
runtime_status: not-implemented|pullable-single-staff|v1-full-page-committed|unknown
task_music_owner: got-format|tromr|dual-lane|unknown
distribution: manifest-pull-int8|manifest-pull-f32-reference|manifest-pull-int8-plus-f32|local-artifact-only|unknown
perf_baseline: absent|f32-losing-row|win|unknown
next_gate: int8|dewarp|default-barline-quality|geometry-normalization|kern|none
split_rescue: absent|experimental-focr-tromr-split|not-checked
sanity_warnings: absent|annotate-only-current|not-checked
music_warning_events: absent|additive-schema-v1|not-checked
residual_skew: absent|lever1-current|not-checked
bd_av64_13_status: open|in_progress|closed|unknown
remaining_quality_gate: tta-voting|single-staff-refined-crop|model-quality|none|unknown
tromr_int8_experiment: absent|live-wip|closed-lossless|negative-evidence|unknown
property_fuzz_plumbing: absent|live-wip|closed-current|unknown
```

### OP-SC: SmolVLM2 Conversion and Decoder Census

Canonical tag: `OP-SC`

When-to-Use Triggers:

- a user asks for SmolVLM2 artifacts, photo description, or VQA,
- `focr convert --model-id smolvlm2` fails or produces suspicious counts,
- a `.focrq` says `model_id=smolvlm2`,
- a docs update might imply SmolVLM2 inference quality/perf is certified by
  conversion alone.

Failure Modes:

- confusing conversion proof with forward/inference route proof,
- confusing decoder-seam proof with full image describe/VQA quality/perf proof,
- claiming the `.focrq` stores the untied SmolVLM2 `lm_head` as int8; runtime
  int8+refine after `4291807` is a separate default-on execution path,
- matching decoder tensors by leaf name and accidentally quantizing SigLIP
  vision tensors,
- ignoring tied-checkpoint rejection.

Prompt Module:

```text
Use OP-SC. Treat SmolVLM2 as conversion-proven, current-source routed, and
sub-epic-C closed, but do not certify quality/perf from conversion alone. Validate the
arch-aware conversion census, high-precision stored untied lm_head, model_id,
license notice, C5 decoder seam proof, C8/C10 close evidence, `4291807`
runtime int8+refine certification, `FOCR_GOT_INT8_LMHEAD=0` kill-switch, and
DISC-003 near-tie ledger before preserving or distributing the artifact.
```

Evidence anchors:

- `scripts/smolvlm2_convert_e2e.sh`
- `src/native_engine/model_arch.rs` `SMOLVLM2`
- `src/native_engine/smolvlm2.rs`
- `src/quant/convert.rs`
- `docs/zoo/smolvlm2-spec.md`
- `br show bd-3jo6.3.2 --json`
- `br show bd-3jo6.3.5 --json`

Exit artifact:

```text
model_id: smolvlm2|other|unknown
convert_census: pass|fail|not-run
tensors: <total/int8/f32-or-unknown>
lm_head: high-precision-untied|tied-rejected|wrongly-quantized|unknown
forward_status: implemented|stale-binary|unknown
next: keep-artifact|rerun-convert|mark-stale-binary|cite-or-rerun-c8-c10
```

### OP-ST: SmolVLM2 Tokenizer Conformance

Canonical tag: `OP-ST`

When-to-Use Triggers:

- a task mentions SmolLM2 tokenizer ids, prompt ids, image token expansion, or
  `PretokScheme`,
- `FOCR_SMOLVLM2_TOKENIZER_JSON`, `FOCR_SMOLVLM2_DIR`, or tokenizer fixtures are
  involved,
- a prompt/IO change depends on SmolVLM2 token counts,
- a docs update might imply tokenizer conformance is still pending.

Failure Modes:

- using the DeepSeek/Baidu pretokenizer for SmolLM2,
- ignoring `tokenizer.json` `pre_tokenizer` and selecting by model id alone,
- silently accepting unknown pretokenizer declarations,
- changing special ids without repinning the source tokenizer and fixture
  corpus,
- treating prompt-image expansion as a raw string count instead of token ids.

Prompt Module:

```text
Use OP-ST. Prove the SmolVLM2 tokenizer through PretokScheme::SmolLm2,
Digits(individual_digits=true), ByteLevel(use_regex=true), pinned special ids
1/49279/2/49190, and the 128/128 token-id/decode exact C6 fixture before
changing prompt, image-slot, or tokenizer claims.
```

Evidence anchors:

- `src/tokenizer/mod.rs`
- `src/tokenizer/pretok.rs`
- `scripts/gen_smolvlm2_token_id_fixtures.py`
- `tests/fixtures/tokenizer_smolvlm2/corpus.txt`
- `tests/fixtures/tokenizer_smolvlm2/expected.json`
- `br show bd-3jo6.3.6 --json`

Exit artifact:

```text
tokenizer_scheme: SmolLm2|DeepSeekV2|unknown
tokenizer_json: <path-or-missing>
special_ids: bos/eos/pad/image=<values-or-unknown>
fixture_gate: pass|skip|fail|not-run
exact_cases: <passed>/<total-or-unknown>
recommendation: keep|repin-fixtures|fix-pretok|mark-stale-binary
```

### OP-SV: SmolVLM2 Vision and Connector Seams

Canonical tag: `OP-SV`

When-to-Use Triggers:

- a task mentions SmolVLM2 image describe/VQA, SigLIP, pixel-shuffle,
  `vision_siglip.rs`, `token_compress.rs`, or `FOCR_SMOLVLM2_DIR`,
- an image-forward issue reaches the vision tower after C7 preprocessing,
- C3/C4/A8/A9 Beads are being edited or cited,
- a source diff changes `gelu_tanh`, patch embedding, position ids, attention
  masking, or connector row order,
- docs might imply vision seams alone prove full describe/VQA quality/perf.

Failure Modes:

- mistaking closed seam proof alone for the closed quality/perf gate,
- using SAM erf GELU or CLIP quick GELU instead of tanh GELU,
- using identity 2-D position ids instead of NaViT bucketized ids,
- applying a causal mask to SigLIP bidirectional attention,
- flattening pixel-shuffle in the wrong row/column order,
- tolerancing `pixel_shuffle` drift even though it should be bit-exact,
- misclassifying `FOCR_SIGLIP_SEQ=1` as the batched path instead of the old
  per-frame sequential SigLIP loop,
- treating `forward_frames_batched` as a quiet-host PERF_LEDGER row instead of a
  byte-identical source-current batching seam,
- treating C3/C4 seams alone as the full C8/C10 describe/VQA proof.

Prompt Module:

```text
Use OP-SV. Classify SmolVLM2 vision status from live Beads and source, then
verify SigLIP 512/1024/768/12x12 bidirectional tanh-GELU seams, NaViT bucketized
position ids, and connector pixel_shuffle scale-4 bit-exactness under
FOCR_SMOLVLM2_DIR. If current source has f1ac972 or later, verify
smolvlm2::vision_rows calls vision_siglip::forward_frames_batched by default,
FOCR_SIGLIP_SEQ=1 is the sequential kill switch, and
batched_frames_match_sequential_byte_for_byte is the proof anchor. End with
seam-certified plus implemented C7/C9 route and source-current batch seam, but
cite DISC-003/C8/C10/A11 for generated-caption quality/perf claims and OP-GB
for any performance interpretation. If the drift is in resize/frame assembly
before SigLIP, use OP-SP first.
```

Evidence anchors:

- `src/native_engine/vision_siglip.rs`
- `src/native_engine/token_compress.rs`
- `src/native_engine/nn.rs` `gelu_tanh`
- `scripts/gen_reference_fixtures_smolvlm2_vision.py`
- `tests/fixtures/smolvlm2/sample_photo.png`
- `tests/fixtures/smolvlm2/vision_oracle_fixtures.json`
- `smolvlm2::vision_rows`
- `vision_siglip::forward_frames_batched`
- `FOCR_SIGLIP_SEQ`
- `batched_frames_match_sequential_byte_for_byte`
- `br show bd-3jo6.3.3 --json`
- `br show bd-3jo6.3.4 --json`
- `br show bd-3jo6.1.8 --json`
- `br show bd-3jo6.1.9 --json`
- `br show bd-3jo6.3.8 --json`

Exit artifact:

```text
bead_status: C3=<status> C4=<status> A8=<status> A9=<status> C8=<status> C10=<status>
siglip_seam: pass|skip|fail|not-run
connector_seam: pass|skip|fail|not-run
siglip_frame_batch: source-current|sequential-kill-switch|absent|not-checked
frames: <n-or-unknown>
image_slots: <n-or-unknown>
forward_status: seam-certified|implemented|unknown
recommendation: keep-narrow|fix-seam|cite-or-rerun-c8-c10|mark-stale-doc
```

### OP-GB: Gauntlet Baseline

Canonical tag: `OP-GB`

When-to-Use Triggers:

- user asks whether focr beats the CPU reference,
- a perf row or roofline claim is being added,
- a benchmark guardrail or frozen baseline claim is being interpreted,
- host load, thread parity, or reference identity is uncertain,
- `artifacts/perf/bd-re8.17/arch.json` is cited.

Failure Modes:

- treating `robot selftest` as a performance win,
- timing focr and reference at different thread budgets,
- running on a noisy host and hiding cv%/loadavg,
- omitting CER/correctness proof for the same timed runs,
- misreading `scripts/bench_guardrail.py` as positive performance evidence
  instead of a regression guardrail,
- moving a frozen baseline without explicit reviewed `--ratchet`,
- chasing artifact-load optimization after `bd-av64.10` evidence has already
  shown SAM attention/vision is the GOT e2e tax,
- treating `FOCR_QKV_FUSED` as opt-in/default-off after closed `bd-241s`,
- treating `FOCR_FUSE_NGRAM_LMHEAD` as a win after `bd-2mo.24` measured it
  inside noise,
- retrying SIMD/polynomial-exp softmax after `bd-av64.10` measured it dead and
  reverted it,
- retrying row-tiled SAM global attention after public `c5e535a` added the
  experiment, public `8bd4037` restored the untiled baseline, and public
  `b757bc0` ledgered the byte-identical-but-slower same-regime measurements,
- treating SmolVLM2 SigLIP frame batching as the whole formal closeout rather
  than byte-identical source-current pass evidence,
- summarizing A11 zoo decode-per-token ratios as end-to-end speedups or as
  proof that all stages improved,
- treating `3f2878d` GOT statics caching as release readiness or the formal
  `bd-av64.10` closeout by itself,
- treating source tag `v0.5.2` / `4cedacd`, the published `v0.5.2` release, or
  README prose as proof that a user's installed binary contains post-release
  source commits such as `efd83e8`, `3f2f97e`, `4291807`, `c248e6d`, or
  `c29a78b`,
- treating shipped `507cebe` mmap-loader correctness as full `bd-2mo.22`
  closure, a perf win, or release-asset proof,
- treating committed OneChart pass-7 statics caching as release readiness,
  broad chart-quality proof, or standalone formal PERF_LEDGER evidence,
- treating committed SmolVLM2 pass-8 statics caching as release readiness,
  public VQA quality, or standalone formal PERF_LEDGER evidence,
- treating the `efd83e8` closeout as an e2e `>=1.0x` win,
- reviving `FOCR_ATTN_GEMM` or broad `FOCR_INT8_KV` from the rejected ledger
  without new bit-exact and corpus evidence.

Prompt Module:

```text
Use OP-GB. Use the quiet-host gauntlet runbook, enforce thread and fixture
fairness, require a fresh raw dir and same-page focr/reference/roofline pairing,
record architecture selftest evidence, and cite ledger rows exactly: G2-closed
for decode-per-token on the current pages, but not a universal stage-win claim.
At/after `3f2f97e`, the gauntlet harness skips AppleDouble `._*.rs` files,
parses the current SAM/CLIP drill-down timing labels, and supports fresh rerun
evidence homes via `OUT_DIR`; do not re-open those as dirty-WIP unless the local
diff is newer than `592426c`. `4291807` certifies the SmolVLM2 untied
`lm_head` int8+refine runtime path with `FOCR_GOT_INT8_LMHEAD=0` as f32-head
kill switch. `c248e6d`/`c4c1684` close `bd-2mo.26` with page_0009 e2e `3.41x`
and page_0014 e2e `2.81x` vs pinned HF bf16. For regression guardrails,
validate bd-1a6h/60d8af4,
scripts/bench_guardrail.py, and benches/.bench-history/baseline.json. The
guardrail consumes gauntlet_focr stage records, fails >10% regressions, logs
NDJSON per stage, refuses perf reporting without an all-green parity receipt,
marks cv_pct > 5 or posture mismatches ineligible, skips-green on absent
fixtures/baselines/receipts, and moves baselines only through reviewed
--ratchet. Do not confuse this with roofline floors; those stay in
gauntlet_row.py/PERF_LEDGER.
For GOT e2e triage, inspect FOCR_TIMING raw stderr and bd-av64.10 first:
sam.hydrate is negligible.
Weights::load is not the bottleneck. Current committed self-relative evidence
includes the bit-identical `01f07fe` pass 1, `f3d3215` pass 2, `0298651` CLIP
pass 3, `f65fded` shared-Linear pass 4, and `f1ac972` SmolVLM2 SigLIP
frame-batched pass 5. Public `origin/main` includes the full row-tile
experiment/reversal/ledger chain: `c5e535a` row-tiled SAM global-attention score
buffers, `8bd4037` restoring the untiled baseline, and `b757bc0` adding
`artifacts/perf/bd-av64.10-rowtile/` plus the negative-evidence row. Treat row
tiling as negative/reverted unless a new target-specific profile proves a
different hardware regime and the replacement avoids multiplying small GEMM
dispatches. Cumulative pass-2 timings: window
attention 1.88s to 0.72s; global
attention 2.10s to 1.66s; sam.forward 5.55s to 4.24s to 3.4-3.6s; GOT forward
6.7s to 5.7s to 4.6-4.8s; unlimited-OCR real page 19.3s to 13.5s. Pass 3
pre-transposes CLIP LinearParams and caches ClipWeights on OcrModel, with
vision.clip 2.49s to 0.77s steady-state. Proof: byte-identical output, armed
GOT/L2 certs, vision_sam 37/37, vision_clip 41/41, full lib 957 green, and
clippy -D clean where cited by the matching pass. Pass 4 moves
`vision_sam::Linear` to a validating `from_row_major` constructor with cached
GEMM-ready `[in,out]` matrices, removing repeated transpose work across SAM,
GOT/OneChart projectors, SmolVLM2, SigLIP, and TrOMR linear consumers;
`3c1b1ea` records the pass-3/pass-4 Beads evidence. Pass 5 stacks all SmolVLM2
frames through `vision_siglip::forward_frames_batched`; `FOCR_SIGLIP_SEQ=1`
forces the old per-frame loop, and
`batched_frames_match_sequential_byte_for_byte` is the byte-identity anchor.
Loaded-host interleaved pairs improved vision+splice 24.00 -> 23.11, 28.51 ->
25.53, and 18.22 -> 16.76; the later `efd83e8` rows are the formal G2 state.
Do not keep SIMD-exp softmax in the remaining-hot-path list: `ab6e083` records
it as a reverted negative in `docs/NEGATIVE_EVIDENCE.md`, and `50d5dad` adds
the durable `artifacts/perf/bd-av64.10-simd-exp/` pointer. Approximate exp had
<=1.1e-8 softmax drift but still forked greedy OCR text, so any retry needs a
new profile and token/output fixture gate. Do not keep row-tiled SAM attention
on the remaining-hot-path list either: `b757bc0` made that negative evidence
public. `3f2878d` is pass-6 committed GOT statics-cache evidence:
`got::GotStatics` caches SAM/projector/embed statics once on `OcrModel`, shared
by sequential and batch GOT paths, with `got.hydrate(cached)`, about 0.8s/page
saved on the cited 2-page sequential loop, GOT sample byte identity, full-lib
959, fmt/clippy/ubs, and the armed batch-vs-sequential gate. Count it as scoped
self-relative source evidence, not release readiness or the formal closeout by
itself. `8de3674` is the committed source/package `v0.5.1` bump plus the
`memmap2` dependency; `4cedacd` is the `v0.5.2` tag and the `v0.5.2` GitHub
release is now published, while post-release source and installed binaries
remain separate facts.
`507cebe` ships the mmap-loader half for open `bd-2mo.22`:
`FOCR_NO_MMAP` / `Backing::{Owned,Mapped}` / `mmap_island` /
`mmap_load_is_byte_identical_to_owned_read`, with `0401df2` documenting that the
remaining bead work is 64B scratch alignment, decode-loop buffer reuse, and
mimalloc measurement. `38ab806` / `a9a406e` then commit the OneChart analog:
`onechart::OnechartStatics` / `OcrModel::onechart_statics` /
`onechart.hydrate(cached)` hydrate SAM/projector/embed once, with
byte-identical chart-data output, full lib 960, fmt/clippy clean, and about
0.10s one-time hydrate. Count it as scoped pass-7 source evidence only.
`9b2a03b` then commits the SmolVLM2 analog:
`smolvlm2::SmolStatics` / `OcrModel::smol_statics` /
`smolvlm2.hydrate(cached)` hydrate SigLIP/projector/embed once, with
byte-identical describe output, lib green, Beads comment 91, and about 0.14s
one-time hydrate. Count it as scoped pass-8 source evidence only. Remaining
candidates are only what fresh timings show. `efd83e8` is the formal closeout:
nine 2026-07-08 rows under artifacts/perf/bd-av64.10-g2r, GOT e2e
0.624->0.885, OneChart 0.546->0.755, SmolVLM2 0.878->0.890, and
decode-per-token 3.046x/2.249x/1.499x. It is closed as measured final state,
not as an e2e >=1.0x win. `3f2f97e` then commits the `bd-2mo.26` gauntlet
harness hardening; `ae7b8f2` records the evidence bundle; `c248e6d` lands the
head-to-head rows; and `c4c1684` closes the bead. Any dirty perf follow-on
after public `592426c` is live-WIP.
`0924479` itself is committed README-only source/binary clarification. Older
v0.4.0 A11 zoo summary rows exist, but prefer the `efd83e8` final rows for this
bead and cite all ratios by stage. Keep them separate from dense-batch
throughput rows and TrOMR's f32 loss baseline.
For GOT setup amortization, validate current public 3f2878d first: OcrModel
owns got::GotStatics, SamWeights, mm_projector_vary, and the widened embed table
hydrate once on the model, sequential and batch GOT page paths reuse them,
FOCR_TIMING logs got.hydrate(cached), and
recognize_batch_matches_sequential_e2e remains byte-identical. Cite the pass-6
measurement as about 0.8s/page saved on the sequential page loop
(got.vision+splice 4.15 -> 3.31s/page on a 2-page batch, one 0.14s hydrate
total). Treat d25dbd7 / got.hydrate(batch) as the predecessor batch-only hoist
with same-binary 3-page attribution 14.47s sequential vs 13.53s batch (~6.5%);
do not attribute the larger SAM-attention campaign to either hoist.
For Baidu int8 decode q/k/v, current source is default-on fused projection:
98cc790/5474ae0 close bd-241s; FOCR_QKV_FUSED=0/off/false/no is the old-path
kill switch. Validate qkv_fused_enabled, fuse_qkv, CachedLayerI8.qkv, and
fused_qkv_gemv_is_byte_identical_to_three_calls before trusting older comments.
Use bd-1waa/bd-3pg7/docs/NEGATIVE_EVIDENCE.md as the ledger boundary: fused
qkv is kept; FOCR_ATTN_GEMM and broad FOCR_INT8_KV were rejected without new
proof. Do not call this prefill fusion.
For ngram-lmhead fusion, validate bd-2mo.24/a0ad299 and
docs/NEGATIVE_EVIDENCE.md: FOCR_FUSE_NGRAM_LMHEAD is opt-in, correct, and
unit-gated by fused_ngram_lmhead_is_byte_identical_to_separate_mask, but the
page_0023 A/B was 16.43s -> 16.40s inside noise. Keep it off unless
multi-image ngram_window=1024 ban sets or a roughly 10x faster decode step make
the lever arithmetic different, then rerun that A/B.
```

Evidence anchors:

- `scripts/gauntlet_runbook.sh`
- `scripts/gauntlet_reference.py`
- `scripts/gauntlet_ref_unlimited.py`
- `artifacts/perf/bd-re8.17/arch.json`
- `docs/PERF_LEDGER.md`
- `br show bd-re8.17 --json`
- `GOT-OCR2 3.37x`
- `OneChart 2.58x`
- `SmolVLM2 1.67x`
- `scripts/bench_guardrail.py`
- `benches/.bench-history/baseline.json`
- `br show bd-1a6h --json`
- `docs/NEGATIVE_EVIDENCE.md`
- `br show bd-2mo.24 --json`
- `br show bd-av64.10 --json`
- `FOCR_TIMING`
- `sam.hydrate`
- `sam.block attn(GLOBAL)`
- `sam.block attn(win)`
- `sam.block mlp`
- `br show bd-241s --json`
- `br show bd-1waa --json`
- `br show bd-3pg7 --json`
- `docs/NEGATIVE_EVIDENCE.md`
- `fused_qkv_gemv_is_byte_identical_to_three_calls`

Exit artifact:

```text
gauntlet_state: runbook-ready|preflight-pass|timed|row-drafted|closed|blocked
reference_backend: hf|onnx|gguf|unknown
threads: <n|unknown>
arch_selftest: pass|not-run|fail
correctness_proof: cer-pass|missing|failed
timing_diagnosis: decode-roofline|sam-attention|artifact-load|unknown
claim: no-perf-claim|partial-evidence|g2-claim
```
