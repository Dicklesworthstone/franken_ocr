# Changelog

All notable changes to `franken_ocr` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it ships a
working artifact.

**Scope note.** This project began on 2026-06-24 as a planning and scaffolding
effort. There are no released versions and no working inference path yet, so this
log currently has a single `Unreleased` section describing the kickoff. Going
forward, history is tracked in git and in the `.beads/` issue graph; capability
waves will be promoted into dated version sections as phases land (see the
roadmap in [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](./COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md)).

## [Unreleased]

Pre-Phase-0 kickoff. Planning, research, and the repository skeleton. No inference
code; the `focr` binary builds and its diagnostic subcommands work, but `ocr`,
`convert`, and `doctor` return "not yet implemented".

### Added

**Planning and specification**

- `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`, the master engineering plan (v2): the
  verified target-model dossier (Baidu Unlimited-OCR as a DeepSeek-OCR-derived 3B
  MoE VLM with R-SWA), the custom `.focrq` quantization format, the full
  per-arch CPU kernel-optimization catalog (tiled SMMLA/AMX and SDOT/VNNI int8/int4
  GEMM, MoE token-grouping, online-softmax R-SWA, many-core/NUMA scaling, fusion,
  vectorized transcendentals, PGO/BOLT), the alien-artifact math families
  (rate-distortion bit-allocation, tail-risk CER bounds, conformal early-exit,
  submodular tensor selection, USL pool-sizing), the three-pillar verification
  gauntlet, and a phased roadmap with hard exit gates.
- `AGENTS.md`, conventions for human and agent contributors, including the
  engineering doctrine (correctness before speed, the validated decoder-only quant
  recipe, the no-nested-rayon-under-lock concurrency rule, the int32-overflow proof
  obligation, honest negative-evidence ledgering).
- Supporting docs: `docs/NEGATIVE_EVIDENCE.md` (seeded with two inherited results
  from sibling projects), `docs/DISCREPANCIES.md`, `docs/PERF_LEDGER.md`.

**Crate skeleton**

- A single-crate Rust 2024 (nightly) package with two binaries from one shared
  entrypoint, `franken_ocr` and the short `focr` alias, `#![forbid(unsafe_code)]`,
  and the separate `release-perf` profiling profile. The dispatch lives in the
  library as `franken_ocr::cli_main()` (`src/cli.rs`); each binary is a thin
  one-line shim — `src/main.rs` (`franken_ocr`) and `src/bin/focr.rs` (`focr`) —
  so each `[[bin]]` has its own source file and cargo emits no "present in
  multiple build targets" warning.
- A synchronous CLI shell (`src/cli.rs` + the two shims) with the planned
  subcommand surface stubbed: working `robot schema` / `robot health` / `robot
  backends` diagnostics, and clear "not yet implemented" results for `ocr`,
  `convert`, `doctor`.
- The stable error type with documented process exit codes (`src/error.rs`) and the
  versioned robot-event schema seed (`src/robot.rs`), both unit-tested.
- The library handle `OcrEngine` (`src/lib.rs`), construction only; `recognize`
  lands in Phase 1.

**Support files**

- `README.md`, `.gitignore` (configured so OCR image fixtures stay committable
  while weights are ignored), `.ubsignore`, `LICENSE` (MIT for the source plus a
  separate notice reproducing Baidu's MIT license for the model weights), and the
  `rust-toolchain.toml` nightly pin.
- Out-of-band script stubs: `scripts/fetch_model.sh` and
  `scripts/gen_reference_fixtures.py`.
- `scripts/check.sh`, the one-command mandatory-check gate (fmt → check
  `--all-targets` → clippy → `cargo test`), stopping on first failure — the repo's
  stand-in for a CI `test` target until CI exists. `cargo test` is a hard
  green-bar gate before any handoff (see AGENTS.md).

### Methodology / evidence

The plan was synthesized from live research into the official Baidu Unlimited-OCR
sources (Hugging Face repo, `config.json`, `modeling_*.py`, `deepencoder.py`, the
arXiv paper) and a study of sibling projects (`frankentorch`'s recent int8 kernel
work, `frankensearch`'s reranker optimization campaign, `franken_whisper` as the
closest single-binary analog, and `asupersync` as the runtime). It was then revised
against an adversarial review (the int32-overflow bound, the R-SWA cross-page
dependence, the bf16-vs-f16 storage correctness fix, the CPU-vs-GPU oracle split,
the full token census, and source-commit pinning).

### Not yet present

No inference engine, no model-specific kernels, no weight converter, no conformance
harness, no benchmarks. These are the subject of the roadmap, not of this release.

[Unreleased]: https://github.com/Dicklesworthstone/franken_ocr/commits/main
