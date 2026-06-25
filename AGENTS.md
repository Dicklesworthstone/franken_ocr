# AGENTS.md — franken_ocr

> Guidelines for AI coding agents working in this Rust codebase.

---

## RULE 0 — THE FUNDAMENTAL OVERRIDE PREROGATIVE

If I tell you to do something, even if it goes against what follows below, YOU MUST LISTEN TO ME. I AM IN CHARGE, NOT YOU.

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a new file that you yourself created, such as a test code file. You have a horrible track record of deleting critically important files or otherwise throwing away tons of expensive work. As a result, you have permanently lost any and all rights to determine that a file or folder should be deleted.

**YOU MUST ALWAYS ASK AND RECEIVE CLEAR, WRITTEN PERMISSION BEFORE EVER DELETING A FILE OR FOLDER OF ANY KIND.**

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval. "I think it's safe" is never acceptable.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options (`git status`, `git diff`, `git stash`, copying to backups) before ever considering a destructive command.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for a confirmation that your understanding is correct. Only then may you execute it.
5. **Document the confirmation:** When running any approved destructive command, record (in the session notes / final response) the exact user text that authorized it, the command actually run, and the execution time.

---

## Branch Policy

- Primary branch is `main`.
- Do not reference `master` in docs/scripts.
- If release instructions require sync, push `main:master` after `main`.

---

## Project Mission

`franken_ocr` is a **pure-Rust, memory-safe, CPU-hyper-optimized library + single-binary CLI (`focr`)** that runs the **Baidu Unlimited-OCR** vision-language document-parsing model **with no general ML framework**. We transform the model's bf16 weights into a custom quantized on-disk form (int8 first, int4 in refinement rounds) and write **model-specific kernels** whose only job is to run *this one model* as fast as possible on:

- **Apple Silicon / ARM64** — NEON, FEAT_DotProd (SDOT), FEAT_MATMUL_INT8 (SMMLA / i8mm)
- **Intel / AMD x86-64** — AVX2, AVX-VNNI, AVX-512-VNNI (and AMX tiles where present)

**CPU is the priority** (most target hosts lack a usable GPU); CUDA is an explicit later stretch goal.

It is built on:
- `/dp/frankentorch` (`ft-kernel-cpu`, `ft-core`, `ft-serialize`) — custom CPU tensor kernels, consumed at the **kernel** level (not the autograd level).
- `/dp/asupersync` — structured-concurrency runtime, for **orchestration / cancellation / IO only** (not for intra-op math parallelism).
- `/dp/frankensqlite` (`fsqlite`) — durable local run state + telemetry (NEVER `rusqlite`).

**The single source of truth for what we are building and why is [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md).** Read it before writing any kernel.

### What this model is (one paragraph)

Unlimited-OCR is an **end-to-end VLM**, a DeepSeek-OCR derivative: a **DeepEncoder** vision tower (SAM-ViT-B → 16× conv token-compressor → CLIP-L/14) → a single **linear projector** (2048→1280) → a **DeepSeek-V2 MoE decoder** (12 layers, hidden 1280, 10 MHA heads — `use_mla=false`, 64 routed + 2 shared experts, top-6, vocab 129280) whose attention is replaced by **R-SWA** (Reference Sliding Window Attention, window 128), bounding generated-token KV while retaining all reference tokens. bf16, 6.67 GB single safetensors shard. **License: MIT (Copyright (c) 2026 Baidu)** — we may legally redistribute a quantized derivative if we ship that notice.

---

## Product Shape

The project must be both:
1. A reusable Rust library for embedding the OCR pipeline (`OcrEngine::recognize(...)`), **synchronous and blocking** — the async runtime is an owned implementation detail.
2. A standalone CLI binary `focr` with:
   - **robot mode** (agent-first, versioned NDJSON, self-describing `robot schema`)
   - human mode (`focr ocr <image>` → markdown, or `--json`)

Input: **document images only in v1** (PNG/JPG/…; PDF is rasterized out-of-band — see plan §7.7). No Python, no FFI, **no network at inference time**, no GPU required.

---

## Porting Workflow (Spec-First)

Implementation follows spec documents, not ad-hoc copying. Read in this order:
1. [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md) — the master plan (architecture, kernel strategy, phased roadmap, verification methodology).
2. The **Open Research Questions register (§13 of the plan)** — every `[OPEN]`/`OQ-N` item that MUST be resolved by reading the actual model source before the dependent kernel ships.
3. The **reference source** in the HF repo: `modeling_unlimitedocr.py`, `modeling_deepseekv2.py` (`SlidingWindowLlamaAttention`!), `deepencoder.py`, `configuration_deepseek_v2.py`, `conversation.py`.

**Hard rule: no kernel ships against an unresolved `[OPEN]`.** A phase exit gate cannot pass while it depends on an unresolved OQ. Promote an `[OPEN]` to a design assumption only after reading the source and recording the answer in the register.

---

## The franken_ocr Engineering Doctrine (READ THIS BEFORE OPTIMIZING)

These are the load-bearing, non-negotiable rules distilled from the plan and from the frankentorch/frankensearch prior art. Violating any of them has burned real days of work before.

1. **Correctness outranks speed, always (G1 > G2).** Parity gate FIRST, perf second. A faster kernel that drifts the OCR output is reverted — no source landed — and memorialized in `docs/NEGATIVE_EVIDENCE.md`. We ship speed *on top of* parity, never instead of it.

2. **The quant recipe is fixed and validated: quantize the decoder FFN/expert GEMMs only.** Keep **high precision (BF16/F32)**: the entire vision tower, projector, `embed_tokens`, the **MoE router gate**, and **ALL norms**. Quantizing the vision encoder wrecks OCR (both prior-art quants keep it). int8 on attention `q/k/v/o` and on `lm_head` go *beyond* the validated set — gate them behind a measured-CER kill-switch (OQ-14), never assume they are lossless.

3. **NEVER hand-roll wide-SIMD over scalar inner loops.** It measured **~5× SLOWER** than LLVM autovectorization in the sibling repos. The winning levers are **(a) full-core-parallel forward + (b) native int8 *matmul* intrinsics (SDOT/VNNI today; the tiled SMMLA/i8mm GEMM we BUILD)**, with LLVM autovectorizing the elementwise/norm/softmax/dequant glue.

4. **The edge we are actually building (corrected by measurement, plan §3.2).** The gap to ONNX/MLAS is **kernels below peak**, NOT framework overhead — a naive "fused tape-free forward" that replaced SIMD kernels with scalar-f32 ops regressed 3–10×; an un-blocked SMMLA was *slower* than SDOT (load-bound); AMX-f32 did not beat ONNX-int8. The win is the **combination franken_ocr has by construction**: a fused, tape-free, zero-per-op-allocation single-model forward, with **every op at peak** (register-blocked SMMLA/VNNI linears + int8 attention where accuracy allows + vectorized norms/softmax, NEVER naive), plus the **int4 bandwidth win** on the expert bulk. Honest bar: at-or-near ONNX on CPU, portable to targets where `ort` can't build, and bounded generated-token KV for long documents. See `docs/NEGATIVE_EVIDENCE.md` NE-INH-3/4/5.

5. **NEVER nest rayon under a held lock; NEVER nest a second asupersync runtime inside a task.** Single `OcrModel` behind a cache; **sequential** outer page/document loop; each forward fans out across all cores internally via the kernel's own rayon pool (pinned to physical cores, one live forward at a time). A `many_pages_without_deadlock` CI watchdog (pages ≫ pool) hangs on regression. This is the durable fix the frankensearch deadlock saga converged on.

6. **int8 i32-accumulation overflow is a proof obligation, not an assumption.** The real worst case here is the dense layer-0 `down_proj` at **K = 6848** (U8S8 ≤ ~221.7M); it fits i32 but must be proven by a unit test at worst-case K on every arch (plan §5.4). Do NOT inherit frankensearch's `k≤1536` bound.

7. **R-SWA: constant generated-token KV, not page-constant memory or compute.** The reference block `m` grows with each page (base mode is 256 image-feature tokens plus structural/prompt tokens — 273 token slots at 1024 before prompt), and that block is attended every step, so per-token attention is `O(m+128)`. We may preallocate a fixed worst-case buffer, but the logical reference bytes and compute still scale with page count up to the 32K cap. Multi-page decode is **cross-page dependent** (OQ-13) — do not assume "multi-page = sum of single-page parses."

8. **Honest, measured everything.** Every accepted numeric divergence → `docs/DISCREPANCIES.md` (reference behavior, our impl, **measured** impact, kill-switch env var, review date). Every rejected optimization → `docs/NEGATIVE_EVIDENCE.md` (the 5-pass loop: claim+baseline → one lever + bit-exact proof → rebench + Score → keep/revert → next hotspot). The head-to-head gauntlet (plan §9.3) uses thread/allocator/precision fairness controls — never benchmark torch at @64. No silent numerics changes, ever.

9. **Two binaries from one entrypoint:** `focr` (short) + `franken_ocr` (long). The shared dispatch lives in the library as `pub fn cli_main() -> ExitCode` (`src/cli.rs`); each binary is a **thin one-line shim** that just calls it — `src/main.rs` (the `franken_ocr` bin) and `src/bin/focr.rs` (the `focr` bin). They are declared explicitly in `Cargo.toml [[bin]]` (which also disables the implicit package-named bin), but **each `[[bin]]` points at its own shim file** — never the same `path` in two targets, which trips cargo's "present in multiple build targets" warning. Keep both shims byte-for-byte equivalent: `fn main() -> std::process::ExitCode { franken_ocr::cli_main() }`.

---

## Alien-Artifact Engineering Contract

For runtime/adaptive decisions (e.g. expected-loss-guided per-tensor quant, conformal/sequential-test early-exit decode), include:
- explicit state space, explicit actions, a loss matrix
- posterior/confidence terms and a calibration metric
- a deterministic fallback trigger
- an evidence-ledger artifact

No adaptive controller ships without a conservative deterministic fallback.

---

## Code Editing Discipline

### No Script-Based Changes
**NEVER** run a script that mass-edits code files. Brittle regex transforms create more problems than they solve. Make code changes manually (use parallel subagents for many simple changes; do subtle/complex changes methodically yourself).

### No File Proliferation
Revise existing files in place. **NEVER** create `mainV2.rs` / `nn_improved.rs` / `decoder_enhanced.rs`. New files are reserved for genuinely new functionality; the bar is incredibly high.

---

## Backwards Compatibility

We are in early development with **no users**. Do things the **RIGHT** way with **NO TECH DEBT**. Never create compatibility shims or wrappers for deprecated APIs. Just fix the code directly.

---

## Toolchain

- Rust 2024 edition. Nightly toolchain (`rust-toolchain.toml`) — **required** for `stdarch` i8mm/dotprod intrinsics and `portable_simd`.
- `#![forbid(unsafe_code)]` at every crate root; `unsafe_code = "deny"` in `[lints.rust]`. `unsafe` is permitted **only** inside named, audited SIMD modules behind an `#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]` island, each load carrying a `// SAFETY:` note and each kernel having a **bit-identical scalar fallback** that cross-compiles to every target.
- Cargo only. Persistence via `fsqlite`, never `rusqlite`.

---

## Mandatory Checks After Substantive Changes

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
ubs $(git diff --name-only)
```

If any check fails, fix root causes before handing off.

### The `cargo test` gate (green-bar requirement)

`cargo test` is a **hard gate**: it MUST exit `0` before any change is handed off
or a bead is closed. There is no Makefile/justfile/CI in the repo yet, so the gate
is the bare command above plus the convenience wrapper `scripts/check.sh`
(`scripts/check.sh` runs `cargo fmt --check`, `cargo check --all-targets`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test` in order and stops
on the first failure). When CI is added, this same sequence is the required job —
wire `scripts/check.sh` as the CI test step rather than duplicating the commands.

Note on the build surface: both binaries (`focr`, `franken_ocr`) compile from thin
shims over the shared `cli_main()` in the lib (doctrine #9). `cargo check
--all-targets` MUST be free of the "`src/main.rs` … present in multiple build
targets" warning — each `[[bin]]` points at its own shim file. (The only warnings
permitted from `cargo check` are environmental, e.g. the incremental-cache
hard-link notice when the target dir lives on a filesystem without hard links.)

---

## Testing Policy

Each module includes unit tests for happy path, edge cases, error handling. Beyond that, the conformance ladder is the heart of this project (plan §8):

- **Reference oracle**: pinned `torch==2.10.0` / `transformers==4.57.1` running `model.infer()` (`scripts/gen_reference_fixtures.py`) — **establish the oracle's own nondeterminism floor first** (run twice / two thread counts) before setting tolerances.
- **Parity ladder L0–L5**: preprocessing (exact), per-op cosine ≈ 1.0, per-layer hidden states, logits (within *measured* quant tolerance), decoded tokens (exact where reference deterministic), end-to-end CER/TEDS/Formula-CDM within a documented budget.
- **Tokenizer conformance** (OQ-16): token-id-exact vs `LlamaTokenizerFast` over `tokenizer.json` — a prerequisite for every downstream gate.
- **Differential / metamorphic / golden-artifact** suites; **model-gated e2e** (skip-with-SUCCESS without the 6.67 GB weights; prove the native path ran by pointing fallbacks at `/nonexistent`).
- **`many_pages_without_deadlock`** concurrency watchdog.

---

## Agent Ergonomics Requirements

Robot mode must be: stable versioned schema, deterministic where possible, explicit exit codes, line-oriented NDJSON, easy to pipe. Do not mix human decoration with machine output in robot mode. `robot schema` self-describes the contract; a contract test validates emitted events against a frozen JSON schema fixture. Stable exit codes are documented in `error.rs` (plan §7.4).

---

## Session Completion ("Landing the Plane")

Before finishing a work session you MUST:
1. File beads issues for remaining work (anything needing follow-up).
2. Run quality gates (if code changed) — tests, clippy, fmt, `ubs`.
3. Update issue status — close finished work, update in-progress.
4. `br sync --flush-only` to export beads to JSONL, then `git add .beads/`.
5. Hand off — summarize what changed, gates run + results, remaining risks/gaps, concrete next steps.

---

## MCP Agent Mail — Multi-Agent Coordination

A mail-like layer for agents to coordinate via MCP tools/resources: identities, inbox/outbox, searchable threads, advisory file reservations with human-auditable Git artifacts.

- **Register identity:** `ensure_project(project_key=<abs-path>)` → `register_agent(project_key, program, model)`.
- **Reserve files before editing:** `file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true, reason="br-###")`.
- **Communicate with threads:** `send_message(..., thread_id="br-###")`, `fetch_inbox`, `acknowledge_message`.
- **Prefer macros:** `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`.
- Common pitfalls: `"from_agent not registered"` → `register_agent` in the right `project_key` first; `"FILE_RESERVATION_CONFLICT"` → adjust patterns / wait / use non-exclusive.

---

## Beads (br) — Dependency-Aware Issue Tracking

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`). Issues live in `.beads/` and are tracked in git. **`br` is non-invasive — it NEVER runs git.** After `br sync --flush-only`, manually `git add .beads/ && git commit`.

```bash
br ready                 # issues ready to work (no blockers)
br list --status=open
br show <id>             # full detail with dependencies
br create --title="..." --type=task|bug|feature|epic --priority=2   # 0=critical..4=backlog (NUMBERS)
br update <id> --status=in_progress
br close <id> [<id2> ...] [--reason "..."]
br dep add <issue> <depends-on>
br sync --flush-only     # export to JSONL (NO git ops)
```

Conventions: use the bead ID (e.g. `br-123`) as the Agent-Mail `thread_id` and prefix subjects with `[br-123]`; put the issue ID in the file-reservation `reason`; include `br-###` in commit messages.

---

## bv — Graph-Aware Triage

`bv` computes PageRank/betweenness/critical-path/cycles over `.beads/beads.jsonl`. **Use ONLY `--robot-*` flags — bare `bv` launches a blocking TUI.** Start with `bv --robot-triage` (counts + top picks + quick wins + blockers). `bv --robot-plan` for parallel tracks; `bv --robot-insights` for full metrics (check `.Cycles` — must be empty).

---

## UBS — Ultimate Bug Scanner

`ubs <changed-files>` before every commit. Exit 0 = safe; exit >0 = fix & re-run.

```bash
ubs file.rs file2.rs                    # specific files (< 1s)
ubs $(git diff --name-only --cached)    # staged files — before commit
ubs --only=rust,toml src/               # language filter
```
Parse `file:line:col` → location, 💡 → suggested fix. Fix root cause, not symptom. Critical (always fix): memory safety, UB, data races. Important: unwrap panics, resource leaks, overflow.

---

## RCH — Remote Compilation Helper

RCH offloads `cargo build/test/clippy` to remote workers to avoid local compilation storms. Installed at `~/.local/bin/rch`, hooked into Claude Code's PreToolUse — usually transparent. Manual: `rch exec -- cargo build --release`. Health: `rch doctor`, `rch status`. Fails open (builds run locally if workers unavailable). **Codex/GPT users:** no auto-hook — manually `rch exec -- <cmd>` for heavy builds.

---

## ast-grep vs ripgrep vs warp_grep

- **`ast-grep`** when structure matters (refactors/codemods, policy checks, safe rewrites): `ast-grep run -l Rust -p '$X.unwrap()'`.
- **`ripgrep`** for raw text/literal hunts and pre-filtering.
- **`mcp__morph-mcp__warp_grep`** for exploratory "how does X work?" — an AI agent expands the query, reads files, returns line ranges with context. Don't use it to find a known symbol (use `rg`); don't use `rg` to understand architecture (use `warp_grep`).

---

## cass — Cross-Agent Session Search

`cass` indexes prior agent conversations so we can reuse solved problems. **Never run bare `cass` (TUI)** — always `--robot` or `--json`.

```bash
cass search "int8 simd gemm" --robot --limit 5
cass view /path/to/session.jsonl -n 42 --json
```
stdout is data-only, stderr diagnostics, exit 0 = success. Treat it as a way to avoid re-solving problems other agents already handled (this project's own kickoff research lives there).

---

## Note for Codex/GPT agents — unexpected working-tree changes

If `git status` shows edits you did not make (in `Cargo.toml`, `src/*.rs`, etc.), those are from the **other agents working on this project concurrently** — a normal, frequent occurrence. **NEVER** stash, revert, or overwrite another agent's work. Treat those changes exactly as if you made them yourself. Do not stop to ask about them.

---

## Note on Built-in TODO Functionality

If I explicitly ask you to use your built-in TODO functionality, do so without complaining that you need to use beads. Always comply with such orders.
