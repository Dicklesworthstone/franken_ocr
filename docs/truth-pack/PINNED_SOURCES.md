# Pinned Sources — Baidu Unlimited-OCR (Truth Pack, Phase −1)

> Resolves beads `PM1-pin-sources` (bd-322.1) and underpins every downstream
> kernel. The target model was released ~2026-06-22 and HF repos are mutable, so
> every fixture, perf ratio, and OQ answer is meaningful **only** relative to
> these exact immutable commits.

## Verified commits (resolved 2026-06-25 via `git ls-remote`)

| Repo | URL | Verified commit (HEAD = refs/heads/main) | Method |
|------|-----|------------------------------------------|--------|
| Hugging Face | https://huggingface.co/baidu/Unlimited-OCR | **`3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`** | `git ls-remote https://huggingface.co/baidu/Unlimited-OCR` |
| GitHub | https://github.com/baidu/Unlimited-OCR | **`7e98affeacba24e95562fbaa234ddb89b856874a`** | `git ls-remote https://github.com/baidu/Unlimited-OCR` |

**Provenance note.** Both verified SHAs are **identical** to the Codex-reported
candidates recorded in the plan — i.e. the model has **not** moved since the plan
was written. All `[REPORTED]` facts derived against these commits remain valid.

## Runtime pin (the oracle MUST use this, NOT config.json's export tag)

Per the HF `README.md` (plan §8.1, §11 reference-version-drift risk): `config.json`
records `transformers_version "4.46.3"` (an export tag), while the **runtime** is:

```
torch == 2.10.0
transformers == 4.57.1
Pillow == 12.1.1
pymupdf == 1.27.2.2     # reference-oracle fixture generation only; runtime PDF ingestion is native Rust (plan §7.7)
```

The reference oracle (`scripts/gen_reference_fixtures.py`) asserts these at runtime
and refuses to proceed on a mismatch — fixtures generated against a different stack
are not comparable.

## License (controls redistribution)

**MIT**, `Copyright (c) 2026 Baidu` — identical on the HF `LICENSE` and the GitHub
`LICENSE` (SHA-256 in `SOURCE_HASHES.md`). We may legally redistribute a
converted/quantized derivative of the weights and our kernels provided we ship the
MIT notice attributing Baidu (see `LICENSE` "THIRD-PARTY MODEL WEIGHTS — NOTICE").

## How to reproduce

```bash
scripts/fetch_sources.sh           # re-fetches the load-bearing sources at HEAD
scripts/fetch_sources.sh --verify  # re-fetches AND checks against SOURCE_HASHES.md
```

The raw source snapshots live under `docs/truth-pack/snapshots/` and are **git-ignored**
(re-fetchable + hash-verifiable); only the hashes, this pin record, and the derived
OQ answers / census / spec are committed.
