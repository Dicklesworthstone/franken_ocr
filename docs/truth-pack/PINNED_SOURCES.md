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
pymupdf == 1.27.2.2     # PDF -> image at 300 DPI (out-of-band; v1 is image-only, plan §7.7)
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

---

# Unlimited-OCR-2 (v2) — pins (bead U2-A1)

> The spec is `docs/planning/UNLIMITED_OCR_2_PLAN.md` §5 and its OQ register is §14
> (indexed in `OQ_INDEX.md`, section OQ-V2). Every v2 fact, fixture and OQ answer is
> meaningful only relative to these commits. Per-file SHA-256s: `SOURCE_HASHES.md`.

## Verified commits

| Repo | Pin | Notes |
|------|-----|-------|
| Hugging Face `baidu/Unlimited-OCR-2` | **`5632329a32e103d5f1add270681e0f5c1233c6e4`** (2026-10-28; repo created 2026-10-26) | The spec's basis. |
| ↳ upstream HEAD, checked 2026-10-29 | `c14071b549f4fcab10909f3c4acc6a2edb366947` | Drift check (below) passed: every model, code, tokenizer and weight file is byte-identical to the pin (same git blob and LFS oids). Only `README.md` (wheel filename), `.gitattributes` and the SGLang wheel (dev10 → dev13) moved. |

Weights (two bf16 shards; index `total_size` 6,687,958,536):

| File | Bytes | SHA-256 (LFS oid) |
|------|-------|-------------------|
| `model-00001-of-00002.safetensors` | 4,998,430,312 | `4c40b740f209fd7682040d4ebf19409208acde5c401ccd2dde426d0f41542f18` |
| `model-00002-of-00002.safetensors` | 1,689,875,264 | `9490b82fa47f89e5110eb5577f8ab1b6694c12121905dbc086ddb10207a4ba0c` |

## Serving reference (windowed decoding, plan §5.10 `[W]`)

| Wheel | SHA-256 | Where |
|-------|---------|-------|
| `sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl` (15,281,032 B) — **analysis basis** | `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d` | `wheel/` at the pin |
| `sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl` (15,286,440 B) — current upstream | `576c71dab3cacad843f6870739494e13b2e8169a69cc038f84395d6503e801ea` | `wheel/` at `c14071b5` |

The windowed-mode notes (`oq/uocr2-windowed-sglang.md`) cite dev10 line numbers; their
section 5 reviews the dev13 delta (no semantic change to the spec; dev13 fixes a dev10
eager-decode bug that dropped RoPE coordinates and page isolation, so any SGLang
reference run uses dev13 or later).

## Reference stack (differs from the v1 oracle pins above)

```
python == 3.10
torch == 2.13.0
transformers == 4.52.1
tokenizers == 0.21.4
huggingface-hub == 0.36.0
```

The v2 oracle (`scripts/gen_reference_fixtures_uocr2.py`, U2-A6) asserts these five at
runtime, and `scripts/check_oracle_provenance.py` holds the same constants and checks
they agree with the generator's. The remote code also imports `torchvision`, `addict`,
`einops` and `easydict`, and Pillow does the base-mode letterbox resize. So the oracle venv
(macOS arm64, uv 0.9.13, 2026-10-29) resolved to:

```
torchvision == 0.28.0   pillow == 12.3.0   numpy == 2.2.6   safetensors == 0.8.0
addict == 2.4.0         einops == 0.8.2    easydict == 1.13
```

Recreate it with `uv venv --python 3.10` and one `uv pip install` of all of the above.
Never share it with the v1 oracle venv.

## License

**MIT.** The GitHub `baidu/Unlimited-OCR` `LICENSE` reads "MIT License, Copyright (c) 2026
Baidu" (SHA-256 in `SOURCE_HASHES.md`), and the HF v2 model card metadata says
`license: mit`. **The HF v2 repo ships no `LICENSE` file.** The notice franken_ocr
redistributes is the GitHub text, as for v1.

## Upstream-drift check (R20)

The repo was two days old at pin time and is still moving. Before relying on new
upstream material — U2-C2 runs this at M1, U2-E4 before publishing — compare HEAD
to the pin:

```bash
UA='franken_ocr-agent (+https://github.com/Dicklesworthstone/franken_ocr)'
git -c http.userAgent="$UA" ls-remote https://huggingface.co/baidu/Unlimited-OCR-2 refs/heads/main
curl -sS -A "$UA" "https://huggingface.co/api/models/baidu/Unlimited-OCR-2/tree/5632329a32e103d5f1add270681e0f5c1233c6e4?recursive=1" > pin.json
curl -sS -A "$UA" "https://huggingface.co/api/models/baidu/Unlimited-OCR-2/tree/main?recursive=1" > head.json
# Diff (path, oid, lfs.oid, size) between the two listings.
```

* **Nothing changed:** record the date and HEAD here.
* **Docs, README or wheel only:** review the delta (as in `oq/uocr2-windowed-sglang.md`
  section 5) and record it here.
* **Any model, code, tokenizer or weight file changed:** STOP. Changes re-enter only
  through this pinning process: re-pin, re-hash, re-read the changed source, and update
  the plan §5 facts and OQ-V2 answers that cite it before any dependent bead proceeds.

Checks run:
- 2026-10-29: HEAD `c14071b5`, wheel/README-only delta, reviewed.
- 2026-10-30 (U2-C2, at M1): HEAD still `c14071b5`. The listings (28 entries each)
  differ from the pin only in `README.md`, `.gitattributes` and the wheel (dev10
  removed, dev13 added with LFS oid `576c71da…`, the SHA-256 above): the delta
  already reviewed. Every model, code, tokenizer and weight file is unchanged.
- 2026-10-30 (U2-E4, before publishing): HEAD still `c14071b5`. The same 28-entry
  listings, the same delta (`README.md`, `.gitattributes`, the dev10 → dev13 wheel).
  Every model, code, tokenizer and weight file is unchanged.
