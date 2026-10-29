# Source Hashes — Baidu Unlimited-OCR load-bearing files (Truth Pack, Phase −1)

> Resolves bead `PM1-hash-sources`. SHA-256 of every load-bearing source, fetched
> at HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5` (see `PINNED_SOURCES.md`).
> Every OQ answer and every `NEGATIVE_EVIDENCE.md` / `DISCREPANCIES.md` provenance
> field cites a `(file_sha256, line range)` against these.

| SHA-256 | File | Bytes | Role |
|---------|------|-------|------|
| `27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9` | `config.json` | 2881 | architecture/dtype/MoE/vision/projector config |
| `354be1f2dcfb72ebb385e25465522ce5413a77c36f3b35fec088a3162a11af99` | `model.safetensors.index.json` | 257611 | tensor names + `total_size` (single shard) |
| `a0cbe8464049da1f891b7a12676de06af4cb54c130995d42f71adc1c30c6e9f3` | `tokenizer_config.json` | 165938 | tokenizer class + special tokens |
| `ab4bd57ce17d62e39e0a39e739de1e407484f090f0b2c7e391312bca7a5b061a` | `special_tokens_map.json` | 801 | bos/eos/pad + additional specials |
| `92588cffb1d7032ec83d0a06c3a5171b41df5cbf432d68765441139a57899328` | `processor_config.json` | 466 | normalize/patch_size/downsample_ratio |
| `268bdcbe12cf37bf5a2debb53faf542e56570958a5d9f3314aab3cab2cf6cb48` | `modeling_unlimitedocr.py` | 53431 | preprocess/tiling, `infer`/`infer_multi`, postprocess |
| `74e36e6bd0ba7bc565ef76464a99baa8e6bccb710ae9c1007b54ac30b855fa4c` | `modeling_deepseekv2.py` | 90162 | `SlidingWindowLlamaAttention` (R-SWA), MoE, RoPE |
| `0ae2fb6d1e5ae8cf100fc32f854830acd08c821a0a1f23a94a76588c222ddcf2` | `deepencoder.py` | 38008 | SAM-ViT-B + CLIP-L/14 + projector |
| `b8470dd616ba8745fce6e27b093aef73a098863cc891b2477dcf9326a36000f7` | `configuration_deepseek_v2.py` | 10720 | DeepSeek-V2 config defaults (rope_theta, etc.) |
| `ec7b6ce89bcda643de1f43269ffa66a7b2e65dc3ed30e427958f776546b4ba03` | `conversation.py` | 9253 | prompt-mode taxonomy / conversation template |
| `ddb4e9e5c97c4e560cf133e2bb2adeb1b1609a467c3f504b7495b66852cb32ef` | `README.md` | 8721 | runtime pin, inference example, modes |
| `d985048c6d69429d685fdbe7557340aa0897c0fd8dc038299b148b9c75dc3383` | `LICENSE` | 1061 | MIT, Copyright (c) 2026 Baidu |

**Large source (hashed, fetched on demand by `scripts/fetch_sources.sh`):**

| SHA-256 | File | Bytes | Role |
|---------|------|-------|------|
| `a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4` | `tokenizer.json` | 9979544 | byte-level **BPE** (base vocab 128000 + 830 added tokens), merges, pre-tokenizer `Sequence` (OQ-16) |
| `2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6` | `model-00001-of-000001.safetensors` | 6672547120 | pinned bf16 weight shard used by parity fixtures and conservative `.focrq` conversion |

## Verify

`scripts/fetch_sources.sh --verify` re-fetches the small sources and checks exactly that
set. By hand (the `exit` stops at the v2 section below, whose rows reuse file names;
`--ignore-missing` skips the large sources unless they sit in the directory):

```bash
cd docs/truth-pack/snapshots && shasum -a 256 -c --ignore-missing <(awk -F'`' '/^# Unlimited-OCR-2/{exit} /\| `[0-9a-f]{64}` \|/{print $2"  "$4}' ../SOURCE_HASHES.md)
```

Any mismatch means the upstream model moved since 2026-06-25 — STOP and re-pin
(`PINNED_SOURCES.md`); every `[REPORTED]`/`[VERIFIED]` fact must be re-confirmed.

---

# Unlimited-OCR-2 (v2) source hashes (bead U2-A1)

> SHA-256 of every v2 source, fetched at HF `baidu/Unlimited-OCR-2` commit
> `5632329a32e103d5f1add270681e0f5c1233c6e4` (see `PINNED_SOURCES.md`) and re-verified
> 2026-10-29. Plan citations `MU`/`MD`/`SB`/`DA` are `modeling_unlimitedocr.py`,
> `modeling_deepseekv2.py`, `_sparse_base_impl.py` and `dash_attention.py`.

| SHA-256 | File | Bytes | Role |
|---------|------|-------|------|
| `607d130a70d5af5a35329aa237acd68651f45504c62cc7ced981066f4e781126` | `config.json` | 5525 | v2 architecture + `sparse_attention_config` (SR-SWA, page2d RoPE) |
| `dcd0479f3a12e5e4413d956197f8a4074408e4901834b38c7e036519134e05af` | `generation_config.json` | 111 | generation defaults |
| `b8470dd616ba8745fce6e27b093aef73a098863cc891b2477dcf9326a36000f7` | `configuration_deepseek_v2.py` | 10720 | identical to v1 |
| `ec7b6ce89bcda643de1f43269ffa66a7b2e65dc3ed30e427958f776546b4ba03` | `conversation.py` | 9253 | identical to v1 |
| `960b460acd816df8e561e73dccd5683acb95445eddc723c4f0d6ed1f462b2f77` | `dash_attention.py` | 71465 | `DA`: Token Sieve router, entmax-1.5, block-sparse attention |
| `419b63388c1645b9b72c716070cdca36cd0be6047049424dc75941d3a00aaf1f` | `deepencoder.py` | 37939 | vision tower; comment-only diff vs v1 |
| `24a15dfd47578ab24a7a4d4e977dc5d7a23dfef0a2a4c4bf73fcf3e2ce7acaf2` | `modeling_deepseekv2.py` | 25572 | `MD`: decoder, SR-SWA layer wiring |
| `f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f` | `modeling_unlimitedocr.py` | 55724 | `MU`: preprocess, `infer`/`infer_multi`, page bookkeeping, decode loop |
| `86da9c5638704bb9338575040568a97b3e10820bbeb426f51c07e866f3423ccd` | `_sparse_base_impl.py` | 221755 | `SB`: page2d RoPE, page state, sparse base |
| `e60d842c210c9f1d6cd2ae328cf9127a75f7941a0cea2a8e9abc14662f770c6c` | `packing_patch.py` | 8110 | training-time packing patch |
| `c3de717ffaca40240c665b8d645f15a9d1999feeabaa94beb35caad877c2f0f4` | `sparse_gather_triton.py` | 23860 | Triton gather kernel (GPU-only reference) |
| `58f1bc46093d47b00744970677e66bfbd7485a2d13810f46d13d867ceda13dd4` | `sparse_gather_e2e_triton.py` | 18058 | Triton end-to-end kernel (GPU-only reference) |
| `9e89493d7938e7710c524dbaff613c3648f0a018262417af78505128004e5cb4` | `special_tokens_map.json` | 549 | bos/eos/pad |
| `ceba7d221f265cca4c98b5e0a1d48c10107270b7b29c12396610d6d0fd64b126` | `tokenizer_config.json` | 166632 | tokenizer class + added tokens (incl. `<PAGE>` 128827) |
| `dc60a31ecb77eb03e166345d18cbe25bc71d2a19cc6fc089d8a9ac1c7f209671` | `model.safetensors.index.json` | 254054 | tensor → shard map; `total_size` 6,687,958,536 |
| `af13efa88a75b862337099a19d76e434e37e1d552a3453a180c178ad16ca6eb9` | `README.md` | 13914 | reference stack, `infer`/`infer_multi`/SGLang usage, `license: mit` |
| `1eb5885f4ddfa4ec6fbff375092377f30b5d6a5e034ec4936213f125bf2f9444` | `Unlimited-OCR-2.pdf` | 1515878 | the paper (Algorithm 1, sec. 3.4) |

**Large sources:**

| SHA-256 | File | Bytes | Role |
|---------|------|-------|------|
| `c55c165e32c5a67830362a53e1f548c8930dcc8469efa30ff3252a982799c589` | `tokenizer.json` | 9980278 | v2 BPE (differs from v1's `a02f8fd5…`) |
| `4c40b740f209fd7682040d4ebf19409208acde5c401ccd2dde426d0f41542f18` | `model-00001-of-00002.safetensors` | 4998430312 | bf16 shard 1 |
| `9490b82fa47f89e5110eb5577f8ab1b6694c12121905dbc086ddb10207a4ba0c` | `model-00002-of-00002.safetensors` | 1689875264 | bf16 shard 2 |
| `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d` | `wheel/sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl` | 15281032 | SGLang serving reference (analysis basis, at the pin) |
| `576c71dab3cacad843f6870739494e13b2e8169a69cc038f84395d6503e801ea` | `wheel/sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl` | 15286440 | current upstream wheel (at HEAD `c14071b5`; delta reviewed) |

**Upstream license text** (GitHub `baidu/Unlimited-OCR`, HEAD `6c5d1906ba638a7cd87bc00f14452a47fdabf16b`,
checked 2026-10-29): `LICENSE` is byte-identical to v1's `d985048c…` above. The HF v2 repo has no `LICENSE` file.

**Derived truth-pack files:**

| SHA-256 | File | Role |
|---------|------|------|
| `b3cd2e7a556d526d3d41376bc0c8caf9b62219190d2932d851bcc8c955193e6a` | `oq/uocr2-windowed-sglang.md` | SGLang windowed-decoding semantics (plan §5.10 `[W]`), with the dev10 → dev13 delta |

## Verify (v2)

```bash
# Upstream files: in a directory holding the downloads at the pin (wheels under wheel/).
shasum -a 256 -c <(awk -F'`' '/^# Unlimited-OCR-2/{v=1} /^\*\*Upstream license/{v=0} v && /\| `[0-9a-f]{64}` \|/{print $2"  "$4}' /path/to/docs/truth-pack/SOURCE_HASHES.md)
# Derived files (re-hash this row whenever the notes change):
cd docs/truth-pack && shasum -a 256 -c <(awk -F'`' '/^\*\*Derived truth-pack files/{v=1} v && /\| `[0-9a-f]{64}` \|/{print $2"  "$4}' SOURCE_HASHES.md)
```
