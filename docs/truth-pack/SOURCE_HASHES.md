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

```bash
cd docs/truth-pack/snapshots && shasum -a 256 -c <(awk -F'`' '/\| `[0-9a-f]{64}` \|/{print $2"  "$4}' ../SOURCE_HASHES.md)
```

Any mismatch means the upstream model moved since 2026-06-25 — STOP and re-pin
(`PINNED_SOURCES.md`); every `[REPORTED]`/`[VERIFIED]` fact must be re-confirmed.
