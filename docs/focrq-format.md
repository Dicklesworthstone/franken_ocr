# `.focrq` Format Specification

This document is the frozen on-disk format contract for franken_ocr quantized
weights. It resolves bead `bd-1es.1` and is the source for the writer
(`bd-1es.2`), reader (`bd-1es.3`), converter (`bd-1es.6`), arch-specific packing
(`bd-2mo.3`), and convert/load determinism tests.

The format is intentionally safetensors-like: one immutable file, a small fixed
binary prefix, one canonical UTF-8 JSON header, and one raw payload blob. The
reader loads the file into one `Vec<u8>` or mmap, validates the prefix and header,
then indexes tensors by byte range. Runtime inference never needs Python,
safetensors, JSON parsing beyond this header, or network access.

## Version

Current `format_version`: **1**.

Any layout change that alters byte interpretation bumps `format_version`. A
loader must refuse a file whose `format_version` is greater than the binary's
supported version and report `FocrError::FormatMismatch` / exit code 7. A loader
may read older versions only when an explicit migration path is implemented and
tested; absent that path, older versions are also rejected as format mismatches.

## Provenance

Every `.focrq` file is tied to the Phase -1 truth pack:

- Hugging Face commit:
  `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`
- GitHub commit:
  `7e98affeacba24e95562fbaa234ddb89b856874a`
- Truth-pack source hashes:
  `docs/truth-pack/SOURCE_HASHES.md`
- Runtime reference pin:
  `torch==2.10.0`, `transformers==4.57.1`, `Pillow==12.1.1`,
  `pymupdf==1.27.2.2`

The fixed prefix stores `source_sha256`, the SHA-256 of the source safetensors
shard used for conversion. The canonical JSON header stores the pinned commits,
the relevant frozen `config.json` fields, and the MIT license notice. A file
without a non-empty Baidu MIT notice is invalid.

## File Layout

All integers are **little-endian**. All offsets and lengths are byte counts.
Readers must use checked arithmetic for every `offset + len` computation.

```
file =
  fixed_prefix
  header_json_utf8
  payload
```

### Fixed Prefix

The fixed prefix is 94 bytes.

| Byte range | Width | Field | Type | Meaning |
|------------|------:|-------|------|---------|
| `0..6` | 6 | `magic` | bytes | Exact bytes `46 4f 43 52 51 00` (`b"FOCRQ\0"`). |
| `6..10` | 4 | `format_version` | `u32` | Current value `1`. |
| `10..14` | 4 | `arch_target` | `u32` enum | Offline packing target, see below. |
| `14..46` | 32 | `source_sha256` | `[u8; 32]` | SHA-256 of source safetensors shard. |
| `46..54` | 8 | `header_len` | `u64` | Length of `header_json_utf8`. |
| `54..86` | 32 | `header_sha256` | `[u8; 32]` | SHA-256 of exact header bytes. |
| `86..94` | 8 | `payload_len` | `u64` | Length of payload blob. |

The header begins at byte 94. The payload begins at `94 + header_len`.
`94 + header_len + payload_len` must equal the file length.

The fixed prefix is deliberately not padded to a natural alignment. Readers must
not cast it to a Rust struct; parse each integer from bytes.

### Header JSON

The header is UTF-8 JSON encoded in canonical form:

- object keys sorted lexicographically
- no insignificant whitespace
- strings escaped by the JSON encoder
- integers represented in base 10
- no floating point values except in `model_config` if a future pinned config
  requires them

The header object has this top-level shape:

```json
{
  "arch_target": "Aarch64Smmla",
  "format_version": 1,
  "license_notice": "Copyright (c) 2026 Baidu. MIT License.",
  "model_config": {},
  "packing_manifest": {},
  "provenance": {},
  "tensor_directory": []
}
```

The JSON `format_version`, `arch_target`, and `provenance.source_sha256_hex`
must duplicate the fixed-prefix values. A mismatch is a format error.

Required top-level fields:

| Field | Type | Required rule |
|-------|------|---------------|
| `format_version` | integer | Equal to fixed prefix and supported binary version. |
| `arch_target` | string enum | Equal to fixed prefix enum. |
| `provenance` | object | Contains pinned commits and source hashes. |
| `license_notice` | string | Non-empty and contains `Copyright (c) 2026 Baidu` and `MIT License`. |
| `model_config` | object | Frozen relevant config fields from truth-pack `config.json`. |
| `tensor_directory` | array | One entry per tensor payload. |
| `packing_manifest` | object | Converter/packing metadata and optional bit-allocation table. |

### Payload

The payload is an unstructured byte blob. Tensor entries name byte ranges inside
the payload. Offsets in tensor entries are relative to the **start of payload**,
not the start of file.

All payload ranges must be non-overlapping unless two directory entries are
explicit aliases with the same `alias_of` field. v1 writers must not emit aliases.

Payload alignment:

- Tensor data ranges start at 64-byte aligned offsets.
- Scale ranges start at 64-byte aligned offsets.
- Padding bytes between ranges must be zero.
- Readers must not require alignment for safety, but validators should reject
  writer output that violates the alignment rule.

## Enumerations

### `arch_target`

Fixed-prefix values:

| Value | JSON string | Meaning |
|------:|-------------|---------|
| `0` | `Generic` | Row-major generic/scalar packing. |
| `1` | `Aarch64Smmla` | aarch64 i8mm/SMMLA prepacked layout. |
| `2` | `X86Vnni` | x86 AVX-VNNI / AVX-512-VNNI layout. |
| `3` | `X86Amx` | x86 AMX-int8 prefill layout. |

Runtime behavior:

- If the file `arch_target` matches the selected backend, use the packed path.
- If the file target is `Generic`, use generic packing on all backends.
- If the file target does not match the selected backend, warn once and either
  use a generic representation embedded in the file or return
  `FormatMismatch` when no compatible packing exists. The loader must never
  silently reinterpret one target's packed bytes as another target's layout.

### `dtype`

| JSON string | Meaning |
|-------------|---------|
| `F32` | Little-endian IEEE-754 f32 payload. |
| `F16` | Reserved for future compatibility. v1 writer must not emit unless explicitly ledgered. |
| `BF16` | Little-endian BF16 payload, stored verbatim from source. |
| `QInt8PerChan` | Signed int8 weights with one f32 scale per output channel. |
| `QInt4PerGroup` | Packed signed int4 weights with f32 scales per group. |

High-precision model tensors are stored as BF16 or F32. BF16 is not narrowed to
F16; BF16 and F16 are both two bytes, and narrowing would be a lossy divergence.

### `packing`

| JSON string | Applies to | Meaning |
|-------------|------------|---------|
| `RowMajor` | all dtypes | Logical row-major order. |
| `Aarch64Smmla2x8` | int8/int4 | Rows interleaved for SMMLA/i8mm micro-kernels. |
| `Aarch64Sdot4x16` | int8/int4 | SDOT-friendly row/block layout. |
| `X86VnniU8S8` | int8/int4 | VNNI U8 activation x S8 weight packing with correction metadata. |
| `X86AmxTile16x16` | int8/int4 | AMX tile-oriented K-panel layout. |

The logical tensor shape and dequantized values must be identical across all
packings produced from the same source tensor.

### `tier`

`tier` is optional for v1 int8 entries and required for int4 entries.

Allowed values:

- `BF16`
- `F32`
- `Int8`
- `Int4G16`
- `Int4G32`

The `tier` records the converter's precision policy. It is provenance, not a
runtime dispatch knob.

## Tensor Directory

Each tensor entry is a JSON object:

```json
{
  "byte_len": 165478400,
  "byte_offset": 0,
  "dtype": "BF16",
  "group_size": null,
  "name": "decoder.embed",
  "packing": "RowMajor",
  "scales_len": 0,
  "scales_offset": null,
  "shape": [129280, 1280],
  "source_name": "model.embed_tokens.weight",
  "tier": "BF16"
}
```

Required fields:

| Field | Type | Rule |
|-------|------|------|
| `name` | string | Internal canonical tensor path. Unique. |
| `source_name` | string | HF state_dict tensor path. |
| `dtype` | enum | One of the dtype strings above. |
| `shape` | array of integers | Logical dequantized shape. |
| `packing` | enum | Physical payload layout. |
| `byte_offset` | integer | Offset into payload. 64-byte aligned for writer output. |
| `byte_len` | integer | Data byte length. Must be non-zero unless the tensor is explicitly empty in config. |
| `scales_offset` | integer or null | Offset into payload for scale bytes. |
| `scales_len` | integer | Scale byte length, zero for unquantized tensors. |
| `group_size` | integer or null | Required for `QInt4PerGroup`, null otherwise. |
| `tier` | enum or null | Precision policy; required for int4. |

Loader validation:

- `name` is unique.
- `byte_offset + byte_len <= payload_len`.
- If `scales_len > 0`, `scales_offset` is non-null and
  `scales_offset + scales_len <= payload_len`.
- Data and scale ranges do not overlap.
- `shape` matches the expected model census for `name`.
- `dtype` is compatible with `tier`.
- `packing` is compatible with `arch_target`.
- Scale count matches `dtype`, shape, and group size.

## Scale Layout

All scales are little-endian f32 arrays stored in the payload range named by
`scales_offset` / `scales_len`.

### `QInt8PerChan`

Quantization is symmetric per output channel:

```
scale[row] = max(abs(w[row, :])) / 127
q[row, k] = round_ties_to_even(clamp(w[row, k] / scale[row], -127, 127))
zero_point = 0
```

Rules:

- `scales_len == shape[0] * 4`.
- `byte_len` is the packed int8 weight byte length for the physical packing.
- Logical dequantization is `f32(q) * scale[row]`.
- If an all-zero row has `max_abs == 0`, writer stores `scale[row] = 1.0` and all
  `q` values zero. This avoids NaN/Inf while preserving the row exactly.

### `QInt4PerGroup`

Int4 is defined now so the v1 reader can reject or inspect it, but int4 writing
lands later.

Rules:

- `group_size` is either 16 or 32.
- Groups are along the K/input dimension within each output row.
- `tier` is `Int4G16` or `Int4G32`.
- Each logical value is signed two's-complement int4 in `[-8, 7]`.
- Two int4 values pack into one byte: low nibble first, then high nibble.
- Scale count is `shape[0] * ceil(shape[1] / group_size)`.
- `scales_len == scale_count * 4`.
- Logical dequantization is `f32(q4) * scale[row, group]`.

Padding inside the final partial group is encoded as zero and ignored by logical
dequantization.

## Frozen `model_config`

`model_config` is the minimal frozen subset of truth-pack `config.json` needed to
validate shape compatibility and reject stale artifacts. It must include at
least:

- `model_type`
- `torch_dtype`
- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `v_head_dim`
- `intermediate_size`
- `moe_intermediate_size`
- `n_routed_experts`
- `num_experts_per_tok`
- `n_shared_experts`
- `vocab_size`
- `max_position_embeddings`
- `sliding_window`
- `use_mla`
- `vision_config`
- `projector_config`
- `source_hashes.config_json_sha256`
- `source_hashes.model_index_sha256`
- `source_hashes.tokenizer_json_sha256`

Readers must compare these values against their compiled model census before
loading tensor bytes. A mismatch is `FormatMismatch`, not a warning.

## `packing_manifest`

`packing_manifest` records converter decisions that are not tensor data:

```json
{
  "converter_version": "franken_ocr 0.1.0",
  "created_utc": "2026-06-25T00:00:00Z",
  "quant_recipe": "decoder-ffn-int8-v1",
  "activation_quant": "dynamic-per-row",
  "bit_allocation_table": null,
  "rounding": "round_ties_to_even",
  "notes": []
}
```

`bit_allocation_table` is reserved for AF-1 rate-distortion allocation. When
present, it maps tensor names to `tier`, `expected_loss`, and `bits_per_weight`.
Readers validate it for consistency but do not use it to reinterpret payload
bytes; the tensor directory remains authoritative.

## High-Precision Set

The converter must store these tensors high precision unless a later bead adds a
measured, kill-switched exception:

- full vision tower
- projector
- `embed_tokens`
- MoE router gates
- all norms

The default validated quantized set is decoder FFN/expert/dense GEMMs. Attention
`q/k/v/o` and `lm_head` int8 are separate measured levers behind kill switches,
not baseline assumptions.

## Loader Algorithm

1. Read the file into one blob or mmap.
2. Check length is at least 94 bytes.
3. Check magic equals `b"FOCRQ\0"`.
4. Parse fixed prefix with little-endian integer reads.
5. Check `format_version` is supported.
6. Check `94 + header_len + payload_len == file_len` with checked arithmetic.
7. Hash header bytes and compare to `header_sha256`.
8. Parse header JSON.
9. Check duplicated fixed-prefix fields match header fields.
10. Check non-empty `license_notice` includes Baidu MIT attribution.
11. Validate `model_config` against the compiled truth-pack census.
12. Validate every tensor directory entry and byte range.
13. Build an immutable map `name -> TensorRange`.
14. Warn on compatible arch mismatch; error on incompatible packing.

No tensor dequantization is required during header sniffing. `native_model_available`
may stop after step 12.

## Writer Determinism

For a fixed source safetensors shard, config, converter version, quant recipe,
and arch target, output must be byte-identical across runs:

- tensor directory sorted by `name`
- canonical JSON header
- deterministic range ordering
- zeroed alignment padding
- deterministic rounding (`round_ties_to_even`)
- no RNG or calibration data in v1 int8 conversion

The writer test must assert:

- `source_sha256` matches the source shard
- high-precision BF16/F32 tensors round-trip byte-identically
- `convert -> load -> reserialize` is byte-identical for v1 artifacts
- all arch packings dequantize to the same logical weights

## Error Mapping

| Condition | Error |
|-----------|-------|
| Missing file | model-not-found / exit 3 |
| Bad magic | `FormatMismatch` / exit 7 |
| Unsupported `format_version` | `FormatMismatch` / exit 7 |
| Invalid JSON header | `FormatMismatch` / exit 7 |
| Missing license notice | `FormatMismatch` / exit 7 |
| Source/config/census mismatch | `FormatMismatch` / exit 7 |
| Out-of-range tensor byte range | `FormatMismatch` / exit 7 |
| Incompatible arch packing | `FormatMismatch` / exit 7 |

Warnings are allowed only for compatible arch fallback, and must name both the
file target and selected backend.

## Minimal Header Example

```json
{
  "arch_target": "Generic",
  "format_version": 1,
  "license_notice": "Copyright (c) 2026 Baidu. MIT License.",
  "model_config": {
    "hidden_size": 1280,
    "max_position_embeddings": 32768,
    "model_type": "deepseek_v2",
    "num_attention_heads": 10,
    "num_hidden_layers": 12,
    "sliding_window": 128,
    "source_hashes": {
      "config_json_sha256": "27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9",
      "model_index_sha256": "354be1f2dcfb72ebb385e25465522ce5413a77c36f3b35fec088a3162a11af99",
      "tokenizer_json_sha256": "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4"
    },
    "torch_dtype": "bfloat16",
    "use_mla": false,
    "vocab_size": 129280
  },
  "packing_manifest": {
    "activation_quant": "dynamic-per-row",
    "bit_allocation_table": null,
    "converter_version": "franken_ocr 0.1.0",
    "created_utc": "2026-06-25T00:00:00Z",
    "quant_recipe": "decoder-ffn-int8-v1",
    "rounding": "round_ties_to_even"
  },
  "provenance": {
    "github_commit": "7e98affeacba24e95562fbaa234ddb89b856874a",
    "hf_commit": "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5",
    "source_sha256_hex": "0000000000000000000000000000000000000000000000000000000000000000"
  },
  "tensor_directory": []
}
```
