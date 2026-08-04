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

The committed v1 fixed prefix is 51 bytes.

| Byte range | Width | Field | Type | Meaning |
|------------|------:|-------|------|---------|
| `0..6` | 6 | `magic` | bytes | Exact bytes `46 4f 43 52 51 00` (`b"FOCRQ\0"`). |
| `6..10` | 4 | `format_version` | `u32` | Current value `1`. |
| `10..11` | 1 | `arch_target` | `u8` enum | Offline packing target, see below. |
| `11..43` | 32 | `source_sha256` | `[u8; 32]` | SHA-256 of source safetensors shard. |
| `43..51` | 8 | `header_len` | `u64` | Length of `header_json_utf8`. |

The header begins at byte 51. The payload begins at `51 + header_len` and
continues to end-of-file. v1 has no fixed-prefix `header_sha256` or
`payload_len`; byte-range validation is performed against the remaining payload
length.

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
  "arch_target": 1,
  "format_version": 1,
  "license_notice": "Baidu Unlimited-OCR - Copyright (c) 2026 Baidu, MIT License",
  "model_config": {},
  "packing_manifest": {},
  "provenance": {},
  "source_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "tensors": {}
}
```

A non-default architecture additionally declares a `model_id` (placed between
`model_config` and `packing_manifest`), e.g. for a GOT-OCR2 artifact:

```json
{
  "arch_target": 0,
  "format_version": 1,
  "license_notice": "GOT-OCR2.0 (StepFun) - Apache-2.0",
  "model_id": "got-ocr2",
  "tensors": {}
}
```

The fixed prefix is authoritative for `format_version`. The writer also emits a
forward-compatible JSON `format_version` mirror, but current readers ignore
unknown header fields. `arch_target` and `source_sha256` are emitted in both
places; the reader prefers the header value when non-default/non-empty and falls
back to the prefix value otherwise.

Required top-level fields:

| Field | Type | Required rule |
|-------|------|---------------|
| `arch_target` | integer | One of the `arch_target` numeric values below. |
| `source_sha256` | hex string | 64 lowercase hex chars when known; empty means use the prefix bytes. |
| `license_notice` | string | The notice the **declared architecture** requires (see `model_id`). For `unlimited-ocr` / v1 artifacts: non-empty and contains `Copyright (c) 2026 Baidu` and `MIT License` (the project writer fills this from `FOCR_MODEL_LICENSE_NOTICE`). For any other arch: equal to that arch's `ModelArch::license_notice` (e.g. GOT-OCR2 → `GOT-OCR2.0 (StepFun) - Apache-2.0`). |
| `tensors` | object | Map from canonical tensor name to one tensor record. |

Forward-compatible optional fields:

| Field | Type | Rule |
|-------|------|------|
| `format_version` | integer | Writer emits the prefix version for traceability. |
| `model_id` | string | The architecture id (`ModelArch::id`) the loader selects from the model registry. **Absent ⇒ the default `unlimited-ocr`** (so every v1 artifact loads unchanged); a non-empty id not in this binary's registry is **refused** (forward-incompatible artifact). The writer omits it for the default arch so re-converting Unlimited-OCR stays byte-identical to v1. |
| `provenance` | object | Contains pinned commits and source hashes when attached. |
| `model_config` | object | Forward-compatible config metadata when attached. Format-v1 readers do not treat it as validated or authoritative; see below. |
| `packing_manifest` | object | Converter/packing metadata and optional bit-allocation table when attached. |

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

In the committed v1 reader/writer, `tier` is an unsigned integer provenance
field, not a JSON string enum. High-precision tensors omit it and readers default
an absent value to `0`. Quantized writer records include it:

- `QInt8PerChan` must use `tier = 0`.
- `QInt4PerGroup` uses `tier` as opaque converter / allocator provenance.
  `group_size`, not `tier`, is the normative runtime contract for 16 vs 32
  element groups.

The `tier` value is not a runtime dispatch knob.

## Tensor Directory

The current committed v1 header stores tensors as an object keyed by canonical
tensor name:

```json
{
  "tensors": {
    "decoder.embed": {
      "byte_len": 165478400,
      "byte_offset": 0,
      "dtype": "BF16",
      "shape": [129280, 1280]
    },
    "decoder.expert.up_proj": {
      "byte_len": 573440,
      "byte_offset": 165478400,
      "dtype": "QInt4PerGroup",
      "group_size": 16,
      "scales_len": 8960,
      "scales_offset": 166051840,
      "shape": [896, 1280],
      "tier": 3
    }
  }
}
```

Required tensor-record fields:

| Field | Type | Rule |
|-------|------|------|
| `dtype` | enum | One of the dtype strings above. |
| `shape` | array of integers | Logical dequantized shape. |
| `byte_offset` | integer | Offset into payload. 64-byte aligned when writer alignment is enabled. |
| `byte_len` | integer | Data byte length. Must be non-zero unless the tensor is explicitly empty in config. |

Quantized tensor-record fields:

| Field | Type | Rule |
|-------|------|------|
| `scales_offset` | integer | Offset into payload for scale bytes. Required for quantized tensors. |
| `scales_len` | integer | Scale byte length. Required for quantized tensors. |
| `group_size` | integer | Required for `QInt4PerGroup`; must be `0` for `QInt8PerChan`. |
| `tier` | integer | Required in writer output for quantized tensors; see `tier` above. |

Loader validation:

- Tensor names are unique by construction as object keys.
- `byte_offset + byte_len <= payload_len`.
- If `scales_len > 0`, `scales_offset + scales_len <= payload_len`.
- Data and scale ranges do not overlap.
- `shape` matches the expected model census for `name`.
- `dtype` is compatible with `tier`.
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

### Dynamic Activation Quantization

The runtime dynamic activation quantizer uses the same deterministic rounding
rule as stored weights. For each activation row:

```
scale_a[row] = max(abs(x[row, :])) / 127
q[row, k] = round_ties_to_even(clamp(x[row, k] / scale_a[row], -127, 127))
zero_point = 0
```

Rules:

- Scalar, row-parallel, and architecture-specific GEMM paths must consume
  activation bytes produced by this exact rule.
- `round_ties_to_even` is banker's rounding. Exact ties such as `±0.5`,
  `±1.5`, and `±2.5` after division round to `0`, `±2`, and `±2` respectively.
- The contract is true division by `scale_a`, not reciprocal multiply. That
  avoids one-ULP boundary flips on non-power-of-two scales.
- Clamp range is `[-127, 127]`; `-128` is outside the symmetric operand domain.
- An all-zero activation row stores `scale_a[row] = 1.0` and all `q` values zero,
  avoiding NaN/Inf and making repeated runs byte-identical.
- U8S8 converter-side activation helpers use the same `round_ties_to_even`
  convention with clamp range `[0, 255]` and an explicit integer zero point.

### `QInt4PerGroup`

Int4 is defined now so the v1 reader and writer can reject or inspect it. Full
converter integration lands later.

Rules:

- `group_size` is either 16 or 32.
- Groups are along the K/input dimension within each output row.
- `shape[1]` must be an exact multiple of `group_size`; partial final groups
  are not part of the v1 format.
- `tier` is numeric converter provenance. `group_size` is the normative group
  contract.
- Each logical value is signed two's-complement int4 in `[-8, 7]`.
- Two int4 values pack into one byte: low nibble first, then high nibble.
- Scale count is `shape[0] * (shape[1] / group_size)`.
- `scales_len == scale_count * 4`.
- Logical dequantization is `f32(q4) * scale[row, group]`.

## Deferred `model_config` contract

There is not yet one canonical, versioned `model_config` object embedded in the
runtime. Format-v1 writers may attach this field for forward-compatible
provenance, but current readers ignore it. They do **not** claim that an attached
object was checked or use it to reinterpret payload bytes.

Unlimited-OCR v1 acceptance instead uses the compiled exact 2,710-tensor
name/shape/dtype manifest, the pinned declared source hash, the exact quant
recipe, and runtime constants compiled from the truth pack. This avoids creating
a second, drifting config schema merely to validate an optional header field.

A future format version may promote `model_config` to a validated contract after
one canonical serialized schema is generated from the truth pack and shared by
the writer and reader. That schema is expected to include at least:

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

Until that canonical object exists, tooling must not describe `model_config` as
validated. Once promoted, readers must compare the complete versioned object
against the compiled truth-pack representation and report a mismatch as
`FormatMismatch`, not a warning.

## `packing_manifest`

`packing_manifest` records converter decisions that are not tensor data:

```json
{
  "converter_version": "franken_ocr 0.1.0",
  "created_utc": "2026-06-25T00:00:00Z",
  "quant_recipe": "unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1",
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
- decoder attention `q/k/v/o`
- `lm_head`

The default validated quantized set is exactly the 2,148 decoder
FFN/expert/dense matrices. The 48 attention projections and `lm_head` are
separate measured levers behind independent kill switches, not baseline
assumptions.

## Loader Algorithm

1. Read the file into one blob or mmap.
2. Check length is at least 51 bytes.
3. Check magic equals `b"FOCRQ\0"`.
4. Parse fixed prefix with little-endian integer reads.
5. Check `format_version` is supported.
6. Check `51 + header_len <= file_len` with checked arithmetic.
7. Parse header JSON.
8. Resolve `arch_target` and `source_sha256` from the header when present, or
   from the fixed prefix otherwise. Reject `arch_target` outside the closed v1
   enum `0..=3`; the bounded production probe also requires the mirrored header
   and prefix values to agree.
9. Resolve `model_id` against the model registry: absent ⇒ the default
   `unlimited-ocr`; a non-empty id the registry lacks ⇒ refuse (forward-incompatible).
10. Check `license_notice` against the resolved arch: the Baidu MIT attribution for
    `unlimited-ocr`, else exact equality with that arch's `ModelArch::license_notice`.
11. For an Unlimited-OCR `.focrq`, validate every present tensor dtype against
    `unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1`: recipe FFNs must be
    `QInt8PerChan`; every other tensor must be BF16 or F32. Reject legacy
    full-int8 artifacts before any forward.
12. Treat optional v1 `model_config` as unvalidated provenance; use the compiled
    exact tensor manifest and runtime constants as the active model contract.
13. Validate every tensor record and byte range in the `tensors` map, including
    scale cardinality, bounds, and global non-overlap of every non-empty data and
    scale interval.
14. Build an immutable map `name -> TensorRange`.
15. Warn on compatible arch mismatch; error on incompatible packing.

No tensor dequantization is required during header sniffing.
`native_model_available` performs the bounded structural, recipe, provenance-
header, and exact schema checks. It does not read or hash the full payload and
therefore does not claim whole-file identity.

## Local-file integrity boundary

`source_sha256` is converter provenance: the CLI hashes the exact source bytes it
read and records that digest in the artifact. At load time the reader validates
the field's syntax, its fixed-prefix/header agreement, and (for Unlimited-OCR)
its equality to the pinned source digest. It does not recompute the source digest
from a file that is no longer present, nor does it hash every local `.focrq`
payload on each load.

Whole-file integrity belongs to these boundaries:

- `focr pull` verifies every part plus the reassembled artifact against the
  committed download manifest before installation.
- Model-gated release tests hash the pinned raw shard and conservative artifact
  before exercising their exact schemas.
- A direct `FOCR_MODEL_PATH` or an already-mutated cache file is trusted local
  input after structural/schema validation. Operators requiring tamper detection
  must verify its whole-file SHA-256 against a trusted manifest before loading.

The owned-buffer default prevents a concurrent truncate from faulting through a
memory map; it does not turn a concurrently modified local file into an atomic or
cryptographically authenticated snapshot. `FOCR_MMAP=1` additionally requires
an inode guaranteed immutable for the process lifetime.

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
| Unknown `arch_target` outside `0..=3` | `FormatMismatch` / exit 7 |
| Invalid JSON header | `FormatMismatch` / exit 7 |
| Missing license notice | `FormatMismatch` / exit 7 |
| Declared source / recipe / exact census mismatch | `FormatMismatch` / exit 7 |
| Out-of-range tensor byte range | `FormatMismatch` / exit 7 |
| Overlapping tensor data/scale ranges | `FormatMismatch` / exit 7 |
| Incompatible arch packing | `FormatMismatch` / exit 7 |

Warnings are allowed only for compatible arch fallback, and must name both the
file target and selected backend.

## Minimal Header Example

```json
{
  "arch_target": 0,
  "format_version": 1,
  "license_notice": "Baidu Unlimited-OCR - Copyright (c) 2026 Baidu, MIT License",
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
  "source_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "tensors": {}
}
```
