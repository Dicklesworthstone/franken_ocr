# franken_ocr Test Logging Contract

This document defines the structured JSON-line log format used by unit,
conformance, model-gated, and end-to-end tests. The goal is diagnostic logs that
are cheap to emit, easy for agents to parse, and strict enough that a typo in a
stage name becomes a failing test rather than hidden prose.

The machine-readable schema lives in
`tests/fixtures/test_log_schema.json`. The validator is
`scripts/check_test_logs.py`.

## Stream Rules

- Emit one JSON object per line.
- Test data goes to stdout or to a captured `.ndjson` file. Human diagnostics go
  to stderr.
- A successful validator run exits 0. Any malformed line exits non-zero.
- `schema_version` is mandatory on every line and currently equals `1`.
- Timing fields are scrubbable. Golden comparisons must ignore `ts` and
  `elapsed_us` unless the test explicitly validates timing.
- Logs may be buffered, but test log writers must flush on drop and flush during
  panic handling so a crashing test leaves the last useful events on disk.
- Do not emit per-inner-loop logs in hot kernels. Emit at setup, stage boundary,
  parity assertion, skip, result, and error boundaries.

## Required Common Fields

Every log line has these fields:

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | integer | Must equal `1`. |
| `ts` | number | Monotonic-relative timestamp. Scrubbable. |
| `test` | string | Rust test function or script test name. |
| `case` | string | Subcase id, fixture id, or corpus item id. |
| `run_seq` | integer | Per-test invocation counter. Use it to group multiple forwards in one test. |
| `event` | enum | `setup`, `stage`, `parity`, `assert`, `skip`, `result`, or `error`. |
| `result` | enum | `pass`, `fail`, `xfail`, `skip`, or `skip_no_model`. |

`trace_id` is optional. Use it when multiple workers or pages can interleave
lines. The tuple `(test, case, run_seq, trace_id)` should identify one logical
pipeline invocation.

## Event Types

### `setup`

Use for fixture selection, model resolution, seed selection, and backend setup.

Additional required fields:

- `seed`

### `stage`

Use for pipeline and decoder-stage boundaries. A stage line should be enough to
answer "what shape, dtype, backend, and elapsed time reached this point?"

Additional required fields:

- `stage`
- `inputs`
- `shapes`
- `dtype`
- `elapsed_us`
- `simd_tier`
- `seed`

`inputs` is an object whose keys name logical inputs and whose values are compact
descriptors, for example:

```json
{"image": "golden/base_001.png", "tokens": 273}
```

`shapes` is an object whose values are shape arrays or shape strings, for
example:

```json
{"hidden": [273, 1280], "sam_tokens": [256, 1024]}
```

Use `layer_idx` on decoder-internal stage lines.

### `parity`

Use for L0-L5 comparisons against the oracle.

Additional required fields:

- `gate`
- `metric`
- `value`
- `tolerance`
- `oracle_fixture`
- `oracle_sha256`
- `nondeterminism_envelope`
- `pass`

When `simd_tier` is present and equals `avx2`, a parity line must also carry an
`avx2_exception` field if it is compared against an i32-exact reference. The
AVX2 tier may use an i16-saturating path; the exception must point at the
corresponding `DISC-NNN` if the divergence is accepted.

### `assert`

Use for ordinary non-parity assertions.

Additional required fields:

- `assertion`
- `pass`

### `skip`

Use for model-gated skips. Missing model weights or missing CUDA reference hosts
are skips, not failures, when the bead explicitly allows skip-with-success.

Additional required fields:

- `reason`

When proving the native path ran in a model-gated test, include:

- `native_path_ran`
- `fallback_target`

If `native_path_ran` is `true`, `fallback_target` must be `/nonexistent`.

### `result`

Use once at the end of a test case or pipeline invocation.

Additional required fields:

- `elapsed_us`

When a result line records `result="fail"`, it must include `diag`.

### `error`

Use for caught `FocrError`s, panics, failed assertions, or malformed fixture
conditions that should be machine-readable.

Additional required fields:

- `diag`

`diag` is an object with:

- `error_kind`
- `focr_exit_code`
- `message`

## Enumerations

### Events

- `setup`
- `stage`
- `parity`
- `assert`
- `skip`
- `result`
- `error`

### Results

- `pass`
- `fail`
- `xfail`
- `skip`
- `skip_no_model`

### Pipeline Stages

- `decode_image`
- `preprocess`
- `tokenize`
- `vision_sam`
- `vision_clip`
- `vision_bridge`
- `connector`
- `prefill`
- `decode`
- `postprocess`

### Decoder-Internal Stages

- `embed`
- `rmsnorm`
- `rope`
- `rswa_attn`
- `moe_router`
- `moe_expert`
- `dense_mlp`
- `lm_head`
- `kv_cache`

### Dtypes

- `f32`
- `bf16`
- `i8`
- `i4`
- `u8`

### SIMD Tiers

- `smmla`
- `sdot`
- `vnni512`
- `vnni256`
- `amx`
- `avx2`
- `scalar`
- `none`

`none` is allowed only for setup, skip, result, and other non-kernel diagnostic
events. Stage lines should use the dispatched tier that actually ran.

### Parity Gates

- `L0`
- `L1`
- `L2`
- `L3`
- `L4`
- `L5`

### Metrics

- `cosine`
- `max_abs_diff`
- `cer`
- `teds`
- `formula_cdm`
- `argmax_match`
- `token_exact`

## Example

```json
{"case":"base_001","event":"stage","inputs":{"image":"tests/fixtures/base_001.png"},"result":"pass","run_seq":0,"schema_version":1,"seed":1234,"shapes":{"hidden":[273,1280]},"simd_tier":"scalar","stage":"preprocess","test":"preprocess_l0","ts":1.2,"dtype":"f32","elapsed_us":412}
{"case":"base_001","event":"parity","gate":"L0","metric":"max_abs_diff","nondeterminism_envelope":{"source":"oracle_nondeterminism_envelope.json","max_abs_diff":0.0},"oracle_fixture":"tests/fixtures/native/base_001/preprocess.npy","oracle_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pass":true,"result":"pass","run_seq":0,"schema_version":1,"test":"preprocess_l0","tolerance":0.0,"ts":1.3,"value":0.0}
{"case":"base_001","event":"result","elapsed_us":500,"result":"pass","run_seq":0,"schema_version":1,"test":"preprocess_l0","ts":1.4}
```

## Validator

Run the schema self-test:

```bash
python3 scripts/check_test_logs.py --self-test
```

Validate captured logs:

```bash
python3 scripts/check_test_logs.py tests/artifacts/some-test.ndjson
```

The validator emits one NDJSON result per checked line and rejects:

- missing required fields
- unknown `event`, `stage`, `result`, `dtype`, `simd_tier`, `gate`, or `metric`
- mismatched `schema_version`
- `result="fail"` without `diag`
- `event="error"` without `diag`
- `native_path_ran=true` without `fallback_target="/nonexistent"`
- AVX2 parity lines without an `avx2_exception`
