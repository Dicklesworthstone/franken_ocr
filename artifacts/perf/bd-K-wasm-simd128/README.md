# bd-K — wasm32 simd128 int8/int4 kernel island (browser lane)

Evidence for the `IsaTier::WasmSimd128` island (`src/simd/wasm128.rs`, commit
`5d63e3a`). Everything here was measured on the BUILT wasm artifact, never from
source: wasm perf claims from a native test are worthless.

## Harness

* **Host**: Apple M4, macOS 15 (Darwin 25.2.0), Node v25.4.0 / V8, 64 GB RAM.
* **Kernel micro-bench** (`kernel_bench_abba.txt`): a scratch `cdylib`
  (`scratchpad/wbench`) that links this crate `--no-default-features` for
  `wasm32-unknown-unknown` with `-C target-feature=+simd128` and exports raw
  `extern "C"` benches. Node instantiates it with an empty import object and
  times with `performance.now()` (wasm has no clock — timing is the caller's
  job). Operands come from a deterministic xorshift over the FULL byte range
  (a sparse matrix would flatter the kernel); every bench returns a checksum so
  the optimizer cannot delete the loop. Interleaved ABBA, 5 pairs, min-of-pairs
  reported; `scalar` calls `simd::scalar::*` / `int4::igemm_s4s8_packed_scalar`
  and `dispatch` calls the public dispatched entrypoint — same process, same
  allocator, same warm caches.
* **Real page** (`page_before_scalar.txt` / `page_after_simd128.txt`): the
  SHIPPED `focr-wasm` module driven from Node against the real 2.80 GiB
  `unlimited-ocr.wasm-int4.focrq` artifact (release `models-unlimited-wasm-v1`,
  parts pinned in `site/model-manifest.js`) on
  `tests/fixtures/got/sample_text.png`. Serial lane (`site/pkg`), single
  thread. Stage timings come from the shipped `set_progress_callback` seam,
  timestamped in JS.
* **before** = the same tree with `src/simd/{dispatch,int4,mod,scalar}.rs` from
  commit `8f79ef1` (pre-island) and no `wasm128.rs` module declaration, built
  with the identical flags into `scratchpad/pkg-before`. Only the kernel island
  differs; every other crate/flag/artifact is byte-identical between the two
  modules. Route asserted on each module before timing (`int8_route()` →
  `Scalar` vs `WasmSimd128`) so a silent scalar fallback cannot masquerade as a
  measurement.

## Headline (warm, best of the interleaved runs)

| stage | before (Scalar) | after (WasmSimd128) | ratio |
|---|---|---|---|
| recognize (end-to-end) | 131.78 s | 48.92 s | **2.69x** |
| prefill | 68.22 s | 10.70 s | 6.38x |
| decode | 33.12 s | 6.05 s | 5.47x |
| vision (f32/bf16, does NOT route through this island) | 30.43 s | 32.16 s | 0.95x |

Vision is now 66% of the page and is the honest next lever; it does not touch
the int8 dispatch at all, so its ~1.0 ratio is the control that proves the
measurement is attributing the win to the right code.

## Correctness

`recognize` output is BYTE-IDENTICAL across every run of both modules:

```
c521e4f0d43c8d40cf8d100adb4a8d71e5148bbe91785de14030afcd723a9b6a  recognize_output.md
```

(six runs: three per module.) In-module parity, run on the built artifact:
`simd::selftest()` reports 0 failures over its 44-case battery (both operand
domains, K tails, the model's real GEMV shapes, and the K=6848 worst-case
overflow stress), and an 11-shape int4 battery — both group sizes, ragged
tails, `k=6848`, `m>1` — is bit-identical in f32 to
`igemm_s4s8_packed_scalar`. The micro-bench additionally checks the
scalar/dispatch checksums match on every shape it times (`checksum =` in the
log).
