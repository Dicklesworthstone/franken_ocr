# bd-2mo.5.1 / bd-2mo.4.1 — the tier truth on Apple M4 (2026-07-09, quiet window)

## Microbench (igemm_quiet_{sdot,smmla,scalar}.txt — cargo bench igemm_tier, FOCR_FORCE_ARCH per tier, error bars 1-10%)

ns/iter at the model's real shapes:

| shape | sdot | smmla | autovec-scalar | scalar/sdot |
|---|---|---|---|---|
| attn_proj m1 (1280x1280) | 93,775 | 530,325 | 21,145 | **4.4x faster** |
| attn_proj m16 | 672,148 | 1,572,608 | 340,669 | 2.0x |
| down m1 (K=6848) | 511,095 | 3,148,190 | 114,818 | **4.5x** |
| down m16 | 3,590,179 | 10,559,862 | 1,837,696 | 2.0x |
| gate_up m1 (N=6848) | 505,927 | 2,827,258 | 114,942 | **4.4x** |
| gate_up m16 | 3,613,625 | 8,384,675 | 1,839,588 | 2.0x |

* **SMMLA loses to SDOT 2.3-6.2x on every shape** (bd-2mo.4.1's verdict:
  the un-blocked-loses prior NE-INH-3 extends to the blocked kernel on
  M4's half-rate i8mm — SMMLA stays deprioritized in dispatch).
* **The LLVM-autovectorized scalar loop beats the hand-written SDOT
  micro-tile on EVERY shape** — NE-INH-1's inherited prior ("hand-written
  wide-SIMD int8 dot ~5x slower than autovec") catching the m=1 GEMV.

## E2E confirmation (real int8 decode, page_0009, interleaved runs, identical 98-token output)

* pre-lever binary: FOCR_FORCE_ARCH=sdot 1.61s/1.58s vs =scalar 1.27s/1.26s
* post-lever binary (default autovec vs FOCR_INT8_AUTOVEC=0):
  1.28s/1.27s vs 1.64s/1.64s — **22% decode win, default ON**

## The lever (src/simd/arm.rs autovec_preferred)

Scoped to the dense igemm_s8s8/u8s8 entrypoints on macOS Apple Silicon:
detect_tier() still reports SDOT truthfully (robot backends/selftest), the
int4 packed path keeps SDOT (its scalar fallback is the ledgered
5.8x-slower unpack), offline-SMMLA packed-B unaffected. Kill-switch
FOCR_INT8_AUTOVEC=0; FOCR_FORCE_ARCH overrides still mean what they say.
Bit-identical by construction (exact integer math; the tier parity suite
holds all routes equal).
