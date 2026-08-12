# Refused timing captures — bd-K2 (wasm vision f32 pass)

Per PERF-RITUAL §3 and §6, timing rows are REFUSED, not averaged, when the host
is contended. This host carried `load averages: 33–54` for the entire pass
(4–8 concurrent `rustc` processes from sibling agent sessions, each well above
the 0.1-core external-process ceiling). Every wall-clock and GF/s number below
is recorded as **attribution-only, `publishable_timing: false`**.

## What that invalidated

`node smoke-unlimited.mjs site/pkg` on the SAME shipped module that the
bd-K-wasm-simd128 row measured at 48.92 s read **180.52 s** (vision 115.01 s,
prefill 37.25 s, decode 28.16 s). The stage *shares* are stable and usable
(vision 63.7% vs the 65.8% baseline — vision is still the bucket); the absolute
numbers are ~3.7x inflated by contention and are NOT comparable to the ledger.

## What survived anyway (load-insensitive)

- **Bit-exactness receipts.** PERF-RITUAL §3: "parity receipts are
  LOAD-INSENSITIVE". Every checksum verdict in this dir is admissible.
- **Structural facts.** The disassembly census in `microkernel-disasm.out` is a
  property of the shipped bytes, not of the clock.
- **Large, consistent, same-run directional ratios.** The +simd128 : scalar-
  fallback sgemm ratio reproduced at 2.79–4.48x on 5/5 shapes across three
  independent runs at loadavg 36/40/50, with both arms interleaved ABBA inside
  the same process. The DIRECTION is admissible; the magnitude is not.

## What was refused outright

- `mc_probe.mjs` (MATMUL_SGEMM_MC ∈ {64,128,256}). Two runs disagreed on the
  winner AND on which arm won the paired sign test; the MC=64 arm alone read
  9.09 / 20.97 GF/s on the same shape 3 minutes apart (cv >> 5%). The BIT-EQUAL
  verdict stands; the speed verdict is UNDECIDED, blocked on a quiet host.
- `glue_probe.mjs` (layer_norm / softmax / bf16-widen). The softmax ratio read
  0.68x then 1.25x on consecutive runs. Only the sdpa result (2.27–5.42x, i.e.
  already vectorized, doctrine #3 applies) is stable enough to act on — and it
  says DO NOT TOUCH, which is the safe direction to be wrong in.
- `relaxed_probe.mjs`. Reported 59.61 GF/s on one shape and 23.17 GF/s on a
  smaller one in the same run — physically incoherent, discarded. Only its
  `checksum DIFFER` verdict is retained (see the NEGATIVE_EVIDENCE row).
