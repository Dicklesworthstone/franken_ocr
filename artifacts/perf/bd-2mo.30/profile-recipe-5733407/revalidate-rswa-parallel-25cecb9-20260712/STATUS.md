# Current-tree R-SWA parallel-attention revalidation

## Verdict

`KEEP` the shipped parallel independent-head R-SWA schedule. The serial oracle
remains available through `FOCR_RSWA_PARALLEL_ATTN=0`.

This receipt-bound focr-versus-fallback A/B clears the `bd-2mo.30.15` gate for
this lever only. It is not a focr-versus-pinned-reference row and therefore is
not eligible for `docs/PERF_LEDGER.md`. The parent revalidation bead remains in
progress for the other retained levers.

## Bound inputs

- Workspace HEAD: `25cecb9c4cbfa599fbf48a45d9bf9748daab4192`.
- Pinned siblings: frankentorch `062cf3671c194f6ab184da98f0559ebc76cff7c7`,
  frankensqlite `cd9990bb16291d8c7c247b75b47faae8d7701adb`, and
  asupersync `53aa5c72f855352148c3a88e6961f7f09adb535c`.
- Source manifest: 22,489 entries, root
  `09a9893d1ae83d7bef652051e3772b2ee6d27eeec9c5768ad25f12a88dda9f0d`.
- Release-perf binary:
  `0e50d513e60956d544af4e7cd707e76888a00de83598563dffd4b0cbc94961c6`
  (96,506,416 bytes).
- Conservative model:
  `573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592`
  (4,157,448,783 bytes), recipe
  `unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1`.
- Tokenizer:
  `a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4`,
  fetched from the pinned HF commit
  `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`.
- Host: x86_64 Linux, dual-socket AMD EPYC 7282, AVX2, system allocator,
  eight focr threads, nightly-2026-07-09.

The release-perf producer ran through `rch exec`; RCH rejected the noncanonical
checkout path and failed open to the same quiet host. The receipt records that
exact command, toolchain, binary, and source closure.

## Method

Both workloads used one discarded warmup and five measured runs per arm in the
harness's balanced interleaved schedule. The only varied setting was
`FOCR_RSWA_PARALLEL_ATTN`: arm A used the shipped default (`<unset>`), while arm
B used the serial control (`0`). Both arms fixed:

```text
FOCR_DECODE_INT8=0 FOCR_INT8_ATTN=0 FOCR_INT8_LMHEAD=0
FOCR_ATTN_GEMM=<unset> FOCR_INT8_KV=<unset>
FOCR_SPEC_DECODE=<unset> FOCR_DECODE_STATELESS=<unset>
FOCR_THREADS=8 allocator=system precision=focr-mixed-ffn-int8
```

The dense workload was `ocr page_0014.png`. The broad workload was sequential
`ocr-batch` over the 20 sorted pinned pages with `FOCR_MAX_NEW_TOKENS=32`.
Pre-run load1 was `0.44` for dense and `0.58` for batch, both below the unforced
`2.0` gate. Post-run load is recorded only as an informational consequence of
the benchmark itself.

A setup-only attempt in a separate fresh directory stopped before warmup when
the copied fixture lacked `tokenizer.json`; it contributed no timing sample.

## Results

| Workload / stage | Parallel p50 | Serial p50 | Parallel / serial result | CV parallel / serial |
|---|---:|---:|---:|---:|
| dense end-to-end | 45.860 s | 49.028 s | 6.46% lower | 3.008% / 0.824% |
| dense decode | 24.260 s | 27.550 s | 11.94% lower | 2.211% / 1.655% |
| dense attention | 5.890 s | 8.646 s | 31.88% lower | 3.848% / 1.202% |
| 20-page end-to-end | 323.910 s | 326.374 s | 0.76% lower | 1.087% / 1.042% |
| 20-page decode | 31.750 s | 34.710 s | 8.53% lower | 1.459% / 1.220% |
| 20-page attention | 7.161 s | 9.932 s | 27.90% lower | 2.606% / 1.115% |

Dense RSS p50 was 10,126,893,056 bytes parallel versus 10,124,070,912 bytes
serial (+0.028%). Batch RSS p50 was 10,695,380,992 versus 10,694,914,048 bytes
(+0.004%).

All five measured outputs and the warmup were byte-identical within each arm
and across arms for both workloads. Dense output SHA-256 is
`7190e15dcbe5a85caf1fc61d2ac27aa2fc997e841965aceb0566cc39c8e13d0a`.
The full batch output identity is bound in `batch20/focr_ab.json`.

The focused median gain exceeds 2%, the broad workload does not regress, every
claimed-stage CV is at most 5%, and output identity holds. The result therefore
adds a second win for this retained lever: `W 2 / L 0 / N 0`.

## Bundle policy

The 96 MB binary, 4.16 GB model, tokenizer, and input images are intentionally
not duplicated in git. Their hashes, sizes, and paths are bound by the included
receipt and A/B documents. This bundle includes both raw schedules, all warmup
and measured stdout/stderr/meta observations, the exact source manifest, host
and load records, and a recursive `SHA256SUMS` manifest.
