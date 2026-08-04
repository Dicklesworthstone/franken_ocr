# Ranked Hotspots

This table ranks current-HEAD evidence. Single-run wall values are attribution
only until the quiet-window baseline is complete. Decoder rows describe the
experimental full-int8 artifact that also quantizes attention and `lm_head`;
they are not a release-eligible conservative-recipe baseline.

## Sparse Page

| Rank | Location | Metric | Value | Category | Evidence |
|---:|---|---|---:|---|---|
| 1 | `vision_sam::forward_with` block stack | wall | 3.71 s block total; about 48% of e2e | CPU/compute | `raw/smoke_sparse.stderr` |
| 2 | SAM MLPs | wall | about 1.6 s across 12 blocks | CPU/GEMM | `raw/smoke_sparse.stderr` |
| 3 | SAM global attention | wall | about 1.3 s across 4 blocks | CPU/attention | `raw/smoke_sparse.stderr` |
| 4 | int8 decode experts | wall | 592 ms of 1.395 s attributed decode | CPU/memory | `raw/smoke_sparse.stderr` |
| 5 | R-SWA decode attention | wall | 486 ms of 1.395 s attributed decode | CPU/mixed | `raw/smoke_sparse.stderr` |

## Dense Page

| Rank | Location | Metric | Value | Category | Evidence |
|---:|---|---|---:|---|---|
| 1 | int8 expert/MLP GEMVs | wall | 2.957 s, 41.8% of attributed decode | CPU/memory | `raw/smoke_dense.stderr` |
| 2 | R-SWA attention and q/k/v/o projections | wall | 2.580 s, 36.5% | CPU/mixed | `raw/smoke_dense.stderr` |
| 3 | SAM block stack | wall | 3.55 s fixed page cost | CPU/compute | `raw/smoke_dense.stderr` |
| 4 | final norm and lm-head | wall | 1.058 s, 15.0% of attributed decode | CPU/memory | `raw/smoke_dense.stderr` |
| 5 | MoE routing | wall | 478 ms, 6.8% | CPU/control | `raw/smoke_dense.stderr` |

## Independent CPU/Contention Signals

| Rank | Frame or signal | Top-of-stack samples | Interpretation | Evidence |
|---:|---|---:|---|---|
| 1 | `simd::scalar::igemm_s8s8` | 17,847 | current Apple autovec integer GEMM/GEMV is the dominant active leaf | `dense-sample.txt` |
| 2 | `rswa::decode_attention` | 1,236 | attention arithmetic is independently visible | `dense-sample.txt` |
| 3 | `ft_kernel_cpu::linear_int8_dynamic_f32` | 1,213 | prefill/other dynamic-int8 linears remain material | `dense-sample.txt` |
| 4 | condition wait plus yield | 42,678 | Rayon scheduling/pool behavior needs a controlled scaling sweep; not proof of a defect | `dense-sample.txt` |
| 5 | mutex waits plus work stealing | 945 | contention is measurable but secondary to active GEMM work | `dense-sample.txt` |

Every listed source frame exceeds the gauntlet's 0.1% attribution floor.
