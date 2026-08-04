# Profiling Hypothesis Ledger

| Hypothesis | Verdict | Evidence | Next test |
|---|---|---|---|
| Sparse pages are fixed-cost vision-bound. | supports | vision is 4.97 s of 7.79 s; SAM is 3.93 s | 20-run sparse baseline and SAM substage distribution |
| Dense pages shift to per-token decode. | supports | decode is 7.34 s of 13.43 s and scales to 495 tokens | output-length sweep on the same page |
| Experts are the largest decoder category. | supports | 41.8% dense and 42.4% sparse attributed decode | distinguish MAC time, weight stalls, and Rayon overhead |
| R-SWA is a co-equal decode target. | supports | 36.5% dense decode plus 1,236 independent leaf samples | split projections, RoPE, score/value, and synchronization |
| lm-head is the dominant decode cost. | rejects | 15.0%, materially below experts and attention | only pursue exact certified-search ideas after higher-EV work |
| Steady-state disk I/O is limiting inference. | rejects | warm mmap runs report zero block I/O | keep cold-start as a separate scenario |
| Handwritten SDOT should replace current autovec. | rejects by current negative evidence | current commit measured autovec 4.4-4.5x faster at m=1 and 22% e2e | retry only on new silicon/LLVM or a multi-row call shape |
| SAM row tiling will improve cache residency. | rejects by current negative evidence | previous row tiling regressed about 3% from dispatch multiplication | retry only as one fused internal kernel after traffic counters support it |
| Approximate/polynomial softmax is a safe vision win. | rejects by current negative evidence | no wall gain and token failures from tiny probability drift | exact-order alternatives only |
| Rayon scheduling is over-provisioned for decode. | unresolved | waits/yields are prominent, but sampling workers naturally includes idle time | AF-5 1/2/4/6/8/10 thread sweep on a quiet host |
| Buffer churn is a top-five cost. | unresolved | peak footprint is about 9.3 GB, but xctrace was blocked before recording | allocator-native or authorized Instruments trace |
| Multi-page execution changes the ranking. | unresolved | no current-HEAD 10/20-page capture yet | profile multi-page and continuous-batch modes separately |

No source optimization is authorized until the unresolved profiling gates in
`README.md` are closed and the candidate scores at least 2.0.
