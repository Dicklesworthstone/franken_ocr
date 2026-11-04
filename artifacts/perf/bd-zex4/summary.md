
## unlimited-ocr

| FOCR_THREADS | reps | real s (best, sum of pages) | user s | sys s | sys share | max RSS GiB | cv % |
|---|---|---|---|---|---|---|---|
| default | 3 | 245.27 | 1033.2 | 4363.9 | 81% | 8.09 | 15.8 |
| 1 | 1 | 483.17 | 455.4 | 16.8 | 4% | 7.39 | — |
| 2 | 1 | 276.03 | 467.9 | 30.4 | 6% | 7.55 | — |
| 4 | 1 | 174.22 | 491.6 | 75.8 | 13% | 7.85 | — |
| 8 | 3 | 99.37 | 520.6 | 139.2 | 21% | 8.51 | 13.5 |
| 12 | 3 | 93.75 | 557.6 | 302.1 | 35% | 9.12 | 7.0 |
| 18 | 3 | 99.04 | 626.9 | 666.0 | 52% | 8.85 | 7.9 |
| 24 | 3 | 109.62 | 691.5 | 1221.4 | 64% | 8.16 | 9.0 |
| 36 | 3 | 248.68 | 1059.7 | 4535.0 | 81% | 8.11 | 16.1 |

Output bytes across widths:
- page_0014: identical
- page_0023: identical
- page_0590: identical

AF-5 USL fit (usl_unlimited-ocr.json): {"alpha": 0.0209, "arch": "apple-m5-ultra-12p24e", "beta": 0.008505, "cap_is_win": true, "chosen_pool_n": 11, "decision": "cap-at-usl-peak", "degenerate": false, "fallback_used": false, "noisy": true, "num_cpus": 36, "op_class": "unlimited-ocr-page-e2e", "peak_n": 11, "peak_n_real": 10.7295, "physical_cores": 36, "predicted_gain_pct": 77.3, "r2": 0.9094, "rmse": 0.4687, "schema_version": 1, "speedup_at_num_cpus": 2.8918, "speedup_at_peak": 5.1282}

## unlimited-ocr-2

| FOCR_THREADS | reps | real s (best, sum of pages) | user s | sys s | sys share | max RSS GiB | cv % |
|---|---|---|---|---|---|---|---|
| default | 3 | 157.70 | 668.5 | 2678.1 | 80% | 8.11 | 15.6 |
| 1 | 1 | 289.95 | 275.0 | 10.8 | 4% | 7.40 | — |
| 2 | 1 | 166.35 | 283.1 | 16.9 | 6% | 7.55 | — |
| 4 | 1 | 104.57 | 303.3 | 40.0 | 12% | 7.93 | — |
| 8 | 3 | 69.92 | 328.4 | 94.7 | 22% | 8.51 | 14.4 |
| 12 | 3 | 64.74 | 348.8 | 190.2 | 35% | 9.14 | 19.8 |
| 18 | 3 | 66.73 | 387.8 | 388.3 | 50% | 8.88 | 23.4 |
| 24 | 3 | 76.95 | 462.0 | 752.4 | 62% | 8.14 | 28.7 |
| 36 | 3 | 151.56 | 625.6 | 2596.1 | 81% | 8.12 | 15.4 |

Output bytes across widths:
- page_0014: identical
- page_0023: identical
- page_0590: identical

AF-5 USL fit (usl_unlimited-ocr-2.json): {"alpha": 0.0534, "arch": "apple-m5-ultra-12p24e", "beta": 0.008598, "cap_is_win": true, "chosen_pool_n": 11, "decision": "cap-at-usl-peak", "degenerate": false, "fallback_used": false, "noisy": true, "num_cpus": 36, "op_class": "unlimited-ocr-2-page-e2e", "peak_n": 11, "peak_n_real": 10.4923, "physical_cores": 36, "predicted_gain_pct": 68.8, "r2": 0.9242, "rmse": 0.347, "schema_version": 1, "speedup_at_num_cpus": 2.6269, "speedup_at_peak": 4.435}
