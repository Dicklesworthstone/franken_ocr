# U2-C5a head-to-head

## Single page (OmniDocBench)

- CER_text(v2) − CER_text(v1): n = 315, mean -0.0694, 95% [-0.0985, -0.0414]; non-inferior at δ = 0.005: True
- TEDS(v1) − TEDS(v2), single-table pages: n = 66, mean -0.0344, 95% [-0.0946, +0.0178]; non-inferior: False
- ED_all(v2) − ED_all(v1): n = 320, mean -0.0662, 95% [-0.0941, -0.0394]
- Sensitivity (not pre-registered): CER_text(v2) − CER_text(v1) over the pages both cells completed, so a timeout is not scored as 1.0: n = 279, mean +0.0021, 95% [-0.0128, +0.0175]
- Sensitivity (registered in the run-status note): every page lost to the 600 s budget rerun alone with no budget ({"v1_single_rerun": 36, "v2_single_rerun": 6}; failed again: {}). CER_text(v2) − CER_text(v1): n = 315, mean -0.0166, 95% [-0.0325, +0.0003]; non-inferior: True. TEDS(v1) − TEDS(v2): n = 66, mean +0.0109, 95% [-0.0149, +0.0400]; non-inferior: False
- v1_single: {"pages": 320, "failed": 36, "failed_by_source": {"academic_literature": 8, "book": 4, "magazine": 1, "newspaper": 23}, "mean_cer_text": 0.21927697829464715, "mean_ed_all": 0.2215908474733089, "mean_teds": 0.8881314797630305, "mean_formula_ed": 0.3635608786875623}
- v2_single: {"pages": 320, "failed": 0, "failed_by_source": {}, "mean_cer_text": 0.14988784197025262, "mean_ed_all": 0.15534606095974596, "mean_teds": 0.922486170834107, "mean_formula_ed": 0.31140470794520114}

| category | metric | n | mean Δ (v2 worse > 0) | 95% interval | regression (τ = 0.02) |
|---|---|---|---|---|---|
| data_source=PPT2PDF | cer_text | 49 | -0.0303 | [-0.0687, +0.0059] | False |
| data_source=academic_literature | cer_text | 44 | -0.1336 | [-0.2540, -0.0207] | False |
| data_source=book | cer_text | 52 | -0.0683 | [-0.1315, -0.0173] | False |
| data_source=colorful_textbook | cer_text | 24 | +0.0164 | [-0.0065, +0.0519] | False |
| data_source=exam_paper | cer_text | 39 | +0.0076 | [-0.0132, +0.0269] | False |
| data_source=historical_document | cer_text | 1 | -0.1828 | [-0.1828, -0.1828] | n < 15 |
| data_source=magazine | cer_text | 28 | -0.0214 | [-0.0629, +0.0024] | False |
| data_source=newspaper | cer_text | 26 | -0.5009 | [-0.6251, -0.3789] | False |
| data_source=note | cer_text | 25 | +0.0707 | [+0.0029, +0.1654] | True |
| data_source=research_report | cer_text | 27 | +0.0150 | [-0.0191, +0.0720] | False |
| language=english | cer_text | 144 | -0.1113 | [-0.1651, -0.0619] | False |
| language=simplified_chinese | cer_text | 143 | -0.0536 | [-0.0820, -0.0266] | False |
| language=en_ch_mixed | cer_text | 23 | +0.0936 | [+0.0124, +0.2007] | True |
| CJK | cer_text | 169 | -0.0344 | [-0.0625, -0.0063] | False |
| layout=1andmore_column | cer_text | 28 | -0.0474 | [-0.1296, +0.0073] | False |
| layout=double_column | cer_text | 34 | -0.0537 | [-0.1272, -0.0035] | False |
| layout=other_layout | cer_text | 67 | -0.1978 | [-0.2755, -0.1256] | False |
| layout=single_column | cer_text | 176 | -0.0083 | [-0.0378, +0.0214] | False |
| layout=three_column | cer_text | 10 | -0.3997 | [-0.6921, -0.1089] | n < 15 |
| special_issue=watermark | cer_text | 17 | +0.0070 | [-0.0146, +0.0312] | False |
| special_issue=colorful_backgroud | cer_text | 47 | -0.0095 | [-0.0424, +0.0185] | False |
| special_issue=fuzzy_scan | cer_text | 5 | -0.0071 | [-0.0185, +0.0007] | n < 15 |
| tables (TEDS) | teds | 66 | -0.0344 | [-0.0946, +0.0178] | False |
| formulas (FormulaED) | formula_ed | 72 | -0.0522 | [-0.1132, +0.0051] | False |

## Documents of 5–10 pages

- ED_all(v1_multi) − ED_all(v2_doc): n = 32, mean +0.4146, 95% [+0.3075, +0.5177]; v2 wins: True
  - marathon: n = 20, mean +0.4270, 95% [+0.2747, +0.5738]; regression: False
  - non-Baidu (born-digital): n = 12, mean +0.3940, 95% [+0.2713, +0.5171]; regression: n < 15
  - en: n = 10, mean +0.3617, 95% [+0.2265, +0.5019]; regression: n < 15
  - zh-containing: n = 22, mean +0.4386, 95% [+0.3011, +0.5741]; regression: False
- ED_all(v1_pp) − ED_all(v2_doc): n = 32, mean -0.0132, 95% [-0.0616, +0.0155]; v2 wins: False
  - marathon: n = 20, mean -0.0222, 95% [-0.1000, +0.0229]; regression: True
  - non-Baidu (born-digital): n = 12, mean +0.0017, 95% [-0.0040, +0.0091]; regression: n < 15
  - en: n = 10, mean -0.0016, 95% [-0.0057, +0.0009]; regression: n < 15
  - zh-containing: n = 22, mean -0.0185, 95% [-0.0887, +0.0226]; regression: False
- v2_doc: {"documents": 32, "failed": 0, "exit_nonzero": 0, "mean_ed_all": 0.10479611198745933}
- v1_multi: {"documents": 32, "failed": 0, "exit_nonzero": 0, "mean_ed_all": 0.5194103372676623}
- v1_pp: {"documents": 32, "failed": 0, "exit_nonzero": 0, "mean_ed_all": 0.09159402101991003}
- per page, v1_multi − v2_doc (both split): n = 36, mean +0.1377, 95% [+0.0589, +0.2304]

## Tail risk (AF-2, `scripts/af2_tail_risk.py`; not pre-registered, gates nothing)

Per-item error over the items both cells scored: the mean, VaR and CVaR at α = 0.1 (CVaR is the mean of the worst 10%), and the EVT p99.9 bound (a GPD tail fit, or the empirical quantile below 8 exceedances). The gate is AF-2's: the v2 cell's CVaR and EVT bound within τ = 0.02 of the v1 cell's.

| set | cell | n | mean | VaR_0.1 | CVaR_0.1 | EVT p99.9 | max | gate |
|---|---|---|---|---|---|---|---|---|
| single page, CER_text | v1_single | 315 | 0.2193 | 1.0000 | 1.0000 | 1.0000 (pwm) | 1.0000 | baseline |
| single page, CER_text | v2_single | 315 | 0.1499 | 0.4435 | 0.6995 | 1.0000 (pwm) | 1.0000 | pass |
| single page, CER_text, budget pages rerun | v1_single | 315 | 0.1663 | 0.5663 | 0.8085 | 1.0000 (pwm) | 0.9995 | baseline |
| single page, CER_text, budget pages rerun | v2_single | 315 | 0.1497 | 0.4435 | 0.6974 | 1.0000 (pwm) | 1.0000 | pass |
| documents, ED_all | v1_multi | 32 | 0.5194 | 0.8594 | 0.9331 | 0.9760 (empirical-fallback) | 0.9768 | baseline |
| documents, ED_all | v2_doc | 32 | 0.1048 | 0.2138 | 0.6339 | 0.8542 (empirical-fallback) | 0.8554 | pass |
| documents, ED_all | v1_pp | 32 | 0.0916 | 0.1935 | 0.4518 | 0.8037 (empirical-fallback) | 0.8195 | baseline |
| documents, ED_all | v2_doc | 32 | 0.1048 | 0.2138 | 0.6339 | 0.8542 (empirical-fallback) | 0.8554 | fail |

## Cost

| cell | pages | wall s/page | peak RSS GiB |
|---|---|---|---|
| v1_single | 320 | 131.9 | 9.68 |
| v2_single | 320 | 92.0 | 9.91 |
| v2_doc | 224 | 55.7 | 9.85 |
| v1_multi | 224 | 100.9 | 8.08 |
| v1_pp | 224 | 47.8 | 9.46 |
