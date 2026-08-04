# page_0590 precision attribution

- Binary, page, model, reference, token cap, and eight-thread topology were fixed.
- Conservative FFN/expert-only int8: normalized CER `1.24286`, 32,694 output
  characters, and 12,000 generated tokens without EOS.
- Experimental full-int8 attention/lm-head: normalized CER `1.63831`, 41,531
  output characters, and 12,000 generated tokens without EOS.
- Full-int8 regressed normalized CER by `0.39545` absolute (`31.82%` relative
  to conservative) while making decode `2.20x` faster.
- Verdict: reject the experimental full-int8 gates. The conservative result is
  materially better but still far outside the release budget, so
  `bd-2mo.30.12` remains an open P0 blocker requiring a deterministic fallback.
- The optimized exact CER kernel was differentially checked against the former
  dynamic program over 16,129 exhaustive and 10,000 randomized Unicode pairs.
