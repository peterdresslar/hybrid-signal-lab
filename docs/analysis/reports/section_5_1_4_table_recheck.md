# Section 5.1.4 Table Recheck

Reference manuscript section:
- [manuscript.md](/Users/peterdresslar/Workspace/cas-capstone-dresslar/docs/manuscript.md): `5.1.4 Constant gain vectors versus shaped gain vectors`

## Summary

I reran the `5.1.4` tables directly from the sweep artifacts in this repository.

- Table 1 is reproducible exactly for all 8 hybrid sweeps.
- For `Olmo block-out`, the manuscript row matches only if the exclusion set is:
  - `constant_0.4`
  - `constant_0.55`
  - `constant_1.8`
  - `constant_2`
  - `constant_2.3`
  - `constant_2.6`
  - `constant_3`
  - `early_boost_bal_0.60`
  - `late_boost_bal_0.60`
- Tables 2 and 3 are not reproducible under either of the two coherent prompt-level winner definitions available in the sweep data:
  - best profile by `target_rank`
  - best profile by `delta_target_prob`

The most coherent reading of the prose around Tables 2 and 3 is the `target_rank` version, because Table 1 is already formulated in terms of rank oracles.

## Table 1: Confirmed

Definition used:
- `Baseline rank`: mean `target_rank` of baseline rows
- `Constant-oracle rank`: per-prompt minimum `target_rank` over `baseline + constant profiles`
- `Full-oracle rank`: per-prompt minimum `target_rank` over `baseline + all retained profiles`
- `Constant share of headroom`: `(baseline - constant_oracle) / (baseline - full_oracle)`

Recomputed values:

| Sweep | Baseline rank | Constant-oracle rank | Full-oracle rank | Constant share of headroom |
|---|---:|---:|---:|---:|
| Qwen 2B attn-contr | 252.8 | 177.7 | 154.9 | 77% |
| Qwen 9B attn-contr | 109.2 | 72.3 | 60.9 | 76% |
| Qwen 35B attn-contr | 109.5 | 58.8 | 53.4 | 90% |
| Olmo attn-contr | 170.3 | 96.9 | 83.9 | 85% |
| Qwen 2B block-out | 252.8 | 139.3 | 85.7 | 68% |
| Qwen 9B block-out | 109.2 | 63.6 | 36.4 | 63% |
| Qwen 35B block-out | 109.5 | 59.3 | 30.8 | 64% |
| Olmo block-out | 170.3 | 110.0 | 53.7 | 52% |

## Tables 2 and 3: Not Confirmed As Written

The manuscript prose says:
- “when the oracle selects the best profile for an individual prompt”
- “oracle champions across both strategies”

The natural implementation of that language is:
- choose, for each prompt, the profile with the best `target_rank`
- classify the winner as `baseline`, `constant`, or `shaped`

Under that rank-based oracle definition, the sweep-level replacement for Table 2 is:

| Sweep | Baseline % | Constant % | Shaped % |
|---|---:|---:|---:|
| Qwen 2B attn-contr | 27.9% | 35.7% | 36.4% |
| Qwen 9B attn-contr | 30.7% | 43.0% | 26.4% |
| Qwen 35B attn-contr | 40.7% | 26.6% | 32.7% |
| Olmo attn-contr | 48.8% | 26.5% | 24.7% |
| Qwen 2B block-out | 26.6% | 17.8% | 55.6% |
| Qwen 9B block-out | 29.8% | 24.2% | 46.0% |
| Qwen 35B block-out | 36.3% | 10.7% | 53.1% |
| Olmo block-out | 46.1% | 13.6% | 40.4% |

And the model-level replacement for Table 3 is:

| Model | Baseline % | Constant % | Shaped % | Top non-baseline champion |
|---|---:|---:|---:|---|
| Qwen 2B | 27.2% | 26.8% | 46.0% | `constant_0.4` (attn, 2.9%) |
| Qwen 9B | 30.2% | 33.6% | 36.2% | `constant_2.3` (attn, 4.5%) |
| Qwen 35B | 38.5% | 18.6% | 42.9% | `constant_0.4` (attn, 4.9%) |
| Olmo Hybrid | 47.4% | 20.1% | 32.5% | `constant_3` (attn, 3.8%) |

## Interpretation

What is currently solid:
- Table 1 is good, provided the `Olmo block-out` note names the full 9-profile exclusion set explicitly.

What needs attention:
- Tables 2 and 3 do not follow from the sweep artifacts under the two obvious prompt-level oracle definitions.
- The manuscript text currently suggests a rank-based prompt oracle, but the printed percentages do not match that.
- Before revising prose around those tables, the winner definition should be fixed explicitly:
  - `target_rank` oracle
  - `target_prob` oracle
  - or a third convention if one was used during drafting

## Notes

I also checked a `delta_target_prob`-based prompt-winner convention using `analysis_prompt_winners.csv`. That convention also fails to reproduce the current Tables 2 and 3, so the mismatch is not just “rank vs. probability.”
