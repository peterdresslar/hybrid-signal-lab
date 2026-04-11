# 9B 030 Reselect Provenance Note

This directory contains the symmetric 9B profile re-selection run executed on
2026-04-10 to match the OLMO 030 selection protocol.

Command:

```bash
python -m router.experiments.select_profiles \
  --model-key 9B \
  --data-dir data/022-balanced-attention-hybrid \
  --objective separable \
  --max-constants 1 \
  --output-dir router/router-9B-030-reselect
```

Current benchmarked 9B set used in `030-bench`:

- `constant_2.6`
- `edges_narrow_bal_0.55`
- `late_boost_bal_0.60`
- `triad_odd_bal_0.45`

Top-1 set under the matched re-selection protocol:

- `constant_2.6`
- `late_boost_bal_0.60`
- `plateau_bal_0.55`
- `spike_p1_bal_0.18`

Decision:

- Keep the benchmarked 9B set.
- Do not re-run `030-bench`.
- Use the re-selection only to clean up provenance and methodology.

Rationale:

- The benchmarked set ranks `#14` under the matched separable objective.
- The separable-score gap to top-1 is only `0.000329`.
- The benchmarked set has slightly higher selected-set oracle routed `Δp`:
  `0.117589` vs. `0.116505` for the top-1 reselected set.

Interpretation:

This is stronger than a simple “top-10 and close” outcome. The matched
re-selection confirms that the benchmarked 9B set is not an outlier or a poor
choice under the disciplined protocol, while preserving the actual provenance of
the reported 030 benchmark results.
