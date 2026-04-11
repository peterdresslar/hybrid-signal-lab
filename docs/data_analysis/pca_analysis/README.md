# PCA Analysis Summary

This folder collects reusable PCA-oriented diagnostics across the 022 run families.

## Notes

- Baseline PCA is computed from baseline `attn_entropy_per_head_final` vectors only.
- The two hybrid families (`022-balanced-attention-hybrid` and `022-balanced-block-hybrid`) share identical baseline PCA within a model, because the baseline forward pass is the same and only the intervention mode differs.
- `tokens_approx` from Battery 4 is used as the prompt-length proxy for the length-residualized summaries.

## Files

- `pca_model_summary.csv`: per-run summary of explained variance, prompt-length loading, and type-structure persistence before and after prompt-length residualization.
