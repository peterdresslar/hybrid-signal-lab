# Analysis Layout

- `head_entropy/`
  - `scripts/`: reproducible head-entropy analysis code
  - `outputs/qwen9b/`: generated metrics and ranked tables for Qwen 9B
- `pca/`
  - `scripts/`: reproducible PCA analysis code
  - `outputs/qwen9b/`: Figure 6 diagnostics, CSV exports, and reports
  - `outputs/cross_model/`: lightweight cross-model PCA summaries
- `reports/`
  - markdown reports and one-off analysis writeups

Manuscript-facing figures live under `docs/figures/manuscript/`.
Diagnostic and exploratory figures live under `docs/figures/diagnostics/`.

Legacy entrypoints remain in `docs/data_analysis/code/` as thin wrappers so
existing commands continue to run.
