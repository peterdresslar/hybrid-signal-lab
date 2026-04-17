# Sequence Data Manifest

Reference for the artifacts written by `uv run -m signal_lab.sequence_analyze`.
This document describes the analyzer output directory, not the upstream raw
`collect_sequences.py` run directory.

Example current outputs:

- `data/040-analysis/9B-analysis/`
- `data/040-analysis/OLMO-analysis/`

The analyzer writes three kinds of artifacts:

- run-level summary files
- one PCA CSV per hidden-state feature family
- one figure directory per hidden-state feature family

## Root Files

| File | Purpose |
|------|---------|
| `sequence_analysis_prompt_scalars.csv` | Prompt-level scalar metadata copied from the collection records: prompt id/type/source, `tokens_approx`, token count, layer count, hidden size, and entropy summaries. Use this as the compact prompt metadata table for joins and quick filtering. |
| `sequence_analysis_family_summary.csv` | One row per feature family with the main PCA diagnostics: feature dimension, first five explained-variance ratios for raw and length-residualized PCA, correlations with prompt length, and task η² on the first two PCs. This is the fastest summary table for comparing families. |
| `sequence_analysis_manifest.json` | Minimal manifest for the analysis bundle: source run directory, output directory, prefix, number of prompts, and the list of feature families present. Useful for provenance and programmatic loading. |

## Feature Families

`sequence_analyze.py` currently emits one PCA CSV and one figure directory for
each of these six feature families:

- `embedding_last_token`: embedding-layer hidden state at the final prompt token
- `final_layer_last_token`: final-layer hidden state at the final prompt token
- `embedding_mean_pool`: mean-pooled embedding-layer hidden state over prompt tokens
- `final_layer_mean_pool`: mean-pooled final-layer hidden state over prompt tokens
- `all_layers_last_token_concat`: concatenation of all layers at the final prompt token
- `all_layers_mean_pool_concat`: concatenation of mean-pooled states from all layers

### More information

1. The embedding layer hidden state can be thought of as the "input" to the model. It is the representation of the prompt before any processing by the model's layers.
2. The final layer hidden state is the output of the model. It is the representation of the prompt after processing by all of the model's layers.
3. The mean-pooled hidden states are the average of the hidden states over all tokens in the prompt. This can be thought of as a summary of the prompt's representation.
4. The concatenated hidden states are the concatenation of the hidden states from all layers. This can be thought of as a summary of the prompt's representation.

By “hidden state” we are referring to the model’s internal activation vector at a given layer and token position: the
learned numerical representation the network carries forward while computing its next-token prediction.

## Per-Family CSVs

For each family `{family}`, the analyzer writes:

| File | Purpose |
|------|---------|
| `{family}_pca.csv` | Prompt-level PCA coordinates for that feature family. Columns include `pc1`-`pc3` and `length_resid_pc1`-`length_resid_pc3`, plus prompt metadata (`prompt_id`, `type`, `source`, `tokens_approx`). <br><br> **Note on Attention Entropy Residualization:** If the `signal_lab.sequence_heads` downstream pipeline has been run, this file will also contain `attn_resid_pc1`-`attn_resid_pc3`. These are double-residualized coordinates that regress out both token length *and* the `global_mean_attn_entropy` (derived directly from `verbose.jsonl` internal attention physics, rather than logit distributions). This validates structural class isolation independent of general sequence ambiguity. This is the main table for downstream plotting, joins, and cross-model prompt-level comparisons. |

## Per-Family Figure Directories


For each family `{family}`, the analyzer writes a directory:

- `{family}_figures/`

That directory contains the following files:

| File | Purpose |
|------|---------|
| `pc1_pc2_raw_vs_length_resid.png` | Two-panel comparison of raw PC1 vs PC2 and length-residualized PC1 vs PC2, with points colored by prompt type. Use this first to see how much of the visible geometry is dominated by prompt length. |
| `pc1_pc2_raw_task.png` | Raw PC1 vs PC2 scatter colored by prompt type. This is the main quick-look view of the first two unadjusted axes. |
| `pc2_pc3_raw_task.png` | Raw PC2 vs PC3 scatter colored by prompt type. Useful when PC1 is mostly a prompt-length axis and the more interesting task structure shifts into later components. |
| `pc1_pc2_length_resid_task.png` | Length-residualized PC1 vs PC2 scatter colored by prompt type. Use this to inspect task structure after linear removal of prompt-length effects. |
| `pc2_pc3_length_resid_task.png` | Length-residualized PC2 vs PC3 scatter colored by prompt type. This is often the follow-up view when the residualized first two axes still do not fully expose class structure. |
| `raw_scree.png` | Scree plot for the raw PCA, showing explained variance for the first components. Use it to judge whether the family is dominated by one or two axes or has more distributed structure. |
| `length_resid_scree.png` | Scree plot for the PCA after residualizing on `tokens_approx`. Use it to see how much variance remains concentrated once prompt length is regressed out. |

## Downstream Intervention and Isolation Tools

The standard `sequence_analyze` pipeline provides baseline coordinates, but targeted hypothesis testing and router panel validation relies on these supplemental pipelines chained *after* the initial run.

### `signal_lab.sequence_heads`
Standalone module designed to perform deep entropy-based double-residualization.
- Automatically consumes `verbose.jsonl` internal run logs to calculate `global_mean_attn_entropy` for each prompt.
- Regresses both token sequence length and mean attention heat out of the unadjusted raw arrays, bounding coordinates as `attn_resid_pc1`-`attn_resid_pc3`.
- Validates the null-hypothesis that task clustering (like Code Comprehension) isn't simply an artifact of prompt-length or model ambiguity.

### `signal_lab.sequence_plot_3d`
Renders 3D Plotly visual equivalents of the initial unadjusted matrices to `<analysis-dir>/3d_plots`. Generates snapshot plots corresponding directly to the underlying manifold adjustments:
- `..._3d_raw.png`
- `..._3d_length_resid.png`
- `..._3d_attn_resid.png`

### `signal_lab.sequence_plot_winners` & `sequence_plot_panel`
Tools mapping specific ML routing interventions *onto* the dimensional representations to prove structural alignment:
- **`sequence_plot_winners`:** Extracts the top-4 raw oracle interventions from `sweep_oracle_winners_v1.csv` and maps them spatially onto the manifolds as `..._top4other_v2.png`.
- **`sequence_plot_panel`:** Used to hard-test explicit custom panels (like `router-9B-040`). Matches the prompt-level delta target probabilities and plots the optimal profile win geometries natively as `..._top4plusoff.png`.

### `signal_lab.optimize_router_panel`
Hardware-accelerated search engine treating the sequence representation manifold as a multi-objective search space.
- Exhaustively evaluates all `K=3` and `K=4` combinations of reliable intervention arrays against the PCA metric embeddings.
- Simultaneously scores **Battery Performance** (how many prompts receive delta coverage) vs **Geometry Isolation** (Silhouette separation scoring).
- Returns the theoretical Pareto optimum geometric topology (e.g., `['constant_2.6', 'constant_1.45', 'plateau_bal_0.55', 'bowl_bal_0.40']`) maximizing downstream routing effectiveness.
