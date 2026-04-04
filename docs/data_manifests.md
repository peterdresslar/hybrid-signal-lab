# Signal Lab Data Manifests

Reference for the file structure of a single model run and a single pairwise comparison. Each item includes what it is and what it is useful for.

Example run: `reanalyze/2B/`. Example comparison: `ks/_comparisons/2B_vs_35B`. Older runs (e.g., `ks/35B/`) follow the same structure but may contain plots that have since been pruned from the default output.

---

## Manifest A: Single Model Run

### Root

| File | Description | Useful for |
|------|-------------|------------|
| `_meta.json` | Run configuration: model ID, architecture config (layers, heads, hidden size), cartridge name, attention layer indices, all g-profile specs with their gain vectors, and prompt battery reference. | Reproducing the run. Confirming which gain vectors were actually applied. Cross-referencing layer indices when interpreting scout heads or PCA results. |
| `main.jsonl` | Raw per-prompt inference results (one JSON object per prompt × profile execution). Primary data artifact from Signal Lab. | Source of truth for all downstream analysis. Any custom analysis or re-aggregation starts here. |
| `verbose.jsonl` | Extended inference log with full token-level probability distributions and attention head entropy values. | Per-head entropy analysis, scout head identification, and any analysis that needs the full token-level distribution rather than just the target token. |
| `errors.jsonl` | Any errors encountered during the sweep (empty if run completed cleanly). | Diagnosing incomplete runs. Should be empty for production data. |

### Analysis CSVs (`analysis/`)

| File | Description | Useful for |
|------|-------------|------------|
| `analysis_report.txt` | Human-readable run summary. Top-8 profile rankings by four criteria (positive cluster strength, sharpness, mean delta_p, balanced score), best/worst type × profile combinations, gain family summary, per-type best profile, top prompt winners/losers. | First thing to read for any run. Orients you to which profiles matter and which types respond. Provides the narrative entry point before drilling into CSVs. |
| `analysis_joined_long.csv` | Core long-format table. One row per prompt × profile. 49 columns including prompt metadata, intervention spec, raw output metrics, baseline metrics, and delta metrics. | The primary analysis surface for custom queries, filtering, and visualization. Any question that crosses prompt-level and profile-level dimensions starts here. |
| `analysis_completion.csv` | Sweep completeness check: unique prompts, reps, expected vs. completed runs, missing runs, error count. | Verifying data integrity before analysis. Catching silent failures or incomplete runs. |
| `analysis_files.csv` | Index of all file paths in the run directory. | Provenance tracking. Confirming which files were generated from which run. |
| `analysis_overall_profile_summary.csv` | Profile-level aggregates. One row per g_profile with mean/median/stdev of delta_target_prob, positive mass, top-k cluster means (top-1 through top-16), best/worst type, balanced score, wins/losses. | Ranking profiles globally. Identifying candidates for top-1/top-2/top-4 profile sets. Comparing a profile's peak performance (top-k clusters) against its average. |
| `analysis_best_profile_by_type.csv` | Best-performing g_profile for each prompt type, with mean delta_p, % positive, p-value. | Quick lookup for type-specific winners. Answering "what profile should I use for code_comprehension?" without scanning the full type × profile table. |
| `analysis_prompt_winners.csv` | Per-prompt best and worst g_profile with their delta_target_prob and delta_rank. | Identifying the full headroom range for each prompt. Finding prompts where intervention makes the biggest difference (positive or negative). Spotting outliers. |
| `analysis_type_gain_summary.csv` | Type × profile detail. One row per (type, g_profile) with mean delta_p, % positive, sign-test p-value, rank changes, entropy changes. | The main table for task-dependent intervention effects. Answering "how does profile X perform on task type Y?" with statistical support. |
| `analysis_type_gain_matrix_delta_p.csv` | Pivot of the above: rows = types, columns = g_profiles, values = mean delta_target_prob. | Generating heatmaps. Visually scanning for type × profile interaction patterns. Compact format for cross-run comparison. |
| `analysis_type_family_summary.csv` | Same as type_gain_summary but aggregated by gain family (e.g., "early_boost" combines 1.3 and 1.5 variants). | Reducing dimensionality for pattern identification. Answering "do early_boost profiles as a class help reasoning tasks?" without per-variant noise. |
| `analysis_type_family_matrix_delta_p.csv` | Pivot of type × family. | Same as the type × profile matrix but at family level. Useful when the profile-level matrix is too wide to scan. |
| `analysis_tier_gain_summary.csv` | Aggregates by prompt length tier × g_profile. | Testing whether intervention sensitivity varies with prompt length/complexity. Answering "do longer prompts respond differently to gain?" |
| `analysis_type_tier_gain_summary.csv` | Three-way breakdown: type × tier × g_profile. | Finest-grained pre-computed summary. Useful when a type-level effect might be driven by one prompt-length tier. |
| `analysis_scout_head_rankings.csv` | Per-profile ranking of individual attention heads by their correlation with intervention benefit. Columns include layer_slot, layer_index, head index, correlation magnitude, AUROC, balanced accuracy, entropy threshold. | Identifying which attention heads predict intervention response. Input to Colony design (adaptive per-prompt routing). Comparing scout head stability across profiles and models. |
| `analysis_warnings.csv` | Data quality warnings (e.g., missing baselines, anomalous values). Empty if clean. | Flagging issues before they contaminate analysis. Should be checked once per run. |

### Analysis JSON (`analysis/`)

| File | Description | Useful for |
|------|-------------|------------|
| `analysis_baseline_attn_pca.json` | PCA decomposition of baseline attention entropy vectors across prompts. | Source data for the baseline PCA plot and for PCA intervention plots. Captures the latent structure of how attention heads respond to different prompt types before intervention. Needed for on-demand regeneration of any PCA intervention plot. |
| `analysis_head_correlations.json` | Full correlation matrix of per-head entropy vs. delta_target_prob for each profile. | Source data for head correlation heatmaps and scout head analysis. Can be queried directly for any profile without regenerating plots. |

### Plots (`analysis/plots/`)

| File | Description | Useful for |
|------|-------------|------------|
| `analysis_plot_manifest.json` | Index of all generated plots with metadata (model, x-metric, file paths). | Provenance. Programmatic discovery of which plots exist for a given run. |
| `analysis_scatter_delta_target_prob_by_type__x-tokens_approx.png` | Delta_target_prob vs. approximate token count, all profiles overlaid, faceted by prompt type. | Checking whether prompt length interacts with intervention effect at the run level. The type faceting keeps it readable despite the profile overlay. |
| `analysis_scatter_baseline_target_prob_vs_delta_target_prob_by_type.png` | Baseline target probability vs. delta_target_prob, faceted by type. | Identifying headroom structure: whether low-baseline ("harder") prompts benefit more from intervention, broken out by task type. One of the most directly interpretable top-level plots. |
| `analysis_baseline_attn_pca.png` | PCA visualization of baseline attention entropy landscape, colored by prompt type. | Seeing how prompt types cluster in attention-head space before any intervention. Reveals whether task types occupy distinct regions of the model's attention geometry, which is the foundation for understanding why intervention effects are type-dependent. Not duplicated by any other plot. |
| `analysis_scout_heads.png` | Heatmap of top scout heads across profiles. | Cross-profile summary of which attention heads are most informative about intervention benefit. Useful for comparing scout head stability across runs and models. |
| `analysis_head_corr__top_positive_cluster__{profile}.png` | Detailed head correlation map for the single strongest positive-cluster profile (filename varies by run). | Exemplar of the head × layer correlation structure for the run's best profile. Shows which specific heads drive the strongest intervention responses. |

### Plot Subdirectories

#### `plots/baseline/`

Baseline characterization. Three scatter plots pairing `baseline_target_prob` on the y-axis against non-redundant x-metrics, all faceted by prompt type.

| File | Useful for |
|------|------------|
| `scatter_baseline_target_prob_by_type__x-baseline_final_entropy_bits.png` | Seeing how each task type distributes across the model's output entropy landscape before intervention. Shows which types the model is confident about vs. uncertain about. Canonical output-entropy view. |
| `scatter_baseline_target_prob_by_type__x-baseline_attn_entropy_mean.png` | Same question but for attention-layer entropy specifically. Distinct clustering structure from output entropy — some types that look similar in output entropy separate in attention entropy, which matters because gain intervention acts on the attention sublayer. |
| `scatter_baseline_target_prob_by_type__x-baseline_top1_top2_logit_margin.png` | Confidence margin view: how close the model's top-1 and top-2 predictions are. Complements the probability view by showing decision-boundary proximity, which can predict whether a small intervention push will flip the model's answer. |
| `summary.json` | Metadata for the baseline plot set. |

#### `plots/best_interventions/`

Index of the top 12 profiles by mean delta_target_prob.

| File | Useful for |
|------|------------|
| `best_interventions.csv` | Quick lookup of which profiles to focus on. Defines the set used for restricted plot generation (head correlations, PCA interventions). |
| `manifest.json` | Same ranking with folder paths. Programmatic routing to per-profile plot directories. |

#### `plots/interventions/{profile}/`

One subdirectory per g_profile (78 total). Each contains 4 scatter plots plus metadata.

| File | Useful for |
|------|------------|
| `scatter_delta_target_prob_by_type__x-baseline_target_prob.png` | The single most interpretable per-profile plot. Shows which prompts a profile helps vs. hurts, broken out by task type, against baseline performance. Reveals headroom structure: typically low-baseline prompts benefit while high-baseline prompts are inert or hurt. |
| `scatter_delta_target_prob_by_type__x-baseline_attn_entropy_mean.png` | Shows whether intervention response tracks the model's attention-layer state rather than just output confidence. Distinct scatter shapes from the target_prob view — important because gain acts on the attention sublayer, so attention entropy may be a better predictor of intervention response than output probability. |
| `scatter_delta_target_prob_by_type__x-tokens_approx.png` | Prompt-length interaction for this profile. Answers whether the profile's effect depends on how long the prompt is. Lower priority than the other two by_type views but useful for the prompt-length secondary question. |
| `scatter_delta_target_prob_all__x-baseline_target_prob.png` | Compact single-panel headroom summary. All prompts on one plot, type-colored. The probability axis separates type clusters more clearly than entropy, and the headroom envelope (low-baseline prompts lift, high-baseline prompts inert or hurt) is immediately visible. Useful as a quick-glance triage before examining the by_type version. |
| `summary.json` | Per-profile metadata (model, profile name, prompt count, file paths). |

#### `plots/head_correlations/`

Per-head correlation heatmaps, restricted to the top 12 profiles from `best_interventions/`.

| File | Useful for |
|------|------------|
| `analysis_head_corr__{profile}.png` | Shows which specific attention heads correlate with intervention benefit for a given top profile. Layer × head grid with correlation sign and magnitude. Useful for identifying candidate scout heads and comparing head-level mechanism across profiles. Restricted to top profiles because incremental value drops rapidly and the full correlation data is available in JSON for any profile. |
| `manifest.json` | Index of which profiles have heatmaps generated. |

#### `plots/pca_interventions/`

PCA intervention scatters, restricted to the top 12 profiles from `best_interventions/`.

| File | Useful for |
|------|------------|
| `analysis_pca_delta__{profile}.png` | Shows prompts in baseline-attention PCA space, colored by delta_target_prob. For top-ranked profiles, can reveal whether intervention response follows attention-space geometry rather than just task labels — i.e., whether nearby prompts in attention space respond similarly to intervention regardless of their type label. Hard to read for weak profiles due to the continuous color scale over a dense point cloud. Restricted to top profiles; PCA data retained in JSON for on-demand regeneration. |
| `manifest.json` | Index of which profiles have PCA plots generated. |

### X-Metric Redundancy Notes

The original 7 x-metrics used in scatter plots cluster into groups where members produce near-identical scatter shapes. This redundancy was verified by side-by-side visual comparison across multiple profiles (`edges_narrow`, `shifted_bump`) on `b4_021_attn_contr/9B`.

- **Output confidence** (near-identical): `baseline_target_prob` ≈ `baseline_target_geo_mean_prob` ≈ `baseline_top1_top2_logit_margin`. The `baseline_target_prob` version is the most directly interpretable; `top1_top2_logit_margin` is retained in baseline plots as a complementary confidence axis.
- **Output entropy** (near-identical): `baseline_final_entropy_bits` ≈ `baseline_mean_entropy_bits`. The `final_entropy_bits` version is retained as the canonical output-entropy view.
- **Attention entropy** (distinct): `baseline_attn_entropy_mean`. Shows attention-specific structure not visible in output entropy. Retained in both baseline and per-profile intervention plots.
- **Prompt length** (distinct, lower value): `tokens_approx`. Retained in per-profile intervention plots for the prompt-length secondary question.

The `_all` (across-all-prompts) view for `x-baseline_target_prob` was selected over `x-baseline_mean_entropy_bits` based on side-by-side comparison: the probability axis separates type clusters more clearly and makes the headroom envelope immediately visible.

---

## Manifest B: Single Pairwise Comparison

Files shown for `ks/_comparisons/2B_vs_35B/`. All comparisons follow this structure.
Model A = 2B, Model B = 35B (labels derived from run directory names).

### CSVs

| File | Description | Useful for |
|------|-------------|------------|
| `compare_report.txt` | Human-readable comparison summary. Largest type × profile gaps, type × family gaps, and prompt-level divergences between the two models. | Quick-reference narrative for cross-model interpretation. Start here before drilling into pairwise CSVs. |
| `compare_files.csv` | Index: which two runs are being compared (slot, label, model ID, run directory paths). | Provenance. Confirming which exact runs were compared. |
| `compare_type_gain_pairwise.csv` | Type × profile comparison. One row per (type, g_profile) with side-by-side metrics for both models plus diff columns. | The primary cross-model analysis surface. Answering "does profile X help model A more than model B on task type Y?" with full statistical detail. |
| `compare_type_family_pairwise.csv` | Same structure, aggregated to gain family level. | Reducing noise for identifying systematic cross-model patterns at the family level. |
| `compare_prompt_pairwise.csv` | Prompt-level comparison. One row per (prompt, g_profile) with side-by-side metrics plus diffs. Largest file. | Identifying prompt-specific divergences between models. Finding prompts where the two models respond to intervention in opposite directions. |
| `compare_warnings.csv` | Data quality warnings for the comparison. | Catching mismatched prompt sets or other comparison-level issues. |

### JSON

| File | Description | Useful for |
|------|-------------|------------|
| `compare_cross_model_scouts.json` | Scout head alignment analysis. Each model's top-k scout heads and cross-model prediction accuracy. | Testing whether one model's scout heads predict the other model's intervention responses. Key for the question of whether intervention mechanisms transfer across architectures. |

### Plots (`plots/`)

| File | Description | Useful for |
|------|-------------|------------|
| `compare_plot_manifest.json` | Index of comparison plots with metadata. | Provenance and programmatic discovery. |
| `compare_cross_model_prediction_summary.png` | Bar chart: can model A's scout heads predict model B's intervention response (and vice versa)? | Visual summary of cross-model scout transferability. Shows whether the attention-level mechanism is architecture-specific or shared. |
| `compare_scout_entropy_alignment.png` | Scatter comparing scout head entropy distributions between models. | Testing whether the two architectures use similar attention patterns even if different specific heads carry the signal. |
| `compare_scout_entropy_vs_delta.png` | Scout head entropy vs. delta_target_prob for both models overlaid. | Checking whether the entropy→benefit relationship is conserved across model scale or architecture. |
