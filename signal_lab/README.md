## Signal Lab

`signal_lab` is the current probing and analysis package for this repository.
It runs hybrid LLMs and full-softmax control models under baseline and
gain-vector interventions, records prompt-level response metrics, and
generates summaries and plots for both single-model and cross-model
comparisons.

The broader `colony` concept is future work. For the current iteration, the
main implemented loop is:

1. build or select a prompt battery
2. run `signal_lab` or `sweep` over one or more models
3. analyze the resulting runs
4. compare runs across models
5. generate simple plots to inspect directionality

Shared model backends and prompt structures live in the top-level `model`
package.

## Intervention Strategies

Gain intervention applies a multiplicative scalar *g* to modulate the
attention sublayer at each softmax attention layer during inference. The
central design question is *where* in the decoder block to apply that scalar.
Signal Lab supports two explicit intervention strategies, plus a backward-compatible default:

**`block_output`** scales the entire decoder block output — attention, feed-forward,
and residual stream together: `g · (h + a + f)`. This was the original
implementation. It is a blunt instrument: because the gain multiplies the
residual and feed-forward pathways alongside the attention output, it conflates
the attention contribution with other information flows. Empirically, effects
under block_output mode are modest — the best overall mean delta_p for 9B is
+0.01, and the productive gain range collapses above ~1.5.

**`attention_contribution`** scales only the attention sublayer output at the
residual add point: `h + g · a + f`. This isolates the causal pathway the
intervention is intended to modulate. The hook target depends on the model's
normalization topology: for pre-norm architectures (Qwen 3.5), gain is applied
at `self_attn`; for post-norm architectures (OLMo Hybrid), gain must be applied
after the post-attention RMSNorm (`post_attention_layernorm`), because RMSNorm
is scale-invariant and would erase the gain if applied before it.

The distinction between these modes is not just operational — it is the primary
axis of experimental comparison in the current data. Under attention_contribution
mode, 9B shows 5–6× higher overall mean delta_p, a productive gain range
extending to ~2.75, and dramatically stronger type-level effects (e.g.,
code_comprehension +0.45, reasoning_numerical +0.37 under the best profile).
The mode × architecture interaction is also significant: OLMO's post-norm
topology constrains the attention contribution's magnitude relative to the
residual, so attention_contribution mode has less leverage on OLMO than on the
pre-norm Qwen models.

**`backend_default`** preserves each backend's legacy behavior for reproducibility.
For hybrid backends this resolves to `block_output`; for raw HF transformer
control models it resolves to `attention_contribution`.

See `docs/data_manifests.md` for the detailed file-level reference to
intervention data, and `data/intervention_modes/DATA_GUIDE.md` for the full
experimental context comparing the two modes.

Generated Signal Lab artifacts now default under `[DATA_DIR]/outputs/signal_lab/`.
You can point `[DATA_DIR]` somewhere else with `--data-dir` or the `DATA_DIR`
environment variable.
The main subtrees are:

- `[DATA_DIR]/outputs/signal_lab/probes/`
- `[DATA_DIR]/outputs/signal_lab/runs/<run_name>/<model_name_short>/`
- `[DATA_DIR]/outputs/signal_lab/runs/<run_name>/<model_name_short>/analysis/`
- `[DATA_DIR]/outputs/signal_lab/runs/<run_name>/_comparisons/<compare_name>/`

## Package Contents

- `signal_lab.signal_lab`: one-off probing CLI for a single prompt
- `signal_lab.sweep`: batch runner over cartridges or battery selections
- `signal_lab.sweep_cartridges`: named gain-profile sweep specs
- `signal_lab.sweep_analyze`: summarize one sweep run directory
- `signal_lab.run_analyze`: analyze every model subfolder under a run collection (see below)
- `signal_lab.sweep_compare`: compare two analyzed run directories
- `signal_lab.sweep_plot_analyze`: plot one analyzed run
- `signal_lab.sweep_plot_compare`: plot one pairwise comparison bundle
- `signal_lab.agent`: inference pipeline used by the CLIs

## Setup

Requires Python `3.12.x` (project pin: `3.12.8`) and is intended to be used with `uv`.

```bash
uv sync
```

For Hugging Face model access, create `.env.development` in the repo root:

```bash
HF_TOKEN=hf_your_token_here
```

`HUGGINGFACE_HUB_TOKEN` is also supported.

Device selection defaults to auto-detection in this order:

1. `cuda`
2. `mps`
3. `cpu`

You can override that globally:

```bash
export COLONY_DEVICE=cuda
```

Or per command:

```bash
uv run -m signal_lab.signal_lab --device cuda ...
```

## Single Prompt Probing

**Important:** The nature of the intervention is critical to the experimentation being performed in `signal_lab`. It generally will make sense to understand the differences and to explicitly set an intervention mode before running even the simplest of passes.

Use `signal_lab.signal_lab` when you want one prompt, one model, and one gain
specification.

Gain is now applied in one of three intervention strategies:

- `backend_default`: preserve each backend's legacy behavior
- `block_output`: scale the full decoder-layer output
- `attention_contribution`: scale only the attention branch contribution at the
  point where it enters the residual stream

For the current hybrid backends, `backend_default` resolves to `block_output`
for `Qwen3.5` hybrid and `OLMo-Hybrid` so older runs remain reproducible. For
raw Hugging Face transformer control models, `backend_default` resolves to
`attention_contribution`.

### Simplest Example

```bash
uv run -m signal_lab.signal_lab \
  --prompt "The color with the shortest wavelength is" \
  --model-key 0_8B \
  --g-function constant \
  --g 1.0 \
  --intervention-strategy backend_default
```

This writes a full JSON summary under `[DATA_DIR]/outputs/signal_lab/probes/`.

### Prompt From A Battery

```bash
uv run -m signal_lab.signal_lab \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-id alg_gen_arithmetic_0000 \
  --model-key 2B \
  --g-function constant \
  --g 1.25 \
  --intervention-strategy attention_contribution
```

### Prompt Filtered By Type/Tier

This selection must resolve to exactly one prompt.

```bash
uv run -m signal_lab.signal_lab \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-type algorithmic \
  --prompt-tier short \
  --prompt-id alg_gen_arithmetic_0000 \
  --model-key 9B \
  --g-function control_points \
  --g-vector 1.5,1.5,1.5,0.5,0.5,0.5
```

### Useful Flags

- `--prompt`: literal string, text file, or prompt id/candidate selector
- `--prompt-battery`: battery directory or JSON collection
- `--prompt-id`, `--prompt-ids`, `--prompt-tier`, `--prompt-type`: battery-based selection helpers
- `--model-key`: one of the registered model keys such as `0_8B`, `2B`, `9B`, `35B`, `OLMO`
- `--g-function`: `constant`, `linear`, `gaussian`, `step`, or `control_points`
- `--g`: shortcut for constant profiles
- `--g-vector`: comma-separated control points for `control_points`
- `--g-params-json`: extra parameters for non-constant functions
- `--intervention-strategy` / `--gain-mode`: `backend_default`, `block_output`, or `attention_contribution`
- `--target-attention-layers`: `native_attention_layers`, `all_layers`, or `every_4th_layer`
- `--mimic-hybrid`: convenience alias for `--target-attention-layers every_4th_layer`
- `--device`: `auto`, `cuda`, `mps`, or `cpu`
- `--output-path`: optional override for the summary JSON location
- `--data-dir`: optional override for `[DATA_DIR]`

## Running Sweeps

Use `signal_lab.sweep` when you want many prompts and/or many gain profiles.

Cartridges now also encode how gain profiles are targeted onto attention layers:

- hybrid-native cartridges such as `kitchen_sink` use each backend's native
  attention-layer selection
- control-model cartridges ending in `_all_layers` target every attention layer
- control-model cartridges ending in `_hybrid_mimic` target every 4th layer to
  mimic the hybrid cadence on transformer-only models

Independently of cartridge attention targeting, sweeps also accept
`--intervention-strategy` to choose *where* the selected gain is applied inside each
decoder block:

- `backend_default`: preserve each backend's legacy semantics
- `block_output`: scale the full decoder-layer output
- `attention_contribution`: scale only the attention residual contribution

For the hybrid backends this means:

- `Qwen3.5` hybrid: `attention_contribution` hooks `self_attn`
- `OLMo-Hybrid`: `attention_contribution` hooks `post_attention_layernorm`

For full-softmax control models loaded via raw HF ids, `TransformerBackend`
uses architecture-aware defaults:

- `Qwen3`: attention contribution is taken at `self_attn`
- `OLMo3`: attention contribution is taken at `post_attention_layernorm`

### Cartridge-Based Sweep

```bash
uv run -m signal_lab.sweep \
  --cartridge uniform_check_lite \
  --run-name uniform_check_lite_demo \
  --model-key 0_8B \
  --intervention-strategy backend_default
```

### Battery-Backed Sweep

```bash
uv run -m signal_lab.sweep \
  --cartridge kitchen_sink \
  --run-name battery3 \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-types algorithmic \
  --prompt-tiers short \
  --model-key 35B \
  --device cuda
```

### Transformer Control Sweep

```bash
uv run -m signal_lab.sweep \
  --cartridge kitchen_sink_all_layers \
  --run-name qwen25_control \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-types algorithmic \
  --prompt-tiers short \
  --model-key Qwen/Qwen2.5-0.5B \
  --device cuda \
  --intervention-strategy attention_contribution
```

For a sparse control run that mimics the hybrid every-4th-layer cadence, swap
the cartridge to `kitchen_sink_hybrid_mimic` (or the corresponding
`fine_grain_kitchen_sink_hybrid_mimic` variant).

You can also override targeting explicitly from the CLI:

```bash
uv run -m signal_lab.sweep \
  --cartridge kitchen_sink_all_layers \
  --run-name qwen25_control_mimic \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-types algorithmic \
  --prompt-tiers short \
  --model-key Qwen/Qwen2.5-0.5B \
  --device cuda \
  --intervention-strategy attention_contribution \
  --mimic-hybrid
```

`--target-attention-layers` defaults to the cartridge's configured targeting. For
transformer-only controls, that generally means:

- `all_layers`: target every attention block
- `every_4th_layer`: mimic the hybrid cadence
- `native_attention_layers`: use the backend default attention-bearing blocks

### Specific Prompt IDs From A Battery

```bash
uv run -m signal_lab.sweep \
  --cartridge kitchen_sink \
  --run-name kitchen_sink_demo \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-ids alg_gen_arithmetic_0000,alg_gen_arithmetic_0001 \
  --model-key OLMO \
  --device cuda
```

### Verbose Sweep

```bash
uv run -m signal_lab.sweep \
  --cartridge kitchen_sink \
  --run-name long_range_debug \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-types long_range_retrieval \
  --model-key 9B \
  --verbose
```

Verbose mode adds `verbose.jsonl` with heavier per-run detail.

Useful cartridge families for control-model studies:

- `kitchen_sink_all_layers`
- `kitchen_sink_hybrid_mimic`
- `fine_grain_kitchen_sink_all_layers`
- `fine_grain_kitchen_sink_hybrid_mimic`
- `balanced_kitchen_sink_all_layers`
- `balanced_kitchen_sink_hybrid_mimic`

Example: run the balanced comprehensive sweep on a transformer-only control
model while targeting every attention layer:

```bash
uv run -m signal_lab.sweep \
  --cartridge balanced_kitchen_sink_all_layers \
  --model-key Q3_8B \
  --intervention-strategy attention_contribution \
  --run-name q3_8b_balanced_all_layers
```

Example: run the same balanced sweep but sparsify targeting to every 4th layer
to mimic the hybrid cadence:

```bash
uv run -m signal_lab.sweep \
  --cartridge balanced_kitchen_sink_hybrid_mimic \
  --model-key Q3_8B \
  --intervention-strategy attention_contribution \
  --run-name q3_8b_balanced_hybrid_mimic
```

### Sweep Outputs

When you use `--run-name`, the default model run directory is:

- `[DATA_DIR]/outputs/signal_lab/runs/<run_name>/<model_name_short>/`

The sweep command rejects an already-populated output directory so that a
checkpoint-like `run_name` does not silently overwrite prior results.

Each model run directory contains:

- `main.jsonl`: core per-run metrics
- `_meta.json`: model/config/prompt-selection metadata
- `errors.jsonl`: failures if any occur
- `verbose.jsonl`: optional detailed output when `--verbose` is enabled

For control sweeps, `_meta.json` also records:

- `attention_targeting`
- `intervention_strategy`
- `gain_intervention_mode`
- `available_attention_layer_indices`
- `target_attention_layer_indices`

## Analyzing A Sweep Run

Sweeps that used `--run-name` lay out each model under a **collection** directory:

- `[DATA_DIR]/outputs/signal_lab/runs/<collection_name>/<model_slug>/`

Example: `runs/fine/35B_20260322_123428/` holds `main.jsonl`, `_meta.json`, and by default analysis lands in `.../35B_.../analysis/`.

### One model (single run directory)

Use `signal_lab.sweep_analyze` on that folder:

```bash
uv run -m signal_lab.sweep_analyze \
  --run-dir [DATA_DIR]/outputs/signal_lab/runs/fine/35B_20260322_123428
```

### Every model in a collection (batch)

Use `signal_lab.run_analyze` with the **parent** of the model folders (the directory that contains `2B_...`, `35B_...`, `OLMO_...`, etc.):

```bash
uv run -m signal_lab.run_analyze \
  --input-dir [DATA_DIR]/outputs/signal_lab/runs/fine
```

By default (when not using `--no-write-files`), `run_analyze` also runs
`sweep_plot_analyze` on each model’s analysis directory. If two or more models
are present, it then runs **all pairwise** `sweep_compare` and
`sweep_plot_compare` steps (for K models that is K*(K-1)/2 pairs). Use
`--no-compare` to skip only the pairwise comparison and comparison-plot steps;
per-model plots still run.

Optional: put all analyses under a separate tree (mirrors each model folder name):

```bash
uv run -m signal_lab.run_analyze \
  --input-dir [DATA_DIR]/outputs/signal_lab/runs/fine \
  --output-parent [DATA_DIR]/outputs/signal_lab/analysis_exports/fine
```

That writes `.../analysis_exports/fine/<model_slug>/analysis/`, and comparisons
to `.../analysis_exports/fine/_comparisons/<model-a>_vs_<model-b>/` (order
matches `--run-a` / `--run-b` in `sweep_compare`).

Machine-readable JSON per model when batching:

```bash
uv run -m signal_lab.run_analyze \
  --input-dir [DATA_DIR]/outputs/signal_lab/runs/fine \
  --json-out-dir [DATA_DIR]/outputs/signal_lab/analysis_exports/fine_json
```

Useful flags: `--data-dir`, `--prefix`, `--no-write-files`, `--no-compare`,
`--compare-prefix`, `--x-metric` / `--x-metrics`, `--no-intervention-folders`
(intervention folders are enabled by default), `--best-interventions-top-n`,
`--disagreement-top-n`, `--plot-dpi`, and `--dry-run`. For a **single**
discovered run, `--json-out` works as in
`sweep_analyze`.

### Example (single run, classic path)

```bash
uv run -m signal_lab.sweep_analyze \
  --run-dir [DATA_DIR]/outputs/signal_lab/runs/battery3/35B
```

### Example With Separate Output Directory

```bash
uv run -m signal_lab.sweep_analyze \
  --run-dir /path/to/raw_run \
  --output-dir /path/to/analysis_run
```

Without `--output-dir`, analysis files are written to `<run-dir>/analysis/`.

### Example Without Writing Files

```bash
uv run -m signal_lab.sweep_analyze \
  --run-dir [DATA_DIR]/outputs/signal_lab/runs/battery3/35B \
  --no-write-files
```

### Main Analysis Artifacts

CSVs and text:

- `analysis_report.txt` — human-readable summary; start here
- `analysis_joined_long.csv` — core long-format table (one row per prompt × profile)
- `analysis_overall_profile_summary.csv` — profile-level aggregates
- `analysis_type_gain_summary.csv` — type × profile detail
- `analysis_type_gain_matrix_delta_p.csv` — type × profile pivot (heatmap-ready)
- `analysis_type_family_summary.csv` — type × family aggregates
- `analysis_type_family_matrix_delta_p.csv` — type × family pivot
- `analysis_best_profile_by_type.csv` — best profile per task type
- `analysis_prompt_winners.csv` — per-prompt best/worst profiles
- `analysis_tier_gain_summary.csv` — tier × profile aggregates
- `analysis_type_tier_gain_summary.csv` — type × tier × profile
- `analysis_scout_head_rankings.csv` — attention heads ranked by intervention-benefit correlation
- `analysis_completion.csv`, `analysis_warnings.csv`, `analysis_files.csv`

JSON:

- `analysis_baseline_attn_pca.json` — PCA of baseline attention entropy vectors
- `analysis_head_correlations.json` — full per-head correlation data

For a detailed description of every file and plot (including what each is useful
for), see `docs/data_manifests.md`.

## Comparing Two Analyzed Runs

Use `signal_lab.sweep_compare` after each run has already been analyzed.

### Minimal Example

```bash
uv run -m signal_lab.sweep_compare \
  --run-a [DATA_DIR]/outputs/signal_lab/runs/checkpoint_name/35B/analysis \
  --run-b [DATA_DIR]/outputs/signal_lab/runs/checkpoint_name/OLMO/analysis
```

### Labeled Cross-Model Example

```bash
uv run -m signal_lab.sweep_compare \
  --run-a [DATA_DIR]/outputs/signal_lab/runs/checkpoint_name/35B/analysis \
  --run-b [DATA_DIR]/outputs/signal_lab/runs/checkpoint_name/OLMO/analysis \
  --label-a qwen35b \
  --label-b olmo \
  --output-dir [DATA_DIR]/outputs/signal_lab/runs/checkpoint_name/_comparisons/qwen35b_vs_olmo \
  --prefix qwen35b_vs_olmo
```

### Main Comparison Artifacts

- `*_report.txt`
- `*_prompt_pairwise.csv`
- `*_type_gain_pairwise.csv`
- `*_type_family_pairwise.csv`
- `*_files.csv`
- `*_warnings.csv`

## Plotting A Single Analyzed Run

Use `signal_lab.sweep_plot_analyze` for single-model scatter plots.

### Basic Example

```bash
uv run -m signal_lab.sweep_plot_analyze \
  --analysis-dir [DATA_DIR]/outputs/signal_lab/runs/battery3/35B/analysis \
  --x-metric tokens_approx
```

### Baseline Difficulty View

```bash
uv run -m signal_lab.sweep_plot_analyze \
  --analysis-dir [DATA_DIR]/outputs/signal_lab/runs/battery3/35B/analysis \
  --x-metric baseline_target_prob
```

### Intervention Folder View

```bash
uv run -m signal_lab.sweep_plot_analyze \
  --analysis-dir [DATA_DIR]/outputs/signal_lab/runs/battery3/35B/analysis \
  --x-metric tokens_approx \
  --intervention-folders \
  --best-interventions-top-n 12
```

If `analysis_joined_long.csv` contains the enriched baseline columns from
`signal_lab.sweep_analyze`, intervention folders will also emit additional
scatter variants for richer prompt discriminators such as baseline entropy,
baseline target geo-mean probability, baseline top-1 vs top-2 logit margin,
and mean attention entropy.

The default x-metrics for intervention folders have been slimmed to reduce
redundancy. The 7 available x-metrics cluster into groups that produce
near-identical scatter shapes (see `docs/data_manifests.md` for details). The
recommended defaults are:

- **by_type views:** `baseline_target_prob` (headroom), `baseline_attn_entropy_mean` (attention-specific), optionally `tokens_approx` (prompt length)
- **single _all view:** `baseline_target_prob` (compact headroom summary)

Head correlation heatmaps and PCA intervention plots are restricted by default
to the top-N profiles from `best_interventions/`. The underlying JSON data for
both is always generated, so any profile can be plotted on demand.

### Multi-Metric Batch Example

```bash
uv run -m signal_lab.sweep_plot_analyze \
  --analysis-dir [DATA_DIR]/outputs/signal_lab/runs/battery3/35B/analysis \
  --x-metrics tokens_approx baseline_target_prob baseline_final_entropy_bits \
  --intervention-folders
```

This produces:

- `plots/baseline/` — baseline characterization scatters (by_type only)
- `plots/interventions/<g_profile>/` — per-profile intervention scatters (slimmed)
- `plots/best_interventions/` — index of top-ranked profiles
- `plots/head_correlations/` — per-head heatmaps (top-N profiles only)
- `plots/pca_interventions/` — PCA delta scatters (top-N profiles only)

### Useful Plot Flags

- `--x-metric`: single primary x-axis for the main plots
- `--x-metrics`: batch-render multiple x-axes in one run
- `--x-metric` / `--x-metrics` choices: `tokens_approx`, `baseline_target_prob`, `baseline_target_geo_mean_prob`, `baseline_final_entropy_bits`, `baseline_mean_entropy_bits`, `baseline_top1_top2_logit_margin`, `baseline_attn_entropy_mean`, `target_prob`, `baseline_target_rank`, `target_rank`
- `--intervention-folders`: generate per-intervention subfolders
- `--best-interventions-top-n`: size of the ranked best-intervention index
- `--label-top-n`: annotate strongest outliers in each panel

## Plotting A Cross-Model Comparison

Use `signal_lab.sweep_plot_compare` on a pairwise comparison bundle.

### Basic Example

```bash
uv run -m signal_lab.sweep_plot_compare \
  --compare-dir [DATA_DIR]/outputs/signal_lab/runs/checkpoint_name/_comparisons/qwen35b_vs_olmo
```

### Intervention Folder View

```bash
uv run -m signal_lab.sweep_plot_compare \
  --compare-dir [DATA_DIR]/outputs/signal_lab/runs/checkpoint_name/_comparisons/qwen35b_vs_olmo \
  --intervention-folders \
  --best-interventions-top-n 12 \
  --disagreement-top-n 12
```

This produces:

- `plots/interventions/<g_profile>/`
- `plots/best_interventions/`
- `plots/biggest_model_disagreements/`

The cross-model intervention folders are intentionally simple and useful for
directional inspection:

- `scatter_delta_prob_all.png`
- `scatter_delta_prob_by_type.png`
- `scatter_baseline_gap_vs_delta_gap_by_type.png`
- `summary.json`

## Suggested Workflow

### One Model

```bash
uv run -m signal_lab.sweep \
  --cartridge kitchen_sink \
  --run-name battery3 \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-types algorithmic \
  --prompt-tiers short \
  --model-key 35B \
  --device cuda

uv run -m signal_lab.sweep_analyze \
  --run-dir [DATA_DIR]/outputs/signal_lab/runs/battery3/35B

uv run -m signal_lab.sweep_plot_analyze \
  --analysis-dir [DATA_DIR]/outputs/signal_lab/runs/battery3/35B/analysis \
  --x-metric baseline_target_prob \
  --intervention-folders
```

### Two Models

```bash
uv run -m signal_lab.sweep_compare \
  --run-a [DATA_DIR]/outputs/signal_lab/runs/battery3/35B/analysis \
  --run-b [DATA_DIR]/outputs/signal_lab/runs/battery3/OLMO/analysis \
  --label-a qwen35b \
  --label-b olmo \
  --output-dir [DATA_DIR]/outputs/signal_lab/runs/battery3/_comparisons/qwen35b_vs_olmo \
  --prefix qwen35b_vs_olmo

uv run -m signal_lab.sweep_plot_compare \
  --compare-dir [DATA_DIR]/outputs/signal_lab/runs/battery3/_comparisons/qwen35b_vs_olmo \
  --intervention-folders
```

## Notes

- Simpler scatter plots tend to be the easiest way to get directionality before
  moving into more aggregated analysis.
- The intervention-folder outputs are meant to make exact `g_profile`
  inspection cheap.
- For current project framing, `signal_lab` is the implemented measurement and
  intervention layer; `colony` is the future collective-signal layer.
- For a complete file-level reference to analysis and plot outputs (including
  what each file is useful for and the rationale for the slimmed plot defaults),
  see `docs/data_manifests.md`.
