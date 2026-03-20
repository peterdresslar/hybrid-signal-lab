## Signal Lab

`signal_lab` is the current probing and analysis package for this repository.
It runs hybrid LLMs under baseline and gain-vector interventions, records
prompt-level response metrics, and generates summaries and plots for both
single-model and cross-model comparisons.

The broader `colony` concept is future work. For the current iteration, the
main implemented loop is:

1. build or select a prompt battery
2. run `signal_lab` or `sweep` over one or more models
3. analyze the resulting runs
4. compare runs across models
5. generate simple plots to inspect directionality

Shared model backends and prompt structures live in the top-level `model`
package.

## Package Contents

- `signal_lab.signal_lab`: one-off probing CLI for a single prompt
- `signal_lab.sweep`: batch runner over cartridges or battery selections
- `signal_lab.sweep_cartridges`: named gain-profile sweep specs
- `signal_lab.sweep_analyze`: summarize one sweep run directory
- `signal_lab.sweep_compare`: compare two analyzed run directories
- `signal_lab.sweep_plot_analyze`: plot one analyzed run
- `signal_lab.sweep_plot_compare`: plot one pairwise comparison bundle
- `signal_lab.agent`: inference pipeline used by the CLIs

## Setup

Requires Python `>=3.13` and is intended to be used with `uv`.

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

Use `signal_lab.signal_lab` when you want one prompt, one model, and one gain
specification.

### Simplest Example

```bash
uv run -m signal_lab.signal_lab \
  --prompt "The color with the shortest wavelength is" \
  --model-key 0_8B \
  --g-function constant \
  --g 1.0
```

This writes a full JSON summary to `signal_lab_output.json`.

### Prompt From A Battery

```bash
uv run -m signal_lab.signal_lab \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-id alg_gen_arithmetic_0000 \
  --model-key 2B \
  --g-function constant \
  --g 1.25
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
- `--device`: `auto`, `cuda`, `mps`, or `cpu`

## Running Sweeps

Use `signal_lab.sweep` when you want many prompts and/or many gain profiles.

### Cartridge-Based Sweep

```bash
uv run -m signal_lab.sweep \
  --cartridge uniform_check_lite \
  --model-key 0_8B
```

### Battery-Backed Sweep

```bash
uv run -m signal_lab.sweep \
  --cartridge kitchen_sink \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-types algorithmic \
  --prompt-tiers short \
  --model-key 35B \
  --device cuda \
  --out-dir results/battery3_qwen35b_alg_short_{timestamp}
```

### Specific Prompt IDs From A Battery

```bash
uv run -m signal_lab.sweep \
  --cartridge kitchen_sink \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-ids alg_gen_arithmetic_0000,alg_gen_arithmetic_0001 \
  --model-key OLMO \
  --device cuda
```

### Verbose Sweep

```bash
uv run -m signal_lab.sweep \
  --cartridge kitchen_sink \
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-types long_range_retrieval \
  --model-key 9B \
  --verbose
```

Verbose mode adds `verbose.jsonl` with heavier per-run detail.

### Sweep Outputs

Each sweep run directory contains:

- `main.jsonl`: core per-run metrics
- `_meta.json`: model/config/prompt-selection metadata
- `errors.jsonl`: failures if any occur
- `verbose.jsonl`: optional detailed output when `--verbose` is enabled

## Analyzing A Sweep Run

Use `signal_lab.sweep_analyze` on one sweep output directory.

### Example

```bash
uv run -m signal_lab.sweep_analyze \
  --run-dir data/sweep_sample
```

### Example With Separate Output Directory

```bash
uv run -m signal_lab.sweep_analyze \
  --run-dir /path/to/raw_run \
  --output-dir /path/to/analysis_run
```

### Example Without Writing Files

```bash
uv run -m signal_lab.sweep_analyze \
  --run-dir data/sweep_sample \
  --no-write-files
```

### Main Analysis Artifacts

- `analysis_report.txt`
- `analysis_joined_long.csv`
- `analysis_type_gain_summary.csv`
- `analysis_type_family_summary.csv`
- `analysis_best_profile_by_type.csv`
- `analysis_prompt_winners.csv`
- `analysis_type_gain_matrix_delta_p.csv`
- `analysis_type_family_matrix_delta_p.csv`

## Comparing Two Analyzed Runs

Use `signal_lab.sweep_compare` after each run has already been analyzed.

### Minimal Example

```bash
uv run -m signal_lab.sweep_compare \
  --run-a /path/to/analysis_run_a \
  --run-b /path/to/analysis_run_b
```

### Labeled Cross-Model Example

```bash
uv run -m signal_lab.sweep_compare \
  --run-a /path/to/analysis35B \
  --run-b /path/to/analysisO \
  --label-a qwen35b \
  --label-b olmo \
  --output-dir /path/to/qwen35b_vs_olmo \
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
  --analysis-dir data/exemplar/analysis35B \
  --x-metric tokens_approx
```

### Baseline Difficulty View

```bash
uv run -m signal_lab.sweep_plot_analyze \
  --analysis-dir data/exemplar/analysis35B \
  --x-metric baseline_target_prob
```

### Intervention Folder View

```bash
uv run -m signal_lab.sweep_plot_analyze \
  --analysis-dir data/exemplar/analysis35B \
  --x-metric tokens_approx \
  --intervention-folders \
  --best-interventions-top-n 12
```

This produces:

- `plots/baseline/`
- `plots/interventions/<g_profile>/`
- `plots/best_interventions/`

### Useful Plot Flags

- `--x-metric`: `tokens_approx`, `baseline_target_prob`, `target_prob`, `baseline_target_rank`, `target_rank`
- `--intervention-folders`: generate per-intervention subfolders
- `--best-interventions-top-n`: size of the ranked best-intervention index
- `--label-top-n`: annotate strongest outliers in each panel

## Plotting A Cross-Model Comparison

Use `signal_lab.sweep_plot_compare` on a pairwise comparison bundle.

### Basic Example

```bash
uv run -m signal_lab.sweep_plot_compare \
  --compare-dir data/exemplar/35B-v-OLMo
```

### Intervention Folder View

```bash
uv run -m signal_lab.sweep_plot_compare \
  --compare-dir data/exemplar/35B-v-OLMo \
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
  --prompt-battery bench/battery/data/battery_3 \
  --prompt-types algorithmic \
  --prompt-tiers short \
  --model-key 35B \
  --device cuda \
  --out-dir results/qwen35b_alg_short_{timestamp}

uv run -m signal_lab.sweep_analyze \
  --run-dir results/qwen35b_alg_short_YYYYMMDD_HHMMSS

uv run -m signal_lab.sweep_plot_analyze \
  --analysis-dir results/qwen35b_alg_short_YYYYMMDD_HHMMSS \
  --x-metric baseline_target_prob \
  --intervention-folders
```

### Two Models

```bash
uv run -m signal_lab.sweep_compare \
  --run-a /path/to/analysis35B \
  --run-b /path/to/analysisO \
  --label-a qwen35b \
  --label-b olmo \
  --output-dir /path/to/qwen35b_vs_olmo \
  --prefix qwen35b_vs_olmo

uv run -m signal_lab.sweep_plot_compare \
  --compare-dir /path/to/qwen35b_vs_olmo \
  --intervention-folders
```

## Notes

- Simpler scatter plots tend to be the easiest way to get directionality before
  moving into more aggregated analysis.
- The intervention-folder outputs are meant to make exact `g_profile`
  inspection cheap.
- For current project framing, `signal_lab` is the implemented measurement and
  intervention layer; `colony` is the future collective-signal layer.
