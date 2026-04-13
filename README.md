# Hybrid Signal Lab

Hybrid Signal Lab is a research codebase for probing hybrid-architecture language models with controlled gain interventions, building prompt batteries, and studying whether prompt-level model state can help predict when intervention will help. The project began as a CAS capstone at Arizona State University under the supervision of Prof. Bryan Daniels.

At a high level, this repository is for:

- running baseline and intervened sweeps on hybrid models,
- analyzing how prompt families separate under those sweeps,
- benchmarking fixed intervention profiles and selected profile panels,
- and experimenting toward eventual routing or collective-signal methods.

The current experimental focus is on Qwen/Qwen3.5 and `allenai/olmo-hybrid`, both of which interleave Gated DeltaNet layers with softmax attention layers. The core intervention multiplies the attention pathway by a gain vector `g` at the attention-bearing layers. At `g = 1.0` the model runs unchanged; moving away from `1.0` amplifies or suppresses the attention contribution and exposes a measurable response surface without retraining.

This front-door README is intentionally brief. The detailed technical story lives in the module READMEs:

- [signal_lab/README.md](signal_lab/README.md): core intervention runtime, sweep execution, hook behavior, and output schema.
- [battery/README.md](battery/README.md): prompt battery construction and data organization.
- [bench/README.md](bench/README.md): benchmark passes and evaluation workflow.
- [router/README.md](router/README.md): profile selection, routing experiments, and current router artifacts.
- [docs/analysis/README.md](docs/analysis/README.md): analysis outputs, diagnostics, and figure-generation organization.
- [docs/figurelib/README.md](docs/figurelib/README.md): reusable plotting utilities.

Additional project notes live in [docs/project_notes.md](docs/project_notes.md).

This repository remains an active research workbench rather than a polished package. We expect to keep developing the ideas here, in some form, after the degree concludes in May 2026.

## Usage

### Setup

Requires Python 3.12.x (pinned locally to `3.12.8`). Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

You will need a Hugging Face token with access to the Qwen models. Create a `.env.development` file in the project root:

```
HF_TOKEN=hf_your_token_here
```

`HUGGINGFACE_HUB_TOKEN` is also supported.

In the Signal Lab docs below, `[DATA_DIR]` means the base data root used for
Signal Lab inputs and outputs. By default this is `data/`, but you can override
it per command with `--data-dir` or via the `DATA_DIR` environment variable.

Device selection defaults to auto-detection (`cuda` -> `mps` -> `cpu`). You can override with either:

```bash
export COLONY_DEVICE=cuda
```

or per command:

```bash
uv run python -m signal_lab.signal_lab --device cuda ...
```

### Running Signal Lab standalone

`signal_lab.signal_lab` is a diagnostic tool for running a single forward pass through the model with a configurable attention scaling profile. It reports top-k logits, entropy, and attention statistics, and writes a full summary under `[DATA_DIR]/outputs/signal_lab/probes/` by default.

```bash
uv run python -m signal_lab.signal_lab --prompt "The color with the shortest wavelength is" --g-function constant --g 1.0
```

- `--prompt` accepts a literal string, a path to a file, or a filename in the `[DATA_DIR]` directory.
- `--g-function` selects the profile family (`constant`, `linear`, `gaussian`, `step`, `control_points`).
- `--g` is a shortcut for the constant profile value.
- `--g-vector` provides comma-separated control points for `control_points`.
- `--g-params-json` provides extra family parameters (for example slope/intercept, gaussian center/width, or step threshold).
- `--device` optionally overrides hardware (`auto`, `cuda`, `mps`, `cpu`).

### Running sweeps

`signal_lab.sweep` automates running prompts across a cartridge-defined set of *g* profile specifications (`g_specs`), collecting per-run metrics (target rank, target probability, final entropy, KL divergence from baseline) into a structured output directory.

```bash
uv run python -m signal_lab.sweep --cartridge uniform_check_lite
```

Options:

- `--cartridge` — required named sweep configuration from `signal_lab/sweep_cartridges.py`.
- `--model-key` — optional model selector (`0_8B`, `2B`, `4B`, `9B`), default `0_8B`.
- `--device` — optional hardware override (`auto`, `cuda`, `mps`, `cpu`).
- `--repetitions` — number of repetitions per prompt/g pair (default `1`).
- `--verbose` — log full top-k and attention entropy to a separate `verbose.jsonl`.
- `--data-dir <path>` — optional base directory to use instead of `data/` for Signal Lab inputs and outputs.
- `--run-name <name>` — checkpoint-like folder name for the default output layout.
- `--out-dir <path>` — optional explicit output directory. If omitted, outputs land in `[DATA_DIR]/outputs/signal_lab/runs/<run-name>/<model-key>/`.

Results are written as JSONL to `<out-dir>/main.jsonl`, with model metadata in `<out-dir>/_meta.json`.

### Short prompts

The short probe prompts and their expected target tokens (used by the sweep to track target rank and probability across *g* values):

| File | Prompt | Target |
|------|--------|--------|
| `short0.txt` | The color with the shortest wavelength is | violet |
| `short1.txt` | 1, 1, 2, 3, 5, 8, 13, 21, | 34 |
| `short2.txt` | The capital of Mongolia is | U |
| `short3.txt` | She opened the door and he opened the | door |
| `short4.txt` | roses are red, violets are blue, sugar is sweet, and | so |
| `short5.txt` | import torch\nimport torch.nn as | nn |
