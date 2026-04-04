# Hybrid Signal Lab

A testbed for hybrid-model probing and collective-signal research, developed as a CAS capstone at Arizona State University (advisor: Prof. Bryan Daniels).

This is a work in progress.

The current implemented stack is centered on `signal_lab`, which probes hybrid-architecture LLMs by applying gain intervention to the attention sublayer at inference time. The broader `colony` concept remains the future collective-signal layer that will eventually generate or adapt those interventions.

The subject models are Qwen/Qwen3.5 and allenai/olmo-hybrid, both built on hybrid architectures that interleave Gated DeltaNet (GDN) layers with softmax attention layers in a 3:1 ratio. A gain vector *g* is applied as a multiplicative scalar to the attention sublayer contribution at each softmax attention layer. At *g* = 1.0 the model runs unmodified (baseline); deviations from 1.0 amplify or suppress the attention pathway relative to the feed-forward and residual stream, letting the system explore the response surface without retraining.

The gain can be applied in two modes. **Block-output mode** scales the entire decoder block output (attention + feed-forward + residual), which was the original implementation. **Attention-contribution mode** scales only the attention sublayer output at the residual add point, isolating the causal pathway the intervention targets. The two modes produce qualitatively different experimental results: attention-contribution mode yields 5–6× stronger effects on 9B, extends the productive gain range, and reveals task-dependent intervention patterns (particularly on code comprehension and numerical reasoning) that are invisible under block-output mode. The mode × architecture interaction is also significant — the pre-norm Qwen models respond strongly to attention-contribution intervention while the post-norm OLMo architecture constrains the intervention's leverage. See `signal_lab/README.md` for the full technical description of intervention modes and hook targets.

See `docs/project_notes.md` for the latest information on the project's direction.

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
