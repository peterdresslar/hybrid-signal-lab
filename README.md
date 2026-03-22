## Peter Dresslar CAS Capstone Spring 2026

A testbed for hybrid-model probing and collective-signal research, developed as a CAS capstone at Arizona State University (advisor: Prof. Bryan Daniels).

The current implemented stack is centered on `signal_lab`, which probes hybrid-architecture LLMs by modulating the balance between recurrent (GDN) and attention layer types at inference time. The broader `colony` concept remains the future collective-signal layer that will eventually generate or adapt those interventions.

The current implementation targets Qwen/Qwen3.5 and allenai/olmo-hybrid, both built on hybrid architectures that interleave Gated DeltaNet (GDN) layers with gated attention layers in a 3:1 ratio. Forward hooks scale the attention-layer residual contributions by a factor *g* (g→0: GDN-dominated, g→1: attention-dominated), letting the system explore the full response surface without retraining.

See `docs/proposal.md` for the full project proposal.

## Usage

### Setup

Requires Python ≥ 3.13. Install with [uv](https://docs.astral.sh/uv/):

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
