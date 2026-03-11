## Peter Dresslar CAS Capstone Spring 2026

A testbed for measuring collective behavior in multi-agent LLM systems, developed as a CAS capstone at Arizona State University (advisor: Prof. Bryan Daniels).

The project investigates whether a colony of hybrid-architecture LLM agents can collectively modulate the balance between recurrent (GDN) and attention layer types at inference time, and whether the resulting collective exhibits the amplification and decomposition predicted by formal theories of collectivity (Daniels et al., 2016). Each agent processes a different window of shared context and broadcasts compressed activation signals to a shared buffer — analogous to pheromone signaling in eusocial insect colonies. The aggregate signal drives a global "g knob" that scales attention-layer contributions across the colony.

The current implementation targets **Qwen3.5-2B**, a hybrid transformer that interleaves Gated DeltaNet (GDN) layers with gated attention layers in a 3:1 ratio. Forward hooks scale the attention-layer residual contributions by a factor *g* (g→0: GDN-dominated, g→1: attention-dominated), letting the system explore the full response surface without retraining.

See `docs/proposal.md` for the full project proposal.

## Usage

### Setup

Requires Python ≥ 3.13. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

You will need a HuggingFace token with access to `Qwen/Qwen3.5-2B`. Create a `.env.development` file in the project root:

```
HF_TOKEN=hf_your_token_here
```

### Running Signal Lab standalone

`signal_lab.py` is a diagnostic tool for running a single forward pass through the model with a given attention scaling factor. It reports top-k logits, entropy, and attention statistics, and writes a full summary to `signal_lab_output.json`.

```bash
uv run python -m colony.signal_lab --prompt "The color with the shortest wavelength is" --g 1.0
```

- `--prompt` accepts a literal string, a path to a file, or a filename in the `data/` directory.
- `--g` sets the attention scaler (default `1.0`, the unmodified model). Values below 1.0 suppress attention layers; values above 1.0 amplify them.

### Running sweeps

`sweep.py` automates running every prompt in the `data/` directory across a range of *g* values, collecting per-run metrics (target rank, target probability, final entropy, KL divergence from baseline) into a structured output directory.

```bash
uv run python -m colony.sweep
```

Options:

- `--granularity` — step size between *g* values (default `0.25`, range 0.0–2.0).
- `--repetitions` — number of repetitions per prompt/g pair (default `1`).
- `--short-only` / `--med-only` — restrict to short or medium prompts.
- `--use-prompt <file>` — run a single specific prompt file.
- `--verbose` — log full top-k and attention entropy to a separate `verbose.jsonl`.
- `--out-dir <path>` — output directory (default `results/sweep_{timestamp}`).

Results are written as JSONL to `<out-dir>/main.jsonl`, with model metadata in `<out-dir>/_meta.json`.
