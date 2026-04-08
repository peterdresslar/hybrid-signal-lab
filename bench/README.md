# Bench

`bench` evaluates baseline, fixed-profile, routed, and oracle behavior on a
small benchmark suite built around the current routing stack.

The harness currently supports:

- `copa`
- `storycloze`
- `gsm8k`

Evaluation modes:

- `baseline`: no intervention, `g = 1.0`
- `fixed`: apply one selected profile to every example
- `routed`: baseline sensing pass, router classification, then intervention
- `oracle`: for scoring tasks, try every routed profile and pick the best

## Task types

Two different scoring regimes are implemented:

- Log-likelihood selection tasks:
  - `COPA`
  - `StoryCloze`
- Generation task:
  - `GSM8K`

For `COPA` and `StoryCloze`, the model scores candidate continuations with
`Agent.score_target()` and chooses the higher-likelihood option.

For `GSM8K`, the model generates a solution and the harness extracts the final
numeric answer from the `#### ...` suffix.

## Usage

Example baseline-only run:

```bash
uv run -m bench.run_bench \
  --model-key 9B \
  --tasks copa storycloze gsm8k \
  --baseline-only \
  --output-dir data/bench/9B
```

Example routed run:

```bash
uv run -m bench.run_bench \
  --model-key 9B \
  --tasks copa storycloze gsm8k \
  --router-model router/router-9B-011/router_model.json \
  --output-dir data/bench/routed_9B
```

The router model JSON should match the model being benchmarked.

## Files

- `run_bench.py`
  Main benchmark driver.
- `tasks.py`
  Dataset loading, prompt formatting, and answer extraction helpers.
- `../bench-diag.sh`
  SLURM helper script for baseline and routed benchmark runs on cluster.

## Notes

- The current benchmark path is Qwen/OLMo router evaluation, not the broader
  Signal Lab sweep workflow.
- The benchmark runner assumes the relevant model/router artifacts already
  exist locally.
