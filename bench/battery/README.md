# Battery Builder — Attention Is In The Air

Generates a ~400-candidate prompt battery for g-profile sweep experiments.
After generation, run a baseline calibration sweep and filter to ~200 items
in the sweet spot (baseline p(tok) ∈ [0.05, 0.85]).

## Target Distribution (oversampled for high-headroom types)

| Type | Target Count | Source |
|------|-------------|--------|
| factual_recall | 80 | COUNTERFACT |
| structural_copying | 60 | Generator |
| reasoning_numerical | 40 | Generator |
| reasoning_tracking | 40 | Generator |
| factual_retrieval | 40 | COUNTERFACT (long-context) |
| long_range_retrieval | 30 | Generator |
| algorithmic | 30 | Generator |
| cultural_memorized | 25 | LAMBADA + curated |
| syntactic_pattern | 25 | Generator + adapted LM_syneval-style agreement templates |
| domain_knowledge | 15 | Generator |
| code_comprehension | 15 | Generator |
| **Total** | **400** | |

## Usage

```bash
# 1. Install dependencies
uv add datasets

# 2. Generate the full candidate battery
uv run -m bench.battery.src.build_battery --outdir battery

# 3. Run baseline calibration sweep (on Sol)
uv run -m bench.battery.src.calibrate --battery battery/all_candidates.json --model Qwen/Qwen3.5-2B-Base

# 4. Filter to sweet spot
python filter_battery.py --results calibration_results.jsonl --output battery_final.json
```

## References

- COUNTERFACT: [Meng et al., "Locating and Editing Factual Associations in GPT"](https://arxiv.org/abs/2202.05262), with the battery using the [`NeelNanda/counterfact-tracing`](https://huggingface.co/datasets/NeelNanda/counterfact-tracing) Hugging Face dataset for `factual_recall` and `factual_retrieval`.
- LAMBADA: [Paperno et al., "The LAMBADA dataset: Word prediction requiring a broad discourse context"](https://arxiv.org/abs/1606.06031), with the battery using the [`EleutherAI/lambada_openai`](https://huggingface.co/datasets/EleutherAI/lambada_openai) Hugging Face dataset for `cultural_memorized`.
- Targeted syntactic evaluation: [Marvin and Linzen, "Targeted Syntactic Evaluation of Language Models"](https://aclanthology.org/D18-1151/) and the accompanying [`BeckyMarvin/LM_syneval`](https://github.com/BeckyMarvin/LM_syneval) templates, which inform the agreement-style `syntactic_pattern` items.

## Output Format

```json
{
  "id": "fr_counterfact_0042",
  "prompt": "The capital of France is",
  "target": " Paris",
  "type": "factual_recall",
  "tier": "short",
  "tokens_approx": 7,
  "source": "counterfact",
  "metadata": {}
}
```
