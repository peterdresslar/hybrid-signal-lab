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
| syntactic_pattern | 25 | Generator |
| domain_knowledge | 15 | Generator |
| code_comprehension | 15 | Generator |
| **Total** | **400** | |

## Usage

```bash
# 1. Install dependencies
uv add datasets

# 2. Generate the full candidate battery
python build_battery.py --output battery_candidates.json

# 3. Run baseline calibration sweep (on Sol)
python calibrate.py --battery battery_candidates.json --model Qwen/Qwen3.5-2B-Base

# 4. Filter to sweet spot
python filter_battery.py --results calibration_results.jsonl --output battery_final.json
```

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
