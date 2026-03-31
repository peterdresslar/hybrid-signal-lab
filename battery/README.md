# Battery

A prompt battery for evaluating inference-time interventions on language models. The battery provides 1,070 cloze-style prompts spanning 11 mechanism types, designed to measure how gain vector interventions affect next-token prediction across qualitatively different tasks.

While this module could function as a standalone prompt generation and calibration toolkit, it was built as part of [Hybrid Signal Lab](../README.md) — a research project on inference-time intervention in hybrid linear attention models, developed as a CAS capstone at Arizona State University (advisor: Prof. Bryan Daniels). The battery serves as the primary input to `signal_lab`'s gain profile sweep experiments.


## What's in the battery

Each prompt is a text prefix with a known continuation. The model's job is to predict the next token(s). By comparing prediction quality with and without gain interventions, we can characterize how different architectural components contribute to different kinds of language processing.

The 11 prompt types test distinct mechanisms:

- **factual_recall** and **factual_retrieval** — stored knowledge retrieval, sourced from COUNTERFACT
- **domain_knowledge** — specialized vocabulary in context, derived from Wikipedia articles
- **cultural_memorized** — passage completion from LAMBADA, testing memorized text
- **reasoning_numerical** and **reasoning_tracking** — arithmetic and state-tracking tasks
- **algorithmic** — procedural operations (sorting, filtering, mapping)
- **structural_copying** — pattern continuation in structured formats (CSV rows, outlines, key-value pairs)
- **long_range_retrieval** — recovering a fact buried in filler paragraphs
- **syntactic_pattern** — grammatical agreement and completion
- **code_comprehension** — predicting Python program outputs


## Quick start

Requires Python 3.12.x (project pin: `3.12.8`) and [uv](https://docs.astral.sh/uv/).

**Build a battery from an existing recipe:**

```bash
uv run -m battery.src.build_battery \
  --outdir battery/data/battery_4 \
  --recipe battery/data/recipes/battery_4_recipe.json
```

This produces per-type JSON files, a combined `all_candidates.json`, and a `manifest.json` recording the build parameters.

**Regenerate the procedural seed pools** (refreshes all `*_seed.json` files in `battery/data/sources/`):

```bash
uv run -m battery.src.build_battery \
  --recipe battery/data/recipes/battery_4_recipe.json \
  --reseed
```

Note: `code_comprehension` reseeding requires an `OPENROUTER_KEY` environment variable. `domain_knowledge` is intentionally excluded from reseeding since its pool is curated from Wikipedia rather than procedurally generated. `structural_copying` is generated inline during the build.

The battery pipeline has three steps. Calibration and analysis outputs live in a data directory of your choosing, separate from the battery build artifacts.

**Step 1: Calibrate** — run prompts through a model and record baseline target-token statistics:

```bash
uv run -m battery.src.calibrate \
  --battery battery/data/battery_4/all_candidates.json \
  --model Qwen/Qwen3.5-9B-Base \
  --output ~/workspace/data/calibration/calibration_qwen9b.jsonl \
  --device cuda
```

Add `--output-attentions` and `--output-hidden-states` for richer diagnostic output.

**Step 2: Analyze** — summarize calibration results by type, tier, family, concept, and difficulty:

```bash
uv run -m battery.src.calibration_analyze \
  --calibration ~/workspace/data/calibration/calibration_qwen9b.jsonl \
  --candidates battery/data/battery_4/all_candidates.json
```

Analysis files are written next to the calibration JSONL by default. Use `--output-dir` to override, or `--battery-dir` to scan an entire directory of calibration files at once.

**Step 3: Annotate** — assign train/test splits informed by cross-model calibration statistics:

```bash
uv run -m battery.src.annotate_battery \
  --analysis-dir ~/workspace/data/calibration/b4 \
  --candidates battery/data/battery_4/all_candidates.json \
  --output battery/data/battery_4/annotation_manifest.json
```

The annotation script reads per-model `analysis_item_cross_model.csv` files produced by Step 2, merges them, and classifies each prompt as `eligible` or `too_hard` based on whether any model assigns the target token a probability above `--hard-threshold` (default 0.01). All eligible items enter a deterministic stratified train/test split (default 80/20 per type). Prompts where no model can engage with the target are assigned to `other` and excluded from the split.

The output manifest maps prompt IDs to `train_prompt`, `test_prompt`, or `other` without mutating the battery itself. Each item also carries informational sub-labels (`eligible_easy`, `eligible_model_separating`, `eligible_sweet_spot`) and cross-model probability statistics for downstream analysis.


## Project structure

```
battery/
├── data/
│   ├── recipes/              # Per-battery count overrides (JSON)
│   ├── sources/              # Durable source pools (*_seed.json, *_pool.json)
│   └── battery_4/            # Built output: per-type JSONs, manifest, calibration
├── src/
│   ├── build_battery.py      # Main builder: assembles prompts from pools + generators
│   ├── calibrate.py          # Runs prompts through a model, records target-token stats
│   ├── calibration_analyze.py # Summarizes calibration JSONL by type, tier, family
│   ├── annotate_battery.py   # Assigns train/test splits from cross-model calibration
│   ├── wikipedia_generate.py # Two-stage pipeline: Wikipedia API → LLM filter → cloze generation
│   ├── algorithmic_generate.py
│   ├── code_generate.py
│   ├── reasoning_numerical_generate.py
│   ├── reasoning_tracking_generate.py
│   ├── long_range_retrieval_generate.py
│   └── syntactic_pattern_generate.py
└── README.md
```


## How the build works

The builder has three source types:

1. **Inline generators** — deterministic, programmatic prompt construction (structural_copying and fallbacks for other types).
2. **External seed pools** — pre-generated JSON files in `battery/data/sources/` that the builder samples from. Most types now use these. The `--reseed` flag runs the standalone generators to refresh them.
3. **HuggingFace datasets** — COUNTERFACT for factual recall/retrieval, LAMBADA for cultural memorized. Downloaded on first use and cached.

A recipe file specifies how many prompts to draw from each type. The builder auto-detects seed pools by filename convention (`{type}_seed.json` or `{type}_pool.json`). If a pool exists, it's used; otherwise the builder falls back to the inline generator.


## Prompt format

```json
{
  "id": "dk_gen-wikipedia-random-battery4_0042",
  "prompt": "In economics, the theory that increasing the money supply leads to proportional increases in price levels is known as the quantity theory of",
  "target": " money",
  "type": "domain_knowledge",
  "tier": "short",
  "tokens_approx": 29,
  "source": "gen-wikipedia-random-battery4",
  "metadata": {}
}
```

Targets are cloze continuations, almost always beginning with a space token (the exception is structural_copying delimiter completions like `)` or `}`). Calibration measures first-token target probability, not full sequence likelihood.

The `tier` field bins prompts by approximate token count: short (≤30), brief (31–80), med (81–200), long (201–500), extended (500+).


## Generating domain knowledge prompts

The `wikipedia_generate.py` script is a standalone two-stage pipeline for creating grounded cloze prompts from Wikipedia:

```bash
uv run -m battery.src.wikipedia_generate battery/data/sources/domain_knowledge_pool.json 120 \
  --min-tokens 50 --append
```

Stage 1 fetches random Wikipedia articles and asks an LLM whether the topic is suitable (rejecting biographies, minor geographic entries, sports, media). Stage 2 sends the full article text to the LLM to generate a passage ending in a domain-specific cloze. The script includes exponential backoff for Wikipedia rate limiting.

Requires `OPENROUTER_KEY`. Default model is Gemini 3 Flash via OpenRouter.


## References

- **COUNTERFACT**: Meng et al., ["Locating and Editing Factual Associations in GPT"](https://arxiv.org/abs/2202.05262). Battery uses [`NeelNanda/counterfact-tracing`](https://huggingface.co/datasets/NeelNanda/counterfact-tracing) on HuggingFace.
- **LAMBADA**: Paperno et al., ["The LAMBADA dataset: Word prediction requiring a broad discourse context"](https://arxiv.org/abs/1606.06031). Battery uses [`EleutherAI/lambada_openai`](https://huggingface.co/datasets/EleutherAI/lambada_openai).
- **Targeted syntactic evaluation**: Marvin and Linzen, ["Targeted Syntactic Evaluation of Language Models"](https://aclanthology.org/D18-1151/) and the [`BeckyMarvin/LM_syneval`](https://github.com/BeckyMarvin/LM_syneval) templates, which inform the agreement-style syntactic_pattern items.
- **Wikipedia-derived prompts**: Curated from random article summaries and full extracts via the [Wikimedia REST API](https://en.wikipedia.org/api/rest_v1/page/random/summary) and [MediaWiki Action API](https://en.wikipedia.org/w/api.php).
