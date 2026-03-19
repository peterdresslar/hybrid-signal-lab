# Independent Signal Analysis of Kitchen-Sink Runs

Generated from raw files in `results/kitchen_sink/` and prompt metadata in `battery/data/`.

This report is intentionally independent of the existing `docs/kitchen_sink_signal_report.md`. The goal here is to describe what signals are actually present in the kitchen-sink sweep results before comparing those findings against any prior writeup.

## Scope

The dataset contains 4 model runs:

- `Qwen/Qwen3.5-0.8B-Base`
- `Qwen/Qwen3.5-2B-Base`
- `Qwen/Qwen3.5-9B-Base`
- `allenai/Olmo-Hybrid-7B`

Each run covers:

- 22 prompts
- 31 `g_profiles`
- 1 replication per prompt/profile pair

So the full dataset contains 2,728 observations.

The prompt battery is shared across models:

- 6 `short`
- 6 `brief`
- 6 `med`
- 4 `long`

The models differ in how many sampled attention control slots were exposed:

- `Qwen3.5-0.8B` and `Qwen3.5-2B`: 6 sampled attention layers
- `Qwen3.5-9B` and `Olmo-Hybrid-7B`: 8 sampled attention layers

## What Counts As a Signal Here

The raw outputs expose several candidate signal families.

### 1. Baseline outcome and confidence signals

These come directly from the baseline `g=1.0` row for each model-prompt pair:

- `target_prob`
- `target_rank`
- `target_seq_logprob`
- `target_avg_logprob`
- `target_first_token_prob`
- `target_first_token_rank`

These are the most obvious potential control signals because they say how confidently the model already prefers the desired target under the unperturbed setting.

### 2. Output-distribution uncertainty signals

- `final_entropy_bits`
- `kl_from_baseline` for non-baseline profiles
- top-`k` logit structure through `top_k_logits`
- top-`k` token identities through `top_k_tokens`

These indicate whether the output distribution is sharp or diffuse and how violently a profile changes it.

### 3. Attention-state signals

From `verbose.jsonl`, each row also carries:

- `mean_entropy_bits`
- `attn_entropy_per_head_final`
- `attention_layer_indices`

These are richer than the final-token confidence signals because they encode where attention is concentrated or diffuse at the sampled layers.

### 4. Surface-shape signals

These are not single fields but properties of the prompt-specific response surface across all 31 profiles:

- best-vs-baseline improvement
- worst-vs-baseline damage
- which profile family wins
- whether early-heavy or late-heavy interventions help
- cross-model similarity of the full response vector

These are not available from one row, but they are clearly present in the sweep results and matter for downstream controller design.

## Core Question

The main practical question is:

Can any baseline-observable or cheaply computed signal predict when a prompt has headroom under `g-profile` intervention, and if so, what sort of intervention is promising?

For this report I define:

- `headroom` = best target probability across all profiles minus baseline target probability
- `damage` = baseline target probability minus worst target probability

These are useful because Experiment 1 only matters if there is recoverable headroom, and a controller is also valuable if it can avoid catastrophic profiles.

## Headroom Exists on Most Prompts

The kitchen-sink data shows real room for improvement.

By model:

- `Qwen3.5-0.8B`: improved on `22/22` prompts, mean headroom `+0.1447`
- `Qwen3.5-2B`: improved on `20/22` prompts, mean headroom `+0.1152`
- `Qwen3.5-9B`: improved on `20/22` prompts, mean headroom `+0.1476`
- `Olmo-Hybrid-7B`: improved on `19/22` prompts, mean headroom `+0.1280`

Pooled, `81/88` model-prompt cases have a non-baseline profile that beats baseline.

This means the dataset contains a real control problem, not just noise.

## The Strongest Baseline Signals

I computed pooled correlations over the 88 baseline model-prompt cases.

### Baseline target probability

- Pearson correlation with headroom: `-0.335`
- Spearman correlation with headroom: `-0.303`

Interpretation: when the baseline model is already very confident on the target, there is usually less room for `g-profile` improvement. This is unsurprising, but it is a genuine and usable triage signal.

### Top-1 logit margin

Using the difference between the top two baseline logits:

- Pearson correlation with headroom: `-0.392`
- Spearman correlation with headroom: `-0.432`

This is one of the cleanest signals in the dataset. Small winner-vs-runner-up margins mark prompts where the current operating point is unstable and more likely to benefit from intervention.

On `Qwen3.5-9B`, this signal is especially strong:

- Pearson `-0.662`
- Spearman `-0.791`

### Final-token entropy

- Pearson correlation with headroom: `0.169`
- Spearman correlation with headroom: `0.354`

This looks more monotonic than linear. Higher entropy often means more recoverable headroom, but the effect is not well described by a single straight-line fit.

### Mean attention entropy

- Pearson correlation with headroom: `0.271`
- Spearman correlation with headroom: `0.412`

This is stronger than final-token entropy in the pooled analysis. When the sampled attention system is more diffuse at baseline, the prompt tends to have more available improvement under perturbation.

Model-specific behavior is especially notable here:

- `Qwen3.5-9B`: Pearson `0.455`, Spearman `0.537`
- `Olmo-Hybrid-7B`: Pearson `0.510`, Spearman `0.496`

So attention entropy looks like a serious candidate signal for Experiment 1.

## Signals That Look Weak

Not every plausible signal survives contact with the data.

### Baseline target rank

`target_rank` is much noisier than `target_prob`. It does not show a stable relationship with headroom and looks inferior as a control feature.

### Early-minus-late attention entropy tilt

I computed a simple scalar:

- mean entropy of early sampled layers minus mean entropy of late sampled layers

Pooled correlation with headroom:

- Pearson `-0.065`
- Spearman `-0.071`

This is effectively null. The data does show early/late asymmetry in the response surfaces, but this simple baseline scalar does not extract it.

### Layerwise entropy spread

The standard deviation of the baseline layer-mean entropies also looks weak and inconsistent across models.

## The Safe and Dangerous Regions of Profile Space

The sweep does not just show where improvement is possible. It also shows which parts of profile space are typically safe and which are typically destructive.

I grouped the 30 non-baseline profiles into families and aggregated their effects.

### Most destructive families

- `ablation`: mean KL from baseline `9.912`, mean target-prob delta `-0.4914`
- `alternating`: mean KL `9.912`, mean delta `-0.4914`
- `uniform`: mean KL `4.016`, mean delta `-0.2806`
- `middle`: mean KL `2.728`, mean delta `-0.1777`

These families often produce severe collapses in target probability.

### Most promising low-KL families

- `ramp_up`: mean KL `0.237`, mean delta `-0.0034`
- `edges`: mean KL `0.266`, mean delta `-0.0150`
- `late_regional`: mean KL `0.272`, mean delta `-0.0485`
- `ramp_down`: mean KL `0.289`, mean delta `-0.0556`

These families are important because they move the distribution modestly on average yet still contain large individual wins.

For example:

- `edges` has a best single gain of `+0.7026`
- `ramp_up` has a best single gain of `+0.5374`
- `late_regional` has a best single gain of `+0.4978`

This suggests a practical controller should search first within a moderate profile subset rather than across the entire 31-profile battery.

## Damage Is Easier to Detect Than Improvement

The negative signal in the dataset is extremely strong.

By model, mean damage from the worst profile is:

- `Qwen3.5-0.8B`: `0.3875`
- `Qwen3.5-2B`: `0.5266`
- `Qwen3.5-9B`: `0.5793`
- `Olmo-Hybrid-7B`: `0.4722`

Many already-solved prompts can be driven nearly to zero probability.

The most catastrophic cases are usually high-confidence baseline prompts combined with extreme suppression or ablation. Examples include:

- `short5` on all three Qwen models, typically wrecked by `constant_0.25`
- `long0` on multiple models, often wrecked by `early_only_2x` or strong uniform suppression
- highly confident cultural or numerical prompts that collapse under `early_only_2x`

This matters because even a coarse controller could be valuable if it merely avoids the most dangerous profile families.

## Early/Late Asymmetry Is Real, But the Simple Scalar Misses It

Although the baseline early-minus-late entropy scalar is weak, the profile comparisons themselves reveal a robust directional bias.

Across the 88 model-prompt cases:

- `early_high_late_low` beats `late_high_early_low` in `79/88` cases
- `late_suppress_0.5` beats `early_suppress_0.5` in `79/88` cases
- `late_boost_1.5` beats `early_boost_1.5` in `60/88` cases
- `ramp_up` beats `ramp_down` in `47/88` cases

So there is a real asymmetry in the response surfaces:

- reducing late attention is often better than reducing early attention
- mixed profiles with stronger early attention and weaker late attention often outperform the inverse pattern

But that asymmetry is not well predicted by the crude scalar summary extracted from the baseline attention entropy vector. The phenomenon is real; the simple probe is inadequate.

## Cross-Model Consistency

The response surfaces are moderately conserved across models, which is important for the later colony and drone/queen experiments.

Average per-prompt correlation of full 31-profile response vectors:

- `Qwen 2B` vs `Qwen 9B`: `0.733`
- `Qwen 0.8B` vs `Qwen 9B`: `0.669`
- `Qwen 2B` vs `Olmo 7B`: `0.670`
- `Qwen 0.8B` vs `Olmo 7B`: `0.546`

This is high enough to support the claim that there is shared structure, but not so high that one should expect a universal controller with no model-specific calibration.

### Most cross-model-stable prompts

The strongest positive correlations tend to occur on:

- algorithmic prompts such as `short1` and `brief1`
- `long0` code comprehension
- some cultural/factual prompts such as `brief4`
- `short5` syntactic pattern

### Least stable prompts

The weakest and sometimes near-zero or negative correlations occur on:

- `brief3` syntactic pattern
- some factual recall items like `short0` and `short2`
- `med2` reasoning tracking
- `brief2` structural copying for some model pairs

So transfer is plausible, but not uniform by prompt type.

## Which Prompt Types Show the Most Headroom

Mean headroom pooled across models by prompt type:

- `factual_retrieval`: `0.3902`
- `structural_copying`: `0.2917`
- `long_range_retrieval`: `0.1766`
- `factual_recall`: `0.1732`
- `syntactic_pattern`: `0.1311`
- `algorithmic`: `0.0774`
- `reasoning_tracking`: `0.0763`
- `cultural_memorized`: `0.0627`
- `domain_knowledge`: `0.0348`
- `reasoning_numerical`: `0.0312`
- `code_comprehension`: `0.0021`

This ranking says the best near-term evaluation set for signal-guided control is probably not the easiest coding or memorized prompts. The real action is in factual retrieval, structural copying, and long-range retrieval.

## Large-Win Cases

The biggest headroom cases in the dataset include:

- `Qwen3.5-0.8B` `brief3` (`syntactic_pattern`): baseline `0.0168`, headroom `+0.7026`, best profile `edges_low`
- `Qwen3.5-0.8B` `med0` (`factual_retrieval`): baseline `0.1130`, headroom `+0.6928`, best profile `constant_0.75`
- `Olmo-Hybrid-7B` `brief2` (`structural_copying`): baseline `0.1446`, headroom `+0.5374`, best profile `ramp_up`
- `Qwen3.5-9B` `med0` (`factual_retrieval`): baseline `0.2711`, headroom `+0.5175`, best profile `edges_low`
- `Qwen3.5-2B` `brief3` (`syntactic_pattern`): baseline `0.0739`, headroom `+0.4978`, best profile `late_suppress_0.5`

These are exactly the sort of cases a controller should be able to identify if the signals are informative.

## Uniform Scaling Has Structure

Even the simple uniform profiles are not random.

Winning uniform settings by model:

- `Qwen3.5-0.8B`: mostly `constant_0.75` and `constant_1.25`
- `Qwen3.5-2B`: mostly `constant_0.75` and `constant_1.25`
- `Qwen3.5-9B`: strongly concentrated on `constant_1.25`
- `Olmo-Hybrid-7B`: mostly `constant_0.75` and `constant_1.25`, with a few `constant_1.5` or `constant_2`

So even before considering structured profiles, there is a nontrivial signal that mild upscaling or mild downscaling often helps, while stronger uniform suppression is usually catastrophic.

## Practical Implications for Experiment 1

If the goal is to build a single-agent controller from baseline-observable signals, the current data suggests the following.

### Most promising input features

- baseline `target_prob`
- baseline `final_entropy_bits`
- baseline `mean_entropy_bits`
- the full vector of layerwise mean attention entropies
- top-1 vs top-2 logit margin

### Less promising features

- target rank
- a scalar early-minus-late entropy difference
- layer entropy variance alone

### Most promising restricted action space

Rather than searching over all 31 profiles, a first controller should probably choose among a safer subset:

- `ramp_up`
- `ramp_down`
- `edges_high`
- `edges_low`
- moderate `late_*` profiles
- mild uniforms like `constant_0.75` and `constant_1.25`

This restricted family set is attractive because it contains many of the best wins while avoiding the worst catastrophic behaviors.

## Bottom Line

There is definitely signal in the kitchen-sink results.

The strongest simple signals are:

- low baseline confidence
- high final-token entropy
- high mean attention entropy
- small top-logit margin

The most important structural signal is that moderate profile families can produce meaningful gains at very low KL cost, while extreme families are overwhelmingly destructive.

The weakest conclusion one can safely draw is:

- a conservative controller should be able to avoid a large fraction of the catastrophic interventions

The stronger conclusion, which the data also supports, is:

- there is probably enough signal to recover nontrivial headroom on a meaningful subset of prompts, especially if the controller uses attention-entropy features and operates over a restricted safe family of `g_profiles`

The main unresolved issue is not whether signal exists. It does. The unresolved issue is how much of the oracle headroom can be captured by a controller that only sees baseline-accessible features.
