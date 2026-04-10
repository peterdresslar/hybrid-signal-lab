# 022-balanced-attention-hybrid

## Scope

This note summarizes the balanced attention-contribution sweep across four hybrid **base-model** subjects:

- `2B`
- `9B`
- `35B` (MoE)
- `OLMO`

Primary source files live under [`data/022-balanced-attention-hybrid`](/Users/peterdresslar/Workspace/hybrid-signal-lab/data/022-balanced-attention-hybrid). A small reproducibility helper for the cross-model summary statistics in this note lives in [`022_balanced_attention_hybrid_summary.py`](/Users/peterdresslar/Workspace/hybrid-signal-lab/docs/data_analysis/code/022_balanced_attention_hybrid_summary.py).

## Headline

Under attention-contribution, this experiment separates into three regimes:

- The dense Qwen pair (`2B`, `9B`) has a productive constant-gain regime.
- The `35B-MoE` variant departs from that dense-Qwen pattern and shifts toward shaped-profile regimes.
- `OLMO` also shifts toward shaped-profile regimes, though with smaller absolute oracle headroom.

So the dominant intervention mechanism differs not just by family, but also by internal model topology: some models reward broad dose-like amplification, while others reward profile geometry more directly.

## Model-Level Summary

### 2B

- Best overall fixed profile: `constant_2` at mean `Δp = +0.05`.
- Best shaped profile: `triad_even_bal_0.45` at `+0.02`.
- All top 8 fixed profiles by mean `Δp` are constants.
- Oracle mean best-profile gain is `0.167`, with `69.5%` of prompts having positive oracle headroom.

This is the clearest small-model instance of the dense-Qwen constant-gain regime.

### 9B

- Best overall fixed profile: `constant_2.3` at mean `Δp = +0.06`.
- Best shaped profile: `edges_asym_late_bal_0.25` at `+0.01`.
- All top 8 fixed profiles by mean `Δp` are constants.
- Oracle mean best-profile gain is `0.1378`, with `76.4%` positive oracle headroom.

This is the strongest larger dense-model expression of the same constant-friendly Qwen response surface.

### 35B

- Best overall fixed profile: `bowl_bal_0.40` at mean `Δp = +0.02`.
- Best constant profile: `constant_1.15` at `+0.01`.
- None of the top 8 fixed profiles are constants.
- Oracle mean best-profile gain is `0.1838`, the highest among the four models here.

The `35B-MoE` run is the sharpest departure from the dense-Qwen pattern. It keeps large oracle headroom, but the usable fixed-profile regime is no longer a constant-amplification story.

### OLMO

- Best overall fixed profile: `mid_late_peak_bal_0.50` at mean `Δp = +0.02`.
- Best constant profile is effectively neutral: `constant_0.85` at roughly `0.00`.
- None of the top 8 fixed profiles are constants.
- Oracle mean best-profile gain is `0.0729`, the smallest absolute oracle among the four models.

This is the clearest indication in this experiment that, for OLMO under attention-contribution, constants are the wrong search space.

## Underexplored Findings

### 1. The “bookend” paradox is real across all four models

`bookend_high_bal_0.40` is a poor fixed profile everywhere, but an extremely frequent oracle winner everywhere.

- `2B`: mean `Δp = -0.02`, but `336` prompt-level oracle wins
- `35B`: mean `Δp = -0.01`, but `248` oracle wins
- `9B`: mean `Δp = -0.01`, but `254` oracle wins
- `OLMO`: mean `Δp = -0.02`, but `271` oracle wins

The wins are not random. They cluster heavily in:

- `factual_recall`
- `syntactic_pattern`
- `domain_knowledge`
- `factual_retrieval`

Interpretation: this profile is not a good deployment default, but it is a recurring niche specialist. That makes it more important for oracle and router analysis than fixed-profile averages would suggest.

### 2. The dense Qwen pair has a remarkably stable type ordering

Averaging `Δp` across the full balanced profile set by prompt type, the rank ordering of type responsiveness is almost unchanged across the dense Qwen pair.

Spearman rank correlation of type-level mean response:

- `2B` vs `9B`: `0.943`

The `35B-MoE` variant remains close in ordering, but it should not be treated as just a larger dense-Qwen model:

- `2B` vs `35B`: `0.980`
- `35B` vs `9B`: `0.964`

By contrast, OLMO diverges sharply:

- `2B` vs `OLMO`: `0.419`
- `35B` vs `OLMO`: `0.382`
- `9B` vs `OLMO`: `0.327`

This suggests a strong invariance inside the dense Qwen pair: attention intervention changes magnitude with scale, but not much of the task-family ordering. The `35B-MoE` model remains nearby in ranking space while still departing strongly in which profile families are actually usable as fixed interventions.

### 3. Oracle headroom is not tightly coupled to fixed-profile quality

The fixed-profile story and the oracle story separate sharply.

- `35B` has the highest oracle mean (`0.1838`) even though its best fixed profile is only `+0.02`.
- `2B` has the second-highest oracle mean (`0.167`) despite looking much simpler at the fixed-profile level.
- `9B`, which looks best at the fixed-profile level, has less oracle headroom than either `2B` or `35B`.

Interpretation: fixed-profile averages are telling us about deployable defaults. Oracle headroom is telling us about latent controllability. Those are not the same quantity, and this experiment makes the distinction unusually sharp.

### 4. PCA concentration itself is family-informative

The first two baseline-attention PCA components explain:

- `2B`: `71.6%` + `8.9%`
- `9B`: `72.7%` + `6.1%`
- `35B`: `69.6%` + `5.0%`
- `OLMO`: `59.4%` + `10.3%`

The dense-Qwen runs concentrate much more of the baseline entropy geometry into PC1. OLMO spreads more variance into PC2. The `35B-MoE` model sits between those regimes. That may matter later for router complexity: the dense Qwen pair may naturally admit lower-dimensional routing decisions than either OLMO or the MoE variant under attention-contribution.

## Interpretation

This experiment does not just say “constants help Qwen and shapes help OLMO.” It says something more structured:

- The experiment exposes a structural split between the dense Qwen pair, the `35B-MoE` comparison model, and OLMO.
- Prompt-level oracle structure remains rich even where fixed-profile means collapse.
- Certain profiles, especially `bookend_high_bal_0.40`, are better understood as sparse specialists than as candidates for global deployment.

Because all four runs use base models rather than instruction-tuned descendants, this comparison is relatively clean: the response surfaces are less likely to be driven by downstream alignment artifacts and more likely to reflect representational and architectural differences in the pretrained backbones.

One additional caution matters for any direct Qwen-vs-Olmo comparison under attention-contribution. The intervention is not implemented at an identical sub-block location in the two architectures. Qwen and Olmo differ in normalization placement, so the gain enters the residual pathway at different points relative to normalization. As a result, cross-model differences here should not be read as purely representational. Some part of the contrast likely reflects the altered gain pathway induced by pre-norm versus post-norm organization itself.

That last point matters for routing. A profile can be globally bad and still be mechanistically important if it repeatedly wins in the same narrow prompt neighborhoods.

## Follow-Up Questions

1. Does `bookend_high_bal_0.40` remain a recurring oracle specialist under block intervention, or is it specific to attention-contribution?
2. Does the strong type-order stability within the Qwen family survive in the pure-transformer `pure-all` and `pure-mimic` controls?
3. Is the larger oracle headroom in `35B` evidence of genuinely richer controllability, or just wider prompt-level variance with lower fixed-profile robustness?
