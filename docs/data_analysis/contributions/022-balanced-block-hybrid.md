# 022-balanced-block-hybrid

## Scope

This note summarizes the balanced block-intervention sweep across four hybrid **base-model** subjects:

- `2B`
- `9B`
- `35B` (MoE)
- `OLMO`

Primary source files live under [`data/022-balanced-block-hybrid`](/Users/peterdresslar/Workspace/hybrid-signal-lab/data/022-balanced-block-hybrid). A small reproducibility helper for the cross-model summary statistics in this note lives in [`022_balanced_block_hybrid_summary.py`](/Users/peterdresslar/Workspace/hybrid-signal-lab/docs/data_analysis/code/022_balanced_block_hybrid_summary.py).

## Headline

This experiment presents a severe fixed-profile environment but a surprisingly rich oracle environment.

- At the fixed-profile level, almost everything is weak, and most constant profiles are actively damaging on average.
- At the oracle level, all four models still retain substantial prompt-level headroom.

So the main property of block intervention in this sweep is not broad usability. It is high prompt-level variance under a generally harsh average regime.

## Model-Level Summary

### 2B

- Best overall fixed profile: `edges_asym_late_bal_0.25` at mean `Δp = +0.01`.
- Best constant profile: `constant_0.85` at roughly `0.00`.
- Mean constant effect across the cartridge: `-0.0947`.
- Mean shaped-profile effect: `-0.0188`.
- Oracle mean best-profile gain is `0.126`, with `77.0%` of prompts showing positive oracle headroom.

The 2B model is the least punitive block case on average, but even here the fixed-profile regime is narrow and fragile.

### 9B

- Best overall fixed profile: `constant_1.15` at mean `Δp = +0.01`.
- Best shaped profile: `edges_asym_early_bal_0.25`, effectively neutral.
- Mean constant effect across the cartridge: `-0.1027`.
- Mean shaped-profile effect: `-0.0232`.
- Oracle mean best-profile gain is `0.1524`, with `83.2%` positive oracle headroom.

This is the clearest example of the experiment’s central tension: the deployable fixed-profile regime is weak, but the prompt-level oracle regime remains rich.

### 35B-MoE

- Best overall fixed profile: `constant_0.85`, effectively `0.00`.
- Best shaped profile: `early_boost_bal_0.15`, also effectively `0.00`.
- Mean constant effect across the cartridge: `-0.1473`, the worst of the four models.
- Mean shaped-profile effect: `-0.0235`.
- Oracle mean best-profile gain is `0.1823`, the highest in the experiment, with `86.8%` positive oracle headroom.

The `35B-MoE` model is the most extreme fixed-vs-oracle split in this experiment: almost unusable by average fixed profile, but highly controllable in principle at the prompt level.

### OLMO

- Best overall fixed profile: `late_boost_bal_0.15` at mean `Δp = +0.01`.
- Best constant profile: `constant_0.95`, effectively `0.00`.
- Mean constant effect across the cartridge: `-0.1043`.
- Mean shaped-profile effect: `-0.0251`.
- Oracle mean best-profile gain is `0.1396`, with `83.8%` positive oracle headroom.

Unlike the attention-contribution experiment, the block setting gives OLMO a genuinely productive shaped-profile niche without making the overall fixed-profile regime gentle.

## Underexplored Findings

### 1. Block intervention is much harsher on constants than on shapes

In all four models, the average constant profile is substantially more negative than the average shaped profile:

- `2B`: constants `-0.0947`, shapes `-0.0188`
- `9B`: constants `-0.1027`, shapes `-0.0232`
- `35B-MoE`: constants `-0.1473`, shapes `-0.0235`
- `OLMO`: constants `-0.1043`, shapes `-0.0251`

This is one of the cleanest cross-model regularities in the experiment. Whatever productive structure exists under block intervention is disproportionately carried by shaped profiles, not by broad constant scaling.

### 2. Oracle headroom stays high even when fixed-profile means collapse

The most important quantitative separation in this experiment is between fixed-profile averages and oracle headroom:

- `35B-MoE`: best fixed profile `≈ 0.00`, oracle mean `0.1823`
- `9B`: best fixed profile `+0.01`, oracle mean `0.1524`
- `OLMO`: best fixed profile `+0.01`, oracle mean `0.1396`
- `2B`: best fixed profile `+0.01`, oracle mean `0.126`

This means block intervention is not “weak.” It is selective. The problem is not lack of latent controllability; the problem is that average fixed interventions fail to access it cleanly.

### 3. The `bookend_high_bal_0.40` specialist persists here too

`bookend_high_bal_0.40` remains a recurring oracle winner despite negative mean performance in every model:

- `2B`: `258` oracle wins, mean `Δp = -0.03`
- `35B-MoE`: `147` oracle wins, mean `Δp = -0.02`
- `9B`: `183` oracle wins, mean `Δp = -0.02`
- `OLMO`: `179` oracle wins, mean `Δp = -0.03`

Its winning prompts again cluster heavily in:

- `factual_recall`
- `syntactic_pattern`
- `domain_knowledge`
- `factual_retrieval`

So this profile is not an artifact of one intervention mode. In both average-harsh and oracle-rich settings, it repeatedly behaves like a narrow retrieval/memorization specialist.

### 4. Type-order structure remains, but it is weaker and less family-coherent

Average type responsiveness is still structured, but much less cleanly than in the attention experiment.

Spearman rank correlations of type-level mean response:

- `35B-MoE` vs `9B`: `0.909`
- `2B` vs `35B-MoE`: `0.818`
- `2B` vs `9B`: `0.691`
- `2B` vs `OLMO`: `0.601`
- `35B-MoE` vs `OLMO`: `0.410`
- `9B` vs `OLMO`: `0.342`

That is still real structure, but it is looser. Block intervention seems to preserve broad task-order regularities while adding more model-specific noise and more destructive tails.

### 5. PCA concentration is largely unchanged despite the harsher response surface

The first two baseline-attention PCA components explain:

- `2B`: `71.6%` + `8.9%`
- `9B`: `72.7%` + `6.1%`
- `35B-MoE`: `69.6%` + `5.0%`
- `OLMO`: `59.4%` + `10.3%`

These are nearly identical to the attention-contribution experiment. So the harsher fixed-profile behavior here is not mirrored by a radically different baseline entropy geometry. The raw sensing geometry appears relatively stable across intervention strategies, while the downstream gain response becomes much more punitive under block intervention.

## Interpretation

The central property of this experiment is a fixed-vs-oracle mismatch:

- Fixed-profile means are weak and often negative.
- Prompt-level oracle headroom is still large.
- Shaped profiles are systematically less damaging than constants.

That makes block intervention look less attractive as a global fixed policy, but still very relevant as a routing or oracle regime. The data say that useful block interventions exist. They are simply buried inside a response surface where the average intervention is unusually costly.

Because these are base models, the pattern is plausibly tied to the pretrained residual dynamics themselves rather than to instruction-tuning artifacts. In that sense, this experiment may be telling us something fairly direct about the fragility of broad residual scaling and the relative safety of more localized block geometry.

## Follow-Up Questions

1. Can a simple router recover meaningful value from the large block oracle headroom without inheriting too much of the destructive average regime?
2. Is the relative safety of shaped profiles under block intervention preserved in the pure-transformer controls?
3. Does the persistent `bookend_high_bal_0.40` specialization indicate a genuine retrieval/memorization subspace that is robust across intervention boundaries?
