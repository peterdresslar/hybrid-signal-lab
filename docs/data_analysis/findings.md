# Findings

This document synthesizes the four experiment-specific contribution notes:

- [022-balanced-attention-hybrid.md](/Users/peterdresslar/Workspace/hybrid-signal-lab/docs/data_analysis/contributions/022-balanced-attention-hybrid.md)
- [022-balanced-block-hybrid.md](/Users/peterdresslar/Workspace/hybrid-signal-lab/docs/data_analysis/contributions/022-balanced-block-hybrid.md)
- [022-pure-all.md](/Users/peterdresslar/Workspace/hybrid-signal-lab/docs/data_analysis/contributions/022-pure-all.md)
- [022-pure-mimic.md](/Users/peterdresslar/Workspace/hybrid-signal-lab/docs/data_analysis/contributions/022-pure-mimic.md)

It is intentionally comparative. The detailed per-experiment summaries and local oddities live in the contribution notes.

## Hybrid: attention vs block

The hybrid experiments reveal a clear strategic split between `attention-contribution` and `block intervention`.

### Attention-contribution is the cleaner fixed-profile regime

In the hybrid runs, attention-contribution is the regime where fixed-profile means are most interpretable.

- The dense Qwen pair (`2B`, `9B`) supports a real constant-gain window.
- `35B-MoE` shifts away from that dense-Qwen pattern toward shaped-profile usefulness.
- `OLMO` under attention-contribution is real but comparatively weak and shape-dependent.

The key property of attention-contribution is that fixed-profile averages and family-level task orderings still carry a large amount of signal. This is the setting where “what kind of intervention helps this model?” can often be answered directly from mean behavior.

### Block intervention is harsher but still rich

The block-intervention runs are not weak in an oracle sense. They are weak in an average fixed-profile sense.

- Average constants are substantially negative in all four hybrid models.
- Average shapes are also negative, but much less so.
- Oracle headroom remains large across all four models.

This creates a distinctive block pattern:

- bad global defaults
- meaningful prompt-level upside
- strong incentive toward routing or sparse specialization rather than fixed deployment

So the cleanest contrast is not “attention works, block does not.” It is:

- attention-contribution is the better fixed-policy regime
- block intervention is the better demonstration of latent prompt-level selectivity under hostile averages

### OLMO is the strongest example of the strategy split

Across the hybrid experiments, OLMO behaves differently under the two strategies in a way that is hard to ignore.

- Under attention-contribution, OLMO is comparatively muted and shape-dependent.
- Under block intervention, OLMO becomes more legible and more productive.

That is consistent with the architectural caveat already noted in the hybrid attention writeup: the Qwen and OLMO attention-contribution hooks do not land at identical sub-block locations relative to normalization. In practical terms, the two intervention strategies are not interchangeable for OLMO. If one wants the strongest OLMO story, the block results are the more natural place to look.

## Pure transformers: all vs mimic

The pure-transformer controls reveal that intervention cadence itself is a major experimental axis.

### Pure-all is stronger but costlier

The `pure-all` setting gives the highest pure-transformer ceilings.

- `Q3_30B` is one of the strongest models in the entire dataset under fixed profiles and oracle headroom.
- `Q3_8B` also supports a clear constant-gain regime.
- `OLMO_3` remains selective, but with meaningful niche wins.

The tradeoff is cost:

- average constants are often negative
- shaped profiles are safer but still not uniformly benign
- the response surface is richer, but also harsher

So `pure-all` is the “full power” control: strongest ceilings, clearest pure-Qwen success case, but higher collateral damage.

### Pure-mimic is gentler than expected

The `pure-mimic` setting turns out not to be a toy reduction. It is a materially different operating regime.

- Average profile effects move much closer to zero in all three models.
- Qwen retains useful constant-gain windows.
- `OLMO_3` benefits especially strongly from the mimic restriction, becoming much less punitive on average.

The right interpretation is not that mimic simply weakens intervention. It compresses the response surface:

- lower average damage
- lower but still substantial oracle headroom
- lower-amplitude, potentially more controllable intervention behavior

This makes `pure-mimic` conceptually important. It shows that depth cadence alone can reshape the cost-benefit profile of intervention, even without changing the underlying model family.

### The pure Qwen pair is robust across both control regimes

`Q3_8B` and `Q3_30B` remain strongly aligned in both pure-transformer experiments.

- In `pure-all`, the alignment is extremely strong.
- In `pure-mimic`, the alignment remains strong but loosens somewhat.

That is useful because it suggests two different kinds of stability:

- family-level stability of task ordering
- regime-level modulation of how sharp or costly the intervention surface becomes

In other words, cadence changes the shape of the response surface, but does not erase the deeper similarity between the two Qwen pure-transformer models.

## Overall summary

Across all four experiments, a small number of findings look robust enough to treat as platform-level observations rather than one-off run artifacts.

### 1. Intervention usefulness is real, but it is regime-dependent

No single strategy dominates everywhere.

- Attention-contribution is the cleanest fixed-policy regime for hybrid Qwen models.
- Block intervention is harsh on average but preserves large oracle headroom, especially useful for OLMO and for routing-style questions.
- Pure-all gives the strongest pure-transformer ceilings.
- Pure-mimic gives the gentlest pure-transformer operating point.

This is the strongest argument for treating `hybrid-signal-lab` as a research platform rather than a one-result pipeline. The intervention boundary and the layer-targeting cadence are first-class scientific variables, not implementation details.

### 2. Fixed-profile means and oracle headroom are different objects

This distinction survives every experiment.

- Some models look excellent under fixed-profile means.
- Some models look mediocre or harsh under fixed-profile means but retain large oracle headroom.

So “how good is intervention here?” is always an incomplete question unless one specifies:

- good as a fixed global policy
- or good as a prompt-selective latent regime

That distinction is central to router design, but it is also central to interpretation. The platform repeatedly finds settings where controllability exists without a good default profile.

### 3. Qwen and OLMO are separated by more than family name

The Qwen-vs-OLMO contrast is robust, but it should not be read lazily.

Relevant differences include:

- dense vs MoE status in some comparisons
- pure vs hybrid structure
- normalization placement
- intervention hook location

So cross-model contrasts are informative, but they are never reducible to one variable. This is especially important in attention-contribution analyses, where Qwen and OLMO are only approximately comparable at the sub-block level.

### 4. `bookend_high_bal_0.40` is now a serious object

By this point, `bookend_high_bal_0.40` has appeared too consistently to dismiss.

It is:

- usually poor or neutral as a fixed profile
- repeatedly strong as a prompt-level oracle winner
- persistently concentrated in retrieval / syntax / memorization-adjacent prompt families

That makes it one of the clearest examples in the whole platform of a sparse specialist profile: globally unattractive, locally recurrent, and mechanistically suggestive.

### 5. Router complexity should be matched to regime complexity

The experiments suggest a natural hierarchy:

- some regimes admit simple off/on or low-state routing
- some regimes need richer profile selection
- some regimes may simply be too harsh on average to justify broad deployment at all

This argues against a single universal router story. The right router depends on the response surface:

- simple switch baselines are meaningful where one constant profile already captures much of the benefit
- richer routing is justified where oracle headroom is large but fixed defaults are poor

## Practical takeaways

If one had to choose where to invest next, the experiments suggest four priority directions:

1. Keep treating attention-contribution and block intervention as first-class alternatives in hybrid analysis.
2. Treat `pure-mimic` as a real control regime, not a throwaway simplification.
3. Continue separating fixed-profile conclusions from oracle/routing conclusions.
4. Investigate `bookend_high_bal_0.40` directly as a recurring sparse specialist rather than as a failed global profile.

## Manuscript contributions

The current manuscript introduction proposes four main contributions. In light of the present evidence, they are not all equally supported at the same strength.

### Contribution 1: effective hybrid geometry

This contribution is well supported and should remain first.

The platform now shows that depth-profiled gain vectors are not merely a local trick for one model or one intervention boundary. Across hybrid and pure-transformer controls, changing the effective depth geometry at inference time produces structured and repeatable changes in model behavior.

If anything, this contribution is now stronger than the current wording suggests, because the experiments show that geometry has multiple axes:

- intervention boundary (`attention-contribution` vs `block`)
- layer targeting cadence (`hybrid`, `pure-all`, `pure-mimic`)
- profile shape itself

So the paper is not just introducing a method. It is introducing a manipulable inference-time geometry framework for studying hybrid and hybrid-like architectures.

### Contribution 2: systematic empirical study with task-dependent effects

This contribution is also well supported, but the last clause should be tightened.

The evidence strongly supports:

- measurable task-dependent effects
- substantial variation across models
- substantial variation across intervention strategies
- substantial variation across layer-targeting regimes

What is less cleanly supported in a strong causal sense is the phrase “consistent with mechanistic interpretability predictions about layer-depth specialization,” at least if stated too confidently. The data are certainly compatible with MI-style depth specialization, and in some cases strongly suggestive of it, but the current experiments are still intervention-response studies rather than direct mechanistic identification.

For a fast preprint, I would soften this part slightly:

- good: “consistent with”
- riskier: “demonstrating” or “establishing”

### Contribution 3: separable headroom across task categories, differing by architecture

This contribution is mostly supported, but should be reframed from “task categories” alone to “task categories plus prompt-level structure.”

The experiments clearly show:

- separable headroom across prompt families
- different usable regimes for Qwen, `35B-MoE`, and OLMO
- strong evidence that the architecture/intervention combination matters

But they also show that manual task bins are not the whole story. In several settings, the decisive structure is:

- weaker at the type-average level
- stronger at the prompt-oracle level
- sometimes concentrated in sparse specialist profiles like `bookend_high_bal_0.40`

So the safest and strongest version of this contribution is not:

- “task categories are separable”

but rather:

- “headroom is structured across task families and further refined at the prompt level, with architecture-specific response surfaces”

That is a better match to the actual data.

### Contribution 4: proof-of-concept router from baseline attention head entropy

This contribution is supported, but should now be stated in a more graduated way.

What the evidence supports strongly:

- simple routers can recover meaningful value in some regimes
- router usefulness depends on regime complexity
- baseline-pass signals do carry enough information to support routing decisions

What the evidence does not support uniformly:

- one common router story across all models and all intervention regimes

The strongest router evidence at present is:

- simple bistate routing for Qwen 9B attention
- likely richer routing need for block and some OLMO settings

So I would make this contribution more pragmatic:

- “We demonstrate proof-of-concept routing from single-pass baseline signals, including simple and multiclass policies, and show that routing value depends strongly on the intervention regime.”

That is more accurate than centering only the more ambitious multiclass router.

## What else should be stated up front?

Two additional contributions now look important enough to foreground in the introduction.

### 1. Intervention regime itself is a scientific variable

The experiments strongly support a claim that is larger than any single model result:

- the same nominal gain-profile family behaves very differently under different intervention boundaries and different layer-targeting cadences

This is not just a technical note. It is one of the paper’s real findings. The paper is showing that:

- `attention-contribution` vs `block`
- `hybrid` vs `pure-all` vs `pure-mimic`

materially reshape the response surface. That is a genuine contribution and should probably appear explicitly in the introduction.

### 2. Sparse specialist profiles are real and recurring

The repeated appearance of `bookend_high_bal_0.40` changes the shape of the contribution story.

One of the clearest outputs of the platform is that some profiles are:

- bad global defaults
- recurrent oracle winners
- concentrated in coherent prompt subdomains

That is a useful scientific contribution in its own right. It suggests that intervention analysis should not focus only on top mean profiles; it should also study sparse specialists as mechanistically informative objects.

## Preprint guidance

Given the deadline, the current evidence is strong enough for a preprint if the claims are stated with the right level of ambition.

I would be comfortable foregrounding:

1. effective hybrid geometry as an inference-time intervention framework
2. systematic cross-regime empirical mapping of response surfaces
3. structured headroom that is architecture-specific and refined below the task-category level
4. proof-of-concept routing from baseline-pass signals, including simple switch baselines

I would be more careful about:

1. treating Qwen-vs-Olmo contrasts as purely architectural rather than partly intervention-location effects
2. overcommitting to task categories as the final level of separability
3. presenting the multiclass router as the only or primary routing result

If the goal is to publish what is solid now and move on, the safest high-level framing is:

- this paper introduces an inference-time geometry framework,
- maps its response surfaces across multiple model and intervention regimes,
- and shows that usable routing signal exists, from simple switches up through profile selection, depending on the regime.
