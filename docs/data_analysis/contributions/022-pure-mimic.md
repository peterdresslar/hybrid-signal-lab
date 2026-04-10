# 022-pure-mimic

## Scope

This note summarizes the `pure-mimic` control experiment across three **base-model** pure-transformer subjects:

- `Q3_8B`
- `Q3_30B`
- `OLMO_3`

In this experiment, the balanced cartridge is applied only to the subset of pure-transformer attention layers that mimic the hybrid cadence. Primary source files live under [`data/022-pure-mimic`](/Users/peterdresslar/Workspace/hybrid-signal-lab/data/022-pure-mimic). A small reproducibility helper for the cross-model summary statistics in this note lives in [`022_pure_mimic_summary.py`](/Users/peterdresslar/Workspace/hybrid-signal-lab/docs/data_analysis/code/022_pure_mimic_summary.py).

## Headline

The `pure-mimic` regime turns out to be surprisingly gentle.

- All three models have near-neutral average profile effects compared with the harsher `pure-all` setting.
- Both Qwen pure-transformer models retain productive constant-gain windows.
- OLMO_3 becomes much less punitive on average while still preserving meaningful oracle structure.

So the mimic cadence is not a trivial degraded control. It appears to isolate a lower-cost intervention regime that still carries real prompt-level structure.

## Model-Level Summary

### Q3_8B

- Best overall fixed profile: `constant_1.45` at mean `Δp = +0.02`.
- Best shaped profile: `late_boost_bal_0.30` at `+0.01`.
- Seven of the top 8 fixed profiles are constants.
- Mean constant effect across the cartridge: `-0.0020`.
- Mean shaped-profile effect: `-0.0031`.
- Oracle mean best-profile gain is `0.1112`, with `77.3%` positive oracle headroom.

This is a much calmer regime than `pure-all`: the fixed-profile ceiling is lower, but the average intervention is also far less destructive.

### Q3_30B

- Best overall fixed profile: `constant_1.8` at mean `Δp = +0.07`.
- Best shaped profile: `late_boost_bal_0.45` at `+0.05`.
- Five of the top 8 fixed profiles are constants.
- Mean constant effect across the cartridge: `+0.0027`.
- Mean shaped-profile effect: `-0.0136`.
- Oracle mean best-profile gain is `0.1306`, with `78.7%` positive oracle headroom.

This is the strongest fixed-profile performer in the experiment, and the only model here with a slightly positive average constant profile family.

### OLMO_3

- Best overall fixed profile: `constant_1.3` at mean `Δp = +0.01`.
- Best shaped profile: `pair_stride_bal_0.25` at `+0.01`.
- Four of the top 8 fixed profiles are constants.
- Mean constant effect across the cartridge: `-0.0067`.
- Mean shaped-profile effect: `-0.0014`.
- Oracle mean best-profile gain is `0.067`, with `71.9%` positive oracle headroom.

This is the most surprising result in the experiment. `OLMO_3` is not especially strong here, but it is far less punitive than in `pure-all`, and both constants and shapes remain viable in small doses.

## Underexplored Findings

### 1. Mimic cadence dramatically softens the pure-transformer response surface

The first-order story of this experiment is not peak gain but gentleness.

Average family means:

- `Q3_8B`: constants `-0.0020`, shapes `-0.0031`
- `Q3_30B`: constants `+0.0027`, shapes `-0.0136`
- `OLMO_3`: constants `-0.0067`, shapes `-0.0014`

All three models are clustered close to zero. That is a very different regime from `pure-all`, where constants were often sharply negative. The mimic cadence seems to preserve a substantial amount of control while suppressing much of the collateral damage.

### 2. The pure Qwen pair remains aligned, but less rigidly than in `pure-all`

Average type responsiveness still tracks well between `Q3_8B` and `Q3_30B`, though not as tightly as in the full-stack setting.

Spearman rank correlation of type-level mean response:

- `Q3_30B` vs `Q3_8B`: `0.836`

That is still strong, but weaker than the `0.955` observed in `pure-all`. The mimic cadence keeps the broad family resemblance while introducing more local reordering.

### 3. OLMO_3 is the clearest beneficiary of the mimic restriction

`OLMO_3` is where the mimic setup looks most meaningful rather than merely reduced.

- Best fixed profile becomes mildly positive (`constant_1.3`, `+0.01`).
- Average constants are only slightly negative (`-0.0067`).
- Average shapes are nearly neutral (`-0.0014`).
- Useful type-level niches remain, especially:
  - `spike_p1_bal_0.18` on `reasoning_numerical` at `+0.08`
  - `tent_bal_0.55` on `long_range_retrieval` at `+0.06`
  - `edges_bal_0.70` on `code_comprehension` at `+0.06`

So the mimic cadence is not simply weakening the intervention. For OLMO_3, it appears to improve the tradeoff between selectivity and damage.

### 4. The `bookend_high_bal_0.40` specialist becomes even more dominant here

`bookend_high_bal_0.40` remains a frequent oracle winner in all three models:

- `Q3_8B`: `255` oracle wins
- `Q3_30B`: `221` oracle wins
- `OLMO_3`: `318` oracle wins

And in `OLMO_3` especially, it becomes almost neutral on average (`-0.0`) while still winning on nearly a third of prompts.

Its winning prompts still cluster in the familiar zone:

- `factual_recall`
- `syntactic_pattern`
- `factual_retrieval`
- `domain_knowledge`
- `structural_copying`

This is the strongest evidence so far that `bookend_high_bal_0.40` is capturing a real sparse subspace rather than surviving by accident.

### 5. The mimic regime preserves oracle structure while lowering the ceiling

Oracle mean best-profile gain:

- `Q3_30B`: `0.1306`
- `Q3_8B`: `0.1112`
- `OLMO_3`: `0.0670`

These oracle values are lower than the `pure-all` setting, but still substantial. That is important because it means the mimic restriction is not simply deleting structure. It appears to compress the intervention response surface into a lower-amplitude but potentially more controllable regime.

## Interpretation

The `pure-mimic` setup is not silly at all. It looks like a meaningful operating point between the harsher full-stack pure-transformer regime and the original hybrid setting.

The core properties of this experiment are:

- lower average damage
- lower but still substantial oracle headroom
- persistent Qwen-vs-OLMO structural differences
- strong survival of narrow specialist profiles, especially `bookend_high_bal_0.40`

That combination makes `pure-mimic` conceptually important. It suggests that intervention cadence alone can materially reshape the cost-benefit profile of gain steering, even when the underlying model is unchanged.

## Follow-Up Questions

1. Does the gentler average behavior in `pure-mimic` make it a better target for simple routers than `pure-all`?
2. Is the improvement in OLMO_3 primarily about reduced damage, or does mimic cadence also sharpen specific niches in a usable way?
3. Does the persistent dominance of `bookend_high_bal_0.40` point to a common retrieval/syntax subspace that is especially exposed when intervention is made sparse in depth?
