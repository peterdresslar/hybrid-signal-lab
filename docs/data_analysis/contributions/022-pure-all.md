# 022-pure-all

## Scope

This note summarizes the `pure-all` control experiment across three **base-model** pure-transformer subjects:

- `Q3_8B`
- `Q3_30B`
- `OLMO_3`

In this experiment, the balanced cartridge is applied across the full softmax-attention layer stack rather than a hybrid cadence. Primary source files live under [`data/022-pure-all`](/Users/peterdresslar/Workspace/hybrid-signal-lab/data/022-pure-all). A small reproducibility helper for the cross-model summary statistics in this note lives in [`022_pure_all_summary.py`](/Users/peterdresslar/Workspace/hybrid-signal-lab/docs/data_analysis/code/022_pure_all_summary.py).

## Headline

The full-transformer setting is productive, but not uniformly gentle.

- The two Qwen pure-transformer models (`Q3_8B`, `Q3_30B`) support a meaningful constant-gain regime.
- `Q3_30B` is the strongest fixed-profile performer in the experiment.
- `OLMO_3` is harsher on average, but still retains substantial oracle headroom and several distinctive niche wins.

So the `pure-all` control does not flatten the response surface. It produces a workable constant-friendly Qwen regime alongside a more selective OLMO regime.

## Model-Level Summary

### Q3_8B

- Best overall fixed profile: `constant_1.45` at mean `Δp = +0.05`.
- Best shaped profile: `ramp_up_bal_0.35` at `+0.03`.
- Six of the top 8 fixed profiles are constants.
- Mean constant effect across the cartridge: `-0.0453`.
- Mean shaped-profile effect: `-0.0193`.
- Oracle mean best-profile gain is `0.1523`, with `82.1%` positive oracle headroom.

This is a productive but still somewhat costly regime: constants win, but the average constant profile is not safe.

### Q3_30B

- Best overall fixed profile: `constant_1.6` at mean `Δp = +0.08`.
- Best shaped profile: `ramp_up_bal_0.50` at `+0.06`.
- Five of the top 8 fixed profiles are constants.
- Mean constant effect across the cartridge: `-0.0247`.
- Mean shaped-profile effect: `-0.0231`.
- Oracle mean best-profile gain is `0.1924`, with `84.3%` positive oracle headroom.

This is the strongest model in the experiment both as a fixed-profile target and as an oracle target. It supports both a strong constant regime and a genuinely competitive shaped regime.

### OLMO_3

- Best overall fixed profile: `constant_0.95`, effectively `0.00`.
- Best shaped profile: `early_mid_high_bal_0.30`, also effectively `0.00`.
- Mean constant effect across the cartridge: `-0.0740`.
- Mean shaped-profile effect: `-0.0208`.
- Oracle mean best-profile gain is `0.1227`, with `83.5%` positive oracle headroom.

`OLMO_3` is the least hospitable fixed-profile environment here, but it is not empty. The oracle layer remains rich, and some of its strongest type-level niches differ from the Qwen models in revealing ways.

## Underexplored Findings

### 1. The pure-Qwen pair is very tightly aligned

Average type responsiveness is extremely similar between `Q3_8B` and `Q3_30B`.

Spearman rank correlation of type-level mean response:

- `Q3_30B` vs `Q3_8B`: `0.955`

That is the strongest rank-order agreement in the experiment. The pure Qwen pair appears to preserve almost the same task-family ordering even while the stronger model raises both fixed-profile and oracle ceilings.

### 2. `Q3_30B` is the cleanest “full-stack gain” success case

`Q3_30B` has:

- the best fixed profile in the experiment (`constant_1.6`, `+0.08`)
- the best shaped profile in the experiment (`ramp_up_bal_0.50`, `+0.06`)
- the highest oracle mean (`0.1924`)

This matters because it means the full-transformer `pure-all` setting is not merely a harsher version of the hybrid setting. At least in this model, it supports a broad productive intervention window that includes both constants and shapes.

### 3. OLMO_3 has an unusual long-range retrieval niche

The strongest type-level OLMO_3 niche is not one of the usual computational families.

Its best per-type profile for `long_range_retrieval` is:

- `middle_bump_bal_0.50` at `+0.06`

Other OLMO_3 type winners include:

- `spike_p1_bal_0.18` on `reasoning_numerical` at `+0.11`
- `constant_1.8` on `structural_copying` at `+0.05`
- `edges_narrow_bal_0.70` on `algorithmic` at `+0.03`

So even though OLMO_3 looks weak at the fixed-profile average level, its useful regions are not trivial. They are just more fragmented and more profile-specific than the Qwen pure-transformer runs.

### 4. The `bookend_high_bal_0.40` specialist persists in the pure-transformer setting

`bookend_high_bal_0.40` again shows up as a frequent oracle winner despite negative mean performance:

- `Q3_8B`: `189` oracle wins, mean `Δp = -0.04`
- `Q3_30B`: `172` oracle wins, mean `Δp = -0.05`
- `OLMO_3`: `174` oracle wins, mean `Δp = -0.02`

The winning prompts again cluster in:

- `factual_recall`
- `syntactic_pattern`
- `domain_knowledge`
- `factual_retrieval`

This is now a robust recurring phenomenon across hybrid and pure-transformer settings alike. Whatever `bookend_high_bal_0.40` is targeting, it is not a quirk of one model family or one intervention boundary.

### 5. Pure-transformer PCA geometry is flatter than the hybrid runs

The first two baseline-attention PCA components explain:

- `Q3_8B`: `53.4%` + `11.9%`
- `Q3_30B`: `54.1%` + `11.1%`
- `OLMO_3`: `51.4%` + `8.0%`

These values are lower than the hybrid runs, where PC1 often held closer to `60–73%` of the variance. So the pure-transformer baseline entropy geometry appears less concentrated in a single dominant axis. That may matter for later router analysis: the pure-transformer controls may need richer decision boundaries even when their fixed-profile regimes are productive.

## Interpretation

The clearest internal split in this experiment is:

- the pure Qwen pair supports a strong, coherent full-stack intervention regime
- `OLMO_3` retains substantial latent headroom but expresses it less cleanly through fixed profiles

This is not just a magnitude difference. The pure Qwen models share a highly stable type-order structure and a clear constant-friendly productive window. `OLMO_3` instead looks selective and fragmented, with its most interesting gains appearing in narrower per-type or per-prompt niches.

At the same time, the experiment preserves one striking invariant: `bookend_high_bal_0.40` continues to act like a sparse retrieval/memorization specialist even when it is globally bad on average. That makes it increasingly difficult to dismiss as a statistical oddity.

## Follow-Up Questions

1. Does the pure-transformer `mimic` cadence preserve the same strong Qwen alignment seen here in `pure-all`?
2. Is `Q3_30B` strong because full-stack intervention is intrinsically cleaner at that scale, or because its MoE structure creates more accessible prompt-level specialization?
3. Can OLMO_3’s long-range retrieval niche be isolated into a small router-ready profile set, or is it too sparse to exploit reliably?
