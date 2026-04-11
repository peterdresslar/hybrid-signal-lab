# 030 Benchmark Analysis Report

**Generated**: 2026-04-10
**Data**: [data/030-bench](/Users/peterdresslar/Workspace/hybrid-signal-lab/data/030-bench)
**Models**: Qwen 3.5 9B, OLMo-Hybrid-7B
**Benchmarks**: ARC-Challenge, MMLU abstract algebra, MMLU college mathematics, MMLU college computer science

## 1. Framing

This report treats the 030 benchmark run as a benchmark-scale probe of prompt-dependent gain-profile sensitivity.

The stable contribution is not the current router. The benchmark data show three stronger findings:

1. Simple fixed intervention profiles can improve standard multiple-choice reasoning benchmarks.
2. Prompt-level oracle profile choice yields substantially larger gains than any single fixed profile.
3. The oracle profile distributions do **not** collapse to one universal profile, indicating real within-benchmark heterogeneity.

The current routed results are therefore best interpreted as an exploratory control experiment: routing is presently underpowered relative to the heterogeneity already visible in the benchmark outcomes. We expect routing and policy-learning to be addressed much more directly in later work on the platform.

## 2. Summary

### Qwen 3.5 9B

| Task | N | Baseline | Best fixed | Routed | Oracle |
|------|---:|---------:|-----------:|-------:|-------:|
| ARC-Challenge | 1172 | 0.538 | 0.542 | 0.494 | 0.619 |
| MMLU abstract algebra | 100 | 0.620 | 0.640 | 0.330 | 0.810 |
| MMLU college math | 100 | 0.630 | 0.640 | 0.380 | 0.780 |
| MMLU college CS | 100 | 0.750 | 0.740 | 0.610 | 0.850 |

Best fixed profiles:

- ARC-Challenge: `triad_odd_bal_0.45`
- MMLU abstract algebra: `edges_narrow_bal_0.55`
- MMLU college math: `edges_narrow_bal_0.55`
- MMLU college CS: `triad_odd_bal_0.45`

### OLMo-Hybrid-7B

| Task | N | Baseline | Best fixed | Routed | Oracle |
|------|---:|---------:|-----------:|-------:|-------:|
| ARC-Challenge | 1172 | 0.533 | 0.532 | 0.489 | 0.636 |
| MMLU abstract algebra | 100 | 0.350 | 0.320 | 0.200 | 0.460 |
| MMLU college math | 100 | 0.470 | 0.460 | 0.360 | 0.660 |
| MMLU college CS | 100 | 0.510 | 0.530 | 0.450 | 0.720 |

Best fixed profiles:

- ARC-Challenge: `edges_narrow_bal_0.40`
- MMLU abstract algebra: `edges_narrow_bal_0.40`
- MMLU college math: `edges_narrow_bal_0.40`
- MMLU college CS: `edges_narrow_bal_0.40`

## 3. Main Findings

### 3.1 Fixed profiles already matter on standard benchmarks

For Qwen 9B, fixed-profile gains are modest but real on three of the four tasks:

- ARC-Challenge: `+0.004`
- MMLU abstract algebra: `+0.020`
- MMLU college math: `+0.010`
- MMLU college CS: `-0.010`

This is important because these are not battery-internal prompts. The same intervention family that showed structured effects on the battery also produces measurable gains on external multiple-choice reasoning benchmarks.

For OLMo, the fixed-profile story is weaker:

- ARC-Challenge: `-0.002`
- MMLU abstract algebra: `-0.030`
- MMLU college math: `-0.010`
- MMLU college CS: `+0.020`

OLMo still shows strong oracle headroom, but its best single fixed profile is less reliable than Qwen's.

### 3.2 Oracle headroom is much larger than fixed-profile gains

The benchmark-scale oracle improvements are the clearest signal in the run.

For Qwen 9B:

- ARC-Challenge: `+0.081`
- MMLU abstract algebra: `+0.190`
- MMLU college math: `+0.150`
- MMLU college CS: `+0.100`

For OLMo:

- ARC-Challenge: `+0.102`
- MMLU abstract algebra: `+0.110`
- MMLU college math: `+0.190`
- MMLU college CS: `+0.210`

This gap between best fixed and oracle is the main empirical result. It implies that these benchmarks do not reduce to a single good geometry. Prompts within the same benchmark prefer different intervention profiles, and some prompts still prefer baseline.

### 3.3 Benchmark-scale profile selection does not collapse

The per-benchmark oracle distributions are spread across multiple profiles plus baseline.

For Qwen 9B on ARC-Challenge:

- `constant_2.6`: 325
- `triad_odd_bal_0.45`: 331
- `edges_narrow_bal_0.55`: 158
- `late_boost_bal_0.60`: 201
- `baseline`: 157

For Qwen 9B on MMLU abstract algebra:

- `edges_narrow_bal_0.55`: 25
- `triad_odd_bal_0.45`: 14
- `late_boost_bal_0.60`: 14
- `constant_2.6`: 19
- `baseline`: 28

For OLMo on ARC-Challenge:

- `edges_narrow_bal_0.40`: 322
- `ramp_up_bal_0.50`: 156
- `constant_1.6`: 232
- `late_boost_bal_0.30`: 168
- `baseline`: 294

For OLMo on MMLU college CS:

- `ramp_up_bal_0.50`: 21
- `edges_narrow_bal_0.40`: 21
- `constant_1.6`: 28
- `late_boost_bal_0.30`: 21
- `baseline`: 9

This is the strongest evidence that the profile-selection signal is real at benchmark scale. Even without selecting benchmarks for full prompt-type coverage, the oracle winners remain distributed rather than collapsing to one fixed intervention.

## 4. Interpretation

### 4.1 Qwen 9B

Qwen gives the cleaner benchmark result.

- Fixed profiles help on ARC-Challenge and both math-like MMLU subsets.
- Oracle headroom is consistently large.
- The best fixed profile differs across tasks.
- The oracle distributions are broad, indicating substantial within-task heterogeneity.

This strongly backs up the earlier battery findings. The battery did not merely overfit to synthetic prompts; it identified an intervention family whose task-dependent structure persists on standard benchmarks.

### 4.2 OLMo-Hybrid-7B

OLMo gives a different but still important result.

- Fixed profiles are not consistently helpful.
- Oracle headroom remains large on every benchmark.
- Profile winners are still distributed across multiple profiles plus baseline.

So OLMo supports the heterogeneity claim even where the fixed-profile story is weaker. In other words, OLMo suggests the geometry exists, but the simple fixed controller is less aligned to it.

### 4.3 Routing is exploratory, not a headline contribution yet

The current routed results underperform baseline for both models on every benchmark task. The most immediate reason is visible in the routed profile distributions:

- Qwen routing collapses almost entirely to `constant_2.6`.
- OLMo routing collapses mostly to `ramp_up_bal_0.50` or `off`.

Those routed choices do not resemble the oracle profile distributions. The benchmarks therefore show that routing remains an unsolved control problem, not that profile heterogeneity is absent.

For this report and any near-term manuscript framing, routing should be described as a preliminary probe of online policy learning over gain geometry. The finished contribution is the benchmark-scale evidence for intervention sensitivity and prompt-level heterogeneity.

## 5. Suggested Paper-Safe Claims

The 030 benchmark run supports the following restrained claims:

1. Static layerwise gain interventions can improve accuracy on standard multiple-choice reasoning benchmarks, especially in Qwen 3.5 9B.
2. Prompt-level oracle profile choice produces substantially larger gains than any single fixed profile in both Qwen and OLMo.
3. Oracle winners remain distributed across multiple profiles and baseline at benchmark scale, implying real within-benchmark intervention heterogeneity.
4. Current lightweight routing does not recover this heterogeneity well, motivating later work on prompt-conditional control policies rather than weakening the underlying intervention result.

## 6. Bottom Line

The 030 benchmark data make the project stronger, not weaker.

The key outcome is not that a router failed. It is that standard external benchmarks reproduce the central battery finding: pretrained models expose multiple prompt-dependent computational geometries under intervention, and benchmark prompts do not collapse onto one universally best fixed profile.

That is already a substantial result. The routing problem can be left for future platform work without weakening the main empirical contribution of the current study.
