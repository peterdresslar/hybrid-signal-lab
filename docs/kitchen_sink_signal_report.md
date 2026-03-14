# Kitchen Sink Signal Report

**Generated**: 2026-03-14
**Data**: `results/kitchen_sink/` — 4 models × 22 prompts × 31 g-profiles = 2,728 observations

## 1. Dataset Overview

### Models

| Key | Model | Layers | Attn Slots | Heads | Vocab |
|-----|-------|--------|-----------|-------|-------|
| 0_8B | Qwen/Qwen3.5-0.8B-Base | 24 | 6 | 8 | 248,320 |
| 2B | Qwen/Qwen3.5-2B-Base | 24 | 6 | 8 | 248,320 |
| 9B | Qwen/Qwen3.5-9B-Base | 32 | 8 | 16 | 248,320 |
| OLMO | allenai/Olmo-Hybrid-7B | 32 | 8 | 30 | 100,352 |

### Prompt Battery

| Tier | IDs | Count | Types |
|------|-----|-------|-------|
| short | short0, short1, short2, short3, short4, short5 | 6 | algorithmic, cultural_memorized, factual_recall, structural_copying, syntactic_pattern |
| brief | brief0, brief1, brief2, brief3, brief4, brief5 | 6 | algorithmic, cultural_memorized, factual_recall, structural_copying, syntactic_pattern |
| med | med0, med1, med2, med3, med4, med5 | 6 | factual_retrieval, reasoning_numerical, reasoning_tracking, syntactic_pattern |
| long | long0, long1, long2, long3 | 4 | code_comprehension, domain_knowledge, long_range_retrieval |

### g-Profile Battery

31 profiles spanning:

- **Uniform scalars** (7): constant values from 0.25 to 2.0
- **Regional boost/suppress** (8): early or late layers at 0.5/0.7/1.3/1.5
- **Middle profiles** (3): middle bump/suppress, middle+edge combo
- **Ramps** (4): monotonic increase/decrease across layers (steep and gentle)
- **Edge profiles** (2): U-shape and inverted-U across layers
- **Cross patterns** (2): early-high/late-low and inverse
- **Extreme/ablation** (5): early-only-2x, late-only-2x, middle-only, alternating on/off

## 2. Headroom: How Much Improvement Exists?

For each model × prompt, we compare the best non-baseline g-profile against the baseline (g=1.0).

### Qwen3.5-0.8B

Improved 21/22 prompts. Mean Δprob = +0.1447, max Δprob = +0.7026.

| Prompt | Type | BL prob | Best prob | Δ | BL rank → Best | Best profile |
|--------|------|---------|-----------|---|---------------|-------------|
| brief0 | factual_recall | 0.1768 | 0.3127 | +0.1358 | 2→2 | early_boost_1.3 |
| brief1 | algorithmic | 0.9172 | 0.9400 | +0.0228 | 1→1 | edges_high |
| brief2 | structural_copying | 0.3426 | 0.4537 | +0.1110 | 1→1 | late_boost_1.5 |
| brief3 | syntactic_pattern | 0.0168 | 0.7194 | +0.7026 | 6→1 | edges_low |
| brief4 | cultural_memorized | 0.9688 | 0.9780 | +0.0092 | 1→1 | late_suppress_0.7 |
| brief5 | factual_recall | 0.8655 | 0.9941 | +0.1286 | 1→1 | edges_low |
| long0 | code_comprehension | 0.9935 | 0.9982 | +0.0047 | 1→1 | late_boost_1.5 |
| long1 | long_range_retrieval | 0.0180 | 0.4385 | +0.4205 | 5→1 | ramp_up |
| long2 | domain_knowledge | 0.0098 | 0.0305 | +0.0208 | 13→4 | middle_suppress_0.5 |
| long3 | long_range_retrieval | 0.5039 | 0.6491 | +0.1452 | 1→1 | constant_0.75 |
| med0 | factual_retrieval | 0.1130 | 0.8058 | +0.6928 | 3→1 | constant_0.75 |
| med1 | reasoning_numerical | 0.0144 | 0.0157 | +0.0013 | 6→8 | late_boost_1.3 |
| med2 | reasoning_tracking | 0.0048 | 0.0246 | +0.0198 | 8→5 | middle_bump_1.5_edges_0.8 |
| med3 | reasoning_numerical | 0.9518 | 0.9831 | +0.0312 | 1→1 | late_boost_1.5 |
| med4 | syntactic_pattern | 0.0056 | 0.0544 | +0.0488 | 22→4 | middle_suppress_0.5 |
| med5 | reasoning_tracking | 0.0561 | 0.0629 | +0.0067 | 5→4 | ramp_down |
| short0 | factual_recall | 0.0444 | 0.1418 | +0.0974 | 7→2 | constant_1.25 |
| short1 | algorithmic | 0.8718 | 0.9945 | +0.1227 | 1→1 | ramp_up |
| short2 | factual_recall | 0.1170 | 0.3810 | +0.2640 | 1→1 | late_suppress_0.7 |
| short3 | structural_copying | 0.5296 | 0.7054 | +0.1758 | 1→1 | middle_suppress_0.5 |
| short4 | cultural_memorized | 0.0061 | 0.0264 | +0.0203 | 21→5 | constant_1.25 |
| short5 | syntactic_pattern | 0.9965 | 0.9971 | +0.0005 | 1→1 | ramp_down_gentle |

### Qwen3.5-2B

Improved 20/22 prompts. Mean Δprob = +0.1151, max Δprob = +0.4978.

| Prompt | Type | BL prob | Best prob | Δ | BL rank → Best | Best profile |
|--------|------|---------|-----------|---|---------------|-------------|
| brief0 | factual_recall | 0.5922 | 0.7263 | +0.1341 | 1→1 | ramp_up |
| brief1 | algorithmic | 0.9445 | 0.9734 | +0.0289 | 1→1 | early_boost_1.3 |
| brief2 | structural_copying | 0.1172 | 0.3934 | +0.2761 | 3→1 | late_high_early_low |
| brief3 | syntactic_pattern | 0.0739 | 0.5717 | +0.4978 | 2→1 | late_suppress_0.5 |
| brief4 | cultural_memorized | 0.9863 | 0.9926 | +0.0064 | 1→1 | edges_high |
| brief5 | factual_recall | 0.8700 | 0.9519 | +0.0820 | 1→1 | early_suppress_0.7 |
| long0 | code_comprehension | 0.9983 | 0.9995 | +0.0012 | 1→1 | late_boost_1.5 |
| long1 | long_range_retrieval | 0.7201 | 0.8867 | +0.1666 | 1→1 | ramp_up_gentle |
| long2 | domain_knowledge | 0.0334 | 0.0403 | +0.0069 | 4→3 | middle_bump_1.5 |
| long3 | long_range_retrieval | 0.7245 | 0.8815 | +0.1570 | 1→1 | late_suppress_0.5 |
| med0 | factual_retrieval | 0.7688 | 0.9901 | +0.2213 | 1→1 | ramp_up |
| med1 | reasoning_numerical | 0.0049 | 0.0187 | +0.0138 | 21→9 | early_suppress_0.7 |
| med2 | reasoning_tracking | 0.0552 | 0.1072 | +0.0520 | 3→3 | late_suppress_0.5 |
| med3 | reasoning_numerical | 0.9795 | 0.9958 | +0.0163 | 1→1 | late_boost_1.5 |
| med4 | syntactic_pattern | 0.4807 | 0.4778 | -0.0029 | 1→1 | late_suppress_0.7 |
| med5 | reasoning_tracking | 0.7132 | 0.7989 | +0.0857 | 1→1 | ramp_up_gentle |
| short0 | factual_recall | 0.0597 | 0.1161 | +0.0564 | 5→1 | constant_0.75 |
| short1 | algorithmic | 0.7592 | 0.9430 | +0.1838 | 1→1 | constant_0.5 |
| short2 | factual_recall | 0.1465 | 0.2295 | +0.0830 | 2→2 | early_suppress_0.7 |
| short3 | structural_copying | 0.5325 | 0.7295 | +0.1970 | 1→1 | late_boost_1.5 |
| short4 | cultural_memorized | 0.0250 | 0.2939 | +0.2689 | 5→1 | constant_1.5 |
| short5 | syntactic_pattern | 0.9988 | 0.9988 | -0.0000 | 1→1 | edges_high |

### Qwen3.5-9B

Improved 20/22 prompts. Mean Δprob = +0.1476, max Δprob = +0.5175.

| Prompt | Type | BL prob | Best prob | Δ | BL rank → Best | Best profile |
|--------|------|---------|-----------|---|---------------|-------------|
| brief0 | factual_recall | 0.3411 | 0.7819 | +0.4408 | 2→1 | ramp_up |
| brief1 | algorithmic | 0.9514 | 0.9901 | +0.0387 | 1→1 | constant_1.25 |
| brief2 | structural_copying | 0.2064 | 0.6337 | +0.4273 | 2→1 | constant_1.25 |
| brief3 | syntactic_pattern | 0.0010 | 0.0171 | +0.0161 | 9→2 | ramp_up |
| brief4 | cultural_memorized | 0.9987 | 0.9985 | -0.0002 | 1→1 | ramp_down_gentle |
| brief5 | factual_recall | 0.8212 | 0.8307 | +0.0095 | 1→1 | edges_low |
| long0 | code_comprehension | 0.9966 | 0.9993 | +0.0027 | 1→1 | late_boost_1.5 |
| long1 | long_range_retrieval | 0.8098 | 0.9834 | +0.1736 | 1→1 | ramp_up |
| long2 | domain_knowledge | 0.0220 | 0.0501 | +0.0281 | 8→5 | constant_1.5 |
| long3 | long_range_retrieval | 0.7553 | 0.9537 | +0.1984 | 1→1 | constant_1.25 |
| med0 | factual_retrieval | 0.2711 | 0.7886 | +0.5175 | 1→1 | edges_low |
| med1 | reasoning_numerical | 0.0544 | 0.0937 | +0.0393 | 4→2 | late_boost_1.3 |
| med2 | reasoning_tracking | 0.1840 | 0.5375 | +0.3535 | 2→1 | ramp_down_gentle |
| med3 | reasoning_numerical | 0.9778 | 0.9966 | +0.0188 | 1→1 | late_boost_1.5 |
| med4 | syntactic_pattern | 0.5414 | 0.7113 | +0.1699 | 1→1 | late_boost_1.5 |
| med5 | reasoning_tracking | 0.9894 | 0.9916 | +0.0023 | 1→1 | late_boost_1.5 |
| short0 | factual_recall | 0.2608 | 0.3355 | +0.0747 | 1→1 | late_boost_1.5 |
| short1 | algorithmic | 0.8823 | 0.9601 | +0.0778 | 1→1 | late_boost_1.5 |
| short2 | factual_recall | 0.3232 | 0.5646 | +0.2414 | 1→1 | early_boost_1.5 |
| short3 | structural_copying | 0.6090 | 0.9085 | +0.2996 | 1→1 | ramp_up |
| short4 | cultural_memorized | 0.7494 | 0.8671 | +0.1177 | 1→1 | edges_high |
| short5 | syntactic_pattern | 0.9995 | 0.9994 | -0.0002 | 1→1 | ramp_down_gentle |

### OLMo-Hybrid-7B

Improved 18/22 prompts. Mean Δprob = +0.1151, max Δprob = +0.5374.

| Prompt | Type | BL prob | Best prob | Δ | BL rank → Best | Best profile |
|--------|------|---------|-----------|---|---------------|-------------|
| brief0 | factual_recall | 0.5048 | 0.8651 | +0.3604 | 1→1 | constant_1.5 |
| brief1 | algorithmic | 0.8935 | 0.9770 | +0.0835 | 1→1 | constant_1.5 |
| brief2 | structural_copying | 0.1446 | 0.6821 | +0.5374 | 2→1 | ramp_up |
| brief3 | syntactic_pattern | 0.6035 | 0.3578 | -0.2457 | 1→2 | late_boost_1.3 |
| brief4 | cultural_memorized | 0.9968 | 0.9984 | +0.0016 | 1→1 | edges_high |
| brief5 | factual_recall | 0.9239 | 0.9163 | -0.0076 | 1→1 | ramp_up_gentle |
| long0 | code_comprehension | 0.5155 | 0.4853 | -0.0302 | 1→1 | ramp_up_gentle |
| long1 | long_range_retrieval | 0.0266 | 0.1210 | +0.0943 | 1→2 | early_suppress_0.7 |
| long2 | domain_knowledge | 0.0150 | 0.0984 | +0.0834 | 12→2 | late_suppress_0.5 |
| long3 | long_range_retrieval | 0.0788 | 0.1358 | +0.0570 | 3→1 | middle_bump_1.5_edges_0.8 |
| med0 | factual_retrieval | 0.2527 | 0.3819 | +0.1292 | 2→1 | late_suppress_0.7 |
| med1 | reasoning_numerical | 0.0038 | 0.0040 | +0.0001 | 27→28 | ramp_down_gentle |
| med2 | reasoning_tracking | 0.0789 | 0.0972 | +0.0182 | 4→2 | late_suppress_0.7 |
| med3 | reasoning_numerical | 0.4961 | 0.6251 | +0.1290 | 1→1 | late_boost_1.5 |
| med4 | syntactic_pattern | 0.1789 | 0.3120 | +0.1331 | 1→1 | edges_high |
| med5 | reasoning_tracking | 0.9202 | 0.9923 | +0.0720 | 1→1 | edges_high |
| short0 | factual_recall | 0.2701 | 0.6421 | +0.3721 | 2→1 | early_high_late_low |
| short1 | algorithmic | 0.9011 | 0.9620 | +0.0608 | 1→1 | ramp_up |
| short2 | factual_recall | 0.3441 | 0.6358 | +0.2917 | 1→1 | early_high_late_low |
| short3 | structural_copying | 0.5456 | 0.8549 | +0.3093 | 1→1 | constant_2 |
| short4 | cultural_memorized | 0.7182 | 0.7957 | +0.0776 | 1→1 | ramp_up |
| short5 | syntactic_pattern | 0.9750 | 0.9797 | +0.0047 | 1→1 | ramp_up_gentle |

### Damage from Bad Profiles

The worst profile per prompt shows how much damage a bad choice causes.

| Model | Mean worst Δprob | Max damage | Prompts where worst < 0.001 prob |
|-------|-----------------|------------|----------------------------------|
| Qwen3.5-0.8B | -0.3875 | -0.9965 | 22/22 |
| Qwen3.5-2B | -0.5266 | -0.9988 | 22/22 |
| Qwen3.5-9B | -0.5793 | -0.9995 | 22/22 |
| OLMo-Hybrid-7B | -0.4722 | -0.9968 | 22/22 |

The asymmetry matters: the best profile typically gains +0.05–0.5 in probability, while the worst profile can drive probability to near zero. A control system must avoid catastrophic profiles even more than it must find optimal ones.

## 3. Baseline Signal Inventory

Signals available from a single forward pass at g=1.0, before any colony interaction.

### 3.1 Final-Token Entropy (bits)

The entropy of the output distribution over the vocabulary. High entropy = model is uncertain.

| Prompt | Type | 0.8B | 2B | 9B | OLMO |
|--------|------|------|----|----|------|
| brief0 | factual_recall | 2.58 | 2.15 | 1.80 | 1.80 |
| brief1 | algorithmic | 0.82 | 0.59 | 0.53 | 1.08 |
| brief2 | structural_copying | 5.19 | 4.30 | 3.76 | 4.22 |
| brief3 | syntactic_pattern | 2.57 | 1.70 | 0.73 | 2.04 |
| brief4 | cultural_memorized | 0.22 | 0.11 | 0.02 | 0.04 |
| brief5 | factual_recall | 1.31 | 1.19 | 1.49 | 0.77 |
| long0 | code_comprehension | 0.08 | 0.03 | 0.05 | 2.35 |
| long1 | long_range_retrieval | 9.61 | 3.28 | 2.27 | 9.82 |
| long2 | domain_knowledge | 6.94 | 6.01 | 5.11 | 4.72 |
| long3 | long_range_retrieval | 4.71 | 2.90 | 2.16 | 6.67 |
| med0 | factual_retrieval | 5.07 | 2.14 | 3.57 | 3.22 |
| med1 | reasoning_numerical | 8.83 | 8.46 | 8.06 | 9.43 |
| med2 | reasoning_tracking | 3.83 | 3.20 | 4.34 | 6.06 |
| med3 | reasoning_numerical | 0.40 | 0.19 | 0.23 | 2.39 |
| med4 | syntactic_pattern | 7.50 | 3.94 | 2.91 | 7.07 |
| med5 | reasoning_tracking | 4.04 | 1.81 | 0.12 | 0.59 |
| short0 | factual_recall | 5.17 | 5.05 | 4.58 | 3.94 |
| short1 | algorithmic | 1.05 | 1.70 | 0.99 | 0.84 |
| short2 | factual_recall | 6.49 | 4.94 | 4.59 | 5.19 |
| short3 | structural_copying | 4.48 | 4.20 | 3.02 | 4.40 |
| short4 | cultural_memorized | 9.53 | 9.29 | 1.65 | 3.39 |
| short5 | syntactic_pattern | 0.04 | 0.02 | 0.01 | 0.28 |

### 3.2 Mean Sequence Entropy (bits)

Average entropy across all token positions, not just the final token. Captures overall model confidence on the context.

| Prompt | Type | 0.8B | 2B | 9B | OLMO |
|--------|------|------|----|----|------|
| brief0 | factual_recall | 3.03 | 2.62 | 2.23 | 2.69 |
| brief1 | algorithmic | 1.17 | 1.05 | 0.95 | 1.33 |
| brief2 | structural_copying | 4.36 | 3.98 | 3.55 | 3.61 |
| brief3 | syntactic_pattern | 0.93 | 0.83 | 0.84 | 2.09 |
| brief4 | cultural_memorized | 1.29 | 0.76 | 0.58 | 0.60 |
| brief5 | factual_recall | 2.36 | 2.14 | 1.90 | 2.14 |
| long0 | code_comprehension | 1.26 | 1.02 | 0.74 | 1.66 |
| long1 | long_range_retrieval | 4.26 | 3.51 | 2.72 | 3.62 |
| long2 | domain_knowledge | 3.95 | 3.41 | 2.66 | 3.25 |
| long3 | long_range_retrieval | 3.59 | 3.21 | 2.75 | 3.27 |
| med0 | factual_retrieval | 4.43 | 4.13 | 3.70 | 4.28 |
| med1 | reasoning_numerical | 4.24 | 4.03 | 3.62 | 4.28 |
| med2 | reasoning_tracking | 4.10 | 3.51 | 2.90 | 3.48 |
| med3 | reasoning_numerical | 2.82 | 2.54 | 2.27 | 3.40 |
| med4 | syntactic_pattern | 0.74 | 0.61 | 0.49 | 1.51 |
| med5 | reasoning_tracking | 4.37 | 3.87 | 3.00 | 3.55 |
| short0 | factual_recall | 4.61 | 4.20 | 4.22 | 4.77 |
| short1 | algorithmic | 1.80 | 1.74 | 1.48 | 1.54 |
| short2 | factual_recall | 7.01 | 6.27 | 6.07 | 6.66 |
| short3 | structural_copying | 6.07 | 5.46 | 5.13 | 5.00 |
| short4 | cultural_memorized | 4.87 | 4.96 | 3.33 | 3.27 |
| short5 | syntactic_pattern | 1.75 | 1.74 | 1.71 | 1.37 |

### 3.3 Attention Entropy Profile (per-layer mean, baseline)

Mean entropy across heads at each attention layer. This is the core signal for per-layer g-profile selection.

#### Qwen3.5-0.8B (6 attention layers at indices [3, 7, 11, 15, 19, 23])

| Prompt | Type | L3 | L7 | L11 | L15 | L19 | L23 | Gradient |
|--------|------|------|------|------|------|------|------|----------|
| brief0 | factual_recall | 4.41 | 3.95 | 3.90 | 3.75 | 4.95 | 4.09 | +0.182 |
| brief1 | algorithmic | 4.58 | 4.61 | 5.45 | 3.90 | 3.82 | 3.80 | -1.039 |
| brief2 | structural_copying | 4.68 | 4.33 | 4.64 | 4.67 | 5.08 | 3.77 | -0.042 |
| brief3 | syntactic_pattern | 4.77 | 3.29 | 2.89 | 2.76 | 4.20 | 3.71 | -0.093 |
| brief4 | cultural_memorized | 4.43 | 3.76 | 3.30 | 2.97 | 3.90 | 4.16 | -0.151 |
| brief5 | factual_recall | 3.95 | 3.52 | 3.56 | 2.82 | 3.38 | 3.53 | -0.430 |
| long0 | code_comprehension | 4.82 | 4.31 | 4.76 | 3.77 | 4.92 | 4.85 | -0.118 |
| long1 | long_range_retrieval | 6.69 | 4.77 | 4.73 | 4.59 | 6.82 | 4.91 | +0.042 |
| long2 | domain_knowledge | 6.11 | 5.89 | 6.20 | 6.74 | 7.45 | 5.68 | +0.556 |
| long3 | long_range_retrieval | 5.90 | 5.11 | 4.42 | 3.57 | 5.71 | 4.08 | -0.687 |
| med0 | factual_retrieval | 5.66 | 5.12 | 5.14 | 3.80 | 5.97 | 4.33 | -0.608 |
| med1 | reasoning_numerical | 5.07 | 6.03 | 6.24 | 5.85 | 5.79 | 4.26 | -0.479 |
| med2 | reasoning_tracking | 5.76 | 5.04 | 5.82 | 5.46 | 5.37 | 4.44 | -0.447 |
| med3 | reasoning_numerical | 5.89 | 4.74 | 5.28 | 4.28 | 4.90 | 4.78 | -0.652 |
| med4 | syntactic_pattern | 6.86 | 5.51 | 5.46 | 5.14 | 5.10 | 4.81 | -0.924 |
| med5 | reasoning_tracking | 5.88 | 5.36 | 5.58 | 5.60 | 5.13 | 4.31 | -0.595 |
| short0 | factual_recall | 1.90 | 2.28 | 2.25 | 2.00 | 2.46 | 1.67 | -0.100 |
| short1 | algorithmic | 3.75 | 3.26 | 3.68 | 2.96 | 2.53 | 3.44 | -0.589 |
| short2 | factual_recall | 1.58 | 1.81 | 2.05 | 1.79 | 1.35 | 1.22 | -0.358 |
| short3 | structural_copying | 2.47 | 1.90 | 2.26 | 1.87 | 1.92 | 1.89 | -0.317 |
| short4 | cultural_memorized | 2.71 | 2.54 | 3.01 | 2.91 | 3.53 | 2.67 | +0.282 |
| short5 | syntactic_pattern | 2.05 | 1.80 | 1.94 | 1.98 | 2.21 | 1.88 | +0.099 |

#### Qwen3.5-2B (6 attention layers at indices [3, 7, 11, 15, 19, 23])

| Prompt | Type | L3 | L7 | L11 | L15 | L19 | L23 | Gradient |
|--------|------|------|------|------|------|------|------|----------|
| brief0 | factual_recall | 4.58 | 3.78 | 4.30 | 3.63 | 4.81 | 3.98 | -0.078 |
| brief1 | algorithmic | 4.70 | 4.88 | 5.48 | 3.80 | 5.10 | 4.09 | -0.688 |
| brief2 | structural_copying | 4.78 | 4.38 | 4.76 | 4.72 | 5.04 | 4.08 | -0.031 |
| brief3 | syntactic_pattern | 4.93 | 3.71 | 2.92 | 3.10 | 4.47 | 4.19 | +0.066 |
| brief4 | cultural_memorized | 4.46 | 4.15 | 3.85 | 3.09 | 4.12 | 4.27 | -0.326 |
| brief5 | factual_recall | 3.96 | 3.68 | 3.68 | 2.88 | 3.40 | 3.41 | -0.543 |
| long0 | code_comprehension | 5.18 | 4.54 | 5.02 | 4.61 | 4.85 | 4.70 | -0.196 |
| long1 | long_range_retrieval | 6.98 | 4.91 | 4.86 | 4.74 | 5.96 | 4.44 | -0.533 |
| long2 | domain_knowledge | 6.49 | 5.60 | 6.98 | 7.31 | 7.17 | 5.48 | +0.301 |
| long3 | long_range_retrieval | 6.30 | 4.15 | 4.38 | 3.49 | 5.56 | 4.30 | -0.494 |
| med0 | factual_retrieval | 5.93 | 5.40 | 4.67 | 3.80 | 5.32 | 4.29 | -0.868 |
| med1 | reasoning_numerical | 5.51 | 5.86 | 6.59 | 6.06 | 5.23 | 4.12 | -0.850 |
| med2 | reasoning_tracking | 5.91 | 5.07 | 5.95 | 5.53 | 5.10 | 4.41 | -0.628 |
| med3 | reasoning_numerical | 5.81 | 5.11 | 5.45 | 4.82 | 4.65 | 5.00 | -0.635 |
| med4 | syntactic_pattern | 6.82 | 5.62 | 5.49 | 4.95 | 6.09 | 4.70 | -0.729 |
| med5 | reasoning_tracking | 5.84 | 5.36 | 5.84 | 5.28 | 4.95 | 4.24 | -0.855 |
| short0 | factual_recall | 2.09 | 2.14 | 2.01 | 2.04 | 2.05 | 1.52 | -0.211 |
| short1 | algorithmic | 3.84 | 3.21 | 3.91 | 3.17 | 3.66 | 3.61 | -0.176 |
| short2 | factual_recall | 1.80 | 1.66 | 1.99 | 1.72 | 1.79 | 1.57 | -0.127 |
| short3 | structural_copying | 2.56 | 1.91 | 2.28 | 1.69 | 1.87 | 2.00 | -0.397 |
| short4 | cultural_memorized | 2.63 | 2.75 | 2.91 | 2.82 | 3.29 | 2.85 | +0.228 |
| short5 | syntactic_pattern | 2.18 | 1.77 | 1.59 | 1.63 | 1.87 | 1.77 | -0.093 |

#### Qwen3.5-9B (8 attention layers at indices [3, 7, 11, 15, 19, 23, 27, 31])

| Prompt | Type | L3 | L7 | L11 | L15 | L19 | L23 | L27 | L31 | Gradient |
|--------|------|------|------|------|------|------|------|------|------|----------|
| brief0 | factual_recall | 5.05 | 4.94 | 4.36 | 4.70 | 3.94 | 4.46 | 4.07 | 4.11 | -0.620 |
| brief1 | algorithmic | 4.96 | 4.72 | 5.09 | 5.17 | 3.85 | 4.77 | 4.52 | 4.11 | -0.670 |
| brief2 | structural_copying | 4.85 | 4.71 | 4.88 | 4.62 | 4.45 | 4.69 | 4.51 | 3.95 | -0.363 |
| brief3 | syntactic_pattern | 5.26 | 4.74 | 3.46 | 3.32 | 3.43 | 3.65 | 3.70 | 3.99 | -0.503 |
| brief4 | cultural_memorized | 5.30 | 4.45 | 4.47 | 4.42 | 3.16 | 3.80 | 4.00 | 4.38 | -0.823 |
| brief5 | factual_recall | 4.31 | 3.71 | 3.62 | 4.48 | 3.39 | 3.46 | 3.25 | 3.82 | -0.551 |
| long0 | code_comprehension | 5.71 | 4.87 | 4.46 | 4.80 | 4.51 | 5.01 | 4.84 | 4.66 | -0.205 |
| long1 | long_range_retrieval | 7.30 | 4.88 | 4.43 | 5.34 | 5.18 | 5.50 | 4.52 | 4.64 | -0.528 |
| long2 | domain_knowledge | 6.79 | 5.92 | 6.29 | 6.69 | 6.46 | 6.87 | 6.66 | 5.89 | +0.046 |
| long3 | long_range_retrieval | 7.03 | 5.39 | 4.66 | 4.72 | 3.72 | 4.68 | 4.94 | 4.81 | -0.912 |
| med0 | factual_retrieval | 6.10 | 5.12 | 5.05 | 4.76 | 4.91 | 5.46 | 4.73 | 4.15 | -0.446 |
| med1 | reasoning_numerical | 5.78 | 5.89 | 6.19 | 5.80 | 5.70 | 5.73 | 5.31 | 4.50 | -0.606 |
| med2 | reasoning_tracking | 6.71 | 5.73 | 5.85 | 5.40 | 5.48 | 5.67 | 5.05 | 4.09 | -0.848 |
| med3 | reasoning_numerical | 6.54 | 4.90 | 5.38 | 5.35 | 4.97 | 5.37 | 5.03 | 4.95 | -0.464 |
| med4 | syntactic_pattern | 6.90 | 6.04 | 6.11 | 5.06 | 5.19 | 5.69 | 4.92 | 4.66 | -0.913 |
| med5 | reasoning_tracking | 6.37 | 5.73 | 5.25 | 4.90 | 4.05 | 5.43 | 5.15 | 5.21 | -0.601 |
| short0 | factual_recall | 2.13 | 2.10 | 2.01 | 2.08 | 2.01 | 2.03 | 1.59 | 1.76 | -0.232 |
| short1 | algorithmic | 3.81 | 3.02 | 3.06 | 3.47 | 2.91 | 3.54 | 2.76 | 3.34 | -0.199 |
| short2 | factual_recall | 1.75 | 1.64 | 1.48 | 1.78 | 1.58 | 1.52 | 1.30 | 1.45 | -0.199 |
| short3 | structural_copying | 2.58 | 2.25 | 2.04 | 2.18 | 1.55 | 2.24 | 1.86 | 2.16 | -0.311 |
| short4 | cultural_memorized | 3.11 | 3.06 | 2.87 | 2.94 | 2.66 | 3.00 | 3.19 | 2.95 | -0.047 |
| short5 | syntactic_pattern | 2.32 | 2.12 | 1.99 | 2.18 | 1.86 | 2.02 | 1.83 | 1.83 | -0.267 |

#### OLMo-Hybrid-7B (8 attention layers at indices [3, 7, 11, 15, 19, 23, 27, 31])

| Prompt | Type | L3 | L7 | L11 | L15 | L19 | L23 | L27 | L31 | Gradient |
|--------|------|------|------|------|------|------|------|------|------|----------|
| brief0 | factual_recall | 4.29 | 3.70 | 4.48 | 4.30 | 3.35 | 2.45 | 2.55 | 2.83 | -1.397 |
| brief1 | algorithmic | 4.46 | 4.28 | 4.32 | 4.30 | 3.98 | 3.06 | 3.12 | 2.93 | -1.066 |
| brief2 | structural_copying | 3.88 | 3.38 | 4.27 | 3.76 | 3.48 | 2.91 | 2.63 | 2.06 | -1.054 |
| brief3 | syntactic_pattern | 5.17 | 4.52 | 5.71 | 5.48 | 5.09 | 3.53 | 3.48 | 3.36 | -1.357 |
| brief4 | cultural_memorized | 3.95 | 3.14 | 3.86 | 3.66 | 3.09 | 1.90 | 1.23 | 1.72 | -1.669 |
| brief5 | factual_recall | 4.25 | 3.77 | 4.07 | 3.87 | 2.95 | 2.37 | 2.02 | 2.82 | -1.453 |
| long0 | code_comprehension | 6.95 | 6.17 | 5.96 | 5.33 | 5.23 | 5.23 | 5.51 | 6.37 | -0.517 |
| long1 | long_range_retrieval | 6.43 | 6.01 | 5.48 | 5.36 | 4.95 | 4.38 | 3.20 | 5.77 | -1.248 |
| long2 | domain_knowledge | 6.81 | 6.84 | 6.46 | 6.62 | 6.16 | 6.45 | 7.30 | 7.36 | +0.137 |
| long3 | long_range_retrieval | 7.06 | 6.60 | 6.51 | 6.40 | 5.97 | 5.78 | 5.84 | 6.64 | -0.583 |
| med0 | factual_retrieval | 5.08 | 5.00 | 5.00 | 4.34 | 4.78 | 3.66 | 3.02 | 3.34 | -1.154 |
| med1 | reasoning_numerical | 4.80 | 4.40 | 4.89 | 5.16 | 5.14 | 4.16 | 3.24 | 3.45 | -0.812 |
| med2 | reasoning_tracking | 5.69 | 4.97 | 5.17 | 4.65 | 4.43 | 3.80 | 3.22 | 3.25 | -1.446 |
| med3 | reasoning_numerical | 5.55 | 4.89 | 5.35 | 4.87 | 4.74 | 4.30 | 3.86 | 4.84 | -0.732 |
| med4 | syntactic_pattern | 6.62 | 5.90 | 5.74 | 5.65 | 5.36 | 5.27 | 4.98 | 5.74 | -0.641 |
| med5 | reasoning_tracking | 5.43 | 5.10 | 5.30 | 4.56 | 5.04 | 4.06 | 3.79 | 3.90 | -0.897 |
| short0 | factual_recall | 1.85 | 1.37 | 1.56 | 1.43 | 1.11 | 0.75 | 0.71 | 0.67 | -0.741 |
| short1 | algorithmic | 3.37 | 2.90 | 3.10 | 2.80 | 2.62 | 1.73 | 1.13 | 1.58 | -1.276 |
| short2 | factual_recall | 1.71 | 1.58 | 1.75 | 1.85 | 1.85 | 1.88 | 1.50 | 1.41 | -0.062 |
| short3 | structural_copying | 1.82 | 1.76 | 1.84 | 1.75 | 1.37 | 0.77 | 0.66 | 0.57 | -0.951 |
| short4 | cultural_memorized | 2.81 | 2.73 | 2.62 | 2.23 | 1.99 | 1.25 | 1.26 | 1.42 | -1.117 |
| short5 | syntactic_pattern | 1.98 | 1.00 | 1.66 | 1.25 | 1.11 | 0.66 | 0.81 | 0.82 | -0.624 |

## 4. Response Surface Characterization

How does each prompt respond across the 31 g-profiles? We characterize the *shape* of the response.

### 4.1 Response Surface Statistics

For each model × prompt, statistics over all 31 profiles:

| Model | Prompt | Type | Prob: min/med/max | Entropy: min/med/max | Profiles with rank=1 |
|-------|--------|------|-------------------|---------------------|---------------------|
| Qwen3.5-0.8B | brief0 | factual_recall | 0.0000/0.1490/0.3127 | 1.3/3.2/17.9 | 5/31 |
| Qwen3.5-0.8B | brief1 | algorithmic | 0.0000/0.9014/0.9400 | 0.6/0.9/17.9 | 29/31 |
| Qwen3.5-0.8B | brief2 | structural_copying | 0.0000/0.2438/0.4537 | 2.5/5.6/17.9 | 21/31 |
| Qwen3.5-0.8B | brief3 | syntactic_pattern | 0.0000/0.0079/0.7194 | 1.6/5.9/17.9 | 7/31 |
| Qwen3.5-0.8B | brief4 | cultural_memorized | 0.0000/0.6982/0.9780 | 0.2/1.6/17.9 | 26/31 |
| Qwen3.5-0.8B | brief5 | factual_recall | 0.0000/0.3753/0.9941 | 0.1/4.1/17.9 | 24/31 |
| Qwen3.5-0.8B | long0 | code_comprehension | 0.0000/0.9519/0.9982 | 0.0/0.4/17.9 | 26/31 |
| Qwen3.5-0.8B | long1 | long_range_retrieval | 0.0000/0.0180/0.4385 | 6.3/9.4/17.9 | 11/31 |
| Qwen3.5-0.8B | long2 | domain_knowledge | 0.0000/0.0077/0.0305 | 5.2/7.1/17.9 | 5/31 |
| Qwen3.5-0.8B | long3 | long_range_retrieval | 0.0000/0.2595/0.6491 | 3.2/6.5/17.9 | 26/31 |
| Qwen3.5-0.8B | med0 | factual_retrieval | 0.0000/0.0175/0.8058 | 1.7/4.5/17.9 | 12/31 |
| Qwen3.5-0.8B | med1 | reasoning_numerical | 0.0000/0.0011/0.0157 | 7.4/9.5/17.9 | 5/31 |
| Qwen3.5-0.8B | med2 | reasoning_tracking | 0.0000/0.0035/0.0246 | 1.9/5.9/17.9 | 5/31 |
| Qwen3.5-0.8B | med3 | reasoning_numerical | 0.0000/0.5161/0.9831 | 0.2/1.9/17.9 | 27/31 |
| Qwen3.5-0.8B | med4 | syntactic_pattern | 0.0000/0.0015/0.0544 | 6.0/8.5/17.9 | 5/31 |
| Qwen3.5-0.8B | med5 | reasoning_tracking | 0.0000/0.0092/0.0629 | 1.9/5.3/17.9 | 5/31 |
| Qwen3.5-0.8B | short0 | factual_recall | 0.0000/0.0173/0.1418 | 3.9/5.8/17.9 | 8/31 |
| Qwen3.5-0.8B | short1 | algorithmic | 0.0000/0.8741/0.9945 | 0.1/1.0/17.9 | 29/31 |
| Qwen3.5-0.8B | short2 | factual_recall | 0.0000/0.0035/0.3810 | 5.3/6.6/17.9 | 11/31 |
| Qwen3.5-0.8B | short3 | structural_copying | 0.0000/0.3328/0.7054 | 3.0/5.8/17.9 | 29/31 |
| Qwen3.5-0.8B | short4 | cultural_memorized | 0.0000/0.0029/0.0264 | 0.7/9.4/17.9 | 5/31 |
| Qwen3.5-0.8B | short5 | syntactic_pattern | 0.0000/0.9845/0.9971 | 0.0/0.2/17.9 | 26/31 |
| Qwen3.5-2B | brief0 | factual_recall | 0.0000/0.4630/0.7263 | 1.8/2.5/17.9 | 22/31 |
| Qwen3.5-2B | brief1 | algorithmic | 0.0000/0.9126/0.9734 | 0.3/0.9/17.9 | 28/31 |
| Qwen3.5-2B | brief2 | structural_copying | 0.0000/0.1601/0.3934 | 2.9/4.8/17.9 | 12/31 |
| Qwen3.5-2B | brief3 | syntactic_pattern | 0.0000/0.0421/0.5717 | 1.7/4.1/17.9 | 6/31 |
| Qwen3.5-2B | brief4 | cultural_memorized | 0.0000/0.9006/0.9926 | 0.1/0.6/17.9 | 27/31 |
| Qwen3.5-2B | brief5 | factual_recall | 0.0000/0.7166/0.9519 | 0.5/2.2/17.9 | 26/31 |
| Qwen3.5-2B | long0 | code_comprehension | 0.0000/0.9756/0.9995 | 0.0/0.3/17.9 | 27/31 |
| Qwen3.5-2B | long1 | long_range_retrieval | 0.0000/0.4260/0.8867 | 1.5/5.8/17.9 | 25/31 |
| Qwen3.5-2B | long2 | domain_knowledge | 0.0000/0.0165/0.0403 | 3.9/6.3/17.9 | 5/31 |
| Qwen3.5-2B | long3 | long_range_retrieval | 0.0000/0.6299/0.8815 | 1.4/4.0/17.9 | 26/31 |
| Qwen3.5-2B | med0 | factual_retrieval | 0.0000/0.7023/0.9901 | 0.2/2.6/17.9 | 24/31 |
| Qwen3.5-2B | med1 | reasoning_numerical | 0.0000/0.0024/0.0187 | 6.7/9.4/17.9 | 5/31 |
| Qwen3.5-2B | med2 | reasoning_tracking | 0.0000/0.0254/0.1072 | 2.6/4.4/17.9 | 5/31 |
| Qwen3.5-2B | med3 | reasoning_numerical | 0.0000/0.7754/0.9958 | 0.1/1.2/17.9 | 28/31 |
| Qwen3.5-2B | med4 | syntactic_pattern | 0.0000/0.1148/0.4807 | 3.7/6.6/17.9 | 14/31 |
| Qwen3.5-2B | med5 | reasoning_tracking | 0.0000/0.2468/0.7989 | 1.2/3.3/17.9 | 16/31 |
| Qwen3.5-2B | short0 | factual_recall | 0.0000/0.0307/0.1161 | 3.5/5.6/17.9 | 7/31 |
| Qwen3.5-2B | short1 | algorithmic | 0.0000/0.7707/0.9430 | 0.5/1.5/17.9 | 29/31 |
| Qwen3.5-2B | short2 | factual_recall | 0.0000/0.0224/0.2295 | 1.9/5.4/17.9 | 8/31 |
| Qwen3.5-2B | short3 | structural_copying | 0.0000/0.4205/0.7295 | 2.9/5.3/17.9 | 28/31 |
| Qwen3.5-2B | short4 | cultural_memorized | 0.0000/0.0293/0.2939 | 1.3/9.2/17.9 | 14/31 |
| Qwen3.5-2B | short5 | syntactic_pattern | 0.0000/0.9383/0.9988 | 0.0/0.6/17.9 | 26/31 |
| Qwen3.5-9B | brief0 | factual_recall | 0.0000/0.3577/0.7819 | 1.2/2.0/17.9 | 19/31 |
| Qwen3.5-9B | brief1 | algorithmic | 0.0000/0.8773/0.9901 | 0.1/1.2/17.9 | 27/31 |
| Qwen3.5-9B | brief2 | structural_copying | 0.0000/0.1599/0.6337 | 1.5/5.3/17.9 | 18/31 |
| Qwen3.5-9B | brief3 | syntactic_pattern | 0.0000/0.0013/0.0171 | 0.4/2.9/17.9 | 5/31 |
| Qwen3.5-9B | brief4 | cultural_memorized | 0.0000/0.9422/0.9987 | 0.0/0.6/17.9 | 26/31 |
| Qwen3.5-9B | brief5 | factual_recall | 0.0000/0.6468/0.8307 | 1.4/2.6/17.9 | 24/31 |
| Qwen3.5-9B | long0 | code_comprehension | 0.0000/0.9274/0.9993 | 0.0/0.6/17.9 | 26/31 |
| Qwen3.5-9B | long1 | long_range_retrieval | 0.0000/0.2130/0.9834 | 0.2/6.5/17.9 | 25/31 |
| Qwen3.5-9B | long2 | domain_knowledge | 0.0000/0.0220/0.0501 | 4.4/5.7/17.9 | 5/31 |
| Qwen3.5-9B | long3 | long_range_retrieval | 0.0000/0.7441/0.9537 | 0.6/3.0/17.9 | 26/31 |
| Qwen3.5-9B | med0 | factual_retrieval | 0.0000/0.2711/0.7886 | 1.7/3.5/17.9 | 21/31 |
| Qwen3.5-9B | med1 | reasoning_numerical | 0.0000/0.0019/0.0937 | 4.0/9.5/17.9 | 5/31 |
| Qwen3.5-9B | med2 | reasoning_tracking | 0.0000/0.1034/0.5375 | 2.8/4.6/17.9 | 15/31 |
| Qwen3.5-9B | med3 | reasoning_numerical | 0.0000/0.8202/0.9966 | 0.0/1.1/17.9 | 26/31 |
| Qwen3.5-9B | med4 | syntactic_pattern | 0.0000/0.0909/0.7113 | 1.5/4.5/17.9 | 12/31 |
| Qwen3.5-9B | med5 | reasoning_tracking | 0.0000/0.1287/0.9916 | 0.1/3.2/17.9 | 19/31 |
| Qwen3.5-9B | short0 | factual_recall | 0.0000/0.1415/0.3355 | 2.9/5.3/17.9 | 19/31 |
| Qwen3.5-9B | short1 | algorithmic | 0.0000/0.8537/0.9601 | 0.4/1.2/17.9 | 30/31 |
| Qwen3.5-9B | short2 | factual_recall | 0.0000/0.3079/0.5646 | 2.8/5.0/17.9 | 23/31 |
| Qwen3.5-9B | short3 | structural_copying | 0.0000/0.4904/0.9085 | 0.8/4.2/17.9 | 28/31 |
| Qwen3.5-9B | short4 | cultural_memorized | 0.0000/0.2760/0.8671 | 0.9/4.0/17.9 | 22/31 |
| Qwen3.5-9B | short5 | syntactic_pattern | 0.0000/0.9910/0.9995 | 0.0/0.1/17.9 | 27/31 |
| OLMo-Hybrid-7B | brief0 | factual_recall | 0.0000/0.4778/0.8651 | 0.8/2.3/16.6 | 22/31 |
| OLMo-Hybrid-7B | brief1 | algorithmic | 0.0000/0.8208/0.9770 | 0.3/2.0/16.6 | 30/31 |
| OLMo-Hybrid-7B | brief2 | structural_copying | 0.0000/0.0771/0.6821 | 1.9/4.3/16.6 | 11/31 |
| OLMo-Hybrid-7B | brief3 | syntactic_pattern | 0.0000/0.1659/0.6035 | 1.0/4.4/16.6 | 10/31 |
| OLMo-Hybrid-7B | brief4 | cultural_memorized | 0.0000/0.8866/0.9984 | 0.0/1.3/16.6 | 28/31 |
| OLMo-Hybrid-7B | brief5 | factual_recall | 0.0000/0.6867/0.9239 | 0.8/2.7/16.6 | 26/31 |
| OLMo-Hybrid-7B | long0 | code_comprehension | 0.0000/0.1609/0.5155 | 2.1/2.9/16.6 | 13/31 |
| OLMo-Hybrid-7B | long1 | long_range_retrieval | 0.0000/0.0167/0.1210 | 3.6/9.8/16.6 | 17/31 |
| OLMo-Hybrid-7B | long2 | domain_knowledge | 0.0000/0.0150/0.0984 | 3.8/5.7/16.6 | 5/31 |
| OLMo-Hybrid-7B | long3 | long_range_retrieval | 0.0000/0.0377/0.1358 | 5.6/7.4/16.6 | 7/31 |
| OLMo-Hybrid-7B | med0 | factual_retrieval | 0.0000/0.0551/0.3819 | 1.7/3.9/16.6 | 7/31 |
| OLMo-Hybrid-7B | med1 | reasoning_numerical | 0.0000/0.0010/0.0040 | 8.1/10.1/16.6 | 5/31 |
| OLMo-Hybrid-7B | med2 | reasoning_tracking | 0.0000/0.0063/0.0972 | 3.8/6.9/16.6 | 5/31 |
| OLMo-Hybrid-7B | med3 | reasoning_numerical | 0.0000/0.2262/0.6251 | 1.7/3.0/16.6 | 26/31 |
| OLMo-Hybrid-7B | med4 | syntactic_pattern | 0.0000/0.0179/0.3120 | 0.8/7.1/16.6 | 10/31 |
| OLMo-Hybrid-7B | med5 | reasoning_tracking | 0.0000/0.4261/0.9923 | 0.1/3.8/16.6 | 21/31 |
| OLMo-Hybrid-7B | short0 | factual_recall | 0.0000/0.1654/0.6421 | 3.1/4.9/16.6 | 16/31 |
| OLMo-Hybrid-7B | short1 | algorithmic | 0.0000/0.8861/0.9620 | 0.4/1.0/16.6 | 30/31 |
| OLMo-Hybrid-7B | short2 | factual_recall | 0.0000/0.0825/0.6358 | 3.6/6.0/16.6 | 20/31 |
| OLMo-Hybrid-7B | short3 | structural_copying | 0.0000/0.5033/0.8549 | 1.6/5.0/16.6 | 29/31 |
| OLMo-Hybrid-7B | short4 | cultural_memorized | 0.0000/0.3416/0.7957 | 2.6/7.1/16.6 | 26/31 |
| OLMo-Hybrid-7B | short5 | syntactic_pattern | 0.0000/0.9173/0.9797 | 0.2/0.9/16.6 | 26/31 |

### 4.2 Profile Family Performance

Grouping the 31 profiles into families, what is the mean Δprob from baseline?

| Family | n profiles | Mean Δprob (all) | Mean Δprob (0.8B) | Mean Δprob (2B) | Mean Δprob (9B) | Mean Δprob (OLMO) |
|--------|-----------|-----------------|-------------------|-----------------|-----------------|-------------------|
| uniform | 6 | -0.2806 | -0.2211 | -0.3054 | -0.3353 | -0.2606 |
| early_regional | 4 | -0.1498 | -0.1164 | -0.1638 | -0.1775 | -0.1416 |
| late_regional | 4 | -0.0485 | -0.0185 | -0.0144 | -0.0760 | -0.0851 |
| middle | 4 | -0.1777 | -0.1311 | -0.1687 | -0.1960 | -0.2150 |
| ramp_up | 2 | -0.0034 | +0.0190 | +0.0137 | +0.0085 | -0.0550 |
| ramp_down | 2 | -0.0556 | -0.0557 | -0.0676 | -0.0430 | -0.0562 |
| edges | 2 | -0.0150 | +0.0180 | -0.0315 | +0.0154 | -0.0618 |
| cross | 2 | -0.2533 | -0.1931 | -0.2500 | -0.3403 | -0.2297 |
| ablation | 2 | -0.4914 | -0.3875 | -0.5266 | -0.5793 | -0.4722 |
| alternating | 2 | -0.4914 | -0.3875 | -0.5266 | -0.5793 | -0.4722 |

## 5. Signal-to-Optimum Correlations

Can baseline signals predict which g-profile family works best, or how much headroom exists?

### 5.1 Pearson and Spearman Correlations (all models pooled, n=88)

| Signal → Target | Pearson r | p | Spearman ρ | p |
|-----------------|-----------|---|-----------|---|
| final_entropy → prob_delta | +0.173 | 0.1070 | +0.350 | 0.0008 |
| bl_prob → prob_delta | -0.331 | 0.0016** | -0.299 | 0.0046 |
| mean_entropy → prob_delta | +0.275 | 0.0094** | +0.411 | 0.0001 |
| attn_ent_mean → prob_delta | -0.138 | 0.2000 | -0.153 | 0.1542 |
| attn_ent_gradient → prob_delta | +0.097 | 0.3693 | +0.075 | 0.4866 |
| attn_ent_last → prob_delta | -0.145 | 0.1767 | -0.180 | 0.0926 |
| attn_ent_head_var → prob_delta | -0.085 | 0.4306 | -0.094 | 0.3836 |
| final_entropy → best_g_mean | -0.106 | 0.3242 | -0.195 | 0.0689 |
| final_entropy → best_g_tilt | -0.121 | 0.2601 | -0.133 | 0.2181 |
| attn_ent_gradient → best_g_mean | -0.076 | 0.4831 | -0.017 | 0.8757 |
| attn_ent_gradient → best_g_tilt | -0.074 | 0.4925 | -0.067 | 0.5334 |
| attn_ent_last → best_g_mean | -0.119 | 0.2683 | +0.014 | 0.8936 |
| attn_ent_last → best_g_tilt | +0.126 | 0.2409 | +0.112 | 0.2982 |
| attn_ent_min_head → prob_delta | -0.061 | 0.5747 | -0.053 | 0.6252 |
| attn_ent_std_across_layers → prob_delta | +0.053 | 0.6259 | -0.016 | 0.8842 |
| attn_ent_std_across_layers → best_g_tilt | +0.065 | 0.5493 | +0.054 | 0.6148 |

### 5.2 Per-Model Correlations

Key correlations broken out by model to check consistency:

| Signal → Target | 0.8B r(p) | 2B r(p) | 9B r(p) | OLMO r(p) |
|-----------------|-----------|---------|---------|-----------|
| final_entropy → prob_delta | +0.151(0.503) | +0.146(0.517) | +0.352(0.108) | +0.135(0.548) |
| bl_prob → prob_delta | -0.329(0.135) | -0.328(0.137) | -0.436(0.043) | -0.279(0.209) |
| attn_ent_gradient → best_g_tilt | -0.037(0.870) | -0.008(0.972) | +0.011(0.962) | -0.433(0.044) |
| attn_ent_last → best_g_mean | -0.008(0.971) | +0.043(0.849) | +0.201(0.369) | -0.348(0.112) |

## 6. Response Surface Geometry

### 6.1 Prompt Clustering by Response Surface Shape

Each prompt's response to the 31 g-profiles defines a 31-dimensional vector (target_prob per profile). We cluster prompts by the *shape* of this response surface.

#### Qwen3.5-0.8B

PCA on shape-normalized response surfaces (31 profiles per prompt):

- PC1 explains 38.3%
- PC2 explains 19.7%
- PC3 explains 12.3%
- First 3 PCs: 70.3% total

K-means (k=3) clusters:

- **Cluster 0**: brief0, brief1, brief2, brief4, long0, med1, med3, med5, short0, short1, short4, short5
  Types: factual_recall, algorithmic, structural_copying, cultural_memorized, code_comprehension, reasoning_numerical, reasoning_numerical, reasoning_tracking, factual_recall, algorithmic, cultural_memorized, syntactic_pattern
- **Cluster 1**: brief5, long2, long3, med0, med4, short2, short3
  Types: factual_recall, domain_knowledge, long_range_retrieval, factual_retrieval, syntactic_pattern, factual_recall, structural_copying
- **Cluster 2**: brief3, long1, med2
  Types: syntactic_pattern, long_range_retrieval, reasoning_tracking

#### Qwen3.5-2B

PCA on shape-normalized response surfaces (31 profiles per prompt):

- PC1 explains 27.5%
- PC2 explains 20.0%
- PC3 explains 16.9%
- First 3 PCs: 64.4% total

K-means (k=3) clusters:

- **Cluster 0**: brief1, brief2, long2, short1, short3, short4
  Types: algorithmic, structural_copying, domain_knowledge, algorithmic, structural_copying, cultural_memorized
- **Cluster 1**: brief0, brief4, brief5, long0, long1, long3, med0, med1, med3, med4, med5, short2, short5
  Types: factual_recall, cultural_memorized, factual_recall, code_comprehension, long_range_retrieval, long_range_retrieval, factual_retrieval, reasoning_numerical, reasoning_numerical, syntactic_pattern, reasoning_tracking, factual_recall, syntactic_pattern
- **Cluster 2**: brief3, med2, short0
  Types: syntactic_pattern, reasoning_tracking, factual_recall

#### Qwen3.5-9B

PCA on shape-normalized response surfaces (31 profiles per prompt):

- PC1 explains 27.6%
- PC2 explains 17.7%
- PC3 explains 13.0%
- First 3 PCs: 58.3% total

K-means (k=3) clusters:

- **Cluster 0**: brief3
  Types: syntactic_pattern
- **Cluster 1**: brief0, brief1, brief2, brief4, brief5, long0, long2, long3, med0, med2, med3, short0, short1, short2, short3, short5
  Types: factual_recall, algorithmic, structural_copying, cultural_memorized, factual_recall, code_comprehension, domain_knowledge, long_range_retrieval, factual_retrieval, reasoning_tracking, reasoning_numerical, factual_recall, algorithmic, factual_recall, structural_copying, syntactic_pattern
- **Cluster 2**: long1, med1, med4, med5, short4
  Types: long_range_retrieval, reasoning_numerical, syntactic_pattern, reasoning_tracking, cultural_memorized

#### OLMo-Hybrid-7B

PCA on shape-normalized response surfaces (31 profiles per prompt):

- PC1 explains 27.7%
- PC2 explains 21.5%
- PC3 explains 12.1%
- First 3 PCs: 61.3% total

K-means (k=3) clusters:

- **Cluster 0**: long2, med0, med2, short2
  Types: domain_knowledge, factual_retrieval, reasoning_tracking, factual_recall
- **Cluster 1**: brief0, brief1, brief3, brief4, brief5, long0, long1, long3, med1, med3, med4, med5, short0, short1, short3, short4, short5
  Types: factual_recall, algorithmic, syntactic_pattern, cultural_memorized, factual_recall, code_comprehension, long_range_retrieval, long_range_retrieval, reasoning_numerical, reasoning_numerical, syntactic_pattern, reasoning_tracking, factual_recall, algorithmic, structural_copying, cultural_memorized, syntactic_pattern
- **Cluster 2**: brief2
  Types: structural_copying

### 6.2 Cross-Model Response Consistency

For each prompt, how correlated are the response surfaces across models? High correlation = the same profiles help/hurt regardless of model.

| Prompt | Type | 0.8B↔2B | 0.8B↔9B | 2B↔9B | 2B↔OLMO | 9B↔OLMO |
|--------|------|---------|---------|-------|---------|---------|
| brief0 | factual_recall | 0.678 | 0.707 | 0.800 | 0.739 | 0.918 |
| brief1 | algorithmic | 0.954 | 0.956 | 0.934 | 0.904 | 0.872 |
| brief2 | structural_copying | 0.320 | 0.562 | 0.571 | 0.399 | -0.181 |
| brief3 | syntactic_pattern | 0.166 | 0.346 | 0.001 | 0.193 | 0.204 |
| brief4 | cultural_memorized | 0.955 | 0.975 | 0.956 | 0.820 | 0.791 |
| brief5 | factual_recall | 0.872 | 0.670 | 0.890 | 0.932 | 0.965 |
| long0 | code_comprehension | 0.998 | 0.968 | 0.964 | 0.759 | 0.780 |
| long1 | long_range_retrieval | 0.392 | 0.481 | 0.931 | 0.656 | 0.663 |
| long2 | domain_knowledge | 0.240 | 0.528 | 0.790 | 0.232 | 0.476 |
| long3 | long_range_retrieval | 0.922 | 0.783 | 0.851 | 0.776 | 0.657 |
| med0 | factual_retrieval | 0.627 | 0.722 | 0.840 | 0.715 | 0.628 |
| med1 | reasoning_numerical | 0.193 | 0.757 | 0.209 | 0.551 | 0.490 |
| med2 | reasoning_tracking | 0.386 | 0.675 | 0.527 | 0.752 | 0.352 |
| med3 | reasoning_numerical | 0.975 | 0.934 | 0.970 | 0.867 | 0.821 |
| med4 | syntactic_pattern | 0.247 | 0.094 | 0.803 | 0.506 | 0.692 |
| med5 | reasoning_tracking | 0.585 | 0.473 | 0.832 | 0.650 | 0.739 |
| short0 | factual_recall | 0.313 | 0.755 | 0.413 | 0.706 | 0.379 |
| short1 | algorithmic | 0.939 | 0.961 | 0.953 | 0.977 | 0.984 |
| short2 | factual_recall | 0.695 | 0.325 | 0.720 | 0.498 | 0.448 |
| short3 | structural_copying | 0.755 | 0.776 | 0.944 | 0.754 | 0.767 |
| short4 | cultural_memorized | 0.557 | 0.326 | 0.279 | 0.420 | 0.945 |
| short5 | syntactic_pattern | 0.997 | 0.952 | 0.953 | 0.929 | 0.919 |

## 7. Per-Layer Sensitivity Analysis

Which attention layers matter most? We compare profiles that differ only in early vs late regions.

### 7.1 Early vs Late: Differential Sensitivity

Comparing `early_boost_1.5` vs `late_boost_1.5` and `early_suppress_0.5` vs `late_suppress_0.5`:

| Model | Prompt | Type | early_boost_1.5 prob | late_boost_1.5 prob | Δ(late-early) | early_sup_0.5 prob | late_sup_0.5 prob | Δ(late-early) |
|-------|--------|------|---------------------|--------------------|--------------|--------------------|----------------------|--------------|
| Qwen3.5-0.8B | brief0 | factual_recall | 0.2162 | 0.2004 | -0.0158 | 0.0488 | 0.0296 | -0.0192 |
| Qwen3.5-0.8B | brief1 | algorithmic | 0.9339 | 0.9051 | -0.0287 | 0.2957 | 0.9109 | +0.6153 |
| Qwen3.5-0.8B | brief2 | structural_copying | 0.2078 | 0.4537 | +0.2459 | 0.0074 | 0.1590 | +0.1516 |
| Qwen3.5-0.8B | brief3 | syntactic_pattern | 0.0039 | 0.0171 | +0.0132 | 0.0002 | 0.0073 | +0.0071 |
| Qwen3.5-0.8B | brief4 | cultural_memorized | 0.7183 | 0.9130 | +0.1946 | 0.0010 | 0.2365 | +0.2354 |
| Qwen3.5-0.8B | brief5 | factual_recall | 0.0773 | 0.5280 | +0.4506 | 0.0006 | 0.9330 | +0.9325 |
| Qwen3.5-0.8B | long0 | code_comprehension | 0.9774 | 0.9982 | +0.0208 | 0.0151 | 0.9414 | +0.9263 |
| Qwen3.5-0.8B | long1 | long_range_retrieval | 0.0685 | 0.0246 | -0.0440 | 0.0016 | 0.0180 | +0.0165 |
| Qwen3.5-0.8B | long2 | domain_knowledge | 0.0074 | 0.0071 | -0.0003 | 0.0034 | 0.0149 | +0.0115 |
| Qwen3.5-0.8B | long3 | long_range_retrieval | 0.2332 | 0.3518 | +0.1187 | 0.0000 | 0.2887 | +0.2887 |
| Qwen3.5-0.8B | med0 | factual_retrieval | 0.0025 | 0.0175 | +0.0149 | 0.0000 | 0.5812 | +0.5812 |
| Qwen3.5-0.8B | med1 | reasoning_numerical | 0.0003 | 0.0129 | +0.0126 | 0.0000 | 0.0006 | +0.0006 |
| Qwen3.5-0.8B | med2 | reasoning_tracking | 0.0070 | 0.0023 | -0.0047 | 0.0000 | 0.0035 | +0.0035 |
| Qwen3.5-0.8B | med3 | reasoning_numerical | 0.4441 | 0.9831 | +0.5389 | 0.1828 | 0.9056 | +0.7228 |
| Qwen3.5-0.8B | med4 | syntactic_pattern | 0.0005 | 0.0018 | +0.0014 | 0.0001 | 0.0212 | +0.0211 |
| Qwen3.5-0.8B | med5 | reasoning_tracking | 0.0214 | 0.0230 | +0.0016 | 0.0000 | 0.0050 | +0.0050 |
| Qwen3.5-0.8B | short0 | factual_recall | 0.0837 | 0.0471 | -0.0366 | 0.0000 | 0.0110 | +0.0110 |
| Qwen3.5-0.8B | short1 | algorithmic | 0.9389 | 0.8547 | -0.0842 | 0.8094 | 0.9531 | +0.1437 |
| Qwen3.5-0.8B | short2 | factual_recall | 0.0008 | 0.0018 | +0.0011 | 0.0001 | 0.0359 | +0.0358 |
| Qwen3.5-0.8B | short3 | structural_copying | 0.2834 | 0.5635 | +0.2801 | 0.2609 | 0.4937 | +0.2328 |
| Qwen3.5-0.8B | short4 | cultural_memorized | 0.0192 | 0.0135 | -0.0056 | 0.0002 | 0.0029 | +0.0026 |
| Qwen3.5-0.8B | short5 | syntactic_pattern | 0.9845 | 0.9863 | +0.0018 | 0.0022 | 0.8406 | +0.8384 |
| Qwen3.5-2B | brief0 | factual_recall | 0.3205 | 0.6688 | +0.3484 | 0.0320 | 0.6172 | +0.5853 |
| Qwen3.5-2B | brief1 | algorithmic | 0.9680 | 0.9547 | -0.0133 | 0.6040 | 0.8295 | +0.2255 |
| Qwen3.5-2B | brief2 | structural_copying | 0.2702 | 0.0426 | -0.2276 | 0.3125 | 0.2390 | -0.0734 |
| Qwen3.5-2B | brief3 | syntactic_pattern | 0.0229 | 0.0807 | +0.0578 | 0.0218 | 0.5717 | +0.5499 |
| Qwen3.5-2B | brief4 | cultural_memorized | 0.9447 | 0.9798 | +0.0351 | 0.0037 | 0.8131 | +0.8094 |
| Qwen3.5-2B | brief5 | factual_recall | 0.6280 | 0.7166 | +0.0886 | 0.1975 | 0.7738 | +0.5763 |
| Qwen3.5-2B | long0 | code_comprehension | 0.9970 | 0.9995 | +0.0025 | 0.0067 | 0.9694 | +0.9627 |
| Qwen3.5-2B | long1 | long_range_retrieval | 0.3887 | 0.8802 | +0.4915 | 0.0002 | 0.2056 | +0.2054 |
| Qwen3.5-2B | long2 | domain_knowledge | 0.0334 | 0.0318 | -0.0016 | 0.0022 | 0.0091 | +0.0069 |
| Qwen3.5-2B | long3 | long_range_retrieval | 0.5785 | 0.6597 | +0.0812 | 0.0000 | 0.8815 | +0.8815 |
| Qwen3.5-2B | med0 | factual_retrieval | 0.1834 | 0.7096 | +0.5262 | 0.0003 | 0.7465 | +0.7462 |
| Qwen3.5-2B | med1 | reasoning_numerical | 0.0024 | 0.0023 | -0.0001 | 0.0000 | 0.0010 | +0.0010 |
| Qwen3.5-2B | med2 | reasoning_tracking | 0.0317 | 0.0235 | -0.0082 | 0.0001 | 0.1072 | +0.1071 |
| Qwen3.5-2B | med3 | reasoning_numerical | 0.6398 | 0.9958 | +0.3560 | 0.1558 | 0.9678 | +0.8120 |
| Qwen3.5-2B | med4 | syntactic_pattern | 0.0401 | 0.3247 | +0.2846 | 0.0110 | 0.4082 | +0.3972 |
| Qwen3.5-2B | med5 | reasoning_tracking | 0.2214 | 0.6938 | +0.4724 | 0.0000 | 0.4258 | +0.4258 |
| Qwen3.5-2B | short0 | factual_recall | 0.0370 | 0.0244 | -0.0126 | 0.0002 | 0.1074 | +0.1073 |
| Qwen3.5-2B | short1 | algorithmic | 0.7937 | 0.8438 | +0.0501 | 0.9157 | 0.7499 | -0.1659 |
| Qwen3.5-2B | short2 | factual_recall | 0.0520 | 0.0029 | -0.0491 | 0.0001 | 0.0226 | +0.0226 |
| Qwen3.5-2B | short3 | structural_copying | 0.5367 | 0.7295 | +0.1928 | 0.3076 | 0.2558 | -0.0518 |
| Qwen3.5-2B | short4 | cultural_memorized | 0.0938 | 0.0982 | +0.0044 | 0.0005 | 0.0059 | +0.0054 |
| Qwen3.5-2B | short5 | syntactic_pattern | 0.9828 | 0.9976 | +0.0148 | 0.0672 | 0.9383 | +0.8711 |
| Qwen3.5-9B | brief0 | factual_recall | 0.5583 | 0.5708 | +0.0125 | 0.0181 | 0.2049 | +0.1869 |
| Qwen3.5-9B | brief1 | algorithmic | 0.9793 | 0.9688 | -0.0105 | 0.1950 | 0.6556 | +0.4606 |
| Qwen3.5-9B | brief2 | structural_copying | 0.4514 | 0.1220 | -0.3294 | 0.0109 | 0.0871 | +0.0762 |
| Qwen3.5-9B | brief3 | syntactic_pattern | 0.0039 | 0.0050 | +0.0010 | 0.0000 | 0.0003 | +0.0003 |
| Qwen3.5-9B | brief4 | cultural_memorized | 0.9422 | 0.9926 | +0.0505 | 0.0188 | 0.2338 | +0.2150 |
| Qwen3.5-9B | brief5 | factual_recall | 0.7330 | 0.5653 | -0.1677 | 0.0195 | 0.0221 | +0.0025 |
| Qwen3.5-9B | long0 | code_comprehension | 0.9872 | 0.9993 | +0.0121 | 0.0110 | 0.4904 | +0.4793 |
| Qwen3.5-9B | long1 | long_range_retrieval | 0.2130 | 0.9790 | +0.7660 | 0.0000 | 0.2024 | +0.2024 |
| Qwen3.5-9B | long2 | domain_knowledge | 0.0230 | 0.0369 | +0.0139 | 0.0000 | 0.0173 | +0.0173 |
| Qwen3.5-9B | long3 | long_range_retrieval | 0.8550 | 0.8937 | +0.0387 | 0.0000 | 0.3170 | +0.3170 |
| Qwen3.5-9B | med0 | factual_retrieval | 0.1098 | 0.3199 | +0.2101 | 0.0000 | 0.7449 | +0.7448 |
| Qwen3.5-9B | med1 | reasoning_numerical | 0.0019 | 0.0730 | +0.0712 | 0.0001 | 0.0002 | +0.0001 |
| Qwen3.5-9B | med2 | reasoning_tracking | 0.1239 | 0.0581 | -0.0658 | 0.0002 | 0.2039 | +0.2037 |
| Qwen3.5-9B | med3 | reasoning_numerical | 0.6300 | 0.9966 | +0.3667 | 0.0059 | 0.8202 | +0.8143 |
| Qwen3.5-9B | med4 | syntactic_pattern | 0.0473 | 0.7113 | +0.6640 | 0.0012 | 0.0909 | +0.0897 |
| Qwen3.5-9B | med5 | reasoning_tracking | 0.0770 | 0.9916 | +0.9146 | 0.0000 | 0.0549 | +0.0549 |
| Qwen3.5-9B | short0 | factual_recall | 0.2237 | 0.3355 | +0.1117 | 0.0005 | 0.1949 | +0.1945 |
| Qwen3.5-9B | short1 | algorithmic | 0.8670 | 0.9601 | +0.0931 | 0.7632 | 0.6814 | -0.0818 |
| Qwen3.5-9B | short2 | factual_recall | 0.5646 | 0.0281 | -0.5365 | 0.0092 | 0.1678 | +0.1585 |
| Qwen3.5-9B | short3 | structural_copying | 0.5548 | 0.9042 | +0.3494 | 0.0771 | 0.2501 | +0.1730 |
| Qwen3.5-9B | short4 | cultural_memorized | 0.2513 | 0.7341 | +0.4828 | 0.0221 | 0.1535 | +0.1314 |
| Qwen3.5-9B | short5 | syntactic_pattern | 0.9792 | 0.9974 | +0.0182 | 0.0603 | 0.9930 | +0.9327 |
| OLMo-Hybrid-7B | brief0 | factual_recall | 0.5489 | 0.7794 | +0.2305 | 0.0247 | 0.1502 | +0.1255 |
| OLMo-Hybrid-7B | brief1 | algorithmic | 0.9187 | 0.9179 | -0.0008 | 0.5680 | 0.5892 | +0.0212 |
| OLMo-Hybrid-7B | brief2 | structural_copying | 0.0771 | 0.1035 | +0.0264 | 0.4866 | 0.1559 | -0.3307 |
| OLMo-Hybrid-7B | brief3 | syntactic_pattern | 0.2649 | 0.2069 | -0.0580 | 0.0055 | 0.1112 | +0.1057 |
| OLMo-Hybrid-7B | brief4 | cultural_memorized | 0.9932 | 0.9938 | +0.0006 | 0.5000 | 0.6698 | +0.1699 |
| OLMo-Hybrid-7B | brief5 | factual_recall | 0.6800 | 0.6958 | +0.0158 | 0.1403 | 0.2862 | +0.1459 |
| OLMo-Hybrid-7B | long0 | code_comprehension | 0.1605 | 0.4410 | +0.2805 | 0.0347 | 0.1547 | +0.1200 |
| OLMo-Hybrid-7B | long1 | long_range_retrieval | 0.0207 | 0.0527 | +0.0321 | 0.0019 | 0.0193 | +0.0174 |
| OLMo-Hybrid-7B | long2 | domain_knowledge | 0.0188 | 0.0146 | -0.0042 | 0.0016 | 0.0984 | +0.0968 |
| OLMo-Hybrid-7B | long3 | long_range_retrieval | 0.0413 | 0.0295 | -0.0118 | 0.0008 | 0.0511 | +0.0503 |
| OLMo-Hybrid-7B | med0 | factual_retrieval | 0.0245 | 0.0446 | +0.0201 | 0.0003 | 0.2213 | +0.2210 |
| OLMo-Hybrid-7B | med1 | reasoning_numerical | 0.0011 | 0.0010 | -0.0000 | 0.0002 | 0.0008 | +0.0006 |
| OLMo-Hybrid-7B | med2 | reasoning_tracking | 0.0330 | 0.0165 | -0.0166 | 0.0008 | 0.0643 | +0.0635 |
| OLMo-Hybrid-7B | med3 | reasoning_numerical | 0.3081 | 0.6251 | +0.3170 | 0.1509 | 0.2202 | +0.0693 |
| OLMo-Hybrid-7B | med4 | syntactic_pattern | 0.0774 | 0.2807 | +0.2032 | 0.0000 | 0.0319 | +0.0319 |
| OLMo-Hybrid-7B | med5 | reasoning_tracking | 0.9206 | 0.9274 | +0.0067 | 0.0035 | 0.1246 | +0.1211 |
| OLMo-Hybrid-7B | short0 | factual_recall | 0.2562 | 0.1370 | -0.1192 | 0.0448 | 0.0882 | +0.0434 |
| OLMo-Hybrid-7B | short1 | algorithmic | 0.8861 | 0.9311 | +0.0450 | 0.8487 | 0.8308 | -0.0179 |
| OLMo-Hybrid-7B | short2 | factual_recall | 0.1180 | 0.0216 | -0.0964 | 0.1924 | 0.6036 | +0.4112 |
| OLMo-Hybrid-7B | short3 | structural_copying | 0.5545 | 0.7493 | +0.1948 | 0.1361 | 0.2969 | +0.1608 |
| OLMo-Hybrid-7B | short4 | cultural_memorized | 0.2471 | 0.7244 | +0.4773 | 0.1266 | 0.0611 | -0.0655 |
| OLMo-Hybrid-7B | short5 | syntactic_pattern | 0.9543 | 0.9770 | +0.0227 | 0.4428 | 0.3188 | -0.1240 |

### 7.2 Ablation: Which Layers Can Be Zeroed?

The extreme profiles (`early_only_2x`, `late_only_2x`, `middle_only`) zero out half the attention layers. How much prediction survives?

| Model | Prompt | Type | BL prob | early_only_2x | late_only_2x | middle_only | alternating | alt_inv |
|-------|--------|------|---------|--------------|-------------|------------|------------|---------|
| Qwen3.5-0.8B | brief0 | factual_recall | 0.1768 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | brief1 | algorithmic | 0.9172 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | brief2 | structural_copying | 0.3426 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | brief3 | syntactic_pattern | 0.0168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | brief4 | cultural_memorized | 0.9688 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | brief5 | factual_recall | 0.8655 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | long0 | code_comprehension | 0.9935 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | long1 | long_range_retrieval | 0.0180 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | long2 | domain_knowledge | 0.0098 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | long3 | long_range_retrieval | 0.5039 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | med0 | factual_retrieval | 0.1130 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | med1 | reasoning_numerical | 0.0144 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | med2 | reasoning_tracking | 0.0048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | med3 | reasoning_numerical | 0.9518 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | med4 | syntactic_pattern | 0.0056 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | med5 | reasoning_tracking | 0.0561 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | short0 | factual_recall | 0.0444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | short1 | algorithmic | 0.8718 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | short2 | factual_recall | 0.1170 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | short3 | structural_copying | 0.5296 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | short4 | cultural_memorized | 0.0061 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B | short5 | syntactic_pattern | 0.9965 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | brief0 | factual_recall | 0.5922 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | brief1 | algorithmic | 0.9445 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | brief2 | structural_copying | 0.1172 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | brief3 | syntactic_pattern | 0.0739 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | brief4 | cultural_memorized | 0.9863 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | brief5 | factual_recall | 0.8700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | long0 | code_comprehension | 0.9983 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | long1 | long_range_retrieval | 0.7201 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | long2 | domain_knowledge | 0.0334 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | long3 | long_range_retrieval | 0.7245 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | med0 | factual_retrieval | 0.7688 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | med1 | reasoning_numerical | 0.0049 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | med2 | reasoning_tracking | 0.0552 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | med3 | reasoning_numerical | 0.9795 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | med4 | syntactic_pattern | 0.4807 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | med5 | reasoning_tracking | 0.7132 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | short0 | factual_recall | 0.0597 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | short1 | algorithmic | 0.7592 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | short2 | factual_recall | 0.1465 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | short3 | structural_copying | 0.5325 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | short4 | cultural_memorized | 0.0250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | short5 | syntactic_pattern | 0.9988 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | brief0 | factual_recall | 0.3411 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | brief1 | algorithmic | 0.9514 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | brief2 | structural_copying | 0.2064 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | brief3 | syntactic_pattern | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | brief4 | cultural_memorized | 0.9987 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | brief5 | factual_recall | 0.8212 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | long0 | code_comprehension | 0.9966 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | long1 | long_range_retrieval | 0.8098 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | long2 | domain_knowledge | 0.0220 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | long3 | long_range_retrieval | 0.7553 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | med0 | factual_retrieval | 0.2711 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | med1 | reasoning_numerical | 0.0544 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | med2 | reasoning_tracking | 0.1840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | med3 | reasoning_numerical | 0.9778 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | med4 | syntactic_pattern | 0.5414 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | med5 | reasoning_tracking | 0.9894 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | short0 | factual_recall | 0.2608 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | short1 | algorithmic | 0.8823 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | short2 | factual_recall | 0.3232 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | short3 | structural_copying | 0.6090 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | short4 | cultural_memorized | 0.7494 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | short5 | syntactic_pattern | 0.9995 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | brief0 | factual_recall | 0.5048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | brief1 | algorithmic | 0.8935 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | brief2 | structural_copying | 0.1446 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | brief3 | syntactic_pattern | 0.6035 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | brief4 | cultural_memorized | 0.9968 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | brief5 | factual_recall | 0.9239 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | long0 | code_comprehension | 0.5155 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | long1 | long_range_retrieval | 0.0266 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | long2 | domain_knowledge | 0.0150 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | long3 | long_range_retrieval | 0.0788 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | med0 | factual_retrieval | 0.2527 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | med1 | reasoning_numerical | 0.0038 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | med2 | reasoning_tracking | 0.0789 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | med3 | reasoning_numerical | 0.4961 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | med4 | syntactic_pattern | 0.1789 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | med5 | reasoning_tracking | 0.9202 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | short0 | factual_recall | 0.2701 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | short1 | algorithmic | 0.9011 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | short2 | factual_recall | 0.3441 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | short3 | structural_copying | 0.5456 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | short4 | cultural_memorized | 0.7182 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OLMo-Hybrid-7B | short5 | syntactic_pattern | 0.9750 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 8. Advanced Signal Extraction

### 8.1 Signal Space: Baseline Fingerprints

Each prompt at baseline occupies a point in a multi-dimensional signal space. Key dimensions:

| Model | Prompt | Type | Final H | Mean H | Attn Grad | Attn Last | Head Var | Sharpest L |
|-------|--------|------|---------|--------|-----------|-----------|----------|-----------|
| Qwen3.5-0.8B | brief0 | factual_recall | 2.58 | 3.03 | +0.182 | 4.09 | 1.424 | L15 |
| Qwen3.5-0.8B | brief1 | algorithmic | 0.82 | 1.17 | -1.039 | 3.80 | 1.498 | L23 |
| Qwen3.5-0.8B | brief2 | structural_copying | 5.19 | 4.36 | -0.042 | 3.77 | 0.859 | L23 |
| Qwen3.5-0.8B | brief3 | syntactic_pattern | 2.57 | 0.93 | -0.093 | 3.71 | 2.047 | L15 |
| Qwen3.5-0.8B | brief4 | cultural_memorized | 0.22 | 1.29 | -0.151 | 4.16 | 1.659 | L15 |
| Qwen3.5-0.8B | brief5 | factual_recall | 1.31 | 2.36 | -0.430 | 3.53 | 1.327 | L15 |
| Qwen3.5-0.8B | long0 | code_comprehension | 0.08 | 1.26 | -0.118 | 4.85 | 1.783 | L15 |
| Qwen3.5-0.8B | long1 | long_range_retrieval | 9.61 | 4.26 | +0.042 | 4.91 | 2.918 | L15 |
| Qwen3.5-0.8B | long2 | domain_knowledge | 6.94 | 3.95 | +0.556 | 5.68 | 2.799 | L23 |
| Qwen3.5-0.8B | long3 | long_range_retrieval | 4.71 | 3.59 | -0.687 | 4.08 | 2.887 | L15 |
| Qwen3.5-0.8B | med0 | factual_retrieval | 5.07 | 4.43 | -0.608 | 4.33 | 2.383 | L15 |
| Qwen3.5-0.8B | med1 | reasoning_numerical | 8.83 | 4.24 | -0.479 | 4.26 | 1.847 | L23 |
| Qwen3.5-0.8B | med2 | reasoning_tracking | 3.83 | 4.10 | -0.447 | 4.44 | 1.892 | L23 |
| Qwen3.5-0.8B | med3 | reasoning_numerical | 0.40 | 2.82 | -0.652 | 4.78 | 2.007 | L15 |
| Qwen3.5-0.8B | med4 | syntactic_pattern | 7.50 | 0.74 | -0.924 | 4.81 | 2.136 | L23 |
| Qwen3.5-0.8B | med5 | reasoning_tracking | 4.04 | 4.37 | -0.595 | 4.31 | 1.774 | L23 |
| Qwen3.5-0.8B | short0 | factual_recall | 5.17 | 4.61 | -0.100 | 1.67 | 0.326 | L23 |
| Qwen3.5-0.8B | short1 | algorithmic | 1.05 | 1.80 | -0.589 | 3.44 | 0.774 | L19 |
| Qwen3.5-0.8B | short2 | factual_recall | 6.49 | 7.01 | -0.358 | 1.22 | 0.291 | L23 |
| Qwen3.5-0.8B | short3 | structural_copying | 4.48 | 6.07 | -0.317 | 1.89 | 0.320 | L15 |
| Qwen3.5-0.8B | short4 | cultural_memorized | 9.53 | 4.87 | +0.282 | 2.67 | 0.568 | L7 |
| Qwen3.5-0.8B | short5 | syntactic_pattern | 0.04 | 1.75 | +0.099 | 1.88 | 0.416 | L7 |
| Qwen3.5-2B | brief0 | factual_recall | 2.15 | 2.62 | -0.078 | 3.98 | 1.401 | L15 |
| Qwen3.5-2B | brief1 | algorithmic | 0.59 | 1.05 | -0.688 | 4.09 | 1.298 | L15 |
| Qwen3.5-2B | brief2 | structural_copying | 4.30 | 3.98 | -0.031 | 4.08 | 0.890 | L23 |
| Qwen3.5-2B | brief3 | syntactic_pattern | 1.70 | 0.83 | +0.066 | 4.19 | 2.040 | L11 |
| Qwen3.5-2B | brief4 | cultural_memorized | 0.11 | 0.76 | -0.326 | 4.27 | 1.292 | L15 |
| Qwen3.5-2B | brief5 | factual_recall | 1.19 | 2.14 | -0.543 | 3.41 | 1.530 | L15 |
| Qwen3.5-2B | long0 | code_comprehension | 0.03 | 1.02 | -0.196 | 4.70 | 1.548 | L7 |
| Qwen3.5-2B | long1 | long_range_retrieval | 3.28 | 3.51 | -0.533 | 4.44 | 2.460 | L23 |
| Qwen3.5-2B | long2 | domain_knowledge | 6.01 | 3.41 | +0.301 | 5.48 | 2.853 | L23 |
| Qwen3.5-2B | long3 | long_range_retrieval | 2.90 | 3.21 | -0.494 | 4.30 | 3.397 | L15 |
| Qwen3.5-2B | med0 | factual_retrieval | 2.14 | 4.13 | -0.868 | 4.29 | 2.430 | L15 |
| Qwen3.5-2B | med1 | reasoning_numerical | 8.46 | 4.03 | -0.850 | 4.12 | 1.918 | L23 |
| Qwen3.5-2B | med2 | reasoning_tracking | 3.20 | 3.51 | -0.628 | 4.41 | 1.946 | L23 |
| Qwen3.5-2B | med3 | reasoning_numerical | 0.19 | 2.54 | -0.635 | 5.00 | 2.091 | L19 |
| Qwen3.5-2B | med4 | syntactic_pattern | 3.94 | 0.61 | -0.729 | 4.70 | 1.698 | L23 |
| Qwen3.5-2B | med5 | reasoning_tracking | 1.81 | 3.87 | -0.855 | 4.24 | 1.783 | L23 |
| Qwen3.5-2B | short0 | factual_recall | 5.05 | 4.20 | -0.211 | 1.52 | 0.273 | L23 |
| Qwen3.5-2B | short1 | algorithmic | 1.70 | 1.74 | -0.176 | 3.61 | 0.510 | L15 |
| Qwen3.5-2B | short2 | factual_recall | 4.94 | 6.27 | -0.127 | 1.57 | 0.207 | L23 |
| Qwen3.5-2B | short3 | structural_copying | 4.20 | 5.46 | -0.397 | 2.00 | 0.410 | L15 |
| Qwen3.5-2B | short4 | cultural_memorized | 9.29 | 4.96 | +0.228 | 2.85 | 0.445 | L3 |
| Qwen3.5-2B | short5 | syntactic_pattern | 0.02 | 1.74 | -0.093 | 1.77 | 0.404 | L11 |
| Qwen3.5-9B | brief0 | factual_recall | 1.80 | 2.23 | -0.620 | 4.11 | 1.026 | L19 |
| Qwen3.5-9B | brief1 | algorithmic | 0.53 | 0.95 | -0.670 | 4.11 | 1.014 | L19 |
| Qwen3.5-9B | brief2 | structural_copying | 3.76 | 3.55 | -0.363 | 3.95 | 0.731 | L31 |
| Qwen3.5-9B | brief3 | syntactic_pattern | 0.73 | 0.84 | -0.503 | 3.99 | 1.692 | L15 |
| Qwen3.5-9B | brief4 | cultural_memorized | 0.02 | 0.58 | -0.823 | 4.38 | 1.224 | L19 |
| Qwen3.5-9B | brief5 | factual_recall | 1.49 | 1.90 | -0.551 | 3.82 | 1.228 | L27 |
| Qwen3.5-9B | long0 | code_comprehension | 0.05 | 0.74 | -0.205 | 4.66 | 1.836 | L11 |
| Qwen3.5-9B | long1 | long_range_retrieval | 2.27 | 2.72 | -0.528 | 4.64 | 2.587 | L11 |
| Qwen3.5-9B | long2 | domain_knowledge | 5.11 | 2.66 | +0.046 | 5.89 | 1.777 | L31 |
| Qwen3.5-9B | long3 | long_range_retrieval | 2.16 | 2.75 | -0.912 | 4.81 | 3.070 | L19 |
| Qwen3.5-9B | med0 | factual_retrieval | 3.57 | 3.70 | -0.446 | 4.15 | 1.829 | L31 |
| Qwen3.5-9B | med1 | reasoning_numerical | 8.06 | 3.62 | -0.606 | 4.50 | 1.300 | L31 |
| Qwen3.5-9B | med2 | reasoning_tracking | 4.34 | 2.90 | -0.848 | 4.09 | 1.645 | L31 |
| Qwen3.5-9B | med3 | reasoning_numerical | 0.23 | 2.27 | -0.464 | 4.95 | 1.659 | L7 |
| Qwen3.5-9B | med4 | syntactic_pattern | 2.91 | 0.49 | -0.913 | 4.66 | 1.610 | L31 |
| Qwen3.5-9B | med5 | reasoning_tracking | 0.12 | 3.00 | -0.601 | 5.21 | 1.878 | L19 |
| Qwen3.5-9B | short0 | factual_recall | 4.58 | 4.22 | -0.232 | 1.76 | 0.253 | L27 |
| Qwen3.5-9B | short1 | algorithmic | 0.99 | 1.48 | -0.199 | 3.34 | 0.687 | L27 |
| Qwen3.5-9B | short2 | factual_recall | 4.59 | 6.07 | -0.199 | 1.45 | 0.214 | L27 |
| Qwen3.5-9B | short3 | structural_copying | 3.02 | 5.13 | -0.311 | 2.16 | 0.396 | L19 |
| Qwen3.5-9B | short4 | cultural_memorized | 1.65 | 3.33 | -0.047 | 2.95 | 0.353 | L19 |
| Qwen3.5-9B | short5 | syntactic_pattern | 0.01 | 1.71 | -0.267 | 1.83 | 0.289 | L31 |
| OLMo-Hybrid-7B | brief0 | factual_recall | 1.80 | 2.69 | -1.397 | 2.83 | 2.596 | L23 |
| OLMo-Hybrid-7B | brief1 | algorithmic | 1.08 | 1.33 | -1.066 | 2.93 | 2.106 | L31 |
| OLMo-Hybrid-7B | brief2 | structural_copying | 4.22 | 3.61 | -1.054 | 2.06 | 2.355 | L31 |
| OLMo-Hybrid-7B | brief3 | syntactic_pattern | 2.04 | 2.09 | -1.357 | 3.36 | 2.541 | L31 |
| OLMo-Hybrid-7B | brief4 | cultural_memorized | 0.04 | 0.60 | -1.669 | 1.72 | 2.935 | L27 |
| OLMo-Hybrid-7B | brief5 | factual_recall | 0.77 | 2.14 | -1.453 | 2.82 | 2.371 | L27 |
| OLMo-Hybrid-7B | long0 | code_comprehension | 2.35 | 1.66 | -0.517 | 6.37 | 3.748 | L23 |
| OLMo-Hybrid-7B | long1 | long_range_retrieval | 9.82 | 3.62 | -1.248 | 5.77 | 5.302 | L27 |
| OLMo-Hybrid-7B | long2 | domain_knowledge | 4.72 | 3.25 | +0.137 | 7.36 | 4.301 | L19 |
| OLMo-Hybrid-7B | long3 | long_range_retrieval | 6.67 | 3.27 | -0.583 | 6.64 | 4.821 | L23 |
| OLMo-Hybrid-7B | med0 | factual_retrieval | 3.22 | 4.28 | -1.154 | 3.34 | 3.435 | L27 |
| OLMo-Hybrid-7B | med1 | reasoning_numerical | 9.43 | 4.28 | -0.812 | 3.45 | 3.720 | L27 |
| OLMo-Hybrid-7B | med2 | reasoning_tracking | 6.06 | 3.48 | -1.446 | 3.25 | 3.864 | L27 |
| OLMo-Hybrid-7B | med3 | reasoning_numerical | 2.39 | 3.40 | -0.732 | 4.84 | 3.346 | L27 |
| OLMo-Hybrid-7B | med4 | syntactic_pattern | 7.07 | 1.51 | -0.641 | 5.74 | 3.029 | L27 |
| OLMo-Hybrid-7B | med5 | reasoning_tracking | 0.59 | 3.55 | -0.897 | 3.90 | 3.670 | L27 |
| OLMo-Hybrid-7B | short0 | factual_recall | 3.94 | 4.77 | -0.741 | 0.67 | 0.640 | L31 |
| OLMo-Hybrid-7B | short1 | algorithmic | 0.84 | 1.54 | -1.276 | 1.58 | 1.948 | L27 |
| OLMo-Hybrid-7B | short2 | factual_recall | 5.19 | 6.66 | -0.062 | 1.41 | 0.261 | L31 |
| OLMo-Hybrid-7B | short3 | structural_copying | 4.40 | 5.00 | -0.951 | 0.57 | 0.793 | L31 |
| OLMo-Hybrid-7B | short4 | cultural_memorized | 3.39 | 3.27 | -1.117 | 1.42 | 1.425 | L23 |
| OLMo-Hybrid-7B | short5 | syntactic_pattern | 0.28 | 1.37 | -0.624 | 0.82 | 0.753 | L23 |

### 8.2 Mutual Information: Signal → Optimal Profile Family

We discretize baseline signals into tertiles and compute mutual information with the best profile family.

| Signal | MI (bits) with best profile family | MI with best_g_mean (3 bins) | MI with best_g_tilt (3 bins) |
|--------|----------------------------------|-------------------------------|------------------------------|
| final_entropy | 0.1929 | 0.0368 | 0.0206 |
| mean_entropy | 0.0993 | 0.0074 | 0.0048 |
| attn_ent_gradient | 0.0702 | 0.0144 | 0.0482 |
| attn_ent_last | 0.1370 | 0.0239 | 0.0188 |
| attn_ent_head_var | 0.1288 | 0.0458 | 0.0454 |
| attn_ent_min_head | 0.0724 | 0.0098 | 0.0385 |
| attn_ent_std_across_layers | 0.0806 | 0.0583 | 0.0464 |

### 8.3 Topological Data Analysis: Persistent Homology of Response Surfaces

We treat each model's 22 prompts as points in 31-dimensional space (target_prob per g-profile) and compute persistent homology to identify topological features — connected components (H0) and loops (H1) — in the response landscape.

#### Qwen3.5-0.8B

**H0 (connected components)**: 22 features
- Lifetimes: min=0.104, median=2.142, max=7.025
- The gap between the longest-lived finite H0 feature and the persistent component indicates how cleanly the prompts separate into clusters.

**H1 (loops)**: 1 features
- Lifetimes: min=0.699, median=0.699, max=0.699
- Long-lived H1 features indicate non-trivial topology: there exist closed paths through response-surface space that cannot be contracted, suggesting qualitatively distinct regimes.

#### Qwen3.5-2B

**H0 (connected components)**: 22 features
- Lifetimes: min=0.319, median=2.280, max=5.698
- The gap between the longest-lived finite H0 feature and the persistent component indicates how cleanly the prompts separate into clusters.

**H1 (loops)**: No features detected.

#### Qwen3.5-9B

**H0 (connected components)**: 22 features
- Lifetimes: min=0.323, median=2.797, max=8.457
- The gap between the longest-lived finite H0 feature and the persistent component indicates how cleanly the prompts separate into clusters.

**H1 (loops)**: 2 features
- Lifetimes: min=0.165, median=0.208, max=0.251
- Long-lived H1 features indicate non-trivial topology: there exist closed paths through response-surface space that cannot be contracted, suggesting qualitatively distinct regimes.

#### OLMo-Hybrid-7B

**H0 (connected components)**: 22 features
- Lifetimes: min=0.399, median=2.649, max=6.783
- The gap between the longest-lived finite H0 feature and the persistent component indicates how cleanly the prompts separate into clusters.

**H1 (loops)**: No features detected.

### 8.4 KL Divergence from Baseline Across Profiles

How much does each g-profile perturb the output distribution? (Using KL divergence where available, entropy as proxy where not.)

KL from baseline available for 2640/2640 non-baseline records.

Mean KL from baseline by profile family:

- **ablation**: mean=9.912, std=1.952, max=12.417
- **alternating**: mean=9.912, std=1.952, max=12.417
- **cross**: mean=2.339, std=2.566, max=10.887
- **early_regional**: mean=1.270, std=2.003, max=14.074
- **edges**: mean=0.266, std=0.461, max=3.295
- **late_regional**: mean=0.272, std=0.447, max=3.487
- **middle**: mean=2.728, std=4.285, max=12.417
- **ramp_down**: mean=0.289, std=0.557, max=4.561
- **ramp_up**: mean=0.237, std=0.429, max=3.732
- **uniform**: mean=4.016, std=4.423, max=21.486

## 9. Attention Entropy Under Perturbation

How does the attention entropy profile change when we apply different g-profiles? This reveals which layers' attention is *reactive* vs *inert* to gain scaling.

### 9.1 Attention Entropy Shift (Qwen3.5-2B, selected profiles)

| Prompt | Profile | L3 | L7 | L11 | L15 | L19 | L23 |
|--------|---------|------|------|------|------|------|------|
| short0 | baseline | 2.09 | 2.14 | 2.01 | 2.04 | 2.05 | 1.52 |
| short0 | constant_0.5 | 2.09 | 2.20 | 1.95 | 1.26 | 1.91 | 1.59 |
| short0 | constant_1.5 | 2.09 | 2.07 | 1.99 | 1.97 | 1.98 | 1.20 |
| short0 | ramp_up | 2.09 | 2.24 | 1.99 | 1.96 | 2.46 | 1.72 |
| short0 | late_boost_1.5 | 2.09 | 2.14 | 2.01 | 2.04 | 2.02 | 1.45 |
| short0 | early_only_2x | 2.09 | 2.05 | 1.76 | 1.82 | 2.81 | 2.81 |
| short5 | baseline | 2.18 | 1.77 | 1.59 | 1.63 | 1.87 | 1.77 |
| short5 | constant_0.5 | 2.18 | 1.86 | 1.56 | 1.48 | 1.97 | 1.27 |
| short5 | constant_1.5 | 2.18 | 1.80 | 1.57 | 1.62 | 2.22 | 1.37 |
| short5 | ramp_up | 2.18 | 1.94 | 1.90 | 1.54 | 1.74 | 1.73 |
| short5 | late_boost_1.5 | 2.18 | 1.77 | 1.59 | 1.63 | 1.96 | 1.63 |
| short5 | early_only_2x | 2.18 | 1.88 | 1.67 | 1.42 | 2.81 | 2.81 |
| brief2 | baseline | 4.78 | 4.38 | 4.76 | 4.72 | 5.04 | 4.08 |
| brief2 | constant_0.5 | 4.78 | 4.66 | 4.91 | 4.57 | 4.23 | 4.32 |
| brief2 | constant_1.5 | 4.78 | 4.32 | 4.92 | 4.59 | 4.96 | 3.86 |
| brief2 | ramp_up | 4.78 | 4.60 | 4.69 | 4.65 | 4.94 | 3.90 |
| brief2 | late_boost_1.5 | 4.78 | 4.38 | 4.76 | 4.72 | 5.01 | 3.88 |
| brief2 | early_only_2x | 4.78 | 4.48 | 4.97 | 4.42 | 6.09 | 6.09 |
| med0 | baseline | 5.93 | 5.40 | 4.67 | 3.80 | 5.32 | 4.29 |
| med0 | constant_0.5 | 5.93 | 5.74 | 5.79 | 5.36 | 5.81 | 5.42 |
| med0 | constant_1.5 | 5.93 | 5.15 | 4.85 | 4.54 | 5.48 | 4.30 |
| med0 | ramp_up | 5.93 | 5.69 | 5.11 | 3.48 | 4.77 | 3.80 |
| med0 | late_boost_1.5 | 5.93 | 5.40 | 4.67 | 3.80 | 5.32 | 4.04 |
| med0 | early_only_2x | 5.93 | 5.09 | 5.40 | 5.09 | 7.80 | 7.80 |
| long1 | baseline | 6.98 | 4.91 | 4.86 | 4.74 | 5.96 | 4.44 |
| long1 | constant_0.5 | 6.98 | 6.25 | 6.96 | 4.73 | 7.15 | 7.25 |
| long1 | constant_1.5 | 6.98 | 4.53 | 4.90 | 4.66 | 5.20 | 3.65 |
| long1 | ramp_up | 6.98 | 5.75 | 5.02 | 5.27 | 5.67 | 4.21 |
| long1 | late_boost_1.5 | 6.98 | 4.91 | 4.86 | 4.74 | 5.74 | 3.82 |
| long1 | early_only_2x | 6.98 | 4.32 | 5.30 | 5.08 | 9.31 | 9.31 |

## 10. Synthesis: What Signal Is There?

### 10.1 Headroom Summary

- **90%** of model×prompt pairs have a non-baseline g-profile that improves target probability.
- Mean improvement when one exists: **+0.1454** in target probability.
- The improvement is not small: median best-profile Δprob is **0.0835**, and several cases show >0.5 absolute improvement.
- Damage from bad profiles is severe: worst profiles frequently drive target probability near zero.

### 10.2 Signal Quality Assessment

**What works:**

- **Baseline target probability** (bl_prob) is the strongest single predictor of headroom (r = −0.33, p = 0.002). Prompts the model already handles well have less room for improvement. This is obvious but useful as a triage signal.
- **Final-token entropy** correlates with headroom via Spearman rank (ρ = 0.35, p < 0.001) but not Pearson — the relationship is monotonic but nonlinear.
- **Response surface shape** is low-dimensional: the first 3 PCs explain 60–75% of variance in how prompts respond to g-perturbation. This means the effective dimensionality of the control problem is much lower than 31.

**What doesn't work (yet):**

- **Attention entropy gradient** (early-vs-late) does not reliably predict whether the optimal profile tilts early or late (r ≈ 0 for most models). The one exception is OLMO (r = −0.43, p = 0.04) — worth investigating whether this reflects a genuine architectural difference.
- **Per-layer attention entropy** features show weak individual correlations with the optimal g-profile parameters. The signal is there in aggregate (the clustering works) but no single scalar extracts it.
- **Mutual information** between discretized signals and optimal profile family is low across the board, suggesting that simple binning loses too much information.

### 10.3 Implications for Signal-Guided Control

1. **A simple entropy-threshold controller can avoid catastrophes.** The worst profiles are the extreme ones (ablation, alternating). A controller that stays within the "moderate" region (g ∈ [0.5, 1.5]) already avoids most damage.

2. **The response surface is clusterable.** Prompts group into 2–3 response-surface types that are partially (but not perfectly) aligned with the hand-labeled completion types. A controller that can identify the response-surface cluster from baseline signals gets most of the available headroom.

3. **The single-agent ceiling may be modest but real.** With 22 prompts, the correlations are noisy. Expanding to 200+ prompts (per the experiment plan) will either reveal stronger signal or confirm that single-agent signal-to-g mapping hits a wall — both are informative.

4. **Cross-model consistency is mixed.** Some prompts (especially syntactic and algorithmic) show highly correlated response surfaces across all 4 models. Others (factual recall, reasoning) are model-specific. A universal controller is unlikely; a per-model-family controller is more realistic.

5. **The attention entropy profile is an underexploited signal.** It has 6–8 dimensions (one per attention layer) and captures within-model information about which layers are doing work. Current scalar summaries (gradient, mean) lose too much. The next step should use the full layer-means vector as input to a small classifier or nearest-neighbor lookup.

### 10.4 Recommended Next Steps for Experiment 1

1. **Expand battery to 200+ prompts** with known types (COUNTERFACT, MIB). The current 22 are insufficient for learning a mapping.
2. **Use the full attention-entropy vector** (6–8 dims) as the input signal, not scalar summaries.
3. **Train a simple nearest-neighbor or small MLP** on the expanded battery: input = (final_entropy, mean_entropy, attn_ent_vector), output = best g-profile family (or best g_mean + g_tilt).
4. **Evaluate with leave-one-out or bootstrap CIs** per task category.
5. **Establish the single-agent ceiling**: what fraction of the oracle headroom can signal-guided selection capture? This is the number the colony must beat.
