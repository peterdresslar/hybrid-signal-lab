# Attention Is In The Air — Capstone Working Document

Peter Dresslar, CAS Capstone, ASU. Advisor: Bryan Daniels. March 2026.

This document tracks the current state of the project. For theoretical framing see proposal.md. For benchmark design see benchmark_strategy.md.

## What we know so far

We can scale the residual contribution of attention layers in hybrid transformer models (Qwen3.5 and OLMo Hybrid) using per-layer gain factors, which we call g-profiles. Forward hooks multiply each attention layer's output by a scalar before it enters the residual stream. This is cheap, deterministic, and non-destructive.

Sweeping g-profiles across a battery of prompts produces response surfaces that are non-monotonic, task-dependent, and model-dependent. We have sweep data from Qwen3.5 (0.8B, 2B, 9B) and OLMo Hybrid (7B), covering 22 prompts across 33 g-profile configurations on OLMo and 23 prompts across 25 configurations on Qwen.

The main findings:

Different prompts have qualitatively different response surfaces. Some have wide flat basins where the model is robust to attention perturbation. Some have narrow optima where attention scaling is critical. Some have multiple peaks suggesting two mechanisms can independently produce the answer.

The shape of the response surface correlates with the semantic type of the prompt, but is not fully determined by it. Factual recall prompts tend to have narrow basins. Syntactic patterns tend to have wide basins. Reasoning tasks tend to have complex, multi-modal surfaces. There is within-type variation that suggests the real computational signature is finer-grained than any human-readable classification.

Early-vs-late attention layer asymmetry is task-dependent. Some prompts improve when early attention is boosted and late attention is suppressed; others show the opposite. This is consistent with the OLMo Hybrid quantization model (Merrill et al., 2026), where attention layers perform discrete quantization of the continuous state maintained by recurrent layers, and different tasks resolve at different depths.

The taxonomy holds across architectures. Qwen3.5 and OLMo Hybrid show qualitatively similar response surface shapes for the same prompt types despite different training data and model families. This suggests the phenomena are architectural.

Some specific numbers: on OLMo 7B, early_high_late_low improves factual recall prompt short0 from p=0.27 to p=0.64. Structural copying prompt brief2 goes from p=0.14 to p=0.68 with ramp_up. Syntactic pattern prompts degrade under any perturbation. At g=0.0, final-token entropy collapses to ~16.6 bits (approximately log2 of the vocabulary size), confirming near-uniform output. The constant_3.0 profile produces NaN on OLMo, so the usable range is roughly g in [0.0, 2.0].

The upshot: a per-layer g-profile is a lightweight intervention that can meaningfully alter model predictions for better or worse depending on the prompt. The question is whether an external signal can guide the choice of g-profile without knowing in advance what task the model is facing.

## The three experiments

### Experiment 1: Single-agent signal-guided g-profile selection

Status: partially complete. Sweep data collected. Signal extraction and closed-loop control not yet built.

The sweep data shows that better-than-baseline g-profiles exist for most prompts. Experiment 1 asks whether we can build a signal-to-g-profile mapping that works in practice.

The approach: run a single forward pass at baseline (g=1.0) and extract lightweight scalar signals — final-token entropy, per-layer attention entropy, KL divergence from a running baseline. Use the sweep data as ground truth to learn a mapping from signal vector to optimal g-profile region. Test on held-out prompts.

The signal is computed from the agent's own forward pass, not from other agents. This establishes the single-agent ceiling. The colony experiments must beat this ceiling to justify their existence.

What remains: expand the prompt battery from ~22 items to ~200+ (see benchmark_strategy.md), run the expanded battery on 2B and 7B, implement signal extraction, build a simple mapping (nearest-neighbor or small MLP), evaluate with bootstrap CIs per task category.

### Experiment 2: Collective g-modulation

Status: not started. Depends on Experiment 1 infrastructure.

A group of agents processes the same prompt, each produces lightweight signals, and the aggregated signal drives a shared g-profile update. The key comparison is collective adaptive vs. individual adaptive from Experiment 1. If the collective finds better g-profiles, the aggregated signal carries information that no single agent's signal contains — amplification in the Daniels et al. (2016) sense.

Colony of N agents (target N=10–50 Qwen3.5-2B instances) each runs a forward pass and reports scalar signals. Signals are aggregated and fed to a controller that updates the shared g-profile. The colony iterates.

Conditions: fixed-g baseline, individual adaptive (Experiment 1), collective adaptive with homogeneous agents, collective adaptive with heterogeneous agents (different temperatures or initial g-profiles), and task-class perturbation (converge on one type, switch, measure adaptation speed).

Measurements from Daniels et al. (2016): amplification (mutual information between collective state and task identity normalized by individual), decomposition (eigenvalue spectrum of agent-agent influence matrix), and task performance (p(tok) compared across conditions).

What remains: build the collective feedback loop, design the controller (fast channel for regime detection, slow channel for stability), determine minimum colony size for measurable amplification.

### Experiment 3: Small-agent fleet improving a single large agent

Status: design phase.

A queen model (OLMo 7B or Qwen3.5-9B) processes a prompt. A fleet of drone models (N × Qwen3.5-2B) pre-process the same prompt and produce lightweight signals. The aggregated drone signal sets the queen's g-profile.

The hypothesis is that drones are individually weaker but collectively their signals contain task-type information that helps the queen find a better operating point than its default. The drones are cheap to run in parallel; the queen is expensive but only runs once with the drone-informed g-profile.

This differs from Experiment 2 in two ways. The colony is heterogeneous — queen and drones are different models, different sizes, potentially different architectures, so the signal must transfer across model boundaries. Communication is asymmetric — drones signal to the queen, the queen does not signal back.

The key question is whether signals from 2B models carry useful information about the optimal g-profile for a 7B model. The sweep data suggests probably yes, since the qualitative response surface taxonomy is preserved across model sizes, but the quantitative mapping will differ.

What remains: validate cross-model signal transfer, design the drone-to-queen signal pathway, determine cost-performance tradeoff, everything from Experiments 1 and 2.

## Execution order

Phase 1 (now): benchmark battery. Write prompt generators for each mechanism type. Pull and format items from COUNTERFACT, LAMBADA, optionally CounterBench. Curate cultural_memorized items. Assemble into a single cartridge of ~400 candidates. Run difficulty calibration sweep on 2B and 7B at baseline only. Filter to ~200 items in the sweet spot.

Phase 2: Experiment 1. Run full g-profile sweep on the calibrated battery. Extract signal vectors from baseline runs. Build signal-to-g-profile mapping. Evaluate on held-out prompts. Write up single-agent results.

Phase 3: Experiment 2. Build the agent signal / buffer / aggregate / g-update loop. Run colony experiments. Compare collective vs. individual adaptive. Compute Daniels et al. measures. Test task-class perturbation.

Phase 4: Experiment 3. Implement cross-model signal pathway. Run queen/drone experiments. Measure cost-performance tradeoff. Compare against Experiments 1 and 2.

## Infrastructure

Compute is on ASU Sol cluster with A100 80GB GPUs. The 7B OLMo fits on a single GPU; the 2B Qwen models can run many instances per GPU for drone experiments. The sweep harness is signal_lab.py and sweep.py, both existing and validated. Data format is JSON prompt batteries with fields id, prompt (or prompt_file), target, type, tier, tokens_approx. Results are JSONL per run in results/docs/runs_data/.

## Open questions

How many agents are needed before collective effects are measurable? Are lightweight scalars sufficient or do we need richer signals like hidden state projections? What are the controller update rates and aggregation functions? Can drone signals from one architecture inform g-profiles on another? The g-profile effect is strongest in the 0.1–0.8 baseline p range — the benchmark battery is designed around this, but real-world prompt distributions are not pre-filtered.