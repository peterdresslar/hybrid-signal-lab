# Collective-AI Backburner

Items parked here are not abandoned---they are deferred from the immediate capstone and AIITA paper scope. Each section notes why it was deferred and what would reactivate it.

## 1. INFORM / Ghosh et al. Comparison Experiment

The original capstone proposal framed the project as a direct comparison with Ghosh et al. (2026) INFORM architecture: replicate their learned router (Gumbel-Softmax collaboration matrix), run the same benchmarks, then show that distributed pheromone signaling outperforms or restructures the relational/intrinsic importance divergence.

**Why deferred:** The AIITA framing is stronger. Rather than comparing orchestration strategies on existing benchmarks, we are demonstrating a novel intervention (collective modulation of hybrid layer balance) that nobody has attempted. The Ghosh et al. divergence finding remains excellent motivation---it goes in the AIITA introduction---but we don't need to reimplement their router.

**Reactivation trigger:** If a reviewer or collaborator wants an apples-to-apples comparison with INFORM on standard MAS benchmarks. Also relevant if we pivot to evaluating Colony on task-completion metrics rather than collective-theoretic metrics.

### Preserved content

- Baseline (direct) orchestration mode: standard learned router following Ghosh et al. INFORM---collaboration matrix C(x), selection distribution via Gumbel-Softmax. In the asynchronous setting, the router updates its collaboration matrix after each batch.
- Task suite from original proposal: GSM8K and/or HumanEval subsets, chosen for overlap with Ghosh et al. evaluation.
- Divergence comparison experiment: Run both orchestration modes on the same task suite. Measure whether distributed control produces a different relationship between relational and intrinsic importance.

### Key references

- Ghosh, S., Nath, S., Manchanda, S., and Chakraborty, T. (2026). Disentangling causal importance from emergent structure in multi-expert orchestration. Preprint, arXiv:2602.04291.
- Dang, Y., Qian, C., et al. (2025). Multi-agent collaboration via evolving orchestration. NeurIPS 2025.

---

## 2. Standard MAS Benchmarks (MARBLE / MultiAgentBench)

The original plan included evaluating Colony against MARBLE (ACL 2025) task-completion benchmarks.

**Why deferred:** MARBLE measures task completion, not collective dynamics. Our contribution is measuring the process, not the outcome. Also, the MARBLE repo appears inactive (no commits in 8+ months as of March 2026). The abstract probe battery designed for AIITA gives us cleaner measurement with full control over what computational regime we exercise.

**Reactivation trigger:** If we need to demonstrate practical task performance to satisfy a venue's reviewers, or if MARBLE becomes active again and gains traction.

---

## 3. Original Weekly Timeline (8-week plan)

The proposal's 4.1–4.4 timeline was designed around the INFORM comparison framing:

- Weeks 1–3: Testbed construction (core loop → full orchestration → scale to N=50)
- Weeks 2–4: Measurement framework (amplification, decomposition, criticality signatures)
- Weeks 4–7: Experiments (divergence comparison, criticality sweep, robustness under ablation)
- Weeks 6–8: Analysis and writing

**Why deferred:** The AIITA experiment has different phases. The testbed already exists (Colony v0.0.1). The next phases are: (1) signal extraction via transformers on Qwen3.5-2B hybrid architecture, (2) forward hook implementation for layer-type modulation, (3) probe battery construction and scoring, (4) collective state aggregation and feedback loop, (5) experimental runs and Daniels et al. measurement. New timeline needed.

**Reactivation trigger:** Elements of the original timeline (especially the measurement framework and ablation experiments) will be adapted into the AIITA plan rather than reused wholesale.

---

## 4. Lotka-Volterra / "Do Android Wolves Dream of Electric Sheep?"

A previously-developed agent-based model where LLM agents set a territorial competition parameter (theta) in a modified Lotka-Volterra predator-prey system. Demonstrated that AI agents can stabilize population dynamics that defeat programmatic functions, with performance varying by information level (high/medium/low prompt detail).

**Why deferred:** Adds a layer of complexity (population dynamics + hybrid architecture tuning) that would require explaining both to Prof. Daniels. The scientific question was also muddled: it tried to serve both "can AIs perform optimally in simulations" and "can AIs emulate humans in simulations" without committing to either.

**Reactivation trigger:** Paper two. Once the collective tuning framework is established via AIITA, port it to Lotka-Volterra. The scientific question becomes sharp: "does collective architectural tuning improve agent performance in a dynamical system with known analytic solutions?" The wolf framing becomes illustration, not justification. The theta parameter maps directly to the Colony signal framework.

### Key resources

- Repository: https://github.com/peterdresslar/do-android-wolves-dream-of-electric-sheep/
- Model uses modified LV with theta as a competition/territoriality factor
- Three conditions demonstrated: constant theta, functional theta (scarcity-sensitive), LLM-decided theta
- LLM wolves outperformed the programmatic function in hard initial conditions (15 sheep / 10 wolves)
- High-information wolves: stability via moderate theta (~0.4). Medium-information: classic boom-bust cycles. Low-information: stability via high territorial aggression (~0.6).

---

## 5. MoE Router Modulation

In Mixture-of-Experts architectures, the router is a softmax over expert logits---structurally identical to a collective voting mechanism. Router temperature controls consensus (low temp) vs. exploration (high temp). A forward hook could scale logits before softmax based on collective state: high collective entropy → cool the router (stabilize), low collective entropy → warm the router (diversify). "Anytime we see router, we should think bees."

**Why deferred:** The small Qwen3.5 models (0.8B–9B) are dense, not MoE. The MoE version is the 397B-A17B flagship. Requires ASU compute or API access to run. The hybrid layer modulation on dense Qwen3.5-2B is the more tractable first experiment.

**Reactivation trigger:** Access to ASU GPU cluster or API for a large MoE model. Same framework, different internal architecture. Natural second paper after AIITA demonstrates the principle on hybrid dense models.

### Architectural notes

- Arcee Trinity Large: 400B MoE, 17B active, gated attention + SWA
- GLM-5: 744B, MLA + sparse attention
- Qwen3.5-397B-A17B: MoE with hybrid DeltaNet + Gated Attention
- Key operation: forward hook on router layer, scale logits by collective-derived temperature before softmax

---

## 6. Activation Steering from Collective State

Inject collective state directly into the residual stream at specific layer boundaries via forward hooks. The steering vector is derived from aggregated peer hidden states rather than predetermined by researchers. This is representation engineering (Zou et al., 2023) but with steering vectors derived from the collective.

**Why deferred:** Most invasive intervention. Requires understanding of representation engineering literature, careful implementation of forward hooks that modify (not just read) activations, and validation that the intervention doesn't degrade base model performance. Likely requires more compute than local M2 MBA for meaningful experiments.

**Reactivation trigger:** After AIITA establishes that residual stream scaling (the gentler intervention) produces measurable collective effects. If scaling works, steering is the natural next step---more targeted, more powerful, and a genuine novel contribution to the representation engineering literature.

### Key references

- Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency. Preprint.
- Li, K., et al. (2024). Inference-time intervention: Eliciting truthful answers from a language model.

---

## 7. Learned Inter-Agent Attention

The signal compression between agents (extract high-dimensional hidden states → compress to low-dimensional pheromone vector → inject into next agent) is structurally identical to cross-attention. A trainable cross-attention module sitting on top of frozen Qwen3.5-2B agents would learn which dimensions of one agent's state are most relevant to others. The compressor discovers "natural pheromone channels."

**Why deferred:** Introduces a training loop on top of the inference-only experiment. For the capstone, a fixed projection (PCA on initial runs, then freeze) is sufficient to demonstrate the principle. The learned version is a follow-up showing that optimizing the inter-agent communication channel improves collective performance beyond any fixed projection.

**Reactivation trigger:** After AIITA establishes baseline results with fixed PCA-based signal compression. The ablation study (1-dim vs. 4-dim vs. 8-dim vs. 16-dim signal) will reveal whether there's headroom for learning to exploit.

---

## 8. Distillation as Collective Dynamics

The teacher-student relationship in model distillation is a two-agent system with asymmetric information flow. The finding that the best model is not necessarily the best teacher (Lambert, 2026) parallels Ghosh et al.'s relational/intrinsic importance divergence. The measurement framework could instrument distillation dynamics: is there amplification between teacher and student? Does the student exhibit causal emergence relative to the teacher on specific tasks?

**Why deferred:** Requires access to a distillation pipeline and compute for training runs. Conceptually exciting but experimentally heavy.

**Reactivation trigger:** Collaboration with a lab that does distillation research (e.g., Ai2/OLMo team, Qwen team). The framing would be: "your distillation pipeline is a two-agent collective---here's what our measurement framework reveals about it."

---

## 9. Marimo Integration

Wire signal_lab.py into a marimo notebook using mo.ui.chat() with a custom model Callable and mo.state() for reactive telemetry display. Real-time visualization of per-layer hidden states, attention patterns, entropy profiles during interactive generation.

**Why deferred:** Nice-to-have visualization tool, not on the critical path to the AIITA experiment. The signal_lab.py script produces JSON output that can be analyzed offline.

**Reactivation trigger:** When the probe battery is running and we want a live dashboard for monitoring collective dynamics during experimental runs. Also useful for demos to Bryan / Colin.

---

## 10. Preserved References (not used in AIITA or companion paper)

- Cemri, M., Pan, M. Z., Yang, S., et al. (2025). Why do multi-agent LLM systems fail? Preprint, arXiv:2503.13657.
- Dang, Y., Qian, C., et al. (2025). Multi-agent collaboration via evolving orchestration. NeurIPS 2025.
- Gundawar, N., et al. (2025). REALM-Bench: A benchmark for evaluating multi-agent systems on real-world, dynamic planning and scheduling tasks. Preprint, arXiv:2502.18836.
- Hu, S., et al. (2025). Emergent social conventions and collective bias in LLM populations. Science Advances.
- Li, Z., et al. (2026). Towards adaptive, scalable, and robust coordination of LLM agents. Preprint, arXiv:2602.08009.
- Lyu (2025). [Centralized routing bottleneck reference from original proposal.]
- Towards Data Science (2025). [17x error rate amplification in unstructured MAS.]
