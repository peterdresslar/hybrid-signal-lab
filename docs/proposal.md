# Attention Is In The Air

**Peter Dresslar**
**CAS Capstone, Arizona State University**
**Advisor: Prof. Bryan Daniels**
**March 2026**

## 1. Problem Statement

Multi-agent systems (MAS) composed of networked large language models (LLMs) are a rapidly growing area of AI engineering, yet the MAS literature lacks formal tools for measuring collective behavior. Daniels et al. (2016) provide exactly such tools---amplification and decomposition---developed for biological collectives including eusocial insect colonies. Honey bee hives achieve robust collective behavior not through direct command but through distributed chemical signaling, information amplification across networks, and phase transitions tuned near critical points (Romanczuk and Daniels, 2023; Lynch and Daniels, 2026). These measures have not been applied to artificial multi-agent systems.

Meanwhile, a parallel development in LLM architecture has introduced hybrid models that interleave fundamentally different computational layer types within a single transformer. Qwen3.5 (Qwen Team, 2026) alternates Gated DeltaNet (GDN) layers---fast, recurrent, locally-focused---with gated attention layers---slower, global, relationally-rich---in a fixed 3:1 ratio. The OLMo Hybrid (Merrill et al., 2026) demonstrates that such hybrids outperform both pure attention and pure recurrent architectures, but the ratio between layer types is static, set at training time. Finding the optimal ratio is expensive: it requires ablation studies across architectures and domains, and there is no guarantee that a ratio optimized for one task generalizes to others.

A separate line of work has begun exploring latent (non-textual) communication between LLM agents. Ramesh and Li (2025) inject one agent's intermediate activations into another's forward pass, achieving 27% improvement over natural language communication. Du et al. (2026) transmit compressed last-hidden-state representations between agents. Shi et al. (2026) selectively share KV pairs between agents based on attention importance scores. These approaches demonstrate that activation-level inter-agent communication is effective, but none of them use the collective signal to modulate architectural parameters, and none measure the resulting collective with formal tools.

**This project asks: can a colony of LLM agents collectively modulate the balance between hybrid layer types at inference time, producing measurably better performance than any fixed configuration---and does the resulting collective exhibit the amplification and decomposition predicted by the formal theory of collectivity?**

## 2. Core Idea: The Colony as Meta-Transformer

The key insight is structural. A colony of hybrid LLM agents, each processing different windows of a shared context, is itself a transformer---one level up. The agents are the heads. The pheromone buffer (shared signal space) is the residual stream. The signal compression is the projection. The collective modulation of each agent's GDN/attention balance is the layer norm---regulating information flow.

Unlike a standard transformer, this meta-transformer has heterogeneous heads. Different agents can occupy different positions in the input sequence (geopositioning) and operate at different temperatures. A single transformer cannot do this; its heads are identical by construction.

This design draws directly on honey bee colony coordination:

- **Geopositioning as attention windows.** Each agent receives a different (overlapping) window of the input context, analogous to scouts stationed at different locations. The GDN layers process the local window (strong on nearby context, lossy on distant context); the attention layers integrate across the window. The inter-agent signal aggregation performs the global integration that would require a single model to have attention over the entire sequence.

- **Swarm density and centroid.** The overlap between agent windows controls redundancy vs. coverage. The centroid of the swarm---where most agents' windows cluster---can shift dynamically based on agent signals. Agents near high-entropy regions (uncertain, finding something unexpected) recruit the swarm's attention toward their region, analogous to scout recruitment in bee colonies.

- **Non-semiotic pheromone signals.** Agents broadcast low-dimensional state signals derived from their internal activations via dimensionality reduction (PCA or lightweight autoencoder on layer-boundary hidden states). These signals need not possess semantic meaning---they are the "chemical signature" of each agent's computational state.

- **Noise as substrate.** Temperature diversity across agents is not incidental---it is a design requirement. High-temperature agents establish the entropy floor against which low-temperature agents' convergence becomes a detectable signal. Without noise, the collective cannot distinguish "easy task, everyone agrees" from "hard task, everyone is confidently wrong." Noise keeps the system exploring the parameter space and prevents lock-in, maintaining the colony near the critical regime where collective dynamics are richest (Shpurov et al., 2024). This mirrors biological colonies, where stochastic individual behavior is the substrate that makes collective signal detection possible.

## 3. Architecture

### 3.1 Hybrid Layer Modulation

In Qwen3.5's hybrid architecture, the residual stream is additive: each layer adds its output to a running sum. Forward hooks can scale the contribution of GDN layers vs. attention layers by a collective-determined factor *g*:

- g → 0: GDN layers dominate (fast, local, compressed processing)
- g → 1: Attention layers dominate (slow, global, relational processing)

The collective sets *g* each turn based on aggregated agent signals. This is a resource allocation decision---the same kind bee colonies make when adjusting the ratio of foragers to scouts.

This approach sidesteps the training encumbrance of hybrid architectures. Rather than training multiple models with different fixed ratios and hoping one generalizes, we train (or use) a single hybrid model and let the collective discover the appropriate balance per task at inference time.

### 3.2 Signal Extraction and Compression

Each agent's forward pass yields hidden states at every layer boundary. In a 32-layer Qwen3.5 model, there are ~8 GDN-to-attention transition points. The signal extraction pipeline:

1. **Extract**: Read hidden state tensors at layer boundaries (7-8 transition points × hidden_dim floats per agent). Computationally free relative to the forward pass.
2. **Compress**: Reduce to a low-dimensional pheromone vector (target: 8–16 dimensions) via PCA or a lightweight autoencoder trained on the first N runs. The bottleneck representation is the pheromone. The compressor discovers the "natural pheromone channels" of the architecture.
3. **Buffer**: Write the compressed signal to a shared buffer with exponential time-decay, so recent signals are strong and old signals fade (matching biological pheromone dynamics).
4. **Aggregate**: The buffer state---mean, variance, trajectory of the signal distribution across agents---drives the collective modulation of *g* and the swarm centroid.

This pipeline is structurally analogous to inter-agent attention heads: the compression learns which dimensions of one agent's internal state are most relevant to the collective, and the injection determines which aspects of the collective state matter for each agent's computation. The key difference from Ramesh and Li (2025) is that the communication is collective (many-to-one-to-many via the buffer) rather than pairwise, and it drives architectural modulation rather than just information transfer.

## 4. Experiment Design

### 4.1 Abstract Probe Battery

Three probe types, each designed to favor a different GDN/attention balance, plus a control, with automatic scoring:

1. **Sequence echo** (GDN-favoring): Present a structured token pattern (e.g., A-B-C-A-B-C...) and ask the model to continue. Pure local pattern completion; a recurrent state machine should excel. Score: exact match of continued pattern.

2. **Needle retrieval** (attention-favoring): Embed a specific token or phrase early in a long context, fill with distractor text, ask "what was the item at position X?" Requires reaching back across full context; GDN's compressed state will have degraded early information. Score: exact retrieval of the target.

3. **Compositional reasoning** (balanced): Present 2–3 facts and ask a question requiring their combination ("A > B, C > A, what's smallest?"). Needs both retention and cross-token interaction. Score: correct logical answer (graded for partial credit on multi-step problems).

4. **Null probe** (control): A task where GDN and attention perform equally (e.g., single-token factual recall). Verifies that collective modulation doesn't degrade baseline performance.

### 4.2 Experimental Conditions

1. **Fixed-g baseline**: Run each probe type with the knob fixed at positions across the full range (g = 0, 0.25, 0.5, 0.75, 1.0). This maps the response surface: which g values favor which probes.
2. **Individual agent adaptive**: A single agent adjusts its own g based on its own hidden states (self-reflection). This controls for whether any benefit comes from the collective vs. just from adaptive inference.
3. **Collective adaptive**: The colony of 50 agents, with signals aggregated and g set collectively. The hypothesis: the collective finds better g values than any fixed setting or individual self-adaptation, especially on mixed task streams.
4. **Perturbation and recovery**: Run a stream of one probe type, then inject a different type. Measure how many turns until the collective re-adapts g. Recovery speed relates directly to amplification---a system with high amplification recovers faster because the collective signal dominates individual noise.

### 4.3 Measurements

Drawing on Daniels et al. (2016):

- **Amplification**: Does the collective g setting carry more information about the task type than any individual agent's signal? Measured as mutual information between collective state and task identity, normalized by individual agent mutual information.
- **Decomposition**: Can the colony's behavior be decomposed into independent agent contributions, or are there irreducible collective effects? Measured via the eigenvalue spectrum of the agent-agent influence matrix.
- **Task performance**: Accuracy on the probe battery, compared across conditions. The prediction: collectively-tuned colony outperforms any fixed configuration on mixed task streams.

Additionally, following Riedl et al. (2024), we can apply partial information decomposition to distinguish redundant, unique, and synergistic components of the inter-agent signal---testing whether the colony exhibits genuine higher-order synergy or merely averages individual estimates.

## 5. Implementation Status

### Colony v0.0.1

The Colony testbed is running locally (Qwen 3.5 2B via Ollama, M2 MBA). Validated: asynchronous turn-taking, signal buffer with exponential decay, thinking traces via Ollama's think parameter, entropy measurement from logprobs.

### Signal Lab

A diagnostic tool (signal_lab.py) for exploring model internals via HuggingFace transformers. Produces per-layer hidden state statistics, attention entropy, logit distributions, and KV cache inspection. Output saved to JSON. Built for Qwen3-1.7B (homogeneous attention); needs updating for Qwen3.5-2B (hybrid GDN + attention).

### Next steps

1. Switch to Qwen3.5-2B via transformers. Map which layers are GDN vs. attention using the model config (full_attention_interval=4).
2. Implement forward hooks for reading layer-boundary hidden states and for scaling layer contributions (the g knob).
3. Build the probe battery with automatic scoring.
4. Implement signal compression (PCA on layer-boundary snapshots from initial runs).
5. Wire the collective feedback loop: agent signals → buffer → aggregate → g modulation.
6. Run the experimental conditions and compute Daniels et al. measurements.

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Hybrid layer scaling may not produce measurable performance differences across probes | Map the full response surface first (fixed-g sweep). If GDN/attention balance doesn't affect probe performance, the experimental premise fails early and cheaply. |
| Signal compression may lose the relevant information | Ablation study: run with 1, 4, 8, 16 signal dimensions and plot collective performance vs. dimensionality. |
| 50 agents on a laptop may be too slow for meaningful experimental runs | Qwen3.5-0.8B as fallback; reduce to 20 agents; MBP upgrade imminent; ASU compute for final runs. |
| Collective effects may not emerge---the colony may behave like 50 independent agents | This is a publishable negative result: it tells us where the threshold for collectivity lies. Compare against the individual-adaptive condition to distinguish "no collective effect" from "no adaptive effect." |

## 7. Relevance

This project sits at the intersection of complexity science and applied AI engineering. It tests whether formal theories of collectivity---developed for biological systems---transfer to the design of artificial multi-agent systems operating on a novel architectural substrate (hybrid attention). The meta-transformer framing (colony as outer transformer with heterogeneous heads) suggests that collective dynamics operate at multiple scales in deep learning systems, from attention heads to layers to agents.

## References

- Daniels, B. C., Ellison, C. J., Krakauer, D. C., and Flack, J. C. (2016). Quantifying collectivity. Current Opinion in Neurobiology, 37:106--113.
- Du, Z., Wang, R., Bai, H., Cao, Z., Zhu, X., Cheng, Y., Zheng, B., Chen, W., and Ying, H. (2026). Enabling agents to communicate entirely in latent space. Preprint, arXiv:2511.09149.
- Lynch, C. M. and Daniels, B. C. (2026). Tuning regimes in ant foraging dynamics depend on the existence of bistability. J. Royal Society Interface, 23(225):20250838.
- Merrill, W., et al. (2026). OLMo Hybrid. Ai2.
- Qwen Team (2026). Qwen3.5 technical report.
- Ramesh, V. and Li, K. (2025). Communicating activations between language model agents. Preprint, arXiv:2501.14082.
- Riedl, C., et al. (2024). Emergent coordination in multi-agent language models. Preprint, arXiv:2510.05174.
- Romanczuk, P. and Daniels, B. C. (2023). Phase transitions and criticality in the collective behavior of animals. In Order, Disorder and Criticality. World Scientific.
- Shi, X., Chiesa, M., Maguire, G. Q., and Kostic, D. (2026). KVComm: Enabling efficient LLM communication through selective KV sharing. Preprint, arXiv:2510.03346.
- Shpurov, I., Froese, T., and Chialvo, D. R. (2024). Beehive scale-free emergent dynamics. Scientific Reports, 14(1):13404.
