# Attention Is In The Air

**Peter Dresslar**

**CAS Capstone, Arizona State University**

**Advisor: Prof. Bryan Daniels**

**March 2026**

## 1. Problem Statement

Multi-agent systems (MAS) composed of networked large language models (LLMs) are a rapidly growing area of AI engineering, yet the MAS literature lacks formal tools for measuring collective behavior. Daniels et al. (2016) provide exactly such tools---amplification and decomposition---developed for biological collectives including eusocial insect colonies. Honey bee hives achieve robust collective behavior not through direct command but through distributed chemical signaling, information amplification across networks, and phase transitions tuned near critical points (Romanczuk and Daniels, 2023; Lynch and Daniels, 2026). These measures have not been applied to artificial multi-agent systems.

Meanwhile, a parallel development in LLM architecture has introduced hybrid models that interleave fundamentally different computational layer types within a single transformer. Qwen3.5 (Qwen Team, 2026) alternates Gated DeltaNet (GDN) layers---fast, recurrent, locally-focused---with gated attention layers---slower, global, relationally-rich---in a fixed 3:1 ratio. The OLMo Hybrid (Merrill et al., 2026) demonstrates that such hybrids outperform both pure attention and pure recurrent architectures, but the ratio between layer types is static, set at training time. Finding the optimal ratio is expensive: it requires ablation studies across architectures and domains, and there is no guarantee that a ratio optimized for one task generalizes to others.

A separate line of work has begun exploring latent (non-textual) communication between LLM agents. Ramesh and Li (2025) inject one agent's intermediate activations into another's forward pass, achieving 27% improvement over natural language communication. Du et al. (2026) transmit compressed last-hidden-state representations between agents. Shi et al. (2026) selectively share KV pairs between agents based on attention importance scores weighted by a Gaussian prior over layer depth, demonstrating that intermediate layers carry the most transferable semantic knowledge and that only 30% of layers' KV pairs are needed for near-optimal communication. These approaches demonstrate that activation-level inter-agent communication is effective, but none of them use the collective signal to modulate architectural parameters, and none measure the resulting collective with formal tools.

**This project asks: can a colony of LLM agents collectively modulate the balance between hybrid layer types at inference time, producing measurably better performance than any fixed configuration---and does the resulting collective exhibit the amplification and decomposition predicted by the formal theory of collectivity?**

## 2. Core Idea: The Colony as Meta-Transformer

The key insight is structural. A colony of hybrid LLM agents, each processing different windows of a shared context, is itself a transformer---one level up. The agents are the heads. The pheromone buffer (shared signal space) is the residual stream. The signal compression is the projection. The collective modulation of each agent's GDN/attention balance is the layer norm---regulating information flow.

Unlike a standard transformer, this meta-transformer has heterogeneous heads. Different agents can occupy different positions in the input sequence (geopositioning) and operate at different temperatures. A single transformer cannot do this; its heads are identical by construction.

This design draws directly on honey bee colony coordination:

- **Geopositioning as attention windows.** Each agent receives a different (overlapping) window of the input context, analogous to scouts stationed at different locations. The GDN layers process the local window (strong on nearby context, lossy on distant context); the attention layers integrate across the window. The inter-agent signal aggregation performs the global integration that would require a single model to have attention over the entire sequence.

- **Swarm density and centroid.** The overlap between agent windows controls redundancy vs. coverage. The centroid of the swarm---where most agents' windows cluster---can shift dynamically based on agent signals. Agents near high-entropy regions (uncertain, finding something unexpected) recruit the swarm's attention toward their region, analogous to scout recruitment in bee colonies.

- **Non-semiotic pheromone signals.** Agents broadcast low-dimensional state signals derived from their internal activations via dimensionality reduction (PCA or lightweight autoencoder on layer-boundary hidden states). These signals need not possess semantic meaning---they are the "chemical signature" of each agent's computational state.

- **Noise as substrate.** Temperature diversity across agents is not incidental---it is a design requirement. Preliminary experiments with attention scaling (Section 5) reveal a chaos zone at low scaling values (g < 0.5) where model predictions become highly unstable and non-monotonic---the model thrashes between distant attractors in prediction space. This stochastic regime is precisely the substrate that makes collective signal detection possible. High-temperature agents establish the entropy floor against which low-temperature agents' convergence becomes a detectable signal. Without noise, the collective cannot distinguish "easy task, everyone agrees" from "hard task, everyone is confidently wrong." Noise keeps the system exploring the parameter space and prevents lock-in, maintaining the colony near the critical regime where collective dynamics are richest (Shpurov et al., 2024). This mirrors biological colonies, where stochastic individual behavior is the substrate that makes collective signal detection possible.

## 3. Architecture

### 3.1 Attention Scaling and the *g* Response Surface

In Qwen3.5's hybrid architecture, the residual stream is additive: each layer adds its output to a running sum. Forward hooks can scale the contribution of attention layers by a factor *g*:

- *g* = 0: Attention layers contribute nothing (equivalent to a pure-recurrent model)
- *g* = 1: Normal operation (the trained balance)
- *g* > 1: Attention layers are amplified beyond their trained contribution

Preliminary sweeps across a battery of short probes (Section 5) reveal that the *g* response surface is far richer than a simple slider between two regimes. The relationship between *g* and prediction quality is non-monotonic, completion-type-dependent, and model-size-dependent. Specific findings include:

- **Phase transitions**: Some prompts exhibit sharp transitions from near-perfect prediction to near-uniform entropy within a single 0.1 step of *g* (e.g., syntactic completions collapsing at *g* ≈ 1.7).
- **Multiple optima**: Cultural/memorized completions can exhibit two distinct *g* regions that produce correct predictions, with a valley of degraded performance between them---suggesting two different internal mechanisms can independently find the answer.
- **Completion-type taxonomy**: The width of the *g* basin correlates with the type of knowledge required. Syntactic pattern completions (e.g., `import torch.nn as` → `nn`) are robust across a wide range (g ≈ 0.8–1.5); algorithmic completions (e.g., Fibonacci continuation) are similarly robust; structural copying tasks (e.g., syntactic parallelism) require moderate attention and degrade cleanly with over-amplification; and factual recall tasks (e.g., "the color with the shortest wavelength is") have narrow optima and never reach high confidence.
- **Model-size effects**: A 0.8B parameter model shows the same qualitative taxonomy as a 2B model but with narrower basins, faster degradation at high *g*, and failure on tasks the larger model handles (cultural completions that never reach rank 1 at any *g* value).

These findings suggest that *g* is not merely a tuning parameter but a diagnostic: the shape of a prompt's *g* response curve reveals the type of computational mechanism the model employs to produce its prediction.

### 3.2 Per-Layer *g* Profiles

The uniform scaling described above applies the same *g* to all hooked attention layers. However, the information hierarchy within transformers is well-established: early layers capture surface patterns, middle layers encode semantic abstractions, and late layers specialize in task-specific predictions (Jawahar et al., 2019; Geva et al., 2020). Shi et al. (2026) exploit this hierarchy by centering a Gaussian prior on middle layers to select the most informative KV pairs for inter-agent communication.

This motivates extending from a scalar *g* to a per-layer profile **g** = (g₁, g₂, ..., gₗ), where each attention layer receives its own scaling factor. The colony's task then becomes navigating a low-dimensional **g**-space rather than optimizing a single scalar. With Qwen3.5's architecture (attention layers at every 4th position), a 24-layer model has only 6 attention layers---making **g** a 6-dimensional vector, tractable for both exhaustive sweeps and collective optimization.

The connection to KVComm is direct: Shi et al. find that layer-wise attention importance scores (how concentrated a layer's attention distribution is) predict which layers' KV pairs are most useful for communication. Our attention entropy measurements per layer per head provide exactly the same information, repurposed as a signal for modulating layer contribution rather than selecting which layers to share.

### 3.3 Signal Extraction and Compression

Each agent's forward pass yields hidden states at every layer boundary. In a 32-layer Qwen3.5 model, there are ~8 GDN-to-attention transition points. The signal extraction pipeline:

1. **Extract**: Read hidden state tensors at layer boundaries (7-8 transition points × hidden_dim floats per agent). Computationally free relative to the forward pass.
2. **Compress**: Reduce to a low-dimensional pheromone vector (target: 8–16 dimensions) via PCA or a lightweight autoencoder trained on the first N runs. The bottleneck representation is the pheromone. The compressor discovers the "natural pheromone channels" of the architecture.
3. **Buffer**: Write the compressed signal to a shared buffer with exponential time-decay, so recent signals are strong and old signals fade (matching biological pheromone dynamics).
4. **Aggregate**: The buffer state---mean, variance, trajectory of the signal distribution across agents---drives the collective modulation of **g** and the swarm centroid.

Lightweight scalar signals---particularly final-token entropy (in bits), KL divergence from a running baseline, and per-layer attention entropy---may prove sufficient for collective *g* modulation without full hidden-state extraction. The sweep experiments demonstrate that these signals are sensitive to both task type and *g* perturbation, making them natural candidates for a fast feedback channel.

This pipeline is structurally analogous to inter-agent attention heads: the compression learns which dimensions of one agent's internal state are most relevant to the collective, and the injection determines which aspects of the collective state matter for each agent's computation. The key difference from Ramesh and Li (2025) is that the communication is collective (many-to-one-to-many via the buffer) rather than pairwise, and it drives architectural modulation rather than just information transfer.

## 4. Experiment Design

### Experiment 1: Attention Scaling Response Surfaces

The first experiment characterizes how attention scaling affects prediction quality across different completion types and model sizes, establishing the empirical foundation for collective *g* modulation.

**Probe corpus.** A battery of short probes classified by the type of knowledge required for completion. The preliminary battery (Section 5) covers five categories: syntactic pattern completion, algorithmic/sequential completion, structural copying, cultural/memorized completion, and factual recall. Each probe has a known single-token target, enabling automatic scoring via target rank and probability in the model's output distribution. This battery will be extended using prompts drawn from established benchmarks---COUNTERFACT (Meng et al., 2022) for factual recall, and MIB (Mueller et al., 2025) for syntactic, arithmetic, and reasoning tasks---to provide statistically powered results across a classified taxonomy.

**Sweep protocol.** For each probe, we sweep *g* from 0.0 to 2.0 at 0.1 granularity, recording: target token rank, target token probability, final-token entropy (bits), mean sequence entropy, KL divergence from the *g*=1.0 baseline distribution, and per-layer per-head attention entropy. These measurements are deterministic under `model.eval()` with `torch.no_grad()`, so single runs are sufficient.

**Per-layer sweeps.** Following the uniform-*g* characterization, we conduct per-layer sweeps: holding all attention layers at *g*=1.0 except one, and sweeping the free layer. This identifies which layers are most sensitive to scaling for each completion type, and whether the information hierarchy (early=surface, middle=semantic, late=task-specific) manifests in the *g* response.

**Cross-model comparison.** All sweeps are conducted on at least two model sizes (Qwen3.5-2B and Qwen3.5-0.8B) to test whether the completion-type taxonomy and sensitivity profiles are preserved across scale.

### Experiment 2: Collective *g* Modulation

The second experiment tests whether a colony of agents can collectively discover and maintain an appropriate **g** configuration, and whether the resulting collective exhibits measurable amplification and decomposition.

**Protocol.** A colony of agents processes a sequential stream of classified probes. Each agent runs a forward pass, reports lightweight scalar signals (entropy, KL divergence, attention entropy), and the collective aggregates these signals to set **g** for the next round.

**Conditions:**

1. **Fixed-*g* baseline**: Performance at fixed *g* values across the probe stream (already characterized in Experiment 1).
2. **Individual adaptive**: A single agent adjusts its own *g* based on its own signals. This controls for whether any benefit comes from the collective vs. just from adaptive inference.
3. **Collective adaptive**: The colony adjusts **g** collectively via aggregated signals. The hypothesis: the collective finds better **g** configurations than any fixed setting or individual self-adaptation, especially on mixed task streams.
4. **Task-class perturbation**: Run a sustained stream of one probe class (e.g., factual recall), allowing the colony to converge on a stable **g** regime, then switch to a different class (e.g., syntactic completion). Measure the transition cost (cumulative excess entropy or degraded target rank during adaptation), convergence speed (rounds to reach new stable **g**), and whether the colony overshoots or oscillates. Recovery speed relates directly to amplification---a system with high amplification recovers faster because the collective signal dominates individual noise.

**Controller design.** The collective *g* update follows a slow-fast architecture inspired by biological gain control. A fast channel detects entropy or KL spikes (signaling regime change) and pushes *g* aggressively in the direction that reduces error. A slow channel damps *g* back toward a running average, preventing overreaction to noise. The response curves from Experiment 1 provide ground truth for validating the controller's behavior.

**Colony size.** [Note: The appropriate colony size is an open design parameter. 50 agents provides statistical power for measuring collective effects but may be computationally prohibitive on local hardware. An alternative is a smaller colony (10–20 agents) with heterogeneous *g* values spanning the response surface, where diversity across agents provides coverage analogous to a larger uniform colony. The heterogeneous design is more biologically faithful---bee colonies exhibit individual behavioral variation that is functional, not merely noise---and may require fewer agents to achieve the same collective sensitivity. Initial experiments will determine the minimum colony size at which amplification becomes measurable.]

### 4.3 Measurements

Drawing on Daniels et al. (2016):

- **Amplification**: Does the collective **g** setting carry more information about the task type than any individual agent's signal? Measured as mutual information between collective state and task identity, normalized by individual agent mutual information.
- **Decomposition**: Can the colony's behavior be decomposed into independent agent contributions, or are there irreducible collective effects? Measured via the eigenvalue spectrum of the agent-agent influence matrix.
- **Task performance**: Accuracy on the probe battery, compared across conditions. The prediction: collectively-tuned colony outperforms any fixed configuration on mixed task streams.

Additionally, following Riedl et al. (2024), we can apply partial information decomposition to distinguish redundant, unique, and synergistic components of the inter-agent signal---testing whether the colony exhibits genuine higher-order synergy or merely averages individual estimates.

## 5. Preliminary Results

Uniform-*g* sweeps have been completed on both Qwen3.5-2B and Qwen3.5-0.8B across a battery of six short probes, at 0.1 granularity from *g*=0.0 to *g*=2.0. Key findings:

**Uniform baseline at *g*=0.0.** Zeroing all attention layers produces a final-token entropy of exactly 17.92 bits (≈ log₂ 248,320, the vocabulary size), confirming a uniform output distribution. This establishes the information-theoretic floor: attention contributes all of the model's discriminative capacity.

**Completion-type taxonomy.** The width and depth of the optimal *g* basin varies systematically with completion type:

| Completion type | Example | Optimal *g* range (2B) | Peak target prob | Basin character |
|----------------|---------|----------------------|-----------------|----------------|
| Syntactic pattern | `import torch.nn as` → `nn` | 0.8–1.5 | 99.9% | Wide, flat, sharp cliff |
| Algorithmic | Fibonacci → `34` | 0.5–1.7 | 91% | Wide, slight sub-baseline peak |
| Structural copying | `opened the door...opened the` → `door` | 0.9–1.6 | 59% | Monotonic rise then fall |
| Cultural/memorized | `roses are red...` → `so` | 0.6, 1.2–1.6 | 30% | Double-peaked, two mechanisms |
| Factual recall | shortest wavelength → `violet` | 0.8–0.9 | 4.8% | Narrow, never confident |

**Model-size scaling.** The 0.8B model preserves the qualitative taxonomy but with uniformly narrower basins and lower peak probabilities. Cultural/memorized completions that reach rank 1 on the 2B model (e.g., "roses are red" → "so") fail to reach rank 1 at any *g* value on the 0.8B model, suggesting a capacity threshold for this completion type.

**Determinism.** Repeated runs under `model.eval()` with `torch.no_grad()` produce identical results, confirming that the response surfaces are properties of the model architecture and weights, not stochastic artifacts.

## 6. Implementation Status

### Colony v0.0.1

The Colony testbed is running locally (Qwen 3.5 2B via Ollama, M2 MBA). Validated: asynchronous turn-taking, signal buffer with exponential decay, thinking traces via Ollama's think parameter, entropy measurement from logprobs.

### Signal Lab

A diagnostic tool (signal_lab.py) for exploring model internals via HuggingFace transformers. Produces attention entropy per layer per head, logit distributions, top-k predictions, target token tracking (rank, probability), final-token and mean-sequence entropy in bits, and KL divergence from baseline. Forward hooks scale attention layer contributions by a configurable factor *g*. Output saved to JSON/JSONL. Sweep harness (sweep.py) automates *g* sweeps across prompt batteries with per-prompt target tokens. Validated on Qwen3.5-2B and Qwen3.5-0.8B.

### Next Steps

1. Extend probe corpus using COUNTERFACT and MIB benchmarks for statistically powered results across the completion-type taxonomy.
2. Implement per-layer *g* profiles and conduct per-layer sensitivity sweeps.
3. Validate target tokens for the 0.8B model (some 2B targets may not be appropriate for the smaller model).
4. Obtain ASU Sol supercomputer access (A100 80GB GPUs, free via Research Computing) for medium-length prompts and larger model experiments.
5. Build the probe battery with automatic scoring for Experiment 2.
6. Implement signal compression (PCA on layer-boundary snapshots from initial runs).
7. Wire the collective feedback loop: agent signals → buffer → aggregate → **g** modulation.
8. Run the experimental conditions and compute Daniels et al. measurements.

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Attention scaling may not produce measurable performance differences across probes | Preliminary sweeps (Section 5) already demonstrate strong completion-type-dependent effects. The experimental premise is validated. |
| Per-layer *g* profiles may not add explanatory power beyond uniform *g* | The uniform-*g* sweeps provide a baseline. If per-layer variation is inert, the colony operates on a 1D manifold instead of 6D, which simplifies the collective optimization. |
| Signal compression may lose the relevant information | Ablation study: run with 1, 4, 8, 16 signal dimensions and plot collective performance vs. dimensionality. Lightweight scalar signals (entropy, KL) may suffice. |
| Colony may be too slow for meaningful experimental runs on local hardware | Qwen3.5-0.8B as fallback; ASU Sol for final runs; reduce colony size and test minimum viable colony for measurable amplification. |
| Collective effects may not emerge---the colony may behave like independent agents | This is a publishable negative result: it tells us where the threshold for collectivity lies. Compare against the individual-adaptive condition to distinguish "no collective effect" from "no adaptive effect." |

## 8. Relevance

This project sits at the intersection of complexity science and applied AI engineering. It tests whether formal theories of collectivity---developed for biological systems---transfer to the design of artificial multi-agent systems operating on a novel architectural substrate (hybrid attention). The meta-transformer framing (colony as outer transformer with heterogeneous heads) suggests that collective dynamics operate at multiple scales in deep learning systems, from attention heads to layers to agents.

The preliminary finding that attention scaling response surfaces are completion-type-dependent and non-monotonic connects the mechanistic interpretability literature (activation patching, causal tracing) to the multi-agent coordination literature in a novel way: the same measurements that reveal what attention does for a single model also provide the signals by which a colony of models can collectively regulate their own attention.

## References

- Daniels, B. C., Ellison, C. J., Krakauer, D. C., and Flack, J. C. (2016). Quantifying collectivity. Current Opinion in Neurobiology, 37:106--113.
- Du, Z., Wang, R., Bai, H., Cao, Z., Zhu, X., Cheng, Y., Zheng, B., Chen, W., and Ying, H. (2026). Enabling agents to communicate entirely in latent space. Preprint, arXiv:2511.09149.
- Geva, M., Schuster, R., Berant, J., and Levy, O. (2020). Transformer feed-forward layers are key-value memories. Preprint, arXiv:2012.14913.
- Jawahar, G., Sagot, B., and Seddah, D. (2019). What does BERT learn about the structure of language? In Proceedings of ACL 2019.
- Lynch, C. M. and Daniels, B. C. (2026). Tuning regimes in ant foraging dynamics depend on the existence of bistability. J. Royal Society Interface, 23(225):20250838.
- Meng, K., Bau, D., Andonian, A., and Belinkov, Y. (2022). Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems 35.
- Merrill, W., et al. (2026). OLMo Hybrid. Ai2.
- Mueller, A., et al. (2025). MIB: A Mechanistic Interpretability Benchmark. In Proceedings of ICML 2025.
- Qwen Team (2026). Qwen3.5 technical report.
- Ramesh, V. and Li, K. (2025). Communicating activations between language model agents. Preprint, arXiv:2501.14082.
- Riedl, C., et al. (2024). Emergent coordination in multi-agent language models. Preprint, arXiv:2510.05174.
- Romanczuk, P. and Daniels, B. C. (2023). Phase transitions and criticality in the collective behavior of animals. In Order, Disorder and Criticality. World Scientific.
- Shi, X., Chiesa, M., Maguire, G. Q., and Kostic, D. (2026). KVComm: Enabling efficient LLM communication through selective KV sharing. In Proceedings of ICLR 2026. arXiv:2510.03346.
- Shpurov, I., Froese, T., and Chialvo, D. R. (2024). Beehive scale-free emergent dynamics. Scientific Reports, 14(1):13404.