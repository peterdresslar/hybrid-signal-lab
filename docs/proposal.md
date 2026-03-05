# Capstone Project Proposal: Collective Control in Multi-Agent Systems

**Peter Dresslar**
**CAS Capstone, Arizona State University**
**Advisor: Prof. Bryan Daniels**
**March 2026**

## 1. Problem Statement

Multi-agent systems (MAS) composed of networked large language models (LLMs) are a rapidly growing area of AI engineering. Current orchestration approaches use direct control: a central router selects which agent to invoke, in what order, and with what context (Dang et al., 2025). Recent interpretability work by Ghosh et al. (2026) reveals a striking finding within this paradigm: relational importance (routing frequency, interaction topology) diverges from intrinsic importance (gradient-based causal attribution). Agents that are frequently routed to often have limited causal influence, while sparsely routed agents can be structurally critical.

This divergence is familiar from complexity science. Daniels et al. (2016) formalize precisely these properties---amplification and decomposition---as measures of collectivity in biological systems, including eusocial insect colonies. Honey bee hives, for instance, achieve robust collective behavior not through direct command but through distributed chemical signaling (e.g., queen mandibular pheromone), information amplification across networks, and phase transitions tuned near critical points (Romanczuk and Daniels, 2023; Lynch and Daniels, 2026).

Yet the MAS literature does not draw on this framework. None of the 31 papers citing Dang et al. (2025) emphasize eigenvectors or collective dynamics per Google Scholar as of this writing. The Ghosh et al. (2026) paper itself reinvents concepts analogous to amplification and decomposition without citing the complexity science origins.

**This project asks: can bee-inspired distributed control, informed by the formal theory of collectivity, produce measurably better orchestration dynamics in a multi-agent LLM system?**

## 2. Current Problems in Multi-Agent LLM Systems

A survey of the recent MAS literature (2025--2026) reveals a set of recurring challenges that motivate alternative orchestration approaches:

1. **Centralized routing bottleneck.** The dominant paradigm uses a central meta-controller for task decomposition and routing, creating a single point of failure whose decision space grows combinatorially with agent count (Lyu, 2025; Li et al., 2026).

2. **Relational vs. intrinsic importance divergence.** Agents that are routed to most frequently are not necessarily the most causally important---an observability finding from Ghosh et al. (2026) that diagnoses the problem but does not propose an orchestration solution.

3. **Coordination collapse ("bag of agents").** Flat topologies with no structure lead to circular logic and hallucination loops, with error rates amplified up to 17x over single-agent baselines in unstructured multi-agent configurations (Towards Data Science, 2025).

4. **Inter-agent misalignment.** The MAST failure taxonomy identifies 14 failure modes across 1600+ annotated traces, with breakdowns in inter-agent information flow---role confusion, contradictory outputs, coordination overload---forming a major failure category distinct from individual agent errors (Cemri et al., 2025).

5. **Fragile degradation under ablation.** When agents are removed or fail, centralized systems tend to collapse rather than degrade gracefully. There is little work on how MAS can maintain function when components drop out.

6. **Token and cost explosion.** Multi-agent systems consume 4--220x more input tokens than single-agent systems due to inter-agent communication overhead (Cemri et al., 2025).

7. **Emergent collective bias.** Decentralized LLM populations spontaneously develop shared conventions and collective biases even when individual agents exhibit none---a coordination phenomenon that is also a safety concern (Hu et al., 2025).

8. **Long-horizon planning fragility.** Individual agents reason well locally, but MAS struggle with tasks where early decisions constrain future options and constraints evolve dynamically (Gundawar et al., 2025).

9. **False emergence (data leakage).** LLMs may reproduce conventions encountered during pretraining rather than genuinely self-organizing, making it methodologically difficult to distinguish real emergent coordination from memorized patterns.

10. **Static vs. dynamic topology.** Most frameworks use fixed agent roles and communication graphs. Newer work argues agents should specialize dynamically and adjust connectivity, but how to achieve this robustly remains open (AgentNet, 2025).

## 3. Core Idea: From Direct to Distributed Orchestration

In a standard MAS orchestrator, a central module directly selects agents and routes tasks. We propose replacing or augmenting this with a *distributed signaling layer* inspired by honey bee colony coordination, optionally using a "queenful" orchestrator. To describe this, we must first borrow from current work in LLM observability and interpretability. 

- **Non-Semiotic Pheromone Signals**: Instead of a router making hard selection decisions based on human-readable text, agents broadcast low-dimensional state signals (analogous to pheromonal fields) that influence---but do not dictate---the routing distribution. These signals need not possess semantic meaning; they can be derived directly from the models' internal state, such as normalized hidden state activations, attention head gradients, or token entropy signatures. The queenful orchestrator operates on these signals rather than computing selections from scratch.

- **Amplification control**: Drawing on Daniels et al. (2016), we tune the degree to which individual agent signals are amplified or dampened by the collective. This provides a control parameter analogous to the queen's QMP influence on hive behavior.

- **Criticality tuning**: Following Romanczuk and Daniels (2023) and Shpurov et al. (2024), we hypothesize that orchestration performance is optimized when the system operates near a critical point---poised between rigid single-mode operation and disordered switching. The distributed signaling layer provides a natural parameter space in which to search for and maintain criticality.

We note here that ant and bee signals degrade or decay by virtue of distance or time. Whether or not this is a problem or an opportunitity is not currently clear, though we hypothesize that introducing artificial signal decay (e.g., an exponential time-decay factoring) may naturally counteract the hallucination loops and coordination collapse observed in flat MAS topologies.

### 3.1 Proposed Signal Types

To implement the non-semiotic pheromone signaling, we will explore several observability metrics derived from the LLMs' internal states:

1. **Confidence and Entropy Signatures**: The average log-probability or entropy of recently generated tokens. Like the intensity of a bee's waggle dance, high-confidence (low-entropy) outputs could structurally amplify an agent's routing priority, naturally driving the system toward certainty.
2. **Hidden State Activations**: Mean-pooled high-dimensional vectors from a model's final layer, projected down to lower dimensions. This acts as a semantic "chemical signature," representing the specific region of latent space the agent is currently exploring. It enables clustering of agents addressing similar sub-problems or the prioritization of agents exploring novel conceptual spaces.
3. **Attention Gradients**: Leveraging intrinsic importance measurements derived from the gradients of the output with respect to the input context. Analogous to localized scent marks, this maps which sections of the shared context an agent considers most critical.

## 4. Approach

### 4.1 Testbed Construction (Weeks 1--3)

Build a lightweight multi-agent testbed in Python. Week 1 focuses on the core turn-taking loop and persistent signal buffer with K=5 agents on a single task; Week 2 extends to the full orchestration layer (both modes) and signal decay mechanics; Week 3 scales to N=50 via asynchronous batching and validates on the full task suite. Components:

1. **Agent pool and asynchronous scaling**: The testbed uses open-weight LLMs (e.g., LLaMA-3.1 8B, Qwen-3 8B, DeepSeek-R1 8B) following Ghosh et al.'s homogeneous consortium design with controlled temperature variation. Compute is provisioned through ASU research computing facilities.

   Rather than requiring all N agents to run concurrently, the testbed uses **asynchronous turn-taking**: a batch of K agents (target: K=5) is loaded at a time, each reading the current state of a persistent signal buffer, performing inference, and writing updated signals back. Agents not currently loaded still have presence in the system through their persisted signals, which decay over time according to the exponential decay schedule described in Section 3. This architecture allows scaling to large consortia (target: N=50) with modest hardware---N=50 requires only K=5 concurrent models across 10 rounds per cycle.

   This design is not merely a compute optimization; it is more biologically faithful than synchronous execution. Bee pheromone signals persist in the physical environment and are read asynchronously by individuals who encounter them. The persistent signal buffer with time-decay replicates this dynamic: signals from recently-active agents are strong, while signals from agents that haven't contributed recently fade naturally. The turn-taking structure also introduces a temporal dimension to the collective dynamics that is absent from synchronous architectures, enabling the study of how signal persistence and decay affect collective behavior.

2. **Task suite**: A subset of benchmark tasks from GSM8K and/or HumanEval, chosen for tractability within compute and time constraints. These overlap with the Ghosh et al. evaluation, enabling direct comparison.

3. **Orchestration layer**: Implement two orchestration modes:
   - **Baseline (direct)**: A standard learned router following the Ghosh et al. INFORM architecture---collaboration matrix C(x), selection distribution via Gumbel-Softmax. In the asynchronous setting, the router updates its collaboration matrix after each batch.
   - **Experimental (distributed)**: A pheromone-inspired signaling layer where agents emit state vectors into the persistent signal buffer, collectively shaping routing through decayed accumulation rather than direct selection. Tunable parameters include amplification gain, decay rate, and batch size K.

### 4.2 Measurement Framework (Weeks 2--4)

Implement collective-theoretic measurements from Daniels et al. (2016):

- **Amplification**: Quantify how much individual agent signals are magnified by the collective routing process. High amplification of a single agent indicates queen-like causal centrality.
- **Decomposition**: Measure the degree to which the system's behavior can be decomposed into independent agent contributions vs. irreducible collective effects.
- **Relational vs. intrinsic importance**: Replicate the Ghosh et al. divergence measurement, then assess whether the distributed control condition reduces or restructures this divergence.

Additionally, monitor for signatures of criticality:
- Power-law distributions in routing weight fluctuations (cf. Shpurov et al., 2024).
- Sensitivity to perturbation (ablation of agents) as a function of the amplification control parameter.
- Bistability or hysteresis in task-switching behavior.

### 4.3 Experiments (Weeks 4--7)

1. **Divergence comparison**: Run both orchestration modes on the same task suite. Measure whether the distributed control condition produces a different relationship between relational and intrinsic importance.

2. **Criticality sweep**: Vary the amplification control parameter across a range and measure task performance alongside collective metrics. Test the hypothesis that performance peaks near a critical point in the collective dynamics.

3. **Robustness under ablation**: Mask agents (following Ghosh et al.'s protocol) and compare routing collapse between the two orchestration modes. The distributed mode should exhibit more graceful degradation if the collective signal provides redundancy.

### 4.4 Analysis and Writing (Weeks 6--8)

Compile results into a methods-focused capstone paper. Even negative results (e.g., distributed control does not outperform direct routing on these benchmarks) would be informative, as they would help characterize the boundary conditions under which collective dynamics matter for engineered systems.

## 5. Minimum Viable Deliverable

Given the 8-week timeline, the minimum viable project is:

1. A working testbed with at least one direct and one distributed orchestration mode.
2. Implemented collective measurements (amplification, decomposition) applied to both modes.
3. A methods section and preliminary results suitable for a capstone paper, with clear identification of what further work would be needed to make the results publication-ready.

The stretch goal is a complete comparison paper demonstrating that criticality tuning produces measurable advantages in multi-agent orchestration.

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Compute constraints for running multiple LLMs | ASU research computing facilities provide GPU access for parallel inference; use quantized models or API-based agents for rapid prototyping during development |
| Collective effects may not emerge at small N | Asynchronous turn-taking allows scaling to N=50 with K=5 concurrent models; start with small N for debugging, then scale up; Shpurov et al. (2024) show scale-free dynamics even in moderately-sized bee colonies |
| Criticality tuning may be sensitive to hyperparameters | Frame the criticality sweep itself as a result---mapping the parameter landscape is valuable even if the optimum is hard to find |
| 8 weeks is tight | Prioritize the testbed and measurement framework; the comparison experiments can be scoped down to a single task if needed |

## 7. Relevance

This project sits at the intersection of complexity science and applied AI engineering. It tests whether formal theories of collectivity---developed in the context of biological systems---transfer productively to the design of artificial multi-agent systems. The superorganism framing (Dresslar, 2026) suggests that AI collectives may satisfy the conditions for Krakauer individuality and Hoel causal emergence without requiring constraint closure, making them a novel class of collective entity. This capstone would provide an early empirical data point on that broader theoretical question.

## 8. Future Work

- **Distillation Potential**: Because these coordination mechanisms rely on low-dimensional vector representations rather than complex meta-prompting, they present a novel opportunity for model distillation. If robust collective intelligence emerges from these compact signals, a smaller model could theoretically be trained to predict and internalize these non-semiotic signaling dynamics, effectively distilling the emergent "superorganism" into a highly efficient monolithic architecture.

## References

- Cemri, M., Pan, M. Z., Yang, S., et al. (2025). Why do multi-agent LLM systems fail? Preprint, arXiv:2503.13657.
- Daniels, B. C., Ellison, C. J., Krakauer, D. C., and Flack, J. C. (2016). Quantifying collectivity. *Current Opinion in Neurobiology*, 37:106--113.
- Dang, Y., Qian, C., et al. (2025). Multi-agent collaboration via evolving orchestration. *NeurIPS 2025*.
- Dresslar, P. (2026). Collective properties of honey bees and an analogy to multi-agent systems. CAS 503 Module 7, Arizona State University.
- Ghosh, S., Nath, S., Manchanda, S., and Chakraborty, T. (2026). Disentangling causal importance from emergent structure in multi-expert orchestration. Preprint, arXiv:2602.04291.
- Gundawar, N., et al. (2025). REALM-Bench: A benchmark for evaluating multi-agent systems on real-world, dynamic planning and scheduling tasks. Preprint, arXiv:2502.18836.
- Hu, S., et al. (2025). Emergent social conventions and collective bias in LLM populations. *Science Advances*.
- Li, Z., et al. (2026). Towards adaptive, scalable, and robust coordination of LLM agents: a dynamic ad-hoc networking perspective. Preprint, arXiv:2602.08009.
- Lynch, C. M. and Daniels, B. C. (2026). Tuning regimes in ant foraging dynamics depend on the existence of bistability. *J. Royal Society Interface*, 23(225):20250838.
- Romanczuk, P. and Daniels, B. C. (2023). Phase transitions and criticality in the collective behavior of animals. In *Order, Disorder and Criticality*. World Scientific.
- Shpurov, I., Froese, T., and Chialvo, D. R. (2024). Beehive scale-free emergent dynamics. *Physics Letters A*, 514:129614.
