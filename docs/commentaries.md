# Commentaries

This file is completely managed by Claude

## AttnRes (Kimi Team, arXiv:2603.15031, March 2026)

### Summary
Replaces fixed residual accumulation (h_l = h_{l-1} + f_{l-1}(h_{l-1})) with learned softmax attention over depth. Each layer selectively aggregates all preceding layer outputs with input-dependent weights via a single learned pseudo-query vector per layer. Block AttnRes partitions layers into ~8 blocks and attends over block-level representations, reducing memory from O(Ld) to O(Nd) while recovering most of the performance gain. Integrated into Kimi Linear (48B/3B active MoE) and pre-trained on 1.4T tokens. Consistent improvements across all benchmarks, particularly on multi-step reasoning (+7.5 on GPQA-Diamond) and code generation (+3.1 on HumanEval). Key theoretical framing: "duality of time and depth"—the same linear-to-softmax transition that transformers applied over the sequence dimension, now applied over the depth dimension.

### Relevance to Our Work

**Direct structural analogy.** Our pheromone buffer is itself a residual stream. The colony's collectively-generated signal accumulates and gets modulated across agents in the same way that layer outputs accumulate and get modulated across depth in a standard transformer. Our gain vector intervention is doing for the colony's external residual stream what AttnRes does for the model's internal one: replacing fixed uniform aggregation with selective, input-dependent weighting. Block AttnRes groups layers into blocks with aggregate representations; our colony groups agents with aggregate signals in the pheromone buffer. The analogy is not metaphorical—it is structural.

**Key difference and our contribution.** AttnRes learns depth-wise attention weights during training via backpropagation through a single model. We are attempting to learn analogous weights at inference time across multiple model instances without gradient access. This is a harder problem and a genuinely different contribution. AttnRes approaches the residual stream from the inside (training-time, single-model architectural change). Our work approaches it from the outside (inference-time, multi-agent collective signal).

**Implications for our intervention framework.** Our gain vector currently operates on standard residual streams in models that use fixed uniform accumulation (Qwen3.5, OLMo). If the field moves toward models where depth-wise aggregation is learned and selective (as AttnRes proposes), the intervention surface area changes fundamentally. A gain vector applied to a standard residual modulates a uniform sum. A gain vector applied to an AttnRes-style residual would interact with the model's own learned depth-routing decisions—either complicating the intervention or opening a richer intervention space. This is a forward-looking observation that belongs in the discussion section.

**Evidence for our broader argument.** AttnRes is strong evidence that the residual stream is becoming an active research target, not just passive infrastructure. The fact that a major lab is redesigning depth-wise information flow from the inside strengthens our claim that intervening on this flow from the outside—through collectively-generated signals—is a meaningful research direction. Both approaches are saying the same underlying thing: uniform fixed accumulation is suboptimal, and there is information to be gained from selective depth-wise modulation.

**Relevance to other teams.** This work should be directly relevant to the AllenAI/OLMo team, particularly regarding how depth-wise information aggregation interacts with hybrid architectures (state-space + attention layers). The question of whether AttnRes-style selective aggregation behaves differently across heterogeneous layer types is open and connects to our interest in hybrid model intervention.

### Citation
Chen, G., Zhang, Y., Su, J., et al. (2026). Attention Residuals. arXiv:2603.15031 [cs.CL].


## LLM-JEPA (Huang, LeCun, Balestriero, arXiv:2509.14252, October 2025)

### Summary
First JEPA-based training objective for LLMs. Adds an embedding-space prediction loss alongside standard next-token prediction, using natural "views" (e.g., text description + code) of the same underlying knowledge. Maintains generative capabilities while improving abstract representation quality. Consistent improvements across Llama3, Gemma2, OpenELM, and OLMo families. Key finding: minimizing the standard LLM loss does NOT implicitly minimize the JEPA objective—the embedding-space structure must be explicitly trained. Promotes near-linear transformations between view representations in embedding space.

### Relevance to Our Work
- Central finding echoes our premise: standard next-token prediction leaves information on the table in the embedding space. Our gain vector intervention and their JEPA loss are different levers targeting the same insight.
- Their "views" concept (multiple representations of the same knowledge) has a structural parallel to our colony design: multiple LLM instances producing different perspectives on the same prompt, with the collective signal capturing what no single forward pass surfaces.
- Tested on OLMo—our primary hybrid model—with positive results, suggesting this model family is particularly amenable to embedding-space interventions.
- Their finding that the JEPA objective acts as a regularizer without harming generative capability is relevant to our concern about whether gain vector interventions might degrade output quality.
- Potential future direction: could a colony-generated signal serve as an inference-time analog of the JEPA training objective? Rather than learning embedding structure through backpropagation, learn it through collective signal generation at inference time.

### Citation
Huang, H., LeCun, Y., & Balestriero, R. (2025). LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures. arXiv:2509.14252 [cs.CL].