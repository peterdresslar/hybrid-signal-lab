# Synopsis of Capstone Options

**Peter Dresslar**

**CAS Capstone, Arizona State University**

**Advisor: Prof. Bryan Daniels**

**March 2026**

Continuing my work from class, I am exploring the ways in which colony-wide signals can be generated and used to modulate the performance of individual AI agents in a multi-agent system (MAS). A major challenge for this work is that LLMs have notoriously opaque internal workings that are difficult to change once training is complete. Of course, contrary to the sensory and signalling capabilities of biological colony agents such as eusocial insects, modern AI agents particularly lack mechanisms for low-dimensional collective signal exchange. However, my limited initial experiments suggest that there may be a couple of control planes on which to experiment. The available literature on this topic---the centroid for which occurs perhaps three months ago---confirms some of these possibilities. There continues to be a stark lack of recognition of collective logic in even the most academic of these articles.

A highly technical but perhaps not [impenetrable blog post](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight) from this month details the many advances in open-source AI models from the first 2+ months of 2026. The roundup illuminates a few interesting directions in particular. First, we might note that there is considerable interest in different architectures for attention in LLMs. Attention mechanisms control how much information from the input context is passed on to the next layer of the model's processing, which can have a major effect on the amount of processing cost/time/energy required for output inference to occur. It also has an effect on output quality, and---critically---this effect is situationally dependent. Particularly compelling is the rise in "hybrid" approaches to attention. We might also note that, while perhaps not quite of the moment as attention architectures, the Feed Forward Network (FFN) architecture in models also sees significant variation with various labs, and this variation is of particular interest due to the nature of the task FFNs perform. FFNs essentially "crystallize" meaning of the gathered layer values, and the fact that many of the annotated variations from the article come with a "router" label should be a tip-off that there is an opportunity for more clever signal modulation.

While the blog post reports solely on non-collective, individual LLMs, the project approach I am pursuing is with collectives in the form of MAS. Multi-agent systems continue to be a significant topic of interest, and are the easiest form to consider when seeking avenues through which collective logic might be applied to the operation of AI agents. For instance, it may simply be of interest in some cases to perform experimental analyses with the goal of answering scientific questions about the logic itself. However, we of course should stay aware of the enormous opportunity for applied work in this space. Here, a non-trivial feature of applications is what we might call the "fungibility of collectivity" in an artificial space. At least to a certain degree, whether we are working with 50 LLM instances or 1 LLM instance is an arbitrary distinction, since the single LLM can simply be polled 50 times. More pointedly, a detected advantage from collectivity might lead to a new model design that replicates the collective logic desired in a single LLM instance. This discussion connects quite explicitly to the examination of organisms and superorganisms we have been discussing (and as outlined in the "companion_paper" document.)

## Project Options

I am evaluating three possible approaches for my Applied Project.

### Option 1: Attention Is In The Air

In this project we build a prototype of a *collective-signal driven architectural model modulation with formal collective logic measures*. We particularly consider an MAS construction that collects colony-wide signal vectors and applies them to agent hybrid attention layer tuning for one, some, or all agents. Questions to consider include:

1. Can we identify, characterize, and quantify system amplification in either homogeneous and heterogenous agent conditions?
2. Can we measure and assign system-wide information?
3. Can we measure an improvement in system characteristics of the prototype in terms of performance or cost?

There is an extended (and constantly updating) discussion of this project at the document, [proposal.md](proposal.md). This project would be most useful for me personally and seems to have the most potential value for the Lab and/or department. It will not be easy and would require risk mitigation and/or fallback strategies. Additionally, this project would require the acquisition of some compute time with ASU's HPC facilities. I expect to have a better feel for the specific scope of AIITA by Tuesday March 17.

### Option 2: Companion Paper

In this project we develop the theoretical argument outlined in the [companion_paper.md](companion_paper.md) document: a formal definition of Kantian Individuals as a synthesis of Krakauer (information preservation), Kauffman (constraint closure), and Hoel (causal emergence), followed by a sharpened definition of superorganisms as Kantian Individuals that lack constraint closure. The paper argues that AI systems are more likely to build superorganisms than organisms, and that this is a feature rather than a limitation. A small empirical demonstration using the Colony testbed would accompany the theoretical discussion, showing that a coupled MAS exhibits measurable amplification and causal emergence while failing the constraint closure test. Questions to consider include:

1. Can we articulate a testable definition of "superorganism" that is both novel and useful to the complexity science and AI communities?
2. Can we demonstrate, even minimally, that an AI collective satisfies the proposed criteria?
3. Does the framing generate predictions that distinguish it from existing accounts of multi-agent coordination?

This project has the advantage of being closest to completion and least dependent on technical risk. The argument is largely written and the experimental component can be scoped to fit available time and compute. The product could represent a satisfying contribution to the literature on individuality and collective behavior, and would serve the Lab's interests in extending the formal framework to artificial systems. However, it may not be the most mercenary approach, and there is the risk that the synthesis framework may be rejected by the reviewer or scientific community. In essence, this approach may be seen to be trading technical risk for conceptual risk. The reviewer's feedback (and possibly venue selection ideas) will be useful to measure this tradeoff.

### Option 3: Literature Review and Synopsis

In this project we produce a survey and research agenda: "Collectivity Measures for Artificial Systems," or similar. We map the emerging intersection of formal collective logic (Daniels et al., Krakauer, Hoel) with multi-agent AI engineering, particularly the recent and rapidly growing literature on latent inter-agent communication (activation sharing, KV pair exchange, state delta trajectories). The review would identify the stark absence of collective-theoretic measurement in the AI literature, catalog the available tools and frameworks, and lay out a concrete research program — including both AIITA and the companion paper's superorganism argument — as proposed future work. Questions to consider include:

1. Can we convincingly demonstrate that formal collectivity measures are applicable to, and missing from, the AI systems literature?
2. Can we synthesize the biological and AI literatures in a way that is useful to researchers in both fields?
3. Does the resulting research agenda provide a clear and actionable roadmap for the Lab?

This project is the most completable within the available timeline and carries the least technical and conceptual risk. The literature work is necessary regardless: it forms the foundation for both other options. It would position the Lab as the authority on this intersection and produce a document that could anchor a grant proposal or a lab website. The tradeoff is that it does not produce an experimental result or a novel theoretical claim, and would appear on a resume as scholarship rather than as building. The risk here is opportunity loss.
