## Router

**Router is currently under development. While it is not prodution-ready for candidate *selection*, the experiments are
currently under use for benchmark experimentation. Thus this is versioned, but not fully-functioning, software.**

`router` selects gain intervention profiles for hybrid language models at
inference time. Given an incoming prompt, it runs a baseline forward pass
(g=1.0), extracts features from the model's internal state, and routes the
prompt to the most advantageous of a small set of pre-selected gain profiles —
or to no intervention at all.

The package targets the **attention_contribution** intervention mode
exclusively. It operates on the same model backends as `signal_lab` but serves
a different purpose: where `signal_lab` sweeps the full profile × prompt space
to map the intervention response surface, `router` exploits that map to make
per-prompt decisions.

### Why routing matters

No single gain profile is universally best. The b4_021 attention-contribution
data shows that optimal intervention varies by prompt: `edges_narrow` yields
+0.45 delta_p on code comprehension prompts for Qwen 9B but damages
long-range retrieval. `late_boost_1.5` helps cultural memorization but does
nothing for numerical reasoning. A fixed profile must compromise; a router can
specialize.

The intervention response also varies by architecture. Qwen 3.5 (pre-norm)
and OLMo Hybrid (post-norm) have different productive gain ranges, different
winning profile families, and different attention head signatures predicting
intervention benefit. The router therefore maintains separate routing logic
per model.

### Architecture

```
router/
  README.md
  __init__.py
  sensing.py            Baseline forward pass, feature extraction
  qwen_router.py        Qwen 3.5 routing: profile selection classifier
  olmo_router.py        OLMo Hybrid routing: profile selection classifier
  profiles.py           Profile set definitions (4 profiles per model)
  evaluate.py           Evaluation harness: oracle vs routed vs fixed vs baseline

  experiments/
    bistate_router.py  Binary off/on baseline using only PCA PC1/PC2
    select_profiles.py  Combinatorial search for optimal 4-profile sets
    score_profile_sets.py
                        Two-stage ranking: oracle shortlist, then CV router scoring
    train_router.py     Train routing classifiers from b4_021 data
    eval_router.py      Cross-validated evaluation of trained routers
```

### Routing pipeline

1. **Sense.** Run the prompt through the model with g=1.0 at all attention
   layers. Extract the per-head attention entropy vector (128 features for
   Qwen 9B, 240 for OLMo) plus output-distribution features (final entropy,
   top-1/top-2 logit margin).

2. **Classify.** Feed the feature vector to a trained multinomial classifier
   (logistic regression on PCA-reduced features). The classifier outputs a
   probability distribution over {profile_1, profile_2, profile_3, profile_4,
   off}.

3. **Intervene.** Apply the selected gain profile using the existing
   `model.backend` hook infrastructure, or skip intervention if the
   classifier selects "off."

### Profile selection

Each model has 4 intervention profiles chosen for **separability**, not just
raw effect size. The goal is to maximize routed performance across the full
prompt distribution, which means choosing profiles that each dominate in
different regions of the feature space.

In current experiments this is a two-stage process:

1. `select_profiles.py` finds high-value candidate sets by oracle-routed
   performance, with optional constraints on coverage, class usage, within-set
   correlation, and number of constant profiles.
2. `score_profile_sets.py` re-ranks the top candidate sets by actual
   cross-validated routing performance using the same PCA/scalar baseline
   features that the deployed router will see.

For Qwen 9B, the b4_021 data shows distinct intervention regimes:
code/numerical prompts respond to high-edge profiles, reasoning-tracking
prompts respond to early-boost, retrieval/memorization prompts respond to
gentle late-boost, and ~20% of prompts are better left at baseline. The 4
profiles are selected to cover these regimes with minimal overlap.

For OLMo Hybrid, intervention effects are smaller but more niche-specific.
Profile selection focuses on the task × profile pairings where intervention
produces reliable improvement (code × triad_odd, retrieval × spike_p5,
structural × bowl, tracking × bookend_suppress).

The profile sets are determined by a combinatorial search over all C(78,4)
candidate sets, scored by oracle-routed mean delta_p at the prompt level.

### Sensing features

The baseline forward pass provides the features the classifier trains on.
The primary signal is per-head attention entropy at each softmax attention
layer — the same data that produces the scout head rankings in `signal_lab`
analysis. PCA analysis of these vectors shows that PC1 alone (72.7% of
variance for 9B, 59.4% for OLMO) strongly predicts intervention sensitivity:
high-PC1 prompts barely respond to any intervention, while low/mid-PC1
prompts show large, profile-dependent effects.

The classifier uses a PCA projection of the entropy vector (first 5–10
components) plus scalar output-distribution features. This keeps the feature
dimension well below the training set size (1,070 prompts from battery 4).

### Training and evaluation

The routing classifier is trained on b4_021 data: for each of the 1,070
prompts, we know the delta_p under all 78 profiles. Given the selected
4-profile set, each prompt's label is the profile (or "off") that maximizes
delta_p. Training uses cross-validation to estimate generalization.

Performance is reported as:

- **Oracle ceiling:** what a perfect router achieves (prompt-level best of
  the 4 profiles + off)
- **Routed:** what the trained classifier achieves
- **Best fixed profile:** what the single best overall profile achieves
- **Baseline:** no intervention (delta_p = 0 everywhere)

The gap between oracle and routed measures how much routing signal the
classifier captures. The gap between routed and best-fixed measures the
practical value of routing over a static intervention.

Before training the full multiclass router, `bistate_router.py` provides a
scientifically cleaner baseline: route between `off` and a single fixed
constant profile using only `PC1` and `PC2` from the baseline attention-entropy
PCA. This measures how much value is available from a simple switch before
asking whether a richer profile-selection router is justified.

### Relationship to signal_lab

`router` depends on `signal_lab` for model loading, intervention hooks, and
inference execution. It does not duplicate the sweep or analysis machinery.
The relationship is:

- `signal_lab` generates the data (sweeps) and analysis (scout heads, PCA)
- `router/experiments/` uses that data to select profiles and train classifiers
- `router/` at runtime uses `model.backend` for inference and hook management

### Relationship to colony (future)

The `colony` concept — collective signal generation through multiple LLM
instances — may return as a higher-level coordination layer. If it does,
`router` would become one component within it: each colony member would use
its own router for per-prompt intervention decisions, and colony-level logic
would coordinate across members. For now, `router` is self-contained.

### References

The router design draws on several lines of work:

Wang, W., Yang, J., & Peng, W. (2025). Semantics-Adaptive Activation
Intervention for LLMs via Dynamic Steering Vectors. *ICLR 2025*.
Constructs per-input dynamic steering vectors by projecting activations onto
contrastive directions identified from critical attention heads and neurons.
The closest published analog to our approach: input-adaptive intervention
on the same model, rather than routing between models.

Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E.,
Kadous, M., & Stoica, I. (2025). RouteLLM: Learning to Route LLMs with
Preference Data. *ICLR 2025*. Trains lightweight classifiers on outcome
data to route queries between model configurations at inference time.
Demonstrates that routers can be trained from existing evaluation data
without new human labels — the methodology we adapt for profile selection
from b4_021 sweep results.

Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2024).
Inference-Time Intervention: Eliciting Truthful Answers from a Language
Model. *NeurIPS 2024*. Trains linear probes on attention head activations
to identify intervention-relevant heads, then applies fixed activation
shifts at inference time. The probe-on-hidden-states methodology informs
our sensing pass feature extraction and the scout head analysis.

Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J.,
Mini, U., & MacDiarmid, M. (2024). Steering Language Models With
Activation Engineering. Formalizes multiplicative and additive intervention
on internal activations, demonstrating that effects are layer-position-
dependent and composable. Provides the theoretical basis for depth-varying
gain profiles.

Li, Z., Zhang, T., Liu, J., & Zhang, H. (2025). Reasoning Models Know
When They're Right: Probing Hidden States for Self-Verification.
Shows that hidden states from a single forward pass encode reliable
information about output correctness, extractable with lightweight probes.
Demonstrates that baseline-pass internal features carry sufficient signal
for inference-time routing decisions.
