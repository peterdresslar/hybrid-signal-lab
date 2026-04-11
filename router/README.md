## Router

**Router is currently exploratory research software. The benchmark work in this
repo uses router artifacts and profile-selection experiments, but routing
itself is not yet a finished headline contribution. The stable result today is
that prompt-level profile heterogeneity is real; online policy learning over
that heterogeneity remains ongoing work.**

`router` selects gain intervention profiles for hybrid language models at
inference time. Given an incoming prompt, it runs a baseline forward pass
(g=1.0), extracts features from the model's internal state, and routes the
prompt to one of a small set of pre-selected gain profiles — or to no
intervention at all.

The package now supports both major intervention regimes used in the study:

- `attention_contribution` for the Qwen balanced-attention experiments
- `block_output` for the OLMO balanced-block experiments

It operates on the same model backends as `signal_lab` but serves a different
purpose: where `signal_lab` sweeps the full profile × prompt space to map the
intervention response surface, `router` uses that map to study profile
selection and prompt-conditional control.

### Why routing matters

No single gain profile is universally best. The balanced 022 sweeps and the
030 benchmark runs show that optimal intervention varies by prompt, task, and
architecture. Fixed profiles can help, but prompt-level oracle selection is
consistently stronger and the winning profile distributions do not collapse to
a single geometry.

The intervention response also varies by architecture. Qwen 3.5 (pre-norm,
attention-contribution routing) and OLMo Hybrid (post-norm, block-output
routing) have different productive gain ranges, different winning profile
families, and different sensing signatures. The router therefore maintains
separate artifacts per model and intervention mode.

### Architecture

```
router/
  README.md
  __init__.py
  sensing.py            Baseline forward pass, feature extraction
  profiles.py           Profile set definitions (4 profiles per model)
  pipeline.py           Runtime helper functions for routed/fixed evaluation
  router.py             Artifact-backed routing classifier

  experiments/
    bistate_router.py   Binary off/on baseline using only PCA PC1/PC2
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

Each model uses a small intervention set chosen for **separability**, not just
raw effect size. The goal is to maximize prompt-level oracle utility with a set
whose members dominate in different regions of the response surface.

In the current 030 study protocol, `select_profiles.py` is run directly under a
matched separable objective with `--max-constants 1`, and the saved selection
artifacts are treated as the methodological receipt for each model.

The candidate pool is discovered from valid non-baseline rows in
`analysis_joined_long.csv`. The nominal balanced sweep library is shared across
models, but the usable candidate count can differ if some model/profile rows
are invalid at analysis time. In the current 022 data this yields:

- Qwen 9B: 87 usable non-baseline profiles
- OLMO: 86 usable non-baseline profiles

The OLMO count is lower because very-high-gain `constant_3` rows are invalid in
the joined analysis and therefore drop out of the effective candidate pool.

### Sensing features

The baseline forward pass provides the features the classifier trains on.
The primary signal is per-head attention entropy at each softmax attention
layer, paired with scalar output-distribution features such as final entropy,
margin, and baseline target probability.

The classifier uses a PCA projection of the entropy vector (typically the first
5–10 components) plus those scalar features. This keeps the feature dimension
well below the training set size (1,070 prompts in the current balanced battery).

### Training and evaluation

The routing classifier is trained on balanced sweep analysis data: for each of
the 1,070 prompts, we know the delta_p under the valid tested profiles. Given
the selected 4-profile set, each prompt's label is the profile (or "off") that
maximizes delta_p. Training uses cross-validation to estimate generalization.

Performance is reported as:

- **Oracle ceiling:** what a perfect router achieves (prompt-level best of
  the 4 profiles + off)
- **Routed:** what the trained classifier achieves
- **Best fixed profile:** what the single best overall profile achieves
- **Baseline:** no intervention (delta_p = 0 everywhere)

The gap between oracle and routed measures how much routing signal the
classifier captures. In the current benchmark study this is still exploratory:
the benchmark contribution comes from fixed-profile effects and prompt-level
oracle headroom, not from claiming routing is already solved.

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
