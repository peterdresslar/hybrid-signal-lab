# Router Series 050 Architecture Brief

This document replaces the earlier `050` sketch with a more direct design path
informed by **Lee et al. (2022), _Discovering sparse control strategies in neural
activity_**. The immediate goal is not to solve routing in one step, but to
identify a small bundle of control-relevant probe directions that can recover a
nontrivial fraction of the benchmark oracle headroom already exposed by the
`040` panel.

The central shift is from **classifying prompts into profile labels** to
**estimating control value along a small number of sparse probe directions**.

## 1. Current State: What `040` Established

The `040` series established three things clearly:

1. **The intervention panel is real.**
   On both Battery 4 and the downstream benchmarks, panel-bounded oracle gains
   are substantial. This is not a null result or a battery-local artifact.

2. **The current router is the weak link.**
   Benchmark runs show a repeated collapse toward a small subset of profiles,
   especially `constant_2.6`, even when benchmark oracle and best-fixed results
   clearly favor milder or shaped profiles such as `constant_1.45`,
   `plateau_bal_0.55`, or `bowl_bal_0.40`.

3. **Flat multinomial classification is the wrong abstraction.**
   The present router is trained to predict a winner label. That objective is
   too coarse for the real control problem, which is better described as:
   - should we intervene at all?
   - if so, what kind of intervention is locally valuable?
   - how much value is plausibly available?

The `040` results therefore motivate a move away from winner classification and
toward **continuous value estimation plus sparse probe selection**.

## 2. Inspiration: Lee et al. (2022)

Lee et al. propose a way to characterize collective control by probing local
sensitivity and identifying a small number of dominant perturbative modes. Two
ideas are especially relevant here:

1. **Control need not be dense.**
   Complex systems can be most sensitive along a small number of dominant
   directions, rather than requiring full-state intervention or dense
   observation.

2. **A useful control surface can be ranked.**
   Candidate perturbative directions can be compared by how much behavioral
   variation they explain, making it possible to retain only the highest-value
   modes.

Applied to effective geometry, this suggests the following interpretation:

- gain profiles are perturbative control directions
- baseline sequence/entropy features are the observable state
- benchmark or sweep `Δp` is the local control value to be predicted
- the right router is likely a **small probe bundle**, not a single winner
  classifier

This paper is therefore the direct methodological inspiration for `050`.

## 3. Secondary Conceptual Frame: Lynch & Daniels (2026)

The more recent Daniels-Lynch paper is not the implementation template for
`050`, but it is important for the longer-term direction of the project.

Its main relevance is conceptual:

- useful collective tuning often depends on operating near regime boundaries
- hysteresis and bistability are functionally important, not incidental
- tuning difficulty depends on the structure of the underlying control surface

These ideas support the future direction of:

- thresholded or sticky intervention states
- hysteresis across longer chats or sequences
- fast-slow control built atop a compact probe bundle

For `050`, however, Lee et al. remains the operational guide.

## 4. Diagnosis: Why `040` Fails

The immediate `040` benchmark failure should be understood as a **representation
and objective mismatch**, not as evidence against effective geometry itself.

The main failure modes are:

1. **Winner labels are too brittle.**
   A prompt whose best profile is only slightly better than its second-best
   profile is treated identically to one with a large winner margin.

2. **Benchmark tasks demand value comparison, not forced class assignment.**
   The router should be comparing the expected utility of profiles, not forcing
   every prompt into a learned bucket.

3. **The useful signal is likely distributed.**
   Current sequence plots suggest that the geometry contains meaningful but
   distributed structure across several PCs and several feature families.

4. **The current controller is too monolithic.**
   It does not separate:
   - interventionability
   - profile family
   - intervention strength
   - confidence / abstention

In short: `040` failed because it tried to compress a structured control problem
into a single multiclass label.

## 5. Path Forward: The `050` Sparse Probe Bundle

`050` will replace label classification with a **bundle of linear value probes**
trained directly on profile utility.

### Core design

For each candidate profile in the panel:

- train a linear probe to predict prompt-level `Δp`
- interpret the resulting weight vector as a candidate control direction
- compare probes by validation utility, not just regression loss

The routing decision becomes:

1. estimate `Δp` for each panel profile
2. identify the best predicted profile
3. abstain if the predicted gain is below a threshold
4. otherwise apply the best predicted profile

This is a better match to the actual control problem because it asks:

- what is the value of this perturbation here?

rather than:

- which label does this prompt belong to?

### Why this is preferable to `040`

- avoids softmax winner forcing
- preserves profile-level utility information
- allows multiple strong constant traces to coexist naturally
- supports thresholding directly on predicted gain
- creates a natural ranking over control modes

## 6. Candidate Banks

`050` should not start from one giant undifferentiated feature slab. It should
start from a small number of explicit candidate banks and let the probes compete.

### Recommended initial banks

1. **Baseline entropy/scalar bank**
   - entropy PCs
   - baseline scalar features

2. **Raw sequence bank**
   - `all_layers_mean_pool_concat`
   - `final_layer_mean_pool`
   - `embedding_last_token`

3. **Length-residualized sequence bank**
   - same families with `length_resid`

4. **Attention-residualized sequence bank**
   - same families with `attn_resid`

The current implementation should prioritize:

- `all_layers_mean_pool_concat`
- `final_layer_mean_pool`
- `embedding_last_token`

because these together cover:

- the strongest “serious” geometry
- the cleanest pooled representation
- the weird but empirically useful shallow switching signal

### Dimensionality

Current `10`-PC runs are likely too compressed for this objective. `050` should
explicitly test broader banks, likely in the range:

- `25`
- `50`
- `100`

with regularization selected by cross-validation.

## 7. Probe Targets

The first probes should not all target exact winner identity. Better targets are:

1. `Δp(profile)` for each panel profile
2. best predicted gain over baseline
3. `off` vs `on`
4. constant-like vs shaped-like
5. mild vs strong intervention regime
6. winner margin / local headroom

This creates the possibility of a layered controller:

- gate probe
- family probe
- profile-value probes

But `050` should begin with the simplest viable form:

- one value probe per profile
- threshold on predicted gain

## 8. Minimal `050` Implementation

### Training artifact

Create:

- `router/experiments/train_probes.py`

This module should:

1. load the same feature banks as the current router infrastructure
2. fit one regularized linear regressor per panel profile
3. evaluate each bank independently and in selected combinations
4. save:
   - probe weights
   - intercepts
   - feature metadata
   - residualization metadata
   - validation metrics

### Runtime artifact

Create:

- `router/probe_router.py`

This runtime should:

1. extract the requested feature bank
2. run all profile probes
3. compare predicted values
4. abstain if best predicted value `< threshold`
5. otherwise apply the best predicted profile

### Decision rule

The initial rule should be:

- `apply argmax_profile(predicted_delta_p)` if `max_predicted_delta_p > tau`
- else `off`

Possible later refinements:

- require margin over second-best
- require positive advantage over `off`
- add OOD or leverage-based penalty

## 9. Success Criterion

`050` does **not** need to solve routing.

The target is:

- recover a meaningful fraction of benchmark oracle headroom

A practical threshold is:

- roughly one quarter of the oracle gap on at least some benchmark tasks

This is enough to establish that the headroom is not merely observational, but
partially capturable by a learned compact control surface.

## 10. What `050` Is Really For

`050` is not just another router attempt.

It is the first implementation step toward a larger project:

- discovering a low-dimensional control surface for effective geometry
- ranking dominant control-relevant directions
- building a sparse probe bundle
- later extending that bundle to hysteretic and multi-timescale control

In that sense, `050` is best understood as:

**the first sparse-control probe infrastructure for effective geometry.**
