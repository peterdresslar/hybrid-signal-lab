State of the PCA

  At present, the PCA work in this project is strongest as a baseline-structure
  analysis rather than as a direct intervention analysis. The principal dataset
  so far is the Qwen 3.5 9B baseline per-prompt attention-head entropy matrix,
  formed by flattening attn_entropy_per_head_final across the eight softmax
  attention layers and their sixteen heads, yielding a 128-dimensional vector
  for each of the 1,070 battery prompts. In other words, the PCA that is
  currently mature is a PCA over final-token baseline attention geometry, not
  over full residual-stream hidden states and not over all prompt tokens. That
  distinction matters, because it defines both what the present results
  genuinely say and what the new analyses may yet change.

  The clearest result from this PCA is that the baseline attention-entropy space
  is very strongly structured, but that the dominant structure is not yet the
  one we ultimately care most about. In Qwen 9B, PC1 absorbs the large majority
  of variance (about 72.7%), with PC2 much smaller (about 6.1%) and PC3 smaller
  still (about 2.1%). This dominant first axis is heavily loaded on prompt
  length: in the cross-model summary, the correlation between PC1 and
  tokens_approx is large in magnitude across all four hybrid runs, roughly (|r|
  \approx 0.74) to (0.82). So the first-order state of the PCA is that the
  baseline attention geometry is not random at all, but its most obvious axis is
  strongly entangled with surface prompt length.

  That said, the subdominant structure is not washed out by this. In Qwen 9B,
  task-category structure survives very clearly beneath PC1. The original Figure
  6 diagnostics showed strong task clustering on the PC2/PC3 plane, with task η²
  around 0.82 on PC2 and 0.53 on PC3. This is one of the more encouraging PCA
  findings in the project so far: once the giant length-aligned axis is no
  longer monopolizing the picture, prompt type is still strongly organizing the
  geometry. In other words, the baseline attention state appears to carry real
  task information beyond simple prompt size.

  Cross-model summaries suggest that this is not a purely Qwen-specific
  artifact. Across the hybrid runs, baseline PCA behaves similarly in broad
  outline: a dominant PC1, substantial prompt-length loading, and persistent
  task structure after length residualization. For Qwen 2B, 9B, and 35B, the
  first residualized PC still explains a large share of remaining variance
  (roughly 0.53 to 0.57), and type η² remains fairly high on the first two
  residualized axes. Olmo is somewhat different in shape: its raw PC1 explains
  less variance (about 0.59), PC2 and PC3 are somewhat larger than in Qwen, and
  the task structure is distributed differently across components. But the broad
  conclusion is the same: baseline attention geometry is structured by task in
  both architectures, even though the exact decomposition differs.

  A second important result is that winner-related structure exists, but is
  weaker and coarser than task structure. In the Qwen 9B diagnostics, oracle-
  winner identity does show up in the PCA space, but much of that signal is
  concentrated in a constant_2.6 versus non-constant_2.6 split rather than in a
  rich separation among many specialist winners. Pooled η² of residual PC2 by
  winner is modest, and within-class analyses suggest that the winner effect is
  not uniformly distributed across task classes. The strongest within-class
  winner structure appears in computational families such as code_comprehension
  and algorithmic, where the PCA seems to carry some information about which
  intervention profile will win, but even there the effect is not especially
  fine-grained. So the present PCA gives some evidence that baseline attention
  state is related to intervention response, but not yet in a way that would
  justify a strong “the PCA predicts the exact best profile” claim.

  A third useful finding is negative in a productive way. The pooled correlation
  between residual PC2 and baseline mean entropy looked substantial when prompts
  were pooled together, but the within-class version of that effect largely
  disappeared. In the current Figure 6c summary, the pooled partial correlation
  is about (-0.307), while the weighted mean within-class correlation is near
  zero. That strongly suggests that a meaningful fraction of the pooled PCA-
  versus-entropy relationship is mediated by class composition rather than by a
  smooth within-class latent continuum. This is important because it prevents us
  from overreading the PCA as a universal scalar “difficulty” or “confidence”
  axis. At the moment, the more defensible interpretation is that the PCA
  captures a mixture of prompt length, task-family structure, and some weaker
  winner-related geometry, rather than a single elegant latent factor.

  So the current “state of the PCA” is fairly clear. The work already supports
  four moderate claims. First, baseline attention-head entropy geometry is
  highly structured and far from isotropic. Second, prompt length is the
  dominant first-order axis in that geometry, especially for Qwen. Third,
  substantial task structure survives beneath that axis and persists after
  residualization. Fourth, there is some winner-related signal, but it is coarse
  and unevenly distributed, not yet strong enough to treat PCA as a mature
  routing solution on its own.

  Just as importantly, the current PCA also has clear limits. It is based on
  baseline-only data, on final-token attention-head entropy summaries, and
  mostly on Qwen 9B diagnostics. It is therefore not yet a full account of
  residual-state organization, nor a full cross-model representation analysis,
  nor a decisive answer to how intervention-sensitive geometry evolves across
  the token sequence. The new residual-stream PCA work and the TDA work are
  therefore not merely more data of the same kind; they are probing a genuinely
  different object. If those analyses succeed, they may sharpen or partly
  replace the current picture. If they do not, the existing PCA still stands as
  evidence that the baseline attention state contains meaningful task-aligned
  structure and a weaker, partial trace of intervention-response structure.

  If you want a more compact closing sentence for internal notes, I would put it
  this way: the PCA has already done enough to justify its inclusion in the
  project, but not yet enough to carry a central mechanistic claim. Right now it
  is best understood as an informative structural probe: strong on baseline
  geometry, suggestive on task organization, intriguing but still limited on
  winner prediction.

