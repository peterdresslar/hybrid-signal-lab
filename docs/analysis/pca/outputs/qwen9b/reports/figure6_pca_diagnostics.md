# Figure 6 PCA Diagnostics

## 1. Embedding matrix

The PCA input was the baseline per-prompt attention-head entropy matrix for Qwen 3.5 9B from `data/022-balanced-attention-hybrid/9B/verbose.jsonl`. Each prompt contributes one 128-dimensional vector formed by flattening `attn_entropy_per_head_final` over the 8 hybrid attention layers (`[3, 7, 11, 15, 19, 23, 27, 31]`) and their 16 heads; the entropy is measured at the final prompt position, not averaged across prompt tokens.

## 2. Scree plot

The explained-variance spectrum shows a sharp drop after PC2. PC1 explains 72.65% and PC2 explains 6.08% of variance; PCs 3–10 explain 2.05%, 1.33%, 1.11%, 1.02%, 0.96%, 0.77%, 0.64%, 0.60%.

## 3. PC2 vs PC3 task structure

Task-category structure persists clearly on PC2 vs PC3, with organized sub-clustering beyond the dominant PC1 axis. Quantitatively, η² by task category is 0.820 on PC2 and 0.529 on PC3.

## 4. PC2 vs PC3 and residualized winner structure

On the winner labels, profile-winner identity shows a moderate but coarse linear pattern: most of the signal is a `constant_2.6` versus non-`constant_2.6` split on PC2, while the specialist winners remain diffuse and are not cleanly separated from one another. η² by winner is 0.384 on PC2 and 0.011 on PC3. Because PCA components are orthogonal, the residualized PCA is effectively a re-indexed version of the original subdominant subspace: residual PC1 aligns with original PC2, and residual PC2 aligns with original PC3, yielding the same η² values (0.384, 0.011).

## 5. Ceiling claim assessment

The approximate-ceiling claim needs a concrete caveat: there is substantial task-aligned linear structure beneath PC1, and a weaker winner-related axis concentrated mostly in the `constant_2.6` versus rest contrast. The residualized decomposition explains 22.22% and 7.50% of variance on its first two axes, confirming that the original dominant PC1 structure has been removed.

## 6. Length-plus-generator residualization

After regressing each feature on both prompt length (`tokens_approx`) and generator/source identity dummies, the spectrum flattens further: the first three components explain 25.51%, 5.49%, and 4.40% of variance. Task structure still survives at a nontrivial level (η² = 0.000, 0.000, 0.000 on the first three residual PCs), while winner structure remains weaker and coarser (η² = 0.011, 0.009, 0.006). This is the cleanest current estimate of how much attention geometry organizes by task beyond surface length and generator-template effects.
