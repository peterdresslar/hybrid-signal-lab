"""
sensing.py — Extract routing features from a baseline forward pass.

Takes the verbose result dict from Agent.run_pass() and produces a
feature vector suitable for the trained routing classifier. The
feature pipeline mirrors what train_router.py uses:

  1. Per-head attention entropy at the final token → flat vector
  2. PCA projection of the entropy vector using saved components
  3. Scalar summary features (entropy, logit margin, etc.)
  4. Concatenation and standardization

The PCA components, standardization parameters, and other artifacts
are loaded from the router_model.json produced by train_router.py.
"""

from __future__ import annotations

import numpy as np


def extract_entropy_vector(pass_result: dict) -> np.ndarray:
    """Extract the flat per-head attention entropy vector from a verbose pass result.

    Args:
        pass_result: dict from Agent.run_pass(return_verbose=True)

    Returns:
        1-D numpy array of shape (n_layers * n_heads,)
    """
    per_layer = pass_result["attn_entropy_per_head_final"]
    flat = []
    for layer_entropies in per_layer:
        flat.extend(layer_entropies)
    return np.array(flat, dtype=np.float64)


def extract_scalar_features(pass_result: dict) -> dict[str, float]:
    """Extract scalar summary features from a verbose pass result.

    Returns a dict with the same keys used during training:
        baseline_attn_entropy_mean
        baseline_final_entropy_bits
        baseline_mean_entropy_bits
        baseline_target_prob
        baseline_top1_top2_logit_margin

    Note: Some features require information not in the standard pass
    result dict. We compute what we can and fall back to 0.0.
    """
    # Attention entropy mean: mean of the flat entropy vector
    per_layer = pass_result.get("attn_entropy_per_head_final", [])
    all_entropies = []
    for layer_e in per_layer:
        all_entropies.extend(layer_e)
    attn_entropy_mean = float(np.mean(all_entropies)) if all_entropies else 0.0

    # Final entropy (output distribution)
    final_entropy = pass_result.get("final_entropy_bits", 0.0)

    # Mean entropy across all positions
    mean_entropy = pass_result.get("mean_entropy_bits", 0.0) or 0.0

    # Target probability from baseline
    target_prob = pass_result.get("target_prob", 0.0) or 0.0

    # Logit margin: top1 - top2
    top_logits = pass_result.get("top_k_logits", None)
    if top_logits is not None and len(top_logits) >= 2:
        logit_margin = float(top_logits[0] - top_logits[1])
    else:
        logit_margin = 0.0

    return {
        "baseline_attn_entropy_mean": attn_entropy_mean,
        "baseline_final_entropy_bits": final_entropy,
        "baseline_mean_entropy_bits": mean_entropy,
        "baseline_target_prob": target_prob,
        "baseline_top1_top2_logit_margin": logit_margin,
    }


def build_feature_vector(
    pass_result: dict,
    *,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    standardization_mean: np.ndarray,
    standardization_std: np.ndarray,
    feature_set: str = "pca+scalar",
) -> np.ndarray:
    """Build the full standardized feature vector for the classifier.

    Args:
        pass_result: dict from Agent.run_pass(return_verbose=True)
        pca_components: (n_pca, entropy_dim) from router_model.json
        pca_mean: (entropy_dim,) centering vector
        standardization_mean: (n_features,) from training
        standardization_std: (n_features,) from training
        feature_set: "pca+scalar", "pca", "scalar", or "raw"

    Returns:
        1-D numpy array ready for classifier input
    """
    entropy_vec = extract_entropy_vector(pass_result)
    scalar_feats = extract_scalar_features(pass_result)

    parts = []

    if feature_set in ("pca", "pca+scalar"):
        # Project entropy vector into PCA space
        centered = entropy_vec - pca_mean
        pca_proj = centered @ pca_components.T  # (n_pca,)
        parts.append(pca_proj)

    if feature_set == "raw":
        parts.append(entropy_vec)

    if feature_set in ("scalar", "pca+scalar"):
        # Scalar features in sorted key order (matches training)
        scalar_names = sorted(scalar_feats.keys())
        scalar_arr = np.array([scalar_feats[k] for k in scalar_names], dtype=np.float64)
        parts.append(scalar_arr)

    x = np.concatenate(parts)

    # Standardize using training statistics
    x = (x - standardization_mean) / standardization_std

    return x
