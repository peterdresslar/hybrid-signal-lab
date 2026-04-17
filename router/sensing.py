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
import torch


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


def extract_num_tokens(pass_result: dict) -> float:
    """Extract actual prompt token count from a baseline pass result."""
    if "num_tokens" in pass_result and pass_result["num_tokens"] is not None:
        return float(pass_result["num_tokens"])
    attention_mask = pass_result.get("attention_mask")
    if attention_mask is not None:
        return float(np.asarray(attention_mask).sum())
    input_ids = pass_result.get("input_ids")
    if input_ids is not None:
        return float(np.asarray(input_ids).size)
    return 0.0


def extract_sequence_family_vector(pass_result: dict, family_name: str) -> np.ndarray:
    """Extract one hidden-state feature family from a baseline pass result."""
    hidden_states = pass_result.get("hidden_states")
    if hidden_states is None:
        raise ValueError(
            f"Pass result is missing hidden_states required for sequence family '{family_name}'."
        )
    if isinstance(hidden_states, torch.Tensor):
        hs = hidden_states.detach().cpu().numpy()
    else:
        hs = np.asarray(hidden_states)

    if family_name == "embedding_last_token":
        vec = hs[0, -1, :]
    elif family_name == "final_layer_last_token":
        vec = hs[-1, -1, :]
    elif family_name == "embedding_mean_pool":
        vec = hs[0].mean(axis=0)
    elif family_name == "final_layer_mean_pool":
        vec = hs[-1].mean(axis=0)
    elif family_name == "all_layers_last_token_concat":
        vec = hs[:, -1, :].reshape(-1)
    elif family_name == "all_layers_mean_pool_concat":
        vec = hs.mean(axis=1).reshape(-1)
    else:
        raise ValueError(f"Unknown sequence family '{family_name}'")

    return np.asarray(vec, dtype=np.float64)


def build_feature_vector(
    pass_result: dict,
    *,
    pca_components: np.ndarray | None,
    pca_mean: np.ndarray | None,
    standardization_mean: np.ndarray,
    standardization_std: np.ndarray,
    sequence_family: str | None = None,
    sequence_pca_components: np.ndarray | None = None,
    sequence_pca_mean: np.ndarray | None = None,
    sequence_residualization: str = "raw",
    sequence_residual_beta: np.ndarray | None = None,
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

    if feature_set in ("pca", "pca+scalar", "pca+sequence_pca", "pca+scalar+sequence_pca"):
        # Project entropy vector into PCA space
        if pca_components is None or pca_mean is None:
            raise ValueError(f"Feature set '{feature_set}' requires entropy PCA artifacts.")
        centered = entropy_vec - pca_mean
        pca_proj = centered @ pca_components.T  # (n_pca,)
        parts.append(pca_proj)

    if feature_set == "raw":
        parts.append(entropy_vec)

    if feature_set in ("scalar", "pca+scalar", "scalar+sequence_pca", "pca+scalar+sequence_pca"):
        # Scalar features in sorted key order (matches training)
        scalar_names = sorted(scalar_feats.keys())
        scalar_arr = np.array([scalar_feats[k] for k in scalar_names], dtype=np.float64)
        parts.append(scalar_arr)

    if "sequence_pca" in feature_set:
        if sequence_family is None or sequence_pca_components is None or sequence_pca_mean is None:
            raise ValueError(f"Feature set '{feature_set}' requires sequence PCA artifacts.")
        sequence_vec = extract_sequence_family_vector(pass_result, sequence_family)
        if sequence_residualization not in ("raw", "length_resid", "attn_resid"):
            raise ValueError(f"Unknown sequence residualization '{sequence_residualization}'.")
        if sequence_residualization != "raw":
            if sequence_residual_beta is None:
                raise ValueError(
                    f"Feature set '{feature_set}' with residualization '{sequence_residualization}' "
                    "requires sequence_residual_beta."
                )
            design_parts = [1.0, extract_num_tokens(pass_result)]
            if sequence_residualization == "attn_resid":
                design_parts.append(scalar_feats["baseline_attn_entropy_mean"])
            design = np.asarray(design_parts, dtype=np.float64)
            sequence_vec = sequence_vec - design @ sequence_residual_beta
        centered = sequence_vec - sequence_pca_mean
        sequence_proj = centered @ sequence_pca_components.T
        parts.append(sequence_proj)

    x = np.concatenate(parts)

    # Standardize using training statistics
    x = (x - standardization_mean) / standardization_std

    return x
