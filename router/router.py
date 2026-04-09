"""
router.py — Inference-time intervention router.

Implements the two-pass routing pipeline:
  1. Run a baseline forward pass (g=1.0) to sense the prompt
  2. Extract features from the baseline pass (sensing.py)
  3. Classify the prompt → select a gain profile or "off"
  4. If a profile is selected, run a second forward pass with that profile

The router loads its classifier weights and feature-engineering
artifacts from router_model.json, produced by train_router.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from model.backend import InterventionMode, normalize_intervention_mode
from router.sensing import build_feature_vector
from router.profiles import get_profile_specs, BASELINE_SPEC


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax (handles 1-D input too)."""
    z = np.atleast_2d(z)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).squeeze()


class InterventionRouter:
    """Predicts the best gain profile for a prompt from baseline features.

    Typical usage::

        router = InterventionRouter.from_artifacts(
            "data/intervention_modes/b4_021_attn_contr/9B/router/router_model.json"
        )
        decision = router.classify(baseline_pass_result)
        # decision = {"profile_name": "edges_narrow", "profile_spec": {...},
        #             "confidence": 0.72, "class_probs": {...}}
    """

    def __init__(
        self,
        *,
        model_key: str,
        class_names: list[str],
        profile_specs: dict[str, dict],
        weights: np.ndarray,
        bias: np.ndarray,
        pca_components: np.ndarray | None,
        pca_mean: np.ndarray | None,
        standardization_mean: np.ndarray,
        standardization_std: np.ndarray,
        feature_set: str,
        intervention_mode: InterventionMode = InterventionMode.ATTENTION_CONTRIBUTION,
    ):
        self.model_key = model_key
        self.class_names = class_names
        self.profile_specs = profile_specs
        self.W = weights
        self.b = bias
        self.pca_components = pca_components
        self.pca_mean = pca_mean
        self.standardization_mean = standardization_mean
        self.standardization_std = standardization_std
        self.feature_set = feature_set
        self.intervention_mode = normalize_intervention_mode(intervention_mode)

    @classmethod
    def from_artifacts(cls, model_path: str | Path) -> "InterventionRouter":
        """Load a trained router from a router_model.json file."""
        model_path = Path(model_path)
        with open(model_path) as f:
            artifacts = json.load(f)

        model_key = artifacts["model_key"]

        pca_components = None
        pca_mean = None
        if "pca_components" in artifacts:
            pca_components = np.array(artifacts["pca_components"], dtype=np.float64)
            pca_mean = np.array(artifacts["pca_mean"], dtype=np.float64)

        # Load profile specs for this router. Newer artifacts may embed the exact
        # runtime profile set directly; otherwise fall back to model defaults.
        if "profile_specs" in artifacts:
            profile_specs = artifacts["profile_specs"]
        else:
            profile_specs = get_profile_specs(model_key)

        return cls(
            model_key=model_key,
            class_names=artifacts["class_names"],
            profile_specs=profile_specs,
            weights=np.array(artifacts["weights"], dtype=np.float64),
            bias=np.array(artifacts["bias"], dtype=np.float64),
            pca_components=pca_components,
            pca_mean=pca_mean,
            standardization_mean=np.array(artifacts["standardization_mean"], dtype=np.float64),
            standardization_std=np.array(artifacts["standardization_std"], dtype=np.float64),
            feature_set=artifacts["feature_set"],
            intervention_mode=artifacts.get("intervention_mode", InterventionMode.ATTENTION_CONTRIBUTION.value),
        )

    def classify(self, baseline_pass_result: dict) -> dict[str, Any]:
        """Classify a prompt and return the routing decision.

        Args:
            baseline_pass_result: dict from Agent.run_pass(return_verbose=True)
                with g_attention_scales = [1.0, ...] (baseline).

        Returns:
            dict with:
                profile_name:  str — selected profile name, or "off"
                profile_spec:  dict — gain spec for g_profile.build_attention_scales_from_spec(),
                               or BASELINE_SPEC if "off"
                confidence:    float — softmax probability of the selected class
                class_probs:   dict — {class_name: probability} for all classes
                is_off:        bool — True if the router chose no intervention
        """
        # Build feature vector
        x = build_feature_vector(
            baseline_pass_result,
            pca_components=self.pca_components,
            pca_mean=self.pca_mean,
            standardization_mean=self.standardization_mean,
            standardization_std=self.standardization_std,
            feature_set=self.feature_set,
        )

        # Classify
        logits = x @ self.W + self.b
        probs = softmax(logits)

        predicted_idx = int(np.argmax(probs))
        predicted_name = self.class_names[predicted_idx]
        confidence = float(probs[predicted_idx])

        class_probs = {name: float(p) for name, p in zip(self.class_names, probs)}

        is_off = (predicted_name == "off")

        if is_off:
            profile_spec = BASELINE_SPEC
        else:
            profile_spec = self.profile_specs[predicted_name]

        return {
            "profile_name": predicted_name,
            "profile_spec": profile_spec,
            "confidence": confidence,
            "class_probs": class_probs,
            "is_off": is_off,
        }

    def __repr__(self) -> str:
        profiles = [n for n in self.class_names if n != "off"]
        return (
            f"InterventionRouter(model={self.model_key}, "
            f"profiles={profiles}, features={self.feature_set}, mode={self.intervention_mode.value})"
        )
