"""
probe_router.py — Runtime value-probe router for `050` artifacts.

Unlike `InterventionRouter`, which predicts a discrete winner label, this router
predicts a continuous delta_p value for each profile and abstains to baseline
when the best predicted value does not exceed a threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from model.backend import InterventionMode, normalize_intervention_mode
from router.profiles import BASELINE_SPEC, get_profile_specs
from router.sensing import build_feature_vector


class ProbeRouter:
    def __init__(
        self,
        *,
        model_key: str,
        profiles: list[str],
        profile_specs: dict[str, dict],
        weights: np.ndarray,
        bias: np.ndarray,
        pca_components: np.ndarray | None,
        pca_mean: np.ndarray | None,
        standardization_mean: np.ndarray,
        standardization_std: np.ndarray,
        feature_set: str,
        intervention_mode: InterventionMode = InterventionMode.ATTENTION_CONTRIBUTION,
        decision_threshold: float | None = None,
        sequence_family: str | None = None,
        sequence_residualization: str = "raw",
        sequence_pca_components: np.ndarray | None = None,
        sequence_pca_mean: np.ndarray | None = None,
        sequence_residual_beta: np.ndarray | None = None,
        probe_type: str = "ridge",
        ridge_reg: float | None = None,
    ):
        self.model_key = model_key
        self.profiles = profiles
        self.profile_specs = profile_specs
        self.W = weights
        self.b = bias
        self.pca_components = pca_components
        self.pca_mean = pca_mean
        self.standardization_mean = standardization_mean
        self.standardization_std = standardization_std
        self.feature_set = feature_set
        self.intervention_mode = normalize_intervention_mode(intervention_mode)
        self.decision_threshold = decision_threshold
        self.sequence_family = sequence_family
        self.sequence_residualization = sequence_residualization
        self.sequence_pca_components = sequence_pca_components
        self.sequence_pca_mean = sequence_pca_mean
        self.sequence_residual_beta = sequence_residual_beta
        self.probe_type = probe_type
        self.ridge_reg = ridge_reg

    @classmethod
    def from_artifacts(cls, model_path: str | Path) -> "ProbeRouter":
        model_path = Path(model_path)
        with open(model_path) as f:
            artifacts = json.load(f)

        model_key = artifacts["model_key"]
        profiles = artifacts["profiles"]
        if "profile_specs" in artifacts:
            profile_specs = artifacts["profile_specs"]
        else:
            available = get_profile_specs(model_key)
            profile_specs = {name: available[name] for name in profiles}

        pca_components = None
        pca_mean = None
        if "pca_components" in artifacts:
            pca_components = np.array(artifacts["pca_components"], dtype=np.float64)
            pca_mean = np.array(artifacts["pca_mean"], dtype=np.float64)

        return cls(
            model_key=model_key,
            profiles=profiles,
            profile_specs=profile_specs,
            weights=np.array(artifacts["weights"], dtype=np.float64),
            bias=np.array(artifacts["bias"], dtype=np.float64),
            pca_components=pca_components,
            pca_mean=pca_mean,
            standardization_mean=np.array(artifacts["standardization_mean"], dtype=np.float64),
            standardization_std=np.array(artifacts["standardization_std"], dtype=np.float64),
            feature_set=artifacts["feature_set"],
            intervention_mode=artifacts.get("intervention_mode", InterventionMode.ATTENTION_CONTRIBUTION.value),
            decision_threshold=artifacts.get("decision_threshold"),
            sequence_family=artifacts.get("sequence_family"),
            sequence_residualization=artifacts.get("sequence_residualization", "raw"),
            sequence_pca_components=(
                np.array(artifacts["sequence_pca_components"], dtype=np.float64)
                if "sequence_pca_components" in artifacts
                else None
            ),
            sequence_pca_mean=(
                np.array(artifacts["sequence_pca_mean"], dtype=np.float64)
                if "sequence_pca_mean" in artifacts
                else None
            ),
            sequence_residual_beta=(
                np.array(artifacts["sequence_residual_beta"], dtype=np.float64)
                if "sequence_residual_beta" in artifacts
                else None
            ),
            probe_type=artifacts.get("probe_type", "ridge"),
            ridge_reg=artifacts.get("ridge_reg"),
        )

    @property
    def requires_hidden_states(self) -> bool:
        return "sequence_pca" in self.feature_set

    def classify(self, baseline_pass_result: dict) -> dict[str, Any]:
        x = build_feature_vector(
            baseline_pass_result,
            pca_components=self.pca_components,
            pca_mean=self.pca_mean,
            standardization_mean=self.standardization_mean,
            standardization_std=self.standardization_std,
            sequence_family=self.sequence_family,
            sequence_pca_components=self.sequence_pca_components,
            sequence_pca_mean=self.sequence_pca_mean,
            sequence_residualization=self.sequence_residualization,
            sequence_residual_beta=self.sequence_residual_beta,
            feature_set=self.feature_set,
        )

        predicted_values = x @ self.W + self.b
        best_idx = int(np.argmax(predicted_values))
        best_value = float(predicted_values[best_idx])
        predicted_name = self.profiles[best_idx]

        threshold = 0.0 if self.decision_threshold is None else float(self.decision_threshold)
        is_off = best_value <= threshold
        profile_spec = BASELINE_SPEC if is_off else self.profile_specs[predicted_name]

        return {
            "profile_name": "off" if is_off else predicted_name,
            "profile_spec": profile_spec,
            "confidence": best_value,
            "predicted_values": {name: float(v) for name, v in zip(self.profiles, predicted_values)},
            "is_off": is_off,
        }

    def __repr__(self) -> str:
        return (
            f"ProbeRouter(model={self.model_key}, profiles={self.profiles}, "
            f"features={self.feature_set}, mode={self.intervention_mode.value}, "
            f"threshold={self.decision_threshold}, probe_type={self.probe_type})"
        )
