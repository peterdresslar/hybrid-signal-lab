"""
pipeline.py — Two-pass routed inference pipeline.

Orchestrates the full routing sequence:
  1. Baseline forward pass → sense features
  2. Classify → select profile or "off"
  3. Intervention forward pass (if not "off") → final result

This module operates on an existing Agent instance and
InterventionRouter, keeping them decoupled.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from model.g_profile import build_attention_scales_from_spec
from model.backend import InterventionMode
from router.router import InterventionRouter
from router.profiles import BASELINE_SPEC


def routed_pass(
    agent: "Agent",
    router: InterventionRouter,
    prompt: str,
    *,
    prompt_id: str | None = None,
    target_token_id: int | None = None,
    return_verbose: bool = False,
    intervention_mode: str | InterventionMode = InterventionMode.ATTENTION_CONTRIBUTION,
) -> dict[str, Any]:
    """Run the full two-pass routed inference pipeline.

    Args:
        agent: A loaded signal_lab Agent instance.
        router: A loaded InterventionRouter instance.
        prompt: The input prompt text.
        prompt_id: Optional identifier for the prompt.
        target_token_id: Optional target token for probability tracking.
        return_verbose: If True, include verbose outputs in both passes.
        intervention_mode: Intervention mode (default: attention_contribution).

    Returns:
        dict with:
            routing_decision: the classifier's output (profile_name, confidence, etc.)
            baseline_result:  result dict from the baseline pass
            intervention_result: result dict from the intervention pass (or None if "off")
            routed_result: the final result (intervention if applied, else baseline)
    """
    n_attn_layers = len(agent.get_attention_layer_indices())

    # --- Pass 1: Baseline ---
    baseline_scales = build_attention_scales_from_spec(
        BASELINE_SPEC,
        attention_slots=n_attn_layers,
    )

    baseline_result = agent.run_pass(
        prompt,
        baseline_scales,
        prompt_id=prompt_id,
        target_token_id=target_token_id,
        return_verbose=True,  # always need verbose for sensing
        return_raw_logits=True,
        intervention_mode=intervention_mode,
    )

    # --- Classify ---
    decision = router.classify(baseline_result)

    # --- Pass 2: Intervention (if not "off") ---
    intervention_result = None
    if not decision["is_off"]:
        intervention_scales = build_attention_scales_from_spec(
            decision["profile_spec"],
            attention_slots=n_attn_layers,
        )

        intervention_result = agent.run_pass(
            prompt,
            intervention_scales,
            prompt_id=prompt_id,
            target_token_id=target_token_id,
            baseline_logits=baseline_result.get("_raw_logits"),
            return_verbose=return_verbose,
            intervention_mode=intervention_mode,
        )

    # --- Assemble result ---
    routed_result = intervention_result if intervention_result is not None else baseline_result

    # Strip internal fields from baseline before returning
    baseline_clean = {k: v for k, v in baseline_result.items() if not k.startswith("_")}

    return {
        "routing_decision": decision,
        "baseline_result": baseline_clean,
        "intervention_result": intervention_result,
        "routed_result": routed_result,
    }


def routed_score_target(
    agent: "Agent",
    router: InterventionRouter,
    prompt: str,
    target_text: str,
    *,
    intervention_mode: str | InterventionMode = InterventionMode.ATTENTION_CONTRIBUTION,
) -> dict[str, Any]:
    """Run routed teacher-forced scoring over a multi-token target.

    Same two-pass structure as routed_pass, but uses Agent.score_target()
    for the intervention pass to get per-token probabilities.
    """
    n_attn_layers = len(agent.get_attention_layer_indices())

    # --- Pass 1: Baseline (verbose, for sensing) ---
    baseline_scales = build_attention_scales_from_spec(
        BASELINE_SPEC,
        attention_slots=n_attn_layers,
    )

    baseline_result = agent.run_pass(
        prompt,
        baseline_scales,
        return_verbose=True,
        intervention_mode=intervention_mode,
    )

    # --- Classify ---
    decision = router.classify(baseline_result)

    # --- Pass 2: Score target with intervention ---
    if decision["is_off"]:
        score_scales = baseline_scales
    else:
        score_scales = build_attention_scales_from_spec(
            decision["profile_spec"],
            attention_slots=n_attn_layers,
        )

    score_result = agent.score_target(
        prompt,
        target_text,
        score_scales,
        intervention_mode=intervention_mode,
    )

    return {
        "routing_decision": decision,
        "score_result": score_result,
    }
