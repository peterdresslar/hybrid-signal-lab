"""
profiles.py — Gain profile specifications for the trained routers.

Each profile is stored as a control_points spec compatible with
model.g_profile.build_attention_scales_from_spec(). The specs are
extracted from sweep_cartridges.py — this module duplicates only the
profiles selected by the combinatorial search (select_profiles.py),
so the router package has no dependency on the sweep infrastructure.
"""

from __future__ import annotations

# -----------------------------------------------------------------------
# 9B selected profiles used in 030-bench (022-balanced-attention-hybrid,
# attention_contribution mode).
#
# Provenance note:
# A symmetric 9B re-selection was later run under the same discipline used for
# OLMO 030: `select_profiles.py --objective separable --max-constants 1`
# against `data/022-balanced-attention-hybrid`, with saved artifacts in
# `router/router-9B-030-reselect/`. The benchmarked set below ranked #14 by the
# separable objective but was retained because it remained extremely close to
# the top-ranked set (Δscore = 0.000329) and achieved slightly *higher* selected
# 4-profile oracle routed Δp (0.117589 vs. 0.116505 for the reselected top-1).
# This keeps the 030 benchmark provenance intact while documenting the matched
# selection protocol.
# -----------------------------------------------------------------------

PROFILES_9B = {
    "constant_2.6": {
        "g_function": "constant",
        "g_params": {"value": 2.6},
    },
    "edges_narrow_bal_0.55": {
        "g_function": "control_points",
        "g_vector": [1.55, 0.725, 0.725, 0.725, 0.725, 1.55],
    },
    "late_boost_bal_0.60": {
        "g_function": "control_points",
        "g_vector": [0.4, 0.4, 0.4, 1.6, 1.6, 1.6],
    },
    "triad_odd_bal_0.45": {
        "g_function": "control_points",
        "g_vector": [1.45, 0.55, 1.45, 0.55, 1.45, 0.55],
    },
}

# -----------------------------------------------------------------------
# Legacy OLMO attention-contribution profiles.
#
# These remain as the fallback/default table for historical artifacts that do
# not embed `profile_specs`, but the current 030 OLMO benchmark/router path uses
# self-contained block-output artifacts in `router/router-OLMO-030/`.
# -----------------------------------------------------------------------

PROFILES_OLMO = {
    "bookend_suppress": {
        "g_function": "control_points",
        "g_vector": [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    },
    "late_boost_4.0": {
        "g_function": "control_points",
        "g_vector": [1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
    },
    "late_high_early_low_3x": {
        "g_function": "control_points",
        "g_vector": [0.0, 0.0, 0.0, 2.5, 2.5, 2.5],
    },
    "triad_odd": {
        "g_function": "control_points",
        "g_vector": [3.0, 1.0, 3.0, 1.0, 3.0, 1.0],
    },
}

# Baseline (no intervention) — shared
BASELINE_SPEC = {
    "g_function": "constant",
    "g_params": {"value": 1.0},
}

# Lookup by model key
PROFILE_SETS = {
    "9B": PROFILES_9B,
    "OLMO": PROFILES_OLMO,
}


def get_profile_specs(model_key: str) -> dict[str, dict]:
    """Return the selected profile specs for a given model key."""
    key = model_key.upper().replace("QWEN", "9B")
    if key not in PROFILE_SETS:
        raise ValueError(f"Unknown model key '{model_key}'. Expected one of: {list(PROFILE_SETS.keys())}")
    return PROFILE_SETS[key]


def get_profile_names(model_key: str) -> list[str]:
    """Return the ordered list of profile names for a model."""
    return list(get_profile_specs(model_key).keys())
