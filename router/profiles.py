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
# 9B selected profiles (b4_021, attention_contribution mode)
# -----------------------------------------------------------------------

PROFILES_9B = {
    "edges_narrow": {
        "g_function": "control_points",
        "g_vector": [3.0, 1.0, 1.0, 1.0, 1.0, 3.0],
    },
    "late_boost_4.0": {
        "g_function": "control_points",
        "g_vector": [1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
    },
    "shifted_ramp_down": {
        "g_function": "control_points",
        "g_vector": [2.75, 2.5, 2.25, 2.0, 1.75, 1.5],
    },
    "tent_steep": {
        "g_function": "control_points",
        "g_vector": [1.0, 2.0, 4.0, 4.0, 2.0, 1.0],
    },
}

# -----------------------------------------------------------------------
# OLMO selected profiles (b4_021, attention_contribution mode)
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
