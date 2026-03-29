from __future__ import annotations


ATTENTION_TARGETING_NATIVE = "native_attention_layers"
ATTENTION_TARGETING_ALL_LAYERS = "all_layers"
ATTENTION_TARGETING_EVERY_4TH = "every_4th_layer"

VALID_ATTENTION_TARGETING = {
    ATTENTION_TARGETING_NATIVE,
    ATTENTION_TARGETING_ALL_LAYERS,
    ATTENTION_TARGETING_EVERY_4TH,
}


def _constant(value: float, *, name: str | None = None) -> dict:
    return {
        "name": name or f"constant_{value:g}",
        "g_function": "constant",
        "g_params": {"value": value},
    }


def _control_points(values: list[float], *, name: str) -> dict:
    return {
        "name": name,
        "g_function": "control_points",
        "g_vector": values,
    }


def _clone_g_specs(g_specs: list[dict]) -> list[dict]:
    return [dict(spec) for spec in g_specs]


def _with_attention_targeting(
    base: dict,
    *,
    attention_targeting: str,
    description_suffix: str = "",
) -> dict:
    variant = dict(base)
    variant["g_specs"] = _clone_g_specs(base["g_specs"])
    variant["attention_targeting"] = attention_targeting
    if description_suffix:
        variant["description"] = f"{base['description']} {description_suffix}".strip()
    return variant


KITCHEN_SINK_G_SPECS = [
    # uniform scalars
    _constant(1.0, name="baseline"),
    _constant(0.25),
    _constant(0.5),
    _constant(0.75),
    _constant(1.25),
    _constant(1.5),
    _constant(2.0),
    # early boost / suppress
    _control_points([1.3, 1.3, 1.3, 1.0, 1.0, 1.0], name="early_boost_1.3"),
    _control_points([1.5, 1.5, 1.5, 1.0, 1.0, 1.0], name="early_boost_1.5"),
    _control_points([0.7, 0.7, 0.7, 1.0, 1.0, 1.0], name="early_suppress_0.7"),
    _control_points([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], name="early_suppress_0.5"),
    # late boost / suppress
    _control_points([1.0, 1.0, 1.0, 1.3, 1.3, 1.3], name="late_boost_1.3"),
    _control_points([1.0, 1.0, 1.0, 1.5, 1.5, 1.5], name="late_boost_1.5"),
    _control_points([1.0, 1.0, 1.0, 0.7, 0.7, 0.7], name="late_suppress_0.7"),
    _control_points([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], name="late_suppress_0.5"),
    # middle bump / suppress
    _control_points([1.0, 1.0, 1.5, 1.5, 1.0, 1.0], name="middle_bump_1.5"),
    _control_points([1.0, 1.0, 0.5, 0.5, 1.0, 1.0], name="middle_suppress_0.5"),
    _control_points([0.8, 1.0, 1.5, 1.5, 1.0, 0.8], name="middle_bump_1.5_edges_0.8"),
    # crossover
    _control_points([1.5, 1.5, 1.5, 0.5, 0.5, 0.5], name="early_high_late_low"),
    _control_points([0.5, 0.5, 0.5, 1.5, 1.5, 1.5], name="late_high_early_low"),
    # ramps
    _control_points([0.6, 0.8, 1.0, 1.2, 1.4, 1.6], name="ramp_up"),
    _control_points([1.6, 1.4, 1.2, 1.0, 0.8, 0.6], name="ramp_down"),
    _control_points([0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="ramp_up_gentle"),
    _control_points([1.3, 1.2, 1.1, 1.0, 0.9, 0.8], name="ramp_down_gentle"),
    # edges
    _control_points([1.4, 1.2, 1.0, 1.0, 1.2, 1.4], name="edges_high"),
    _control_points([0.6, 0.8, 1.0, 1.0, 0.8, 0.6], name="edges_low"),
    # extreme / stress
    _control_points([2.0, 2.0, 2.0, 0.0, 0.0, 0.0], name="early_only_2x"),
    _control_points([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], name="late_only_2x"),
    _control_points([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], name="middle_only"),
    _control_points([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], name="alternating"),
    _control_points([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], name="alternating_inv"),
]


FINE_GRAIN_KITCHEN_SINK_G_SPECS = [
    # baseline
    _constant(1.0, name="baseline"),
    # uniform scalars: 0.8 to 1.2 in 0.1 steps (excluding 1.0)
    _constant(0.8),
    _constant(0.9),
    _constant(1.1),
    _constant(1.2),
    # early boost / suppress (subtle)
    _control_points([1.1, 1.1, 1.1, 1.0, 1.0, 1.0], name="early_boost_1.1"),
    _control_points([1.2, 1.2, 1.2, 1.0, 1.0, 1.0], name="early_boost_1.2"),
    _control_points([0.9, 0.9, 0.9, 1.0, 1.0, 1.0], name="early_suppress_0.9"),
    _control_points([0.8, 0.8, 0.8, 1.0, 1.0, 1.0], name="early_suppress_0.8"),
    # late boost / suppress (subtle)
    _control_points([1.0, 1.0, 1.0, 1.1, 1.1, 1.1], name="late_boost_1.1"),
    _control_points([1.0, 1.0, 1.0, 1.2, 1.2, 1.2], name="late_boost_1.2"),
    _control_points([1.0, 1.0, 1.0, 0.9, 0.9, 0.9], name="late_suppress_0.9"),
    _control_points([1.0, 1.0, 1.0, 0.8, 0.8, 0.8], name="late_suppress_0.8"),
    # middle bump / suppress (subtle)
    _control_points([1.0, 1.0, 1.1, 1.1, 1.0, 1.0], name="middle_bump_1.1"),
    _control_points([1.0, 1.0, 1.2, 1.2, 1.0, 1.0], name="middle_bump_1.2"),
    _control_points([1.0, 1.0, 0.9, 0.9, 1.0, 1.0], name="middle_suppress_0.9"),
    _control_points([1.0, 1.0, 0.8, 0.8, 1.0, 1.0], name="middle_suppress_0.8"),
    # crossover (subtle)
    _control_points([1.2, 1.2, 1.2, 0.8, 0.8, 0.8], name="early_high_late_low_0.2"),
    _control_points([0.8, 0.8, 0.8, 1.2, 1.2, 1.2], name="late_high_early_low_0.2"),
    _control_points([1.1, 1.1, 1.1, 0.9, 0.9, 0.9], name="early_high_late_low_0.1"),
    _control_points([0.9, 0.9, 0.9, 1.1, 1.1, 1.1], name="late_high_early_low_0.1"),
    # ramps (fine)
    _control_points([0.8, 0.88, 0.96, 1.04, 1.12, 1.2], name="ramp_up_fine"),
    _control_points([1.2, 1.12, 1.04, 0.96, 0.88, 0.8], name="ramp_down_fine"),
    _control_points([0.9, 0.94, 0.98, 1.02, 1.06, 1.1], name="ramp_up_very_gentle"),
    _control_points([1.1, 1.06, 1.02, 0.98, 0.94, 0.9], name="ramp_down_very_gentle"),
    # edges (subtle)
    _control_points([1.2, 1.1, 1.0, 1.0, 1.1, 1.2], name="edges_high_subtle"),
    _control_points([0.8, 0.9, 1.0, 1.0, 0.9, 0.8], name="edges_low_subtle"),
]


CARTRIDGES = {
    "test_cartridge": {
        "description": "Test cartridge for checking baseline end-to-end behavior.",
        "g_specs": [
            _constant(1.0, name="baseline"),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt": "The color with the shortest wavelength is",
        "target": "violet",
        "model_key": "0_8B",
    },
    "test_cartridge_prompts": {
        "description": "Test cartridge for checking prompt tier loading.",
        "g_specs": [
            _constant(1.0, name="baseline"),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief"],
        "model_key": "0_8B",
    },
    "uniform_check_lite": {
        "description": "Uniform scalar values across depth using model-agnostic function specs.",
        "g_specs": [
            _constant(0.5),
            _constant(0.75),
            _constant(1.0, name="baseline"),
            _constant(1.25),
            _constant(1.5),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief"],
        "model_key": "0_8B",
    },
    "uniform_check": {
        "description": "Uniform scalar values across depth using model-agnostic function specs.",
        "g_specs": [
            _constant(0.5),
            _constant(0.75),
            _constant(1.0, name="baseline"),
            _constant(1.25),
            _constant(1.5),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief", "med", "long"],
        "model_key": "0_8B",
    },
    "uniform_type_smoke": {
        "description": "One representative prompt from each battery_3 prompt type across five balanced uniform gain levels.",
        "g_specs": [
            _constant(0.5),
            _constant(0.75),
            _constant(1.0, name="baseline"),
            _constant(1.25),
            _constant(1.5),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_battery": "battery/data/battery_3",
        "prompt_ids": [
            "sc_gen_alpha_seq_0000",        # structural_copying
            "rn_gen_0000",                  # reasoning_numerical
            "rt_gen_0000",                  # reasoning_tracking
            "alg_gen_arithmetic_0000",      # algorithmic
            "sp_gen_agreement_0000",        # syntactic_pattern
            "lr_gen_0000",                  # long_range_retrieval
            "dk_gen-wikipedia-random_0001", # domain_knowledge
            "cc_gen_0000",                  # code_comprehension
            "fr_counterfact_0000",          # factual_recall
            "fret_counterfact_0000",        # factual_retrieval
            "cm_lambada_0000",              # cultural_memorized
        ],
        "model_key": "0_8B",
    },
    "middle_bump_lite": {
        "description": "Middle-depth influence probe via normalized-depth control points.",
        "g_specs": [
            _constant(1.0, name="baseline"),
            _control_points([1.0, 1.0, 1.3, 1.3, 1.0, 1.0], name="middle_bump_1.3"),
            _control_points([1.0, 1.0, 1.5, 1.5, 1.0, 1.0], name="middle_bump_1.5"),
            _control_points([1.0, 1.0, 0.7, 0.7, 1.0, 1.0], name="middle_suppress_0.7"),
            _control_points([1.0, 1.0, 0.5, 0.5, 1.0, 1.0], name="middle_suppress_0.5"),
            _control_points([0.8, 1.0, 1.3, 1.3, 1.0, 0.8], name="middle_bump_1.3_edges_0.8"),
            _control_points([0.8, 1.0, 1.5, 1.5, 1.0, 0.8], name="middle_bump_1.5_edges_0.8"),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief"],
        "model_key": "0_8B",
    },
    "early_vs_late_lite": {
        "description": "Detailed early-vs-late probe with normalized-depth control-point profiles.",
        "g_specs": [
            _constant(1.0, name="baseline"),
            _control_points([1.1, 1.1, 1.1, 1.0, 1.0, 1.0], name="early_boost_1.1"),
            _control_points([1.3, 1.3, 1.3, 1.0, 1.0, 1.0], name="early_boost_1.3"),
            _control_points([1.5, 1.5, 1.5, 1.0, 1.0, 1.0], name="early_boost_1.5"),
            _control_points([1.0, 1.0, 1.0, 1.1, 1.1, 1.1], name="late_boost_1.1"),
            _control_points([1.0, 1.0, 1.0, 1.3, 1.3, 1.3], name="late_boost_1.3"),
            _control_points([1.0, 1.0, 1.0, 1.5, 1.5, 1.5], name="late_boost_1.5"),
            _control_points([0.9, 0.9, 0.9, 1.0, 1.0, 1.0], name="early_suppress_0.9"),
            _control_points([0.7, 0.7, 0.7, 1.0, 1.0, 1.0], name="early_suppress_0.7"),
            _control_points([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], name="early_suppress_0.5"),
            _control_points([1.0, 1.0, 1.0, 0.9, 0.9, 0.9], name="late_suppress_0.9"),
            _control_points([1.0, 1.0, 1.0, 0.7, 0.7, 0.7], name="late_suppress_0.7"),
            _control_points([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], name="late_suppress_0.5"),
            _control_points([1.3, 1.3, 1.3, 0.7, 0.7, 0.7], name="early_high_late_low_1"),
            _control_points([1.5, 1.5, 1.5, 0.5, 0.5, 0.5], name="early_high_late_low_2"),
            _control_points([1.2, 1.2, 1.2, 0.8, 0.8, 0.8], name="early_high_late_low_3"),
            _control_points([0.7, 0.7, 0.7, 1.3, 1.3, 1.3], name="late_high_early_low_1"),
            _control_points([0.5, 0.5, 0.5, 1.5, 1.5, 1.5], name="late_high_early_low_2"),
            _control_points([0.8, 0.8, 0.8, 1.2, 1.2, 1.2], name="late_high_early_low_3"),
            _control_points([0.6, 0.8, 1.0, 1.2, 1.4, 1.6], name="ramp_up"),
            _control_points([1.6, 1.4, 1.2, 1.0, 0.8, 0.6], name="ramp_down"),
            _control_points([0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="ramp_up_gentle"),
            _control_points([1.3, 1.2, 1.1, 1.0, 0.9, 0.8], name="ramp_down_gentle"),
            _control_points([1.4, 1.2, 1.0, 1.0, 1.2, 1.4], name="edges_high"),
            _control_points([0.6, 0.8, 1.0, 1.0, 0.8, 0.6], name="edges_low"),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief"],
        "model_key": "0_8B",
    },
    "early_vs_late": {
        "description": "Detailed early-vs-late probe with normalized-depth control-point profiles.",
        "g_specs": [
            _constant(1.0, name="baseline"),
            _control_points([1.1, 1.1, 1.1, 1.0, 1.0, 1.0], name="early_boost_1.1"),
            _control_points([1.3, 1.3, 1.3, 1.0, 1.0, 1.0], name="early_boost_1.3"),
            _control_points([1.5, 1.5, 1.5, 1.0, 1.0, 1.0], name="early_boost_1.5"),
            _control_points([1.0, 1.0, 1.0, 1.1, 1.1, 1.1], name="late_boost_1.1"),
            _control_points([1.0, 1.0, 1.0, 1.3, 1.3, 1.3], name="late_boost_1.3"),
            _control_points([1.0, 1.0, 1.0, 1.5, 1.5, 1.5], name="late_boost_1.5"),
            _control_points([0.9, 0.9, 0.9, 1.0, 1.0, 1.0], name="early_suppress_0.9"),
            _control_points([0.7, 0.7, 0.7, 1.0, 1.0, 1.0], name="early_suppress_0.7"),
            _control_points([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], name="early_suppress_0.5"),
            _control_points([1.0, 1.0, 1.0, 0.9, 0.9, 0.9], name="late_suppress_0.9"),
            _control_points([1.0, 1.0, 1.0, 0.7, 0.7, 0.7], name="late_suppress_0.7"),
            _control_points([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], name="late_suppress_0.5"),
            _control_points([1.3, 1.3, 1.3, 0.7, 0.7, 0.7], name="early_high_late_low_1"),
            _control_points([1.5, 1.5, 1.5, 0.5, 0.5, 0.5], name="early_high_late_low_2"),
            _control_points([1.2, 1.2, 1.2, 0.8, 0.8, 0.8], name="early_high_late_low_3"),
            _control_points([0.7, 0.7, 0.7, 1.3, 1.3, 1.3], name="late_high_early_low_1"),
            _control_points([0.5, 0.5, 0.5, 1.5, 1.5, 1.5], name="late_high_early_low_2"),
            _control_points([0.8, 0.8, 0.8, 1.2, 1.2, 1.2], name="late_high_early_low_3"),
            _control_points([0.6, 0.8, 1.0, 1.2, 1.4, 1.6], name="ramp_up"),
            _control_points([1.6, 1.4, 1.2, 1.0, 0.8, 0.6], name="ramp_down"),
            _control_points([0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="ramp_up_gentle"),
            _control_points([1.3, 1.2, 1.1, 1.0, 0.9, 0.8], name="ramp_down_gentle"),
            _control_points([1.4, 1.2, 1.0, 1.0, 1.2, 1.4], name="edges_high"),
            _control_points([0.6, 0.8, 1.0, 1.0, 0.8, 0.6], name="edges_low"),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief", "med", "long"],
        "model_key": "0_8B",
    },
    "kitchen_sink": {
        "description": "Comprehensive sweep: uniform scalars, early/late, middle, ramps, edges, and extreme profiles across all prompt tiers.",
        "g_specs": _clone_g_specs(KITCHEN_SINK_G_SPECS),
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief", "med", "long", "extended"],
        "model_key": "0_8B",
    },
    "fine_grain_kitchen_sink": {
        "description": "Fine-grained sweep at 0.1 steps in the 0.8–1.2 gain range. "
        "Uniform scalars plus shaped profiles within the narrow band where gentle interventions showed signal.",
        "g_specs": _clone_g_specs(FINE_GRAIN_KITCHEN_SINK_G_SPECS),
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief", "med", "long", "extended"],
        "model_key": "0_8B",
    },
    "kitchen_sink_all_layers": _with_attention_targeting(
        {
            "description": "Comprehensive sweep: uniform scalars, early/late, middle, ramps, edges, and extreme profiles across all prompt tiers.",
            "g_specs": KITCHEN_SINK_G_SPECS,
            "prompt_tiers": ["short", "brief", "med", "long", "extended"],
            "model_key": "0_8B",
        },
        attention_targeting=ATTENTION_TARGETING_ALL_LAYERS,
        description_suffix="Control variant targeting every attention layer.",
    ),
    "kitchen_sink_hybrid_mimic": _with_attention_targeting(
        {
            "description": "Comprehensive sweep: uniform scalars, early/late, middle, ramps, edges, and extreme profiles across all prompt tiers.",
            "g_specs": KITCHEN_SINK_G_SPECS,
            "prompt_tiers": ["short", "brief", "med", "long", "extended"],
            "model_key": "0_8B",
        },
        attention_targeting=ATTENTION_TARGETING_EVERY_4TH,
        description_suffix="Control variant targeting every 4th layer to mimic the hybrid cadence.",
    ),
    "fine_grain_kitchen_sink_all_layers": _with_attention_targeting(
        {
            "description": "Fine-grained sweep at 0.1 steps in the 0.8–1.2 gain range. "
            "Uniform scalars plus shaped profiles within the narrow band where gentle interventions showed signal.",
            "g_specs": FINE_GRAIN_KITCHEN_SINK_G_SPECS,
            "prompt_tiers": ["short", "brief", "med", "long", "extended"],
            "model_key": "0_8B",
        },
        attention_targeting=ATTENTION_TARGETING_ALL_LAYERS,
        description_suffix="Control variant targeting every attention layer.",
    ),
    "fine_grain_kitchen_sink_hybrid_mimic": _with_attention_targeting(
        {
            "description": "Fine-grained sweep at 0.1 steps in the 0.8–1.2 gain range. "
            "Uniform scalars plus shaped profiles within the narrow band where gentle interventions showed signal.",
            "g_specs": FINE_GRAIN_KITCHEN_SINK_G_SPECS,
            "prompt_tiers": ["short", "brief", "med", "long", "extended"],
            "model_key": "0_8B",
        },
        attention_targeting=ATTENTION_TARGETING_EVERY_4TH,
        description_suffix="Control variant targeting every 4th layer to mimic the hybrid cadence.",
    ),
}


def list_cartridges() -> list[str]:
    return sorted(CARTRIDGES.keys())


def get_cartridge(name: str) -> dict:
    if name not in CARTRIDGES:
        available = ", ".join(list_cartridges())
        raise ValueError(f"Unknown cartridge '{name}'. Available: {available}")
    cart = dict(CARTRIDGES[name])
    cart["name"] = name
    cart["g_specs"] = _clone_g_specs(cart["g_specs"])
    cart.setdefault("attention_targeting", ATTENTION_TARGETING_NATIVE)
    if cart["attention_targeting"] not in VALID_ATTENTION_TARGETING:
        valid = ", ".join(sorted(VALID_ATTENTION_TARGETING))
        raise ValueError(
            f"Cartridge '{name}' has invalid attention_targeting={cart['attention_targeting']!r}. "
            f"Expected one of: {valid}"
        )
    return cart
