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


def _balanced_control_points(
    deviations: list[float],
    *,
    amplitude: float,
    name: str,
) -> dict:
    if not deviations:
        raise ValueError("deviations must be non-empty.")
    mean_dev = sum(float(v) for v in deviations) / len(deviations)
    if abs(mean_dev) > 1e-9:
        raise ValueError(
            f"Balanced profile template '{name}' must have mean 0.0; got {mean_dev:.6f}."
        )
    values = [1.0 + amplitude * float(v) for v in deviations]
    if min(values) < 0.0:
        raise ValueError(
            f"Balanced profile '{name}' produced negative gain values: {values}."
        )
    return _control_points(values, name=name)


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


BALANCED_SWEEP_G_SPECS = [
    # ===== CONSTANTS: keep pure dose-response coverage =====
    _constant(1.0, name="baseline"),
    _constant(0.4),
    _constant(0.55),
    _constant(0.7),
    _constant(0.85),
    _constant(0.95),
    _constant(1.05),
    _constant(1.15),
    _constant(1.3),
    _constant(1.45),
    _constant(1.6),
    _constant(1.8),
    _constant(2.0),
    _constant(2.3),
    _constant(2.6),
    _constant(3.0),
    # ===== EARLY / LATE CONTRAST (mean-preserving) =====
    _balanced_control_points([1, 1, 1, -1, -1, -1], amplitude=0.15, name="early_boost_bal_0.15"),
    _balanced_control_points([1, 1, 1, -1, -1, -1], amplitude=0.3, name="early_boost_bal_0.30"),
    _balanced_control_points([1, 1, 1, -1, -1, -1], amplitude=0.45, name="early_boost_bal_0.45"),
    _balanced_control_points([1, 1, 1, -1, -1, -1], amplitude=0.6, name="early_boost_bal_0.60"),
    _balanced_control_points([-1, -1, -1, 1, 1, 1], amplitude=0.15, name="late_boost_bal_0.15"),
    _balanced_control_points([-1, -1, -1, 1, 1, 1], amplitude=0.3, name="late_boost_bal_0.30"),
    _balanced_control_points([-1, -1, -1, 1, 1, 1], amplitude=0.45, name="late_boost_bal_0.45"),
    _balanced_control_points([-1, -1, -1, 1, 1, 1], amplitude=0.6, name="late_boost_bal_0.60"),
    # ===== MIDDLE BUMP / VALLEY (mean-preserving) =====
    _balanced_control_points([-0.5, -0.5, 1, 1, -0.5, -0.5], amplitude=0.2, name="middle_bump_bal_0.20"),
    _balanced_control_points([-0.5, -0.5, 1, 1, -0.5, -0.5], amplitude=0.35, name="middle_bump_bal_0.35"),
    _balanced_control_points([-0.5, -0.5, 1, 1, -0.5, -0.5], amplitude=0.5, name="middle_bump_bal_0.50"),
    _balanced_control_points([-0.5, -0.5, 1, 1, -0.5, -0.5], amplitude=0.65, name="middle_bump_bal_0.65"),
    _balanced_control_points([0.5, 0.5, -1, -1, 0.5, 0.5], amplitude=0.2, name="middle_valley_bal_0.20"),
    _balanced_control_points([0.5, 0.5, -1, -1, 0.5, 0.5], amplitude=0.35, name="middle_valley_bal_0.35"),
    _balanced_control_points([0.5, 0.5, -1, -1, 0.5, 0.5], amplitude=0.5, name="middle_valley_bal_0.50"),
    _balanced_control_points([0.5, 0.5, -1, -1, 0.5, 0.5], amplitude=0.65, name="middle_valley_bal_0.65"),
    # ===== EDGES =====
    _balanced_control_points([0.8, 0.2, -1, -1, 0.2, 0.8], amplitude=0.25, name="edges_bal_0.25"),
    _balanced_control_points([0.8, 0.2, -1, -1, 0.2, 0.8], amplitude=0.4, name="edges_bal_0.40"),
    _balanced_control_points([0.8, 0.2, -1, -1, 0.2, 0.8], amplitude=0.55, name="edges_bal_0.55"),
    _balanced_control_points([0.8, 0.2, -1, -1, 0.2, 0.8], amplitude=0.7, name="edges_bal_0.70"),
    _balanced_control_points([1, -0.5, -0.5, -0.5, -0.5, 1], amplitude=0.25, name="edges_narrow_bal_0.25"),
    _balanced_control_points([1, -0.5, -0.5, -0.5, -0.5, 1], amplitude=0.4, name="edges_narrow_bal_0.40"),
    _balanced_control_points([1, -0.5, -0.5, -0.5, -0.5, 1], amplitude=0.55, name="edges_narrow_bal_0.55"),
    _balanced_control_points([1, -0.5, -0.5, -0.5, -0.5, 1], amplitude=0.7, name="edges_narrow_bal_0.70"),
    _balanced_control_points([1, 0.5, -0.5, -0.5, -0.25, -0.25], amplitude=0.25, name="edges_asym_early_bal_0.25"),
    _balanced_control_points([1, 0.5, -0.5, -0.5, -0.25, -0.25], amplitude=0.45, name="edges_asym_early_bal_0.45"),
    _balanced_control_points([-0.25, -0.25, -0.5, -0.5, 0.5, 1], amplitude=0.25, name="edges_asym_late_bal_0.25"),
    _balanced_control_points([-0.25, -0.25, -0.5, -0.5, 0.5, 1], amplitude=0.45, name="edges_asym_late_bal_0.45"),
    # ===== RAMPS =====
    _balanced_control_points([-1, -0.6, -0.2, 0.2, 0.6, 1], amplitude=0.2, name="ramp_up_bal_0.20"),
    _balanced_control_points([-1, -0.6, -0.2, 0.2, 0.6, 1], amplitude=0.35, name="ramp_up_bal_0.35"),
    _balanced_control_points([-1, -0.6, -0.2, 0.2, 0.6, 1], amplitude=0.5, name="ramp_up_bal_0.50"),
    _balanced_control_points([1, 0.6, 0.2, -0.2, -0.6, -1], amplitude=0.2, name="ramp_down_bal_0.20"),
    _balanced_control_points([1, 0.6, 0.2, -0.2, -0.6, -1], amplitude=0.35, name="ramp_down_bal_0.35"),
    _balanced_control_points([1, 0.6, 0.2, -0.2, -0.6, -1], amplitude=0.5, name="ramp_down_bal_0.50"),
    # ===== BOWL / TENT / PLATEAU =====
    _balanced_control_points([1, 0, -1, -1, 0, 1], amplitude=0.25, name="bowl_bal_0.25"),
    _balanced_control_points([1, 0, -1, -1, 0, 1], amplitude=0.4, name="bowl_bal_0.40"),
    _balanced_control_points([1, 0, -1, -1, 0, 1], amplitude=0.55, name="bowl_bal_0.55"),
    _balanced_control_points([-1, 0, 1, 1, 0, -1], amplitude=0.25, name="tent_bal_0.25"),
    _balanced_control_points([-1, 0, 1, 1, 0, -1], amplitude=0.4, name="tent_bal_0.40"),
    _balanced_control_points([-1, 0, 1, 1, 0, -1], amplitude=0.55, name="tent_bal_0.55"),
    _balanced_control_points([-1, 0.5, 0.5, 0.5, 0.5, -1], amplitude=0.25, name="plateau_bal_0.25"),
    _balanced_control_points([-1, 0.5, 0.5, 0.5, 0.5, -1], amplitude=0.4, name="plateau_bal_0.40"),
    _balanced_control_points([-1, 0.5, 0.5, 0.5, 0.5, -1], amplitude=0.55, name="plateau_bal_0.55"),
    # ===== SPIKES =====
    _balanced_control_points([5, -1, -1, -1, -1, -1], amplitude=0.12, name="spike_p1_bal_0.12"),
    _balanced_control_points([5, -1, -1, -1, -1, -1], amplitude=0.18, name="spike_p1_bal_0.18"),
    _balanced_control_points([-1, -1, 5, -1, -1, -1], amplitude=0.12, name="spike_p3_bal_0.12"),
    _balanced_control_points([-1, -1, 5, -1, -1, -1], amplitude=0.18, name="spike_p3_bal_0.18"),
    _balanced_control_points([-1, -1, -1, -1, -1, 5], amplitude=0.12, name="spike_p6_bal_0.12"),
    _balanced_control_points([-1, -1, -1, -1, -1, 5], amplitude=0.18, name="spike_p6_bal_0.18"),
    # ===== ALTERNATING / PERIODIC =====
    _balanced_control_points([1, -1, 1, -1, 1, -1], amplitude=0.25, name="triad_odd_bal_0.25"),
    _balanced_control_points([1, -1, 1, -1, 1, -1], amplitude=0.45, name="triad_odd_bal_0.45"),
    _balanced_control_points([-1, 1, -1, 1, -1, 1], amplitude=0.25, name="triad_even_bal_0.25"),
    _balanced_control_points([-1, 1, -1, 1, -1, 1], amplitude=0.45, name="triad_even_bal_0.45"),
    _balanced_control_points([1, -0.5, -0.5, 1, -0.5, -0.5], amplitude=0.25, name="pair_stride_bal_0.25"),
    _balanced_control_points([1, -0.5, -0.5, 1, -0.5, -0.5], amplitude=0.45, name="pair_stride_bal_0.45"),
    _balanced_control_points([-0.5, 1, 1, -0.5, -0.5, -0.5], amplitude=0.25, name="quarter_wave_bal_0.25"),
    _balanced_control_points([-0.5, 1, 1, -0.5, -0.5, -0.5], amplitude=0.45, name="quarter_wave_bal_0.45"),
    _balanced_control_points([1, -1, 0.5, -0.5, 0.5, -0.5], amplitude=0.25, name="sawtooth_bal_0.25"),
    _balanced_control_points([1, -1, 0.5, -0.5, 0.5, -0.5], amplitude=0.45, name="sawtooth_bal_0.45"),
    _balanced_control_points([-1, 1, -0.5, 0.5, -0.5, 0.5], amplitude=0.25, name="sawtooth_inv_bal_0.25"),
    _balanced_control_points([-1, 1, -0.5, 0.5, -0.5, 0.5], amplitude=0.45, name="sawtooth_inv_bal_0.45"),
    # ===== THREE-ZONE / ASYMMETRIC REGION PROBES =====
    _balanced_control_points([0.5, 0.5, 0.5, 0.5, -1, -1], amplitude=0.3, name="early_mid_high_bal_0.30"),
    _balanced_control_points([0.5, 0.5, 0.5, 0.5, -1, -1], amplitude=0.5, name="early_mid_high_bal_0.50"),
    _balanced_control_points([-1, -1, 0.5, 0.5, 0.5, 0.5], amplitude=0.3, name="mid_late_high_bal_0.30"),
    _balanced_control_points([-1, -1, 0.5, 0.5, 0.5, 0.5], amplitude=0.5, name="mid_late_high_bal_0.50"),
    _balanced_control_points([0.5, 0.5, -1, -1, 0.5, 0.5], amplitude=0.3, name="early_late_high_bal_0.30"),
    _balanced_control_points([0.5, 0.5, -1, -1, 0.5, 0.5], amplitude=0.5, name="early_late_high_bal_0.50"),
    _balanced_control_points([0, 1, 1, 0, -1, -1], amplitude=0.3, name="early_mid_peak_bal_0.30"),
    _balanced_control_points([0, 1, 1, 0, -1, -1], amplitude=0.5, name="early_mid_peak_bal_0.50"),
    _balanced_control_points([-1, -1, 0, 1, 1, 0], amplitude=0.3, name="mid_late_peak_bal_0.30"),
    _balanced_control_points([-1, -1, 0, 1, 1, 0], amplitude=0.5, name="mid_late_peak_bal_0.50"),
    _balanced_control_points([1, 1, -0.5, -0.5, -0.5, -0.5], amplitude=0.4, name="bookend_high_bal_0.40"),
]


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
    "attention_pilot": {
        "description": "Wide-range shape-family survey for attention_contribution mode. "
        "3x deviation from baseline, 2x profile density vs kitchen_sink. "
        "Use with --intervention-strategy attention_contribution.",
        "g_specs": [
            # --- uniform scalars: original + 3x-wide ---
            _constant(1.0, name="baseline"),
            # original constants
            _constant(0.25),
            _constant(0.5),
            _constant(0.75),
            _constant(1.25),
            _constant(1.5),
            _constant(2.0),
            # 3x-wide constants (new)
            _constant(0.0, name="constant_0.0"),
            _constant(3.0),
            _constant(4.0),
            # intermediate density (new)
            _constant(1.75),
            _constant(2.5),
            # --- early boost / suppress: original + 3x ---
            _control_points([1.3, 1.3, 1.3, 1.0, 1.0, 1.0], name="early_boost_1.3"),
            _control_points([1.5, 1.5, 1.5, 1.0, 1.0, 1.0], name="early_boost_1.5"),
            _control_points([1.9, 1.9, 1.9, 1.0, 1.0, 1.0], name="early_boost_1.9"),
            _control_points([2.5, 2.5, 2.5, 1.0, 1.0, 1.0], name="early_boost_2.5"),
            _control_points([0.7, 0.7, 0.7, 1.0, 1.0, 1.0], name="early_suppress_0.7"),
            _control_points([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], name="early_suppress_0.5"),
            _control_points([0.1, 0.1, 0.1, 1.0, 1.0, 1.0], name="early_suppress_0.1"),
            _control_points([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], name="early_suppress_0.0"),
            # --- late boost / suppress: original + 3x ---
            _control_points([1.0, 1.0, 1.0, 1.3, 1.3, 1.3], name="late_boost_1.3"),
            _control_points([1.0, 1.0, 1.0, 1.5, 1.5, 1.5], name="late_boost_1.5"),
            _control_points([1.0, 1.0, 1.0, 1.9, 1.9, 1.9], name="late_boost_1.9"),
            _control_points([1.0, 1.0, 1.0, 2.5, 2.5, 2.5], name="late_boost_2.5"),
            _control_points([1.0, 1.0, 1.0, 0.7, 0.7, 0.7], name="late_suppress_0.7"),
            _control_points([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], name="late_suppress_0.5"),
            _control_points([1.0, 1.0, 1.0, 0.1, 0.1, 0.1], name="late_suppress_0.1"),
            _control_points([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], name="late_suppress_0.0"),
            # --- middle bump / suppress: original + 3x ---
            _control_points([1.0, 1.0, 1.5, 1.5, 1.0, 1.0], name="middle_bump_1.5"),
            _control_points([1.0, 1.0, 2.5, 2.5, 1.0, 1.0], name="middle_bump_2.5"),
            _control_points([1.0, 1.0, 0.5, 0.5, 1.0, 1.0], name="middle_suppress_0.5"),
            _control_points([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], name="middle_suppress_0.0"),
            _control_points([0.8, 1.0, 1.5, 1.5, 1.0, 0.8], name="middle_bump_1.5_edges_0.8"),
            _control_points([0.4, 1.0, 2.5, 2.5, 1.0, 0.4], name="middle_bump_2.5_edges_0.4"),
            # --- crossover: original + 3x ---
            _control_points([1.5, 1.5, 1.5, 0.5, 0.5, 0.5], name="early_high_late_low"),
            _control_points([2.5, 2.5, 2.5, 0.0, 0.0, 0.0], name="early_high_late_low_3x"),
            _control_points([0.5, 0.5, 0.5, 1.5, 1.5, 1.5], name="late_high_early_low"),
            _control_points([0.0, 0.0, 0.0, 2.5, 2.5, 2.5], name="late_high_early_low_3x"),
            # --- ramps: original + 3x ---
            _control_points([0.6, 0.8, 1.0, 1.2, 1.4, 1.6], name="ramp_up"),
            _control_points([0.0, 0.4, 1.0, 1.6, 2.2, 2.8], name="ramp_up_3x"),
            _control_points([1.6, 1.4, 1.2, 1.0, 0.8, 0.6], name="ramp_down"),
            _control_points([2.8, 2.2, 1.6, 1.0, 0.4, 0.0], name="ramp_down_3x"),
            _control_points([0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="ramp_up_gentle"),
            _control_points([0.4, 0.7, 1.0, 1.3, 1.6, 1.9], name="ramp_up_wide"),
            _control_points([1.3, 1.2, 1.1, 1.0, 0.9, 0.8], name="ramp_down_gentle"),
            _control_points([1.9, 1.6, 1.3, 1.0, 0.7, 0.4], name="ramp_down_wide"),
            # --- edges: original + 3x ---
            _control_points([1.4, 1.2, 1.0, 1.0, 1.2, 1.4], name="edges_high"),
            _control_points([2.2, 1.6, 1.0, 1.0, 1.6, 2.2], name="edges_high_3x"),
            _control_points([0.6, 0.8, 1.0, 1.0, 0.8, 0.6], name="edges_low"),
            _control_points([0.0, 0.4, 1.0, 1.0, 0.4, 0.0], name="edges_low_3x"),
            # --- extreme / stress: original + 3x ---
            _control_points([2.0, 2.0, 2.0, 0.0, 0.0, 0.0], name="early_only_2x"),
            _control_points([4.0, 4.0, 4.0, 0.0, 0.0, 0.0], name="early_only_4x"),
            _control_points([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], name="late_only_2x"),
            _control_points([0.0, 0.0, 0.0, 4.0, 4.0, 4.0], name="late_only_4x"),
            _control_points([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], name="middle_only"),
            _control_points([0.0, 0.0, 3.0, 3.0, 0.0, 0.0], name="middle_only_3x"),
            _control_points([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], name="alternating"),
            _control_points([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], name="alternating_inv"),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief", "med", "long", "extended"],
        "model_key": "0_8B",
    },
       "attention_pilot_lite": {
        "description": "Wide-range shape-family survey for attention_contribution mode. "
        "3x deviation from baseline, 2x profile density vs kitchen_sink. "
        "Capped to roughly 5 prompts per prompt type for pilot-sized runs. "
        "Use with --intervention-strategy attention_contribution.",
        "g_specs": [
            # --- uniform scalars: original + 3x-wide ---
            _constant(1.0, name="baseline"),
            # original constants
            _constant(0.25),
            _constant(0.5),
            _constant(0.75),
            _constant(1.25),
            _constant(1.5),
            _constant(2.0),
            # 3x-wide constants (new)
            _constant(0.0, name="constant_0.0"),
            _constant(3.0),
            _constant(4.0),
            # intermediate density (new)
            _constant(1.75),
            _constant(2.5),
            # --- early boost / suppress: original + 3x ---
            _control_points([1.3, 1.3, 1.3, 1.0, 1.0, 1.0], name="early_boost_1.3"),
            _control_points([1.5, 1.5, 1.5, 1.0, 1.0, 1.0], name="early_boost_1.5"),
            _control_points([1.9, 1.9, 1.9, 1.0, 1.0, 1.0], name="early_boost_1.9"),
            _control_points([2.5, 2.5, 2.5, 1.0, 1.0, 1.0], name="early_boost_2.5"),
            _control_points([0.7, 0.7, 0.7, 1.0, 1.0, 1.0], name="early_suppress_0.7"),
            _control_points([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], name="early_suppress_0.5"),
            _control_points([0.1, 0.1, 0.1, 1.0, 1.0, 1.0], name="early_suppress_0.1"),
            _control_points([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], name="early_suppress_0.0"),
            # --- late boost / suppress: original + 3x ---
            _control_points([1.0, 1.0, 1.0, 1.3, 1.3, 1.3], name="late_boost_1.3"),
            _control_points([1.0, 1.0, 1.0, 1.5, 1.5, 1.5], name="late_boost_1.5"),
            _control_points([1.0, 1.0, 1.0, 1.9, 1.9, 1.9], name="late_boost_1.9"),
            _control_points([1.0, 1.0, 1.0, 2.5, 2.5, 2.5], name="late_boost_2.5"),
            _control_points([1.0, 1.0, 1.0, 0.7, 0.7, 0.7], name="late_suppress_0.7"),
            _control_points([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], name="late_suppress_0.5"),
            _control_points([1.0, 1.0, 1.0, 0.1, 0.1, 0.1], name="late_suppress_0.1"),
            _control_points([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], name="late_suppress_0.0"),
            # --- middle bump / suppress: original + 3x ---
            _control_points([1.0, 1.0, 1.5, 1.5, 1.0, 1.0], name="middle_bump_1.5"),
            _control_points([1.0, 1.0, 2.5, 2.5, 1.0, 1.0], name="middle_bump_2.5"),
            _control_points([1.0, 1.0, 0.5, 0.5, 1.0, 1.0], name="middle_suppress_0.5"),
            _control_points([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], name="middle_suppress_0.0"),
            _control_points([0.8, 1.0, 1.5, 1.5, 1.0, 0.8], name="middle_bump_1.5_edges_0.8"),
            _control_points([0.4, 1.0, 2.5, 2.5, 1.0, 0.4], name="middle_bump_2.5_edges_0.4"),
            # --- crossover: original + 3x ---
            _control_points([1.5, 1.5, 1.5, 0.5, 0.5, 0.5], name="early_high_late_low"),
            _control_points([2.5, 2.5, 2.5, 0.0, 0.0, 0.0], name="early_high_late_low_3x"),
            _control_points([0.5, 0.5, 0.5, 1.5, 1.5, 1.5], name="late_high_early_low"),
            _control_points([0.0, 0.0, 0.0, 2.5, 2.5, 2.5], name="late_high_early_low_3x"),
            # --- ramps: original + 3x ---
            _control_points([0.6, 0.8, 1.0, 1.2, 1.4, 1.6], name="ramp_up"),
            _control_points([0.0, 0.4, 1.0, 1.6, 2.2, 2.8], name="ramp_up_3x"),
            _control_points([1.6, 1.4, 1.2, 1.0, 0.8, 0.6], name="ramp_down"),
            _control_points([2.8, 2.2, 1.6, 1.0, 0.4, 0.0], name="ramp_down_3x"),
            _control_points([0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="ramp_up_gentle"),
            _control_points([0.4, 0.7, 1.0, 1.3, 1.6, 1.9], name="ramp_up_wide"),
            _control_points([1.3, 1.2, 1.1, 1.0, 0.9, 0.8], name="ramp_down_gentle"),
            _control_points([1.9, 1.6, 1.3, 1.0, 0.7, 0.4], name="ramp_down_wide"),
            # --- edges: original + 3x ---
            _control_points([1.4, 1.2, 1.0, 1.0, 1.2, 1.4], name="edges_high"),
            _control_points([2.2, 1.6, 1.0, 1.0, 1.6, 2.2], name="edges_high_3x"),
            _control_points([0.6, 0.8, 1.0, 1.0, 0.8, 0.6], name="edges_low"),
            _control_points([0.0, 0.4, 1.0, 1.0, 0.4, 0.0], name="edges_low_3x"),
            # --- extreme / stress: original + 3x ---
            _control_points([2.0, 2.0, 2.0, 0.0, 0.0, 0.0], name="early_only_2x"),
            _control_points([4.0, 4.0, 4.0, 0.0, 0.0, 0.0], name="early_only_4x"),
            _control_points([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], name="late_only_2x"),
            _control_points([0.0, 0.0, 0.0, 4.0, 4.0, 4.0], name="late_only_4x"),
            _control_points([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], name="middle_only"),
            _control_points([0.0, 0.0, 3.0, 3.0, 0.0, 0.0], name="middle_only_3x"),
            _control_points([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], name="alternating"),
            _control_points([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], name="alternating_inv"),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief", "med", "long", "extended"],
        "max_prompts_per_type": 5,
        "model_key": "0_8B",
    },
    "pilot_wide": {
        "description": "Extended profile search for attention_contribution mode. "
        "All 58 attention_pilot profiles plus ~58 new profiles exploring "
        "deeper edges family, single-position spikes/valleys, nonlinear ramps, "
        "bowl/tent shapes, shifted-up patterns, and high-amplitude stress tests. "
        "Use with --intervention-strategy attention_contribution.",
        "g_specs": [
            # ===== ORIGINAL attention_pilot profiles (58) =====
            # --- uniform scalars: original + 3x-wide ---
            _constant(1.0, name="baseline"),
            _constant(0.25),
            _constant(0.5),
            _constant(0.75),
            _constant(1.25),
            _constant(1.5),
            _constant(2.0),
            _constant(0.0, name="constant_0.0"),
            _constant(3.0),
            _constant(4.0),
            _constant(1.75),
            _constant(2.5),
            # --- early boost / suppress: original + 3x ---
            _control_points([1.3, 1.3, 1.3, 1.0, 1.0, 1.0], name="early_boost_1.3"),
            _control_points([1.5, 1.5, 1.5, 1.0, 1.0, 1.0], name="early_boost_1.5"),
            _control_points([1.9, 1.9, 1.9, 1.0, 1.0, 1.0], name="early_boost_1.9"),
            _control_points([2.5, 2.5, 2.5, 1.0, 1.0, 1.0], name="early_boost_2.5"),
            _control_points([0.7, 0.7, 0.7, 1.0, 1.0, 1.0], name="early_suppress_0.7"),
            _control_points([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], name="early_suppress_0.5"),
            _control_points([0.1, 0.1, 0.1, 1.0, 1.0, 1.0], name="early_suppress_0.1"),
            _control_points([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], name="early_suppress_0.0"),
            # --- late boost / suppress: original + 3x ---
            _control_points([1.0, 1.0, 1.0, 1.3, 1.3, 1.3], name="late_boost_1.3"),
            _control_points([1.0, 1.0, 1.0, 1.5, 1.5, 1.5], name="late_boost_1.5"),
            _control_points([1.0, 1.0, 1.0, 1.9, 1.9, 1.9], name="late_boost_1.9"),
            _control_points([1.0, 1.0, 1.0, 2.5, 2.5, 2.5], name="late_boost_2.5"),
            _control_points([1.0, 1.0, 1.0, 0.7, 0.7, 0.7], name="late_suppress_0.7"),
            _control_points([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], name="late_suppress_0.5"),
            _control_points([1.0, 1.0, 1.0, 0.1, 0.1, 0.1], name="late_suppress_0.1"),
            _control_points([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], name="late_suppress_0.0"),
            # --- middle bump / suppress: original + 3x ---
            _control_points([1.0, 1.0, 1.5, 1.5, 1.0, 1.0], name="middle_bump_1.5"),
            _control_points([1.0, 1.0, 2.5, 2.5, 1.0, 1.0], name="middle_bump_2.5"),
            _control_points([1.0, 1.0, 0.5, 0.5, 1.0, 1.0], name="middle_suppress_0.5"),
            _control_points([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], name="middle_suppress_0.0"),
            _control_points([0.8, 1.0, 1.5, 1.5, 1.0, 0.8], name="middle_bump_1.5_edges_0.8"),
            _control_points([0.4, 1.0, 2.5, 2.5, 1.0, 0.4], name="middle_bump_2.5_edges_0.4"),
            # --- crossover: original + 3x ---
            _control_points([1.5, 1.5, 1.5, 0.5, 0.5, 0.5], name="early_high_late_low"),
            _control_points([2.5, 2.5, 2.5, 0.0, 0.0, 0.0], name="early_high_late_low_3x"),
            _control_points([0.5, 0.5, 0.5, 1.5, 1.5, 1.5], name="late_high_early_low"),
            _control_points([0.0, 0.0, 0.0, 2.5, 2.5, 2.5], name="late_high_early_low_3x"),
            # --- ramps: original + 3x ---
            _control_points([0.6, 0.8, 1.0, 1.2, 1.4, 1.6], name="ramp_up"),
            _control_points([0.0, 0.4, 1.0, 1.6, 2.2, 2.8], name="ramp_up_3x"),
            _control_points([1.6, 1.4, 1.2, 1.0, 0.8, 0.6], name="ramp_down"),
            _control_points([2.8, 2.2, 1.6, 1.0, 0.4, 0.0], name="ramp_down_3x"),
            _control_points([0.8, 0.9, 1.0, 1.1, 1.2, 1.3], name="ramp_up_gentle"),
            _control_points([0.4, 0.7, 1.0, 1.3, 1.6, 1.9], name="ramp_up_wide"),
            _control_points([1.3, 1.2, 1.1, 1.0, 0.9, 0.8], name="ramp_down_gentle"),
            _control_points([1.9, 1.6, 1.3, 1.0, 0.7, 0.4], name="ramp_down_wide"),
            # --- edges: original + 3x ---
            _control_points([1.4, 1.2, 1.0, 1.0, 1.2, 1.4], name="edges_high"),
            _control_points([2.2, 1.6, 1.0, 1.0, 1.6, 2.2], name="edges_high_3x"),
            _control_points([0.6, 0.8, 1.0, 1.0, 0.8, 0.6], name="edges_low"),
            _control_points([0.0, 0.4, 1.0, 1.0, 0.4, 0.0], name="edges_low_3x"),
            # --- extreme / stress: original + 3x ---
            _control_points([2.0, 2.0, 2.0, 0.0, 0.0, 0.0], name="early_only_2x"),
            _control_points([4.0, 4.0, 4.0, 0.0, 0.0, 0.0], name="early_only_4x"),
            _control_points([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], name="late_only_2x"),
            _control_points([0.0, 0.0, 0.0, 4.0, 4.0, 4.0], name="late_only_4x"),
            _control_points([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], name="middle_only"),
            _control_points([0.0, 0.0, 3.0, 3.0, 0.0, 0.0], name="middle_only_3x"),
            _control_points([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], name="alternating"),
            _control_points([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], name="alternating_inv"),
            # ===== NEW profiles (58) =====
            # --- constant density fills ---
            _constant(1.1),
            _constant(1.35),
            _constant(1.65),
            _constant(1.85),
            _constant(2.25),
            _constant(2.75),
            # --- high-amplitude constants ---
            _constant(5.0),
            _constant(6.0),
            _constant(8.0),
            # --- intermediate shaped fills ---
            _control_points([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], name="early_boost_2.0"),
            _control_points([3.0, 3.0, 3.0, 1.0, 1.0, 1.0], name="early_boost_3.0"),
            _control_points([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], name="late_boost_2.0"),
            _control_points([1.0, 1.0, 1.0, 3.0, 3.0, 3.0], name="late_boost_3.0"),
            _control_points([0.3, 0.3, 0.3, 1.0, 1.0, 1.0], name="early_suppress_0.3"),
            _control_points([1.0, 1.0, 1.0, 0.3, 0.3, 0.3], name="late_suppress_0.3"),
            _control_points([1.0, 1.0, 2.0, 2.0, 1.0, 1.0], name="middle_bump_2.0"),
            _control_points([1.0, 1.0, 3.5, 3.5, 1.0, 1.0], name="middle_bump_3.5"),
            # --- single-position spikes at 3x ---
            _control_points([3.0, 1.0, 1.0, 1.0, 1.0, 1.0], name="spike_p1"),
            _control_points([1.0, 3.0, 1.0, 1.0, 1.0, 1.0], name="spike_p2"),
            _control_points([1.0, 1.0, 3.0, 1.0, 1.0, 1.0], name="spike_p3"),
            _control_points([1.0, 1.0, 1.0, 3.0, 1.0, 1.0], name="spike_p4"),
            _control_points([1.0, 1.0, 1.0, 1.0, 3.0, 1.0], name="spike_p5"),
            _control_points([1.0, 1.0, 1.0, 1.0, 1.0, 3.0], name="spike_p6"),
            # --- position valleys ---
            _control_points([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], name="valley_p1"),
            _control_points([1.0, 1.0, 0.0, 1.0, 1.0, 1.0], name="valley_p3"),
            _control_points([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], name="valley_p6"),
            # --- extended edges (deepening OLMO's best family) ---
            _control_points([3.0, 2.0, 1.0, 1.0, 2.0, 3.0], name="edges_high_5x"),
            _control_points([3.8, 2.4, 1.0, 1.0, 2.4, 3.8], name="edges_high_7x"),
            _control_points([5.0, 3.0, 1.0, 1.0, 3.0, 5.0], name="edges_high_10x"),
            _control_points([3.0, 2.0, 1.0, 1.0, 1.0, 1.0], name="edges_asym_early"),
            _control_points([1.0, 1.0, 1.0, 1.0, 2.0, 3.0], name="edges_asym_late"),
            _control_points([3.0, 1.0, 1.0, 1.0, 1.0, 3.0], name="edges_narrow"),
            _control_points([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], name="bookend_suppress"),
            # --- nonlinear ramps ---
            _control_points([3.0, 2.0, 1.0, 0.5, 0.25, 0.0], name="decay_fast"),
            _control_points([0.0, 0.25, 0.5, 1.0, 2.0, 3.0], name="grow_fast"),
            _control_points([2.0, 1.8, 1.5, 1.2, 1.0, 0.8], name="decay_gentle"),
            _control_points([0.8, 1.0, 1.2, 1.5, 1.8, 2.0], name="grow_gentle"),
            # --- shifted-up shapes (all above baseline) ---
            _control_points([1.5, 1.75, 2.0, 2.25, 2.5, 2.75], name="shifted_ramp_up"),
            _control_points([2.75, 2.5, 2.25, 2.0, 1.75, 1.5], name="shifted_ramp_down"),
            _control_points([2.0, 2.0, 3.0, 3.0, 2.0, 2.0], name="shifted_bump"),
            _control_points([2.5, 1.75, 1.25, 1.25, 1.75, 2.5], name="shifted_edges"),
            _control_points([3.0, 3.0, 3.0, 2.0, 2.0, 2.0], name="flat_2_early_3"),
            # --- alternating pairs at 3x ---
            _control_points([1.0, 3.0, 1.0, 1.0, 3.0, 1.0], name="pair_quarter"),
            _control_points([3.0, 1.0, 3.0, 1.0, 3.0, 1.0], name="triad_odd"),
            _control_points([1.0, 3.0, 1.0, 3.0, 1.0, 3.0], name="triad_even"),
            # --- three-zone patterns ---
            _control_points([2.0, 2.0, 2.0, 2.0, 1.0, 1.0], name="early_mid_high"),
            _control_points([1.0, 1.0, 2.0, 2.0, 2.0, 2.0], name="mid_late_high"),
            _control_points([2.0, 2.0, 1.0, 1.0, 2.0, 2.0], name="early_late_high"),
            _control_points([0.5, 1.0, 2.0, 2.0, 1.0, 0.5], name="quarter_wave"),
            _control_points([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], name="sawtooth"),
            # --- high-amplitude shaped ---
            _control_points([1.0, 1.0, 1.0, 4.0, 4.0, 4.0], name="late_boost_4.0"),
            _control_points([4.0, 4.0, 4.0, 1.0, 1.0, 1.0], name="early_boost_4.0"),
            # --- bowl / tent shapes ---
            _control_points([2.0, 1.0, 0.5, 0.5, 1.0, 2.0], name="bowl"),
            _control_points([3.0, 1.0, 0.0, 0.0, 1.0, 3.0], name="bowl_deep"),
            _control_points([1.0, 1.5, 2.5, 2.5, 1.5, 1.0], name="tent"),
            _control_points([1.0, 2.0, 4.0, 4.0, 2.0, 1.0], name="tent_steep"),
            _control_points([1.0, 2.5, 2.5, 2.5, 2.5, 1.0], name="plateau"),
        ],
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief", "med", "long", "extended"],
        "max_prompts_per_type": 5,
        "model_key": "0_8B",
    },
    "attn_kitchen_sink": {
        "description": "Comprehensive shared sweep for both attention_contribution and "
        "block_output interventions. Keeps the original broad constant coverage, but "
        "replaces most shaped profiles with mean-centered geometries whose average gain "
        "is exactly 1.0. This reduces the cartridge's bias toward net gain amplification "
        "while preserving dense shape-family coverage for prompt-level separability.",
        "g_specs": _clone_g_specs(BALANCED_SWEEP_G_SPECS),
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief", "med", "long", "extended"],
        "model_key": "0_8B",
    },
    "balanced_kitchen_sink": {
        "description": "Alias of attn_kitchen_sink: shared comprehensive sweep with "
        "mean-centered shaped profiles and broad constant controls. Use with either "
        "--intervention-strategy attention_contribution or --intervention-strategy block_output.",
        "g_specs": _clone_g_specs(BALANCED_SWEEP_G_SPECS),
        "attention_targeting": ATTENTION_TARGETING_NATIVE,
        "prompt_tiers": ["short", "brief", "med", "long", "extended"],
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
