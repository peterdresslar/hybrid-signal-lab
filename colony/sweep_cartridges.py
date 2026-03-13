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


CARTRIDGES = {
    "test_cartridge": {
        "description": "Test cartridge for checking baseline end-to-end behavior.",
        "g_specs": [
            _constant(1.0, name="baseline"),
        ],
        "prompt": "The color with the shortest wavelength is",
        "target": "violet",
        "model_key": "0_8B",
    },
    "test_cartridge_prompts": {
        "description": "Test cartridge for checking prompt tier loading.",
        "g_specs": [
            _constant(1.0, name="baseline"),
        ],
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
        "prompt_tiers": ["short", "brief"],
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
        "prompt_tiers": ["short", "brief"],
        "model_key": "0_8B",
    },
}


def list_cartridges() -> list[str]:
    return sorted(CARTRIDGES.keys())


def get_cartridge(name: str) -> dict:
    if name not in CARTRIDGES:
        available = ", ".join(list_cartridges())
        raise ValueError(f"Unknown cartridge '{name}'. Available: {available}")
    cart = dict(CARTRIDGES[name])
    cart["name"] = name
    cart["g_specs"] = [dict(spec) for spec in cart["g_specs"]]
    return cart
