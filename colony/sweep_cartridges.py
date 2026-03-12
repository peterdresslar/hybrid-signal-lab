import numpy as np

CARTRIDGES = {
    "test_cartridge": {
        "description": "Test cartridge for checking that the vector path matches the scalar path.",
        "g_vectors": [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
    },
    "uniform_check": {
        "description": "Uniform scalar values expressed as vectors. Validates that vector path matches scalar path.",
        "g_vectors": [
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75, 0.75, 0.75, 0.75],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        ],
    },
    "middle_bump": {
        "description": "Test middle-layer influence hypothesis. Layers 3-4 (0-indexed) get elevated gain.",
        "g_vectors": [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.3, 1.3, 1.0, 1.0],
            [1.0, 1.0, 1.5, 1.5, 1.0, 1.0],
            [1.0, 1.0, 0.7, 0.7, 1.0, 1.0],
            [1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
            [0.8, 1.0, 1.3, 1.3, 1.0, 0.8],
            [0.8, 1.0, 1.5, 1.5, 1.0, 0.8],
        ],
    },
    "early_vs_late": {
        "description": "Detailed early-vs-late probe: symmetric boosts/suppression, ramps, and antagonistic mixes across depth.",
        "g_vectors": [
            # baseline
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

            # early-only boosts (late fixed at 1.0)
            [1.1, 1.1, 1.1, 1.0, 1.0, 1.0],
            [1.3, 1.3, 1.3, 1.0, 1.0, 1.0],
            [1.5, 1.5, 1.5, 1.0, 1.0, 1.0],

            # late-only boosts (early fixed at 1.0)
            [1.0, 1.0, 1.0, 1.1, 1.1, 1.1],
            [1.0, 1.0, 1.0, 1.3, 1.3, 1.3],
            [1.0, 1.0, 1.0, 1.5, 1.5, 1.5],

            # early-only suppression
            [0.9, 0.9, 0.9, 1.0, 1.0, 1.0],
            [0.7, 0.7, 0.7, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],

            # late-only suppression
            [1.0, 1.0, 1.0, 0.9, 0.9, 0.9],
            [1.0, 1.0, 1.0, 0.7, 0.7, 0.7],
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],

            # antagonistic: early high / late low
            [1.3, 1.3, 1.3, 0.7, 0.7, 0.7],
            [1.5, 1.5, 1.5, 0.5, 0.5, 0.5],
            [1.2, 1.2, 1.2, 0.8, 0.8, 0.8],

            # antagonistic: late high / early low
            [0.7, 0.7, 0.7, 1.3, 1.3, 1.3],
            [0.5, 0.5, 0.5, 1.5, 1.5, 1.5],
            [0.8, 0.8, 0.8, 1.2, 1.2, 1.2],

            # smooth depth ramps
            [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],  # increasing
            [1.6, 1.4, 1.2, 1.0, 0.8, 0.6],  # decreasing
            [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],  # gentle increasing
            [1.3, 1.2, 1.1, 1.0, 0.9, 0.8],  # gentle decreasing

            # edge-anchored variants
            [1.4, 1.2, 1.0, 1.0, 1.2, 1.4],  # high at edges, neutral center
            [0.6, 0.8, 1.0, 1.0, 0.8, 0.6],  # low at edges, neutral center
        ],
    },
}


def get_cartridge(name: str) -> dict:
    if name not in CARTRIDGES:
        available = ", ".join(CARTRIDGES.keys())
        raise ValueError(f"Unknown cartridge '{name}'. Available: {available}")
    cart = CARTRIDGES[name]
    return {
        "name": name,
        "description": cart["description"],
        "g_vectors": [np.array(v) for v in cart["g_vectors"]],
    }
