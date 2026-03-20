from __future__ import annotations

from typing import Any

import numpy as np

VALID_G_FUNCTIONS = {"constant", "linear", "gaussian", "step", "control_points"}


def attention_slot_positions(
    attention_slots: int,
    *,
    depth_min: float = 0.0,
    depth_max: float = 1.0,
) -> np.ndarray:
    if attention_slots <= 0:
        raise ValueError("attention_slots must be positive.")
    if depth_max <= depth_min:
        raise ValueError("depth_max must be greater than depth_min.")
    if attention_slots == 1:
        return np.array([(depth_min + depth_max) / 2.0], dtype=float)
    return np.linspace(depth_min, depth_max, attention_slots, dtype=float)


def printable_scales(scales: np.ndarray, *, decimals: int = 6) -> list[float]:
    return [round(float(v), decimals) for v in scales.tolist()]


def _apply_clipping(scales: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    clip_min = params.get("clip_min")
    clip_max = params.get("clip_max")
    if clip_min is None and clip_max is None:
        return scales
    lo = -np.inf if clip_min is None else float(clip_min)
    hi = np.inf if clip_max is None else float(clip_max)
    return np.clip(scales, lo, hi)


def build_attention_scales(
    g_function: str,
    *,
    attention_slots: int,
    g_vector: list[float] | np.ndarray | None = None,
    g_params: dict[str, Any] | None = None,
    depth_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    params = dict(g_params or {})
    g_function = g_function.strip().lower()
    if g_function not in VALID_G_FUNCTIONS:
        valid = ", ".join(sorted(VALID_G_FUNCTIONS))
        raise ValueError(f"Unknown g_function '{g_function}'. Expected one of: {valid}.")

    x = attention_slot_positions(
        attention_slots,
        depth_min=depth_range[0],
        depth_max=depth_range[1],
    )

    if g_function == "constant":
        value = float(params.get("value", 1.0))
        scales = np.full(attention_slots, value, dtype=float)
    elif g_function == "linear":
        intercept = float(params.get("intercept", 1.0))
        slope = float(params.get("slope", 0.0))
        scales = intercept + slope * x
    elif g_function == "gaussian":
        baseline = float(params.get("baseline", 1.0))
        amplitude = float(params.get("amplitude", 0.0))
        mu = float(params.get("mu", (depth_range[0] + depth_range[1]) / 2.0))
        sigma = float(params.get("sigma", 0.2))
        if sigma <= 0:
            raise ValueError("gaussian sigma must be > 0.")
        scales = baseline + amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
    elif g_function == "step":
        left = float(params.get("left", 1.0))
        right = float(params.get("right", 1.0))
        threshold = float(params.get("threshold", (depth_range[0] + depth_range[1]) / 2.0))
        scales = np.where(x < threshold, left, right).astype(float)
    else:
        if g_vector is None:
            raise ValueError("g_vector is required for g_function='control_points'.")
        y = np.asarray(g_vector, dtype=float)
        if y.ndim != 1 or y.size < 2:
            raise ValueError("g_vector for control_points must contain at least 2 values.")

        raw_x_positions = params.get("x_positions")
        if raw_x_positions is None:
            xp = np.linspace(depth_range[0], depth_range[1], y.size, dtype=float)
        else:
            xp = np.asarray(raw_x_positions, dtype=float)
            if xp.ndim != 1 or xp.size != y.size:
                raise ValueError("x_positions length must match g_vector length.")
            if not np.all(np.diff(xp) > 0):
                raise ValueError("x_positions must be strictly increasing.")
        scales = np.interp(x, xp, y).astype(float)

    return _apply_clipping(scales, params)


def build_attention_scales_from_spec(
    spec: dict[str, Any],
    *,
    attention_slots: int,
) -> np.ndarray:
    g_function = str(spec.get("g_function", "constant"))
    g_vector = spec.get("g_vector")
    g_params = spec.get("g_params") or {}
    depth_range_raw = spec.get("depth_range", [0.0, 1.0])
    if not isinstance(depth_range_raw, (list, tuple)) or len(depth_range_raw) != 2:
        raise ValueError("depth_range must be a list/tuple of two numbers, e.g. [0.0, 1.0].")
    depth_range = (float(depth_range_raw[0]), float(depth_range_raw[1]))
    return build_attention_scales(
        g_function,
        attention_slots=attention_slots,
        g_vector=g_vector,
        g_params=g_params,
        depth_range=depth_range,
    )
