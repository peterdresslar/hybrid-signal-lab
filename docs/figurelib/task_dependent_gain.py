from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from .common import configure_matplotlib, prettify_type


PROFILE_ORDER = [
    "constant_1.35",
    "constant_2",
    "constant_2.75",
    "edges_narrow",
    "early_boost_3.0",
    "middle_bump_2.5",
]

PROFILE_LABELS = {
    "constant_1.35": "constant 1.35",
    "constant_2": "constant 2.0",
    "constant_2.75": "constant 2.75",
    "edges_narrow": "edges narrow",
    "early_boost_3.0": "early boost 3.0",
    "middle_bump_2.5": "middle bump 2.5",
}


def load_heatmap_matrix(type_gain_summary_path: Path) -> tuple[list[str], np.ndarray]:
    with type_gain_summary_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    values_by_type: dict[str, dict[str, float]] = {}
    responsiveness: dict[str, list[float]] = {}
    for row in rows:
        g_profile = row["g_profile"]
        if g_profile not in PROFILE_ORDER:
            continue
        prompt_type = row["type"]
        value = float(row["mean_delta_target_prob"])
        values_by_type.setdefault(prompt_type, {})[g_profile] = value
        responsiveness.setdefault(prompt_type, []).append(abs(value))

    type_order = [
        name
        for name, _ in sorted(
            responsiveness.items(),
            key=lambda item: sum(item[1]) / len(item[1]),
            reverse=True,
        )
    ]
    matrix = np.array(
        [[values_by_type[prompt_type][profile] for profile in PROFILE_ORDER] for prompt_type in type_order]
    )
    return type_order, matrix


def load_profile_curves(meta_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    meta = json.loads(meta_path.read_text())
    attention_indices = np.array(meta["target_attention_layer_indices"], dtype=float)
    x_target = (attention_indices - attention_indices.min()) / (attention_indices.max() - attention_indices.min())

    curves: dict[str, np.ndarray] = {}
    for spec in meta["g_specs"]:
        name = spec["name"]
        if name not in PROFILE_ORDER:
            continue

        if spec["g_function"] == "constant":
            value = float(spec["g_params"]["value"])
            curves[name] = np.full_like(x_target, fill_value=value, dtype=float)
            continue

        source_y = np.array(spec["g_vector"], dtype=float)
        source_x = np.linspace(0.0, 1.0, len(source_y))
        curves[name] = np.interp(x_target, source_x, source_y)

    missing = [profile for profile in PROFILE_ORDER if profile not in curves]
    if missing:
        raise ValueError(f"Missing profile curves: {missing}")

    return x_target, curves


def _draw_top_panel(fig: plt.Figure, gs, x_target: np.ndarray, curves: dict[str, np.ndarray], row: int = 0) -> None:
    line_color = "#666666"
    fill_color = "#F5F5F5"
    for idx, profile in enumerate(PROFILE_ORDER):
        ax = fig.add_subplot(gs[row, idx])
        ax.set_facecolor(fill_color)
        ax.axhline(1.0, color="#CCCCCC", linewidth=0.5, linestyle="--", zorder=1)
        ax.plot(x_target, curves[profile], color=line_color, linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 4.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(PROFILE_LABELS[profile], fontsize=10, pad=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#BBBBBB")
        if idx == 0:
            ax.set_ylabel("gain", fontsize=10)


def _draw_heatmap(ax, matrix: np.ndarray):
    norm = SymLogNorm(linthresh=0.08, vmin=-0.6, vmax=0.6)
    x_edges = np.arange(matrix.shape[1] + 1) - 0.5
    y_edges = np.arange(matrix.shape[0] + 1) - 0.5
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap="RdBu",
        norm=norm,
        edgecolors="#CCCCCC",
        linewidth=0.5,
        shading="flat",
    )
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    return im


def _format_colorbar(cbar) -> None:
    ticks = [-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["-0.3", "-0.1", "-0.05", "0", "0.05", "0.1", "0.3"])
    cbar.set_label("Mean Δp (target probability)")


def plot_task_dependent_gain_response(
    *,
    type_gain_summary_path: Path,
    meta_path: Path,
    output_combined: Path,
    output_top: Path,
    output_bottom: Path,
    caption_path: Path,
) -> None:
    configure_matplotlib(font_family="serif", font_size=11)
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    type_order, matrix = load_heatmap_matrix(type_gain_summary_path)
    x_target, curves = load_profile_curves(meta_path)

    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 4], hspace=0.03, wspace=0.12)
    _draw_top_panel(fig, gs, x_target, curves)
    heatmap_ax = fig.add_subplot(gs[1, :])
    im = _draw_heatmap(heatmap_ax, matrix)
    heatmap_ax.set_xticks(np.arange(len(PROFILE_ORDER)))
    heatmap_ax.set_xticklabels([PROFILE_LABELS[name] for name in PROFILE_ORDER], rotation=35, ha="right")
    heatmap_ax.set_yticks(np.arange(len(type_order)))
    heatmap_ax.set_yticklabels([prettify_type(name) for name in type_order])
    heatmap_ax.set_ylabel("Prompt type")
    heatmap_ax.set_xlabel("Gain profile")
    cbar = fig.colorbar(im, ax=heatmap_ax, fraction=0.035, pad=0.02)
    _format_colorbar(cbar)
    fig.savefig(output_combined, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 1.45), constrained_layout=True)
    gs = fig.add_gridspec(1, 6, wspace=0.12)
    _draw_top_panel(fig, gs, x_target, curves, row=0)
    fig.savefig(output_top, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4.7), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    im = _draw_heatmap(ax, matrix)
    ax.set_xticks(np.arange(len(PROFILE_ORDER)))
    ax.set_xticklabels([PROFILE_LABELS[name] for name in PROFILE_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(type_order)))
    ax.set_yticklabels([prettify_type(name) for name in type_order])
    ax.set_xlabel("Gain profile")
    ax.set_ylabel("Prompt type")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    _format_colorbar(cbar)
    fig.savefig(output_bottom, dpi=300, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 1. Task-dependent response of Qwen3.5-9B to six representative gain profiles "
        "under attention-contribution intervention. Top: gain shape over normalized attention-layer "
        "depth. Bottom: mean change in target-token probability (Δp) for each prompt type and profile; "
        "rows are ordered by average absolute responsiveness across the six profiles."
    )
    caption_path.write_text(caption + "\n")
