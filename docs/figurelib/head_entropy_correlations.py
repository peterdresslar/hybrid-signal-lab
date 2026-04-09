from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .common import configure_matplotlib


PROFILE_ORDER = [
    "edges_narrow",
    "late_boost_4.0",
    "shifted_ramp_down",
    "tent_steep",
]


def load_head_correlations(path: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    obj = json.loads(path.read_text())
    layer_indices = list(obj["layer_indices"])
    matrices: dict[str, np.ndarray] = {}
    for entry in obj["profiles"]:
        matrices[entry["g_profile"]] = np.array(entry["correlation_matrix"], dtype=float)
    return layer_indices, matrices


def plot_head_entropy_correlation_heatmaps(
    *,
    correlations_path: Path,
    output_path: Path,
    profiles: list[str] | None = None,
    figsize: tuple[float, float] = (10.5, 7.6),
    vlim: float = 0.45,
    scout_threshold: float = 0.30,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    layer_indices, matrices = load_head_correlations(correlations_path)
    chosen_profiles = profiles or PROFILE_ORDER
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    image = None

    x_edges = np.arange(17)
    y_edges = np.arange(len(layer_indices) + 1)

    for idx, (ax, profile) in enumerate(zip(axes, chosen_profiles)):
        matrix = matrices[profile]
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            matrix,
            cmap="RdBu_r",
            norm=norm,
            shading="flat",
            edgecolors="none",
        )
        ax.set_title(profile, fontsize=11, pad=6)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, len(layer_indices))
        ax.invert_yaxis()
        ax.set_xticks(np.arange(16) + 0.5)
        ax.set_xticklabels([str(i) for i in range(16)])
        ax.set_yticks(np.arange(len(layer_indices)) + 0.5)
        ax.set_yticklabels([str(layer) for layer in layer_indices])
        ax.tick_params(length=0)

        if idx // 2 == 1:
            ax.set_xlabel("Head index")
        if idx % 2 == 0:
            ax.set_ylabel("Attention layer index")

        scout_positions = np.argwhere(np.abs(matrix) > scout_threshold)
        for row, col in scout_positions:
            ax.add_patch(
                mpatches.Rectangle(
                    (col, row),
                    1,
                    1,
                    fill=False,
                    edgecolor="#2F2F2F",
                    linewidth=0.6,
                )
            )

    cbar = fig.colorbar(image, ax=axes, fraction=0.030, pad=0.03)
    cbar.set_label("Pearson r")

    fig.subplots_adjust(wspace=0.12, hspace=0.18, right=0.90)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_head_entropy_shared_vs_specific(
    *,
    correlations_path: Path,
    output_path: Path,
    shared_profiles: list[str] | None = None,
    contrast_profiles: list[str] | None = None,
    figsize: tuple[float, float] = (12.0, 4.6),
    shared_vlim: float = 0.45,
    contrast_vlim: float = 0.16,
    shared_threshold: float = 0.30,
    contrast_threshold: float = 0.08,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    layer_indices, matrices = load_head_correlations(correlations_path)
    chosen_shared = shared_profiles or PROFILE_ORDER
    chosen_contrasts = contrast_profiles or ["edges_narrow", "tent_steep"]

    mean_matrix = np.mean([matrices[profile] for profile in chosen_shared], axis=0)
    contrast_matrices = {
        profile: matrices[profile] - mean_matrix
        for profile in chosen_contrasts
    }

    x_edges = np.arange(17)
    y_edges = np.arange(len(layer_indices) + 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    shared_norm = mcolors.TwoSlopeNorm(vmin=-shared_vlim, vcenter=0.0, vmax=shared_vlim)
    contrast_norm = mcolors.TwoSlopeNorm(vmin=-contrast_vlim, vcenter=0.0, vmax=contrast_vlim)

    panels = [
        (
            axes[0],
            mean_matrix,
            "Shared responsiveness scaffold\n(mean across 4 router profiles)",
            shared_norm,
            shared_threshold,
        ),
        (
            axes[1],
            contrast_matrices[chosen_contrasts[0]],
            f"{chosen_contrasts[0]} minus shared mean",
            contrast_norm,
            contrast_threshold,
        ),
        (
            axes[2],
            contrast_matrices[chosen_contrasts[1]],
            f"{chosen_contrasts[1]} minus shared mean",
            contrast_norm,
            contrast_threshold,
        ),
    ]

    shared_image = None
    contrast_image = None

    for idx, (ax, matrix, title, norm, threshold) in enumerate(panels):
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            matrix,
            cmap="RdBu_r",
            norm=norm,
            shading="flat",
            edgecolors="none",
        )
        if idx == 0:
            shared_image = image
        else:
            contrast_image = image
        ax.set_title(title, fontsize=10.5, pad=7)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, len(layer_indices))
        ax.invert_yaxis()
        ax.set_xticks(np.arange(16) + 0.5)
        ax.set_xticklabels([str(i) for i in range(16)])
        ax.set_yticks(np.arange(len(layer_indices)) + 0.5)
        ax.set_yticklabels([str(layer) for layer in layer_indices])
        ax.tick_params(length=0)
        ax.set_xlabel("Head index")
        if idx == 0:
            ax.set_ylabel("Attention layer index")

        scout_positions = np.argwhere(np.abs(matrix) > threshold)
        for row, col in scout_positions:
            ax.add_patch(
                mpatches.Rectangle(
                    (col, row),
                    1,
                    1,
                    fill=False,
                    edgecolor="#2F2F2F",
                    linewidth=0.55,
                )
            )

    fig.subplots_adjust(wspace=0.18, bottom=0.24)

    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()

    shared_cax = fig.add_axes([pos0.x0 + 0.12 * pos0.width, 0.11, 0.76 * pos0.width, 0.028])
    contrast_left = pos1.x0
    contrast_right = pos2.x1
    contrast_width = contrast_right - contrast_left
    contrast_cax = fig.add_axes([contrast_left + 0.12 * contrast_width, 0.11, 0.76 * contrast_width, 0.028])

    cbar_shared = fig.colorbar(shared_image, cax=shared_cax, orientation="horizontal")
    cbar_shared.set_label("Pearson r")
    cbar_contrast = fig.colorbar(contrast_image, cax=contrast_cax, orientation="horizontal")
    cbar_contrast.set_label("Δ relative to shared mean")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
