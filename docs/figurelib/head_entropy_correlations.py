from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .common import configure_matplotlib
from router.experiments.train_router import load_baseline_entropy_vectors


PROFILE_ORDER = [
    "constant_2.6",
    "edges_narrow_bal_0.55",
    "late_boost_bal_0.60",
    "triad_odd_bal_0.45",
]


def load_head_correlations(path: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    obj = json.loads(path.read_text())
    layer_indices = list(obj["layer_indices"])
    matrices: dict[str, np.ndarray] = {}
    for entry in obj["profiles"]:
        matrices[entry["g_profile"]] = np.array(entry["correlation_matrix"], dtype=float)
    return layer_indices, matrices


def load_delta_by_prompt(joined_path: Path, profile_name: str) -> dict[str, float]:
    deltas: dict[str, float] = {}
    with joined_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["g_profile"] != profile_name:
                continue
            try:
                deltas[row["prompt_id"]] = float(row["delta_target_prob"])
            except (TypeError, ValueError):
                continue
    return deltas


def load_oracle_positive_constant_label_by_prompt(
    joined_path: Path,
    *,
    min_delta: float = 0.0,
) -> dict[str, float]:
    """Binary label for whether any non-baseline constant profile helps a prompt."""
    best_delta_by_prompt: dict[str, float] = {}
    with joined_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["g_profile"] == "baseline":
                continue
            if row["g_function"] != "constant":
                continue
            try:
                delta = float(row["delta_target_prob"])
            except (TypeError, ValueError):
                continue
            prompt_id = row["prompt_id"]
            best_delta_by_prompt[prompt_id] = max(
                best_delta_by_prompt.get(prompt_id, float("-inf")),
                delta,
            )

    return {
        prompt_id: float(best_delta > min_delta)
        for prompt_id, best_delta in best_delta_by_prompt.items()
    }


def load_constant_response_polarity_by_prompt(joined_path: Path) -> dict[str, float]:
    """Signed label for whether constant-profile intervention helps or hurts on average."""
    values_by_prompt: dict[str, list[float]] = {}
    with joined_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["g_profile"] == "baseline":
                continue
            if row["g_function"] != "constant":
                continue
            try:
                delta = float(row["delta_target_prob"])
            except (TypeError, ValueError):
                continue
            values_by_prompt.setdefault(row["prompt_id"], []).append(delta)

    labels: dict[str, float] = {}
    for prompt_id, values in values_by_prompt.items():
        mean_delta = float(np.mean(values))
        if mean_delta > 0:
            labels[prompt_id] = 1.0
        elif mean_delta < 0:
            labels[prompt_id] = -1.0
        else:
            labels[prompt_id] = 0.0
    return labels


def _corr_map_from_vectors(
    prompt_ids: list[str],
    vectors: dict[str, np.ndarray],
    deltas: dict[str, float],
    n_layers: int,
) -> np.ndarray:
    X = np.stack([vectors[pid] for pid in prompt_ids], axis=0)
    y = np.array([deltas[pid] for pid in prompt_ids], dtype=float)
    y_centered = y - y.mean()
    y_norm = np.linalg.norm(y_centered)
    if y_norm == 0:
        return np.zeros((n_layers, X.shape[1] // n_layers), dtype=float)

    X_centered = X - X.mean(axis=0, keepdims=True)
    x_norm = np.linalg.norm(X_centered, axis=0)
    denom = x_norm * y_norm
    corr = np.divide(
        X_centered.T @ y_centered,
        denom,
        out=np.zeros_like(x_norm),
        where=denom > 0,
    )
    n_heads = X.shape[1] // n_layers
    return corr.reshape(n_layers, n_heads)


def _within_prompt_zscore(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for pid, vec in vectors.items():
        mean = vec.mean()
        std = vec.std()
        if std <= 0:
            out[pid] = np.zeros_like(vec)
        else:
            out[pid] = (vec - mean) / std
    return out


def _within_layer_center(vectors: dict[str, np.ndarray], n_layers: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for pid, vec in vectors.items():
        n_heads = vec.shape[0] // n_layers
        mat = vec.reshape(n_layers, n_heads)
        centered = mat - mat.mean(axis=1, keepdims=True)
        out[pid] = centered.reshape(-1)
    return out


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
    figsize: tuple[float, float] = (9.6, 4.6),
    shared_vlim: float = 0.45,
    shared_threshold: float = 0.30,
    top_k: int = 12,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    layer_indices, matrices = load_head_correlations(correlations_path)
    chosen_shared = shared_profiles or PROFILE_ORDER
    mean_matrix = np.mean([matrices[profile] for profile in chosen_shared], axis=0)

    recurrence = np.zeros_like(mean_matrix, dtype=int)
    for profile in chosen_shared:
        matrix = matrices[profile]
        flat = np.abs(matrix).ravel()
        if top_k >= flat.size:
            selected = np.ones_like(flat, dtype=bool)
        else:
            cutoff = np.partition(flat, -top_k)[-top_k]
            selected = flat >= cutoff
        recurrence += selected.reshape(matrix.shape).astype(int)

    x_edges = np.arange(17)
    y_edges = np.arange(len(layer_indices) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    shared_norm = mcolors.TwoSlopeNorm(vmin=-shared_vlim, vcenter=0.0, vmax=shared_vlim)
    recur_cmap = plt.get_cmap("YlOrRd", len(chosen_shared) + 1)
    recur_norm = mcolors.BoundaryNorm(
        boundaries=np.arange(len(chosen_shared) + 2) - 0.5,
        ncolors=recur_cmap.N,
    )

    panels = [
        (
            axes[0],
            mean_matrix,
            "Shared responsiveness scaffold\n(mean across 4 router profiles)",
            "RdBu_r",
            shared_norm,
        ),
        (
            axes[1],
            recurrence,
            f"Scout-head recurrence across profiles\n(top {top_k} |r| heads per profile)",
            recur_cmap,
            recur_norm,
        ),
    ]

    images = []

    for idx, (ax, matrix, title, cmap, norm) in enumerate(panels):
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            matrix,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="none",
        )
        images.append(image)
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

        if idx == 0:
            scout_positions = np.argwhere(np.abs(matrix) > shared_threshold)
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
        else:
            for row, col in np.argwhere(recurrence > 0):
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    str(int(recurrence[row, col])),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#2F2F2F" if recurrence[row, col] <= 2 else "white",
                )

    fig.subplots_adjust(wspace=0.18, bottom=0.24)

    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()

    shared_cax = fig.add_axes([pos0.x0 + 0.12 * pos0.width, 0.11, 0.76 * pos0.width, 0.028])
    recur_cax = fig.add_axes([pos1.x0 + 0.12 * pos1.width, 0.11, 0.76 * pos1.width, 0.028])

    cbar_shared = fig.colorbar(images[0], cax=shared_cax, orientation="horizontal")
    cbar_shared.set_label("Pearson r")
    cbar_recur = fig.colorbar(images[1], cax=recur_cax, orientation="horizontal")
    cbar_recur.set_label("Recurrence count")
    cbar_recur.set_ticks(np.arange(len(chosen_shared) + 1))

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_head_entropy_bistate(
    *,
    data_dir: Path,
    model_key: str,
    correlations_path: Path,
    joined_path: Path,
    output_path: Path,
    profile_name: str,
    figsize: tuple[float, float] = (9.4, 4.6),
    corr_vlim: float = 0.45,
    diff_vlim: float = 0.40,
    corr_threshold: float = 0.30,
    diff_threshold: float = 0.18,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    layer_indices, matrices = load_head_correlations(correlations_path)
    corr_matrix = matrices[profile_name]

    deltas = load_delta_by_prompt(joined_path, profile_name)
    entropy_vectors = load_baseline_entropy_vectors(data_dir, model_key)
    prompt_ids = sorted(set(deltas) & set(entropy_vectors))
    label = np.array([1 if deltas[pid] > 0 else 0 for pid in prompt_ids], dtype=int)
    entropy = np.stack([entropy_vectors[pid] for pid in prompt_ids], axis=0)

    n_layers = len(layer_indices)
    n_heads = entropy.shape[1] // n_layers
    if n_layers * n_heads != entropy.shape[1]:
        raise ValueError("Entropy vector size does not match layer count")

    pos_mean = entropy[label == 1].mean(axis=0)
    nonpos_mean = entropy[label == 0].mean(axis=0)
    diff_matrix = (pos_mean - nonpos_mean).reshape(n_layers, n_heads)

    x_edges = np.arange(n_heads + 1)
    y_edges = np.arange(n_layers + 1)
    corr_norm = mcolors.TwoSlopeNorm(vmin=-corr_vlim, vcenter=0.0, vmax=corr_vlim)
    diff_norm = mcolors.TwoSlopeNorm(vmin=-diff_vlim, vcenter=0.0, vmax=diff_vlim)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    panels = [
        (
            axes[0],
            corr_matrix,
            f"Correlation with Δp\n({profile_name})",
            corr_norm,
            corr_threshold,
            "Pearson r",
        ),
        (
            axes[1],
            diff_matrix,
            "Entropy difference by bistate label\n(intervene-positive minus off)",
            diff_norm,
            diff_threshold,
            "Δ entropy (bits)",
        ),
    ]

    images = []
    for idx, (ax, matrix, title, norm, threshold, _) in enumerate(panels):
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            matrix,
            cmap="RdBu_r",
            norm=norm,
            shading="flat",
            edgecolors="none",
        )
        images.append(image)
        ax.set_title(title, fontsize=10.5, pad=7)
        ax.set_xlim(0, n_heads)
        ax.set_ylim(0, n_layers)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(n_heads) + 0.5)
        ax.set_xticklabels([str(i) for i in range(n_heads)])
        ax.set_yticks(np.arange(n_layers) + 0.5)
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

    fig.subplots_adjust(wspace=0.16, bottom=0.24)

    for ax, image, (_, _, _, _, _, label_txt) in zip(axes, images, panels, strict=True):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + 0.12 * pos.width, 0.11, 0.76 * pos.width, 0.028])
        cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        cbar.set_label(label_txt)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_head_entropy_profile_comparison(
    *,
    correlations_path: Path,
    output_path: Path,
    profiles: list[str],
    figsize: tuple[float, float] = (9.4, 4.6),
    vlim: float = 0.45,
    scout_threshold: float = 0.30,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    layer_indices, matrices = load_head_correlations(correlations_path)
    x_edges = np.arange(17)
    y_edges = np.arange(len(layer_indices) + 1)
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    fig, axes = plt.subplots(1, len(profiles), figsize=figsize, sharey=True)
    if len(profiles) == 1:
        axes = [axes]

    images = []
    for idx, (ax, profile) in enumerate(zip(axes, profiles, strict=True)):
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
        images.append(image)
        ax.set_title(f"Correlation with Δp\n({profile})", fontsize=10.5, pad=7)
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

        scout_positions = np.argwhere(np.abs(matrix) > scout_threshold)
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

    fig.subplots_adjust(wspace=0.16, bottom=0.24)

    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    width = right - left
    cax = fig.add_axes([left + 0.18 * width, 0.11, 0.64 * width, 0.028])
    cbar = fig.colorbar(images[0], cax=cax, orientation="horizontal")
    cbar.set_label("Pearson r")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_head_entropy_normalization_comparison(
    *,
    data_dir: Path,
    model_key: str,
    correlations_path: Path,
    joined_path: Path,
    output_path: Path,
    profiles: list[str],
    figsize: tuple[float, float] = (10.0, 10.5),
    vlim: float = 0.45,
    scout_threshold: float = 0.30,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=10.5)

    layer_indices, raw_corr_matrices = load_head_correlations(correlations_path)
    n_layers = len(layer_indices)
    entropy_vectors = load_baseline_entropy_vectors(data_dir, model_key)

    transformed_sets = {
        "Raw entropy correlation": None,
        "Within-prompt z-score": _within_prompt_zscore(entropy_vectors),
        "Within-layer centered": _within_layer_center(entropy_vectors, n_layers),
    }

    x_edges = np.arange(17)
    y_edges = np.arange(n_layers + 1)
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    fig, axes = plt.subplots(len(transformed_sets), len(profiles), figsize=figsize, sharex=True, sharey=True)

    if len(transformed_sets) == 1:
        axes = np.array([axes])

    def resolve_target_values(target_name: str) -> dict[str, float]:
        if target_name == "oracle_any_constant_positive":
            return load_oracle_positive_constant_label_by_prompt(joined_path)
        if target_name == "constant_response_polarity":
            return load_constant_response_polarity_by_prompt(joined_path)
        return load_delta_by_prompt(joined_path, target_name)

    def resolve_title(target_name: str) -> str:
        if target_name == "oracle_any_constant_positive":
            return "Any constant oracle Δp > 0"
        if target_name == "constant_response_polarity":
            return "Constant-response polarity"
        return target_name

    for row_idx, (row_title, transformed) in enumerate(transformed_sets.items()):
        for col_idx, profile in enumerate(profiles):
            ax = axes[row_idx, col_idx]
            if transformed is None and profile in raw_corr_matrices:
                matrix = raw_corr_matrices[profile]
            else:
                target_values = resolve_target_values(profile)
                vector_source = entropy_vectors if transformed is None else transformed
                prompt_ids = sorted(set(target_values) & set(vector_source))
                matrix = _corr_map_from_vectors(prompt_ids, vector_source, target_values, n_layers)

            image = ax.pcolormesh(
                x_edges,
                y_edges,
                matrix,
                cmap="RdBu_r",
                norm=norm,
                shading="flat",
                edgecolors="none",
            )
            ax.set_xlim(0, 16)
            ax.set_ylim(0, n_layers)
            ax.invert_yaxis()
            ax.set_xticks(np.arange(16) + 0.5)
            ax.set_xticklabels([str(i) for i in range(16)])
            ax.set_yticks(np.arange(n_layers) + 0.5)
            ax.set_yticklabels([str(layer) for layer in layer_indices])
            ax.tick_params(length=0)

            if row_idx == 0:
                ax.set_title(resolve_title(profile), fontsize=11, pad=7)
            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\n\nAttention layer index")
            if row_idx == len(transformed_sets) - 1:
                ax.set_xlabel("Head index")

            scout_positions = np.argwhere(np.abs(matrix) > scout_threshold)
            for row, col in scout_positions:
                ax.add_patch(
                    mpatches.Rectangle(
                        (col, row),
                        1,
                        1,
                        fill=False,
                        edgecolor="#2F2F2F",
                        linewidth=0.5,
                    )
                )

    fig.subplots_adjust(wspace=0.08, hspace=0.16, bottom=0.10)
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Pearson r")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
