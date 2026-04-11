from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .common import configure_matplotlib


def _robust_norm(matrix: np.ndarray, *, lower_q: float = 0.05, upper_q: float = 0.99) -> mcolors.Normalize:
    vmin = float(np.quantile(matrix, lower_q))
    vmax = float(np.quantile(matrix, upper_q))
    if vmax <= vmin:
        vmax = float(np.max(matrix))
        vmin = float(np.min(matrix))
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def plot_head_entropy_diagnostics(
    *,
    metrics_path: Path,
    output_path: Path,
    figsize: tuple[float, float] = (12.0, 4.6),
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    metrics = json.loads(metrics_path.read_text())
    layer_indices = metrics["layer_indices"]
    variance = np.array(metrics["variance"]["matrix"], dtype=float)
    max_auc = np.array(metrics["auc"]["max_auc_matrix"], dtype=float)
    selection_freq = np.array(metrics["sparse_logistic"]["selection_frequency_matrix"], dtype=float)

    n_layers, n_heads = variance.shape
    x_edges = np.arange(n_heads + 1)
    y_edges = np.arange(n_layers + 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    variance_norm = mcolors.Normalize(vmin=float(np.min(variance)), vmax=float(np.max(variance)))
    auc_norm = mcolors.Normalize(vmin=0.5, vmax=max(0.75, float(np.max(max_auc))))
    sel_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    panels = [
        (axes[0], variance, "Baseline entropy variance", "viridis", variance_norm, None),
        (axes[1], max_auc, "Max one-vs-rest AUC", "magma", auc_norm, None),
        (axes[2], selection_freq, "Sparse logistic selection frequency", "cividis", sel_norm, None),
    ]

    images = []
    for idx, (ax, matrix, title, cmap, norm, _) in enumerate(panels):
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

    fig.subplots_adjust(wspace=0.2, bottom=0.24)
    for idx, (ax, _, _, _, _, _) in enumerate(panels):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + 0.12 * pos.width, 0.11, 0.76 * pos.width, 0.028])
        cbar = fig.colorbar(images[idx], cax=cax, orientation="horizontal")
        if idx == 0:
            cbar.set_label("Variance")
        elif idx == 1:
            cbar.set_label("AUC")
        else:
            cbar.set_label("Selection frequency")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_figure5a_variance_vs_residual_auc(
    *,
    raw_metrics_path: Path,
    length_resid_metrics_path: Path,
    output_path: Path,
    figsize: tuple[float, float] = (8.4, 4.4),
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    raw_metrics = json.loads(raw_metrics_path.read_text())
    resid_metrics = json.loads(length_resid_metrics_path.read_text())
    layer_indices = raw_metrics["layer_indices"]
    variance = np.array(raw_metrics["variance"]["matrix"], dtype=float)
    max_auc = np.array(resid_metrics["auc"]["max_auc_matrix"], dtype=float)

    n_layers, n_heads = variance.shape
    x_edges = np.arange(n_heads + 1)
    y_edges = np.arange(n_layers + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    variance_norm = mcolors.Normalize(vmin=float(np.min(variance)), vmax=float(np.max(variance)))
    auc_norm = _robust_norm(max_auc, lower_q=0.05, upper_q=0.99)

    panels = [
        (axes[0], variance, "Baseline entropy variance", "viridis", variance_norm),
        (axes[1], max_auc, "Length-residualized max AUC", "cividis", auc_norm),
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

    fig.subplots_adjust(wspace=0.18, bottom=0.23)
    for idx, ax in enumerate(axes):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + 0.14 * pos.width, 0.10, 0.72 * pos.width, 0.03])
        cbar = fig.colorbar(images[idx], cax=cax, orientation="horizontal")
        cbar.set_label("Variance" if idx == 0 else "AUC")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_shared_vs_specialist_auc(
    *,
    shared_metrics_path: Path,
    specialist_metrics_path: Path,
    output_path: Path,
    figsize: tuple[float, float] = (8.6, 4.4),
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    shared_metrics = json.loads(shared_metrics_path.read_text())
    specialist_metrics = json.loads(specialist_metrics_path.read_text())
    layer_indices = shared_metrics["layer_indices"]
    shared_auc = np.array(shared_metrics["auc"]["max_auc_matrix"], dtype=float)
    specialist_auc = np.array(specialist_metrics["max_auc_matrix"], dtype=float)
    residual_loss = shared_auc - specialist_auc

    n_layers, n_heads = shared_auc.shape
    x_edges = np.arange(n_heads + 1)
    y_edges = np.arange(n_layers + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    shared_norm = mcolors.Normalize(vmin=0.49, vmax=0.81)
    specialist_norm = _robust_norm(residual_loss, lower_q=0.01, upper_q=0.99)

    panels = [
        (axes[0], shared_auc, "Shared interventionability AUC", "cividis", shared_norm),
        (axes[1], residual_loss, "Residual loss after subtraction", "Blues", specialist_norm),
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
        else:
            flat_idx = np.argsort(matrix.ravel())[:5]
            seen: set[tuple[int, int]] = set()
            for flat in flat_idx:
                layer_pos, head_idx = np.unravel_index(flat, matrix.shape)
                if (layer_pos, head_idx) in seen:
                    continue
                seen.add((layer_pos, head_idx))
                ax.scatter(
                    head_idx + 0.5,
                    layer_pos + 0.5,
                    s=24,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.7,
                )
                ax.text(
                    head_idx + 0.62,
                    layer_pos + 0.32,
                    f"L{layer_indices[layer_pos]}H{head_idx}",
                    fontsize=6.5,
                    color="black",
                )

    fig.subplots_adjust(wspace=0.18, bottom=0.23)
    for idx, ax in enumerate(axes):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + 0.14 * pos.width, 0.10, 0.72 * pos.width, 0.03])
        cbar = fig.colorbar(images[idx], cax=cax, orientation="horizontal")
        cbar.set_label("AUC" if idx == 0 else "AUC loss")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
