from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.figurelib.common import configure_matplotlib, prettify_type, qualitative_11_color_map
from router.experiments.train_router import load_baseline_entropy_vectors


TYPE_LEGEND_ORDER = [
    "code_comprehension",
    "reasoning_numerical",
    "algorithmic",
    "reasoning_tracking",
    "structural_copying",
    "syntactic_pattern",
    "factual_recall",
    "factual_retrieval",
    "domain_knowledge",
    "cultural_memorized",
    "long_range_retrieval",
]

WINNER_ORDER = [
    "constant_2.6",
    "edges_narrow_bal_0.55",
    "late_boost_bal_0.60",
    "triad_odd_bal_0.45",
    "off",
]

WINNER_COLORS = {
    "constant_2.6": "#4C78A8",
    "edges_narrow_bal_0.55": "#F58518",
    "late_boost_bal_0.60": "#54A24B",
    "triad_odd_bal_0.45": "#B279A2",
    "off": "#9E9E9E",
}

WINNER_DISPLAY = {
    "constant_2.6": "constant 2.6",
    "edges_narrow_bal_0.55": "edges narrow",
    "late_boost_bal_0.60": "late boost",
    "triad_odd_bal_0.45": "triad odd",
    "off": "off",
}

ANNOTATION_OFFSETS = {
    "algorithmic": (0.2, -0.6),
    "code_comprehension": (-0.2, 0.3),
    "reasoning_numerical": (0.25, 0.45),
    "reasoning_tracking": (-0.15, 0.65),
    "domain_knowledge": (-0.2, 0.3),
    "long_range_retrieval": (-0.2, -0.45),
    "cultural_memorized": (-0.25, -0.3),
    "factual_retrieval": (0.2, -0.35),
    "structural_copying": (0.1, 0.35),
    "syntactic_pattern": (0.15, -0.15),
    "factual_recall": (0.15, 0.25),
}


def load_saved_points(pca_path: Path) -> tuple[list[dict], np.ndarray]:
    obj = json.loads(pca_path.read_text())
    return obj["points"], np.array(obj["explained_variance_ratio"], dtype=float)


def load_baseline_matrix(data_dir: Path, model_key: str, battery_json: Path) -> tuple[list[str], list[str], np.ndarray]:
    entropy_vectors = load_baseline_entropy_vectors(data_dir, model_key)
    battery_rows = json.loads(battery_json.read_text())
    type_by_prompt = {row["id"]: row["type"] for row in battery_rows}

    prompt_ids = sorted(pid for pid in entropy_vectors if pid in type_by_prompt)
    types = [type_by_prompt[pid] for pid in prompt_ids]
    X = np.stack([entropy_vectors[pid] for pid in prompt_ids], axis=0)
    return prompt_ids, types, X


def derive_winner_labels(joined_long_path: Path, profiles: list[str], prompt_ids: list[str]) -> list[str]:
    allowed_profiles = set(profiles + ["baseline"])
    by_prompt: dict[str, dict[str, float]] = defaultdict(dict)
    with joined_long_path.open(newline="") as f:
        for row in csv.DictReader(f):
            gp = row["g_profile"]
            if gp not in allowed_profiles:
                continue
            try:
                by_prompt[row["prompt_id"]][gp] = float(row["delta_target_prob"])
            except (TypeError, ValueError):
                continue

    labels: list[str] = []
    for prompt_id in prompt_ids:
        values = by_prompt.get(prompt_id, {})
        best_label = "off"
        best_value = 0.0
        for profile in profiles:
            candidate = values.get(profile, float("-inf"))
            if candidate > best_value:
                best_value = candidate
                best_label = profile
        labels.append(best_label)
    return labels


def compute_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    _U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    coords = X_centered @ Vt.T
    explained = (S**2) / (S**2).sum()
    return coords, explained, X_mean


def residualize_on_scalar(X: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(scalar), scalar])
    beta = np.linalg.lstsq(design, X, rcond=None)[0]
    return X - design @ beta


def residualize_on_design(X: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta = np.linalg.lstsq(design, X, rcond=None)[0]
    return X - design @ beta


def residualize_pc1(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords, explained, X_mean = compute_pca(X)
    pc1_scores = coords[:, 0:1]
    _, _, Vt = np.linalg.svd(X - X_mean, full_matrices=False)
    pc1 = Vt[0:1, :]
    X_resid = (X - X_mean) - (pc1_scores @ pc1)
    resid_coords, resid_explained, _ = compute_pca(X_resid)
    return resid_coords, resid_explained


def eta_squared(values: np.ndarray, groups: list[str]) -> float:
    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    buckets: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values.tolist(), groups):
        buckets[group].append(value)
    ss_between = sum(len(bucket) * (float(np.mean(bucket)) - grand_mean) ** 2 for bucket in buckets.values())
    return float(ss_between / ss_total)


def _axis_limits(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_pad = 0.05 * (float(x.max()) - float(x.min()))
    y_pad = 0.08 * (float(y.max()) - float(y.min()))
    return (float(x.min()) - x_pad, float(x.max()) + x_pad), (float(y.min()) - y_pad, float(y.max()) + y_pad)


def plot_task_scatter(
    coords: np.ndarray,
    explained: np.ndarray,
    types: list[str],
    output_path: Path,
    *,
    annotate: bool,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    x = coords[:, 0]
    y = coords[:, 1]
    xlim, ylim = _axis_limits(x, y)
    type_order = [t for t in TYPE_LEGEND_ORDER if t in set(types)]
    type_colors = qualitative_11_color_map(type_order)

    fig, ax = plt.subplots(figsize=(7.3, 5.8))
    for prompt_type in type_order:
        idx = [i for i, t in enumerate(types) if t == prompt_type]
        ax.scatter(
            x[idx],
            y[idx],
            s=18,
            alpha=0.78,
            color=type_colors[prompt_type],
            linewidths=0,
            label=prettify_type(prompt_type),
        )

    if annotate:
        for prompt_type in type_order:
            idx = [i for i, t in enumerate(types) if t == prompt_type]
            cx = float(np.mean(x[idx]))
            cy = float(np.mean(y[idx]))
            dx, dy = ANNOTATION_OFFSETS.get(prompt_type, (0.0, 0.0))
            ax.text(
                cx + dx,
                cy + dy,
                prettify_type(prompt_type),
                fontsize=8.5,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
                zorder=5,
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.grid(False)
    ax.set_facecolor("white")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_single_scatter(
    coords: np.ndarray,
    explained_pair: tuple[float, float],
    labels: list[str],
    label_order: list[str],
    colors: dict[str, str],
    output_path: Path,
    *,
    xlabel: str,
    ylabel: str,
    display_labels: dict[str, str] | None = None,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    x = coords[:, 0]
    y = coords[:, 1]
    xlim, ylim = _axis_limits(x, y)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for label in label_order:
        idx = [i for i, v in enumerate(labels) if v == label]
        if not idx:
            continue
        ax.scatter(
            x[idx],
            y[idx],
            s=18,
            alpha=0.78 if label != "off" else 0.6,
            color=colors[label],
            linewidths=0,
            label=(display_labels or {}).get(label, label),
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"{xlabel} ({explained_pair[0] * 100:.1f}%)")
    ax.set_ylabel(f"{ylabel} ({explained_pair[1] * 100:.1f}%)")
    ax.grid(False)
    ax.set_facecolor("white")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    fig.subplots_adjust(bottom=0.23)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_scree(explained: np.ndarray, output_path: Path, n_components: int = 15) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    n = min(n_components, explained.shape[0])
    xs = np.arange(1, n + 1)
    ys = explained[:n] * 100.0
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(xs, ys, color="#4C78A8", alpha=0.85, width=0.78)
    ax.plot(xs, ys, color="#1F3552", linewidth=1.1, marker="o", markersize=3)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_xticks(xs)
    ax.grid(False)
    ax.text(1, ys[0] + 1.0, f"PC1 {ys[0]:.1f}%", ha="center", va="bottom", fontsize=9)
    if n >= 2:
        ax.text(2, ys[1] + 1.0, f"PC2 {ys[1]:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(
    *,
    report_path: Path,
    explained: np.ndarray,
    resid_explained: np.ndarray,
    pc23_type_eta: tuple[float, float],
    pc23_winner_eta: tuple[float, float],
    resid_type_eta: tuple[float, float],
    resid_winner_eta: tuple[float, float],
    length_source_explained: np.ndarray,
    length_source_type_eta: tuple[float, float, float],
    length_source_winner_eta: tuple[float, float, float],
) -> None:
    scree_shape = (
        "sharp drop after PC2" if explained[2] < 0.03 else "gradual decay across PC3–PC10"
    )
    if max(pc23_type_eta) >= 0.20:
        pc23_task_desc = "Task-category structure persists clearly on PC2 vs PC3, with organized sub-clustering beyond the dominant PC1 axis."
    elif max(pc23_type_eta) >= 0.08:
        pc23_task_desc = "Task-category structure persists weakly on PC2 vs PC3, indicating some finer task sub-structure beneath PC1."
    else:
        pc23_task_desc = "Task-category structure is limited on PC2 vs PC3 and mostly diffuse once PC1 is suppressed."

    winner_desc = (
        "profile-winner identity shows a moderate but coarse linear pattern: most of the signal is a "
        "`constant_2.6` versus non-`constant_2.6` split on PC2, while the specialist winners remain diffuse "
        "and are not cleanly separated from one another."
    )

    if max(pc23_winner_eta + resid_winner_eta) < 0.08 and explained[2] < 0.03:
        ceiling = "The approximate-ceiling claim broadly holds as written."
    elif max(pc23_winner_eta + resid_winner_eta) < 0.25:
        ceiling = "The approximate-ceiling claim should be softened slightly: subdominant linear structure exists, but it remains weak relative to the dominant task axis."
    else:
        ceiling = "The approximate-ceiling claim needs a concrete caveat: there is substantial task-aligned linear structure beneath PC1, and a weaker winner-related axis concentrated mostly in the `constant_2.6` versus rest contrast."

    report = f"""# Figure 6 PCA Diagnostics

## 1. Embedding matrix

The PCA input was the baseline per-prompt attention-head entropy matrix for Qwen 3.5 9B from `data/022-balanced-attention-hybrid/9B/verbose.jsonl`. Each prompt contributes one 128-dimensional vector formed by flattening `attn_entropy_per_head_final` over the 8 hybrid attention layers (`[3, 7, 11, 15, 19, 23, 27, 31]`) and their 16 heads; the entropy is measured at the final prompt position, not averaged across prompt tokens.

## 2. Scree plot

The explained-variance spectrum shows a {scree_shape}. PC1 explains {explained[0] * 100:.2f}% and PC2 explains {explained[1] * 100:.2f}% of variance; PCs 3–10 explain {", ".join(f"{v * 100:.2f}%" for v in explained[2:10])}.

## 3. PC2 vs PC3 task structure

{pc23_task_desc} Quantitatively, η² by task category is {pc23_type_eta[0]:.3f} on PC2 and {pc23_type_eta[1]:.3f} on PC3.

## 4. PC2 vs PC3 and residualized winner structure

On the winner labels, {winner_desc} η² by winner is {pc23_winner_eta[0]:.3f} on PC2 and {pc23_winner_eta[1]:.3f} on PC3. Because PCA components are orthogonal, the residualized PCA is effectively a re-indexed version of the original subdominant subspace: residual PC1 aligns with original PC2, and residual PC2 aligns with original PC3, yielding the same η² values ({resid_winner_eta[0]:.3f}, {resid_winner_eta[1]:.3f}).

## 5. Ceiling claim assessment

{ceiling} The residualized decomposition explains {resid_explained[0] * 100:.2f}% and {resid_explained[1] * 100:.2f}% of variance on its first two axes, confirming that the original dominant PC1 structure has been removed.

## 6. Length-plus-generator residualization

After regressing each feature on both prompt length (`tokens_approx`) and generator/source identity dummies, the spectrum flattens further: the first three components explain {length_source_explained[0] * 100:.2f}%, {length_source_explained[1] * 100:.2f}%, and {length_source_explained[2] * 100:.2f}% of variance. Task structure still survives at a nontrivial level (η² = {length_source_type_eta[0]:.3f}, {length_source_type_eta[1]:.3f}, {length_source_type_eta[2]:.3f} on the first three residual PCs), while winner structure remains weaker and coarser (η² = {length_source_winner_eta[0]:.3f}, {length_source_winner_eta[1]:.3f}, {length_source_winner_eta[2]:.3f}). This is the cleanest current estimate of how much attention geometry organizes by task beyond surface length and generator-template effects.
"""
    report_path.write_text(report)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    manuscript_figures = repo_root / "docs" / "figures" / "manuscript"
    docs_figures = repo_root / "docs" / "figures" / "diagnostics"
    manuscript_figures.mkdir(parents=True, exist_ok=True)
    docs_figures.mkdir(parents=True, exist_ok=True)
    docs_figures.mkdir(parents=True, exist_ok=True)
    report_dir = repo_root / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    data_dir = repo_root / "data" / "022-balanced-attention-hybrid"
    analysis_dir = data_dir / "9B" / "analysis"
    battery_json = repo_root / "battery" / "data" / "battery_4" / "all_candidates.json"
    pca_json = analysis_dir / "analysis_baseline_attn_pca.json"
    joined_long = analysis_dir / "analysis_joined_long.csv"

    profiles = ["constant_2.6", "edges_narrow_bal_0.55", "late_boost_bal_0.60", "triad_odd_bal_0.45"]

    saved_points, saved_explained = load_saved_points(pca_json)
    prompt_ids, types, X = load_baseline_matrix(data_dir, "9B", battery_json)
    coords, explained, _ = compute_pca(X)
    winners = derive_winner_labels(joined_long, profiles, prompt_ids)
    resid_coords, resid_explained = residualize_pc1(X)
    battery_rows = json.loads(battery_json.read_text())
    meta_by_prompt = {row["id"]: row for row in battery_rows}
    prompt_length = np.array([float(meta_by_prompt[pid].get("tokens_approx", 0.0)) for pid in prompt_ids], dtype=float)
    prompt_source = [str(meta_by_prompt[pid].get("source", "unknown")) for pid in prompt_ids]
    X_length_resid = residualize_on_scalar(X, prompt_length)
    length_resid_coords, length_resid_explained, _ = compute_pca(X_length_resid)
    source_order = sorted(set(prompt_source))
    source_to_idx = {name: i for i, name in enumerate(source_order)}
    source_dummy = np.zeros((len(prompt_ids), max(len(source_order) - 1, 0)), dtype=float)
    for i, source in enumerate(prompt_source):
        idx = source_to_idx[source]
        if idx > 0:
            source_dummy[i, idx - 1] = 1.0
    length_source_design = np.column_stack([np.ones_like(prompt_length), prompt_length, source_dummy])
    X_length_source_resid = residualize_on_design(X, length_source_design)
    length_source_coords, length_source_explained, _ = compute_pca(X_length_source_resid)

    if abs(explained[0] - saved_explained[0]) > 0.002 or abs(explained[1] - saved_explained[1]) > 0.002:
        raise RuntimeError(
            f"Reconstructed PCA does not match saved Figure 6 baseline: "
            f"recomputed=({explained[0]:.6f}, {explained[1]:.6f}) "
            f"saved=({saved_explained[0]:.6f}, {saved_explained[1]:.6f})"
        )

    # overwrite the manuscript figure with single-panel task-colored PCA
    plot_task_scatter(coords[:, :2], explained[:2], types, manuscript_figures / "figure6.png", annotate=True)
    plot_scree(explained, docs_figures / "figure6_scree.png")
    plot_single_scatter(
        coords[:, 1:3],
        (explained[1], explained[2]),
        types,
        [t for t in TYPE_LEGEND_ORDER if t in set(types)],
        qualitative_11_color_map([t for t in TYPE_LEGEND_ORDER if t in set(types)]),
        docs_figures / "figure6_pc2_pc3_task.png",
        xlabel="PC2",
        ylabel="PC3",
    )
    plot_single_scatter(
        coords[:, 1:3],
        (explained[1], explained[2]),
        winners,
        WINNER_ORDER,
        WINNER_COLORS,
        docs_figures / "figure6_pc2_pc3_winner.png",
        xlabel="PC2",
        ylabel="PC3",
        display_labels=WINNER_DISPLAY,
    )
    plot_single_scatter(
        resid_coords[:, :2],
        (resid_explained[0], resid_explained[1]),
        types,
        [t for t in TYPE_LEGEND_ORDER if t in set(types)],
        qualitative_11_color_map([t for t in TYPE_LEGEND_ORDER if t in set(types)]),
        docs_figures / "figure6_resid_task.png",
        xlabel="Residual PC1",
        ylabel="Residual PC2",
    )
    plot_single_scatter(
        resid_coords[:, :2],
        (resid_explained[0], resid_explained[1]),
        winners,
        WINNER_ORDER,
        WINNER_COLORS,
        docs_figures / "figure6_resid_winner.png",
        xlabel="Residual PC1",
        ylabel="Residual PC2",
        display_labels=WINNER_DISPLAY,
    )
    plot_task_scatter(
        length_resid_coords[:, :2],
        length_resid_explained[:2],
        types,
        docs_figures / "figure6_length_resid_task.png",
        annotate=True,
    )
    plot_single_scatter(
        length_resid_coords[:, :2],
        (length_resid_explained[0], length_resid_explained[1]),
        winners,
        WINNER_ORDER,
        WINNER_COLORS,
        docs_figures / "figure6_length_resid_winner.png",
        xlabel="Length-resid PC1",
        ylabel="Length-resid PC2",
        display_labels=WINNER_DISPLAY,
    )
    plot_single_scatter(
        length_resid_coords[:, 1:3],
        (length_resid_explained[1], length_resid_explained[2]),
        types,
        [t for t in TYPE_LEGEND_ORDER if t in set(types)],
        qualitative_11_color_map([t for t in TYPE_LEGEND_ORDER if t in set(types)]),
        docs_figures / "figure6_length_resid_pc2_pc3_task.png",
        xlabel="Length-resid PC2",
        ylabel="Length-resid PC3",
    )
    plot_scree(length_resid_explained, docs_figures / "figure6_length_resid_scree.png")
    plot_task_scatter(
        length_source_coords[:, :2],
        length_source_explained[:2],
        types,
        docs_figures / "figure6_length_source_resid_task.png",
        annotate=True,
    )
    plot_single_scatter(
        length_source_coords[:, :2],
        (length_source_explained[0], length_source_explained[1]),
        winners,
        WINNER_ORDER,
        WINNER_COLORS,
        docs_figures / "figure6_length_source_resid_winner.png",
        xlabel="Length+source-resid PC1",
        ylabel="Length+source-resid PC2",
        display_labels=WINNER_DISPLAY,
    )
    plot_single_scatter(
        length_source_coords[:, 1:3],
        (length_source_explained[1], length_source_explained[2]),
        types,
        [t for t in TYPE_LEGEND_ORDER if t in set(types)],
        qualitative_11_color_map([t for t in TYPE_LEGEND_ORDER if t in set(types)]),
        docs_figures / "figure6_length_source_resid_pc2_pc3_task.png",
        xlabel="Length+source-resid PC2",
        ylabel="Length+source-resid PC3",
    )
    plot_scree(length_source_explained, docs_figures / "figure6_length_source_resid_scree.png")

    pc23_type_eta = (eta_squared(coords[:, 1], types), eta_squared(coords[:, 2], types))
    pc23_winner_eta = (eta_squared(coords[:, 1], winners), eta_squared(coords[:, 2], winners))
    resid_type_eta = (eta_squared(resid_coords[:, 0], types), eta_squared(resid_coords[:, 1], types))
    resid_winner_eta = (eta_squared(resid_coords[:, 0], winners), eta_squared(resid_coords[:, 1], winners))

    write_report(
        report_path=report_dir / "figure6_pca_diagnostics.md",
        explained=explained,
        resid_explained=resid_explained,
        pc23_type_eta=pc23_type_eta,
        pc23_winner_eta=pc23_winner_eta,
        resid_type_eta=resid_type_eta,
        resid_winner_eta=resid_winner_eta,
        length_source_explained=length_source_explained,
        length_source_type_eta=(
            eta_squared(length_source_coords[:, 0], types),
            eta_squared(length_source_coords[:, 1], types),
            eta_squared(length_source_coords[:, 2], types),
        ),
        length_source_winner_eta=(
            eta_squared(length_source_coords[:, 0], winners),
            eta_squared(length_source_coords[:, 1], winners),
            eta_squared(length_source_coords[:, 2], winners),
        ),
    )

    summary = {
        "explained_variance_ratio_first_10": [round(float(v), 6) for v in explained[:10]],
        "residual_explained_variance_ratio_first_10": [round(float(v), 6) for v in resid_explained[:10]],
        "length_residual_explained_variance_ratio_first_10": [round(float(v), 6) for v in length_resid_explained[:10]],
        "pc23_type_eta": [round(float(v), 6) for v in pc23_type_eta],
        "pc23_winner_eta": [round(float(v), 6) for v in pc23_winner_eta],
        "resid_type_eta": [round(float(v), 6) for v in resid_type_eta],
        "resid_winner_eta": [round(float(v), 6) for v in resid_winner_eta],
        "corr_pc1_length": round(float(np.corrcoef(coords[:, 0], prompt_length)[0, 1]), 6),
        "corr_pc2_length": round(float(np.corrcoef(coords[:, 1], prompt_length)[0, 1]), 6),
        "length_resid_type_eta": [
            round(float(eta_squared(length_resid_coords[:, 0], types)), 6),
            round(float(eta_squared(length_resid_coords[:, 1], types)), 6),
            round(float(eta_squared(length_resid_coords[:, 2], types)), 6),
        ],
        "length_resid_winner_eta": [
            round(float(eta_squared(length_resid_coords[:, 0], winners)), 6),
            round(float(eta_squared(length_resid_coords[:, 1], winners)), 6),
            round(float(eta_squared(length_resid_coords[:, 2], winners)), 6),
        ],
        "length_source_residual_explained_variance_ratio_first_10": [
            round(float(v), 6) for v in length_source_explained[:10]
        ],
        "length_source_resid_type_eta": [
            round(float(eta_squared(length_source_coords[:, 0], types)), 6),
            round(float(eta_squared(length_source_coords[:, 1], types)), 6),
            round(float(eta_squared(length_source_coords[:, 2], types)), 6),
        ],
        "length_source_resid_winner_eta": [
            round(float(eta_squared(length_source_coords[:, 0], winners)), 6),
            round(float(eta_squared(length_source_coords[:, 1], winners)), 6),
            round(float(eta_squared(length_source_coords[:, 2], winners)), 6),
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
