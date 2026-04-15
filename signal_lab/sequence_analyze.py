"""Analyze baseline sequence-collection runs into compact representation artifacts.

This module is meant to bolt directly onto ``signal_lab.collect_sequences`` output.
It reads the saved per-prompt ``.pt`` state files and exports a compact analysis
bundle suitable for downstream PCA work without having to move the raw state
artifacts around.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from signal_lab.paths import DATA_DIR_ENV_VAR, configure_data_dir, default_analysis_output_dir, resolve_input_path
from docs.figurelib.common import configure_matplotlib, prettify_type, qualitative_11_color_map

DEFAULT_PREFIX = "sequence_analysis"
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


def resolve_run_dir(path_str: str) -> Path:
    run_dir = resolve_input_path(path_str).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(
            f"Expected a sequence collection directory with manifest.json and records.jsonl: {run_dir}"
        )
    return run_dir


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_centered = X - X.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    coords = X_centered @ vt.T
    explained = (s**2) / (s**2).sum()
    return coords, explained


def residualize_on_scalar(X: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(scalar), scalar])
    beta = np.linalg.lstsq(design, X, rcond=None)[0]
    return X - design @ beta


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


def axis_limits(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_pad = 0.05 * (float(x.max()) - float(x.min()))
    y_pad = 0.08 * (float(y.max()) - float(y.min()))
    return (float(x.min()) - x_pad, float(x.max()) + x_pad), (float(y.min()) - y_pad, float(y.max()) + y_pad)


def plot_task_scatter(
    coords: np.ndarray,
    explained_pair: tuple[float, float],
    types: list[str],
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    x = coords[:, 0]
    y = coords[:, 1]
    xlim, ylim = axis_limits(x, y)
    type_order = [t for t in TYPE_LEGEND_ORDER if t in set(types)]
    type_colors = qualitative_11_color_map(type_order)

    fig, ax = plt.subplots(figsize=(7.1, 5.6))
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
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"{xlabel} ({explained_pair[0] * 100:.1f}%)")
    ax.set_ylabel(f"{ylabel} ({explained_pair[1] * 100:.1f}%)")
    ax.grid(False)
    ax.set_facecolor("white")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=9,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.subplots_adjust(bottom=0.23)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_compare(
    *,
    types: list[str],
    raw_coords: np.ndarray,
    raw_explained: np.ndarray,
    resid_coords: np.ndarray,
    resid_explained: np.ndarray,
    output_path: Path,
    family_name: str,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    type_order = [t for t in TYPE_LEGEND_ORDER if t in set(types)]
    colors = qualitative_11_color_map(type_order)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5))
    panels = [
        (axes[0], raw_coords[:, 0], raw_coords[:, 1], raw_explained[:2], "Raw PC1 vs PC2", "PC1", "PC2"),
        (
            axes[1],
            resid_coords[:, 0],
            resid_coords[:, 1],
            resid_explained[:2],
            "Length-residualized PC1 vs PC2",
            "Length-resid PC1",
            "Length-resid PC2",
        ),
    ]
    for ax, x, y, explained_pair, title, xlabel, ylabel in panels:
        xlim, ylim = axis_limits(x, y)
        for prompt_type in type_order:
            idx = [i for i, t in enumerate(types) if t == prompt_type]
            ax.scatter(
                x[idx],
                y[idx],
                s=18,
                alpha=0.78,
                color=colors[prompt_type],
                linewidths=0,
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"{xlabel} ({explained_pair[0] * 100:.1f}%)")
        ax.set_ylabel(f"{ylabel} ({explained_pair[1] * 100:.1f}%)")
        ax.grid(False)
        ax.set_facecolor("white")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=colors[t], markeredgewidth=0)
        for t in type_order
    ]
    labels = [prettify_type(t) for t in type_order]
    fig.suptitle(f"{family_name}: raw vs length-residualized PCA", fontsize=13, y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.22)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_scree(explained: np.ndarray, output_path: Path, *, title: str, n_components: int = 15) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    n = min(n_components, explained.shape[0])
    xs = np.arange(1, n + 1)
    ys = explained[:n] * 100.0
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(xs, ys, color="#4C78A8", alpha=0.85, width=0.78)
    ax.plot(xs, ys, color="#1F3552", linewidth=1.1, marker="o", markersize=3)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_xticks(xs)
    ax.grid(False)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def extract_feature_families(state_payload: dict[str, Any]) -> dict[str, np.ndarray]:
    hidden_states = state_payload["hidden_states"]
    if isinstance(hidden_states, torch.Tensor):
        hs = hidden_states.detach().cpu().numpy()
    else:
        hs = np.asarray(hidden_states)

    # hs shape: (L+1, N, d)
    embedding_last = hs[0, -1, :]
    final_layer_last = hs[-1, -1, :]
    embedding_mean = hs[0].mean(axis=0)
    final_layer_mean = hs[-1].mean(axis=0)
    concat_last = hs[:, -1, :].reshape(-1)
    concat_mean = hs.mean(axis=1).reshape(-1)

    return {
        "embedding_last_token": embedding_last.astype(np.float32, copy=False),
        "final_layer_last_token": final_layer_last.astype(np.float32, copy=False),
        "embedding_mean_pool": embedding_mean.astype(np.float32, copy=False),
        "final_layer_mean_pool": final_layer_mean.astype(np.float32, copy=False),
        "all_layers_last_token_concat": concat_last.astype(np.float32, copy=False),
        "all_layers_mean_pool_concat": concat_mean.astype(np.float32, copy=False),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_feature_family(
    *,
    family_name: str,
    matrix: np.ndarray,
    prompt_ids: list[str],
    prompt_types: list[str],
    sources: list[str],
    tokens: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    coords, explained = compute_pca(matrix)
    resid_matrix = residualize_on_scalar(matrix, tokens)
    resid_coords, resid_explained = compute_pca(resid_matrix)
    figures_dir = output_dir / f"{family_name}_figures"
    ensure_dir(figures_dir)

    rows: list[dict[str, Any]] = []
    for i, prompt_id in enumerate(prompt_ids):
        rows.append(
            {
                "prompt_id": prompt_id,
                "type": prompt_types[i],
                "source": sources[i],
                "tokens_approx": round(float(tokens[i]), 3),
                "pc1": round(float(coords[i, 0]), 6),
                "pc2": round(float(coords[i, 1]), 6),
                "pc3": round(float(coords[i, 2]), 6),
                "length_resid_pc1": round(float(resid_coords[i, 0]), 6),
                "length_resid_pc2": round(float(resid_coords[i, 1]), 6),
                "length_resid_pc3": round(float(resid_coords[i, 2]), 6),
            }
        )
    write_csv(output_dir / f"{family_name}_pca.csv", rows)
    plot_compare(
        types=prompt_types,
        raw_coords=coords,
        raw_explained=explained,
        resid_coords=resid_coords,
        resid_explained=resid_explained,
        output_path=figures_dir / "pc1_pc2_raw_vs_length_resid.png",
        family_name=family_name,
    )
    plot_task_scatter(
        coords[:, :2],
        (explained[0], explained[1]),
        prompt_types,
        figures_dir / "pc1_pc2_raw_task.png",
        title=f"{family_name}: raw PC1 vs PC2",
        xlabel="PC1",
        ylabel="PC2",
    )
    plot_task_scatter(
        coords[:, 1:3],
        (explained[1], explained[2]),
        prompt_types,
        figures_dir / "pc2_pc3_raw_task.png",
        title=f"{family_name}: raw PC2 vs PC3",
        xlabel="PC2",
        ylabel="PC3",
    )
    plot_task_scatter(
        resid_coords[:, :2],
        (resid_explained[0], resid_explained[1]),
        prompt_types,
        figures_dir / "pc1_pc2_length_resid_task.png",
        title=f"{family_name}: length-resid PC1 vs PC2",
        xlabel="Length-resid PC1",
        ylabel="Length-resid PC2",
    )
    plot_task_scatter(
        resid_coords[:, 1:3],
        (resid_explained[1], resid_explained[2]),
        prompt_types,
        figures_dir / "pc2_pc3_length_resid_task.png",
        title=f"{family_name}: length-resid PC2 vs PC3",
        xlabel="Length-resid PC2",
        ylabel="Length-resid PC3",
    )
    plot_scree(explained, figures_dir / "raw_scree.png", title=f"{family_name}: raw scree")
    plot_scree(resid_explained, figures_dir / "length_resid_scree.png", title=f"{family_name}: length-resid scree")

    summary = {
        "family": family_name,
        "n_prompts": int(matrix.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "explained_variance_ratio_first_5": [round(float(v), 6) for v in explained[:5]],
        "length_resid_explained_variance_ratio_first_5": [round(float(v), 6) for v in resid_explained[:5]],
        "corr_pc1_tokens": round(float(np.corrcoef(coords[:, 0], tokens)[0, 1]), 6),
        "corr_pc2_tokens": round(float(np.corrcoef(coords[:, 1], tokens)[0, 1]), 6),
        "corr_length_resid_pc1_tokens": round(float(np.corrcoef(resid_coords[:, 0], tokens)[0, 1]), 6),
        "corr_length_resid_pc2_tokens": round(float(np.corrcoef(resid_coords[:, 1], tokens)[0, 1]), 6),
        "task_eta_raw_pc1": round(float(eta_squared(coords[:, 0], prompt_types)), 6),
        "task_eta_raw_pc2": round(float(eta_squared(coords[:, 1], prompt_types)), 6),
        "task_eta_length_resid_pc1": round(float(eta_squared(resid_coords[:, 0], prompt_types)), 6),
        "task_eta_length_resid_pc2": round(float(eta_squared(resid_coords[:, 1], prompt_types)), 6),
        "figures_dir": str(figures_dir),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a signal_lab sequence-collection directory into compact PCA-ready artifacts."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Sequence collection directory produced by signal_lab.collect_sequences.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <run-dir>/analysis).",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Artifact filename prefix (default: {DEFAULT_PREFIX}).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"Optional base directory override. Also supports {DATA_DIR_ENV_VAR}.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap on prompts to analyze, in records order.",
    )
    args = parser.parse_args()

    configure_data_dir(args.data_dir)
    run_dir = resolve_run_dir(args.run_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_analysis_output_dir(run_dir)
    ensure_dir(output_dir)

    records = load_jsonl(run_dir / "records.jsonl")
    if args.max_prompts is not None:
        records = records[: args.max_prompts]
    if not records:
        raise ValueError("No records found to analyze.")

    prompt_ids: list[str] = []
    prompt_types: list[str] = []
    sources: list[str] = []
    tokens: list[float] = []
    scalar_rows: list[dict[str, Any]] = []
    feature_rows: dict[str, list[np.ndarray]] = defaultdict(list)

    for record in records:
        prompt_id = str(record["prompt_id"])
        state_path = run_dir / str(record["state_file"])
        state_payload = torch.load(state_path, map_location="cpu", weights_only=False)

        prompt_ids.append(prompt_id)
        prompt_types.append(str(record.get("prompt_type", "")))
        sources.append(str(record.get("source", "")))
        tokens.append(float(record.get("tokens_approx", 0.0) or 0.0))

        scalar_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_type": record.get("prompt_type", ""),
                "source": record.get("source", ""),
                "tokens_approx": record.get("tokens_approx", ""),
                "num_tokens": record.get("num_tokens", ""),
                "num_layers_plus_embedding": record.get("num_layers_plus_embedding", ""),
                "hidden_size": record.get("hidden_size", ""),
                "final_entropy_bits": record.get("final_entropy_bits", ""),
                "mean_entropy_bits": record.get("mean_entropy_bits", ""),
                "state_file": record.get("state_file", ""),
            }
        )

        features = extract_feature_families(state_payload)
        for family_name, vec in features.items():
            feature_rows[family_name].append(vec)

    write_csv(output_dir / f"{args.prefix}_prompt_scalars.csv", scalar_rows)

    token_array = np.array(tokens, dtype=float)
    family_summaries: list[dict[str, Any]] = []
    for family_name, rows in feature_rows.items():
        matrix = np.stack(rows, axis=0)
        summary = analyze_feature_family(
            family_name=family_name,
            matrix=matrix,
            prompt_ids=prompt_ids,
            prompt_types=prompt_types,
            sources=sources,
            tokens=token_array,
            output_dir=output_dir,
        )
        family_summaries.append(summary)

    write_csv(output_dir / f"{args.prefix}_family_summary.csv", family_summaries)

    manifest = {
        "run_kind": "sequence_collection_analysis",
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "prefix": args.prefix,
        "n_prompts": len(prompt_ids),
        "feature_families": [row["family"] for row in family_summaries],
    }
    with (output_dir / f"{args.prefix}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote analysis files:")
    print(f"- {output_dir / f'{args.prefix}_prompt_scalars.csv'}")
    print(f"- {output_dir / f'{args.prefix}_family_summary.csv'}")
    for family_name in feature_rows:
        print(f"- {output_dir / f'{family_name}_pca.csv'}")
    print(f"- {output_dir / f'{args.prefix}_manifest.json'}")


if __name__ == "__main__":
    main()
