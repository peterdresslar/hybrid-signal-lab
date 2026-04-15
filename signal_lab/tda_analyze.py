"""Topological data analysis helpers for Battery 4 representation clouds.

Supports two primary inputs:

1. Baseline last-token attention-head entropy clouds from sweep runs.
2. Hidden-state feature families exported by ``signal_lab.sequence_analyze``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser

from docs.figurelib.common import configure_matplotlib, prettify_type, qualitative_11_color_map
from signal_lab.paths import DATA_DIR_ENV_VAR, configure_data_dir, resolve_input_path

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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_entropy_cloud(run_dir: Path, battery_json: Path) -> tuple[list[str], list[str], np.ndarray]:
    verbose_path = run_dir / "verbose.jsonl"
    if not verbose_path.is_file():
        raise FileNotFoundError(f"Missing verbose.jsonl in {run_dir}")

    battery_rows = json.loads(battery_json.read_text())
    type_by_prompt = {row["id"]: row["type"] for row in battery_rows}

    by_prompt: dict[str, np.ndarray] = {}
    with verbose_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("g_attention_scales") != [1.0] * 8:
                continue
            pid = row["prompt_id"]
            if pid not in type_by_prompt:
                continue
            raw = row.get("attn_entropy_per_head_final")
            if raw is None:
                continue
            vec = np.asarray(raw, dtype=np.float32).reshape(-1)
            by_prompt[pid] = vec

    prompt_ids = sorted(by_prompt)
    types = [type_by_prompt[pid] for pid in prompt_ids]
    X = np.stack([by_prompt[pid] for pid in prompt_ids], axis=0)
    return prompt_ids, types, X


def load_sequence_family(analysis_dir: Path, family_name: str) -> tuple[list[str], list[str], np.ndarray]:
    csv_path = analysis_dir / f"{family_name}_pca.csv"
    manifest_path = analysis_dir / "sequence_analysis_manifest.json"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing PCA CSV for family {family_name}: {csv_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing sequence analysis manifest: {manifest_path}")

    source_run_dir = Path(json.loads(manifest_path.read_text())["source_run_dir"])
    records_by_prompt = {row["prompt_id"]: row for row in load_jsonl(source_run_dir / "records.jsonl")}

    prompt_ids: list[str] = []
    prompt_types: list[str] = []
    vectors: list[np.ndarray] = []
    for pid, record in sorted(records_by_prompt.items()):
        state_path = source_run_dir / str(record["state_file"])
        payload = __import__("torch").load(state_path, map_location="cpu", weights_only=False)
        hidden_states = payload["hidden_states"]
        hs = hidden_states.detach().cpu().numpy() if hasattr(hidden_states, "detach") else np.asarray(hidden_states)

        if family_name == "embedding_last_token":
            vec = hs[0, -1, :]
        elif family_name == "final_layer_last_token":
            vec = hs[-1, -1, :]
        elif family_name == "embedding_mean_pool":
            vec = hs[0].mean(axis=0)
        elif family_name == "final_layer_mean_pool":
            vec = hs[-1].mean(axis=0)
        elif family_name == "all_layers_last_token_concat":
            vec = hs[:, -1, :].reshape(-1)
        elif family_name == "all_layers_mean_pool_concat":
            vec = hs.mean(axis=1).reshape(-1)
        else:
            raise ValueError(f"Unsupported family name: {family_name}")

        prompt_ids.append(pid)
        prompt_types.append(str(record.get("prompt_type", "")))
        vectors.append(np.asarray(vec, dtype=np.float32))

    X = np.stack(vectors, axis=0)
    return prompt_ids, prompt_types, X


def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma == 0.0] = 1.0
    return (X - mu) / sigma


def pca_reduce(X: np.ndarray, n_components: int) -> np.ndarray:
    X_centered = X - X.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ vt[:n_components].T


def subsample(
    prompt_ids: list[str],
    types: list[str],
    X: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[list[str], list[str], np.ndarray]:
    if X.shape[0] <= max_points:
        return prompt_ids, types, X
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=max_points, replace=False))
    return [prompt_ids[i] for i in idx], [types[i] for i in idx], X[idx]


def compute_pca_coords(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_centered = X - X.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    coords = X_centered @ vt.T
    explained = (s**2) / (s**2).sum()
    return coords, explained


def plot_point_cloud(coords: np.ndarray, explained: np.ndarray, types: list[str], out_path: Path, title: str) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    type_order = [t for t in TYPE_LEGEND_ORDER if t in set(types)]
    colors = qualitative_11_color_map(type_order)
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    for prompt_type in type_order:
        idx = [i for i, t in enumerate(types) if t == prompt_type]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=16, alpha=0.78, color=colors[prompt_type], linewidths=0, label=prettify_type(prompt_type))
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title(title, fontsize=12)
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.23)
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_diagrams(diagrams: list[np.ndarray], out_path: Path, title: str) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    fig, axes = plt.subplots(1, max(1, len(diagrams)), figsize=(6.0 * max(1, len(diagrams)), 4.8))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    finite_max = 0.0
    for dgm in diagrams:
        if dgm.size:
            finite = dgm[np.isfinite(dgm[:, 1])]
            if finite.size:
                finite_max = max(finite_max, float(finite[:, 1].max()))
            finite_max = max(finite_max, float(dgm[:, 0].max()))
    finite_max = max(finite_max, 1.0)

    for dim, ax in enumerate(axes):
        dgm = diagrams[dim] if dim < len(diagrams) else np.empty((0, 2))
        ax.plot([0, finite_max], [0, finite_max], color="#999999", linewidth=1.0)
        if dgm.size:
            births = dgm[:, 0]
            deaths = dgm[:, 1].copy()
            inf_mask = ~np.isfinite(deaths)
            deaths[inf_mask] = finite_max * 1.02
            ax.scatter(births, deaths, s=18, alpha=0.8, color="#4C78A8", linewidths=0)
            if np.any(inf_mask):
                ax.scatter(births[inf_mask], deaths[inf_mask], s=28, alpha=0.9, color="#F58518", marker="^", linewidths=0)
        ax.set_title(f"H{dim}", fontsize=12)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_xlim(0, finite_max * 1.05)
        ax.set_ylim(0, finite_max * 1.05)
        ax.grid(False)
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lifetimes(diagrams: list[np.ndarray], out_path: Path, title: str) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    rows = []
    for dim, dgm in enumerate(diagrams):
        if dgm.size == 0:
            continue
        lifetimes = dgm[:, 1] - dgm[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        lifetimes = np.sort(lifetimes)[::-1]
        if lifetimes.size == 0:
            continue
        rows.append((dim, lifetimes))
    for dim, lifetimes in rows:
        ax.plot(np.arange(1, lifetimes.size + 1), lifetimes, marker="o", markersize=3, linewidth=1.1, label=f"H{dim}")
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("Lifetime")
    ax.set_title(title, fontsize=12)
    ax.grid(False)
    if rows:
        ax.legend(frameon=False)
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summarize_diagrams(diagrams: list[np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dim, dgm in enumerate(diagrams):
        finite = dgm[np.isfinite(dgm[:, 1])] if dgm.size else np.empty((0, 2))
        lifetimes = finite[:, 1] - finite[:, 0] if finite.size else np.array([])
        summary[f"h{dim}_count"] = int(dgm.shape[0]) if dgm.size else 0
        summary[f"h{dim}_finite_count"] = int(finite.shape[0]) if finite.size else 0
        summary[f"h{dim}_max_lifetime"] = round(float(lifetimes.max()), 6) if lifetimes.size else 0.0
        summary[f"h{dim}_mean_lifetime"] = round(float(lifetimes.mean()), 6) if lifetimes.size else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ripser/TDA on baseline entropy or sequence-state feature clouds.")
    parser.add_argument("--mode", choices=["entropy", "sequence"], required=True)
    parser.add_argument("--run-dir", required=True, help="Sweep run dir (entropy) or sequence collection dir (sequence).")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and summaries.")
    parser.add_argument("--data-dir", default=None, help=f"Optional base directory override. Also supports {DATA_DIR_ENV_VAR}.")
    parser.add_argument("--battery-json", default="battery/data/battery_4/all_candidates.json", help="Battery JSON for prompt metadata (entropy mode).")
    parser.add_argument("--family", default="final_layer_last_token", help="Sequence feature family in sequence mode.")
    parser.add_argument("--max-points", type=int, default=400, help="Maximum prompts to include in the point cloud.")
    parser.add_argument("--pca-dim", type=int, default=24, help="Optional PCA reduction dimension before ripser.")
    parser.add_argument("--maxdim", type=int, default=2, help="Maximum homology dimension for ripser.")
    parser.add_argument("--coeff", type=int, default=2, help="Field coefficient for ripser.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for prompt subsampling.")
    args = parser.parse_args()

    configure_data_dir(args.data_dir)
    run_dir = resolve_input_path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    if args.mode == "entropy":
        battery_json = resolve_input_path(args.battery_json).expanduser().resolve()
        prompt_ids, types, X = load_entropy_cloud(run_dir, battery_json)
        label = "entropy_last_token_heads"
    else:
        analysis_dir = run_dir / "analysis"
        if not analysis_dir.is_dir():
            analysis_dir = run_dir
        prompt_ids, types, X = load_sequence_family(analysis_dir, args.family)
        label = args.family

    prompt_ids, types, X = subsample(prompt_ids, types, X, args.max_points, args.seed)
    X_std = standardize(X)
    X_work = pca_reduce(X_std, min(args.pca_dim, X_std.shape[1], X_std.shape[0])) if args.pca_dim > 0 else X_std

    coords, explained = compute_pca_coords(X_work)
    result = ripser(X_work, maxdim=args.maxdim, coeff=args.coeff)
    diagrams = result["dgms"]

    plot_point_cloud(coords[:, :2], explained, types, output_dir / f"{label}_pc1_pc2.png", f"{label}: PCA view for TDA input")
    plot_diagrams(diagrams, output_dir / f"{label}_persistence_diagrams.png", f"{label}: persistence diagrams")
    plot_lifetimes(diagrams, output_dir / f"{label}_lifetimes.png", f"{label}: persistence lifetimes")

    summary = {
        "mode": args.mode,
        "label": label,
        "n_points": len(prompt_ids),
        "feature_dim_raw": int(X.shape[1]),
        "feature_dim_tda_input": int(X_work.shape[1]),
        "maxdim": args.maxdim,
        "coeff": args.coeff,
        "subsample_seed": args.seed,
        "point_ids_preview": prompt_ids[:10],
        "explained_variance_ratio_first_5": [round(float(v), 6) for v in explained[:5]],
    }
    summary.update(summarize_diagrams(diagrams))
    with (output_dir / f"{label}_tda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(output_dir / f"{label}_pc1_pc2.png")
    print(output_dir / f"{label}_persistence_diagrams.png")
    print(output_dir / f"{label}_lifetimes.png")
    print(output_dir / f"{label}_tda_summary.json")


if __name__ == "__main__":
    main()
