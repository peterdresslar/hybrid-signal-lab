from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.analysis.pca.scripts.pca_figure6_diagnostics import (
    compute_pca,
    derive_winner_labels,
    load_baseline_matrix,
    residualize_on_scalar,
)
from docs.figurelib.common import configure_matplotlib

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional local dependency
    AutoTokenizer = None


MODEL_ID = "Qwen/Qwen3.5-9B-Base"
PROFILES = ["constant_2.6", "edges_narrow_bal_0.55", "late_boost_bal_0.60", "triad_odd_bal_0.45"]
SOURCE_ORDER = [
    "gen_alpha_seq",
    "gen_alt",
    "gen_delim",
    "gen_indent",
    "gen_mirror",
    "gen_num_seq",
    "gen_struct",
    "gen_table",
    "gen_word_list",
]
SOURCE_COLORS = {
    "gen_alpha_seq": "#4C78A8",
    "gen_alt": "#F58518",
    "gen_delim": "#54A24B",
    "gen_indent": "#E45756",
    "gen_mirror": "#72B7B2",
    "gen_num_seq": "#B279A2",
    "gen_struct": "#FF9DA6",
    "gen_table": "#9D755D",
    "gen_word_list": "#BAB0AC",
}
SOURCE_LABELS = {
    "gen_alpha_seq": "alpha seq",
    "gen_alt": "alternation",
    "gen_delim": "delimited",
    "gen_indent": "indent",
    "gen_mirror": "mirror",
    "gen_num_seq": "number seq",
    "gen_struct": "structured",
    "gen_table": "table",
    "gen_word_list": "word list",
}
ANNOTATION_OFFSETS = {
    "gen_alpha_seq": (0.15, -0.15),
    "gen_alt": (0.15, 0.15),
    "gen_delim": (-0.15, 0.15),
    "gen_indent": (0.0, 0.2),
    "gen_mirror": (-0.2, -0.05),
    "gen_num_seq": (0.1, -0.15),
    "gen_struct": (-0.1, 0.15),
    "gen_table": (-0.15, -0.15),
    "gen_word_list": (0.1, 0.15),
}


def _axis_limits(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_pad = 0.06 * (float(x.max()) - float(x.min()))
    y_pad = 0.08 * (float(y.max()) - float(y.min()))
    return (float(x.min()) - x_pad, float(x.max()) + x_pad), (float(y.min()) - y_pad, float(y.max()) + y_pad)


def _load_tokenizer():
    if AutoTokenizer is None:
        return None
    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        return None


def _count_tokens(tokenizer, prompt: str) -> int | None:
    if tokenizer is None:
        return None
    try:
        return int(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    except Exception:
        return None


def plot_structural_copying(
    coords: np.ndarray,
    explained: np.ndarray,
    sources: list[str],
    output_path: Path,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    x = coords[:, 0]
    y = coords[:, 1]
    xlim, ylim = _axis_limits(x, y)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for source in SOURCE_ORDER:
        idx = [i for i, s in enumerate(sources) if s == source]
        if not idx:
            continue
        ax.scatter(
            x[idx],
            y[idx],
            s=22,
            alpha=0.82,
            color=SOURCE_COLORS[source],
            linewidths=0,
            label=SOURCE_LABELS[source],
        )
        cx = float(np.mean(x[idx]))
        cy = float(np.mean(y[idx]))
        dx, dy = ANNOTATION_OFFSETS.get(source, (0.0, 0.0))
        ax.text(
            cx + dx,
            cy + dy,
            SOURCE_LABELS[source],
            fontsize=8.5,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
            zorder=5,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"Length-residual PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"Length-residual PC2 ({explained[1] * 100:.1f}%)")
    ax.grid(False)
    ax.set_facecolor("white")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    fig.subplots_adjust(bottom=0.27)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    docs_figures = REPO_ROOT / "docs" / "figures" / "diagnostics"
    csv_dir = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "csv"
    docs_figures.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "data" / "022-balanced-attention-hybrid"
    analysis_dir = data_dir / "9B" / "analysis"
    battery_json = REPO_ROOT / "battery" / "data" / "battery_4" / "all_candidates.json"
    prompt_ids, types, X = load_baseline_matrix(data_dir, "9B", battery_json)
    raw_coords, _raw_explained, _ = compute_pca(X)
    winners = derive_winner_labels(analysis_dir / "analysis_joined_long.csv", PROFILES, prompt_ids)

    battery_rows = json.loads(battery_json.read_text())
    prompt_meta = {row["id"]: row for row in battery_rows}
    prompt_length = np.array([float(prompt_meta[pid].get("tokens_approx", 0.0)) for pid in prompt_ids], dtype=float)
    X_length_resid = residualize_on_scalar(X, prompt_length)
    length_resid_coords, length_resid_explained, _ = compute_pca(X_length_resid)

    keep_idx = [i for i, t in enumerate(types) if t == "structural_copying"]
    sc_prompt_ids = [prompt_ids[i] for i in keep_idx]
    sc_coords = length_resid_coords[keep_idx, :2]
    sc_sources = [prompt_meta[pid].get("source", "") for pid in sc_prompt_ids]

    tokenizer = _load_tokenizer()

    rows = []
    for out_i, src_i in enumerate(keep_idx):
        prompt_id = prompt_ids[src_i]
        meta = prompt_meta[prompt_id]
        prompt = meta["prompt"]
        rows.append(
            {
                "prompt_id": prompt_id,
                "type": "structural_copying",
                "source": meta.get("source", ""),
                "winner": winners[src_i],
                "prompt": prompt,
                "target": meta.get("target", ""),
                "tier": meta.get("tier", ""),
                "tokens_approx": meta.get("tokens_approx", ""),
                "qwen_token_count": _count_tokens(tokenizer, prompt),
                "char_len": len(prompt),
                "length_resid_pc1": round(float(length_resid_coords[src_i, 0]), 6),
                "length_resid_pc2": round(float(length_resid_coords[src_i, 1]), 6),
                "length_resid_pc3": round(float(length_resid_coords[src_i, 2]), 6),
                "pc1": round(float(raw_coords[src_i, 0]), 6),
                "pc2": round(float(raw_coords[src_i, 1]), 6),
                "pc3": round(float(raw_coords[src_i, 2]), 6),
            }
        )

    rows.sort(key=lambda row: (row["source"], row["length_resid_pc1"]))

    csv_path = csv_dir / "figure6_structural_copying_length_resid_source.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_structural_copying(
        sc_coords,
        length_resid_explained[:2],
        sc_sources,
        docs_figures / "figure6_structural_copying_length_resid_source.png",
    )

    centroid_rows = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, source in enumerate(sc_sources):
        grouped[source].append(i)
    for source in SOURCE_ORDER:
        idx = grouped.get(source, [])
        if not idx:
            continue
        centroid_rows.append(
            {
                "source": source,
                "display": SOURCE_LABELS[source],
                "n_prompts": len(idx),
                "centroid_pc1": round(float(np.mean(sc_coords[idx, 0])), 6),
                "centroid_pc2": round(float(np.mean(sc_coords[idx, 1])), 6),
            }
        )

    centroid_csv = csv_dir / "figure6_structural_copying_length_resid_source_centroids.csv"
    with centroid_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(centroid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(centroid_rows)

    summary = {
        "n_structural_copying_prompts": len(rows),
        "source_counts": dict(Counter(sc_sources)),
        "tokenizer_loaded": tokenizer is not None,
        "figure_path": str(docs_figures / "figure6_structural_copying_length_resid_source.png"),
        "csv_path": str(csv_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
