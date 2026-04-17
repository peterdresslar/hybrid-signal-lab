"""
Project sweep top-4 intervention profiles onto sequence PCA maps.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from docs.figurelib.common import configure_matplotlib
from signal_lab.paths import DATA_DIR_ENV_VAR, configure_data_dir, resolve_input_path


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def axis_limits(values: list[float], pad_frac: float = 0.08) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    span = hi - lo
    pad = span * pad_frac if span > 1e-12 else 1.0
    return lo - pad, hi + pad


def get_top_4_winners(oracle_path: Path, model_label: str) -> tuple[dict[str, str], list[str]]:
    """Return an array mapping prompt_id -> winner mapping, and the exact top 4 profiles."""
    rows = load_csv_rows(oracle_path)
    
    # Filter constraints
    filtered = [r for r in rows if r["model_label"] == model_label and r["winner_scope"] == "full_library" and r["winner_objective"] == "delta_target_prob_max"]
    
    freq = {}
    for r in filtered:
        profile = r["winner_profile"]
        if profile != "baseline":
            freq[profile] = freq.get(profile, 0) + 1
            
    # Sort by frequency descending and grab top 4
    top4 = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)[:4]
    
    mapping = {}
    for r in filtered:
        profile = r["winner_profile"]
        if profile in top4:
            mapping[r["prompt_id"]] = profile
        else:
            mapping[r["prompt_id"]] = "Other"
            
    return mapping, top4


def export_winners_3d(
    *,
    rows: list[dict],
    family_name: str,
    output_path: Path,
    mode: str,
    top4_labels: list[str],
    elev: float,
    azim: float,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    
    if mode == "attn_resid":
        x_key, y_key, z_key = "attn_resid_pc1", "attn_resid_pc2", "attn_resid_pc3"
        axis_labels = ("Length+Attn-resid PC1", "Length+Attn-resid PC2", "Length+Attn-resid PC3")
    else:
        x_key, y_key, z_key = "pc1", "pc2", "pc3"
        axis_labels = ("PC1", "PC2", "PC3")

    types = [row["winner_group"] for row in rows]
    type_order = top4_labels + ["Other"]
    
    # Distinct qualitative colors for top 4, grey for 'Other'
    color_palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#b0b0b0"]
    colors = {t: color_palette[i] for i, t in enumerate(type_order)}
    
    x_vals = [float(row[x_key]) for row in rows]
    y_vals = [float(row[y_key]) for row in rows]
    z_vals = [float(row[z_key]) for row in rows]

    fig = plt.figure(figsize=(9.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot "Other" first so it falls to the background visually
    for prompt_type in reversed(type_order):
        idx = [i for i, t in enumerate(types) if t == prompt_type]
        alpha = 0.2 if prompt_type == "Other" else 0.85
        s = 10 if prompt_type == "Other" else 28
        ax.scatter(
            [x_vals[i] for i in idx],
            [y_vals[i] for i in idx],
            [z_vals[i] for i in idx],
            s=s,
            alpha=alpha,
            color=colors[prompt_type],
            linewidths=0,
            label=prompt_type,
            depthshade=False,
        )

    ax.set_xlim(*axis_limits(x_vals))
    ax.set_ylim(*axis_limits(y_vals))
    ax.set_zlim(*axis_limits(z_vals))
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_title(f"Top 4 Sweep Winners vs Other\n{family_name} ({mode})", fontsize=12)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title="Oracle Profile",
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-analysis-dir", required=True)
    parser.add_argument("--oracle-export", required=True)
    parser.add_argument("--model-label", required=True, choices=["Qwen 9B", "Olmo Hybrid"])
    parser.add_argument("--mode", default="attn_resid", choices=["attn_resid", "raw"])
    args = parser.parse_args()

    analysis_dir = Path(args.sequence_analysis_dir).expanduser().resolve()
    oracle_path = Path(args.oracle_export).expanduser().resolve()
    plot_dir = analysis_dir / "3d_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    mapping, top4 = get_top_4_winners(oracle_path, args.model_label)
    
    for pca_file in sorted(analysis_dir.glob("*_pca.csv")):
        family_name = pca_file.name.removesuffix("_pca.csv")
        rows = load_csv_rows(pca_file)
        
        valid_rows = []
        for r in rows:
            if r["prompt_id"] in mapping:
                r["winner_group"] = mapping[r["prompt_id"]]
                valid_rows.append(r)
                
        if not valid_rows:
            continue
            
        out_name = f"{family_name}_3d_{args.mode}_top4other_v2.png"
        out_path = plot_dir / out_name
        
        export_winners_3d(
            rows=valid_rows,
            family_name=family_name,
            output_path=out_path,
            mode=args.mode,
            top4_labels=top4,
            elev=24.0,
            azim=-58.0,
        )
        print(f"Mapped {len(valid_rows)} points: {out_path}")

if __name__ == "__main__":
    main()
