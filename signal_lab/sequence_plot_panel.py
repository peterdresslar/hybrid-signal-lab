"""
Project restricted router panel interventions onto sequence PCA maps.
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


def get_panel_winners(joined_csv_path: Path, panel: list[str]) -> dict[str, str]:
    """Map each prompt_id to the best performing panel profile (or baseline)."""
    rows = load_csv_rows(joined_csv_path)
    allowed = set(panel) | {"baseline"}
    
    # prompt_id -> list of (profile, delta_prob)
    candidates = {}
    for r in rows:
        pid = r["prompt_id"]
        prof = r["g_profile"]
        if prof not in allowed:
            continue
            
        raw_val = r.get("delta_target_prob", "")
        # Baseline might have blank delta_target_prob, treat as 0.0
        val = float(raw_val) if raw_val not in ("", None) else 0.0
        
        if pid not in candidates:
            candidates[pid] = []
        candidates[pid].append((prof, val))
        
    mapping = {}
    for pid, scores in candidates.items():
        # Fallback to baseline if none of the panel yields a positive improvement
        # or if baseline simply outperforms
        best_prof = "baseline"
        best_val = 0.0
        for prof, val in scores:
            if val > best_val:
                best_val = val
                best_prof = prof
        mapping[pid] = best_prof
        
    return mapping


def export_panel_3d(
    *,
    rows: list[dict],
    family_name: str,
    output_path: Path,
    mode: str,
    panel: list[str],
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
    type_order = panel + ["baseline"]
    
    # 4 distinct colors for the panel, grey for baseline (off)
    color_palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#b0b0b0"]
    colors = {t: color_palette[i] for i, t in enumerate(type_order)}
    
    x_vals = [float(row[x_key]) for row in rows]
    y_vals = [float(row[y_key]) for row in rows]
    z_vals = [float(row[z_key]) for row in rows]

    fig = plt.figure(figsize=(9.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot baseline first so it falls to the background visually
    for prompt_type in reversed(type_order):
        idx = [i for i, t in enumerate(types) if t == prompt_type]
        alpha = 0.15 if prompt_type == "baseline" else 0.85
        s = 10 if prompt_type == "baseline" else 28
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
    ax.set_title(f"Router Panel vs Baseline\n{family_name} ({mode})", fontsize=12)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title="Router Profile",
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-analysis-dir", required=True)
    parser.add_argument("--joined-long-csv", required=True)
    parser.add_argument("--panel", nargs="+", required=True)
    parser.add_argument("--mode", default="attn_resid", choices=["attn_resid", "raw"])
    args = parser.parse_args()

    analysis_dir = Path(args.sequence_analysis_dir).expanduser().resolve()
    joined_csv_path = Path(args.joined_long_csv).expanduser().resolve()
    plot_dir = analysis_dir / "3d_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    mapping = get_panel_winners(joined_csv_path, args.panel)
    
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
            
        out_name = f"{family_name}_3d_{args.mode}_top4plusoff.png"
        out_path = plot_dir / out_name
        
        export_panel_3d(
            rows=valid_rows,
            family_name=family_name,
            output_path=out_path,
            mode=args.mode,
            panel=args.panel,
            elev=24.0,
            azim=-58.0,
        )
        print(f"Mapped {len(valid_rows)} points: {out_path}")

if __name__ == "__main__":
    main()
