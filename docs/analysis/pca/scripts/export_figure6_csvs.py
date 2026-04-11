from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

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


PROFILES = ["constant_2.6", "edges_narrow_bal_0.55", "late_boost_bal_0.60", "triad_odd_bal_0.45"]


def main() -> None:
    out_dir = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "data" / "022-balanced-attention-hybrid"
    analysis_dir = data_dir / "9B" / "analysis"
    battery_json = REPO_ROOT / "battery" / "data" / "battery_4" / "all_candidates.json"

    prompt_ids, types, X = load_baseline_matrix(data_dir, "9B", battery_json)
    coords, explained, _ = compute_pca(X)
    winners = derive_winner_labels(analysis_dir / "analysis_joined_long.csv", PROFILES, prompt_ids)

    battery_rows = json.loads(battery_json.read_text())
    prompt_meta = {row["id"]: row for row in battery_rows}
    prompt_length = np.array([float(prompt_meta[pid].get("tokens_approx", 0.0)) for pid in prompt_ids], dtype=float)
    X_length_resid = residualize_on_scalar(X, prompt_length)
    length_resid_coords, length_resid_explained, _ = compute_pca(X_length_resid)

    base_rows = []
    for i, prompt_id in enumerate(prompt_ids):
        meta = prompt_meta[prompt_id]
        base_rows.append(
            {
                "prompt_id": prompt_id,
                "type": types[i],
                "winner": winners[i],
                "prompt": meta["prompt"],
                "target": meta["target"],
                "tier": meta.get("tier", ""),
                "source": meta.get("source", ""),
                "tokens_approx": meta.get("tokens_approx", ""),
                "pc1": round(float(coords[i, 0]), 6),
                "pc2": round(float(coords[i, 1]), 6),
                "pc3": round(float(coords[i, 2]), 6),
                "length_resid_pc1": round(float(length_resid_coords[i, 0]), 6),
                "length_resid_pc2": round(float(length_resid_coords[i, 1]), 6),
                "length_resid_pc3": round(float(length_resid_coords[i, 2]), 6),
            }
        )

    files = {
        "figure6_main_task.csv": base_rows,
        "figure6_pc2_pc3_task.csv": base_rows,
        "figure6_pc2_pc3_winner.csv": base_rows,
        "figure6_length_resid_task.csv": base_rows,
        "figure6_length_resid_winner.csv": base_rows,
        "figure6_length_resid_pc2_pc3_task.csv": base_rows,
    }

    for name, rows in files.items():
        path = out_dir / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "output_dir": str(out_dir),
        "n_rows": len(base_rows),
        "explained_variance_ratio_first_3": [round(float(v), 6) for v in explained[:3]],
        "length_residual_explained_variance_ratio_first_3": [round(float(v), 6) for v in length_resid_explained[:3]],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
