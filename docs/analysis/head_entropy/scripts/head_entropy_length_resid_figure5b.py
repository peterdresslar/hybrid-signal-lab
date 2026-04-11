from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.analysis.head_entropy.scripts.head_entropy_diagnostics import (
    DEFAULT_9B_PROFILES,
    build_dataset,
    one_vs_rest_auc,
    residualize_dataset_on_scalar,
    specialist_vs_anchor_auc,
)
from docs.figurelib.head_entropy_diagnostics import plot_shared_vs_specialist_auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Length-residualized Figure 5b head-entropy diagnostics.")
    parser.add_argument("--data-dir", default="data/022-balanced-attention-hybrid")
    parser.add_argument("--model-key", default="9B")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_9B_PROFILES)
    parser.add_argument("--battery-json", default="battery/data/battery_4/all_candidates.json")
    parser.add_argument("--output-dir", default="docs/analysis/head_entropy/outputs/qwen9b/length_resid")
    parser.add_argument("--figure-path", default="docs/figures/diagnostics/figure_head_entropy_specialist_auc_qwen9b_length_resid.png")
    parser.add_argument("--n-permutations-auc", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    battery_json = Path(args.battery_json)
    output_dir = Path(args.output_dir)
    figure_path = Path(args.figure_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(data_dir=data_dir, model_key=args.model_key, profiles=list(args.profiles))
    battery_rows = json.loads(battery_json.read_text())
    token_map = {row["id"]: float(row.get("tokens_approx", 0.0)) for row in battery_rows}
    tokens = np.array([token_map[pid] for pid in dataset.prompt_ids], dtype=float)
    resid_dataset = residualize_dataset_on_scalar(dataset, tokens)

    auc = one_vs_rest_auc(
        resid_dataset,
        n_permutations=args.n_permutations_auc,
        random_state=args.random_state,
    )
    specialist_auc = specialist_vs_anchor_auc(
        resid_dataset,
        anchor_profile=args.profiles[0],
        n_permutations=args.n_permutations_auc,
        random_state=args.random_state + 1,
    )

    shared_path = output_dir / "metrics_summary.json"
    specialist_path = output_dir / "specialist_vs_anchor_summary.json"
    shared_summary = {
        "model_key": args.model_key,
        "data_dir": str(data_dir),
        "profiles": list(args.profiles),
        "layer_indices": resid_dataset.layer_indices,
        "n_prompts": len(resid_dataset.prompt_ids),
        "length_covariate": "tokens_approx",
        "entropy_definition": (
            "Per-head attention entropy at the final prompt position "
            "(`attn[0, :, -1, :]`), residualized feature-wise on battery `tokens_approx`."
        ),
        "auc": {
            "max_auc_matrix": auc["max_auc_matrix"].tolist(),
            "best_profile_matrix": auc["best_profile_matrix"].tolist(),
            "top_rankings": auc["rankings"][:100],
        },
    }
    specialist_summary = {
        "model_key": args.model_key,
        "data_dir": str(data_dir),
        "profiles": list(args.profiles),
        "layer_indices": resid_dataset.layer_indices,
        "n_prompts": len(resid_dataset.prompt_ids),
        "length_covariate": "tokens_approx",
        "anchor_profile": specialist_auc["anchor_profile"],
        "specialist_names": specialist_auc["specialist_names"],
        "max_auc_matrix": specialist_auc["max_auc_matrix"].tolist(),
        "best_specialist_matrix": specialist_auc["best_specialist_matrix"].tolist(),
        "top_rankings": specialist_auc["rankings"][:100],
    }
    shared_path.write_text(json.dumps(shared_summary, indent=2))
    specialist_path.write_text(json.dumps(specialist_summary, indent=2))

    plot_shared_vs_specialist_auc(
        shared_metrics_path=shared_path,
        specialist_metrics_path=specialist_path,
        output_path=figure_path,
    )

    print(
        json.dumps(
            {
                "shared_metrics_path": str(shared_path),
                "specialist_metrics_path": str(specialist_path),
                "figure_path": str(figure_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
