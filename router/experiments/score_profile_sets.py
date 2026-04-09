"""
score_profile_sets.py — Two-stage router profile-set scoring.

Stage 1:
    Enumerate profile sets and keep the top-N candidates by oracle utility.

Stage 2:
    Score each candidate set by actual learnability using the same baseline
    features and cross-validation pipeline as train_router.py. This ranks sets
    by routed mean delta_p rather than oracle ceiling alone.

Usage:
    python -m router.experiments.score_profile_sets \
        --model-key 9B \
        --data-dir data/022-balanced-attention-hybrid \
        --shortlist-size 40 \
        --top-n 10 \
        --feature-set pca+scalar \
        --n-pca 10 \
        --max-constants 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .select_profiles import load_delta_matrix as load_full_delta_matrix
from .select_profiles import search
from .train_router import (
    assign_labels,
    build_feature_matrix,
    cross_validate,
    load_baseline_entropy_vectors,
    load_scalar_features,
)


def _submatrix(full_matrix: np.ndarray, full_profiles: list[str], selected_profiles: list[str]) -> np.ndarray:
    idx = [full_profiles.index(name) for name in selected_profiles]
    return full_matrix[:, idx]


def evaluate_candidate_sets(
    *,
    prompt_ids: list[str],
    full_profiles: list[str],
    full_matrix: np.ndarray,
    candidate_sets: list[dict],
    entropy_vectors: dict[str, np.ndarray],
    scalar_features: dict[str, dict[str, float]],
    feature_set: str,
    n_pca: int,
    n_folds: int,
    lr: float,
    n_iter: int,
    reg: float,
) -> tuple[list[dict], list[str], np.ndarray | None, np.ndarray | None]:
    X, feature_names, pca_components, pca_mean = build_feature_matrix(
        prompt_ids,
        entropy_vectors,
        scalar_features,
        n_pca=n_pca,
        feature_set=feature_set,
    )

    scored: list[dict] = []
    for rank0, candidate in enumerate(candidate_sets, 1):
        profiles = candidate["profiles"]
        delta_matrix = _submatrix(full_matrix, full_profiles, profiles)
        labels, class_names = assign_labels(delta_matrix, profiles)
        results = cross_validate(
            X,
            labels,
            delta_matrix,
            n_classes=len(class_names),
            class_names=class_names,
            n_folds=n_folds,
            lr=lr,
            n_iter=n_iter,
            reg=reg,
        )

        scored.append(
            {
                "stage1_rank": rank0,
                "profiles": profiles,
                "oracle_mean_delta_p": float(candidate["mean_oracle_routed_delta_p"]),
                "stage1_routing_advantage": float(candidate["routing_advantage"]),
                "stage1_coverage_pct": float(candidate["coverage_pct"]),
                "winner_entropy": float(candidate.get("winner_entropy", 0.0)),
                "min_profile_wins": int(candidate.get("min_profile_wins", 0)),
                "mean_abs_pairwise_corr": float(candidate.get("mean_abs_pairwise_corr", 0.0)),
                "routed_mean_delta_p": float(results["routed_mean_delta_p"]),
                "cv_accuracy": float(results["overall_accuracy"]),
                "routing_efficiency": float(
                    results["routed_mean_delta_p"] / results["oracle_mean_delta_p"]
                    if results["oracle_mean_delta_p"] > 0
                    else 0.0
                ),
                "off_predictions": int(results["off_predictions"]),
                "true_class_distribution": results["true_class_distribution"],
                "predicted_class_distribution": results["predicted_class_distribution"],
                "fold_results": results["fold_results"],
            }
        )

    scored.sort(
        key=lambda r: (
            -r["routed_mean_delta_p"],
            -r["cv_accuracy"],
            -r["oracle_mean_delta_p"],
        )
    )
    return scored, feature_names, pca_components, pca_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage profile-set scoring for router experiments.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--shortlist-size", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--min-profile-wins", type=int, default=0)
    parser.add_argument("--max-mean-abs-corr", type=float, default=None)
    parser.add_argument("--max-constants", type=int, default=None)
    parser.add_argument("--feature-set", default="pca+scalar", choices=["pca", "raw", "scalar", "pca+scalar"])
    parser.add_argument("--n-pca", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = data_dir / args.model_key / "analysis"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / args.model_key / "router"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading full delta matrix for {args.model_key}...")
    prompt_ids, full_profiles, full_matrix = load_full_delta_matrix(analysis_dir)
    print(f"  {len(prompt_ids)} prompts × {len(full_profiles)} profiles")

    print("\nStage 1: oracle shortlist search...")
    t0 = time.time()
    shortlist = search(
        full_matrix,
        full_profiles,
        top_n=args.shortlist_size,
        min_coverage=args.min_coverage,
        objective="oracle",
        min_profile_wins=args.min_profile_wins,
        max_mean_abs_corr=args.max_mean_abs_corr,
        max_constants=args.max_constants,
    )
    print(f"Stage 1 done in {time.time() - t0:.1f}s")

    print("\nLoading baseline features once for Stage 2...")
    entropy_vectors = load_baseline_entropy_vectors(data_dir, args.model_key)
    scalar_features = load_scalar_features(analysis_dir)

    print(
        f"\nStage 2: cross-validated router scoring on top {len(shortlist)} candidate sets "
        f"({args.feature_set}, n_pca={args.n_pca})..."
    )
    t1 = time.time()
    scored, feature_names, pca_components, pca_mean = evaluate_candidate_sets(
        prompt_ids=prompt_ids,
        full_profiles=full_profiles,
        full_matrix=full_matrix,
        candidate_sets=shortlist,
        entropy_vectors=entropy_vectors,
        scalar_features=scalar_features,
        feature_set=args.feature_set,
        n_pca=args.n_pca,
        n_folds=args.n_folds,
        lr=args.lr,
        n_iter=args.n_iter,
        reg=args.reg,
    )
    print(f"Stage 2 done in {time.time() - t1:.1f}s")

    top_scored = scored[: args.top_n]
    print("\nTop candidate sets by cross-validated routed delta_p:")
    print(f"{'rank':>4}  {'routed':>8}  {'acc':>6}  {'oracle':>8}  {'eff':>6}  {'cov%':>5}  profiles")
    print("-" * 120)
    for rank, row in enumerate(top_scored, 1):
        print(
            f"{rank:>4}  {row['routed_mean_delta_p']:>8.4f}  {row['cv_accuracy']:>6.3f}  "
            f"{row['oracle_mean_delta_p']:>8.4f}  {row['routing_efficiency']:>6.2%}  "
            f"{row['stage1_coverage_pct']:>5.1f}  {', '.join(row['profiles'])}"
        )

    output = {
        "model_key": args.model_key,
        "data_dir": str(data_dir),
        "shortlist_size": args.shortlist_size,
        "top_n": args.top_n,
        "feature_set": args.feature_set,
        "n_pca": args.n_pca,
        "n_folds": args.n_folds,
        "hyperparameters": {"lr": args.lr, "n_iter": args.n_iter, "reg": args.reg},
        "constraints": {
            "min_coverage": args.min_coverage,
            "min_profile_wins": args.min_profile_wins,
            "max_mean_abs_corr": args.max_mean_abs_corr,
            "max_constants": args.max_constants,
        },
        "feature_names": feature_names,
        "shortlist": shortlist,
        "top_sets": top_scored,
    }
    if pca_components is not None:
        output["pca_components"] = pca_components.tolist()
        output["pca_mean"] = pca_mean.tolist() if pca_mean is not None else None

    output_path = output_dir / "profile_set_cv_ranking.json"
    output_path.write_text(json.dumps(output, indent=2))

    report_path = output_dir / "profile_set_cv_ranking_report.txt"
    with report_path.open("w") as f:
        f.write(f"Profile-Set CV Ranking: {args.model_key}\n")
        f.write(f"{'=' * 72}\n\n")
        f.write(f"Data: {data_dir}\n")
        f.write(f"Feature set: {args.feature_set}\n")
        f.write(f"n_pca: {args.n_pca}\n")
        f.write(f"Shortlist size: {args.shortlist_size}\n")
        f.write(f"Top-N reported: {args.top_n}\n")
        f.write(f"Constraints: min_coverage={args.min_coverage}, min_profile_wins={args.min_profile_wins}, ")
        f.write(f"max_mean_abs_corr={args.max_mean_abs_corr}, max_constants={args.max_constants}\n\n")
        f.write("Top sets by cross-validated routed delta_p:\n\n")
        for rank, row in enumerate(top_scored, 1):
            f.write(
                f"  #{rank:>2}  routed={row['routed_mean_delta_p']:.4f}  "
                f"acc={row['cv_accuracy']:.3f}  oracle={row['oracle_mean_delta_p']:.4f}  "
                f"eff={row['routing_efficiency']:.1%}  cov={row['stage1_coverage_pct']:.0f}%\n"
            )
            for profile in row["profiles"]:
                f.write(f"        {profile}\n")
            f.write("\n")

    print(f"\nResults saved to {output_path}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
