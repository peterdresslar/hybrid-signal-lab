"""
train_probes.py — Train per-profile linear value probes for effective geometry.

This is the first `050`-series training path. Instead of predicting a winner
label, it fits one regularized linear regressor per profile to predict prompt-
level delta_target_prob directly. Routing is then defined procedurally:

  1. predict delta_p for every profile in the panel
  2. choose the profile with the highest predicted value
  3. abstain to baseline if the predicted value is below a threshold

The feature-extraction path intentionally reuses the `train_router.py` stack so
raw, length-residualized, and attention-residualized sequence banks remain
runtime-compatible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from model.backend import InterventionMode, normalize_intervention_mode
from router.experiments.train_router import (
    build_feature_matrix,
    load_baseline_entropy_vectors,
    load_delta_matrix,
    load_profile_specs,
    load_scalar_features,
    load_sequence_family_vectors,
)


def fit_ridge(X: np.ndarray, y: np.ndarray, reg: float) -> tuple[np.ndarray, float]:
    """Fit a ridge regressor on already-standardized features."""
    y_mean = float(y.mean())
    y_centered = y - y_mean
    gram = X.T @ X
    rhs = X.T @ y_centered
    w = np.linalg.solve(gram + reg * np.eye(X.shape[1], dtype=np.float64), rhs)
    return w, y_mean


def predict_ridge_bundle(
    X: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Predict per-profile delta_p values."""
    return X @ weights + bias


def choose_profiles(
    pred_matrix: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose best profile or abstain based on predicted values.

    Returns:
        chosen_idx: profile column indices, or -1 for off
        chosen_val: max predicted value per prompt
    """
    best_idx = pred_matrix.argmax(axis=1)
    best_val = pred_matrix[np.arange(pred_matrix.shape[0]), best_idx]
    chosen_idx = np.where(best_val > threshold, best_idx, -1)
    return chosen_idx, best_val


def summarize_routed_delta(
    chosen_idx: np.ndarray,
    delta_matrix: np.ndarray,
) -> tuple[float, dict[str, int]]:
    """Return mean routed delta_p and prediction counts."""
    routed = []
    counts: dict[str, int] = {"off": 0}
    for i, pred in enumerate(chosen_idx):
        if pred < 0:
            routed.append(0.0)
            counts["off"] += 1
        else:
            routed.append(max(0.0, float(delta_matrix[i, pred])))
    return float(np.mean(routed)), counts


def profile_regression_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 0.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    corr = 0.0
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"mse": mse, "mae": mae, "r2": r2, "corr": corr}


def cross_validate_probes(
    X: np.ndarray,
    delta_matrix: np.ndarray,
    profile_names: list[str],
    *,
    n_folds: int,
    reg: float,
    thresholds: list[float],
) -> dict:
    n = X.shape[0]
    n_profiles = delta_matrix.shape[1]
    np.random.seed(42)
    perm = np.random.permutation(n)
    folds = np.array_split(perm, n_folds)

    oof_pred = np.zeros((n, n_profiles), dtype=np.float64)
    fold_results: list[dict] = []

    for fold_idx in range(n_folds):
        print(f"  fold {fold_idx + 1}/{n_folds}")
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])

        mu = X[train_idx].mean(axis=0)
        std = X[train_idx].std(axis=0)
        std[std == 0] = 1.0
        X_train = (X[train_idx] - mu) / std
        X_test = (X[test_idx] - mu) / std

        weights = np.zeros((X.shape[1], n_profiles), dtype=np.float64)
        bias = np.zeros(n_profiles, dtype=np.float64)
        per_profile_metrics: dict[str, dict[str, float]] = {}

        for j, profile_name in enumerate(profile_names):
            w, b = fit_ridge(X_train, delta_matrix[train_idx, j], reg=reg)
            weights[:, j] = w
            bias[j] = b
            preds = X_test @ w + b
            per_profile_metrics[profile_name] = profile_regression_summary(
                delta_matrix[test_idx, j], preds
            )

        fold_pred = predict_ridge_bundle(X_test, weights, bias)
        oof_pred[test_idx] = fold_pred

        threshold_metrics = {}
        for threshold in thresholds:
            chosen_idx, _ = choose_profiles(fold_pred, threshold=threshold)
            routed_delta = []
            off_count = 0
            for local_i, global_i in enumerate(test_idx):
                pred = chosen_idx[local_i]
                if pred < 0:
                    routed_delta.append(0.0)
                    off_count += 1
                else:
                    routed_delta.append(max(0.0, float(delta_matrix[global_i, pred])))
            threshold_metrics[str(threshold)] = {
                "routed_mean_delta_p": float(np.mean(routed_delta)),
                "off_predictions": off_count,
            }

        oracle = np.maximum(delta_matrix[test_idx].max(axis=1), 0.0).mean()
        fold_results.append(
            {
                "fold": fold_idx,
                "n_test": len(test_idx),
                "oracle_mean_delta_p": float(oracle),
                "threshold_metrics": threshold_metrics,
                "per_profile_regression": per_profile_metrics,
            }
        )

    threshold_summary = {}
    for threshold in thresholds:
        chosen_idx, _ = choose_profiles(oof_pred, threshold=threshold)
        routed_mean_delta_p, counts = summarize_routed_delta(chosen_idx, delta_matrix)
        pred_counts = {"off": counts.get("off", 0)}
        for j, profile_name in enumerate(profile_names):
            pred_counts[profile_name] = int((chosen_idx == j).sum())
        threshold_summary[str(threshold)] = {
            "routed_mean_delta_p": routed_mean_delta_p,
            "predicted_profile_distribution": pred_counts,
        }

    oracle_mean_delta_p = float(np.maximum(delta_matrix.max(axis=1), 0.0).mean())
    best_threshold = max(
        thresholds,
        key=lambda t: threshold_summary[str(t)]["routed_mean_delta_p"],
    )

    per_profile_regression = {}
    for j, profile_name in enumerate(profile_names):
        per_profile_regression[profile_name] = profile_regression_summary(
            delta_matrix[:, j], oof_pred[:, j]
        )

    return {
        "n_folds": n_folds,
        "fold_results": fold_results,
        "oracle_mean_delta_p": oracle_mean_delta_p,
        "threshold_summary": threshold_summary,
        "best_threshold": float(best_threshold),
        "best_threshold_routed_mean_delta_p": threshold_summary[str(best_threshold)]["routed_mean_delta_p"],
        "per_profile_regression": per_profile_regression,
        "oof_pred": oof_pred,
    }


def parse_thresholds(values: list[float] | None) -> list[float]:
    if not values:
        return [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    return sorted({float(v) for v in values})


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Train per-profile value probes.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--profiles", nargs="+", required=True, help="One or more profile names")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-pca", type=int, default=10)
    parser.add_argument("--sequence-n-pca", type=int, default=10)
    parser.add_argument(
        "--feature-set",
        default="pca+scalar",
        choices=[
            "pca",
            "raw",
            "scalar",
            "pca+scalar",
            "sequence_pca",
            "pca+sequence_pca",
            "scalar+sequence_pca",
            "pca+scalar+sequence_pca",
        ],
    )
    parser.add_argument("--sequence-states-dir", default=None)
    parser.add_argument(
        "--sequence-family",
        default=None,
        choices=[
            "embedding_last_token",
            "final_layer_last_token",
            "embedding_mean_pool",
            "final_layer_mean_pool",
            "all_layers_last_token_concat",
            "all_layers_mean_pool_concat",
        ],
    )
    parser.add_argument(
        "--sequence-residualization",
        default="raw",
        choices=["raw", "length_resid", "attn_resid"],
    )
    parser.add_argument("--ridge-reg", type=float, default=1.0)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument(
        "--intervention-mode",
        default="auto",
        help='Intervention mode for runtime artifact: "attention_contribution", "block_output", or "auto".',
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = data_dir / args.model_key / "analysis"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / args.model_key / "probe_router"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.intervention_mode == "auto":
        inferred_mode = (
            InterventionMode.BLOCK_OUTPUT.value
            if "block" in str(data_dir).lower()
            else InterventionMode.ATTENTION_CONTRIBUTION.value
        )
    else:
        inferred_mode = args.intervention_mode
    intervention_mode = normalize_intervention_mode(inferred_mode)

    thresholds = parse_thresholds(args.thresholds)

    print(f"Model: {args.model_key}")
    print(f"Profiles: {args.profiles}")
    print(f"Feature set: {args.feature_set}")
    print(f"Intervention mode: {intervention_mode.value}")
    if "sequence_pca" in args.feature_set:
        print(f"Sequence family: {args.sequence_family}")
        print(f"Sequence residualization: {args.sequence_residualization}")
    print(f"Threshold grid: {thresholds}")
    print()

    print("Loading baseline entropy vectors from verbose.jsonl...")
    t0 = time.time()
    entropy_vectors = load_baseline_entropy_vectors(data_dir, args.model_key)
    print(f"  {len(entropy_vectors)} vectors in {time.time()-t0:.1f}s")

    print("Loading scalar features from joined_long...")
    scalar_features = load_scalar_features(analysis_dir)
    print(f"  {len(scalar_features)} prompts")

    sequence_vectors = None
    sequence_token_counts = None
    if "sequence_pca" in args.feature_set:
        if not args.sequence_states_dir or not args.sequence_family:
            raise ValueError(
                "Sequence feature sets require both --sequence-states-dir and --sequence-family."
            )
        print(f"Loading sequence vectors ({args.sequence_family}) from states dir...")
        t0 = time.time()
        sequence_vectors, sequence_token_counts = load_sequence_family_vectors(
            Path(args.sequence_states_dir), args.sequence_family
        )
        print(f"  {len(sequence_vectors)} vectors in {time.time()-t0:.1f}s")

    print("Loading delta matrix for selected profiles...")
    prompt_ids, delta_matrix = load_delta_matrix(analysis_dir, args.profiles)
    print(f"  {delta_matrix.shape[0]} prompts × {delta_matrix.shape[1]} profiles")

    print(f"\nBuilding feature matrix ({args.feature_set}, n_pca={args.n_pca})...")
    (
        X,
        feature_names,
        pca_components,
        pca_mean,
        sequence_pca_components,
        sequence_pca_mean,
        sequence_residual_beta,
    ) = build_feature_matrix(
        prompt_ids,
        entropy_vectors,
        scalar_features,
        sequence_vectors=sequence_vectors,
        sequence_token_counts=sequence_token_counts,
        n_pca=args.n_pca,
        sequence_n_pca=args.sequence_n_pca,
        feature_set=args.feature_set,
        sequence_residualization=args.sequence_residualization,
    )
    print(f"  Feature matrix: {X.shape}")

    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    results = cross_validate_probes(
        X,
        delta_matrix,
        args.profiles,
        n_folds=args.n_folds,
        reg=args.ridge_reg,
        thresholds=thresholds,
    )

    best_fixed_dp = float(np.maximum(delta_matrix, 0.0).mean(axis=0).max())

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {args.model_key}")
    print(f"{'=' * 60}")
    print(f"  Best fixed profile:            {best_fixed_dp:.4f}")
    print(f"  Oracle (perfect routing):      {results['oracle_mean_delta_p']:.4f}")
    print(f"  Best threshold:                {results['best_threshold']:.4f}")
    print(f"  Routed (best threshold):       {results['best_threshold_routed_mean_delta_p']:.4f}")
    print(
        f"  Routing efficiency:            "
        f"{results['best_threshold_routed_mean_delta_p'] / results['oracle_mean_delta_p']:.1%} of oracle"
    )

    print("\nThreshold summary:")
    for threshold in thresholds:
        summary = results["threshold_summary"][str(threshold)]
        print(
            f"  t={threshold:>6.3f}  routed={summary['routed_mean_delta_p']:.4f}  "
            f"pred={summary['predicted_profile_distribution']}"
        )

    print("\nPer-profile regression summary:")
    for profile_name in args.profiles:
        metrics = results["per_profile_regression"][profile_name]
        print(
            f"  {profile_name:<24} "
            f"corr={metrics['corr']:.3f}  r2={metrics['r2']:.3f}  "
            f"mse={metrics['mse']:.6f}  mae={metrics['mae']:.6f}"
        )

    save_results = {
        "model_key": args.model_key,
        "profiles": args.profiles,
        "intervention_mode": intervention_mode.value,
        "feature_set": args.feature_set,
        "n_pca": args.n_pca,
        "sequence_n_pca": args.sequence_n_pca if "sequence_pca" in args.feature_set else 0,
        "sequence_family": args.sequence_family,
        "sequence_residualization": args.sequence_residualization if "sequence_pca" in args.feature_set else "raw",
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "n_folds": args.n_folds,
        "ridge_reg": args.ridge_reg,
        "thresholds": thresholds,
        "best_fixed_delta_p": best_fixed_dp,
        "oracle_delta_p": results["oracle_mean_delta_p"],
        "best_threshold": results["best_threshold"],
        "best_threshold_routed_delta_p": results["best_threshold_routed_mean_delta_p"],
        "routing_efficiency": results["best_threshold_routed_mean_delta_p"] / results["oracle_mean_delta_p"],
        "threshold_summary": results["threshold_summary"],
        "per_profile_regression": results["per_profile_regression"],
        "fold_results": results["fold_results"],
    }

    output_path = output_dir / "train_probes_results.json"
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\nTraining final probes on full dataset...")
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_std = (X - mu) / std

    weights = np.zeros((X.shape[1], len(args.profiles)), dtype=np.float64)
    bias = np.zeros(len(args.profiles), dtype=np.float64)
    for j, profile_name in enumerate(args.profiles):
        w, b = fit_ridge(X_std, delta_matrix[:, j], reg=args.ridge_reg)
        weights[:, j] = w
        bias[j] = b

    artifacts = {
        "model_key": args.model_key,
        "profiles": args.profiles,
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "n_pca": args.n_pca,
        "sequence_n_pca": args.sequence_n_pca if "sequence_pca" in args.feature_set else 0,
        "sequence_family": args.sequence_family,
        "sequence_residualization": args.sequence_residualization if "sequence_pca" in args.feature_set else "raw",
        "intervention_mode": intervention_mode.value,
        "probe_type": "ridge",
        "ridge_reg": args.ridge_reg,
        "decision_threshold": results["best_threshold"],
        "thresholds_evaluated": thresholds,
        "standardization_mean": mu.tolist(),
        "standardization_std": std.tolist(),
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "profile_specs": load_profile_specs(args.profiles),
    }
    if pca_components is not None:
        artifacts["pca_components"] = pca_components.tolist()
        artifacts["pca_mean"] = pca_mean.tolist()
    if sequence_pca_components is not None:
        artifacts["sequence_pca_components"] = sequence_pca_components.tolist()
        artifacts["sequence_pca_mean"] = sequence_pca_mean.tolist()
    if sequence_residual_beta is not None:
        artifacts["sequence_residual_beta"] = sequence_residual_beta.tolist()

    model_path = output_dir / "probe_router_model.json"
    with open(model_path, "w") as f:
        json.dump(artifacts, f)
    print(f"Probe artifacts saved to {model_path}")


if __name__ == "__main__":
    main()
