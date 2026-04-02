#!/usr/bin/env python3
"""Binary gate classifier for gain profile interventions.

Trains a logistic regression classifier to predict whether applying a specific
gain profile will improve target-token probability for a given prompt. Evaluates
under four conditions: baseline, always-on, classifier-gated, and oracle-gated.

Usage:

    uv run -m battery.src.classify_gate \
      --sweep-data ~/workspace/data/sl-runs/b4/35B/main.jsonl \
      --annotation battery/data/battery_4/annotation_manifest.json \
      --profile ramp_down \
      --output ~/workspace/data/classifier/gate_results.json

Requires scikit-learn: pip install scikit-learn
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a binary gate classifier for gain interventions."
    )
    parser.add_argument(
        "--sweep-data",
        type=str,
        required=True,
        help="Path to main.jsonl from a sweep run.",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        required=True,
        help="Path to annotation_manifest.json with train/test splits.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="ramp_down",
        help="Gain profile to train the gate for (default: ramp_down).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Delta target_prob threshold for positive label (default: 0.0, any improvement).",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["A", "B", "C"],
        default="A",
        help="Feature set: A=baseline, B=A+profile_stats, C=B+entropy features.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write results JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-type results.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sweep(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load sweep JSONL into {prompt_id: {profile: row}} structure."""
    data: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            data[row["prompt_id"]][row["g_profile"]] = row
    return dict(data)


def load_annotation(path: Path) -> dict[str, dict[str, Any]]:
    """Load annotation manifest, return {prompt_id: item} for annotated items."""
    manifest = json.loads(Path(path).read_text())
    items = {}
    for item in manifest["items"]:
        items[item["id"]] = item
    return items


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Type encoding (one-hot)
ALL_TYPES = [
    "algorithmic", "code_comprehension", "cultural_memorized",
    "domain_knowledge", "factual_recall", "factual_retrieval",
    "long_range_retrieval", "reasoning_numerical", "reasoning_tracking",
    "structural_copying", "syntactic_pattern",
]

# Tier encoding
ALL_TIERS = ["short", "brief", "med", "long", "extended"]


def extract_features_a(
    prompt_id: str,
    sweep_data: dict[str, dict[str, dict[str, Any]]],
    annotation: dict[str, dict[str, Any]],
) -> np.ndarray | None:
    """Feature Set A: baseline characteristics.

    Features:
    - type (one-hot, 11 dims)
    - tier (one-hot, 5 dims)
    - tokens_approx (1 dim)
    - baseline_target_prob (1 dim)
    - baseline_target_rank (1 dim, log-transformed)
    - baseline_entropy (1 dim)
    """
    if prompt_id not in sweep_data:
        return None
    baseline = sweep_data[prompt_id].get("baseline")
    if baseline is None:
        return None

    ann = annotation.get(prompt_id, {})

    # Type one-hot
    ptype = ann.get("type", "")
    type_vec = [1.0 if t == ptype else 0.0 for t in ALL_TYPES]

    # Tier one-hot
    tier = ann.get("tier", "")
    tier_vec = [1.0 if t == tier else 0.0 for t in ALL_TIERS]

    tokens = float(ann.get("tokens_approx", 0) or 0)
    base_prob = float(baseline.get("target_prob", 0))
    base_rank = float(baseline.get("target_rank", 1))
    base_entropy = float(baseline.get("final_entropy_bits", 0))

    features = (
        type_vec
        + tier_vec
        + [tokens, base_prob, np.log1p(base_rank), base_entropy]
    )
    return np.array(features, dtype=np.float64)


def extract_features_b(
    prompt_id: str,
    sweep_data: dict[str, dict[str, dict[str, Any]]],
    annotation: dict[str, dict[str, Any]],
) -> np.ndarray | None:
    """Feature Set B: A + profile response statistics.

    Additional features computed from the sweep data across all profiles:
    - mean delta_prob across all non-baseline profiles
    - std delta_prob
    - max delta_prob
    - fraction of profiles that helped
    """
    feat_a = extract_features_a(prompt_id, sweep_data, annotation)
    if feat_a is None:
        return None

    baseline = sweep_data[prompt_id].get("baseline")
    if baseline is None:
        return None
    base_prob = float(baseline.get("target_prob", 0))

    deltas = []
    for prof, row in sweep_data[prompt_id].items():
        if prof == "baseline":
            continue
        delta = float(row.get("target_prob", 0)) - base_prob
        deltas.append(delta)

    if not deltas:
        return None

    deltas_arr = np.array(deltas)
    profile_features = [
        float(np.mean(deltas_arr)),
        float(np.std(deltas_arr)),
        float(np.max(deltas_arr)),
        float(np.sum(deltas_arr > 0) / len(deltas_arr)),
    ]
    return np.concatenate([feat_a, np.array(profile_features)])


def extract_features_c(
    prompt_id: str,
    sweep_data: dict[str, dict[str, dict[str, Any]]],
    annotation: dict[str, dict[str, Any]],
) -> np.ndarray | None:
    """Feature Set C: B + entropy response features.

    Additional features:
    - mean entropy across all profiles
    - entropy range (max - min)
    - baseline KL divergence mean (from profiles that report kl_from_baseline)
    """
    feat_b = extract_features_b(prompt_id, sweep_data, annotation)
    if feat_b is None:
        return None

    entropies = []
    kl_values = []
    for prof, row in sweep_data[prompt_id].items():
        ent = row.get("final_entropy_bits")
        if ent is not None:
            entropies.append(float(ent))
        kl = row.get("kl_from_baseline")
        if kl is not None:
            kl_values.append(float(kl))

    if not entropies:
        return None

    ent_arr = np.array(entropies)
    entropy_features = [
        float(np.mean(ent_arr)),
        float(np.max(ent_arr) - np.min(ent_arr)),
        float(np.mean(kl_values)) if kl_values else 0.0,
    ]
    return np.concatenate([feat_b, np.array(entropy_features)])


FEATURE_EXTRACTORS = {
    "A": extract_features_a,
    "B": extract_features_b,
    "C": extract_features_c,
}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(
    sweep_data: dict[str, dict[str, dict[str, Any]]],
    annotation: dict[str, dict[str, Any]],
    profile: str,
    threshold: float,
    feature_set: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix and label vector for a given split.

    Returns (X, y, prompt_ids).
    """
    extractor = FEATURE_EXTRACTORS[feature_set]
    X_rows = []
    y_rows = []
    ids = []

    for prompt_id, ann in annotation.items():
        if ann.get("split") != split:
            continue

        features = extractor(prompt_id, sweep_data, annotation)
        if features is None:
            continue

        # Compute label: did the profile help?
        baseline = sweep_data.get(prompt_id, {}).get("baseline")
        intervention = sweep_data.get(prompt_id, {}).get(profile)
        if baseline is None or intervention is None:
            continue

        base_prob = float(baseline["target_prob"])
        interv_prob = float(intervention["target_prob"])
        delta = interv_prob - base_prob
        label = 1.0 if delta > threshold else 0.0

        X_rows.append(features)
        y_rows.append(label)
        ids.append(prompt_id)

    if not X_rows:
        return np.array([]), np.array([]), []

    return np.array(X_rows), np.array(y_rows), ids


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_conditions(
    sweep_data: dict[str, dict[str, dict[str, Any]]],
    annotation: dict[str, dict[str, Any]],
    profile: str,
    threshold: float,
    predictions: dict[str, float],
    test_ids: list[str],
) -> dict[str, Any]:
    """Evaluate four conditions on test set.

    Conditions:
    1. baseline: no intervention (delta = 0 for all)
    2. always_on: apply profile to every prompt
    3. classifier_gated: apply profile only when classifier predicts positive
    4. oracle_gated: apply profile only when it actually helps

    Returns mean delta_target_prob under each condition.
    """
    results = {
        "baseline": [],
        "always_on": [],
        "classifier_gated": [],
        "oracle_gated": [],
    }

    for pid in test_ids:
        baseline_row = sweep_data.get(pid, {}).get("baseline")
        intervention_row = sweep_data.get(pid, {}).get(profile)
        if baseline_row is None or intervention_row is None:
            continue

        base_prob = float(baseline_row["target_prob"])
        interv_prob = float(intervention_row["target_prob"])
        delta = interv_prob - base_prob
        truly_helps = delta > threshold

        # baseline: 0 delta
        results["baseline"].append(0.0)

        # always_on: always apply
        results["always_on"].append(delta)

        # classifier_gated: apply only if predicted positive
        pred = predictions.get(pid, 0.0)
        if pred >= 0.5:
            results["classifier_gated"].append(delta)
        else:
            results["classifier_gated"].append(0.0)

        # oracle: apply only when it truly helps
        if truly_helps:
            results["oracle_gated"].append(delta)
        else:
            results["oracle_gated"].append(0.0)

    summary = {}
    for condition, deltas in results.items():
        if deltas:
            arr = np.array(deltas)
            summary[condition] = {
                "mean_delta": float(np.mean(arr)),
                "median_delta": float(np.median(arr)),
                "sum_delta": float(np.sum(arr)),
                "n_applied": int(np.sum(np.array(deltas) != 0.0)),
                "n_total": len(deltas),
            }
        else:
            summary[condition] = {
                "mean_delta": 0.0,
                "median_delta": 0.0,
                "sum_delta": 0.0,
                "n_applied": 0,
                "n_total": 0,
            }

    return summary


def evaluate_per_type(
    sweep_data: dict[str, dict[str, dict[str, Any]]],
    annotation: dict[str, dict[str, Any]],
    profile: str,
    threshold: float,
    predictions: dict[str, float],
    test_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Per-type breakdown of evaluation conditions."""
    types: dict[str, list[str]] = defaultdict(list)
    for pid in test_ids:
        ann = annotation.get(pid, {})
        ptype = ann.get("type", "unknown")
        types[ptype].append(pid)

    per_type = {}
    for ptype, pids in sorted(types.items()):
        per_type[ptype] = evaluate_conditions(
            sweep_data, annotation, profile, threshold, predictions, pids
        )
    return per_type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Late import so the script can show --help without sklearn installed
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )
    except ImportError:
        print("scikit-learn is required: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    print(f"Loading sweep data from {args.sweep_data}...")
    sweep_data = load_sweep(Path(args.sweep_data))
    print(f"  {len(sweep_data)} prompts loaded")

    print(f"Loading annotation from {args.annotation}...")
    annotation = load_annotation(Path(args.annotation))
    print(f"  {len(annotation)} items loaded")

    # Check profile exists
    sample_pid = next(iter(sweep_data))
    if args.profile not in sweep_data[sample_pid]:
        available = sorted(sweep_data[sample_pid].keys())
        print(f"Profile '{args.profile}' not found. Available: {available}", file=sys.stderr)
        sys.exit(1)

    # Build datasets
    print(f"\nBuilding datasets (feature_set={args.feature_set}, profile={args.profile})...")
    X_train, y_train, train_ids = build_dataset(
        sweep_data, annotation, args.profile, args.threshold, args.feature_set, "train_prompt"
    )
    X_test, y_test, test_ids = build_dataset(
        sweep_data, annotation, args.profile, args.threshold, args.feature_set, "test_prompt"
    )

    print(f"  Train: {len(train_ids)} prompts ({int(y_train.sum())}/{len(y_train)} positive = {y_train.mean():.1%})")
    print(f"  Test:  {len(test_ids)} prompts ({int(y_test.sum())}/{len(y_test)} positive = {y_test.mean():.1%})")

    if len(train_ids) == 0 or len(test_ids) == 0:
        print("Insufficient data for training/testing.", file=sys.stderr)
        sys.exit(1)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    print("\nTraining logistic regression...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = None

    print(f"\nClassifier Performance:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    if auc is not None:
        print(f"  AUC:       {auc:.3f}")

    # Build predictions dict for evaluation
    predictions = {pid: float(p) for pid, p in zip(test_ids, y_prob)}

    # Four-condition evaluation
    print(f"\nFour-Condition Evaluation (profile={args.profile}):")
    conditions = evaluate_conditions(
        sweep_data, annotation, args.profile, args.threshold, predictions, test_ids
    )
    for cond, stats in conditions.items():
        applied = stats["n_applied"]
        total = stats["n_total"]
        print(f"  {cond:20s}: mean_delta={stats['mean_delta']:+.6f}  "
              f"applied={applied}/{total}")

    # Per-type breakdown
    if args.verbose:
        print(f"\nPer-Type Breakdown:")
        per_type = evaluate_per_type(
            sweep_data, annotation, args.profile, args.threshold, predictions, test_ids
        )
        for ptype, type_conditions in per_type.items():
            print(f"\n  {ptype}:")
            for cond, stats in type_conditions.items():
                print(f"    {cond:20s}: mean_delta={stats['mean_delta']:+.6f}  "
                      f"applied={stats['n_applied']}/{stats['n_total']}")

    # Feature importance
    feature_names = (
        [f"type_{t}" for t in ALL_TYPES]
        + [f"tier_{t}" for t in ALL_TIERS]
        + ["tokens_approx", "baseline_prob", "log_baseline_rank", "baseline_entropy"]
    )
    if args.feature_set in ("B", "C"):
        feature_names += ["mean_delta_all", "std_delta_all", "max_delta_all", "frac_helped"]
    if args.feature_set == "C":
        feature_names += ["mean_entropy_all", "entropy_range", "mean_kl"]

    coefs = clf.coef_[0]
    importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 10 Feature Importances (by |coefficient|):")
    for name, coef in importance[:10]:
        print(f"  {name:30s}: {coef:+.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "profile": args.profile,
            "feature_set": args.feature_set,
            "threshold": args.threshold,
            "train_n": len(train_ids),
            "test_n": len(test_ids),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "classifier_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
            },
            "conditions": conditions,
            "feature_importance": [
                {"feature": name, "coefficient": float(coef)}
                for name, coef in importance
            ],
        }

        if args.verbose:
            result["per_type"] = evaluate_per_type(
                sweep_data, annotation, args.profile, args.threshold, predictions, test_ids
            )

        output_path.write_text(json.dumps(result, indent=2))
        print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
