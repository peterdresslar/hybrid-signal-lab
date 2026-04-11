"""
select_profiles.py — Combinatorial search for optimal 4-profile sets.

For each model, find the 4 gain profiles that maximize prompt-level utility
under either:

- `oracle`: mean oracle-routed delta_p over the selected set
- `separable`: oracle utility plus tie-break terms favoring broad class usage
  and lower within-set response correlation

"Oracle-routed" means each prompt is assigned whichever of the 4 profiles
(or baseline/off) gives the best delta_target_prob.

The candidate pool is discovered directly from valid non-baseline rows in
`analysis_joined_long.csv` from the prior probing sweep associated with the
model under study. In the balanced hybrid probing runs used for the current
router selections, this yields about 86–87 usable profiles per model,
depending on whether any model/profile rows were invalid or missing in the
analysis output. That corresponds to about 2.1–2.2M 4-profile combinations,
which remains feasible on a single core in a few minutes.

Usage:
    python -m router.experiments.select_profiles \
        --model-key 9B \
        --data-dir data/022-balanced-attention-hybrid

    python -m router.experiments.select_profiles \
        --model-key OLMO \
        --data-dir data/022-balanced-block-hybrid

Optional flags:
    --top-n              Number of top sets to report (default: 20)
    --min-coverage       Minimum fraction of prompts where at least one
                         profile in the set beats baseline (default: 0.0)
    --objective          Search objective: "oracle" or "separable"
                         (default: oracle)
    --min-profile-wins   Minimum number of oracle-assigned prompts for
                         every selected profile (default: 0)
    --max-mean-abs-corr  Optional upper bound on mean absolute pairwise
                         profile correlation within the set
    --max-constants      Optional upper bound on number of constant_*
                         profiles allowed in a set
    --output-dir         Directory for results (default: <data-dir>/<model-key>/router/)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np


def load_delta_matrix(analysis_dir: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Load the prompt × profile delta_target_prob matrix from analysis_joined_long.csv.

    Returns:
        prompt_ids: list of unique prompt IDs (length P)
        profiles:   list of non-baseline profile names (length N)
        matrix:     P × N array of delta_target_prob values
    """
    joined_path = analysis_dir / "analysis_joined_long.csv"
    if not joined_path.exists():
        raise FileNotFoundError(f"Missing {joined_path}")

    # First pass: collect all (prompt_id, g_profile, delta_target_prob) triples
    triples: list[tuple[str, str, float]] = []
    with open(joined_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g_profile = row["g_profile"]
            if g_profile == "baseline":
                continue
            try:
                delta = float(row["delta_target_prob"])
            except (ValueError, TypeError):
                continue
            triples.append((row["prompt_id"], g_profile, delta))

    # Build index maps
    prompt_set: dict[str, int] = {}
    profile_set: dict[str, int] = {}
    for pid, gp, _ in triples:
        if pid not in prompt_set:
            prompt_set[pid] = len(prompt_set)
        if gp not in profile_set:
            profile_set[gp] = len(profile_set)

    prompt_ids = sorted(prompt_set, key=prompt_set.get)  # type: ignore[arg-type]
    profiles = sorted(profile_set, key=profile_set.get)  # type: ignore[arg-type]

    # Rebuild clean index maps from sorted lists
    pid_idx = {pid: i for i, pid in enumerate(prompt_ids)}
    prof_idx = {gp: j for j, gp in enumerate(profiles)}

    # Fill matrix
    matrix = np.zeros((len(prompt_ids), len(profiles)), dtype=np.float64)
    for pid, gp, delta in triples:
        matrix[pid_idx[pid], prof_idx[gp]] = delta

    return prompt_ids, profiles, matrix


def _normalized_entropy(counts: np.ndarray) -> float:
    """Return entropy normalized to [0, 1]."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    if len(probs) <= 1:
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy / float(np.log(len(counts)))


def score_profile_set(
    matrix: np.ndarray,
    indices: tuple[int, ...],
    pairwise_corr: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Score a candidate set of profile column indices.

    For each prompt, the oracle picks max(0, max(delta_p across selected profiles)).
    The zero floor represents the "off" option (baseline).

    Returns:
        Dictionary with oracle utility plus separability diagnostics.
    """
    sub = matrix[:, list(indices)]                  # P × 4
    per_prompt_best = np.maximum(sub.max(axis=1), 0.0)  # floor at 0 (off)
    mean_routed = float(per_prompt_best.mean())

    # Best fixed: which single column has highest overall mean (clipped at 0)?
    col_means = np.maximum(sub, 0.0).mean(axis=0)
    mean_best_fixed = float(col_means.max())

    best_vals = sub.max(axis=1)
    coverage = int((best_vals > 0).sum())
    best_idx = sub.argmax(axis=1)
    active_idx = best_idx[best_vals > 0]
    winner_counts = np.bincount(active_idx, minlength=len(indices))
    min_profile_wins = int(winner_counts.min()) if len(winner_counts) else 0
    winner_entropy = _normalized_entropy(winner_counts.astype(np.float64))

    mean_abs_corr = 0.0
    if pairwise_corr is not None and len(indices) > 1:
        corr_vals = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                corr_vals.append(abs(float(pairwise_corr[indices[i], indices[j]])))
        mean_abs_corr = float(np.mean(corr_vals)) if corr_vals else 0.0

    # Separable score favors strong oracle utility, broad usage across selected
    # profiles, and lower within-set response correlation. The scale keeps the
    # oracle term dominant while still breaking ties toward more learnable sets.
    separable_score = (
        mean_routed
        + 0.020 * winner_entropy
        + 0.020 * (1.0 - mean_abs_corr)
        + 0.010 * (min_profile_wins / max(matrix.shape[0], 1))
    )

    return {
        "mean_oracle_routed_delta_p": mean_routed,
        "mean_best_fixed_delta_p": mean_best_fixed,
        "routing_advantage": mean_routed - mean_best_fixed,
        "coverage": coverage,
        "coverage_pct": coverage / matrix.shape[0] * 100.0,
        "winner_entropy": winner_entropy,
        "min_profile_wins": min_profile_wins,
        "mean_abs_pairwise_corr": mean_abs_corr,
        "separable_score": separable_score,
    }


def compute_pairwise_corr(matrix: np.ndarray) -> np.ndarray:
    """Precompute profile-response correlations across prompts."""
    corr = np.corrcoef(matrix, rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def search(
    matrix: np.ndarray,
    profiles: list[str],
    top_n: int = 20,
    min_coverage: float = 0.0,
    objective: str = "oracle",
    min_profile_wins: int = 0,
    max_mean_abs_corr: float | None = None,
    max_constants: int | None = None,
) -> list[dict]:
    """Enumerate all C(N, 4) profile sets and return the top scoring ones."""
    n_profiles = matrix.shape[1]
    n_prompts = matrix.shape[0]
    min_cov_count = int(min_coverage * n_prompts)
    pairwise_corr = compute_pairwise_corr(matrix)

    total_combos = 1
    for i in range(4):
        total_combos = total_combos * (n_profiles - i) // (i + 1)

    print(f"Searching {total_combos:,} combinations of 4 from {n_profiles} profiles "
          f"across {n_prompts} prompts...")

    results: list[tuple[float, float, dict[str, float | int], tuple[int, ...]]] = []
    report_interval = max(total_combos // 20, 1)
    t0 = time.time()

    for count, combo in enumerate(combinations(range(n_profiles), 4)):
        if count % report_interval == 0 and count > 0:
            elapsed = time.time() - t0
            rate = count / elapsed
            remaining = (total_combos - count) / rate
            print(f"  {count:>10,} / {total_combos:,}  "
                  f"({count/total_combos*100:.0f}%)  "
                  f"{remaining:.0f}s remaining")

        if max_constants is not None:
            constant_count = sum(1 for idx in combo if profiles[idx].startswith("constant_"))
            if constant_count > max_constants:
                continue

        metrics = score_profile_set(matrix, combo, pairwise_corr=pairwise_corr)
        coverage = int(metrics["coverage"])
        if coverage < min_cov_count:
            continue
        if int(metrics["min_profile_wins"]) < min_profile_wins:
            continue
        if max_mean_abs_corr is not None and float(metrics["mean_abs_pairwise_corr"]) > max_mean_abs_corr:
            continue

        primary = (
            float(metrics["mean_oracle_routed_delta_p"])
            if objective == "oracle"
            else float(metrics["separable_score"])
        )
        secondary = float(metrics["mean_oracle_routed_delta_p"])
        results.append((primary, secondary, metrics, combo))

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({total_combos / elapsed:,.0f} combos/sec)")

    # Sort by chosen objective, then oracle mean as a stable tiebreaker.
    results.sort(key=lambda x: (-x[0], -x[1]))

    top_results = []
    for rank, (primary, secondary, metrics, combo) in enumerate(results[:top_n], 1):
        profile_names = [profiles[i] for i in combo]
        top_results.append({
            "rank": rank,
            "profiles": profile_names,
            "objective": objective,
            "objective_score": round(primary, 6),
            "mean_oracle_routed_delta_p": round(float(metrics["mean_oracle_routed_delta_p"]), 6),
            "mean_best_fixed_delta_p": round(float(metrics["mean_best_fixed_delta_p"]), 6),
            "routing_advantage": round(float(metrics["routing_advantage"]), 6),
            "coverage": int(metrics["coverage"]),
            "coverage_pct": round(float(metrics["coverage_pct"]), 1),
            "winner_entropy": round(float(metrics["winner_entropy"]), 6),
            "min_profile_wins": int(metrics["min_profile_wins"]),
            "mean_abs_pairwise_corr": round(float(metrics["mean_abs_pairwise_corr"]), 6),
        })

    return top_results


def compute_baselines(matrix: np.ndarray, profiles: list[str]) -> dict:
    """Compute reference points for comparison."""
    n_prompts = matrix.shape[0]

    # Oracle over ALL profiles (upper bound)
    all_oracle = float(np.maximum(matrix.max(axis=1), 0.0).mean())

    # Best single fixed profile
    profile_means = matrix.mean(axis=0)
    best_fixed_idx = int(profile_means.argmax())
    best_fixed_mean = float(profile_means[best_fixed_idx])

    # Best single fixed profile (with off option = clipped at 0)
    clipped_means = np.maximum(matrix, 0.0).mean(axis=0)
    best_fixed_clipped_idx = int(clipped_means.argmax())
    best_fixed_clipped_mean = float(clipped_means[best_fixed_clipped_idx])

    return {
        "oracle_all_78_profiles": round(all_oracle, 6),
        "best_fixed_profile": profiles[best_fixed_idx],
        "best_fixed_mean_delta_p": round(best_fixed_mean, 6),
        "best_fixed_profile_clipped": profiles[best_fixed_clipped_idx],
        "best_fixed_clipped_mean_delta_p": round(best_fixed_clipped_mean, 6),
        "n_prompts": n_prompts,
        "n_profiles": matrix.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal 4-profile sets for gain intervention routing."
    )
    parser.add_argument(
        "--model-key", required=True,
        help="Model subdirectory name (e.g. 9B, OLMO)"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to intervention_modes run directory (e.g. data/intervention_modes/b4_021_attn_contr)"
    )
    parser.add_argument("--top-n", type=int, default=20, help="Number of top sets to report")
    parser.add_argument("--min-coverage", type=float, default=0.0,
                        help="Minimum coverage fraction (0.0–1.0)")
    parser.add_argument("--objective", choices=["oracle", "separable"], default="oracle",
                        help="Ranking objective for candidate sets")
    parser.add_argument("--min-profile-wins", type=int, default=0,
                        help="Minimum oracle winner count for every selected profile")
    parser.add_argument("--max-mean-abs-corr", type=float, default=None,
                        help="Optional upper bound on mean absolute pairwise correlation within a set")
    parser.add_argument("--max-constants", type=int, default=None,
                        help="Optional upper bound on number of constant_* profiles in a set")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = data_dir / args.model_key / "analysis"
    if not analysis_dir.exists():
        print(f"Error: analysis directory not found: {analysis_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else data_dir / args.model_key / "router"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for {args.model_key}...")
    prompt_ids, profiles, matrix = load_delta_matrix(analysis_dir)
    print(f"  {len(prompt_ids)} prompts × {len(profiles)} profiles")

    baselines = compute_baselines(matrix, profiles)
    print(f"\nReference points:")
    print(f"  Oracle (all {baselines['n_profiles']} profiles): "
          f"{baselines['oracle_all_78_profiles']:.4f}")
    print(f"  Best fixed profile: {baselines['best_fixed_profile']} "
          f"({baselines['best_fixed_mean_delta_p']:.4f})")
    print(f"  Best fixed (clipped): {baselines['best_fixed_profile_clipped']} "
          f"({baselines['best_fixed_clipped_mean_delta_p']:.4f})")
    print()

    top_results = search(
        matrix,
        profiles,
        top_n=args.top_n,
        min_coverage=args.min_coverage,
        objective=args.objective,
        min_profile_wins=args.min_profile_wins,
        max_mean_abs_corr=args.max_mean_abs_corr,
        max_constants=args.max_constants,
    )

    # Display results
    print(f"\nTop {len(top_results)} profile sets by {args.objective} objective:")
    print(f"{'rank':>4}  {'score':>8}  {'routed':>8}  {'adv':>7}  {'cov%':>5}  {'ent':>5}  {'minw':>5}  {'corr':>5}  profiles")
    print("-" * 120)
    for r in top_results:
        print(f"{r['rank']:>4}  {r['objective_score']:>8.4f}  {r['mean_oracle_routed_delta_p']:>8.4f}  "
              f"{r['routing_advantage']:>7.4f}  {r['coverage_pct']:>5.1f}  {r['winner_entropy']:>5.2f}  "
              f"{r['min_profile_wins']:>5}  {r['mean_abs_pairwise_corr']:>5.2f}  {', '.join(r['profiles'])}")

    # Save results
    output = {
        "model_key": args.model_key,
        "data_dir": str(data_dir),
        "objective": args.objective,
        "min_profile_wins": args.min_profile_wins,
        "max_mean_abs_corr": args.max_mean_abs_corr,
        "max_constants": args.max_constants,
        "baselines": baselines,
        "top_sets": top_results,
    }
    output_path = output_dir / "profile_selection.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save a concise human-readable report
    report_path = output_dir / "profile_selection_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Profile Selection: {args.model_key}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Data: {data_dir}\n")
        f.write(f"Prompts: {baselines['n_prompts']}\n")
        f.write(f"Candidate profiles: {baselines['n_profiles']}\n\n")

        f.write(f"Reference points:\n")
        f.write(f"  Baseline (no intervention):     0.0000\n")
        f.write(f"  Best fixed profile:             "
                f"{baselines['best_fixed_clipped_mean_delta_p']:.4f}  "
                f"({baselines['best_fixed_profile_clipped']})\n")
        f.write(f"  Oracle (all profiles):          "
                f"{baselines['oracle_all_78_profiles']:.4f}\n\n")

        f.write(f"Objective: {args.objective}\n")
        f.write(f"Min coverage: {args.min_coverage:.2f}\n")
        f.write(f"Min profile wins: {args.min_profile_wins}\n")
        f.write(f"Max mean abs corr: {args.max_mean_abs_corr}\n")
        f.write(f"Max constants: {args.max_constants}\n\n")

        f.write(f"Top {len(top_results)} profile sets (4 profiles + off):\n\n")
        for r in top_results:
            f.write(f"  #{r['rank']:>2}  score={r['objective_score']:.4f}  "
                    f"routed={r['mean_oracle_routed_delta_p']:.4f}  "
                    f"advantage={r['routing_advantage']:.4f}  "
                    f"coverage={r['coverage_pct']:.0f}%  "
                    f"entropy={r['winner_entropy']:.2f}  "
                    f"minwins={r['min_profile_wins']}  "
                    f"corr={r['mean_abs_pairwise_corr']:.2f}\n")
            for p in r["profiles"]:
                f.write(f"        {p}\n")
            f.write("\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
