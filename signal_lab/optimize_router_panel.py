"""
Exhaustively search for Pareto-optimal router panels balancing performance wins and geometric spatial isolation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def _score_chunk(chunk: list[tuple], scores_mat: np.ndarray, labels_base: np.ndarray, metric_coords: np.ndarray) -> list[tuple[float, int, tuple[int, ...]]]:
    """Worker function for multiprocessing."""
    results = []
    for combo in chunk:
        combo_arr = np.array(combo)
        # scores_mat is [n_prompts, n_filtered_profiles]. Slice to combo.
        sub_scores = scores_mat[:, combo_arr]
        
        # Best profile in combo per prompt
        best_idx_local = np.argmax(sub_scores, axis=1)
        best_vals = np.take_along_axis(sub_scores, best_idx_local[:, None], axis=1).squeeze(1)
        
        # If best isn't > 0, assign baseline (0). Else assign the actual profile index.
        wins_mask = best_vals > 0.0
        coverage = int(np.sum(wins_mask))
        
        assigned_labels = np.zeros(scores_mat.shape[0], dtype=np.int32)
        assigned_labels[wins_mask] = combo_arr[best_idx_local[wins_mask]] + 1 # +1 to avoid collision with baseline=0
        
        # If there's only 1 label assigned globally (e.g. baseline or 1 profile entirely dominates),
        # silhouette_score throws a ValueError.
        unique_labels = np.unique(assigned_labels)
        if len(unique_labels) < 2:
            results.append((-1.0, coverage, combo))
            continue
            
        sil = float(silhouette_score(metric_coords, assigned_labels))
        results.append((sil, coverage, combo))
        
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-analysis-dir", required=True)
    parser.add_argument("--joined-long-csv", required=True)
    parser.add_argument("--oracle-export", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--min-wins", type=int, default=10)
    parser.add_argument("--k-sizes", type=int, nargs="+", default=[3, 4])
    args = parser.parse_args()

    analysis_dir = Path(args.sequence_analysis_dir).expanduser().resolve()
    pca_csv = analysis_dir / "embedding_last_token_pca.csv"
    joined_csv = Path(args.joined_long_csv).expanduser().resolve()
    oracle_csv = Path(args.oracle_export).expanduser().resolve()

    # 1. Identify robust profiles from oracle sweeps
    sweep_df = pd.read_csv(oracle_csv)
    sub_sweep = sweep_df[(sweep_df.model_label == args.model_label) & 
                         (sweep_df.winner_scope == "full_library") & 
                         (sweep_df.winner_objective == "delta_target_prob_max")]
    
    wins = sub_sweep[sub_sweep.winner_profile != "baseline"].winner_profile.value_counts()
    robust_profiles = wins[wins >= args.min_wins].index.tolist()
    print(f"Discovered {len(robust_profiles)} robust profiles (>= {args.min_wins} wins).")
    
    # 2. Extract Cartesian Coordinates
    pca_df = pd.read_csv(pca_csv)
    prompt_ids = pca_df["prompt_id"].tolist()
    prompt_id_to_idx = {pid: i for i, pid in enumerate(prompt_ids)}
    n_prompts = len(prompt_ids)
    
    # [n_prompts, 3] layout for geometry isolation
    metric_coords = pca_df[["attn_resid_pc1", "attn_resid_pc2", "attn_resid_pc3"]].values

    # 3. Build Score Matrix [n_prompts, n_robust_profiles]
    joined_df = pd.read_csv(joined_csv)
    # Fast filtering to relevant rows
    joined_df = joined_df[joined_df.g_profile.isin(robust_profiles)]
    joined_df = joined_df[joined_df.prompt_id.isin(prompt_id_to_idx)]
    
    # We map back into massive array
    profile_to_idx = {p: i for i, p in enumerate(robust_profiles)}
    scores_mat = np.zeros((n_prompts, len(robust_profiles)), dtype=np.float32)
    
    for _, row in joined_df.iterrows():
        p_idx = prompt_id_to_idx[row["prompt_id"]]
        prof_idx = profile_to_idx[row["g_profile"]]
        val = row["delta_target_prob"]
        if pd.isna(val):
            val = 0.0
        scores_mat[p_idx, prof_idx] = val

    labels_base = np.zeros(n_prompts, dtype=np.int32)
    
    all_combos = []
    for k in args.k_sizes:
        import math
        cnt = math.comb(len(robust_profiles), k)
        print(f"Generating {cnt} combinations for K={k}...")
        all_combos.extend(list(itertools.combinations(range(len(robust_profiles)), k)))
        
    print(f"Total panels to evaluate: {len(all_combos)}")
    
    # Split into chunks for multiprocessing
    num_workers = max(1, cpu_count() - 2)
    chunk_size = max(1, len(all_combos) // (num_workers * 4))
    chunks = [all_combos[i:i + chunk_size] for i in range(0, len(all_combos), chunk_size)]
    
    start_time = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_score_chunk, chunk, scores_mat, labels_base, metric_coords) for chunk in chunks]
        for f in futures:
            results.extend(f.result())
            
    print(f"Searched {len(results)} configurations in {time.time() - start_time:.1f} seconds.")
    
    # Compute Pareto Frontier (Max Coverage vs Max Silhouette)
    # Sort primarily by Coverage (wins) descending, then Silhouette
    results.sort(key=lambda x: (x[1], x[0]), reverse=True)
    
    pareto_front = []
    max_sil_so_far = -1.0
    
    for sil, cov, combo in results:
        if sil > max_sil_so_far:
            pareto_front.append((sil, cov, combo))
            max_sil_so_far = sil

    # The frontier comes out with Highest-Coverage, Lowest-Sil first. 
    # Let's print out the landscape from High Coverage down to High Sil
    print()
    print("================== PARETO FRONTIER ==================")
    best_all_rounder = None
    best_all_score = -1
    
    for i, (sil, cov, combo) in enumerate(pareto_front):
        profile_names = [robust_profiles[idx] for idx in combo]
        print(f"Win Coverage: {cov:>4} | Silhouette: {sil:>6.4f} | Profiles: {profile_names}")
        
        # A simple blended metric to auto-pick a great center-of-frontier candidate for plotting
        # Silhouette varies ~0.2 to 0.5. Coverage varies ~400 to 700.
        # We normalize loosely based on observation:
        blend = (cov / 1070.0) + (sil * 2.0)
        if blend > best_all_score:
            best_all_score = blend
            best_all_rounder = profile_names

    if best_all_rounder:
        print(f"\nRecommended optimal blended panel chosen for rendering: {best_all_rounder}")
        
        # Output choice script to stdout so we can pipe/read it
        import json
        with open("best_panel_args.json", "w") as f:
            json.dump(best_all_rounder, f)


if __name__ == "__main__":
    main()
