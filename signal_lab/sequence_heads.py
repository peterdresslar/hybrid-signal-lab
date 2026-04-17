"""
Pipeline to extract global mean attention entropy from verbose.jsonl, residualize PCA coordinates,
and render companion 3D plots for structural task isolation verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

from signal_lab.paths import DATA_DIR_ENV_VAR, configure_data_dir, resolve_input_path
from signal_lab.sequence_plot_3d import plot_pca_3d


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def extract_global_attn_heat(verbose_path: Path) -> dict[str, float]:
    """Extracts the global scalar mean of the final-token attention entropy heatmap for each prompt."""
    logging.info(f"Extracting global attention heat from {verbose_path}...")
    heat_lookup = {}
    with verbose_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_id = record["prompt_id"]
            matrix = record.get("attn_entropy_per_head_final")
            
            if not matrix:
                raise ValueError(f"Prompt {prompt_id} is missing 'attn_entropy_per_head_final' matrix.")
            
            # matrix is typically List[List[float]] (layers x heads)
            all_vals = []
            for layer in matrix:
                all_vals.extend(layer)
            
            if not all_vals:
                raise ValueError(f"Prompt {prompt_id} yielded empty attention entropy.")
            
            heat_lookup[prompt_id] = sum(all_vals) / len(all_vals)
            
            if idx % 200 == 0:
                logging.info(f"  Processed {idx} prompts...")
                
    logging.info(f"Extracted {len(heat_lookup)} heat scalars.")
    return heat_lookup


def residualize_scalar_series(values: list[float], indicator: list[float]) -> list[float]:
    """Simple single-variable OLS residualization."""
    n = len(values)
    if n == 0 or len(indicator) != n:
        return values
    mean_x = sum(indicator) / n
    mean_y = sum(values) / n
    ss_xx = sum((x - mean_x) ** 2 for x in indicator)
    if ss_xx <= 1e-12:
        return [y - mean_y for y in values]
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(indicator, values))
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    return [y - (intercept + slope * x) for x, y in zip(indicator, values)]


def infer_model_label(analysis_dir: Path, verbose_path: Path) -> str:
    for candidate in (analysis_dir.name, verbose_path.parent.name):
        term = candidate.upper()
        if "OLMO" in term:
            return "OLMO"
        if "9B" in term:
            return "9B"
    return analysis_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Residualize global attention entropy from sequence PCA maps.")
    parser.add_argument("--sequence-analysis-dir", required=True, help="Directory with raw *_pca.csv outputs.")
    parser.add_argument("--verbose-jsonl", required=True, help="Path to the source verbose.jsonl for extracting heat maps.")
    parser.add_argument("--data-dir", default=None, help=f"Optional base directory override. Also supports {DATA_DIR_ENV_VAR}.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D elevation angle for companion plots.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D azimuth angle for companion plots.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    configure_data_dir(args.data_dir)

    analysis_dir = resolve_input_path(args.sequence_analysis_dir).expanduser().resolve()
    verbose_path = resolve_input_path(args.verbose_jsonl).expanduser().resolve()
    
    if not analysis_dir.is_dir():
        raise NotADirectoryError(f"Sequence analysis dir not found: {analysis_dir}")
    if not verbose_path.is_file():
        raise FileNotFoundError(f"Verbose jsonl not found: {verbose_path}")

    model_label = infer_model_label(analysis_dir, verbose_path)

    # 1. Parse global head entropies
    heat_lookup = extract_global_attn_heat(verbose_path)

    # 2. Iterate and residualize existing *_pca.csv mappings
    pca_files = sorted(analysis_dir.glob("*_pca.csv"))
    if not pca_files:
        logging.warning("No *_pca.csv files found to process.")
        return

    plot_dir = analysis_dir / "3d_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    for pca_file in pca_files:
        family_name = pca_file.name.removesuffix("_pca.csv")
        logging.info(f"Residualizing family: {family_name}")
        
        rows = load_csv_rows(pca_file)
        
        # We assume length_resid_pcX already exist in standard pipeline
        val_x = []
        val_y = []
        val_z = []
        heat_indicators = []
        
        valid_rows = []
        for r in rows:
            prompt_id = r["prompt_id"]
            if prompt_id not in heat_lookup:
                logging.warning(f"  Missing heat scalar for {prompt_id}, dropping row from plotting pipeline...")
                continue
            
            valid_rows.append(r)
            
            val_x.append(float(r["length_resid_pc1"]))
            val_y.append(float(r["length_resid_pc2"]))
            val_z.append(float(r["length_resid_pc3"]))
            heat_indicators.append(heat_lookup[prompt_id])
            
        heat_resid_x = residualize_scalar_series(val_x, heat_indicators)
        heat_resid_y = residualize_scalar_series(val_y, heat_indicators)
        heat_resid_z = residualize_scalar_series(val_z, heat_indicators)
        
        for idx, r in enumerate(valid_rows):
            r["attn_resid_pc1"] = f"{heat_resid_x[idx]:.12f}"
            r["attn_resid_pc2"] = f"{heat_resid_y[idx]:.12f}"
            r["attn_resid_pc3"] = f"{heat_resid_z[idx]:.12f}"

        # Rewrite CSV immediately to save updated namespace
        write_csv(pca_file, valid_rows)
        
        # Generate new companion 3D plots natively
        out_path = plot_dir / f"{family_name}_3d_attn_resid.png"
        plot_pca_3d(
            rows=valid_rows,
            family_name=family_name,
            model_label=model_label,
            output_path=out_path,
            mode="attn_resid",
            elev=args.elev,
            azim=args.azim,
        )
        
        logging.info(f"  Saved companion plot: {out_path}")

    logging.info("Pipeline sweep completed.")


if __name__ == "__main__":
    main()
