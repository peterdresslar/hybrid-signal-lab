"""Join sequence PCA bundles with sweep winner summaries for prompt-level inspection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from docs.figurelib.common import configure_matplotlib
from router.profiles import get_profile_names
from signal_lab.paths import (
    DATA_DIR_ENV_VAR,
    configure_data_dir,
    resolve_input_path,
)
from signal_lab.signal_lab import resolve_prompt_collection

DEFAULT_PREFIX = "sequence_interpret"
DEFAULT_BATTERY = "battery/data/battery_4"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_dir(path_str: str, *, required_files: list[str]) -> Path:
    path = resolve_input_path(path_str).expanduser().resolve()
    missing = [name for name in required_files if not (path / name).is_file()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Expected files in {path}: {missing_str}")
    return path


def to_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def to_int(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def build_battery_lookup(prompt_battery: str) -> dict[str, dict[str, Any]]:
    prompts = resolve_prompt_collection(prompt_battery)
    lookup: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        lookup[prompt.id] = {
            "prompt": prompt.prompt_text,
            "target": prompt.target or "",
            "battery_type": prompt.type or "",
            "battery_source": prompt.source or "",
            "battery_tokens_approx": prompt.tokens_approx,
        }
    return lookup


def build_prompt_winner_lookup(prompt_winners_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in prompt_winners_rows:
        if row.get("rep") not in {None, "", "1"}:
            continue
        prompt_id = row["prompt_id"]
        lookup[prompt_id] = {
            "winner_best_g_profile": row.get("best_g_profile", ""),
            "winner_best_g_family": row.get("best_g_family", ""),
            "winner_best_delta_target_prob": row.get("best_delta_target_prob", ""),
            "winner_best_delta_target_rank": row.get("best_delta_target_rank", ""),
            "winner_worst_g_profile": row.get("worst_g_profile", ""),
            "winner_worst_g_family": row.get("worst_g_family", ""),
            "winner_worst_delta_target_prob": row.get("worst_delta_target_prob", ""),
            "winner_worst_delta_target_rank": row.get("worst_delta_target_rank", ""),
        }
    return lookup


def build_joined_long_lookup(joined_long_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in joined_long_rows:
        if row.get("rep") not in {None, "", "1"}:
            continue
        grouped[row["prompt_id"]].append(row)

    lookup: dict[str, dict[str, Any]] = {}
    for prompt_id, rows in grouped.items():
        baseline_row = next((row for row in rows if row.get("g_profile") == "baseline"), None)
        intervention_rows = [row for row in rows if row.get("g_profile") != "baseline"]
        sorted_by_delta = sorted(
            intervention_rows,
            key=lambda row: (to_float(row.get("delta_target_prob")) or float("-inf")),
            reverse=True,
        )
        best_row = sorted_by_delta[0] if sorted_by_delta else None
        second_row = sorted_by_delta[1] if len(sorted_by_delta) > 1 else None
        top3 = sorted_by_delta[:3]

        payload: dict[str, Any] = {}
        if baseline_row is not None:
            payload.update(
                {
                    "baseline_target_prob": baseline_row.get("baseline_target_prob", ""),
                    "baseline_target_rank": baseline_row.get("baseline_target_rank", ""),
                    "baseline_target_geo_mean_prob": baseline_row.get("baseline_target_geo_mean_prob", ""),
                    "baseline_final_entropy_bits": baseline_row.get("baseline_final_entropy_bits", ""),
                    "baseline_mean_entropy_bits": baseline_row.get("baseline_mean_entropy_bits", ""),
                    "baseline_attn_entropy_mean": baseline_row.get("baseline_attn_entropy_mean", ""),
                    "intervention_strategy": baseline_row.get("intervention_strategy", ""),
                    "cartridge": baseline_row.get("cartridge", ""),
                    "model": baseline_row.get("model", ""),
                    "tier": baseline_row.get("tier", ""),
                    "type": baseline_row.get("type", ""),
                    "source": baseline_row.get("source", ""),
                    "prompt": baseline_row.get("prompt", ""),
                    "target": baseline_row.get("target", ""),
                }
            )
        if best_row is not None:
            best_delta = to_float(best_row.get("delta_target_prob")) or 0.0
            second_delta = to_float(second_row.get("delta_target_prob")) if second_row is not None else None
            margin = "" if second_delta is None else round(best_delta - second_delta, 6)
            payload.update(
                {
                    "oracle_best_g_profile": best_row.get("g_profile", ""),
                    "oracle_best_g_family": best_row.get("g_family", ""),
                    "oracle_best_delta_target_prob": best_row.get("delta_target_prob", ""),
                    "oracle_best_delta_target_rank": best_row.get("delta_target_rank", ""),
                    "oracle_best_delta_final_entropy_bits": best_row.get("delta_final_entropy_bits", ""),
                    "oracle_runner_up_g_profile": second_row.get("g_profile", "") if second_row else "",
                    "oracle_runner_up_g_family": second_row.get("g_family", "") if second_row else "",
                    "oracle_runner_up_delta_target_prob": second_row.get("delta_target_prob", "") if second_row else "",
                    "oracle_winner_margin_delta_target_prob": margin,
                    "oracle_top3_g_profiles": "|".join(row.get("g_profile", "") for row in top3),
                }
            )
        lookup[prompt_id] = payload
    return lookup


def build_prompt_table(
    *,
    battery_lookup: dict[str, dict[str, Any]],
    scalars_rows: list[dict[str, str]],
    winners_lookup: dict[str, dict[str, Any]],
    joined_long_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scalar_row in scalars_rows:
        prompt_id = scalar_row["prompt_id"]
        battery_row = battery_lookup.get(prompt_id, {})
        joined_row = joined_long_lookup.get(prompt_id, {})
        winner_row = winners_lookup.get(prompt_id, {})
        row = {
            "prompt_id": prompt_id,
            "type": scalar_row.get("prompt_type", "") or joined_row.get("type", "") or battery_row.get("battery_type", ""),
            "source": scalar_row.get("source", "") or joined_row.get("source", "") or battery_row.get("battery_source", ""),
            "prompt": joined_row.get("prompt", "") or battery_row.get("prompt", ""),
            "target": joined_row.get("target", "") or battery_row.get("target", ""),
            "tokens_approx": scalar_row.get("tokens_approx", "") or battery_row.get("battery_tokens_approx", ""),
            "num_tokens": scalar_row.get("num_tokens", ""),
            "num_layers_plus_embedding": scalar_row.get("num_layers_plus_embedding", ""),
            "hidden_size": scalar_row.get("hidden_size", ""),
            "final_entropy_bits": scalar_row.get("final_entropy_bits", ""),
            "mean_entropy_bits": scalar_row.get("mean_entropy_bits", ""),
            "baseline_target_prob": joined_row.get("baseline_target_prob", ""),
            "baseline_target_rank": joined_row.get("baseline_target_rank", ""),
            "baseline_target_geo_mean_prob": joined_row.get("baseline_target_geo_mean_prob", ""),
            "baseline_final_entropy_bits": joined_row.get("baseline_final_entropy_bits", ""),
            "baseline_mean_entropy_bits": joined_row.get("baseline_mean_entropy_bits", ""),
            "baseline_attn_entropy_mean": joined_row.get("baseline_attn_entropy_mean", ""),
            "intervention_strategy": joined_row.get("intervention_strategy", ""),
            "cartridge": joined_row.get("cartridge", ""),
            "model": joined_row.get("model", ""),
        }
        row.update(winner_row)
        row.update(joined_row)
        rows.append(row)
    return rows


def build_family_join_rows(
    family_rows: list[dict[str, str]],
    prompt_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_row in family_rows:
        prompt_id = family_row["prompt_id"]
        row = dict(prompt_lookup.get(prompt_id, {}))
        row.update(
            {
                "family_type": family_row.get("type", ""),
                "family_source": family_row.get("source", ""),
                "family_tokens_approx": family_row.get("tokens_approx", ""),
                "pc1": family_row.get("pc1", ""),
                "pc2": family_row.get("pc2", ""),
                "pc3": family_row.get("pc3", ""),
                "length_resid_pc1": family_row.get("length_resid_pc1", ""),
                "length_resid_pc2": family_row.get("length_resid_pc2", ""),
                "length_resid_pc3": family_row.get("length_resid_pc3", ""),
            }
        )
        rows.append(row)
    return rows


def infer_model_key(sequence_dir: Path, sweep_dir: Path) -> str:
    for candidate in (sequence_dir.name, sweep_dir.parent.name, sweep_dir.name):
        token = candidate.upper()
        if "OLMO" in token:
            return "OLMO"
        if "9B" in token:
            return "9B"
    raise ValueError(f"Could not infer model key from {sequence_dir} and {sweep_dir}")


def resolve_router_panel_profiles(model_key: str) -> list[str]:
    if model_key == "9B":
        return get_profile_names("9B")
    if model_key == "OLMO":
        router_model_path = Path("router/router-OLMO-030/router_model.json")
        with router_model_path.open("r", encoding="utf-8") as f:
            router_model = json.load(f)
        profile_specs = router_model.get("profile_specs", {})
        profiles = list(profile_specs.keys())
        if len(profiles) != 4:
            raise ValueError(f"Expected 4 embedded OLMO router profiles, found {len(profiles)}")
        return profiles
    raise ValueError(f"Unsupported model key for router panel profiles: {model_key}")


def plot_family_router_panels(
    *,
    family_name: str,
    family_rows: list[dict[str, str]],
    joined_long_rows: list[dict[str, str]],
    profiles_to_plot: list[str],
    model_label: str,
    output_dir: Path,
) -> list[str]:
    coords_by_prompt: dict[str, dict[str, float]] = {}
    for row in family_rows:
        prompt_id = row["prompt_id"]
        coords_by_prompt[prompt_id] = {
            "pc1": float(row["pc1"]),
            "pc2": float(row["pc2"]),
            "length_resid_pc1": float(row["length_resid_pc1"]),
            "length_resid_pc2": float(row["length_resid_pc2"]),
        }

    delta_lookup: dict[str, dict[str, float]] = {profile: {} for profile in profiles_to_plot}
    for row in joined_long_rows:
        if row.get("rep") not in {None, "", "1"}:
            continue
        g_profile = row.get("g_profile", "")
        if g_profile not in delta_lookup:
            continue
        prompt_id = row.get("prompt_id", "")
        delta = to_float(row.get("delta_target_prob"))
        if delta is None or not math.isfinite(delta):
            continue
        delta_lookup[g_profile][prompt_id] = delta

    all_deltas = [
        delta
        for profile in profiles_to_plot
        for delta in delta_lookup[profile].values()
    ]
    if not all_deltas:
        return []

    vmax = max(abs(delta) for delta in all_deltas)
    if vmax < 1e-9:
        vmax = 0.01
    gamma = 0.4
    transformed_limit = vmax**gamma

    configure_matplotlib(font_family="sans-serif", font_size=11)
    written_paths: list[str] = []
    panel_specs = [
        ("raw", "pc1", "pc2", "PC1", "PC2"),
        ("length_resid", "length_resid_pc1", "length_resid_pc2", "Length-resid PC1", "Length-resid PC2"),
    ]

    for panel_key, x_key, y_key, xlabel_base, ylabel_base in panel_specs:
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.8), squeeze=False)
        axes_flat = axes.flatten()
        scatter_ref = None
        for ax, profile in zip(axes_flat, profiles_to_plot):
            x_vals: list[float] = []
            y_vals: list[float] = []
            colors: list[float] = []
            for prompt_id, coords in coords_by_prompt.items():
                delta = delta_lookup[profile].get(prompt_id)
                if delta is None:
                    continue
                x_vals.append(coords[x_key])
                y_vals.append(coords[y_key])
                colors.append(math.copysign(abs(delta) ** gamma, delta))

            if not x_vals:
                ax.set_visible(False)
                continue

            scatter_ref = ax.scatter(
                x_vals,
                y_vals,
                c=colors,
                cmap="RdBu_r",
                vmin=-transformed_limit,
                vmax=transformed_limit,
                s=20,
                alpha=0.72,
                edgecolors="none",
            )
            ax.set_title(profile, fontsize=11)
            ax.set_xlabel(xlabel_base)
            ax.set_ylabel(ylabel_base)
            ax.grid(False)

        for ax in axes_flat[len(profiles_to_plot):]:
            ax.set_visible(False)

        if scatter_ref is not None:
            cbar = fig.colorbar(scatter_ref, ax=axes_flat.tolist(), pad=0.02, shrink=0.92)
            ticks = [-transformed_limit, -transformed_limit / 2, 0.0, transformed_limit / 2, transformed_limit]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(
                [f"{math.copysign(abs(t) ** (1.0 / gamma), t):.3f}" for t in ticks]
            )
            cbar.set_label("Δ target prob")

        fig.suptitle(f"{model_label}: {family_name} router panel ({panel_key})", fontsize=13, y=0.98)
        fig.tight_layout(rect=(0.0, 0.0, 0.96, 0.96))
        out_path = output_dir / f"{family_name}_router_panel_{panel_key}.png"
        fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written_paths.append(str(out_path))

    return written_paths


def residualize_scalar_series(values: list[float], indicator: list[float]) -> list[float]:
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


def plot_family_winner_residual_panels(
    *,
    family_name: str,
    family_rows: list[dict[str, str]],
    prompt_lookup: dict[str, dict[str, Any]],
    profiles_to_plot: list[str],
    model_label: str,
    output_dir: Path,
) -> str | None:
    x_vals: list[float] = []
    y_vals: list[float] = []
    types: list[str] = []
    prompt_ids: list[str] = []
    for row in family_rows:
        prompt_id = row["prompt_id"]
        meta = prompt_lookup.get(prompt_id, {})
        x_vals.append(float(row["pc1"]))
        y_vals.append(float(row["pc2"]))
        types.append(str(meta.get("type", row.get("type", ""))))
        prompt_ids.append(prompt_id)

    if not x_vals:
        return None

    unique_types = sorted({t for t in types if t})
    configure_matplotlib(font_family="sans-serif", font_size=11)
    palette = plt.get_cmap("tab10")
    type_colors = {t: palette(i % 10) for i, t in enumerate(unique_types)}

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.8), squeeze=False)
    axes_flat = axes.flatten()
    for ax, profile in zip(axes_flat, profiles_to_plot):
        indicator = [
            1.0 if prompt_lookup.get(pid, {}).get("oracle_best_g_profile", "") == profile else 0.0
            for pid in prompt_ids
        ]
        resid_x = residualize_scalar_series(x_vals, indicator)
        resid_y = residualize_scalar_series(y_vals, indicator)
        for prompt_type in unique_types:
            idx = [i for i, t in enumerate(types) if t == prompt_type]
            ax.scatter(
                [resid_x[i] for i in idx],
                [resid_y[i] for i in idx],
                s=18,
                alpha=0.72,
                color=type_colors[prompt_type],
                linewidths=0,
                label=prompt_type,
            )
        win_count = int(sum(indicator))
        ax.set_title(f"{profile}\nresid on winner==1 (n={win_count})", fontsize=11)
        ax.set_xlabel("Residualized PC1")
        ax.set_ylabel("Residualized PC2")
        ax.grid(False)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=type_colors[t], markeredgewidth=0)
        for t in unique_types
    ]
    fig.legend(handles, unique_types, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"{model_label}: {family_name} winner-residualized PCA coordinates",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(bottom=0.12, top=0.90, wspace=0.22, hspace=0.26)
    out_path = output_dir / f"{family_name}_winner_residual_panels.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join sequence PCA outputs with sweep winner summaries for prompt-level interpretation."
    )
    parser.add_argument("--sequence-analysis-dir", required=True, help="Directory produced by signal_lab.sequence_analyze.")
    parser.add_argument("--sweep-analysis-dir", required=True, help="Sweep analysis directory containing analysis_joined_long.csv.")
    parser.add_argument(
        "--prompt-battery",
        default=DEFAULT_BATTERY,
        help=f"Prompt battery used by both runs (default: {DEFAULT_BATTERY}).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <sequence-analysis-dir>/interpret).",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Artifact filename prefix (default: {DEFAULT_PREFIX}).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"Optional base directory override. Also supports {DATA_DIR_ENV_VAR}.",
    )
    args = parser.parse_args()

    configure_data_dir(args.data_dir)
    sequence_dir = resolve_dir(
        args.sequence_analysis_dir,
        required_files=["sequence_analysis_manifest.json", "sequence_analysis_prompt_scalars.csv"],
    )
    sweep_dir = resolve_dir(
        args.sweep_analysis_dir,
        required_files=["analysis_joined_long.csv", "analysis_prompt_winners.csv"],
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (sequence_dir / "interpret").resolve()
    )
    ensure_dir(output_dir)

    battery_lookup = build_battery_lookup(args.prompt_battery)
    scalars_rows = load_csv_rows(sequence_dir / "sequence_analysis_prompt_scalars.csv")
    prompt_winner_rows = load_csv_rows(sweep_dir / "analysis_prompt_winners.csv")
    joined_long_rows = load_csv_rows(sweep_dir / "analysis_joined_long.csv")
    winners_lookup = build_prompt_winner_lookup(prompt_winner_rows)
    joined_long_lookup = build_joined_long_lookup(joined_long_rows)
    model_key = infer_model_key(sequence_dir, sweep_dir)
    router_panel_profiles = resolve_router_panel_profiles(model_key)

    prompt_table_rows = build_prompt_table(
        battery_lookup=battery_lookup,
        scalars_rows=scalars_rows,
        winners_lookup=winners_lookup,
        joined_long_lookup=joined_long_lookup,
    )
    prompt_lookup = {row["prompt_id"]: row for row in prompt_table_rows}
    write_csv(output_dir / f"{args.prefix}_prompt_table.csv", prompt_table_rows)

    family_csv_paths = sorted(sequence_dir.glob("*_pca.csv"))
    family_manifest_rows: list[dict[str, Any]] = []
    panel_dir = output_dir / "router_panels"
    ensure_dir(panel_dir)
    panel_manifest_rows: list[dict[str, Any]] = []
    winner_residual_manifest_rows: list[dict[str, Any]] = []
    for family_path in family_csv_paths:
        family_name = family_path.name.removesuffix("_pca.csv")
        family_rows = load_csv_rows(family_path)
        joined_rows = build_family_join_rows(family_rows, prompt_lookup)
        family_output_path = output_dir / f"{family_name}_interpret.csv"
        write_csv(family_output_path, joined_rows)
        plot_paths = plot_family_router_panels(
            family_name=family_name,
            family_rows=family_rows,
            joined_long_rows=joined_long_rows,
            profiles_to_plot=router_panel_profiles,
            model_label=model_key,
            output_dir=panel_dir,
        )
        winner_residual_path = plot_family_winner_residual_panels(
            family_name=family_name,
            family_rows=family_rows,
            prompt_lookup=prompt_lookup,
            profiles_to_plot=router_panel_profiles,
            model_label=model_key,
            output_dir=panel_dir,
        )
        family_manifest_rows.append(
            {
                "family": family_name,
                "source_pca_csv": str(family_path),
                "interpret_csv": str(family_output_path),
                "n_rows": len(joined_rows),
            }
        )
        panel_manifest_rows.append(
            {
                "family": family_name,
                "profiles": "|".join(router_panel_profiles),
                "raw_panel_png": next((path for path in plot_paths if path.endswith("_raw.png")), ""),
                "length_resid_panel_png": next((path for path in plot_paths if path.endswith("_length_resid.png")), ""),
            }
        )
        winner_residual_manifest_rows.append(
            {
                "family": family_name,
                "profiles": "|".join(router_panel_profiles),
                "winner_residual_panel_png": winner_residual_path or "",
                "note": "Residualized on binary winner indicator separately for each profile using exported PCA coordinates.",
            }
        )

    write_csv(output_dir / f"{args.prefix}_family_manifest.csv", family_manifest_rows)
    write_csv(output_dir / f"{args.prefix}_router_panel_manifest.csv", panel_manifest_rows)
    write_csv(output_dir / f"{args.prefix}_winner_residual_manifest.csv", winner_residual_manifest_rows)
    manifest = {
        "run_kind": "sequence_interpret",
        "sequence_analysis_dir": str(sequence_dir),
        "sweep_analysis_dir": str(sweep_dir),
        "prompt_battery": args.prompt_battery,
        "output_dir": str(output_dir),
        "prefix": args.prefix,
        "n_prompts": len(prompt_table_rows),
        "families": [row["family"] for row in family_manifest_rows],
        "router_panel_profiles": router_panel_profiles,
        "router_panel_dir": str(panel_dir),
        "winner_residual_note": "Winner-residual plots are coordinate-space diagnostics, not full feature-matrix residualized PCA reruns.",
    }
    write_json(output_dir / f"{args.prefix}_manifest.json", manifest)

    print("Wrote interpret files:")
    print(f"- {output_dir / f'{args.prefix}_prompt_table.csv'}")
    print(f"- {output_dir / f'{args.prefix}_family_manifest.csv'}")
    print(f"- {output_dir / f'{args.prefix}_router_panel_manifest.csv'}")
    print(f"- {output_dir / f'{args.prefix}_winner_residual_manifest.csv'}")
    print(f"- {output_dir / f'{args.prefix}_manifest.json'}")
    for row in family_manifest_rows:
        print(f"- {row['interpret_csv']}")
    for row in panel_manifest_rows:
        if row["raw_panel_png"]:
            print(f"- {row['raw_panel_png']}")
        if row["length_resid_panel_png"]:
            print(f"- {row['length_resid_panel_png']}")
    for row in winner_residual_manifest_rows:
        if row["winner_residual_panel_png"]:
            print(f"- {row['winner_residual_panel_png']}")


if __name__ == "__main__":
    main()
