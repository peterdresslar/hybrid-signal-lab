from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "docs" / "analysis" / "oracle_winners" / "exports"

OLMO_BLOCK_COLLAPSE_V1 = {
    "constant_0.4",
    "constant_0.55",
    "constant_1.8",
    "constant_2",
    "constant_2.3",
    "constant_2.6",
    "constant_3",
    "early_boost_bal_0.60",
    "late_boost_bal_0.60",
}

SWEEPS = [
    {
        "experiment": "022-balanced-attention-hybrid",
        "sweep": "Qwen 2B attn-contr",
        "model_label": "Qwen 2B",
        "model_family": "qwen_hybrid",
        "intervention_family": "attn_contribution",
        "path": DATA_ROOT / "022-balanced-attention-hybrid" / "2B" / "analysis" / "analysis_joined_long.csv",
        "exclusion_policy": "none",
        "excluded_profiles": set(),
    },
    {
        "experiment": "022-balanced-attention-hybrid",
        "sweep": "Qwen 9B attn-contr",
        "model_label": "Qwen 9B",
        "model_family": "qwen_hybrid",
        "intervention_family": "attn_contribution",
        "path": DATA_ROOT / "022-balanced-attention-hybrid" / "9B" / "analysis" / "analysis_joined_long.csv",
        "exclusion_policy": "none",
        "excluded_profiles": set(),
    },
    {
        "experiment": "022-balanced-attention-hybrid",
        "sweep": "Qwen 35B attn-contr",
        "model_label": "Qwen 35B",
        "model_family": "qwen_hybrid",
        "intervention_family": "attn_contribution",
        "path": DATA_ROOT / "022-balanced-attention-hybrid" / "35B" / "analysis" / "analysis_joined_long.csv",
        "exclusion_policy": "none",
        "excluded_profiles": set(),
    },
    {
        "experiment": "022-balanced-attention-hybrid",
        "sweep": "Olmo attn-contr",
        "model_label": "Olmo Hybrid",
        "model_family": "olmo_hybrid",
        "intervention_family": "attn_contribution",
        "path": DATA_ROOT / "022-balanced-attention-hybrid" / "OLMO" / "analysis" / "analysis_joined_long.csv",
        "exclusion_policy": "none",
        "excluded_profiles": set(),
    },
    {
        "experiment": "022-balanced-block-hybrid",
        "sweep": "Qwen 2B block-out",
        "model_label": "Qwen 2B",
        "model_family": "qwen_hybrid",
        "intervention_family": "block_output",
        "path": DATA_ROOT / "022-balanced-block-hybrid" / "2B" / "analysis" / "analysis_joined_long.csv",
        "exclusion_policy": "none",
        "excluded_profiles": set(),
    },
    {
        "experiment": "022-balanced-block-hybrid",
        "sweep": "Qwen 9B block-out",
        "model_label": "Qwen 9B",
        "model_family": "qwen_hybrid",
        "intervention_family": "block_output",
        "path": DATA_ROOT / "022-balanced-block-hybrid" / "9B" / "analysis" / "analysis_joined_long.csv",
        "exclusion_policy": "none",
        "excluded_profiles": set(),
    },
    {
        "experiment": "022-balanced-block-hybrid",
        "sweep": "Qwen 35B block-out",
        "model_label": "Qwen 35B",
        "model_family": "qwen_hybrid",
        "intervention_family": "block_output",
        "path": DATA_ROOT / "022-balanced-block-hybrid" / "35B" / "analysis" / "analysis_joined_long.csv",
        "exclusion_policy": "none",
        "excluded_profiles": set(),
    },
    {
        "experiment": "022-balanced-block-hybrid",
        "sweep": "Olmo block-out",
        "model_label": "Olmo Hybrid",
        "model_family": "olmo_hybrid",
        "intervention_family": "block_output",
        "path": DATA_ROOT / "022-balanced-block-hybrid" / "OLMO" / "analysis" / "analysis_joined_long.csv",
        "exclusion_policy": "olmo_block_collapse_v1",
        "excluded_profiles": OLMO_BLOCK_COLLAPSE_V1,
    },
]

BENCH_RUNS = [
    {
        "benchmark_run": "routed_9B",
        "model_label": "Qwen 9B",
        "model_family": "qwen_hybrid",
        "path": DATA_ROOT / "030-bench" / "routed_9B",
    },
    {
        "benchmark_run": "routed_OLMO",
        "model_label": "Olmo Hybrid",
        "model_family": "olmo_hybrid",
        "path": DATA_ROOT / "030-bench" / "routed_OLMO",
    },
]


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def safe_float(value: str | None) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def classify_family(row: dict) -> str:
    gp = row["g_profile"]
    if gp == "baseline":
        return "baseline"
    return "constant" if row["g_family"] == "constant" else "shaped"


def export_sweep_oracles() -> list[dict]:
    out_rows: list[dict] = []
    scopes = ("full_library", "constant_only")
    objectives = ("target_rank_min", "delta_target_prob_max")

    for spec in SWEEPS:
        rows = load_csv(spec["path"])
        by_prompt: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for row in rows:
            rank = safe_float(row.get("target_rank"))
            if rank is None:
                continue
            gp = row["g_profile"]
            if gp != "baseline" and gp in spec["excluded_profiles"]:
                continue
            by_prompt[(row["prompt_id"], row["rep"])].append(row)

        for (prompt_id, rep), prompt_rows in by_prompt.items():
            sample = next(r for r in prompt_rows if r["g_profile"] == "baseline")
            for scope in scopes:
                scoped_rows = []
                for row in prompt_rows:
                    fam = classify_family(row)
                    if scope == "constant_only" and fam == "shaped":
                        continue
                    scoped_rows.append(row)

                for objective in objectives:
                    winner_row: dict
                    if objective == "target_rank_min":
                        winner_row = min(scoped_rows, key=lambda r: safe_float(r["target_rank"]) or float("inf"))
                    else:
                        non_baseline = [r for r in scoped_rows if r["g_profile"] != "baseline"]
                        positive = [r for r in non_baseline if (safe_float(r.get("delta_target_prob")) or 0.0) > 0]
                        if positive:
                            winner_row = max(positive, key=lambda r: safe_float(r["delta_target_prob"]) or -float("inf"))
                        else:
                            winner_row = next(r for r in scoped_rows if r["g_profile"] == "baseline")

                    winner_family = classify_family(winner_row)
                    out_rows.append(
                        {
                            "export_version": "v1",
                            "source_kind": "sweep",
                            "experiment": spec["experiment"],
                            "sweep": spec["sweep"],
                            "model_label": spec["model_label"],
                            "model_family": spec["model_family"],
                            "intervention_family": spec["intervention_family"],
                            "winner_scope": scope,
                            "winner_objective": objective,
                            "exclusion_policy": spec["exclusion_policy"],
                            "prompt_id": prompt_id,
                            "rep": rep,
                            "type": sample["type"],
                            "tier": sample["tier"],
                            "source": sample["source"],
                            "target": sample["target"],
                            "tokens_approx": sample["tokens_approx"],
                            "winner_profile": winner_row["g_profile"],
                            "winner_family": winner_family,
                            "winner_target_rank": winner_row["target_rank"],
                            "winner_target_prob": winner_row["target_prob"],
                            "winner_delta_target_prob": winner_row.get("delta_target_prob", ""),
                            "baseline_target_rank": sample["target_rank"],
                            "baseline_target_prob": sample["target_prob"],
                        }
                    )
    return out_rows


def export_bench_oracles() -> list[dict]:
    out_rows: list[dict] = []
    for run in BENCH_RUNS:
        for records_path in sorted(run["path"].glob("*_records.jsonl")):
            task = records_path.name.removesuffix("_records.jsonl")
            with records_path.open() as f:
                for line in f:
                    record = json.loads(line)
                    if record.get("condition") != "oracle":
                        continue
                    profile = record.get("oracle_profile", "baseline")
                    winner_family = "baseline" if profile == "baseline" else ("constant" if profile.startswith("constant_") else "shaped")
                    out_rows.append(
                        {
                            "export_version": "v1",
                            "source_kind": "bench",
                            "benchmark_run": run["benchmark_run"],
                            "task": task,
                            "model_label": run["model_label"],
                            "model_family": run["model_family"],
                            "winner_scope": "selected_panel",
                            "winner_objective": "benchmark_oracle_recorded",
                            "example_id": record["example_id"],
                            "winner_profile": profile,
                            "winner_family": winner_family,
                            "oracle_margin": record.get("oracle_margin", ""),
                            "predicted": record.get("predicted", ""),
                            "correct_idx": record.get("correct_idx", ""),
                            "correct": record.get("correct", ""),
                        }
                    )
    return out_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_schema(path: Path) -> None:
    payload = {
        "export_version": "v1",
        "notes": {
            "purpose": "Versioned source-of-truth exports for oracle winner analyses used in manuscript tables and figures.",
            "olmo_block_collapse_policy": sorted(OLMO_BLOCK_COLLAPSE_V1),
        },
        "sweep_winner_definitions": {
            "full_library": "Compare baseline against all retained non-baseline profiles.",
            "constant_only": "Compare baseline against retained constant profiles only.",
            "target_rank_min": "Winner is the profile with minimum target_rank.",
            "delta_target_prob_max": "Winner is the profile with maximum positive delta_target_prob, else baseline fallback.",
        },
        "bench_winner_definitions": {
            "selected_panel": "Oracle profiles recorded in benchmark oracle records; restricted to the deployed benchmark panel plus baseline.",
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "sweep_oracle_winners_v1.csv", export_sweep_oracles())
    write_csv(OUT_DIR / "bench_oracle_winners_v1.csv", export_bench_oracles())
    write_schema(OUT_DIR / "oracle_winner_schema_v1.json")


if __name__ == "__main__":
    main()
