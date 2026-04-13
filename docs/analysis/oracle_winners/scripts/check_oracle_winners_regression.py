from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPORT_DIR = REPO_ROOT / "docs" / "analysis" / "oracle_winners" / "exports"

EXPECTED_SWEEP_ROWS = 34_240
EXPECTED_BENCH_ROWS = 2_944

EXPECTED_OLMO_BLOCK_EXCLUSIONS = [
    "constant_0.4",
    "constant_0.55",
    "constant_1.8",
    "constant_2",
    "constant_2.3",
    "constant_2.6",
    "constant_3",
    "early_boost_bal_0.60",
    "late_boost_bal_0.60",
]

EXPECTED_TABLE_5_1_4_RANK = {
    "Qwen 2B attn-contr": (252.8, 177.7, 154.9, 77),
    "Qwen 9B attn-contr": (109.2, 72.3, 60.9, 76),
    "Qwen 35B attn-contr": (109.5, 58.8, 53.4, 90),
    "Olmo attn-contr": (170.3, 96.9, 83.9, 85),
    "Qwen 2B block-out": (252.8, 139.3, 85.7, 68),
    "Qwen 9B block-out": (109.2, 63.6, 36.4, 63),
    "Qwen 35B block-out": (109.5, 59.3, 30.8, 64),
    "Olmo block-out": (170.3, 110.0, 53.7, 52),
}


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def main() -> None:
    sweep_path = EXPORT_DIR / "sweep_oracle_winners_v1.csv"
    bench_path = EXPORT_DIR / "bench_oracle_winners_v1.csv"
    schema_path = EXPORT_DIR / "oracle_winner_schema_v1.json"

    sweep_rows = load_csv(sweep_path)
    bench_rows = load_csv(bench_path)
    schema = json.loads(schema_path.read_text())

    assert_equal(schema["export_version"], "v1", "schema version")
    assert_equal(len(sweep_rows), EXPECTED_SWEEP_ROWS, "sweep row count")
    assert_equal(len(bench_rows), EXPECTED_BENCH_ROWS, "bench row count")
    assert_equal(
        schema["notes"]["olmo_block_collapse_policy"],
        EXPECTED_OLMO_BLOCK_EXCLUSIONS,
        "Olmo block-out exclusion policy",
    )

    rank_rows = [row for row in sweep_rows if row["winner_objective"] == "target_rank_min"]
    for sweep_name, expected in EXPECTED_TABLE_5_1_4_RANK.items():
        baseline = [
            float(row["baseline_target_rank"])
            for row in rank_rows
            if row["sweep"] == sweep_name and row["winner_scope"] == "full_library"
        ]
        constant = [
            float(row["winner_target_rank"])
            for row in rank_rows
            if row["sweep"] == sweep_name and row["winner_scope"] == "constant_only"
        ]
        full = [
            float(row["winner_target_rank"])
            for row in rank_rows
            if row["sweep"] == sweep_name and row["winner_scope"] == "full_library"
        ]
        reconstructed = (
            round(statistics.mean(baseline), 1),
            round(statistics.mean(constant), 1),
            round(statistics.mean(full), 1),
            round(
                100
                * (
                    (statistics.mean(baseline) - statistics.mean(constant))
                    / (statistics.mean(baseline) - statistics.mean(full))
                )
            ),
        )
        assert_equal(reconstructed, expected, f"5.1.4 rank table row for {sweep_name}")

    bench_task_counts = Counter(row["task"] for row in bench_rows)
    assert_equal(
        dict(bench_task_counts),
        {
            "arc_challenge": 2344,
            "mmlu_abstract_algebra": 200,
            "mmlu_college_cs": 200,
            "mmlu_college_math": 200,
        },
        "bench task counts",
    )

    print("oracle_winners regression checks passed")


if __name__ == "__main__":
    main()
