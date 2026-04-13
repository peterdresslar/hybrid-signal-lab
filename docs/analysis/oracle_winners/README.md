# Oracle Winners

This directory defines the versioned source-of-truth exports for prompt-level and example-level oracle winner analyses used in the manuscript.

The immediate goal is to eliminate ambiguity around what “winner” means in each context.

## Current exports

- `exports/sweep_oracle_winners_v1.csv`
  - Prompt-level oracle winners derived from the `022-*` sweep artifacts.
  - Includes multiple explicit winner definitions:
    - `target_rank_min`
    - `delta_target_prob_max`
  - Includes multiple scopes:
    - `full_library`
    - `constant_only`

- `exports/bench_oracle_winners_v1.csv`
  - Example-level oracle winners derived from `030-bench` routed benchmark runs.
  - Uses the oracle profile recorded in the benchmark `*_records.jsonl` files.

- `exports/oracle_winner_schema_v1.json`
  - Version metadata and policy notes.

## Why this exists

The repo historically had multiple winner-like artifacts:

- `analysis_prompt_winners.csv`
- prompt-level rank oracles reconstructed from `analysis_joined_long.csv`
- benchmark oracle choices recorded in `*_records.jsonl`

Those are not interchangeable. This directory makes the definitions explicit and versioned.

## Canonical definitions in v1

### Sweep winners

Rows in `sweep_oracle_winners_v1.csv` are keyed by:

- `experiment`
- `sweep`
- `winner_scope`
- `winner_objective`
- `prompt_id`
- `rep`

Definitions:

- `winner_scope = full_library`
  - compare `baseline` against all retained non-baseline profiles

- `winner_scope = constant_only`
  - compare `baseline` against retained constant profiles only

- `winner_objective = target_rank_min`
  - winner is the profile with the minimum `target_rank`

- `winner_objective = delta_target_prob_max`
  - winner is the profile with the maximum `delta_target_prob`
  - baseline is used as fallback when no retained non-baseline profile has positive `delta_target_prob`

### Bench winners

Rows in `bench_oracle_winners_v1.csv` are keyed by:

- `benchmark_run`
- `task`
- `example_id`

Definition:

- winner is the `oracle_profile` recorded in the `condition == "oracle"` record
- this is panel-restricted by construction, because the benchmark runs only evaluate the deployed profile panel plus baseline

## Special policy: Olmo block-output collapse

`Olmo block-out` has a documented collapse region. In v1 sweep exports, this is encoded explicitly as:

- `exclusion_policy = olmo_block_collapse_v1`

Excluded profiles:

- `constant_0.4`
- `constant_0.55`
- `constant_1.8`
- `constant_2`
- `constant_2.3`
- `constant_2.6`
- `constant_3`
- `early_boost_bal_0.60`
- `late_boost_bal_0.60`

All other sweeps use:

- `exclusion_policy = none`

## Regeneration

Run:

```bash
python docs/analysis/oracle_winners/scripts/export_oracle_winners.py
```

Regression check:

```bash
python docs/analysis/oracle_winners/scripts/check_oracle_winners_regression.py
```
