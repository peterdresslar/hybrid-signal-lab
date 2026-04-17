"""
run_bench.py — Run scoring benchmarks with baseline and routed inference.

Evaluates multiple-choice log-likelihood tasks under four conditions:
  1. Baseline (g=1.0, no intervention)
  2. Best fixed profile (single best gain profile applied to all prompts)
  3. Routed (two-pass: baseline → classify → intervene)
  4. Oracle (try baseline + all fixed profiles, pick best per-example)
Supported tasks:
  - COPA
  - StoryCloze
  - ARC-Challenge
  - MMLU subsets (abstract algebra, college mathematics, college CS)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
import builtins
from functools import partial

import numpy as np
import dotenv

from signal_lab.agent import Agent
from model.backend import InterventionMode
from model.g_profile import build_attention_scales_from_spec

from bench.tasks import (
    load_copa,
    load_storycloze,
    load_arc_challenge,
    load_mmlu,
    ScoringExample,
)
from router.router import InterventionRouter
from router.profiles import BASELINE_SPEC

dotenv.load_dotenv(".env.development")
dotenv.load_dotenv(".env")
dotenv.load_dotenv(".env.slurm")

print = partial(builtins.print, flush=True)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

class ProgressHeartbeat:
    """Emit periodic liveness updates during slow benchmark loops."""

    def __init__(self, task_name: str, condition: str, total: int, interval_s: float = 30.0):
        self.task_name = task_name
        self.condition = condition
        self.total = total
        self.interval_s = interval_s
        self.start_time = time.time()
        self.last_ping_time = self.start_time

    def maybe_ping(
        self,
        completed: int,
        *,
        metric_name: str = "acc",
        metric_value: float | None = None,
        force: bool = False,
    ) -> None:
        now = time.time()
        if not force and (now - self.last_ping_time) < self.interval_s:
            return

        elapsed = now - self.start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = self.total - completed
        eta_s = (remaining / rate) if rate > 0 else None

        message = (
            f"    [{self.task_name}/{self.condition}] "
            f"{completed}/{self.total} done"
        )
        if metric_value is not None:
            message += f"  {metric_name}={metric_value:.3f}"
        message += f"  elapsed={elapsed:.1f}s"
        if eta_s is not None:
            message += f"  eta={eta_s:.1f}s"
        print(message)
        self.last_ping_time = now

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_continuation(
    agent: Agent,
    context: str,
    continuation: str,
    g_scales: np.ndarray,
    intervention_mode: InterventionMode = InterventionMode.ATTENTION_CONTRIBUTION,
) -> float:
    """Score a single continuation and return its total log-probability."""
    result = agent.score_target(
        context,
        continuation,
        g_scales,
        intervention_mode=intervention_mode,
    )
    if result is None:
        return float("-inf")
    return result["target_seq_logprob"]


def run_scoring_example_baseline(
    agent: Agent,
    example: ScoringExample,
    baseline_scales: np.ndarray,
    intervention_mode: InterventionMode = InterventionMode.ATTENTION_CONTRIBUTION,
) -> dict[str, Any]:
    """Run a single scoring example under baseline conditions."""
    scores = []
    for cont in example.continuations:
        lp = score_continuation(
            agent,
            example.context,
            cont,
            baseline_scales,
            intervention_mode=intervention_mode,
        )
        scores.append(lp)

    predicted = int(np.argmax(scores))
    correct = predicted == example.correct_idx

    return {
        "example_id": example.example_id,
        "condition": "baseline",
        "scores": scores,
        "predicted": predicted,
        "correct_idx": example.correct_idx,
        "correct": correct,
    }


def run_scoring_example_routed(
    agent: Agent,
    router: InterventionRouter,
    example: ScoringExample,
    baseline_scales: np.ndarray,
    profile_scales: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Run a single scoring example with routing.

    1. Run baseline forward pass (verbose) for sensing
    2. Router classifies → picks profile or "off"
    3. Score continuations under the selected profile
    """
    # Sensing pass
    baseline_result = agent.run_pass(
        example.context,
        baseline_scales,
        return_verbose=True,
        return_hidden_states=router.requires_hidden_states,
        intervention_mode=router.intervention_mode,
    )

    # Classify
    decision = router.classify(baseline_result)

    # Score continuations under selected profile
    if decision["is_off"]:
        scales = baseline_scales
    else:
        scales = profile_scales[decision["profile_name"]]

    scores = []
    for cont in example.continuations:
        lp = score_continuation(
            agent,
            example.context,
            cont,
            scales,
            intervention_mode=router.intervention_mode,
        )
        scores.append(lp)

    predicted = int(np.argmax(scores))
    correct = predicted == example.correct_idx

    return {
        "example_id": example.example_id,
        "condition": "routed",
        "profile": decision["profile_name"],
        "confidence": decision["confidence"],
        "scores": scores,
        "predicted": predicted,
        "correct_idx": example.correct_idx,
        "correct": correct,
    }


def _threshold_slug(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def run_scoring_example_fixed(
    agent: Agent,
    example: ScoringExample,
    profile_name: str,
    scales: np.ndarray,
    intervention_mode: InterventionMode = InterventionMode.ATTENTION_CONTRIBUTION,
) -> dict[str, Any]:
    """Run a single scoring example under a fixed profile."""
    scores = []
    for cont in example.continuations:
        lp = score_continuation(
            agent,
            example.context,
            cont,
            scales,
            intervention_mode=intervention_mode,
        )
        scores.append(lp)

    predicted = int(np.argmax(scores))
    correct = predicted == example.correct_idx

    return {
        "example_id": example.example_id,
        "condition": f"fixed_{profile_name}",
        "scores": scores,
        "predicted": predicted,
        "correct_idx": example.correct_idx,
        "correct": correct,
    }


def run_scoring_example_oracle(
    agent: Agent,
    example: ScoringExample,
    baseline_scales: np.ndarray,
    profile_scales: dict[str, np.ndarray],
    intervention_mode: InterventionMode = InterventionMode.ATTENTION_CONTRIBUTION,
) -> dict[str, Any]:
    """Run a single scoring example under oracle routing.

    Tries baseline + all profiles, picks whichever gets it right
    (or if multiple get it right, whichever has the largest margin).
    """
    all_results = {}

    # Baseline
    scores_bl = []
    for cont in example.continuations:
        scores_bl.append(
            score_continuation(
                agent,
                example.context,
                cont,
                baseline_scales,
                intervention_mode=intervention_mode,
            )
        )
    all_results["baseline"] = scores_bl

    # Each profile
    for pname, pscales in profile_scales.items():
        scores_p = []
        for cont in example.continuations:
            scores_p.append(
                score_continuation(
                    agent,
                    example.context,
                    cont,
                    pscales,
                    intervention_mode=intervention_mode,
                )
            )
        all_results[pname] = scores_p

    # Pick the condition that gets the highest margin for the correct answer
    best_condition = "baseline"
    best_margin = scores_bl[example.correct_idx] - max(
        s for i, s in enumerate(scores_bl) if i != example.correct_idx
    )

    for pname, pscores in all_results.items():
        if pname == "baseline":
            continue
        margin = pscores[example.correct_idx] - max(
            s for i, s in enumerate(pscores) if i != example.correct_idx
        )
        if margin > best_margin:
            best_margin = margin
            best_condition = pname

    best_scores = all_results[best_condition]
    predicted = int(np.argmax(best_scores))
    correct = predicted == example.correct_idx

    return {
        "example_id": example.example_id,
        "condition": "oracle",
        "oracle_profile": best_condition,
        "oracle_margin": float(best_margin),
        "scores": best_scores,
        "predicted": predicted,
        "correct_idx": example.correct_idx,
        "correct": correct,
    }
# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_scoring_task(
    task_name: str,
    examples: list[ScoringExample],
    agent: Agent,
    *,
    router: InterventionRouter | None = None,
    router_thresholds: list[float] | None = None,
    baseline_only: bool = False,
    output_dir: Path,
) -> dict[str, Any]:
    """Run a full log-likelihood scoring task."""
    n_attn = len(agent.get_attention_layer_indices())
    baseline_scales = build_attention_scales_from_spec(BASELINE_SPEC, attention_slots=n_attn)

    # Build profile scales
    profile_scales = {}
    if router and not baseline_only:
        specs = router.profile_specs
        for pname, pspec in specs.items():
            profile_scales[pname] = build_attention_scales_from_spec(pspec, attention_slots=n_attn)

    results = {"task": task_name, "n_examples": len(examples), "conditions": {}}
    all_records = []

    # --- Baseline ---
    print(f"\n  [{task_name}] Running baseline...")
    t0 = time.time()
    baseline_records = []
    baseline_heartbeat = ProgressHeartbeat(task_name, "baseline", len(examples))
    for i, ex in enumerate(examples):
        rec = run_scoring_example_baseline(
            agent,
            ex,
            baseline_scales,
            intervention_mode=router.intervention_mode if router else InterventionMode.ATTENTION_CONTRIBUTION,
        )
        baseline_records.append(rec)
        acc_so_far = sum(r["correct"] for r in baseline_records) / len(baseline_records)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(examples)}  acc={acc_so_far:.3f}")
        baseline_heartbeat.maybe_ping(i + 1, metric_value=acc_so_far)

    bl_acc = sum(r["correct"] for r in baseline_records) / len(baseline_records)
    results["conditions"]["baseline"] = {
        "accuracy": bl_acc,
        "n_correct": sum(r["correct"] for r in baseline_records),
        "elapsed": time.time() - t0,
    }
    all_records.extend(baseline_records)
    print(f"  [{task_name}] Baseline: {bl_acc:.3f} ({time.time()-t0:.1f}s)")

    if baseline_only:
        _save_task_results(results, all_records, task_name, output_dir)
        return results

    # --- Routed ---
    if router:
        routed_thresholds = router_thresholds or [router.decision_threshold]
        original_threshold = router.decision_threshold

        for threshold in routed_thresholds:
            router.decision_threshold = threshold
            threshold_slug = _threshold_slug(threshold)
            condition_name = "routed" if len(routed_thresholds) == 1 else f"routed_t{threshold_slug}"

            print(f"\n  [{task_name}] Running routed (threshold={threshold:.2f})...")
            t0 = time.time()
            routed_records = []
            routed_heartbeat = ProgressHeartbeat(task_name, condition_name, len(examples))
            for i, ex in enumerate(examples):
                rec = run_scoring_example_routed(agent, router, ex, baseline_scales, profile_scales)
                if len(routed_thresholds) > 1:
                    rec["condition"] = condition_name
                    rec["router_decision_threshold"] = threshold
                routed_records.append(rec)
                acc_so_far = sum(r["correct"] for r in routed_records) / len(routed_records)
                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(examples)}  acc={acc_so_far:.3f}")
                routed_heartbeat.maybe_ping(i + 1, metric_value=acc_so_far)

            rt_acc = sum(r["correct"] for r in routed_records) / len(routed_records)
            results["conditions"][condition_name] = {
                "accuracy": rt_acc,
                "n_correct": sum(r["correct"] for r in routed_records),
                "elapsed": time.time() - t0,
                "profile_distribution": _count_profiles(routed_records),
                "router_decision_threshold": threshold,
            }
            all_records.extend(routed_records)
            print(f"  [{task_name}] {condition_name}: {rt_acc:.3f} ({time.time()-t0:.1f}s)")

        router.decision_threshold = original_threshold

    # --- Best fixed ---
    if profile_scales:
        best_fixed_acc = 0.0
        best_fixed_name = None
        for pname, pscales in profile_scales.items():
            print(f"\n  [{task_name}] Running fixed {pname}...")
            t0 = time.time()
            fixed_records = []
            fixed_heartbeat = ProgressHeartbeat(task_name, f"fixed_{pname}", len(examples))
            for i, ex in enumerate(examples):
                rec = run_scoring_example_fixed(
                    agent,
                    ex,
                    pname,
                    pscales,
                    intervention_mode=router.intervention_mode,
                )
                fixed_records.append(rec)
                acc_so_far = sum(r["correct"] for r in fixed_records) / len(fixed_records)
                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(examples)}  acc={acc_so_far:.3f}")
                fixed_heartbeat.maybe_ping(i + 1, metric_value=acc_so_far)

            f_acc = sum(r["correct"] for r in fixed_records) / len(fixed_records)
            results["conditions"][f"fixed_{pname}"] = {
                "accuracy": f_acc,
                "n_correct": sum(r["correct"] for r in fixed_records),
                "elapsed": time.time() - t0,
            }
            all_records.extend(fixed_records)
            print(f"  [{task_name}] Fixed {pname}: {f_acc:.3f} ({time.time()-t0:.1f}s)")

            if f_acc > best_fixed_acc:
                best_fixed_acc = f_acc
                best_fixed_name = pname

        results["best_fixed"] = {"profile": best_fixed_name, "accuracy": best_fixed_acc}

    # --- Oracle ---
    if profile_scales:
        print(f"\n  [{task_name}] Running oracle...")
        t0 = time.time()
        oracle_records = []
        oracle_heartbeat = ProgressHeartbeat(task_name, "oracle", len(examples))
        for i, ex in enumerate(examples):
            rec = run_scoring_example_oracle(
                agent,
                ex,
                baseline_scales,
                profile_scales,
                intervention_mode=router.intervention_mode,
            )
            oracle_records.append(rec)
            acc_so_far = sum(r["correct"] for r in oracle_records) / len(oracle_records)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(examples)}  acc={acc_so_far:.3f}")
            oracle_heartbeat.maybe_ping(i + 1, metric_value=acc_so_far)

        or_acc = sum(r["correct"] for r in oracle_records) / len(oracle_records)
        results["conditions"]["oracle"] = {
            "accuracy": or_acc,
            "n_correct": sum(r["correct"] for r in oracle_records),
            "elapsed": time.time() - t0,
            "oracle_profile_distribution": _count_profiles(oracle_records, key="oracle_profile"),
        }
        all_records.extend(oracle_records)
        print(f"  [{task_name}] Oracle:   {or_acc:.3f} ({time.time()-t0:.1f}s)")

    _save_task_results(results, all_records, task_name, output_dir)
    return results
def _count_profiles(records: list[dict], key: str = "profile") -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        p = r.get(key, "unknown")
        counts[p] = counts.get(p, 0) + 1
    return counts


def _save_task_results(results: dict, records: list[dict], task_name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{task_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / f"{task_name}_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def maybe_limit_examples(
    examples: list[ScoringExample],
    limit: int | None,
) -> list[ScoringExample]:
    """Optionally truncate a loaded benchmark for diagnostic runs."""
    if limit is None:
        return examples
    return examples[:limit]


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict], model_key: str):
    """Print a summary table across all tasks."""
    routed_conditions: list[str] = []
    for r in all_results:
        for condition_name in r["conditions"]:
            if condition_name.startswith("routed") and condition_name not in routed_conditions:
                routed_conditions.append(condition_name)

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK SUMMARY: {model_key}")
    print(f"{'=' * 70}")
    header = f"  {'Task':<15} {'N':>5} {'Baseline':>10} {'Best Fix':>10}"
    for condition_name in routed_conditions:
        header += f" {condition_name:>12}"
    header += f" {'Oracle':>10}"
    print(header)
    print(f"  {'-' * max(62, len(header) - 2)}")

    for r in all_results:
        task = r["task"]
        n = r["n_examples"]
        bl = r["conditions"].get("baseline", {}).get("accuracy", "—")
        ora = r["conditions"].get("oracle", {}).get("accuracy", "—")
        bf = r.get("best_fixed", {}).get("accuracy", "—")

        bl_s = f"{bl:.3f}" if isinstance(bl, float) else bl
        bf_s = f"{bf:.3f}" if isinstance(bf, float) else bf
        ora_s = f"{ora:.3f}" if isinstance(ora, float) else ora

        row = f"  {task:<15} {n:>5} {bl_s:>10} {bf_s:>10}"
        for condition_name in routed_conditions:
            rt = r["conditions"].get(condition_name, {}).get("accuracy", "—")
            rt_s = f"{rt:.3f}" if isinstance(rt, float) else rt
            row += f" {rt_s:>12}"
        row += f" {ora_s:>10}"
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks.")
    parser.add_argument("--model-key", required=True, help="Model key (9B, OLMO)")
    parser.add_argument("--tasks", nargs="+", default=["copa", "storycloze"],
                        choices=[
                            "copa",
                            "storycloze",
                            "arc_challenge",
                            "mmlu_abstract_algebra",
                            "mmlu_college_math",
                            "mmlu_college_cs",
                        ],
                        help="Which benchmarks to run")
    parser.add_argument("--router-model", default=None,
                        help="Path to router_model.json (omit for baseline-only)")
    parser.add_argument(
        "--router-decision-threshold",
        type=float,
        default=None,
        help="Optional abstention threshold override. Higher values make the router choose baseline/off more often.",
    )
    parser.add_argument(
        "--router-decision-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Evaluate multiple routed thresholds in one benchmark pass. Baseline, fixed, and oracle are shared.",
    )
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only baseline condition (no routing)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for results")
    parser.add_argument("--device", default="auto",
                        help="Device (cuda, cuda:0, mps, cpu, auto)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional maximum number of examples per task (for diagnostics)")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model {args.model_key} on {device}...")
    agent = Agent.from_model_key(args.model_key, device)
    print(f"  Model loaded.")

    # Load router
    router = None
    if args.router_model and not args.baseline_only:
        print(f"Loading router from {args.router_model}...")
        router = InterventionRouter.from_artifacts(args.router_model)
        if args.router_decision_threshold is not None:
            router.decision_threshold = args.router_decision_threshold
        print(f"  {router}")

    router_thresholds = None
    if args.router_decision_thresholds:
        router_thresholds = args.router_decision_thresholds
    elif args.router_decision_threshold is not None:
        router_thresholds = [args.router_decision_threshold]

    all_results = []

    # Run tasks
    for task_name in args.tasks:
        print(f"\n{'=' * 50}")
        print(f"  Task: {task_name}")
        print(f"{'=' * 50}")

        if task_name == "copa":
            examples = maybe_limit_examples(load_copa("validation"), args.limit)
            print(f"  Loaded {len(examples)} COPA examples")
            result = run_scoring_task(
                "copa", examples, agent,
                router=router, router_thresholds=router_thresholds,
                baseline_only=args.baseline_only,
                output_dir=output_dir,
            )

        elif task_name == "storycloze":
            examples = maybe_limit_examples(load_storycloze(), args.limit)
            print(f"  Loaded {len(examples)} StoryCloze examples")
            result = run_scoring_task(
                "storycloze", examples, agent,
                router=router, router_thresholds=router_thresholds,
                baseline_only=args.baseline_only,
                output_dir=output_dir,
            )

        elif task_name == "arc_challenge":
            examples = maybe_limit_examples(load_arc_challenge("test"), args.limit)
            print(f"  Loaded {len(examples)} ARC-Challenge examples")
            result = run_scoring_task(
                "arc_challenge", examples, agent,
                router=router, router_thresholds=router_thresholds,
                baseline_only=args.baseline_only,
                output_dir=output_dir,
            )

        elif task_name.startswith("mmlu_"):
            subset_map = {
                "mmlu_abstract_algebra": "abstract_algebra",
                "mmlu_college_math": "college_mathematics",
                "mmlu_college_cs": "college_computer_science",
            }
            subset = subset_map[task_name]
            examples = maybe_limit_examples(load_mmlu(subset, "test"), args.limit)
            print(f"  Loaded {len(examples)} {task_name} examples")
            result = run_scoring_task(
                task_name, examples, agent,
                router=router, router_thresholds=router_thresholds,
                baseline_only=args.baseline_only,
                output_dir=output_dir,
            )

        all_results.append(result)

    # Summary
    if all_results:
        print_summary(all_results, args.model_key)

        # Save combined summary
        summary_path = output_dir / "bench_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "model_key": args.model_key,
                "device": device,
                "router_decision_threshold": args.router_decision_threshold,
                "router_decision_thresholds": router_thresholds,
                "tasks": all_results,
            }, f, indent=2)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
