"""
run_bench.py — Run benchmarks with baseline and routed inference.

Evaluates COPA, StoryCloze, and GSM8K under four conditions:
  1. Baseline (g=1.0, no intervention)
  2. Best fixed profile (single best gain profile applied to all prompts)
  3. Routed (two-pass: baseline → classify → intervene)
  4. Oracle (for log-likelihood tasks: try all 4 profiles, pick best per-example)

For log-likelihood tasks (COPA, StoryCloze):
  - Each candidate continuation is scored via Agent.score_target()
  - The model picks whichever continuation has higher total log-prob
  - Metric: accuracy (% of examples where correct continuation is chosen)

For generation tasks (GSM8K):
  - The model generates a chain-of-thought answer via greedy decoding
  - The final numerical answer is extracted and compared to the reference
  - Metric: accuracy (% correct)

Usage:
    python -m bench.run_bench \
        --model-key 9B \
        --tasks copa storycloze gsm8k \
        --router-model data/intervention_modes/b4_021_attn_contr/9B/router/router_model.json \
        --output-dir data/bench/9B

    For baseline-only (overnight run, no router needed):
    python -m bench.run_bench \
        --model-key 9B \
        --tasks copa storycloze \
        --baseline-only \
        --output-dir data/bench/9B
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import dotenv

from signal_lab.agent import Agent
from model.backend import InterventionMode
from model.g_profile import build_attention_scales_from_spec

from bench.tasks import (
    load_copa,
    load_storycloze,
    load_gsm8k,
    extract_gsm8k_answer,
    ScoringExample,
    GenerationExample,
)
from router.router import InterventionRouter
from router.profiles import get_profile_specs, BASELINE_SPEC

dotenv.load_dotenv(".env.development")
dotenv.load_dotenv(".env")
dotenv.load_dotenv(".env.slurm")


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
) -> dict[str, Any]:
    """Run a single scoring example under baseline conditions."""
    scores = []
    for cont in example.continuations:
        lp = score_continuation(agent, example.context, cont, baseline_scales)
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
        intervention_mode=InterventionMode.ATTENTION_CONTRIBUTION,
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
        lp = score_continuation(agent, example.context, cont, scales)
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


def run_scoring_example_fixed(
    agent: Agent,
    example: ScoringExample,
    profile_name: str,
    scales: np.ndarray,
) -> dict[str, Any]:
    """Run a single scoring example under a fixed profile."""
    scores = []
    for cont in example.continuations:
        lp = score_continuation(agent, example.context, cont, scales)
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
) -> dict[str, Any]:
    """Run a single scoring example under oracle routing.

    Tries baseline + all profiles, picks whichever gets it right
    (or if multiple get it right, whichever has the largest margin).
    """
    all_results = {}

    # Baseline
    scores_bl = []
    for cont in example.continuations:
        scores_bl.append(score_continuation(agent, example.context, cont, baseline_scales))
    all_results["baseline"] = scores_bl

    # Each profile
    for pname, pscales in profile_scales.items():
        scores_p = []
        for cont in example.continuations:
            scores_p.append(score_continuation(agent, example.context, cont, pscales))
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
# Generation helpers (GSM8K)
# ---------------------------------------------------------------------------

def run_generation_example_baseline(
    agent: Agent,
    example: GenerationExample,
    baseline_scales: np.ndarray,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Run a single generation example under baseline conditions."""
    result = agent.generate(
        example.prompt,
        baseline_scales,
        max_new_tokens=max_new_tokens,
        intervention_mode=InterventionMode.ATTENTION_CONTRIBUTION,
    )
    predicted = extract_gsm8k_answer(result["generated_text"])
    correct = predicted is not None and predicted == example.reference_answer

    return {
        "example_id": example.example_id,
        "condition": "baseline",
        "generated_text": result["generated_text"],
        "predicted_answer": predicted,
        "reference_answer": example.reference_answer,
        "correct": correct,
        "elapsed": result["elapsed_time"],
    }


def run_generation_example_fixed(
    agent: Agent,
    example: GenerationExample,
    profile_name: str,
    scales: np.ndarray,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Run a single generation example under a fixed profile."""
    result = agent.generate(
        example.prompt,
        scales,
        max_new_tokens=max_new_tokens,
        intervention_mode=InterventionMode.ATTENTION_CONTRIBUTION,
    )
    predicted = extract_gsm8k_answer(result["generated_text"])
    correct = predicted is not None and predicted == example.reference_answer

    return {
        "example_id": example.example_id,
        "condition": f"fixed_{profile_name}",
        "generated_text": result["generated_text"],
        "predicted_answer": predicted,
        "reference_answer": example.reference_answer,
        "correct": correct,
        "elapsed": result["elapsed_time"],
    }


def run_generation_example_routed(
    agent: Agent,
    router: InterventionRouter,
    example: GenerationExample,
    baseline_scales: np.ndarray,
    profile_scales: dict[str, np.ndarray],
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Run a single generation example with routing.

    1. Baseline sensing pass (verbose) on the prompt
    2. Router classifies → picks profile or "off"
    3. Generate under the selected profile
    """
    # Sensing pass
    baseline_result = agent.run_pass(
        example.prompt,
        baseline_scales,
        return_verbose=True,
        intervention_mode=InterventionMode.ATTENTION_CONTRIBUTION,
    )

    decision = router.classify(baseline_result)

    if decision["is_off"]:
        scales = baseline_scales
    else:
        scales = profile_scales[decision["profile_name"]]

    result = agent.generate(
        example.prompt,
        scales,
        max_new_tokens=max_new_tokens,
        intervention_mode=InterventionMode.ATTENTION_CONTRIBUTION,
    )
    predicted = extract_gsm8k_answer(result["generated_text"])
    correct = predicted is not None and predicted == example.reference_answer

    return {
        "example_id": example.example_id,
        "condition": "routed",
        "profile": decision["profile_name"],
        "confidence": decision["confidence"],
        "generated_text": result["generated_text"],
        "predicted_answer": predicted,
        "reference_answer": example.reference_answer,
        "correct": correct,
        "elapsed": result["elapsed_time"],
    }


def run_generation_example_oracle(
    agent: Agent,
    example: GenerationExample,
    baseline_scales: np.ndarray,
    profile_scales: dict[str, np.ndarray],
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Run a single generation example under oracle routing.

    Tries baseline + all profiles. Picks whichever gets the correct answer.
    If multiple get it right, picks baseline (conservative). If none get it
    right, returns the baseline result.
    """
    # Baseline first
    bl_result = agent.generate(
        example.prompt,
        baseline_scales,
        max_new_tokens=max_new_tokens,
        intervention_mode=InterventionMode.ATTENTION_CONTRIBUTION,
    )
    bl_predicted = extract_gsm8k_answer(bl_result["generated_text"])
    bl_correct = bl_predicted is not None and bl_predicted == example.reference_answer

    if bl_correct:
        # Baseline already correct — oracle picks baseline
        return {
            "example_id": example.example_id,
            "condition": "oracle",
            "oracle_profile": "baseline",
            "generated_text": bl_result["generated_text"],
            "predicted_answer": bl_predicted,
            "reference_answer": example.reference_answer,
            "correct": True,
            "elapsed": bl_result["elapsed_time"],
        }

    # Try each profile
    for pname, pscales in profile_scales.items():
        p_result = agent.generate(
            example.prompt,
            pscales,
            max_new_tokens=max_new_tokens,
            intervention_mode=InterventionMode.ATTENTION_CONTRIBUTION,
        )
        p_predicted = extract_gsm8k_answer(p_result["generated_text"])
        p_correct = p_predicted is not None and p_predicted == example.reference_answer

        if p_correct:
            return {
                "example_id": example.example_id,
                "condition": "oracle",
                "oracle_profile": pname,
                "generated_text": p_result["generated_text"],
                "predicted_answer": p_predicted,
                "reference_answer": example.reference_answer,
                "correct": True,
                "elapsed": p_result["elapsed_time"],
            }

    # None got it right — return baseline result
    return {
        "example_id": example.example_id,
        "condition": "oracle",
        "oracle_profile": "baseline",
        "generated_text": bl_result["generated_text"],
        "predicted_answer": bl_predicted,
        "reference_answer": example.reference_answer,
        "correct": False,
        "elapsed": bl_result["elapsed_time"],
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
    baseline_only: bool = False,
    output_dir: Path,
) -> dict[str, Any]:
    """Run a full log-likelihood scoring task."""
    n_attn = len(agent.get_attention_layer_indices())
    baseline_scales = build_attention_scales_from_spec(BASELINE_SPEC, attention_slots=n_attn)

    # Build profile scales
    profile_scales = {}
    if router and not baseline_only:
        specs = get_profile_specs(router.model_key)
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
        rec = run_scoring_example_baseline(agent, ex, baseline_scales)
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
        print(f"\n  [{task_name}] Running routed...")
        t0 = time.time()
        routed_records = []
        routed_heartbeat = ProgressHeartbeat(task_name, "routed", len(examples))
        for i, ex in enumerate(examples):
            rec = run_scoring_example_routed(agent, router, ex, baseline_scales, profile_scales)
            routed_records.append(rec)
            acc_so_far = sum(r["correct"] for r in routed_records) / len(routed_records)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(examples)}  acc={acc_so_far:.3f}")
            routed_heartbeat.maybe_ping(i + 1, metric_value=acc_so_far)

        rt_acc = sum(r["correct"] for r in routed_records) / len(routed_records)
        results["conditions"]["routed"] = {
            "accuracy": rt_acc,
            "n_correct": sum(r["correct"] for r in routed_records),
            "elapsed": time.time() - t0,
            "profile_distribution": _count_profiles(routed_records),
        }
        all_records.extend(routed_records)
        print(f"  [{task_name}] Routed:   {rt_acc:.3f} ({time.time()-t0:.1f}s)")

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
                rec = run_scoring_example_fixed(agent, ex, pname, pscales)
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
            rec = run_scoring_example_oracle(agent, ex, baseline_scales, profile_scales)
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


def run_generation_task(
    task_name: str,
    examples: list[GenerationExample],
    agent: Agent,
    *,
    router: InterventionRouter | None = None,
    baseline_only: bool = False,
    max_new_tokens: int = 512,
    output_dir: Path,
) -> dict[str, Any]:
    """Run a full generation task (e.g., GSM8K)."""
    n_attn = len(agent.get_attention_layer_indices())
    baseline_scales = build_attention_scales_from_spec(BASELINE_SPEC, attention_slots=n_attn)

    profile_scales = {}
    if router and not baseline_only:
        specs = get_profile_specs(router.model_key)
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
        rec = run_generation_example_baseline(agent, ex, baseline_scales, max_new_tokens)
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
        print(f"\n  [{task_name}] Running routed...")
        t0 = time.time()
        routed_records = []
        routed_heartbeat = ProgressHeartbeat(task_name, "routed", len(examples))
        for i, ex in enumerate(examples):
            rec = run_generation_example_routed(
                agent, router, ex, baseline_scales, profile_scales, max_new_tokens,
            )
            routed_records.append(rec)
            acc_so_far = sum(r["correct"] for r in routed_records) / len(routed_records)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(examples)}  acc={acc_so_far:.3f}")
            routed_heartbeat.maybe_ping(i + 1, metric_value=acc_so_far)

        rt_acc = sum(r["correct"] for r in routed_records) / len(routed_records)
        results["conditions"]["routed"] = {
            "accuracy": rt_acc,
            "n_correct": sum(r["correct"] for r in routed_records),
            "elapsed": time.time() - t0,
            "profile_distribution": _count_profiles(routed_records),
        }
        all_records.extend(routed_records)
        print(f"  [{task_name}] Routed:   {rt_acc:.3f} ({time.time()-t0:.1f}s)")

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
                rec = run_generation_example_fixed(agent, ex, pname, pscales, max_new_tokens)
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
            rec = run_generation_example_oracle(
                agent, ex, baseline_scales, profile_scales, max_new_tokens,
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


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict], model_key: str):
    """Print a summary table across all tasks."""
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK SUMMARY: {model_key}")
    print(f"{'=' * 70}")
    print(f"  {'Task':<15} {'N':>5} {'Baseline':>10} {'Best Fix':>10} {'Routed':>10} {'Oracle':>10}")
    print(f"  {'-' * 62}")

    for r in all_results:
        task = r["task"]
        n = r["n_examples"]
        bl = r["conditions"].get("baseline", {}).get("accuracy", "—")
        rt = r["conditions"].get("routed", {}).get("accuracy", "—")
        ora = r["conditions"].get("oracle", {}).get("accuracy", "—")
        bf = r.get("best_fixed", {}).get("accuracy", "—")

        bl_s = f"{bl:.3f}" if isinstance(bl, float) else bl
        bf_s = f"{bf:.3f}" if isinstance(bf, float) else bf
        rt_s = f"{rt:.3f}" if isinstance(rt, float) else rt
        ora_s = f"{ora:.3f}" if isinstance(ora, float) else ora

        print(f"  {task:<15} {n:>5} {bl_s:>10} {bf_s:>10} {rt_s:>10} {ora_s:>10}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks.")
    parser.add_argument("--model-key", required=True, help="Model key (9B, OLMO)")
    parser.add_argument("--tasks", nargs="+", default=["copa", "storycloze"],
                        choices=["copa", "storycloze", "gsm8k"],
                        help="Which benchmarks to run")
    parser.add_argument("--router-model", default=None,
                        help="Path to router_model.json (omit for baseline-only)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only baseline condition (no routing)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for results")
    parser.add_argument("--device", default="auto",
                        help="Device (cuda, cuda:0, mps, cpu, auto)")
    parser.add_argument("--gsm8k-limit", type=int, default=None,
                        help="Limit GSM8K examples (for testing)")
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
        print(f"  {router}")

    all_results = []

    # Run tasks
    for task_name in args.tasks:
        print(f"\n{'=' * 50}")
        print(f"  Task: {task_name}")
        print(f"{'=' * 50}")

        if task_name == "copa":
            examples = load_copa("validation")
            print(f"  Loaded {len(examples)} COPA examples")
            result = run_scoring_task(
                "copa", examples, agent,
                router=router, baseline_only=args.baseline_only,
                output_dir=output_dir,
            )

        elif task_name == "storycloze":
            examples = load_storycloze()
            print(f"  Loaded {len(examples)} StoryCloze examples")
            result = run_scoring_task(
                "storycloze", examples, agent,
                router=router, baseline_only=args.baseline_only,
                output_dir=output_dir,
            )

        elif task_name == "gsm8k":
            examples = load_gsm8k("test", n_examples=args.gsm8k_limit)
            print(f"  Loaded {len(examples)} GSM8K examples")
            result = run_generation_task(
                "gsm8k", examples, agent,
                router=router, baseline_only=args.baseline_only,
                max_new_tokens=512,
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
                "tasks": all_results,
            }, f, indent=2)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
