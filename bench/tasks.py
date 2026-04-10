"""
tasks.py — Benchmark task definitions.

Each task knows how to:
  1. Load its dataset from HuggingFace
  2. Format examples into (context, continuation) pairs for log-likelihood scoring
  3. Score model outputs into a final metric

Evaluation approach:
  - All active benchmarks are log-likelihood tasks.
  - The model scores each candidate continuation and picks the one with
    higher total log-prob.
  - This maps directly to Agent.score_target().
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from datasets import load_dataset


@dataclass
class ScoringExample:
    """A single log-likelihood comparison example."""
    example_id: str
    context: str
    continuations: list[str]     # candidate completions
    correct_idx: int             # index of the correct continuation
    metadata: dict = field(default_factory=dict)


# COPA
# ---------------------------------------------------------------------------

def load_copa(split: str = "validation") -> list[ScoringExample]:
    """Load COPA from SuperGLUE.

    COPA: Choice of Plausible Alternatives.
    Given a premise and a question (cause or effect), choose between
    two alternatives.

    ~100 validation examples, ~400 train, ~500 test (test labels hidden).
    """
    ds = load_dataset("super_glue", "copa", split=split)
    examples = []

    for i, row in enumerate(ds):
        premise = row["premise"].strip()
        question = row["question"]  # "cause" or "effect"

        if question == "cause":
            context = f"{premise} This happened because"
        else:
            context = f"{premise} As a result,"

        c1 = " " + row["choice1"][0].lower() + row["choice1"][1:]
        c2 = " " + row["choice2"][0].lower() + row["choice2"][1:]

        examples.append(ScoringExample(
            example_id=f"copa_{split}_{i}",
            context=context,
            continuations=[c1, c2],
            correct_idx=row["label"],
            metadata={"question_type": question},
        ))

    return examples


# ---------------------------------------------------------------------------
# StoryCloze
# ---------------------------------------------------------------------------

def load_storycloze(split: str = "test", seed: int = 42) -> list[ScoringExample]:
    """Load StoryCloze from lecslab/story_cloze on HuggingFace.

    Given a multi-sentence story prompt, choose the correct ending from
    two options (chosen vs. rejected). Publicly accessible, no data-use
    agreement required.

    The dataset has columns: prompt, chosen, rejected.
    We randomize continuation order per-example (seeded) so position
    bias doesn't contaminate log-likelihood evaluation.

    Splits available: train (~1871), validation (~187), test (~1684).

    Args:
        split: dataset split (default: "test")
        seed: random seed for continuation order shuffling
    """
    ds = load_dataset("lecslab/story_cloze", split=split)

    rng = random.Random(seed)
    examples = []

    for i, row in enumerate(ds):
        context = row["prompt"].strip()
        chosen = " " + row["chosen"].strip()
        rejected = " " + row["rejected"].strip()

        # Randomize order to avoid position bias; track correct index
        if rng.random() < 0.5:
            continuations = [chosen, rejected]
            correct_idx = 0
        else:
            continuations = [rejected, chosen]
            correct_idx = 1

        examples.append(ScoringExample(
            example_id=f"storycloze_{split}_{i}",
            context=context,
            continuations=continuations,
            correct_idx=correct_idx,
        ))

    return examples


# ---------------------------------------------------------------------------
# ARC-Challenge
# ---------------------------------------------------------------------------

def load_arc_challenge(split: str = "test") -> list[ScoringExample]:
    """Load ARC-Challenge multiple-choice science questions."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    examples = []

    for i, row in enumerate(ds):
        question = row["question"].strip()
        labels = row["choices"]["label"]
        choice_texts = row["choices"]["text"]
        answer_key = row["answerKey"]

        if answer_key not in labels:
            continue

        continuations = [" " + choice.strip() for choice in choice_texts]
        correct_idx = labels.index(answer_key)

        examples.append(ScoringExample(
            example_id=f"arc_challenge_{split}_{i}",
            context=f"{question}\nAnswer:",
            continuations=continuations,
            correct_idx=correct_idx,
            metadata={
                "labels": labels,
                "answer_key": answer_key,
            },
        ))

    return examples


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

def load_mmlu(subset: str, split: str = "test") -> list[ScoringExample]:
    """Load a single MMLU subset using standard letter-scoring format."""
    ds = load_dataset("cais/mmlu", subset, split=split)
    examples = []

    for i, row in enumerate(ds):
        question = row["question"].strip()
        choices = [choice.strip() for choice in row["choices"]]
        answer = int(row["answer"])

        if len(choices) != 4:
            continue
        if not 0 <= answer < len(choices):
            continue

        context_lines = [question]
        for label, choice in zip(("A", "B", "C", "D"), choices):
            context_lines.append(f"{label}. {choice}")
        context_lines.append("Answer:")

        examples.append(ScoringExample(
            example_id=f"mmlu_{subset}_{split}_{i}",
            context="\n".join(context_lines),
            continuations=[" A", " B", " C", " D"],
            correct_idx=answer,
            metadata={"subset": subset, "choices": choices},
        ))

    return examples
