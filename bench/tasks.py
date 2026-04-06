"""
tasks.py — Benchmark task definitions.

Each task knows how to:
  1. Load its dataset from HuggingFace
  2. Format examples into (context, continuation) pairs for log-likelihood scoring
  3. Score model outputs into a final metric

Evaluation approach:
  - COPA and StoryCloze are log-likelihood tasks: the model scores two
    candidate continuations and picks the one with higher log-prob.
    This maps directly to Agent.score_target().
  - GSM8K is a generation task: the model produces a chain-of-thought
    answer and we extract the final number. This uses greedy decoding
    with a few-shot prompt prefix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass
class ScoringExample:
    """A single log-likelihood comparison example."""
    example_id: str
    context: str
    continuations: list[str]     # candidate completions
    correct_idx: int             # index of the correct continuation
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationExample:
    """A single generation example (e.g., GSM8K)."""
    example_id: str
    prompt: str                  # full prompt including few-shot prefix
    reference_answer: str        # the correct final answer (e.g., "42")
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
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

def load_storycloze(split: str = "2018/val") -> list[ScoringExample]:
    """Load StoryCloze.

    Given a 4-sentence story, choose the correct 5th sentence from two options.

    NOTE: StoryCloze requires accepting a data use agreement on HuggingFace.
    If the dataset is not accessible, you can download it manually from
    https://cs.rochester.edu/nlp/rocstories/ and place it in data/bench/.

    ~1871 validation examples.
    """
    try:
        ds = load_dataset("story_cloze", split.split("/")[0], split=split.split("/")[1])
    except Exception:
        # Fallback: try the LSDSem version
        try:
            ds = load_dataset("LSDSem/story_cloze", split="test")
        except Exception as e:
            raise RuntimeError(
                f"Could not load StoryCloze. You may need to accept the data use "
                f"agreement on HuggingFace or download manually. Error: {e}"
            )

    examples = []
    for i, row in enumerate(ds):
        # Build the 4-sentence context
        context_sentences = [
            row["input_sentence_1"],
            row["input_sentence_2"],
            row["input_sentence_3"],
            row["input_sentence_4"],
        ]
        context = " ".join(context_sentences)

        c1 = " " + row["sentence_quiz1"]
        c2 = " " + row["sentence_quiz2"]

        # answer_right_ending is 1-indexed
        correct_idx = row["answer_right_ending"] - 1

        examples.append(ScoringExample(
            example_id=f"storycloze_{i}",
            context=context,
            continuations=[c1, c2],
            correct_idx=correct_idx,
        ))

    return examples


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

GSM8K_FEW_SHOT_PREFIX = """Solve the following math problem step by step. After your reasoning, write the final answer as a number on a new line after "#### ".

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.
#### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
#### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
#### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.
#### 8

"""


def load_gsm8k(split: str = "test", n_examples: int | None = None) -> list[GenerationExample]:
    """Load GSM8K math problems.

    ~1319 test examples. Each example is a word problem with a
    chain-of-thought solution ending in "#### <number>".

    Args:
        split: dataset split (default: "test")
        n_examples: optional limit on number of examples to load
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    examples = []
    for i, row in enumerate(ds):
        if n_examples is not None and i >= n_examples:
            break

        question = row["question"]
        answer_text = row["answer"]

        # Extract the final numerical answer after ####
        match = re.search(r"####\s*(.+)", answer_text)
        reference = match.group(1).strip().replace(",", "") if match else ""

        prompt = GSM8K_FEW_SHOT_PREFIX + f"Question: {question}\nAnswer:"

        examples.append(GenerationExample(
            example_id=f"gsm8k_{split}_{i}",
            prompt=prompt,
            reference_answer=reference,
            metadata={"question": question, "full_answer": answer_text},
        ))

    return examples


def extract_gsm8k_answer(generated_text: str) -> str | None:
    """Extract the final numerical answer from a GSM8K generation.

    Looks for #### <number> pattern. Returns None if not found.
    """
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", generated_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None
