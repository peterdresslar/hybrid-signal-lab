#!/usr/bin/env python3
"""
Generate algorithmic prompts from curated procedural families.

This generator is intentionally programmatic rather than LLM-driven.
The goal is to create short, deterministic prompts with exact answers
while supporting richer diversity than the original inlined generator.

Usage:
    python algorithmic_generate.py output.json 80
    python algorithmic_generate.py output.json 20 --append
"""

import argparse
import json
import os
import random
from pathlib import Path


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def tier_from_tokens(n: int) -> str:
    if n <= 30:
        return "short"
    if n <= 80:
        return "brief"
    if n <= 200:
        return "med"
    if n <= 500:
        return "long"
    return "extended"


def make_id(idx: int) -> str:
    return f"alg_seed_{idx:04d}"


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def wrap_item(prompt: str, target: str, family: str, concept: str, difficulty: str, source: str) -> dict:
    tok_count = approx_tokens(prompt)
    return {
        "prompt": prompt,
        "target": target if target.startswith(" ") else " " + str(target),
        "type": "algorithmic",
        "tokens_approx": tok_count,
        "tier": tier_from_tokens(tok_count),
        "source": source,
        "metadata": {
            "family": family,
            "concept": concept,
            "difficulty": difficulty,
        },
    }


def gen_arithmetic_chain(rng: random.Random) -> dict:
    a = rng.randint(5, 40)
    b = rng.randint(2, 15)
    c = rng.randint(1, 10)
    result = a + b - c
    prompt = f"Start with {a}. Add {b}. Then subtract {c}. Result:"
    return wrap_item(prompt, str(result), "arithmetic", "three_step_update", "easy", "alg_arithmetic")


def gen_two_stage_arithmetic(rng: random.Random) -> dict:
    nums = rng.sample(range(2, 20), 4)
    answer = (nums[0] + nums[1]) * (nums[2] - nums[3] if nums[2] > nums[3] else nums[3] - nums[2])
    prompt = (
        f"Add {nums[0]} and {nums[1]}. Then multiply that result by the absolute difference "
        f"between {nums[2]} and {nums[3]}. Result:"
    )
    return wrap_item(prompt, str(answer), "arithmetic", "two_stage_arithmetic", "medium", "alg_arithmetic2")


def gen_count_occurrences(rng: random.Random) -> dict:
    word = rng.choice(["red", "blue", "cat", "dog", "sun", "tree"])
    total_words = rng.randint(8, 14)
    positions = sorted(rng.sample(range(total_words), rng.randint(2, 5)))
    words = []
    vocab = ["red", "blue", "cat", "dog", "sun", "tree", "bird", "fish", "leaf", "stone"]
    for i in range(total_words):
        words.append(word if i in positions else rng.choice(vocab))
    actual_count = words.count(word)
    prompt = f'Count how many times "{word}" appears in: ' + " ".join(words) + "\nAnswer:"
    return wrap_item(prompt, str(actual_count), "counting", "token_count", "easy", "alg_counting")


def gen_reverse_word(rng: random.Random) -> dict:
    word = rng.choice(["planet", "window", "silver", "magnet", "forest", "bridge"])
    prompt = f"Reverse this word: {word}\nAnswer:"
    return wrap_item(prompt, word[::-1], "string_transform", "reverse_word", "easy", "alg_reverse")


def gen_first_letters(rng: random.Random) -> dict:
    words = rng.sample(["red", "blue", "green", "yellow", "black", "white", "orange"], 3)
    answer = "".join(w[0] for w in words)
    prompt = "Take the first letter of each word and combine them: " + ", ".join(words) + "\nAnswer:"
    return wrap_item(prompt, answer, "string_transform", "first_letters", "easy", "alg_initials")


def gen_last_letter(rng: random.Random) -> dict:
    word = rng.choice(["planet", "window", "silver", "magnet", "forest", "bridge"])
    prompt = f"What is the last letter of the word {word}?\nAnswer:"
    return wrap_item(prompt, word[-1], "string_transform", "last_letter", "easy", "alg_last_letter")


def gen_every_second_reverse(rng: random.Random) -> dict:
    words = rng.sample(["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"], 6)
    selected = words[1::2]
    selected.reverse()
    answer = selected[1]
    prompt = (
        "Take every second word starting from the second item in this list, reverse that smaller list, "
        f"and return its {ordinal(2)} item: " + ", ".join(words) + "\nAnswer:"
    )
    return wrap_item(prompt, answer, "string_transform", "every_second_reverse", "medium", "alg_string_rule")


def gen_select_extreme(rng: random.Random) -> dict:
    nums = rng.sample(range(1, 40), 5)
    choose_max = rng.choice([True, False])
    answer = max(nums) if choose_max else min(nums)
    mode = "largest" if choose_max else "smallest"
    prompt = "Choose the " + mode + " number from: " + ", ".join(map(str, nums)) + "\nAnswer:"
    return wrap_item(prompt, str(answer), "selection", f"{mode}_value", "easy", "alg_select")


def gen_select_after_transform(rng: random.Random) -> dict:
    nums = rng.sample(range(2, 20), 5)
    transformed = [n * 2 if n % 2 == 0 else n + 3 for n in nums]
    answer = max(transformed)
    prompt = (
        "Transform each number using this rule: if it is even, double it; if it is odd, add 3. "
        "Then give the largest transformed value from: "
        + ", ".join(map(str, nums))
        + "\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "selection", "largest_after_transform", "medium", "alg_select_transform")


def gen_sort_position(rng: random.Random) -> dict:
    nums = rng.sample(range(1, 30), 5)
    idx = rng.randint(1, 3)
    ordered = sorted(nums)
    prompt = (
        "Sort these numbers in ascending order and give the "
        f"{ordinal(idx + 1)} number: " + ", ".join(map(str, nums)) + "\nAnswer:"
    )
    return wrap_item(prompt, str(ordered[idx]), "sorting", "sorted_position", "medium", "alg_sort")


def gen_sort_dedup_position(rng: random.Random) -> dict:
    base = rng.sample(range(1, 20), 5)
    seq = base + [rng.choice(base), rng.choice(base)]
    rng.shuffle(seq)
    deduped = sorted(set(seq))
    idx = min(len(deduped) - 1, rng.randint(1, 3))
    prompt = (
        "Remove duplicate numbers, sort the remaining values in ascending order, and give the "
        f"{ordinal(idx + 1)} number from: " + ", ".join(map(str, seq)) + "\nAnswer:"
    )
    return wrap_item(prompt, str(deduped[idx]), "sorting", "sort_dedup_position", "medium", "alg_sort_dedup")


def gen_parity_divisibility(rng: random.Random) -> dict:
    n = rng.randint(10, 99)
    divisor = rng.choice([2, 3, 4, 5, 6])
    answer = " yes" if n % divisor == 0 else " no"
    prompt = f"Is {n} divisible by {divisor}? Answer yes or no.\nAnswer:"
    return wrap_item(prompt, answer, "parity_divisibility", "divisible_check", "easy", "alg_divisible")


def gen_parity_sum_rule(rng: random.Random) -> dict:
    nums = rng.sample(range(1, 15), 6)
    answer = sum((n // 2) if n % 2 == 0 else (n + 1) for n in nums)
    prompt = (
        "For each number, use this rule: if it is even, halve it; if it is odd, add 1. "
        "Then sum the results for: " + ", ".join(map(str, nums)) + "\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "parity_divisibility", "parity_sum_rule", "medium", "alg_parity_rule")


def gen_filter_sum(rng: random.Random) -> dict:
    nums = rng.sample(range(1, 20), 6)
    threshold = rng.randint(6, 12)
    answer = sum(n for n in nums if n > threshold)
    prompt = (
        "Sum only the numbers greater than "
        f"{threshold} from: " + ", ".join(map(str, nums)) + "\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "filtering", "sum_above_threshold", "medium", "alg_filter_sum")


def gen_filter_count_dual_rule(rng: random.Random) -> dict:
    nums = rng.sample(range(1, 25), 8)
    answer = sum(1 for n in nums if n > 8 and n % 2 == 0)
    prompt = (
        "Count how many numbers are both greater than 8 and even in: "
        + ", ".join(map(str, nums))
        + "\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "filtering", "count_even_above_threshold", "medium", "alg_filter_count")


def gen_position_extraction(rng: random.Random) -> dict:
    words = rng.sample(["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta"], 5)
    idx = rng.randint(1, 3)
    prompt = f"What is the word in position {idx + 1} of this list: " + ", ".join(words) + "\nAnswer:"
    return wrap_item(prompt, words[idx], "extraction", "nth_item", "easy", "alg_extract")


def gen_extraction_after_rotation(rng: random.Random) -> dict:
    words = rng.sample(["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta"], 5)
    shift = rng.randint(1, 3)
    rotated = words[shift:] + words[:shift]
    idx = rng.randint(1, 3)
    prompt = (
        f"Rotate this list left by {shift} positions and give the {ordinal(idx + 1)} item: "
        + ", ".join(words)
        + "\nAnswer:"
    )
    return wrap_item(prompt, rotated[idx], "extraction", "rotation_then_extract", "medium", "alg_extract_rotate")


def gen_running_transform(rng: random.Random) -> dict:
    start = rng.randint(2, 10)
    ops = [
        ("double", lambda x: x * 2),
        ("add 3", lambda x: x + 3),
        ("subtract 4", lambda x: x - 4),
        ("halve", lambda x: x // 2),
    ]
    chosen = rng.sample(ops, 3)
    value = start
    steps = [f"Start with {start}."]
    for label, fn in chosen:
        value = fn(value)
        steps.append(label.capitalize() + ".")
    prompt = " ".join(steps) + " Result:"
    return wrap_item(prompt, str(value), "transforms", "running_transform", "medium", "alg_transform")


def gen_running_rule_list(rng: random.Random) -> dict:
    nums = rng.sample(range(1, 12), 5)
    transformed = []
    for n in nums:
        if n % 2 == 0:
            transformed.append(n + 4)
        else:
            transformed.append(n * 2)
    answer = transformed[-1] - transformed[0]
    prompt = (
        "Transform each number using this rule: if it is even, add 4; if it is odd, double it. "
        "Then subtract the first transformed value from the last transformed value for: "
        + ", ".join(map(str, nums))
        + "\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "transforms", "list_rule_then_difference", "medium", "alg_transform_list")


def gen_remove_duplicates(rng: random.Random) -> dict:
    base = rng.sample(["red", "blue", "green", "yellow", "black"], 4)
    seq = base + [rng.choice(base), rng.choice(base)]
    rng.shuffle(seq)
    deduped = []
    for item in seq:
        if item not in deduped:
            deduped.append(item)
    prompt = "Remove duplicates while keeping first occurrence order: " + ", ".join(seq) + "\nHow many items remain?"
    return wrap_item(prompt, str(len(deduped)), "deduplication", "count_unique_ordered", "medium", "alg_dedup")


def gen_symbol_mapping(rng: random.Random) -> dict:
    mapping = {"A": 2, "B": 3, "C": 5, "D": 7}
    seq = rng.choices(list(mapping.keys()), k=5)
    answer = sum(mapping[s] for s in seq)
    prompt = (
        "Use this mapping A=2, B=3, C=5, D=7. Sum the values for: "
        + ", ".join(seq)
        + "\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "mapping", "symbol_sum", "medium", "alg_mapping")


FAMILIES = [
    ("arithmetic", 2, gen_arithmetic_chain),
    ("arithmetic", 3, gen_two_stage_arithmetic),
    ("counting", 2, gen_count_occurrences),
    ("string_transform", 1, gen_reverse_word),
    ("string_transform", 2, gen_first_letters),
    ("string_transform", 1, gen_last_letter),
    ("string_transform", 3, gen_every_second_reverse),
    ("selection", 1, gen_select_extreme),
    ("selection", 3, gen_select_after_transform),
    ("sorting", 3, gen_sort_position),
    ("sorting", 4, gen_sort_dedup_position),
    ("parity_divisibility", 1, gen_parity_divisibility),
    ("parity_divisibility", 3, gen_parity_sum_rule),
    ("filtering", 3, gen_filter_sum),
    ("filtering", 3, gen_filter_count_dual_rule),
    ("extraction", 1, gen_position_extraction),
    ("extraction", 3, gen_extraction_after_rotation),
    ("transforms", 3, gen_running_transform),
    ("transforms", 4, gen_running_rule_list),
    ("deduplication", 3, gen_remove_duplicates),
    ("mapping", 3, gen_symbol_mapping),
]


def family_limit(num_prompts: int) -> int:
    return max(4, (num_prompts + 4) // 5)


def concept_limit(num_prompts: int) -> int:
    return max(3, (num_prompts + 24) // 25)


def existing_prompts(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    try:
        with open(output_path) as f:
            items = json.load(f)
        return {item["prompt"] for item in items}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return set()


def generate_items(num_prompts: int, seed: int, existing: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    seen_prompts = set(existing or set())
    family_counts: dict[str, int] = {}
    concept_counts: dict[tuple[str, str], int] = {}
    max_per_family = family_limit(num_prompts)
    max_per_concept = concept_limit(num_prompts)
    attempts = 0
    max_attempts = max(80, num_prompts * 12)

    while len(items) < num_prompts and attempts < max_attempts:
        attempts += 1
        eligible = [(family, weight, fn) for family, weight, fn in FAMILIES if family_counts.get(family, 0) < max_per_family]
        if not eligible:
            eligible = FAMILIES
        families = [family for family, _, _ in eligible]
        weights = [weight for _, weight, _ in eligible]
        family, _, fn = rng.choices(eligible, weights=weights, k=1)[0]
        item = fn(rng)
        concept_key = (item["metadata"]["family"], item["metadata"]["concept"])
        if item["prompt"] in seen_prompts:
            continue
        if concept_counts.get(concept_key, 0) >= max_per_concept:
            continue

        seen_prompts.add(item["prompt"])
        family_counts[family] = family_counts.get(family, 0) + 1
        concept_counts[concept_key] = concept_counts.get(concept_key, 0) + 1
        items.append(item)

    for idx, item in enumerate(items):
        item["id"] = make_id(idx)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate algorithmic prompts from procedural families.")
    parser.add_argument("output", type=str, help="Output JSON path")
    parser.add_argument("num_prompts", type=int, help="Number of prompts to generate")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()

    current_items = []
    existing = set()
    if args.append and os.path.exists(args.output):
        with open(args.output) as f:
            current_items = json.load(f)
        existing = {item["prompt"] for item in current_items}

    new_items = generate_items(
        num_prompts=args.num_prompts,
        seed=args.seed,
        existing=existing,
    )

    all_items = current_items + new_items
    for idx, item in enumerate(all_items):
        item["id"] = make_id(idx)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_items, f, indent=2)

    print(f"Wrote {len(all_items)} total items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
