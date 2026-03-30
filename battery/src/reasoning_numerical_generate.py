#!/usr/bin/env python3
"""
Generate reasoning_numerical prompts from curated numerical reasoning families.

The goal is to keep the category more reasoning-oriented than the algorithmic
generator while increasing diversity beyond a single monolithic family.

Usage:
    python reasoning_numerical_generate.py output.json 90
    python reasoning_numerical_generate.py output.json 20 --append
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
    return f"rn_seed_{idx:04d}"


def wrap_item(prompt: str, target: str, family: str, concept: str, difficulty: str, source: str) -> dict:
    tok_count = approx_tokens(prompt)
    return {
        "prompt": prompt,
        "target": target if target.startswith(" ") else " " + str(target),
        "type": "reasoning_numerical",
        "tokens_approx": tok_count,
        "tier": tier_from_tokens(tok_count),
        "source": source,
        "metadata": {
            "family": family,
            "concept": concept,
            "difficulty": difficulty,
        },
    }


def gen_multi_step_word_problem(rng: random.Random) -> dict:
    start = rng.randint(10, 40)
    bought = rng.randint(3, 12)
    gave = rng.randint(1, min(8, start + bought - 1))
    left = start + bought - gave
    item = rng.choice(["marbles", "coins", "stickers", "books"])
    person = rng.choice(["Mia", "Noah", "Lena", "Owen", "Sara"])
    prompt = (
        f"{person} had {start} {item}. Then {person.lower()} got {bought} more and later gave away {gave}. "
        f"How many {item} does {person.lower()} have now?\nAnswer:"
    )
    return wrap_item(prompt, str(left), "word_problem", "two_step_quantity_change", "medium", "rn_word")


def gen_rate_time_distance(rng: random.Random) -> dict:
    speed = rng.choice([20, 30, 40, 50, 60])
    hours = rng.randint(2, 6)
    dist = speed * hours
    vehicle = rng.choice(["train", "bus", "boat", "car"])
    prompt = (
        f"A {vehicle} travels at {speed} miles per hour for {hours} hours. "
        "How far does it travel?\nAnswer:"
    )
    return wrap_item(prompt, str(dist), "rate", "distance_from_rate_time", "easy", "rn_rate")


def gen_remaining_after_groups(rng: random.Random) -> dict:
    total = rng.randint(20, 60)
    group = rng.randint(3, 8)
    used_groups = rng.randint(2, 5)
    used = group * used_groups
    remaining = total - used
    thing = rng.choice(["apples", "cards", "tokens", "blocks"])
    prompt = (
        f"There are {total} {thing}. {used_groups} groups of {group} are used. "
        f"How many {thing} remain?\nAnswer:"
    )
    return wrap_item(prompt, str(remaining), "grouping", "remaining_after_groups", "medium", "rn_group")


def gen_compare_totals(rng: random.Random) -> dict:
    a1, a2 = rng.randint(5, 20), rng.randint(5, 20)
    b1, b2 = rng.randint(5, 20), rng.randint(5, 20)
    total_a = a1 + a2
    total_b = b1 + b2
    answer = " first" if total_a > total_b else " second" if total_b > total_a else " equal"
    prompt = (
        f"First total: {a1} + {a2}. Second total: {b1} + {b2}. "
        "Which is larger: first, second, or equal?\nAnswer:"
    )
    return wrap_item(prompt, answer, "comparison", "compare_two_totals", "easy", "rn_compare")


def gen_discount_price(rng: random.Random) -> dict:
    price = rng.choice([20, 25, 30, 40, 50, 60, 80])
    pct = rng.choice([10, 20, 25, 50])
    final = price - (price * pct // 100)
    prompt = f"An item costs {price} dollars and is discounted by {pct}%. What is the final price?\nAnswer:"
    return wrap_item(prompt, str(final), "percentages", "discount_price", "medium", "rn_percent")


def gen_ratio_sharing(rng: random.Random) -> dict:
    total = rng.choice([18, 24, 30, 36, 42])
    a = rng.randint(1, 4)
    b = rng.randint(1, 4)
    share_a = total * a // (a + b)
    prompt = (
        f"A total of {total} points is shared in the ratio {a}:{b}. "
        "How many points does the first part receive?\nAnswer:"
    )
    return wrap_item(prompt, str(share_a), "ratios", "ratio_share", "medium", "rn_ratio")


def gen_signed_change(rng: random.Random) -> dict:
    temp = rng.randint(-5, 15)
    up = rng.randint(3, 10)
    down = rng.randint(2, 8)
    final = temp + up - down
    prompt = (
        f"The temperature starts at {temp} degrees, rises by {up}, then falls by {down}. "
        "What is the final temperature?\nAnswer:"
    )
    return wrap_item(prompt, str(final), "signed_numbers", "signed_change", "medium", "rn_signed")


def gen_average_of_values(rng: random.Random) -> dict:
    nums = rng.sample(range(4, 20), 4)
    total = sum(nums)
    if total % 4 != 0:
        nums[-1] += 4 - (total % 4)
    avg = sum(nums) // 4
    prompt = "What is the average of these numbers: " + ", ".join(map(str, nums)) + "\nAnswer:"
    return wrap_item(prompt, str(avg), "averages", "average_of_four", "medium", "rn_average")


def gen_sequence_accumulation(rng: random.Random) -> dict:
    start = rng.randint(2, 10)
    step = rng.randint(2, 6)
    length = rng.randint(4, 6)
    seq = [start + i * step for i in range(length)]
    answer = seq[0] + seq[-1]
    prompt = (
        "A sequence starts at "
        f"{start} and increases by {step}. After writing {length} numbers, add the first and last. Result:\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "sequences", "first_last_sum", "medium", "rn_sequence")


def gen_constraint_count(rng: random.Random) -> dict:
    nums = rng.sample(range(1, 25), 8)
    answer = sum(1 for n in nums if n > 10 and n % 3 == 0)
    prompt = (
        "How many numbers are both greater than 10 and divisible by 3 in: "
        + ", ".join(map(str, nums))
        + "\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "constraint_counting", "count_divisible_and_large", "medium", "rn_constraints")


def gen_modulo_reasoning(rng: random.Random) -> dict:
    n = rng.randint(20, 120)
    divisor = rng.choice([3, 4, 5, 6, 7, 8, 9])
    answer = n % divisor
    prompt = f"What is the remainder when {n} is divided by {divisor}?\nAnswer:"
    return wrap_item(prompt, str(answer), "modulo", "remainder", "medium", "rn_modulo")


def gen_unit_conversion(rng: random.Random) -> dict:
    if rng.choice([True, False]):
        meters = rng.randint(2, 25)
        answer = meters * 100
        prompt = f"Convert {meters} meters to centimeters.\nAnswer:"
    else:
        hours = rng.randint(2, 12)
        answer = hours * 60
        prompt = f"Convert {hours} hours to minutes.\nAnswer:"
    return wrap_item(prompt, str(answer), "unit_conversion", "simple_unit_conversion", "easy", "rn_units")


def gen_ordering_gap(rng: random.Random) -> dict:
    nums = rng.sample(range(5, 50), 5)
    ordered = sorted(nums)
    answer = ordered[-1] - ordered[0]
    prompt = (
        "Find the difference between the largest and smallest numbers in: "
        + ", ".join(map(str, nums))
        + "\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "ordering", "range_gap", "medium", "rn_ordering")


def gen_table_reasoning(rng: random.Random) -> dict:
    a = rng.randint(3, 12)
    b = rng.randint(3, 12)
    c = rng.randint(3, 12)
    answer = max(a, b, c) + min(a, b, c)
    prompt = (
        "Scores are:\n"
        f"Ada: {a}\n"
        f"Bo: {b}\n"
        f"Cy: {c}\n"
        "Add the highest and lowest score.\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "table_reasoning", "max_plus_min", "medium", "rn_table")


def gen_brief_multi_step_word_problem(rng: random.Random) -> dict:
    start = rng.randint(15, 45)
    morning = rng.randint(4, 12)
    afternoon = rng.randint(3, 10)
    gave = rng.randint(2, 8)
    answer = start + morning + afternoon - gave
    item = rng.choice(["marbles", "stickers", "tickets", "cards"])
    person = rng.choice(["Mia", "Noah", "Lena", "Owen", "Sara"])
    prompt = (
        f"{person} started the day with {start} {item}. In the morning, {person.lower()} received {morning} more. "
        f"Later that afternoon, {person.lower()} found {afternoon} extra {item} in a drawer. "
        f"Before going home, {person.lower()} gave {gave} {item} to a friend. "
        f"How many {item} does {person.lower()} have now?\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "word_problem", "three_step_quantity_change_brief", "medium", "rn_word_brief")


def gen_brief_rate_trip(rng: random.Random) -> dict:
    speed1 = rng.choice([20, 30, 40, 50])
    time1 = rng.randint(2, 4)
    speed2 = rng.choice([25, 35, 45, 55])
    time2 = rng.randint(1, 3)
    answer = speed1 * time1 + speed2 * time2
    vehicle = rng.choice(["train", "bus", "boat", "car"])
    prompt = (
        f"A {vehicle} travels for {time1} hours at {speed1} miles per hour, then continues for "
        f"{time2} more hours at {speed2} miles per hour. "
        f"What total distance does the {vehicle} travel?\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "rate", "two_leg_trip_distance", "medium", "rn_rate_brief")


def gen_brief_ratio_context(rng: random.Random) -> dict:
    total = rng.choice([24, 30, 36, 42, 48, 54])
    a = rng.randint(1, 4)
    b = rng.randint(1, 4)
    first = total * a // (a + b)
    second = total - first
    answer = second - first
    prompt = (
        f"A total of {total} points is split between two teams in the ratio {a}:{b}. "
        "After the split, compare the teams' shares. "
        "How many more points does the larger share have than the smaller share?\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "ratios", "ratio_difference_brief", "medium", "rn_ratio_brief")


def gen_brief_table_reasoning(rng: random.Random) -> dict:
    a = rng.randint(5, 18)
    b = rng.randint(5, 18)
    c = rng.randint(5, 18)
    d = rng.randint(5, 18)
    answer = (a + d) - (b + c)
    prompt = (
        "A small score table is shown below:\n"
        f"Ada: {a}\n"
        f"Bo: {b}\n"
        f"Cy: {c}\n"
        f"Di: {d}\n"
        "Add Ada and Di's scores together. Then subtract the sum of Bo and Cy's scores. "
        "What is the result?\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "table_reasoning", "paired_sum_difference_brief", "medium", "rn_table_brief")


def gen_brief_average_reasoning(rng: random.Random) -> dict:
    nums = rng.sample(range(6, 24), 4)
    total = sum(nums)
    adjust = (4 - (total % 4)) % 4
    nums[-1] += adjust
    avg = sum(nums) // 4
    answer = avg + nums[0]
    prompt = (
        "Consider these numbers: "
        + ", ".join(map(str, nums))
        + ". First find their average. Then add the first number in the list to that average. "
        "What is the result?\nAnswer:"
    )
    return wrap_item(prompt, str(answer), "averages", "average_then_add_first_brief", "medium", "rn_average_brief")


FAMILIES = [
    ("word_problem", 4, gen_multi_step_word_problem),
    ("word_problem", 4, gen_brief_multi_step_word_problem),
    ("rate", 2, gen_rate_time_distance),
    ("rate", 3, gen_brief_rate_trip),
    ("grouping", 3, gen_remaining_after_groups),
    ("comparison", 2, gen_compare_totals),
    ("percentages", 3, gen_discount_price),
    ("ratios", 3, gen_ratio_sharing),
    ("ratios", 3, gen_brief_ratio_context),
    ("signed_numbers", 3, gen_signed_change),
    ("averages", 3, gen_average_of_values),
    ("averages", 3, gen_brief_average_reasoning),
    ("sequences", 3, gen_sequence_accumulation),
    ("constraint_counting", 3, gen_constraint_count),
    ("modulo", 3, gen_modulo_reasoning),
    ("unit_conversion", 2, gen_unit_conversion),
    ("ordering", 3, gen_ordering_gap),
    ("table_reasoning", 3, gen_table_reasoning),
    ("table_reasoning", 3, gen_brief_table_reasoning),
]


def family_limit(num_prompts: int) -> int:
    return max(8, (num_prompts + 7) // 8)


def concept_limit(num_prompts: int) -> int:
    return max(8, (num_prompts + 14) // 15)


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
    max_attempts = max(160, num_prompts * 18)

    while len(items) < num_prompts and attempts < max_attempts:
        attempts += 1
        eligible = [(family, weight, fn) for family, weight, fn in FAMILIES if family_counts.get(family, 0) < max_per_family]
        if not eligible:
            eligible = FAMILIES
        family, _, fn = rng.choices(eligible, weights=[weight for _, weight, _ in eligible], k=1)[0]
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
    parser = argparse.ArgumentParser(description="Generate reasoning_numerical prompts from procedural families.")
    parser.add_argument("output", type=str, help="Output JSON path")
    parser.add_argument("num_prompts", type=int, help="Number of prompts to generate")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    parser.add_argument("--seed", type=int, default=47)
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
