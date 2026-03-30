#!/usr/bin/env python3
"""
Generate reasoning_tracking prompts from curated state-tracking families.

This generator is designed to preserve the battery's brief-heavy,
multi-step tracking character while increasing diversity across
entity, location, possession, and overwrite-style tasks.

Usage:
    python reasoning_tracking_generate.py output.json 90
    python reasoning_tracking_generate.py output.json 20 --append
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
    return f"rt_seed_{idx:04d}"


def wrap_item(prompt: str, target: str, family: str, concept: str, difficulty: str, source: str) -> dict:
    tok_count = approx_tokens(prompt)
    return {
        "prompt": prompt,
        "target": target if target.startswith(" ") else " " + str(target),
        "type": "reasoning_tracking",
        "tokens_approx": tok_count,
        "tier": tier_from_tokens(tok_count),
        "source": source,
        "metadata": {
            "family": family,
            "concept": concept,
            "difficulty": difficulty,
        },
    }


NAMES = [
    "Alice", "Ben", "Cara", "Dylan", "Eva", "Finn", "Grace", "Hugo",
    "Iris", "Jack", "Kira", "Lena", "Maya", "Noah", "Owen", "Pia",
    "Quinn", "Rosa", "Sam", "Tara",
]
OBJECTS = ["book", "coin", "key", "notebook", "ticket", "card", "lantern", "map"]
LOCATIONS = ["kitchen", "garden", "garage", "office", "library", "hall", "attic", "bedroom"]


def gen_location_tracking(rng: random.Random) -> dict:
    person = rng.choice(NAMES)
    start, mid, end = rng.sample(LOCATIONS, 3)
    prompt = (
        f"{person} started in the {start}. Then {person.lower()} walked to the {mid}. "
        f"After that, {person.lower()} went to the {end}. Where is {person.lower()} now?\nAnswer:"
    )
    return wrap_item(prompt, end, "location", "three_step_location", "easy", "rt_location")


def gen_object_transfer(rng: random.Random) -> dict:
    a, b, c = rng.sample(NAMES, 3)
    obj = rng.choice(OBJECTS)
    holder1, holder2 = a, b
    prompt = (
        f"{a} had the {obj}. Then {a} gave the {obj} to {b}. "
        f"Later, {b} passed the {obj} to {c}. Who has the {obj} now?\nAnswer:"
    )
    return wrap_item(prompt, c, "transfer", "two_transfer_chain", "easy", "rt_transfer")


def gen_possession_swap(rng: random.Random) -> dict:
    a, b = rng.sample(NAMES, 2)
    obj1, obj2 = rng.sample(OBJECTS, 2)
    prompt = (
        f"{a} has the {obj1} and {b} has the {obj2}. "
        f"They swap items. After the swap, who has the {obj1}?\nAnswer:"
    )
    return wrap_item(prompt, b, "swap", "simple_swap", "easy", "rt_swap")


def gen_instruction_overwrite(rng: random.Random) -> dict:
    person = rng.choice(NAMES)
    color1, color2, color3 = rng.sample(["red", "blue", "green", "yellow", "black"], 3)
    prompt = (
        f"{person} paints a box {color1}. Then {person.lower()} paints the same box {color2}. "
        f"Finally, {person.lower()} paints it {color3}. What color is the box at the end?\nAnswer:"
    )
    return wrap_item(prompt, color3, "overwrite", "final_state_overwrite", "easy", "rt_overwrite")


def gen_brief_entity_tracking(rng: random.Random) -> dict:
    a, b, c = rng.sample(NAMES, 3)
    obj = rng.choice(OBJECTS)
    loc1, loc2 = rng.sample(LOCATIONS, 2)
    prompt = (
        f"In the morning, {a} placed the {obj} in the {loc1}. "
        f"At noon, {b} moved the {obj} from the {loc1} to the {loc2}. "
        f"In the evening, {c} checked the {loc2} but did not move the {obj}. "
        f"Where is the {obj} now?\nAnswer:"
    )
    return wrap_item(prompt, loc2, "location", "object_moved_between_locations", "medium", "rt_object_location")


def gen_brief_possession_chain(rng: random.Random) -> dict:
    a, b, c, d = rng.sample(NAMES, 4)
    obj = rng.choice(OBJECTS)
    prompt = (
        f"{a} started with the {obj}. {a} handed it to {b}. "
        f"{b} later gave it to {c}. After using it, {c} passed it to {d}. "
        f"Who has the {obj} at the end?\nAnswer:"
    )
    return wrap_item(prompt, d, "transfer", "three_transfer_chain", "medium", "rt_transfer_brief")


def gen_brief_role_tracking(rng: random.Random) -> dict:
    a, b, c = rng.sample(NAMES, 3)
    prompt = (
        f"{a} was first in line, {b} was second, and {c} was third. "
        f"Then {c} moved ahead of {b}. After that, {b} moved ahead of {a}. "
        f"Who is second in line now?\nAnswer:"
    )
    answer = a
    return wrap_item(prompt, answer, "ordering", "line_reordering", "medium", "rt_order")


def gen_brief_container_tracking(rng: random.Random) -> dict:
    obj = rng.choice(OBJECTS)
    cont1, cont2, cont3 = rng.sample(["box", "bag", "drawer", "basket", "cabinet"], 3)
    prompt = (
        f"The {obj} was first put in the {cont1}. Then it was moved to the {cont2}. "
        f"Later, it was removed from the {cont2} and placed in the {cont3}. "
        f"Which container holds the {obj} now?\nAnswer:"
    )
    return wrap_item(prompt, cont3, "containers", "container_move_chain", "medium", "rt_container")


def gen_brief_state_update(rng: random.Random) -> dict:
    person = rng.choice(NAMES)
    mood1, mood2, mood3 = rng.sample(["happy", "tired", "calm", "nervous", "excited"], 3)
    prompt = (
        f"{person} felt {mood1} in the morning. After a long walk, {person.lower()} felt {mood2}. "
        f"At the end of the day, after some good news, {person.lower()} felt {mood3}. "
        f"How does {person.lower()} feel at the end?\nAnswer:"
    )
    return wrap_item(prompt, mood3, "overwrite", "final_state_update_brief", "medium", "rt_state")


def gen_brief_reference_tracking(rng: random.Random) -> dict:
    a, b = rng.sample(NAMES, 2)
    obj1, obj2 = rng.sample(OBJECTS, 2)
    prompt = (
        f"{a} put the {obj1} on the table and the {obj2} on the shelf. "
        f"Then {b} moved the {obj1} to the shelf and moved the {obj2} to the drawer. "
        f"Which item is on the shelf now?\nAnswer:"
    )
    return wrap_item(prompt, obj1, "reference", "which_item_in_location", "medium", "rt_reference")


def gen_brief_nested_tracking(rng: random.Random) -> dict:
    person = rng.choice(NAMES)
    outer, inner = rng.sample(["box", "bag", "drawer", "cabinet"], 2)
    obj = rng.choice(OBJECTS)
    prompt = (
        f"{person} placed the {obj} inside a {inner}, then put the {inner} inside a {outer}. "
        f"Later, {person.lower()} removed the {inner} from the {outer} but did not open it. "
        f"Where is the {obj} now?\nAnswer:"
    )
    return wrap_item(prompt, inner, "nested_tracking", "object_inside_inner_container", "medium", "rt_nested")


FAMILIES = [
    ("location", 3, gen_location_tracking),
    ("transfer", 3, gen_object_transfer),
    ("swap", 2, gen_possession_swap),
    ("overwrite", 2, gen_instruction_overwrite),
    ("location", 4, gen_brief_entity_tracking),
    ("transfer", 4, gen_brief_possession_chain),
    ("ordering", 3, gen_brief_role_tracking),
    ("containers", 3, gen_brief_container_tracking),
    ("overwrite", 3, gen_brief_state_update),
    ("reference", 3, gen_brief_reference_tracking),
    ("nested_tracking", 3, gen_brief_nested_tracking),
]


def family_limit(num_prompts: int) -> int:
    return max(8, (num_prompts + 7) // 8)


def concept_limit(num_prompts: int) -> int:
    return max(6, (num_prompts + 14) // 15)


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
    max_attempts = max(120, num_prompts * 14)

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
    parser = argparse.ArgumentParser(description="Generate reasoning_tracking prompts from procedural families.")
    parser.add_argument("output", type=str, help="Output JSON path")
    parser.add_argument("num_prompts", type=int, help="Number of prompts to generate")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    parser.add_argument("--seed", type=int, default=48)
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
