#!/usr/bin/env python3
"""
Generate syntactic_pattern prompts from curated structural families.

This generator focuses on short, controlled syntax-sensitive prompts:
- agreement across intervening material
- clause and coordinator completion
- quote/bracket closure
- list and punctuation continuation
- correlative and conditional constructions

Usage:
    python syntactic_pattern_generate.py output.json 60
    python syntactic_pattern_generate.py output.json 20 --append
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
    return f"sp_seed_{idx:04d}"


def wrap_item(prompt: str, target: str, family: str, concept: str, difficulty: str, source: str, **extra_metadata) -> dict:
    tok_count = approx_tokens(prompt)
    metadata = {
        "family": family,
        "concept": concept,
        "difficulty": difficulty,
    }
    metadata.update(extra_metadata)
    return {
        "prompt": prompt,
        "target": target if target.startswith(" ") else " " + str(target),
        "type": "syntactic_pattern",
        "tokens_approx": tok_count,
        "tier": tier_from_tokens(tok_count),
        "source": source,
        "metadata": metadata,
    }


SUBJECTS_SING = ["The cat", "The dog", "The teacher", "The scientist", "The child", "The painter", "The driver"]
SUBJECTS_PLUR = ["The cats", "The dogs", "The teachers", "The scientists", "The children", "The painters", "The drivers"]
ATTRACTOR_NPS = [
    "near the old barns",
    "beside the broken windows",
    "from the northern village",
    "with the red markings",
    "under the heavy curtains",
    "that the students admired",
    "by the narrow staircase",
]
VERBS = [
    ("runs", "run"),
    ("walks", "walk"),
    ("speaks", "speak"),
    ("writes", "write"),
    ("waits", "wait"),
    ("arrives", "arrive"),
]
NOUNS = ["apples", "books", "papers", "tools", "lamps", "maps", "flowers"]
ADJS = ["careful", "quiet", "steady", "bright", "patient", "kind"]
NATURAL_NOUNS = ["music", "medicine", "rest", "rain", "advice", "company"]


def gen_agreement_attractor(rng: random.Random) -> dict:
    singular = rng.choice([True, False])
    subj = rng.choice(SUBJECTS_SING if singular else SUBJECTS_PLUR)
    interv = rng.choice(ATTRACTOR_NPS)
    verb_sing, verb_plur = rng.choice(VERBS)
    target = verb_sing if singular else verb_plur
    prompt = f"{subj} {interv}"
    return wrap_item(
        prompt, target, "agreement", "agreement_with_intervener", "medium", "sp_agreement",
        subject_number="singular" if singular else "plural",
        distance=len(interv.split()),
    )


def gen_agreement_relative_clause(rng: random.Random) -> dict:
    singular = rng.choice([True, False])
    subj = rng.choice(SUBJECTS_SING if singular else SUBJECTS_PLUR)
    rc_noun = rng.choice(["students", "visitors", "workers", "neighbors", "artists"])
    verb_sing, verb_plur = rng.choice(VERBS)
    target = verb_sing if singular else verb_plur
    prompt = f"{subj} that the {rc_noun} admired"
    return wrap_item(
        prompt, target, "agreement", "agreement_relative_clause", "hard", "sp_agreement_rc",
        subject_number="singular" if singular else "plural",
        distance=len(prompt.split()) - len(subj.split()),
    )


def gen_correlative_construction(rng: random.Random) -> dict:
    kind = rng.choice(["either_or", "not_only"])
    if kind == "either_or":
        prompt, target = rng.choice([
            ("Either the music will help, or the medicine will", " work"),
            ("Either the rain will stop, or the wind will", " weaken"),
            ("Either the advice will help, or the company will", " comfort her"),
            ("Either the rest will help, or the tea will", " calm her"),
            ("Either the medicine will help, or the sleep will", " be enough"),
            ("Either the light will return, or the guide will", " continue"),
        ])
        concept = "either_or_completion"
    else:
        prompt, target = rng.choice([
            ("Not only was the lecture long, but it was also surprisingly", " clear"),
            ("Not only was the room quiet, but it was also unexpectedly", " warm"),
            ("Not only was the path steep, but it was also remarkably", " narrow"),
            ("Not only was the speech brief, but it was also unusually", " careful"),
            ("Not only was the plan simple, but it was also extremely", " useful"),
        ])
        concept = "not_only_but_also"
    return wrap_item(prompt, target, "correlative", concept, "medium", "sp_correlative")


def gen_conditional_parallel(rng: random.Random) -> dict:
    action1, action2 = rng.sample([
        "stay inside", "go outside", "light the fire", "close the gate", "open the windows",
        "cover the chairs", "move the boxes", "wait by the door", "start the engine",
    ], 2)
    weather1, weather2 = rng.sample(["rains", "snows", "clears", "warms", "freezes", "darkens"], 2)
    prompt = f"If it {weather1}, we {action1}; if it {weather2}, we"
    target = f" {action2}"
    return wrap_item(prompt, target, "parallelism", "if_then_parallel", "medium", "sp_conditional")


def gen_series_continuation(rng: random.Random) -> dict:
    actions = rng.sample(["eat", "drink", "rest", "sleep", "wait", "leave", "listen", "watch", "work"], 3)
    prompt = f"First we {actions[0]}, then we {actions[1]}, and finally we"
    target = f" {actions[2]}"
    return wrap_item(prompt, target, "parallelism", "series_continuation", "easy", "sp_series")


def gen_list_punctuation(rng: random.Random) -> dict:
    container = rng.choice(["The box held", "The bag contained", "The shelf displayed", "The cart carried"])
    items = rng.sample(NOUNS, 3)
    prompt = f"{container} {items[0]}, {items[1]}, and"
    target = f" {items[2]}"
    return wrap_item(prompt, target, "list_syntax", "coordinated_np_list", "easy", "sp_list")


def gen_clause_subordinator(rng: random.Random) -> dict:
    clause_type = rng.choice(["although", "because", "while", "unless", "after", "before", "when"])
    if clause_type == "although":
        prompt = "Although the road was narrow, the cart"
        target = " passed"
    elif clause_type == "because":
        prompt = "Because the lantern was broken, the guide"
        target = " waited"
    elif clause_type == "while":
        prompt = "While the guests were arriving, the host"
        target = " smiled"
    elif clause_type == "unless":
        prompt = "Unless the rain stops soon, the workers"
        target = " will leave"
    elif clause_type == "after":
        prompt = "After the bell rang, the students"
        target = " entered"
    elif clause_type == "before":
        prompt = "Before the market opened, the merchants"
        target = " arrived"
    else:
        prompt = "When the signal changed, the driver"
        target = " slowed"
    return wrap_item(prompt, target, "clause_completion", f"{clause_type}_main_clause", "medium", "sp_clause")


def gen_quote_closure(rng: random.Random) -> dict:
    opener, answer = rng.choice([
        ('She whispered, "Please close the', ' door."'),
        ('He said, "Bring me the', ' map."'),
        ('The note read, "Meet at the', ' gate."'),
        ('The sign warned, "Do not cross the', ' line."'),
        ('She asked, "Did you lock the', ' door?"'),
        ('He called out, "Wait by the', ' stairs."'),
    ])
    return wrap_item(opener, answer, "punctuation", "quote_closure", "easy", "sp_quote")


def gen_parenthetical_closure(rng: random.Random) -> dict:
    prompt, target = rng.choice([
        ("The oldest tool (kept in the attic", ")"),
        ("Her final answer [written on the board", "]"),
        ("The meeting time (listed near the entrance", ")"),
        ("The missing page [tucked inside the folder", "]"),
        ("The smaller key (found beneath the desk", ")"),
        ("The final note [left by the window", "]"),
    ])
    return wrap_item(prompt, target, "punctuation", "bracket_closure", "easy", "sp_bracket")


def gen_attachment_style_completion(rng: random.Random) -> dict:
    prompt, target = rng.choice([
        ("The reporter interviewed the painter with the", " microphone"),
        ("The guard watched the traveler with the", " telescope"),
        ("The child greeted the teacher with a", " smile"),
        ("The guide followed the visitor with the", " lantern"),
        ("The student approached the speaker with a", " question"),
    ])
    return wrap_item(prompt, target, "attachment", "pp_attachment_continuation", "medium", "sp_attachment")


def gen_comparative_pair(rng: random.Random) -> dict:
    verb = rng.choice(["gets", "becomes", "looks"])
    prompt = f"The more you practice, the better it"
    target = f" {verb}"
    return wrap_item(prompt, target, "correlative", "the_more_the_more", "easy", "sp_comparative")


def gen_coordinator_completion(rng: random.Random) -> dict:
    prompt, target, concept = rng.choice([
        ("She wanted to leave early, but she", " stayed", "but_clause_completion"),
        ("The path was steep, yet the hikers", " continued", "yet_clause_completion"),
        ("He opened the window, and the room", " cooled", "and_clause_completion"),
        ("The warning was clear, but the crowd", " hesitated", "but_clause_completion"),
        ("The rain stopped, and the children", " cheered", "and_clause_completion"),
        ("The signal was weak, yet the radio", " worked", "yet_clause_completion"),
        ("The door was heavy, but the porter", " pushed", "but_clause_completion"),
    ])
    return wrap_item(prompt, target, "coordination", concept, "medium", "sp_coord")


def gen_complementizer_completion(rng: random.Random) -> dict:
    prompt, target = rng.choice([
        ("She said that the answer was", " correct"),
        ("They believed that the map was", " accurate"),
        ("He realized that the gate was", " locked"),
        ("We knew that the signal was", " late"),
        ("The teacher explained that the test was", " over"),
        ("She noticed that the room was", " empty"),
        ("They discovered that the path was", " blocked"),
    ])
    return wrap_item(prompt, target, "complement_clause", "that_clause_completion", "easy", "sp_comp")


def gen_colon_or_dash_continuation(rng: random.Random) -> dict:
    prompt, target, concept = rng.choice([
        ("She carried only three things:", " a map, a lantern, and a key", "colon_list_continuation"),
        ("The note contained a single instruction:", " wait outside", "colon_clause_continuation"),
        ("He had one clear goal -", " to finish early", "dash_infinitive_continuation"),
        ("There was only one real problem -", " the door was locked", "dash_clause_continuation"),
        ("The report ended with one recommendation:", " close the gate", "colon_clause_continuation"),
        ("They needed only two supplies:", " rope and water", "colon_list_continuation"),
        ("Her only plan -", " to leave quietly", "dash_infinitive_continuation"),
    ])
    return wrap_item(prompt, target, "punctuation", concept, "medium", "sp_colon_dash")


def gen_pronoun_agreement(rng: random.Random) -> dict:
    prompt, target = rng.choice([
        ("Every student handed in", " their paper"),
        ("Each worker returned to", " their station"),
        ("Every child looked for", " their coat"),
        ("Each guest checked", " their seat"),
        ("Every visitor picked up", " their badge"),
        ("Each runner waited for", " their turn"),
    ])
    return wrap_item(prompt, target, "agreement", "distributive_pronoun_completion", "medium", "sp_pronoun")


FAMILIES = [
    ("agreement", 3, gen_agreement_attractor),
    ("agreement", 3, gen_agreement_relative_clause),
    ("agreement", 2, gen_pronoun_agreement),
    ("correlative", 3, gen_correlative_construction),
    ("parallelism", 3, gen_conditional_parallel),
    ("parallelism", 3, gen_series_continuation),
    ("list_syntax", 3, gen_list_punctuation),
    ("clause_completion", 4, gen_clause_subordinator),
    ("coordination", 3, gen_coordinator_completion),
    ("complement_clause", 3, gen_complementizer_completion),
    ("punctuation", 3, gen_quote_closure),
    ("punctuation", 3, gen_parenthetical_closure),
    ("punctuation", 3, gen_colon_or_dash_continuation),
    ("attachment", 2, gen_attachment_style_completion),
    ("correlative", 2, gen_comparative_pair),
]


def family_limit(num_prompts: int) -> int:
    return max(10, (num_prompts + 5) // 6)


def concept_limit(num_prompts: int) -> int:
    return max(10, (num_prompts + 9) // 10)


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
    max_attempts = max(180, num_prompts * 24)

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
    parser = argparse.ArgumentParser(description="Generate syntactic_pattern prompts from procedural families.")
    parser.add_argument("output", type=str, help="Output JSON path")
    parser.add_argument("num_prompts", type=int, help="Number of prompts to generate")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    parser.add_argument("--seed", type=int, default=49)
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
