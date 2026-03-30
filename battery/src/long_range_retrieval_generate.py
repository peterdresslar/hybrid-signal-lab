#!/usr/bin/env python3
"""
Generate long_range_retrieval prompts from curated delayed-retrieval families.

This generator emphasizes retrieval geometry rather than state tracking:
- where the key fact appears in the context
- how many distractors intervene
- what kind of attribute must be retrieved

Usage:
    python long_range_retrieval_generate.py output.json 80
    python long_range_retrieval_generate.py output.json 20 --append
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
    return f"lrr_seed_{idx:04d}"


def wrap_item(
    prompt: str,
    target: str,
    family: str,
    concept: str,
    difficulty: str,
    source: str,
    **extra_metadata,
) -> dict:
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
        "type": "long_range_retrieval",
        "tokens_approx": tok_count,
        "tier": tier_from_tokens(tok_count),
        "source": source,
        "metadata": metadata,
    }


NAMES = [
    "Alice", "Ben", "Cara", "Dylan", "Eva", "Finn", "Grace", "Hugo",
    "Iris", "Jack", "Kira", "Lena", "Maya", "Noah", "Owen", "Pia",
    "Quinn", "Rosa", "Sam", "Tara",
]
OBJECTS = ["lantern", "map", "notebook", "key", "ticket", "card", "medallion", "compass"]
LOCATIONS = ["kitchen", "garden", "garage", "office", "library", "hall", "attic", "bedroom"]
COLORS = ["blue", "red", "green", "white", "gold", "silver", "amber", "violet"]
CITIES = ["Avalon", "Meridian", "Northwatch", "Silverport", "Eastholm", "Dunmere", "Harrowgate"]
ROLES = ["gardener", "messenger", "archivist", "teacher", "merchant", "carpenter", "painter"]
FOODS = [
    "pasta", "sushi", "chocolate", "tacos", "dumplings", "soup", "curry", "bread",
    "rice", "cheese", "berries", "noodles",
]
MATERIALS = ["oak", "bronze", "marble", "glass", "copper", "linen", "clay", "iron"]
NUMBERS = [12, 15, 18, 21, 24, 27, 30, 36, 42, 48]

FILLER_BANKS = {
    "civic": [
        "Several visitors spent the afternoon discussing local events in quiet conversation.",
        "A caretaker checked the lamps one by one as evening approached.",
        "A messenger crossed the courtyard carrying notes for the town council.",
        "Children played near the fountain until they were called home for dinner.",
        "Several workers paused to compare schedules before returning to their tasks.",
        "By sunset, the central square had become quieter as people returned indoors.",
        "A few late visitors lingered to finish conversations before the doors closed.",
    ],
    "market": [
        "Merchants arranged their goods carefully while customers drifted between the stalls.",
        "The market road was crowded with carts loaded with supplies and tools.",
        "Vendors called out prices while shoppers compared baskets of produce and cloth.",
        "Near the edge of the square, porters stacked crates beside waiting wagons.",
        "Two traders argued amiably over weights, measures, and delivery times.",
        "A clerk moved from stall to stall recording payments in a small ledger.",
    ],
    "archive": [
        "The library remained unusually busy, with readers moving between the shelves.",
        "The walls of the hall were decorated with maps, notices, and faded paintings.",
        "Along the corridor, small tables held baskets, tools, and folded papers.",
        "An assistant carefully copied names and dates from one register to another.",
        "Stacks of documents were arranged in neat bundles beside the reference desk.",
        "A senior archivist paused to compare two copies of the same report.",
    ],
    "travel": [
        "Travelers resting nearby exchanged stories about roads, weather, and inns.",
        "In the distance, bells rang softly from a tower near the main gate.",
        "A pair of riders arrived late and asked for directions before moving on.",
        "Dust from the road settled slowly over the fence near the entrance.",
        "Carters checked their wheels and harnesses before setting out again.",
        "The watch at the gate recorded the names of those entering after noon.",
    ],
    "festival": [
        "In the nearby square, musicians were rehearsing for a festival later that week.",
        "Colorful ribbons had been tied to poles along the edge of the plaza.",
        "Workers tested lantern hooks and banners before the evening ceremony began.",
        "A group of children practiced their steps near the bandstand.",
        "The stage manager walked the route twice to confirm the order of events.",
        "Several cooks were already preparing trays for the celebration meal.",
    ],
    "garden": [
        "The weather was mild, and a light breeze moved through the open windows.",
        "The old stone paths were lined with flowering plants and trimmed hedges.",
        "The air smelled faintly of wood smoke and fresh bread from a nearby bakery.",
        "Bees drifted between blossoms near the far wall of the courtyard.",
        "Water ran quietly through a narrow channel beside the planted beds.",
        "A gardener paused to trim a row of herbs growing near the path.",
    ],
}


def add_fillers(rng: random.Random, count: int, theme: str | None = None) -> list[str]:
    banks = list(FILLER_BANKS.keys())
    if theme is None:
        theme = rng.choice(banks)
    available = list(FILLER_BANKS[theme])
    other_themes = [name for name in banks if name != theme]
    rng.shuffle(available)
    fillers = available[: min(count, len(available))]
    while len(fillers) < count:
        bank_name = rng.choice(other_themes)
        fillers.append(rng.choice(FILLER_BANKS[bank_name]))
    return fillers


def difficulty_from_distractors(n_fillers: int) -> str:
    if n_fillers >= 10:
        return "hard"
    if n_fillers >= 6:
        return "medium"
    return "easy"


def distractor_level(n_fillers: int) -> str:
    if n_fillers >= 10:
        return "high"
    if n_fillers >= 6:
        return "medium"
    return "light"


def join_context(fact_sentences: list[str], query_sentence: str) -> str:
    return " ".join(fact_sentences + [query_sentence])


def build_early_prompt(
    rng: random.Random, fact_sentence: str, query_sentence: str, n_fillers: int, theme: str | None = None
) -> tuple[str, str]:
    fillers = add_fillers(rng, n_fillers, theme=theme)
    prompt = join_context([fact_sentence] + fillers, query_sentence)
    return prompt, "early"


def build_mid_prompt(
    rng: random.Random, fact_sentence: str, query_sentence: str, n_fillers: int, theme: str | None = None
) -> tuple[str, str]:
    fillers = add_fillers(rng, n_fillers, theme=theme)
    split = max(1, len(fillers) // 2)
    prompt = join_context(fillers[:split] + [fact_sentence] + fillers[split:], query_sentence)
    return prompt, "mid"


def gen_early_fact_late_query(rng: random.Random) -> dict:
    name = rng.choice(NAMES)
    schema = rng.choice(["food", "material", "number"])
    if schema == "food":
        answer = rng.choice(FOODS)
        fact = f"At the beginning of the meeting, everyone noted that {name}'s favorite food was {answer}."
        query = f"Much later, someone asked about {name}. {name}'s favorite food was"
        concept = "favorite_food_after_delay"
        answer_type = "food"
    elif schema == "material":
        obj = rng.choice(OBJECTS)
        answer = rng.choice(MATERIALS)
        fact = f"At the start of the inspection, the group noted that {name}'s {obj} was made of {answer}."
        query = f"Much later, the record returned to {name}. {name}'s {obj} was made of"
        concept = "object_material_after_delay"
        answer_type = "material"
    else:
        answer = str(rng.choice(NUMBERS))
        obj = rng.choice(["markers", "tickets", "stones", "tokens"])
        fact = f"At the opening count, {name} was said to be carrying {answer} {obj}."
        query = f"Much later, someone asked about the earlier count. {name} had"
        concept = "object_count_after_delay"
        answer_type = "number"
    n_fillers = rng.randint(6, 11)
    prompt, fact_position = build_early_prompt(rng, fact, query, n_fillers, theme=rng.choice(["civic", "festival", "garden"]))
    return wrap_item(
        prompt, answer, "early_fact_late_query", concept,
        difficulty_from_distractors(n_fillers), "lrr_early",
        fact_position=fact_position, distractor_level=distractor_level(n_fillers), answer_type=answer_type,
        num_entities=1,
    )


def gen_mid_fact_late_query(rng: random.Random) -> dict:
    city = rng.choice(CITIES)
    schema = rng.choice(["color", "role", "number"])
    if schema == "color":
        answer = rng.choice(COLORS)
        fact = f"In the middle of the report, it was stated that the ceremonial flag of {city} was {answer}."
        query = f"At the end of the report, the writer returned to the topic. The ceremonial flag of {city} was"
        concept = "city_flag_color_after_delay"
        answer_type = "color"
    elif schema == "role":
        answer = rng.choice(ROLES)
        fact = f"In the middle of the report, it was stated that the official trade representative of {city} was the {answer}."
        query = f"At the end of the report, the writer returned to the topic. The official trade representative of {city} was the"
        concept = "city_official_role_after_delay"
        answer_type = "role"
    else:
        answer = str(rng.choice(NUMBERS))
        fact = f"In the middle of the report, it was stated that {city} maintained {answer} storage vaults near the river."
        query = f"At the end of the report, the writer returned to the topic. {city} maintained"
        concept = "city_count_after_delay"
        answer_type = "number"
    n_fillers = rng.randint(7, 12)
    prompt, fact_position = build_mid_prompt(rng, fact, query, n_fillers, theme=rng.choice(["archive", "travel", "civic"]))
    return wrap_item(
        prompt, answer, "mid_fact_late_query", concept,
        difficulty_from_distractors(n_fillers), "lrr_mid",
        fact_position=fact_position, distractor_level=distractor_level(n_fillers), answer_type=answer_type,
        num_entities=1,
    )


def gen_multiple_entities_single_attribute(rng: random.Random) -> dict:
    people = rng.sample(NAMES, 4)
    roles = rng.sample(ROLES, 4)
    target_idx = rng.randrange(4)
    target_name = people[target_idx]
    target_role = roles[target_idx]
    intro = "During the planning session, several people introduced themselves."
    fact_sentences = [intro] + [
        f"{person} said they worked as the {role}."
        for person, role in zip(people, roles)
    ]
    query = f"After many other details were discussed, the note-taker asked one final question. {target_name} worked as the"
    fillers = add_fillers(rng, rng.randint(5, 9), theme=rng.choice(["civic", "archive"]))
    prompt = join_context(fact_sentences[:2] + fillers[:2] + fact_sentences[2:] + fillers[2:], query)
    return wrap_item(
        prompt, target_role, "multiple_entities_single_attribute", "retrieve_role_from_entity_list",
        "medium", "lrr_entity_attr",
        fact_position="early-mid", distractor_level=distractor_level(len(fillers)), answer_type="role",
        num_entities=4,
    )


def gen_attribute_binding(rng: random.Random) -> dict:
    people = rng.sample(NAMES, 3)
    schema = rng.choice(["object", "location"])
    target_idx = rng.randrange(3)
    target_name = people[target_idx]
    sentences = []
    if schema == "object":
        objs = rng.sample(OBJECTS, 3)
        locs = rng.sample(LOCATIONS, 3)
        target_answer = objs[target_idx]
        for person, obj, loc in zip(people, objs, locs):
            sentences.append(f"{person} was assigned the {obj} and was told to wait in the {loc}.")
        query = f"At the end of the announcement, someone asked what item belonged to {target_name}. {target_name} had the"
        concept = "entity_object_binding"
        answer_type = "object"
    else:
        objs = rng.sample(OBJECTS, 3)
        locs = rng.sample(LOCATIONS, 3)
        target_answer = locs[target_idx]
        for person, obj, loc in zip(people, objs, locs):
            sentences.append(f"{person} was assigned the {obj} and was told to wait in the {loc}.")
        query = f"At the end of the announcement, someone asked where {target_name} had been told to wait. {target_name} was told to wait in the"
        concept = "entity_location_binding"
        answer_type = "location"
    fillers = add_fillers(rng, rng.randint(6, 10), theme=rng.choice(["civic", "market", "archive"]))
    prompt = join_context(sentences[:1] + fillers[:3] + sentences[1:] + fillers[3:], query)
    return wrap_item(
        prompt, target_answer, "attribute_binding", concept,
        difficulty_from_distractors(len(fillers)), "lrr_binding",
        fact_position="distributed", distractor_level=distractor_level(len(fillers)), answer_type=answer_type,
        num_entities=3,
    )


def gen_story_distractor_density(rng: random.Random) -> dict:
    obj = rng.choice(OBJECTS)
    location = rng.choice(LOCATIONS)
    density = rng.choice(["light", "medium", "high"])
    filler_count = {"light": 4, "medium": 8, "high": 12}[density]
    fact = f"Before anything else happened, the {obj} was placed in the {location}."
    query = f"After the long story concluded, the narrator reminded the reader that the {obj} was in the"
    prompt, fact_position = build_early_prompt(rng, fact, query, filler_count, theme=rng.choice(["travel", "festival", "market"]))
    return wrap_item(
        prompt, location, "story_distractor_density", "same_fact_varied_distractor_load",
        difficulty_from_distractors(filler_count), "lrr_density",
        fact_position=fact_position, distractor_level=density, answer_type="location",
        num_entities=1,
    )


def gen_list_context_retrieval(rng: random.Random) -> dict:
    schema = rng.choice(["city_color", "person_role", "object_location"])
    target_idx = rng.randrange(5)
    if schema == "city_color":
        items = rng.sample(CITIES, 5)
        values = rng.sample(COLORS, 5)
        target_item = items[target_idx]
        target_answer = values[target_idx]
        listings = [f"{item} was associated with the color {value}." for item, value in zip(items, values)]
        query = f"At the very end of the catalog, the summary returned to one entry. The color associated with {target_item} was"
        concept = "catalog_attribute_lookup_color"
        answer_type = "color"
    elif schema == "person_role":
        items = rng.sample(NAMES, 5)
        values = rng.sample(ROLES, 5)
        target_item = items[target_idx]
        target_answer = values[target_idx]
        listings = [f"{item} was listed with the role {value}." for item, value in zip(items, values)]
        query = f"At the very end of the catalog, the summary returned to one entry. The role listed for {target_item} was"
        concept = "catalog_attribute_lookup_role"
        answer_type = "role"
    else:
        items = rng.sample(OBJECTS, 5)
        values = rng.sample(LOCATIONS, 5)
        target_item = items[target_idx]
        target_answer = values[target_idx]
        listings = [f"The {item} was stored in the {value}." for item, value in zip(items, values)]
        query = f"At the very end of the catalog, the summary returned to one entry. The {target_item} was stored in the"
        concept = "catalog_attribute_lookup_location"
        answer_type = "location"
    fillers = add_fillers(rng, rng.randint(5, 8), theme=rng.choice(["archive", "market", "travel"]))
    prompt = join_context(listings[:2] + fillers[:2] + listings[2:] + fillers[2:], query)
    return wrap_item(
        prompt, target_answer, "list_context_retrieval", concept,
        "medium", "lrr_list",
        fact_position="distributed", distractor_level=distractor_level(len(fillers)), answer_type=answer_type,
        num_entities=5,
    )


def gen_two_fact_conjunction(rng: random.Random) -> dict:
    person = rng.choice(NAMES)
    schema = rng.choice(["role_origin", "object_location"])
    if schema == "role_origin":
        first_answer = rng.choice(CITIES)
        role = rng.choice(ROLES)
        fact1 = f"Early in the report, it was noted that {person} came from {first_answer}."
        fact2 = f"Several paragraphs later, the report explained that {person} worked as the {role}."
        query = f"In the closing line, the writer combined both details. {person} was the {role} from"
        concept = "combine_role_and_origin"
        answer_type = "city"
    else:
        first_answer = rng.choice(LOCATIONS)
        obj = rng.choice(OBJECTS)
        fact1 = f"Early in the report, it was noted that {person} had left the {obj} in the {first_answer}."
        fact2 = f"Several paragraphs later, the report explained that the {obj} was still the item associated with {person}."
        query = f"In the closing line, the writer combined both details. The {obj} associated with {person} had been left in the"
        concept = "combine_object_and_location"
        answer_type = "location"
    fillers_a = add_fillers(rng, rng.randint(3, 5), theme=rng.choice(["travel", "civic"]))
    fillers_b = add_fillers(rng, rng.randint(3, 5), theme=rng.choice(["archive", "market"]))
    prompt = join_context([fact1] + fillers_a + [fact2] + fillers_b, query)
    return wrap_item(
        prompt, first_answer, "two_fact_conjunction", concept,
        "hard", "lrr_two_fact",
        fact_position="early+mid", distractor_level="medium", answer_type=answer_type,
        num_entities=1,
    )


FAMILIES = [
    ("early_fact_late_query", 4, gen_early_fact_late_query),
    ("mid_fact_late_query", 4, gen_mid_fact_late_query),
    ("multiple_entities_single_attribute", 4, gen_multiple_entities_single_attribute),
    ("attribute_binding", 4, gen_attribute_binding),
    ("story_distractor_density", 3, gen_story_distractor_density),
    ("list_context_retrieval", 3, gen_list_context_retrieval),
    ("two_fact_conjunction", 3, gen_two_fact_conjunction),
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
        eligible = [
            (family, weight, fn)
            for family, weight, fn in FAMILIES
            if family_counts.get(family, 0) < max_per_family
        ]
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
    parser = argparse.ArgumentParser(description="Generate long_range_retrieval prompts from procedural families.")
    parser.add_argument("output", type=str, help="Output JSON path")
    parser.add_argument("num_prompts", type=int, help="Number of prompts to generate")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    parser.add_argument("--seed", type=int, default=50)
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
