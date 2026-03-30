#!/usr/bin/env python3
"""
Battery builder for g-profile sweep experiments.

Generates ~445 candidate prompts across 11 mechanism types,
oversampled toward high-headroom types (factual_recall, structural_copying, reasoning).

Sources:
  - COUNTERFACT (HuggingFace) for factual_recall and factual_retrieval
  - LAMBADA (HuggingFace) for cultural_memorized
  - Programmatic generators for all other types
  - Curated Wikipedia-derived domain knowledge pools loaded from JSON
  - Optional externally generated code-comprehension pool

Output: JSON battery file compatible with signal_lab.py / sweep.py
"""

import argparse
import json
import random
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def tier_from_tokens(n: int) -> str:
    if n <= 30:
        return "short"
    elif n <= 80:
        return "brief"
    elif n <= 200:
        return "med"
    elif n <= 500:
        return "long"
    else:
        return "extended"

def make_id(type_tag: str, source: str, idx: int) -> str:
    return f"{type_tag}_{source}_{idx:04d}"

def approx_tokens(text: str) -> int:
    """Rough token count: split on whitespace, multiply by 1.3 for subword."""
    return int(len(text.split()) * 1.3)

# ---------------------------------------------------------------------------
# COUNTERFACT extraction
# ---------------------------------------------------------------------------

def load_counterfact(cache_dir: str = None):
    """Download and return COUNTERFACT dataset."""
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train", cache_dir=cache_dir)
    return ds


def _is_good_factual_recall_prompt(prompt: str) -> bool:
    """Filter out CounterFact prompt patterns that read awkwardly as clozes."""
    blocked_phrases = [
        " is a part of the continent of",
    ]
    return not any(phrase in prompt for phrase in blocked_phrases)

def extract_factual_recall(ds, n: int = 80, seed: int = 42) -> list[dict]:
    """Extract factual recall prompts from COUNTERFACT.

    These are short cloze-style prompts where the model must complete
    a factual statement. We filter for single-token targets.
    """
    rng = random.Random(seed)

    # COUNTERFACT has fields: prompt, subject, target_true, target_false, etc.
    candidates = []
    for row in ds:
        prompt = row.get("prompt", "")
        target = row.get("target_true", "")
        subject = row.get("subject", "")

        if not prompt or not target:
            continue
        if not _is_good_factual_recall_prompt(prompt):
            continue

        # We want prompts that end mid-sentence (cloze style)
        # Target should be reasonably short (1-3 words for single-token focus)
        target_words = target.strip().split()
        if len(target_words) > 3:
            continue

        # Ensure target starts with space (for token alignment)
        target_clean = " " + target.strip() if not target.startswith(" ") else target

        tok_count = approx_tokens(prompt)

        candidates.append({
            "prompt": prompt,
            "target": target_clean,
            "type": "factual_recall",
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
            "source": "counterfact",
            "metadata": {"subject": subject}
        })

    # Sample, preferring items in the sweet-spot token range
    rng.shuffle(candidates)

    # Stratify by tier to get variety
    by_tier = {}
    for c in candidates:
        by_tier.setdefault(c["tier"], []).append(c)

    selected = []
    # Aim for: 40% short, 30% brief, 20% med, 10% long
    tier_targets = {"short": int(n * 0.4), "brief": int(n * 0.3),
                    "med": int(n * 0.2), "long": n - int(n * 0.4) - int(n * 0.3) - int(n * 0.2)}

    for tier, target_count in tier_targets.items():
        pool = by_tier.get(tier, [])
        selected.extend(pool[:target_count])

    # Fill remainder from any tier
    remaining = n - len(selected)
    used_prompts = {s["prompt"] for s in selected}
    extras = [c for c in candidates if c["prompt"] not in used_prompts]
    selected.extend(extras[:remaining])

    for i, item in enumerate(selected):
        item["id"] = make_id("fr", "counterfact", i)

    return selected[:n]


def extract_factual_retrieval(ds, n: int = 40, seed: int = 43) -> list[dict]:
    """Extract factual retrieval prompts from COUNTERFACT.

    These are longer-context versions where we embed the factual question
    in a paragraph of context, forcing the model to retrieve across distance.
    """
    rng = random.Random(seed)

    # We'll create longer prompts by adding contextual framing
    context_templates = [
        "In a comprehensive review of {subject}, researchers noted several key facts. Among them, {prompt}",
        "The following information pertains to {subject}. After careful examination of the available evidence, we can state that {prompt}",
        "Historical records about {subject} reveal many interesting details. One well-established fact is that {prompt}",
        "According to multiple authoritative sources discussing {subject}, it is widely accepted that {prompt}",
        "When examining the topic of {subject} in depth, one important detail stands out: {prompt}",
        "A detailed analysis of {subject} shows that across various references, {prompt}",
        "Scholars studying {subject} have documented that {prompt}",
        "In the context of understanding {subject} and its significance, we note that {prompt}",
    ]

    candidates = []
    for row in ds:
        prompt = row.get("prompt", "")
        target = row.get("target_true", "")
        subject = row.get("subject", "")

        if not prompt or not target:
            continue

        target_words = target.strip().split()
        if len(target_words) > 3:
            continue

        target_clean = " " + target.strip() if not target.startswith(" ") else target

        # Wrap in context template
        template = rng.choice(context_templates)
        full_prompt = template.format(subject=subject, prompt=prompt)

        tok_count = approx_tokens(full_prompt)

        candidates.append({
            "prompt": full_prompt,
            "target": target_clean,
            "type": "factual_retrieval",
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
            "source": "counterfact_extended",
            "metadata": {"subject": subject, "base_prompt": prompt}
        })

    rng.shuffle(candidates)
    selected = candidates[:n]

    for i, item in enumerate(selected):
        item["id"] = make_id("fret", "counterfact", i)

    return selected


# ---------------------------------------------------------------------------
# LAMBADA extraction
# ---------------------------------------------------------------------------

def load_lambada(cache_dir: str = None):
    """Download and return LAMBADA dataset."""
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test", cache_dir=cache_dir)
    return ds

def extract_cultural_memorized(ds_lambada, n: int = 25, seed: int = 44) -> list[dict]:
    """Extract cultural/memorized completion prompts.

    LAMBADA provides passages where the final word requires broad context.
    We supplement with hand-crafted cultural memorization prompts.
    """
    rng = random.Random(seed)

    candidates = []

    # From LAMBADA: passages where last word is the target
    for row in ds_lambada:
        text = row.get("text", "")
        if not text:
            continue

        words = text.strip().split()
        if len(words) < 5:
            continue

        target_word = words[-1]
        prompt_text = " ".join(words[:-1])

        # Filter for clean single-word targets
        if not target_word.isalpha():
            continue
        if len(target_word) < 2:
            continue

        target_clean = " " + target_word
        tok_count = approx_tokens(prompt_text)

        candidates.append({
            "prompt": prompt_text,
            "target": target_clean,
            "type": "cultural_memorized",
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
            "source": "lambada",
            "metadata": {}
        })

    rng.shuffle(candidates)

    # Also add some hand-crafted cultural memorization items
    cultural_items = [
        {"prompt": "To be, or not to be, that is the", "target": " question"},
        {"prompt": "I think, therefore I", "target": " am"},
        {"prompt": "E = mc", "target": " squared"},  # Note: target may need adjustment per tokenizer
        {"prompt": "Four score and seven years ago our fathers brought forth on this continent, a new", "target": " nation"},
        {"prompt": "It was the best of times, it was the worst of", "target": " times"},
        {"prompt": "In the beginning God created the heavens and the", "target": " earth"},
        {"prompt": "We hold these truths to be self-evident, that all men are created", "target": " equal"},
        {"prompt": "Ask not what your country can do for you, ask what you can do for your", "target": " country"},
        {"prompt": "The only thing we have to fear is fear", "target": " itself"},
        {"prompt": "One small step for man, one giant leap for", "target": " mankind"},
    ]

    for i, item in enumerate(cultural_items):
        tok_count = approx_tokens(item["prompt"])
        item.update({
            "type": "cultural_memorized",
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
            "source": "curated",
            "metadata": {}
        })

    # Combine: mostly LAMBADA, supplement with curated
    n_lambada = n - len(cultural_items)
    selected = candidates[:max(0, n_lambada)] + cultural_items
    rng.shuffle(selected)
    selected = selected[:n]

    for i, item in enumerate(selected):
        item["id"] = make_id("cm", item.get("source", "mixed"), i)

    return selected


# ---------------------------------------------------------------------------
# Generators: Structural Copying
# ---------------------------------------------------------------------------

def generate_structural_copying(n: int = 60, seed: int = 45) -> list[dict]:
    """Generate structural copying prompts.

    These test the model's ability to reproduce patterns:
    - List continuation (A, B, C → D)
    - Format copying (repeat a structure)
    - Sequence mirroring
    """
    rng = random.Random(seed)
    items = []

    # Type 1: Alphabetic list continuation
    for i in range(n // 4):
        start = rng.randint(0, 20)
        length = rng.randint(3, 8)
        letters = [chr(ord('A') + (start + j) % 26) for j in range(length)]
        next_letter = chr(ord('A') + (start + length) % 26)

        sep = rng.choice([", ", " ", " - ", "; "])
        prompt = sep.join(letters) + sep

        items.append({
            "prompt": prompt,
            "target": " " + next_letter,
            "type": "structural_copying",
            "source": "gen_alpha_seq",
            "metadata": {"pattern": "alpha_continuation"}
        })

    # Type 2: Number list continuation
    for i in range(n // 4):
        # Arithmetic sequences
        start = rng.randint(1, 50)
        step = rng.choice([1, 2, 3, 5, 10])
        length = rng.randint(3, 7)
        nums = [str(start + j * step) for j in range(length)]
        target_num = str(start + length * step)

        sep = rng.choice([", ", " ", "; "])
        prompt = sep.join(nums) + sep

        items.append({
            "prompt": prompt,
            "target": " " + target_num,
            "type": "structural_copying",
            "source": "gen_num_seq",
            "metadata": {"pattern": "arithmetic", "start": start, "step": step}
        })

    # Type 3: Repeated structure copying
    structures = [
        ("Name: {name}\nAge: {age}\nCity: {city}\n\nName: {name2}\nAge: {age2}\nCity:",
         lambda rng: {
             "name": rng.choice(["Alice", "Bob", "Carol", "Dave", "Eve"]),
             "age": str(rng.randint(20, 60)),
             "city": rng.choice(["London", "Paris", "Tokyo", "Berlin", "Cairo"]),
             "name2": rng.choice(["Frank", "Grace", "Hank", "Iris", "Jack"]),
             "age2": str(rng.randint(20, 60)),
         },
         lambda vals: " " + rng.choice(["London", "Paris", "Tokyo", "Berlin", "Cairo", "Rome", "Delhi"])),

        ("Item: apple, Color: red\nItem: banana, Color: yellow\nItem: grass, Color:",
         lambda rng: {},
         lambda vals: " green"),

        ("Q: What is 2+2? A: 4\nQ: What is 3+3? A: 6\nQ: What is 5+5? A:",
         lambda rng: {},
         lambda vals: " 10"),
    ]

    for i in range(n // 4):
        template, val_fn, target_fn = rng.choice(structures)
        vals = val_fn(rng)
        prompt = template.format(**vals) if vals else template
        target = target_fn(vals)

        items.append({
            "prompt": prompt,
            "target": target,
            "type": "structural_copying",
            "source": "gen_struct",
            "metadata": {"pattern": "structure_copy"}
        })

    # Type 4: Word list pattern copying
    word_categories = {
        "animals": ["cat", "dog", "bird", "fish", "horse", "lion", "bear", "wolf", "deer", "frog"],
        "colors": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "brown"],
        "fruits": ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "kiwi", "lemon", "mango"],
        "countries": ["France", "Japan", "Brazil", "Egypt", "India", "Canada", "Italy", "Spain", "China", "Kenya"],
    }

    for i in range(n - len(items)):
        cat = rng.choice(list(word_categories.keys()))
        words = rng.sample(word_categories[cat], min(5, len(word_categories[cat])))
        target_word = rng.choice([w for w in word_categories[cat] if w not in words])

        # Present as a list with one more to complete
        prompt = f"List of {cat}: " + ", ".join(words) + ","

        items.append({
            "prompt": prompt,
            "target": " " + target_word,
            "type": "structural_copying",
            "source": "gen_word_list",
            "metadata": {"pattern": "category_list", "category": cat}
        })

    for i, item in enumerate(items[:n]):
        tok_count = approx_tokens(item["prompt"])
        item.update({
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
        })
        item["id"] = make_id("sc", item["source"], i)

    return items[:n]


# ---------------------------------------------------------------------------
# Generators: Algorithmic
# ---------------------------------------------------------------------------

def generate_algorithmic(n: int = 30, seed: int = 46) -> list[dict]:
    """Generate algorithmic completion prompts.

    These test the model's ability to execute simple algorithms:
    - Arithmetic
    - String manipulation
    - Sorting
    - Counting
    """
    rng = random.Random(seed)
    items = []

    # Type 1: Simple arithmetic
    for i in range(n // 3):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        op = rng.choice(["+", "-", "*"])

        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        else:
            # Keep multiplication small
            a = rng.randint(1, 20)
            b = rng.randint(1, 20)
            result = a * b

        prompt = f"{a} {op} {b} ="

        items.append({
            "prompt": prompt,
            "target": " " + str(result),
            "type": "algorithmic",
            "source": "gen_arithmetic",
            "metadata": {"operation": op, "a": a, "b": b}
        })

    # Type 2: Counting
    for i in range(n // 3):
        word = rng.choice(["the", "a", "is", "cat", "dog", "and", "to"])
        sentence_words = []
        target_count = rng.randint(1, 5)
        total_words = rng.randint(8, 20)

        filler = ["big", "small", "red", "quick", "lazy", "old", "new", "bright",
                   "dark", "warm", "cold", "soft", "hard", "fast", "slow"]

        positions = sorted(rng.sample(range(total_words), min(target_count, total_words)))

        for j in range(total_words):
            if j in positions:
                sentence_words.append(word)
            else:
                sentence_words.append(rng.choice(filler))

        actual_count = sentence_words.count(word)
        sentence = " ".join(sentence_words)
        prompt = f'How many times does "{word}" appear in: "{sentence}"? Answer:'

        items.append({
            "prompt": prompt,
            "target": " " + str(actual_count),
            "type": "algorithmic",
            "source": "gen_counting",
            "metadata": {"word": word, "count": actual_count}
        })

    # Type 3: Letter/word reversal and manipulation
    for i in range(n - len(items)):
        task_type = rng.choice(["reverse_word", "first_letters", "last_letter"])

        if task_type == "reverse_word":
            word = rng.choice(["hello", "world", "python", "model", "brain", "heart",
                              "light", "stone", "river", "cloud"])
            reversed_word = word[::-1]
            prompt = f'Reverse the word "{word}": '
            target = reversed_word

        elif task_type == "first_letters":
            words = rng.sample(["Apple", "Banana", "Cherry", "Date", "Elderberry",
                               "Fig", "Grape", "Honeydew", "Ice", "Jackfruit",
                               "Kiwi", "Lemon", "Mango", "Nectarine", "Orange"],
                              rng.randint(3, 6))
            first_letters = "".join(w[0] for w in words)
            prompt = "First letters of " + ", ".join(words) + ":"
            target = " " + first_letters

        else:  # last_letter
            word = rng.choice(["elephant", "giraffe", "penguin", "dolphin", "butterfly",
                              "mountain", "rainbow", "thunder", "whisper", "crystal"])
            prompt = f'The last letter of "{word}" is'
            target = " " + word[-1]

        items.append({
            "prompt": prompt,
            "target": target if target.startswith(" ") else " " + target,
            "type": "algorithmic",
            "source": f"gen_{task_type}",
            "metadata": {"task": task_type}
        })

    for i, item in enumerate(items[:n]):
        tok_count = approx_tokens(item["prompt"])
        item.update({
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
        })
        item["id"] = make_id("alg", item["source"], i)

    return items[:n]


# ---------------------------------------------------------------------------
# Generators: Reasoning (Numerical + Tracking)
# ---------------------------------------------------------------------------

def generate_reasoning_numerical(n: int = 40, seed: int = 47) -> list[dict]:
    """Generate numerical reasoning prompts.

    Multi-step arithmetic word problems where the model must
    track quantities and compute a final answer.
    """
    rng = random.Random(seed)
    items = []

    # Simpler approach: generate directly
    names_pool = [
        "Alice", "Ava", "Ben", "Bob", "Caleb", "Carol", "Charlotte", "Chloe",
        "Daniel", "Dave", "Eli", "Elena", "Ella", "Emma", "Ethan", "Eva",
        "Eve", "Felix", "Finn", "Frank", "Gabriel", "Grace", "Hank", "Hazel",
        "Henry", "Iris", "Isaac", "Ivy", "Jack", "Jade", "James", "Jasper",
        "Julia", "Kai", "Kate", "Kim", "Leo", "Liam", "Lila", "Lily",
        "Lucas", "Lucy", "Luna", "Maya", "Mia", "Milo", "Naomi", "Natalie",
        "Nathan", "Nina", "Noah", "Nora", "Oliver", "Olivia", "Owen", "Paige",
        "Penelope", "Quinn", "Ruby", "Ryan", "Sadie", "Sam", "Sofia", "Stella",
        "Theo", "Thomas", "Vera", "Violet", "Willow", "Wyatt", "Zoe",
    ]
    objects_pool = [
        "apples", "backpacks", "balloons", "baseballs", "beads", "blocks",
        "books", "bottles", "bracelets", "candies", "cards", "carrots",
        "chairs", "chocolates", "coins", "cookies", "crayons", "cups",
        "envelopes", "erasers", "flowers", "folders", "gems", "glasses",
        "hats", "jars", "kites", "lanterns", "markers", "marbles", "mugs",
        "notebooks", "oranges", "packages", "paintbrushes", "papers",
        "pencils", "pears", "pebbles", "photos", "plates", "postcards",
        "puzzles", "rocks", "rulers", "seashells", "seeds", "shells",
        "shirts", "spoons", "stars", "stickers", "stones", "tickets",
        "tokens", "toys", "umbrellas", "vouchers", "widgets",
    ]

    for i in range(n):
        task_variant = i % 8
        name1, name2 = rng.sample(names_pool, 2)
        obj = rng.choice(objects_pool)

        if task_variant == 0:
            # Addition + subtraction
            a = rng.randint(5, 30)
            b = rng.randint(1, a - 1)
            c = rng.randint(1, 15)
            result = a - b + c
            prompt = f"{name1} had {a} {obj}. {name1} gave {b} to {name2} and then found {c} more. How many {obj} does {name1} have now? Answer:"
            target = f" {result}"

        elif task_variant == 1:
            # Multi-step multiplication
            groups = rng.randint(2, 6)
            per = rng.randint(2, 8)
            extra = rng.randint(0, 5)
            result = groups * per + extra
            prompt = f"There are {groups} boxes with {per} {obj} in each box, plus {extra} extra {obj}. The total number of {obj} is"
            target = f" {result}"

        elif task_variant == 2:
            # Age comparison
            age1 = rng.randint(15, 50)
            diff = rng.randint(2, 15)
            direction = rng.choice(["older", "younger"])
            age2 = age1 + diff if direction == "older" else age1 - diff
            prompt = f"{name1} is {age1} years old. {name2} is {diff} years {direction} than {name1}. {name2} is"
            target = f" {age2}"

        elif task_variant == 3:
            # Fraction/division
            total = rng.choice([12, 15, 18, 20, 24, 30, 36])
            divisor = rng.choice([d for d in [2, 3, 4, 5, 6] if total % d == 0])
            result = total // divisor
            prompt = f"{name1} has {total} {obj} and wants to split them equally among {divisor} friends. Each friend gets"
            target = f" {result}"

        elif task_variant == 4:
            # Two-step comparison
            a = rng.randint(5, 30)
            b_more = rng.randint(1, 10)
            b = a + b_more
            c_less = rng.randint(1, min(b - 1, 10))
            c = b - c_less
            prompt = f"{name1} has {a} {obj}. {name2} has {b_more} more than {name1}. {rng.choice(names_pool)} has {c_less} fewer than {name2}. {rng.choice(names_pool)} has"
            # This gets ambiguous with random names, let's simplify
            name3 = rng.choice([n for n in names_pool if n not in (name1, name2)])
            prompt = f"{name1} has {a} {obj}. {name2} has {b_more} more than {name1}. {name3} has {c_less} fewer than {name2}. How many does {name3} have? Answer:"
            target = f" {c}"

        elif task_variant == 5:
            # Repeated addition across days
            start = rng.randint(4, 20)
            day1 = rng.randint(1, 8)
            day2 = rng.randint(1, 8)
            day3 = rng.randint(1, 8)
            result = start + day1 + day2 + day3
            prompt = (
                f"{name1} started with {start} {obj}. On Monday {name1} got {day1} more, "
                f"on Tuesday got {day2} more, and on Wednesday got {day3} more. "
                f"How many {obj} does {name1} have in total? Answer:"
            )
            target = f" {result}"

        elif task_variant == 6:
            # Equal groups after removing some items
            groups = rng.choice([2, 3, 4, 5, 6, 8])
            per_group = rng.randint(2, 8)
            removed = rng.randint(1, 8)
            remaining = groups * per_group
            total = remaining + removed
            result = per_group
            prompt = (
                f"A box had {total} {obj}. {name1} removed {removed} of them. "
                f"The remaining {obj} were divided equally into {groups} bags. "
                f"How many {obj} were placed in each bag? Answer:"
            )
            target = f" {result}"

        else:
            # Combined purchase and sharing
            start = rng.randint(10, 30)
            bought = rng.randint(2, 12)
            given = rng.randint(1, min(8, start + bought - 1))
            final_total = start + bought - given
            multiplier = rng.choice([2, 3, 4])
            result = final_total * multiplier
            prompt = (
                f"{name1} had {start} {obj}. Then {name1} bought {bought} more and gave "
                f"{given} to {name2}. If {multiplier} people each have that same final number "
                f"of {obj}, how many {obj} do they have altogether? Answer:"
            )
            target = f" {result}"

        tok_count = approx_tokens(prompt)
        items.append({
            "id": make_id("rn", "gen", i),
            "prompt": prompt,
            "target": target,
            "type": "reasoning_numerical",
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
            "source": "gen_reasoning_num",
            "metadata": {"variant": task_variant}
        })

    return items[:n]


def generate_reasoning_tracking(n: int = 40, seed: int = 48) -> list[dict]:
    """Generate state-tracking reasoning prompts.

    These require tracking the state of entities through a narrative:
    - Object location tracking
    - Possession tracking
    - State change tracking
    """
    rng = random.Random(seed)
    items = []

    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
    locations = ["kitchen", "bedroom", "garden", "garage", "living room",
                 "bathroom", "attic", "basement", "office", "hallway"]
    objects = ["book", "key", "ball", "hat", "phone", "cup", "pen", "ring", "bag", "toy"]
    containers = ["box", "drawer", "shelf", "basket", "cabinet", "bag", "pocket"]

    for i in range(n):
        variant = i % 4

        if variant == 0:
            # Object movement tracking (Sally-Anne style)
            n1, n2 = rng.sample(names, 2)
            obj = rng.choice(objects)
            loc1, loc2, loc3 = rng.sample(locations, 3)

            prompt = (f"{n1} put the {obj} in the {loc1}. "
                     f"{n1} went to the {loc2}. "
                     f"{n2} moved the {obj} to the {loc3}. "
                     f"Where is the {obj} now? The {obj} is in the")
            target = f" {loc3}"

        elif variant == 1:
            # Multi-hop object tracking
            n1, n2, n3 = rng.sample(names, 3)
            obj = rng.choice(objects)
            locs = rng.sample(locations, 4)

            prompt = (f"The {obj} starts in the {locs[0]}. "
                     f"{n1} moves it to the {locs[1]}. "
                     f"{n2} moves it to the {locs[2]}. "
                     f"{n3} moves it to the {locs[3]}. "
                     f"Where is the {obj}? Answer: the")
            target = f" {locs[3]}"

        elif variant == 2:
            # Container nesting
            obj = rng.choice(objects)
            c1, c2 = rng.sample(containers, 2)
            loc = rng.choice(locations)

            prompt = (f"The {obj} is in the {c1}. "
                     f"The {c1} is in the {c2}. "
                     f"The {c2} is in the {loc}. "
                     f"Where is the {obj}? The {obj} is in the")
            target = f" {c1}"  # Immediate container

        else:
            # Possession tracking with transfers
            n1, n2, n3 = rng.sample(names, 3)
            obj = rng.choice(objects)

            # Chain of transfers
            steps = rng.randint(2, 4)
            people = [n1, n2, n3]
            current_holder = rng.choice(people)
            narrative = [f"{current_holder} has the {obj}."]

            for s in range(steps):
                receiver = rng.choice([p for p in people if p != current_holder])
                narrative.append(f"{current_holder} gives the {obj} to {receiver}.")
                current_holder = receiver

            prompt = " ".join(narrative) + f" Who has the {obj}?"
            target = f" {current_holder}"

        tok_count = approx_tokens(prompt)
        items.append({
            "id": make_id("rt", "gen", i),
            "prompt": prompt,
            "target": target,
            "type": "reasoning_tracking",
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
            "source": "gen_reasoning_track",
            "metadata": {"variant": variant}
        })

    return items[:n]


# ---------------------------------------------------------------------------
# Generators: Syntactic Pattern
# ---------------------------------------------------------------------------

def _lm_syneval_switch_number(words: list[str], verb: bool = False) -> list[str]:
    """Switch singular/plural number for LM_syneval-style terminals."""
    switched = []
    for word in words:
        if word.split()[0] == "is":
            switched.append(" ".join(["are"] + word.split()[1:]))
        elif verb:
            if len(word.split()) > 1:
                switched.append(" ".join([word.split()[0][:-1]] + word.split()[1:]))
            else:
                switched.append(word[:-1])
        else:
            switched.append(word + "s")
    return switched


def _lm_syneval_get_case_name(
    preterms: list[str],
    match: tuple[list[int], list[int]],
    vary: list[int],
    opt: str = "sing",
    v_opt: str = "sing",
) -> str:
    """Recreate LM_syneval's variant naming to filter sing_* cases."""
    parts = [opt]
    for group in match:
        parts.extend(preterms[idx] for idx in group)
    if vary:
        parts.append(v_opt)
        parts.extend(preterms[idx] for idx in vary)
    return "_".join(parts)


def _lm_syneval_switch_numbers(
    base_sent: list[list[str]],
    variables: list[int],
    preterms: list[str],
) -> list[list[str]]:
    """Switch the number for selected terminal slots."""
    new_sent = base_sent[:]
    for idx in variables:
        new_sent[idx] = _lm_syneval_switch_number(
            new_sent[idx],
            verb=preterms[idx].endswith("V"),
        )
    return new_sent


def _lm_syneval_make_variable_sents(
    terminals: dict[str, list[str]],
    preterms: list[str],
    match: tuple[list[int], list[int]],
    vary: list[int],
) -> dict[str, tuple[list[list[str]], list[list[str]]]]:
    """Generate grammatical/ungrammatical sentence templates per case variant."""
    all_sents = {}
    base_sent = [terminals[p] for p in preterms]
    prefixes = ["sing", "plur"]

    for i, prefix in enumerate(prefixes):
        singular_grammatical = base_sent[:]
        plural_grammatical = _lm_syneval_switch_numbers(base_sent, vary, preterms)

        singular_ungrammatical = _lm_syneval_switch_numbers(
            singular_grammatical,
            match[1],
            preterms,
        )
        plural_ungrammatical = _lm_syneval_switch_numbers(
            plural_grammatical,
            match[1],
            preterms,
        )

        if i == 1:
            singular_ungrammatical = _lm_syneval_switch_numbers(
                singular_grammatical,
                match[0],
                preterms,
            )
            plural_ungrammatical = _lm_syneval_switch_numbers(
                plural_grammatical,
                match[0],
                preterms,
            )

        singular_grammatical = _lm_syneval_switch_numbers(
            singular_grammatical,
            match[0] + match[1],
            preterms,
        )
        plural_grammatical = _lm_syneval_switch_numbers(
            plural_grammatical,
            match[0] + match[1],
            preterms,
        )

        singular_name = _lm_syneval_get_case_name(preterms, match, vary, opt=prefix, v_opt="sing")
        all_sents[singular_name] = (singular_grammatical, singular_ungrammatical)

        if vary:
            plural_name = _lm_syneval_get_case_name(preterms, match, vary, opt=prefix, v_opt="plur")
            all_sents[plural_name] = (plural_grammatical, plural_ungrammatical)

    return all_sents


def _lm_syneval_expand_sentence(
    sent: list[list[str]],
    terminals: dict[str, list[str]],
    partial: str = "",
):
    """Expand LM_syneval terminal lists into concrete sentences."""
    if len(sent) == 1:
        for word in sent[0]:
            repeats_number_variant = (
                word.split(" ")[0] + " " + " ".join(word.split(" ")[1:]) in partial
                if word.endswith("s") and len(word.split()) > 1
                else False
            )
            repeats_singular_variant = (
                word.split(" ")[0][:-1] + " " + " ".join(word.split(" ")[1:]) in partial
                if word.endswith("s") and len(word.split()) > 1
                else False
            )
            repeats_copula_variant = (
                (word.startswith("is") and "are " + word[3:] in partial)
                or (word.startswith("are") and "is " + word[4:] in partial)
            )

            if (
                word not in partial
                and word not in terminals["D"]
                and word not in terminals["C"]
                and not repeats_number_variant
                and not repeats_singular_variant
                and not repeats_copula_variant
            ):
                yield partial + word
    else:
        for word in sent[0]:
            for expanded in _lm_syneval_expand_sentence(
                sent[1:],
                terminals=terminals,
                partial=partial + word + " ",
            ):
                yield expanded


def _lm_syneval_pair_to_item(
    good_sentence: str,
    bad_sentence: str,
    case_name: str,
    variant_name: str,
) -> dict | None:
    """Convert a grammatical/ungrammatical pair into a local cloze item."""
    good_tokens = good_sentence.split()
    bad_tokens = bad_sentence.split()
    if not good_tokens or not bad_tokens:
        return None

    first_diff = next(
        (idx for idx, (good_tok, bad_tok) in enumerate(zip(good_tokens, bad_tokens)) if good_tok != bad_tok),
        None,
    )
    if first_diff is None:
        return None

    good_end = len(good_tokens)
    bad_end = len(bad_tokens)
    while good_end > first_diff and bad_end > first_diff and good_tokens[good_end - 1] == bad_tokens[bad_end - 1]:
        good_end -= 1
        bad_end -= 1

    # Only keep pairs where the contrast occurs at the end, so the shared
    # prefix is a valid cloze prompt rather than a truncated sentence.
    if good_end != len(good_tokens) or bad_end != len(bad_tokens):
        return None

    good_span = good_tokens[first_diff:good_end]
    bad_span = bad_tokens[first_diff:bad_end]
    prefix = good_tokens[:first_diff]

    if len(prefix) < 2:
        return None
    if not (1 <= len(good_span) <= 4 and 1 <= len(bad_span) <= 4):
        return None

    prompt = " ".join(prefix)
    target = " " + " ".join(good_span)

    return {
        "prompt": prompt,
        "target": target,
        "type": "syntactic_pattern",
        "source": "lm_syneval",
        "metadata": {
            "pattern": "agreement_minimal_pair",
            "case": case_name,
            "variant": variant_name,
            "contrast": " ".join(bad_span),
        },
    }


def _build_lm_syneval_agreement_candidates(seed: int = 49) -> list[dict]:
    """Adapt a subset of LM_syneval agreement cases into cloze items."""
    rng = random.Random(seed)

    terminals = {
        "D": ["the"],
        "C": ["that"],
        "MS": ["author", "pilot", "surgeon", "farmer", "manager", "customer", "officer", "teacher", "senator", "consultant"],
        "ES": ["guard", "chef", "architect", "skater", "dancer", "minister", "taxi driver", "assistant", "executive", "parent"],
        "IS": ["movie", "book", "game", "song", "picture", "painting", "novel", "poem", "show"],
        "MV": ["laughs", "swims", "smiles", "is tall", "is old", "is young", "is short"],
        "EV": ["likes", "admires", "hates", "loves"],
        "IV": ["is good", "is bad", "is new", "is popular", "is unpopular", "brings joy to people", "interests people"],
        "P": ["next to", "behind", "in front of", "near", "to the side of", "across from"],
        "IP": ["from", "by"],
        "BS": ["mechanic", "banker"],
        "BV": ["said", "thought", "knew"],
        "AND": ["and"],
        "LMV": [
            "knows many different foreign languages",
            "likes to watch television shows",
            "is twenty three years old",
            "enjoys playing tennis with colleagues",
            "writes in a journal every day",
        ],
    }

    rules = {
        "simple_agrmt": (["D", "MS", "MV"], ([1], [2]), []),
        "prep_anim": (["D", "MS", "P", "D", "ES", "MV"], ([1], [5]), [4]),
        "prep_inanim": (["D", "IS", "IP", "D", "ES", "IV"], ([1], [5]), [4]),
        "subj_rel": (["D", "MS", "C", "EV", "D", "ES", "MV"], ([1, 3], [6]), [5]),
        "sent_comp": (["D", "BS", "BV", "D", "MS", "MV"], ([4], [5]), [1]),
        "vp_coord": (["D", "MS", "MV", "AND", "MV"], ([1, 2], [4]), []),
        "long_vp_coord": (["D", "MS", "LMV", "AND", "LMV"], ([1, 2], [4]), []),
        "obj_rel_across_anim": (["D", "MS", "C", "D", "ES", "EV", "MV"], ([1], [6]), [4, 5]),
        "obj_rel_within_anim": (["D", "MS", "C", "D", "ES", "EV", "MV"], ([4], [5]), [1, 6]),
        "obj_rel_no_comp_across_anim": (["D", "MS", "D", "ES", "EV", "MV"], ([1], [5]), [3, 4]),
        "obj_rel_no_comp_within_anim": (["D", "MS", "D", "ES", "EV", "MV"], ([3], [4]), [1, 5]),
    }

    candidates = []
    seen = set()

    for case_name, (preterms, match, vary) in rules.items():
        variant_templates = _lm_syneval_make_variable_sents(terminals, preterms, match, vary)
        for variant_name, (good_template, bad_template) in variant_templates.items():
            if not variant_name.startswith("plur_"):
                continue

            good_sentences = list(_lm_syneval_expand_sentence(good_template, terminals))
            bad_sentences = list(_lm_syneval_expand_sentence(bad_template, terminals))

            for good_sentence, bad_sentence in zip(good_sentences, bad_sentences):
                item = _lm_syneval_pair_to_item(good_sentence, bad_sentence, case_name, variant_name)
                if item is None:
                    continue

                key = (item["prompt"], item["target"])
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(item)

    rng.shuffle(candidates)
    return candidates


def generate_syntactic_pattern(n: int = 25, seed: int = 49) -> list[dict]:
    """Generate syntactic pattern completion prompts.

    These test the model's syntactic prediction ability:
    - Subject-verb agreement across distance
    - Garden path completions
    - Structural parallelism
    """
    rng = random.Random(seed)
    items = []

    # Type 1: Subject-verb agreement with intervening material
    subjects_sing = ["The cat", "The dog", "The teacher", "The scientist", "The child"]
    subjects_plur = ["The cats", "The dogs", "The teachers", "The scientists", "The children"]
    interveners = [
        "that the students admired",
        "near the old oak trees",
        "behind the tall buildings",
        "with the red markings",
        "from the small village",
        "next to the broken windows",
        "in the large auditorium",
    ]
    verbs_sing = ["runs", "walks", "speaks", "writes", "lives"]
    verbs_plur = ["run", "walk", "speak", "write", "live"]

    for i in range(n // 3):
        is_singular = rng.random() > 0.5
        subj = rng.choice(subjects_sing if is_singular else subjects_plur)
        interv = rng.choice(interveners)
        verb = rng.choice(verbs_sing if is_singular else verbs_plur)

        prompt = f"{subj} {interv}"
        target = f" {verb}"

        items.append({
            "prompt": prompt,
            "target": target,
            "type": "syntactic_pattern",
            "source": "gen_agreement",
            "metadata": {"singular": is_singular, "distance": len(interv.split())}
        })

    # Type 2: Structural parallelism
    parallel_templates = [
        ("She likes reading, writing, and", " painting"),
        ("The old man sat down, picked up his book, and began to", " read"),
        ("They came, they saw, they", " conquered"),
        ("Not only is she smart, but she is also very", " kind"),
        ("The more you practice, the better you", " get"),
        ("If it rains, we stay inside; if it shines, we go", " outside"),
        ("He ran quickly, jumped high, and landed", " softly"),
        ("First we eat, then we drink, and finally we", " sleep"),
    ]

    for i in range(n // 3):
        template = rng.choice(parallel_templates)
        prompt, target = template

        items.append({
            "prompt": prompt,
            "target": target,
            "type": "syntactic_pattern",
            "source": "gen_parallel",
            "metadata": {"pattern": "parallelism"}
        })

    # Type 3: LM_syneval-style agreement extraction
    agreement_candidates = _build_lm_syneval_agreement_candidates(seed=seed)
    for item in agreement_candidates[:max(0, n - len(items))]:
        items.append(item)

    unique_items = []
    seen_prompts = set()
    for item in items:
        if item["prompt"] in seen_prompts:
            continue
        seen_prompts.add(item["prompt"])
        unique_items.append(item)

    if len(unique_items) < n:
        for item in agreement_candidates:
            if item["prompt"] in seen_prompts:
                continue
            seen_prompts.add(item["prompt"])
            unique_items.append(item)
            if len(unique_items) >= n:
                break

    for i, item in enumerate(unique_items[:n]):
        if item["prompt"] and item["prompt"][0].islower():
            item["prompt"] = item["prompt"][0].upper() + item["prompt"][1:]
        tok_count = approx_tokens(item["prompt"])
        item.update({
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
        })
        item["id"] = make_id("sp", item.get("source", "gen"), i)

    return unique_items[:n]


# ---------------------------------------------------------------------------
# Generators: Long-Range Retrieval
# ---------------------------------------------------------------------------

def generate_long_range_retrieval(n: int = 30, seed: int = 50) -> list[dict]:
    """Generate long-range retrieval prompts.

    A fact is stated early in the context, then the model must retrieve it
    after intervening material.
    """
    rng = random.Random(seed)
    items = []

    facts = [
        ("The password is {word}.", "What is the password? The password is", " {word}"),
        ("The meeting is at {time}.", "When is the meeting? It is at", " {time}"),
        ("The color of the house is {color}.", "What color is the house? The house is", " {color}"),
        ("{name}'s favorite food is {food}.", "What is {name}'s favorite food?", " {food}"),
        ("The capital of the kingdom is {city}.", "What is the capital? The capital is", " {city}"),
    ]

    fillers = [
        "The weather was pleasant that day, with a gentle breeze blowing through the trees.",
        "Several merchants had set up their stalls along the main road, selling various goods.",
        "In the distance, the mountains rose majestically against the clear blue sky.",
        "The river flowed quietly through the valley, its waters reflecting the afternoon sun.",
        "Children played in the streets, their laughter echoing between the stone buildings.",
        "An old scholar sat in the library, surrounded by countless volumes of ancient texts.",
        "The market was bustling with activity as traders from distant lands displayed their wares.",
        "A group of travelers rested beneath a large oak tree, sharing stories of their journeys.",
        "The garden was filled with flowers of every color, tended by a dedicated gardener.",
        "Night fell slowly over the town, and lanterns were lit one by one along the pathways.",
        "A blacksmith worked at his forge, the rhythmic sound of his hammer ringing through the air.",
        "The festival preparations were underway, with decorations being hung across every doorway.",
    ]

    words = ["diamond", "phoenix", "crystal", "emerald", "cascade", "horizon", "twilight"]
    times = ["3pm", "noon", "midnight", "dawn", "sunset", "9am"]
    colors = ["blue", "red", "green", "white", "golden", "silver"]
    food_names = [("Emma", "pasta"), ("Liam", "sushi"), ("Ava", "chocolate"),
                 ("Noah", "pizza"), ("Sophia", "tacos")]
    cities = ["Avalon", "Meridian", "Northwatch", "Silverport", "Eastholm"]

    for i in range(n):
        fact_template = rng.choice(facts)
        fact_text, question, target_template = fact_template

        # Fill in the specific value
        if "{word}" in fact_text:
            word = rng.choice(words)
            fact_filled = fact_text.format(word=word)
            target = target_template.format(word=word)
            question_filled = question
        elif "{time}" in fact_text:
            time = rng.choice(times)
            fact_filled = fact_text.format(time=time)
            target = target_template.format(time=time)
            question_filled = question
        elif "{color}" in fact_text:
            color = rng.choice(colors)
            fact_filled = fact_text.format(color=color)
            target = target_template.format(color=color)
            question_filled = question
        elif "{name}" in fact_text:
            name, food = rng.choice(food_names)
            fact_filled = fact_text.format(name=name, food=food)
            target = target_template.format(food=food)
            question_filled = question.format(name=name)
        elif "{city}" in fact_text:
            city = rng.choice(cities)
            fact_filled = fact_text.format(city=city)
            target = target_template.format(city=city)
            question_filled = question

        # Add varying amounts of filler and insert the fact at a randomized
        # early position, biased toward the front half of the context.
        n_fillers = rng.randint(3, 8)
        sampled_fillers = rng.sample(fillers, min(n_fillers, len(fillers)))
        insertion_slots = len(sampled_fillers) + 1
        max_insert_idx = max(1, insertion_slots // 2)
        insert_mode = max(0, round(max_insert_idx * 0.6))
        fact_insert_idx = min(
            max_insert_idx,
            int(round(rng.triangular(0, max_insert_idx, insert_mode))),
        )

        context_parts = (
            sampled_fillers[:fact_insert_idx]
            + [fact_filled]
            + sampled_fillers[fact_insert_idx:]
            + [question_filled]
        )
        prompt = " ".join(context_parts)

        tok_count = approx_tokens(prompt)
        items.append({
            "id": make_id("lr", "gen", i),
            "prompt": prompt,
            "target": target,
            "type": "long_range_retrieval",
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
            "source": "gen_long_range",
            "metadata": {
                "n_fillers": n_fillers,
                "fact_insert_idx": fact_insert_idx,
                "fact_insert_frac": round(fact_insert_idx / max(1, len(sampled_fillers)), 3),
            }
        })

    return items[:n]


# ---------------------------------------------------------------------------
# Generators: Domain Knowledge
# ---------------------------------------------------------------------------

def generate_domain_knowledge(n: int = 60, seed: int = 51) -> list[dict]:
    """Generate domain knowledge prompts.

    These test specialized knowledge in science, history, geography, etc.
    Longer prompts that establish a domain context.

    Sources:
      - 18 hand-crafted "easy" items (curated_domain)
      - 60 items curated from random Wikipedia pages (gen-wikipedia-random)
      - 20 long-form Wikipedia-derived items (gen-wikipedia-random-long)
    """
    rng = random.Random(seed)

    # Hand-crafted domain knowledge items with specific targets
    domain_items = [
        {
            "prompt": "In organic chemistry, the process by which a carboxylic acid reacts with an alcohol to form an ester and water is called",
            "target": " esterification",
            "source": "curated_domain",
        },
        {
            "prompt": "The phenomenon in quantum mechanics where measuring one particle instantaneously affects another particle regardless of distance is known as quantum",
            "target": " entanglement",
            "source": "curated_domain",
        },
        {
            "prompt": "In machine learning, the technique of randomly dropping neurons during training to prevent overfitting is called",
            "target": " dropout",
            "source": "curated_domain",
        },
        {
            "prompt": "The treaty that ended World War I and imposed heavy reparations on Germany was the Treaty of",
            "target": " Versailles",
            "source": "curated_domain",
        },
        {
            "prompt": "In economics, the theory that increasing the money supply leads to proportional increases in price levels is known as the quantity theory of",
            "target": " money",
            "source": "curated_domain",
        },
        {
            "prompt": "The deepest known point in Earth's oceans, located in the western Pacific, is called the Mariana",
            "target": " Trench",
            "source": "curated_domain",
        },
        {
            "prompt": "In genetics, the molecule that carries amino acids to the ribosome during protein synthesis is transfer",
            "target": " RNA",
            "source": "curated_domain",
        },
        {
            "prompt": "The architectural style characterized by pointed arches, ribbed vaults, and flying buttresses that dominated medieval Europe is called",
            "target": " Gothic",
            "source": "curated_domain",
        },
        {
            "prompt": "In music theory, a chord consisting of a root, major third, and perfect fifth is called a major",
            "target": " triad",
            "source": "curated_domain",
        },
        {
            "prompt": "The mathematical constant approximately equal to 2.71828, which is the base of the natural logarithm, is known as Euler's",
            "target": " number",
            "source": "curated_domain",
        },
        {
            "prompt": "In neuroscience, the gap between two neurons where neurotransmitters are released to transmit signals is called the synaptic",
            "target": " cleft",
            "source": "curated_domain",
        },
        {
            "prompt": "The effect where light from distant galaxies is shifted toward longer wavelengths due to the expansion of the universe is called",
            "target": " redshift",
            "source": "curated_domain",
        },
        {
            "prompt": "In computer science, the data structure that follows the Last-In-First-Out principle is called a",
            "target": " stack",
            "source": "curated_domain",
        },
        {
            "prompt": "The philosophical thought experiment involving a cat that is simultaneously alive and dead until observed was proposed by",
            "target": " Schrödinger",
            "source": "curated_domain",
        },
        {
            "prompt": "In ecology, the maximum population size that an environment can sustain indefinitely is called the carrying",
            "target": " capacity",
            "source": "curated_domain",
        },
        {
            "prompt": "The ancient trade route connecting China to the Mediterranean, facilitating the exchange of silk, spices, and ideas, was known as the Silk",
            "target": " Road",
            "source": "curated_domain",
        },
        {
            "prompt": "In thermodynamics, the law stating that entropy of an isolated system always increases is the second law of",
            "target": " thermodynamics",
            "source": "curated_domain",
        },
        {
            "prompt": "The cognitive bias where people overestimate their abilities or knowledge is known as the Dunning-",
            "target": "Kruger",  # No leading space: follows hyphen
            "source": "curated_domain",
        },
    ]

    # Items curated from random Wikipedia pages (harder, more obscure)
    _data_dir = Path(__file__).resolve().parent.parent / "data"
    _sources_dir = _data_dir / "sources"

    def _load_json_data(filename: str) -> list[dict]:
        """Load durable source data, preferring data/sources during cleanup."""
        candidate_paths = [
            _sources_dir / filename,
            _data_dir / filename,  # Temporary fallback during layout migration.
        ]
        for path in candidate_paths:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        raise FileNotFoundError(
            f"Could not find {filename} in {_sources_dir} or {_data_dir}"
        )

    wikipedia_items = _load_json_data("wikipedia_domain_items.json")
    long_items = _load_json_data("long_domain_items.json")

    def _normalize_domain_item(item: dict, fallback_source: str) -> dict:
        """Preserve optional metadata/source from JSON-backed domain items."""
        normalized = dict(item)
        normalized["source"] = normalized.get("source", fallback_source)
        normalized["metadata"] = normalized.get("metadata", {})
        return normalized

    # JSON-backed domain items live in:
    #   battery/data/sources/wikipedia_domain_items.json
    #   battery/data/sources/long_domain_items.json
    all_items = (
        [_normalize_domain_item(item, "curated_domain") for item in domain_items]
        + [_normalize_domain_item(item, "gen-wikipedia-random") for item in wikipedia_items]
        + [_normalize_domain_item(item, "gen-wikipedia-random-long") for item in long_items]
    )


    rng.shuffle(all_items)
    selected = all_items[:n]

    for i, item in enumerate(selected):
        tok_count = approx_tokens(item["prompt"])
        item.update({
            "id": make_id("dk", item.get("source", "curated"), i),
            "type": "domain_knowledge",
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
            "metadata": item.get("metadata", {})
        })

    return selected


# ---------------------------------------------------------------------------
# Generators: Code Comprehension
# ---------------------------------------------------------------------------

def generate_code_comprehension(n: int = 15, seed: int = 52) -> list[dict]:
    """Generate code comprehension prompts.

    These present short code snippets and ask what they output.
    """
    rng = random.Random(seed)

    code_items = [
        {
            "prompt": '```python\nx = 5\ny = 3\nprint(x + y)\n```\nOutput:',
            "target": " 8",
        },
        {
            "prompt": '```python\ndef f(n):\n    return n * 2\nprint(f(7))\n```\nOutput:',
            "target": " 14",
        },
        {
            "prompt": '```python\nwords = ["hello", "world"]\nprint(len(words))\n```\nOutput:',
            "target": " 2",
        },
        {
            "prompt": '```python\nx = [1, 2, 3, 4, 5]\nprint(x[2])\n```\nOutput:',
            "target": " 3",
        },
        {
            "prompt": '```python\nx = "hello"\nprint(x.upper())\n```\nOutput:',
            "target": " HELLO",
        },
        {
            "prompt": '```python\nfor i in range(3):\n    pass\nprint(i)\n```\nOutput:',
            "target": " 2",
        },
        {
            "prompt": '```python\nd = {"a": 1, "b": 2}\nprint(d["b"])\n```\nOutput:',
            "target": " 2",
        },
        {
            "prompt": '```python\nx = 10\nif x > 5:\n    x = x - 3\nprint(x)\n```\nOutput:',
            "target": " 7",
        },
        {
            "prompt": '```python\ns = "abcdef"\nprint(s[:3])\n```\nOutput:',
            "target": " abc",
        },
        {
            "prompt": '```python\nx = [1, 2, 3]\nx.append(4)\nprint(len(x))\n```\nOutput:',
            "target": " 4",
        },
        {
            "prompt": '```python\ndef add(a, b=10):\n    return a + b\nprint(add(5))\n```\nOutput:',
            "target": " 15",
        },
        {
            "prompt": '```python\nx = 17\nprint(x % 5)\n```\nOutput:',
            "target": " 2",
        },
        {
            "prompt": '```python\nx = True\ny = False\nprint(x and y)\n```\nOutput:',
            "target": " False",
        },
        {
            "prompt": '```python\nprint(max(3, 7, 1, 9, 4))\n```\nOutput:',
            "target": " 9",
        },
        {
            "prompt": '```python\ns = "hello world"\nprint(s.count("l"))\n```\nOutput:',
            "target": " 3",
        },
        {
            "prompt": '```python\nx = [3, 1, 4, 1, 5]\nx.sort()\nprint(x[0])\n```\nOutput:',
            "target": " 1",
        },
    ]

    rng.shuffle(code_items)
    selected = code_items[:n]

    for i, item in enumerate(selected):
        tok_count = approx_tokens(item["prompt"])
        item.update({
            "id": make_id("cc", "gen", i),
            "type": "code_comprehension",
            "tokens_approx": tok_count,
            "tier": tier_from_tokens(tok_count),
            "source": "gen_code",
            "metadata": {}
        })

    return selected


# ---------------------------------------------------------------------------
# Type registry: maps type name -> (generator_fn, target_count, needs_dataset)
# ---------------------------------------------------------------------------

# Generator specs for types that don't need HuggingFace downloads
GENERATED_TYPES = {
    "structural_copying":   (generate_structural_copying,   60),
    "reasoning_numerical":  (generate_reasoning_numerical,  40),
    "reasoning_tracking":   (generate_reasoning_tracking,   40),
    "algorithmic":          (generate_algorithmic,           30),
    "syntactic_pattern":    (generate_syntactic_pattern,     25),
    "long_range_retrieval": (generate_long_range_retrieval,  30),
    "domain_knowledge":     (generate_domain_knowledge,      60),
    "code_comprehension":   (generate_code_comprehension,    15),
}

# Dataset-sourced types (need HuggingFace)
DATASET_TYPES = {
    "factual_recall":    (extract_factual_recall,    80, "counterfact"),
    "factual_retrieval": (extract_factual_retrieval, 40, "counterfact"),
    "cultural_memorized":(extract_cultural_memorized,25, "lambada"),
}


def load_recipe(recipe_path: str | None) -> dict[str, int]:
    """Load optional per-type count overrides from a JSON recipe."""
    if recipe_path is None:
        return {}

    with open(recipe_path) as f:
        data = json.load(f)

    if "types" in data:
        data = data["types"]

    recipe: dict[str, int] = {}
    for type_name, value in data.items():
        if isinstance(value, dict):
            count = value.get("count")
        else:
            count = value

        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid recipe count for {type_name}: {count!r}")

        recipe[type_name] = count

    return recipe


def default_code_pool_path() -> str | None:
    """Return the default external code pool path when present."""
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "data" / "sources" / "code_comprehension_seed.json",
        root / "data" / "code_comprehension_seed.json",
        root / "data" / "battery_4" / "code_comprehension_seed.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def load_external_type_pool(pool_path: str, type_name: str, n: int, seed: int) -> list[dict]:
    """Load a pre-generated item pool for one type and sample n items."""
    with open(pool_path) as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError(f"External pool must be a JSON list: {pool_path}")

    filtered = [item for item in items if item.get("type") == type_name]
    if not filtered:
        raise ValueError(f"No items of type '{type_name}' found in external pool: {pool_path}")

    filtered = deduplicate(filtered)
    rng = random.Random(seed)
    rng.shuffle(filtered)
    return filtered[:n]


def build_type(
    type_name: str,
    cache_dir: str = None,
    seed: int = 42,
    n_override: int | None = None,
    code_pool_path: str | None = None,
) -> list[dict]:
    """Build items for a single type."""

    if type_name == "code_comprehension" and code_pool_path:
        _, default_n = GENERATED_TYPES[type_name]
        n = default_n if n_override is None else n_override
        print(f"Loading external code pool for {type_name} (n={n}) from {code_pool_path}...")
        return load_external_type_pool(code_pool_path, type_name, n=n, seed=seed)

    if type_name in GENERATED_TYPES:
        gen_fn, n = GENERATED_TYPES[type_name]
        if n_override is not None:
            n = n_override
        print(f"Generating {type_name} (n={n})...")
        return gen_fn(n=n, seed=seed)

    elif type_name in DATASET_TYPES:
        extract_fn, n, dataset_key = DATASET_TYPES[type_name]
        if n_override is not None:
            n = n_override

        if dataset_key == "counterfact":
            print(f"Loading COUNTERFACT for {type_name}...")
            ds = load_counterfact(cache_dir)
            print(f"  Loaded {len(ds)} items")
            return extract_fn(ds, n=n, seed=seed)

        elif dataset_key == "lambada":
            print(f"Loading LAMBADA for {type_name}...")
            ds = load_lambada(cache_dir)
            print(f"  Loaded {len(ds)} items")
            return extract_fn(ds, n=n, seed=seed)

    raise ValueError(f"Unknown type: {type_name}")


def deduplicate(items: list[dict]) -> list[dict]:
    """Remove duplicate prompts."""
    seen = set()
    unique = []
    for item in items:
        h = hashlib.md5(item["prompt"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(item)
    return unique


def report(items: list[dict], label: str = ""):
    """Print distribution summary."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    type_counts, tier_counts = {}, {}
    for item in items:
        type_counts[item["type"]] = type_counts.get(item["type"], 0) + 1
        tier_counts[item["tier"]] = tier_counts.get(item["tier"], 0) + 1

    print(f"  Total: {len(items)}")
    print("  Types:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:25s}: {c}")
    print("  Tiers:")
    for t, c in sorted(tier_counts.items()):
        print(f"    {t:10s}: {c}")


# ---------------------------------------------------------------------------
# Main: one file per type + manifest
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build prompt battery for g-profile experiments.\n"
                    "Outputs one JSON file per type into --outdir, plus a manifest."
    )
    parser.add_argument("--outdir", type=str, default="battery",
                        help="Output directory (one JSON per type + manifest)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory for datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recipe", type=str, default=None,
                        help="Optional JSON recipe with per-type prompt counts")
    parser.add_argument("--code-pool", type=str, default=None,
                        help="Optional JSON pool for code_comprehension items")
    parser.add_argument("--no-datasets", action="store_true",
                        help="Skip HuggingFace dataset downloads (generators only)")
    parser.add_argument("--types", type=str, nargs="*", default=None,
                        help="Build only these types (default: all)")
    parser.add_argument("--smoke", action="store_true",
                        help="Build exactly one prompt per selected type")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine which types to build
    all_type_names = list(GENERATED_TYPES.keys())
    if not args.no_datasets:
        all_type_names += list(DATASET_TYPES.keys())

    if args.types:
        type_names = [t for t in args.types if t in all_type_names]
        skipped = [t for t in args.types if t not in all_type_names]
        if skipped:
            print(f"WARNING: unknown types skipped: {skipped}")
    else:
        type_names = all_type_names

    # Build each type into its own file
    manifest = {"types": {}, "seed": args.seed, "smoke": args.smoke}
    all_items = []
    n_override = 1 if args.smoke else None
    recipe_overrides = load_recipe(args.recipe)
    code_pool_path = args.code_pool or default_code_pool_path()

    if args.recipe:
        manifest["recipe"] = str(args.recipe)
    if code_pool_path:
        manifest["code_pool"] = str(code_pool_path)

    for type_name in type_names:
        try:
            type_n_override = n_override
            if type_n_override is None:
                type_n_override = recipe_overrides.get(type_name)
            items = build_type(
                type_name,
                cache_dir=args.cache_dir,
                seed=args.seed,
                n_override=type_n_override,
                code_pool_path=code_pool_path,
            )
            items = deduplicate(items)
        except Exception as e:
            print(f"  ERROR building {type_name}: {e}")
            continue

        # Write per-type file
        type_file = outdir / f"{type_name}.json"
        with open(type_file, "w") as f:
            json.dump(items, f, indent=2)

        manifest["types"][type_name] = {
            "file": f"{type_name}.json",
            "count": len(items),
            "tiers": {},
        }
        for item in items:
            tier = item["tier"]
            manifest["types"][type_name]["tiers"][tier] = \
                manifest["types"][type_name]["tiers"].get(tier, 0) + 1

        all_items.extend(items)
        print(f"  -> {type_file}: {len(items)} items")

    # Write manifest
    manifest["total"] = len(all_items)
    manifest_file = outdir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    # Also write a combined file for convenience (sweep.py can use either)
    combined_file = outdir / "all_candidates.json"
    with open(combined_file, "w") as f:
        json.dump(all_items, f, indent=2)

    report(all_items, f"Battery: {outdir}")
    print(f"\n  Manifest: {manifest_file}")
    print(f"  Combined: {combined_file}")
    print(f"  Per-type: {len(type_names)} files in {outdir}/")


if __name__ == "__main__":
    main()
