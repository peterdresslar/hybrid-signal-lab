#!/usr/bin/env python3
"""
Generate code-comprehension prompts from seeded programming concepts.

Workflow:
1. Sample a concept seed from curated subfamilies.
2. Ask an LLM to synthesize a short Python snippet whose output is the target.
3. Execute the snippet locally to validate determinism and exact output.
4. Write validated prompts as battery items.

Usage:
    python code_generate.py output.json 50
    python code_generate.py output.json 20 --append

Requires OPENROUTER_KEY environment variable.
"""

import argparse
import contextlib
import io
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import textwrap
import urllib.error
import urllib.request


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3-flash-preview"


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
    return f"cc_seed_{idx:04d}"


SEED_FAMILIES = {
    "strings": [
        ("easy", "slice a string and print the result"),
        ("easy", "count a character in a string"),
        ("easy", "replace a substring and print the new string"),
        ("medium", "split a string and print one selected part"),
        ("easy", "strip whitespace and print the cleaned string"),
        ("medium", "combine split and indexing to print a selected token"),
        ("medium", "slice after concatenating two short strings"),
    ],
    "lists": [
        ("easy", "append to a list and print its length"),
        ("medium", "sort a list and print one element"),
        ("easy", "reverse a list and print the first item"),
        ("easy", "sum a short list of integers"),
        ("medium", "mutate a list and print a selected value"),
        ("medium", "append and then index from the end of a list"),
        ("medium", "sort a list after mutation and print a middle element"),
    ],
    "dicts": [
        ("easy", "update a dictionary and print one key's value"),
        ("medium", "look up a key after mutation"),
        ("easy", "compute a simple derived value from two dictionary entries"),
        ("medium", "mutate a dictionary twice and print a derived total"),
        ("medium", "store a list inside a dictionary and print one nested value"),
    ],
    "booleans": [
        ("easy", "evaluate a boolean expression and print the result"),
        ("medium", "use an if/else branch and print a final value"),
        ("easy", "combine comparison operators and print a boolean"),
        ("medium", "use an if/else branch that selects between arithmetic results"),
    ],
    "loops": [
        ("medium", "run a short for-loop accumulator and print the final total"),
        ("medium", "iterate through a list and count matches"),
        ("medium", "use range in a loop and print the final counter-derived value"),
        ("medium", "update a running value inside a conditional loop and print the final result"),
        ("medium", "loop through a list and accumulate only values that satisfy a condition"),
    ],
    "functions": [
        ("easy", "define a short function and print its return value"),
        ("easy", "use a default argument in a function"),
        ("easy", "call a helper function once and print the result"),
        ("medium", "call the same function twice with different arguments and print a combined result"),
        ("medium", "use a short helper function inside another expression"),
    ],
    "indexing": [
        ("easy", "index into a list and print the selected element"),
        ("medium", "index into a tuple after a simple transformation"),
        ("easy", "use negative indexing and print the result"),
        ("medium", "slice a list and then index into the slice"),
        ("medium", "reverse a sequence and print a selected indexed value"),
    ],
    "aliasing": [
        ("hard", "create a list alias, mutate through one name, and print through the other"),
        ("hard", "copy a list with slicing, mutate the copy, and print from the original"),
        ("hard", "store a list in two variables, mutate one, and print a selected element"),
    ],
    "order_effects": [
        ("medium", "append, sort, and then print an indexed value"),
        ("medium", "reverse a list after extending it and print the first value"),
        ("easy", "update a variable twice in sequence and print the final result"),
    ],
    "nested_structures": [
        ("medium", "index into a list of dictionaries and print one nested value"),
        ("medium", "look up a key in a dictionary of lists and print one element"),
        ("hard", "mutate a nested list and print a selected nested value"),
    ],
    "shadowing": [
        ("hard", "reuse a variable name inside a loop and print the final outer variable"),
        ("hard", "reassign a variable after using it in a helper expression and print the final value"),
        ("hard", "overwrite a variable with a derived value and then use it in a branch"),
    ],
    "branch_mutation": [
        ("hard", "mutate a list inside an if/else branch and print a selected element"),
        ("hard", "update a dictionary in one branch and print a derived value afterward"),
        ("hard", "append to a list only under a condition and print the final structure"),
    ],
    "comprehensions": [
        ("hard", "use a list comprehension with a condition and print one selected result"),
        ("hard", "use a comprehension with a helper function and print the final list"),
        ("hard", "build a derived list with a comprehension and print its length or one value"),
    ],
}


GENERATION_SYSTEM_PROMPT = """You are generating short Python code-comprehension prompts for a language model battery.

You will be given a programming concept seed.

Your task:
1. Write a very short Python snippet.
2. The snippet must be deterministic and self-contained.
3. It must end with exactly one print statement whose output is the answer.
4. Avoid input(), randomness, datetime, file I/O, imports, exceptions, or environment dependence.
5. Keep it small: ideally 3-8 lines.
6. Prefer one-line or short outputs.
7. The output must be unambiguous and easy to validate exactly.
8. Use only basic Python features: strings, lists, dicts, arithmetic, loops, boolean logic, indexing, simple functions.

Output ONLY a JSON object with these keys:
  "code": the Python code snippet as a string, no markdown fences
  "family": a short family label
  "concept": a short concept label
  "difficulty": one of easy, medium, hard

No other text."""


def call_llm(messages: list[dict], api_key: str, model: str, max_tokens: int = 300, temperature: float = 0.7) -> str | None:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/hybrid-signal-lab",
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"].get("content")
            return content.strip() if content else None
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError, TimeoutError, ConnectionError) as e:
        print(f"  [warn] LLM call failed: {e}", file=sys.stderr)
        return None


def parse_json_response(content: str) -> dict | None:
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    if content.startswith("json"):
        content = content[4:].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def execute_code(code: str, python_bin: str, timeout_sec: int = 2) -> tuple[bool, str | None]:
    """Run code in a subprocess and capture stdout."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        path = tmp.name

    try:
        proc = subprocess.run(
            [python_bin, path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, None
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)

    if proc.returncode != 0:
        return False, None

    stdout = proc.stdout.rstrip("\n")
    stderr = proc.stderr.strip()
    if stderr:
        return False, None
    return True, stdout


def validate_candidate(result: dict, python_bin: str) -> dict | None:
    code = result.get("code", "").strip()
    if not code:
        return None
    if "input(" in code or "import " in code:
        return None

    ok1, out1 = execute_code(code, python_bin)
    ok2, out2 = execute_code(code, python_bin)
    if not ok1 or not ok2 or out1 is None or out2 is None or out1 != out2:
        return None

    if len(out1) == 0 or len(out1) > 30:
        return None
    if "\n" in out1:
        return None

    prompt = f"```python\n{code}\n```\nOutput:"
    tok_count = approx_tokens(prompt)

    return {
        "prompt": prompt,
        "target": " " + out1,
        "type": "code_comprehension",
        "tokens_approx": tok_count,
        "tier": tier_from_tokens(tok_count),
        "source": "seed_codegen",
        "metadata": {
            "family": result.get("family", ""),
            "concept": result.get("concept", ""),
            "difficulty": result.get("difficulty", ""),
        },
    }


def existing_prompts(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    try:
        with open(output_path) as f:
            items = json.load(f)
        return {item["prompt"] for item in items}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return set()


def build_seed_pool() -> list[tuple[str, str, str]]:
    pool: list[tuple[str, str, str]] = []
    for family, seeds in SEED_FAMILIES.items():
        for difficulty, seed in seeds:
            pool.append((family, difficulty, seed))
    return pool


def family_limit(num_prompts: int) -> int:
    """Soft cap for any single family within a generated batch."""
    return max(3, (num_prompts + 5) // 6)


def difficulty_weights() -> list[tuple[str, int]]:
    return [("easy", 2), ("medium", 5), ("hard", 4)]


def weighted_choice(rng: random.Random, weighted_items: list[tuple[str, int]]) -> str:
    labels, weights = zip(*weighted_items)
    return rng.choices(labels, weights=weights, k=1)[0]


def generate_items(num_prompts: int, api_key: str, model: str, python_bin: str, seed: int, existing: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    seed_pool = build_seed_pool()
    items: list[dict] = []
    seen_prompts = set(existing or set())
    seen_concepts: set[tuple[str, str]] = set()
    family_counts = {family: 0 for family in SEED_FAMILIES}
    max_per_family = family_limit(num_prompts)
    attempts = 0
    max_attempts = max(60, num_prompts * 20)

    while len(items) < num_prompts and attempts < max_attempts:
        attempts += 1
        eligible_families = [f for f, count in family_counts.items() if count < max_per_family]
        if not eligible_families:
            eligible_families = list(SEED_FAMILIES.keys())
        family = rng.choice(eligible_families)
        target_difficulty = weighted_choice(rng, difficulty_weights())
        family_seed_pool = [(difficulty, seed) for difficulty, seed in SEED_FAMILIES[family] if difficulty == target_difficulty]
        if not family_seed_pool:
            family_seed_pool = SEED_FAMILIES[family]
        difficulty, concept_seed = rng.choice(family_seed_pool)
        user_prompt = (
            f"Family: {family}\n"
            f"Target difficulty: {difficulty}\n"
            f"Concept seed: {concept_seed}\n\n"
            "Generate one short Python snippet following the rules."
        )
        content = call_llm(
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            api_key=api_key,
            model=model,
        )
        if not content:
            continue

        parsed = parse_json_response(content)
        if not parsed:
            continue

        parsed.setdefault("family", family)
        parsed.setdefault("concept", concept_seed)
        parsed.setdefault("difficulty", difficulty)
        item = validate_candidate(parsed, python_bin)
        if not item:
            continue

        if item["prompt"] in seen_prompts:
            continue
        concept_key = (item["metadata"]["family"], item["metadata"]["concept"])
        if concept_key in seen_concepts:
            continue
        if family_counts.get(item["metadata"]["family"], 0) >= max_per_family:
            continue
        if item["metadata"]["family"] == "booleans" and item["target"].strip() in {"True", "False"} and family_counts["booleans"] >= max(2, max_per_family - 1):
            continue

        seen_prompts.add(item["prompt"])
        seen_concepts.add(concept_key)
        family_counts[item["metadata"]["family"]] = family_counts.get(item["metadata"]["family"], 0) + 1
        items.append(item)
        print(f"  accepted {len(items)}/{num_prompts}: {item['metadata']['family']} :: {item['metadata']['concept']}")

    for idx, item in enumerate(items):
        item["id"] = make_id(idx)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate code-comprehension prompts from seeded concept families.")
    parser.add_argument("output", type=str, help="Output JSON path")
    parser.add_argument("num_prompts", type=int, help="Number of prompts to generate")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter used for validation")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_KEY is not set.", file=sys.stderr)
        return 1

    current_items = []
    existing = set()
    if args.append and os.path.exists(args.output):
        with open(args.output) as f:
            current_items = json.load(f)
        existing = {item["prompt"] for item in current_items}

    new_items = generate_items(
        num_prompts=args.num_prompts,
        api_key=api_key,
        model=args.model,
        python_bin=args.python_bin,
        seed=args.seed,
        existing=existing,
    )

    all_items = current_items + new_items
    for idx, item in enumerate(all_items):
        item["id"] = make_id(idx)

    with open(args.output, "w") as f:
        json.dump(all_items, f, indent=2)

    print(f"\nWrote {len(all_items)} total items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
