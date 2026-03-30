#!/usr/bin/env python3
"""
Generate cloze-style domain knowledge prompts from random Wikipedia articles.

Usage:
    python wikipedia_generate.py output.json 120
    python wikipedia_generate.py output.json 30 --min-tokens 200
    python wikipedia_generate.py output.json 50 --append

Requires OPENROUTER_KEY environment variable.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse


# ---------------------------------------------------------------------------
# Wikipedia fetching
# ---------------------------------------------------------------------------

WIKI_RANDOM_URL = "https://en.wikipedia.org/api/rest_v1/page/random/summary"


def fetch_random_article_summary() -> dict | None:
    """Fetch a random Wikipedia article summary. Returns dict with title, extract."""
    req = urllib.request.Request(
        WIKI_RANDOM_URL,
        headers={"User-Agent": "hybrid-signal-lab-battery/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return {
                "title": data.get("title", ""),
                "extract": data.get("extract", ""),
                "description": data.get("description", ""),
                "content_urls": data.get("content_urls", {}),
            }
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  [warn] Wikipedia fetch failed: {e}", file=sys.stderr)
        return None


def fetch_full_article_text(title: str) -> str | None:
    """Fetch full article extract via the MediaWiki API (plain text extracts)."""
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "exlimit": "1",
        "format": "json",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "hybrid-signal-lab-battery/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  [warn] Full article fetch failed for '{title}': {e}", file=sys.stderr)
    return None


def is_good_article(summary: dict, min_extract_chars: int = 150) -> bool:
    """Filter for articles with enough prose to generate a good cloze."""
    extract = summary.get("extract", "")
    title = summary.get("title", "")

    if len(extract) < min_extract_chars:
        return False

    # Skip list/disambiguation/index articles
    lower_title = title.lower()
    skip_patterns = [
        "list of", "index of", "outline of", "disambiguation",
        "glossary of", "timeline of", "table of",
    ]
    if any(p in lower_title for p in skip_patterns):
        return False

    return True


# ---------------------------------------------------------------------------
# LLM utilities
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3-flash-preview"


def call_llm(messages: list, api_key: str, model: str, max_tokens: int = 256, temperature: float = 0.3) -> str | None:
    """Generic LLM call via OpenRouter. Returns raw content string or None."""
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
            if content is None:
                return None
            return content.strip()
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  [warn] LLM call failed: {e}", file=sys.stderr)
        return None


def parse_json_response(content: str) -> dict | None:
    """Parse JSON from LLM response, stripping markdown fences if present."""
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


# ---------------------------------------------------------------------------
# Stage 1: Article filter
# ---------------------------------------------------------------------------

FILTER_SYSTEM_PROMPT = """You are a filter for a language model evaluation battery. You will receive the title and summary of a Wikipedia article. Your job is to decide whether this article can yield a good domain-knowledge cloze prompt.

A GOOD article for our purposes:
- Is about a scientific concept, technical process, historical event, cultural practice, natural phenomenon, or specialized domain topic
- Contains at least one domain-specific term, technical concept, or classification that could serve as a cloze target
- The target term should be INFERABLE from a well-written description, not just a proper noun that requires rote memorization

A BAD article for our purposes:
- Is primarily a biography where the only natural cloze target is the person's name
- Is about a specific village, municipality, or minor geographic location
- Is about a specific sports season, match, or athlete's career statistics
- Is about a specific album, song, TV episode, or media release
- Is about a specific building, road, or infrastructure unless it illustrates a notable architectural/engineering concept
- Is about a specific military unit, election, or political term unless it illustrates a broader concept

Output ONLY a JSON object with two keys:
  "accept": true or false
  "reason": a short phrase explaining why (e.g., "good: marine biology concept" or "reject: biography of minor politician")

No other text."""


def filter_article(title: str, extract: str, api_key: str, model: str) -> tuple[bool, str]:
    """Stage 1: Ask LLM whether this article is suitable for cloze generation.
    Returns (accepted, reason)."""
    content = call_llm(
        messages=[
            {"role": "system", "content": FILTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {title}\n\nSummary: {extract[:1500]}"},
        ],
        api_key=api_key,
        model=model,
        max_tokens=128,
        temperature=0.1,
    )
    if not content:
        return False, "LLM filter call failed"

    result = parse_json_response(content)
    if not result:
        return False, "unparseable filter response"

    return result.get("accept", False), result.get("reason", "no reason given")


# ---------------------------------------------------------------------------
# Stage 2: Cloze generation
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """You are generating a cloze-style prompt for a language model evaluation battery.

You will receive the text of a Wikipedia article. Your task:

1. Write a passage that draws ONLY on facts stated in the article. Do not add information from your own knowledge.
2. Let the article's depth guide your passage length:
   - For a short or narrowly-focused article, write a concise passage of 1-3 sentences.
   - For a rich, detailed article with substantial content, write a longer passage — up to several hundred words — that builds context progressively like a mini-essay from a specialist reference work.
   - In general, prefer LONGER passages when the article supports it. Dense, well-contextualized prompts are more valuable than short ones.
3. The final clause must be an incomplete sentence whose completion is a single key term, name, or short phrase from the article.
4. Prefer targets that are domain-specific vocabulary, technical terms, or classifications rather than extremely obscure proper nouns. Proper nouns are acceptable when they are the natural focal point of the article.
5. The target should be determinable from the passage content — a knowledgeable reader should be able to infer the answer.
6. Do NOT reveal the target word/phrase anywhere earlier in the passage.

Output ONLY a JSON object with exactly two keys:
  "prompt": the passage text ending mid-sentence (no trailing space)
  "target": the completion, with a leading space (e.g., " phototransistor")

Do not include any other text, explanation, markdown formatting, or commentary. Output raw JSON only."""


def generate_cloze(article_text: str, title: str, api_key: str, model: str = DEFAULT_MODEL) -> dict | None:
    """Stage 2: Call LLM to generate a cloze prompt from article text."""

    article_truncated = article_text[:6000]

    content = call_llm(
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Wikipedia article: \"{title}\"\n\n{article_truncated}"},
        ],
        api_key=api_key,
        model=model,
        max_tokens=2048,
        temperature=0.7,
    )
    if not content:
        return None

    result = parse_json_response(content)
    if not result:
        print(f"  [warn] unparseable generation response", file=sys.stderr)
        return None

    if "prompt" not in result or "target" not in result:
        print(f"  [warn] LLM response missing keys: {list(result.keys())}", file=sys.stderr)
        return None

    # Ensure target has leading space
    target = result["target"]
    if not target.startswith(" "):
        target = " " + target
    result["target"] = target

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Rough token count: split on whitespace, multiply by 1.3 for subword."""
    return int(len(text.split()) * 1.3)


def generate_items(
    num_prompts: int,
    min_tokens: int,
    api_key: str,
    source_tag: str = "gen-wikipedia-random",
    model: str = DEFAULT_MODEL,
    max_attempts_per_prompt: int = 20,
) -> list[dict]:
    """Generate num_prompts cloze items from random Wikipedia articles."""

    items = []
    attempts = 0
    max_total_attempts = num_prompts * max_attempts_per_prompt

    while len(items) < num_prompts and attempts < max_total_attempts:
        attempts += 1
        print(f"[{len(items) + 1}/{num_prompts}] Attempt {attempts}...", end=" ")
        sys.stdout.flush()

        # Unconditional rate limiting to respect Wikipedia API limits
        time.sleep(1.2)

        # 1. Fetch random article
        summary = fetch_random_article_summary()
        if not summary:
            print("skip (fetch failed)")
            continue

        title = summary["title"]
        print(f"'{title}'", end=" ")

        # 2. Check article quality from summary (cheap heuristic filter)
        if not is_good_article(summary):
            print("skip (too short or list article)")
            continue

        # 3. Stage 1: LLM filter — is this article suitable for a cloze?
        accepted, reason = filter_article(title, summary["extract"], api_key, model)
        if not accepted:
            print(f"skip (filter: {reason})")
            time.sleep(0.3)
            continue
        print(f"[filter: {reason}]", end=" ")

        # 4. Fetch full article text for better context
        full_text = fetch_full_article_text(title)
        if not full_text or len(full_text) < 500:
            # Fall back to summary extract
            full_text = summary["extract"]
            if len(full_text) < 300:
                print("skip (insufficient text)")
                continue

        # 5. Stage 2: Generate cloze via LLM
        result = generate_cloze(full_text, title, api_key, model=model)
        if not result:
            print("skip (LLM failed)")
            continue

        # 6. Basic validation
        prompt = result["prompt"]
        target = result["target"]

        prompt_tokens = approx_tokens(prompt)
        prompt_words = len(prompt.split())

        if prompt_tokens < min_tokens:
            print(f"skip (too short: ~{prompt_tokens} tokens, min {min_tokens})")
            continue

        if len(target.strip()) == 0:
            print("skip (empty target)")
            continue

        # Check target doesn't appear in prompt
        target_clean = target.strip().lower()
        if target_clean in prompt.lower():
            print(f"skip (target '{target.strip()}' appears in prompt)")
            continue

        item = {
            "prompt": prompt,
            "target": target,
            "source": source_tag,
        }

        items.append(item)
        print(f"OK (~{prompt_tokens} tokens, {prompt_words} words, target: '{target.strip()}')")

    return items


def main():
    parser = argparse.ArgumentParser(
        description="Generate cloze prompts from random Wikipedia articles."
    )
    parser.add_argument("output", type=str, help="Output JSON file path")
    parser.add_argument("num_prompts", type=int, help="Number of prompts to generate")
    parser.add_argument(
        "--min-tokens", type=int, default=50,
        help="Minimum token count for generated prompts (default: 50)"
    )
    parser.add_argument(
        "--source-tag", type=str, default="gen-wikipedia-random-battery4",
        help="Source tag for items (default: gen-wikipedia-random-battery4)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"OpenRouter model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing output file instead of overwriting"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_KEY")
    if not api_key:
        print("Error: OPENROUTER_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    model = args.model

    print(f"Generating {args.num_prompts} prompts (min ~{args.min_tokens} tokens)")
    print(f"Model: {model}")
    print(f"Source tag: {args.source_tag}")
    print(f"Output: {args.output}")
    print()

    items = generate_items(
        num_prompts=args.num_prompts,
        min_tokens=args.min_tokens,
        api_key=api_key,
        source_tag=args.source_tag,
        model=model,
    )

    # Handle append mode
    if args.append and os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        print(f"\nAppending {len(items)} new items to {len(existing)} existing items.")
        items = existing + items

    with open(args.output, "w") as f:
        json.dump(items, f, indent=2)

    print(f"\nDone. Wrote {len(items)} total items to {args.output}")


if __name__ == "__main__":
    main()
