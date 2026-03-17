#!/usr/bin/env python3
"""Prompt adapters for battery calibration.

These adapters let calibration present the same underlying battery item in a
task-appropriate way without changing the stored candidate data itself.
"""


def _identity_prompt(prompt: str) -> str:
    """Leave prompts unchanged for types that already calibrate naturally."""
    return prompt


def _factual_suffix_prompt(prompt: str) -> str:
    """Add a light answer cue without changing the factual prompt content."""
    return f"{prompt}\nAnswer:"


def _sentence_suffix_prompt(prompt: str) -> str:
    """Add a light continuation cue for bare sentence fragments."""
    return f"{prompt}\nContinuation:"


ADAPTERS = {
    "identity": _identity_prompt,
    "factual_suffix_v2": _factual_suffix_prompt,
    "sentence_suffix_v2": _sentence_suffix_prompt,
}


TYPE_TO_ADAPTER = {
    "factual_recall": "factual_suffix_v2",
    "factual_retrieval": "factual_suffix_v2",
    "syntactic_pattern": "sentence_suffix_v2",
}


def get_adapter_name(item: dict) -> str:
    """Return the adapter name to use for a battery item."""
    return TYPE_TO_ADAPTER.get(item["type"], "identity")


def adapt_prompt(item: dict) -> tuple[str, str]:
    """Return the adapted prompt and adapter name for a battery item."""
    adapter_name = get_adapter_name(item)
    adapter = ADAPTERS[adapter_name]
    return adapter(item["prompt"]), adapter_name
