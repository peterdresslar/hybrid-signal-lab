#!/usr/bin/env python3
"""Prompt adapters for battery calibration.

Calibration currently uses identity rendering only. The adapter layer remains
so artifacts record prompt-render policy explicitly and can be versioned if
task-specific adapters are reintroduced later.
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


PROMPT_RENDER_VERSION = "prompt_render_v4_identity_only"

# Active adapter mapping for the current prompt-rendering version.
TYPE_TO_ADAPTER = {}



def get_adapter_name(item: dict) -> str:
    """Return the adapter name to use for a battery item."""
    return TYPE_TO_ADAPTER.get(item["type"], "identity")


def adapt_prompt(item: dict) -> tuple[str, str, str]:
    """Return the adapted prompt, adapter name, and render version."""
    adapter_name = get_adapter_name(item)
    adapter = ADAPTERS[adapter_name]
    return adapter(item["prompt"]), adapter_name, PROMPT_RENDER_VERSION
