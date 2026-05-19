"""
LLM Client — src/unstruc/text/llm_client.py
============================================

Era 5 of the NLP evolution story: Foundation Models.

Scaling transformers to billions of parameters and training on internet-scale
corpora unlocks emergent abilities — reasoning, translation, coding — all from
predicting the next token.  No task-specific training or labelled data is
required for zero-shot classification; a handful of examples suffices for
few-shot classification.

This module is intentionally thin: it contains only API plumbing and response
parsing.  The system prompt encodes the task definition; no business logic
belongs here.

API key must be set via ANTHROPIC_API_KEY environment variable (load from
a .env file using python-dotenv before calling any function).

Public API
----------
  zero_shot(text)                          → int (1=positive, 0=negative)
  few_shot(text, examples)                 → int (1=positive, 0=negative)
  classify_batch(texts, mode, examples)    → list[int]
"""

from __future__ import annotations

import os
from typing import Literal

import anthropic

_CLIENT: anthropic.Anthropic | None = None

_MODEL         = "claude-haiku-4-5-20251001"
_SYSTEM_PROMPT = (
    "You are a sentiment classifier for Airbnb guest reviews. "
    "Classify the review as POSITIVE or NEGATIVE. "
    "Respond with exactly one word: POSITIVE or NEGATIVE."
)


def _get_client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _CLIENT


# ── Single-sample functions ───────────────────────────────────────────────────

def zero_shot(text: str, model: str = _MODEL) -> int:
    """
    Classify a single review with no examples.

    Demonstrates emergent instruction-following: the model has never seen
    Airbnb data during training yet produces reliable sentiment labels
    purely from the system prompt.

    Returns 1 (positive) or 0 (negative).
    """
    response = _get_client().messages.create(
        model    = model,
        max_tokens = 5,
        system   = _SYSTEM_PROMPT,
        messages = [{"role": "user", "content": str(text)}],
    )
    label = response.content[0].text.strip().upper()
    return 1 if "POSITIVE" in label else 0


def few_shot(text: str, examples: list[dict], model: str = _MODEL) -> int:
    """
    Classify a single review using in-context examples.

    Args:
        text     : review string to classify
        examples : list of {"text": str, "label": int} dicts
                   where label 1=positive, 0=negative
        model    : Anthropic model ID

    Returns 1 (positive) or 0 (negative).
    """
    example_lines = "\n".join(
        f"Review: {ex['text']}\n"
        f"Label: {'POSITIVE' if ex['label'] == 1 else 'NEGATIVE'}"
        for ex in examples
    )
    prompt = f"{example_lines}\n\nReview: {text}\nLabel:"

    response = _get_client().messages.create(
        model    = model,
        max_tokens = 5,
        system   = _SYSTEM_PROMPT,
        messages = [{"role": "user", "content": prompt}],
    )
    label = response.content[0].text.strip().upper()
    return 1 if "POSITIVE" in label else 0


# ── Batch wrapper ─────────────────────────────────────────────────────────────

def classify_batch(
    texts,
    mode:     Literal["zero_shot", "few_shot"] = "zero_shot",
    examples: list[dict] = None,
    model:    str        = _MODEL,
) -> list[int]:
    """
    Classify a list of review strings.

    Args:
        texts    : list / Series of review strings
        mode     : "zero_shot" or "few_shot"
        examples : required when mode="few_shot"; list of labelled dicts
        model    : Anthropic model ID

    Returns:
        List of binary labels (1=positive, 0=negative).

    Note: each call makes one API request.  Keep len(texts) small (≤200)
    to control cost in a classroom setting.
    """
    texts_list = list(texts)
    if mode == "zero_shot":
        return [zero_shot(t, model=model) for t in texts_list]
    return [few_shot(t, examples=examples or [], model=model) for t in texts_list]
