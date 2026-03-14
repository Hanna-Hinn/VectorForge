"""Shared LLM-as-judge helper for evaluation evaluators.

Provides a thin wrapper around the LLM provider to send a prompt and
parse structured JSON responses.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from vectorforge.exceptions import EvaluationError
from vectorforge.llm.base import BaseLLMProvider
from vectorforge.llm.types import LLMRequestConfig

logger = logging.getLogger(__name__)


async def judge(
    llm: BaseLLMProvider,
    prompt: str,
    *,
    model: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Send a prompt to the judge LLM and parse the JSON response.

    Args:
        llm: The LLM provider to use.
        prompt: The evaluation prompt.
        model: Model override (uses provider default if empty).
        temperature: Sampling temperature (0 for deterministic).
        max_tokens: Response length limit.

    Returns:
        Parsed JSON dict from the LLM response.

    Raises:
        EvaluationError: If the LLM call or JSON parsing fails.
    """
    config = LLMRequestConfig(
        model=model or llm.default_model(),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    messages = [
        {"role": "system", "content": "You are an evaluation judge. Respond only with valid JSON."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = await llm.generate(messages, config)
        return _parse_json(response.content)
    except json.JSONDecodeError as exc:
        msg = f"Judge LLM returned invalid JSON: {exc}"
        raise EvaluationError(msg) from exc
    except Exception as exc:
        if isinstance(exc, EvaluationError):
            raise
        msg = f"Judge LLM call failed: {exc}"
        raise EvaluationError(msg) from exc


def _parse_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response text.

    Handles cases where the LLM wraps JSON in markdown code blocks.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (```json and ```)
        inner = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        cleaned = inner.strip()
    return json.loads(cleaned)
