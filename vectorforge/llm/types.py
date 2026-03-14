"""LLM request/response types for the generation layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMRequestConfig(BaseModel):
    """Configuration for a single LLM generation request.

    Args:
        model: The model identifier (e.g. ``gpt-4o``, ``claude-sonnet-4-20250514``).
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling threshold.
        frequency_penalty: Penalise frequent tokens.
        presence_penalty: Penalise already-present tokens.
        stop_sequences: Sequences that stop generation.
    """

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """Structured response from an LLM provider."""

    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, object] = Field(default_factory=dict)
