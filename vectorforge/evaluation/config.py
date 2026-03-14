"""Evaluation configuration — settings for the evaluation module."""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationConfig(BaseSettings):
    """Configuration for the evaluation module.

    All settings are loaded from environment variables with the
    ``VECTORFORGE_EVALUATION_`` prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="VECTORFORGE_EVALUATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = False
    schedule_interval_hours: int = 24
    sample_size: int = 50
    sample_strategy: str = "recent"
    max_concurrent_evaluators: int = 3
    evaluation_timeout_seconds: int = 300
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o-mini"

    # Thresholds — trigger recommendations when scores fall below
    faithfulness_threshold: float = 0.7
    relevance_threshold: float = 0.6
    hallucination_threshold: float = 0.3
    coverage_threshold: float = 0.5

    @field_validator("sample_size")
    @classmethod
    def validate_sample_size(cls, v: int) -> int:
        """Ensure sample_size is positive."""
        if v < 1:
            msg = f"sample_size must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("schedule_interval_hours")
    @classmethod
    def validate_schedule_interval(cls, v: int) -> int:
        """Ensure schedule interval is at least 1."""
        if v < 1:
            msg = f"schedule_interval_hours must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator(
        "faithfulness_threshold",
        "relevance_threshold",
        "hallucination_threshold",
        "coverage_threshold",
    )
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure thresholds are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            msg = f"Threshold must be between 0.0 and 1.0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("sample_strategy")
    @classmethod
    def validate_sample_strategy(cls, v: str) -> str:
        """Ensure sample strategy is valid."""
        allowed = {"recent", "random", "worst_performing"}
        if v not in allowed:
            msg = f"sample_strategy must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v
