"""Pydantic settings models for VectorForge configuration.

All configuration is loaded from environment variables with the VECTORFORGE_ prefix.
Each sub-config uses its own ENV prefix (e.g., VECTORFORGE_DB_, VECTORFORGE_CHUNKING_).
A .env file is automatically loaded if present.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vectorforge.config.defaults import (
    DEFAULT_BREAKPOINT_THRESHOLD,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_DB_DATABASE,
    DEFAULT_DB_HOST,
    DEFAULT_DB_MAX_OVERFLOW,
    DEFAULT_DB_POOL_SIZE,
    DEFAULT_DB_PORT,
    DEFAULT_DB_USER,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_METRICS_FLUSH_INTERVAL_SECONDS,
    DEFAULT_S3_REGION,
    DEFAULT_STORAGE_BACKEND,
    DEFAULT_STORAGE_THRESHOLD_MB,
)


def _env_config(env_prefix: str) -> SettingsConfigDict:
    """Build a SettingsConfigDict with shared env-file settings.

    Args:
        env_prefix: The environment variable prefix for the config section.

    Returns:
        A SettingsConfigDict with the common .env loading behaviour.
    """
    return SettingsConfigDict(
        env_prefix=env_prefix,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class DatabaseConfig(BaseSettings):
    """PostgreSQL connection configuration."""

    model_config = _env_config("VECTORFORGE_DB_")

    host: str = DEFAULT_DB_HOST
    port: int = DEFAULT_DB_PORT
    database: str = DEFAULT_DB_DATABASE
    user: str = DEFAULT_DB_USER
    password: str = ""
    pool_size: int = DEFAULT_DB_POOL_SIZE
    max_overflow: int = DEFAULT_DB_MAX_OVERFLOW
    echo_sql: bool = False

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Ensure port is within valid TCP range."""
        if not 1 <= v <= 65535:
            msg = f"port must be between 1 and 65535, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("pool_size")
    @classmethod
    def validate_pool_size(cls, v: int) -> int:
        """Ensure pool_size is positive."""
        if v < 1:
            msg = f"pool_size must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @property
    def database_url(self) -> str:
        """Build the async PostgreSQL connection URL.

        User and password are URL-encoded to handle special characters
        (e.g., ``@``, ``/``, ``%``, ``:``) safely.
        """
        return (
            f"postgresql+asyncpg://{quote_plus(self.user)}:{quote_plus(self.password)}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class StorageConfig(BaseSettings):
    """Document storage configuration (PostgreSQL or S3)."""

    model_config = _env_config("VECTORFORGE_STORAGE_")

    default_backend: str = DEFAULT_STORAGE_BACKEND
    threshold_mb: int = DEFAULT_STORAGE_THRESHOLD_MB
    s3_bucket: str = ""
    s3_region: str = DEFAULT_S3_REGION
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_endpoint_url: str = ""

    @field_validator("threshold_mb")
    @classmethod
    def validate_threshold(cls, v: int) -> int:
        """Ensure threshold is positive."""
        if v < 1:
            msg = f"threshold_mb must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    def is_s3_configured(self) -> bool:
        """Check if S3 backend is properly configured."""
        return bool(self.s3_bucket)


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    model_config = _env_config("VECTORFORGE_EMBEDDING_")

    default_provider: str = DEFAULT_EMBEDDING_PROVIDER
    default_model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Ensure dimensions is positive."""
        if v < 1:
            msg = f"dimensions must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Ensure batch_size is positive."""
        if v < 1:
            msg = f"batch_size must be >= 1, got {v}"
            raise ValueError(msg)
        return v


class ChunkingConfig(BaseSettings):
    """Document chunking configuration."""

    model_config = _env_config("VECTORFORGE_CHUNKING_")

    strategy: str = DEFAULT_CHUNKING_STRATEGY
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    separators: list[str] | None = None
    model_name: str | None = None
    embedding_provider: str | None = None
    breakpoint_threshold: float = DEFAULT_BREAKPOINT_THRESHOLD

    @model_validator(mode="after")
    def validate_overlap_less_than_size(self) -> ChunkingConfig:
        """Ensure chunk_overlap is strictly less than chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            msg = f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            raise ValueError(msg)
        return self


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    model_config = _env_config("VECTORFORGE_LLM_")

    default_provider: str = DEFAULT_LLM_PROVIDER
    default_model: str = DEFAULT_LLM_MODEL
    temperature: float = DEFAULT_LLM_TEMPERATURE
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    system_prompt: str = ""

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is within valid range."""
        if not 0.0 <= v <= 2.0:
            msg = f"temperature must be between 0.0 and 2.0, got {v}"
            raise ValueError(msg)
        return v


class MonitoringConfig(BaseSettings):
    """Monitoring, logging, and health check configuration."""

    model_config = _env_config("VECTORFORGE_MONITORING_")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        DEFAULT_LOG_LEVEL  # type: ignore[assignment]
    )
    log_format: Literal["json", "text"] = (
        DEFAULT_LOG_FORMAT  # type: ignore[assignment]
    )
    log_file: str | None = None
    metrics_enabled: bool = True
    metrics_flush_interval_seconds: int = DEFAULT_METRICS_FLUSH_INTERVAL_SECONDS
    health_check_timeout_seconds: int = DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS

    @field_validator("metrics_flush_interval_seconds")
    @classmethod
    def validate_flush_interval(cls, v: int) -> int:
        """Ensure flush interval is positive."""
        if v < 1:
            msg = f"metrics_flush_interval_seconds must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("health_check_timeout_seconds")
    @classmethod
    def validate_health_timeout(cls, v: int) -> int:
        """Ensure health check timeout is positive."""
        if v < 1:
            msg = f"health_check_timeout_seconds must be >= 1, got {v}"
            raise ValueError(msg)
        return v


class VectorForgeConfig(BaseSettings):
    """Root configuration aggregating all sub-configs.

    Each sub-config reads its own ENV variables independently.
    The root config is the single entry point used by application code.
    """

    model_config = _env_config("VECTORFORGE_")

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


def load_config() -> VectorForgeConfig:
    """Load and validate the full VectorForge configuration from environment.

    Returns:
        Fully validated VectorForgeConfig instance.

    Raises:
        pydantic.ValidationError: If any configuration value is invalid.
    """
    return VectorForgeConfig()
