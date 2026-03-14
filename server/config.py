"""API-specific configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    """Configuration for the REST API server.

    All values are loaded from environment variables prefixed with
    ``VECTORFORGE_API_``.  Defaults are tuned for local development.
    """

    model_config = SettingsConfigDict(
        env_prefix="VECTORFORGE_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["*"]
    api_key: str = ""
    auth_required: bool = False
    log_requests: bool = True
