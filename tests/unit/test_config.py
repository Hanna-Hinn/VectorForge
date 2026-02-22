"""Unit tests for VectorForge configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vectorforge.config.settings import (
    ChunkingConfig,
    DatabaseConfig,
    EmbeddingConfig,
    LLMConfig,
    MonitoringConfig,
    StorageConfig,
    VectorForgeConfig,
)


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config loads with sensible defaults."""
        monkeypatch.delenv("VECTORFORGE_DB_HOST", raising=False)
        monkeypatch.delenv("VECTORFORGE_DB_PORT", raising=False)
        monkeypatch.delenv("VECTORFORGE_DB_USER", raising=False)
        monkeypatch.delenv("VECTORFORGE_DB_DATABASE", raising=False)
        config = DatabaseConfig(_env_file=None, password="secret")
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "vectorforge"
        assert config.user == "vectorforge"
        assert config.pool_size == 10
        assert config.max_overflow == 5
        assert config.echo_sql is False

    def test_database_url(self) -> None:
        """database_url builds the correct async connection string."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            database="mydb",
            user="admin",
            password="pw",
        )
        assert config.database_url == "postgresql+asyncpg://admin:pw@db.example.com:5433/mydb"

    def test_database_url_encodes_special_chars(self) -> None:
        """database_url URL-encodes special characters in user and password."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="db",
            user="user@org",
            password="p@ss/w%rd",
        )
        url = config.database_url
        assert "user%40org" in url
        assert "p%40ss%2Fw%25rd" in url

    def test_invalid_port_raises(self) -> None:
        """Port outside 1-65535 raises ValidationError."""
        with pytest.raises(ValidationError, match="port must be between 1 and 65535"):
            DatabaseConfig(port=99999, password="x")

    def test_zero_port_raises(self) -> None:
        """Port 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="port must be between 1 and 65535"):
            DatabaseConfig(port=0, password="x")

    def test_negative_pool_size_raises(self) -> None:
        """pool_size < 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="pool_size must be >= 1"):
            DatabaseConfig(pool_size=0, password="x")

    def test_env_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config reads from VECTORFORGE_DB_ prefixed env vars."""
        monkeypatch.setenv("VECTORFORGE_DB_HOST", "envhost")
        monkeypatch.setenv("VECTORFORGE_DB_PORT", "9999")
        monkeypatch.setenv("VECTORFORGE_DB_PASSWORD", "envpw")
        config = DatabaseConfig()
        assert config.host == "envhost"
        assert config.port == 9999
        assert config.password == "envpw"


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_defaults(self) -> None:
        """Storage config has sensible defaults."""
        config = StorageConfig()
        assert config.default_backend == "pg"
        assert config.threshold_mb == 10
        assert config.s3_bucket == ""

    def test_is_s3_configured(self) -> None:
        """is_s3_configured returns True only when bucket is set."""
        config = StorageConfig()
        assert config.is_s3_configured() is False

        config_with_s3 = StorageConfig(s3_bucket="my-bucket")
        assert config_with_s3.is_s3_configured() is True

    def test_invalid_threshold(self) -> None:
        """threshold_mb < 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="threshold_mb must be >= 1"):
            StorageConfig(threshold_mb=0)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_defaults(self) -> None:
        """Embedding config has correct defaults."""
        config = EmbeddingConfig()
        assert config.default_provider == "voyage"
        assert config.default_model == "voyage-3"
        assert config.dimensions == 1024
        assert config.batch_size == 100

    def test_invalid_dimensions(self) -> None:
        """dimensions < 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="dimensions must be >= 1"):
            EmbeddingConfig(dimensions=0)

    def test_invalid_batch_size(self) -> None:
        """batch_size < 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="batch_size must be >= 1"):
            EmbeddingConfig(batch_size=0)


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_defaults(self) -> None:
        """Chunking config has correct defaults."""
        config = ChunkingConfig()
        assert config.strategy == "recursive"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_overlap_exceeds_size_raises(self) -> None:
        """chunk_overlap >= chunk_size raises ValidationError."""
        with pytest.raises(ValidationError, match=r"chunk_overlap.*must be <.*chunk_size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=200)

    def test_overlap_equals_size_raises(self) -> None:
        """chunk_overlap == chunk_size raises ValidationError."""
        with pytest.raises(ValidationError, match=r"chunk_overlap.*must be <.*chunk_size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_valid_overlap(self) -> None:
        """Valid overlap/size combination passes."""
        config = ChunkingConfig(chunk_size=500, chunk_overlap=100)
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_defaults(self) -> None:
        """LLM config has correct defaults."""
        config = LLMConfig()
        assert config.default_provider == "openai"
        assert config.default_model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_invalid_temperature_high(self) -> None:
        """temperature > 2.0 raises ValidationError."""
        with pytest.raises(ValidationError, match=r"temperature must be between 0\.0 and 2\.0"):
            LLMConfig(temperature=3.0)

    def test_invalid_temperature_negative(self) -> None:
        """temperature < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError, match=r"temperature must be between 0\.0 and 2\.0"):
            LLMConfig(temperature=-0.1)


class TestMonitoringConfig:
    """Tests for MonitoringConfig."""

    def test_defaults(self) -> None:
        """Monitoring config has correct defaults."""
        config = MonitoringConfig()
        assert config.log_level == "INFO"
        assert config.log_format == "json"
        assert config.log_file is None
        assert config.metrics_enabled is True
        assert config.health_check_timeout_seconds == 5

    def test_invalid_log_level(self) -> None:
        """Invalid log_level raises ValidationError."""
        with pytest.raises(ValidationError):
            MonitoringConfig(log_level="INVALID")

    def test_invalid_log_format(self) -> None:
        """Invalid log_format raises ValidationError."""
        with pytest.raises(ValidationError):
            MonitoringConfig(log_format="yaml")

    def test_invalid_flush_interval(self) -> None:
        """metrics_flush_interval_seconds < 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="metrics_flush_interval_seconds must be >= 1"):
            MonitoringConfig(metrics_flush_interval_seconds=0)

    def test_invalid_health_timeout(self) -> None:
        """health_check_timeout_seconds < 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="health_check_timeout_seconds must be >= 1"):
            MonitoringConfig(health_check_timeout_seconds=0)


class TestVectorForgeConfig:
    """Tests for the root VectorForgeConfig."""

    def test_creates_with_defaults(self) -> None:
        """Root config aggregates all sub-configs with their defaults."""
        config = VectorForgeConfig()
        assert config.database.host == "localhost"
        assert config.embedding.default_provider == "voyage"
        assert config.chunking.chunk_size == 1000
        assert config.monitoring.log_level == "INFO"

    def test_nested_config_independence(self) -> None:
        """Sub-configs load independently from each other."""
        config = VectorForgeConfig(
            database=DatabaseConfig(host="custom-host", password="pw"),
        )
        assert config.database.host == "custom-host"
        assert config.storage.default_backend == "pg"
