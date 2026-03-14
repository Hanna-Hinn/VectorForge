"""Unit tests for VectorForge storage backends and router."""

from __future__ import annotations

import pytest

from vectorforge.config.settings import StorageConfig
from vectorforge.exceptions import StorageError
from vectorforge.storage.postgres import PostgresStorageBackend
from vectorforge.storage.router import StorageRouter

# ---------------------------------------------------------------------------
# PostgresStorageBackend tests
# ---------------------------------------------------------------------------


class TestPostgresStorageBackend:
    """Tests for the PostgreSQL storage backend (in-memory cache)."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self) -> None:
        backend = PostgresStorageBackend()
        key = await backend.store("doc-1", b"hello world")
        assert key == "doc-1"
        content = await backend.retrieve("doc-1")
        assert content == b"hello world"

    @pytest.mark.asyncio
    async def test_retrieve_missing_key_raises(self) -> None:
        backend = PostgresStorageBackend()
        with pytest.raises(StorageError, match="not found"):
            await backend.retrieve("nonexistent")

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        backend = PostgresStorageBackend()
        await backend.store("doc-2", b"data")
        await backend.delete("doc-2")
        with pytest.raises(StorageError):
            await backend.retrieve("doc-2")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key_is_noop(self) -> None:
        backend = PostgresStorageBackend()
        await backend.delete("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# StorageRouter tests
# ---------------------------------------------------------------------------


class TestStorageRouter:
    """Tests for the storage router without S3 configured."""

    def _pg_only_config(self) -> StorageConfig:
        return StorageConfig(
            default_backend="pg",
            threshold_mb=5,
            s3_bucket="",
        )

    def test_pg_only_select_backend(self) -> None:
        router = StorageRouter(self._pg_only_config())
        _backend, name = router.select_backend(100)
        assert name == "pg"

    def test_pg_only_large_still_goes_to_pg(self) -> None:
        """Without S3, even large content goes to PostgreSQL."""
        router = StorageRouter(self._pg_only_config())
        _backend, name = router.select_backend(100 * 1024 * 1024)
        assert name == "pg"

    @pytest.mark.asyncio
    async def test_store_returns_pg(self) -> None:
        router = StorageRouter(self._pg_only_config())
        key, backend_name = await router.store("k1", b"data")
        assert backend_name == "pg"
        assert key == "k1"

    @pytest.mark.asyncio
    async def test_retrieve_pg(self) -> None:
        router = StorageRouter(self._pg_only_config())
        await router.store("k2", b"content")
        result = await router.retrieve("k2", "pg")
        assert result == b"content"

    @pytest.mark.asyncio
    async def test_retrieve_unknown_backend_raises(self) -> None:
        router = StorageRouter(self._pg_only_config())
        with pytest.raises(StorageError, match="Unknown storage backend"):
            await router.retrieve("k", "gcs")

    @pytest.mark.asyncio
    async def test_retrieve_s3_when_not_configured_raises(self) -> None:
        router = StorageRouter(self._pg_only_config())
        with pytest.raises(StorageError, match="S3 backend requested but not configured"):
            await router.retrieve("k", "s3")

    @pytest.mark.asyncio
    async def test_delete_pg(self) -> None:
        router = StorageRouter(self._pg_only_config())
        await router.store("k3", b"data")
        await router.delete("k3", "pg")
        with pytest.raises(StorageError):
            await router.retrieve("k3", "pg")

    def test_s3_backend_is_none(self) -> None:
        router = StorageRouter(self._pg_only_config())
        assert router.s3_backend is None
        assert router.pg_backend is not None


class TestStorageRouterWithS3Config:
    """Tests for the storage router with S3 configured (routing logic only)."""

    def _s3_config(self) -> StorageConfig:
        return StorageConfig(
            default_backend="pg",
            threshold_mb=1,
            s3_bucket="test-bucket",
            s3_region="us-east-1",
            s3_access_key="AKID",
            s3_secret_key="SECRET",
        )

    def test_small_content_routes_to_pg(self) -> None:
        router = StorageRouter(self._s3_config())
        _, name = router.select_backend(100)
        assert name == "pg"

    def test_large_content_routes_to_s3(self) -> None:
        router = StorageRouter(self._s3_config())
        _, name = router.select_backend(2 * 1024 * 1024)  # 2MB > 1MB threshold
        assert name == "s3"

    def test_s3_backend_is_configured(self) -> None:
        router = StorageRouter(self._s3_config())
        assert router.s3_backend is not None
