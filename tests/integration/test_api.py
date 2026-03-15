"""Integration tests for the VectorForge REST API server.

Uses FastAPI's TestClient (via httpx) with mocked state to verify
route behaviour, middleware, and dependency injection without a real
database or external services.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import APIConfig
from vectorforge.analytics.types import AnalyticsSummary, LatencyStats, QueryFrequency
from vectorforge.exceptions import DuplicateError, NotFoundError
from vectorforge.models.domain import Collection, Document, DocumentStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 1, 1, tzinfo=UTC)
_COLLECTION_ID = uuid.uuid4()
_DOCUMENT_ID = uuid.uuid4()

_SAMPLE_COLLECTION = Collection(
    id=_COLLECTION_ID,
    name="test-collection",
    description="A test collection",
    embedding_config={"default_provider": "voyage"},
    chunking_config=None,
    created_at=_NOW,
    updated_at=None,
)

_SAMPLE_DOCUMENT = Document(
    id=_DOCUMENT_ID,
    collection_id=_COLLECTION_ID,
    source_uri="test.txt",
    content_type="text/plain",
    raw_content="Hello world",
    status=DocumentStatus.INDEXED,
    content_size_bytes=11,
    metadata={},
    created_at=_NOW,
    updated_at=None,
)


def _mock_app() -> FastAPI:
    """Create a FastAPI app with mocked lifespan state."""
    config = APIConfig(auth_required=False)

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Mock db engine
        mock_engine = MagicMock()
        mock_engine.health_check = AsyncMock(return_value=True)

        mock_session = AsyncMock()
        mock_engine.get_session = lambda: _mock_session_cm(mock_session)
        mock_engine._session_factory = MagicMock()

        app.state.db_engine = mock_engine
        app.state.api_config = config
        app.state.vf_config = MagicMock()

        # Mock registries
        embedding_registry = MagicMock()
        embedding_registry.list_providers.return_value = ["voyage"]
        app.state.embedding_registry = embedding_registry

        llm_registry = MagicMock()
        llm_registry.list_providers.return_value = ["openai"]
        app.state.llm_registry = llm_registry

        # Mock health checker
        health_checker = MagicMock()
        app.state.health_checker = health_checker

        # Mock lifespan singletons used by dependency functions
        app.state.vector_store = MagicMock()
        app.state.loader_registry = MagicMock()
        app.state.chunker_registry = MagicMock()
        app.state.storage_router = MagicMock()

        yield

    from starlette.middleware.cors import CORSMiddleware

    from server.middleware import ErrorHandlerMiddleware, RequestLoggingMiddleware
    from server.routes.analytics import router as analytics_router
    from server.routes.collections import router as collections_router
    from server.routes.documents import router as documents_router
    from server.routes.query import router as query_router
    from server.routes.status import router as status_router

    app = FastAPI(lifespan=_lifespan)
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(collections_router, prefix="/api")
    app.include_router(documents_router, prefix="/api")
    app.include_router(query_router, prefix="/api")
    app.include_router(analytics_router, prefix="/api")
    app.include_router(status_router, prefix="/api")
    return app


@asynccontextmanager
async def _mock_session_cm(mock_session: AsyncMock) -> AsyncIterator[AsyncMock]:
    yield mock_session


@pytest.fixture()
def client() -> TestClient:  # type: ignore[misc]
    """Provide a TestClient with a mocked FastAPI app."""
    app = _mock_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Collection Endpoints
# ---------------------------------------------------------------------------


class TestCollectionRoutes:
    """Tests for /api/collections endpoints."""

    def test_list_collections_empty(self, client: TestClient) -> None:
        with patch(
            "server.routes.collections.CollectionRepository"
        ) as MockRepo:
            MockRepo.return_value.find_all = AsyncMock(return_value=[])
            resp = client.get("/api/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert data["collections"] == []

    def test_list_collections_with_data(self, client: TestClient) -> None:
        with patch(
            "server.routes.collections.CollectionRepository"
        ) as MockRepo:
            MockRepo.return_value.find_all = AsyncMock(
                return_value=[_SAMPLE_COLLECTION]
            )
            resp = client.get("/api/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["collections"]) == 1
        assert data["collections"][0]["name"] == "test-collection"

    def test_create_collection(self, client: TestClient) -> None:
        with patch(
            "server.routes.collections.CollectionRepository"
        ) as MockRepo:
            MockRepo.return_value.create = AsyncMock(
                return_value=_SAMPLE_COLLECTION
            )
            resp = client.post(
                "/api/collections",
                json={"name": "test-collection", "description": "A test"},
            )
        assert resp.status_code == 201
        assert resp.json()["name"] == "test-collection"

    def test_create_collection_duplicate(self, client: TestClient) -> None:
        with patch(
            "server.routes.collections.CollectionRepository"
        ) as MockRepo:
            MockRepo.return_value.create = AsyncMock(
                side_effect=DuplicateError("already exists")
            )
            resp = client.post(
                "/api/collections",
                json={"name": "existing"},
            )
        assert resp.status_code == 409
        assert resp.json()["error"] == "duplicate"

    def test_get_collection(self, client: TestClient) -> None:
        with patch(
            "server.routes.collections.CollectionRepository"
        ) as MockRepo:
            MockRepo.return_value.find_by_id = AsyncMock(
                return_value=_SAMPLE_COLLECTION
            )
            resp = client.get(f"/api/collections/{_COLLECTION_ID}")
        assert resp.status_code == 200
        assert resp.json()["id"] == str(_COLLECTION_ID)

    def test_get_collection_not_found(self, client: TestClient) -> None:
        random_id = uuid.uuid4()
        with patch(
            "server.routes.collections.CollectionRepository"
        ) as MockRepo:
            MockRepo.return_value.find_by_id = AsyncMock(return_value=None)
            resp = client.get(f"/api/collections/{random_id}")
        assert resp.status_code == 404

    def test_delete_collection(self, client: TestClient) -> None:
        with patch(
            "server.routes.collections.CollectionRepository"
        ) as MockRepo:
            MockRepo.return_value.delete = AsyncMock(return_value=None)
            resp = client.delete(f"/api/collections/{_COLLECTION_ID}")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Collection deleted"

    def test_delete_collection_not_found(self, client: TestClient) -> None:
        random_id = uuid.uuid4()
        with patch(
            "server.routes.collections.CollectionRepository"
        ) as MockRepo:
            MockRepo.return_value.delete = AsyncMock(
                side_effect=NotFoundError("not found")
            )
            resp = client.delete(f"/api/collections/{random_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Document Endpoints
# ---------------------------------------------------------------------------


class TestDocumentRoutes:
    """Tests for /api/documents and /api/collections/:id/documents."""

    def test_list_documents(self, client: TestClient) -> None:
        with (
            patch("server.routes.documents.CollectionRepository") as MockColRepo,
            patch("server.routes.documents.DocumentRepository") as MockDocRepo,
        ):
            MockColRepo.return_value.find_by_id = AsyncMock(
                return_value=_SAMPLE_COLLECTION
            )
            MockDocRepo.return_value.find_by_collection = AsyncMock(
                return_value=[_SAMPLE_DOCUMENT]
            )
            MockDocRepo.return_value.count_by_collection = AsyncMock(
                return_value=1
            )
            resp = client.get(f"/api/collections/{_COLLECTION_ID}/documents")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["documents"][0]["source_uri"] == "test.txt"

    def test_list_documents_collection_not_found(
        self, client: TestClient,
    ) -> None:
        random_id = uuid.uuid4()
        with patch(
            "server.routes.documents.CollectionRepository"
        ) as MockColRepo:
            MockColRepo.return_value.find_by_id = AsyncMock(return_value=None)
            resp = client.get(f"/api/collections/{random_id}/documents")
        assert resp.status_code == 404

    def test_get_document(self, client: TestClient) -> None:
        with patch(
            "server.routes.documents.DocumentRepository"
        ) as MockDocRepo:
            MockDocRepo.return_value.find_by_id = AsyncMock(
                return_value=_SAMPLE_DOCUMENT
            )
            resp = client.get(f"/api/documents/{_DOCUMENT_ID}")
        assert resp.status_code == 200
        assert resp.json()["id"] == str(_DOCUMENT_ID)

    def test_get_document_not_found(self, client: TestClient) -> None:
        random_id = uuid.uuid4()
        with patch(
            "server.routes.documents.DocumentRepository"
        ) as MockDocRepo:
            MockDocRepo.return_value.find_by_id = AsyncMock(return_value=None)
            resp = client.get(f"/api/documents/{random_id}")
        assert resp.status_code == 404

    def test_delete_document(self, client: TestClient) -> None:
        with patch(
            "server.routes.documents.DocumentRepository"
        ) as MockDocRepo:
            MockDocRepo.return_value.delete = AsyncMock(return_value=None)
            resp = client.delete(f"/api/documents/{_DOCUMENT_ID}")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Document deleted"


# ---------------------------------------------------------------------------
# Status Endpoints
# ---------------------------------------------------------------------------


class TestStatusRoutes:
    """Tests for /api/status endpoints."""

    def test_system_status(self, client: TestClient) -> None:
        mock_health = MagicMock()
        mock_health.status = "healthy"
        mock_health.components = []
        mock_health.checked_at = _NOW

        with patch(
            "server.dependencies.get_health_checker",
            return_value=MagicMock(check_all=AsyncMock(return_value=mock_health)),
        ):
            # We need to patch on the app state instead
            client.app.state.health_checker = MagicMock()  # type: ignore[attr-defined]
            client.app.state.health_checker.check_all = AsyncMock(  # type: ignore[attr-defined]
                return_value=mock_health,
            )
            resp = client.get("/api/status")

        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_list_providers(self, client: TestClient) -> None:
        resp = client.get("/api/status/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "embedding_providers" in data
        assert "llm_providers" in data


# ---------------------------------------------------------------------------
# Analytics Endpoints
# ---------------------------------------------------------------------------


class TestAnalyticsRoutes:
    """Tests for /api/analytics endpoints."""

    def test_get_summary(self, client: TestClient) -> None:
        summary = AnalyticsSummary(
            total_queries=42,
            unique_queries=10,
            latency=LatencyStats(
                avg_ms=100.0,
                min_ms=50.0,
                max_ms=200.0,
                p50_ms=90.0,
                p95_ms=180.0,
                sample_count=42,
            ),
            top_queries=[QueryFrequency(query_text="test", count=5)],
            volume=[],
        )
        with patch(
            "server.routes.analytics.QueryAnalyticsService"
        ) as MockSvc:
            MockSvc.return_value.get_summary = AsyncMock(return_value=summary)
            resp = client.get(f"/api/analytics/{_COLLECTION_ID}/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_queries"] == 42
        assert data["latency"]["avg_ms"] == 100.0

    def test_get_top_queries(self, client: TestClient) -> None:
        queries = [QueryFrequency(query_text="test", count=5)]
        with patch(
            "server.routes.analytics.QueryAnalyticsService"
        ) as MockSvc:
            MockSvc.return_value.get_top_queries = AsyncMock(
                return_value=queries
            )
            resp = client.get(
                f"/api/analytics/{_COLLECTION_ID}/top-queries?limit=5"
            )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["queries"]) == 1

    def test_get_latency_stats(self, client: TestClient) -> None:
        stats = LatencyStats(
            avg_ms=100.0,
            min_ms=50.0,
            max_ms=200.0,
            p50_ms=90.0,
            p95_ms=180.0,
            sample_count=42,
        )
        with patch(
            "server.routes.analytics.QueryAnalyticsService"
        ) as MockSvc:
            MockSvc.return_value.get_latency_stats = AsyncMock(
                return_value=stats
            )
            resp = client.get(f"/api/analytics/{_COLLECTION_ID}/latency")
        assert resp.status_code == 200
        data = resp.json()
        assert data["avg_ms"] == 100.0

    def test_get_latency_no_data(self, client: TestClient) -> None:
        with patch(
            "server.routes.analytics.QueryAnalyticsService"
        ) as MockSvc:
            MockSvc.return_value.get_latency_stats = AsyncMock(
                return_value=None
            )
            resp = client.get(f"/api/analytics/{_COLLECTION_ID}/latency")
        assert resp.status_code == 200
        assert resp.json() is None


# ---------------------------------------------------------------------------
# Auth Tests
# ---------------------------------------------------------------------------


class TestAuth:
    """Tests for API key authentication."""

    @contextmanager
    def _make_auth_app(self) -> Generator[TestClient, None, None]:
        """Create a test client that requires API key auth."""
        config = APIConfig(auth_required=True, api_key="test-secret")

        @asynccontextmanager
        async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
            app.state.db_engine = MagicMock()
            app.state.db_engine._session_factory = MagicMock()
            app.state.db_engine.get_session = lambda: _mock_session_cm(AsyncMock())
            app.state.api_config = config
            app.state.vf_config = MagicMock()
            app.state.embedding_registry = MagicMock()
            app.state.llm_registry = MagicMock()
            app.state.health_checker = MagicMock()
            app.state.vector_store = MagicMock()
            app.state.loader_registry = MagicMock()
            app.state.chunker_registry = MagicMock()
            app.state.storage_router = MagicMock()
            yield

        from server.middleware import ErrorHandlerMiddleware
        from server.routes.collections import router as collections_router

        app = FastAPI(lifespan=_lifespan)
        app.add_middleware(ErrorHandlerMiddleware)
        app.include_router(collections_router, prefix="/api")
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    def test_missing_api_key(self) -> None:
        with self._make_auth_app() as client:
            with patch(
                "server.routes.collections.CollectionRepository"
            ) as MockRepo:
                MockRepo.return_value.find_all = AsyncMock(return_value=[])
                resp = client.get("/api/collections")
            assert resp.status_code == 401

    def test_wrong_api_key(self) -> None:
        with self._make_auth_app() as client:
            with patch(
                "server.routes.collections.CollectionRepository"
            ) as MockRepo:
                MockRepo.return_value.find_all = AsyncMock(return_value=[])
                resp = client.get(
                    "/api/collections",
                    headers={"X-Api-Key": "wrong-key"},
                )
            assert resp.status_code == 401

    def test_correct_api_key(self) -> None:
        with self._make_auth_app() as client:
            with patch(
                "server.routes.collections.CollectionRepository"
            ) as MockRepo:
                MockRepo.return_value.find_all = AsyncMock(return_value=[])
                resp = client.get(
                    "/api/collections",
                    headers={"X-Api-Key": "test-secret"},
                )
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Middleware Tests
# ---------------------------------------------------------------------------


class TestIngestRoutes:
    """Tests for document ingestion endpoints."""

    def _override_ingestion(
        self, client: TestClient, mock_svc: AsyncMock,
    ) -> None:
        from server.dependencies import get_ingestion_service

        client.app.dependency_overrides[get_ingestion_service] = lambda: mock_svc  # type: ignore[union-attr]

    def _clear_overrides(self, client: TestClient) -> None:
        client.app.dependency_overrides.clear()  # type: ignore[union-attr]

    def test_ingest_document(self, client: TestClient) -> None:
        mock_svc = AsyncMock()
        mock_svc.ingest = AsyncMock(return_value=_SAMPLE_DOCUMENT)
        self._override_ingestion(client, mock_svc)

        with patch(
            "server.routes.documents.CollectionRepository",
        ) as MockColRepo:
            MockColRepo.return_value.find_by_id = AsyncMock(
                return_value=_SAMPLE_COLLECTION,
            )
            resp = client.post(
                f"/api/collections/{_COLLECTION_ID}/documents",
                json={"source": "test.txt"},
            )

        self._clear_overrides(client)

        assert resp.status_code == 201
        assert resp.json()["source_uri"] == "test.txt"

    def test_ingest_document_collection_not_found(
        self, client: TestClient,
    ) -> None:
        random_id = uuid.uuid4()
        with patch(
            "server.routes.documents.CollectionRepository",
        ) as MockColRepo:
            MockColRepo.return_value.find_by_id = AsyncMock(return_value=None)
            resp = client.post(
                f"/api/collections/{random_id}/documents",
                json={"source": "test.txt"},
            )
        assert resp.status_code == 404

    def test_batch_ingest(self, client: TestClient) -> None:
        mock_svc = AsyncMock()
        mock_svc.ingest = AsyncMock(return_value=_SAMPLE_DOCUMENT)
        self._override_ingestion(client, mock_svc)

        with patch(
            "server.routes.documents.CollectionRepository",
        ) as MockColRepo:
            MockColRepo.return_value.find_by_id = AsyncMock(
                return_value=_SAMPLE_COLLECTION,
            )
            resp = client.post(
                f"/api/collections/{_COLLECTION_ID}/documents/batch",
                json=[{"source": "a.txt"}, {"source": "b.txt"}],
            )

        self._clear_overrides(client)

        assert resp.status_code == 200
        data = resp.json()
        assert data["succeeded"] == 2
        assert data["failed"] == 0
        assert len(data["results"]) == 2

    def test_batch_ingest_partial_failure(self, client: TestClient) -> None:
        mock_svc = AsyncMock()
        mock_svc.ingest = AsyncMock(
            side_effect=[_SAMPLE_DOCUMENT, NotFoundError("file not found")],
        )
        self._override_ingestion(client, mock_svc)

        with patch(
            "server.routes.documents.CollectionRepository",
        ) as MockColRepo:
            MockColRepo.return_value.find_by_id = AsyncMock(
                return_value=_SAMPLE_COLLECTION,
            )
            resp = client.post(
                f"/api/collections/{_COLLECTION_ID}/documents/batch",
                json=[{"source": "good.txt"}, {"source": "bad.txt"}],
            )

        self._clear_overrides(client)

        assert resp.status_code == 200
        data = resp.json()
        assert data["succeeded"] == 1
        assert data["failed"] == 1

    def test_upload_document(self, client: TestClient) -> None:
        mock_svc = AsyncMock()
        mock_svc.ingest = AsyncMock(return_value=_SAMPLE_DOCUMENT)
        self._override_ingestion(client, mock_svc)

        with patch(
            "server.routes.documents.CollectionRepository",
        ) as MockColRepo:
            MockColRepo.return_value.find_by_id = AsyncMock(
                return_value=_SAMPLE_COLLECTION,
            )
            resp = client.post(
                f"/api/collections/{_COLLECTION_ID}/documents/upload",
                files={"file": ("test.txt", b"Hello world", "text/plain")},
                data={"metadata": '{"source": "upload-test"}'},
            )

        self._clear_overrides(client)

        assert resp.status_code == 201
        assert resp.json()["source_uri"] == "test.txt"
        # Verify the temp file was cleaned up (ingest was called with a path)
        call_kwargs = mock_svc.ingest.call_args
        assert call_kwargs is not None

    def test_upload_document_collection_not_found(
        self, client: TestClient,
    ) -> None:
        random_id = uuid.uuid4()
        with patch(
            "server.routes.documents.CollectionRepository",
        ) as MockColRepo:
            MockColRepo.return_value.find_by_id = AsyncMock(return_value=None)
            resp = client.post(
                f"/api/collections/{random_id}/documents/upload",
                files={"file": ("test.txt", b"Hello world", "text/plain")},
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Query Endpoints
# ---------------------------------------------------------------------------


class TestQueryRoutes:
    """Tests for /api/query endpoints."""

    def test_sync_query(self, client: TestClient) -> None:
        from vectorforge.pipeline.context import SourceCitation
        from vectorforge.pipeline.types import QueryResult

        result = QueryResult(
            query="What is RAG?",
            answer="RAG is Retrieval-Augmented Generation.",
            sources=[
                SourceCitation(
                    document_source="guide.md",
                    chunk_index=0,
                    score=0.95,
                    snippet="RAG combines retrieval...",
                ),
            ],
            retrieval_latency_ms=50.0,
            generation_latency_ms=200.0,
            total_latency_ms=250.0,
        )

        mock_svc = AsyncMock()
        mock_svc.query = AsyncMock(return_value=result)

        from server.dependencies import get_query_service

        client.app.dependency_overrides[get_query_service] = lambda: mock_svc  # type: ignore[union-attr]

        resp = client.post(
            "/api/query",
            json={
                "query": "What is RAG?",
                "collection_id": str(_COLLECTION_ID),
            },
        )

        client.app.dependency_overrides.clear()  # type: ignore[union-attr]

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "RAG is Retrieval-Augmented Generation."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_source"] == "guide.md"
        assert data["total_latency_ms"] == 250.0

    def test_stream_query(self, client: TestClient) -> None:
        async def _mock_stream(
            query: str, collection_id: object, config: object,
        ) -> AsyncIterator[str]:
            yield "Hello"
            yield " world"

        mock_svc = AsyncMock()
        mock_svc.query_stream = _mock_stream

        from server.dependencies import get_query_service

        client.app.dependency_overrides[get_query_service] = lambda: mock_svc  # type: ignore[union-attr]

        resp = client.post(
            "/api/query/stream",
            json={
                "query": "Hello?",
                "collection_id": str(_COLLECTION_ID),
            },
        )

        client.app.dependency_overrides.clear()  # type: ignore[union-attr]

        assert resp.status_code == 200
        # SSE responses contain "data:" lines
        body = resp.text
        assert "token" in body
        assert "done" in body


# ---------------------------------------------------------------------------
# Middleware Tests
# ---------------------------------------------------------------------------


class TestMiddleware:
    """Tests for error handling and request logging middleware."""

    def test_request_id_header(self, client: TestClient) -> None:
        with patch(
            "server.routes.collections.CollectionRepository"
        ) as MockRepo:
            MockRepo.return_value.find_all = AsyncMock(return_value=[])
            resp = client.get("/api/collections")
        assert "x-request-id" in resp.headers

    def test_error_handler_catches_vectorforge_errors(
        self, client: TestClient,
    ) -> None:
        with patch(
            "server.routes.collections.CollectionRepository"
        ) as MockRepo:
            MockRepo.return_value.find_by_id = AsyncMock(
                side_effect=NotFoundError("not found")
            )
            resp = client.get(f"/api/collections/{uuid.uuid4()}")
        assert resp.status_code == 404
        assert resp.json()["error"] == "not_found"
