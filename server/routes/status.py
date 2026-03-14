"""System status and health check endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from server.dependencies import ApiKey, EmbeddingReg, HealthCheck, LLMReg
from server.schemas import (
    ComponentHealthResponse,
    ProvidersResponse,
    SystemHealthResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/status", tags=["status"])


@router.get("", response_model=SystemHealthResponse)
async def system_status(
    health_checker: HealthCheck,
) -> SystemHealthResponse:
    """Deep health check — probes all registered components.

    This endpoint is public (no auth required) for use by load
    balancers and uptime monitors.
    """
    health = await health_checker.check_all()
    logger.info("Health check: %s (%d components)", health.status, len(health.components))
    return SystemHealthResponse(
        status=health.status,
        components=[
            ComponentHealthResponse(
                name=c.name,
                status=c.status,
                latency_ms=c.latency_ms,
                message=c.message,
            )
            for c in health.components
        ],
        checked_at=health.checked_at,
    )


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers(
    embedding_registry: EmbeddingReg,
    llm_registry: LLMReg,
    _key: ApiKey,
) -> ProvidersResponse:
    """List all registered embedding and LLM providers."""
    return ProvidersResponse(
        embedding_providers=embedding_registry.list_providers(),
        llm_providers=llm_registry.list_providers(),
    )
