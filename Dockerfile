# ============================================================================
# VectorForge Backend — Multi-stage Dockerfile
# ============================================================================
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for asyncpg and psycopg2 (build stage only needs these)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Build stage — install Python deps into a virtual-env
# ---------------------------------------------------------------------------
FROM base AS builder

COPY pyproject.toml ./
# Create a minimal structure so pip can resolve the package
COPY vectorforge/__init__.py vectorforge/__init__.py
COPY server/__init__.py server/__init__.py

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[server,litellm]"

# ---------------------------------------------------------------------------
# Runtime stage — slim image with only what is needed
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libpq is needed at runtime for asyncpg
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY vectorforge/ vectorforge/
COPY server/ server/
COPY alembic.ini ./

# Must listen on 0.0.0.0 inside the container
ENV VECTORFORGE_API_HOST=0.0.0.0
ENV VECTORFORGE_API_PORT=8000

EXPOSE 8000

# Run migrations then start the API server
CMD ["sh", "-c", "python -m alembic upgrade head && python -m server"]
