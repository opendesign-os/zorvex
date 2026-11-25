ARG PY_VERSION=3.12
FROM python:${PY_VERSION}-slim-bookworm AS base
ARG PY_VERSION

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create non-root user for runtime
RUN addgroup --system app && adduser --system --ingroup app app

# -----------------------------
# Builder stage
# -----------------------------
FROM base AS builder

# Retrieve the uv binary directly from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /bin/uv

# Install system build dependencies required for compiling Python extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy zorvex-api package files and workspace lockfile
COPY libs/zorvex-api/pyproject.toml ./
COPY uv.lock ./

# Install the locked runtime dependencies.
RUN uv export --frozen --no-dev --no-emit-project --format=requirements-txt > requirements.txt && \
    uv pip install --system --compile-bytecode -r requirements.txt && \
    rm requirements.txt

# Copy the actual project source code and forced includes (alembic).
COPY libs/zorvex-api/src/ ./src/
COPY libs/zorvex-api/alembic.ini ./alembic.ini
COPY libs/zorvex-api/alembic/ ./alembic/

# Install the project package itself.
RUN uv pip install --system --compile-bytecode --no-deps .

# -----------------------------
# Final, minimal runtime image
# -----------------------------
FROM base AS final
ARG PY_VERSION

# Install only minimal runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage.
COPY --from=builder /usr/local/lib/python${PY_VERSION}/site-packages/ /usr/local/lib/python${PY_VERSION}/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy runtime assets required by the app and alembic
COPY libs/zorvex-api/alembic.ini ./alembic.ini
COPY libs/zorvex-api/alembic/ ./alembic/
COPY zorvex.json ./zorvex.json

EXPOSE 2026

# Run as non-root
USER app

# The image has zorvex-api only (no CLI), so serve it with uvicorn.
CMD ["uvicorn", "zorvex_api.main:app", "--host", "0.0.0.0", "--port", "2026"]
