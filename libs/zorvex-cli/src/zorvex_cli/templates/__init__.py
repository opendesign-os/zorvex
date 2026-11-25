"""The files of a new project, and the Docker files ``zorvex dev`` and ``zorvex up`` add to a project without them."""

from collections.abc import Mapping
from importlib import resources
from string import Template
from typing import NamedTuple


class ProjectTemplate(NamedTuple):
    name: str
    description: str
    files: dict[str, str]
    """Its own files, besides SHARED_FILES: the source under templates/ -> the destination in the project."""


# Every project's files: the source under templates/ -> the destination, $-substituted like the contents.
SHARED_FILES = {
    "shared/README.md.template": "README.md",
    "shared/pyproject.toml.template": "pyproject.toml",
    "shared/env.example.template": ".env.example",
    "shared/gitignore": ".gitignore",
    "shared/__init__.py.template": "src/$slug/__init__.py",
    "shared/context.py.template": "src/$slug/context.py",
    "shared/prompts.py.template": "src/$slug/prompts.py",
    "shared/state.py.template": "src/$slug/state.py",
    "shared/utils.py.template": "src/$slug/utils.py",
}

TEMPLATES = {
    "simple-chatbot": ProjectTemplate(
        name="New Zorvex Project",
        description="A simple chatbot with message memory.",
        files={"simple-chatbot/graph.py.template": "src/$slug/graph.py"},
    ),
    "react-agent": ProjectTemplate(
        name="ReAct Agent",
        description="An agent with tools that reasons and acts step by step.",
        files={
            "react-agent/graph.py.template": "src/$slug/graph.py",
            "react-agent/tools.py.template": "src/$slug/tools.py",
        },
    ),
}


def render(name: str, variables: Mapping[str, str] | None = None) -> str:
    """The bundled file *name*, a path under templates/, with its ``$variables`` substituted."""
    text = resources.files(__name__).joinpath(name).read_text(encoding="utf-8")
    return Template(text).safe_substitute(variables or {})


# The generative UI bundler (zorvex_api/api/ui.py), pinned: `zorvex dev` and the
# Dockerfile build with one release.
UI_BUNDLER = "@langchain/langgraph-ui@1.8.1"


def docker_files(*, ui: bool = False) -> dict[str, str]:
    """docker-compose.yml, Dockerfile and .dockerignore, by name.

    Args:
        ui: Bundle zorvex.json's ui components into the image, as the server serves them.
    """
    dockerfile = _DOCKERFILE
    if ui:
        dockerfile = _UI_STAGE + _DOCKERFILE.replace(
            "COPY zorvex.json .\n", "COPY zorvex.json .\nCOPY --from=ui /app/.zorvex/ui ./.zorvex/ui\n"
        )
    return {"docker-compose.yml": _COMPOSE, "Dockerfile": dockerfile, ".dockerignore": _DOCKERIGNORE}


# The server's defaults (DATABASE_URI) match the postgres service, so a project runs without settings.
_COMPOSE = """\
# PostgreSQL, Redis and the server: `zorvex up` starts them all, `zorvex dev` only postgres.
# In .env, POSTGRES_PORT moves PostgreSQL's host port (DATABASE_URI must follow it for
# `zorvex dev`), and PORT the server's.

services:
  postgres:
    image: pgvector/pgvector:pg18
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: zorvex
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres", "-d", "zorvex"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build: .
    restart: unless-stopped
    ports:
      - "${PORT:-2026}:${PORT:-2026}"
    env_file:
      - .env
    environment:
      - DATABASE_URI=postgresql://postgres:postgres@postgres:5432/zorvex
      - REDIS_URI=redis://redis:6379/0
      - PORT=${PORT:-2026}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-fs", "http://localhost:${PORT:-2026}/ok"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 30s

volumes:
  postgres_data:
  redis_data:
"""

# Bundles zorvex.json's ui components when the image builds, read from the zorvex.json it copies.
_UI_STAGE = f"""\
# -----------------------------
# UI stage: zorvex.json's ui components, bundled with {UI_BUNDLER}
# -----------------------------
FROM node:22-slim AS ui
WORKDIR /app
COPY . .
RUN LANGGRAPH_UI="$(node -p 'JSON.stringify(require("./zorvex.json").ui || {{}})')" \\
    LANGGRAPH_UI_CONFIG="$(node -p 'JSON.stringify(require("./zorvex.json").ui_config || {{}})')" \\
    npx -y {UI_BUNDLER} build -o .zorvex/ui

"""

# The project itself is not installed: the server puts zorvex.json's dependencies (./src) on sys.path.
_DOCKERFILE = """\
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create non-root user for runtime
RUN addgroup --system app && adduser --system --ingroup app app

# -----------------------------
# Builder stage: the dependencies, as uv.lock pins them when there is one
# -----------------------------
FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY uv.loc[k] ./
RUN uv sync --no-dev --no-install-project --compile-bytecode

# -----------------------------
# Runtime stage: zorvex.json and the code it points to (./src)
# -----------------------------
FROM base AS final

RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates \\
    curl \\
    libpq5 \\
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY zorvex.json .
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 2026

USER app

CMD ["zorvex", "serve"]
"""

_DOCKERIGNORE = """\
.env
.venv
.git
.zorvex
**/__pycache__
node_modules
"""
