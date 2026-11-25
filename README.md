# Zorvex

[![PyPI](https://img.shields.io/pypi/v/zorvex-cli)](https://pypi.org/project/zorvex-cli/)
[![CI](https://github.com/opendesign-os/zorvex/actions/workflows/ci.yml/badge.svg)](https://github.com/opendesign-os/zorvex/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)

Zorvex is a self-hosted server for [LangGraph](https://github.com/langchain-ai/langgraph) agents. It serves the
same HTTP API as the LangSmith Deployment Agent Server, so the LangGraph SDKs, and the clients built on them, work
against it unchanged: on your own PostgreSQL, with no license key.

- **The Agent Server API**: assistants and their versions, threads with their state and history, background,
  streamed and awaited runs, crons and the store, as `langgraph-api` 0.18 serves them.
- **Streaming**: the SDK's stream modes over SSE, resumable streams (`stream_resumable`, `Last-Event-ID`), the
  thread stream, and the Agent Protocol v2 event stream over SSE and WebSocket.
- **Human in the loop**: interrupts, resuming with a `Command`, state updates, runs from any checkpoint.
- **Interop**: the deployment's assistants as an MCP server (`/mcp`) and as A2A agents (`/a2a/{assistant_id}`).
- **Operations**: custom auth, custom routes, webhooks, thread and store TTLs, semantic store search (pgvector), run
  priority, OpenTelemetry tracing and Prometheus metrics.
- **Scaling out**: with Redis, runs go to workers, and a worker that dies has its runs picked up again.

## Quickstart

You need Python 3.12 or later, and Docker or Podman for the local PostgreSQL.

```bash
uv tool install zorvex-cli            # or: pip install zorvex-cli
zorvex new ./my-agent -t react-agent
cd my-agent
cp .env.example .env                  # set OPENAI_API_KEY, or your model's key
uv sync
uv run zorvex dev
```

`zorvex dev` starts PostgreSQL in a container when it is not running yet, then serves the project's graphs on
<http://127.0.0.1:2026>, reloading on code changes. The API reference is at <http://127.0.0.1:2026/docs>.

Talk to it with the LangGraph SDK (`pip install langgraph-sdk`):

```python
import asyncio

from langgraph_sdk import get_client


async def main() -> None:
    client = get_client(url="http://127.0.0.1:2026")
    thread = await client.threads.create()
    async for part in client.runs.stream(
        thread["thread_id"],
        "my_agent",  # the graph's name in zorvex.json
        input={"messages": [{"role": "user", "content": "What is 12.5 times 4, plus 7?"}]},
        stream_mode="updates",
    ):
        print(part.event, part.data)


asyncio.run(main())
```

## The `zorvex` command

| Command | What it does |
|---|---|
| `zorvex new [PATH]` | Creates a project from a template (`-t simple-chatbot` or `-t react-agent`) |
| `zorvex dev` | Runs the development server with reload, starting PostgreSQL if needed |
| `zorvex serve` | Runs the production server, as a container's command |
| `zorvex up [SERVICES]` | Builds and starts PostgreSQL and the server with Docker or Podman Compose |
| `zorvex down` | Stops them (`-v` also removes their volumes) |
| `zorvex validate` | Checks `zorvex.json` |
| `zorvex db upgrade \| downgrade \| current \| history` | Database migrations |

`zorvex <command> --help` lists each command's options. Docker or Podman is picked automatically; set
`ZORVEX_CONTAINER_RUNTIME=docker` or `podman` to choose.

## Configuration

A project is described by its `zorvex.json`. It takes the runtime keys of LangGraph's `langgraph.json`, not its
image-build keys such as `python_version` or `dockerfile_lines`:

```json
{
  "dependencies": ["./src"],
  "graphs": {
    "my_agent": "./src/my_agent/graph.py:graph"
  },
  "env": ".env",
  "auth": { "path": "./src/my_agent/auth.py:auth" },
  "http": { "cors": { "allow_origins": ["https://app.example.com"] } },
  "store": { "index": { "dims": 1536, "embed": "openai:text-embedding-3-small" } },
  "checkpointer": { "ttl": { "strategy": "delete", "default_ttl": 10080 } }
}
```

| Key | What it sets |
|---|---|
| `graphs` | Each graph's name and `file.py:variable`: a compiled graph, or a factory that builds one |
| `dependencies` | Directories put on `sys.path` before the graphs load |
| `env` | A `.env` file, or the variables themselves |
| `auth.path` | A `langgraph_sdk.Auth`: who the caller is, and what they may read and change |
| `http` | Custom routes (`app`), CORS, the headers runs see, `disable_*` switches, `mount_prefix` |
| `store` | The semantic search index (needs pgvector) and item TTLs |
| `checkpointer` | Thread TTLs (minutes), the serializer, durability |
| `webhooks` | Which URLs runs may call back, and the headers they get |
| `priority.path` | A callback that gives each thread its run priority |
| `ui` | A graph's generative UI components |

Every object rejects unknown keys, so a misspelled key stops the server from starting; `zorvex validate` checks the
file first.

The server reads its settings from the environment. The ones you are likely to set:

| Variable | Default | |
|---|---|---|
| `DATABASE_URI` | `postgresql://postgres:postgres@localhost:5432/zorvex` | PostgreSQL, as a libpq URL |
| `REDIS_URI` | unset | Set it to run several instances, with runs on workers |
| `ZORVEX_CONFIG` | `zorvex.json` | The project file |
| `RUN_MIGRATIONS_ON_STARTUP` | `true` | Apply database migrations as the server starts |
| `N_JOBS_PER_WORKER` | `10` | Runs a worker executes at once |
| `BG_JOB_TIMEOUT_SECS` | `3600` | The longest a run may take |
| `LOG_LEVEL`, `LOG_FORMAT` | `INFO`, `console` | `LOG_FORMAT=json` for log collectors |
| `OTEL_TRACES_EXPORTER` | `none` | `otlp` or `console`, with the standard `OTEL_*` variables |
| `ZORVEX_WEBHOOK_SECRET` | unset | Signs webhooks the Standard Webhooks way (`whsec_...`) |

Every setting, with its default and meaning, is in
[`settings.py`](libs/zorvex-api/src/zorvex_api/settings.py).

## Deploying

- **Docker**: a project from `zorvex new` has a `Dockerfile` and a `docker-compose.yml`; `zorvex up --wait` builds the
  image and starts it with PostgreSQL. In a container the server runs with `zorvex serve`.
- **Authentication**: without `auth.path`, every request is one anonymous user. Set it up before the server is
  reachable by others.
- **Several instances**: point them at one PostgreSQL and one Redis (`REDIS_URI`). Runs are queued to workers with
  leases, so the runs of an instance that dies are picked up by the others, and streams work from any instance.
- **Migrations**: the server applies them as it starts. With several replicas you can set
  `RUN_MIGRATIONS_ON_STARTUP=false` and run `zorvex db upgrade` once per release instead.
- **Probes**: `/ok` for liveness (`/ok?check_db=1` also checks PostgreSQL), `/info` for the version and features,
  `/metrics` for Prometheus.

## API compatibility

Zorvex follows the Agent Server API of LangSmith Deployment (`langgraph-api` 0.18): its documented endpoints, with
their request fields and answers, and every call the LangGraph SDK makes. The spec is
[`docs/openapi.json`](docs/openapi.json).

Not supported: JavaScript graphs, custom or MongoDB checkpointers and custom stores, encryption at rest, and
LangSmith API-key authentication. The LangSmith control plane (deployments, revisions, autoscaling) is not part of
Zorvex.

## Developing Zorvex

```bash
git clone https://github.com/opendesign-os/zorvex.git
cd zorvex
make dev-install    # the workspace and its git hooks
make run            # zorvex dev on this repository's zorvex.json
```

| | |
|---|---|
| `libs/zorvex-api/` | The server (FastAPI) and its database migrations |
| `libs/zorvex-cli/` | The `zorvex` command and the project templates |
| `docs/openapi.json` | The API spec, generated with `make openapi` |
| `scripts/` | The spec export and the release version check |

`make ci-check` runs CI's code checks: format, lint, types (ty) and security (bandit). After a change to the API, run
`make openapi`; CI fails when the committed spec differs from the code. `make build` builds both packages into
`dist/`.

### Releasing

1. Set the same `version` in `libs/zorvex-api/pyproject.toml` and `libs/zorvex-cli/pyproject.toml`; when the minor
   version changes, also zorvex-cli's `zorvex-api~=` requirement. Then `uv lock`.
2. Commit, tag the commit `vX.Y.Z` and push the tag.

The release workflow checks the tag against the versions, publishes zorvex-api and then zorvex-cli to PyPI with
trusted publishing (GitHub environments `zorvex-api` and `zorvex-cli`), and creates the GitHub release.

## License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
