# zorvex-cli

The command line of [Zorvex](https://github.com/opendesign-os/zorvex), a self-hosted server for
[LangGraph](https://github.com/langchain-ai/langgraph) agents that serves the LangSmith Deployment Agent Server API.
The LangGraph SDKs, and the clients built on them, work against it unchanged.

## Install

```bash
uv tool install zorvex-cli    # or: pip install zorvex-cli
```

The server, [zorvex-api](https://pypi.org/project/zorvex-api/), comes with it. You need Python 3.12 or later, and
Docker or Podman for the local PostgreSQL.

## Quickstart

```bash
zorvex new ./my-agent -t react-agent
cd my-agent
uv sync
uv run zorvex dev
```

`zorvex dev` starts PostgreSQL in a container when it is not running yet, then serves the project's graphs on
<http://127.0.0.1:2026>, reloading on code changes. The API reference is at <http://127.0.0.1:2026/docs>. The first
run creates `.env` from `.env.example`: set `OPENAI_API_KEY` there, or your model provider's key, and start it again.

```python
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2026")
thread = await client.threads.create()
async for part in client.runs.stream(
    thread["thread_id"],
    "my_agent",  # the graph's name in zorvex.json
    input={"messages": [{"role": "user", "content": "What is 12.5 times 4, plus 7?"}]},
    stream_mode="updates",
):
    print(part.event, part.data)
```

## Commands

| Command | What it does |
|---|---|
| `zorvex new [PATH]` | Creates a project from a template (`-t simple-chatbot` or `-t react-agent`) |
| `zorvex dev` | Runs the development server with reload, starting PostgreSQL if needed |
| `zorvex serve` | Runs the production server, as a container's command |
| `zorvex up [SERVICES]` | Builds and starts PostgreSQL, Redis and the server with Docker or Podman Compose |
| `zorvex down` | Stops them (`-v` also removes their volumes) |
| `zorvex validate` | Checks `zorvex.json` |
| `zorvex db upgrade \| downgrade \| current \| history` | Database migrations |

`zorvex <command> --help` lists each command's options.

**The project.** Every command but `new` works on a project: the `zorvex.json` in the current directory, or the one
`-c/--config` or `ZORVEX_CONFIG` names. Its environment is the file `-e/--env-file` names, else zorvex.json's `env` (a
file relative to it, or the variables themselves), else the `.env` beside zorvex.json, which `dev` and `up` create
from `.env.example` when missing. Variables already set in the shell keep their values. So `zorvex db` reaches the
database `zorvex dev` uses. `up` and `down` take no `-e`.

**`dev` and `serve`.**

- `--host`, `--port`: `127.0.0.1` and `2026` for `dev`; `0.0.0.0` and `PORT`, else `2026`, for `serve`.
- `--no-reload` (`dev`): no restart on code changes.
- `--debug [HOST:]PORT` (`dev`): runs the server under debugpy (`pip install 'zorvex-cli[debug]'`) and waits for a
  debugger to attach. The host is 127.0.0.1 unless given; any other exposes the unauthenticated debugger.
- `--no-db-check`, `-f/--file` (`dev`): leave PostgreSQL alone, or start it from another Compose file.
- Any other argument goes to uvicorn, after `--` when it looks like one of zorvex's own:
  `zorvex dev --ssl-certfile cert.pem --ssl-keyfile key.pem`.

**Containers.** `dev`, `up` and `down` run Compose with Docker or Podman: `ZORVEX_CONTAINER_RUNTIME=docker` or
`podman` chooses; unset, it is Docker when its daemon runs, else Podman, else Docker. A runtime that is not running is
started with the platform's usual command (`docker desktop start`, `open -a Docker`, `podman machine start`, ...),
which Zorvex shows when it fails. The Compose file is the `docker-compose.yml` beside zorvex.json, or the one `-f`
names; `dev` and `up` create it, the `Dockerfile` and a `.dockerignore` where missing. Its PostgreSQL is the one the
server connects to by default, so a new project needs no settings; `POSTGRES_PORT` moves its host port, with
`DATABASE_URI` to match.

## Learn more

Configuration, deployment and API compatibility are covered in the
[Zorvex README](https://github.com/opendesign-os/zorvex#readme).

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
