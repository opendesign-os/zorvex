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
cp .env.example .env          # set OPENAI_API_KEY, or your model's key
uv sync
uv run zorvex dev
```

`zorvex dev` starts PostgreSQL in a container when it is not running yet, then serves the project's graphs on
<http://127.0.0.1:2026>, reloading on code changes. The API reference is at <http://127.0.0.1:2026/docs>.

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
| `zorvex up [SERVICES]` | Builds and starts PostgreSQL and the server with Docker or Podman Compose |
| `zorvex down` | Stops them (`-v` also removes their volumes) |
| `zorvex validate` | Checks `zorvex.json` |
| `zorvex db upgrade \| downgrade \| current \| history` | Database migrations |

`zorvex <command> --help` lists each command's options. Docker or Podman is picked automatically; set
`ZORVEX_CONTAINER_RUNTIME=docker` or `podman` to choose.

## Learn more

Configuration, deployment and API compatibility are covered in the
[Zorvex README](https://github.com/opendesign-os/zorvex#readme).

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
