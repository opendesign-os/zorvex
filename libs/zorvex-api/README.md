# zorvex-api

The server of [Zorvex](https://github.com/opendesign-os/zorvex): a FastAPI app that runs
[LangGraph](https://github.com/langchain-ai/langgraph) graphs behind the LangSmith Deployment Agent Server API, on
PostgreSQL. The LangGraph SDKs, and the clients built on them, work against it unchanged.

To create and run projects, install [zorvex-cli](https://pypi.org/project/zorvex-cli/) instead: it brings this
package along with the `zorvex` command.

## Running it directly

```bash
pip install zorvex-api
export DATABASE_URI=postgresql://zorvex:secret@localhost:5432/zorvex   # the zorvex.json in the working directory serves
uvicorn zorvex_api.main:app --host 0.0.0.0 --port 2026
```

The server applies its database migrations as it starts. On Windows, add
`--loop zorvex_api.utils.event_loop:selector_loop_factory`: psycopg's async mode needs a selector event loop.

## What it serves

- `/assistants`, `/threads`, `/runs`, `/runs/crons` and `/store`: the Agent Server API, as `langgraph-api` 0.18 serves
  it, with SSE streaming and resumable streams.
- `/threads/{thread_id}/stream/events` and `/threads/{thread_id}/commands`: the Agent Protocol v2 event stream, over
  SSE and WebSocket.
- `/mcp`: the deployment's assistants as an MCP server.
- `/a2a/{assistant_id}`: each assistant as an A2A agent, with its agent card.
- `/ok`, `/info` and `/metrics`: liveness, version and features, Prometheus metrics; `/docs` for the API reference.

## Learn more

Configuration (`zorvex.json` and the environment), deployment and API compatibility are covered in the
[Zorvex README](https://github.com/opendesign-os/zorvex#readme).

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
