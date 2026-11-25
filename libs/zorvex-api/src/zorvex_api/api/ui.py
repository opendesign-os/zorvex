"""Generative UI, as the Agent Server serves it: the components zorvex.json ``ui`` names, bundled by
@langchain/langgraph-ui into ``.zorvex/ui`` beside zorvex.json (``zorvex dev`` runs the bundler; the generated
Dockerfile bundles them in the image), for a client's ``LoadExternalComponent``.

``POST /ui/{graph_id}`` answers the tags that load a graph's bundle and render one of its components into the
client's shadow root; the bundle's files are public under ``GET /ui/{graph_id}/{file}``.
"""

import html
import json
import re
from pathlib import Path
from urllib.parse import quote

from anyio import Path as AsyncPath
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, Response

from zorvex_api.config import config_dir
from zorvex_api.core.auth_deps import require_auth

router = APIRouter(tags=["UI"], dependencies=[Depends(require_auth)])


def ui_dir() -> Path:
    """Where the bundler writes: ``public/<graph_id>/`` holds each graph's files, ``schemas.json`` lists them."""
    return config_dir() / ".zorvex" / "ui"


@router.post("/ui/{graph_id}", response_class=HTMLResponse, include_in_schema=False)
async def ui_component(graph_id: str, request: Request) -> Response:
    """The tags that load *graph_id*'s bundle and render the component the body names."""
    schemas_file = AsyncPath(ui_dir() / "schemas.json")
    schemas = json.loads(await schemas_file.read_text(encoding="utf-8")) if await schemas_file.exists() else {}
    if graph_id not in schemas:
        return Response(f"UI not found for graph '{graph_id}'", status_code=404)
    name = html.escape(json.dumps((await request.json()).get("name")), quote=True)
    renderer = "__LGUI_" + re.sub(r"[^a-zA-Z0-9]", "_", graph_id)
    root = request.scope.get("root_path", "")
    tags: list[str] = []
    for asset in schemas[graph_id].get("assets", []):
        file = Path(asset).name
        url = html.escape(f"{root}/ui/{quote(graph_id, safe='')}/{quote(file, safe='')}", quote=True)
        if file.endswith(".css"):
            tags.append(f'<link rel="stylesheet" href="{url}" />')
        elif file.endswith(".js"):
            tags.append(f'<script src="{url}" onload=\'{renderer}.render({name}, "{{{{shadowRootId}}}}")\'></script>')
    return HTMLResponse("\n".join(tags))
