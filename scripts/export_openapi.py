"""Export the OpenAPI spec to docs/openapi.json

Generates the spec from the FastAPI app, normalizes the title/version to
match the Zorvex project defaults, and removes any custom-route endpoints
that are project-specific rather than part of the standard API.
"""

import json
import logging
from pathlib import Path

# Importing the app logs its startup; keep the export quiet.
logging.disable(logging.CRITICAL)

import structlog  # noqa: E402

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
)

from zorvex_api import __version__  # noqa: E402
from zorvex_api.main import OPENAPI_TAGS, app  # noqa: E402


def main() -> None:
    """Export the OpenAPI spec to docs/openapi.json."""
    schema = app.openapi()

    # Normalize title and description to standard Zorvex values
    schema["info"]["title"] = "Zorvex"
    schema["info"]["version"] = __version__
    schema["info"]["description"] = "Production-ready Agent Protocol server"

    # Remove custom-route endpoints and untagged paths (root, custom routes)
    core_tags: set[str] = {t["name"] for t in OPENAPI_TAGS}
    paths_to_remove: list[str] = []
    for path, methods in schema.get("paths", {}).items():
        for _method, info in methods.items():
            tags = set(info.get("tags", []))
            if not tags or not tags & core_tags:
                paths_to_remove.append(path)
                break

    for path in paths_to_remove:
        del schema["paths"][path]

    # Keep only core tags
    schema["tags"] = [t for t in schema.get("tags", []) if t["name"] in core_tags]

    output_path = Path(__file__).parent.parent / "docs" / "openapi.json"
    output_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(schema['paths'])} endpoints to {output_path}")


if __name__ == "__main__":
    main()
