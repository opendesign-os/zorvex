"""Check that a release tag names the version both packages carry.

usage: check_release_version.py vX.Y.Z

zorvex-api and zorvex-cli are released together at the tag's version (the tag
without its leading ``v``): each pyproject.toml must carry it, and zorvex-cli's
requirement on zorvex-api must admit it, so the CLI installed from PyPI runs on
the API released with it. Exits 1 and prints what disagrees otherwise.
"""

import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent.parent


def _project(package: str) -> dict:
    return tomllib.loads((ROOT / "libs" / package / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def problems(tag: str) -> list[str]:
    """What keeps *tag* from being released; none when it can be."""
    if not tag.startswith("v"):
        return [f"A release tag is the version with a leading v, as in v1.2.3, not {tag!r}"]
    try:
        version = Version(tag[1:])
    except InvalidVersion:
        return [f"{tag[1:]!r} is not a version"]

    api, cli = _project("zorvex-api"), _project("zorvex-cli")
    found = [
        f"{project['name']} is at {project['version']} in its pyproject.toml, the tag is {version}"
        for project in (api, cli)
        if Version(project["version"]) != version
    ]
    requirements = [Requirement(spec) for spec in cli.get("dependencies", [])]
    api_requirement = next((req for req in requirements if req.name == api["name"]), None)
    if api_requirement is None:
        found.append(f"{cli['name']} does not depend on {api['name']}")
    elif not api_requirement.specifier.contains(version, prereleases=True):
        found.append(f"{cli['name']} requires {api_requirement}, which {api['name']} {version} does not satisfy")
    return found


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    found = problems(sys.argv[1])
    for problem in found:
        # A GitHub Actions annotation; plain text anywhere else.
        print(f"::error::{problem}")
    if not found:
        print(f"zorvex-api and zorvex-cli are both at {sys.argv[1][1:]}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
