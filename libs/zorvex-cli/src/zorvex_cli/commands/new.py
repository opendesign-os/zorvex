"""``zorvex new``: a project from a template."""

import json
import logging
import os
import re
import sys
from pathlib import Path
from string import Template

import click
from rich.console import Console

from zorvex_cli import __version__
from zorvex_cli.templates import SHARED_FILES, TEMPLATES, docker_files, render

console = Console()
logger = logging.getLogger(__name__)


def slugify(name: str) -> str:
    """A Python identifier for *name*: "My Project" -> "my_project", "MyApp 2.0" -> "myapp_20"."""
    slug = name.lower().replace(" ", "_").replace("-", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if slug and slug[0].isdigit():
        slug = "project_" + slug
    return slug or "zorvex_project"


def _is_interactive() -> bool:
    """Whether stdin is a terminal, so questions can be asked (not in CI or with piped input)."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        logger.debug("Failed to determine TTY status for stdin", exc_info=True)
        return False


def _prompt_template() -> str:
    """Ask which template to start from."""
    click.echo()
    click.echo(click.style("Choose a template:", bold=True))
    for number, template in enumerate(TEMPLATES.values(), 1):
        click.echo(f"  {number}. {template.name} - {template.description}")
    click.echo()
    number = click.prompt("Enter your choice", default=1, type=click.IntRange(1, len(TEMPLATES)))
    return list(TEMPLATES)[number - 1]


def _write_file(path: Path, content: str, force: bool) -> bool:
    """Write *content* to *path* unless it exists and *force* is off; whether it was written."""
    if path.exists() and not force:
        console.print(f"  [yellow]SKIP[/yellow]    {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    console.print(f"  [green]CREATE[/green]  {path}")
    return True


@click.command()
@click.argument("path", default=".", required=False)
@click.option(
    "--template",
    "-t",
    type=click.Choice(list(TEMPLATES)),
    default=None,
    help="The template to start from; asked for when omitted.",
)
@click.option("--name", "-n", default=None, help="Project name (defaults to directory name).")
@click.option("--force", is_flag=True, help="Overwrite existing files.")
def new(path: str, template: str | None, name: str | None, force: bool) -> None:
    """Create a new Zorvex project from a template.

    \b
    - zorvex.json: the graphs Zorvex serves
    - pyproject.toml, README.md, .gitignore
    - .env.example: what .env starts from (the first `zorvex dev` creates .env)
    - src/<slug>/graph.py: the template's graph
    - src/<slug>/state.py, prompts.py, context.py, utils.py: the modules it imports
    - src/<slug>/tools.py: its tools (ReAct agent only)
    - docker-compose.yml, Dockerfile, .dockerignore: PostgreSQL, Redis and the server

    Examples:

    \b
        zorvex new                                  # Interactive mode
        zorvex new ./my-agent                       # Create at path
        zorvex new ./my-agent -t simple-chatbot     # New Zorvex Project
        zorvex new ./my-agent -t react-agent        # ReAct Agent
        zorvex new ./my-agent -t simple-chatbot -n "My Agent"
    """
    interactive = _is_interactive()

    # Ask where, unless the path, or both the template and the name, are given.
    if path == "." and interactive and not (template and name):
        path = click.prompt(click.style("Where should I create the project?", bold=True), default=".")
    project_path = Path(path).resolve()

    # The named template, else the one chosen (the first when nobody can be asked).
    if template is None:
        template = _prompt_template() if interactive else next(iter(TEMPLATES))
    selected = TEMPLATES[template]
    click.echo(f"You selected: {selected.name} - {selected.description}")

    project_name = name or project_path.name
    slug = slugify(project_name)
    variables = {"project_name": project_name, "slug": slug, "zorvex_version": __version__}
    config = {"name": project_name, "dependencies": ["./src"], "graphs": {slug: f"./src/{slug}/graph.py:graph"}}
    files = {
        "zorvex.json": json.dumps(config, indent=2) + "\n",
        **{
            Template(destination).substitute(variables): render(source, variables)
            for source, destination in {**SHARED_FILES, **selected.files}.items()
        },
        **docker_files(),
    }

    click.echo()
    click.echo(click.style("Creating project...", bold=True))
    created = sum(_write_file(project_path / relative, content, force) for relative, content in files.items())

    click.echo()
    click.echo(f"New project created at {project_path}")
    click.echo()
    if created < len(files):
        console.print(f"[green]{created}[/green] files created, [yellow]{len(files) - created}[/yellow] files skipped")
        click.echo()

    click.echo(click.style("Next steps:", bold=True))
    if project_path != Path.cwd().resolve():
        try:
            location = os.path.relpath(project_path)
        except ValueError:  # on another drive, on Windows
            location = str(project_path)
        click.echo(f"  cd {location}")
    click.echo("  uv sync                    # Install dependencies")
    click.echo("  uv run zorvex dev          # Start developing!")
    click.echo()
    click.echo("The first `zorvex dev` creates .env from .env.example: put your model provider's key there.")
