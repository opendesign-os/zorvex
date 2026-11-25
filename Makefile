.PHONY: help dev-install format check openapi build clean run

help:
	@echo "make dev-install  Install the workspace and its git hooks"
	@echo "make run          Run the dev server (zorvex dev)"
	@echo "make format       Format and fix the code with ruff"
	@echo "make check        Run CI's checks: format, lint, types, security, generated spec"
	@echo "make openapi      Regenerate docs/openapi.json from the code"
	@echo "make build        Build zorvex-api and zorvex-cli into dist/, as a release does"
	@echo "make clean        Remove caches and build output"

dev-install:
	uv sync --all-packages
	uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

run:
	uv run zorvex dev

format:
	uv run ruff format .
	uv run ruff check --fix .

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check libs/zorvex-api/src/ libs/zorvex-cli/src/
	uv run bandit -q -c pyproject.toml -r libs/zorvex-api/src/ libs/zorvex-cli/src/
	$(MAKE) openapi
	git diff --exit-code -- docs/openapi.json

openapi:
	uv run --package zorvex-api python scripts/export_openapi.py

build:
	rm -rf dist
	uv build --all-packages --out-dir dist

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ty_cache .ruff_cache dist
