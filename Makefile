.PHONY: help install dev-install setup-hooks format lint type-check security clean run ci-check openapi env-example build

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make dev-install   - Install all dependencies + git hooks"
	@echo "  make setup-hooks   - Reinstall git hooks (if needed)"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make type-check    - Run ty type checking"
	@echo "  make security      - Run security checks with bandit"
	@echo "  make openapi       - Regenerate docs/openapi.json from code"
	@echo "  make env-example   - Regenerate .env.example from the zorvex-cli template"
	@echo "  make ci-check      - Run all CI checks locally"
	@echo "  make build         - Build zorvex-api and zorvex-cli into dist/, as a release does"
	@echo "  make clean         - Clean cache files"
	@echo "  make run           - Run the dev server (zorvex dev)"

install:
	uv sync --all-packages --no-dev

dev-install:
	uv sync --all-packages
	@uv run pre-commit install
	@uv run pre-commit install --hook-type commit-msg
	@echo ""
	@echo "Done! Dependencies installed and git hooks set up."

setup-hooks:
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	@echo ""
	@echo "Git hooks reinstalled!"

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

type-check:
	uv run ty check libs/zorvex-api/src/ libs/zorvex-cli/src/

security:
	uv run bandit -c pyproject.toml -r libs/zorvex-api/src/ libs/zorvex-cli/src/

openapi:
	uv run --package zorvex-api python scripts/export_openapi.py

env-example:
	uv run --package zorvex-cli python -c "from zorvex_cli.templates import render_env_example; print(render_env_example({'slug': 'zorvex'}), end='')" > .env.example

ci-check: format lint
	-uv run ty check libs/zorvex-api/src/ libs/zorvex-cli/src/
	-uv run bandit -c pyproject.toml -r libs/zorvex-api/src/ libs/zorvex-cli/src/
	@echo ""
	@echo "All CI checks completed! (ty and bandit are non-blocking)"

build:
	rm -rf dist
	uv build --all-packages --out-dir dist

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ty_cache .ruff_cache dist 2>/dev/null || true

run:
	uv run zorvex dev
