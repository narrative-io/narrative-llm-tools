.PHONY: lint
lint:
	uv run mypy . --strict
	uv run ruff check .
	uv run black . --check

.PHONY: format
format:
	uv run ruff check --fix .
	uv run black .