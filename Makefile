.PHONY: lint
lint:
	uv run mypy . --strict
	uv run ruff check .
	uv run black . --check

.PHONY: format
format:
	uv run ruff check --fix .
	uv run black .

.PHONY: test
test:
	uv run pytest

.PHONY: pre-commit
pre-commit: format lint test

.PHONY: install-hooks
install-hooks:
	pre-commit install