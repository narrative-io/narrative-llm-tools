# Narrative LLM Tools - Development Guide

## Build Commands
- `make format`: Run ruff and black formatters
- `make lint`: Run mypy, ruff, and black in check mode
- `make test`: Run all tests
- `uv run pytest tests/path/to/test_file.py::test_function`: Run specific test
- `make pre-commit`: Run format, lint, and test (use before commits)

## Code Style Guidelines
- **Type Annotations**: Strict typing required - use mypy with strict mode
- **Line Length**: 100 characters max
- **Formatting**: Uses ruff format and black
- **Imports**: Use ruff's isort configuration
- **Python Version**: Target Python 3.10-3.12
- **Error Handling**: Log exceptions with traceback using `logger.error`, include context
- **Testing**: Write pytest tests for all functionality with good coverage
- **Naming**:
  - snake_case for functions, variables, and modules
  - PascalCase for classes
  - Use descriptive names that convey purpose
- **Comments**: Document complex logic and public API functions with docstrings
- **Caching**: Use LRUCache from cachetools for expensive operations

Project leverages pydantic for data validation and uses cachetools for performance optimization.
