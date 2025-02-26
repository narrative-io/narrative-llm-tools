# Narrative LLM Tools - Development Guide

## Build Commands
- `make format`: Run ruff and black formatters
- `make lint`: Run mypy, ruff, and black in check mode
- `make test`: Run all tests
- `uv run pytest tests/path/to/test_file.py::test_function`: Run specific test
- `make pre-commit`: Run format, lint, and test (use before commits)

## Workflow Best Practices
- **Before Committing**: Always run `make format` on modified files before adding them to avoid pre-commit hook failures
- **Adding New Files**: Run `make format` on new files before the first commit to ensure proper EOL and whitespace formatting
- **Automatic Checks**: The project uses pre-commit hooks that enforce formatting, linting, and testing standards
- **Handling Hook Failures**: If pre-commit hooks fail, run `make format` and add the files again before committing

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
