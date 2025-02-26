# Contributing Guide

Thank you for your interest in contributing to the Narrative LLM Tools project! This guide will help you get started with the development process.

## Prerequisites

Before contributing, make sure you have:

- Python 3.10 or higher installed
- A GitHub account
- Basic knowledge of Git and GitHub workflow

## Getting Started

### Setting Up Your Development Environment

1. Fork the repository on GitHub.

2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/narrative-llm-tools.git
   cd narrative-llm-tools
   ```

3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

5. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Project Structure

```
narrative-llm-tools/
├── docs/                 # Documentation
├── narrative_llm_tools/  # Main package
│   ├── cli/              # Command-line interface
│   ├── handlers/         # Integration handlers (HuggingFace, etc.)
│   ├── narrative_api/    # Narrative API integration
│   ├── rest_api_client/  # REST API client
│   ├── reward_functions/ # Reward functions for LLM evaluation
│   ├── state/            # Conversation state management
│   ├── tools/            # Tool definitions and utilities
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── pyproject.toml        # Project configuration
├── Makefile              # Build commands
└── README.md             # Project overview
```

## Development Workflow

### Creating a New Feature

1. Create a new branch for your feature:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Make your changes, following the code style guidelines.

3. Add tests for your feature in the `tests/` directory.

4. Run the tests to ensure everything works:
   ```bash
   make test
   ```

5. Format and lint your code:
   ```bash
   make format
   make lint
   ```

6. Commit your changes with a descriptive message:
   ```bash
   git commit -m "feat: add your feature description"
   ```

7. Push your branch to GitHub:
   ```bash
   git push origin feat/your-feature-name
   ```

8. Open a Pull Request on GitHub.

### Fixing a Bug

1. Create a new branch:
   ```bash
   git checkout -b fix/bug-description
   ```

2. Fix the bug and add tests that demonstrate the fix.

3. Run tests, format, and lint as described above.

4. Commit with a descriptive message:
   ```bash
   git commit -m "fix: description of the bug fix"
   ```

5. Push and open a Pull Request.

## Code Style Guidelines

We follow strict code style guidelines to ensure consistency across the codebase:

- Use **mypy** with strict mode for type checking
- Follow **black** and **ruff** formatting rules
- Maximum line length is 100 characters
- Use snake_case for functions, variables, and modules
- Use PascalCase for classes
- Include docstrings for all public functions and classes
- Add type annotations to all functions and methods

## Testing

We use pytest for testing. All new features should include tests:

```bash
# Run all tests
make test

# Run a specific test
uv run pytest tests/path/to/test_file.py::test_function
```

## Pre-commit Hooks

The project uses pre-commit hooks to enforce code style. Install them with:

```bash
pre-commit install
```

This will automatically run formatters and linters when you commit code.

## Documentation

When adding new features, please update the documentation:

1. Add docstrings to your code following Google style guidelines
2. Update or create markdown files in the `docs/` directory
3. If adding a new module, ensure it's included in the API reference

## Release Process

Our release process follows these steps:

1. Bump the version in `pyproject.toml`
2. Update `release_notes.md` with new changes
3. Create a new tag: `git tag v0.1.x`
4. Push the tag: `git push origin v0.1.x`
5. CI will automatically create a new release

## Getting Help

If you need help or have questions:

- Open an issue on GitHub
- Reach out to the maintainers
- Check the existing documentation and issues

Thank you for contributing to Narrative LLM Tools!
