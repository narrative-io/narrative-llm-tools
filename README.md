# Narrative LLM Tools

A set of tools to use when calling Narrative I/O LLMs. This package provides utilities for managing LLM conversations, handling REST API interactions, and enforcing JSON schema formats.

## Features

- REST API client with configurable authentication and parameter handling
- Conversation state management for LLM interactions
- JSON schema validation and enforcement for LLM outputs
- Tool management system with support for both REST and non-REST tools
- Caching system for format enforcers
- Comprehensive test coverage

## Installation

You can install the package using pip:

```bash
pip install narrative-llm-tools
```

## Requirements

- Python ≥ 3.11
- Dependencies:
  - requests ≥ 2.32.3
  - pydantic ≥ 2.10.4
  - jmespath ≥ 1.0.1
  - lm-format-enforcer ≥ 0.10.9
  - cachetools ≥ 5.5.0
  - transformers ≥ 4.47.1
  - torch == 2.5.1

## License

MIT License