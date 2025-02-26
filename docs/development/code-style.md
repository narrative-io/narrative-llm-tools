# Code Style Guidelines

Narrative LLM Tools follows a consistent code style to ensure readability and maintainability. This guide outlines our code style requirements and best practices.

## General Guidelines

- Target Python 3.10 and above
- Use strict typing (mypy with strict mode)
- Maximum line length of 100 characters
- Follow formatting rules from ruff and black
- Document all public functions, classes, and modules

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | `snake_case` | `user_message`, `api_key` |
| Functions | `snake_case` | `get_messages()`, `add_user_input()` |
| Classes | `PascalCase` | `ConversationState`, `RestApiClient` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_TOKENS`, `DEFAULT_TIMEOUT` |
| Type Aliases | `PascalCase` | `MessageType`, `ToolSchema` |
| Modules | `snake_case` | `json_schema_tools.py`, `conversation_state.py` |

## Type Annotations

All code should include proper type annotations:

```python
def add_numbers(a: int, b: int) -> int:
    return a + b

def process_messages(messages: list[dict[str, str]]) -> None:
    for message in messages:
        print(message["content"])

# Type aliases for complex types
from typing import TypeAlias, Dict, List, Optional

Message: TypeAlias = Dict[str, str]
MessageList: TypeAlias = List[Message]

def get_messages(limit: Optional[int] = None) -> MessageList:
    # ...
```

## Imports

Organize imports according to the following order:

1. Standard library imports
2. Third-party imports
3. Local application imports

Separate each import group with a blank line:

```python
import json
import os
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel, Field

from narrative_llm_tools.tools import define_tool_schema
from narrative_llm_tools.utils.format_enforcer import FormatEnforcer
```

## Docstrings

We follow Google's docstring style. Every public function and class should have a docstring:

```python
def format_message(message: str, max_length: int = 100) -> str:
    """Format a message with proper length constraints.

    Args:
        message: The message to format
        max_length: Maximum allowed length for the message

    Returns:
        The formatted message trimmed to max_length if necessary

    Raises:
        ValueError: If message is empty or max_length is negative
    """
    if not message:
        raise ValueError("Message cannot be empty")
    if max_length < 0:
        raise ValueError("Max length cannot be negative")

    return message[:max_length]
```

## Error Handling

Use explicit error handling with appropriate exception types:

```python
def parse_json(data: str) -> dict:
    """Parse JSON data with proper error handling.

    Args:
        data: JSON string to parse

    Returns:
        Parsed JSON as a dictionary

    Raises:
        ValueError: If data is not valid JSON
    """
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e
```

## Logging

Use the standard logging module with appropriate log levels:

```python
import logging

logger = logging.getLogger(__name__)

def process_request(request_data: dict) -> dict:
    """Process an API request."""
    logger.debug("Processing request: %s", request_data)

    try:
        # Process the request
        result = do_something(request_data)
        logger.info("Request processed successfully")
        return result
    except Exception as e:
        logger.error("Error processing request: %s", e, exc_info=True)
        raise
```

## Testing

Write tests for all code you contribute:

- Use pytest as the testing framework
- Aim for high code coverage
- Write both unit and integration tests
- Test edge cases and error scenarios

```python
# Example test function
def test_conversation_state_initialization():
    """Test that conversation state initializes correctly."""
    state = ConversationState()
    assert state.status == ConversationStatus.RUNNING
    assert len(state.raw_messages) == 0
    assert state.tool_round_count == 0
```

## Code Organization

- Keep functions focused on a single responsibility
- Limit function length (aim for under 50 lines)
- Group related functionality in modules
- Use classes to encapsulate state and behavior

## Comments

- Use comments sparingly to explain "why" not "what"
- Keep comments up-to-date with code changes
- Use TODO comments for incomplete code (but try to minimize these)

```python
# Good comment
# Use caching to avoid expensive recalculation for common schemas
cached_result = schema_cache.get(schema_hash)

# Avoid obvious comments
# Get the message  # Unnecessary
message = get_message()
```

## Formatting with Tools

The project uses these tools for enforcing code style:

- **black**: Code formatting
- **ruff**: Linting with isort configuration
- **mypy**: Static type checking

Run them with the provided Makefile commands:

```bash
# Format code
make format

# Check code style without modifying
make lint

# Run all checks including type checking
make pre-commit
```

## Examples

### Good Example

```python
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class MessageProcessor:
    """Process and validate conversation messages.

    This class handles message validation, transformation, and storage
    for conversation state management.
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialize the message processor.

        Args:
            max_history: Maximum number of messages to retain
        """
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the conversation.

        Args:
            role: The role of the message sender (user, assistant, etc.)
            content: The message content

        Raises:
            ValueError: If role or content is invalid
        """
        if not role or not content:
            raise ValueError("Role and content cannot be empty")

        message = {"role": role, "content": content}
        self.messages.append(message)

        # Trim history if it exceeds max size
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

        logger.debug("Added message from %s, message count: %d", role, len(self.messages))
```

By following these guidelines, we maintain a consistent, readable, and maintainable codebase.
