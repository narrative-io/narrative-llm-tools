# Conversation State Management

The Narrative LLM Tools library provides a robust system for managing the state of conversations with LLMs, ensuring proper handling of multi-turn interactions, tool usage, and state transitions.

## Overview

The conversation state system is designed to:

- Track messages between users and assistants
- Manage tool calls and their responses
- Enforce conversation structure and rules
- Control state transitions based on conversation flow
- Limit tool usage to prevent infinite loops

## Key Components

### ConversationState

This is the primary class used for managing conversations:

```python
from narrative_llm_tools.state import ConversationState

# Create a new conversation
conversation = ConversationState()

# Add a system message (optional, a default is used if none provided)
conversation.add_system_message("You are a helpful assistant. Use the provided tools.")

# Add a user message
conversation.add_user_message("Can you tell me about the weather in New York?")
```

### Message Types

The library supports several types of messages:

- **System Messages**: Configure assistant behavior
- **User Messages**: Input from the user
- **Assistant Messages**: Responses from the LLM
- **Tool Catalog Messages**: Define available tools
- **Tool Call Messages**: Tool invocations from the assistant
- **Tool Response Messages**: Results returned from tools

### Conversation Status

Conversation status is tracked using an enum:

```python
from narrative_llm_tools.state.conversation_state import ConversationStatus

# Check the current status
print(conversation.status)  # e.g., ConversationStatus.RUNNING

# Valid statuses include:
# - RUNNING: Normal conversation flow
# - WAITING_TOOL_RESPONSE: Waiting for a tool to complete
# - COMPLETED: Conversation has ended
# - ERROR: An error has occurred
```

## Using ConversationState

### Basic Conversation Flow

```python
# Initialize conversation
conversation = ConversationState()

# User message
conversation.add_user_message("What's the weather in Paris?")

# Get messages for API request
messages = conversation.get_messages()

# Make API call (using NarrativeApiClient)
response = client.chat_completion(
    model="narrative-io/cogenome-13b",
    messages=messages
)

# Add assistant's response
assistant_message = response.choices[0].message.content
conversation.add_assistant_message(assistant_message)

# Continue with another user message
conversation.add_user_message("And how about London?")
```

### Working with Tools

The conversation state system handles tool usage automatically:

```python
from narrative_llm_tools.tools import define_tool_schema

# Define a weather tool
weather_tool = define_tool_schema(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
)

# Create conversation with tool
conversation = ConversationState(tools=[weather_tool])
conversation.add_user_message("What's the weather in Tokyo?")

# Get messages for API call
messages = conversation.get_messages()

# Make API call with tools
response = client.chat_completion(
    model="narrative-io/cogenome-13b",
    messages=messages,
    tools=[weather_tool],
    tool_choice="auto"
)

# Process tool call
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    conversation.add_assistant_message(
        response.choices[0].message.content,
        tool_calls=response.choices[0].message.tool_calls
    )

    # Status now changes to WAITING_TOOL_RESPONSE
    print(conversation.status)  # ConversationStatus.WAITING_TOOL_RESPONSE

    # Execute the tool (simplified example)
    tool_result = {"temperature": 25, "conditions": "sunny"}

    # Add tool response
    conversation.add_tool_response(
        tool_call_id=tool_call.id,
        content=tool_result
    )

    # Status returns to RUNNING
    print(conversation.status)  # ConversationStatus.RUNNING
```

### Advanced Features

#### Message Filtering

ConversationState provides methods to get filtered views of the conversation:

```python
# Get all messages
all_messages = conversation.get_messages()

# Get last N messages
recent_messages = conversation.get_messages(limit=5)

# Get messages without system or tool catalog
visible_messages = conversation.get_messages(include_system=False, include_tool_catalog=False)
```

#### Tool Rounds Limiting

To prevent infinite tool call loops, you can set a maximum number of tool rounds:

```python
# Limit to 3 rounds of tool usage
conversation = ConversationState(max_tool_rounds=3)

# Check if limit is reached
if conversation.exceeded_tool_rounds():
    print("Maximum tool rounds reached.")
```

#### Conversation Completion

Explicitly mark a conversation as complete:

```python
# End the conversation
conversation.complete()
print(conversation.status)  # ConversationStatus.COMPLETED

# Further message additions will raise exceptions
try:
    conversation.add_user_message("This won't work")
except Exception as e:
    print(f"Error: {e}")
```

## State Transitions

The conversation progresses through a series of states:

1. **RUNNING**: Normal conversation flow
2. **WAITING_TOOL_RESPONSE**: After tool calls, waiting for tool execution
3. **COMPLETED**: Conversation ended (either naturally or explicitly)
4. **ERROR**: An error occurred in processing

Transitions follow strict rules to ensure conversation integrity:

```
RUNNING → WAITING_TOOL_RESPONSE: When a tool is called
WAITING_TOOL_RESPONSE → RUNNING: When all tool responses are received
RUNNING → COMPLETED: When conversation is explicitly completed
Any state → ERROR: When an error occurs
```

## Best Practices

1. **Create a new conversation for each user session**:
   ```python
   conversation = ConversationState()
   ```

2. **Include system messages for consistent behavior**:
   ```python
   conversation.add_system_message("You are a helpful assistant...")
   ```

3. **Track tool usage with proper state management**:
   ```python
   # After tool call
   conversation.add_assistant_message(..., tool_calls=...)

   # After tool response
   conversation.add_tool_response(...)
   ```

4. **Limit tool rounds to prevent infinite loops**:
   ```python
   conversation = ConversationState(max_tool_rounds=5)
   ```

5. **Check conversation status before operations**:
   ```python
   if conversation.status != ConversationStatus.COMPLETED:
       # Continue conversation
   ```

## Example: Complete Conversation with Tools

Here's a complete example showing a conversation with tool usage:

```python
from narrative_llm_tools.state import ConversationState
from narrative_llm_tools.tools import define_tool_schema

# Define a calculator tool
calculator_tool = define_tool_schema(
    name="calculator",
    description="Perform a calculation",
    parameters={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["operation", "a", "b"]
    }
)

# Create conversation
conversation = ConversationState(tools=[calculator_tool])
conversation.add_system_message("You are a helpful math assistant.")
conversation.add_user_message("What is 25 + 17?")

# LLM generates tool call (simulated here)
tool_calls = [
    {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": '{"operation": "add", "a": 25, "b": 17}'
        }
    }
]
conversation.add_assistant_message(
    "I'll calculate that for you.",
    tool_calls=tool_calls
)

# Execute tool (in real app, you'd implement the actual calculator)
result = 42
conversation.add_tool_response(
    tool_call_id="call_123",
    content={"result": result}
)

# LLM generates final response (simulated)
conversation.add_assistant_message("The result of 25 + 17 is 42.")

# User continues
conversation.add_user_message("And what's 42 × 2?")

# Get messages for next API call
messages = conversation.get_messages()
# ... continue conversation
```

## Related Documentation

- [REST API Client Guide](rest-api-client.md): Details on using the API client with conversational state
- [JSON Schema Tools Guide](json-schema-tools.md): Information on defining and using tools
- [API Reference](../api/state.md): Complete API documentation for the state management system
