# Quick Start Guide

This guide will help you get up and running with Narrative LLM Tools quickly.

## Basic Usage

### Setting Up a REST API Client

The REST API client is the primary way to interact with Narrative I/O's LLM services:

```python
from narrative_llm_tools.rest_api_client import NarrativeApiClient

# Initialize the client
client = NarrativeApiClient(
    api_key="your_api_key",  # Replace with your actual API key
    base_url="https://api.narrative.io/v1"  # Default URL
)
```

### Making a Simple Chat Completion Request

```python
response = client.chat_completion(
    model="narrative-io/cogenome-13b",  # Choose a model
    messages=[
        {"role": "user", "content": "Tell me about Narrative I/O"}
    ],
    temperature=0.7,
)

# Print the response
print(response.choices[0].message.content)
```

## Managing Conversations

For multi-turn conversations, use the `ConversationState` class:

```python
from narrative_llm_tools.state import ConversationState

# Create a conversation
conversation = ConversationState()

# Add a user message
conversation.add_user_message("What can you do?")

# Get a completion
response = client.chat_completion(
    model="narrative-io/cogenome-13b",
    messages=conversation.get_messages(),
)

# Update conversation with the response
conversation.add_assistant_message(response.choices[0].message.content)

# Continue the conversation
conversation.add_user_message("Can you give me an example?")

# Get another completion
response = client.chat_completion(
    model="narrative-io/cogenome-13b",
    messages=conversation.get_messages(),
)
```

## Working with Tools

Narrative LLM models support tool use for more advanced capabilities:

```python
from narrative_llm_tools.tools import define_tool_schema

# Define a weather tool
weather_tool = define_tool_schema(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature"
            }
        },
        "required": ["location"]
    }
)

# Create a conversation with tools
conversation = ConversationState()
conversation.add_user_message("What's the weather in New York?")

# Get a completion with tool access
response = client.chat_completion(
    model="narrative-io/cogenome-13b",
    messages=conversation.get_messages(),
    tools=[weather_tool],
    tool_choice="auto",
)

# Process tool calls (if any)
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    # Here you would implement the actual tool functionality

    # Add the tool response back to the conversation
    tool_response = {"temperature": 72, "conditions": "sunny"}
    conversation.add_tool_result(
        tool_call_id=tool_call.id,
        function_name=tool_call.function.name,
        result=tool_response
    )
```

## Using Format Enforcement

To enforce specific output formats:

```python
from narrative_llm_tools.utils.format_enforcer import FormatEnforcer

# Define a schema for a product review
product_review_schema = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
        "title": {"type": "string"},
        "review": {"type": "string"},
        "pros": {"type": "array", "items": {"type": "string"}},
        "cons": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["rating", "title", "review"]
}

# Create a format enforcer
enforcer = FormatEnforcer(schema=product_review_schema)

# Use the enforcer with the API client
response = client.chat_completion(
    model="narrative-io/cogenome-13b",
    messages=[
        {"role": "user", "content": "Write a review for the latest iPhone"}
    ],
    format_enforcer=enforcer
)

# The response will be enforced to match the schema
review = response.choices[0].message.content
print(review)
```

## Next Steps

Now that you understand the basics, check out these resources for more advanced usage:

- [REST API Client Guide](../guides/rest-api-client.md)
- [Conversation State Guide](../guides/conversation-state.md)
- [JSON Schema Tools Guide](../guides/json-schema-tools.md)
- [Reward Functions Documentation](../reward_functions/index.md)
