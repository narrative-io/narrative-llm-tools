# Narrative LLM Tools

A comprehensive toolkit for working with Narrative I/O LLMs and managing AI-powered conversations.

## Overview

Narrative LLM Tools is a Python library that provides utilities for:

- Managing LLM conversations
- Handling REST API interactions
- Enforcing JSON schema formats
- Validating and scoring completions
- Managing tool calls and responses

## Key Features

- **🤗 HuggingFace Integration**: Ready-to-use handler for Inference Endpoints
- **🌐 REST API Client**: Easy access to Narrative I/O's LLM services
- **💬 Conversation State**: Sophisticated state management for multi-turn conversations
- **✅ JSON Schema Validation**: Tools for ensuring proper format and structure
- **🛠️ Tool Management**: Built-in support for tool definition and validation
- **📊 Reward Functions**: Scoring mechanisms for evaluating LLM outputs
- **💾 Caching System**: Performance optimizations for format enforcement

## Supported LLMs

This toolkit is designed for use with [Narrative I/O LLMs](https://huggingface.co/spaces/narrative-io/README). These models are optimized for tool use and structured reasoning.

## Getting Started

Check out the [Installation](getting-started/installation.md) guide to set up the package, then follow the [Quick Start](getting-started/quick-start.md) to make your first API call.

## Example

```python
from narrative_llm_tools.rest_api_client import NarrativeApiClient
from narrative_llm_tools.state import ConversationState

# Initialize the client
client = NarrativeApiClient(api_key="your_api_key")

# Create a conversation
conversation = ConversationState()
conversation.add_user_message("What's the weather in New York?")

# Get a completion
response = client.chat_completion(
    model="narrative-io/cogenome-13b",
    messages=conversation.get_messages(),
)

# Update conversation with the response
conversation.add_assistant_message(response.choices[0].message.content)

# Print the response
print(response.choices[0].message.content)
```
