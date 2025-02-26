# Narrative Chat Templates

The Narrative LLM Tools package includes Jinja2 templates for formatting conversations with language models. These templates handle the proper structuring of messages, system prompts, and tool interactions.

## Available Templates

### 1. Standard Chat Template (`narrative_chat_template.j2`)

The standard template formats conversations between users, assistants, and tools with appropriate header tokens.

**Key Features:**
- Configurable tokens for message boundaries
- Support for system messages
- Tool catalog integration
- Tool calls and responses formatting
- Optional generation prompting

**Template Parameters:**
- `bos_token`: Begin of sequence token (default: `<|begin_of_text |>`)
- `tool_calls_token`: Token indicating tool calls (default: `<|tool_calls |>`)
- `tool_catalog_start_token`: Token marking tool catalog start (default: `<|tool_catalog_start |>`)
- `tool_catalog_end_token`: Token marking tool catalog end (default: `<|tool_catalog_end |>`)
- `add_generation_prompt`: Whether to add prompt for model generation (default: `false`)
- `force_tool_calls`: Whether to force tool calls in generated responses (default: `true`)

### 2. Chain of Thought Template (`narrative_chat_template_cot.j2`)

An extension of the standard template that adds support for chain-of-thought reasoning.

**Additional Features:**
- Thought process tracking with dedicated tokens
- Sequential thought rendering
- Chain-of-thought prompt generation

**Additional Parameters:**
- `use_chain_of_thought`: Whether to use chain-of-thought formatting (default: `false`)

## Message Structure

Templates expect messages with the following structure:
```python
{
    "role": "user" | "assistant" | "system" | "tool" | "tool_response" | "tool_calls" | "tool_catalog",
    "content": "Message content",
    "thoughts": ["Thought 1", "Thought 2"]  # Optional, only used in CoT template
}
```

## Usage Examples

When rendering these templates, you'll typically provide:

1. A list of message objects as described above
2. Optional configuration parameters to customize behavior

The templates will format the conversation appropriately for consumption by compatible language models.
