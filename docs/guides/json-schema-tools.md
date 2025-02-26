# JSON Schema Tools

Narrative LLM Tools provides powerful utilities for working with JSON Schema, particularly for defining tools, validating data, and enforcing output formats.

## Overview

JSON Schema is a vocabulary that allows you to validate, document, and interact with JSON data. In the context of LLMs, it's particularly useful for:

- Defining the structure of tool inputs and outputs
- Validating that LLM-generated content follows specific formats
- Enforcing structure during generation to improve first-pass success rates

This guide covers the main components related to JSON Schema in the Narrative LLM Tools library.

## Tool Definitions

### Creating Tool Schemas

The `define_tool_schema` function is the primary way to create tool definitions:

```python
from narrative_llm_tools.tools import define_tool_schema

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
```

This creates a properly formatted tool that can be passed to the API client or used with conversation state.

### Working with Multiple Tools

You can define multiple tools and use them together:

```python
tools = [
    define_tool_schema(
        name="search",
        description="Search for information on the web",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    ),
    define_tool_schema(
        name="calculator",
        description="Perform a mathematical calculation",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    )
]

# Use with API client
response = client.chat_completion(
    model="narrative-io/cogenome-13b",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)
```

### Tool Schema Utilities

The `JsonSchemaTools` class provides utilities for working with collections of tools:

```python
from narrative_llm_tools.tools.json_schema_tools import JsonSchemaTools

# Create a tool collection
tool_collection = JsonSchemaTools(tools=tools)

# Filter tools by name
filtered_tools = tool_collection.remove_tool_by_name("calculator")

# Get a JSON schema array representation
schema_array = tool_collection.to_dict()
```

## Schema Validation

### Validating Tool Calls

You can validate that a tool call conforms to its schema:

```python
from narrative_llm_tools.tools.json_schema_tools import validate_tool_call_against_schema

# Tool call from LLM response
tool_call = {
    "name": "get_weather",
    "arguments": {"location": "New York", "unit": "celsius"}
}

# Validate against schema
is_valid = validate_tool_call_against_schema(tool_call, weather_tool)

if is_valid:
    print("Tool call is valid!")
else:
    print("Tool call doesn't match schema")
```

### Common Validation Scenarios

The library supports various validation scenarios:

```python
# Check required fields
tool_call_missing_required = {
    "name": "get_weather",
    "arguments": {"unit": "celsius"}
}
is_valid = validate_tool_call_against_schema(tool_call_missing_required, weather_tool)
# Result: False (missing "location")

# Check enum values
tool_call_invalid_enum = {
    "name": "get_weather",
    "arguments": {"location": "New York", "unit": "kelvin"}
}
is_valid = validate_tool_call_against_schema(tool_call_invalid_enum, weather_tool)
# Result: False (invalid enum value for "unit")
```

## Format Enforcement

For the best results, you can enforce schemas during generation rather than just validating afterward.

### FormatEnforcer

The `FormatEnforcer` class constrains LLM outputs to follow specific formats:

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

# Use with API client
response = client.chat_completion(
    model="narrative-io/cogenome-13b",
    messages=[
        {"role": "user", "content": "Write a review for the latest iPhone"}
    ],
    format_enforcer=enforcer
)

# The response will be enforced to match the schema
print(response.choices[0].message.content)
```

### Caching for Performance

The library uses an LRU cache to optimize format enforcer creation:

```python
from narrative_llm_tools.utils.format_enforcer import get_format_enforcer

# Get a cached format enforcer (created if not in cache)
enforcer1 = get_format_enforcer(schema=product_review_schema)
enforcer2 = get_format_enforcer(schema=product_review_schema)

# Both are the same instance (cached)
print(enforcer1 is enforcer2)  # True
```

## Integration with Reward Functions

JSON schema validation is a key component of the reward functions used for evaluating LLM outputs:

```python
from narrative_llm_tools.reward_functions import tool_calls_validity_reward

# Define a prompt with a schema
prompt_with_schema = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "tool_catalog", "content": "[{...schema here...}]"},
    {"role": "user", "content": "What's the weather in New York?"}
]

# LLM completion to evaluate
completion = "<|tool_calls|>[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"New York\"}}]<|eot_id|>"

# Validate completion
score = tool_calls_validity_reward([completion], [prompt_with_schema])
# Result: [1.0] if valid
```

## Advanced Use Cases

### Custom Validation Logic

You can build custom validation logic on top of the base tools:

```python
from narrative_llm_tools.tools.json_schema_tools import validate_json_against_schema

def validate_weather_response(response_json):
    schema = {
        "type": "object",
        "properties": {
            "temperature": {"type": "number"},
            "conditions": {"type": "string"},
            "location": {"type": "string"}
        },
        "required": ["temperature", "conditions", "location"]
    }

    return validate_json_against_schema(response_json, schema)
```

### REST API Tool Integration

JSON Schema tools also support REST API integration:

```python
from narrative_llm_tools.tools.json_schema_tools import define_rest_api_tool

weather_api_tool = define_rest_api_tool(
    name="get_weather_api",
    description="Get weather data from an API",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    },
    rest_api_config={
        "url": "https://api.weather.com/current/{location}",
        "method": "GET",
        "parameter_location": "QUERY"
    }
)
```

## Best Practices

1. **Define Schemas Precisely**: Include required fields, descriptions, and constraints
2. **Use Enums**: When appropriate, use enum values to restrict choices
3. **Test Validation**: Verify your schemas work with both valid and invalid inputs
4. **Use Format Enforcement**: For critical applications, use format enforcement during generation
5. **Keep Schemas Simple**: Avoid overly complex schemas that might confuse the LLM

## Next Steps

- Check out the [Reward Functions](../reward_functions/index.md) documentation for more on validation
- Explore the [Format Enforcer](format-enforcer.md) for advanced format control
- See the [API Reference](../api/tools.md) for complete details on schema tools
