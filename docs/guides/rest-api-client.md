# REST API Client Guide

The REST API client in Narrative LLM Tools provides a flexible way to interact with REST APIs, particularly the Narrative I/O LLM services.

## Overview

Narrative LLM Tools offers a powerful REST API client designed to:

- Simplify API interactions with configurable endpoints
- Handle authentication securely
- Process and validate responses
- Support various parameter locations (query, body, path)
- Provide detailed logging and error handling

## RestApiClient

The core `RestApiClient` class is a general-purpose client for interacting with any REST API.

### Basic Usage

```python
from narrative_llm_tools.rest_api_client import RestApiClient, RestApiConfig
from narrative_llm_tools.rest_api_client.types import HttpMethod, ParameterLocation

# Create a client
client = RestApiClient(
    name="weather_api",
    config=RestApiConfig(
        url="https://api.example.com/weather/{city}",
        method=HttpMethod.GET,
        parameter_location=ParameterLocation.QUERY
    )
)

# Make a call
response = client.call({"city": "London", "units": "metric"})
print(response.body)
```

### Configuration Options

The `RestApiConfig` class provides numerous options for configuring API endpoints:

| Option | Description | Example |
| ------ | ----------- | ------- |
| `url` | Endpoint URL with optional path parameters | `"https://api.example.com/users/{user_id}"` |
| `method` | HTTP method (GET, POST, PUT, DELETE) | `HttpMethod.POST` |
| `auth` | Authentication configuration | `BearerTokenAuth(env_var="API_TOKEN")` |
| `parameter_location` | Where to place parameters (query or body) | `ParameterLocation.BODY` |
| `query_path` | Optional JMESPath for response filtering | `"data.results"` |
| `response_behavior` | Controls how API responses are handled | See documentation |

### Advanced Features

#### Path Parameters

Path parameters in the URL template (indicated by `{param_name}`) are automatically extracted from the parameters passed to `call()`:

```python
client = RestApiClient(
    name="user_api",
    config=RestApiConfig(
        url="https://api.example.com/users/{user_id}/profile",
        method=HttpMethod.GET
    )
)

# The user_id parameter will be used in the URL
response = client.call({"user_id": "123"})
# Request URL: https://api.example.com/users/123/profile
```

#### Authentication

The client supports Bearer token authentication via environment variables:

```python
from narrative_llm_tools.rest_api_client.types import BearerTokenAuth

client = RestApiClient(
    name="protected_api",
    config=RestApiConfig(
        url="https://api.example.com/protected",
        method=HttpMethod.GET,
        auth=BearerTokenAuth(env_var="API_TOKEN")
    )
)
```

This will automatically add an `Authorization: Bearer <token>` header to requests, using the value from the specified environment variable.

#### Response Filtering

Use JMESPath expressions to extract specific data from responses:

```python
client = RestApiClient(
    name="search_api",
    config=RestApiConfig(
        url="https://api.example.com/search",
        method=HttpMethod.GET,
        query_path="results[0].items"
    )
)

# The response will only contain the data matching the query_path
response = client.call({"q": "python"})
```

## Building a NarrativeApiClient

To create a client specifically for Narrative I/O's LLM services, you can extend the base `RestApiClient`:

```python
from narrative_llm_tools.rest_api_client import RestApiClient, RestApiConfig
from narrative_llm_tools.rest_api_client.types import HttpMethod, ParameterLocation, BearerTokenAuth

class NarrativeApiClient:
    def __init__(self, api_key=None, base_url="https://api.narrative.io/v1"):
        self.api_key = api_key
        self.base_url = base_url

        # Create the completion client
        self.completion_client = RestApiClient(
            name="narrative_completion",
            config=RestApiConfig(
                url=f"{base_url}/chat/completions",
                method=HttpMethod.POST,
                parameter_location=ParameterLocation.BODY,
                auth=BearerTokenAuth(env_var="NARRATIVE_API_KEY") if not api_key else None
            )
        )

    def chat_completion(self, model, messages, temperature=0.7, tools=None, tool_choice=None, format_enforcer=None):
        """
        Get a chat completion from the Narrative I/O API.

        Args:
            model (str): The model to use
            messages (list): List of message objects
            temperature (float): Temperature for generation
            tools (list): Optional list of tool schemas
            tool_choice (str): Tool choice setting ("auto", "none")
            format_enforcer (FormatEnforcer): Optional format enforcer instance

        Returns:
            The API response containing the completion
        """
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice or "auto"

        # Apply format enforcer if provided
        if format_enforcer:
            params["format"] = format_enforcer.get_format_settings()

        # Use the api_key if provided, otherwise rely on environment variable
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return self.completion_client.call(params, headers=headers)
```

## Error Handling

The client provides detailed error handling with informative exceptions:

```python
from narrative_llm_tools.rest_api_client.exceptions import ApiError

try:
    response = client.call(params)
except ApiError as e:
    print(f"API Error: {e.status_code} - {e.message}")
    print(f"Response body: {e.response_body}")
```

## Performance Considerations

For high-performance applications:

1. **Reuse clients**: Create the client once and reuse it for multiple calls
2. **Use connection pooling**: The client supports connection pooling internally
3. **Monitor response times**: The client logs timing information for performance tracking

## Next Steps

- Check out the [API Reference](../api/rest-api-client.md) for complete details
- Explore [Conversation State](conversation-state.md) to manage multi-turn interactions
- Learn about [JSON Schema Tools](json-schema-tools.md) for defining tool schemas
