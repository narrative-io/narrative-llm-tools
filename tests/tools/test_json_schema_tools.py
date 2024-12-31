import pytest
from pydantic import ValidationError
from narrative_llm_tools.rest_api_client.types import BearerTokenAuth, RestApiConfig
from narrative_llm_tools.rest_api_client.rest_api_client import RestApiClient

from narrative_llm_tools.tools.json_schema_tools import (
    NameProperty, ParametersProperty, ToolProperties, ToolSchema,
    Tool, ItemsSchema, JsonSchemaTools
)

@pytest.fixture
def sample_config():
    return RestApiConfig(
        url="https://api.example.com/users/{user_id}",
        method="GET",
        auth=BearerTokenAuth(env_var="API_TOKEN"),
        parameter_location="query"
    )

@pytest.fixture
def example_tool_schema(sample_config):
    return ToolSchema(
        type="object",
        required=["name", "parameters"],
        properties=ToolProperties(
            name=NameProperty(
                type="string",
                enum=["example_tool"],
                description="An example tool"
            ),
            parameters=ParametersProperty(
                type="object",
                properties={
                    "example_param": {"type": "string", "description": "An example parameter"}
                },
                required=["example_param"],
                additionalProperties=False
            )
        ),
        restApi=sample_config,
        additionalProperties=False
    )

@pytest.fixture
def json_schema_tools(example_tool_schema):
    return JsonSchemaTools(
        type="array",
        items=ItemsSchema(anyOf=[example_tool_schema])
    )

# Tests for ToolSchema
def test_tool_schema_hash(example_tool_schema):
    hashed_value = hash(example_tool_schema)
    assert isinstance(hashed_value, int)

def test_tool_schema_equality(example_tool_schema):
    assert example_tool_schema == example_tool_schema

def test_user_response_tool():
    user_response_tool = ToolSchema.user_response_tool()
    assert user_response_tool.properties.name.enum == ["respond_to_user"]
    assert "response" in user_response_tool.properties.parameters.properties

# Tests for JsonSchemaTools
def test_get_rest_apis(json_schema_tools):
    apis = json_schema_tools.get_rest_apis()
    assert "example_tool" in apis
    assert isinstance(apis["example_tool"], RestApiClient)

def test_remove_tool_by_name(json_schema_tools):
    updated_tools = json_schema_tools.remove_tool_by_name("example_tool")
    assert len(updated_tools.items.anyOf) == 0

def test_remove_rest_api_tools(json_schema_tools):
    updated_tools = json_schema_tools.remove_rest_api_tools()
    assert len(updated_tools.items.anyOf) == 0

def test_clear_tools(json_schema_tools):
    json_schema_tools.clear_tools()
    assert len(json_schema_tools.items.anyOf) == 0

def test_with_user_response_tool(json_schema_tools):
    updated_tools = json_schema_tools.with_user_response_tool()
    assert len(updated_tools.items.anyOf) == 2
    assert updated_tools.items.anyOf[-1].properties.name.enum == ["respond_to_user"]

def test_only_user_response_tool():
    json_tools = JsonSchemaTools.only_user_response_tool()
    assert len(json_tools.items.anyOf) == 1
    assert json_tools.items.anyOf[0].properties.name.enum == ["respond_to_user"]

# Validation Tests
def test_invalid_name_property():
    with pytest.raises(ValidationError):
        NameProperty(type="integer", enum=["invalid"])

def test_invalid_parameters_property():
    with pytest.raises(ValidationError):
        ParametersProperty(type="array", properties={"key": {"type": "string"}})

def test_invalid_tool_schema():
    with pytest.raises(ValidationError):
        ToolSchema(
            type="array",
            required=["invalid"],
            properties=None
        )
