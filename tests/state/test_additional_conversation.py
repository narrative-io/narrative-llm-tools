import json
import pytest
from jsonschema import SchemaError
from narrative_llm_tools.state.conversation import (
    Conversation,
    validate_tool_catalog_schema,
    validate_tool_response_structure,
    validate_tool_response_matching,
    extract_enumerated_names,
    generate_tool_response_json_schema,
)


def test_validate_tool_response_structure_missing_fields():
    """Test that validate_tool_response_structure detects missing fields."""
    response = {"name": "test_tool"}  # Missing 'content'
    errors = validate_tool_response_structure(response, 0)
    assert len(errors) == 1
    assert "must have exactly 'name' and 'content' fields" in errors[0]


def test_validate_tool_response_structure_extra_fields():
    """Test that validate_tool_response_structure detects extra fields."""
    response = {"name": "test_tool", "content": "result", "extra": "field"}
    errors = validate_tool_response_structure(response, 0)
    assert len(errors) == 1
    assert "must have exactly 'name' and 'content' fields" in errors[0]


def test_validate_tool_response_structure_wrong_types():
    """Test that validate_tool_response_structure detects wrong types."""
    response = {"name": 123, "content": ["not", "a", "string"]}
    errors = validate_tool_response_structure(response, 0)
    assert len(errors) == 1
    assert "'name' and 'content' must be strings" in errors[0]


def test_validate_tool_response_matching_with_mismatched_names():
    """Test validate_tool_response_matching with mismatched names."""
    response = {"name": "wrong_tool", "content": "result"}
    prev_call = {"name": "test_tool", "parameters": {"param": "value"}}
    errors = validate_tool_response_matching(response, prev_call, 0)
    assert len(errors) == 1
    assert "name 'wrong_tool' does not match tool call name 'test_tool'" in errors[0]


def test_validate_tool_response_matching_with_missing_prev_name():
    """Test validate_tool_response_matching with missing name in prev_call."""
    response = {"name": "test_tool", "content": "result"}
    prev_call = {"parameters": {"param": "value"}}  # Missing 'name'
    errors = validate_tool_response_matching(response, prev_call, 0)
    assert len(errors) == 1
    assert "name 'test_tool' does not match tool call name 'None'" in errors[0]


def test_extract_enumerated_names_invalid_schema():
    """Test extract_enumerated_names with various invalid schemas."""
    invalid_schemas = [
        {},  # Empty schema
        {"items": {}},  # Missing anyOf
        None,  # None schema
        {"items": {"anyOf": [{"properties": {}}]}},  # Missing name property
    ]

    for schema in invalid_schemas:
        result = extract_enumerated_names(schema)
        assert result == set()


def test_validate_tool_catalog_schema_invalid_json():
    """Test validate_tool_catalog_schema with invalid JSON."""
    invalid_json = """
    {
        "type": "array",
        missing_quotes: "value"
    }
    """
    schema, errors = validate_tool_catalog_schema(invalid_json)
    assert schema is None
    assert len(errors) == 1
    assert "must be valid JSON" in errors[0]


def test_validate_tool_catalog_schema_empty_string():
    """Test validate_tool_catalog_schema with empty string."""
    schema, errors = validate_tool_catalog_schema("")
    assert schema is None
    assert len(errors) == 1
    assert "must be valid JSON" in errors[0]


def test_validate_tool_catalog_schema_with_schema_error(monkeypatch):
    """Test validate_tool_catalog_schema when jsonschema.Draft7Validator.check_schema raises SchemaError."""
    # Create a valid JSON that will trigger SchemaError
    valid_json = '{"type": "invalid_type"}'

    # Mock the check_schema method to raise SchemaError
    def mock_check_schema(schema):
        raise SchemaError("Invalid schema type")

    # Apply the mock
    monkeypatch.setattr("jsonschema.Draft7Validator.check_schema", mock_check_schema)

    # Run the test
    schema, errors = validate_tool_catalog_schema(valid_json)
    assert schema is None
    assert len(errors) == 1
    assert "Invalid schema type" in errors[0]


def test_generate_tool_response_json_schema():
    """Test generate_tool_response_json_schema with a set of tool names."""
    tool_names = frozenset(["tool1", "tool2", "tool3"])
    schema = generate_tool_response_json_schema(tool_names)

    assert schema["type"] == "array"
    assert schema["items"]["type"] == "object"
    assert "properties" in schema["items"]
    assert "name" in schema["items"]["properties"]
    assert "content" in schema["items"]["properties"]
    assert schema["items"]["properties"]["name"]["type"] == "string"
    assert set(schema["items"]["properties"]["name"]["enum"]) == set(tool_names)
    assert schema["items"]["properties"]["content"]["type"] == "string"
    assert schema["items"]["required"] == ["name", "content"]
