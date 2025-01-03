import json
import pytest
from narrative_llm_tools.state.messages import ToolResponseMessage, parse_json_array

def test_parse_json_array():
    # Test valid JSON array without schema
    content = '[{"name": "test", "content": "value"}]'
    errors, parsed = parse_json_array(content, None)
    assert not errors
    assert parsed == [{"name": "test", "content": "value"}]

    # Test valid JSON array with schema
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["name", "content"]
        }
    }
    errors, parsed = parse_json_array(content, schema)
    assert not errors
    assert parsed == [{"name": "test", "content": "value"}]

    # Test invalid JSON
    content = '[{"name": "test", "content": "value"'  # Missing closing brackets
    errors, parsed = parse_json_array(content, None)
    assert len(errors) == 1
    assert "Invalid JSON in 'value'" in errors[0]
    assert parsed == []

    # Test non-array JSON
    content = '{"name": "test"}'
    errors, parsed = parse_json_array(content, None)
    assert len(errors) == 1
    assert "Expected a list" in errors[0]
    assert parsed == []

    # Test JSON array that doesn't match schema
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "different": {"type": "string"}
            },
            "required": ["different"]
        }
    }
    content = '[{"name": "test", "content": "value"}]'
    errors, parsed = parse_json_array(content, schema)
    assert len(errors) == 1
    assert "Does not match tool_catalog schema" in errors[0]

    # Test empty array
    content = '[]'
    errors, parsed = parse_json_array(content, None)
    assert not errors
    assert parsed == []

    # Test malformed JSON
    content = 'not json at all'
    errors, parsed = parse_json_array(content, None)
    assert len(errors) == 1
    assert "Invalid JSON in 'value'" in errors[0]
    assert parsed == []

import pytest
from narrative_llm_tools.state.messages import ToolCatalogMessage

def test_valid_tool_catalog():
    # Valid JSON Schema
    valid_schema = {
                    "type": "array",
                    "items": {
                    "anyOf": [
                        {
                        "type": "object",
                        "required": ["name", "parameters"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": ["my_generic_tool"],
                                "description": "The unique name of this tool."
                            },
                            "parameters": {
                            "type": "object",
                            "properties": {
                                "param1": {
                                    "type": "string",
                                    "description": "A required string parameter for this tool."
                                },
                                "param2": {
                                    "type": "integer",
                                    "description": "An optional integer parameter for this tool."
                                }
                            },
                            "required": ["param1"],
                            "additionalProperties": False
                            }
                        },
                        "additionalProperties": False
                        },
                        {
                        "type": "object",
                        "required": ["name", "parameters"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": ["text_summarizer"],
                                "description": "This tool is a text summarizer."
                            },
                            "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The text to summarize."
                                },
                                "summary_length": {
                                    "type": "integer",
                                    "description": "Desired length of the summary."
                                }
                            },
                            "required": ["text"],
                            "additionalProperties": False
                            }
                        },
                        "additionalProperties": False
                        }
                    ]
                    }
                }
    
    message = ToolCatalogMessage(
        value=json.dumps(valid_schema)
    )
    assert message.value == json.dumps(valid_schema)

def test_invalid_json():
    with pytest.raises(Exception):
        ToolCatalogMessage(
            from_="tool_catalog",
            value="{ invalid json }"
        )

def test_invalid_json_schema():
    # Schema with invalid type
    invalid_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "invalid_type"}  # Invalid type
        }
    }
    
    with pytest.raises(Exception):
        ToolCatalogMessage(
            from_="tool_catalog",
            value=json.dumps(invalid_schema)
        )

def test_non_string_value():
    with pytest.raises(Exception):
        ToolCatalogMessage(
            from_="tool_catalog",
            value={"type": "object"}  # Dictionary instead of string
        )


def test_empty_value():
    with pytest.raises(ValueError):
        ToolCatalogMessage(
            from_="tool_catalog",
            value=""
        )

def test_complex_valid_schema():
    complex_schema = {
        "type": "object",
        "required": ["name", "parameters"],
        "properties": {
            "name": {"type": "string"},
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1}
                },
                "required": ["query"]
            }
        }
    }
    
    message = ToolCatalogMessage(
        value=json.dumps(complex_schema)
    )
    assert message.value == json.dumps(complex_schema)

def test_invalid_from_field():
    with pytest.raises(ValueError):
        ToolCatalogMessage(
            from_="user",  # Invalid value for from_
            value=json.dumps({"type": "object"})
        )

def test_valid_tool_response():
    valid_json = '''[
        {"name": "tool1", "content": "result1"},
        {"name": "tool2", "content": "result2"}
    ]'''
    message = ToolResponseMessage(value=valid_json)
    assert message.value == valid_json
    assert message.from_ == "tool"

def test_valid_tool_response_with_tool_response_from():
    valid_json = '[{"name": "tool1", "content": "result1"}]'
    message = ToolResponseMessage(value=valid_json)
    assert message.value == valid_json
    assert message.from_ == "tool"  # Should default to "tool"

def test_invalid_json():
    invalid_json = 'not json'
    with pytest.raises(Exception):
        ToolResponseMessage(value=invalid_json)

def test_non_array_json():
    non_array = '{"name": "tool1", "content": "result1"}'
    with pytest.raises(Exception):
        ToolResponseMessage(value=non_array)

def test_missing_required_fields():
    missing_fields = '[{"name": "tool1"}]'  # missing content
    with pytest.raises(Exception):
        ToolResponseMessage(value=missing_fields)

def test_extra_fields():
    extra_fields = '[{"name": "tool1", "content": "result1", "extra": "field"}]'
    with pytest.raises(Exception):
        ToolResponseMessage(value=extra_fields)

def test_non_string_value():
    with pytest.raises(Exception):
        ToolResponseMessage(value=123)

def test_empty_array():
    empty_array = '[]'
    message = ToolResponseMessage(value=empty_array)
    assert message.value == empty_array

def test_invalid_from_field():
    valid_json = '[{"name": "tool1", "content": "result1"}]'
    with pytest.raises(ValueError):
        ToolResponseMessage(from_="invalid", value=valid_json)