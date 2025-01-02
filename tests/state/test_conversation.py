import json
import pytest
from narrative_llm_tools.state.conversation import (
    Conversation,
    validate_conversation_object,
    validate_line,
    extract_enumerated_names,
    generate_tool_response_json_schema,
    validate_tool_catalog_schema,
    validate_tool_response_matching,
    validate_tool_response_structure,
)

# Test fixtures
@pytest.fixture
def valid_tool_catalog():
    return {
        "type": "array",
        "items": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": ["tool1"]},
                        "parameters": {"type": "object"}
                    },
                    "required": ["name", "parameters"]
                }
            ]
        }
    }

@pytest.fixture
def valid_conversation():
    return {
        "conversations": [
            {"from": "system", "value": "System message"},
            {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""},
            {"from": "user", "value": "User message"},
            {"from": "assistant", "value": """[{"name": "tool1", "parameters": {}}]"""},
        ]
    }

# Test Conversation validation
def test_valid_conversation(valid_conversation):
    conv = Conversation(**valid_conversation)
    assert len(conv.conversations) == 4

def test_conversation_too_short():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": []})

def test_invalid_conversation_message():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "user", "value": "User message", "too_much": "info"}]})

    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"role": "user", "value": "User message"}]})

def test_empty_content():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "user", "value": ""}]})


def test_multiple_system_messages():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "system", "value": "System message"}, {"from": "user", "value": "User message"}]})

def test_system_message_not_first():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "user", "value": "User message"}, {"from": "system", "value": "System message"}, {"from": "user", "value": "User message"}]})

def test_multiple_tool_catalog_messages():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "user", "value": "User message"}]})

def test_tool_catalog_not_second():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "user", "value": "User message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "user", "value": "User message"}]})

def test_tool_catalog_not_first_without_system():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "user", "value": "User message"}]})

def two_consecutive_user_messages():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "user", "value": "User message"}, {"from": "user", "value": "User message"}]})

def assistant_message_not_array():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "assistant", "value": "Assistant message"}]})

def tool_response_not_array():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "assistant", "value": """[{"name": "tool1", "parameters": {}}]"""}, {"from": "tool_response", "value": "Tool response message"}]})

def tool_response_not_after_assistant():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "user", "value": "User message"}, {"from": "tool_response", "value": '[{"name": "tool1", "parameters": "{"test": "test"}'}]})

def tool_not_after_assistant():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "user", "value": "User message"}, {"from": "tool", "value": '[{"name": "tool1", "parameters": "{"test": "test"}'}]})

def test_tool_call_not_in_tool_catalog():
    with pytest.raises(ValueError):
        Conversation(**{"conversations": [{"from": "system", "value": "System message"}, {"from": "tool_catalog", "value": """{"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}}"""}, {"from": "user", "value": "User message"}, {"from": "tool_call", "value": '[{"name": "tool1", "parameters": "{"test": "test"}'}]})

def test_validate_line_empty():
    """Test validation of empty lines"""
    result = validate_line(("   ", 1))
    assert result.line_number == 1
    assert result.errors == ["Line 1: Empty line is not allowed."]


def test_validate_line_invalid_json():
    """Test validation of invalid JSON"""
    result = validate_line(("not json", 2))
    assert result.line_number == 2
    assert any("Invalid JSON" in error for error in result.errors)


def test_validate_line_minimal_valid_conversation():
    """Test validation of a minimal valid conversation"""
    valid_conversation = {
        "conversations": [
            {"from": "tool_catalog", "value": json.dumps({
                "type": "array",
                "items": {"anyOf": [{"type": "object", "properties": {"name": {"type": "string", "enum": ["tool1"]}, "parameters": {"type": "object"}}, "required": ["name", "parameters"]}]}
            })},
            {"from": "user", "value": "hello"},
            {"from": "assistant", "value": "[]"}
        ]
    }
    result = validate_line((json.dumps(valid_conversation), 1))
    assert result.line_number == 1
    assert not result.errors


def test_validate_line_invalid_conversation():
    """Test validation of an invalid conversation structure"""
    invalid_conversation = {
        "conversations": [
            {"from": "user", "value": "hello"}  # Missing required messages
        ]
    }
    result = validate_line((json.dumps(invalid_conversation), 3))
    assert result.line_number == 3
    assert any("must have at least 3 messages" in error for error in result.errors)


def test_validate_line_missing_required_fields():
    """Test validation of conversation with missing required fields"""
    invalid_conversation = {
        "conversations": [
            {"value": "test"}  # Missing 'from' field
        ]
    }
    result = validate_line((json.dumps(invalid_conversation), 4))
    assert result.line_number == 4
    assert len(result.errors) > 0


@pytest.mark.parametrize("input_data,expected_error_pattern", [
    (
        {"wrong_key": []},
        "conversations"  # Error should mention missing 'conversations' field
    ),
    (
        {"conversations": "not_a_list"},
        "list"  # Error should mention that conversations must be a list
    ),
])
def test_validate_line_invalid_structures(input_data, expected_error_pattern):
    """Test validation of various invalid conversation structures"""
    result = validate_line((json.dumps(input_data), 5))
    assert result.line_number == 5
    assert any(expected_error_pattern in error.lower() for error in result.errors)


def test_valid_tool_catalog_schema():
    valid_schema = '''
    {
        "type": "array",
        "items": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["tool1"]
                        },
                        "parameters": {
                            "type": "object"
                        }
                    }
                }
            ]
        }
    }
    '''
    schema, errors = validate_tool_catalog_schema(valid_schema)
    assert errors == []
    assert schema is not None
    assert schema["type"] == "array"

def test_invalid_json():
    invalid_json = '''
    {
        "type": "array",
        missing_quotes: "value"
    }
    '''
    schema, errors = validate_tool_catalog_schema(invalid_json)
    assert schema is None
    assert len(errors) == 1
    assert "must be valid JSON" in errors[0]

def test_invalid_schema():
    invalid_schema = '''
    {
        "type": "invalid_type",
        "items": {}
    }
    '''
    schema, errors = validate_tool_catalog_schema(invalid_schema)
    assert schema is None
    assert len(errors) == 1

def test_empty_string():
    schema, errors = validate_tool_catalog_schema("")
    assert schema is None
    assert len(errors) == 1
    assert "must be valid JSON" in errors[0]

def test_lru_cache_behavior():
    # Test that the cache is working
    valid_schema = '{"type": "array", "items": {}}'
    
    # First call
    result1, errors1 = validate_tool_catalog_schema(valid_schema)
    
    # Second call (should use cache)
    result2, errors2 = validate_tool_catalog_schema(valid_schema)
    
    assert result1 == result2
    assert errors1 == errors2
    assert errors1 == []

def test_extract_enumerated_names_basic():
    schema = {
        "items": {
            "anyOf": [
                {
                    "properties": {
                        "name": {
                            "enum": ["tool1", "tool2"]
                        }
                    }
                }
            ]
        }
    }
    assert extract_enumerated_names(schema) == {"tool1", "tool2"}

def test_extract_enumerated_names_multiple_schemas():
    schema = {
        "items": {
            "anyOf": [
                {
                    "properties": {
                        "name": {
                            "enum": ["tool1"]
                        }
                    }
                },
                {
                    "properties": {
                        "name": {
                            "enum": ["tool2"]
                        }
                    }
                }
            ]
        }
    }
    assert extract_enumerated_names(schema) == {"tool1", "tool2"}

def test_extract_enumerated_names_empty_schema():
    schema = {
        "items": {
            "anyOf": []
        }
    }
    assert extract_enumerated_names(schema) == set()

def test_extract_enumerated_names_missing_enum():
    schema = {
        "items": {
            "anyOf": [
                {
                    "properties": {
                        "name": {
                            "type": "string"
                        }
                    }
                }
            ]
        }
    }
    assert extract_enumerated_names(schema) == set()

def test_extract_enumerated_names_invalid_schema():
    invalid_schemas = [
        {},  # Empty schema
        {"items": {}},  # Missing anyOf
        None,  # None schema
        {"items": {"anyOf": [{"properties": {}}]}},  # Missing name property
    ]
    
    for schema in invalid_schemas:
        assert extract_enumerated_names(schema) == set()

def test_validate_tool_response_structure():
    # Valid case
    valid_response = {"name": "test_tool", "content": "test result"}
    assert validate_tool_response_structure(valid_response, 0) == []

    # Not a dict
    assert validate_tool_response_structure(["not", "a", "dict"], 0) == [
        "Response at index 0 must be an object"
    ]

    # Missing fields
    missing_content = {"name": "test_tool"}
    assert validate_tool_response_structure(missing_content, 1) == [
        "Response at index 1 must have exactly 'name' and 'content' fields"
    ]

    # Extra fields
    extra_field = {"name": "test_tool", "content": "result", "extra": "field"}
    assert validate_tool_response_structure(extra_field, 2) == [
        "Response at index 2 must have exactly 'name' and 'content' fields"
    ]

    # Wrong types
    wrong_types = {"name": 123, "content": ["not", "a", "string"]}
    assert validate_tool_response_structure(wrong_types, 3) == [
        "Response at index 3: 'name' and 'content' must be strings"
    ]

def test_validate_tool_response_matching():
    # Valid case
    response = {"name": "test_tool", "content": "result"}
    prev_call = {"name": "test_tool", "parameters": {"param": "value"}}
    assert validate_tool_response_matching(response, prev_call, 0) == []

    # Mismatched names
    response_mismatch = {"name": "wrong_tool", "content": "result"}
    assert validate_tool_response_matching(response_mismatch, prev_call, 0) == [
        "Response at index 0: name 'wrong_tool' does not match tool call name 'test_tool'"
    ]

    # Missing name in prev_call
    prev_call_no_name = {"parameters": {"param": "value"}}
    assert validate_tool_response_matching(response, prev_call_no_name, 0) == [
        "Response at index 0: name 'test_tool' does not match tool call name 'None'"
    ]