import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer

from narrative_llm_tools.tools.json_schema_tools import JsonSchemaTools
from narrative_llm_tools.utils.format_enforcer import get_format_enforcer


class SimpleSchema(BaseModel):
    name: str
    age: int

@pytest.fixture
def test_schema():
    return {
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

def test_get_format_enforcer_basic(test_schema):
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    schema = JsonSchemaTools.model_validate(test_schema)

    # Execute
    enforcer = get_format_enforcer(tokenizer, schema)

    # Assert
    assert callable(enforcer)

def test_format_enforcer_caching(test_schema):
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    schema = JsonSchemaTools.model_validate(test_schema)

    # Execute
    enforcer1 = get_format_enforcer(tokenizer, schema)
    enforcer2 = get_format_enforcer(tokenizer, schema)

    # Assert
    assert enforcer1 is enforcer2  # Should return cached instance

def test_format_enforcer_different_schemas(test_schema):
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    schema1 = JsonSchemaTools.model_validate(test_schema)
    schema2 = JsonSchemaTools.model_validate({
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
                        }]}})

    # Execute
    enforcer1 = get_format_enforcer(tokenizer, schema1)
    enforcer2 = get_format_enforcer(tokenizer, schema2)

    # Assert
    assert enforcer1 is not enforcer2  # Should be different instances

def test_format_enforcer_invalid_schema():
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    with pytest.raises(Exception):
        invalid_schema = JsonSchemaTools.model_validate({
            "type": "invalid_type",  # Invalid schema type
            "properties": {}
        })

        get_format_enforcer(tokenizer, invalid_schema)

def test_format_enforcer_functionality(test_schema):
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    schema = JsonSchemaTools.model_validate(test_schema)

    enforcer = get_format_enforcer(tokenizer, schema)

    # Create a simple input tensor
    input_ids = tokenizer.encode('{"', return_tensors='pt')[0]

    # Execute
    allowed_tokens = enforcer(0, input_ids)

    # Assert
    assert isinstance(allowed_tokens, list)
    assert len(allowed_tokens) > 0
