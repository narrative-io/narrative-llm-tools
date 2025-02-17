import json
import pytest
from unittest.mock import patch

# Directly import the function under test
from narrative_llm_tools.reward_functions.cot_with_tool_call.format import (
    tool_calls_validity_reward
)

# If you have the same JSON schema in a fixture, you can do so here:
mock_json_schema = {
    "title": "Math Function Call Array",
    "description": "A list of simple math function definitions to execute",
    "type": "array",
    "items": {
        "title": "Function Definition",
        "description": "Define either an add or subtract operation",
        "type": "object",
        "properties": {
            "name": {
                "title": "Function Name",
                "description": "The identifier for the function to be invoked",
                "type": "string",
                "enum": ["add", "subtract"]
            },
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "title": "First Number",
                        "description": "The first number in the operation",
                        "type": "number"
                    },
                    "b": {
                        "title": "Second Number",
                        "description": "The second number in the operation",
                        "type": "number"
                    }
                },
                "required": ["a", "b"]
            }
        },
        "required": ["name", "parameters"]
    }
}

@pytest.fixture
def tool_catalog_prompts():
    return [{
        "role": "tool_catalog",
        "content": json.dumps(mock_json_schema)
    }]

@pytest.fixture
def invalid_tool_catalog_prompts():
    return [{
        "role": "tool_catalog",
        "content": "not a valid json"
    }]

def test_valid_tool_call_completion(tool_catalog_prompts):
    completion = [{
        "role": "assistant",
        "content": """<|start_thought|>I should add these numbers<|end_thought|>
        <|tool_calls|>[{"name": "add", "parameters": {"a": 5, "b": 3}}]
        <|eot_id|>"""
    }]
    assert tool_calls_validity_reward(completions=[completion], prompts=[tool_catalog_prompts]) == [1.0]

def test_invalid_tool_call_completion(tool_catalog_prompts):
    completion = [{
        "role": "assistant",
        "content": """<|start_thought|>I should add these numbers<|end_thought|>
        <|tool_calls|>[{}]
        <|eot_id|>"""
    }]  
    assert tool_calls_validity_reward(completions=[completion], prompts=[tool_catalog_prompts]) == [0.0]

def test_no_tool_calls_completion(tool_catalog_prompts):
    completion = [{
        "role": "assistant",
        "content": """<|start_thought|>I should add these numbers<|end_thought|>
        <|eot_id|>"""
    }]
    assert tool_calls_validity_reward(completions=[completion], prompts=[tool_catalog_prompts]) == [0.0]

def test_malformed_schema_fallback(invalid_tool_catalog_prompts):
    completion = [{
        "role": "assistant",
        "content": """<|start_thought|>I should add these numbers<|end_thought|>
        <|tool_calls|>[{"name": "add", "parameters": {"a": 5, "b": 3}}]
        <|eot_id|>"""
    }]
    assert tool_calls_validity_reward(completions=[completion], prompts=[invalid_tool_catalog_prompts]) == [0.0]

def test_malformed_tool_call_fallback(tool_catalog_prompts):
    completion = [{
        "role": "assistant",
        "content": """<|start_thought|>I should add these numbers<|end_thought|>
        <|tool_calls|>invalid-json
        <|eot_id|>"""
    }]
    assert tool_calls_validity_reward(completions=[completion], prompts=[tool_catalog_prompts]) == [0.0]
