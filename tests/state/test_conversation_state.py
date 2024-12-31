import pytest
from pydantic import ValidationError
from narrative_llm_tools.tools.json_schema_tools import JsonSchemaTools, ToolSchema, Tool
from narrative_llm_tools.state.conversation_state import (
    ConversationMessage,
    ConversationState,
    ConversationStatus,
)

# Mock JsonSchemaTools for testing
class MockJsonSchemaTools(JsonSchemaTools):
    def __init__(self, tools):
        super().__init__(
            type="array",
            items={
                "anyOf": [ToolSchema.model_validate(tool) for tool in tools]
            }
        )

    @staticmethod
    def only_user_response_tool():
        return JsonSchemaTools.only_user_response_tool()

@pytest.fixture
def mock_tools_catalog():
    return MockJsonSchemaTools([
        {
            "type": "object",
            "required": ["name", "parameters"],
            "properties": {
                "name": {"type": "string", "enum": ["rest_api_tool_1"]},
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            },
            "restApi": {
                "url": "http://example.com/users/",
                "method": "GET",
                "parameter_location": "query"
            },
        },
        {
            "type": "object",
            "required": ["name", "parameters"],
            "properties": {
                "name": {"type": "string", "enum": ["non_rest_tool"]},
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            },
        },
    ])

@pytest.fixture
def sample_conversation_state(mock_tools_catalog):
    return ConversationState(
        raw_messages=[],
        max_tool_rounds=5,
        tool_choice="required",
        tools_catalog=mock_tools_catalog,
        pipeline_params={},
        status=ConversationStatus.RUNNING,
    )

# Test ConversationMessage validation
def test_conversation_message_validation():
    valid_message = ConversationMessage(role="user", content="Hello")
    assert valid_message.role == "user"

    with pytest.raises(ValidationError):
        ConversationMessage(role="invalid_role", content="Hello")

# Test ConversationState initialization
def test_conversation_state_initialization(sample_conversation_state):
    assert sample_conversation_state.max_tool_rounds == 5
    assert sample_conversation_state.status == ConversationStatus.RUNNING

# Test adding a message
def test_add_message(sample_conversation_state):
    message = ConversationMessage(role="tool_calls", content="[]")
    sample_conversation_state.add_message(message)
    assert sample_conversation_state.raw_messages == [message]

# Test invalid status transitions
def test_invalid_status_transition(sample_conversation_state):
    with pytest.raises(ValueError):
        sample_conversation_state.transition_to(ConversationStatus.COMPLETED)
        sample_conversation_state.transition_to(ConversationStatus.RUNNING)

# Test valid status transitions
def test_valid_status_transitions(sample_conversation_state):
    sample_conversation_state.transition_to(ConversationStatus.WRAP_THINGS_UP)
    assert sample_conversation_state.status == ConversationStatus.WRAP_THINGS_UP

# Test message injection
def test_messages_property(sample_conversation_state):
    system_message = ConversationMessage(role="system", content="You are a helpful assistant.")
    catalog_message = ConversationMessage(
        role="tool_catalog", content="{}"
    )
    sample_conversation_state.raw_messages = [system_message]
    messages = sample_conversation_state.messages
    assert len(messages) == 2
    assert messages[0] == system_message
    assert messages[1].role == "tool_catalog"

# Test tool removal
def test_remove_tool(sample_conversation_state):
    sample_conversation_state.remove_tool("rest_api_tool_1")
    assert "rest_api_tool_1" not in [tool.properties.name.enum[0] for tool in sample_conversation_state.tools_catalog.items.anyOf]

# Test exceeded tool rounds
def test_exceeded_tool_rounds(sample_conversation_state):
    sample_conversation_state.raw_messages = [
        ConversationMessage(role="tool_calls", content="[]") for _ in range(5)
    ]
    assert sample_conversation_state.exceeded_tool_rounds()

# Test from_api_request
def test_from_api_request():
    request_data = {
        "inputs": [
            {"role": "user", "content": "Hello"},
            {"role": "tool_response", "content": "response"},
        ],
        "tool_choice": "auto",
        "max_tool_rounds": 3,
        "tools": {
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
                },
    }
    state = ConversationState.from_api_request(request_data)
    assert state.max_tool_rounds == 3
    assert state.tool_choice == "auto"
    assert len(state.raw_messages) == 2
    assert state.raw_messages[0].role == "user"

# Test parse_tool_calls_content
def test_parse_tool_calls_content(sample_conversation_state):
    content = '[{"name": "rest_api_tool_1", "parameters": {"param1": "value1"}}]'
    tool_calls = sample_conversation_state.parse_tool_calls_content(content)
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "rest_api_tool_1"

# Test invalid JSON parsing
def test_invalid_json_parsing(sample_conversation_state):
    with pytest.raises(ValueError):
        sample_conversation_state.parse_tool_calls_content("invalid json")

def test_has_rest_api_tools():
    # Create a ConversationState with a REST API tool
    rest_api_tool = {
                    "type": "array",
                    "items": {
                    "anyOf": [
                        {
                        "type": "object",
                        "restApi": {
                            "url": "http://example.com/users/{user_id}",
                            "method": "GET",
                            "parameter_location": "query"
                        },
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
                            },
                        },
                        "additionalProperties": False
                        }
                    ]
                    }
                }
    
    state = ConversationState(
        raw_messages=[],
        pipeline_params={},
        tools_catalog=JsonSchemaTools.model_validate(rest_api_tool)
    )

    # Test with content that includes a REST API tool call
    content_with_rest = '[{"name": "my_generic_tool", "parameters": {"param_1": "London", "param_2": 123}}]'
    assert state._has_rest_api_tools(content_with_rest) is True

    # Test with content that doesn't include a REST API tool call
    content_without_rest = '[{"name": "text_summarizer", "parameters": {"text": "Hello"}}]'
    assert state._has_rest_api_tools(content_without_rest) is False

    # Test with empty tool call
    content_empty = '[]'
    assert state._has_rest_api_tools(content_empty) is False

    # Test with invalid JSON should raise ValueError
    with pytest.raises(ValueError):
        state._has_rest_api_tools("invalid json")

def test_handle_tool_response():
    """Test handling of tool response messages and state transitions."""
    # Initial setup with a conversation in WAITING_TOOL_RESPONSE state
    state = ConversationState(
        raw_messages=[
            ConversationMessage(role="user", content="Hello"),
            ConversationMessage(role="tool_calls", content='[{"name": "rest_api_tool", "args": {}}]')
        ],
        pipeline_params={},
        status=ConversationStatus.WAITING_TOOL_RESPONSE
    )

    # Add a tool response
    tool_response = ConversationMessage(
        role="tool_response",
        content='{"result": "some data"}'
    )
    state.add_message(tool_response)

    # Verify the message was added
    assert state.raw_messages[-1] == tool_response
    # Verify state transitioned to RUNNING
    assert state.status == ConversationStatus.RUNNING

def test_handle_tool_response_last_round():
    """Test handling of tool response when it's the last allowed round."""
    # Setup conversation with max_tool_rounds=2 and one tool call already made
    state = ConversationState(
        raw_messages=[
            ConversationMessage(role="user", content="Hello"),
            ConversationMessage(role="tool_calls", content='[{"name": "rest_api_tool", "args": {}}]')
        ],
        max_tool_rounds=2,
        pipeline_params={},
        status=ConversationStatus.WAITING_TOOL_RESPONSE
    )

    # Add a tool response
    tool_response = ConversationMessage(
        role="tool_response",
        content='{"result": "some data"}'
    )
    state.add_message(tool_response)

    # Verify the message was added
    assert state.raw_messages[-1] == tool_response
    # Verify state transitioned to WRAP_THINGS_UP since it's the last round
    assert state.status == ConversationStatus.WRAP_THINGS_UP

from unittest import TestCase
from narrative_llm_tools.state.conversation_state import ConversationState, ConversationMessage, ConversationStatus
import json

class TestHandleToolCall(TestCase):
    def setUp(self):
        # Basic conversation state setup with default values
        self.state = ConversationState(
            raw_messages=[],
            pipeline_params={},
            max_tool_rounds=3
        )

    def test_response_to_user_completes_conversation(self):
        # Setup a tool call that responds to user (non-REST API tool)
        message = ConversationMessage(
            role="tool_calls",
            content=json.dumps([{"name": "respond_to_user", "parameters": {"response": "Hello"}}])
        )
        
        self.state._handle_tool_call(message)
        
        self.assertEqual(self.state.status, ConversationStatus.COMPLETED)
        self.assertEqual(len(self.state.raw_messages), 1)

    def test_invalid_tool_call_content(self):
        # Test with invalid JSON content
        message = ConversationMessage(
            role="tool_calls",
            content="invalid json"
        )
        
        with self.assertRaises(ValueError):
            self.state._handle_tool_call(message)

@pytest.fixture(autouse=True)
def reset_conversation_state(mock_tools_catalog):
    """Reset the conversation state before each test"""
    return ConversationState(
        raw_messages=[],
        max_tool_rounds=5,
        tool_choice="required",
        tools_catalog=mock_tools_catalog,
        pipeline_params={},
        status=ConversationStatus.RUNNING,
    )

# Update the test to use the fixture
def test_multiple_tool_calls_in_one_message(reset_conversation_state):
    state = reset_conversation_state
    message = ConversationMessage(
        role="tool_calls",
        content=json.dumps([
            {"name": "rest_api_tool_1", "parameters": {}},
            {"name": "rest_api_tool_1", "parameters": {}}
        ])
    )
    
    state._handle_tool_call(message)
    
    assert len(state.raw_messages) == 1
    assert state.status == ConversationStatus.WAITING_TOOL_RESPONSE