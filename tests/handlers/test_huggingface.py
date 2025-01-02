import json
import logging
import pytest
from unittest.mock import MagicMock, patch

from narrative_llm_tools.state.conversation_state import (
    ConversationMessage,
    ConversationState,
    ConversationStatus,
)
from narrative_llm_tools.tools import Tool
from narrative_llm_tools.handlers.huggingface import (
    EndpointHandler,
    HandlerResponse,
    PipelineConfigurationError,
    PipelineExecutionError,
    ModelOutputError,
)


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """
    Return a mock pipeline object that behaves like a text-generation pipeline.
    """
    pipeline_mock = MagicMock()
    # The pipeline will return a list of dictionaries with "generated_text"
    pipeline_mock.return_value = [{"generated_text": '[{"name": "test_tool", "parameters": {}}]'}]
    return pipeline_mock


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """
    Return a mock tokenizer that has `apply_chat_template` method.
    """
    tokenizer_mock = MagicMock()
    tokenizer_mock.apply_chat_template.return_value = "formatted conversation"
    return tokenizer_mock


@pytest.fixture
def endpoint_handler(mock_pipeline: MagicMock, mock_tokenizer: MagicMock) -> EndpointHandler:
    """
    Fixture for creating an EndpointHandler with mock dependencies.
    """
    handler = EndpointHandler(path="gpt2")
    
    # Overwrite the pipeline created in `_create_pipeline()` with our mock.
    handler.pipeline = mock_pipeline
    # Overwrite the tokenizer with our mock.
    handler.pipeline.tokenizer = mock_tokenizer
    return handler


def test_pipeline_creation_failure():
    """
    Test if `EndpointHandler` raises a PipelineConfigurationError
    when pipeline creation fails.
    """
    with patch("narrative_llm_tools.handlers.huggingface.pipeline", side_effect=Exception("Mocked pipeline failure")):
        with pytest.raises(PipelineConfigurationError) as exc_info:
            EndpointHandler(path="fake_thing_that_should_fail")
        assert "Failed to create pipeline: Mocked pipeline failure" in str(exc_info.value)


def test_endpoint_handler_call_basic(endpoint_handler: EndpointHandler):
    """
    Test the basic call flow of the EndpointHandler to ensure it returns
    the expected list of tool calls from the pipeline.
    """
    # Prepare input data
    data = {
        "inputs": [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "User message."},
        ]
    }

    result = endpoint_handler(data)

    # Since our mock pipeline returns the JSON with "test_tool",
    # we expect the final output to be a list with one dict for "test_tool".
    assert isinstance(result["tool_calls"], list)
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "test_tool"
    assert result["tool_calls"][0]["parameters"] == {}


def test_endpoint_handler_call_with_logging(endpoint_handler: EndpointHandler, caplog):
    """
    Test setting log_level in the data dictionary to ensure logging is changed.
    """
    data = {
        "log_level": "DEBUG",
        "inputs": [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "User message."},
        ],
    }

    with caplog.at_level(logging.DEBUG):
        endpoint_handler(data)

    # Check that debug logs were generated.
    debug_logs = [rec.message for rec in caplog.records if rec.levelno == logging.DEBUG]
    assert len(debug_logs) > 0  # We expect some debug messages from the conversation flow.


def test_endpoint_handler_invalid_json_output(endpoint_handler: EndpointHandler):
    """
    Test the scenario when pipeline returns invalid JSON for 'generated_text'.
    """
    # Overwrite the mock pipeline's return to give invalid JSON
    endpoint_handler.pipeline.return_value = [{"generated_text": '{invalid_json'}]

    data = {
        "inputs": [
            {"role": "user", "content": "Hello!"},
        ]
    }

    with pytest.raises(ModelOutputError) as exc_info:
        endpoint_handler(data)
    assert "Failed to parse model output as JSON" in str(exc_info.value)


def test_endpoint_handler_no_generated_text(endpoint_handler: EndpointHandler):
    """
    Test the scenario when pipeline's return lacks 'generated_text' key.
    """
    endpoint_handler.pipeline.return_value = [{}]  # no 'generated_text'

    data = {
        "inputs": [
            {"role": "user", "content": "Hello!"},
        ]
    }

    with pytest.raises(ModelOutputError) as exc_info:
        endpoint_handler(data)
    assert "No generated_text found in the model output" in str(exc_info.value)


def test_endpoint_handler_empty_output(endpoint_handler: EndpointHandler):
    """
    Test the scenario when pipeline returns an empty list,
    which might happen if generation fails or returns nothing.
    """
    endpoint_handler.pipeline.return_value = []
    data = {
        "inputs": [
            {"role": "user", "content": "Hello!"},
        ]
    }

    result = endpoint_handler(data)
    # Expect an empty list of tool calls
    assert result == HandlerResponse(tool_calls=[], warnings=None).model_dump()


def test_endpoint_handler_pipeline_execution_error(endpoint_handler: EndpointHandler):
    """
    Test if a PipelineExecutionError is raised when pipeline call fails at runtime.
    """
    endpoint_handler.pipeline.side_effect = Exception("Pipeline runtime error")

    data = {
        "inputs": [{"role": "user", "content": "Trigger error."}],
    }

    with pytest.raises(PipelineExecutionError) as exc_info:
        endpoint_handler(data)
    assert "Failed to generate prediction: Pipeline runtime error" in str(exc_info.value)


def test_endpoint_handler_process_conversation_turn(endpoint_handler: EndpointHandler):
    """
    Test the internal `_process_conversation_turn` logic by injecting a
    conversation state in RUNNING status and verifying it adds the 'tool_calls' message.
    """
    state = ConversationState(
        raw_messages=[
            ConversationMessage(role="user", content="Hello!"),
        ],
        status=ConversationStatus.RUNNING,
        pipeline_params={},
    )

    endpoint_handler._process_conversation_turn(state)

    # The conversation turn should add a "tool_calls" message
    # We expect a single "tool_calls" entry with our mocked pipeline response
    assert len(state.messages) == 4
    assert state.messages[-1].role == "tool_calls"

    # The content should be a list with one tool call matching our mock return
    tool_calls = json.loads(state.messages[-1].content)
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "test_tool"


def test_endpoint_handler_execute_tool_calls_no_catalog(endpoint_handler: EndpointHandler, caplog):
    """
    Test `_execute_tool_calls` with no rest API catalog in the conversation state
    (e.g. no tools were provided).
    """
    state = ConversationState(raw_messages=[], status=ConversationStatus.RUNNING, pipeline_params={})
    tool_calls = [Tool(name="some_tool", parameters={"param": "value"})]

    with caplog.at_level(logging.INFO):
        endpoint_handler._execute_tool_calls(tool_calls, state)

    info_logs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("No rest API catalog is available" in msg for msg in info_logs)

    assert len(state.messages) == 2


def test_endpoint_handler_format_model_output(endpoint_handler: EndpointHandler):
    """
    Test parsing valid JSON from the pipeline's model output.
    """
    mock_output = [{"generated_text": '[{"name": "tool1", "parameters": {"p": 1}}]'}]
    tools = endpoint_handler._format_model_output(mock_output)
    assert len(tools) == 1
    assert tools[0].name == "tool1"
    assert tools[0].parameters == {"p": 1}


def test_endpoint_handler_format_conversation(endpoint_handler: EndpointHandler, mock_tokenizer: MagicMock):
    """
    Test that `_format_conversation` calls the tokenizer's `apply_chat_template` properly.
    """
    state = ConversationState(
        raw_messages=[ConversationMessage(role="user", content="Hello!")],
        status=ConversationStatus.RUNNING,
        pipeline_params={},
    )
    prompt = endpoint_handler._format_conversation(state)

    mock_tokenizer.apply_chat_template.assert_called_once()
    assert prompt == "formatted conversation"
