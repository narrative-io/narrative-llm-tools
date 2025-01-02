import json
import logging
from collections.abc import Hashable
from typing import Any, Literal, Protocol

from pydantic import BaseModel
from torch import Tensor
from transformers import pipeline  # type: ignore

from narrative_llm_tools.rest_api_client.types import RestApiResponse
from narrative_llm_tools.state.conversation_state import (
    ConversationMessage,
    ConversationState,
    ConversationStatus,
    ToolResponse,
)
from narrative_llm_tools.tools import Tool
from narrative_llm_tools.utils.format_enforcer import get_format_enforcer

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class HandlerResponse(BaseModel):
    """Response from the handler."""

    tool_calls: list[dict[str, Any]]
    warnings: list[str] | None


class ModelConfig(BaseModel):
    """Configuration for the model and tokenizer."""

    pipeline_type: Literal["text-generation"] = "text-generation"
    path: str
    max_new_tokens: int = 4096
    device_map: str = "auto"
    begin_token: str = "<|begin_of_text|>"
    eot_token: str = "<|eot_id|>"


class FormatEnforcer(Protocol):
    def __call__(self, batch_id: int, sent: Tensor) -> list[int]: ...


class Tokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        force_tool_calls: bool,
        tokenize: bool,
        bos_token: str,
        eot_token: str,
    ) -> str: ...


class Pipeline(Protocol):
    tokenizer: Tokenizer

    def __call__(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...


class FreezeError(Exception):
    """Exception raised when an object cannot be converted to a hashable form."""

    pass


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class EndpointError(Exception):
    """Base exception class for endpoint-related errors."""

    pass


class ToolConfigurationError(EndpointError):
    """Raised when there's an error in tool configuration or validation."""

    pass


class PipelineConfigurationError(EndpointError):
    """Raised when there's an error in pipeline configuration."""

    pass


class PipelineExecutionError(EndpointError):
    """Raised when the model pipeline fails to execute properly."""

    pass


class ModelOutputError(EndpointError):
    """Raised when there's an error parsing or processing model output."""

    pass


class AuthenticationError(EndpointError):
    """Raised when there's an error authenticating with an API."""

    pass


class EndpointHandler:
    def __init__(self, path: str = "") -> None:
        """
        Initialize the EndpointHandler with the provided model path.

        Args:
            path (str, optional): The path or identifier of the model. Defaults to "".
        """
        self.config = ModelConfig(path=path)

        try:
            self.pipeline: Pipeline = self._create_pipeline()
        except Exception as e:
            logger.error(f"Failed to create pipeline: {str(e)}")
            raise PipelineConfigurationError(f"Failed to create pipeline: {str(e)}") from e

        self.format_enforcers: dict[Hashable, FormatEnforcer] = {}

    def _create_pipeline(self) -> Pipeline:
        """Create and configure the model pipeline."""
        pipe = pipeline(
            self.config.pipeline_type,
            model=self.config.path,
            max_new_tokens=self.config.max_new_tokens,
            device_map=self.config.device_map,
        )
        return pipe  # type: ignore

    def __call__(self, data: dict[str, Any]) -> HandlerResponse:
        """
        Generate model output given a conversation and optional tools/parameters.

        Args:
            data (dict): Expected keys:
                - "inputs" (list): A list of message dictionaries. Each message has:
                    {
                        "role": "system"|"user",
                        "content": "some text"
                    }
                - "tools" (optional, dict): A JSON schema defining the tool set.
                - "log_level" (optional, str): Logging level ('DEBUG',
                                        'INFO',
                                        'WARNING',
                                        'ERROR',
                                        'CRITICAL')

                **Example Schema:**
                ```json
                {
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
                            "additionalProperties": false
                            }
                        },
                        "additionalProperties": false
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
                            "additionalProperties": false
                            }
                        },
                        "additionalProperties": false
                        }
                    ]
                    }
                }
                ```

                **Example Data Matching the Above Schema:**
                ```json
                [
                    {
                    "name": "my_generic_tool",
                    "parameters": {
                        "param1": "some input",
                        "param2": 42
                    }
                    },
                    {
                    "name": "text_summarizer",
                    "parameters": {
                        "text": "This is a long article about JSON schemas.",
                        "summary_length": 3
                    }
                    }
                ]
                ```

                Additional keys in `data` can be used as pipeline
                  parameters (e.g., temperature, top_k).

        Returns:
            list: A list of prediction dictionaries from the pipeline.

        Example:
            ```python
            handler = EndpointHandler(path="path/to/model")
            response = handler({
                "inputs": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Summarize the following: 'Lorem ipsum...'"}
                ],
                "tools": [
                    {
                        "name": "text_summarizer",
                        "parameters": {
                            "text": "Lorem ipsum dolor sit amet...",
                            "summary_length": 3
                        }
                    }
                ],
                "temperature": 0.7
            })
            ```
        """
        if "log_level" in data:
            log_level = data["log_level"].upper()
            if hasattr(logging, log_level):
                logger.setLevel(getattr(logging, log_level))

        logger.info(f"Received data: {data}")

        try:
            conversation_state = ConversationState.from_api_request(data)
            logger.info(f"Conversation state after initialization: {conversation_state}")

            while conversation_state.status in {
                ConversationStatus.RUNNING,
                ConversationStatus.WRAP_THINGS_UP,
            }:
                self._process_conversation_turn(conversation_state)

            return_msg = json.loads(conversation_state.get_last_message().content)

            if not isinstance(return_msg, list):
                raise ModelOutputError("Model output is not a list of tool calls.")

            for tool_call in return_msg:
                if not isinstance(tool_call, dict):
                    raise ModelOutputError("Model output is not a list of tool calls.")

            return HandlerResponse(tool_calls=return_msg, warnings=None)

        except (
            ValidationError,
            ModelOutputError,
            PipelineConfigurationError,
            PipelineExecutionError,
            ToolConfigurationError,
        ):
            raise
        except (ValueError, TypeError) as e:
            logger.error(f"Input validation error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing request: {str(e)}", exc_info=True)
            raise

    def _process_conversation_turn(self, state: ConversationState) -> None:
        """Process a single turn of the conversation."""
        conversation_text = self._format_conversation(state)
        format_enforcer = get_format_enforcer(self.pipeline.tokenizer, state.update_current_tools())
        model_output = self._generate_prediction(
            conversation_text, format_enforcer, state.pipeline_params
        )

        tool_calls = self._format_model_output(model_output)
        serialized = [tool.model_dump() for tool in tool_calls]
        state.add_message(ConversationMessage(role="tool_calls", content=json.dumps(serialized)))

        if state.only_called_rest_api_tools(tool_calls):
            self._execute_tool_calls(tool_calls, state)

    def _execute_tool_calls(self, tool_calls: list[Tool], state: ConversationState) -> None:
        """Execute tool calls and update conversation state."""
        logger.debug(f"Executing tool calls: {tool_calls}")
        rest_api_catalog = state.get_rest_api_catalog()

        if not rest_api_catalog:
            logger.info("No rest API catalog is available, skipping all tool calls.")
            return

        tool_responses: list[ToolResponse] = []
        return_to_user = False
        for tool in tool_calls:
            logger.info(f"Calling tool '{tool.name}' with parameters {tool.parameters}")
            try:
                api_client = rest_api_catalog[tool.name]
                api_response: RestApiResponse = api_client.call(tool.parameters)
                api_client_behavior = (
                    api_client.config.response_behavior.get(api_response.status)
                    if api_client.config.response_behavior.get(api_response.status)
                    else api_client.config.response_behavior.get("default")
                )

                logger.info(f"API response: {api_response}, behavior: {api_client_behavior}")

                if api_response.type == "json" and api_client_behavior == "return_to_llm":
                    tool_responses.append(ToolResponse(name=tool.name, content=api_response.body))
                elif (
                    api_response.type == "json" and api_client_behavior == "return_response_to_user"
                ):
                    tool_responses.append(ToolResponse(name=tool.name, content=api_response.body))
                    return_to_user = True
                elif (
                    api_response.type == "json"
                    and api_client_behavior == "return_request_to_user"
                    and api_response.request
                ):
                    tool_responses.append(
                        ToolResponse(name=tool.name, content=api_response.request)
                    )
                    return_to_user = True
                else:
                    raise ToolConfigurationError(
                        f"Failed to call tool {tool.name}: "
                        f"{api_response.status} - {api_response.body}"
                    )
            except Exception as e:
                logger.error(f"Failed to call tool {tool.name}: {e}")
                tool_responses.append(
                    ToolResponse(name=tool.name, content=f"Failed to call tool: {e}")
                )
                state.remove_tool(tool.name)

        serialized_responses = [response.model_dump() for response in tool_responses]
        state.add_message(
            ConversationMessage(role="tool_response", content=json.dumps(serialized_responses))
        )

        if return_to_user and state.status != ConversationStatus.COMPLETED:
            state.transition_to(ConversationStatus.COMPLETED)

    def _format_model_output(self, model_output: list[dict[str, Any]]) -> list[Tool]:
        """Format the model output into a list of dictionaries."""
        if not model_output:
            return []

        output = model_output[0]
        generated_text: str | None = (
            output.get("generated_text") if type(output.get("generated_text")) is str else None
        )

        if generated_text is None:
            raise ModelOutputError("No generated_text found in the model output.")

        try:
            logger.debug(f"Generated text: {generated_text}")
            parsed_output: list[Tool] = [
                Tool.model_validate(tool) for tool in json.loads(generated_text)
            ]
            return parsed_output
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse generated_text as JSON: {e}")
            raise ModelOutputError(f"Failed to parse model output as JSON: {str(e)}") from e

    def _generate_prediction(
        self,
        formatted_input: str,
        prefix_function: FormatEnforcer | None,
        pipeline_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate prediction using the pipeline."""
        try:
            return self.pipeline(
                formatted_input,
                return_full_text=False,
                prefix_allowed_tokens_fn=prefix_function,
                **pipeline_params,
            )
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise PipelineExecutionError(f"Failed to generate prediction: {str(e)}") from e

    def _format_conversation(self, state: ConversationState) -> str:
        """
        Format the conversation into a string for the pipeline.
        """
        messages = [{"role": msg.role, "content": msg.content} for msg in state.messages]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            force_tool_calls=True,
            tokenize=False,
            bos_token=self.config.begin_token,
            eot_token=self.config.eot_token,
        )
        logger.debug(f"Formatted conversation: {prompt}")
        return prompt
