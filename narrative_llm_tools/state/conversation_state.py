import json
import logging
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, field_validator

from narrative_llm_tools.rest_api_client.rest_api_client import RestApiClient
from narrative_llm_tools.tools.json_schema_tools import JsonSchemaTools, Tool

logger = logging.getLogger(__name__)


class ConversationMessage(BaseModel):
    """
    Represents a single message in a conversation. The `role` indicates who/what
    generated the message (e.g. user, assistant, system, tool, etc.).
    """

    role: Literal["user", "assistant", "system", "tool_response", "tool_catalog", "tool_calls"]
    content: str


class ToolResponse(BaseModel):
    """
    Represents a structured response from a tool.
    """

    name: str
    content: str


class ConversationStatus(Enum):
    """
    Possible statuses for a ConversationState. The conversation can be running,
    waiting for a tool response, completed, etc.
    """

    RUNNING = "running"
    WAITING_TOOL_RESPONSE = "waiting_tool_response"
    COMPLETED = "completed"
    WRAP_THINGS_UP = "wrap_things_up"
    TOOL_ROUNDS_EXCEEDED = "tool_rounds_exceeded"


#: Valid status transitions for conversation flow.
VALID_TRANSITIONS = {
    ConversationStatus.RUNNING: [
        ConversationStatus.WRAP_THINGS_UP,
        ConversationStatus.COMPLETED,
        ConversationStatus.WAITING_TOOL_RESPONSE,
        ConversationStatus.TOOL_ROUNDS_EXCEEDED,
    ],
    ConversationStatus.WRAP_THINGS_UP: [
        ConversationStatus.COMPLETED,
        ConversationStatus.WAITING_TOOL_RESPONSE,
    ],
    ConversationStatus.WAITING_TOOL_RESPONSE: [
        ConversationStatus.RUNNING,
        ConversationStatus.WRAP_THINGS_UP,
    ],
    ConversationStatus.COMPLETED: [],
    ConversationStatus.TOOL_ROUNDS_EXCEEDED: [],
}


class ConversationState(BaseModel):
    """
    Manages the state of a conversation, including messages, tools, status, and
    allowed transitions. Provides methods to add new messages (tool calls or
    tool responses), validate transitions, and keep track of rounds of tool usage.
    """

    RESERVED_KEYS: ClassVar[set[str]] = {
        "inputs",
        "tools",
        "tool_choice",
        "max_tool_rounds",
        "log_level",
    }

    raw_messages: list[ConversationMessage]
    max_tool_rounds: int = 5
    tool_choice: Literal["required", "auto", "none"] = "required"
    tools_catalog: JsonSchemaTools = JsonSchemaTools.only_user_response_tool()
    pipeline_params: dict[str, Any]
    status: ConversationStatus = ConversationStatus.RUNNING

    model_config = {
        "validate_assignment": True,
    }

    @classmethod
    def from_api_request(cls, request_data: dict[str, Any]) -> "ConversationState":
        """
        Creates a ConversationState instance from an API request dictionary,
        extracting pipeline params and properly instantiating any needed tools.
        """
        # Extract pipeline parameters (excluding reserved and private keys)
        pipeline_params = {
            k: v
            for k, v in request_data.items()
            if k not in cls.RESERVED_KEYS and not k.startswith("_")
        }

        tools_data = request_data.get("tools", {})
        tools_instance = (
            JsonSchemaTools.model_validate(tools_data)
            if tools_data
            else JsonSchemaTools.only_user_response_tool()
        )

        tool_choice = request_data.get("tool_choice", "required")
        status = (
            ConversationStatus.WRAP_THINGS_UP
            if tool_choice == "none"
            else ConversationStatus.RUNNING
        )
        max_tool_rounds = request_data.get("max_tool_rounds", 5)

        return cls(
            raw_messages=request_data["inputs"],
            tool_choice=tool_choice,
            max_tool_rounds=max_tool_rounds,
            tools_catalog=tools_instance,
            pipeline_params=pipeline_params,
            status=status,
        )

    @field_validator("max_tool_rounds", mode="before")
    def validate_positive(cls, value: int) -> int:
        """
        Validates that `max_tool_rounds` is a positive integer.
        """
        if value <= 0:
            raise ValueError("max_tool_rounds must be a positive integer.")
        return value

    @field_validator("raw_messages")
    def validate_messages(cls, value: Any) -> list[ConversationMessage]:
        """
        Validates that messages is a list of valid ConversationMessage items.
        """
        if not isinstance(value, list):
            raise ValueError("'messages' must be a list of ConversationMessage objects.")
        for message in value:
            ConversationMessage.model_validate(message)
        return value

    @property
    def rest_api_names(self) -> set[str]:
        """
        Returns a set of tool names corresponding to REST API tools in the catalog.
        """
        return set(self.get_rest_api_catalog().keys())

    def has_system_message(self) -> bool:
        """Checks whether there's at least one system message in the conversation."""
        return any(msg.role == "system" for msg in self.raw_messages)

    def has_tool_catalog_message(self) -> bool:
        """Checks whether there's at least one tool_catalog message in the conversation."""
        return any(msg.role == "tool_catalog" for msg in self.raw_messages)

    def get_last_message(self) -> ConversationMessage:
        """Returns the most recent message in the conversation."""
        return self.raw_messages[-1]

    def responded_to_user(self, content: str) -> bool:
        """
        Checks if the content includes any non-REST API tool calls, which indicates
        a response to the user (since REST API tools are for data gathering only).
        """
        tool_calls = self.parse_tool_calls_content(content)
        return any(tool.name not in self.rest_api_names for tool in tool_calls)

    def only_called_rest_api_tools(self, tool_calls: list[Tool]) -> bool:
        """
        Returns True if all called tools are in the set of known REST API tool names.
        """
        return all(tool.name in self.rest_api_names for tool in tool_calls)

    def _has_non_rest_tool(self) -> bool:
        """
        Internal helper to check if there's at least one non-REST API tool available.
        """
        return len(self.rest_api_names) != len(self.tools_catalog.items.anyOf)

    def _has_rest_api_tools(self, content: str) -> bool:
        """Checks if the given content calls any REST API tools."""
        tool_calls = self.parse_tool_calls_content(content)
        return any(tool.name in self.rest_api_names for tool in tool_calls)

    def parse_tool_calls_content(self, content: str) -> list[Tool]:
        """
        Parses a JSON list of tool calls from string content and returns them as a list of `Tool`.
        """
        try:
            tool_calls_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError("Tool calls content must be valid JSON.") from e

        if not isinstance(tool_calls_data, list):
            raise ValueError("Tool calls must be a list of tool call objects.")

        return [Tool.model_validate(item) for item in tool_calls_data]

    def last_round(self) -> bool:
        """
        Checks if the current round is the last one before max_tool_rounds is reached.
        """
        return self.tool_calls_count == (self.max_tool_rounds - 1)

    def exceeded_tool_rounds(self) -> bool:
        """Checks if the number of tool calls so far has exceeded the maximum."""
        return self.tool_calls_count >= self.max_tool_rounds

    def must_respond(self) -> bool:
        """
        Returns True if we have hit exactly max_tool_rounds, indicating the user
        must be responded to.
        """
        return self.tool_calls_count == self.max_tool_rounds

    def can_respond(self) -> bool:
        """
        Returns True if there is at least one non-REST API tool available
        (e.g., to respond to the user).
        """
        return self._has_non_rest_tool()

    def get_rest_api_catalog(self) -> dict[str, RestApiClient]:
        """Returns all REST API tools from the current catalog."""
        return self.tools_catalog.get_rest_apis()

    def remove_tool(self, tool_name: str) -> None:
        """
        Removes the specified tool from the catalog if it exists.
        """
        self.tools_catalog = self.tools_catalog.remove_tool_by_name(tool_name)

    @property
    def tool_calls_count(self) -> int:
        """
        Returns the total number of tool calls so far.
        """
        return sum(1 for msg in self.raw_messages if msg.role == "tool_calls")

    def _tool_catalog_message(self) -> ConversationMessage:
        """
        Internal helper to produce a ConversationMessage containing the current tools.
        """
        return ConversationMessage(
            role="tool_catalog",
            content=json.dumps(self.tools_catalog.model_dump(), separators=(",", ":")),
        )

    def add_message(self, message: ConversationMessage) -> None:
        """
        Adds a new message (only tool_calls or tool responses) to the conversation.

        Args:
            message: The `ConversationMessage` to add.

        Raises:
            ValueError: If the message role is invalid or adding it violates state constraints.
        """
        if message.role == "tool_calls":
            self._handle_tool_call(message)
        elif message.role == "tool_response":
            self._handle_tool_response(message)
        else:
            raise ValueError(f"Invalid message role: {message.role}")

        logger.info(f"Conversation state after adding message: {self}")

    def _handle_tool_call(self, message: ConversationMessage) -> None:
        """
        Handles adding a tool_call message and performing relevant state transitions.
        """
        tool_calls = self.parse_tool_calls_content(message.content)
        self.raw_messages.append(message)

        if self.responded_to_user(message.content):
            self.transition_to(ConversationStatus.COMPLETED)
        elif self.exceeded_tool_rounds():
            self.transition_to(ConversationStatus.TOOL_ROUNDS_EXCEEDED)
        elif self.last_round():
            self.transition_to(ConversationStatus.WRAP_THINGS_UP)
        elif self.only_called_rest_api_tools(tool_calls):
            self.transition_to(ConversationStatus.WAITING_TOOL_RESPONSE)

    def _handle_tool_response(self, message: ConversationMessage) -> None:
        """
        Handles adding a tool response message and updating state accordingly.
        """
        self.raw_messages.append(message)

        if self.status == ConversationStatus.WAITING_TOOL_RESPONSE:
            if self.last_round():
                self.transition_to(ConversationStatus.WRAP_THINGS_UP)
            else:
                self.transition_to(ConversationStatus.RUNNING)

    def transition_to(self, new_status: ConversationStatus) -> None:
        """
        Transitions the conversation to a new status if it is a valid move.
        """
        allowed_transitions = VALID_TRANSITIONS.get(self.status, [])
        if new_status not in allowed_transitions:
            raise ValueError(f"Invalid transition from {self.status.value} to {new_status.value}.")
        logger.info(f"Transitioning from {self.status.value} to {new_status.value}")
        self.status = new_status
        self.update_current_tools()

    @property
    def messages(self) -> list[ConversationMessage]:
        """
        Returns the conversation's messages, automatically injecting:
          - A default system message at the start if one doesn't exist.
          - A tool_catalog message immediately after the system message.
        """
        if not self.has_system_message():
            # Inject system message and catalog at the front
            return [
                ConversationMessage(role="system", content="You are a helpful assistant."),
                self._tool_catalog_message(),
            ] + self.raw_messages

        # Otherwise, locate the system message and insert tool_catalog right after it
        system_msg_index = next(
            (i for i, msg in enumerate(self.raw_messages) if msg.role == "system"),
            None,
        )

        if system_msg_index is None:  # This should never happen
            return self.raw_messages

        # If system message is found, we insert the catalog message right after it.
        return (
            self.raw_messages[: system_msg_index + 1]
            + [self._tool_catalog_message()]
            + self.raw_messages[system_msg_index + 1 :]
        )

    def _remove_rest_api_tools(self) -> None:
        """
        Removes all REST API tools from the catalog.
        """
        self.tools_catalog = self.tools_catalog.remove_rest_api_tools()

    def update_current_tools(self) -> JsonSchemaTools:
        """
        Returns the appropriate tool catalog for the current conversation state:
          - If status is WRAP_THINGS_UP, only return user-response tool.
          - If status is RUNNING but there's no way to respond, return a catalog
            that includes a user-response tool. Otherwise, return the current tools.
        """
        if len(self.tools_catalog.items.anyOf) == 0:
            self.tools_catalog = JsonSchemaTools.only_user_response_tool()
        elif self.status == ConversationStatus.WRAP_THINGS_UP:
            logger.info(
                "Removing rest API tools.  "
                "We started with {len(self.tools_catalog.items.anyOf)} tools.",
            )
            self._remove_rest_api_tools()
            logger.info(
                "After removing rest API tools, "
                "we have {len(self.tools_catalog.items.anyOf)} tools.",
            )
            if len(self.tools_catalog.items.anyOf) == 0:
                self.tools_catalog = JsonSchemaTools.only_user_response_tool()
        elif self.status == ConversationStatus.RUNNING:
            if not self.can_respond():
                self.tools_catalog = self.tools_catalog.with_user_response_tool()
        elif self.status in [
            ConversationStatus.WAITING_TOOL_RESPONSE,
            ConversationStatus.COMPLETED,
            ConversationStatus.TOOL_ROUNDS_EXCEEDED,
        ]:
            pass
        else:
            raise ValueError(f"Invalid state for retrieving tools: {self.status}")

        return self.tools_catalog
