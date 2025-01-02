import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jsonschema
from jsonschema import SchemaError
from pydantic import BaseModel, ValidationError, model_validator

from narrative_llm_tools.state.messages import Message, MessageWrapper, parse_json_array


class Conversation(BaseModel):
    conversations: list[Message]

    @model_validator(mode="after")
    def validate_conversation_structure(self) -> "Conversation":
        all_errors = []

        conv = self.conversations

        if len(conv) < 3:
            all_errors.append(f"'conversation' must have at least 3 messages. Found {len(conv)}.")

        system_count = 0
        tool_catalog_count = 0
        last_role = None
        found_system = False
        tool_catalog_schema = None
        assistant_call_indices = []
        user_count = 0

        for i, message in enumerate(conv):
            msg_errors = []

            try:
                _ = MessageWrapper(message=message)
            except ValidationError as e:
                msg_errors.append(f"Message is not valid: {e}")
                all_errors.extend(msg_errors)
                continue

            role = message.from_
            content = message.value

            # Check content is a non-empty string
            if not content or not isinstance(content, str):
                msg_errors.append("Message 'content' must be a non-empty string.")

            # --- Role-based checks -----------------------------------

            # 1) system role
            if role == "system":
                system_count += 1
                if system_count > 1:
                    msg_errors.append("Multiple 'system' messages found in a single conversation.")
                if i != 0:  # must be the first message if present
                    msg_errors.append("'system' message must appear as the very first message.")
                found_system = True

            # 2) tool_catalog role
            if role == "tool_catalog":
                tool_catalog_count += 1
                if tool_catalog_count > 1:
                    msg_errors.append(
                        "Multiple 'tool_catalog' messages " "found in a single conversation."
                    )

                # Must be second if system exists, else first
                if found_system and i != 1:
                    msg_errors.append(
                        "If 'system' is present, 'tool_catalog' must be the second message."
                    )
                elif (not found_system) and i != 0:
                    msg_errors.append(
                        "If 'system' is not present, 'tool_catalog' must be the first message."
                    )

                # The content of tool_catalog must be valid JSON and represent a JSON Schema.
                tool_catalog_schema, schema_errors = validate_tool_catalog_schema(content)
                if schema_errors:
                    msg_errors.extend(schema_errors)

            # 3) user role
            if role == "user":
                user_count += 1
                # No two user messages may appear consecutively
                if last_role == "user":
                    msg_errors.append("No two 'user' messages may appear consecutively.")

            # 4) assistant / tool_call (synonyms)
            #    Content must parse to an array. Each array element must validate
            #    against the schema in tool_catalog.
            if role in ("assistant", "tool_calls"):
                assistant_call_indices.append(i)
                # Must parse content into an array
                array_parse_error, arr = parse_json_array(content, tool_catalog_schema)
                if array_parse_error:
                    msg_errors.extend(array_parse_error)
                else:
                    pass

            # 5) tool_response
            if role in ("tool_response", "tool"):
                # Must appear immediately after an assistant/tool_call message
                if last_role not in ("assistant", "tool_calls"):
                    msg_errors.append(
                        "'tool_response' must appear immediately after an "
                        "'assistant'/'tool_call' message."
                    )
                # Content must be valid JSON that parses to an array
                array_parse_error, arr = parse_json_array(
                    content,
                    (
                        generate_tool_response_json_schema(
                            frozenset(extract_enumerated_names(tool_catalog_schema))
                        )
                        if tool_catalog_schema
                        else None
                    ),
                )
                if array_parse_error:
                    msg_errors.extend(array_parse_error)
                else:
                    # Validate array length and structure vs. preceding assistant/tool_call
                    # We'll do that after we collect everything, or inline here.

                    # The preceding assistant/tool_call message is conversation[i-1]
                    prev_content = conv[i - 1].value
                    # Parse the previous array
                    _, prev_arr = parse_json_array(prev_content, tool_catalog_schema)

                    if len(arr) != len(prev_arr):
                        msg_errors.append(
                            "tool_response array length must match the preceding "
                            "'assistant'/'tool_call' array."
                        )
                    else:
                        for idx, (response, prev_call) in enumerate(
                            zip(arr, prev_arr, strict=False)
                        ):
                            structure_errors = validate_tool_response_structure(response, idx)
                            if structure_errors:
                                msg_errors.extend(structure_errors)
                                continue

                            matching_errors = validate_tool_response_matching(
                                response, prev_call, idx
                            )
                            msg_errors.extend(matching_errors)

            all_errors.extend(msg_errors)
            last_role = role

        # Must have exactly 1 tool_catalog
        if tool_catalog_count == 0:
            all_errors.append("Missing required 'tool_catalog' message.")
        elif tool_catalog_count > 1:
            # Already reported above, but let's ensure we mention it here as well
            pass

        # Must appear at least one user message
        if user_count == 0:
            all_errors.append("Must have at least one 'user' message.")

        # Must end with assistant or tool_call
        if conv and conv[-1].from_ not in ("assistant", "tool_calls"):
            all_errors.append("The conversation must end with 'assistant' or 'tool_call'.")

        # If we have a valid tool_catalog schema, attempt to do a deeper validation:
        if tool_catalog_schema:
            # For each assistant/tool_call message, parse their array and validate each element
            for i in assistant_call_indices:
                msg_content: str = conv[i].value
                parse_err, arr = parse_json_array(msg_content, tool_catalog_schema)
                if parse_err:
                    # Already reported above, but let's skip schema validation here
                    continue

                enumerated_names = extract_enumerated_names(tool_catalog_schema)

                for idx, el in enumerate(arr):
                    if not isinstance(el, dict):
                        all_errors.append(f"Message {i}: array element {idx} must be an object.")
                        continue
                    if set(el.keys()) != {"name", "parameters"}:
                        all_errors.append(
                            f"Message {i}: array element {idx} "
                            "must have exactly 'name' and 'parameters'.",
                        )
                    # Check name
                    if "name" in el:
                        name_val = el["name"]
                        if name_val not in enumerated_names:
                            all_errors.append(
                                f"Message {i}: array element {idx} has "
                                f"'name'='{name_val}' not in {enumerated_names}.",
                            )
                    # Check parameters
                    if "parameters" in el:
                        if not isinstance(el["parameters"], dict):
                            all_errors.append(f"Message {i}: 'parameters' must be an object.")

        if all_errors:
            raise ValueError(all_errors)

        return self


def validate_conversation_object(obj: Any, line_number: int) -> list[str]:
    """
    Validate a single conversation using Pydantic models.
    """
    try:
        Conversation(**obj)
        return []
    except ValueError as e:
        return [f"Line {line_number}: {str(e)}"]


@dataclass
class ValidationResult:
    line_number: int
    errors: list[str]


def validate_line(args: tuple[str, int]) -> ValidationResult:
    line, line_number = args
    if not line.strip():
        return ValidationResult(line_number, [f"Line {line_number}: Empty line is not allowed."])

    try:
        conversation_obj = json.loads(line)
    except json.JSONDecodeError as e:
        return ValidationResult(line_number, [f"Line {line_number}: Invalid JSON - {str(e)}"])

    line_errors = validate_conversation_object(conversation_obj, line_number)
    return ValidationResult(line_number, line_errors)


def extract_enumerated_names(tool_catalog_schema: Mapping[str, Any]) -> set[str]:
    """
    Example function to gather all enumerated tool names from the
    'tool_catalog' JSON Schema. The spec indicates each object in
    tool_catalog_schema["items"]["anyOf"] has an enum for 'name'.
    """
    enumerated = set()

    # Attempt to parse out the "anyOf" array
    try:
        any_of = tool_catalog_schema["items"]["anyOf"]
        for obj_schema in any_of:
            name_node = obj_schema.get("properties", {}).get("name", {})
            if "enum" in name_node:
                for name_val in name_node["enum"]:
                    enumerated.add(name_val)
    except (KeyError, TypeError):
        pass

    return enumerated


@lru_cache(maxsize=1000)
def generate_tool_response_json_schema(tool_names: set[str]) -> Mapping[str, Any]:
    """
    Generate a JSON Schema for the 'tool_response' message.
    """
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": list(tool_names)},
                "content": {"type": "string"},
            },
            "required": ["name", "content"],
        },
    }


@lru_cache(maxsize=1000)
def validate_tool_catalog_schema(schema_str: str) -> tuple[Any, list[str]]:
    """Cached validation of tool catalog schema"""
    errors = []
    try:
        schema = json.loads(schema_str)
        jsonschema.Draft7Validator.check_schema(schema)
        return schema, []
    except json.JSONDecodeError:
        errors.append("'tool_catalog' message 'content' must be valid JSON (a JSON Schema).")
    except SchemaError as e:
        errors.append(f"'tool_catalog' message contains invalid JSON Schema: {str(e)}")
    except Exception as e:
        errors.append(f"Unexpected error validating 'tool_catalog' message: {str(e)}")
    return None, errors


def validate_tool_response_structure(response: dict[str, Any], idx: int) -> list[str]:
    """Validate the structure of a single tool response object."""
    errors = []

    if not isinstance(response, dict):
        errors.append(f"Response at index {idx} must be an object")  # type: ignore[unreachable]
        return errors

    if set(response.keys()) != {"name", "content"}:
        errors.append(f"Response at index {idx} must have exactly 'name' and 'content' fields")
        return errors

    if not isinstance(response["name"], str) or not isinstance(response["content"], str):
        errors.append(f"Response at index {idx}: 'name' and 'content' must be strings")

    return errors


def validate_tool_response_matching(
    response: dict[str, Any], prev_call: dict[str, Any], idx: int
) -> list[str]:
    """Validate that a tool response matches its corresponding tool call."""
    errors = []
    if response["name"] != prev_call.get("name"):
        errors.append(
            f"Response at index {idx}: name '{response['name']}' does not match "
            f"tool call name '{prev_call.get('name')}'"
        )
    return errors
