import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal

import jsonschema
from pydantic import BaseModel, Field, field_validator


def parse_json_array(
    content_str: str, tool_catalog_schema: Mapping[str, Any] | None
) -> tuple[list[str], list[Any]]:
    """
    Tries to parse a string into JSON and ensure it's an array (list).
    Returns (error_message, parsed_array) where error_message is None if success.
    """
    errors: list[str] = []

    try:
        parsed = json.loads(content_str)

        if not isinstance(parsed, list):
            errors.append(f"Invalid JSON in 'value': Expected a list, got {type(parsed)}")
            return (errors, [])

        if tool_catalog_schema:
            jsonschema.validate(instance=parsed, schema=tool_catalog_schema)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in 'value': {e}")
    except jsonschema.ValidationError as e:
        errors.append(f"Invalid JSON in 'value'.  Does not match tool_catalog schema: {e}")
    except Exception as e:
        errors.append(f"Unexpected error: {e}")

    return (errors, parsed)


class BaseMessage(BaseModel):
    from_: str = Field(alias="from")
    value: str

    @field_validator("value")
    def validate_value(cls, v: Any) -> str:
        if not v:
            raise ValueError("Message 'content' must be a non-empty string")
        if isinstance(v, str):
            return v
        raise ValueError(f"Message 'content' must be a string, got {type(v)}")


class SystemMessage(BaseMessage):
    from_: Literal["system"] = Field(alias="from")


class ToolCatalogMessage(BaseMessage):
    from_: Literal["tool_catalog"] = Field(alias="from")

    @field_validator("value")
    def validate_catalog(cls, v: Any) -> str:
        try:
            schema = json.loads(v)
            jsonschema.Draft7Validator.check_schema(schema)
        except json.JSONDecodeError as err:
            raise ValueError(
                "'tool_catalog' message 'content' ", "must be valid JSON (a JSON Schema)."
            ) from err
        except jsonschema.ValidationError as e:
            raise ValueError(
                "'tool_catalog' message contains ", f"invalid JSON Schema: {str(e)}"
            ) from e

        if isinstance(v, str):
            return v
        raise ValueError(f"Message 'content' must be a string, got {type(v)}")


class UserMessage(BaseMessage):
    from_: Literal["user"] = Field(alias="from")


class ToolCallMessage(BaseMessage):
    from_: Literal["assistant", "tool_calls"] = Field(alias="from")

    @field_validator("value")
    def validate_tool_calls(cls, v: Any) -> str:
        errors, parsed = parse_json_array(v, None)
        if errors:
            raise ValueError(errors)
        if not isinstance(parsed, list):
            raise ValueError("Tool calls must be a JSON array")
        for item in parsed:
            if not isinstance(item, dict) or set(item.keys()) != {"name", "parameters"}:
                raise ValueError("Each tool call must have exactly 'name' and 'parameters' fields")
        if isinstance(v, str):
            return v
        raise ValueError(f"Message 'content' must be a string, got {type(v)}")


class ToolResponseMessage(BaseMessage):
    from_: Literal["tool_response", "tool"] = Field(alias="from")

    @field_validator("value")
    def validate_response(cls, v: Any) -> str:
        errors, parsed = parse_json_array(v, None)
        if errors:
            raise ValueError(errors)
        if not isinstance(parsed, list):
            raise ValueError("Tool response must be a JSON array")
        for item in parsed:
            if not isinstance(item, dict) or set(item.keys()) != {"name", "content"}:
                raise ValueError("Each response must have exactly 'name' and 'content' fields")
        if isinstance(v, str):
            return v
        raise ValueError(f"Message 'content' must be a string, got {type(v)}")


Message = Annotated[
    SystemMessage | ToolCatalogMessage | UserMessage | ToolCallMessage | ToolResponseMessage,
    Field(discriminator="from_"),
]


class MessageWrapper(BaseModel):
    message: Message
