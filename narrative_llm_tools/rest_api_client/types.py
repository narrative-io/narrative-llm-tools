from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class ParameterLocation(str, Enum):
    QUERY = "query"
    BODY = "body"


class BearerTokenAuth(BaseModel):
    env_var: str

    def __hash__(self) -> int:
        return hash(self.env_var)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, BearerTokenAuth) and self.env_var == other.env_var


class Behavior(BaseModel):
    behavior_type: str
    response: str | None = None

    def __hash__(self) -> int:
        return hash(self.behavior_type)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Behavior) and self.behavior_type == other.behavior_type


class ReturnToLlmBehavior(Behavior):
    behavior_type: Literal["return_to_llm"] = "return_to_llm"
    response: str | None = None


class ReturnResponseToUserBehavior(Behavior):
    behavior_type: Literal["return_response_to_user"] = "return_response_to_user"
    response: str | None = None


class ReturnRequestToUserBehavior(Behavior):
    behavior_type: Literal["return_request_to_user"] = "return_request_to_user"
    response: str | None = None


class RestApiResponse(BaseModel):
    status: int
    type: Literal["json", "text"]
    body: str
    request: str | None = None


class RestApiConfig(BaseModel):
    url: str
    method: HttpMethod
    auth: BearerTokenAuth | None = None
    response_behavior: dict[str | Literal["default"], Behavior] = {
        "default": ReturnToLlmBehavior(response=None),
    }
    query_path: str | None = None
    parameter_location: ParameterLocation | None = None

    model_config = {"frozen": True}

    def __init__(self, **data: Any) -> None:
        if "parameter_location" not in data or data["parameter_location"] is None:
            if data.get("method") in [HttpMethod.PUT, HttpMethod.POST]:
                data["parameter_location"] = ParameterLocation.BODY
            elif data.get("method") == HttpMethod.GET:
                data["parameter_location"] = ParameterLocation.QUERY
            elif data.get("method") == HttpMethod.DELETE:
                data["parameter_location"] = ParameterLocation.BODY
        super().__init__(**data)

    def __hash__(self) -> int:
        return hash(
            (
                self.url,
                self.method,
                self.auth,
                (
                    tuple(self.response_behavior.items())
                    if self.response_behavior is not None
                    else None
                ),
                self.query_path,
                self.parameter_location,
            )
        )

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, RestApiConfig)
            and self.url == other.url
            and self.method == other.method
            and self.auth == other.auth
            and self.response_behavior == other.response_behavior
            and self.query_path == other.query_path
            and self.parameter_location == other.parameter_location
        )
