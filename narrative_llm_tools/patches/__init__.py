import json
from typing import Any

import jinja2.sandbox

OriginalEnv = jinja2.sandbox.ImmutableSandboxedEnvironment


class MySandboxedEnvironment(OriginalEnv):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.filters["from_json"] = self.from_json

    @staticmethod
    def from_json(x: str) -> Any:
        try:
            return json.loads(x)
        except json.JSONDecodeError as e:
            raise jinja2.exceptions.TemplateError(f"Invalid JSON string: {str(e)}") from e


jinja2.sandbox.ImmutableSandboxedEnvironment = MySandboxedEnvironment  # type: ignore
