from typing import Protocol

from pydantic import BaseModel
from torch import Tensor


class TransformersPrefixAllowedTokensFn(Protocol):
    def __call__(self, batch_id: int, sent: Tensor) -> list[int]: ...


class Message(BaseModel):
    role: str
    content: str
