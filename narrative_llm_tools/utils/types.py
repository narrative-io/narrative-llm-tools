from typing import Protocol

from torch import Tensor


class TransformersPrefixAllowedTokensFn(Protocol):
    def __call__(self, batch_id: int, sent: Tensor) -> list[int]: ...
