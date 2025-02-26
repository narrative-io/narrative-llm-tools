from typing import Any, Protocol, TypeVar, overload

from pydantic import BaseModel

T = TypeVar("T", dict[str, Any], dict[str, list[Any]])


class DatasetTransformationFunction(Protocol):
    """Protocol for a data processing function that can operate in different modes.

    This protocol defines a flexible interface for dataset transformation functions that can handle
    both single examples and batches of data, with optional support for indices
    and rank information.

    The protocol supports eight different calling patterns through method overloading:
    1. Single example processing
    2. Single example with index
    3. Single example with rank
    4. Single example with both index and rank
    5. Batch processing
    6. Batch processing with indices
    7. Batch processing with ranks
    8. Batch processing with both indices and ranks

    All implementations must handle dictionary inputs and outputs, with batch operations working on
    dictionary values as lists.
    """

    @overload
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Base case: Single example without indices or rank."""
        ...

    @overload
    def __call__(self, example: dict[str, Any], idx: int) -> dict[str, Any]:
        """Single example with index only."""
        ...

    @overload
    def __call__(self, example: dict[str, Any], rank: int) -> dict[str, Any]:
        """Single example with rank only."""
        ...

    @overload
    def __call__(self, example: dict[str, Any], idx: int, rank: int) -> dict[str, Any]:
        """Single example with both index and rank."""
        ...

    @overload
    def __call__(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Batch case without indices or rank."""
        ...

    @overload
    def __call__(self, batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        """Batch case with indices only."""
        ...

    @overload
    def __call__(self, batch: dict[str, list[Any]], ranks: list[int]) -> dict[str, list[Any]]:
        """Batch case with ranks only."""
        ...

    @overload
    def __call__(
        self, batch: dict[str, list[Any]], indices: list[int], ranks: list[int]
    ) -> dict[str, list[Any]]:
        """Batch case with both indices and ranks."""
        ...


class GRPORecord(BaseModel):
    """Represents a structured record for GRPO (Group Relative Policy Optimization) format.

    This model defines the structure for conversation-based data, containing a prompt
    (as a list of messages) and a completion string.

    Attributes:
        prompt (list[Message]): A list of messages representing the conversation,
                                not including the desired output.
        completion (str): The final response or completion for the conversation
                          which is what we will score.
    """

    prompt: list[dict[str, Any]]
    ground_truth: str


def grpo_conversation_transform(
    cfg: Any, *args: Any, **kwargs: Any
) -> DatasetTransformationFunction:
    """Creates a transformation function for processing conversation data into GRPO format.

    This function factory creates a transformation function that converts conversation data
    into a standardized GRPO (Group Relative Policy Optimization) format. The returned function
    supports both single-example and batch processing, with optional support for indices
    and ranks.

    Args:
        cfg: Configuration object for customizing the transformation.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        A callable that implements the DatasetTransformationFunction protocol.

    Raises:
        ValueError: If the input data structure is invalid or missing required fields.
    """

    def determine_if_batch(*args: Any, **kwargs: Any) -> bool:
        """Determines if the input represents a batch or single example."""
        if not args:
            return False

        data = args[0]
        if isinstance(data, tuple) and len(data) == 1:
            data = data[0]

        return "Batch" in type(data).__name__

    def transform_single_record(record: dict[str, Any]) -> dict[str, Any]:
        # We should only have a single key in our dict that isn't reserved (starting with "_")
        # We'll use that as the key that is holding the conversation
        group_name = next(k for k in record.keys() if not k.startswith("_"))
        if group_name is None:
            raise ValueError("No conversation found in record")

        conversation = record[group_name]
        if not isinstance(conversation, list):
            raise ValueError("Conversation is not a list")

        completion = conversation[-1]["content"]
        if not isinstance(completion, str):
            raise ValueError("Completion is not a string")

        prompt = conversation[:-1]
        result = GRPORecord(prompt=prompt, ground_truth=completion)
        return result.model_dump()

    def transform_fn(example: dict[str, Any]) -> dict[str, Any]:
        return transform_single_record(example)

    def transform_fn_with_idx(example: dict[str, Any], idx: int) -> dict[str, Any]:
        return transform_single_record(example)

    def transform_fn_with_rank(example: dict[str, Any], rank: int) -> dict[str, Any]:
        return transform_single_record(example)

    def transform_fn_with_idx_rank(example: dict[str, Any], idx: int, rank: int) -> dict[str, Any]:
        return transform_single_record(example)

    # Batch cases
    def transform_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        results = []
        batch_size = len(next(iter(batch.values())))
        for i in range(batch_size):
            record = {k: v[i] for k, v in batch.items()}
            results.append(transform_single_record(record))
        return {k: [d[k] for d in results] for k in results[0].keys()}

    def transform_batch_with_indices(
        batch: dict[str, list[Any]], indices: list[int]
    ) -> dict[str, list[Any]]:
        results = []
        batch_size = len(next(iter(batch.values())))
        for i in range(batch_size):
            record = {k: v[i] for k, v in batch.items()}
            results.append(transform_single_record(record))
        return {k: [d[k] for d in results] for k in results[0].keys()}

    def transform_batch_with_ranks(
        batch: dict[str, list[Any]], ranks: list[int]
    ) -> dict[str, list[Any]]:
        results = []
        batch_size = len(next(iter(batch.values())))
        for i in range(batch_size):
            record = {k: v[i] for k, v in batch.items()}
            results.append(transform_single_record(record))
        return {k: [d[k] for d in results] for k in results[0].keys()}

    def transform_batch_with_indices_ranks(
        batch: dict[str, list[Any]], indices: list[int], ranks: list[int]
    ) -> dict[str, list[Any]]:
        results = []
        batch_size = len(next(iter(batch.values())))
        for i in range(batch_size):
            record = {k: v[i] for k, v in batch.items()}
            results.append(transform_single_record(record))
        return {k: [d[k] for d in results] for k in results[0].keys()}

    @overload
    def wrapper(example: dict[str, Any]) -> dict[str, Any]: ...

    @overload
    def wrapper(example: dict[str, Any], idx: int) -> dict[str, Any]: ...

    @overload
    def wrapper(example: dict[str, Any], rank: int) -> dict[str, Any]: ...

    @overload
    def wrapper(example: dict[str, Any], idx: int, rank: int) -> dict[str, Any]: ...

    @overload
    def wrapper(batch: dict[str, list[Any]]) -> dict[str, list[Any]]: ...

    @overload
    def wrapper(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]: ...

    @overload
    def wrapper(batch: dict[str, list[Any]], ranks: list[int]) -> dict[str, list[Any]]: ...

    @overload
    def wrapper(
        batch: dict[str, list[Any]], indices: list[int], ranks: list[int]
    ) -> dict[str, list[Any]]: ...

    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any] | dict[str, list[Any]]:
        """Main transformation function that handles all input patterns.

        This wrapper function implements the DatasetTransformationFunction protocol by detecting
        the input pattern and routing to the appropriate transformation function.

        Args:
            *args: Variable length argument list. The first argument must be either a single
                  example or batch dictionary. Optional second and third arguments can be
                  indices and/or ranks.
            **kwargs: Arbitrary keyword arguments (unused).

        Returns:
            Transformed data in either single example or batch format, depending on input.
        """
        is_batch = determine_if_batch(*args, **kwargs)
        first_arg = args[0]

        if is_batch:
            if len(args) == 1:
                return transform_batch(first_arg)
            elif len(args) == 2:
                if isinstance(args[1][0], int):  # Check first element of second argument
                    return transform_batch_with_indices(first_arg, args[1])
                else:
                    return transform_batch_with_ranks(first_arg, args[1])
            else:
                return transform_batch_with_indices_ranks(first_arg, args[1], args[2])
        else:
            if len(args) == 1:
                return transform_fn(first_arg)
            elif len(args) == 2:
                if isinstance(args[1], int):
                    return transform_fn_with_idx(first_arg, args[1])
                else:
                    return transform_fn_with_rank(first_arg, args[1])
            else:
                return transform_fn_with_idx_rank(first_arg, args[1], args[2])

    return wrapper
