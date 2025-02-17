import json
import re
from collections.abc import Iterator
from typing import Any, Protocol

from jsonschema import ValidationError, validate

StringOrMessage = str | list[dict[str, Any]]


class RewardFn(Protocol):
    """Protocol defining the interface for reward functions in a completion evaluation system.

    This protocol specifies the call signature for reward functions that evaluate model
    completions and assign numerical scores. Reward functions implementing this protocol
    are used to assess various aspects of model outputs, such as format compliance,
    content quality, or specific behavioral metrics.

    Call Signature:
        __call__(completions: list[list[dict[str, Any]]], **kwargs: Any) -> list[float]

    Args:
        completions: A nested list structure representing a batch of conversation turns.
            Each inner list contains message dictionaries with keys like 'role' and
            'content' representing a single conversation.
            Example structure:
            [
                [  # First conversation
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"}
                ],
                [  # Second conversation
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "It's sunny"}
                ]
            ]
        prompts: A list of prompts, one per completion.
        **kwargs: Additional keyword arguments to allow for flexible parameter passing
            and future extensibility.

    Returns:
        A list of float values between 0.0 and 1.0, one score per completion in the
        input batch. Higher values typically indicate better performance or stronger
        adherence to desired criteria.

    Example Implementation:
        >>> class FormatReward(RewardFn):
        ...     def __call__(self, completions, prompts, **kwargs):
        ...         return [1.0 for _ in completions]  # Perfect format score
    """

    def __call__(
        self,
        completions: list[StringOrMessage],
        prompts: list[StringOrMessage] | None = None,
        **kwargs: list[Any],
    ) -> list[float]: ...


def get_first_message_content_by_role(
    messages: list[dict[str, Any]],
    role: str,
    role_property: str = "role",
    content_property: str = "content",
) -> str:
    """Extract content from the first message matching the specified role in a conversation.

    This function searches through a list of message dictionaries and returns the content
    of the first message that matches the specified role. It's commonly used to extract
    specific parts of a conversation (e.g., assistant responses, user inputs) from a
    chat-like message structure.

    Args:
        messages: A list of message dictionaries, where each dictionary represents a
            conversation turn and contains at least 'role' and 'content' keys.
            Example: [{'role': 'user', 'content': 'Hello'},
                     {'role': 'assistant', 'content': 'Hi'}]
        role: The role to search for (e.g., 'user', 'assistant', 'system')

    Returns:
        The content string from the first message matching the specified role.
        Returns an empty string if no matching role is found or if the matching
        message has no content.

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "Be helpful"},
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there"}
        ... ]
        >>> get_content_by_role(messages, "user")
        'Hello'
        >>> get_content_by_role(messages, "unknown")
        ''

    Notes:
        - Only returns content from the first matching message if multiple exist
    """
    message = next((msg for msg in messages if msg.get(role_property, "") == role), None)
    return message.get(content_property, "") if message else ""


def count_thoughts(text: str) -> tuple[int, int]:
    """Count the number of thought segments and their total character length in a text.

    This function searches for thought segments enclosed in <|start_thought|> and
    <|end_thought|> tags within the input text. It counts both the number of distinct
    thought segments and the total number of characters contained within all thoughts
    (excluding the tags themselves).

    Args:
        text: The input string to analyze. May contain zero or more thought segments
            enclosed in thought tags.

    Returns:
        A tuple of two integers where:
        - First element is the count of thought segments found
        - Second element is the total character length of all thought contents combined
          (excluding the tag markers)

    Example:
        >>> text = "<|start_thought|>one<|end_thought|> two <|start_thought|>three<|end_thought|>"
        >>> count_thoughts(text)
        (2, 17)  # 2 thoughts with combined length of "Check weather" + "Call API"

    Notes:
        - Uses non-greedy regex matching to handle nested or sequential thought segments
        - Counts only the characters within the thought segments, not the tag markers
        - Empty thought segments (<|start_thought|><|end_thought|>) are counted as
          segments with zero length
        - The DOTALL flag allows matching thoughts containing newlines
    """
    pattern = r"<\|start_thought\|>(.*?)<\|end_thought\|>"
    matches = re.findall(pattern, text, re.DOTALL)
    total_chars = sum(len(match) for match in matches)
    return len(matches), total_chars


def format_reward(
    completions: list[StringOrMessage],
    prompts: list[StringOrMessage] | None = None,
    **kwargs: list[Any],
) -> list[float]:
    """Evaluates if model completions follow the expected format structure with thoughts
       and tool calls.

    This function acts as a reward function that checks if each completion in the input follows
    a specific pattern with optional thought blocks followed by a tool calls block. The expected
    format is:
    - Zero or more thought blocks enclosed in <|start_thought|> and <|end_thought|> tags
    - A mandatory tool calls block enclosed in <|tool_calls|> tags containing a JSON array
    - An end-of-text identifier <|eot_id|>

    Args:
        completions: A list of conversation turns, where each turn is a list of message
            dictionaries. Each message dictionary contains role and content keys.
        **kwargs: Additional keyword arguments (unused but included for compatibility
            with reward function interface)

    Returns:
        A list of float rewards, one per completion, where:
        - 1.0 indicates the completion matches the expected format
        - 0.0 indicates the completion does not match the format

    Example:
        Valid format:
        <|start_thought|>I should check the weather<|end_thought|>
        <|tool_calls|>[{"type": "function", "name": "get_weather"}]
        <|eot_id|>
    """
    pattern = (
        r"^(?:<\|start_thought\|>.*?<\|end_thought\|>\s*)*"
        r"<\|tool_calls\|>\[.*?\]\s*"
        r"<\|eot_id\|>$"
    )

    rewards = []
    for completion in completions:
        content = (
            get_first_message_content_by_role(completion, "assistant")
            if not isinstance(completion, str)
            else completion
        )
        match = bool(re.match(pattern, content, re.DOTALL | re.MULTILINE))
        rewards.append(1.0 if match else 0.0)

    return rewards


def thought_steps_reward(
    completions: list[StringOrMessage],
    prompts: list[StringOrMessage] | None = None,
    **kwargs: list[Any],
) -> list[float]:
    """Reward function that checks for the presence and quality of thought steps."""
    rewards = []

    for completion in completions:
        content = (
            get_first_message_content_by_role(completion, "assistant")
            if isinstance(completion, list)
            else completion
        )
        num_thoughts, total_chars = count_thoughts(content)

        # Combine both quantity and quality metrics
        thought_count_score = min(1.0, num_thoughts / 3)  # Encourage at least 3 thoughts
        length_score = min(total_chars / 1000, 1.0)  # Cap at 1000 chars

        # Weight both aspects
        combined_score = 0.7 * thought_count_score + 0.3 * length_score
        rewards.append(combined_score)

    return rewards


def validate_tool_call_against_schema(tool_call: str, schema: dict[str, Any]) -> bool:
    """Validate a tool call string against a JSON schema."""
    try:
        tool_call_json = json.loads(tool_call)
        validate(instance=tool_call_json, schema=schema)
        return True
    except (json.JSONDecodeError, ValidationError):
        return False


def _validate_tool_call_structure(call: Any) -> bool:
    """Validate the basic structure of a single tool call."""
    if not isinstance(call, dict):
        return False

    required_fields = {"name", "parameters"}
    required_params = {"attribute_id", "expression", "type"}

    return (
        all(field in call for field in required_fields)
        and isinstance(call["parameters"], dict)
        and all(param in call["parameters"] for param in required_params)
    )


def tool_calls_validity_reward(
    completions: list[StringOrMessage],
    prompts: list[StringOrMessage] | None = None,
    **kwargs: list[Any],
) -> list[float]:
    """Reward function that validates the structure and content of tool calls."""
    rewards = []

    for i, completion in enumerate(completions):
        content = (
            get_first_message_content_by_role(completion, "assistant")
            if isinstance(completion, list)
            else completion
        )

        try:
            # Extract and parse tool calls
            tool_calls_match = re.search(r"<\|tool_calls\|>(\[.*?\])", content, re.DOTALL)
            if not tool_calls_match:
                rewards.append(0.0)
                continue

            tool_calls = json.loads(tool_calls_match.group(1))
            if not isinstance(tool_calls, list):
                rewards.append(0.0)
                continue

            # Validate against schema if available
            if isinstance(prompts, list) and i < len(prompts):
                prompt = prompts[i]
                if isinstance(prompt, list):
                    if tool_catalog_str := get_first_message_content_by_role(
                        prompt, "tool_catalog"
                    ):
                        try:
                            schema = json.loads(tool_catalog_str)
                            if not validate_tool_call_against_schema(
                                tool_calls_match.group(1), schema
                            ):
                                rewards.append(0.0)
                                continue
                        except json.JSONDecodeError:
                            pass

            # Validate structure of each tool call
            rewards.append(1.0 if all(map(_validate_tool_call_structure, tool_calls)) else 0.0)

        except json.JSONDecodeError:
            rewards.append(0.0)

    return rewards


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.5) -> RewardFn:
    """Creates a reward function that penalizes repetitive content in thoughts."""

    def zipngram(text: str, ngram_size: int) -> Iterator[tuple[str, ...]]:
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)], strict=False)

    def repetition_penalty_reward(
        completions: list[StringOrMessage],
        prompts: list[StringOrMessage] | None = None,
        **kwargs: list[Any],
    ) -> list[float]:
        rewards = []

        for completion in completions:
            content = (
                get_first_message_content_by_role(completion, "assistant")
                if isinstance(completion, list)
                else completion
            )

            thoughts = re.findall(r"<\|start_thought\|>(.*?)<\|end_thought\|>", content, re.DOTALL)

            if not thoughts:
                rewards.append(0.0)
                continue

            combined_thoughts = " ".join(thoughts)

            if len(combined_thoughts.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(combined_thoughts, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)

        return rewards

    return repetition_penalty_reward


def combine_rewards(
    reward_functions: list[tuple[RewardFn, float]],
    completions: list[StringOrMessage],
    prompts: list[StringOrMessage] | None = None,
    **kwargs: list[Any],
) -> list[float]:
    """Combines multiple reward functions with weights."""
    if not reward_functions:
        return [0.0] * len(completions)

    # Normalize weights to sum to 1.0
    functions, weights = zip(*reward_functions, strict=False)
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Calculate combined rewards
    combined_rewards = [0.0] * len(completions)

    for func, weight in zip(functions, normalized_weights, strict=False):
        if "prompts" in func.__code__.co_varnames and prompts is not None:
            rewards = func(prompts=prompts, completions=completions, **kwargs)
        else:
            rewards = func(completions=completions, **kwargs)

        for i, reward in enumerate(rewards):
            combined_rewards[i] += reward * weight

    return combined_rewards


def get_default_reward_function(*, include_schema_validation: bool = True) -> RewardFn:
    """Returns a default reward function with pre-configured weights."""

    def default_reward_function(
        completions: list[StringOrMessage],
        prompts: list[StringOrMessage] | None = None,
        **kwargs: list[Any],
    ) -> list[float]:
        reward_functions = [
            (format_reward, 0.25),
            (thought_steps_reward, 0.35),
            (tool_calls_validity_reward, 0.75),
            (get_repetition_penalty_reward(), 0.15),
        ]

        return combine_rewards(
            reward_functions=reward_functions, prompts=prompts, completions=completions, **kwargs
        )

    return default_reward_function
