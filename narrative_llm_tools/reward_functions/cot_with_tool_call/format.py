### https://huggingface.co/docs/trl/main/en/grpo_trainer

import json
import re
from collections.abc import Callable
from typing import Any

from jsonschema import ValidationError, validate


def get_content_by_role(messages: list[dict[str, Any]], role: str) -> str:
    """Extract content from the first message matching the specified role.

    Args:
        messages: List of message dictionaries containing 'role' and 'content'
        role: The role to search for (e.g. 'user', 'assistant', 'tool_catalog')

    Returns:
        Content string from the first matching message, or empty string if no match
    """
    message = next((msg for msg in messages if msg.get("role", "") == role), None)
    return message.get("content", "") if message else ""


def count_thoughts(text: str) -> tuple[int, int]:
    """Count thought segments and their total character length in a string.

    Args:
        text: String to analyze

    Returns:
        tuple containing:
        - Number of thought segments
        - Total characters between thought tags
    """
    pattern = r"<\|start_thought\|>(.*?)<\|end_thought\|>"
    matches = re.findall(pattern, text, re.DOTALL)

    total_chars = sum(len(match) for match in matches)

    return len(matches), total_chars


def cot_has_thoughts_reward_function(
    *, completions: list[list[dict[str, Any]]], **kwargs: Any
) -> list[float]:
    """Calculate rewards based on presence of chain-of-thought segments.

    Assigns a binary reward (1.0 or 0.0) to each completion based on whether
    it contains at least one thought segment marked with <|start_thought|>
    and <|end_thought|> tags.

    Args:
        completions: List of text completions to evaluate
        **kwargs: Additional dataset columns (unused)

    Returns:
        List of float rewards, where:
        - 1.0 indicates presence of at least one thought segment
        - 0.0 indicates no thought segments found
    """

    rewards: list[float] = []

    for completion in completions:
        content = get_content_by_role(completion, "assistant")
        thoughts_count, _ = count_thoughts(content)
        if thoughts_count > 0:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def cot_thoughts_length_reward_function(
    *, completions: list[list[dict[str, Any]]], **kwargs: Any
) -> list[float]:
    """Calculate rewards based on the length of chain-of-thought segments.

    Assigns a reward based on the total character length of all thought segments
    found in each completion.

    Args:
        completions: List of text completions to evaluate
        **kwargs: Additional dataset columns (unused)

    Returns:
        List of float rewards, where:
        - Reward should scale with the length of the thought segments
          up to a max of 1000 characters
        - Rewards are normalized to a range of 0.0 to 1.0
    """

    rewards: list[float] = []

    for completion in completions:
        _, total_chars = count_thoughts(get_content_by_role(completion, "assistant"))
        rewards.append(min(total_chars / 1000, 1.0))

    return rewards


def extract_tool_calls(completion: str) -> list[str]:
    """Extract tool calls from a completion string.

    Args:
        completion: String to analyze

    Returns:
        List of tool calls found in the completion
    """
    pattern = r"<\|tool_call\|>(.*?)<\|end_tool_call\|>"
    return re.findall(pattern, completion, re.DOTALL)


def has_tool_call_reward_function(
    *, completions: list[list[dict[str, Any]]], **kwargs: Any
) -> list[float]:
    """Calculate rewards based on presence of tool calls in completions.

    Assigns a binary reward (1.0 or 0.0) to each completion based on whether
    it contains a tool call tag.
    """

    rewards: list[float] = []

    for completion in completions:
        tool_calls = extract_tool_calls(get_content_by_role(completion, "assistant"))
        rewards.append(1.0 if len(tool_calls) == 1 else 0.0)

    return rewards


def tool_call_is_valid_json_reward_function(
    *, completions: list[list[dict[str, Any]]], **kwargs: Any
) -> list[float]:
    """Calculate rewards based on whether tool calls are valid JSON."""

    rewards: list[float] = []

    for completion in completions:
        tool_calls = extract_tool_calls(get_content_by_role(completion, "assistant"))

        if len(tool_calls) == 1:
            tool_call = tool_calls[0]

            try:
                json.loads(tool_call)
                rewards.append(1.0)
            except json.JSONDecodeError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards


def validate_tool_call_against_schema(tool_call: str, schema: dict[str, Any]) -> bool:
    """Validate a tool call string against a JSON schema.

    Args:
        tool_call: String containing the tool call to validate
        schema: JSON schema to validate against

    Returns:
        True if the tool call is valid JSON and matches the schema, False otherwise
    """
    try:
        tool_call_json = json.loads(tool_call)
        validate(instance=tool_call_json, schema=schema)
        return True
    except (json.JSONDecodeError, ValidationError):
        return False


def tool_call_is_correct_schema_reward_function(
    *, prompts: list[list[dict[str, Any]]], completions: list[list[dict[str, Any]]], **kwargs: Any
) -> list[float]:
    """Calculate rewards based on whether tool calls are valid JSON."""

    rewards: list[float] = []

    for prompt, completion in zip(prompts, completions, strict=True):
        tool_catalog_str = get_content_by_role(prompt, "tool_catalog")
        tool_catalog_json_schema = json.loads(tool_catalog_str)
        content = get_content_by_role(completion, "assistant")
        tool_calls = extract_tool_calls(content)

        if len(tool_calls) == 1 and validate_tool_call_against_schema(
            tool_calls[0], tool_catalog_json_schema
        ):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def combine_reward_functions(
    reward_functions: list[tuple[Callable[..., list[float]], float]],
    *,
    prompts: list[list[dict[str, Any]]] | None = None,
    completions: list[list[dict[str, Any]]],
    **kwargs: Any,
) -> list[float]:
    """Combine multiple reward functions with weights into a single reward.

    Args:
        reward_functions: List of tuples containing (reward_function, weight)
        prompts: Optional list of prompts if needed by any reward functions
        completions: List of completions to evaluate
        **kwargs: Additional arguments passed to reward functions

    Returns:
        List of combined rewards normalized between 0.0 and 1.0
    """
    if not reward_functions:
        return [0.0] * len(completions)

    # Normalize weights to sum to 1.0
    functions, weights = zip(*reward_functions, strict=False)
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Calculate combined rewards
    combined_rewards: list[float] = [0.0] * len(completions)

    for func, weight in zip(functions, normalized_weights, strict=False):
        if "prompts" in func.__code__.co_varnames and prompts is not None:
            rewards = func(prompts=prompts, completions=completions, **kwargs)
        else:
            rewards = func(completions=completions, **kwargs)

        for i, reward in enumerate(rewards):
            combined_rewards[i] += reward * weight

    return combined_rewards


def cot_with_tool_call_reward_function(
    *, prompts: list[list[dict[str, Any]]], completions: list[list[dict[str, Any]]], **kwargs: Any
) -> list[float]:
    return combine_reward_functions(
        reward_functions=[
            (cot_has_thoughts_reward_function, 0.25),
            (cot_thoughts_length_reward_function, 0.1),
            (has_tool_call_reward_function, 0.33),
            (tool_call_is_valid_json_reward_function, 0.66),
            (tool_call_is_correct_schema_reward_function, 1.0),
        ],
        prompts=prompts,
        completions=completions,
    )
