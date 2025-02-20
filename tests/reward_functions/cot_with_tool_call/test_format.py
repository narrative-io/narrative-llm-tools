from typing import Any
import pytest
from narrative_llm_tools.reward_functions.cot_with_tool_call import (
    RewardFn,
    StringOrMessages,
    format_reward,
    thought_steps_reward,
    get_repetition_penalty_reward,
    get_first_message_content_by_role,
    count_thoughts,
    combine_rewards,
    get_default_reward_function,
    validate_reward_fn_inputs,
    validate_tool_call_against_schema
)

# Test data
mock_json_schema = {
  "title": "Math Function Call Array",
  "description": "A list of simple math function definitions to execute",
  "type": "array",
  "items": {
    "title": "Function Definition",
    "description": "Define either an add or subtract operation",
    "type": "object",
    "properties": {
      "name": {
        "title": "Function Name",
        "description": "The identifier for the function to be invoked",
        "type": "string",
        "enum": ["add", "subtract"]
      },
      "parameters": {
        "type": "object",
        "properties": {
          "a": {
            "title": "First Number",
            "description": "The first number in the operation",
            "type": "number"
          },
          "b": {
            "title": "Second Number",
            "description": "The second number in the operation",
            "type": "number"
          }
        },
        "required": ["a", "b"]
      }
    },
    "required": ["name", "parameters"]
  }
}

class SimpleRewardFn(RewardFn):
    """Simple reward function that returns 1.0 for non-empty messages and 0.0 for empty ones."""
    def __call__(
        self,
        completions: list[StringOrMessages],
        prompts: list[StringOrMessages] | None = None,
        **kwargs: list[Any]
    ) -> list[float]:
        rewards = []
        for completion in completions:
            if isinstance(completion, str):
                rewards.append(1.0 if completion else 0.0)
            else:
                # For message lists, check if there's any assistant message with content
                has_content = any(
                    msg.get("role") == "assistant" and msg.get("content")
                    for msg in completion
                )
                rewards.append(1.0 if has_content else 0.0)
        return rewards


def test_reward_fn_protocol():
    """Test that our implementation correctly follows the RewardFn protocol."""
    reward_fn = SimpleRewardFn()
    assert isinstance(reward_fn, RewardFn)


def test_reward_fn_with_string_completions():
    """Test reward function with string completions."""
    reward_fn = SimpleRewardFn()
    completions = ["Hello", "", "World"]

    rewards = reward_fn(completions=completions)

    assert len(rewards) == len(completions)
    assert rewards == [1.0, 0.0, 1.0]


def test_reward_fn_with_message_completions():
    """Test reward function with message list completions."""
    reward_fn = SimpleRewardFn()
    completions = [
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ],
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""}
        ],
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"}
        ]
    ]

    rewards = reward_fn(completions=completions)

    assert len(rewards) == len(completions)
    assert rewards == [1.0, 0.0, 1.0]


def test_reward_fn_with_prompts():
    """Test reward function with both completions and prompts."""
    reward_fn = SimpleRewardFn()
    completions = ["Hello", "World"]
    prompts = ["What's your greeting?", "Say something"]

    rewards = reward_fn(completions=completions, prompts=prompts)

    assert len(rewards) == len(completions)
    assert rewards == [1.0, 1.0]


def test_reward_fn_with_kwargs():
    """Test reward function with additional kwargs."""
    reward_fn = SimpleRewardFn()
    completions = ["Hello"]

    rewards = reward_fn(
        completions=completions,
        extra_param=["something"],
        another_param=["else"]
    )

    assert len(rewards) == len(completions)
    assert rewards == [1.0]

def test_validate_reward_fn_inputs_valid():
    """Test validate_reward_fn_inputs with valid inputs."""
    completions = ["comp1", "comp2", "comp3"]
    prompts = ["prompt1", "prompt2", "prompt3"]
    kwargs = {
        "scores": [1.0, 2.0, 3.0],
        "metadata": ["a", "b", "c"]
    }

    # Should not raise any exceptions
    validate_reward_fn_inputs(completions, prompts, **kwargs)

    # Test with None prompts
    validate_reward_fn_inputs(completions, None, **kwargs)

    # Test with empty kwargs
    validate_reward_fn_inputs(completions, prompts, **{})

def test_validate_reward_fn_inputs_invalid_prompts():
    """Test validate_reward_fn_inputs with mismatched prompts length."""
    completions = ["comp1", "comp2", "comp3"]
    prompts = ["prompt1", "prompt2"]  # One less than completions

    with pytest.raises(ValueError, match=r"prompts length \(2\) != completions length \(3\)"):
        validate_reward_fn_inputs(completions, prompts, **{})

def test_validate_reward_fn_inputs_invalid_kwargs():
    """Test validate_reward_fn_inputs with invalid kwargs."""
    completions = ["comp1", "comp2", "comp3"]

    # Test with mismatched kwargs length
    kwargs_wrong_length = {"scores": [1.0, 2.0]}  # One less than completions
    with pytest.raises(ValueError, match=r"kwargs\[scores\] length \(2\) != completions length \(3\)"):
        validate_reward_fn_inputs(completions, None, **kwargs_wrong_length)

def test_validate_reward_fn_inputs_multiple_kwargs():
    """Test validate_reward_fn_inputs with multiple kwargs having different issues."""
    completions = ["comp1", "comp2"]
    kwargs = {
        "valid": [1.0, 2.0],  # Valid
        "wrong_type": "not a list",  # Invalid type
        "wrong_length": [1.0],  # Wrong length
    }

    # Fix wrong_type and test wrong_length
    kwargs["wrong_type"] = [1.0, 2.0]
    with pytest.raises(ValueError, match=r"kwargs\[wrong_length\] length \(1\) != completions length \(2\)"):
        validate_reward_fn_inputs(completions, None, **kwargs)

def test_get_content_by_role():
    messages = [
        {"role": "system", "content": "system message"},
        {"role": "user", "content": "user message"},
        {"role": "assistant", "content": "assistant message"}
    ]

    assert get_first_message_content_by_role(messages, "assistant") == "assistant message"
    assert get_first_message_content_by_role(messages, "user") == "user message"
    assert get_first_message_content_by_role(messages, "nonexistent") == ""
    assert get_first_message_content_by_role([], "assistant") == ""

def test_count_thoughts():
    text = """<|start_thought|>First thought<|end_thought|>
    <|start_thought|>Second thought<|end_thought|>"""
    count, chars = count_thoughts(text)
    assert count == 2
    assert chars == len("First thought") + len("Second thought")

    # Empty text
    count, chars = count_thoughts("")
    assert count == 0
    assert chars == 0

    # Malformed tags
    count, chars = count_thoughts("<|start_thought|>Incomplete")
    assert count == 0
    assert chars == 0

def test_thought_steps_reward():
    # Test with varying number of thoughts
    single_thought = [{
        "role": "assistant",
        "content": "<|start_thought|>One thought<|end_thought|>"
    }]

    three_thoughts = [{
        "role": "assistant",
        "content": """
        <|start_thought|>First thought<|end_thought|>
        <|start_thought|>Second thought<|end_thought|>
        <|start_thought|>Third thought<|end_thought|>
        """
    }]

    rewards = thought_steps_reward(completions=[single_thought, three_thoughts])
    assert len(rewards) == 2
    assert rewards[0] < rewards[1]  # Three thoughts should score higher

def test_repetition_penalty_reward():
    penalty_func = get_repetition_penalty_reward(ngram_size=2)

    # Test with repetitive content
    repetitive = [{
        "role": "assistant",
        "content": """
        <|start_thought|>The cat the cat the cat<|end_thought|>
        """
    }]

    # Test with varied content
    varied = [{
        "role": "assistant",
        "content": """
        <|start_thought|>The cat sat on the mat<|end_thought|>
        """
    }]

    rewards = penalty_func([repetitive, varied])
    assert rewards[0] < rewards[1]  # Repetitive content should be penalized

def test_format_reward():
    completion = [{
        "role": "assistant",
        "content": "<|start_thought|>Test thought<|end_thought|>\n<|tool_calls|>[]\n<|eot_id|>"
    }]

    assert format_reward([completion]) == [1.0]

def test_format_reward_with_no_thoughts():
    completion = [{
        "role": "assistant",
        "content": "<|tool_calls|>[]\n<|eot_id|>"
    }]

    assert format_reward([completion]) == [1.0]

def test_format_reward_with_thoughts_after_tool_calls():
    completion = [{
        "role": "assistant",
        "content": "<|tool_calls|>[]<|start_thought|>Test thought<|end_thought|><|eot_id|>"
    }]

    assert format_reward([completion]) == [0.0]

def test_combine_rewards():
    completion = [{
        "role": "assistant",
        "content": "<|start_thought|>Test thought<|end_thought|>\n<|tool_calls|>[]\n<|eot_id|>"
    }]

    reward_functions = [
        (format_reward, 0.5),
        (thought_steps_reward, 0.5)
    ]

    rewards = combine_rewards(reward_functions, [completion])
    assert len(rewards) == 1
    assert 0 <= rewards[0] <= 1.0

def test_default_reward_function():
    default_func = get_default_reward_function()

    valid_completion = [{
        "role": "assistant",
        "content": """
        <|start_thought|>First thought<|end_thought|>
        <|tool_calls|>[{
            "name": "test_tool",
            "parameters": {
                "attribute_id": "123",
                "expression": "test",
                "type": "string"
            }
        }]
        <|eot_id|>
        """
    }]

    rewards = default_func(completions=[valid_completion])
    assert len(rewards) == 1
    assert 0 <= rewards[0] <= 1.0

def test_validate_tool_call_against_schema():
    valid_add = '''[{
        "name": "add",
        "parameters": {
            "a": 5,
            "b": 3
        }
    }]'''
    assert validate_tool_call_against_schema(valid_add, mock_json_schema) is True

    # Test valid subtract function
    valid_subtract = '''[{
        "name": "subtract",
        "parameters": {
            "a": 10,
            "b": 4
        }
    }]'''
    assert validate_tool_call_against_schema(valid_subtract, mock_json_schema) is True

    # Test invalid function name
    invalid_function = '''[{
        "name": "multiply",
        "parameters": {
            "a": 5,
            "b": 3
        }
    }]'''
    assert validate_tool_call_against_schema(invalid_function, mock_json_schema) is False

    # Test invalid parameter types
    invalid_params = '''[{
        "name": "add",
        "parameters": {
            "a": "5",
            "b": "3"
        }
    }]'''
    assert validate_tool_call_against_schema(invalid_params, mock_json_schema) is False

    # Test missing required parameter
    missing_param = '''[{
        "name": "add",
        "parameters": {
            "a": 5
        }
    }]'''
    assert validate_tool_call_against_schema(missing_param, mock_json_schema) is False

    # Test multiple valid operations
    multiple_valid = '''[
        {
            "name": "add",
            "parameters": {
                "a": 5,
                "b": 3
            }
        },
        {
            "name": "subtract",
            "parameters": {
                "a": 10,
                "b": 4
            }
        }
    ]'''
    assert validate_tool_call_against_schema(multiple_valid, mock_json_schema) is True

    # Test invalid JSON syntax
    invalid_json = '''[{
        "name": "add",
        "parameters": {
            "a": 5,
            "b": 3,
        }
    }]'''  # Note the extra comma
    assert validate_tool_call_against_schema(invalid_json, mock_json_schema) is False

    # Test empty array
    empty_array = '[]'
    assert validate_tool_call_against_schema(empty_array, mock_json_schema) is True

    # Test non-array input
    non_array = '''{
        "name": "add",
        "parameters": {
            "a": 5,
            "b": 3
        }
    }'''
    assert validate_tool_call_against_schema(non_array, mock_json_schema) is False


def test_combine_rewards_empty_functions():
    # Setup
    reward_functions = []
    completions = [
        "Some completion 1",
        "Some completion 2",
        "Some completion 3"
    ]

    # Execute
    result = combine_rewards(reward_functions, completions)

    # Assert
    assert len(result) == len(completions)
    assert all(reward == 0.0 for reward in result)
    assert result == [0.0, 0.0, 0.0]
