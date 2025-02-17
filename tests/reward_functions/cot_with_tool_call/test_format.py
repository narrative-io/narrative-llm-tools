import pytest
from narrative_llm_tools.reward_functions.cot_with_tool_call.format import (
    format_reward,
    thought_steps_reward,
    tool_calls_validity_reward,
    get_repetition_penalty_reward,
    get_first_message_content_by_role,
    count_thoughts,
    combine_rewards,
    get_default_reward_function
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

def test_tool_calls_validity_reward():
    valid_tool_call = [{
        "role": "assistant",
        "content": """
        <|start_thought|>Thought<|end_thought|>
        <|tool_calls|>[{
            "name": "test_tool",
            "parameters": {
                "attribute_id": "123",
                "expression": "test",
                "type": "string"
            }
        }]
        """
    }]
    
    invalid_tool_call = [{
        "role": "assistant",
        "content": """
        <|start_thought|>Thought<|end_thought|>
        <|tool_calls|>[{
            "name": "test_tool",
            "parameters": {
                "missing_required": "field"
            }
        }]
        """
    }]
    
    rewards = tool_calls_validity_reward([valid_tool_call, invalid_tool_call])
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0

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

@pytest.mark.parametrize("invalid_input", [
    {"role": "assistant"},  # Missing content
    {"content": ""},  # Missing role
])
def test_edge_cases(invalid_input):
    """Test various edge cases with invalid inputs"""
    # Fix: Properly wrap the invalid input as a completion
    completion = [[invalid_input]] if invalid_input is not None else [[]]
    
    # Test each function with invalid input
    assert format_reward(completion) == [0.0]
    assert thought_steps_reward(completion) == [0.0]
    assert tool_calls_validity_reward(completion) == [0.0]
    assert get_repetition_penalty_reward()(completion) == [0.0]