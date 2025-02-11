from narrative_llm_tools.reward_functions.cot_with_tool_call.format import format_reward

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

def test_format_reward():
    # Test valid format
    valid_completion = [{
        "role": "assistant",
        "content": "<|start_thought|>Testing thought<|end_thought|>\n<|tool_calls|>[]\n<|eot_id|>"
    }]
    
    # Test invalid format (missing eot_id)
    invalid_completion = [{
        "role": "assistant",
        "content": "<|start_thought|>Testing thought<|end_thought|>\n<|tool_calls|>[]"
    }]
    
    # Test multiple thoughts
    multiple_thoughts = [{
        "role": "assistant",
        "content": "<|start_thought|>First thought<|end_thought|>\n<|start_thought|>Second thought<|end_thought|>\n<|tool_calls|>[]\n<|eot_id|>"
    }]
    
    completions = [valid_completion, invalid_completion, multiple_thoughts]
    rewards = format_reward(completions)
    
    assert len(rewards) == 3
    assert rewards[0] == 1.0  # Valid format
    assert rewards[1] == 0.0  # Invalid format
    assert rewards[2] == 1.0  # Multiple thoughts valid