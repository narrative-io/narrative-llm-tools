from narrative_llm_tools.reward_functions.cot_with_tool_call.format import cot_has_thoughts_reward_function, count_thoughts, extract_tool_calls, get_content_by_role, tool_call_is_valid_json_reward_function, validate_tool_call_against_schema

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

def test_cot_has_thoughts_reward_function():
    # Test case with thoughts
    completion_with_thoughts = [{
        "role": "assistant",
        "content": "Let me think about this. <|start_thought|>This is a thought process<|end_thought|> Here's my answer."
    }]
    
    # Test case without thoughts
    completion_without_thoughts = [{
        "role": "assistant",
        "content": "Here's my answer without any thoughts."
    }]
    
    # Test both cases together
    completions = [completion_with_thoughts, completion_without_thoughts]
    
    rewards = cot_has_thoughts_reward_function(completions=completions)
    
    assert len(rewards) == 2
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0
    
def test_get_content_by_role_finds_matching_message():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "tool_catalog", "content": "Available tools"},
    ]
    
    assert get_content_by_role(messages, "user") == "Hello"
    assert get_content_by_role(messages, "assistant") == "Hi there"
    assert get_content_by_role(messages, "tool_catalog") == "Available tools"


def test_get_content_by_role_returns_empty_for_missing_role():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    
    assert get_content_by_role(messages, "nonexistent_role") == ""


def test_get_content_by_role_handles_empty_messages():
    assert get_content_by_role([], "any_role") == ""


def test_get_content_by_role_handles_missing_content():
    messages = [
        {"role": "user"},  # Missing content field
        {"role": "assistant", "content": "Hi there"},
    ]
    
    assert get_content_by_role(messages, "user") == ""


def test_get_content_by_role_returns_first_matching_role():
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Second"},  # Second user message
    ]
    
    assert get_content_by_role(messages, "user") == "First"
    
def test_count_thoughts_empty_string():
    assert count_thoughts("") == (0, 0)


def test_count_thoughts_no_thoughts():
    text = "This is a text without any thought tags"
    assert count_thoughts(text) == (0, 0)


def test_count_thoughts_single_thought():
    text = "Before <|start_thought|>This is a thought<|end_thought|> After"
    assert count_thoughts(text) == (1, 17)  # "This is a thought" is 16 chars


def test_count_thoughts_multiple_thoughts():
    text = """
    <|start_thought|>First thought<|end_thought|>
    Middle text
    <|start_thought|>Second thought<|end_thought|>
    """
    assert count_thoughts(text) == (2, 27)  # "First thought" + "Second thought" = 23 chars


def test_count_thoughts_multiline_thought():
    text = """
    <|start_thought|>This is a
    multiline thought
    with three lines<|end_thought|>
    """
    assert count_thoughts(text) == (1, 52)


def test_count_thoughts_nested_tags():
    text = """
    <|start_thought|>Outer
    <|start_thought|>Inner<|end_thought|>
    thought<|end_thought|>
    """
    assert count_thoughts(text) == (1, 32)


def test_count_thoughts_incomplete_tags():
    text = """
    <|start_thought|>Incomplete thought
    <|start_thought|>Complete thought<|end_thought|>
    """
    assert count_thoughts(text) == (1, 56)
    
def test_extract_single_tool_call():
    completion = """Some text before
<|tool_call|>{"name": "test"}|<|end_tool_call|>
Some text after"""
    
    result = extract_tool_calls(completion)
    assert len(result) == 1
    assert result[0] == '{"name": "test"}|'

def test_extract_multiple_tool_calls():
    completion = """
<|tool_call|>{"first": "call"}<|end_tool_call|>
Middle text
<|tool_call|>{"second": "call"}<|end_tool_call|>
"""
    result = extract_tool_calls(completion)
    assert len(result) == 2
    assert result[0] == '{"first": "call"}'
    assert result[1] == '{"second": "call"}'

def test_no_tool_calls():
    completion = "Just some regular text without any tool calls"
    result = extract_tool_calls(completion)
    assert len(result) == 0
    assert result == []

def test_multiline_tool_call():
    completion = """<|tool_call|>{
    "name": "test",
    "parameters": {
        "key": "value"
    }
}<|end_tool_call|>"""
    result = extract_tool_calls(completion)
    assert len(result) == 1
    assert result[0].strip() == '{\n    "name": "test",\n    "parameters": {\n        "key": "value"\n    }\n}'


def test_nested_tool_calls():
    completion = """<|tool_call|>{
    "outer": "call",
    "nested": "<|tool_call|>nested<|end_tool_call|>"
}<|end_tool_call|>"""
    result = extract_tool_calls(completion)
    assert len(result) == 1
    assert "outer" in result[0]
    assert "nested" in result[0]
    
def test_valid_json_tool_call():
    completions = [[{
        "role": "assistant",
        "content": "Here's my tool call: <|tool_call|>{\"name\": \"test\"}<|end_tool_call|>"
    }]]
    
    rewards = tool_call_is_valid_json_reward_function(completions=completions)
    assert rewards == [1.0]


def test_invalid_json_tool_call():
    completions = [[{
        "role": "assistant",
        "content": "Here's my tool call: <|tool_call|>{invalid json}<|end_tool_call|>"
    }]]
    
    rewards = tool_call_is_valid_json_reward_function(completions=completions)
    assert rewards == [0.0]


def test_no_tool_call():
    completions = [[{
        "role": "assistant",
        "content": "No tool call here"
    }]]
    
    rewards = tool_call_is_valid_json_reward_function(completions=completions)
    assert rewards == [0.0]


def test_multiple_tool_calls():
    completions = [[{
        "role": "assistant",
        "content": """
            <|tool_call|>{\"first\": \"call\"}<|end_tool_call|>
            <|tool_call|>{\"second\": \"call\"}<|end_tool_call|>
        """
    }]]
    
    rewards = tool_call_is_valid_json_reward_function(completions=completions)
    assert rewards == [0.0]

def test_validate_tool_call_against_schema():
    tool_call = '[{"name": "add", "parameters": {"a": 1, "b": 2}}]'
    assert validate_tool_call_against_schema(tool_call, mock_json_schema) == True
    
    tool_call = '[{"name": "add", "parameters": {"a": 1, "b": 2}}, {"name": "subtract", "parameters": {"a": 1, "b": 2}}]'
    assert validate_tool_call_against_schema(tool_call, mock_json_schema) == True
    
    tool_call = '{"name": "add", "parameters": {"a": 1, "b": 2}}'
    assert validate_tool_call_against_schema(tool_call, mock_json_schema) == False

    tool_call = '{"name": "test", "parameters": {"a": 1}}'
    assert validate_tool_call_against_schema(tool_call, mock_json_schema) == False
