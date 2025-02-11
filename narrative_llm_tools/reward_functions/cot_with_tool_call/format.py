import re
import json
from typing import Iterator, List, Dict, Any, Tuple, Callable, Optional
from jsonschema import ValidationError, validate
from collections.abc import Callable


def get_content_by_role(messages: List[Dict[str, Any]], role: str) -> str:
    """Extract content from the first message matching the specified role."""
    message = next((msg for msg in messages if msg.get("role", "") == role), None)
    return message.get("content", "") if message else ""


def count_thoughts(text: str) -> Tuple[int, int]:
    """Count thought segments and their total character length."""
    pattern = r"<\|start_thought\|>(.*?)<\|end_thought\|>"
    matches = re.findall(pattern, text, re.DOTALL)
    total_chars = sum(len(match) for match in matches)
    return len(matches), total_chars


def format_reward(completions: List[List[Dict[str, Any]]], **kwargs: Any) -> List[float]:
    """Reward function that checks if the completion follows the expected format."""
    pattern = (
        r"^(?:<\|start_thought\|>.*?<\|end_thought\|>\s*)*"  # Multiple thought blocks
        r"<\|tool_calls\|>\[.*?\]"  # Tool calls block
        r"<\|eot_id\|>$"  # End tag
    )
    
    rewards = []
    for completion in completions:
        content = get_content_by_role(completion, "assistant")
        match = bool(re.match(pattern, content, re.DOTALL | re.MULTILINE))
        rewards.append(1.0 if match else 0.0)
    
    return rewards


def thought_steps_reward(completions: List[List[Dict[str, Any]]], **kwargs: Any) -> List[float]:
    """Reward function that checks for the presence and quality of thought steps."""
    rewards = []
    
    for completion in completions:
        content = get_content_by_role(completion, "assistant")
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


def tool_calls_validity_reward(
    completions: List[List[Dict[str, Any]]], 
    prompts: Optional[List[List[Dict[str, Any]]]] = None,
    **kwargs: Any
) -> List[float]:
    """Reward function that validates the structure and content of tool calls."""
    rewards = []
    
    for i, completion in enumerate(completions):
        content = get_content_by_role(completion, "assistant")
        try:
            # Extract tool calls section
            tool_calls_match = re.search(r'<\|tool_calls\|>(\[.*?\])', content, re.DOTALL)
            if not tool_calls_match:
                rewards.append(0.0)
                continue
                
            tool_calls = json.loads(tool_calls_match.group(1))
            
            # Basic structure validation
            if not isinstance(tool_calls, list):
                rewards.append(0.0)
                continue
            
            # Schema validation if provided
            if prompts:
                tool_catalog_str = get_content_by_role(prompts[i], "tool_catalog")
                if tool_catalog_str:
                    try:
                        schema = json.loads(tool_catalog_str)
                        if not validate_tool_call_against_schema(tool_calls_match.group(1), schema):
                            rewards.append(0.0)
                            continue
                    except json.JSONDecodeError:
                        pass  # Skip schema validation if catalog is invalid
            
            # Validate each tool call structure
            valid_calls = all(
                isinstance(call, dict) and
                "name" in call and
                "parameters" in call and
                isinstance(call["parameters"], dict) and
                "attribute_id" in call["parameters"] and
                "expression" in call["parameters"] and
                "type" in call["parameters"]
                for call in tool_calls
            )
            
            rewards.append(1.0 if valid_calls else 0.0)
            
        except json.JSONDecodeError:
            rewards.append(0.0)
            
    return rewards


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.5) -> Callable[[List[List[Dict[str, Any]]], Dict[str, Any]], List[float]]:
    """Creates a reward function that penalizes repetitive content in thoughts."""
    
    def zipngram(text: str, ngram_size: int) -> Iterator[Tuple[str, ...]]:
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])
    
    def repetition_penalty_reward(completions: List[List[Dict[str, Any]]], **kwargs: Any) -> List[float]:
        rewards = []
        
        for completion in completions:
            content = get_content_by_role(completion, "assistant")
            thoughts = re.findall(
                r'<\|start_thought\|>(.*?)<\|end_thought\|>', 
                content, 
                re.DOTALL
            )
            
            if not thoughts:
                rewards.append(0.0)
                continue
                
            combined_thoughts = ' '.join(thoughts)
            
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
    reward_functions: List[Tuple[Callable[..., List[float]], float]],
    completions: List[List[Dict[str, Any]]],
    prompts: Optional[List[List[Dict[str, Any]]]] = None,
    **kwargs: Any
) -> List[float]:
    """Combines multiple reward functions with weights."""
    if not reward_functions:
        return [0.0] * len(completions)

    # Normalize weights to sum to 1.0
    functions, weights = zip(*reward_functions)
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Calculate combined rewards
    combined_rewards = [0.0] * len(completions)

    for func, weight in zip(functions, normalized_weights):
        if "prompts" in func.__code__.co_varnames and prompts is not None:
            rewards = func(prompts=prompts, completions=completions, **kwargs)
        else:
            rewards = func(completions=completions, **kwargs)

        for i, reward in enumerate(rewards):
            combined_rewards[i] += reward * weight

    return combined_rewards


def get_default_reward_function(
    *, include_schema_validation: bool = True
) -> Callable[..., List[float]]:
    """Returns a default reward function with pre-configured weights."""
    
    def default_reward_function(
        *, 
        prompts: Optional[List[List[Dict[str, Any]]]] = None,
        completions: List[List[Dict[str, Any]]],
        **kwargs: Any
    ) -> List[float]:
        reward_functions = [
            (format_reward, 0.25),
            (thought_steps_reward, 0.35),
            (tool_calls_validity_reward, 0.75),
            (get_repetition_penalty_reward(), 0.15)
        ]
        
        return combine_rewards(
            reward_functions=reward_functions,
            prompts=prompts,
            completions=completions,
            **kwargs
        )
    
    return default_reward_function