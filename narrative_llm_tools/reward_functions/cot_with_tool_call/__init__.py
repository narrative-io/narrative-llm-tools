import json
import re
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from jsonschema import ValidationError, validate

StringOrMessage = str | list[dict[str, Any]]


@runtime_checkable
class RewardFn(Protocol):
    """Protocol defining the interface for reward functions in a completion evaluation system.

    This protocol specifies the call signature for reward functions that evaluate model
    completions and assign numerical scores. Reward functions implementing this protocol
    are used to assess various aspects of model outputs, such as format compliance,
    content quality, or specific behavioral metrics.

    Call Signature:
        __call__(completions: list[list[dict[str, Any]]],
            **kwargs: dict[str, list[Any]]) -> list[float]

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
        **kwargs: dict[str, list[Any]],
    ) -> list[float]: ...


def validate_reward_fn_inputs(
    completions: list[StringOrMessage],
    prompts: list[StringOrMessage] | None,
    **kwargs: dict[str, list[Any]],
) -> None:
    """Validates that input lists for reward function evaluation have consistent lengths.

    This function ensures that all input sequences provided to a reward function have the
    same length as the completions list. This is crucial for maintaining consistency when
    processing multiple samples in parallel, where each index position across all inputs
    should correspond to the same sample.

    Args:
        completions (list[StringOrMessage]): A list of model completions to be evaluated.
            Each element can be either a string or a Message object. This list serves as
            the reference length for validation.

        prompts (list[StringOrMessage] | None): Optional list of prompts corresponding to
            the completions. If provided, must have the same length as completions. Each
            element can be either a string or a Message object.

        **kwargs (dict[str, list[Any]]): Additional keyword arguments where each value
            must be a list with the same length as completions. This allows for flexible
            inclusion of supplementary data needed for reward computation (e.g., reference
            answers, metadata, etc.).

    Raises:
        ValueError: If any input list has a different length than the completions list.
            The error message will specify which input (prompts or kwargs key) caused
            the mismatch and include the mismatched lengths.

    Example:
        >>> completions = ["response1", "response2"]
        >>> prompts = ["prompt1", "prompt2"]
        >>> metadata = ["meta1", "meta2"]
        >>> # This will not raise an error - all lengths match
        >>> validate_reward_fn_inputs(completions, prompts, metadata=metadata)

        >>> # This will raise ValueError - prompts length doesn't match
        >>> validate_reward_fn_inputs(completions, ["single_prompt"], metadata=metadata)
        ValueError: prompts length (1) != completions length (2)
    """
    completion_len = len(completions)
    if prompts is not None and len(prompts) != completion_len:
        raise ValueError(
            f"prompts length ({len(prompts)}) != completions length ({completion_len})"
        )

    for key, value in kwargs.items():
        if len(value) != completion_len:
            raise ValueError(
                f"kwargs[{key}] length ({len(value)}) != completions length ({completion_len})"
            )


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
    **kwargs: dict[str, list[Any]],
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
    validate_reward_fn_inputs(completions, prompts, **kwargs)
    pattern = (
        r"^(?:<\|start_thought\|>[^<]*?<\|end_thought\|>\s*)*"
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
    **kwargs: dict[str, list[Any]],
) -> list[float]:
    """Evaluates the quality of thought step reasoning in model responses.

    This reward function assesses responses based on two key metrics:
    1. The number of distinct thought steps (weighted at 70%)
    2. The total length/detail of the reasoning (weighted at 30%)

    The function combines these metrics to produce a final score between 0 and 1 for each
    completion. It incentivizes structured thinking with at least 3 distinct thought steps
    and rewards detailed explanations up to 1000 characters.

    Args:
        completions (list[StringOrMessage]): List of model responses to evaluate. Each
            element can be either a string containing the response text directly, or a
            list of messages where the assistant's response should be extracted.

        prompts (list[StringOrMessage] | None, optional): List of prompts that generated
            the completions. Currently not used in scoring but maintained for consistency
            with reward function interface. Defaults to None.

        **kwargs (dict[str, list[Any]]): Additional keyword arguments. Currently not used
            in scoring but maintained for interface consistency.

    Returns:
        list[float]: A list of reward scores, one for each completion, where each score
            is between 0 and 1. Higher scores indicate better structured reasoning with:
            - More thought steps (optimal: 3 or more steps)
            - More detailed explanations (optimal: 1000+ characters)

    Scoring Details:
        - Thought Count Score (70% weight):
            * 1.0 for 3 or more thought steps
            * Proportional score (steps/3) for fewer steps

        - Length Score (30% weight):
            * 1.0 for 1000+ characters
            * Proportional score (chars/1000) for shorter responses

    Example:
        >>> completions = [
        ...     "Step 1: Analyze the problem\\nStep 2: Consider options\\nStep 3: Decide",
        ...     "Quick answer without steps"
        ... ]
        >>> rewards = thought_steps_reward(completions)
        >>> # First completion will score higher due to clear thought steps
        >>> # Second completion will score lower due to lack of steps

    Notes:
        - For message list inputs, only the first assistant message is evaluated
        - The function assumes thought steps are clearly delineated in the text
        - Empty or very short responses will receive low scores
        - Additional thought steps beyond 3 don't increase the thought count score
        - Very long responses don't increase the length score beyond 1000 chars
    """
    validate_reward_fn_inputs(completions, prompts, **kwargs)
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
    """Validates if a JSON tool call string conforms to a specified JSON schema.

    This function performs two-step validation:
    1. Checks if the input string is valid JSON
    2. Validates the parsed JSON against the provided schema using jsonschema.validate

    Args:
        tool_call (str): A string containing a JSON-formatted tool call to validate.
            Must be a valid JSON string that can be parsed by json.loads().

        schema (dict[str, Any]): A dictionary representing a JSON schema that defines
            the expected structure and types of the tool call. Should follow JSON Schema
            specification format.

    Returns:
        bool: True if both conditions are met:
            - The tool_call string is valid JSON
            - The parsed JSON conforms to the provided schema
            False in any of these cases:
            - The tool_call string is not valid JSON
            - The parsed JSON doesn't match the schema specification

    Example:
        >>> schema = {
        ...     "title": "Math Function Call Array",
        ...     "type": "array",
        ...     "items": {
        ...         "type": "object",
        ...         "properties": {
        ...             "name": {
        ...                 "type": "string",
        ...                 "enum": ["add", "subtract"]
        ...             },
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "a": {"type": "number"},
        ...                     "b": {"type": "number"}
        ...                 },
        ...                 "required": ["a", "b"]
        ...             }
        ...         },
        ...         "required": ["name", "parameters"]
        ...     }
        ... }

        # Valid tool calls
        >>> validate_tool_call_against_schema(
        ...     '[{"name": "add", "parameters": {"a": 5, "b": 3}}]',
        ...     schema
        ... )
        True

        # Invalid: wrong operation name
        >>> validate_tool_call_against_schema(
        ...     '[{"name": "multiply", "parameters": {"a": 5, "b": 3}}]',
        ...     schema
        ... )
        False

        # Invalid: missing required parameter
        >>> validate_tool_call_against_schema(
        ...     '[{"name": "add", "parameters": {"a": 5}}]',
        ...     schema
        ... )
        False

        # Invalid: malformed JSON
        >>> validate_tool_call_against_schema(
        ...     '[{bad json here}]',
        ...     schema
        ... )
        False

    Notes:
        - Uses jsonschema.validate() for schema validation
        - Silently catches both JSONDecodeError and ValidationError
        - Does not provide details about validation failures
        - Schema validation checks both structure (required fields) and
          value constraints (types, enums, etc.)
    """
    try:
        tool_call_json = json.loads(tool_call)
        validate(instance=tool_call_json, schema=schema)
        return True
    except (json.JSONDecodeError, ValidationError):
        return False


def tool_calls_validity_reward(
    completions: list[StringOrMessage],
    prompts: list[StringOrMessage] | None = None,
    **kwargs: dict[str, list[Any]],
) -> list[float]:
    """Evaluates the validity of tool calls in model responses against provided schemas.

    This reward function checks if tool calls within model responses are:
    1. Properly formatted within <|tool_calls|> and <|eot_id|> tags
    2. Valid JSON
    3. Conform to the schema provided in the prompt's tool catalog (if available)

    The function implements a binary reward system: responses receive either 1.0 for
    fully valid tool calls or 0.0 for any validation failures.

    Args:
        completions (list[StringOrMessage]): List of model responses to evaluate. Each
            element can be either:
            - A string containing the response text
            - A list of messages where the assistant's response should be extracted

        prompts (list[StringOrMessage] | None, optional): List of prompts that generated
            the completions. Important for schema validation as they may contain tool
            catalogs. If not provided, defaults to empty strings. Each element can be:
            - A string (will not enable schema validation)
            - A list of messages where tool_catalog message contains the schema

        **kwargs (dict[str, list[Any]]): Additional keyword arguments. Currently not
            used in the reward calculation but maintained for interface consistency.

    Returns:
        list[float]: A list of reward scores, one for each completion, where:
            - 1.0: Tool calls are present, well-formatted, and validate against schema
            - 0.0: Any of these cases:
                * No tool calls present
                * Tool calls improperly formatted
                * Invalid JSON in tool calls
                * Tool calls don't match schema
                * Any other validation failure

    Example Schema:
        {
          "title": "Math Function Call Array",
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string",
                "enum": ["add", "subtract"]
              },
              "parameters": {
                "type": "object",
                "properties": {
                  "a": {"type": "number"},
                  "b": {"type": "number"}
                },
                "required": ["a", "b"]
              }
            },
            "required": ["name", "parameters"]
          }
        }

    Example Usage:
        >>> schema_prompt = [
        ...     {"role": "tool_catalog", "content": '<schema_json_above>'}
        ... ]
        >>> completions = [
        ...     # Valid tool call - will receive 1.0
        ...     "<|tool_calls|>[{
        ...         \"name\": \"add\",
        ...         \"parameters\": {\"a\": 5, \"b\": 3}
        ...     }]<|eot_id|>",
        ...     # Invalid tool calls - will receive 0.0
        ...     "<|tool_calls|>[{
        ...         \"name\": \"multiply\",  # not in enum
        ...         \"parameters\": {\"a\": 5, \"b\": 3}
        ...     }]<|eot_id|>",
        ...     # Missing parameters - will receive 0.0
        ...     "<|tool_calls|>[{
        ...         \"name\": \"add\",
        ...         \"parameters\": {\"a\": 5}  # missing required 'b'
        ...     }]<|eot_id|>",
        ...     # Invalid JSON - will receive 0.0
        ...     "<|tool_calls|>[{bad json here}]<|eot_id|>"
        ... ]
        >>> rewards = tool_calls_validity_reward(completions, [schema_prompt] * 4)
        >>> # Returns [1.0, 0.0, 0.0, 0.0]

    Notes:
        - Uses strict=False in zip() for explicit handling of unequal length iterables
        - Silently catches all exceptions and returns 0.0 for any validation failure
        - Tool catalog schema must be valid JSON
        - For message lists, only first assistant/tool_catalog messages are used
        - Empty or missing prompts are handled gracefully
        - Schema validation checks both structure and value constraints (e.g., enums)
    """
    validate_reward_fn_inputs(completions, prompts, **kwargs)
    rewards = []
    prompts_iter: list[StringOrMessage] | list[None] = (
        prompts if prompts is not None else [""] * len(completions)
    )

    for completion, prompt in zip(completions, prompts_iter, strict=False):
        content = (
            get_first_message_content_by_role(completion, "assistant")
            if isinstance(completion, list)
            else completion
        )
        reward = 0.0

        try:
            tool_calls_match = re.search(r"<\|tool_calls\|>(.*?)<\|eot_id\|>", content, re.DOTALL)

            if tool_calls_match:
                tool_calls = json.loads(tool_calls_match.group(1))

                if isinstance(prompt, list):
                    tool_catalog_str = get_first_message_content_by_role(prompt, "tool_catalog")
                    if tool_catalog_str:
                        schema = json.loads(tool_catalog_str)
                        if validate_tool_call_against_schema(json.dumps(tool_calls), schema):
                            reward = 1.0

        except Exception:
            pass

        rewards.append(reward)

    return rewards


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.5) -> RewardFn:
    """Creates a reward function that penalizes repetitive content in thought steps.

    This factory function generates a reward function that evaluates text for repetitive
    patterns using n-gram analysis. It specifically targets content within thought tags
    (<|start_thought|> and <|end_thought|>) and assigns penalties based on the proportion
    of repeated n-grams.

    Args:
        ngram_size (int, optional): The size of n-grams to analyze for repetition.
            Larger values detect repetition of longer phrases. Defaults to 3, which
            catches repetition of three-word phrases.

        max_penalty (float, optional): The maximum negative reward that can be applied
            for highly repetitive content. Should be negative. Defaults to -0.5.

    Returns:
        RewardFn: A reward function with the following signature:
            (completions: list[StringOrMessage],
             prompts: list[StringOrMessage] | None = None,
             **kwargs: dict[str, list[Any]]) -> list[float]

    Calculation Details:
        1. Extracts text between <|start_thought|> and <|end_thought|> tags
        2. Combines all thoughts into a single text
        3. Generates n-grams of specified size
        4. Calculates repetition score as: 1 - (unique_ngrams / total_ngrams)
        5. Scales the final penalty by multiplying repetition score with max_penalty

    Reward Values:
        - 0.0: Returned for texts that:
            * Have no thought tags
            * Have fewer words than ngram_size
        - Between max_penalty and 0.0: Based on repetition level
            * More repetitive → Closer to max_penalty
            * Less repetitive → Closer to 0.0

    Example:
        >>> reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-0.3)
        >>> completions = [
        ...     "<|start_thought|>Each step is unique.|end_thought|>",
        ...     "<|start_thought|>Step step step.|end_thought|>"
        ... ]
        >>> rewards = reward_fn(completions)
        >>> # First completion will get ~0.0 (no repetition)
        >>> # Second completion will get closer to -0.3 (high repetition)

    Notes:
        - Case insensitive: converts all text to lowercase before analysis
        - Ignores content outside of thought tags
        - Uses word-level n-grams (splits on whitespace)
        - Empty or short texts receive 0.0 instead of a penalty
        - The helper function zipngram() creates n-grams efficiently using zip
    """

    def zipngram(text: str, ngram_size: int) -> Iterator[tuple[str, ...]]:
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)], strict=False)

    def repetition_penalty_reward(
        completions: list[StringOrMessage],
        prompts: list[StringOrMessage] | None = None,
        **kwargs: dict[str, list[Any]],
    ) -> list[float]:
        validate_reward_fn_inputs(completions, prompts, **kwargs)
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
    **kwargs: dict[str, list[Any]],
) -> list[float]:
    """Combines multiple reward functions using weighted averaging.

    This function takes a list of reward functions and their corresponding weights,
    applies each function to the input completions, and combines the results using
    weighted averaging. Weights are automatically normalized to sum to 1.0.

    Args:
        reward_functions (list[tuple[RewardFn, float]]): List of tuples, where each tuple
            contains:
            - A reward function with signature:
                (completions: list[StringOrMessage],
                 prompts: list[StringOrMessage] | None,
                 **kwargs: dict[str, list[Any]]) -> list[float]
            - A float weight indicating the relative importance of that function

        completions (list[StringOrMessage]): List of model responses to evaluate. Each
            element can be either a string or a Message object. All reward functions
            will be applied to these completions.

        prompts (list[StringOrMessage] | None, optional): List of prompts that generated
            the completions. Must have same length as completions if provided. Passed to
            each reward function. Defaults to None.

        **kwargs (dict[str, list[Any]]): Additional keyword arguments passed to each
            reward function. Each value must be a list with same length as completions.

    Returns:
        list[float]: A list of combined reward scores, one for each completion. Each
            score is a weighted average of individual reward function outputs.

    Calculation Details:
        1. Weights are normalized by dividing each by their sum
        2. Each reward function is applied to all completions
        3. For each completion, its rewards are multiplied by normalized weights
        4. Weighted rewards are summed for final combined score

    Example:
        >>> thought_reward = thought_steps_reward  # Rewards structured thinking
        >>> repetition_penalty = get_repetition_penalty_reward()  # Penalizes repetition
        >>> reward_fns = [
        ...     (thought_reward, 0.7),  # 70% weight on thought structure
        ...     (repetition_penalty, 0.3)  # 30% weight on avoiding repetition
        ... ]
        >>> completions = ["Step 1...", "Step 2..."]
        >>> combined_scores = combine_rewards(reward_fns, completions)

    Notes:
        - If reward_functions is empty, returns list of 0.0 with length matching completions
        - Uses strict=False in zip() for explicit handling of unequal length iterables
        - All reward functions receive identical inputs (completions, prompts, kwargs)
        - Input validation ensures all lists have matching lengths
        - Original weights don't need to sum to 1.0 - they will be normalized
    """
    validate_reward_fn_inputs(completions, prompts, **kwargs)
    if not reward_functions:
        return [0.0] * len(completions)

    # Normalize weights to sum to 1.0
    functions, weights = zip(*reward_functions, strict=False)
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Calculate combined rewards
    combined_rewards = [0.0] * len(completions)

    for func, weight in zip(functions, normalized_weights, strict=False):
        rewards = func(completions=completions, prompts=prompts, **kwargs)

        for i, reward in enumerate(rewards):
            combined_rewards[i] += reward * weight

    return combined_rewards


def get_default_reward_function(*, include_schema_validation: bool = True) -> RewardFn:
    """Creates a pre-configured reward function combining multiple evaluation criteria.

    This factory function returns a reward function that combines several aspects of
    response quality with pre-configured weights:
    - Format correctness (25%): Evaluates structural formatting
    - Thought steps (35%): Rewards clear reasoning steps
    - Tool call validity (75%): Ensures proper tool usage
    - Repetition penalty (15%): Discourages redundant content

    The combined weights are automatically normalized during reward calculation.

    Args:
        include_schema_validation (bool, optional): Keyword-only argument that controls
            whether tool call schema validation is included in the evaluation.
            Defaults to True.

    Returns:
        RewardFn: A reward function with the following signature:
            (completions: list[StringOrMessage],
             prompts: list[StringOrMessage] | None = None,
             **kwargs: dict[str, list[Any]]) -> list[float]

    The returned function evaluates completions using these components:
        1. format_reward (0.25):
           - Checks for proper formatting and structure
           - Ensures required elements are present

        2. thought_steps_reward (0.35):
           - Evaluates quality and quantity of reasoning steps
           - Rewards clear, structured thinking

        3. tool_calls_validity_reward (0.75):
           - Verifies proper tool usage and formatting
           - Ensures tool calls meet schema requirements

        4. repetition_penalty_reward (0.15):
           - Detects and penalizes redundant content
           - Uses 3-word ngrams for analysis

    Example:
        >>> reward_fn = get_default_reward_function()
        >>> completions = [
        ...     "Let me think step by step...<tool>valid_call</tool>",
        ...     "Quick answer without structure"
        ... ]
        >>> rewards = reward_fn(completions)
        >>> # First completion will score higher due to thought steps and valid tool use
        >>> # Second completion will score lower due to missing structure

    Notes:
        - Weights are automatically normalized during combination
        - Tool validation can be disabled if schema validation isn't needed
        - The repetition penalty uses default settings (3-word ngrams, -0.5 max penalty)
        - All component reward functions receive the same inputs
        - The function internally uses combine_rewards for weight handling
    """

    def default_reward_function(
        completions: list[StringOrMessage],
        prompts: list[StringOrMessage] | None = None,
        **kwargs: dict[str, list[Any]],
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
