# Reward Functions for Completion Evaluation

This repository defines a set of **reward functions** that can be used—individually or in combination—to evaluate model-generated responses (or "completions"). Each reward function implements a consistent interface and returns a list of float scores between 0 and 1, unless otherwise noted. These scores measure different aspects of a completion, such as format correctness, step-by-step reasoning, tool call validity, and more.

## Table of Contents
1. [Overview of the RewardFn Protocol](#overview-of-the-rewardfn-protocol)
2. [Reward Functions](#reward-functions)
    - [format_reward](#format_reward)
    - [thought_steps_reward](#thought_steps_reward)
    - [tool_calls_validity_reward](#tool_calls_validity_reward)
    - [get_repetition_penalty_reward](#get_repetition_penalty_reward)
    - [combine_rewards](#combine_rewards)
    - [get_default_reward_function](#get_default_reward_function)
3. [Examples and Usage](#examples-and-usage)


## Overview of the RewardFn Protocol

All reward functions implement (or conform to) the following protocol (originally defined by HuggingFace [here](https://github.com/huggingface/trl/blob/ba036576d4a62d91da0388b7e727f6656f4c08d7/trl/trainer/grpo_trainer.py#L108)):

```python
class RewardFn(Protocol):
    def __call__(
        self,
        completions: list[StringOrMessage],
        prompts: list[StringOrMessage] | None = None,
        **kwargs: dict[str, list[Any]],
    ) -> list[float]:
        ...
```

- **`completions`**: A list of model responses to evaluate. Each element can be either:
  - A **string** of text (the model's response), or
  - A **list of messages**, where each message is a dictionary with keys like `role` and `content`.
- **`prompts`**: An optional list of prompt texts or messages that correspond to each completion.
- **`**kwargs`**: Additional lists of arguments that may be needed for evaluation. Every list provided in `kwargs` must match the length of `completions`.

> **Return value**: A list of floating-point scores (typically) between 0.0 and 1.0 (unless otherwise noted). The length of this list matches `len(completions)`.

## Reward Functions

Below is a quick reference for the reward functions defined in this codebase. You can use them individually or chain them together with [`combine_rewards`](#combine_rewards).

### `format_reward`
**Purpose**: Checks whether a completion follows a strict format:
- Zero or more thought blocks:
  `<|start_thought|> ... <|end_thought|>`
- A mandatory tool calls block:
  `<|tool_calls|>[ ... ]`
- An end-of-text marker:
  `<|eot_id|>`

**Return**: `1.0` if the format is correct; `0.0` otherwise.

<details>
  <summary>Key Points</summary>

  - Uses a regex to verify the presence and ordering of these tags.
  - Ideal for ensuring output structure, especially if your system needs well-defined JSON within `<|tool_calls|>`.

  **Example**:
  ```python
  completions = [
      # Correct format
      "<|start_thought|>I should check the weather<|end_thought|>\n<|tool_calls|>[{\"type\": \"function\"}]\n<|eot_id|>",
      # Incorrect format (missing <|eot_id|>)
      "<|start_thought|>Test<|end_thought|>\n<|tool_calls|>[]"
  ]
  scores = format_reward(completions)
  print(scores)  # [1.0, 0.0]
  ```
</details>

---

### `thought_steps_reward`
**Purpose**: Encourages **structured reasoning** by rewarding:
1. **Number of thought segments** (at least 3 is ideal).
2. **Total length of thought content** (up to 1000 characters).

**Return**: A float in `[0.0, 1.0]` for each completion. A higher score implies more thorough and well-structured reasoning.

<details>
  <summary>Scoring Breakdown</summary>

  - Thought Count Score (70% weight)
    - 1.0 for 3 or more thought segments (e.g., 3 `<|start_thought|> ... <|end_thought|>` blocks).
    - For fewer than 3 segments, score is `num_thoughts / 3`.

  - Length Score (30% weight)
    - 1.0 for 1000+ characters combined in all thoughts.
    - For shorter total length, score is `length / 1000`.

  Final = `0.7 * thought_count_score + 0.3 * length_score`.

  **Example**:
  ```python
  completions = [
      "<|start_thought|>First step<|end_thought|><|start_thought|>Second<|end_thought|><|start_thought|>Third<|end_thought|>",
      "<|start_thought|>Only one step<|end_thought|>"
  ]
  scores = thought_steps_reward(completions)
  print(scores)  # [~1.0, a lower value]
  ```
</details>

---

### `tool_calls_validity_reward`
**Purpose**: Validates the **tool call** portion of the completion:
1. Must appear between `<|tool_calls|>` and `<|eot_id|>`.
2. Must be valid JSON.
3. Optionally, must conform to a JSON schema if provided in the prompt.

**Return**: `1.0` if the tool call is valid; `0.0` otherwise.

<details>
  <summary>Key Points</summary>

  - Looks for `<|tool_calls|> ... <|eot_id|>` block in the completion.
  - If the corresponding **prompt** (or messages in the prompt) includes a `"tool_catalog"` role with a schema, it is used for validation.
  - Fails with `0.0` if the JSON is malformed or doesn't match the schema.

  **Example**:
  ```python
  # Suppose we have a prompt that includes a tool catalog schema
  prompt_with_schema = [
      {"role": "tool_catalog", "content": '{"type": "array", ...}'}
  ]
  completion = "<|tool_calls|>[{\"name\": \"add\", \"parameters\": {\"a\": 5, \"b\": 3}}]<|eot_id|>"

  score = tool_calls_validity_reward([completion], [prompt_with_schema])
  print(score)  # [1.0 if valid]
  ```
</details>

---

### `get_repetition_penalty_reward`
This function returns a **factory** that creates a reward function to penalize repeated **n-grams** within **thought segments**.

```python
def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.5) -> RewardFn
```
- **`ngram_size`**: The size of n-grams to analyze. Default is 3 (e.g., three-word sequences).
- **`max_penalty`**: The most negative penalty for high repetition. Default is `-0.5`.

**Resulting Reward Function**:
- Extracts text from `<|start_thought|> ... <|end_thought|>` segments.
- Calculates how many n-grams are repeated.
- Returns a score in `[max_penalty, 0.0]`. `0.0` means no repetition or insufficient words for n-gram analysis. Approaching `max_penalty` means heavy repetition.

<details>
  <summary>Example</summary>

  ```python
  repetition_reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-0.3)

  completions = [
      "<|start_thought|>Unique words each time<|end_thought|>",
      "<|start_thought|>Repeat repeat repeat<|end_thought|>"
  ]
  scores = repetition_reward_fn(completions)
  print(scores)
  # e.g., [0.0, ~-0.3]
  ```
</details>

---

### `combine_rewards`
**Purpose**: **Weighted** combination of multiple reward functions. Takes a list of `(reward_function, weight)` pairs, applies each function, and computes a normalized weighted average.

```python
def combine_rewards(
    reward_functions: list[tuple[RewardFn, float]],
    completions: list[StringOrMessage],
    prompts: list[StringOrMessage] | None = None,
    **kwargs: dict[str, list[Any]],
) -> list[float]
```

**Process**:
1. Normalize weights so they sum to 1.
2. Call each reward function to get scores.
3. Combine the scores by multiplying each function’s results by its normalized weight, then summing.

<details>
  <summary>Example</summary>

  ```python
  # Suppose we want to combine "format_reward" and "thought_steps_reward"
  reward_fns = [
      (format_reward, 0.4),
      (thought_steps_reward, 0.6)
  ]

  completions = [
      "<|start_thought|>Step 1<|end_thought|><|tool_calls|>[]<|eot_id|>",
      "Missing eot tag"
  ]
  combined_scores = combine_rewards(reward_fns, completions)
  print(combined_scores)  # Weighted average
  ```
</details>

---

### `get_default_reward_function`
Returns a single reward function that **combines** the following:
1. **`format_reward`** (weight = 0.25)
2. **`thought_steps_reward`** (weight = 0.35)
3. **`tool_calls_validity_reward`** (weight = 0.75)
4. **`repetition_penalty_reward`** (weight = 0.15, using defaults)

Weights are internally normalized during computation. This function is useful if you want a **quick** default evaluation metric that checks:
- Format correctness
- Reasoning steps
- Valid tool usage
- Non-repetitive content

<details>
  <summary>Example</summary>

  ```python
  reward_fn = get_default_reward_function(include_schema_validation=True)
  completions = [
      "<|start_thought|>I should clarify the question<|end_thought|>\n<|tool_calls|>[{\"name\": \"search\", \"parameters\": {}}]<|eot_id|>"
  ]
  scores = reward_fn(completions)
  print(scores)  # e.g., [somewhere around 1.0 if all checks pass]
  ```
</details>

## Examples and Usage

Below is a simple demonstration of combining two reward functions (format and thought steps), then applying them to a few completions:

```python
from typing import Any

# Suppose we have the reward functions available in the same namespace
from reward_functions import (
    format_reward,
    thought_steps_reward,
    combine_rewards
)

# Example completions
completions = [
    "<|start_thought|>Thinking...<|end_thought|> <|tool_calls|>[]<|eot_id|>",
    "An invalid response with no tags."
]

# Combine with weighting
reward_fns = [
    (format_reward, 0.5),
    (thought_steps_reward, 0.5)
]

# Evaluate
scores = combine_rewards(reward_fns, completions)
print(scores)  # e.g., [some_value, 0.0]
```

If you want an **all-in-one** evaluation without manually specifying weights, just use:

```python
from reward_functions import get_default_reward_function

default_fn = get_default_reward_function()
scores = default_fn(completions)
print(scores)
```
