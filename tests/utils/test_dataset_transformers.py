from datasets import Dataset
import pytest
from typing import Any, Dict, List

from narrative_llm_tools.utils.dataset_transformers import (
    GRPORecord,
    grpo_conversation_transform,
)


def test_grpo_record_model():
    """Test the GRPORecord model validates correctly."""
    record = GRPORecord(
        prompt=[{"role": "user", "content": "Hello"}],
        ground_truth="Hi there"
    )
    assert record.prompt[0]["role"] == "user"
    assert record.prompt[0]["content"] == "Hello"
    assert record.ground_truth == "Hi there"


def test_model_dumps():
    """Test that the GRPORecord model dumps to dict correctly."""
    record = GRPORecord(
        prompt=[{"role": "user", "content": "Hello"}],
        ground_truth="Hi there"
    )

    dumped = record.model_dump()
    assert isinstance(dumped, dict)
    assert "prompt" in dumped
    assert "ground_truth" in dumped
    assert dumped["prompt"][0]["role"] == "user"
    assert dumped["ground_truth"] == "Hi there"


def test_grpo_record_with_multiple_messages():
    """Test the GRPORecord model with multiple messages."""
    record = GRPORecord(
        prompt=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ],
        ground_truth="I'm doing well, thanks for asking!"
    )

    assert len(record.prompt) == 3
    assert record.prompt[0]["role"] == "user"
    assert record.prompt[1]["role"] == "assistant"
    assert record.prompt[2]["role"] == "user"
    assert record.ground_truth == "I'm doing well, thanks for asking!"


def test_non_list_conversation():
    """Test handling of non-list conversation."""
    transform = grpo_conversation_transform({})
    invalid_record = {"conversation": "not a list"}

    with pytest.raises(ValueError, match="Conversation is not a list"):
        transform(invalid_record)


def test_model_validation_with_invalid_prompt_type():
    """Test that the GRPORecord model validates prompt field type."""
    with pytest.raises(ValueError):
        GRPORecord(prompt="not a list", ground_truth="test")


def test_model_validation_with_invalid_prompt_items():
    """Test that the GRPORecord model validates prompt list items."""
    with pytest.raises(ValueError):
        GRPORecord(prompt=[1, 2, 3], ground_truth="test")


def test_model_validation_with_invalid_ground_truth():
    """Test that the GRPORecord model validates ground_truth field type."""
    with pytest.raises(ValueError):
        GRPORecord(prompt=[{"role": "user", "content": "Hello"}], ground_truth=123)


def test_model_validation_with_empty_prompt():
    """Test that an empty prompt is valid (confirmed by the failing test)."""
    record = GRPORecord(prompt=[], ground_truth="test")
    assert record.prompt == []
    assert record.ground_truth == "test"


def test_model_validation_with_complex_prompt():
    """Test that the GRPORecord model validates with a complex prompt."""
    record = GRPORecord(
        prompt=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello", "metadata": {"id": "123"}},
            {"role": "assistant", "content": "Hi there", "metadata": {"confidence": 0.95}},
            {"role": "user", "content": "How are you?"}
        ],
        ground_truth="I'm doing well, thanks for asking!"
    )

    assert len(record.prompt) == 4
    assert record.prompt[0]["role"] == "system"
    assert record.prompt[1]["metadata"]["id"] == "123"
    assert record.prompt[2]["metadata"]["confidence"] == 0.95


def test_grpo_conversation_transform():
    # Create a sample dataset
    sample_data = {
        "conversations": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "It's sunny"}
            ]
        ],
        "_metadata": [
            {"timestamp": "2024-01-01"},
            {"timestamp": "2024-01-02"}
        ]
    }

    dataset = Dataset.from_dict(sample_data)

    # Create transform function
    transform_fn = grpo_conversation_transform(cfg=None)

    expected_prompt = [{"role": "user", "content": "Hello"}]
    expected_ground_truth = "Hi there!"
    expected_prompt_2 = [{"role": "user", "content": "What's the weather?"}]
    expected_ground_truth_2 = "It's sunny"

    # Test dataset.map with batch_size=1 (single example processing)
    single_result = dataset.map(transform_fn, batched=False)
    assert single_result[0]["prompt"] == expected_prompt
    assert single_result[0]["ground_truth"] == expected_ground_truth
    assert single_result[1]["prompt"] == expected_prompt_2
    assert single_result[1]["ground_truth"] == expected_ground_truth_2

    # Test dataset.map with batch_size=2 (batch processing)
    batch_result = dataset.map(transform_fn, batched=True, batch_size=2)
    assert batch_result[0]["prompt"] == expected_prompt
    assert batch_result[0]["ground_truth"] == expected_ground_truth
    assert batch_result[1]["prompt"] == expected_prompt_2
    assert batch_result[1]["ground_truth"] == expected_ground_truth_2

    ## Result should be the same if we batch or not
    assert single_result[0] == batch_result[0]
    assert single_result[1] == batch_result[1]