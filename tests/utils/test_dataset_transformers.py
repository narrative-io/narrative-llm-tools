from datasets import Dataset
from narrative_llm_tools.utils.dataset_transformers import grpo_conversation_transform

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
