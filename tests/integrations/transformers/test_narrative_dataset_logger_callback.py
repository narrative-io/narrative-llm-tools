import json
import pytest
from unittest.mock import Mock

from narrative_llm_tools.integrations.transformers.narrative_dataset_logger_callback import (
    LogBuffer,
)
from narrative_llm_tools.integrations.axolotl.narrative_dataset_logger_plugin.args import (
    NarrativeDatasetLoggerArgs,
)
from narrative_llm_tools.narrative_api.dataset import Dataset

# Test fixtures
@pytest.fixture
def mock_api_client():
    return Mock()

@pytest.fixture
def logger_args():
    return NarrativeDatasetLoggerArgs(
        api_base_url="https://api.example.com",
        api_key="test-key",
        dataset_name="test-dataset",
        dataset_description="Test dataset",
        upload_every_n_steps=2,
        log_params_to_save={"loss": float, "accuracy": float},
        create_dataset=True,
    )

@pytest.fixture
def mock_dataset():
    return Dataset(
        id=123,
        name="test-dataset",
        description="Test dataset",
        properties={},
    )

# LogBuffer Tests
def test_log_buffer():
    buffer = LogBuffer(entries=[], chunk_size=2)
    
    # Test adding entries
    entry1 = {"step": 1, "loss": 0.5}
    buffer.add_entry(entry1)
    assert len(buffer.entries) == 1
    assert not buffer.is_ready_for_upload()
    
    # Test ready for upload
    entry2 = {"step": 2, "loss": 0.3}
    buffer.add_entry(entry2)
    assert buffer.is_ready_for_upload()
    
    # Test serialization
    expected_jsonl = f"{json.dumps(entry1)}\n{json.dumps(entry2)}"
    assert buffer.serialize() == expected_jsonl
    
    # Test clear
    buffer.clear()
    assert len(buffer.entries) == 0
