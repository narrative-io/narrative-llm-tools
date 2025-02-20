from datasets import Dataset
import tempfile
from trl import GRPOTrainer, GRPOConfig
import torch

from narrative_llm_tools.reward_functions.cot_with_tool_call import adjust_scores_by_length

def reward_func(completions, **kwargs):
    """Reward function that gives higher scores to longer completion content."""
    contents = [completion[0]["content"] for completion in completions]
    base_rewards = [1.0] * len(contents)
    adjusted_rewards = adjust_scores_by_length(base_rewards, [len(content) for content in contents])
    return adjusted_rewards


def test_training_reward_func_conversational():
    # Test if trainer can handle reward function with conversational format
    dataset = Dataset.from_dict({"prompt": [[ { "content": "What is better than ugly?", "role": "user" } ], [ { "content": "What is better than implicit?", "role": "user" } ]]})

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = GRPOConfig(
            output_dir=tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=32,
            report_to="none",
            num_train_epochs=1
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert (trainer.state.log_history[-1]["train_loss"]) is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param)
