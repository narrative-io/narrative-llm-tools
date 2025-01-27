import unittest
import torch
from narrative_llm_tools.reward_models.json_similarity_reward_model import JsonSimilarityRewardModel, JsonDifferenceReason

class TestJsonSimilarityRewardModel(unittest.TestCase):
    def setUp(self):
        self.model = JsonSimilarityRewardModel()

    def test_compare_values_primitives(self):
        # Test exact matches
        self.assertEqual(self.model.compare_values(42, 42), (None, 1.0))
        self.assertEqual(self.model.compare_values(True, True), (None, 1.0))
        self.assertEqual(self.model.compare_values(None, None), (None, 1.0))
        
        # Test mismatches
        self.assertEqual(self.model.compare_values(42, 43), (None, 0.0))
        self.assertEqual(self.model.compare_values(True, False), (None, 0.0))
        
        # Test type mismatches
        self.assertEqual(
            self.model.compare_values(42, "42"),
            (JsonDifferenceReason.TYPE_MISMATCH, 0.0)
        )

    def test_compare_strings(self):
        # Test exact string match
        self.assertEqual(self.model.compare_values("hello", "hello"), (None, 1.0))
        
        # Test similar strings
        error, similarity = self.model.compare_values("hello", "helo")
        self.assertIsNone(error)
        self.assertTrue(0.0 < similarity < 1.0)

    def test_compare_arrays(self):
        # Test empty arrays
        self.assertEqual(self.model.compare_arrays([], []), (None, 1.0))
        
        # Test exact match
        self.assertEqual(
            self.model.compare_arrays([1, 2, 3], [1, 2, 3]),
            (None, 1.0)
        )
        
        # Test different order
        error, similarity = self.model.compare_arrays([1, 2, 3], [3, 1, 2])
        self.assertIsNone(error)
        self.assertEqual(similarity, 1.0)
        
        # Test different lengths
        error, similarity = self.model.compare_arrays([1, 2], [1, 2, 3])
        self.assertIsNone(error)
        self.assertTrue(similarity < 1.0)

    def test_compare_objects(self):
        # Test exact match
        obj1 = {"name": "John", "age": 30}
        obj2 = {"name": "John", "age": 30}
        self.assertEqual(self.model.compare_objects(obj1, obj2), (None, 1.0))
        
        # Test missing key
        obj3 = {"name": "John"}
        self.assertEqual(
            self.model.compare_objects(obj1, obj3),
            (JsonDifferenceReason.MISSING_KEY, 0.0)
        )
        
        # Test similar values
        obj4 = {"name": "Jon", "age": 30}
        error, similarity = self.model.compare_objects(obj1, obj4)
        self.assertIsNone(error)
        self.assertTrue(0.0 < similarity < 1.0)

    def test_forward(self):
        # Test valid JSON strings
        predicted = ['{"name": "John"}', '{"age": 30}']
        target = ['{"name": "Jon"}', '{"age": 30}']
        rewards = self.model(predicted, target)
        
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertEqual(rewards.shape, torch.Size([2]))
        self.assertTrue(torch.all(rewards >= 0.0))
        self.assertTrue(torch.all(rewards <= 1.0))
        
        # Test invalid JSON
        predicted = ['{"name": "John"}', 'invalid json']
        target = ['{"name": "John"}', '{"age": 30}']
        rewards = self.model(predicted, target)
        self.assertEqual(rewards[1].item(), 0.0)

if __name__ == '__main__':
    unittest.main()