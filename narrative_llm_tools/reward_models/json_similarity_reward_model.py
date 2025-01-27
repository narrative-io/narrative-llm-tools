import json
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from similarity.jarowinkler import JaroWinkler  # type: ignore


class JsonDifferenceReason(Enum):
    ACTUAL_NOT_JSON = "ActualNotJson"
    EXPECTED_NOT_JSON = "ExpectedNotJson"
    TYPE_MISMATCH = "TypeMismatch"
    VALUE_MISMATCH = "ValueMismatch"
    MISSING_KEY = "MissingKey"
    EXTRA_KEY = "ExtraKey"
    ARRAY_LENGTH_MISMATCH = "ArrayLengthMismatch"


class JsonSimilarityRewardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def compare_values(
        self,
        expected: dict[str, Any] | list[Any] | str | int | float | bool | None,
        actual: dict[str, Any] | list[Any] | str | int | float | bool | None,
    ) -> tuple[JsonDifferenceReason | None, float]:
        """Compare two JSON values and return error (if any) and similarity score."""

        # Handle None/null values
        if expected is None and actual is None:
            return None, 1.0

        # Type mismatch check
        if type(expected) is not type(actual):
            return JsonDifferenceReason.TYPE_MISMATCH, 0.0

        # Handle different types
        if isinstance(expected, bool):
            return None, 1.0 if expected == actual else 0.0

        elif isinstance(expected, int | float):
            return None, 1.0 if expected == actual else 0.0

        elif isinstance(expected, str):
            return None, JaroWinkler().similarity(expected, actual)

        elif isinstance(expected, list) and isinstance(actual, list):
            return self.compare_arrays(expected, actual)

        elif isinstance(expected, dict) and isinstance(actual, dict):
            return self.compare_objects(expected, actual)

        return JsonDifferenceReason.TYPE_MISMATCH, 0.0

    def compare_arrays(
        self, expected: list[Any], actual: list[Any]
    ) -> tuple[JsonDifferenceReason | None, float]:
        """Compare two arrays using a greedy matching approach."""
        if not expected and not actual:
            return None, 1.0

        # Create similarity matrix
        similarity_matrix = []
        for exp_item in expected:
            row = []
            for act_item in actual:
                error, sim = self.compare_values(exp_item, act_item)
                if error:
                    return error, 0.0
                row.append(sim)
            similarity_matrix.append(row)

        # Pad shorter arrays with zeros
        max_len = max(len(expected), len(actual))
        while len(similarity_matrix) < max_len:
            similarity_matrix.append([0.0] * len(actual))
        for row in similarity_matrix:
            while len(row) < max_len:
                row.append(0.0)

        # Greedy matching
        total_similarity = 0.0
        remaining_matrix = [row[:] for row in similarity_matrix]
        while remaining_matrix:
            best_score = 0.0
            best_i = best_j = 0

            for i, row in enumerate(remaining_matrix):
                for j, sim in enumerate(row):
                    if sim > best_score:
                        best_score = sim
                        best_i, best_j = i, j

            total_similarity += best_score
            remaining_matrix.pop(best_i)
            for row in remaining_matrix:
                row.pop(best_j)

        return None, total_similarity / max_len

    def compare_objects(
        self, expected: dict[str, Any], actual: dict[str, Any]
    ) -> tuple[JsonDifferenceReason | None, float]:
        """Compare two JSON objects."""
        # Check for missing keys
        if not all(key in actual for key in expected):
            return JsonDifferenceReason.MISSING_KEY, 0.0

        total_similarity = 0.0
        for key, exp_value in expected.items():
            act_value = actual[key]
            error, similarity = self.compare_values(exp_value, act_value)
            if error:
                return error, 0.0
            total_similarity += similarity

        return None, total_similarity / len(expected)

    def forward(self, predicted_json: list[str], target_json: list[str]) -> torch.Tensor:
        """
        Forward pass for the reward model.

        Args:
            predicted_json: List of predicted JSON strings
            target_json: List of target JSON strings

        Returns:
            torch.Tensor: Tensor of reward scores between 0 and 1
        """
        batch_size = len(predicted_json)
        rewards = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            try:
                predicted = json.loads(predicted_json[i])
                target = json.loads(target_json[i])
            except json.JSONDecodeError:
                rewards[i] = 0.0
                continue

            error, similarity = self.compare_values(target, predicted)
            rewards[i] = 0.0 if error else similarity

        return rewards
