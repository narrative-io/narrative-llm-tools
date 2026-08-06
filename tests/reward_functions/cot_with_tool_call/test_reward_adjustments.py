import pytest
from narrative_llm_tools.reward_functions.cot_with_tool_call import adjust_scores_by_length

def test_input_validation():
    """Test that mismatched input lengths raise ValueError"""
    scores = [1.0, 0.5]
    lengths = [10]
    with pytest.raises(ValueError, match="Scores and lengths must have the same length"):
        adjust_scores_by_length(scores, lengths)

def test_no_correct_answers():
    """Test that scores are returned unchanged when no answers meet the threshold"""
    scores = [0.5, 0.8, 0.9]
    lengths = [10, 20, 30]
    result = adjust_scores_by_length(scores, lengths)
    assert result == scores

def test_single_correct_answer():
    """Test that a single correct answer remains unchanged"""
    scores = [1.0, 0.5, 0.8]
    lengths = [10, 20, 30]
    result = adjust_scores_by_length(scores, lengths)
    assert result == scores

def test_equal_length_correct_answers():
    """Test that correct answers of equal length remain unchanged"""
    scores = [1.0, 0.5, 1.0]
    lengths = [10, 20, 10]
    result = adjust_scores_by_length(scores, lengths)
    assert result == scores

def test_length_penalty():
    """Test that longer correct answers receive appropriate penalties"""
    scores = [1.0, 0.5, 1.0]
    lengths = [10, 20, 30]  # min=10, max=30, k=20
    result = adjust_scores_by_length(scores, lengths)

    # First correct answer (shortest) should remain 1.0
    assert result[0] == 1.0
    # Second answer (incorrect) should remain unchanged
    assert result[1] == 0.5
    # Third answer (correct, longest) should be penalized
    # excess_length = 20, k = 20
    # penalty = 1 / (1 + (20/20)^2) = 1/2
    assert result[2] == pytest.approx(0.5, rel=1e-6)

def test_custom_threshold():
    """Test that custom threshold works correctly"""
    scores = [0.9, 0.5, 0.9]
    lengths = [10, 20, 30]
    result = adjust_scores_by_length(scores, lengths, correct_threshold=0.9)

    # Similar calculation as above but with 0.9 threshold
    assert result[0] == 0.9
    assert result[1] == 0.5
    assert result[2] == pytest.approx(0.45, rel=1e-6)

def test_floating_point_comparison():
    """Test that floating point comparisons work correctly"""
    scores = [1.0, 1.0 - 1e-7, 0.5]  # Second score is very close to 1.0
    lengths = [10, 20, 30]
    result = adjust_scores_by_length(scores, lengths)

    # Both near-1.0 scores should be treated as correct
    assert result[0] == 1.0
    # For the second score:
    # k = max_length - min_length among correct scores = 20 - 10 = 10
    # excess_length = 20 - 10 = 10
    # penalty = 1 / (1 + (10/10)^2) = 1/2
    assert abs(result[1] - (1.0 - 1e-7) * 0.5) < 1e-6
    assert result[2] == 0.5

def test_empty_inputs():
    """Test that empty inputs return empty list"""
    result = adjust_scores_by_length([], [])
    assert result == []

def test_all_scores_below_threshold():
    """Test behavior when all scores are below threshold"""
    scores = [0.8, 0.7, 0.6]
    lengths = [10, 20, 30]
    result = adjust_scores_by_length(scores, lengths)
    assert result == scores

def test_negative_lengths():
    """Test that function works with negative lengths"""
    scores = [1.0, 0.5, 1.0]
    lengths = [-10, 0, 10]
    result = adjust_scores_by_length(scores, lengths)

    # First correct answer (shortest at -10) should remain 1.0
    assert result[0] == 1.0
    # Second answer (incorrect) should remain unchanged
    assert result[1] == 0.5
    # Third answer (correct, longest) should be penalized
    # excess_length = 20, k = 20
    # penalty = 1 / (1 + (20/20)^2) = 1/2
    assert result[2] == pytest.approx(0.5, rel=1e-6)
