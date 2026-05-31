"""Tests for Mod 4: src/analysis/individual_thresholds.py"""

import numpy as np
import pytest

from src.analysis.individual_thresholds import find_individual_model_threshold


def _perfect_probs(n=100, n_pos=20, seed=0):
    """Return (y, probs) where positive examples have highest probs."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    probs = np.zeros(n)
    probs[:n_pos] = rng.uniform(0.7, 1.0, n_pos)
    probs[n_pos:] = rng.uniform(0.0, 0.3, n - n_pos)
    return y, probs


def test_perfect_classifier_max_profit():
    """A perfect classifier should yield max possible profit."""
    n, n_pos = 200, 30
    y, probs = _perfect_probs(n, n_pos)
    K = 5
    max_profit, threshold, idx, *_ = find_individual_model_threshold(y, probs, K)
    expected = n_pos * 10 - K * 200  # 0 FP
    assert max_profit == pytest.approx(expected), (
        f"Expected {expected}, got {max_profit}"
    )


def test_output_shapes():
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, n)
    probs = rng.uniform(0, 1, n)
    profit, threshold, idx, sorted_probs, sorted_labels, cum_profit = (
        find_individual_model_threshold(y, probs, K=3)
    )
    assert sorted_probs.shape == (n,)
    assert sorted_labels.shape == (n,)
    assert len(cum_profit) == min(n, 1000)


def test_sorted_descending():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 100)
    probs = rng.uniform(0, 1, 100)
    _, _, _, sorted_probs, *_ = find_individual_model_threshold(y, probs, K=2)
    assert np.all(sorted_probs[:-1] >= sorted_probs[1:]), (
        "sorted_probs should be non-increasing"
    )


def test_hard_cap_respected():
    """Even with 2000 samples, only first 1000 are considered."""
    rng = np.random.default_rng(3)
    n = 2000
    y = rng.integers(0, 2, n)
    probs = rng.uniform(0, 1, n)
    *_, cum_profit = find_individual_model_threshold(y, probs, K=1)
    assert len(cum_profit) == 1000, "cumulative_profit length should be capped at 1000"


def test_profit_formula():
    """Manual verification with a known sequence."""
    # Sequence sorted by prob descending: TP, TP, FP, TP
    y = np.array([1, 1, 0, 1])
    probs = np.array([0.9, 0.8, 0.7, 0.6])
    K = 1
    max_profit, threshold, idx, _, _, cum_profit = (
        find_individual_model_threshold(y, probs, K)
    )
    variable_cost = K * 200
    expected_at_each = [
        1 * 10 - 0 * 5 - variable_cost,   # after 1st customer: TP=1, FP=0
        2 * 10 - 0 * 5 - variable_cost,   # after 2nd: TP=2, FP=0
        2 * 10 - 1 * 5 - variable_cost,   # after 3rd: TP=2, FP=1
        3 * 10 - 1 * 5 - variable_cost,   # after 4th: TP=3, FP=1
    ]
    np.testing.assert_array_almost_equal(cum_profit, expected_at_each)
    assert max_profit == max(expected_at_each)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        find_individual_model_threshold(np.array([0, 1]), np.array([0.5]), K=1)


def test_k_zero_raises():
    with pytest.raises(ValueError, match="K must be"):
        find_individual_model_threshold(np.array([0, 1]), np.array([0.4, 0.6]), K=0)
