"""Tests for Mod 1: src/data_manager.py"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data_manager import load_and_filter_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_dummy_data(tmp_path: Path, n_samples=50, n_features=10):
    """Create minimal x_train.txt, y_train.txt, and selected_features.json."""
    rng = np.random.default_rng(0)

    feature_names = [f"V{i}" for i in range(n_features)]
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)

    x_path = tmp_path / "x_train.txt"
    header = " ".join(feature_names)
    np.savetxt(x_path, X, header=header, comments="", fmt="%.6f")

    y_path = tmp_path / "y_train.txt"
    np.savetxt(y_path, y, fmt="%d")

    json_path = tmp_path / "selected_features.json"
    # Reverse order so the "best" feature is V9
    ranked = list(reversed(feature_names))
    json_path.write_text(json.dumps(ranked))

    return x_path, y_path, json_path, ranked, X, y


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_basic_load(tmp_path):
    x_path, y_path, json_path, ranked, X, y = _write_dummy_data(tmp_path)
    K = 3
    X_filt, y_out, names = load_and_filter_features(K, json_path, x_path, y_path)

    assert X_filt.shape == (50, K), "Wrong filtered shape"
    assert y_out.shape == (50,), "Wrong y shape"
    assert names == ranked[:K], "Wrong feature names returned"


def test_full_feature_set(tmp_path):
    x_path, y_path, json_path, ranked, X, y = _write_dummy_data(tmp_path)
    K = 10
    X_filt, y_out, names = load_and_filter_features(K, json_path, x_path, y_path)
    assert X_filt.shape == (50, 10)


def test_feature_order_preserved(tmp_path):
    x_path, y_path, json_path, ranked, X_orig, y = _write_dummy_data(tmp_path)
    K = 5
    X_filt, _, names = load_and_filter_features(K, json_path, x_path, y_path)
    # The columns should be in the order from ranked, not from original X
    assert names == ranked[:K]


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------

def test_k_too_large(tmp_path):
    x_path, y_path, json_path, *_ = _write_dummy_data(tmp_path)
    with pytest.raises(ValueError, match="exceeds"):
        load_and_filter_features(999, json_path, x_path, y_path)


def test_k_zero(tmp_path):
    x_path, y_path, json_path, *_ = _write_dummy_data(tmp_path)
    with pytest.raises(ValueError, match="K must be"):
        load_and_filter_features(0, json_path, x_path, y_path)


def test_missing_x_train(tmp_path):
    _, y_path, json_path, *_ = _write_dummy_data(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_and_filter_features(3, json_path, tmp_path / "nope.txt", y_path)


def test_missing_y_train(tmp_path):
    x_path, _, json_path, *_ = _write_dummy_data(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_and_filter_features(3, json_path, x_path, tmp_path / "nope.txt")


def test_missing_json(tmp_path):
    x_path, y_path, *_ = _write_dummy_data(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_and_filter_features(3, tmp_path / "nope.json", x_path, y_path)


def test_unknown_feature_in_json(tmp_path):
    x_path, y_path, json_path, *_ = _write_dummy_data(tmp_path)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps(["DOES_NOT_EXIST", "V0"]))
    with pytest.raises(ValueError, match="not present"):
        load_and_filter_features(1, bad_json, x_path, y_path)


def test_invalid_json_format(tmp_path):
    x_path, y_path, json_path, *_ = _write_dummy_data(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")  # dict, not list
    with pytest.raises(ValueError):
        load_and_filter_features(1, bad, x_path, y_path)
