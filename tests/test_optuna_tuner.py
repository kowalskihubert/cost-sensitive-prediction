"""Tests for Mod 2: src/tuning/optuna_tuner.py"""

import numpy as np
import pytest

from src.tuning.optuna_tuner import tune_hyperparameters, SUPPORTED_MODELS


def _make_data(n=200, p=8, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + rng.standard_normal(n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("model_type", SUPPORTED_MODELS)
def test_returns_dict_and_df(model_type, tmp_path):
    X, y = _make_data()
    params, history = tune_hyperparameters(
        X, y, model_type, n_trials=3, trial_csv_dir=tmp_path
    )
    assert isinstance(params, dict), "best_params should be a dict"
    assert len(params) > 0, "best_params should not be empty"
    assert len(history) == 3, "trial history should have one row per trial"
    assert "value" in history.columns, "trial history should have 'value' column"


def test_logreg_param_keys(tmp_path):
    X, y = _make_data()
    params, _ = tune_hyperparameters(X, y, "logistic_regression", n_trials=3)
    assert "C" in params
    assert "penalty" in params
    assert params["penalty"] in ("l1", "l2")
    assert 0.001 <= params["C"] <= 10


def test_lgbm_param_keys(tmp_path):
    X, y = _make_data()
    params, _ = tune_hyperparameters(X, y, "lgbm", n_trials=3)
    for key in ("max_depth", "learning_rate", "colsample_bytree", "num_leaves", "min_data_in_leaf"):
        assert key in params, f"missing key: {key}"


def test_rf_param_keys(tmp_path):
    X, y = _make_data()
    params, _ = tune_hyperparameters(X, y, "random_forest", n_trials=3)
    assert "max_depth" in params
    assert "min_samples_leaf" in params


def test_trial_csv_written(tmp_path):
    X, y = _make_data()
    tune_hyperparameters(X, y, "logistic_regression", n_trials=2, trial_csv_dir=tmp_path)
    csv = tmp_path / "logistic_regression_trials.csv"
    assert csv.exists(), "Trial CSV should be written when trial_csv_dir is given"


def test_invalid_model_type():
    X, y = _make_data()
    with pytest.raises(ValueError, match="model_type"):
        tune_hyperparameters(X, y, "svm", n_trials=2)
