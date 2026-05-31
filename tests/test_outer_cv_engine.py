"""Tests for Mod 3: src/cv/outer_cv_engine.py"""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from src.cv.outer_cv_engine import run_outer_cv_with_calibration


def _make_data(n=300, p=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + rng.standard_normal(n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("model_type,params", [
    ("logistic_regression", {"C": 1.0, "penalty": "l2"}),
    ("lgbm", {
        "max_depth": 3, "learning_rate": 0.05,
        "colsample_bytree": 0.8, "num_leaves": 20,
        "min_data_in_leaf": 10,
    }),
    ("random_forest", {"max_depth": 4, "min_samples_leaf": 10}),
])
def test_oof_shape_and_range(model_type, params):
    X, y = _make_data()
    oof, models, folds = run_outer_cv_with_calibration(X, y, model_type, params, n_outer_folds=5)

    assert oof.shape == (300,), f"OOF shape wrong for {model_type}"
    assert np.all((oof >= 0) & (oof <= 1)), "OOF probs outside [0, 1]"
    assert len(models) == 5, "Should return one model per outer fold"


def test_oof_all_rows_covered():
    X, y = _make_data()
    params = {"C": 1.0, "penalty": "l2"}
    oof, _, folds = run_outer_cv_with_calibration(
        X, y, "logistic_regression", params, n_outer_folds=5
    )
    # Every row should have been in exactly one validation fold → no zeros
    # from being skipped (unless the true prob happened to be 0).
    # We verify fold assignments cover all rows exactly once.
    assert len(folds) == 300
    assert set(folds) == {0, 1, 2, 3, 4}


def test_models_are_calibrated_classifiers():
    X, y = _make_data()
    params = {"C": 1.0, "penalty": "l2"}
    _, models, _ = run_outer_cv_with_calibration(
        X, y, "logistic_regression", params, n_outer_folds=3
    )
    for m in models:
        assert isinstance(m, CalibratedClassifierCV), (
            "Each fold model must be a CalibratedClassifierCV"
        )


def test_models_can_predict_on_new_data():
    X, y = _make_data(n=300)
    params = {"C": 1.0, "penalty": "l2"}
    _, models, _ = run_outer_cv_with_calibration(
        X, y, "logistic_regression", params, n_outer_folds=3
    )
    X_new = np.random.default_rng(99).standard_normal((20, 5))
    for m in models:
        probs = m.predict_proba(X_new)[:, 1]
        assert probs.shape == (20,)
        assert np.all((probs >= 0) & (probs <= 1))


def test_invalid_model_type():
    X, y = _make_data()
    with pytest.raises(ValueError, match="Unknown model_type"):
        run_outer_cv_with_calibration(X, y, "svm", {})
