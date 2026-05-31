"""Integration test for Mod 5: src/experiments/master_loop.py.

Uses tiny synthetic data (50 samples, 5 features, K_range=[1,2], 2 Optuna
trials) so the full loop completes in a few seconds.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.master_loop import run_all_experiments


@pytest.fixture()
def tiny_experiment(tmp_path):
    """Set up minimal x_train, y_train, and selected_features.json."""
    rng = np.random.default_rng(0)
    n, p = 100, 5
    feature_names = [f"V{i}" for i in range(p)]
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + rng.standard_normal(n) > 0).astype(int)

    x_path = tmp_path / "x_train.txt"
    header = " ".join(feature_names)
    np.savetxt(x_path, X, header=header, comments="", fmt="%.6f")

    y_path = tmp_path / "y_train.txt"
    np.savetxt(y_path, y, fmt="%d")

    json_path = tmp_path / "selected_features.json"
    json_path.write_text(json.dumps(feature_names))

    results_dir = tmp_path / "results"

    return {
        "x_path": x_path,
        "y_path": y_path,
        "json_path": json_path,
        "results_dir": results_dir,
        "n": n,
        "p": p,
        "feature_names": feature_names,
    }


def test_csv_created(tiny_experiment):
    cfg = tiny_experiment
    run_all_experiments(
        zosia_features_json_path=cfg["json_path"],
        x_train_path=cfg["x_path"],
        y_train_path=cfg["y_path"],
        models=["logistic_regression"],
        K_range=range(1, 3),
        n_optuna_trials=2,
        results_dir=cfg["results_dir"],
    )
    csv = cfg["results_dir"] / "experiment_results.csv"
    assert csv.exists(), "experiment_results.csv must be written"


def test_csv_columns(tiny_experiment):
    cfg = tiny_experiment
    df = run_all_experiments(
        zosia_features_json_path=cfg["json_path"],
        x_train_path=cfg["x_path"],
        y_train_path=cfg["y_path"],
        models=["random_forest"],
        K_range=range(1, 2),
        n_optuna_trials=2,
        results_dir=cfg["results_dir"],
    )
    expected_cols = {"K", "Model", "BestParams", "OOFProfit", "OptThreshold",
                     "OuterModels_Pkl_Path", "OOFPredictions_Npy_Path"}
    assert expected_cols.issubset(set(df.columns)), (
        f"Missing columns: {expected_cols - set(df.columns)}"
    )


def test_artefacts_created(tiny_experiment):
    cfg = tiny_experiment
    run_all_experiments(
        zosia_features_json_path=cfg["json_path"],
        x_train_path=cfg["x_path"],
        y_train_path=cfg["y_path"],
        models=["logistic_regression"],
        K_range=range(1, 2),
        n_optuna_trials=2,
        results_dir=cfg["results_dir"],
    )
    exp_dir = cfg["results_dir"] / "K01_logistic_regression"
    assert exp_dir.is_dir(), "Experiment sub-directory must be created"

    npy = exp_dir / "oof_predictions.npy"
    assert npy.exists(), "oof_predictions.npy must exist"
    oof = np.load(npy)
    assert oof.shape == (cfg["n"],), f"OOF shape should be ({cfg['n']},)"
    assert np.all((oof >= 0) & (oof <= 1)), "OOF probs must be in [0,1]"

    for fold_i in range(5):
        pkl = exp_dir / f"fold_{fold_i}_model.pkl"
        assert pkl.exists(), f"Pickle for fold {fold_i} must exist"
        with open(pkl, "rb") as fh:
            model = pickle.load(fh)
        assert hasattr(model, "predict_proba"), "Loaded model must have predict_proba"


def test_resume_skips_done(tiny_experiment):
    """Running the loop twice should not duplicate rows in the CSV."""
    cfg = tiny_experiment
    run_all_experiments(
        zosia_features_json_path=cfg["json_path"],
        x_train_path=cfg["x_path"],
        y_train_path=cfg["y_path"],
        models=["logistic_regression"],
        K_range=range(1, 2),
        n_optuna_trials=2,
        results_dir=cfg["results_dir"],
    )
    df2 = run_all_experiments(
        zosia_features_json_path=cfg["json_path"],
        x_train_path=cfg["x_path"],
        y_train_path=cfg["y_path"],
        models=["logistic_regression"],
        K_range=range(1, 2),
        n_optuna_trials=2,
        results_dir=cfg["results_dir"],
    )
    assert len(df2) == 1, "Second run must not duplicate rows"


def test_multiple_models(tiny_experiment):
    cfg = tiny_experiment
    df = run_all_experiments(
        zosia_features_json_path=cfg["json_path"],
        x_train_path=cfg["x_path"],
        y_train_path=cfg["y_path"],
        models=["logistic_regression", "random_forest"],
        K_range=range(1, 3),
        n_optuna_trials=2,
        results_dir=cfg["results_dir"],
    )
    assert len(df) == 4, "Should have 2 K values × 2 models = 4 rows"
