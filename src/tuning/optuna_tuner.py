"""Mod 2: Optuna Hyperparameter Tuner — 3-fold inner CV, log-loss minimisation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("logistic_regression", "lgbm", "random_forest")


def _build_model(model_type: str, params: dict[str, Any]):
    """Instantiate a base estimator (no pipeline) from hyper-param dict."""
    if model_type == "logistic_regression":
        # sklearn >= 1.8: penalty is deprecated; use l1_ratio instead.
        # l1_ratio=1 → L1, l1_ratio=0 → L2 (ElasticNet under the hood).
        l1_ratio = 1.0 if params.get("penalty") == "l1" else 0.0
        return LogisticRegression(
            C=params["C"],
            l1_ratio=l1_ratio,
            solver="saga",  # saga supports both L1 and L2 via ElasticNet
            max_iter=1000,
            random_state=42,
        )
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            colsample_bytree=params["colsample_bytree"],
            num_leaves=params["num_leaves"],
            min_data_in_leaf=params["min_data_in_leaf"],
            n_estimators=200,
            random_state=42,
            verbose=-1,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            n_estimators=200,
            random_state=42,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {SUPPORTED_MODELS}"
        )


def _suggest_params(trial: optuna.Trial, model_type: str) -> dict[str, Any]:
    """Map Optuna trial suggestions to a parameter dictionary."""
    if model_type == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
    elif model_type == "lgbm":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "num_leaves": trial.suggest_int("num_leaves", 10, 50),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 20),
        }
    elif model_type == "random_forest":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
        }
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")


def _cv_log_loss(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    params: dict[str, Any],
    n_splits: int = 3,
) -> float:
    """Compute mean log-loss across stratified K-fold splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    losses: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        base = _build_model(model_type, params)

        if model_type in ("logistic_regression",):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", base)])
        else:
            pipe = Pipeline([("clf", base)])

        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_val)[:, 1]
        losses.append(log_loss(y_val, probs))

    return float(np.mean(losses))


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    n_trials: int = 50,
    trial_csv_dir: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run Optuna hyperparameter search using 3-fold stratified CV log-loss.

    Parameters
    ----------
    X_train : np.ndarray, shape (n, p)
    y_train : np.ndarray, shape (n,)
    model_type : str
        One of ``'logistic_regression'``, ``'lgbm'``, ``'random_forest'``.
    n_trials : int
        Number of Optuna trials to run.
    trial_csv_dir : path, optional
        If given, saves a CSV of all trials to
        ``<trial_csv_dir>/<model_type>_trials.csv``.

    Returns
    -------
    best_params : dict[str, Any]
    trial_history : pd.DataFrame
        One row per trial with columns ``trial_number``, ``value``, and one
        column per hyper-parameter.
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"model_type must be one of {SUPPORTED_MODELS}, got '{model_type}'"
        )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_type)
        return _cv_log_loss(X_train, y_train, model_type, params)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    logger.info(
        "Best log-loss=%.4f for %s | params=%s",
        study.best_value,
        model_type,
        best_params,
    )

    # --- build trial history dataframe ---
    rows = []
    for t in study.trials:
        row = {"trial_number": t.number, "value": t.value}
        row.update(t.params)
        rows.append(row)
    trial_history = pd.DataFrame(rows)

    if trial_csv_dir is not None:
        trial_csv_dir = Path(trial_csv_dir)
        trial_csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = trial_csv_dir / f"{model_type}_trials.csv"
        trial_history.to_csv(csv_path, index=False)
        logger.info("Trial history saved to %s", csv_path)

    return best_params, trial_history
