"""Mod 3: Outer Nested CV Engine — OOF predictions + 5 calibrated fold models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _build_base_pipeline(model_type: str, params: dict[str, Any]) -> Pipeline:
    """Build a sklearn Pipeline for a given model type and params.

    StandardScaler is applied for LogisticRegression (required); for tree
    models it is omitted to avoid unnecessary transformation.
    """
    if model_type == "logistic_regression":
        # sklearn >= 1.8: penalty is deprecated; use l1_ratio instead.
        l1_ratio = 1.0 if params.get("penalty") == "l1" else 0.0
        estimator = LogisticRegression(
            C=params["C"],
            l1_ratio=l1_ratio,
            solver="saga",  # saga supports both L1 and L2 via ElasticNet
            max_iter=1000,
            random_state=42,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])

    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            colsample_bytree=params["colsample_bytree"],
            num_leaves=params["num_leaves"],
            min_data_in_leaf=params["min_data_in_leaf"],
            n_estimators=200,
            random_state=42,
            verbose=-1,
        )
        return Pipeline([("clf", estimator)])

    elif model_type == "random_forest":
        estimator = RandomForestClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            n_estimators=200,
            random_state=42,
        )
        return Pipeline([("clf", estimator)])

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from: 'logistic_regression', 'lgbm', 'random_forest'"
        )


def run_outer_cv_with_calibration(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    best_params: dict[str, Any],
    n_outer_folds: int = 5,
) -> tuple[np.ndarray, list[CalibratedClassifierCV], np.ndarray]:
    """Run 5-fold stratified outer CV with Platt-scaled calibration.

    For each outer fold:
    - Wrap the base model pipeline in ``CalibratedClassifierCV(cv=3,
      method='sigmoid')`` and fit on the training fold only.
    - Predict calibrated probabilities on the held-out validation fold.
    - Store the fitted calibrated model.

    The 5000 OOF predictions are concatenated in **original row order** — no
    shuffling artefacts.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    model_type : str
        One of ``'logistic_regression'``, ``'lgbm'``, ``'random_forest'``.
    best_params : dict
        Hyperparameters returned by :func:`tune_hyperparameters`.
    n_outer_folds : int
        Number of outer CV folds (default 5).

    Returns
    -------
    oof_probs : np.ndarray, shape (n_samples,)
        Out-of-fold positive-class probabilities in original row order.
    calibrated_models : list[CalibratedClassifierCV]
        One fitted calibrated model per outer fold (length == n_outer_folds).
    fold_assignments : np.ndarray, shape (n_samples,)
        Integer fold index (0-based) for each training row.
    """
    n_samples = X.shape[0]
    oof_probs = np.zeros(n_samples, dtype=float)
    fold_assignments = np.zeros(n_samples, dtype=int)
    calibrated_models: list[CalibratedClassifierCV] = []

    skf = StratifiedKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=42
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold = y[train_idx]

        base_pipeline = _build_base_pipeline(model_type, best_params)

        # Platt scaling (sigmoid) calibration fitted on the training fold.
        # cv=3 means CalibratedClassifierCV itself does an inner 3-fold split
        # inside the training fold to learn the sigmoid mapping.
        calibrated = CalibratedClassifierCV(
            estimator=base_pipeline, cv=3, method="sigmoid"
        )
        calibrated.fit(X_train_fold, y_train_fold)

        val_probs = calibrated.predict_proba(X_val_fold)[:, 1]
        oof_probs[val_idx] = val_probs
        fold_assignments[val_idx] = fold_idx
        calibrated_models.append(calibrated)

        logger.debug(
            "Fold %d/%d complete — val size=%d",
            fold_idx + 1,
            n_outer_folds,
            len(val_idx),
        )

    logger.info(
        "Outer CV complete for %s — OOF mean prob=%.4f",
        model_type,
        oof_probs.mean(),
    )

    return oof_probs, calibrated_models, fold_assignments
