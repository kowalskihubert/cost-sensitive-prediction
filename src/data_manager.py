"""Mod 1: Data Manager — load training data and filter to top-K features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_and_filter_features(
    K: int,
    zosia_features_json_path: str | Path,
    x_train_path: str | Path = "x_train.txt",
    y_train_path: str | Path = "y_train.txt",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load training data and return the top-K feature subset.

    Parameters
    ----------
    K:
        Number of top features to select (must be >= 1).
    zosia_features_json_path:
        Path to Zosia's ``selected_features.json`` — an ordered list of
        feature names, best-first.
    x_train_path:
        Path to ``x_train.txt`` (space/tab-separated, header row with feature
        names).
    y_train_path:
        Path to ``y_train.txt`` (one label per line, no header).

    Returns
    -------
    X_filtered : np.ndarray, shape (n_samples, K)
    y          : np.ndarray, shape (n_samples,)
    feature_names : list[str]
        The K feature names that were selected (in order).
    """
    zosia_features_json_path = Path(zosia_features_json_path)
    x_train_path = Path(x_train_path)
    y_train_path = Path(y_train_path)

    # --- validate inputs ---
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    for p in (zosia_features_json_path, x_train_path, y_train_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # --- load Zosia's ranking ---
    with zosia_features_json_path.open() as f:
        ranked_features: list[str] = json.load(f)

    if not isinstance(ranked_features, list) or len(ranked_features) == 0:
        raise ValueError(
            f"selected_features.json must be a non-empty JSON list, "
            f"got: {type(ranked_features)}"
        )

    if K > len(ranked_features):
        raise ValueError(
            f"K={K} exceeds the number of features in Zosia's ranking "
            f"({len(ranked_features)})"
        )

    selected = ranked_features[:K]

    # --- load training data ---
    X_full = pd.read_csv(x_train_path, sep=r"\s+", header=0)
    y = np.loadtxt(y_train_path, dtype=int)

    # --- validate feature names ---
    missing = [f for f in selected if f not in X_full.columns]
    if missing:
        raise ValueError(
            f"The following features from Zosia's JSON are not present in "
            f"x_train.txt: {missing}"
        )

    if X_full.shape[0] != y.shape[0]:
        raise ValueError(
            f"Row count mismatch: x_train has {X_full.shape[0]} rows, "
            f"y_train has {y.shape[0]} rows."
        )

    X_filtered = X_full[selected].to_numpy(dtype=float)

    return X_filtered, y, selected
