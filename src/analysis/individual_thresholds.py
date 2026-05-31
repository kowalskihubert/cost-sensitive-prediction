"""Mod 4: Empirical threshold finder for individual model OOF predictions."""

from __future__ import annotations

import numpy as np


def find_individual_model_threshold(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    K: int,
    max_selected: int = 1000,
) -> tuple[float, float, int, np.ndarray, np.ndarray, np.ndarray]:
    """Find the probability cut-off that maximises OOF profit for one model.

    Sorts customers descending by predicted probability and sweeps through
    every possible threshold, computing cumulative profit at each step.

    Profit formula (from EXPERIMENT_PLAN.md §1):
        Profit = (cumulative_TP × 10) - (cumulative_FP × 5) - (K × 200)

    The hard constraint ``total_selected <= max_selected`` (default 1000) is
    enforced: only the first ``max_selected`` rows are considered.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        True binary labels (0 / 1).
    oof_probs : np.ndarray, shape (n,)
        Out-of-fold positive-class probabilities.
    K : int
        Number of features used (determines variable cost).
    max_selected : int
        Hard cap on number of selected customers (default 1000).

    Returns
    -------
    max_profit : float
        Maximum achievable profit.
    opt_threshold : float
        Probability value corresponding to the last included customer at the
        optimal cut-off.
    opt_index : int
        0-based index into the *sorted* arrays at which the maximum is hit.
        ``opt_index + 1`` customers are selected.
    sorted_probs : np.ndarray
        OOF probabilities sorted descending.
    sorted_labels : np.ndarray
        True labels sorted by descending OOF probability.
    cumulative_profit : np.ndarray
        Profit computed at every threshold position (length == min(n, max_selected)).
    """
    y_true = np.asarray(y_true, dtype=int)
    oof_probs = np.asarray(oof_probs, dtype=float)

    if y_true.shape != oof_probs.shape:
        raise ValueError(
            f"y_true and oof_probs must have the same shape, "
            f"got {y_true.shape} vs {oof_probs.shape}"
        )
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    # Sort descending by probability
    order = np.argsort(-oof_probs)
    sorted_probs = oof_probs[order]
    sorted_labels = y_true[order]

    # Apply hard cap
    n_consider = min(len(sorted_labels), max_selected)
    sorted_probs_cap = sorted_probs[:n_consider]
    sorted_labels_cap = sorted_labels[:n_consider]

    cumulative_tp = np.cumsum(sorted_labels_cap)
    cumulative_fp = np.cumsum(1 - sorted_labels_cap)
    variable_cost = K * 200

    cumulative_profit = (
        cumulative_tp * 10 - cumulative_fp * 5 - variable_cost
    ).astype(float)

    opt_index = int(np.argmax(cumulative_profit))
    max_profit = float(cumulative_profit[opt_index])
    opt_threshold = float(sorted_probs_cap[opt_index])

    return (
        max_profit,
        opt_threshold,
        opt_index,
        sorted_probs,
        sorted_labels,
        cumulative_profit,
    )
