"""Mod 5: Master Experiment Loop — runs all (K, model) combinations.

Output
------
experiments/results/experiment_results.csv
    Columns: K, Model, BestParams, OOFProfit, OptThreshold,
             OuterModels_Pkl_Path, OOFPredictions_Npy_Path

experiments/results/K{K}_{model}/
    oof_predictions.npy   — OOF probability array (5000,)
    fold_assignments.npy  — fold index per training row
    fold_{i}_model.pkl    — fitted CalibratedClassifierCV for fold i (i=0..4)

experiments/results/trials/
    {model}_trials.csv    — Optuna trial history (one file per (K, model))
    (actually per (K, model) call, but named by model type with K prefix)
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.analysis.individual_thresholds import find_individual_model_threshold
from src.cv.outer_cv_engine import run_outer_cv_with_calibration
from src.data_manager import load_and_filter_features
from src.tuning.optuna_tuner import tune_hyperparameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODELS = ("logistic_regression", "lgbm", "random_forest")
DEFAULT_K_RANGE = range(1, 21)
RESULTS_DIR = Path("experiments/results")


def run_all_experiments(
    zosia_features_json_path: str | Path,
    x_train_path: str | Path = "x_train.txt",
    y_train_path: str | Path = "y_train.txt",
    models: Sequence[str] = DEFAULT_MODELS,
    K_range: range = DEFAULT_K_RANGE,
    n_optuna_trials: int = 50,
    results_dir: str | Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Run the full grid of (K, model) experiments and persist all outputs.

    For each (K, model) combination:
    1. Load top-K features (Mod 1).
    2. Tune hyperparameters with Optuna — 3-fold inner CV, log-loss (Mod 2).
    3. Run 5-fold outer CV with Platt-scaled calibration (Mod 3).
    4. Find empirical OOF profit-maximising threshold (Mod 4).
    5. Save pickled models, OOF arrays, and a CSV row.

    Parameters
    ----------
    zosia_features_json_path :
        Path to Zosia's ``selected_features.json``.
    x_train_path :
        Path to ``x_train.txt``.
    y_train_path :
        Path to ``y_train.txt``.
    models :
        Sequence of model type strings to evaluate.
    K_range :
        Range of K values (number of features) to sweep.
    n_optuna_trials :
        Number of Optuna trials per (K, model).
    results_dir :
        Root directory for all output artefacts.

    Returns
    -------
    pd.DataFrame
        The ``experiment_results.csv`` table (also written to disk).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "experiment_results.csv"

    # Load pre-existing results so the loop is restartable
    existing: set[tuple[int, str]] = set()
    records: list[dict] = []
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        for _, row in df_existing.iterrows():
            existing.add((int(row["K"]), str(row["Model"])))
            records.append(row.to_dict())
        logger.info(
            "Resuming — %d existing results found in %s", len(existing), csv_path
        )

    total = len(list(K_range)) * len(list(models))
    done = 0

    for K in K_range:
        for model_type in models:
            done += 1
            tag = f"K={K}, model={model_type}"

            if (K, model_type) in existing:
                logger.info("[%d/%d] Skipping %s (already done)", done, total, tag)
                continue

            logger.info("[%d/%d] Starting %s", done, total, tag)
            t0 = time.time()

            # --- Mod 1: load data ---
            try:
                X, y, feature_names = load_and_filter_features(
                    K,
                    zosia_features_json_path,
                    x_train_path=x_train_path,
                    y_train_path=y_train_path,
                )
            except Exception as exc:
                logger.error("Mod 1 failed for %s: %s", tag, exc)
                continue

            # --- Mod 2: hyperparameter tuning ---
            trials_dir = results_dir / "trials" / f"K{K:02d}"
            try:
                best_params, trial_history = tune_hyperparameters(
                    X,
                    y,
                    model_type,
                    n_trials=n_optuna_trials,
                    trial_csv_dir=trials_dir,
                )
            except Exception as exc:
                logger.error("Mod 2 failed for %s: %s", tag, exc)
                continue

            # --- Mod 3: outer CV ---
            try:
                oof_probs, calibrated_models, fold_assignments = (
                    run_outer_cv_with_calibration(X, y, model_type, best_params)
                )
            except Exception as exc:
                logger.error("Mod 3 failed for %s: %s", tag, exc)
                continue

            # --- Mod 4: threshold / profit ---
            try:
                (
                    max_profit,
                    opt_threshold,
                    opt_index,
                    _sorted_probs,
                    _sorted_labels,
                    _cum_profit,
                ) = find_individual_model_threshold(y, oof_probs, K)
            except Exception as exc:
                logger.error("Mod 4 failed for %s: %s", tag, exc)
                continue

            # --- persist artefacts ---
            exp_dir = results_dir / f"K{K:02d}_{model_type}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            npy_path = exp_dir / "oof_predictions.npy"
            fold_path = exp_dir / "fold_assignments.npy"
            np.save(npy_path, oof_probs)
            np.save(fold_path, fold_assignments)

            pkl_paths: list[str] = []
            for fold_i, model_obj in enumerate(calibrated_models):
                pkl_path = exp_dir / f"fold_{fold_i}_model.pkl"
                with open(pkl_path, "wb") as fh:
                    pickle.dump(model_obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
                pkl_paths.append(str(pkl_path))

            # Save feature list for this K so Mateusz doesn't need to re-open JSON
            features_path = exp_dir / "feature_names.json"
            with open(features_path, "w") as fh:
                json.dump(feature_names, fh)

            elapsed = time.time() - t0
            logger.info(
                "  Done in %.1fs | OOF profit=%.0f | threshold=%.4f | "
                "opt_index=%d",
                elapsed,
                max_profit,
                opt_threshold,
                opt_index,
            )

            record = {
                "K": K,
                "Model": model_type,
                "BestParams": json.dumps(best_params),
                "OOFProfit": max_profit,
                "OptThreshold": opt_threshold,
                "OptIndex": opt_index,
                "OuterModels_Pkl_Path": str(exp_dir),
                "OOFPredictions_Npy_Path": str(npy_path),
                "FoldAssignments_Npy_Path": str(fold_path),
                "FeatureNames_Path": str(features_path),
                "ElapsedSeconds": elapsed,
            }
            records.append(record)
            existing.add((K, model_type))

            # Checkpoint after every experiment so partial runs are usable
            df = pd.DataFrame(records)
            df.to_csv(csv_path, index=False)

    df_final = pd.DataFrame(records)
    df_final.to_csv(csv_path, index=False)
    logger.info(
        "All experiments done. Results written to %s (%d rows).",
        csv_path,
        len(df_final),
    )
    return df_final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all (K, model) experiments."
    )
    parser.add_argument(
        "--features",
        default="selected_features.json",
        help="Path to Zosia's selected_features.json",
    )
    parser.add_argument(
        "--x_train", default="x_train.txt", help="Path to x_train.txt"
    )
    parser.add_argument(
        "--y_train", default="y_train.txt", help="Path to y_train.txt"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials per (K, model)",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=20,
        help="Maximum K to evaluate (inclusive)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model types to evaluate",
    )
    args = parser.parse_args()

    run_all_experiments(
        zosia_features_json_path=args.features,
        x_train_path=args.x_train,
        y_train_path=args.y_train,
        models=args.models,
        K_range=range(1, args.k_max + 1),
        n_optuna_trials=args.n_trials,
    )
