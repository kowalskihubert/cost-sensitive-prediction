# Mateusz Handoff: Hubert's Implementation is Complete

This document describes everything Hubert implemented and the exact contract
Mateusz must consume to build his part (Mods 6–9: per-K ensemble, calibration,
test inference, submission generation).

---

## 1. What Has Been Implemented (Hubert's Scope)

| Module | File | Entry-point |
|--------|------|-------------|
| Mod 1 – Data Manager | `src/data_manager.py` | `load_and_filter_features(K, json_path, ...)` |
| Mod 2 – Optuna Tuner | `src/tuning/optuna_tuner.py` | `tune_hyperparameters(X, y, model_type, ...)` |
| Mod 3 – Outer CV Engine | `src/cv/outer_cv_engine.py` | `run_outer_cv_with_calibration(X, y, model_type, best_params)` |
| Mod 4 – Individual Threshold | `src/analysis/individual_thresholds.py` | `find_individual_model_threshold(y_true, oof_probs, K)` |
| Mod 5 – Master Loop | `src/experiments/master_loop.py` | `run_all_experiments(zosia_features_json_path, ...)` |

Mateusz does **not** need to modify any of these files. He only reads their
outputs from disk.

---

## 2. How to Run Hubert's Experiment Loop

### Prerequisites

- `x_train.txt` and `y_train.txt` in the repo root (5000 × 500 and 5000 × 1).
- `selected_features.json` produced by Zosia in the repo root (ordered list of
  feature name strings, best first). Example: `["V52", "V211", "V8", ...]`.

### Run command

```bash
uv run python -m src.experiments.master_loop \
    --features selected_features.json \
    --x_train x_train.txt \
    --y_train y_train.txt \
    --n_trials 50 \
    --k_max 20
```

Progress is logged to stdout. The loop checkpoints after every (K, model) pair
so it can be safely interrupted and resumed.

Expected total wall-clock time (50 Optuna trials × 20 K × 3 models): 2–4 hours
on a modern laptop; roughly 2–4 min per (K, model) pair.

---

## 3. Primary Output: `experiments/results/experiment_results.csv`

This is **the main file Mateusz reads**. One row per (K, model) combination —
60 rows total (K ∈ {1..20} × {logistic_regression, lgbm, random_forest}).

| Column | Type | Description |
|--------|------|-------------|
| `K` | int | Number of features used |
| `Model` | str | `logistic_regression` / `lgbm` / `random_forest` |
| `BestParams` | JSON str | Dict of best Optuna hyperparameters |
| `OOFProfit` | float | Max OOF profit for this (K, Model) individual model |
| `OptThreshold` | float | Probability threshold achieving `OOFProfit` |
| `OptIndex` | int | 0-based sorted-index of the threshold cut-off |
| `OuterModels_Pkl_Path` | str | Path to the directory containing the 5 fold pickle files |
| `OOFPredictions_Npy_Path` | str | Path to `oof_predictions.npy` |
| `FoldAssignments_Npy_Path` | str | Path to `fold_assignments.npy` |
| `FeatureNames_Path` | str | Path to `feature_names.json` (the K feature names used) |
| `ElapsedSeconds` | float | Wall-clock time for this experiment |

---

## 4. Directory Layout of Artefacts

```
experiments/
└── results/
    ├── experiment_results.csv          ← Mateusz's primary input
    ├── K01_logistic_regression/
    │   ├── oof_predictions.npy         ← shape (5000,), float64
    │   ├── fold_assignments.npy        ← shape (5000,), int, values 0–4
    │   ├── fold_0_model.pkl            ← fitted CalibratedClassifierCV
    │   ├── fold_1_model.pkl
    │   ├── fold_2_model.pkl
    │   ├── fold_3_model.pkl
    │   ├── fold_4_model.pkl
    │   └── feature_names.json          ← list of K feature name strings
    ├── K01_lgbm/
    │   └── (same structure)
    ├── K01_random_forest/
    │   └── (same structure)
    ├── K02_logistic_regression/
    │   └── ...
    ...
    ├── K20_random_forest/
    │   └── ...
    └── trials/
        └── K01/
            ├── logistic_regression_trials.csv
            ├── lgbm_trials.csv
            └── random_forest_trials.csv
```

---

## 5. How to Load Hubert's Outputs (code examples)

### Load OOF predictions for a given (K, model)

```python
import numpy as np
import pandas as pd

df = pd.read_csv("experiments/results/experiment_results.csv")
row = df[(df["K"] == 5) & (df["Model"] == "lgbm")].iloc[0]

oof_probs = np.load(row["OOFPredictions_Npy_Path"])  # shape (5000,)
```

### Load all OOF predictions into a dict (the format Mod 6 expects)

```python
import numpy as np
import pandas as pd

df = pd.read_csv("experiments/results/experiment_results.csv")

oof_dict = {}
for _, row in df.iterrows():
    key = (int(row["K"]), str(row["Model"]))
    oof_dict[key] = np.load(row["OOFPredictions_Npy_Path"])
```

### Load the 5 fold models for a given (K, model)

```python
import pickle
from pathlib import Path

model_dir = Path(row["OuterModels_Pkl_Path"])
fold_models = []
for i in range(5):
    with open(model_dir / f"fold_{i}_model.pkl", "rb") as f:
        fold_models.append(pickle.load(f))

# Each model: sklearn CalibratedClassifierCV wrapping a Pipeline.
# It already contains the fitted StandardScaler (for LogReg).
# Call: fold_models[i].predict_proba(X_test_filtered)[:, 1]
```

### Load feature names (to filter x_test.txt)

```python
import json

with open(row["FeatureNames_Path"]) as f:
    feature_names = json.load(f)   # list of K strings, e.g. ["V52", "V211", ...]
```

---

## 6. Contracts Mateusz Must Satisfy

### Input to Mod 6 (Per-K Ensemble Weight Optimizer)

```python
# Signature from WORK_DIVISION.md:
optimize_per_k_ensemble_weights(
    oof_predictions_dict: dict[tuple[int, str], np.ndarray],  # (K, model) → (5000,)
    y_train: np.ndarray,                                       # (5000,)
    K_range: range = range(1, 21),
) -> dict[int, dict]
```

- Load `y_train.txt` with `np.loadtxt("y_train.txt", dtype=int)`.
- Build `oof_predictions_dict` as shown above.

### Input to Mod 7 (Calibrator & K Selector)

```python
# Mod 7 expects the output of Mod 6 + y_train.
calibrate_and_select_best_k(per_k_ensembles, y_train, visualization_dir=None)
```

### Input to Mod 8 (Test Predictor)

```python
predict_on_test_set(
    x_test_path: str,
    best_k_config: dict,        # output of Mod 7
    ensemble_models_dir: str,   # "experiments/results/"
)
```

For test-time inference with a given ensemble member:
1. Load `x_test.txt` as a pandas DataFrame (same `pd.read_csv(..., sep=r"\s+", header=0)` pattern as Mod 1).
2. Filter columns to `feature_names` from `feature_names.json`.
3. Load the 5 fold models from `fold_{i}_model.pkl`.
4. For each fold model: call `model.predict_proba(X_test_filtered)[:, 1]`.
   **Do not re-apply a scaler** — it is already inside the Pipeline wrapped by the CalibratedClassifierCV.
5. Average the 5 probabilities → one vector per ensemble member.
6. Blend with weights, then calibrate.

---

## 7. Key Design Decisions Mateusz Should Know

### Calibration method
All fold models use **Platt Scaling (sigmoid)** via
`CalibratedClassifierCV(cv=3, method='sigmoid')`. This is intentional (see
EXPERIMENT_PLAN.md §4.3). Do NOT use isotonic.

### StandardScaler is inside the Pipeline
For `logistic_regression`, `StandardScaler` is part of the `Pipeline` stored in
the pickle. It was fitted on the outer training fold. There is **no separate
scaler object to load**. Just call `model.predict_proba(X_raw_filtered)`.

### LGBM feature name alignment
LGBM was trained from numpy arrays (not DataFrames), so `predict_proba` should
receive numpy arrays, not pandas DataFrames, to avoid sklearn warnings.

### Profit formula
```
Profit = (TP × 10) - (FP × 5) - (K × 200)
```
Hard constraint: at most 1000 customers selected.
Variable cost is global: regardless of ensemble size, it is K × 200 once.

### Safety margin on threshold (EXPERIMENT_PLAN.md §5.1)
After Mod 7 finds `tau_opt`, enforce `tau_opt = max(tau_opt, 0.38)` before
writing the submission.

---

## 8. Tests

All tests for Hubert's modules live in `tests/`:

```
tests/
├── test_data_manager.py          (10 tests)
├── test_optuna_tuner.py          (7 tests)
├── test_outer_cv_engine.py       (7 tests)
├── test_individual_thresholds.py (7 tests)
└── test_master_loop.py           (5 tests)
```

Run them:

```bash
uv run pytest tests/ -v -W ignore::FutureWarning -W ignore::UserWarning
```

All 37 tests pass on a clean checkout.

---

## 9. Project Tooling

- **Python 3.14**, managed via `uv`.
- All commands must be prefixed with `uv run`.
- `pyproject.toml` already lists all dependencies.
- Use `uv add <pkg>` to add more packages.
