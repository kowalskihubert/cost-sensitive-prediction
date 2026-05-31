# Work Division: Implementation Tasks

This document assigns implementation work between Hubert and Mateusz. Zosia handles variable rankings (EXPERIMENT_PLAN.md §1, Contract).

---

## Hubert: Core Experimental Engine

Hubert owns all tasks related to **training individual models, hyperparameter optimization, and generating OOF predictions** across all feature subsets and model families. His output is a complete experimental record ready for ensemble selection.

### Responsibility 1: Data Manager (Mod 1)
- **File(s):** `src/data_manager.py`
- **Function:** `load_and_filter_features(K, zosia_features_json_path)`
- **Input:** 
  - `x_train.txt`, `y_train.txt` (from project root)
  - Zosia's `selected_features.json` (ordered list of feature names)
  - Integer `K` (number of top features to select)
- **Output:** 
  - `X_filtered` (5000 × K), `y` (5000,)
  - Metadata: column names for the selected features
- **Details:**
  - Robust error handling for missing files, format mismatches
  - Verify that Zosia's JSON is valid and all feature names exist in `x_train.txt`

### Responsibility 2: Optuna Hyperparameter Tuner (Mod 2)
- **File(s):** `src/tuning/optuna_tuner.py`
- **Function:** `tune_hyperparameters(X_train, y_train, model_type, n_trials=50)`
- **Input:**
  - Training data `X_train`, `y_train`
  - Model type string: `'logistic_regression'`, `'lgbm'`, `'random_forest'`
  - Optional: number of trials
- **Output:**
  - Dictionary of best hyperparameters found
  - Trial history (for diagnostics)
- **Details:**
  - Implement 3-fold Stratified CV inside Optuna
  - **Metric:** Minimize Log-Loss (not AUC, not F1)
  - **Hyperparameter bounds** (from EXPERIMENT_PLAN.md):
    - LogReg: `C ∈ [0.001, 10]`, `penalty ∈ {l1, l2}`
    - LGBM: `max_depth ∈ [2, 5]`, `learning_rate ∈ [0.01, 0.1]`, `colsample_bytree ∈ [0.5, 1.0]`, `num_leaves ∈ [10, 50]`, `min_data_in_leaf ∈ [5, 20]`
    - RF: `max_depth ∈ [3, 8]`, `min_samples_leaf ∈ [5, 20]`
  - Save trial history to CSV for inspection
- **No overlap:** Mateusz does NOT touch this module.

### Responsibility 3: Outer Nested CV Engine (Mod 3)
- **File(s):** `src/cv/outer_cv_engine.py`
- **Function:** `run_outer_cv_with_calibration(X, y, model_type, best_params)`
- **Input:**
  - Full training data `X` (5000 × K), `y` (5000,)
  - Model type and best hyperparameters
- **Output:**
  - OOF probability array (5000,) — predictions for all training rows
  - 5 fitted calibrated models (stored as pickles) for later test-time use
  - Metadata: fold assignments (which training rows belong to which fold)
- **Details:**
  - Implement 5-Fold Stratified CV
  - For each fold:
    - Wrap model in `CalibratedClassifierCV(cv=3, method='isotonic')` *on the training fold only*
    - Fit and predict on the validation fold
    - Store the calibrated model
  - Apply `StandardScaler` inside Pipeline (fit on train, apply to val)
  - Concatenate all OOF predictions in original row order
- **Deliverable:** A reusable set of 5 fitted, calibrated models + OOF array

### Responsibility 4: Master Experiment Loop (Mod 5)
- **File(s):** `src/experiments/master_loop.py`
- **Function:** `run_all_experiments(zosia_features_json_path, models=['logistic_regression', 'lgbm', 'random_forest'], K_range=range(1, 21))`
- **Input:**
  - Path to Zosia's features JSON
  - Optional: list of model types, range of K values
- **Output:**
  - `experiment_results.csv` with columns: `K, Model, BestParams, OOFProfit, OuterModels_Pkl_Path, OOFPredictions_Npy_Path`
  - Organized directory: `experiments/results/` containing pickle files and numpy arrays for every (K, Model) pair
- **Details:**
  - Loop over K ∈ {1, ..., 20}
  - Loop over model types
  - Call Mod 1 → Mod 2 → Mod 3 sequentially
  - After Mod 3, compute OOF profit (no calibration validation yet — that's Mateusz's job)
  - Log progress, timing, and any warnings
  - **No overlap:** Hubert does NOT compute Cap/Lift curves or ensemble weights.

### Responsibility 5: Utilities & Helpers
- **File(s):** `src/utils/profit_calculator.py`
- **Function:** `calculate_oof_profit(y_true, oof_probs, K, sort_order=True)`
- **Input:** True labels, OOF predictions, number of variables
- **Output:** 
  - Cumulative profit array (sorted by probability descending)
  - Maximum profit achievable
  - Number of customers at maximum profit point
- **Details:**
  - Formula: `Profit = (TP * 10) - (FP * 5) - (K * 200)`
  - Return sorted arrays for downstream Cap/Lift analysis (Mateusz's use)

---

## Mateusz: Post-Experiment Analysis & Deployment

Mateusz owns all tasks related to **ensemble selection, calibration, threshold validation, and test-time inference**. His work begins after Hubert's `experiment_results.csv` is complete.

### Responsibility 1: Empirical Threshold Optimizer with Cap/Lift Curves (Mod 4)
- **File(s):** `src/analysis/threshold_optimizer.py`
- **Function:** `find_optimal_threshold_with_caplift(y_true, oof_probs, K, visualization_dir=None)`
- **Input:**
  - True labels (5000,)
  - OOF predictions (5000,) — already calibrated (from Hubert's Mod 3)
  - Number of variables K (for profit calculation)
  - Optional: directory to save plots
- **Output:**
  - Optimal threshold `τ_K` (probability value)
  - Maximum achievable profit
  - Index of cut-off (how many customers to select)
  - Cap and Lift curve data (numpy arrays or DataFrames)
  - Plots: cap curve, lift curve, profit curve
- **Details:**
  - Sort OOF predictions descending, bind true labels
  - Compute cumulative TP, FP, profit iteratively
  - Generate cap curve: cumulative true positive rate vs. % population
  - Generate lift curve: performance vs. baseline
  - Detect kink points (use Savitzky-Golay or manually inspect 2nd derivative)
  - Find `τ_K` at the point maximizing profit
  - **Safety Margin Rule:** If `τ_K < 0.38`, enforce `τ_K = 0.38`
  - Save plots to `visualizations/` for the report
- **No overlap:** Hubert does NOT generate Cap/Lift curves.

### Responsibility 2: Ensemble Weight Optimizer (Mod 6)
- **File(s):** `src/ensemble/weight_optimizer.py`
- **Function:** `optimize_ensemble_weights(oof_predictions_dict, y_train, n_models=5)`
- **Input:**
  - Dictionary: `{(K, Model_Name): oof_probs_array}` — all experiments from `experiment_results.csv`
  - True training labels (5000,)
  - Number of top models to ensemble (default 5-7)
- **Output:**
  - List of top N ensemble candidates: `[(K_i, Model_i), ...]`
  - Optimal weights: `{(K_i, Model_i): w_i}`
  - Validation profit (meta-validation set)
- **Details:**
  - Parse `experiment_results.csv` and load all OOF arrays from disk
  - Rank all (K, Model) pairs by their individual OOF profit (descending)
  - Select top N
  - Stratified split: 3000 rows (meta-train), 2000 rows (meta-val)
  - Grid search or Optuna: find weights `w_i` to maximize meta-train profit
  - Validate on meta-val set
  - Return weights and best ensemble configuration
- **Deliverable:** `ensemble_config.json` containing ensemble members and their weights

### Responsibility 3: Ensemble Calibrator (Mod 7)
- **File(s):** `src/ensemble/calibrator.py`
- **Function:** `calibrate_ensemble(P_blend, y_train, method='isotonic')`
- **Input:**
  - Blended OOF predictions (weighted average of top N models)
  - True training labels
- **Output:**
  - Fitted calibration transformer (pickled)
  - Calibration quality metrics (reliability curve, Brier score, etc.)
- **Details:**
  - Compute `P_blend = sum(w_i * P_i)` using ensemble weights
  - Fit `CalibratedClassifierCV(cv=3, method=method)` on the full training set
  - Generate reliability curve (calibration plot)
  - Compare ensemble calibration to best individual model
  - Save transformer for test-time use
- **Deliverable:** `ensemble_calibrator.pkl`

### Responsibility 4: Threshold Validation & Cap/Lift for Ensemble
- **File(s):** `src/analysis/threshold_optimizer.py` (extension of Responsibility 1)
- **Function:** `apply_and_validate_ensemble_threshold(P_calibrated_ensemble, y_true, K_union)`
- **Input:**
  - Calibrated ensemble OOF predictions
  - True labels
  - Union of all K values used in the ensemble
- **Output:**
  - Final threshold `τ_opt` (including any safety margin)
  - Cap/Lift validation plots
  - Comparison table: individual top model vs. ensemble
- **Details:**
  - Run `find_optimal_threshold_with_caplike` on the ensemble OOF
  - Verify that ensemble Cap/Lift curve is at least as good as the best individual model
  - Document kink points and safety margin decisions
  - Save validation report

### Responsibility 5: Test-Time Configuration Lock-in & Inference (TEST_TASK.md Steps 1-3)
- **File(s):** `src/inference/test_predictor.py`
- **Function:** `predict_on_test_set(x_test_path, ensemble_config, calibrated_ensemble_transformer, ensemble_models_dir)`
- **Input:**
  - `x_test.txt` (5000 × 500)
  - Ensemble configuration (members, weights)
  - Calibration transformer
  - Directory containing 5 trained models for each ensemble member
- **Output:**
  - Final calibrated ensemble predictions (5000,) for the test set
  - Metadata: which feature indices were used (union across ensemble members)
- **Details:**
  - For each ensemble member (K_i, Model_i):
    - Load `x_test.txt`, filter to K_i features
    - Load the 5 fitted calibrated models from the outer CV
    - Apply `StandardScaler` (fitted during training) to test data
    - Predict with each of 5 models, average → P_i_test
  - Blend: `P_blend_test = sum(w_i * P_i_test)`
  - Apply calibration transformer: `P_final = calibrator.transform(P_blend_test)`
  - Return `P_final` (5000,)
- **No overlap:** Hubert does NOT touch test inference.

### Responsibility 6: Final Selection & Deliverable Generation (TEST_TASK.md Steps 4-5)
- **File(s):** `src/inference/submit.py`
- **Function:** `generate_submission(P_final, tau_opt, ensemble_config, output_dir)`
- **Input:**
  - Final calibrated test predictions (5000,)
  - Threshold `τ_opt`
  - Ensemble configuration (for variable list)
  - Output directory
- **Output:**
  - `STUDENT1_STUDENT2_STUDENT3_obs.txt` (selected customer indices, one per line, max 1000)
  - `STUDENT1_STUDENT2_STUDENT3_vars.txt` (union of variable indices used in ensemble)
- **Details:**
  - Sort test predictions descending by probability
  - Select customers where `P_final >= τ_opt` (up to 1000 max)
  - Write indices to file (verify 0-based vs. 1-based convention from data spec)
  - Union all variable indices from ensemble members
  - Verify file formats match example files
- **Deliverable:** Two `.txt` files ready for submission

---

## Summary: No Overlap, Clear Sequencing

| Phase | Owner | Task |
|-------|-------|------|
| **Phase 1: Experiments** | Hubert | Run all (K, Model) combinations, generate OOF predictions and `experiment_results.csv` |
| **Phase 2: Ensemble Selection** | Mateusz | Rank experiments, select top N, find optimal weights |
| **Phase 3: Calibration & Validation** | Mateusz | Calibrate ensemble, validate with Cap/Lift curves, determine `τ_opt` |
| **Phase 4: Test Inference** | Mateusz | Load test data, apply ensemble models, generate final predictions |
| **Phase 5: Submission** | Mateusz | Cut-off at threshold, generate `.txt` files |

**Key Principle:** Hubert produces `experiment_results.csv` with trained models and OOF arrays. Mateusz takes that output and produces the final submission. No code conflicts, clean handoff via files and CSVs.

---

## File Structure (for reference)

```
src/
├── data_manager.py                 (Hubert)
├── tuning/
│   └── optuna_tuner.py            (Hubert)
├── cv/
│   └── outer_cv_engine.py         (Hubert)
├── experiments/
│   ├── master_loop.py             (Hubert)
│   └── results/                   (Hubert output: pickles + numpy arrays)
├── utils/
│   └── profit_calculator.py       (Hubert)
├── analysis/
│   └── threshold_optimizer.py     (Mateusz)
├── ensemble/
│   ├── weight_optimizer.py        (Mateusz)
│   └── calibrator.py              (Mateusz)
└── inference/
    ├── test_predictor.py          (Mateusz)
    └── submit.py                  (Mateusz)

experiments/
├── results/
│   ├── experiment_results.csv     (Hubert output)
│   └── (K, Model) subdirs with pickles + numpy arrays
└── visualizations/                (Mateusz output: Cap/Lift plots)
```
