# Work Division: Implementation Tasks

This document assigns implementation work between Hubert and Mateusz. Zosia handles variable rankings (EXPERIMENT_PLAN.md §1, Contract).

---

## Hubert: Core Experimental Engine

Hubert owns all tasks related to **training individual models, hyperparameter optimization, and generating OOF predictions** across all feature subsets and model families. His output is a complete experimental record ready for per-K ensemble selection.

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
    - LogReg: `C in [0.001, 10]`, `penalty in {l1, l2}`
    - LGBM: `max_depth in [2, 5]`, `learning_rate in [0.01, 0.1]`, `colsample_bytree in [0.5, 1.0]`, `num_leaves in [10, 50]`, `min_data_in_leaf in [5, 20]`
    - RF: `max_depth in [3, 8]`, `min_samples_leaf in [5, 20]`
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
  - Metadata: fold assignments
- **Details:**
  - Implement 5-Fold Stratified CV
  - For each fold:
    - Wrap model in `CalibratedClassifierCV(cv=3, method='sigmoid')` *on the training fold only*. **Use Platt Scaling (sigmoid), NOT isotonic**.
    - Fit and predict on the validation fold
    - Store the calibrated model
  - Apply `StandardScaler` inside Pipeline (fit on train, apply to val)
  - Concatenate all OOF predictions in original row order
- **Deliverable:** A reusable set of 5 fitted, calibrated models + OOF array

### Responsibility 4: Individual Model Threshold Finding (Mod 4)
- **File(s):** `src/analysis/individual_thresholds.py`
- **Function:** `find_individual_model_threshold(y_true, oof_probs, K)`
- **Input:** True labels, OOF predictions, number of variables
- **Output:** 
  - Maximum profit achievable
  - Corresponding probability threshold
  - Index where maximum is achieved
- **Details:**
  - Sort OOF predictions descending, bind labels
  - Calculate cumulative TP, FP, profit
  - Formula: Profit = (TP × 10) - (FP × 5) - (K × 200)
  - Return sorted arrays for downstream Cap/Lift analysis

### Responsibility 5: Master Experiment Loop (Mod 5)
- **File(s):** `src/experiments/master_loop.py`
- **Function:** `run_all_experiments(zosia_features_json_path, models=['logistic_regression', 'lgbm', 'random_forest'], K_range=range(1, 21))`
- **Input:**
  - Path to Zosia's features JSON
  - Optional: list of model types, range of K values
- **Output:**
  - `experiment_results.csv` with columns: `K, Model, BestParams, OOFProfit, OptThreshold, OuterModels_Pkl_Path, OOFPredictions_Npy_Path`
  - Organized directory: `experiments/results/` containing pickle files and numpy arrays for every (K, Model) pair
- **Details:**
  - Loop over K in {1, ..., 20}
  - Loop over model types (LogReg, LGBM, RF)
  - Call Mod 1 → Mod 2 → Mod 3 → Mod 4 sequentially
  - Log progress, timing, and any warnings
  - **No overlap:** Hubert does NOT compute ensembles or select best K.

---

## Mateusz: Post-Experiment Analysis & Deployment

Mateusz owns all tasks related to **per-K ensemble selection, calibration, global K optimization, threshold validation, and test-time inference**. His work begins after Hubert's `experiment_results.csv` is complete.

### Responsibility 1: Per-K Ensemble Weight Optimizer (Mod 6)
- **File(s):** `src/ensemble/per_k_weight_optimizer.py`
- **Function:** `optimize_per_k_ensemble_weights(oof_predictions_dict, y_train, K_range=range(1,21))`
- **Input:**
  - Dictionary: `{(K, Model_Name): oof_probs_array}` — all experiments from `experiment_results.csv`
  - True training labels (5000,)
  - Range of K values to process
- **Output:**
  - Dictionary of per-K configurations: `{K: {members: [...], weights: {...}, profit: ...}}`
- **Details:**
  - For each K:
    - Extract the 3 OOF arrays: LogReg(K), LGBM(K), RF(K)
    - Rank by individual profit
    - Select top 2-3 models for ensemble (or single best if dominant)
    - Use **Grid Search** on weights: w_i in {0.0, 0.1, 0.2, ..., 1.0}
    - Optimize on **full OOF data** (all 5000 rows) to maximize (TP × 10) - (FP × 5) - (K × 200)
    - Store best weights w^(K)_best and profit Profit^(K)_max
  - Return: Complete per-K ensemble registry
- **No overlap:** Hubert does NOT compute ensemble weights.

### Responsibility 2: Per-K Ensemble Calibrator & Global K Selection (Mod 7)
- **File(s):** `src/ensemble/calibrator_and_k_selector.py`
- **Function:** `calibrate_and_select_best_k(per_k_ensembles, y_train, visualization_dir=None)`
- **Input:**
  - Per-K ensemble registry (from Mod 6)
  - True training labels
  - Optional: directory for plots
- **Output:**
  - Best K* configuration: (K*, members, weights, calibrator, tau_opt, profit)
  - Per-K calibration plots and Cap/Lift curves (for the report)
- **Details:**
  - For each K from 1 to 20:
    - Compute blended OOF: P^(K)_blend = sum(w^(K)_i × P_i)
    - Fit `CalibratedClassifierCV(cv=3, method='sigmoid')` on P^(K)_blend. **Use Platt Scaling, NOT isotonic**
    - Generate Cap/Lift curves and verify ensemble improves over best individual model
    - Record: (K, ensemble_weights, calibrator, profit^(K)_max)
  - Global Selection: Find K* = argmax Profit^(K)_max
  - Apply Safety Margin: if tau_opt < 0.38, enforce tau_opt = 0.38
  - Return: Best K* configuration with all metadata
- **Deliverable:** `best_k_config.pkl` and calibration visualizations

### Responsibility 3: Test-Time Configuration & Inference
- **File(s):** `src/inference/test_predictor.py`
- **Function:** `predict_on_test_set(x_test_path, best_k_config, ensemble_models_dir)`
- **Input:**
  - `x_test.txt` (5000 × 500)
  - Best K* configuration and calibrator
  - Directory containing 5 trained models for each ensemble member
- **Output:**
  - Final calibrated ensemble predictions (5000,) for the test set
  - Metadata: variable indices used (all from K*)
- **Details:**
  - For each ensemble member at K*:
    - Load `x_test.txt`, filter to K* features
    - Load the 5 fitted calibrated models
    - Apply StandardScaler (fitted during training)
    - Predict with each of 5 models, average → P_i_test
  - Blend: P_blend_test = sum(w_i × P_i_test)
  - Apply calibration transformer: P_final = calibrator.transform(P_blend_test)
  - Return P_final (5000,)
- **No overlap:** Hubert does NOT touch test inference.

### Responsibility 4: Final Selection & Deliverable Generation
- **File(s):** `src/inference/submit.py`
- **Function:** `generate_submission(P_final, tau_opt, K*, output_dir)`
- **Input:**
  - Final calibrated test predictions (5000,)
  - Threshold tau_opt
  - Optimal K* value (for variable list)
  - Output directory
- **Output:**
  - `STUDENT1_STUDENT2_STUDENT3_obs.txt` (selected customer indices, one per line, max 1000)
  - `STUDENT1_STUDENT2_STUDENT3_vars.txt` (variable indices: first K* features from Zosia's JSON)
- **Details:**
  - Sort test predictions descending by probability
  - Select customers where P_final >= tau_opt (up to 1000 max)
  - Write indices (verify 0-based vs. 1-based convention)
  - Write K* variable indices to vars file
  - Verify file formats match examples
- **Deliverable:** Two `.txt` files ready for submission

---

## Summary: No Overlap, Clear Sequencing

| Phase | Owner | Task |
|-------|-------|------|
| **Phase 1: Experiments (Mod 1-5)** | Hubert | Run all (K, Model) combinations, generate OOF predictions and experiment_results.csv |
| **Phase 2: Per-K Ensembles (Mod 6)** | Mateusz | For each K: select top models, optimize weights via grid search on full OOF |
| **Phase 3: Calibration & K Selection (Mod 7)** | Mateusz | For each K: calibrate ensemble (Platt), generate Cap/Lift. Select K* = argmax Profit^(K) |
| **Phase 4: Test Inference** | Mateusz | Load test data, apply K* ensemble (5-fold averaging + blending + calibration) |
| **Phase 5: Submission** | Mateusz | Cut-off at tau_opt, generate `.txt` files |

**Key Principle:** Hubert produces `experiment_results.csv` with trained models and OOF arrays. Mateusz takes that output and:
1. Builds per-K ensembles (not cross-K, which would violate cost constraints)
2. Selects the single best K* globally
3. Produces the final submission

No code conflicts, clean handoff via files and CSVs.

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
├── analysis/
│   ├── individual_thresholds.py   (Hubert)
│   └── per_k_threshold_optimizer.py (Mateusz)
├── ensemble/
│   ├── per_k_weight_optimizer.py  (Mateusz)
│   └── calibrator_and_k_selector.py (Mateusz)
└── inference/
    ├── test_predictor.py          (Mateusz)
    └── submit.py                  (Mateusz)

experiments/
├── results/
│   ├── experiment_results.csv     (Hubert output)
│   └── (K, Model) subdirs with pickles + numpy arrays
└── visualizations/                (Mateusz output: Cap/Lift plots, calibration curves)
```
