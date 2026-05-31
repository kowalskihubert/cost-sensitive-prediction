# Implementation Plan

To enable executing the precise Nested-CV optimization strategy defined in `EXPERIMENT_PLAN.md`, the codebase should be strictly modular and built around Scikit-Learn, Optuna, and Pandas.

## 1. Zosia's Contract (Feature Handoff)
- **The Contract:** Zosia will output an ordered list of selected feature names (best first) as a JSON file, e.g., `selected_features.json`.
- **Format Example:** `["V52", "V211", "V8", "V312", ...]`
- **Usage:** Hubert and Mati's pipeline will open this file, read the list, and use a parameter `top_k` to select `features[:top_k]` for reading subset data.

## 2. Pipeline Modules

### Mod 1: Data Manager
- **Input:** `x_train.txt`, `y_train.txt`, K (int).
- **Operation:** Select top K features from Zosia's list. Return X, y.

### Mod 2: Optuna Hyperparameter Tuner
- **Function:** `tune_hyperparameters(X_train, y_train, model_type)`
- **Logic:** 
  - Define a 3-Fold Stratified CV on X_train.
  - Use Optuna to search parameters (e.g., C for LogReg, max_depth for LGBM).
  - **Metric:** Calculate log_loss across the 3 folds.
  - Return the best parameter dictionary.

### Mod 3: The Outer Nested CV Engine
- **Function:** `run_outer_cv(X, y, model_type, best_params)`
- **Logic:**
  - Initialize a 5-Fold Stratified CV.
  - For each fold (T_outer, V_outer):
    - Instantiate the base model with best_params.
    - Apply scaling pipeline (if Model falls under SVM/LogReg).
    - Wrap in CalibratedClassifierCV(cv=3, method='sigmoid'). **Use Platt Scaling (sigmoid), not isotonic**.
    - fit() on T_outer.
    - predict_proba() on V_outer.
    - Store the fitted calibrated model for later inference on the test set.
  - Return concatenated Out-Of-Fold (OOF) probabilities for the entire dataset (5000 rows), plus the 5 fitted models.

### Mod 4: Empirical Threshold Optimizer (Individual Models)
- **Function:** `find_individual_model_threshold(y_true, oof_probs, K)`
- **Logic:**
  - Bind y_true to oof_probs and sort descending by oof_probs.
  - Iterate to calculate cumulative TP, FP, and expected profit at each threshold.
  - Apply the profit formula: Profit = (TP × 10) - (FP × 5) - (K × 200).
  - Return the maximum theoretical profit, the corresponding probability threshold tau_K, and the index where maximum is achieved.

### Mod 5: The Master Experiment Loop (Hubert)
- Iterate over K in {1, ..., 20}.
- Iterate over model types (LogReg, LGBM, RF).
- Execute Mod 1 -> Mod 2 -> Mod 3 -> Mod 4.
- Append a record [K, Model, BestParams, MaxOOFProfit, OptThreshold, CapLiftMetrics, OuterModels, OOFPredictions] to our tracking mechanism and save to experiment_results.csv.
- Store raw OOF predictions for each model to enable downstream ensemble combination.

### Mod 6: Per-K Ensemble Weight Optimizer (Mateusz)
- **Function:** `optimize_per_k_ensemble_weights(oof_predictions_dict, y_train, K_range=range(1,21))`
- **Logic:**
  - Input: Dictionary mapping (K, Model_Name) -> OOF predictions array (5000,).
  - For each fixed K:
    - Extract the 3 OOF arrays: LogReg(K), LGBM(K), RF(K).
    - Rank by individual profit. Select top 2-3 models for ensemble (or single best if dominant).
    - Use Grid Search on weights: w_i in {0.0, 0.1, 0.2, ..., 1.0}.
    - Optimize on **full OOF data** (all 5000 rows) to maximize (TP × 10) - (FP × 5) - (K × 200).
    - Store best weights w^(K)_best and profit Profit^(K)_max.
  - Return: Dictionary of per-K ensemble weights and profits.

### Mod 7: Per-K Ensemble Calibrator & Global K Selection (Mateusz)
- **Function:** `calibrate_and_select_best_k(per_k_ensembles, oof_predictions_dict, y_train)`
- **Logic:**
  - For each K from 1 to 20:
    - Compute blended OOF: P^(K)_blend = sum(w^(K)_i * P_i) for ensemble members.
    - Fit CalibratedClassifierCV(cv=3, method='sigmoid') on P^(K)_blend. **Use Platt Scaling, not isotonic**.
    - Generate Cap/Lift curves and verify ensemble improves over best individual model.
    - Record: (K, ensemble_weights, calibrator, profit^(K)_max).
  - Global Selection: Find K* = argmax Profit^(K)_max across all K.
  - Return: K*, ensemble configuration, calibrator, threshold tau_opt.
