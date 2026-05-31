# Implementation Plan

To enable executing the precise Nested-CV optimization strategy defined in `EXPERIMENT_PLAN.md`, the codebase should be strictly modular and built around Scikit-Learn, Optuna, and Pandas.

## 1. Zosia's Contract (Feature Handoff)
- **The Contract:** Zosia will output an ordered list of selected feature names (best first) as a JSON file, e.g., `selected_features.json`.
- **Format Example:** `["V52", "V211", "V8", "V312", ...]`
- **Usage:** Our pipeline will read this list and slice the first `K` columns from the dataset.

## 2. Pipeline Modules

### Mod 1: Data Manager
- **Input:** `x_train.txt`, `y_train.txt`, $K$ (int).
- **Operation:** Select top $K$ features from Zosia's list. Return `X, y`.

### Mod 2: Optuna Hyperparameter Tuner
- **Function:** `tune_hyperparameters(X_train, y_train, model_type)`
- **Logic:** 
  - Define a 3-Fold Stratified CV on `X_train`.
  - Use Optuna to search parameters (e.g., `C` for LogReg, `max_depth` for LGBM).
  - **Metric:** Calculate `log_loss` across the 3 folds.
  - Return the best parameter dictionary.

### Mod 3: The Outer Nested CV Engine
- **Function:** `run_outer_cv(X, y, model_type, best_params)`
- **Logic:**
  - Initialize a 5-Fold Stratified CV.
  - For each fold ($T_{outer}$, $V_{outer}$):
    - Instantiate the base model with `best_params`.
    - Apply scaling pipeline (if Model falls under SVM/LogReg).
    - Wrap in `CalibratedClassifierCV(cv=3, method='isotonic')` (learning calibration from inner splits of $T_{outer}$).
    - `fit()` on $T_{outer}$.
    - `predict_proba()` on $V_{outer}$.
    - Store the fitted calibrated model for later inference on the test set.
  - Return concatenated Out-Of-Fold (OOF) probabilities for the entire dataset (5000 rows), plus the 5 fitted models.

### Mod 4: Empirical Threshold Optimizer with Cap/Lift Validation
- **Function:** `find_optimal_threshold_with_curves(y_true, oof_probs, K)`
- **Logic:**
  - Bind `y_true` to `oof_probs` and sort descending by `oof_probs`.
  - Iterate to calculate cumulative TP, FP, and expected profit at each threshold.
  - Apply the profit formula: $\text{Profit} = (TP \times 10) - (FP \times 5) - (K \times 200)$.
  - **Cap/Lift Generation:**
    - Compute cap curve: $\text{Cap}(\alpha) = \frac{\sum_{i=1}^{\lfloor \alpha \times N \rfloor} \text{TP}_i}{\text{Total\_Positives}}$ for $\alpha \in [0, 1]$.
    - Compute lift curve: $\text{Lift}(\alpha) = \frac{\text{Cap}(\alpha) / \alpha}{\text{Base\_Rate}}$.
    - Detect kinks: Look for where the second derivative changes sign (Savitzky-Golay filter can help).
  - **Threshold Selection:** Find index $i \le 1000$ maximizing cumulative profit. Extract the probability at index $i$ as $\tau_K$.
  - **Validation:** If $\tau_K$ resides in a low-lift region (post-kink), enforce $\tau_K := \max(\tau_K, 0.38)$.
  - Return: maximum profit, $\tau_K$, the Cap/Lift plot artifacts (for visualization).

### Mod 5: The Master Experiment Loop
- Iterate over $K \in \{1, \dots, 20\}$.
- Iterate over model types (LogReg, LGBM, RF).
- Execute Mod 1 -> Mod 2 -> Mod 3 -> Mod 4.
- Append a record `[K, Model, BestParams, MaxOOFProfit, OptThreshold, CapLiftMetrics, OuterModels, OOFPredictions]` to our tracking mechanism and save to `experiment_results.csv`.
- Store raw OOF predictions for each model to enable downstream ensemble combination.

### Mod 6: Ensemble Weight Optimizer
- **Function:** `optimize_ensemble_weights(oof_predictions_dict, y_train)`
- **Logic:**
  - Input: Dictionary mapping $(K, \text{Model})$ -> OOF predictions array (5000,).
  - Select top $N$ candidates by profit from `experiment_results.csv`.
  - Randomly split training data: 3000 (meta-train) and 2000 (meta-val).
  - Use grid search or Optuna to find $w_1, \dots, w_N$ maximizing cumulative profit on meta-train.
  - Validate on meta-val. Record ensemble weights and validation profit.
  - Return: optimal weights dictionary.

### Mod 7: Ensemble Calibrator
- **Function:** `calibrate_ensemble(P_blend, y_train)`
- **Logic:**
  - Input: Blended OOF predictions (weighted average of top $N$ models).
  - Apply `CalibratedClassifierCV(cv=3, method='isotonic')` to learn the final calibration.
  - Return: fitted calibration transformer.
  - Generate Cap/Lift curves and compare to top individual model.
