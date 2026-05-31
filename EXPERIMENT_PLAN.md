# Precise Experiment Flow & Optimization Strategy

## 1. The Optimization Problem & Strategy
We are optimizing a discontinuous, non-differentiable step-function:
$$ \text{Total Profit} = (TP \times 10) - (FP \times 5) - (K \times 200) $$
With a hard constraint: $\text{Total Selected} \le 1000$.

Because the cost of features ($K$) is global and massive, **we cannot jointly optimize $K$, Hyperparameters, and Threshold concurrently with gradient or Bayesian methods out-of-the-box.** 

**Our Strategic Choice: Grid-Search over $K$, Bayesian Optimization for Hyperparams, Per-$K$ Ensemble, Empirical Maximization for Thresholds.**
1. **Outer Grid Loop:** Iterate over $K \in \{1, 2, 3, \dots, 20\}$ variables based on Zosia's ranked list.
2. **Inner Bayesian Tuning:** For a given $K$, optimize hyperparameters using Optuna targeting Log-Loss.
3. **Per-$K$ Ensemble:** For each $K$, ensemble the best models that all use that $K$ value.
4. **Threshold Selection:** Find empirical threshold using Cap/Lift curves.

## 2. Cross-Validation Architecture (Nested CV)
To avoid data leakage during hyperparameter tuning and calibration, we use a **Nested Stratified CV Design**.

### Outer Loop (5-Fold Stratified CV) - For Threshold & Generalization
- **Splits:** 5 Folds. For each iteration, 4 Folds ($T_{\text{outer}}$) and 1 Fold ($V_{\text{outer}}$).
- **Purpose:** To generate an unbiased, perfectly clean array of out-of-fold (OOF) probabilities of size exactly equal to the train set (5000x1). 

### Inner Loop (3-Fold Stratified CV) - For Hyperparameters & Calibration
- **Splits:** Inside $T_{\text{outer}}$, we perform another 3-Fold split. 
- **Purpose 1 (Optuna):** Optuna searches hyperparameters comparing Log-Loss across these 3 inner folds.
- **Purpose 2 (Calibration):** We use Scikit-Learn's `CalibratedClassifierCV(cv=3, method='sigmoid')` over $T_{\text{outer}}$. It trains the base estimators on inner training splits and learns the calibration mapping (Platt/sigmoid) on the inner validation splits. The final meta-estimator then predicts on the pristine $V_{\text{outer}}$.

## 3. Precise Flow of a Single Experiment (for fixed $K$ and Model Family)
For every model family (Logistic Regression, LightGBM, Random Forest) and for every $K \in \{1 \dots 20\}$:

1. **Feature Slicing:** Take top $K$ features from Zosia's list.
2. **Hyperparameter Tuning (Optuna):** 
   - Run 50 trials optimizing for minimum **Log-Loss** across a 3-Fold CV.
   - *LogReg bounds:* `C` $\in [0.001, 10]$, `penalty` $\in \{l1, l2\}$.
   - *LightGBM bounds:* `max_depth` $\in [2, 5]$, `learning_rate` $\in [0.01, 0.1]$, `colsample_bytree` $\in [0.5, 1.0]$, `num_leaves` $\in [10, 50]$, `min_data_in_leaf` $\in [5, 20]$.
   - *Random Forest bounds:* `max_depth` $\in [3, 8]$, `min_samples_leaf` $\in [5, 20]`.
3. **Outer CV Execution:**
   - Instantiate the best hyperparameters found by Optuna.
   - Wrap the model in `CalibratedClassifierCV(cv=3, method='sigmoid')` (using **Platt Scaling, not isotonic**).
   - Run the 5-Fold outer loop, generating OOF probabilities for all 5000 training rows.
4. **Probability Calibration Check:**
   - Plot a Reliability Curve (Calibration curve) on the concatenated OOF predictions against true labels. Verify the curve is close to the diagonal.
5. **Threshold Empirical Maximization (Preliminary):**
   - Bind true labels to the 5000 OOF probabilities.
   - Sort descending by probability.
   - Calculate `cumulative_TP`, `cumulative_FP`, and `cumulative_profit` iteratively.
   - `cumulative_profit = (cumulative_TP * 10) - (cumulative_FP * 5) - (K * 200)`.
   - Find the index $i \le 1000$ that yields $\max(\text{cumulative\_profit})$ at this individual model level.
   - Record to `experiment_results.csv`: $(K, \text{Model}, \text{BestParams}, \text{MaxProfit}, \text{OptThreshold})$.

## 4. Per-$K$ Ensemble Selection, Calibration, and Final Comparison

### Critical Design Note
Ensembles must be constructed **separately for each fixed value of $K$**. Mixing models with different $K$ values (e.g., LGBM($K=5$) + LogReg($K=20$)) would require paying for the union of all used variables. Since your submission file will contain all unique variables, you'd pay for 20 variables even though one model only needs 5. **Therefore, we compare ensembles at each fixed $K$, then select the single best $K$.**

### 4.1 For Each $K \in \{1, \dots, 20\}$: Ensemble of Same-$K$ Models
For every fixed $K$, you now have OOF predictions and profits from three model families: LogReg($K$), LGBM($K$), RF($K$).
1. **Model Selection within $K$:** Rank these three models by their individual OOF profit (from experiment results).
2. **Decide on Ensemble:** 
   - If the top model significantly dominates (profit difference > 200 EUR), use it alone.
   - Otherwise, create an ensemble of the top 2-3 models at this $K$ to reduce variance and capture different signal.
3. **Cost Control Benefit:** All models in the ensemble use the same $K$, so you pay exactly $K \times 200$ once, regardless of ensemble size.

### 4.2 Per-$K$ Ensemble Weight Optimization
For a given $K$, if you have $N_K \le 3$ candidate OOF arrays $P^{(K)}_1, \dots, P^{(K)}_{N_K}$ (each 5000,):
1. **Weight Search on Full OOF Data:** Optimize weights directly on the complete OOF matrix (all 5000 rows):
   $$P^{(K)}_{\text{blend}} = \sum_{i=1}^{N_K} w^{(K)}_i \cdot P^{(K)}_i$$
   - Use **Grid Search** with $w^{(K)}_i \in \\{0.0, 0.1, 0.2, \dots, 1.0\\}$ (at most $11^3 = 1331$ combinations for 3 models).
   - Metric: Maximize cumulative profit on the full OOF:
     $$\text{Profit}^{(K)}_\text{blend} = (TP_{\text{blend}} \times 10) - (FP_{\text{blend}} \times 5) - (K \times 200)$$
   - **Rationale:** OOF predictions are already \"out-of-fold\" (each row's prediction comes from a model not trained on that row), so no additional validation split is needed. Grid search with discrete weight steps naturally regularizes against overfitting the weights. This preserves all 5000 rows for weight optimization, which is critical given your limited training data.
2. **Record:** Store the best weights $w^{(K)}_{\text{best}}$ and corresponding maximum profit $\text{Profit}^{(K)}_{\text{max}}$.

### 4.3 Per-$K$ Ensemble Calibration
For the winning ensemble at fixed $K$:
1. Compute the blended OOF predictions: $P^{(K)}_{\text{blend}} = \sum_{i=1}^{N_K} w^{(K)}_{\text{best},i} \cdot P^{(K)}_i$ (across all 5000 training rows).
2. Apply `CalibratedClassifierCV(cv=3, method='sigmoid')` to the blended OOF to learn the final calibration mapping. **Use Platt Scaling (sigmoid method), NOT isotonic**, because:
   - Platt Scaling is mathematically stable at the tail of the probability distribution (where you will select your top 500-1000 customers).
   - Isotonic Regression can overfit, creating flat plateaus and assigning identical probabilities to entire segments, which is problematic for ranking stability in the tail.
3. Store this fitted `sigmoid` calibration transformer for test-time use.
4. **Cap/Lift Validation:** Generate Cap/Lift curves on the calibrated blended OOF predictions ($P^{(K)}_{\text{calibrated}}$). Verify that the ensemble achieves the same or better lift compared to the best individual model at this $K$.

### 4.4 Global Comparison Across All $K$
Once steps 4.1-4.3 are complete for every $K \in \\{1, \dots, 20\\}$, you have 20 candidate solutions, each with:
- Ensemble weights $w^{(K)}_{\text{best}}$
- Calibrated blended OOF predictions $P^{(K)}_{\text{calibrated}}$
- Maximum achievable profit on OOF: $\text{Profit}^{(K)}_{\text{max}}$ (already subtracting $K \times 200$)

**Selection Rule:** Choose the optimal $K^*$ such that:
$$K^* = \arg\max_K \text{Profit}^{(K)}_{\text{max}}$$

This $K^*$ is your final configuration: you will use exactly $K^*$ variables, their corresponding ensemble of models, and the calibrated threshold derived from $P^{(K^*)}_\text{calibrated}$.

## 5. Detailed Risks & Operational Attention Points
1. **The Cost of Overestimating Max Profit Depth:**
   - The threshold loop naturally finds a mathematical peak. However, if the peak resides in an unstable zone (e.g. adding the 400th customer bumps profit by just +5 EUR), the test set might dip negative on that tail. 
   - *Rule:* Apply a **Margin of Safety**. If $P \approx 0.333$ is break-even mathematically, we should enforce a minimum cutoff of $P > 0.38$ to absorb distribution shifts.
2. **Optuna Metric Misalignment:**
   - Tuning hyperparams on AUC or F1 is strictly forbidden. F1 depends on a hardcoded 0.5 threshold, and AUC ignores absolute probability calibration. We must tune on **Log-Loss** (or Brier Score) to ensure $P = 0.40$ actually means a 40% True Positive rate.
3. **Feature Scaling (Crucial for LogReg & SVM):**
   - Must be strictly executed *inside* the CV folds. `StandardScaler` must `fit` on train folds and `transform` on validation/test folds to prevent data leakage. Use SKLearn `Pipeline`.
4. **Zosia's Rankings Stability:**
   - We must verify how stable Zosia's top $K$ features are. If a 5-fold CV on her end yields completely different top 10 features per fold, the baseline is unstable. We require her ranking to be built on a robust ensemble importance (e.g., averaged SHAP values across folds).
5. **Per-$K$ Isolation:**
   - Ensure that when you compute the final $K^*$, you are NOT mixing features from different $K$ values. Only use the exact $K^*$ features and their ensemble models.
