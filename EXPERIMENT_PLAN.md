# Precise Experiment Flow & Optimization Strategy

## 1. The Optimization Problem & Strategy
We are optimizing a discontinuous, non-differentiable step-function:
$$ \text{Total Profit} = (TP \times 10) - (FP \times 5) - (K \times 200) $$
With a hard constraint: $\text{Total Selected} \le 1000$.

Because the cost of features ($K$) is global and massive, **we cannot jointly optimize $K$, Hyperparameters, and Threshold concurrently with gradient or Bayesian methods out-of-the-box.** 

**Our Strategic Choice: Grid-Search over $K$, Bayesian Optimization for Hyperparams, Empirical Maximization for Thresholds.**
1. **Outer Grid Loop:** Iterate over $K \in \{1, 2, 3, \dots, 20\}$ variables based on Zosia's ranked list. (The search space is small enough to brute-force $K$).
2. **Inner Bayesian Tuning:** For a given $K$, optimize hyperparameters using Optuna. The surrogate metric will be **Log-Loss** (Cross-Entropy). Why? Because Log-Loss rigorously forces the model to output accurate probability distributions, which is strictly required for expected value calculations. Optimizing discrete profit during hyperparameter tuning creates flat gradients and traps Optuna in local optima.
3. **Threshold Selection:** A completely deterministic step. Once OOF (Out-Of-Fold) probabilities are generated and calibrated, we will sort them and empirically find the threshold $\tau$ that maximizes profit on the validation set.

## 2. Cross-Validation Architecture (Nested CV)
To avoid data leakage during hyperparameter tuning and calibration, we use a **Nested Stratified CV Design**.

### Outer Loop (5-Fold Stratified CV) - For Threshold & Generalization
- **Splits:** 5 Folds. For each iteration, 4 Folds ($T_{\text{outer}}$) and 1 Fold ($V_{\text{outer}}$).
- **Purpose:** To generate an unbiased, perfectly clean array of out-of-fold (OOF) probabilities of size exactly equal to the train set (5000x1). 

### Inner Loop (3-Fold Stratified CV) - For Hyperparameters & Calibration
- **Splits:** Inside $T_{\text{outer}}$, we perform another 3-Fold split. 
- **Purpose 1 (Optuna):** Optuna searches hyperparameters comparing Log-Loss across these 3 inner folds.
- **Purpose 2 (Calibration):** We use Scikit-Learn’s `CalibratedClassifierCV(cv=3, method='isotonic')` over $T_{\text{outer}}$. It trains the base estimators on inner training splits and learns the calibration mapping (Platt or Isotonic) on the inner validation splits. The final meta-estimator then predicts on the pristine $V_{\text{outer}}$.

## 3. Precise Flow of a Single Experiment
For every model family (Logistic Regression, LightGBM, Random Forest, SVM) and for every $K \in \{1 \dots 20\}$:

1. **Feature Slicing:** Take top $K$ features from Zosia’s list.
2. **Hyperparameter Tuning (Optuna):** 
   - Run 50 trials optimizing for minimum **Log-Loss** across a 3-Fold CV.
   - *LogReg bounds:* `C` $\in [0.001, 10]$, `penalty` $\in \{l1, l2\}$.
   - *LightGBM bounds:* `max_depth` $\in [2, 5]$, `learning_rate` $\in [0.01, 0.1]$, `colsample_bytree` $\in [0.5, 1.0]$. (Note: extremely constrained depths to prevent overfitting on 5000 rows).
3. **Outer CV Execution:**
   - Instantiate the best hyperparameters found by Optuna.
   - Wrap the model in `CalibratedClassifierCV`.
   - Run the 5-Fold outer loop, generating OOF probabilities for all 5000 training rows.
4. **Probability Calibration Check:**
   - Plot a Reliability Curve (Calibration curve) on the concatenated OOF predictions against true labels. If the line severely deviates from the diagonal, switch calibration from Isotonic (can overfit) to Sigmoid (Platt).
5. **Threshold Empirical Maximization with Cap/Lift Curves:**
   - Bind true labels to the 5000 OOF probabilities.
   - Sort descending by probability.
   - Calculate `cumulative_TP`, `cumulative_FP`, and `cumulative_profit` iteratively:
     - `cumulative_profit = (cumulative_TP * 10) - (cumulative_FP * 5) - (K * 200)`.
   - Generate **Cap/Lift Curves** on the OOF:
     - Cap Curve: Plot $$\frac{\text{cumulative\_TP}}{\text{total\_positive}}$$ vs. fraction of population targeted (0 to 1).
     - Lift Curve: Plot $$\frac{\text{cumulative\_TP} / \text{cumulative\_total}}{\text{base\_rate}}$$ vs. fraction of population.
     - Identify **kink points** (inflection points) where the curve flattens significantly, indicating where ROI degrades.
   - Find the index $i \le 1000$ that yields $\max(\text{cumulative\_profit})$.
   - **Validation Rule:** The profit-maximizing threshold $\tau_K$ must align with or precede the first visible kink in the Cap/Lift curves. If the peak is beyond the kink, apply a Safety Margin: $\tau_K := \max(\tau_K, 0.38)$ to absorb distribution shift risk.
   - Record $(K, \text{Model}, \text{Profit}_K, \tau_K, \text{Cap\_Lift\_plots})$ to `experiment_results.csv`.

## 4. Final Ensemble Selection & Calibration

### 4.1 Model Ranking & Selection for Ensemble
Once all experiments across $K \in \{1 \dots 20\}$ and model families are complete:
1. **Global Ranking:** Sort all $(K, \text{Model})$ pairs by their profit scores in descending order.
2. **Ensemble Candidate Pool:** Select the top $N$ performers (e.g., top 5-7 configurations), regardless of their $K$ value or model family. This ensures diversity: you might get LGBM with $K=5$, LogReg with $K=7$, RF with $K=5$, etc.
3. **Rationale:** Different models capture different signal. Combining them reduces overfitting and variance.

### 4.2 Ensemble Weight Optimization
Given $N$ candidate OOF probability arrays $P_1, \dots, P_N$ (each of size 5000):
1. **Nested Split:** Further split the training data into a meta-training set (3000 rows) and meta-validation set (2000 rows).
2. **Weight Search:** Using the meta-training set, find weights $w_1, \dots, w_N$ (with $\sum w_i = 1$) that maximize profit:
   $$P_{\text{blend}} = \sum_{i=1}^{N} w_i \cdot P_i$$
   - Use a simple grid search: $w_i \in \{0.0, 0.1, 0.2, \dots, 1.0\}$ or Optuna-based optimization.
   - Metric: maximize cumulative profit on meta-training OOF.
3. **Validation:** Verify on the meta-validation set that the blended predictions yield higher profit than any individual model.

### 4.3 Ensemble Calibration
After finding optimal weights:
1. Concatenate the blended OOF predictions ($P_{\text{blend}}$) across all 5000 training rows.
2. Apply `CalibratedClassifierCV(cv=3, method='isotonic')` using the concatenated $P_{\text{blend}}$ and true labels to learn the final calibration mapping.
3. Store this calibration transformer for test-time use.
4. **Cap/Lift Validation:** Generate Cap/Lift curves on the calibrated blended OOF predictions. Verify that the ensemble achieves the same or better lift than the top individual model. 

## 5. Detailed Risks & Operational Attention Points
1. **The Cost of Overestimating Max Profit Depth:**
   - The threshold loop naturally finds a mathematical peak. However, if the peak resides in an unstable zone (e.g. adding the 400th customer bumps profit by just +5 EUR but degrades total probability confidence), the test set might dip negative on that tail. 
   - *Rule:* Apply a **Margin of Safety**. If $P \approx 0.333$ is break-even mathematically, we should enforce a minimum cutoff of $P > 0.38$ to absorb distribution shifts.
2. **Optuna Metric Misalignment:**
   - Tuning hyperparams on AUC or F1 is strictly forbidden. F1 depends on a hardcoded 0.5 threshold, and AUC ignores absolute probability calibration. We must tune on **Log-Loss** (or Brier Score) to ensure $P = 0.40$ actually means a 40% True Positive rate.
3. **Feature Scaling (Crucial for LogReg & SVM):**
   - Must be strictly executed *inside* the CV folds. `StandardScaler` must `fit` on train folds and `transform` on validation/test folds to prevent data leakage. Use SKLearn `Pipeline`.
4. **Zosia's Rankings Stability:**
   - We must verify how stable Zosia's top $K$ features are. If a 5-fold CV on her end yields completely different top 10 features per fold, the baseline is unstable. We require her ranking to be built on a robust ensemble importance (e.g., averaged SHAP values across folds).
