# Test Prediction and Submission Task

This document strictly defines how we evaluate on `x_test.txt` once the rigorous Nested-CV experiments have concluded and the best configuration is identified.

## Step 1: Configuration Lock-in with Ensemble & Calibration
1. Parse the output `experiment_results.csv`.
2. Identify the **top $N$ best ensemble candidates** (e.g., top 5-7 configurations), which may include different $K$ values and model families.
3. Load the **ensemble weights** (from Mod 6) that were optimized to maximize profit.
4. Extract:
   - The list of $(K_i, \text{Model}_i)$ pairs forming the ensemble.
   - Their optimized weights $w_1, \dots, w_N$.
   - The empirically optimized threshold $\tau_{opt}$ (computed from the calibrated blended OOF predictions).
   - The fitted **Calibration Transformer** (Isotonic Regression model learned on blended OOF in Mod 7).

## Step 2: Prepare the Multi-Model 5-Fold Ensembles
For each of the $N$ ensemble members:
- Load the 5 fitted, perfectly calibrated outer-fold models (`CalibratedClassifierCV` wrappers) that were saved during training.
- **Do not retrain** on the full 5000 train rows, as we would lose the exact calibration spaces those 5 models established.

## Step 3: Test-Time Ensemble Inference
For each of the $N$ ensemble members (with their respective $K_i$ features and weights $w_i$):
1. Load `x_test.txt` and filter it to the $K_i$ features specific to that model.
2. Ensure the same `StandardScaler` transformations that were fitted on the respective outer training folds are applied individually.
3. Pass the test data through the 5 calibrated models (from the 5 outer folds).
4. Capture `predict_proba` from each of the 5 models and average them per test customer. This gives one probability vector per ensemble member.
5. **Combine all $N$ ensemble members** using the optimized weights:
   $$P_{\\text{blend,test}} = \\sum_{i=1}^{N} w_i \\cdot P_{i,\\text{test}}$$
6. **Apply the Calibration Transformer** (learned during training in Mod 7):
   $$P_{\\text{final}} = \\text{CalibratedTransformer}(P_{\\text{blend,test}})$$
7. This yields the final, calibrated, high-stability probability vector for the test set.

## Step 4: Final Selection & Threshold-Based Cut-off
1. Sort the 5000 test customers descending by their final calibrated ensemble probability $P_{\text{final}}$ (from Step 3).
2. Select customers one by one from the top down **using the empirically determined threshold $\tau_{opt}$ (learned from the calibrated blended OOF ensemble)**.
3. **Stopping Conditions** (Stop adding customers if EITHER is hit):
   - We reach the hard cap of 1,000 customers, **OR**
   - The probability of the next customer drops below $\tau_{opt}$.
4. **Outcome:** If 380 customers have $P_{\text{final}} \ge \tau_{opt}$, submit exactly 380. If 950 customers qualify before hitting the cap, submit all 950. The threshold is the decision rule, not a fixed target count.

## Step 5: Deliverable Generation
1. Write the final selected original indices to `STUDENT1_STUDENT2_STUDENT3_obs.txt`. (Verify dataset documentation to see if indices are 0-based or 1-based, and ensure format conformity).
2. Write the variable indices to `STUDENT1_STUDENT2_STUDENT3_vars.txt`. Since the ensemble includes $N$ models with potentially different feature sets, union all variables used across the ensemble and output their combined indices.
3. Wrap `.txt` artifacts, the compiled LaTeX report (`report.pdf`), presentation (`presentation.pdf`), and the clean `code/` folder into the final ZIP file structure required by the assignment guidelines.
