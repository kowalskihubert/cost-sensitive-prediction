# Test Prediction and Submission Task

This document strictly defines how we evaluate on `x_test.txt` once the rigorous Nested-CV experiments have concluded and the best configuration is identified.

## Step 1: Configuration Lock-in with Best K* Selection
1. Parse the output from Mod 7 (per-K ensemble calibration).
2. **Select the single best K* = argmax Profit^(K)_max** across all K in {1, ..., 20}.
3. Extract:
   - K* (the optimal number of variables).
   - The exact list of target variables (first K* features from Zosia's JSON).
   - The ensemble members (which models form the ensemble at K*).
   - The optimized ensemble weights w^(K*)_best.
   - The empirically optimized threshold tau_opt (from calibrated blended OOF at K*).
   - The fitted Calibration Transformer (Sigmoid/Platt Scaling model learned at K*).
4. **Verify:** Only features from the top K* of Zosia's ranking are used. No cross-K feature mixing.

## Step 2: Prepare the K*-Specific 5-Fold Ensembles
For each of the N_K* ensemble members at K*:
- Load the 5 fitted, perfectly calibrated outer-fold models (CalibratedClassifierCV wrappers) from the training phase.
- **Do not retrain** on the full 5000 train rows, as we would lose the exact calibration spaces those 5 models established.

## Step 3: Test-Time Ensemble Inference
For each of the N_K* ensemble members:
1. Load `x_test.txt` and filter it to the K* features.
2. Ensure the same StandardScaler transformations that were fitted on the respective outer training folds are applied individually.
3. Pass the test data through the 5 calibrated models (from the 5 outer folds).
4. Capture predict_proba from each of the 5 models and average them per test customer. This gives one probability vector per ensemble member.
5. **Combine all N_K* ensemble members** using the optimized weights:
   P_blend,test = sum_i w^(K*)_i * P_i,test
6. **Apply the Calibration Transformer** (learned during training):
   P_final = CalibratedTransformer(P_blend,test)
7. This yields the final, calibrated, high-stability probability vector for the test set.

## Step 4: Final Selection & Threshold-Based Cut-off
1. Sort the 5000 test customers descending by their final calibrated ensemble probability P_final.
2. Select customers one by one from the top down **using the empirically determined threshold tau_opt** (learned from the calibrated blended OOF ensemble at K*).
3. **Stopping Conditions** (Stop adding customers if EITHER is hit):
   - We reach the hard cap of 1,000 customers, **OR**
   - The probability of the next customer drops below tau_opt.
4. **Outcome:** If 380 customers have P_final >= tau_opt, submit exactly 380. If 950 customers qualify before hitting the cap, submit all 950. The threshold is the decision rule, not a fixed target count.

## Step 5: Deliverable Generation
1. Write the final selected original indices to `STUDENT1_STUDENT2_STUDENT3_obs.txt`. (Verify dataset documentation to see if indices are 0-based or 1-based, and ensure format conformity).
2. Write the K* variable indices to `STUDENT1_STUDENT2_STUDENT3_vars.txt`. Since all ensemble members at K* use the same K* features, this is simply the first K* feature indices from Zosia's JSON.
3. Wrap `.txt` artifacts, the compiled LaTeX report (`report.pdf`), presentation (`presentation.pdf`), and the clean `code/` folder into the final ZIP file structure required by the assignment guidelines.
