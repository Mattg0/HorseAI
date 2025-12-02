# Isotonic Calibration Fix

## Problem

When making predictions with both TabNet and RF models, the RF model (wrapped in `CalibratedRegressor`) was failing with a feature mismatch error:

```
ValueError: The feature names should match those that were passed during fit.
```

**Error location**: `isotonic_calibration.py`, line 355 in `predict()`
```python
raw_predictions = self.base_regressor.predict(X)  # <- Error here
```

## Root Cause

In `predict_quinte.py`, the prediction code was modifying the feature matrix `X` for TabNet predictions, and then passing the **same modified X** to the RF model.

### Broken Flow (BEFORE fix):

```python
# predict_quinte.py, lines 502-517

X = <original features after feature selection for TabNet>  # 45-46 features

# TabNet predictions
if self.tabnet_model and self.scaler:
    if hasattr(self.scaler, 'feature_names_in_'):
        X = X[self.scaler.feature_names_in_]  # ← MODIFIES X to match scaler
    X_scaled = self.scaler.transform(X)
    tabnet_preds = self.tabnet_model.predict(X_scaled)

# RF predictions
if self.rf_model:
    rf_preds = self.rf_model.predict(X)  # ← Uses MODIFIED X! (45-46 features)
    # But RF model expects 92 features!
    # ERROR: Feature mismatch
```

**Issue**:
1. X is modified for TabNet (filtered to 45-46 selected features)
2. RF model receives the modified X
3. RF model's `CalibratedRegressor` expects 92 original features
4. Feature mismatch causes error in isotonic calibration

## Solution

Create **separate copies** of X for TabNet and RF predictions:

### Fixed Flow (AFTER fix):

```python
# predict_quinte.py, lines 372-527

if self.feature_selector is not None:
    # Get ORIGINAL features (92)
    original_training_features = self.feature_selector.original_features

    # Prepare X_full with all original features
    X_full = result_df[available_features].copy()  # 92 features
    X_full = cleaner.apply_transformations(X_full)
    X_full = X_full.fillna(0)

    # SAVE FOR RF BEFORE SELECTION
    X_rf = X_full.copy()  # ← RF gets 92 original features

    # Apply feature selection FOR TABNET ONLY
    X_tabnet = self.feature_selector.transform(X_full)  # ← TabNet gets 45 selected features

else:
    # No feature selection
    X = result_df[available_features].copy()
    X = cleaner.apply_transformations(X)
    X = X.fillna(0)

    # Both use same features
    X_rf = X.copy()
    X_tabnet = X.copy()

# TabNet predictions
if self.tabnet_model:
    X_scaled = self.scaler.transform(X_tabnet)  # Uses 45 selected features
    tabnet_preds = self.tabnet_model.predict(X_scaled)

# RF predictions
if self.rf_model:
    rf_preds = self.rf_model.predict(X_rf)  # Uses 92 original features
    # No error!
```

**Fix**:
1. Create `X_full` with all original features (92)
2. Save `X_rf = X_full.copy()` **BEFORE** applying feature selection
3. Apply feature selection to create `X_tabnet` with selected features (45)
4. RF gets original features (92), TabNet gets selected features (45)

## Why This Matters

### TabNet with Feature Selection:
- Expects: 45-46 **selected** features (after feature selection)
- Gets: X_tabnet with selected features
- ✅ Works correctly

### RF with Isotonic Calibration:
- Expects: 92 **original** features (no feature selection)
- Gets: X_rf with original features
- ✅ Works correctly

### Before Fix:
- Both models were sharing the same X variable
- X was modified for TabNet → RF received wrong features
- ❌ Error in CalibratedRegressor.predict()

## Files Modified

✅ **[race_prediction/predict_quinte.py](race_prediction/predict_quinte.py)** (Lines 372-527)
- Lines 395-404: Create `X_full` with all original features (92)
- Line 404: Save `X_rf = X_full.copy()` **BEFORE** feature selection
- Line 408: Apply feature selection to create `X_tabnet` (92 → 45)
- Lines 441-442: When no feature selection, create both copies
- Line 526: RF uses `X_rf` (92 features)
- Line 516-519: TabNet uses `X_tabnet` (45 features)

✅ **[race_prediction/race_predict.py](race_prediction/race_predict.py)** (No changes needed)
- Already uses separate feature sets for RF and TabNet
- `predict_with_rf(X_rf)` uses X_rf from `extract_rf_features()`
- `predict_with_tabnet(race_df)` uses race_df directly
- No shared variable issue

## Key Insight

The **order of operations matters**:

**WRONG**:
```python
X = prepare_features()
X = apply_feature_selection(X)  # Modifies X
X = reorder_for_scaler(X)        # Modifies X again
rf_pred = rf_model.predict(X)    # RF gets modified X ❌
tabnet_pred = tabnet_model.predict(X)  # TabNet gets modified X ✅
```

**CORRECT**:
```python
X = prepare_features()
X = apply_feature_selection(X)

X_rf = X.copy()                  # Copy for RF
X_tabnet = X.copy()              # Copy for TabNet

X_tabnet = reorder_for_scaler(X_tabnet)  # Only modify TabNet copy

rf_pred = rf_model.predict(X_rf)         # RF gets original ✅
tabnet_pred = tabnet_model.predict(X_tabnet)  # TabNet gets modified ✅
```

## Feature Selection Context

This issue only appeared after implementing automatic feature selection because:

1. **Before feature selection**: Both models used the same 92 features
   - X could be shared between models without issues

2. **After feature selection**: TabNet uses 45-46 selected features
   - X is modified for TabNet → Can't be shared
   - RF still expects 92 original features → Needs separate copy

## Testing

To verify the fix works:

```bash
# Retrain models with feature selection
python model_training/historical/train_quinte_model.py

# Make predictions (should work now without isotonic calibration errors)
python race_prediction/predict_quinte.py --date 2025-10-30 --verbose
```

**Expected output**:
```
[QuintePrediction] Using TabNet feature selection pipeline...
[QuintePrediction] ✓ Feature selection applied: 92 → 45 features
[QuintePrediction]   Running TabNet model...
[QuintePrediction]   Running Random Forest model...
✅ Predictions generated successfully
```

**No error** from `CalibratedRegressor.predict()`!

## Critical Timing Issue

**The key insight**: X_rf must be saved **BEFORE** feature selection is applied!

### WRONG (First Attempt):
```python
X = prepare_features()              # 92 features
X = apply_feature_selection(X)      # 92 → 45 features
X_rf = X.copy()                     # ← RF gets 45 features (WRONG!)
```

### CORRECT (Final Fix):
```python
X_full = prepare_features()         # 92 features
X_rf = X_full.copy()                # ← RF gets 92 features (CORRECT!)
X_tabnet = apply_selection(X_full)  # ← TabNet gets 45 features
```

## Summary

✅ **Problem**: RF model received modified features meant for TabNet
✅ **Root Cause**: X_rf was saved AFTER feature selection was applied
✅ **Fix**: Save X_rf BEFORE feature selection, then apply selection for X_tabnet
✅ **Result**: Each model gets the features it expects
✅ **Status**: Isotonic calibration now works correctly with feature selection

This fix ensures that:
- **TabNet** gets selected features (45-46)
- **RF** gets original features (92)
- **CalibratedRegressor** (isotonic calibration) works without errors
- **Both models** can coexist in the same prediction pipeline

**Key Learning**: When models need different feature sets, always save copies BEFORE applying transformations that reduce the feature set!
