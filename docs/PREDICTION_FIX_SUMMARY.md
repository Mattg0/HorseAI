# Prediction Feature Selection Fix

## Problem

When trying to make predictions with a model trained with feature selection, the following error occurred:

```
ValueError: The feature names should match those that were passed during fit.
Feature names seen at fit time, yet now missing:
- che_bytype_consistency
- che_bytype_nb_courses
- che_global_consistency
- ...
```

## Root Cause

The prediction code was applying feature selection in the wrong order:

1. **Broken Flow** (BEFORE fix):
   ```
   Calculate all features (92/115)
   ↓
   Align to model's expected features (45-46 SELECTED features)
   ↓
   Try to apply feature selector → ERROR!
   (Selector expects 92/115 ORIGINAL features but only has 45-46)
   ```

2. **Issue**: The feature selector's `transform()` expects the FULL original feature set (92/115), but we were giving it only the selected features (45-46) after alignment.

## Solution

### 1. Store Original Features in Feature Selector

Updated `TabNetFeatureSelector` to save and load the original feature list:

**File**: [core/feature_selection/tabnet_feature_selector.py](core/feature_selection/tabnet_feature_selector.py)

**Changes**:
- Line 48: Added `self.original_features = None` in `__init__()`
- Line 172: Store `self.original_features = list(X.columns)` in `fit()`
- Line 175: Save `'original_features': list(X.columns)` in metadata
- Line 269: Load `self.original_features` from metadata in `load()`

**Result**: Now feature selector knows both:
- Original features (92/115) - what it expects as input
- Selected features (45-46) - what it outputs after selection

---

### 2. Fix Prediction Flow

Updated prediction code to apply feature selection BEFORE aligning:

**Correct Flow** (AFTER fix):
```
Calculate all features (92/115)
↓
Align to ORIGINAL features (92/115) from feature selector
↓
Apply feature selector: 92/115 → 45-46
↓
Scale with StandardScaler (expects 45-46)
↓
Predict
```

#### predict_quinte.py

**File**: [race_prediction/predict_quinte.py](race_prediction/predict_quinte.py)

**Changes** (Lines 366-439):

```python
if self.feature_selector is not None:
    # Get ORIGINAL features from selector (before selection)
    original_training_features = self.feature_selector.original_features

    # Align to original features
    aligned_X = pd.DataFrame(0.0, index=range(len(X_tabnet)), columns=original_training_features)
    for feature in available_features:
        aligned_X[feature] = result_df[feature]

    # Apply feature selection
    X = self.feature_selector.transform(aligned_X)  # 92 → 45
```

#### race_predict.py

**File**: [race_prediction/race_predict.py](race_prediction/race_predict.py)

**Changes** (Lines 754-816):

```python
if self.tabnet_feature_selector is not None:
    # Get original features from selector (before selection)
    expected_features = self.tabnet_feature_selector.original_features

    # Create aligned DataFrame with original features
    aligned_X = pd.DataFrame(0.0, index=range(len(X_tabnet)), columns=expected_features)
    for feature in available_features:
        aligned_X[feature] = X_tabnet[feature]

    # Apply feature selection
    X_df = self.tabnet_feature_selector.transform(aligned_X)  # 115 → 46
```

---

## Key Changes Summary

| Component | Before | After |
|-----------|--------|-------|
| **TabNetFeatureSelector** | Only stores selected features | Stores both original AND selected features |
| **Prediction alignment** | Aligns to selected features (45-46) | Aligns to original features (92/115) |
| **Feature selection** | Applied after alignment → ERROR | Applied after alignment to original → SUCCESS |
| **Scaler input** | Wrong shape | Correct shape (selected features) |

---

## How It Works Now

### During Training

1. Calculate all features (92 or 115)
2. Feature selector fits on all features
   - Stores `original_features`: [feat1, feat2, ..., feat92]
   - Stores `selected_features`: [feat1, feat5, ..., feat45] (45 selected)
3. Transform: 92 → 45
4. Train TabNet on 45 features
5. Save:
   - `feature_selector.json`: Contains both original_features and selected_features
   - `tabnet_scaler.joblib`: Expects 45 features
   - `tabnet_model.zip`: Expects 45 features

### During Prediction

1. Calculate all features (92 or 115)
2. Load `feature_selector.json` → Get original_features and selected_features
3. Align to original_features (92): Fill with 0 if missing
4. Apply selector.transform(): 92 → 45
5. Scale: 45 features (matches scaler)
6. Predict: 45 features (matches model)

---

## Files Modified

### Core Feature Selection

✅ **[core/feature_selection/tabnet_feature_selector.py](core/feature_selection/tabnet_feature_selector.py)**
- Lines 48, 172, 175, 269: Store and load original_features
- Line 253: Made `load()` a classmethod

### Prediction Scripts

✅ **[race_prediction/predict_quinte.py](race_prediction/predict_quinte.py)**
- Lines 366-439: Rewritten to align to original features first, then apply selection

✅ **[race_prediction/race_predict.py](race_prediction/race_predict.py)**
- Lines 754-816: Rewritten to align to original features first, then apply selection

---

## Testing

To verify the fix works:

```bash
# 1. Train a new model (will save original_features in feature_selector.json)
python model_training/historical/train_quinte_model.py

# 2. Make predictions (should now work without errors)
python race_prediction/predict_quinte.py --date 2025-10-30 --verbose
```

**Expected output**:
```
[QuintePrediction] Using TabNet feature selection pipeline...
[QuintePrediction] ✓ Feature matrix before selection: 92 features
[QuintePrediction] ✓ Feature selection applied: 92 → 45 features
[QuintePrediction] ✓ Final feature matrix: 45 features (model expects 92)
```

---

## Backward Compatibility

**Old models** (trained before this fix):
- `feature_selector.json` doesn't have `original_features` field
- Falls back to using `selected_features` as original
- May still work if features are calculated correctly, but not guaranteed

**Solution**: Retrain models to get the updated `feature_selector.json` with `original_features`.

---

## Summary

✅ **Problem**: Feature selector applied to wrong feature set (selected instead of original)
✅ **Fix**: Store original features in selector, align to original before selection
✅ **Result**: Prediction now works correctly with feature selection
✅ **Action**: Retrain models to get updated feature_selector.json

**Status**: 🎉 **FIXED** - Predictions now work with automatic feature selection!
