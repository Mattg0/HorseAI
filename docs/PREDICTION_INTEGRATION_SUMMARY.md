# Prediction Code Updated for Feature Selection

## Overview

Prediction code has been updated to automatically load and apply TabNet feature selection. When you make predictions with a model trained with feature selection, the prediction code will:

1. Load `feature_selector.json` when loading the model
2. Apply the same feature selection used during training
3. Make predictions on the selected features

**No manual intervention required** - feature selection happens automatically during prediction!

---

## Files Updated

### 1. predict_quinte.py (Quinte Race Predictions)

**Location**: [race_prediction/predict_quinte.py](race_prediction/predict_quinte.py)

**Changes**:
- Lines 69-70: Added `self.feature_selector` and `self.tabnet_model_path` attributes
- Lines 118-128: Load `feature_selector.json` when loading TabNet model
- Lines 401-410: Apply feature selection before scaling in `predict()` method

**How it works**:
```python
# When loading model:
feature_selector_path = self.tabnet_model_path / "feature_selector.json"
if feature_selector_path.exists():
    self.feature_selector = TabNetFeatureSelector.load(str(feature_selector_path))
    print(f"✓ Loaded feature selector: {len(self.feature_selector.selected_features)} selected features")

# During prediction:
if self.feature_selector is not None:
    print(f"Applying TabNet feature selection...")
    original_feature_count = len(X.columns)
    X = self.feature_selector.transform(X)
    print(f"✓ Feature selection applied: {original_feature_count} → {len(X.columns)} features")
```

**Output example**:
```
[QuintePrediction] Loading quinté models...
[QuintePrediction] ✓ Loaded TabNet quinté model from models/2025-10-30/2years_123456_quinte_tabnet/tabnet_model.zip
[QuintePrediction]   Features: 92
[QuintePrediction] ✓ Loaded feature selector: 45 selected features
[QuintePrediction] Generating predictions...
[QuintePrediction] ✓ Feature matrix after transformations: 92 features
[QuintePrediction] Applying TabNet feature selection...
[QuintePrediction] ✓ Feature selection applied: 92 → 45 features
[QuintePrediction] ✓ Final feature matrix: 45 features (model expects 92)
```

---

### 2. race_predict.py (General Race Predictions)

**Location**: [race_prediction/race_predict.py](race_prediction/race_predict.py)

**Changes**:
- Line 533: Added `self.tabnet_feature_selector = None` attribute
- Lines 575-588: Load `feature_selector.json` when loading TabNet model
- Lines 808-815: Apply feature selection before scaling in `predict_with_tabnet()` method

**How it works**:
```python
# When loading model:
feature_selector_file = tabnet_path / "feature_selector.json"
if feature_selector_file.exists():
    self.tabnet_feature_selector = TabNetFeatureSelector.load(str(feature_selector_file))
    print(f"Loaded TabNet feature selector: {len(self.tabnet_feature_selector.selected_features)} selected features")

# During prediction:
if self.tabnet_feature_selector is not None:
    print(f"Applying TabNet feature selection: {len(X_df.columns)} → {len(self.tabnet_feature_selector.selected_features)} features")
    X_df = self.tabnet_feature_selector.transform(X_df)
    print(f"TabNet features after selection: {X_df.shape}")
```

**Output example**:
```
Loaded TabNet model from: models/2025-10-30/2years_123456/tabnet_model/tabnet_model.zip
TabNet expects 115 features
Loaded TabNet feature selector: 46 selected features
TabNet features aligned: (14, 115)
Applying TabNet feature selection: 115 → 46 features
TabNet features after selection: (14, 46)
TabNet features scaled using training scaler
```

---

## Backward Compatibility

The prediction code is **fully backward compatible**:

✅ **New models** (with feature selection):
- `feature_selector.json` exists → Feature selection applied automatically
- Output: `"Loaded feature selector: X selected features"`

✅ **Old models** (without feature selection):
- `feature_selector.json` doesn't exist → Uses all features (existing behavior)
- Output: `"No feature selector found (using all X features)"`

No errors or issues when using old models!

---

## Testing

### Test with New Model (Feature Selection)

```bash
# Train a new model with feature selection
python model_training/historical/train_quinte_model.py

# Make predictions - should show feature selection messages
python race_prediction/predict_quinte.py --date 2025-10-30 --verbose
```

**Expected output**:
```
✓ Loaded feature selector: 45 selected features
Applying TabNet feature selection...
✓ Feature selection applied: 92 → 45 features
```

### Test with Old Model (No Feature Selection)

```bash
# Use an old model without feature_selector.json
# Edit config.yaml to point to old model

# Make predictions - should work normally
python race_prediction/predict_quinte.py --date 2025-10-30 --verbose
```

**Expected output**:
```
No feature selector found (using all 92 features)
✓ Final feature matrix: 92 features (model expects 92)
```

---

## How Feature Selection Works During Prediction

### Flow Diagram

```
1. Load Race Data
   ↓
2. Calculate ALL Features (92 or 115)
   ├─ Standard features
   ├─ Quinté features (if applicable)
   ├─ Global/weighted/bytype variants
   └─ Musique calculations
   ↓
3. Apply Transformations (log transforms, etc.)
   ↓
4. Load Feature Selector (if exists)
   ↓
5. Apply Feature Selection ← NEW!
   ├─ Keep only selected features (e.g., 45 out of 92)
   └─ Remove correlated/sparse features
   ↓
6. Scale Features (StandardScaler)
   ↓
7. Make TabNet Predictions
```

### Key Points

1. **ALL features calculated**: Prediction calculates the same complete feature set as before
2. **Selection happens before scaling**: Features are filtered before StandardScaler
3. **Same features as training**: Uses the exact same selected features from training
4. **No manual steps**: Everything automatic - just run prediction as before

---

## Verification

### Check Feature Selector Was Loaded

Look for this message in prediction output:
```
✓ Loaded feature selector: 45 selected features
```

### Check Feature Selection Was Applied

Look for this message during prediction:
```
Applying TabNet feature selection...
✓ Feature selection applied: 92 → 45 features
```

### Verify Selected Features Match Training

```python
import json

# Load feature selector from model
with open('models/.../tabnet_model/feature_selector.json', 'r') as f:
    selector = json.load(f)

print(f"Selected features: {len(selector['selected_features'])}")
print(f"Features: {selector['selected_features'][:10]}")  # First 10
```

---

## Summary

✅ **predict_quinte.py**: Updated to load and apply feature selection for Quinte predictions
✅ **race_predict.py**: Updated to load and apply feature selection for General predictions
✅ **Backward compatible**: Works with both new and old models
✅ **Automatic**: No code changes needed when making predictions
✅ **Consistent**: Uses exact same features as training

## What This Means

When you:
1. ✅ Train a new model → Feature selection happens automatically
2. ✅ Save the model → `feature_selector.json` saved automatically
3. ✅ Make predictions → Feature selection applied automatically

**Just retrain your models and start predicting - everything else is automatic!**
