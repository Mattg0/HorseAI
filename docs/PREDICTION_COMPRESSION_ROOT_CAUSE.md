# Prediction Compression Root Cause Analysis

**Date**: 2025-11-09
**Issue**: RF predictions severely compressed (89% compression)
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

RF predictions are compressed to a span of **1.1-2.2 positions** instead of the expected **~10-15 positions** for typical race fields. This 89% compression is caused by **isotonic calibration baked into the trained models** during the training phase.

---

## The Problem

### Diagnostic Results

From analyzing 20 recent races:

```
📊 PREDICTED_POSITION (Final blended predictions)
   Average span: 2.24 (Expected: ~10.4)
   ❌ SEVERE COMPRESSION! 78% compressed

📊 RF PREDICTIONS
   Average span: 1.13 (Expected: ~10.4)
   ❌ RF COMPRESSION! 89% compressed

📊 TABNET PREDICTIONS
   Average span: 21.58 (Expected: ~10.4)
   ⚠️  TABNET SCALE ISSUE! Values too high (avg max: 43.0)
```

### Worst Cases

| Race    | Field Size | Predicted Span | Compression |
|---------|------------|----------------|-------------|
| 1621441 | 7          | 0.18           | 97%         |
| 1621412 | 4          | 0.50           | 83%         |
| 1621445 | 11         | 0.76           | 92%         |

**Example**: In a 16-horse race, all predictions clustered between 3.09 and 4.88 (span: 1.79) instead of spreading from 1 to 16.

---

## Root Cause

### 1. Isotonic Calibration in Training

**File**: `model_training/historical/train_quinte_model.py:598`

```python
self.rf_model = CalibratedRegressor(
    base_regressor=base_rf,
    clip_min=1.0
)

self.rf_model.fit(X_train, y_train)
```

**File**: `model_training/historical/train_race_model.py:262`

```python
self.rf_model = CalibratedRegressor(
    base_regressor=base_rf,
    clip_min=1.0  # Race positions start at 1
)
```

### 2. What CalibratedRegressor Does

**File**: `model_training/regressions/isotonic_calibration.py:280-358`

The `CalibratedRegressor` class:
1. Trains the base Random Forest model
2. Gets predictions on training/calibration data
3. Fits an `IsotonicRegression` to map raw predictions → actual positions
4. **Automatically applies isotonic calibration** when `.predict()` is called

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    # Get raw predictions from base model
    raw_predictions = self.base_regressor.predict(X)

    # Apply calibration ← THIS COMPRESSES THE OUTPUT
    calibrated_predictions = self.calibrator.predict(raw_predictions)

    return calibrated_predictions
```

### 3. Why Isotonic Calibration Compresses

Isotonic regression fits a **monotonic (non-decreasing) function** to map predictions to targets. If:

- Training data has poor discrimination (features don't separate horses well)
- Training data is noisy or has limited samples
- Model outputs cluster around the mean

Then isotonic regression will:
- Over-fit to the training calibration set
- Compress the output range to minimize MAE/RMSE on training data
- Sacrifice spread for accuracy on average cases

**Result**: All predictions map to a narrow range (4-6) instead of full field (1-16).

### 4. Prediction Pipeline

```
1. Raw RF Model → predictions with some spread
2. IsotonicRegression (baked into CalibratedRegressor) → COMPRESSION HERE ❌
3. AdaptiveCalibratorManager → (empty, not applied)
4. Blending with TabNet → slight mixing
5. Competitive Analysis → minor adjustments
6. Bias Calibration → tries to fix but data already compressed
```

The compression happens **in step 2**, before any other calibration layers.

---

## Why Bias Calibration Failed

The bias calibration system correctly detected biases in the compressed predictions:

```
✓ Systematic bias: -2.49 positions (t=-28.2, p=0.000) [SIGNIFICANT]
✓ Odds-based bias detected: 6 significant categories
✓ Post position bias detected: 4 significant categories
```

However, when applied to validation data, calibration **made performance worse**:

```
❌ CALIBRATION REJECTED
MAE before: 3.856
MAE after: 4.559 (↓ -18%)
```

**Why?** The predictions are so compressed (all between 3-5) that:
1. There's no room for meaningful correction
2. Any adjustment just shifts the compressed cluster
3. The underlying spread is lost, bias correction can't recover it

---

## Solutions

### Option A: Disable Isotonic Calibration (Recommended)

**Pros**:
- Immediate fix
- Restores full prediction spread
- Can apply bias calibration on uncalibrated predictions

**Cons**:
- May lose some accuracy (isotonic calibration often improves MAE)
- Need to retrain models

**Implementation**:
```python
# In train_quinte_model.py and train_race_model.py
# BEFORE:
self.rf_model = CalibratedRegressor(
    base_regressor=base_rf,
    clip_min=1.0
)

# AFTER:
self.rf_model = base_rf  # Use raw RF without isotonic calibration
```

### Option B: Fix Isotonic Calibration Training

**Pros**:
- Keep isotonic calibration benefits
- More sophisticated solution

**Cons**:
- Requires debugging isotonic calibration training
- May need larger calibration dataset
- Need to retrain models

**Investigation needed**:
1. Check calibration data quality and size
2. Review isotonic regression parameters (y_min, y_max, out_of_bounds)
3. Ensure calibration set has good spread
4. Consider using cross-validation for calibration

### Option C: Use Raw Predictions from base_regressor

**Pros**:
- No retraining needed
- Quick test

**Cons**:
- Requires code changes in prediction pipeline
- Loses any benefits of isotonic calibration

**Implementation**:
```python
# In predict_quinte.py:560-570
# BEFORE:
if hasattr(self.rf_model, 'predict'):
    raw_rf_preds = self.rf_model.predict(X_rf)  # This applies isotonic

# AFTER:
if hasattr(self.rf_model, 'predict_raw'):
    raw_rf_preds = self.rf_model.predict_raw(X_rf)  # Bypass isotonic
elif hasattr(self.rf_model, 'base_regressor'):
    raw_rf_preds = self.rf_model.base_regressor.predict(X_rf)  # Direct to RF
else:
    raw_rf_preds = self.rf_model.predict(X_rf)
```

---

## Recommended Action Plan

### Phase 1: Immediate Testing (No Retraining)

1. **Modify prediction code** to use `base_regressor.predict()` instead of `CalibratedRegressor.predict()`
2. **Run predictions** on recent races to verify spread improves
3. **Measure MAE** on test set to quantify accuracy impact

### Phase 2: Permanent Fix (Requires Retraining)

1. **Disable CalibratedRegressor** in training scripts
2. **Retrain RF models** without isotonic calibration
3. **Apply bias calibration** to raw predictions (now uncalibrated)
4. **Compare performance**:
   - Raw RF vs Isotonic-calibrated RF
   - Raw RF + Bias calibration vs Current system

### Phase 3: Enhanced Calibration (If Needed)

If disabling isotonic hurts accuracy:

1. **Debug isotonic calibration training**:
   - Increase calibration set size
   - Use separate validation set (not training set)
   - Adjust y_min, y_max bounds
2. **Try alternative calibration methods**:
   - Platt scaling
   - Beta calibration
   - Quantile regression
3. **Hybrid approach**:
   - Light isotonic calibration (less aggressive)
   - Bias calibration on top

---

## Files Affected

### Training Scripts
- `model_training/historical/train_quinte_model.py:598` - Quinté RF training
- `model_training/historical/train_race_model.py:262` - General RF training

### Prediction Scripts
- `race_prediction/predict_quinte.py:560-570` - Quinté predictions
- `race_prediction/race_predict.py:437` - General predictions

### Calibration Code
- `model_training/regressions/isotonic_calibration.py:280-358` - CalibratedRegressor class
- `model_training/regressions/adaptive_calibrator.py` - Adaptive isotonic calibrator (not used currently)

### Diagnostic Scripts
- `scripts/diagnose_prediction_compression.py` - Compression analysis
- `scripts/debug_rf_calibration.py` - Raw vs calibrated comparison

---

## Next Steps

1. ✅ Root cause identified (isotonic calibration in training)
2. ⏳ Test Option C (use base_regressor) to validate hypothesis
3. ⏳ Retrain models without CalibratedRegressor (Option A)
4. ⏳ Benchmark performance before/after
5. ⏳ Re-enable bias calibration on uncalibrated predictions

---

## References

- **Isotonic Regression**: Monotonic regression that maps predictions to calibrated values
- **CalibratedRegressor**: Wrapper class in `isotonic_calibration.py:280`
- **Diagnostic Report**: `scripts/diagnose_prediction_compression.py` output shows 89% compression
- **Training Data**: Models in `models/2025-11-03/` directory

---

**Conclusion**: The prediction compression is NOT a bug in the bias calibration system or the prediction pipeline. It's an **expected consequence** of isotonic calibration applied during model training. The calibration was likely trained on data with poor spread, causing it to compress all predictions to a narrow range. The fix is to either disable isotonic calibration or retrain it with better data.
