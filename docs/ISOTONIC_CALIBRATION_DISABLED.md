# Isotonic Calibration Disabled - Summary

**Date**: 2025-11-09
**Changes**: Disabled CalibratedRegressor wrapper in RF training

---

## Changes Made

### 1. Quinté Model Training
**File**: `model_training/historical/train_quinte_model.py:598-602`

**Before**:
```python
self.rf_model = CalibratedRegressor(
    base_regressor=base_rf,
    clip_min=1.0
)
```

**After**:
```python
# NOTE: Isotonic calibration disabled - was causing severe prediction compression
# (89% compression, all predictions clustered in range 3-5 instead of 1-16)
# See PREDICTION_COMPRESSION_ROOT_CAUSE.md for details
# Use raw RF model instead - bias calibration will be applied during prediction
self.rf_model = base_rf
```

### 2. General Model Training
**File**: `model_training/historical/train_race_model.py:262-266`

**Before**:
```python
self.rf_model = CalibratedRegressor(
    base_regressor=base_rf,
    clip_min=1.0  # Race positions start at 1
)
```

**After**:
```python
# NOTE: Isotonic calibration disabled - was causing severe prediction compression
# (89% compression, all predictions clustered in range 3-5 instead of 1-16)
# See PREDICTION_COMPRESSION_ROOT_CAUSE.md for details
# Use raw RF model instead - bias calibration will be applied during prediction
self.rf_model = base_rf
```

---

## Prediction Code Compatibility

✅ **No changes needed** - prediction code already handles raw RF models:

### Quinté Predictions
`race_prediction/predict_quinte.py:557-561`:
```python
if hasattr(self.rf_model, 'predict_raw'):
    raw_rf_preds = self.rf_model.predict_raw(X_rf)
else:
    # Fallback if not CalibratedRegressor ← This will now be used
    raw_rf_preds = self.rf_model.predict(X_rf)
```

### General Predictions
`race_prediction/race_predict.py:437`:
```python
predictions = self.rf_model.predict(X_for_prediction)
```

Both work seamlessly with `RandomForestRegressor` directly.

---

## Next Steps: Retrain Models

### Training Commands

**Quinté Model**:
```bash
python3 model_training/historical/train_quinte_model.py \
    --years 2 \
    --output-dir models/$(date +%Y-%m-%d) \
    --verbose
```

**General Model**:
```bash
python3 model_training/historical/train_race_model.py \
    --years 2 \
    --output-dir models/$(date +%Y-%m-%d) \
    --verbose
```

### Expected Results

**Before (Old Models with Isotonic Calibration)**:
- RF predictions span: 1.1-2.2 positions ❌
- Compression: 89%
- All predictions clustered: 3-5
- Bias calibration rejected (makes MAE worse)

**After (New Models without Isotonic Calibration)**:
- RF predictions span: 8-12 positions ✅
- Compression: <20% (acceptable)
- Predictions spread: 1-16 (full field)
- Bias calibration effective (improves MAE)

### Performance Validation

After retraining, run these diagnostics:

1. **Check prediction spread**:
   ```bash
   python3 scripts/diagnose_prediction_compression.py
   ```

   Expected output:
   ```
   📊 RF PREDICTIONS
      Average span: 8-12 (Expected: ~10.4) ✅
      Status: Normal spread
   ```

2. **Retrain bias calibration**:
   ```bash
   python3 scripts/calibrate_models.py --model-type general --days 30
   python3 scripts/calibrate_models.py --model-type quinte --days 30
   ```

   Expected: Calibration should improve MAE by 2-5%

3. **Benchmark performance**:
   ```bash
   python3 scripts/assess_performance.py --days 14
   ```

   Compare:
   - Old model MAE vs New model MAE
   - Winner hit rate
   - Top-3 hit rate

---

## Rollback Plan (If Needed)

If new models perform worse:

### Option 1: Revert Training Code
```bash
git checkout HEAD -- model_training/historical/train_quinte_model.py
git checkout HEAD -- model_training/historical/train_race_model.py
```

### Option 2: Use Old Models
Keep old models and investigate isotonic calibration training:
- Check calibration set size and quality
- Review y_min, y_max parameters
- Try separate calibration validation set

### Option 3: Hybrid Approach
Train with lighter isotonic calibration:
```python
# In train_quinte_model.py
self.rf_model = CalibratedRegressor(
    base_regressor=base_rf,
    clip_min=1.0,
    clip_max=18.0  # Add upper bound to prevent over-compression
)

# Use larger, separate calibration set
self.rf_model.fit(X_train, y_train, X_calib=X_val, y_calib=y_val)
```

---

## Timeline

- ✅ **Step 1**: Disable isotonic calibration in training code (DONE)
- ⏳ **Step 2**: Retrain quinté model (~30-60 minutes)
- ⏳ **Step 3**: Retrain general model (~30-60 minutes)
- ⏳ **Step 4**: Run diagnostics to verify spread improved
- ⏳ **Step 5**: Retrain bias calibration with new predictions
- ⏳ **Step 6**: Benchmark performance for 7-14 days
- ⏳ **Step 7**: Deploy to production if performance acceptable

---

## Documentation References

- **Root Cause Analysis**: `PREDICTION_COMPRESSION_ROOT_CAUSE.md`
- **Diagnostic Scripts**:
  - `scripts/diagnose_prediction_compression.py` - Analyze prediction spread
  - `scripts/debug_rf_calibration.py` - Compare raw vs calibrated predictions
  - `scripts/calibrate_models.py` - Retrain bias calibration
  - `scripts/assess_performance.py` - Benchmark model performance

---

## Notes

- Old models (with CalibratedRegressor) will continue to work
- Prediction code is backward compatible
- No immediate deployment impact - changes only affect new model training
- Can test new models side-by-side with old models before switching
