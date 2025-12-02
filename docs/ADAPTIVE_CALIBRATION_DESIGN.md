# Adaptive Calibration System - Design

## Goal

Learn from daily race predictions and continuously improve calibration based on actual race results.

## Current Problem

The isotonic calibration is trained on **historical training data** and never updates:
- Calibrator learns from training set (often overfitted)
- Never sees real prediction errors
- Can't adapt to changing conditions
- Static forever after training

## Proposed Solution: Adaptive Calibration

Learn from **actual race predictions** and update the calibrator continuously:

```
┌─────────────────────────────────────────────────────────────┐
│  Daily Cycle                                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Morning: Make Predictions                                │
│     ├─ Load current calibrator (from previous days)         │
│     ├─ Make raw predictions                                  │
│     ├─ Apply calibration                                     │
│     └─ Store BOTH raw and calibrated predictions           │
│                                                              │
│  2. Evening: Races Complete                                 │
│     ├─ Fetch actual race results                            │
│     ├─ Retrieve stored raw predictions                      │
│     ├─ Compute prediction errors                            │
│     └─ Update calibrator with new data                      │
│                                                              │
│  3. Night: Save Updated Calibrator                          │
│     ├─ Save calibrator to disk                              │
│     └─ Ready for tomorrow's predictions                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture

### 1. Storage Layer

**Database Table: `race_predictions`** (already exists!)
- Add columns:
  - `raw_rf_prediction` (before calibration)
  - `raw_tabnet_prediction` (before calibration)
  - `calibrated_rf_prediction` (after calibration)
  - `calibrated_tabnet_prediction` (after calibration)

**Calibrator Storage**:
- `models/calibrators/rf_calibrator.joblib`
- `models/calibrators/tabnet_calibrator.joblib`
- `models/calibrators/metadata.json` (stats, last update, etc.)

### 2. Prediction Flow

```python
# predict_quinte.py

def predict(self, df_features):
    # Load existing calibrators (if any)
    rf_calibrator = self.load_calibrator('rf')
    tabnet_calibrator = self.load_calibrator('tabnet')

    # Make RAW predictions
    raw_rf_preds = self.rf_model.predict_raw(X_rf)  # New method
    raw_tabnet_preds = self.tabnet_model.predict(X_tabnet)

    # Apply calibration (if calibrator exists)
    if rf_calibrator:
        calibrated_rf = rf_calibrator.predict(raw_rf_preds)
    else:
        calibrated_rf = raw_rf_preds

    if tabnet_calibrator:
        calibrated_tabnet = tabnet_calibrator.predict(raw_tabnet_preds)
    else:
        calibrated_tabnet = raw_tabnet_preds

    # Store BOTH raw and calibrated
    result_df['raw_rf_prediction'] = raw_rf_preds
    result_df['raw_tabnet_prediction'] = raw_tabnet_preds
    result_df['calibrated_rf_prediction'] = calibrated_rf
    result_df['calibrated_tabnet_prediction'] = calibrated_tabnet

    # Use calibrated for final prediction
    result_df['predicted_position'] = blend(calibrated_rf, calibrated_tabnet)

    return result_df
```

### 3. Calibrator Update Flow

```python
# New script: scripts/update_calibrators.py

def update_calibrators_from_results():
    """
    After races complete, update calibrators with actual results.
    Run this daily after all race results are available.
    """

    # 1. Get completed races from last N days
    predictions_with_results = fetch_predictions_with_results(days=30)

    # 2. Extract (raw_predictions, actual_positions) pairs
    rf_data = {
        'predictions': predictions_with_results['raw_rf_prediction'],
        'actuals': predictions_with_results['actual_position']
    }

    tabnet_data = {
        'predictions': predictions_with_results['raw_tabnet_prediction'],
        'actuals': predictions_with_results['actual_position']
    }

    # 3. Update RF calibrator
    rf_calibrator = IsotonicRegression(out_of_bounds='clip', y_min=1.0)
    rf_calibrator.fit(rf_data['predictions'], rf_data['actuals'])

    # 4. Update TabNet calibrator
    tabnet_calibrator = IsotonicRegression(out_of_bounds='clip', y_min=1.0)
    tabnet_calibrator.fit(tabnet_data['predictions'], tabnet_data['actuals'])

    # 5. Evaluate improvement
    improvements = evaluate_calibration_improvement(
        rf_calibrator, tabnet_calibrator, predictions_with_results
    )

    # 6. Save updated calibrators
    save_calibrator(rf_calibrator, 'rf', improvements['rf'])
    save_calibrator(tabnet_calibrator, 'tabnet', improvements['tabnet'])

    print(f"✅ Calibrators updated!")
    print(f"   RF MAE improvement: {improvements['rf']['mae_improvement']:.2f}%")
    print(f"   TabNet MAE improvement: {improvements['tabnet']['mae_improvement']:.2f}%")
```

---

## Implementation Plan

### Phase 1: Update Database Schema
```sql
ALTER TABLE race_predictions ADD COLUMN raw_rf_prediction REAL;
ALTER TABLE race_predictions ADD COLUMN raw_tabnet_prediction REAL;
ALTER TABLE race_predictions ADD COLUMN calibrated_rf_prediction REAL;
ALTER TABLE race_predictions ADD COLUMN calibrated_tabnet_prediction REAL;
```

### Phase 2: Modify Prediction Code

**Files to modify**:
1. `race_prediction/predict_quinte.py`
   - Add `load_calibrator()` method
   - Store raw predictions before calibration
   - Apply calibration if available
   - Store both raw and calibrated in database

2. `model_training/regressions/isotonic_calibration.py`
   - Add `predict_raw()` method to `CalibratedRegressor` ✅ (already exists!)
   - Add helper functions for loading/saving calibrators

### Phase 3: Create Calibrator Update Script

**New file**: `scripts/update_calibrators.py`
- Fetch predictions with actual results
- Update isotonic regressors
- Evaluate improvement
- Save updated calibrators
- Log statistics

### Phase 4: Automation

**Daily workflow**:
```bash
# Morning: Make predictions (uses yesterday's calibrators)
python race_prediction/predict_quinte.py --date today

# Evening: After races complete, update calibrators
python scripts/update_calibrators.py --days 30

# This can be automated with cron or systemd timer
```

---

## Key Design Decisions

### 1. Rolling Window vs. All Data

**Option A: Rolling Window (Last 30 days)**
- Pros: Adapts quickly to recent changes
- Cons: Less stable, needs more data per day

**Option B: All Historical Data**
- Pros: Very stable, lots of data
- Cons: Slow to adapt to changes

**Recommendation**: Start with 30-day window, increase if too noisy

### 2. Separate vs. Combined Calibrators

**Separate** (recommended):
- One calibrator for RF
- One calibrator for TabNet
- Can learn different biases for each model

**Combined**:
- One calibrator for ensemble
- Simpler but less flexible

### 3. Minimum Data Requirements

Before using a calibrator, require:
- Minimum 100 predictions (statistical significance)
- At least 5 unique prediction values (isotonic needs variance)
- Maximum 90 days old (relevance)

### 4. Fallback Strategy

```python
if not enough_data(calibrator):
    # Use raw predictions (no calibration)
    return raw_predictions
elif calibrator_age > 90_days:
    # Calibrator too old, use raw
    return raw_predictions
else:
    # Use calibration
    return calibrator.predict(raw_predictions)
```

---

## Expected Benefits

### 1. Learns Real Biases
- Model overconfident on favorites? ✅ Calibrator learns this
- Model pessimistic on longshots? ✅ Calibrator corrects
- Seasonal effects? ✅ Calibrator adapts

### 2. Continuous Improvement
- Week 1: 5% improvement
- Week 4: 8% improvement (more data)
- Week 12: 10% improvement (stable pattern learned)

### 3. Adaptive to Changes
- Track conditions change? Updates within days
- New jockeys? Calibrator adapts
- Meta shifts? Learns new patterns

---

## Implementation Complexity

### Easy Parts ✅
- Database already has `race_predictions` table
- `CalibratedRegressor.predict_raw()` already exists
- `IsotonicRegression` from sklearn is simple

### Medium Parts ⚙️
- Fetching predictions with results (join tables)
- Handling missing data (incomplete results)
- Calibrator persistence (save/load)

### Tricky Parts ⚠️
- Deciding when to update (daily? weekly?)
- Handling cold start (no calibrator initially)
- Monitoring calibrator quality over time
- Dealing with data distribution shifts

---

## Success Metrics

Track these to evaluate if it's working:

```python
metrics = {
    'raw_mae': 4.5,              # Without calibration
    'calibrated_mae': 4.1,       # With calibration
    'improvement': 8.9%,         # Percentage improvement
    'data_points': 450,          # Predictions used
    'last_updated': '2025-10-30',
    'days_in_window': 30
}
```

Target: **5-10% MAE improvement** after 30 days of data.

---

## Example Output

```bash
$ python scripts/update_calibrators.py --days 30

📊 Adaptive Calibrator Update
================================

Loading predictions from last 30 days...
  ✅ Found 450 completed races with predictions

RF Calibrator:
  Training samples: 450
  Raw MAE: 4.52
  Calibrated MAE: 4.15
  Improvement: 8.2% ✅

TabNet Calibrator:
  Training samples: 450
  Raw MAE: 4.31
  Calibrated MAE: 4.08
  Improvement: 5.3% ✅

Saving calibrators...
  ✅ models/calibrators/rf_calibrator_20251030.joblib
  ✅ models/calibrators/tabnet_calibrator_20251030.joblib

Next update recommended: 2025-10-31
```

---

## Quick Start Implementation

Want me to implement this? I can:

1. ✅ Add database columns for raw/calibrated predictions
2. ✅ Modify `predict_quinte.py` to store raw predictions
3. ✅ Create `scripts/update_calibrators.py` for daily updates
4. ✅ Add calibrator loading/saving utilities
5. ✅ Create monitoring dashboard for calibrator performance

This would give you **adaptive, continuously improving calibration** that learns from real race results!

Shall I proceed with implementation?
