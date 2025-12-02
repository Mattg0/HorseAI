# Quinté Prediction Storage Implementation

## Overview

Quinté predictions now have a **dedicated storage system** separate from general race predictions, with Quinté-specific fields and analysis capabilities.

## New Database Tables

### 1. `quinte_predictions` Table

Stores horse-level predictions for Quinté races with extended metadata.

**Key Fields:**

**Race Information:**
- `race_id` - Unique race identifier
- `race_date` - Date of the race (YYYY-MM-DD)
- `track` - Hippodrome name
- `race_number` - Prix number
- `race_name` - Prize name

**Model Predictions:**
- `quinte_rf_prediction` - Quinté-specific RF model
- `quinte_tabnet_prediction` - Quinté-specific TabNet model
- `general_rf_prediction` - General RF model (for blending)
- `general_tabnet_prediction` - General TabNet model (for blending)

**Ensemble & Blending:**
- `quinte_weight` - Weight for Quinté model (e.g., 0.20)
- `general_weight` - Weight for General model (e.g., 0.80)
- `ensemble_weight_rf` - RF weight in ensemble
- `ensemble_weight_tabnet` - TabNet weight in ensemble
- `ensemble_prediction` - Weighted average

**Competitive Analysis:**
- `competitive_adjustment` - Position adjustment
- `primary_advantage_type` - Type of advantage (speed, track, class, form)
- `advantage_strength` - Magnitude of advantage

**Calibration:**
- `calibrated_rf_prediction` - Calibrated RF prediction
- `calibrated_tabnet_prediction` - Calibrated TabNet prediction
- `calibration_applied` - Boolean flag

**Horse Information:**
- `horse_id` - Database horse ID
- `horse_number` - Race numero
- `horse_name` - Horse name

**Quinté-Specific:**
- `is_favorite` - Boolean flag for favorites
- `quinte_score` - Quinté performance score
- `quinte_form_rating` - Form rating for Quinté races

**Results (post-race):**
- `actual_result` - Actual finishing position
- `was_in_quinte` - Boolean (top 5?)

### 2. `quinte_race_summary` Table

Stores race-level summary information.

**Fields:**
- `race_id` - Unique identifier
- `race_date`, `track`, `race_number`, `race_name`
- `predicted_quinte` - Top 5 predicted horses (comma-separated)
- `predicted_winner` - Predicted winner numero
- `total_horses` - Number of horses
- `quinte_accuracy` - Post-race: how many of top 5 were correct
- `winner_correct` - Post-race: boolean
- `quinte_weight`, `general_weight` - Blend weights used
- `calibration_applied` - Boolean

## Implementation

### New File: `quinte_prediction_storage.py`

```python
from race_prediction.quinte_prediction_storage import QuintePredictionStorage

# Initialize storage
storage = QuintePredictionStorage(db_path="data/hippique2.db", verbose=True)

# Store predictions
storage.store_quinte_predictions(
    race_id="1234567",
    predictions_data=[...],  # List of horse predictions
    race_metadata={...}      # Race-level info
)

# Retrieve predictions
predictions = storage.get_race_predictions(race_id="1234567")
daily_predictions = storage.get_predictions_by_date(race_date="2025-10-15")
summaries = storage.get_race_summary(race_date="2025-10-15")

# Update results after race
storage.update_actual_results(
    race_id="1234567",
    results={horse_id: position, ...}
)
```

### Modified: `predict_quinte.py`

**Changed initialization:**
```python
# OLD
from .simple_prediction_storage import SimplePredictionStorage
self.prediction_storage = SimplePredictionStorage(...)

# NEW
from .quinte_prediction_storage import QuintePredictionStorage
self.prediction_storage = QuintePredictionStorage(...)
```

**Changed storage call:**
```python
# OLD
self.prediction_storage.store_race_predictions(
    race_id=race_comp,
    predictions_data=predictions_data
)

# NEW
self.prediction_storage.store_quinte_predictions(
    race_id=race_comp,
    predictions_data=predictions_data,
    race_metadata=race_metadata  # NEW: race-level info
)
```

## Benefits

### 1. Quinté-Specific Fields
- Track Quinté model vs General model performance separately
- Store blend weights for analysis
- Record calibration status

### 2. Better Analysis
- Dedicated indexes for Quinté analysis queries
- Race-level summaries for quick statistics
- Track top 5 accuracy (was_in_quinte field)

### 3. Separation of Concerns
- Quinté predictions don't clutter general `race_predictions` table
- Different schema optimized for Quinté analysis
- Can have different retention policies

### 4. Post-Race Analysis
```sql
-- Quinté accuracy for a date
SELECT
    race_id,
    COUNT(*) as total_horses,
    SUM(CASE WHEN was_in_quinte = 1 THEN 1 ELSE 0 END) as predicted_in_quinte,
    SUM(CASE WHEN actual_result <= 5 AND was_in_quinte = 1 THEN 1 ELSE 0 END) as correct_quinte
FROM quinte_predictions
WHERE race_date = '2025-10-15'
  AND actual_result IS NOT NULL
GROUP BY race_id;
```

## Database Schema

```sql
-- View Quinté predictions for a race
SELECT
    horse_number,
    horse_name,
    quinte_rf_prediction,
    quinte_tabnet_prediction,
    general_rf_prediction,
    general_tabnet_prediction,
    final_prediction,
    predicted_rank,
    actual_result
FROM quinte_predictions
WHERE race_id = '1234567'
ORDER BY predicted_rank;

-- Race summary
SELECT
    race_date,
    track,
    race_name,
    predicted_quinte,
    predicted_winner,
    quinte_accuracy,
    winner_correct
FROM quinte_race_summary
WHERE race_date = '2025-10-15';
```

## Migration Notes

**No migration needed!** The new tables are created automatically:
- First time `QuintePredictionStorage` is instantiated, tables are created
- Old `race_predictions` data is untouched
- Both systems can coexist

**Future:** You can optionally migrate old Quinté predictions from `race_predictions` to `quinte_predictions` if needed.

## Comparison: Old vs New

| Aspect | Old (race_predictions) | New (quinte_predictions) |
|--------|------------------------|--------------------------|
| Storage | Mixed with general races | Dedicated Quinté table |
| Model tracking | RF + TabNet only | Quinté models + General models |
| Blend weights | Basic ensemble weights | Quinté/General blend + ensemble |
| Calibration | Not tracked | Calibration status recorded |
| Race summary | None | Dedicated summary table |
| Quinté analysis | Manual queries needed | Built-in Quinté fields |
| Post-race | Generic actual_result | was_in_quinte + accuracy |

## Usage Example

```python
# Run Quinté prediction (from UI or CLI)
from race_prediction.predict_quinte import QuintePredictionEngine

engine = QuintePredictionEngine(verbose=True)
result = engine.run_prediction(
    race_date='2025-10-15',
    output_dir='predictions',
    store_to_db=True  # Stores to quinte_predictions table
)

# Query predictions
from race_prediction.quinte_prediction_storage import QuintePredictionStorage

storage = QuintePredictionStorage(db_path="data/hippique2.db")

# Get all predictions for a race
predictions = storage.get_race_predictions(race_id="1234567")
print(predictions[['horse_name', 'final_prediction', 'predicted_rank']])

# Get all Quinté races for a date
daily = storage.get_predictions_by_date(race_date="2025-10-15")
print(f"Predicted {len(daily)} horses across Quinté races")

# Get race summaries
summaries = storage.get_race_summary(race_date="2025-10-15")
print(summaries[['track', 'race_name', 'predicted_quinte', 'predicted_winner']])
```

## Files Created/Modified

**Created:**
- `race_prediction/quinte_prediction_storage.py` - New storage class

**Modified:**
- `race_prediction/predict_quinte.py` - Updated to use new storage

**Database:**
- New table: `quinte_predictions`
- New table: `quinte_race_summary`
- Indexes created automatically

## Next Steps (Optional)

1. **Migrate historical data** - Move old Quinté predictions from `race_predictions` to `quinte_predictions`
2. **Add analysis tools** - Create helper functions for common Quinté analysis queries
3. **Performance tracking** - Build dashboards using `quinte_race_summary` data
4. **Calibration tuning** - Use Quinté-specific calibration data for model improvements

The system is ready to use! All new Quinté predictions will automatically go to the dedicated tables. 🎉
