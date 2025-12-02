# Quinté Comparison Update

## Changes Made

Updated the **Quinté Prediction > Generate Comparison Report** feature in UIApp.py with two critical fixes:

1. **🐛 Fixed Critical Bug**: General predictions were comparing `horse_id` against `numero`, causing zero matches!
2. **📊 Database Migration**: Updated to use dedicated `quinte_predictions` table instead of CSV files

## Critical Bug Fix

### The Problem
The comparison was **never finding any matches** for general predictions because:
- `actual_results` contains numero values (e.g., "14-16-3-5-7")
- General predictions were using `horse_id` from `race_predictions` table
- Comparing `horse_id` (e.g., 1748959) against `numero` (e.g., 14) = **always zero matches**

### The Solution
Changed to extract general predictions from `daily_race.prediction_results` JSON field, which contains both:
- `numero` - Horse number in race (matches actual_results)
- `predicted_rank` - Predicted finishing position

Now comparisons work correctly: `numero` vs `numero` ✅

## What Was Changed

### Before (Old System)
- Loaded Quinté predictions from CSV files in `predictions/` directory
- Used `glob()` to find latest `quinte_predictions_*.csv` file
- Required predictions to be saved to CSV files
- Could only compare most recent prediction file

### After (New System)
- Loads Quinté predictions directly from `quinte_predictions` database table
- Uses SQL query with date filtering
- No dependency on CSV files
- Can compare all predictions stored in database for selected date range

## Database Query

The comparison now uses this query to load Quinté predictions:

```sql
SELECT
    race_id,
    horse_number as numero,
    horse_id,
    final_prediction as predicted_position,
    predicted_rank
FROM quinte_predictions
WHERE race_id IN (race_ids_list)
ORDER BY race_id, predicted_rank
```

## Updated Code Location

**File**: [UI/UIApp.py](UI/UIApp.py)
**Lines**: 1322-1505 (Generate Comparison button handler)

### Key Changes:

1. **🐛 CRITICAL FIX - General predictions now use numero** (lines 1412-1432):
   ```python
   # OLD - Using horse_id (WRONG!)
   pred_query = """
   SELECT race_id, horse_id, final_prediction
   FROM race_predictions
   """
   df_general_preds = pd.read_sql_query(pred_query, conn)
   general_top5 = df_general_preds.head(5)['horse_id'].tolist()  # ❌ horse_id

   # NEW - Using numero from prediction_results JSON (CORRECT!)
   pred_data = json.loads(race['prediction_results'])
   predictions_list = pred_data.get('predictions', [])
   sorted_preds = sorted(predictions_list, key=lambda x: x.get('predicted_rank', 999))
   general_top5 = [int(p['numero']) for p in sorted_preds[:5]]  # ✅ numero
   ```

2. **Removed CSV file loading** (lines 1388-1401):
   ```python
   # OLD - Loading from CSV files
   predictions_dir = Path('predictions')
   quinte_pred_files = list(predictions_dir.glob("quinte_predictions_*.csv"))
   df_quinte_preds = pd.read_csv(latest_file)
   ```

3. **Added database query for Quinté predictions** (lines 1376-1387):
   ```python
   # NEW - Loading from database
   quinte_query = f"""
   SELECT race_id, horse_number as numero, horse_id,
          final_prediction, predicted_rank
   FROM quinte_predictions
   WHERE race_id IN ({comp_placeholders})
   ORDER BY race_id, predicted_rank
   """
   df_quinte_preds = pd.read_sql_query(quinte_query, conn, params=race_comps)
   ```

4. **Added warning for empty predictions** (lines 1401-1404):
   ```python
   if len(df_quinte_preds) == 0:
       st.warning("⚠️ No Quinté predictions found in database for selected date range.
                   Please run Quinté predictions first.")
   ```

4. **Simplified prediction processing** (lines 1441-1451):
   - No need to handle None case for `df_quinte_preds`
   - Already sorted by `predicted_rank` from query
   - Direct column access with proper data types

## How to Use

### 1. Run Quinté Predictions First

Before using the comparison feature, ensure you have Quinté predictions in the database:

```python
# From UI: Quinté Prediction > Run Quinté Prediction
# Or from CLI:
python race_prediction/predict_quinte.py --date 2025-11-03
```

This will populate the `quinte_predictions` table.

### 2. Generate Comparison Report

1. Go to **Quinté Prediction** tab in UI
2. Scroll to **Compare Models** section
3. Select date range (start date → end date)
4. Click **📊 Generate Comparison Report**

### 3. View Results

The comparison will show:
- **Quinté Model** vs **General Model** performance
- Top 5 accuracy for each model
- Winner prediction accuracy
- Quinté Désordre (all 5 correct) statistics
- Detailed race-by-race comparison

## Benefits

### ✅ Advantages of Database Storage

1. **No file dependencies** - All predictions in one place
2. **Date range filtering** - Compare any date range in database
3. **Historical tracking** - Keep all predictions indefinitely
4. **Better performance** - SQL queries faster than file I/O
5. **Consistency** - Single source of truth for predictions
6. **Concurrent access** - Multiple users can query simultaneously

### ✅ Data Integrity

- Predictions stored atomically with race metadata
- Unique constraints prevent duplicates (race_id + horse_id)
- Indexed for fast lookups by race_id and date
- Includes all model outputs (Quinté RF, TabNet, General RF, TabNet)

## Troubleshooting

### Error: "No Quinté predictions found in database"

**Cause**: The `quinte_predictions` table is empty for the selected date range.

**Solution**:
1. Run Quinté predictions for the desired date range
2. Verify predictions were stored:
   ```sql
   SELECT COUNT(*) FROM quinte_predictions WHERE race_date BETWEEN '2025-11-01' AND '2025-11-03';
   ```

### Error: "cannot access local variable 'pd' where it is not associated with a value"

**Cause**: This was a scoping issue in the old code (now fixed).

**Solution**: Code has been updated to import `sqlite3` at function scope (line 1323).

## Migration Notes

### Existing CSV Predictions

Old predictions stored in CSV files are **not migrated** to the database automatically. The comparison feature will only use predictions stored in the `quinte_predictions` table.

If you need historical data:
1. Re-run predictions for historical dates
2. Or manually import CSV data to database (script available on request)

### Backward Compatibility

The general model predictions still use the `race_predictions` table (no changes). Only Quinté-specific predictions use the new dedicated table.

## Testing

To verify the update works:

```bash
# 1. Check table exists and query works
python3 -c "
import sqlite3
import pandas as pd
conn = sqlite3.connect('data/hippique2.db')
df = pd.read_sql_query('SELECT COUNT(*) as count FROM quinte_predictions', conn)
print(f'Quinté predictions in DB: {df.iloc[0][\"count\"]}')
conn.close()
"

# 2. Run a Quinté prediction to populate data
python race_prediction/predict_quinte.py --date 2025-11-03

# 3. Generate comparison in UI
# Go to UI > Quinté Prediction > Generate Comparison
```

## Files Modified

1. **[UI/UIApp.py](UI/UIApp.py)** (lines 1322-1505)
   - Removed CSV file loading logic
   - Added database query for `quinte_predictions` table
   - Added empty data warning
   - Simplified prediction processing logic

## Related Documentation

- [QUINTE_STORAGE_IMPLEMENTATION.md](QUINTE_STORAGE_IMPLEMENTATION.md) - Quinté storage system details
- [race_prediction/quinte_prediction_storage.py](race_prediction/quinte_prediction_storage.py) - Storage class implementation
- [race_prediction/predict_quinte.py](race_prediction/predict_quinte.py) - Quinté prediction engine

## Summary

The Quinté comparison feature now:
- ✅ Uses dedicated `quinte_predictions` database table
- ✅ No longer depends on CSV files
- ✅ Supports date range filtering from database
- ✅ Warns when no predictions found
- ✅ More reliable and faster
- ✅ Better integrated with the pipeline

All predictions must now be in the database for comparison. Simply run Quinté predictions from the UI to populate the data.
