# Quinté Batch Prediction Feature

## Overview

Added a new **Batch Prediction** feature to the Quinté Predictions section in UIApp that allows predicting all Quinté races across a date range.

## UI Changes

### New Tab Structure

The Quinté Predictions section now has two tabs:

1. **📅 Single Date** - Predict Quinté races for a single date (existing functionality)
2. **📊 Batch Prediction** - Predict all Quinté races across a date range (NEW)

### Batch Prediction Tab Features

#### Date Range Selection
- Start Date (default: 30 days ago)
- End Date (default: today)

#### Race Status Display
Shows all Quinté races in the selected date range with:
- Date
- Track
- Race Name
- Status: ✅ Predicted or ⏳ Pending

**Summary Statistics:**
- Total Quinté races found
- Races already predicted
- Races needing predictions

#### Prediction Buttons

**1. 🏇 Predict All Quinté Races** (Primary button)
- Predicts only races that don't have Quinté predictions yet
- Processes each unique date sequentially
- Shows progress bar and status for each date
- Displays summary: total races, horses analyzed, failed dates (if any)
- Automatically refreshes the page when complete

**2. 🔁 Force Reprediction**
- Repredicts ALL Quinté races in the date range
- Overwrites existing predictions
- Useful for rerunning with updated models or calibrators

## Database Integration

### Query Used

```sql
SELECT
    dr.comp,
    dr.jour,
    dr.hippo,
    dr.prixnom,
    CASE
        WHEN qp.race_id IS NOT NULL THEN 1
        ELSE 0
    END as has_quinte_prediction
FROM daily_race dr
LEFT JOIN quinte_predictions qp ON dr.comp = qp.race_id
WHERE dr.quinte = 1
AND dr.jour BETWEEN ? AND ?
GROUP BY dr.comp
ORDER BY dr.jour DESC, dr.comp
```

This checks if races have entries in the `quinte_predictions` table to determine prediction status.

## Implementation Details

### File Modified
**[UI/UIApp.py](UI/UIApp.py)** (lines 1184-1443)

### Key Components

#### 1. Tab Structure (lines 1192-1193)
```python
tab_single, tab_batch = st.tabs(["📅 Single Date", "📊 Batch Prediction"])
```

#### 2. Batch Date Range (lines 1261-1274)
```python
batch_start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
batch_end_date = st.date_input("End Date", value=datetime.now())
```

#### 3. Race Status Query (lines 1283-1306)
Queries `daily_race` and `quinte_predictions` tables to determine which races need predictions.

#### 4. Batch Prediction Logic (lines 1332-1391)
```python
for idx, race_date in enumerate(dates_to_predict):
    result = predictor.run_prediction(
        race_date=race_date,
        output_dir='predictions',
        store_to_db=True
    )
    # Track successes and failures
```

## User Workflow

### Typical Usage

1. **Navigate**: Go to UI → Quinté Predictions
2. **Select Tab**: Click "📊 Batch Prediction"
3. **Set Date Range**: Choose start and end dates (default: last 30 days)
4. **Review Status**: See which races already have predictions
5. **Predict**: Click "🏇 Predict All Quinté Races"
6. **Monitor Progress**: Watch the progress bar for each date
7. **Review Results**: See summary of races predicted and horses analyzed

### Example Scenario

**Date Range:** September 1, 2025 - November 3, 2025

**Status Display:**
```
Found 45 Quinté races:
- ✅ 12 already predicted
- ⏳ 33 need predictions
```

**Click "Predict All Quinté Races":**
- Processes 33 races across multiple dates
- Shows progress: "Processing 2025-09-05 (5/20)..."
- Updates log: "✅ 2025-09-05: 2 races, 28 horses"

**Result:**
```
✅ Batch prediction complete!
- 33 races predicted
- 462 horses analyzed
```

## Benefits

### ✅ Advantages

1. **Time Savings** - Predict multiple dates at once instead of one by one
2. **Completeness** - Easy to ensure all Quinté races have predictions
3. **Visibility** - Clear status display shows what's done and what's pending
4. **Safety** - Only predicts races that need it (unless force reprediction)
5. **Auditability** - Detailed logging of successes and failures
6. **Integration** - Uses existing QuintePredictionEngine and dedicated storage

### 📊 Performance

- Sequential processing (one date at a time)
- Progress tracking for each date
- Error handling: continues even if one date fails
- Stores all predictions to `quinte_predictions` database table

## Comparison with General Predictions

| Feature | General Predictions | Quinté Predictions |
|---------|---------------------|-------------------|
| **Batch Mode** | Background jobs via CLI | In-UI sequential processing |
| **Progress** | Database-tracked jobs | Real-time progress bar |
| **Storage** | `race_predictions` table | `quinte_predictions` table |
| **Models** | RF + TabNet general | Quinté-specific + General blend |
| **Scope** | All races | Quinté races only |

## Error Handling

### Failed Dates
If a date fails to predict:
- Error is logged
- Date is added to `failed_dates` list
- Processing continues with next date
- Summary shows failed dates at the end

### Common Issues

**Issue**: No Quinté races found
**Cause**: Date range has no Quinté races or wrong database
**Solution**: Check date range or sync races first

**Issue**: "All Quinté races already have predictions!"
**Cause**: Clicked "Predict All" but nothing needs prediction
**Solution**: Use "Force Reprediction" to overwrite existing predictions

## Future Enhancements

Potential improvements:

1. **Background Processing** - Like general predictions, use background jobs
2. **Selective Prediction** - Checkboxes to select specific races
3. **Date Filtering** - Filter by track, distance, race type
4. **Parallel Processing** - Predict multiple dates simultaneously
5. **Progress Persistence** - Resume interrupted batch jobs

## Testing

To test the feature:

```bash
# 1. Sync some Quinté races
# Go to UI > Execute Prediction > Refresh Races

# 2. Go to Quinté Predictions > Batch Prediction tab

# 3. Select date range (e.g., last 30 days)

# 4. Click "Predict All Quinté Races"

# 5. Verify predictions stored:
sqlite3 data/hippique2.db "SELECT COUNT(*) FROM quinte_predictions WHERE race_date >= '2025-10-01';"

# 6. Check comparison feature works:
# Go to Compare Models section and generate comparison
```

## Related Files

- [UI/UIApp.py](UI/UIApp.py) - UI implementation
- [race_prediction/predict_quinte.py](race_prediction/predict_quinte.py) - Prediction engine
- [race_prediction/quinte_prediction_storage.py](race_prediction/quinte_prediction_storage.py) - Storage class
- [QUINTE_STORAGE_IMPLEMENTATION.md](QUINTE_STORAGE_IMPLEMENTATION.md) - Storage system docs

## Summary

The new batch prediction feature makes it easy to:
- ✅ Predict all Quinté races across any date range
- ✅ See clearly what's done and what's pending
- ✅ Track progress and handle errors gracefully
- ✅ Store predictions to dedicated Quinté database
- ✅ Maintain compatibility with single-date predictions

This completes the Quinté prediction workflow with both single-date and batch modes! 🎉
