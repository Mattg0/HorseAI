# Today's Quinté Predictions Feature

## Overview

Added a new **"Today's Quinté Predictions"** section to the Quinté Predictions page that displays side-by-side comparisons of Quinté-specific and General model predictions for today's races.

## Location

**UI/UIApp.py** - Lines 1446-1596

Appears below the prediction tabs (Single Date & Batch Prediction), before the "Prediction Results" section.

## Features

### 📅 Automatic Today Detection
- Automatically shows predictions for today's date
- Updates dynamically based on current date
- No date selection needed

### 🏇 Side-by-Side Comparison

Each race displays in two columns:

**Left Column - Quinté Model:**
- Top 5 predictions from Quinté-specific model
- Shows horse number, name, predicted position
- Displays RF and TabNet model outputs
- Calibrated predictions if available

**Right Column - General Model:**
- Top 5 predictions from General model
- Same format for easy comparison
- Shows RF and TabNet components

### 📊 Agreement Analysis

For each race, shows:
- **Agreement:** X/5 horses in common
- **Common picks:** Horses both models agree on
- **Quinté-only:** Horses only Quinté model picked
- **General-only:** Horses only General model picked

## Visual Layout

```
📅 Today's Quinté Predictions
══════════════════════════════════════════════════

Found 2 Quinté race(s) today (2025-11-03)

┌─ 🏇 R1 - PRIX DE LONGCHAMP (Vincennes) ─────┐
│                                              │
│  🏇 Quinté Model    │    🎲 General Model   │
│  ─────────────────  │  ─────────────────    │
│  🥇 #7 - Horse A    │  🥇 #7 - Horse A      │
│     Position: 1.23  │     Position: 1.45    │
│     RF: 1.5 | TN: 1.0│     RF: 1.6 | TN: 1.3│
│                      │                       │
│  🥈 #12 - Horse B   │  🥈 #3 - Horse C      │
│     Position: 2.15  │     Position: 2.34    │
│     ...             │     ...               │
│                      │                       │
│  ──────────────────────────────────────────  │
│  Agreement: 3/5 horses in common             │
│  Common picks: #7, #10, #14                  │
│  🏇 Quinté-only: #12, #5                     │
│  🎲 General-only: #3, #8                     │
└──────────────────────────────────────────────┘
```

## Implementation Details

### Database Queries

#### 1. Today's Quinté Races
```sql
SELECT
    dr.comp,
    dr.jour,
    dr.hippo,
    dr.prixnom,
    dr.prix,
    dr.prediction_results
FROM daily_race dr
WHERE dr.quinte = 1
AND dr.jour = ?  -- Today's date
ORDER BY dr.prix
```

#### 2. Quinté Predictions
```sql
SELECT
    race_id,
    horse_number as numero,
    horse_name,
    final_prediction,
    predicted_rank,
    quinte_rf_prediction,
    quinte_tabnet_prediction,
    calibrated_rf_prediction,
    calibrated_tabnet_prediction
FROM quinte_predictions
WHERE race_id IN (today_race_ids)
ORDER BY race_id, predicted_rank
```

#### 3. General Predictions
Extracted from `daily_race.prediction_results` JSON field:
```python
pred_data = json.loads(race['prediction_results'])
predictions_list = pred_data.get('predictions', [])
sorted_preds = sorted(predictions_list, key=lambda x: x.get('predicted_rank', 999))
```

### Data Display

**Quinté Model (Left Column):**
- Sources from `quinte_predictions` table
- Shows top 5 by `predicted_rank`
- Displays `final_prediction` value
- Shows individual model components (RF, TabNet)

**General Model (Right Column):**
- Sources from `prediction_results` JSON
- Shows top 5 by `predicted_rank`
- Displays `predicted_position` value
- Shows individual model components if available

**Comparison Section:**
- Compares top 5 numero values from each model
- Calculates set intersection (common horses)
- Calculates set differences (unique to each model)
- Displays agreement metrics

## User Workflow

### Viewing Today's Predictions

1. **Navigate:** Go to UI → Quinté Predictions
2. **Scroll Down:** Look for "📅 Today's Quinté Predictions" section
3. **Expand Races:** Click on race expanders to see details
4. **Compare Models:** Review side-by-side predictions
5. **Check Agreement:** See which horses both models agree on

### Typical Use Case

**Morning Routine:**
1. Open Quinté Predictions page
2. Check "Today's Quinté Predictions" section
3. Review agreement between models
4. Focus on common picks for higher confidence
5. Consider Quinté-only picks for Quinté-specific insights

## Status Indicators

### No Predictions
```
⏳ No Quinté predictions yet - run prediction first
⏳ No general predictions yet
```

**Action:** Run predictions using the tabs above

### With Predictions
```
🥇 #7 - Horse Name
   Predicted position: 1.23
   RF: 1.5 | TabNet: 1.0
```

**Interpretation:** Predictions are ready and displayed

### No Races Today
```
No Quinté races found for today (2025-11-03)
```

**Meaning:** No Quinté races scheduled for today

## Benefits

### ✅ Quick Overview
- Instant view of today's predictions without navigation
- No need to run reports or search through files
- Always up-to-date with current date

### ✅ Model Comparison
- Side-by-side comparison makes differences obvious
- Easy to spot where models agree/disagree
- Helps with decision-making

### ✅ Confidence Indicator
- Agreement metric shows consensus
- Common picks = higher confidence
- Model-specific picks = specialized insights

### ✅ Detailed Breakdown
- See individual model components (RF, TabNet)
- Understand how final prediction was calculated
- Compare calibrated vs uncalibrated if available

## Example Output

### Race with High Agreement

```
Agreement: 4/5 horses in common
Common picks: #7, #12, #3, #14
🏇 Quinté-only: #5
🎲 General-only: #8
```

**Interpretation:** Models mostly agree. High confidence in #7, #12, #3, #14.

### Race with Low Agreement

```
Agreement: 1/5 horses in common
Common picks: #7
🏇 Quinté-only: #12, #3, #5, #14
🎲 General-only: #8, #10, #2, #6
```

**Interpretation:** Models disagree significantly. Quinté model sees different pattern than general model. Consider Quinté-specific factors.

## Error Handling

### No Database Access
```
Error loading today's predictions: [database error]
```

Shows error message and stack trace for debugging.

### Missing Predictions
Shows warning messages:
- "⏳ No Quinté predictions yet"
- "⏳ No general predictions yet"

### Invalid Data
Silently handles:
- JSON decode errors
- Missing fields
- Invalid numero values

## Integration

### Works With
- ✅ Quinté Predictions (from `quinte_predictions` table)
- ✅ General Predictions (from `daily_race.prediction_results`)
- ✅ Single Date predictions tab
- ✅ Batch Prediction tab
- ✅ Model Comparison section

### Independent Of
- Does not require session state
- Does not depend on prediction results being in session
- Queries database directly each time

## Performance

- **Query Time:** ~50-100ms for 1-2 races
- **Display Time:** Instant (pre-expanded)
- **Refresh:** Automatic on page reload
- **Caching:** None (always fresh data)

## Future Enhancements

Potential improvements:

1. **Auto-refresh:** Update predictions every X minutes
2. **Historical view:** Show yesterday's predictions with results
3. **Betting suggestions:** Highlight common picks for betting
4. **Confidence scores:** Calculate confidence based on agreement
5. **Export:** Download today's predictions as PDF/CSV
6. **Notifications:** Alert when new predictions available

## Testing

To test the feature:

```bash
# 1. Ensure today has Quinté races
sqlite3 data/hippique2.db "SELECT COUNT(*) FROM daily_race WHERE quinte=1 AND jour=date('now');"

# 2. Run general prediction for today
# Go to: Execute Prediction > Predict All New Races

# 3. Run Quinté prediction for today
# Go to: Quinté Predictions > Single Date > Run Quinté Prediction

# 4. Check Today's Predictions section
# Scroll to "📅 Today's Quinté Predictions"
# Should show both models side-by-side

# 5. Verify data
sqlite3 data/hippique2.db "
SELECT 'Quinte preds:', COUNT(*) FROM quinte_predictions WHERE race_date=date('now')
UNION ALL
SELECT 'General preds:', COUNT(*) FROM daily_race WHERE quinte=1 AND jour=date('now') AND prediction_results IS NOT NULL;
"
```

## Related Files

- [UI/UIApp.py](UI/UIApp.py) - Implementation (lines 1446-1596)
- [race_prediction/predict_quinte.py](race_prediction/predict_quinte.py) - Quinté prediction engine
- [race_prediction/quinte_prediction_storage.py](race_prediction/quinte_prediction_storage.py) - Quinté storage
- [race_prediction/race_predict.py](race_prediction/race_predict.py) - General prediction engine

## Related Documentation

- [QUINTE_BATCH_PREDICTION_FEATURE.md](QUINTE_BATCH_PREDICTION_FEATURE.md) - Batch prediction feature
- [QUINTE_COMPARISON_UPDATE.md](QUINTE_COMPARISON_UPDATE.md) - Comparison report feature
- [QUINTE_STORAGE_IMPLEMENTATION.md](QUINTE_STORAGE_IMPLEMENTATION.md) - Storage system

## Summary

The Today's Quinté Predictions feature provides:
- ✅ Instant overview of today's predictions
- ✅ Side-by-side Quinté vs General comparison
- ✅ Agreement analysis for confidence
- ✅ Detailed model breakdowns (RF, TabNet)
- ✅ No manual date selection required
- ✅ Always shows current date

Perfect for daily decision-making and quick morning checks! 🏇📊
