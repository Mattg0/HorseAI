# Testing Two-Stage Quinté Predictions

## Quick Validation Checklist

### Option 1: UIApp Testing (Recommended)

1. **Launch UIApp**
   ```bash
   streamlit run UI/UIApp.py
   ```

2. **Navigate to Quinté Predictions Tab**

3. **Select a Date**
   - Choose any date that has quinté races
   - Today's date or recent date with loaded races

4. **Run Prediction**
   - Click "🏇 Run Quinté Prediction"
   - Watch console output (if verbose)

5. **Verify Results**

   **✓ Check Top 3** (Should match previous behavior)
   - Look at predicted positions 1-3
   - These should be horses with good form and low-to-mid odds (3-10)

   **✓ Check Positions 4-5** (Should show NEW behavior)
   - **BEFORE**: Typically had horses with odds 10-18
   - **AFTER**: Should now have more horses with odds 15-35 (longshots)

   **✓ Console Output** (If verbose=True)
   ```
   ================================================================================
   APPLYING TWO-STAGE REFINEMENT FOR POSITIONS 4-5
   ================================================================================

   --- Race R1-C4-20250102 (14 horses) ---

   Stage 1 - Top 3 (unchanged):
     # 1: HORSE NAME 1        (odds   3.5, pos 1.23)
     # 5: HORSE NAME 2        (odds   4.2, pos 2.45)
     # 3: HORSE NAME 3        (odds   6.8, pos 3.12)

   Longshots boosted (odds 15-35): 3 horses
     # 7: LONGSHOT HORSE 1    (odds  18.5) - boosted -2.5
     # 9: LONGSHOT HORSE 2    (odds  22.0) - boosted -2.5

   Stage 2 - Positions 4-5 (adjusted):
     # 7: LONGSHOT HORSE 1    (odds  18.5) - 6.45 → 3.95 (adj -2.5)
     # 9: LONGSHOT HORSE 2    (odds  22.0) - 7.12 → 4.62 (adj -2.5)

   ================================================================================
   TWO-STAGE REFINEMENT COMPLETE
   ================================================================================
   ```

   **✓ Database Check** (Optional)
   - Predictions are stored in `quinte_predictions` table
   - Check `final_prediction` column for each horse
   - Top 5 should have your expected horses

### Option 2: Command Line Testing

**Run validation script:**
```bash
# Test with synthetic data only (fast, no DB required)
python3 scripts/validate_two_stage_prediction.py --isolated

# Test with real race data from database
python3 scripts/validate_two_stage_prediction.py
```

**Run prediction directly:**
```bash
# Predict today's quinté races
python3 race_prediction/predict_quinte.py

# Predict specific date
python3 race_prediction/predict_quinte.py --date 2025-01-02 --verbose
```

### Option 3: Compare Before/After

**If you want to compare old vs new predictions:**

1. **Disable two-stage** (comment out line 599 in predict_quinte.py):
   ```python
   # result_df = self._apply_two_stage_refinement(result_df)
   ```

2. **Run prediction** and save results:
   ```bash
   python3 race_prediction/predict_quinte.py --date 2025-01-02 --output-dir predictions/old
   ```

3. **Re-enable two-stage** (uncomment line 599)

4. **Run prediction again** and save:
   ```bash
   python3 race_prediction/predict_quinte.py --date 2025-01-02 --output-dir predictions/new
   ```

5. **Compare CSV files**:
   ```bash
   # Look for differences in positions 4-5
   diff predictions/old/*.csv predictions/new/*.csv
   ```

## What to Look For

### ✅ Success Indicators

1. **No crashes or errors**
2. **Top 3 predictions unchanged** (same horses as before)
3. **Positions 4-5 have MORE longshots** (odds 15-35)
4. **Positions 4-5 have FEWER mid-odds horses** (odds 10-18)
5. **Console shows two-stage refinement logs** (if verbose)
6. **5 unique horses in top 5** (no duplicates)

### ❌ Failure Indicators

1. **AssertionError**: Validation checks failed
2. **Top 3 changed**: Top 3 horses are different from base model
3. **Duplicates in top 5**: Same horse appears twice
4. **Missing columns**: `cotedirect` or `predicted_position` not found
5. **Silent failure**: No two-stage logs appear (if verbose=True)

## Example Test Session

```bash
# Terminal 1: Start UIApp with verbose logging
export VERBOSE=1
streamlit run UI/UIApp.py

# Terminal 2: Watch logs (optional)
tail -f models/*/logs/*.log
```

**In UIApp:**
1. Go to "Quinté Predictions" tab
2. Select date: "2025-01-02"
3. Click "Run Quinté Prediction"
4. Wait for completion (~10-30 seconds per race)
5. Review results in UI

**Expected output:**
- Top 3: Low odds favorites (3-8)
- Positions 4-5: Higher odds horses (15-35)

## Rollback Instructions

**If results are worse or errors occur:**

1. Open [race_prediction/predict_quinte.py](../race_prediction/predict_quinte.py)
2. Go to line 599
3. Comment out the two-stage call:
   ```python
   # Apply competitive field analysis to enhance predictions
   result_df = self._apply_competitive_analysis(result_df)

   # DISABLED: Two-stage refinement
   # result_df = self._apply_two_stage_refinement(result_df)

   return result_df
   ```
4. Save and restart UIApp

The system will revert to original predictions immediately.

## Monitoring Long-Term

**Track these metrics over 20-30 races:**

1. **Position 4 accuracy**: Target 55-60% (was 38%)
2. **Position 5 accuracy**: Target 60-65% (was 49%)
3. **Top 3 accuracy**: Should stay ~70% (was 74%, 69%, 51%)
4. **Longshots in top 5**: Count how many horses with odds 15-35 appear
5. **Win rate on quinté bets**: Track bonus4, bonus3, quinté désordre

**Use the comparison feature in UIApp** (Quinté Results tab) to track accuracy over time.

---
**Last Updated**: 2025-12-03
**Status**: ✅ Implementation complete, ready for testing
