# Two-Stage Quinté Prediction Implementation

## Overview

Implemented two-stage prediction logic in `QuintePredictionEngine` to fix positions 4-5 accuracy based on failure analysis of 66 races.

## Problem Identified

From failure analysis:
- **Positions 1-3**: GOOD (74%, 69%, 51% success) ✓
- **Position 4**: TERRIBLE (38% success, drift +3.21) ✗
- **Position 5**: CHAOTIC (49% success, high variance) ✗

**Root Cause**: Model predicts horse quality well (winners), but positions 4-5 depend on race dynamics (trips, pace) not quality. Missing longshots: 92 horses at odds 20-35 finished top 5 but predicted at positions 10-12.

## Solution: Two-Stage Approach

### Stage 1: Keep Top 3 (Model Excels Here)
- Use existing model predictions for positions 1-3
- No changes to winner/podium prediction logic

### Stage 2: Recalculate Positions 4-5 with Odds-Based Adjustments

#### Critical Fix #1: Boost Longshots
- **Target**: Horses with odds 15-35
- **Adjustment**: -2.5 positions (move UP in ranking)
- **Rationale**: Analysis shows these horses often finish 4-5 but model predicts them at 6-12

#### Critical Fix #2: Penalize Mid-Odds at Predicted 4-5
- **Target**: Horses with odds 10-18 currently predicted at positions 4-5
- **Adjustment**: +1.5 positions (move DOWN in ranking)
- **Rationale**: Analysis shows these fail 62% of the time

## Implementation Details

### File Modified
- [race_prediction/predict_quinte.py](../race_prediction/predict_quinte.py)

### Methods Added

1. **`_apply_two_stage_refinement(df_predictions: pd.DataFrame) -> pd.DataFrame`** (Line 729-914)
   - Main refinement logic
   - Processes each race separately
   - Returns DataFrame with refined predictions

### Integration Point

In `QuintePredictionEngine.predict()` method (Line 598-599):
```python
# Apply competitive field analysis to enhance predictions
result_df = self._apply_competitive_analysis(result_df)

# Apply two-stage refinement for positions 4-5 (based on failure analysis)
result_df = self._apply_two_stage_refinement(result_df)

return result_df
```

### Key Features

#### Fail-Fast Validations
```python
# Check required columns
assert 'cotedirect' in df.columns, "Missing cotedirect (odds) column"
assert 'predicted_position' in df.columns, "Missing predicted_position column"
assert len(race_df) >= 10, f"Too few horses: {len(race_df)}"

# Stage validations
assert len(top3) == 3, f"Stage 1 failed: got {len(top3)} horses, expected 3"
assert len(new_45) == 2, f"Stage 2 failed: got {len(new_45)} horses, expected 2"

# Final validation
assert len(final_top5) == 5, f"Final top 5 has {len(final_top5)} horses, expected 5"
assert final_top5['numero'].duplicated().sum() == 0, "Duplicate horses in top 5!"
```

#### Comprehensive Logging (Verbose Mode)
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
  #11: LONGSHOT HORSE 3    (odds  28.0) - boosted -2.5

Mid-odds penalized (odds 10-18 at pos 4-5): 2 horses
  # 4: MID ODDS HORSE 1    (odds  12.5, pos 4.32) - penalized +1.5
  # 6: MID ODDS HORSE 2    (odds  14.2, pos 4.89) - penalized +1.5

Stage 2 - Positions 4-5 (adjusted):
  # 7: LONGSHOT HORSE 1    (odds  18.5) - 6.45 → 3.95 (adj -2.5)
  # 9: LONGSHOT HORSE 2    (odds  22.0) - 7.12 → 4.62 (adj -2.5)

================================================================================
TWO-STAGE REFINEMENT COMPLETE
================================================================================
```

## Testing

### UIApp Integration (No Changes Required!)
The implementation is fully backward compatible. UIApp continues to work without modifications:

```python
from race_prediction.predict_quinte import QuintePredictionEngine

# Create predictor (same as before)
predictor = QuintePredictionEngine(verbose=True)

# Run prediction (same as before)
result = predictor.run_prediction(race_date='2025-01-02')
```

### Expected Behavior

After running predictions via UIApp:

1. **Top 3 positions**: Should remain similar to previous predictions (model is good here)
2. **Positions 4-5**: Should show MORE longshots (odds 15-35) and FEWER mid-odds horses (10-18)
3. **Console output** (if verbose=True): Shows detailed two-stage refinement logs

### Validation Checkpoints

1. ✓ **No duplicate horses** in top 5
2. ✓ **Exactly 5 horses** selected for top 5
3. ✓ **Top 3 unchanged** from model predictions
4. ✓ **Odds-based adjustments** applied only to positions 4-5 selection

## Expected Impact

Based on failure analysis:
- **Position 4 accuracy**: Should improve from 38% → estimated 55-60%
- **Position 5 accuracy**: Should improve from 49% → estimated 60-65%
- **Longshot inclusion**: More horses with odds 15-35 in positions 4-5
- **Top 3 accuracy**: Should remain unchanged (already good at 74%, 69%, 51%)

## Rollback

If results are worse:
1. Comment out line 599 in `predict_quinte.py`:
   ```python
   # result_df = self._apply_two_stage_refinement(result_df)
   ```
2. System reverts to original behavior

## Next Steps

1. **Run predictions** on upcoming quinté races via UIApp
2. **Monitor positions 4-5 accuracy** over 20-30 races
3. **Compare** old vs new predictions:
   - Count longshots in positions 4-5
   - Track position 4-5 hit rate
4. **Fine-tune parameters** if needed:
   - Longshot odds range (currently 15-35)
   - Boost/penalize amounts (currently -2.5/+1.5)

## Files Modified

- [race_prediction/predict_quinte.py](../race_prediction/predict_quinte.py) - Main implementation
- [docs/TWO_STAGE_PREDICTION_IMPLEMENTATION.md](TWO_STAGE_PREDICTION_IMPLEMENTATION.md) - This document

---
**Implementation Date**: 2025-12-03
**Based On**: Failure analysis of 66 quinté races (see analysis results)
