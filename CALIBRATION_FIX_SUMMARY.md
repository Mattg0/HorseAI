# Calibration System Fix - Summary

**Date**: 2025-11-20
**Issue**: All predictions showing as 1.0 in UI due to broken calibration

## Root Cause

The calibration system was trained on **garbage data** from older races in the database:

1. **Older races (before ~Oct 2025)** had wrong values in `predicted_position_uncalibrated`:
   - Values ranged from 19-70 (WAY TOO HIGH)
   - Should be in 5-15 range for normal predictions
   - These were likely stored incorrectly during an earlier prediction run

2. **Bias detector calculated massive errors**:
   - With predicted_position mean=19.35 vs actual_position mean=6.81
   - Calculated error = 12.52 positions over-prediction
   - Generated huge negative corrections (-8 to -22)

3. **When applied, these corrections collapsed all predictions to 1.0**:
   - Example: predicted_position_uncalibrated = 37.4
   - After correction: 37.4 + (-36) = 1.0

## Fixes Applied

### 1. Validation Filter (calibrate_models.py lines 210-234)
Added filter to **reject races with garbage prediction values** before calibration:
```python
# Flag races where predicted positions are suspiciously high
bad_races = race_stats[
    (race_stats['mean'] > 15) |  # Mean way too high
    (race_stats['max'] > 35)     # Max way too high
]['race_id'].tolist()
```

**Result**:
- Filtered out 1,435 races with garbage data
- Remaining 986 races have reasonable predictions (mean=8.07)

### 2. Calibration Rejection Logic (calibrate_models.py lines 296-315)
The system now correctly **rejects calibrations that hurt performance**:
- General model: Rejected (-2% improvement)
- Quinté model: Rejected (-108% improvement)

Both models are reasonably well-calibrated without bias corrections.

### 3. Broken Calibration Files Removed
- `general_calibration.json` → `general_calibration.json.rejected`
- `quinte_calibration.json` → `quinte_calibration.json.rejected`

## Current Status

✅ **Calibration system fixed** - will no longer generate broken corrections
✅ **No active calibration files** - models use uncalibrated predictions
⚠️ **Database contains broken predictions** - need to regenerate

## Required Action: Regenerate Predictions

The existing predictions in the database are corrupted with `predicted_position = 1.0` values. You need to **re-run predictions** for recent races:

### Option 1: Re-predict Recent Races
```bash
# Delete broken predictions from database
sqlite3 data/hippique2.db "UPDATE daily_race SET prediction_results = NULL WHERE jour >= '2025-10-01';"

# Re-run predictions for recent date range
python race_prediction/race_predict.py --start-date 2025-10-01 --end-date 2025-11-20
```

### Option 2: Quick Fix for Display (SQL Update)
If predictions can't be regenerated, you can patch the display by using `predicted_position_uncalibrated` or rank-based reconstruction.

**However, note**: Some recent races (Nov 20) have broken `predicted_position_uncalibrated` values (25-37 range), so full regeneration is recommended.

## Validation Results

After fixes, calibration training data now shows:
- **predicted_position**: mean=8.07, max=34.63 ✓
- **actual_position**: mean=6.17, max=18.00 ✓
- **Error**: mean=1.89 ✓ (reasonable)
- **Corrections**: -0.5 to -7.7 ✓ (reasonable but still rejected)

## Files Modified

1. `/Users/mattg0/Docs/HorseAIv2/scripts/calibrate_models.py`
   - Lines 210-234: Added garbage data filtering
   - Lines 277-295: Added debug output
   - Lines 356-364: Fixed return value consistency (3-tuple)

2. Calibration files moved:
   - `models/calibration/general_calibration.json.broken_20251120`
   - `models/calibration/general_calibration.json.rejected`
   - `models/calibration/quinte_calibration.json.rejected`

## Prevention

The validation filter will now automatically reject:
- Races where mean predicted_position > 15
- Races where max predicted_position > 35

This prevents garbage data from poisoning future calibrations.

## Next Steps

1. **Regenerate predictions** for races from 2025-10-01 onwards
2. **Verify UI** shows correct predictions after regeneration
3. **Monitor calibration health** periodically with: `python scripts/calibrate_models.py --check`

The calibration system is now robust and will correctly reject harmful calibrations.
