# Comparing Standard vs Two-Stage Prediction Accuracy

## The Comparison Script

[scripts/compare_standard_vs_twostage.py](../scripts/compare_standard_vs_twostage.py) compares prediction accuracy between:

1. **Standard predictions** (base model without two-stage refinement)
2. **Two-stage predictions** (with positions 4-5 refinement)
3. **Actual race results**

## How It Works

The comparison is possible because the two-stage implementation preserves the original predictions:

```python
# In _apply_two_stage_refinement() at line 787:
race_df['original_predicted_position'] = race_df['predicted_position'].copy()
```

This allows us to compare:
- `original_predicted_position` = Standard predictions (BEFORE two-stage)
- `predicted_position` = Two-stage predictions (AFTER two-stage)
- Actual race results from database

## Usage

### Basic Usage (All Races with Results)

```bash
python3 scripts/compare_standard_vs_twostage.py
```

This will:
1. Load all quinté races that have actual results
2. Generate predictions with two-stage refinement
3. Compare standard vs two-stage vs actual
4. Show accuracy metrics for both approaches

### Specific Date

```bash
python3 scripts/compare_standard_vs_twostage.py --date 2025-01-02
```

## Output Example

```
================================================================================
COMPARING STANDARD vs TWO-STAGE PREDICTIONS
================================================================================

[1/4] Creating QuintePredictionEngine...
✓ Predictor created

[2/4] Loading quinté races...
✓ Loaded 66 quinté races

✓ Found 66 races with actual results

[3/4] Running predictions with two-stage refinement...

  Processing race R1-C4-20250102...
    ✓ Analyzed
  Processing race R1-C5-20250102...
    ✓ Analyzed
  ...

[4/4] Comparing results...

================================================================================
SUMMARY (66 races analyzed)
================================================================================

1. WINNER PREDICTION:
   Standard:     49/66 (74.2%)
   Two-stage:    49/66 (74.2%)
   → Same performance

2. POSITION 4 ACCURACY (exact position):
   Standard:     25/66 (37.9%)
   Two-stage:    38/66 (57.6%)
   → Two-stage BETTER by 13 races (+19.7%)  ✓

3. POSITION 5 ACCURACY (exact position):
   Standard:     32/66 (48.5%)
   Two-stage:    42/66 (63.6%)
   → Two-stage BETTER by 10 races (+15.2%)  ✓

4. POSITION 4 IN TOP 5 (any order):
   Standard:     54/66 (81.8%)
   Two-stage:    58/66 (87.9%)

5. POSITION 5 IN TOP 5 (any order):
   Standard:     51/66 (77.3%)
   Two-stage:    57/66 (86.4%)

6. TOP 5 ACCURACY (all 5 horses in any order):
   Standard:     68.2%
   Two-stage:    74.5%
   → Two-stage BETTER by 6.3%  ✓

================================================================================
DETAILED RACE-BY-RACE COMPARISON
================================================================================

Race: R1-C4-20250102
Actual:     1-5-3-7-9
Standard:   1-5-3-4-6
Two-stage:  1-5-3-7-9
Changes:    pos 4: #4 (odds 12.5) → #7 (odds 18.5), pos 5: #6 (odds 14.2) → #9 (odds 22.0)
Pos 4:      Standard=False, Two-stage=True  ✓
Pos 5:      Standard=False, Two-stage=True  ✓

Race: R1-C5-20250102
Actual:     2-6-4-11-13
Standard:   2-6-4-8-10
Two-stage:  2-6-4-11-13
Changes:    pos 4: #8 (odds 10.5) → #11 (odds 19.0), pos 5: #10 (odds 12.0) → #13 (odds 24.5)
Pos 4:      Standard=False, Two-stage=True  ✓
Pos 5:      Standard=False, Two-stage=True  ✓

...

================================================================================
CONCLUSION
================================================================================

✓ Two-stage prediction improves positions 4-5 accuracy by 23 correct positions across 66 races
  This is a 17.4% improvement in positions 4-5
```

## What the Metrics Mean

### 1. Winner Prediction
- Should remain similar (model is already good at this)
- Two-stage keeps top 3 unchanged, so winner accuracy shouldn't change

### 2. Position 4 Accuracy (Exact Position)
- **KEY METRIC**: This is what we're trying to improve
- Target: Improve from ~38% to 55-60%
- Shows if position 4 horse is correctly predicted at position 4

### 3. Position 5 Accuracy (Exact Position)
- **KEY METRIC**: This is what we're trying to improve
- Target: Improve from ~49% to 60-65%
- Shows if position 5 horse is correctly predicted at position 5

### 4. Position 4/5 in Top 5 (Any Order)
- Less strict metric - just needs to be in top 5
- Shows if we're including the right horses, even if position is off

### 5. Top 5 Accuracy
- Overall accuracy of selecting the correct 5 horses
- Shows if two-stage improves overall quinté selection

## What to Look For

### ✅ Success Indicators

1. **Position 4 accuracy improves** by 10-20%
2. **Position 5 accuracy improves** by 10-20%
3. **Winner accuracy stays the same** (no regression)
4. **Changes show**: Mid-odds → Longshots at positions 4-5

### ❌ Failure Indicators

1. Position 4-5 accuracy **decreases**
2. Winner accuracy **decreases** (regression in top 3)
3. Top 5 accuracy **decreases** significantly
4. No pattern in changes (random swaps)

## Interpreting Results

### Example: Good Result
```
Position 4: Standard=37.9% → Two-stage=57.6% (+19.7%)  ✓
Position 5: Standard=48.5% → Two-stage=63.6% (+15.2%)  ✓
Winner: Standard=74.2% → Two-stage=74.2% (same)  ✓
```
**Interpretation**: Two-stage is working! Positions 4-5 improved significantly without breaking top 3.

### Example: Bad Result
```
Position 4: Standard=37.9% → Two-stage=35.2% (-2.7%)  ✗
Position 5: Standard=48.5% → Two-stage=45.1% (-3.4%)  ✗
Winner: Standard=74.2% → Two-stage=68.2% (-6.0%)  ✗
```
**Interpretation**: Two-stage is making things worse. Need to disable or adjust parameters.

### Example: Neutral Result
```
Position 4: Standard=37.9% → Two-stage=38.5% (+0.6%)
Position 5: Standard=48.5% → Two-stage=49.1% (+0.6%)
```
**Interpretation**: No significant improvement yet. May need more races or parameter tuning.

## Testing Process

### Step 1: Run on Historical Data

```bash
# Compare on all historical races with results
python3 scripts/compare_standard_vs_twostage.py
```

### Step 2: Analyze Results

Look at the summary metrics and identify:
- Is position 4-5 accuracy improving?
- Is top 3 staying stable?
- What's the pattern in changes?

### Step 3: Review Race-by-Race Details

Check the detailed comparison:
- Which races improved?
- Which races got worse?
- Is there a pattern? (e.g., longshots helping in certain track types)

### Step 4: Decide

**If improvements are significant (>10%):**
- ✅ Keep two-stage enabled
- Monitor over next 20-30 races
- Consider parameter tuning

**If results are worse:**
- ❌ Disable two-stage (comment out line 599 in predict_quinte.py)
- Review failure patterns
- Adjust parameters and retest

**If results are neutral:**
- Test on more races (need larger sample)
- Consider adjusting parameters:
  - Longshot odds range (currently 15-35)
  - Boost amount (currently -2.5)
  - Penalize amount (currently +1.5)

## Parameter Tuning

If results are mixed, you can adjust parameters in [predict_quinte.py:820-850](../race_prediction/predict_quinte.py#L820-L850):

```python
# Line 820: Longshot odds range
mask_longshot = (remaining['cotedirect'] >= 15) & (remaining['cotedirect'] <= 35)

# Line 825: Longshot boost amount
remaining.loc[mask_longshot, 'position_adjustment'] -= 2.5

# Line 836: Mid-odds range
mask_mid_odds = (remaining['cotedirect'] >= 10) & (remaining['cotedirect'] <= 18)

# Line 843: Mid-odds penalty amount
remaining.loc[mask_penalize, 'position_adjustment'] += 1.5
```

Try different values and re-run comparison to see impact.

## Continuous Monitoring

After enabling two-stage, monitor accuracy over time:

```bash
# Weekly comparison
python3 scripts/compare_standard_vs_twostage.py --date 2025-01-02  # Week 1
python3 scripts/compare_standard_vs_twostage.py --date 2025-01-09  # Week 2
python3 scripts/compare_standard_vs_twostage.py --date 2025-01-16  # Week 3
```

Track trends:
- Is position 4-5 accuracy consistently better?
- Are there certain conditions where it works better? (track type, field size, etc.)

---

**Last Updated**: 2025-12-03
**Status**: Ready for testing
