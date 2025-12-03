# Quick Start: Quinté Failure Analysis

## TL;DR

```bash
# Install dependencies (if needed)
pip install matplotlib seaborn scipy

# Run the analysis
python3 scripts/analyze_quinte_failures.py

# View results
ls analysis_output/
cat analysis_output/quinte_failure_analysis_*.txt
```

## What You'll Get

### 📊 4 Key Visualizations
1. **Positional Drift Heatmap** - Where predicted positions actually finish
2. **Success Rate Chart** - Which positions fail most often
3. **Odds Analysis** - Are we biasing toward favorites/outsiders?
4. **Position 4-5 Deep Dive** - Detailed analysis of problem positions

### 📄 Comprehensive Report
Text file with:
- Positional drift statistics
- Missed quinté patterns
- Favorite/outsider bias analysis
- Position 4-5 specific failures
- **Actionable recommendations**

## Expected Results (Based on Your Data)

Given your current performance:
- ✅ Winner accuracy: 32.8% (20/66)
- ⚠️  Bonus 3: 18.2% (12/66)
- ❌ Bonus 4: 6.1% (4/66)
- ❌ Quinté Désordre: 1.5% (1/66)

**The analysis will show:**

### Problem: Position 4-5 Drift
```
PREDICTED POSITION 4
  Success rate: ~60-70% (should be 80%+)
  Average drift: +2 to +3 positions
  → Often finish 6th-8th instead of 4th
```

### Problem: Missing Mid-Pack Horses
```
ACTUAL POSITION 4-5 horses
  We predicted them at rank: 6-8 (not in top 5)
  → We're systematically underestimating certain horses
```

### Problem: Possible Favorite Bias
```
Favorites (odds ≤5): Average drift +1.5
  → We rank them too high
Outsiders (odds >15): Average drift -0.8
  → We rank them too low
```

## Interpreting Results

### Drift Values
- **Drift = 0**: Perfect prediction
- **Drift = +2**: Horse finished 2 positions worse than predicted
- **Drift = -2**: Horse finished 2 positions better than predicted

### Success Rate (for predicted positions)
- **>80%**: Excellent - position is reliable
- **70-80%**: Good - minor calibration needed
- **60-70%**: Fair - significant improvement needed
- **<60%**: Poor - major issue, investigate immediately

### Red Flags to Look For

1. **Position 4-5 success rate <70%**
   - → Need position-specific calibration
   - → Consider widening prediction margin for these positions

2. **False positives concentrated at ranks 4-5**
   - → Model is overconfident about these positions
   - → May need uncertainty estimation

3. **False negatives (missed horses) ranked 6-10**
   - → Model has blind spot for certain horse types
   - → Need additional features or feature engineering

4. **Favorite bias (positive drift for low odds)**
   - → Reduce weight on recent form
   - → Increase weight on race-specific factors

5. **Outsider blind spot (negative drift for high odds)**
   - → Add "upset potential" features
   - → Consider form trends, not just absolute form

## Common Patterns & Solutions

### Pattern 1: "Good at Top 3, Bad at 4-5"
**Symptom**: Positions 1-3 have >75% success, positions 4-5 have <65%

**Cause**: Model confident about clear winners/placers, uncertain about mid-pack

**Solution**:
- Train separate model for positions 4-10
- Add features capturing mid-pack battles (pace, position changes)
- Use wider confidence intervals for positions 4-5

### Pattern 2: "Favorite Overestimation"
**Symptom**: Favorites (low odds) have positive drift (+1 to +2)

**Cause**: Over-weighting recent strong performances

**Solution**:
- Reduce weight on recent form features
- Add regression to mean features
- Consider class/competition level more heavily

### Pattern 3: "Missing Outsiders in Top 5"
**Symptom**: Horses with odds >15 often finish top 5 but predicted 8-12

**Cause**: Model doesn't capture upset potential

**Solution**:
- Add trend features (improving form)
- Add jockey/trainer hot streaks
- Consider pace/trip factors (suited to today's race)

### Pattern 4: "Winner Correct, Rest Wrong"
**Symptom**: 30%+ of races have correct winner but missed quinté

**Cause**: Winner prediction uses different signals than placer prediction

**Solution**:
- Consider two-stage model: winner + placers
- Add position-specific features
- Use different feature weights for each position

## Next Steps After Analysis

1. **Identify the primary pattern** (see above)

2. **Implement targeted fix**:
   ```python
   # Example: Position-specific calibration
   if predicted_rank in [4, 5]:
       predicted_position += calibration_offset  # Add uncertainty margin
   ```

3. **Re-run predictions** on same test set

4. **Compare results**:
   ```bash
   # Before
   python3 scripts/analyze_quinte_failures.py --output-dir analysis_before

   # After fix
   python3 scripts/analyze_quinte_failures.py --output-dir analysis_after

   # Compare
   diff analysis_before/quinte_failure_analysis_*.txt \
        analysis_after/quinte_failure_analysis_*.txt
   ```

5. **Iterate** until:
   - Positions 4-5 success rate >75%
   - Bonus 4 success >12%
   - Quinté Désordre success >4%

## Key Metrics to Track

Monitor these after each improvement:

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Position 4 success rate | ? | 80% | 🔴 HIGH |
| Position 5 success rate | ? | 80% | 🔴 HIGH |
| Avg drift position 4 | ? | 0±1 | 🔴 HIGH |
| Avg drift position 5 | ? | 0±1 | 🔴 HIGH |
| Favorite bias (drift) | ? | 0±0.5 | 🟡 MED |
| Bonus 4 success | 6.1% | 12% | 🔴 HIGH |
| Quinté Désordre | 1.5% | 4% | 🔴 HIGH |

## Validation

After implementing fixes, you should see:

**Before Fix:**
```
Position 4 success rate: 62%
Position 5 success rate: 58%
Bonus 4: 6.1%
Quinté Désordre: 1.5%
```

**After Fix (Target):**
```
Position 4 success rate: 78%+
Position 5 success rate: 76%+
Bonus 4: 12%+
Quinté Désordre: 4%+
```

**Improvement needed: ~25-30% increase in positions 4-5 accuracy**

## Questions?

**Q: Why focus on positions 4-5?**
A: Bonus 4 requires 4/5 correct. Quinté Désordre requires 5/5. If positions 4-5 each have 65% success, combined probability is only ~42%. If they reach 80%, combined is ~64% - a 50% improvement.

**Q: Can I fix just position 5?**
A: No, both 4 and 5 must improve. A chain is only as strong as its weakest link.

**Q: How much data do I need?**
A: Minimum 30 races (recommended 50+) for reliable patterns. Current 66 races is good.

**Q: Should I retrain the model?**
A: Not necessarily. First try:
1. Calibration adjustments
2. Feature engineering
3. Ensemble methods
4. Position-specific models

Only retrain if those don't work.

## Commands Cheatsheet

```bash
# Basic run
python3 scripts/analyze_quinte_failures.py

# With verbose output
python3 scripts/analyze_quinte_failures.py --verbose

# Specific date
python3 scripts/analyze_quinte_failures.py --date 2025-11-03

# Custom output
python3 scripts/analyze_quinte_failures.py --output-dir my_results

# View results quickly
tail -100 analysis_output/quinte_failure_analysis_*.txt

# Check predictions in DB
sqlite3 data/hippique2.db "SELECT COUNT(*), COUNT(DISTINCT race_id) FROM quinte_predictions;"

# Check results in DB
sqlite3 data/hippique2.db "SELECT COUNT(*) FROM daily_race WHERE actual_results IS NOT NULL;"
```

## Success Criteria

You've succeeded when:
- ✅ Position 4-5 success rates are both >75%
- ✅ Average drift for positions 4-5 is between -0.5 and +0.5
- ✅ Bonus 4 success rate >10%
- ✅ Quinté Désordre success rate >3%
- ✅ No significant favorite/outsider bias (drift <|1.0|)

Then you'll have a quinté prediction system that's not just good at winners, but **excellent at full top-5 predictions**! 🎯
