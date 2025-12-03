# Quinté Prediction Failure Analysis

## Overview

This script provides **deep diagnostic analysis** of why the HorseAI quinté prediction system has:
- ✅ **Excellent winner accuracy** (32.8%)
- ❌ **Poor Quinté Désordre success** (1.5%)
- ❌ **Low Bonus 4 success** (6.1%)

## Problem Statement

The model can identify winners but fails to predict the complete top 5, specifically struggling with **positions 4-5**. This script identifies exactly why.

## What It Analyzes

### 1. **Positional Drift Analysis**
For each predicted position (1-5):
- Where do they actually finish?
- Average drift (positive = finished worse than predicted)
- Success rate (% that finish in top 5)
- Distribution of actual positions

**Example Output:**
```
PREDICTED POSITION 4
─────────────────────────────────────────────────────────────────────────────
  Predictions analyzed: 66
  Success rate (finished top 5): 62.1%  ← Should be 80%+
  Exact position accuracy: 15.2%
  Average drift: +2.8 positions (+ = finished worse)

  Most common actual positions:
    Position  6: 12 times (18.2%)  ← Major issue!
    Position  7:  8 times (12.1%)
    Position  5:  7 times (10.6%)
```

### 2. **Missed Quinté Analysis**
Focuses on races where:
- ✅ Winner was predicted correctly
- ❌ Quinté (top 5) was missed

Identifies:
- **False Positives**: Horses predicted top 5 but finished 6+
- **False Negatives**: Horses that finished top 5 but we predicted 6+
- Which predicted positions fail most often
- Odds patterns (are we over-predicting favorites?)

**Example Output:**
```
RACES WITH WINNER CORRECT BUT QUINTÉ MISSED
─────────────────────────────────────────────────────────────────────────────
  Total races analyzed: 66
  Winner correct, quinté missed: 20 (30.3%)  ← Key failure mode

  FALSE POSITIVES (predicted top 5, finished 6+):
    Total: 42 horses
    Predicted rank distribution:
      Rank 4: 18 (42.9%)  ← Position 4 fails most!
      Rank 5: 16 (38.1%)  ← Position 5 second most
      Rank 3:  8 (19.0%)
    Average odds: 15.2 (median: 12.5)
    Average actual position: 7.8
```

### 3. **Favorite vs Outsider Analysis**
Analyzes prediction accuracy by odds categories:
- **Favorites** (odds ≤ 5)
- **Mid-range** (odds 5-15)
- **Outsiders** (odds > 15)

Identifies bias:
- Do we over-predict favorites?
- Do we miss outsiders that place well?

**Example Output:**
```
Favorite (≤5):
  Horses analyzed: 142
  Avg predicted rank: 3.2
  Avg actual position: 4.8
  Avg drift: +1.6 (+ = we overestimated)  ← Overestimating favorites!
  Precision: 68.2% (of predicted top 5, % actually top 5)
  Recall: 71.4% (of actual top 5, % we predicted)

Outsider (>15):
  Avg drift: -0.8 (- = we underestimated)  ← Missing outsiders!
```

### 4. **Position 4-5 Deep Dive**
Specific analysis of the problematic positions:
- Success rate of predicted positions 4-5
- Where do predicted 4-5 actually finish?
- What ranks did we assign to horses that actually finished 4-5?
- Odds patterns

**Example Output:**
```
PREDICTED POSITION 4-5 ANALYSIS
─────────────────────────────────────────────────────────────────────────────
  Total predictions at positions 4-5: 132
  Actually finished top 5: 64.4%  ← Should be 80%+
  Actually finished 4-5: 24.2%   ← Low precision

  Actual finish positions:
    Position  4: 18 (13.6%)
    Position  5: 14 (10.6%)
    Position  6: 24 (18.2%)  ← Most common!
    Position  7: 19 (14.4%)
    Position  8: 12 ( 9.1%)

ACTUAL POSITION 4-5 ANALYSIS
─────────────────────────────────────────────────────────────────────────────
  Total horses that finished 4-5: 132

  Our predicted ranks for these horses:
    Rank  4: 18 (13.6%) ✓
    Rank  5: 14 (10.6%) ✓
    Rank  6: 22 (16.7%) ✗  ← We ranked them 6th!
    Rank  7: 18 (13.6%) ✗
    Rank  3: 15 (11.4%) ✗  ← Sometimes ranked too high

  Correctly predicted at rank 4-5: 32/132 (24.2%)  ← Very low!
  Predicted in top 5: 85/132 (64.4%)
```

## Visualizations Generated

The script creates 4 high-quality PNG visualizations:

### 1. `positional_drift_heatmap.png`
Confusion matrix showing predicted rank vs actual position
- Diagonal = exact predictions (highlighted in blue)
- Off-diagonal = drift
- Heatmap colors: Green (good) → Yellow → Red (bad)

### 2. `success_rate_by_position.png`
Bar chart showing success rate (finished top 5) for each predicted position
- Target line at 80% (ideal success rate)
- **Positions 4-5 should be near 80% but likely much lower**

### 3. `odds_distribution_comparison.png`
Histogram and box plot comparing:
- Odds of horses we predicted top 5
- Odds of horses that actually finished top 5
- Identifies if we're biased toward favorites/outsiders

### 4. `position_4_5_detailed_analysis.png`
4-panel visualization:
- Where predicted 4-5 actually finish
- What we predicted for actual 4-5
- Drift distribution for predicted 4-5
- Success rate by position (zoomed to 1-10)

## Usage

### Prerequisites

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- sqlite3 (built-in)

### Basic Usage

Analyze all predictions in the database:
```bash
python3 scripts/analyze_quinte_failures.py
```

### Advanced Usage

Analyze a specific date:
```bash
python3 scripts/analyze_quinte_failures.py --date 2025-11-03
```

Require minimum number of races:
```bash
python3 scripts/analyze_quinte_failures.py --min-races 50
```

Custom output directory:
```bash
python3 scripts/analyze_quinte_failures.py --output-dir my_analysis
```

Verbose output:
```bash
python3 scripts/analyze_quinte_failures.py --verbose
```

## Output Files

All outputs are saved to `analysis_output/` (or custom directory):

1. **`quinte_failure_analysis_YYYYMMDD_HHMMSS.txt`**
   - Comprehensive text report
   - All analysis sections
   - Key findings and recommendations

2. **`positional_drift_heatmap.png`**
   - Visual confusion matrix (predicted vs actual)

3. **`success_rate_by_position.png`**
   - Bar chart of success rates by position

4. **`odds_distribution_comparison.png`**
   - Odds analysis for predicted vs actual top 5

5. **`position_4_5_detailed_analysis.png`**
   - 4-panel deep dive into positions 4-5

## Expected Insights

Based on the problem description, the analysis will likely reveal:

### ✅ What's Working
- Position 1 (winner) prediction is excellent (32.8% accuracy)
- Positions 1-3 likely have good success rates (>70%)

### ❌ What's Failing
- **Positions 4-5 have low success rates** (<65%)
- Horses predicted at positions 4-5 often finish 6-8
- Horses that actually finish 4-5 are ranked 6-8 by the model
- Possible favorite bias (over-predicting low-odds horses)
- Possible outsider blind spot (missing medium-odds horses)

### 🔧 Recommended Actions

The script will suggest specific improvements based on the patterns found:

1. **Position-Specific Calibration**
   - Positions 4-5 may need separate calibration
   - Consider increasing uncertainty for these positions

2. **Feature Engineering**
   - Add features that capture mid-pack dynamics
   - Consider race pace, position battles, etc.

3. **Model Architecture**
   - Consider ensemble approach: winner model + placer model
   - Position-specific models may improve accuracy

4. **Odds Adjustment**
   - If favorite bias detected, reduce weight on recent form
   - If outsider blind spot, add upset potential features

5. **Competitive Adjustment**
   - Current competitive adjustment may not work well for positions 4-5
   - Consider position-specific adjustments

## Technical Details

### Data Source
- **Database**: `data/hippique2.db` (configured in `config.yaml`)
- **Predictions Table**: `quinte_predictions`
- **Results Table**: `daily_race.actual_results`
- **Metadata Table**: `partant` (for horse names, odds, jockey)

### Key Metrics

**Positional Drift**
```
drift = actual_position - predicted_rank
```
- Positive drift = finished worse than predicted
- Negative drift = finished better than predicted

**Success Rate**
```
success_rate = % of predicted position X that finish in top 5
```
- Target: 80%+ for positions 1-5
- Current issue: Positions 4-5 likely <70%

**Precision (Top 5)**
```
precision = correct_top5 / predicted_top5
```
- Of horses we predicted top 5, how many actually finished top 5?

**Recall (Top 5)**
```
recall = correct_top5 / actual_top5
```
- Of horses that finished top 5, how many did we predict?

## Troubleshooting

### No data found
```
⚠ No predictions with results found
```
**Solution**: Run predictions first using `predict_quinte.py`

### Insufficient races
```
⚠ Warning: Only 15 races found (minimum recommended: 30)
```
**Solution**:
- Run more predictions
- Use `--min-races 10` to lower threshold
- Combine multiple prediction dates

### Missing dependencies
```
ModuleNotFoundError: No module named 'matplotlib'
```
**Solution**:
```bash
pip install -r requirements.txt
```

## Performance

- **Runtime**: ~10-30 seconds for 66 races
- **Memory**: ~100MB peak usage
- **Output size**: ~2MB (4 PNG images + text report)

## Integration

This analysis script complements:
- [`compare_quinte_results.py`](../race_prediction/compare_quinte_results.py) - High-level metrics
- **This script** - Deep diagnostic analysis of failures

Use together:
1. Run `compare_quinte_results.py` to get overall accuracy metrics
2. Run `analyze_quinte_failures.py` to diagnose why failures occur
3. Implement recommended improvements
4. Re-run both scripts to verify improvements

## Example Workflow

```bash
# 1. Make predictions
python3 race_prediction/predict_quinte.py --date 2025-11-03

# 2. Check overall performance
python3 race_prediction/compare_quinte_results.py --date 2025-11-03

# 3. Diagnose failures
python3 scripts/analyze_quinte_failures.py --date 2025-11-03 --verbose

# 4. Review outputs
ls -lh analysis_output/
cat analysis_output/quinte_failure_analysis_*.txt
open analysis_output/*.png  # View visualizations

# 5. Implement fixes based on recommendations

# 6. Re-test
python3 scripts/analyze_quinte_failures.py --date 2025-11-10
```

## Questions?

If you encounter issues or have questions:
1. Check the error message in the console output
2. Review the Troubleshooting section above
3. Verify database has prediction data: `sqlite3 data/hippique2.db "SELECT COUNT(*) FROM quinte_predictions;"`
4. Check config.yaml has correct database path

## License

Part of the HorseAI project.
