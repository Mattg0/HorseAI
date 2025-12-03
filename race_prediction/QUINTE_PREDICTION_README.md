# Quinté Prediction and Evaluation

This directory contains scripts for predicting Quinté+ races and evaluating prediction accuracy.

## Overview

The quinté prediction system consists of three main components:

1. **Training** (`model_training/historical/train_quinte_model.py`) - Train specialized quinté models
2. **Prediction** (`race_prediction/predict_quinte.py`) - Generate predictions for daily quinté races
3. **Evaluation** (`race_prediction/compare_quinte_results.py`) - Compare predictions with actual results

## Workflow

### 1. Train Quinté Models (One-time)

Train specialized Random Forest and TabNet models on historical quinté data:

```bash
python model_training/historical/train_quinte_model.py
```

Models are saved to `models/YYYY-MM-DD/` with the quinté suffix:
- `2years_HHMMSS_quinte_rf/` - Random Forest model
- `2years_HHMMSS_quinte_tabnet/` - TabNet model

**Note:** Training uses batch loading optimization and completes in minutes.

### 2. Generate Predictions

Predict quinté races from the `daily_race` table where `quinte=1`:

```bash
# Predict today's quinté races
python race_prediction/predict_quinte.py

# Predict specific date
python race_prediction/predict_quinte.py --date 2025-10-10

# Custom output directory
python race_prediction/predict_quinte.py --date 2025-10-10 --output-dir my_predictions

# Verbose mode
python race_prediction/predict_quinte.py --date 2025-10-10 --verbose
```

**Output files:**
- `predictions/quinte_predictions_YYYY-MM-DD_HHMMSS.csv` - CSV format
- `predictions/quinte_predictions_YYYY-MM-DD_HHMMSS.json` - JSON format (grouped by race)

**CSV columns:**
- `comp` - Race identifier
- `jour` - Race date
- `hippo` - Racecourse
- `numero` - Horse number
- `nom` - Horse name
- `jockey` - Jockey name
- `entraineur` - Trainer name
- `cotedirect` - Direct odds
- `predicted_position` - Ensemble prediction (60% TabNet + 40% RF)
- `predicted_position_tabnet` - TabNet prediction
- `predicted_position_rf` - Random Forest prediction
- `predicted_rank` - Rank within race (1 = predicted winner)

### 3. Evaluate Predictions

Compare predictions with actual results from the `quinte_results` table:

```bash
# Compare predictions with actual results
python race_prediction/compare_quinte_results.py --predictions predictions/quinte_predictions_2025-10-10_102030.csv

# Specify date for results lookup
python race_prediction/compare_quinte_results.py --predictions predictions/file.csv --actual-date 2025-10-10

# Custom output directory
python race_prediction/compare_quinte_results.py --predictions predictions/file.csv --output-dir results

# Verbose mode
python race_prediction/compare_quinte_results.py --predictions predictions/file.csv --verbose
```

**Output files:**
- `predictions/quinte_comparison_HHMMSS.csv` - Detailed comparison with errors
- `predictions/quinte_metrics_HHMMSS.json` - Performance metrics

**Metrics calculated:**
- **MAE** (Mean Absolute Error) - Average position error
- **RMSE** (Root Mean Squared Error) - Penalizes large errors more
- **Exact Accuracy** - % of horses with exact position predicted
- **Winner Accuracy** - % of race winners correctly predicted
- **Top 5 Accuracy** - % of quinté placings (top 5) correctly predicted
- **Rank Correlation** - Spearman correlation per race

**Example output:**
```
============================================================
QUINTÉ PREDICTION EVALUATION
============================================================

Overall Performance:
  Total Races: 5
  Total Horses: 85
  Mean Absolute Error: 2.847 positions
  RMSE: 3.542 positions

Accuracy Metrics:
  Exact Position: 18.8% (16/85)
  Winner Prediction: 40.0% (2/5)
  Top 5 (Quinté): 65.2% (30/46)

Per-Race Performance:
------------------------------------------------------------
Race            Track           Horses   MAE      Corr     Winner
------------------------------------------------------------
R20251010C1     Longchamp       18       2.34     0.72     ✓
R20251010C2     Vincennes       16       3.12     0.68     ✗
...
```

## Prediction Pipeline

### Data Flow

```
daily_race (quinte=1)
    ↓
Expand participants JSON
    ↓
Calculate standard features (FeatureCalculator)
    ↓
Batch-load quinté historical data
    ↓
Calculate quinté-specific features (QuinteFeatureCalculator)
    ↓
Apply quinté models (RF + TabNet)
    ↓
Generate ensemble predictions
    ↓
Save to CSV/JSON
```

### Feature Engineering

**Standard Features** (from `FeatureCalculator`):
- Horse career statistics
- Jockey/trainer performance
- Track conditions
- Musique pattern features

**Quinté-Specific Features** (from `QuinteFeatureCalculator`):
- `quinte_career_starts` - Total quinté races
- `quinte_win_rate` - Win % in quinté races
- `quinte_top5_rate` - Top 5 % in quinté
- `avg_quinte_position` - Average finish position
- `days_since_last_quinte` - Recency in quinté
- `quinte_handicap_specialist` - Better in handicaps
- `quinte_conditions_specialist` - Better in conditions races
- `quinte_large_field_ability` - Performance in 15+ fields
- `quinte_track_condition_fit` - Performance in current conditions
- `is_handicap_quinte` - Race type
- `handicap_division` - Handicap category (0/1/2)
- `purse_level_category` - Prize level (Low/Med/High)
- `field_size_category` - Number of runners (14-15/16-17/18+)
- Track conditions (PH/DUR/PS/PSF)
- Weather conditions (Clear/Rain/Cloudy)
- Post position biases

### Model Ensemble

Predictions use a weighted ensemble:
- **60% TabNet** - Better at capturing complex patterns
- **40% Random Forest** - Robust baseline predictions

Individual model predictions are also saved for analysis.

## Data Requirements

### Training Data
- `historical_quinte` table - Historical quinté races
- `quinte_results` table - Historical quinté results

### Prediction Data
- `daily_race` table - Daily races where `quinte=1`

### Evaluation Data
- Predictions file (CSV or JSON)
- `quinte_results` table - Actual results

## Performance

### Training
- **~2,000 races** in training dataset
- **~31,000 horses** with features
- **Training time:** ~5-10 minutes (with batch optimization)
- **Batch loading:** 1.77 seconds for all historical data

### Prediction
- **Feature calculation:** ~1-2 seconds per race
- **Model inference:** Nearly instantaneous
- **Total time:** ~10-20 seconds for typical daily quinté races

## Tips

1. **Daily workflow:**
   ```bash
   # Morning: Generate predictions for today
   python race_prediction/predict_quinte.py

   # Evening: Compare with actual results
   python race_prediction/compare_quinte_results.py --predictions predictions/latest.csv
   ```

2. **Backtesting:**
   ```bash
   # Test on historical dates
   for date in 2025-09-01 2025-09-02 2025-09-03; do
       python race_prediction/predict_quinte.py --date $date
       python race_prediction/compare_quinte_results.py --predictions predictions/quinte_predictions_${date}_*.csv
   done
   ```

3. **Model retraining:**
   - Retrain models periodically (e.g., monthly) to incorporate new data
   - Compare old vs new model performance before deploying

## Troubleshooting

**No quinté races found:**
- Check that `daily_race` table has races with `quinte=1`
- Verify the date format is YYYY-MM-DD

**No results for comparison:**
- Ensure `quinte_results` table is populated
- Check that race identifiers (`comp`) match between predictions and results

**Missing features:**
- Verify `historical_quinte` table has sufficient historical data
- Check that feature columns exist in the model's expected features

**Poor prediction accuracy:**
- Consider retraining with more recent data
- Adjust ensemble weights based on validation performance
- Review feature importance to identify key predictive factors

## Next Steps

1. **UI Integration:** Add quinté prediction to the Streamlit UI
2. **Automated Pipeline:** Schedule daily predictions automatically
3. **Live Monitoring:** Track prediction accuracy over time
4. **Model Improvements:** Experiment with feature engineering and hyperparameters
