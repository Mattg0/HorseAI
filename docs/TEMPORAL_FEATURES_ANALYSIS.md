# Temporal Features Analysis for General Model

**Date**: October 21, 2025
**Question**: Is it required to implement the temporal feature approach for the general model?

## Executive Summary

**Answer: YES, temporal features ARE REQUIRED for the general model** to prevent data leakage. The general model currently uses several career statistics features that suffer from the same data leakage problem that was fixed in the quinté models.

## Current General Model Features (TabNet - 57 features)

### ⚠️ **CRITICAL: Features with Data Leakage** (4 features)

These features currently include data from the current race, which creates data leakage:

1. **`victoirescheval`** - Total career wins (includes current race if won)
2. **`placescheval`** - Total career places (includes current race if placed)
3. **`coursescheval`** - Total career races (includes current race)
4. **`gainsCarriere`** - Not directly listed but likely used in calculations

### 📊 **Derived Features That May Have Leakage** (3 features)

These are calculated from the leaking features above:

5. **`ratio_victoires`** - Calculated from victoirescheval/coursescheval
6. **`ratio_places`** - Calculated from placescheval/coursescheval
7. **`gains_par_course`** - Calculated from gainsCarriere/coursescheval

### ✅ **Phase 1 Features (Already Temporal)** (11 features)

These Phase 1 features are calculated using temporal logic and DO NOT leak:

- `career_strike_rate` - Uses historical data only
- `earnings_per_race` - Calculated temporally
- `earnings_trend` - Temporal calculation
- `last_race_position_normalized` - Last race data
- `last_race_odds_normalized` - Last race data
- `last_race_field_size_factor` - Last race data
- `distance_consistency` - Historical consistency
- `vha_normalized` - Normalized rating
- `claiming_tax_trend` - Trend calculation
- `class_stability` - Historical stability
- `distance_comfort` - Historical distance performance

### 🎯 **Other Non-Leaking Features** (39 features)

The remaining features are either:
- Market data (odds, probabilities)
- Static data (age, distance, temperature, handicap)
- Race context (trainer changes, equipment changes, class drops)
- Jockey/Horse performance metrics at specific venues

## Complete Feature List by Category

### Market & Odds (2)
- `cotedirect` - Direct odds
- `coteprob` - Probability from odds

### Race Context (6)
- `handicapDistance` - Handicap distance
- `recence` - Days since last race
- `dist` - Race distance
- `temperature` - Weather temperature
- `age` - Horse age
- `gainsAnneeEnCours` - Earnings this year

### ⚠️ Career Statistics (WITH LEAKAGE) (7)
- `victoirescheval` ❌ LEAKS
- `placescheval` ❌ LEAKS
- `coursescheval` ❌ LEAKS
- `ratio_victoires` ❌ DERIVED LEAKAGE
- `ratio_places` ❌ DERIVED LEAKAGE
- `gains_par_course` ❌ DERIVED LEAKAGE
- `efficacite_couple` ❌ LIKELY LEAKS

### Jockey/Trainer Stats (10)
- `nbVictCouple` - Jockey wins with horse
- `TxVictCouple` - Jockey/horse win rate
- `pourcVictChevalHippo` - Horse win % at track
- `pourcPlaceChevalHippo` - Horse place % at track
- `perf_cheval_hippo` - Horse performance at track
- `perf_jockey_hippo` - Jockey performance at track
- `joc_global_avg_pos` - Jockey global average position
- `joc_global_recent_perf` - Jockey recent performance
- `joc_global_consistency` - Jockey consistency
- `joc_global_pct_top3` - Jockey top 3 percentage

### Equipment Changes (5)
- `blinkers_first_time` - First time with blinkers
- `blinkers_high_impact_change` - Significant blinker change
- `barefoot_to_shod` - Shoeing change
- `major_shoeing_change` - Major shoeing change
- `equipment_momentum_score` - Equipment change impact

### Advanced Metrics (16)
- `equipment_optimization_score`
- `che_global_avg_pos` - Horse global average
- `che_global_recent_perf` - Horse recent performance
- `che_global_consistency` - Horse consistency
- `che_global_pct_top3` - Horse top 3 percentage
- `che_weighted_avg_pos` - Weighted horse average
- `che_weighted_recent_perf` - Weighted recent performance
- `joc_weighted_avg_pos` - Weighted jockey average
- `joc_weighted_recent_perf` - Weighted jockey recent
- `class_drop_pct` - Class change percentage
- `purse_ratio` - Prize money ratio
- `moving_up_in_class` - Class elevation indicator
- `speed_figure_proxy` - Speed proxy metric
- `market_confidence_shift` - Market movement
- `trainer_change` - Trainer change indicator
- `recence_x_class_drop` - Interaction feature

### ✅ Phase 1 Temporal Features (11)
- `career_strike_rate` ✓
- `earnings_per_race` ✓
- `earnings_trend` ✓
- `last_race_position_normalized` ✓
- `last_race_odds_normalized` ✓
- `last_race_field_size_factor` ✓
- `distance_consistency` ✓
- `vha_normalized` ✓
- `claiming_tax_trend` ✓
- `class_stability` ✓
- `distance_comfort` ✓

## Impact Analysis

### Leakage Severity: **HIGH**

The 7 features with data leakage represent **12.3%** of all features (7/57), but their impact is significant because:

1. **They are core career statistics** - These features have high predictive power when they leak
2. **They directly correlate with race outcome** - More wins/places = better current performance indicator
3. **They create unrealistic training accuracy** - The model learns to use future information
4. **They fail in production** - When predicting actual future races, these features don't have the leaked information

### Expected Performance Impact

Based on the quinté model experience:

- **Training accuracy may drop slightly** (2-5%) when leakage is fixed
- **Real-world prediction accuracy will IMPROVE** significantly
- **Model reliability increases** - Predictions become trustworthy

## Implementation Required

### 1. Apply Temporal Feature Calculator

The same `TemporalFeatureCalculator` created for quinté models should be applied to the general model pipeline:

**File**: `core/calculators/temporal_feature_calculator.py`

**Method**: `batch_calculate_all_horses(df)` - Calculates career stats using ONLY historical data before race date

### 2. Update Training Pipeline

**File**: `model_training/historical/train_race_model.py`

**Location**: In `load_and_prepare_data()` method, after loading historical data

**Change Required**:
```python
# Current (in orchestrator.prepare_complete_dataset):
complete_df = self.prepare_features(df)
complete_df = self.apply_embeddings(complete_df)

# Should become:
# Step 1: Apply temporal calculations FIRST
from core.calculators.temporal_feature_calculator import TemporalFeatureCalculator
temporal_calc = TemporalFeatureCalculator(db_path)
df_temporal = temporal_calc.batch_calculate_all_horses(df)

# Step 2: Continue with normal feature engineering
complete_df = self.prepare_features(df_temporal)
complete_df = self.apply_embeddings(complete_df)
```

### 3. Update Prediction Pipeline

**File**: `race_prediction/race_predict.py`

**Location**: In `prepare_race_data()` method

**Change Required**: Same temporal calculation before feature preparation

### 4. Feature Cleanup

Apply the same feature cleanup as quinté model:

**Remove**:
- Raw leaking features: `victoirescheval`, `placescheval`, `coursescheval`
- Derived leaking features: `ratio_victoires`, `ratio_places`, `gains_par_course`, `efficacite_couple`

**Keep**:
- All temporal Phase 1 features (already properly calculated)
- All market, context, and equipment features
- All properly calculated jockey/trainer stats

## Comparison with Quinté Model

### Quinté Model (Already Fixed)
- ✅ Uses `use_temporal=True` in training
- ✅ Uses `use_temporal=True` in prediction
- ✅ Applies `FeatureCleaner` to remove leaking features
- ✅ Uses batch temporal calculation (efficient)

### General Model (Needs Fix)
- ❌ Does NOT use temporal calculation
- ❌ Still uses raw career statistics with leakage
- ❌ No feature cleanup applied
- ❌ Inconsistent with quinté model approach

## Recommended Action Plan

### Phase 1: Analysis (COMPLETE)
- ✅ Identify which features have leakage
- ✅ Document current feature usage
- ✅ Compare with quinté model implementation

### Phase 2: Implementation (TODO)

1. **Update `FeatureEmbeddingOrchestrator`**
   - Add `use_temporal` parameter to `prepare_complete_dataset()`
   - Call `TemporalFeatureCalculator.batch_calculate_all_horses()` when enabled

2. **Update `train_race_model.py`**
   - Pass `use_temporal=True` to orchestrator
   - Apply `FeatureCleaner` to remove leaking features

3. **Update `race_predict.py`**
   - Apply same temporal calculation in prediction
   - Ensure feature consistency with training

4. **Test and Validate**
   - Run training with temporal features
   - Compare accuracy metrics (expect slight drop in training, improvement in validation)
   - Verify no leakage with verification script

### Phase 3: Deployment
- Retrain general model with temporal features
- Update production prediction pipeline
- Monitor real-world performance improvement

## Conclusion

**YES, temporal features ARE REQUIRED for the general model.**

The current implementation has the same data leakage issues that were identified and fixed in the quinté model. Implementing temporal features will:

1. ✅ **Eliminate data leakage** - Use only historical data before race date
2. ✅ **Improve real-world performance** - Model learns from valid historical patterns
3. ✅ **Ensure consistency** - Both quinté and general models use same approach
4. ✅ **Increase reliability** - Predictions are based on truly available information

The implementation is straightforward since the infrastructure already exists from the quinté model fix. It's primarily a matter of:
- Integrating `TemporalFeatureCalculator` into the general model pipeline
- Applying `FeatureCleaner` to remove leaking features
- Retraining the model with corrected features

**Priority: HIGH** - This should be implemented before the next model training cycle to ensure prediction quality.
