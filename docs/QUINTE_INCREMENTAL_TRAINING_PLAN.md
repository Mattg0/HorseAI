# Quinté Incremental Training Implementation Plan

**Date**: October 21, 2025
**Priority**: HIGH
**Goal**: Implement incremental training for quinté model to learn from prediction failures and improve over time

---

## Executive Summary

This document outlines the implementation plan for adding incremental training to the quinté prediction model. The goal is to analyze where the model fails in quinté predictions (quinté désordre, bonus 3, bonus 4) and implement correction strategies to improve performance iteratively.

## Current State Analysis

### Existing Infrastructure (General Model)

✅ **Already Implemented** (`model_training/regressions/regression_enhancement.py`):
- `IncrementalTrainingPipeline` class for general model
- Fetches completed races from `daily_race` with predictions and results
- Performs regression analysis on prediction errors
- Trains incremental model if improvement > 5% threshold
- Archives races to `historical_races` upon success
- Supports RF, TabNet, LSTM, and alternative models

### Quinté Model Current State

✅ **Current Quinté Features** (47 features):
1. **Quinté-Specific Features** (9):
   - `quinte_career_starts` - Total quinté races
   - `quinte_win_rate` - Win rate in quintés
   - `quinte_top5_rate` - Top 5 finish rate in quintés
   - `avg_quinte_position` - Average position in quintés
   - `days_since_last_quinte` - Recency in quintés
   - `quinte_handicap_specialist` - Performance in handicap quintés
   - `quinte_conditions_specialist` - Track condition specialist
   - `quinte_large_field_ability` - Large field performance
   - `quinte_track_condition_fit` - Track condition fit score

2. **Race Context Features** (6):
   - `purse_level_category` - Prize money category
   - `field_size_category` - Field size category
   - `track_condition_*` - Track surface indicators (PH, DUR, PS, PSF)
   - `post_position_bias` - Post position advantage

3. **Horse/Jockey Performance** (18):
   - `che_weighted_*` - Horse weighted statistics (6 features)
   - `joc_weighted_*` - Jockey weighted statistics (6 features)
   - `pourcVictChevalHippo` - Horse win % at track
   - `pourcPlaceChevalHippo` - Horse place % at track
   - `pourcVictJockHippo` - Jockey win % at track
   - `pourcPlaceJockHippo` - Jockey place % at track
   - `perf_cheval_hippo` - Horse performance at track

4. **Basic Features** (14):
   - Career stats: `victoirescheval`, `placescheval`, `coursescheval`
   - Ratios: `ratio_places`, `gains_par_course`
   - Physical: `age`, `dist`, `temperature`, `forceVent`, `numero`
   - Transformed: `recence_log`, `cotedirect_log`

### Quinté Evaluation Metrics

Current metrics from `compare_quinte_results.py`:
- ✅ **Quinté Désordre**: All 5 predicted horses finish in actual top 5 (any order)
- ✅ **Bonus 3**: All actual top 3 are in predicted top 5
- ✅ **Bonus 4**: All actual top 4 are in predicted top 5
- ✅ **Top 5/Top 6 comparison**: Different prediction strategies
- ✅ **MAE**: Mean absolute error in position predictions
- ✅ **Exact order %**: Percentage of exact position matches

---

## Problem Definition

### Quinté-Specific Challenges

1. **Complex Multi-Horse Prediction**
   - Must predict top 5 horses from field of 12-20
   - Order matters less than selection (quinté désordre)
   - Bonus metrics reward capturing podium horses

2. **Current Failure Modes**
   - Missing key horses from top 5 (quinté désordre failures)
   - Not capturing all podium horses (bonus 3 failures)
   - Not capturing top 4 (bonus 4 failures)
   - Over/under-predicting specific position ranges

3. **Learning Opportunity**
   - Each failed race shows which horses were missed
   - Patterns in failures (e.g., always missing longshots, favorites, specific track conditions)
   - Competitive field analysis may be under/over-weighted

---

## Implementation Plan

### Phase 1: Analysis & Design ⏳

#### Task 1.1: Analyze Existing Infrastructure
**Status**: In Progress
**File**: `model_training/regressions/regression_enhancement.py`

**Action Items**:
- ✅ Review `IncrementalTrainingPipeline` class
- ⏳ Identify which components are reusable for quinté
- ⏳ Document differences between general and quinté incremental training
- ⏳ Analyze how `get_completed_races()` works with quinté data

**Deliverable**: Analysis document comparing general vs quinté incremental training needs

---

#### Task 1.2: Review Quinté Model Architecture
**Status**: Pending
**Files**:
- `model_training/historical/train_quinte_model.py`
- `race_prediction/predict_quinte.py`

**Action Items**:
- ⏳ Document quinté model architecture (RF + TabNet)
- ⏳ Identify failure points in prediction pipeline
- ⏳ Review feature importance from trained models
- ⏳ Analyze competitive field analysis impact

**Deliverable**: Quinté model architecture diagram with failure points highlighted

---

#### Task 1.3: Design Quinté Incremental Training Strategy
**Status**: Pending

**Key Design Questions**:

1. **What constitutes a "failure"?**
   - Quinté désordre miss (0 points vs 500+ points)
   - Bonus 3 miss (missing podium horses)
   - Bonus 4 miss (missing top 4)
   - Combination of metrics?

2. **How to weight failures?**
   - Quinté désordre failures: HIGH weight (biggest payout impact)
   - Bonus 4 failures: MEDIUM weight (decent payout)
   - Bonus 3 failures: MEDIUM weight (decent payout)
   - MAE: LOW weight (continuous optimization)

3. **What data to use for retraining?**
   - Only failed races (high error)
   - Recent races (time-weighted)
   - Specific failure patterns (e.g., all races where favorites were missed)
   - Balanced dataset (mix of successes and failures)

4. **Correction strategies**:
   - Adjust competitive weighting
   - Retrain on hard examples (failed races)
   - Feature importance reweighting
   - Ensemble model adjustments

**Deliverable**: Quinté incremental training strategy document

---

### Phase 2: Implementation 🔨

#### Task 2.1: Create Quinté Prediction Error Analysis Module
**Status**: Pending
**New File**: `model_training/regressions/quinte_error_analyzer.py`

**Components**:

```python
class QuinteErrorAnalyzer:
    """
    Analyzes quinté prediction failures and identifies patterns.
    """

    def analyze_race_failures(self, race_data: pd.DataFrame) -> Dict:
        """
        Analyze why a quinté race prediction failed.

        Returns:
            - failure_type: 'quinte_desordre', 'bonus3', 'bonus4', or 'success'
            - missed_horses: Horses that should have been in top 5
            - false_positives: Horses predicted in top 5 but weren't
            - position_errors: MAE and per-horse position errors
            - pattern_insights: Common patterns in failures
        """
        pass

    def identify_failure_patterns(self, failed_races: List[Dict]) -> Dict:
        """
        Identify common patterns across failed quinté predictions.

        Patterns to detect:
            - Missed favorites (favorites not in predicted top 5)
            - Missed longshots (high odds horses that placed)
            - Track condition bias (failures on specific surfaces)
            - Field size bias (failures in large/small fields)
            - Competitive field issues (under/over-weighting)
            - Specific jockey/horse combinations
        """
        pass

    def calculate_failure_weights(self, failures: List[Dict]) -> pd.DataFrame:
        """
        Calculate sample weights for retraining based on failure severity.

        Weight priority:
            1. Quinté désordre misses: 10x weight
            2. Bonus 4 misses: 5x weight
            3. Bonus 3 misses: 3x weight
            4. High MAE: 2x weight
            5. Success: 1x weight (keep for balance)
        """
        pass
```

**Deliverable**: Working error analyzer module with unit tests

---

#### Task 2.2: Implement Quinté Incremental Training Pipeline
**Status**: Pending
**New File**: `model_training/regressions/quinte_incremental_trainer.py`

**Components**:

```python
class QuinteIncrementalTrainer:
    """
    Incremental training pipeline specifically for quinté predictions.
    Extends/adapts IncrementalTrainingPipeline for quinté use case.
    """

    def __init__(self, model_path: str = None, db_name: str = None):
        """Initialize with quinté-specific model paths and config."""
        # Load quinté RF and TabNet models
        # Initialize QuinteErrorAnalyzer
        # Setup quinté-specific thresholds
        pass

    def get_completed_quinte_races(self, date_from: str, date_to: str) -> List[Dict]:
        """
        Fetch completed quinté races with predictions and actual results.

        Query:
            SELECT * FROM daily_race
            WHERE quinte = 1
            AND actual_results IS NOT NULL
            AND prediction_results IS NOT NULL
            AND jour BETWEEN date_from AND date_to
        """
        pass

    def extract_failure_data(self, races: List[Dict]) -> pd.DataFrame:
        """
        Extract and prepare data from failed quinté predictions.

        Steps:
            1. Identify failures using QuinteErrorAnalyzer
            2. Calculate sample weights based on failure severity
            3. Extract features (same as training: 47 quinté features)
            4. Add failure metadata (missed horses, error type)
        """
        pass

    def train_on_failures(self, failure_data: pd.DataFrame) -> Dict:
        """
        Retrain quinté models on failed predictions.

        Strategies:
            1. Sample weighting: Give higher weight to failed races
            2. Hard example mining: Focus on worst failures
            3. Feature reweighting: Adjust feature importance
            4. Competitive adjustment: Tune competitive weighting

        Models to update:
            - Quinté RF model (incremental)
            - Quinté TabNet model (fine-tuning)
        """
        pass

    def evaluate_improvement(self, validation_data: pd.DataFrame) -> Dict:
        """
        Evaluate if incremental training improved quinté metrics.

        Metrics to check:
            - Quinté désordre success rate (before vs after)
            - Bonus 3 success rate
            - Bonus 4 success rate
            - MAE (should not degrade)

        Threshold:
            - Must improve quinté désordre by >= 5% OR
            - Must improve bonus 4 by >= 10% without degrading others
        """
        pass

    def save_incremental_quinte_model(self, results: Dict) -> str:
        """
        Save improved quinté model with versioning.

        Path: models/YYYY-MM-DD/quinte_incremental_v{version}/
        Files:
            - rf_model.joblib
            - tabnet_model.zip
            - metadata.json (with improvement metrics)
        """
        pass
```

**Deliverable**: Working incremental trainer for quinté

---

#### Task 2.3: Create Correction Strategy Module
**Status**: Pending
**New File**: `model_training/regressions/quinte_correction_strategy.py`

**Components**:

```python
class QuinteCorrectionStrategy:
    """
    Implements specific correction strategies for quinté failures.
    """

    def adjust_competitive_weighting(self, failure_patterns: Dict) -> float:
        """
        Adjust competitive field analysis weight based on failures.

        If missing favorites → Increase base model weight (reduce competitive)
        If missing longshots → Increase competitive weight
        """
        pass

    def reweight_features(self, failure_analysis: Dict) -> Dict[str, float]:
        """
        Suggest feature importance adjustments.

        Example:
            If failures correlated with track conditions:
                → Increase weight of track_condition features
            If failures with specific jockeys:
                → Increase weight of jockey features
        """
        pass

    def generate_hard_examples_dataset(self, failures: pd.DataFrame) -> pd.DataFrame:
        """
        Create focused training dataset from worst failures.

        Selection criteria:
            - Top 20% worst quinté désordre failures
            - Races where predicted and actual have 0 overlap
            - Recent failures (last 30 days higher weight)
        """
        pass

    def suggest_model_adjustments(self, error_analysis: Dict) -> List[str]:
        """
        Generate actionable suggestions for model improvement.

        Returns list of suggestions like:
            - "Increase competitive_weight by 0.1 (missing underdogs)"
            - "Add more weight to quinte_large_field_ability feature"
            - "Retrain on races with field_size > 15"
        """
        pass
```

**Deliverable**: Correction strategy module with analysis tools

---

#### Task 2.4: Integrate with Existing Framework
**Status**: Pending
**Files to Modify**:
- `model_training/regressions/regression_enhancement.py`
- `UI/UIhelper.py`

**Actions**:

1. **Add Quinté Support to `regression_enhancement.py`**:
```python
def run_quinte_incremental_training(
    self,
    date_from: str,
    date_to: str,
    focus_on_failures: bool = True
) -> Dict:
    """
    Run incremental training pipeline for quinté model.

    Args:
        date_from: Start date for race selection
        date_to: End date for race selection
        focus_on_failures: If True, focus on failed predictions

    Returns:
        Training results and improvement metrics
    """
    # Use QuinteIncrementalTrainer
    # Analyze failures
    # Apply corrections
    # Validate improvement
    # Save if successful
    pass
```

2. **Add UIHelper Method**:
```python
def execute_quinte_incremental_training(
    self,
    date_from: str,
    date_to: str,
    progress_callback=None
) -> Dict:
    """Execute quinté incremental training with UI feedback."""
    pass
```

**Deliverable**: Integrated quinté incremental training in existing framework

---

### Phase 3: Testing & Validation 🧪

#### Task 3.1: Test on Historical Failed Races
**Status**: Pending

**Test Cases**:
1. Load last 30 days of quinté races
2. Identify all quinté désordre failures
3. Run incremental training on failures
4. Validate on separate test set
5. Measure improvement in success rates

**Success Criteria**:
- Quinté désordre success rate improves by >= 5%
- Bonus 3/4 success rates don't degrade
- MAE stays within 5% of baseline

**Deliverable**: Test results report

---

#### Task 3.2: Validate Correction Strategies
**Status**: Pending

**Tests**:
1. **Competitive Weighting Test**:
   - Run predictions with adjusted weights
   - Compare results on failed races

2. **Feature Reweighting Test**:
   - Retrain with adjusted feature importance
   - Validate on holdout set

3. **Hard Examples Test**:
   - Train on hard examples only
   - Measure generalization to new races

**Deliverable**: Validation report with metrics

---

### Phase 4: UI Integration 🖥️

#### Task 4.1: Create Quinté Incremental Training UI
**Status**: Pending
**File**: `UI/UIApp.py`

**Components**:

```python
# New tab or section in existing Incremental Training tab
if st.checkbox("🎯 Quinté-Specific Training"):
    st.markdown("### Quinté Incremental Training")

    # Date range selector
    date_from = st.date_input("From Date")
    date_to = st.date_input("To Date")

    # Training options
    focus_failures = st.checkbox("Focus on Failed Predictions", value=True)
    min_improvement = st.slider("Minimum Improvement %", 1, 20, 5)

    # Failure analysis display
    if st.button("Analyze Quinté Failures"):
        # Show failure breakdown
        # Display patterns
        # Show correction suggestions

    # Incremental training button
    if st.button("Run Quinté Incremental Training"):
        # Execute training
        # Show progress
        # Display results
```

**Features**:
- Date range selection
- Failure analysis visualization
- Training progress tracking
- Before/after metrics comparison
- Model versioning display

**Deliverable**: Working UI for quinté incremental training

---

## Technical Architecture

### Data Flow

```
1. Fetch Completed Quinté Races
   ↓
   daily_race (quinte=1, has results & predictions)

2. Error Analysis
   ↓
   QuinteErrorAnalyzer
   - Identify failures (désordre, bonus 3/4)
   - Calculate failure weights
   - Extract patterns

3. Correction Strategy
   ↓
   QuinteCorrectionStrategy
   - Adjust competitive weights
   - Reweight features
   - Generate hard examples

4. Incremental Training
   ↓
   QuinteIncrementalTrainer
   - Train on weighted failures
   - Fine-tune models
   - Validate improvement

5. Model Deployment
   ↓
   If improved >= threshold:
     - Save new model version
     - Archive processed races
     - Update config
```

### Database Schema

**Tables Used**:
- `daily_race` - Source of completed races with predictions
- `race_predictions` - Detailed prediction data with competitive scores
- `historical_races` - Archived races (after successful training)

**Key Columns**:
- `daily_race.quinte` - Quinté race flag
- `daily_race.actual_results` - Actual finish order (e.g., "13-10-7-1-8")
- `daily_race.prediction_results` - JSON with predictions
- `race_predictions.competitive_adjustment` - Competitive field scores

---

## Metrics & Success Criteria

### Training Metrics

1. **Quinté Désordre Success Rate**
   - Baseline: Current success rate (e.g., 14%)
   - Target: +5% improvement (e.g., 19%)

2. **Bonus 3 Success Rate**
   - Baseline: Current rate (e.g., 25%)
   - Target: +5% improvement or maintain

3. **Bonus 4 Success Rate**
   - Baseline: Current rate (e.g., 30%)
   - Target: +5% improvement or maintain

4. **Mean Absolute Error (MAE)**
   - Baseline: Current MAE
   - Target: Within 5% of baseline (don't degrade)

### Deployment Criteria

Deploy new incremental model if **ANY** of:
- Quinté désordre rate improves >= 5%
- (Bonus 3 + Bonus 4) combined improves >= 8%
- MAE improves >= 10% without degrading quinté metrics

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Overfitting to failures | HIGH | Use validation set, regularization, early stopping |
| Degrading general performance | MEDIUM | Monitor MAE, keep baseline model if degradation |
| Insufficient failure data | MEDIUM | Accumulate more races before training |
| Computational cost | LOW | Use efficient batch processing |

### Business Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Model becomes unstable | HIGH | Version control, rollback capability |
| Improvement too small | MEDIUM | Set realistic thresholds, iterate |
| False improvements | MEDIUM | Rigorous validation on unseen data |

---

## Timeline

### Sprint 1: Foundation (Week 1)
- ✅ Create implementation plan (this document)
- ⏳ Analyze existing infrastructure
- ⏳ Review quinté model architecture
- ⏳ Design error analysis module

### Sprint 2: Core Implementation (Week 2)
- ⏳ Implement QuinteErrorAnalyzer
- ⏳ Implement QuinteIncrementalTrainer
- ⏳ Create correction strategies
- ⏳ Unit tests for all modules

### Sprint 3: Integration (Week 3)
- ⏳ Integrate with regression_enhancement.py
- ⏳ Add UI components
- ⏳ End-to-end testing
- ⏳ Validation on historical data

### Sprint 4: Production (Week 4)
- ⏳ Final testing
- ⏳ Documentation
- ⏳ Deployment
- ⏳ Monitoring setup

---

## Dependencies

### Code Dependencies
- ✅ `IncrementalTrainingPipeline` (existing)
- ✅ `QuinteRaceModel` (existing)
- ✅ `compare_quinte_results.py` (for metrics)
- ✅ `TemporalFeatureCalculator` (for feature consistency)
- ✅ `FeatureCleaner` (for data quality)

### Data Dependencies
- ✅ `daily_race` table with quinté predictions
- ✅ `race_predictions` table with competitive scores
- ✅ Historical quinté races with results

### Infrastructure Dependencies
- ✅ Model versioning system
- ✅ Database archiving system
- ✅ UI framework (Streamlit)

---

## Success Metrics

### Phase 1 Success
- ✅ Complete analysis of existing infrastructure
- ✅ Documented quinté failure patterns
- ✅ Design approved

### Phase 2 Success
- All modules implemented and tested
- Integration complete
- Unit test coverage > 80%

### Phase 3 Success
- Validation tests pass
- Improvement metrics meet thresholds
- No regression in baseline metrics

### Phase 4 Success
- UI functional and user-friendly
- Production deployment successful
- Monitoring in place

---

## Next Steps

1. ⏳ **Task 1.1**: Complete analysis of `IncrementalTrainingPipeline`
2. ⏳ **Task 1.2**: Document quinté model architecture
3. ⏳ **Task 1.3**: Finalize incremental training strategy
4. ⏳ **Task 2.1**: Begin implementation of `QuinteErrorAnalyzer`

---

## Appendix

### A. Quinté Feature List (47 features)

**Quinté-Specific** (9):
- quinte_career_starts, quinte_win_rate, quinte_top5_rate
- avg_quinte_position, days_since_last_quinte
- quinte_handicap_specialist, quinte_conditions_specialist
- quinte_large_field_ability, quinte_track_condition_fit

**Race Context** (6):
- purse_level_category, field_size_category
- track_condition_PH, track_condition_DUR, track_condition_PS, track_condition_PSF

**Horse Performance** (12):
- che_weighted_avg_pos, che_weighted_recent_perf, che_weighted_consistency
- che_weighted_pct_top3, che_weighted_total_races, che_weighted_dnf_rate
- pourcVictChevalHippo, pourcPlaceChevalHippo, perf_cheval_hippo
- victoirescheval, placescheval, coursescheval

**Jockey Performance** (6):
- joc_weighted_avg_pos, joc_weighted_recent_perf, joc_weighted_consistency
- joc_weighted_pct_top3, joc_weighted_total_races, joc_weighted_dnf_rate

**Basic/Derived** (14):
- age, dist, temperature, forceVent, numero
- ratio_places, gains_par_course
- pourcVictJockHippo, pourcPlaceJockHippo
- post_position_bias, recence_log, cotedirect_log

### B. Reference Implementation: General Model Incremental Training

See: `model_training/regressions/regression_enhancement.py`
- Lines 33-240: `IncrementalTrainingPipeline` class
- Lines 423-478: `analyze_model_performance()`
- Lines 479-575: `train_incremental_model()`
- Lines 675-1067: `run_incremental_training_pipeline()`

---

**Document Status**: Draft v1.0
**Last Updated**: October 21, 2025
**Next Review**: After Task 1.3 completion
