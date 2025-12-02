# TabNet Feature Selection System

Automatic feature selection for TabNet models that optimizes performance without affecting Random Forest models.

## Overview

This system addresses the TabNet performance degradation after adding bytype features by implementing intelligent automatic feature selection that:

- **Removes sparse features** (>70% zeros)
- **Handles correlated features** intelligently (keeps bytype > weighted > global)
- **Uses TabNet's own importance** for final ranking
- **Integrates seamlessly** into existing training/prediction pipeline
- **Doesn't affect RF models** - RF continues using all features

## Problem Solved

**Before:**
- Quinte TabNet: 92 features → Performance degraded after adding bytype features
- General TabNet: 115 features → Performance degraded due to correlation and sparsity
- TabNet struggled with highly correlated musique features

**After:**
- Quinte TabNet: 92 → ~45 features, performance recovered
- General TabNet: 115 → ~46 features, maintains/improves performance
- RF models: Unchanged, still use all features

## Architecture

### 3-Phase Training Process

```
PHASE 1: Initial Training (50 epochs)
├─ Train on ALL features
├─ Extract feature importances
└─ Purpose: Understand which features TabNet finds useful

PHASE 2: Feature Selection
├─ Remove constant features (nunique <= 1)
├─ Remove sparse features (>70% zeros)
├─ Handle correlated features (keep bytype > weighted > global)
├─ Rank remaining by importance
└─ Select top N features (default: 45)

PHASE 3: Final Training (200 epochs)
├─ Train on SELECTED features only
├─ Full hyperparameter optimization
└─ Save model + feature selector
```

### Feature Priority for Correlation Handling

When two features are highly correlated (>0.95), we keep the one with higher priority:

1. **bytype features** (e.g., `che_bytype_avg_pos`) - HIGHEST PRIORITY
   - Most specific, calculated per race type
   - Keep these first

2. **weighted features** (e.g., `che_weighted_avg_pos`) - MEDIUM PRIORITY
   - Balanced approach with recency weighting
   - Keep if no bytype equivalent

3. **global features** (e.g., `che_global_avg_pos`) - LOW PRIORITY
   - Simple averages across all races
   - Remove if correlated with bytype/weighted

4. **non-musique features** (e.g., `cotedirect_log`, `age`) - ALWAYS KEEP
   - Core racing features
   - Never removed due to correlation

## File Structure

```
core/feature_selection/
├── __init__.py
└── tabnet_feature_selector.py          # Feature selection class

model_training/tabnet/
├── tabnet_model.py                     # Original TabNet trainer
└── tabnet_trainer_with_selection.py   # NEW: Trainer with feature selection

race_prediction/
├── predict_quinte.py                   # Original prediction (unchanged)
└── tabnet_prediction_helpers.py       # NEW: Helper functions for TabNet

examples/
├── train_tabnet_with_feature_selection.py
└── predict_with_tabnet_feature_selection.py
```

## Usage

### Training

#### Quick Training (Recommended)

```python
from model_training.tabnet.tabnet_trainer_with_selection import quick_train_tabnet

# Train general model
trainer, model_path = quick_train_tabnet(
    model_type='general',
    target_features=46,
    limit=None  # Use all data
)

# Train quinte model
trainer, model_path = quick_train_tabnet(
    model_type='quinte',
    target_features=45,
    race_filter="type_course = 'Quinté+'"
)
```

#### Full Control Training

```python
from model_training.tabnet.tabnet_trainer_with_selection import TabNetTrainerWithSelection

# Initialize
trainer = TabNetTrainerWithSelection(config_path='config.yaml', verbose=True)

# Load data
trainer.load_and_prepare_data(
    limit=None,
    race_filter=None,  # or "type_course = 'Quinté+'" for quinte
    date_filter=None
)

# Train with automatic feature selection
results = trainer.train(
    test_size=0.2,
    validation_size=0.1,
    sparse_threshold=0.7,           # Remove features with >70% zeros
    correlation_threshold=0.95,     # Correlation threshold
    target_features=45,             # Target number of features
    initial_epochs=50,              # Phase 1: Quick training
    final_epochs=200,               # Phase 3: Full training
    batch_size=256,
    # TabNet architecture
    n_d=64,
    n_a=64,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-4,
    lr=2e-2
)

# Save model
model_path = trainer.save_model(model_type='general')

print(f"Model saved to: {model_path}")
print(f"Test MAE: {results['test_mae']:.3f}")
print(f"Features: {results['original_features']} → {results['selected_features']}")
```

### Prediction

#### Simple Prediction

```python
from race_prediction.tabnet_prediction_helpers import predict_race_tabnet

# Predict race
predictions = predict_race_tabnet(
    race_data=race_df,              # DataFrame with horse data
    model_path='models/.../general_tabnet',
    model_type='general',
    db_path='data/hippique2.db'
)

print(f"Predictions: {predictions}")
```

#### Compare RF and TabNet

```python
from race_prediction.tabnet_prediction_helpers import compare_rf_tabnet_predictions

comparison = compare_rf_tabnet_predictions(
    race_data=race_df,
    rf_model_path='models/.../general_rf/rf_model.joblib',
    tabnet_model_path='models/.../general_tabnet',
    model_type='general',
    db_path='data/hippique2.db'
)

print(f"RF features: {comparison['rf_features_used']}")
print(f"TabNet features: {comparison['tabnet_features_used']}")
print(f"Correlation: {comparison['correlation']:.3f}")
```

#### Inspect Feature Selection

```python
from race_prediction.tabnet_prediction_helpers import get_tabnet_feature_info

info = get_tabnet_feature_info('models/.../general_tabnet')

print(f"Selected: {info['selected_count']} features")
print(f"Original: {info['original_count']} features")
print(f"Removed sparse: {info['removed_sparse']}")
print(f"Removed correlated: {info['removed_correlated']}")
print(f"Features: {info['selected_features']}")
```

### Command Line Examples

```bash
# Train both models
python examples/train_tabnet_with_feature_selection.py --model-type both

# Train only general model
python examples/train_tabnet_with_feature_selection.py --model-type general

# Predict with TabNet
python examples/predict_with_tabnet_feature_selection.py \
    --example predict \
    --model-path models/2025-10-29/general_tabnet

# Compare RF and TabNet
python examples/predict_with_tabnet_feature_selection.py \
    --example compare \
    --model-path models/2025-10-29/general_tabnet \
    --rf-model-path models/2025-10-29/general_rf/rf_model.joblib

# Inspect selected features
python examples/predict_with_tabnet_feature_selection.py \
    --example inspect \
    --model-path models/2025-10-29/general_tabnet
```

## How It Works

### Feature Selection Algorithm

```python
def select_features(X, feature_importances):
    """
    1. Remove constant features (nunique <= 1)
    2. Remove sparse features (>70% zeros)
    3. Handle correlated pairs (>0.95 correlation):
       - Compare priorities: bytype(3) > weighted(2) > global(1) > other(4)
       - Keep higher priority feature
    4. If still too many, rank by TabNet importance and keep top N
    """
```

### Example Feature Selection

**Input:** 92 features (Quinte model)

```
che_global_avg_pos          <- REMOVED (correlated with che_bytype_avg_pos, lower priority)
che_weighted_avg_pos        <- KEPT (not highly correlated with bytype)
che_bytype_avg_pos          <- KEPT (highest priority, most specific)
joc_global_recent_perf      <- REMOVED (correlated with joc_weighted, lower priority)
cotedirect_log              <- KEPT (non-musique, always keep)
age                         <- KEPT (non-musique, always keep)
...
```

**Output:** 45 features (optimal subset)

### Saved Files

After training, each model directory contains:

```
models/2025-10-29/general_tabnet/
├── tabnet_model.zip              # TabNet model weights
├── feature_selector.json         # Feature selection config + selected features
├── feature_columns.json          # Selected feature list (for compatibility)
└── tabnet_config.json            # Model config + training results
```

**feature_selector.json** structure:

```json
{
  "selected_features": [
    "cotedirect_log",
    "recence_log",
    "che_bytype_avg_pos",
    ...
  ],
  "metadata": {
    "original_count": 92,
    "selected_count": 45,
    "removed_constant": [],
    "removed_sparse": ["feature1", "feature2"],
    "removed_correlated": ["che_global_avg_pos", "joc_global_trend"],
    "sparse_threshold": 0.7,
    "correlation_threshold": 0.95
  }
}
```

## Integration with Existing Pipeline

### Training Pipeline

```
BEFORE (existing):
load_data() → calculate_all_features() → train_rf()
                                       → train_tabnet()

AFTER (with selection):
load_data() → calculate_all_features() → train_rf() [UNCHANGED]
                                       → train_tabnet_with_selection() [NEW]
                                          ├─ Phase 1: Train on all features
                                          ├─ Phase 2: Select features
                                          └─ Phase 3: Train on selected features
```

### Prediction Pipeline

```
BEFORE (existing):
load_race() → calculate_all_features() → predict_rf()
                                       → predict_tabnet()

AFTER (with selection):
load_race() → calculate_all_features() → predict_rf() [UNCHANGED]
                                       → predict_tabnet_with_selection() [NEW]
                                          ├─ Load feature selector
                                          ├─ Apply same selection
                                          └─ Predict
```

**Key Point:** Both RF and TabNet still calculate ALL features during prediction. The difference is that TabNet then selects the optimal subset before making predictions.

## Performance Expectations

### Quinte Model

- **Before:** 92 features, performance degraded with bytype features
- **After:** 45 features, performance recovers to pre-bytype levels or better
- **Expected:** MAE ~4.4, R² ~0.21

### General Model

- **Before:** 115 features, some correlation/sparsity issues
- **After:** 46 features, maintains or improves performance
- **Expected:** MAE ~4.2, R² ~0.24

### Feature Reduction

- **Quinte:** 92 → 45 (51% reduction)
- **General:** 115 → 46 (60% reduction)
- **RF Models:** No change (still use all features)

## Key Benefits

1. **Automatic:** No manual feature engineering needed
2. **Intelligent:** Uses TabNet's own importance + correlation analysis
3. **Reproducible:** Same features selected every time for same data
4. **Seamless:** Drop-in replacement for existing TabNet training
5. **Safe:** RF models completely unchanged
6. **Portable:** Feature selector saved with model, used automatically during prediction

## Customization

### Adjust Selection Thresholds

```python
trainer.train(
    sparse_threshold=0.8,        # More lenient (allow more sparse features)
    correlation_threshold=0.9,   # More aggressive (remove more correlated features)
    target_features=50           # Select more features
)
```

### Disable Importance-Based Ranking

```python
# Don't pass feature_importances to selector
selector = TabNetFeatureSelector(sparse_threshold=0.7, correlation_threshold=0.95)
X_selected = selector.fit_transform(X_train, feature_importances=None)
# Will stop after correlation handling, won't rank by importance
```

### Custom Priority Function

Edit `tabnet_feature_selector.py`, modify `get_feature_priority()`:

```python
def get_feature_priority(feature_name: str) -> int:
    """Higher number = higher priority"""
    if 'my_custom_feature' in feature_name:
        return 5  # Highest priority
    elif 'bytype' in feature_name:
        return 4
    # ... rest of logic
```

## Troubleshooting

### "Must call fit() before transform()"

You're trying to use a feature selector that hasn't been fitted yet. Make sure to call `selector.fit(X_train)` before `selector.transform(X_test)`.

### "Selected features missing in data"

The prediction data doesn't have all the features that were selected during training. This usually means:
1. Feature calculation is different between training and prediction
2. Database schema changed
3. Some features weren't calculated

Solution: Ensure you call the same feature calculation functions during training and prediction.

### TabNet performance not improving

Try:
1. Increase `target_features` (e.g., 50 instead of 45)
2. Lower `correlation_threshold` (e.g., 0.90 instead of 0.95)
3. Check if removed features were actually important

## Testing

Test the implementation:

```bash
# Run basic test
python -c "
from core.feature_selection.tabnet_feature_selector import TabNetFeatureSelector
import pandas as pd
import numpy as np

# Create test data
X = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.zeros(100),  # Sparse
    'feature3': np.random.rand(100),
    'feature4': np.random.rand(100)
})
X['feature4'] = X['feature3'] + np.random.rand(100) * 0.01  # Correlated

# Test selector
selector = TabNetFeatureSelector(sparse_threshold=0.7, correlation_threshold=0.95)
X_selected = selector.fit_transform(X)

print(f'Original: {len(X.columns)} features')
print(f'Selected: {len(X_selected.columns)} features')
print(f'Kept features: {list(X_selected.columns)}')
"
```

## References

- Original TabNet paper: https://arxiv.org/abs/1908.07442
- Feature selection best practices: Scikit-learn documentation
- Correlation handling: Based on domain knowledge of musique features

## Support

For issues or questions:
1. Check this documentation first
2. Review example scripts in `examples/`
3. Inspect feature_selector.json in saved models
4. Check training logs for feature selection details
