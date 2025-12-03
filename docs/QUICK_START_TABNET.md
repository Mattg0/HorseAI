# TabNet with Feature Selection - Quick Start

> **TL;DR**: TabNet models now automatically select optimal features during training. RF models are unchanged.

## What Changed

- **TabNet**: Now uses automatic feature selection (92 → 45 features for Quinte, 115 → 46 for General)
- **Random Forest**: No changes, still uses all features
- **Performance**: TabNet performance recovers/improves with optimized feature set

## Training (Choose One)

### Option 1: Command Line (Easiest)

```bash
# Train both General and Quinte models
python examples/train_tabnet_with_feature_selection.py --model-type both

# Train only General model
python examples/train_tabnet_with_feature_selection.py --model-type general

# Train only Quinte model
python examples/train_tabnet_with_feature_selection.py --model-type quinte
```

### Option 2: Python (Quick)

```python
from model_training.tabnet.tabnet_trainer_with_selection import quick_train_tabnet

# General model
trainer, model_path = quick_train_tabnet(model_type='general', target_features=46)

# Quinte model
trainer, model_path = quick_train_tabnet(model_type='quinte', target_features=45)
```

### Option 3: Python (Full Control)

```python
from model_training.tabnet.tabnet_trainer_with_selection import TabNetTrainerWithSelection

trainer = TabNetTrainerWithSelection(verbose=True)
trainer.load_and_prepare_data()
results = trainer.train(
    sparse_threshold=0.7,         # Remove features with >70% zeros
    correlation_threshold=0.95,   # Remove highly correlated
    target_features=45            # Target number of features
)
model_path = trainer.save_model(model_type='general')
```

## Prediction

### Option 1: Using Helper Function

```python
from race_prediction.tabnet_prediction_helpers import predict_race_tabnet

predictions = predict_race_tabnet(
    race_data=race_df,
    model_path='models/.../general_tabnet',
    model_type='general',
    db_path='data/hippique2.db'
)
```

### Option 2: Manual

```python
from race_prediction.tabnet_prediction_helpers import load_tabnet_with_selector
from core.calculators.static_feature_calculator import FeatureCalculator

# Calculate ALL features
df = FeatureCalculator.calculate_all_features(race_df, use_temporal=True, db_path='...')

# Load model and selector
model, selector, feature_list = load_tabnet_with_selector('models/.../general_tabnet')

# Select features and predict
X_selected = selector.transform(df)
predictions = model.predict(X_selected.values)
```

## Verification

### Check Feature Selection

```python
from race_prediction.tabnet_prediction_helpers import get_tabnet_feature_info

info = get_tabnet_feature_info('models/.../general_tabnet')
print(f"Selected: {info['selected_count']} features")
print(f"Original: {info['original_count']} features")
print(f"Removed sparse: {info['removed_sparse']}")
print(f"Removed correlated: {info['removed_correlated']}")
```

### Compare RF and TabNet

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

## Saved Model Structure

```
models/2025-10-29/general_tabnet/
├── tabnet_model.zip              # Model weights
├── feature_selector.json         # Feature selection config
├── feature_columns.json          # Selected feature list
└── tabnet_config.json            # Model config + training results
```

## What Gets Selected

### Removed Features

- **Constant features** (nunique <= 1)
- **Sparse features** (>70% zeros)
- **Correlated features** with lower priority:
  - `che_global_*` removed if correlated with `che_bytype_*`
  - `che_weighted_*` removed if correlated with `che_bytype_*`
  - Similar for `joc_*` features

### Kept Features

- **Core racing features**: `cotedirect_log`, `recence_log`, `age`, etc.
- **Most specific musique features**: `che_bytype_*`, `joc_bytype_*`
- **Important non-musique features**: Based on TabNet importance scores

## Expected Feature Counts

| Model Type | Original | Selected | Reduction |
|-----------|----------|----------|-----------|
| Quinte TabNet | 92 | ~45 | 51% |
| Quinte RF | 92 | 92 | 0% (unchanged) |
| General TabNet | 115 | ~46 | 60% |
| General RF | 115 | 115 | 0% (unchanged) |

## Troubleshooting

### Training takes too long?

Reduce `initial_epochs` (Phase 1) from 50 to 30:

```python
trainer.train(initial_epochs=30, final_epochs=200)
```

### Too many features removed?

Increase thresholds:

```python
trainer.train(
    sparse_threshold=0.8,        # More lenient
    correlation_threshold=0.98,  # More lenient
    target_features=50           # Keep more features
)
```

### Features missing during prediction?

Make sure you calculate ALL features during prediction (same as training):

```python
df = FeatureCalculator.calculate_all_features(df, use_temporal=True, db_path='...')
```

The feature selector will then select the appropriate subset automatically.

## Full Documentation

See [TABNET_FEATURE_SELECTION.md](TABNET_FEATURE_SELECTION.md) for complete documentation.

## Tests

Run tests to verify everything works:

```bash
python tests/test_tabnet_feature_selector.py
```

Expected output: `5/5 tests passed`

## Key Points

1. **RF models unchanged** - They still use all features
2. **TabNet optimized** - Automatic feature selection during training
3. **Seamless integration** - Drop-in replacement for existing TabNet training
4. **Reproducible** - Same features selected every time
5. **Automatic** - No manual feature engineering needed

---

**Ready to use?** Just run:

```bash
python examples/train_tabnet_with_feature_selection.py --model-type both
```

That's it! The system handles everything else automatically.
