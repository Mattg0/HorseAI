# Metadata Saving Fix: Feature Columns Now Saved for General Models

## Problem

The script [extract_general_model_features.py](extract_general_model_features.py) expected `metadata.json` with feature information, but general model training never created it!

### Before Fix

**General Model Directory:**
```
models/2025-10-26/2years_185022/
├── rf_model.joblib              ✅ Saved
├── feature_engineer.joblib      ✅ Saved
├── model_config.json            ✅ Saved (no feature info)
├── feature_columns.json         ❌ NOT saved
└── metadata.json                ❌ NOT saved
```

**Quinte Model Directory (worked correctly):**
```
models/2025-10-20/2years_165822_quinte_rf/
├── rf_model.joblib              ✅ Saved
├── feature_columns.json         ✅ Saved
└── model_config.json            ✅ Saved
```

### Impact

1. ❌ `extract_general_model_features.py` failed to find features
2. ❌ No feature tracking for model versioning
3. ❌ Difficult to debug which features were used
4. ❌ Inconsistent behavior between general and quinte models

## Solution Implemented

### Files Modified

1. **[utils/model_manager.py](utils/model_manager.py)** - Updated `save_models()` method
2. **[model_training/historical/train_race_model.py](model_training/historical/train_race_model.py)** - Updated `save_models()` method
3. **[extract_general_model_features.py](extract_general_model_features.py)** - Made extraction more robust

### Changes in Detail

#### 1. `model_manager.py` - Save Feature Columns

**Added parameters:**
```python
def save_models(self, ..., rf_feature_columns=None, tabnet_feature_columns=None):
```

**Save RF features:**
```python
if rf_feature_columns:
    feature_path = save_path / "feature_columns.json"
    with open(feature_path, 'w') as f:
        json.dump(rf_feature_columns, f, indent=2)
    print(f"✅ Saved {len(rf_feature_columns)} RF feature names")
```

**Save TabNet features:**
```python
if tabnet_feature_columns:
    tabnet_feature_path = save_path / "tabnet_feature_columns.json"
    with open(tabnet_feature_path, 'w') as f:
        json.dump(tabnet_feature_columns, f, indent=2)
    print(f"✅ Saved {len(tabnet_feature_columns)} TabNet feature names")
```

**Update config:**
```python
config_data = {
    ...,
    'feature_count': len(rf_feature_columns) if rf_feature_columns else None
}
```

#### 2. `train_race_model.py` - Extract and Pass Features

**Extract feature names from training results:**
```python
def save_models(self, orchestrator=None):
    # Extract RF feature names
    rf_feature_columns = None
    if self.training_results:
        rf_results = self.training_results.get('rf_results', {})
        if 'feature_names' in rf_results:
            rf_feature_columns = rf_results['feature_names']
            print(f"📋 Extracted {len(rf_feature_columns)} RF feature names")

    # Extract TabNet feature names
    tabnet_feature_columns = None
    if self.training_results:
        tabnet_results = self.training_results.get('tabnet_results', {})
        if tabnet_results.get('status') == 'success':
            if 'feature_names' in tabnet_results:
                tabnet_feature_columns = tabnet_results['feature_names']

    # Pass to model_manager
    model_manager.save_models(
        rf_model=self.rf_model,
        feature_state=feature_state,
        rf_feature_columns=rf_feature_columns,
        tabnet_feature_columns=tabnet_feature_columns
    )
```

#### 3. `extract_general_model_features.py` - Robust Extraction

**Added priority-based feature extraction:**

```python
# Method 1: Check for feature_columns.json (NEW: most reliable)
feature_columns_file = Path(rf_path) / 'feature_columns.json'
if feature_columns_file.exists():
    with open(feature_columns_file, 'r') as f:
        rf_features = json.load(f)
    return rf_features

# Method 2: Check for metadata.json (legacy)
metadata_file = Path(rf_path) / 'metadata.json'
if metadata_file.exists():
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        if 'feature_columns' in metadata:
            return metadata['feature_columns']

# Method 3: Try model.feature_names_in_ (sklearn fallback)
if hasattr(rf_model, 'feature_names_in_'):
    return list(rf_model.feature_names_in_)

# Method 4: Try model.base_regressor.feature_names_in_ (wrapped model)
if hasattr(rf_model, 'base_regressor') and hasattr(rf_model.base_regressor, 'feature_names_in_'):
    return list(rf_model.base_regressor.feature_names_in_)

# Method 5: Fallback to generic names
if hasattr(rf_model, 'n_features_in_'):
    return [f"feature_{i}" for i in range(rf_model.n_features_in_)]
```

## After Fix

**General Model Directory (NEW):**
```
models/DATE/DB_TIME/
├── rf_model.joblib                    ✅ Model
├── feature_columns.json               ✅ NEW - RF feature names
├── tabnet_feature_columns.json        ✅ NEW - TabNet feature names (if TabNet trained)
├── model_config.json                  ✅ Updated - includes feature_count
├── feature_engineer.joblib            ✅ Feature state
└── logs/                              ✅ Training logs
```

**Model Config JSON (Enhanced):**
```json
{
  "db_type": "2years",
  "created_at": "2025-10-29T12:00:00",
  "is_quinte": false,
  "model_suffix": "",
  "feature_count": 108
}
```

**Feature Columns JSON:**
```json
[
  "cotedirect",
  "recence",
  "numero",
  "age",
  "che_global_avg_pos",
  "che_weighted_avg_pos",
  "joc_global_avg_pos",
  ...
]
```

## Benefits

1. ✅ **Consistent behavior** - General models now save features like Quinte models
2. ✅ **Extract script works** - Can now find features reliably
3. ✅ **Better tracking** - Easy to see what features each model version uses
4. ✅ **Easier debugging** - Can verify feature alignment between training and prediction
5. ✅ **Robust fallbacks** - Multiple methods to find features (5 fallback levels)
6. ✅ **Future-proof** - Works with both new and old model directories

## Testing

### Test 1: Feature Saving Logic
```bash
python3 test_feature_saving.py
```
✅ Verifies that features are saved correctly to JSON files

### Test 2: Feature Extraction Logic
```bash
python3 test_extract_features.py
```
✅ Verifies that features can be loaded from multiple sources with correct priority

### After Next Training

After training a new model, you should see:

```
===== SAVING ALL TRAINED MODELS =====
  📋 Extracted 108 RF feature names from training results
  📋 Extracted 37 TabNet feature names from training results
  ✅ Saved 108 RF feature names to feature_columns.json
  ✅ Saved 37 TabNet feature names to tabnet_feature_columns.json
Models saved to: models/2025-10-29/2years_120530
```

Then verify with:
```bash
python3 extract_general_model_features.py
```

Expected output:
```
============================================================
EXTRACTING RF MODEL FEATURES
============================================================
RF model path: models/2025-10-29/2years_120530
✅ Found feature_columns.json: models/2025-10-29/2years_120530/feature_columns.json
✅ Loaded 108 RF features from feature_columns.json

============================================================
EXTRACTING TABNET MODEL FEATURES
============================================================
TabNet model path: models/2025-10-29/2years_120530
✅ Found tabnet_feature_columns.json: models/2025-10-29/2years_120530/tabnet_feature_columns.json
✅ Loaded 37 TabNet features from tabnet_feature_columns.json

✅ Saved RF features to: rf_features.json
✅ Saved TabNet features to: tabnet_features.json
```

## Backwards Compatibility

The enhanced extraction script maintains backwards compatibility:

1. **Old models without feature_columns.json** - Will use sklearn's `feature_names_in_` attribute
2. **Old models with metadata.json** - Will load from there (legacy support)
3. **New models** - Will use `feature_columns.json` (preferred)

## Related Changes

This fix complements the recent feature gap fix:
- [FEATURE_GAP_FIX_SUMMARY.md](FEATURE_GAP_FIX_SUMMARY.md) - Fixed missing features in general model
- [GENERAL_MODEL_FEATURE_FIX.md](GENERAL_MODEL_FEATURE_FIX.md) - Detailed feature fix documentation

Together, these fixes ensure:
1. General model has all necessary features (feature gap fix)
2. Features are properly saved and tracked (this fix)
3. Easy to verify what features are being used (this fix)
