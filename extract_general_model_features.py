#!/usr/bin/env python3
"""
Extract feature list from general model (RF + TabNet)
This script loads the trained models and outputs the features they expect.
"""

import json
import joblib
from pathlib import Path
from utils.model_manager import get_model_manager
from utils.env_setup import AppConfig

def extract_rf_features():
    """Extract features used by Random Forest model"""
    print("="*60)
    print("EXTRACTING RF MODEL FEATURES")
    print("="*60)

    model_manager = get_model_manager()
    model_paths = model_manager.get_all_model_paths()

    rf_path = model_paths.get('rf')
    if not rf_path:
        print("❌ No RF model path found")
        return None

    print(f"RF model path: {rf_path}")

    # Load RF model
    rf_model_file = Path(rf_path) / 'rf_model.joblib'
    if not rf_model_file.exists():
        print(f"❌ RF model file not found: {rf_model_file}")
        return None

    print(f"Loading RF model from: {rf_model_file}")
    rf_model = joblib.load(rf_model_file)

    # Try multiple methods to get feature names (in order of preference)
    rf_features = None

    # Method 1: Check for feature_columns.json (NEW: most reliable)
    feature_columns_file = Path(rf_path) / 'feature_columns.json'
    if feature_columns_file.exists():
        print(f"✅ Found feature_columns.json: {feature_columns_file}")
        with open(feature_columns_file, 'r') as f:
            rf_features = json.load(f)
            print(f"✅ Loaded {len(rf_features)} RF features from feature_columns.json")
            return rf_features

    # Method 2: Check for metadata.json (legacy)
    metadata_file = Path(rf_path) / 'metadata.json'
    if metadata_file.exists():
        print(f"✅ Found metadata.json: {metadata_file}")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            if 'feature_columns' in metadata:
                rf_features = metadata['feature_columns']
                print(f"✅ Loaded {len(rf_features)} RF features from metadata.json")
                return rf_features

    # Method 3: Try to get from sklearn model's feature_names_in_ attribute
    if hasattr(rf_model, 'feature_names_in_'):
        rf_features = list(rf_model.feature_names_in_)
        print(f"✅ Found {len(rf_features)} RF features from model.feature_names_in_")
        return rf_features

    # Method 4: Try to get from base_regressor (for wrapped models like CalibratedRegressor)
    if hasattr(rf_model, 'base_regressor') and hasattr(rf_model.base_regressor, 'feature_names_in_'):
        rf_features = list(rf_model.base_regressor.feature_names_in_)
        print(f"✅ Found {len(rf_features)} RF features from model.base_regressor.feature_names_in_")
        return rf_features

    # Method 5: Fallback - just get feature count
    if hasattr(rf_model, 'n_features_in_'):
        print(f"⚠️  RF model has {rf_model.n_features_in_} features but no feature names")
        rf_features = [f"feature_{i}" for i in range(rf_model.n_features_in_)]
        return rf_features

    print("❌ Could not extract RF feature names")
    return None


def extract_tabnet_features():
    """Extract features used by TabNet model"""
    print("\n" + "="*60)
    print("EXTRACTING TABNET MODEL FEATURES")
    print("="*60)

    model_manager = get_model_manager()
    model_paths = model_manager.get_all_model_paths()

    tabnet_path = model_paths.get('tabnet')
    if not tabnet_path:
        print("❌ No TabNet model path found")
        return None

    print(f"TabNet model path: {tabnet_path}")

    # Try multiple methods to get feature names (in order of preference)
    tabnet_features = None

    # Method 1: Check for tabnet_feature_columns.json (NEW: most reliable)
    tabnet_feature_file = Path(tabnet_path) / 'tabnet_feature_columns.json'
    if tabnet_feature_file.exists():
        print(f"✅ Found tabnet_feature_columns.json: {tabnet_feature_file}")
        with open(tabnet_feature_file, 'r') as f:
            tabnet_features = json.load(f)
            print(f"✅ Loaded {len(tabnet_features)} TabNet features from tabnet_feature_columns.json")
            return tabnet_features

    # Method 2: Check for metadata.json (legacy)
    metadata_file = Path(tabnet_path) / 'metadata.json'
    if metadata_file.exists():
        print(f"✅ Found metadata.json: {metadata_file}")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            if 'feature_columns' in metadata:
                tabnet_features = metadata['feature_columns']
                print(f"✅ Loaded {len(tabnet_features)} TabNet features from metadata.json")
                return tabnet_features

    # Method 3: Check for feature_columns.json (same as RF)
    feature_columns_file = Path(tabnet_path) / 'feature_columns.json'
    if feature_columns_file.exists():
        print(f"✅ Found feature_columns.json: {feature_columns_file}")
        with open(feature_columns_file, 'r') as f:
            tabnet_features = json.load(f)
            print(f"✅ Loaded {len(tabnet_features)} TabNet features from feature_columns.json")
            return tabnet_features

    print("❌ Could not find TabNet feature information")
    print("   Checked: tabnet_feature_columns.json, metadata.json, feature_columns.json")
    return None


def main():
    """Main extraction function"""
    print("\n🔍 EXTRACTING GENERAL MODEL FEATURES\n")

    # Extract RF features
    rf_features = extract_rf_features()

    # Extract TabNet features
    tabnet_features = extract_tabnet_features()

    # Save to JSON files
    output_dir = Path('.')

    if rf_features:
        rf_output = output_dir / 'rf_features.json'
        with open(rf_output, 'w') as f:
            json.dump({
                'model': 'RandomForest',
                'feature_count': len(rf_features),
                'features': rf_features
            }, f, indent=2)
        print(f"\n✅ Saved RF features to: {rf_output}")

    if tabnet_features:
        tabnet_output = output_dir / 'tabnet_features.json'
        with open(tabnet_output, 'w') as f:
            json.dump({
                'model': 'TabNet',
                'feature_count': len(tabnet_features),
                'features': tabnet_features
            }, f, indent=2)
        print(f"✅ Saved TabNet features to: {tabnet_output}")

    # Compare features
    if rf_features and tabnet_features:
        print("\n" + "="*60)
        print("FEATURE COMPARISON")
        print("="*60)

        rf_set = set(rf_features)
        tabnet_set = set(tabnet_features)

        common = rf_set & tabnet_set
        rf_only = rf_set - tabnet_set
        tabnet_only = tabnet_set - rf_set

        print(f"Common features: {len(common)}")
        print(f"RF only: {len(rf_only)}")
        print(f"TabNet only: {len(tabnet_only)}")

        # Save comparison
        comparison = {
            'rf_features': len(rf_features),
            'tabnet_features': len(tabnet_features),
            'common_features': len(common),
            'rf_only_count': len(rf_only),
            'tabnet_only_count': len(tabnet_only),
            'common_feature_list': sorted(list(common)),
            'rf_only_list': sorted(list(rf_only)),
            'tabnet_only_list': sorted(list(tabnet_only))
        }

        comparison_output = output_dir / 'feature_comparison.json'
        with open(comparison_output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✅ Saved feature comparison to: {comparison_output}")


if __name__ == '__main__':
    main()
