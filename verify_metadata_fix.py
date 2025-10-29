#!/usr/bin/env python3
"""
Verification script to check if models have feature metadata saved.
Run this on both old and new models to see the difference.
"""

import json
from pathlib import Path
from utils.model_manager import get_model_manager

def check_model_metadata(model_path):
    """Check what metadata exists for a model"""
    print(f"\nChecking: {model_path}")
    print("-" * 60)

    model_dir = Path(model_path)
    if not model_dir.exists():
        print("  ❌ Model directory not found")
        return

    # Check for various files
    files_to_check = {
        'rf_model.joblib': 'RF Model',
        'feature_columns.json': 'Feature Columns (NEW)',
        'tabnet_feature_columns.json': 'TabNet Features (NEW)',
        'metadata.json': 'Metadata (Legacy)',
        'model_config.json': 'Model Config',
        'feature_engineer.joblib': 'Feature Engineer'
    }

    found_files = []
    for filename, description in files_to_check.items():
        filepath = model_dir / filename
        if filepath.exists():
            found_files.append(filename)
            print(f"  ✅ {description}: {filename}")

            # Show feature count if it's a feature file
            if filename in ['feature_columns.json', 'tabnet_feature_columns.json', 'metadata.json']:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        print(f"     → {len(data)} features")
                    elif isinstance(data, dict) and 'feature_columns' in data:
                        print(f"     → {len(data['feature_columns'])} features")
                except Exception as e:
                    print(f"     ⚠️  Could not read: {e}")

            # Show feature_count from config
            if filename == 'model_config.json':
                try:
                    with open(filepath, 'r') as f:
                        config = json.load(f)
                    if 'feature_count' in config:
                        print(f"     → feature_count: {config['feature_count']}")
                except Exception:
                    pass
        else:
            print(f"  ❌ {description}: missing")

    # Verdict
    print()
    has_new_format = 'feature_columns.json' in found_files
    has_old_format = 'metadata.json' in found_files

    if has_new_format:
        print("  ✅ STATUS: New format (feature_columns.json exists)")
    elif has_old_format:
        print("  ⚠️  STATUS: Old format (only metadata.json)")
    else:
        print("  ❌ STATUS: No feature metadata (extract script will fail)")


def main():
    """Check all configured models"""
    print("="*80)
    print("MODEL METADATA VERIFICATION")
    print("="*80)

    # Get model paths from config
    model_manager = get_model_manager()

    # Check RF model
    print("\n🔍 CHECKING RF MODEL")
    rf_path = model_manager.get_all_model_paths().get('rf')
    if rf_path:
        check_model_metadata(rf_path)
    else:
        print("\n  ❌ No RF model configured")

    # Check TabNet model
    print("\n🔍 CHECKING TABNET MODEL")
    tabnet_path = model_manager.get_all_model_paths().get('tabnet')
    if tabnet_path:
        check_model_metadata(tabnet_path)
    else:
        print("\n  ⚠️  No TabNet model configured")

    # Check Quinte RF model (for comparison)
    print("\n🔍 CHECKING QUINTE RF MODEL (for comparison)")
    quinte_rf_path = model_manager.get_all_model_paths().get('rf_quinte')
    if quinte_rf_path:
        check_model_metadata(quinte_rf_path)
    else:
        print("\n  ⚠️  No Quinte RF model configured")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nAfter retraining with the fix, you should see:")
    print("  ✅ feature_columns.json")
    print("  ✅ tabnet_feature_columns.json (if TabNet trained)")
    print("  ✅ model_config.json with feature_count")
    print("\nOld models will show:")
    print("  ❌ No feature_columns.json")
    print("  ❌ No feature metadata")
    print()

if __name__ == "__main__":
    main()
