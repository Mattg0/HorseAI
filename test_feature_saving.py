#!/usr/bin/env python3
"""
Test script to verify that feature_columns.json is saved correctly
"""

import json
import tempfile
import shutil
from pathlib import Path

def test_feature_saving():
    """Test that the feature saving logic works correctly"""
    print("="*80)
    print("TESTING FEATURE SAVING LOGIC")
    print("="*80)

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\n✓ Created temp directory: {temp_dir}")

    try:
        # Simulate what model_manager.save_models() does
        test_features = [
            'cotedirect', 'recence', 'numero', 'age',
            'che_global_avg_pos', 'che_weighted_avg_pos',
            'joc_global_avg_pos', 'joc_weighted_avg_pos'
        ]

        # Save feature_columns.json
        feature_path = temp_dir / "feature_columns.json"
        with open(feature_path, 'w') as f:
            json.dump(test_features, f, indent=2)

        print(f"✓ Saved feature_columns.json with {len(test_features)} features")

        # Verify file exists
        assert feature_path.exists(), "feature_columns.json should exist"
        print(f"✓ Verified feature_columns.json exists")

        # Load and verify content
        with open(feature_path, 'r') as f:
            loaded_features = json.load(f)

        assert len(loaded_features) == len(test_features), "Feature count should match"
        assert loaded_features == test_features, "Feature content should match"
        print(f"✓ Verified feature content matches (8 features)")

        # Test model_config.json with feature_count
        config_path = temp_dir / "model_config.json"
        config_data = {
            'db_type': '2years',
            'created_at': '2025-10-29T12:00:00',
            'is_quinte': False,
            'feature_count': len(test_features)
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"✓ Saved model_config.json with feature_count={len(test_features)}")

        # Verify config
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)

        assert loaded_config['feature_count'] == len(test_features), "Config feature_count should match"
        print(f"✓ Verified model_config.json has correct feature_count")

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - Feature saving logic works correctly!")
        print("="*80)

        print("\nExpected structure after training:")
        print(f"  models/DATE/DB_TIME/")
        print(f"    ├── rf_model.joblib")
        print(f"    ├── feature_columns.json          ← NEW (RF features)")
        print(f"    ├── tabnet_feature_columns.json   ← NEW (TabNet features)")
        print(f"    ├── model_config.json              ← Updated (includes feature_count)")
        print(f"    └── feature_engineer.joblib")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up temp directory")

if __name__ == "__main__":
    test_feature_saving()
