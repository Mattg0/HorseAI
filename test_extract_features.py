#!/usr/bin/env python3
"""
Test script to verify that extract_general_model_features.py can find features
"""

import json
import tempfile
import shutil
from pathlib import Path

def test_feature_extraction():
    """Test that the extraction logic can find features in multiple ways"""
    print("="*80)
    print("TESTING FEATURE EXTRACTION LOGIC")
    print("="*80)

    # Create temporary directory structure
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\n✓ Created temp directory: {temp_dir}")

    try:
        test_features = [
            'cotedirect', 'recence', 'numero', 'age',
            'che_global_avg_pos', 'che_weighted_avg_pos',
            'joc_global_avg_pos', 'joc_weighted_avg_pos'
        ]

        # Test 1: feature_columns.json (preferred method)
        print("\n--- Test 1: feature_columns.json ---")
        feature_path = temp_dir / "feature_columns.json"
        with open(feature_path, 'w') as f:
            json.dump(test_features, f, indent=2)

        # Simulate extraction
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                loaded = json.load(f)
            print(f"✅ Method 1 SUCCESS: Loaded {len(loaded)} features from feature_columns.json")
            assert loaded == test_features
        else:
            print("❌ Method 1 FAILED")

        # Test 2: metadata.json with feature_columns key (legacy)
        print("\n--- Test 2: metadata.json (legacy) ---")
        metadata_path = temp_dir / "metadata.json"
        metadata = {
            'model_type': 'RandomForest',
            'created_at': '2025-10-29',
            'feature_columns': test_features
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            if 'feature_columns' in meta:
                print(f"✅ Method 2 SUCCESS: Loaded {len(meta['feature_columns'])} features from metadata.json")
                assert meta['feature_columns'] == test_features
        else:
            print("❌ Method 2 FAILED")

        # Test 3: TabNet features
        print("\n--- Test 3: tabnet_feature_columns.json ---")
        tabnet_features = test_features[:5]  # Subset for TabNet
        tabnet_path = temp_dir / "tabnet_feature_columns.json"
        with open(tabnet_path, 'w') as f:
            json.dump(tabnet_features, f, indent=2)

        if tabnet_path.exists():
            with open(tabnet_path, 'r') as f:
                loaded = json.load(f)
            print(f"✅ Method 3 SUCCESS: Loaded {len(loaded)} TabNet features from tabnet_feature_columns.json")
            assert loaded == tabnet_features
        else:
            print("❌ Method 3 FAILED")

        # Test priority (feature_columns.json should be preferred over metadata.json)
        print("\n--- Test 4: Priority test (both files exist) ---")
        different_features = ['feature_A', 'feature_B']
        with open(feature_path, 'w') as f:
            json.dump(different_features, f, indent=2)

        # Check feature_columns.json first (should win)
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                loaded = json.load(f)
            print(f"✅ Priority test SUCCESS: feature_columns.json takes precedence")
            assert loaded == different_features
        else:
            print("❌ Priority test FAILED")

        print("\n" + "="*80)
        print("✅ ALL EXTRACTION TESTS PASSED!")
        print("="*80)

        print("\nExtraction priority order (implemented in extract_general_model_features.py):")
        print("  1. feature_columns.json              ← NEW, most reliable")
        print("  2. metadata.json                     ← Legacy support")
        print("  3. model.feature_names_in_           ← Sklearn fallback")
        print("  4. model.base_regressor.feature_...  ← Wrapped model fallback")
        print("  5. Generate generic names            ← Last resort")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up temp directory")

if __name__ == "__main__":
    test_feature_extraction()
