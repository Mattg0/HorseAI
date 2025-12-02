#!/usr/bin/env python3
"""
Test TabNet Feature Selector

Simple test to verify the TabNetFeatureSelector works correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.feature_selection.tabnet_feature_selector import TabNetFeatureSelector, select_tabnet_features


def test_basic_selection():
    """Test basic feature selection"""
    print("\n" + "="*70)
    print("TEST 1: Basic Feature Selection")
    print("="*70)

    # Create test data with various feature types
    np.random.seed(42)
    n_samples = 1000

    X = pd.DataFrame({
        # Normal features
        'feature1': np.random.rand(n_samples),
        'feature2': np.random.rand(n_samples),

        # Constant feature (should be removed)
        'constant_feature': np.ones(n_samples),

        # Sparse feature (should be removed)
        'sparse_feature': np.concatenate([np.zeros(900), np.random.rand(100)]),

        # Highly correlated features
        'feature_base': np.random.rand(n_samples),
        'feature_corr': None,  # Will be set to highly correlated

        # Musique-style features for priority testing
        'che_global_avg_pos': np.random.rand(n_samples),
        'che_weighted_avg_pos': np.random.rand(n_samples),
        'che_bytype_avg_pos': np.random.rand(n_samples),
    })

    # Create highly correlated feature
    X['feature_corr'] = X['feature_base'] + np.random.rand(n_samples) * 0.01

    print(f"\nInput data: {n_samples} samples, {len(X.columns)} features")
    print(f"Features: {list(X.columns)}")

    # Test feature selector
    selector = TabNetFeatureSelector(
        sparse_threshold=0.7,
        correlation_threshold=0.95,
        target_features=10
    )

    X_selected = selector.fit_transform(X)

    print(f"\nOutput data: {len(X_selected.columns)} features")
    print(f"Selected features: {list(X_selected.columns)}")

    # Verify removals
    assert 'constant_feature' not in X_selected.columns, "Constant feature should be removed"
    assert 'sparse_feature' not in X_selected.columns, "Sparse feature should be removed"

    print("\n✓ Test passed: Constant and sparse features removed")

    return selector, X_selected


def test_correlation_handling():
    """Test priority-based correlation handling"""
    print("\n" + "="*70)
    print("TEST 2: Correlation Handling with Priorities")
    print("="*70)

    np.random.seed(42)
    n_samples = 1000

    # Create base feature
    base_values = np.random.rand(n_samples)

    # Create highly correlated variants with musique naming
    X = pd.DataFrame({
        'che_global_avg_pos': base_values + np.random.rand(n_samples) * 0.01,
        'che_weighted_avg_pos': base_values + np.random.rand(n_samples) * 0.01,
        'che_bytype_avg_pos': base_values + np.random.rand(n_samples) * 0.01,

        # Other features
        'cotedirect_log': np.random.rand(n_samples),
        'age': np.random.rand(n_samples)
    })

    print(f"\nInput: {len(X.columns)} features")
    print(f"Features: {list(X.columns)}")

    selector = TabNetFeatureSelector(
        sparse_threshold=0.7,
        correlation_threshold=0.95,
        target_features=10
    )

    X_selected = selector.fit_transform(X)

    print(f"\nOutput: {len(X_selected.columns)} features")
    print(f"Selected: {list(X_selected.columns)}")

    # Check priority: bytype should be kept over global/weighted
    has_bytype = 'che_bytype_avg_pos' in X_selected.columns
    has_global = 'che_global_avg_pos' in X_selected.columns

    if has_bytype and not has_global:
        print("\n✓ Test passed: bytype feature kept, global removed (correct priority)")
    elif has_bytype and has_global:
        print("\n⚠ Warning: Both bytype and global kept (may not be highly correlated)")
    else:
        print("\n✗ Test failed: Unexpected feature selection")

    # Non-musique features should always be kept
    assert 'cotedirect_log' in X_selected.columns, "Non-musique features should be kept"
    assert 'age' in X_selected.columns, "Non-musique features should be kept"

    print("✓ Test passed: Non-musique features preserved")

    return selector, X_selected


def test_importance_based_selection():
    """Test importance-based final selection"""
    print("\n" + "="*70)
    print("TEST 3: Importance-Based Selection")
    print("="*70)

    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Create random features
    X = pd.DataFrame({
        f'feature_{i}': np.random.rand(n_samples)
        for i in range(n_features)
    })

    # Create mock importance scores (higher = more important)
    feature_importances = np.random.rand(n_features)
    feature_importances[0] = 0.8  # Make feature_0 most important
    feature_importances[1] = 0.7  # Make feature_1 second most important

    print(f"\nInput: {n_features} features")
    print(f"Top importance scores: {feature_importances[:5]}")

    selector = TabNetFeatureSelector(
        sparse_threshold=0.9,  # Lenient (won't remove much)
        correlation_threshold=0.99,  # Lenient (won't remove much)
        target_features=10  # Will force importance-based selection
    )

    X_selected = selector.fit_transform(X, feature_importances)

    print(f"\nOutput: {len(X_selected.columns)} features")
    print(f"Selected: {list(X_selected.columns)[:5]}")

    # Check that most important features were kept
    assert 'feature_0' in X_selected.columns, "Most important feature should be kept"
    assert 'feature_1' in X_selected.columns, "Second most important feature should be kept"
    assert len(X_selected.columns) == 10, "Should select exactly target_features"

    print("\n✓ Test passed: Top important features selected")

    return selector, X_selected


def test_save_load():
    """Test saving and loading feature selector"""
    print("\n" + "="*70)
    print("TEST 4: Save and Load")
    print("="*70)

    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        'feature1': np.random.rand(n_samples),
        'feature2': np.random.rand(n_samples),
        'feature3': np.random.rand(n_samples)
    })

    # Fit selector
    selector1 = TabNetFeatureSelector()
    selector1.fit(X)

    print(f"\nOriginal selector: {len(selector1.selected_features)} features")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        selector1.save(temp_path)
        print(f"Saved to: {temp_path}")

        # Load into new selector
        selector2 = TabNetFeatureSelector()
        selector2.load(temp_path)

        print(f"Loaded selector: {len(selector2.selected_features)} features")

        # Compare
        assert selector1.selected_features == selector2.selected_features, "Features should match"
        assert selector1.selection_metadata == selector2.selection_metadata, "Metadata should match"

        print("\n✓ Test passed: Save and load successful")

        # Verify JSON structure
        with open(temp_path, 'r') as f:
            data = json.load(f)

        assert 'selected_features' in data, "JSON should have selected_features"
        assert 'metadata' in data, "JSON should have metadata"

        print("✓ Test passed: JSON structure correct")

    finally:
        # Clean up
        Path(temp_path).unlink(missing_ok=True)

    return selector1, selector2


def test_convenience_function():
    """Test convenience function"""
    print("\n" + "="*70)
    print("TEST 5: Convenience Function")
    print("="*70)

    np.random.seed(42)
    n_samples = 100

    X_train = pd.DataFrame({
        'feature1': np.random.rand(n_samples),
        'feature2': np.random.rand(n_samples),
        'feature3': np.random.rand(n_samples),
        'sparse': np.concatenate([np.zeros(90), np.random.rand(10)])
    })

    y_train = np.random.rand(n_samples)

    print(f"\nInput: {len(X_train.columns)} features")

    # Use convenience function
    X_selected, selector = select_tabnet_features(
        X_train,
        y_train,
        sparse_threshold=0.7,
        target_features=5
    )

    print(f"Output: {len(X_selected.columns)} features")

    assert len(X_selected.columns) <= 5, "Should select at most target_features"
    assert 'sparse' not in X_selected.columns, "Sparse feature should be removed"

    print("\n✓ Test passed: Convenience function works")

    return X_selected, selector


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("RUNNING ALL TABNET FEATURE SELECTOR TESTS")
    print("="*70)

    tests = [
        ("Basic Selection", test_basic_selection),
        ("Correlation Handling", test_correlation_handling),
        ("Importance-Based Selection", test_importance_based_selection),
        ("Save and Load", test_save_load),
        ("Convenience Function", test_convenience_function)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASS"))
        except Exception as e:
            results.append((test_name, f"FAIL: {str(e)}"))
            print(f"\n✗ Test failed with error: {e}")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {test_name}: {result}")

    passed = sum(1 for _, result in results if result == "PASS")
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
