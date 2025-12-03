#!/usr/bin/env python3
"""
Example: Predict with TabNet using Automatic Feature Selection

This example shows how to make predictions with TabNet models that use
feature selection. The same features selected during training are automatically
applied during prediction.

Usage:
    python examples/predict_with_tabnet_feature_selection.py --model-path models/.../general_tabnet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from race_prediction.tabnet_prediction_helpers import (
    load_tabnet_with_selector,
    predict_race_tabnet,
    get_tabnet_feature_info,
    compare_rf_tabnet_predictions
)
from utils.env_setup import get_sqlite_dbpath


def example_single_race_prediction(model_path: str, db_path: str):
    """Example: Predict a single race using TabNet"""

    print("\n" + "="*70)
    print("EXAMPLE: SINGLE RACE PREDICTION WITH TABNET")
    print("="*70)

    # 1. Load some race data from database
    import sqlite3
    conn = sqlite3.connect(db_path)
    query = """
    SELECT *
    FROM daily_race
    WHERE prediction_results IS NOT NULL
    LIMIT 1
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if len(df) == 0:
        print("No races found in database")
        return

    race_data = df.iloc[0:10]  # First 10 horses

    print(f"\n1. Loaded race data: {len(race_data)} horses")

    # 2. Get model feature info
    print(f"\n2. Loading model from: {model_path}")
    feature_info = get_tabnet_feature_info(model_path)

    print(f"   Selected features: {feature_info['selected_count']}")
    print(f"   Original features: {feature_info['original_count']}")
    print(f"   Removed sparse: {feature_info['removed_sparse']}")
    print(f"   Removed correlated: {feature_info['removed_correlated']}")

    # 3. Make predictions
    print("\n3. Making predictions...")
    predictions = predict_race_tabnet(
        race_data=race_data,
        model_path=model_path,
        model_type='general',
        db_path=db_path
    )

    print(f"\n4. Predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"   Horse {i}: {pred:.2f}")

    # Rank horses by prediction
    ranked_indices = np.argsort(predictions)
    print(f"\n5. Predicted order (best to worst):")
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"   {rank}. Horse {idx+1} (predicted: {predictions[idx]:.2f})")


def example_compare_rf_tabnet(rf_model_path: str,
                                tabnet_model_path: str,
                                db_path: str):
    """Example: Compare RF and TabNet predictions"""

    print("\n" + "="*70)
    print("EXAMPLE: COMPARE RF AND TABNET PREDICTIONS")
    print("="*70)

    # Load some race data
    import sqlite3
    conn = sqlite3.connect(db_path)
    query = """
    SELECT *
    FROM daily_race
    WHERE prediction_results IS NOT NULL
    LIMIT 1
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if len(df) == 0:
        print("No races found in database")
        return

    race_data = df.iloc[0:10]

    print(f"\n1. Loaded race data: {len(race_data)} horses")

    # Compare predictions
    print("\n2. Comparing RF and TabNet predictions...")
    comparison = compare_rf_tabnet_predictions(
        race_data=race_data,
        rf_model_path=rf_model_path,
        tabnet_model_path=tabnet_model_path,
        model_type='general',
        db_path=db_path
    )

    print(f"\n3. Comparison Results:")
    print(f"   RF features used:     {comparison['rf_features_used']}")
    print(f"   TabNet features used: {comparison['tabnet_features_used']}")
    print(f"   Feature reduction:    {(1 - comparison['tabnet_features_used']/comparison['rf_features_used'])*100:.1f}%")
    print(f"   Correlation:          {comparison['correlation']:.3f}")
    print(f"   Mean abs difference:  {comparison['mean_absolute_difference']:.3f}")

    print(f"\n4. Side-by-side predictions:")
    print(f"   {'Horse':<10} {'RF':>8} {'TabNet':>8} {'Diff':>8}")
    print(f"   {'-'*40}")

    for i, (rf_pred, tabnet_pred) in enumerate(zip(comparison['rf_predictions'],
                                                     comparison['tabnet_predictions']), 1):
        diff = abs(rf_pred - tabnet_pred)
        print(f"   Horse {i:<4} {rf_pred:>8.2f} {tabnet_pred:>8.2f} {diff:>8.2f}")


def example_feature_inspection(model_path: str):
    """Example: Inspect which features were selected"""

    print("\n" + "="*70)
    print("EXAMPLE: INSPECT SELECTED FEATURES")
    print("="*70)

    print(f"\nModel path: {model_path}")

    # Get feature info
    info = get_tabnet_feature_info(model_path)

    if 'error' in info:
        print(f"\nError: {info['error']}")
        return

    print(f"\nFeature Selection Summary:")
    print(f"  Original features:    {info['original_count']}")
    print(f"  Selected features:    {info['selected_count']}")
    print(f"  Reduction:            {(1 - info['selected_count']/info['original_count'])*100:.1f}%")

    print(f"\nRemoved Features:")
    print(f"  Constant features:    {info['removed_constant']}")
    print(f"  Sparse features:      {info['removed_sparse']}")
    print(f"  Correlated features:  {info['removed_correlated']}")

    print(f"\nSelection Thresholds:")
    print(f"  Sparse threshold:     {info['sparse_threshold']}")
    print(f"  Correlation threshold: {info['correlation_threshold']}")

    print(f"\nSelected Features ({info['selected_count']}):")
    for i, feature in enumerate(info['selected_features'], 1):
        print(f"  {i:3d}. {feature}")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description='Example predictions with TabNet feature selection'
    )
    parser.add_argument(
        '--example',
        choices=['predict', 'compare', 'inspect'],
        default='predict',
        help='Which example to run'
    )
    parser.add_argument(
        '--model-path',
        required=True,
        help='Path to TabNet model directory'
    )
    parser.add_argument(
        '--rf-model-path',
        help='Path to RF model (for compare example)'
    )
    parser.add_argument(
        '--db-path',
        help='Path to database (optional, uses config default)'
    )

    args = parser.parse_args()

    # Get database path
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = get_sqlite_dbpath('2years')

    # Run selected example
    if args.example == 'predict':
        example_single_race_prediction(args.model_path, db_path)

    elif args.example == 'compare':
        if not args.rf_model_path:
            print("Error: --rf-model-path required for compare example")
            return
        example_compare_rf_tabnet(args.rf_model_path, args.model_path, db_path)

    elif args.example == 'inspect':
        example_feature_inspection(args.model_path)


if __name__ == "__main__":
    main()
