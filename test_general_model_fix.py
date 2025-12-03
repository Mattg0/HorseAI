#!/usr/bin/env python3
"""
Test script to verify that the general model training fix will include all features.
This simulates the complete training pipeline after the fix.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import json
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from core.data_cleaning.feature_cleanup import FeatureCleaner
from core.orchestrators.feature_selector import ModelFeatureSelector
from utils.env_setup import AppConfig

def main():
    print("="*80)
    print("TESTING GENERAL MODEL TRAINING PIPELINE (After Fix)")
    print("="*80)

    # Initialize components (same as training)
    config = AppConfig()
    db_path = config.get_sqlite_dbpath('2years')
    orchestrator = FeatureEmbeddingOrchestrator(sqlite_path=db_path, verbose=False)

    print(f"\n1. Loading historical data from database...")

    # Load a small sample (same as training does)
    df_races = orchestrator.load_historical_data(limit=10, use_cache=False)
    expanded_df = orchestrator._expand_participants(df_races)

    print(f"   Loaded {len(expanded_df)} participants from {df_races.shape[0]} races")
    print(f"   Initial columns: {len(expanded_df.columns)}")

    # Check what's already in the data from database
    print(f"\n2. Checking features from database participants JSON:")
    initial_features = list(expanded_df.columns)
    print(f"   Has recence: {'recence' in initial_features}")
    print(f"   Has cotedirect: {'cotedirect' in initial_features}")
    print(f"   Has numero: {'numero' in initial_features}")

    che_features = [f for f in initial_features if f.startswith('che_')]
    joc_features = [f for f in initial_features if f.startswith('joc_')]
    print(f"   che_* features: {len(che_features)}")
    print(f"   joc_* features: {len(joc_features)}")

    # Apply the FIX: prepare_complete_dataset now calls FeatureCalculator
    print(f"\n3. Running orchestrator.prepare_complete_dataset() [WITH FIX]...")
    complete_df = orchestrator.prepare_complete_dataset(
        expanded_df,
        use_cache=False,
        use_temporal=True  # Same as training
    )

    print(f"   After preparation: {len(complete_df.columns)} columns")

    # Check features after FeatureCalculator
    after_features = list(complete_df.columns)
    print(f"\n4. Features after FeatureCalculator:")
    print(f"   Has recence: {'recence' in after_features}")
    print(f"   Has cotedirect: {'cotedirect' in after_features}")
    print(f"   Has numero: {'numero' in after_features}")

    che_features = [f for f in after_features if f.startswith('che_')]
    joc_features = [f for f in after_features if f.startswith('joc_')]
    print(f"   che_* features: {len(che_features)}")
    print(f"   joc_* features: {len(joc_features)}")

    # Apply transformations (same as training)
    print(f"\n5. Applying feature cleanup and transformations...")
    cleaner = FeatureCleaner()
    complete_df = cleaner.clean_features(complete_df)
    complete_df = cleaner.apply_transformations(complete_df)

    print(f"   After cleanup: {len(complete_df.columns)} columns")

    # Check final features
    final_features = list(complete_df.columns)
    print(f"\n6. Final features after transformations:")
    print(f"   Has recence: {'recence' in final_features}")
    print(f"   Has recence_log: {'recence_log' in final_features}")
    print(f"   Has cotedirect: {'cotedirect' in final_features}")
    print(f"   Has cotedirect_log: {'cotedirect_log' in final_features}")
    print(f"   Has numero: {'numero' in final_features}")

    che_features = [f for f in final_features if f.startswith('che_')]
    joc_features = [f for f in final_features if f.startswith('joc_')]
    couple_features = [f for f in final_features if 'couple' in f.lower()]

    print(f"   che_* features: {len(che_features)}")
    print(f"   joc_* features: {len(joc_features)}")
    print(f"   couple features: {len(couple_features)}")

    # Extract RF features (what model will actually use)
    print(f"\n7. Extracting RF features (what model will be trained on)...")
    X_rf, y_rf = orchestrator.extract_rf_features(complete_df)

    print(f"   RF training features: {len(X_rf.columns)}")

    # Check what RF model will use
    rf_features = list(X_rf.columns)
    print(f"\n8. RF Model will have:")
    print(f"   recence_log: {'recence_log' in rf_features}")
    print(f"   cotedirect_log: {'cotedirect_log' in rf_features}")
    print(f"   numero: {'numero' in rf_features}")

    che_rf = [f for f in rf_features if f.startswith('che_')]
    joc_rf = [f for f in rf_features if f.startswith('joc_')]

    print(f"   che_* features: {len(che_rf)}")
    if che_rf:
        print(f"      Sample: {che_rf[:5]}")

    print(f"   joc_* features: {len(joc_rf)}")
    if joc_rf:
        print(f"      Sample: {joc_rf[:5]}")

    # Compare to Quinte model
    print(f"\n9. Comparing to Quinte model:")
    with open('models/2025-10-20/2years_165822_quinte_rf/feature_columns.json') as f:
        quinte_features = set(json.load(f))

    rf_features_set = set(rf_features)

    missing = quinte_features - rf_features_set
    quinte_only = [f for f in missing if 'quinte' in f.lower()]
    non_quinte_missing = [f for f in missing if 'quinte' not in f.lower()]

    print(f"   Quinte model: {len(quinte_features)} features")
    print(f"   General model (after fix): {len(rf_features)} features")
    print(f"   Missing from General: {len(missing)}")
    print(f"      Quinte-specific (expected): {len(quinte_only)}")
    print(f"      Non-quinte missing (should be ~0): {len(non_quinte_missing)}")

    if non_quinte_missing and len(non_quinte_missing) <= 20:
        print(f"\n   Non-quinte features still missing:")
        for f in sorted(non_quinte_missing):
            print(f"      - {f}")

    print(f"\n{'='*80}")
    if len(non_quinte_missing) <= 10:
        print("✅ FIX VERIFIED: General model will have all core features after retraining!")
        print(f"   Only {len(quinte_only)} quinte-specific features missing (expected)")
    else:
        print(f"⚠️  WARNING: Still missing {len(non_quinte_missing)} non-quinte features")
        print("   May need additional fixes")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
