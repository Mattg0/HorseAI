#!/usr/bin/env python3
"""
Performance Assessment Script for Quinte and General Models

Compares two horse racing prediction models:
- Quinte model: Specialized for Quinte+ races
- General model: All race types

Outputs:
- Text report: assessment_results/assessment_report.txt
- JSON data: assessment_results/assessment_data.json
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
DB_PATH = "data/hippique2.db"
QUINTE_PREDICTIONS_PATH = "predictions/quinte_predictions_all_2025-09-23_to_2025-10-26_20251026_143757.json"

# Model feature paths
QUINTE_RF_PATH = "models/2025-10-26/2years_120713_quinte_rf"
QUINTE_TABNET_PATH = "models/2025-10-26/2years_120713_quinte_tabnet"
GENERAL_RF_PATH = "models/2025-10-29/2years_200717"
GENERAL_TABNET_PATH = "models/2025-10-29/2years_200721"

OUTPUT_DIR = "assessment_results"


def load_general_predictions_from_db():
    """Load general model predictions from daily_race table"""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT comp, jour, hippo, prediction_results, actual_results
    FROM daily_race
    WHERE prediction_results IS NOT NULL
      AND actual_results IS NOT NULL
      AND actual_results <> 'pending'
    ORDER BY jour DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Parse JSON fields
    predictions_list = []
    for _, row in df.iterrows():
        pred_data = json.loads(row['prediction_results'])
        actual = row['actual_results']

        # Skip invalid results
        if not actual or actual.lower() in ['pending', 'null', 'none']:
            continue

        try:
            # Parse actual results string like "11-7-8-3-10-4-1"
            actual_order = actual.split('-')
            actual_positions = {int(num): pos+1 for pos, num in enumerate(actual_order)}
        except (ValueError, AttributeError):
            continue

        # Extract predictions
        for horse_pred in pred_data.get('predictions', []):
            numero = horse_pred['numero']
            predictions_list.append({
                'race_id': row['comp'],
                'race_date': row['jour'],
                'hippo': row['hippo'],
                'horse_numero': numero,
                'predicted_position': horse_pred['predicted_position'],
                'predicted_rank': horse_pred['predicted_rank'],
                'actual_position': actual_positions.get(numero, 999)
            })

    return pd.DataFrame(predictions_list)


def load_quinte_predictions_from_json():
    """Load quinte predictions from JSON file"""
    with open(QUINTE_PREDICTIONS_PATH) as f:
        data = json.load(f)

    # Get actual results from database
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT comp, actual_results FROM daily_race WHERE actual_results IS NOT NULL"
    actual_df = pd.read_sql(query, conn)
    conn.close()

    # Build actual results map
    actual_map = {}
    for _, row in actual_df.iterrows():
        actual_results = row['actual_results']
        # Skip non-numeric results like "pending"
        if not actual_results or actual_results.lower() in ['pending', 'null', 'none']:
            continue
        try:
            actual_order = actual_results.split('-')
            actual_map[row['comp']] = {int(num): pos+1 for pos, num in enumerate(actual_order)}
        except (ValueError, AttributeError):
            # Skip races with invalid result format
            continue

    # Parse JSON predictions
    predictions_list = []
    for race in data:
        comp = race['comp']
        actual_positions = actual_map.get(comp, {})

        for horse_pred in race.get('predictions', []):
            numero = horse_pred['numero']
            predictions_list.append({
                'race_id': comp,
                'race_date': race['jour'],
                'hippo': race['hippo'],
                'horse_numero': numero,
                'predicted_position': horse_pred['predicted_position'],
                'predicted_rank': horse_pred.get('predicted_rank', 0),
                'actual_position': actual_positions.get(numero, 999)
            })

    return pd.DataFrame(predictions_list)


def load_feature_lists():
    """Load feature lists from all 4 models: Quinte RF, Quinte TabNet, General RF, General TabNet"""

    def load_features_from_model(model_path, model_name):
        """Load features from a model directory"""
        model_path = Path(model_path)

        # Try feature_columns.json first
        feature_file = model_path / "feature_columns.json"
        if feature_file.exists():
            with open(feature_file) as f:
                return json.load(f)

        # Try tabnet_config.json for TabNet models
        tabnet_config = model_path / "tabnet_config.json"
        if tabnet_config.exists():
            with open(tabnet_config) as f:
                config = json.load(f)
                if 'feature_columns' in config:
                    return config['feature_columns']

        # Try tabnet_feature_columns.json
        tabnet_features = model_path / "tabnet_feature_columns.json"
        if tabnet_features.exists():
            with open(tabnet_features) as f:
                return json.load(f)

        print(f"  WARNING: Could not find features for {model_name} at {model_path}")
        return []

    print("\nLoading model features...")

    # Load Quinte RF features
    quinte_rf_features = load_features_from_model(QUINTE_RF_PATH, "Quinte RF")
    print(f"  Quinte RF: {len(quinte_rf_features)} features")

    # Load Quinte TabNet features
    quinte_tabnet_features = load_features_from_model(QUINTE_TABNET_PATH, "Quinte TabNet")
    print(f"  Quinte TabNet: {len(quinte_tabnet_features)} features")

    # Load General RF features
    general_rf_features = load_features_from_model(GENERAL_RF_PATH, "General RF")
    print(f"  General RF: {len(general_rf_features)} features")

    # Load General TabNet features
    general_tabnet_features = load_features_from_model(GENERAL_TABNET_PATH, "General TabNet")
    print(f"  General TabNet: {len(general_tabnet_features)} features")

    return quinte_rf_features, quinte_tabnet_features, general_rf_features, general_tabnet_features


def validate_predictions(df, model_name):
    """Validate and clean prediction data"""
    print(f"\n{model_name} - Data Validation:")
    print(f"  Total predictions: {len(df)}")
    print(f"  Unique races: {df['race_id'].nunique()}")

    # Filter out invalid actual positions (999 = DNF/not found)
    df_clean = df[df['actual_position'] < 50].copy()
    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"  Removed {removed} predictions with invalid actual positions")

    # Check for issues
    if df_clean['predicted_position'].isna().any():
        print(f"  WARNING: {df_clean['predicted_position'].isna().sum()} null predictions")

    return df_clean


def calculate_metrics(df, model_name):
    """Calculate performance metrics"""
    results = {}

    # Overall error metrics - use predicted_rank vs actual_position
    errors = np.abs(df['predicted_rank'] - df['actual_position'])
    results['mae'] = float(errors.mean())
    results['rmse'] = float(np.sqrt((errors ** 2).mean()))
    results['median_error'] = float(errors.median())

    # Per-race accuracy metrics
    race_results = []
    for race_id, race_df in df.groupby('race_id'):
        # Sort by predicted_rank to get predicted order
        race_df = race_df.sort_values('predicted_rank')
        predicted_top1 = set(race_df.head(1)['horse_numero'].tolist())
        predicted_top3 = set(race_df.head(3)['horse_numero'].tolist())
        predicted_top5 = set(race_df.head(5)['horse_numero'].tolist())

        # Sort by actual position to get actual order
        race_df_actual = race_df.sort_values('actual_position')
        actual_top1 = set(race_df_actual.head(1)['horse_numero'].tolist())
        actual_top3 = set(race_df_actual.head(3)['horse_numero'].tolist())
        actual_top5 = set(race_df_actual.head(5)['horse_numero'].tolist())

        race_results.append({
            'race_id': race_id,
            'winner_correct': len(predicted_top1 & actual_top1) > 0,
            'top3_exact': predicted_top3 == actual_top3,
            'top5_exact': predicted_top5 == actual_top5,
            'top3_overlap': len(predicted_top3 & actual_top3),
            'top5_overlap': len(predicted_top5 & actual_top5)
        })

    race_results_df = pd.DataFrame(race_results)

    results['n_races'] = len(race_results_df)
    results['n_predictions'] = len(df)
    results['winner_accuracy'] = float(race_results_df['winner_correct'].mean())
    results['top3_exact'] = float(race_results_df['top3_exact'].mean())
    results['top5_exact'] = float(race_results_df['top5_exact'].mean())
    results['avg_top3_overlap'] = float(race_results_df['top3_overlap'].mean())
    results['avg_top5_overlap'] = float(race_results_df['top5_overlap'].mean())

    return results


def compare_features(quinte_rf_features, quinte_tabnet_features, general_rf_features, general_tabnet_features):
    """Compare feature lists between all 4 models"""

    # Convert to sets for comparison
    quinte_rf_set = set(quinte_rf_features)
    quinte_tabnet_set = set(quinte_tabnet_features)
    general_rf_set = set(general_rf_features)
    general_tabnet_set = set(general_tabnet_features)

    comparison = {
        'counts': {
            'quinte_rf': len(quinte_rf_features),
            'quinte_tabnet': len(quinte_tabnet_features),
            'general_rf': len(general_rf_features),
            'general_tabnet': len(general_tabnet_features)
        }
    }

    # 1. RF Models Comparison: Quinte RF vs General RF
    comparison['rf_models'] = {
        'shared': sorted(list(quinte_rf_set & general_rf_set)),
        'shared_count': len(quinte_rf_set & general_rf_set),
        'in_quinte_only': sorted(list(quinte_rf_set - general_rf_set)),
        'in_quinte_only_count': len(quinte_rf_set - general_rf_set),
        'in_general_only': sorted(list(general_rf_set - quinte_rf_set)),
        'in_general_only_count': len(general_rf_set - quinte_rf_set)
    }

    # Analyze quinte-specific vs non-quinte features
    quinte_specific = [f for f in comparison['rf_models']['in_quinte_only'] if 'quinte' in f.lower()]
    non_quinte_missing = [f for f in comparison['rf_models']['in_quinte_only'] if 'quinte' not in f.lower()]
    comparison['rf_models']['quinte_specific_features'] = sorted(quinte_specific)
    comparison['rf_models']['quinte_specific_count'] = len(quinte_specific)
    comparison['rf_models']['non_quinte_missing_from_general'] = sorted(non_quinte_missing)
    comparison['rf_models']['non_quinte_missing_count'] = len(non_quinte_missing)

    # 2. TabNet Models Comparison: Quinte TabNet vs General TabNet
    comparison['tabnet_models'] = {
        'shared': sorted(list(quinte_tabnet_set & general_tabnet_set)),
        'shared_count': len(quinte_tabnet_set & general_tabnet_set),
        'in_quinte_only': sorted(list(quinte_tabnet_set - general_tabnet_set)),
        'in_quinte_only_count': len(quinte_tabnet_set - general_tabnet_set),
        'in_general_only': sorted(list(general_tabnet_set - quinte_tabnet_set)),
        'in_general_only_count': len(general_tabnet_set - quinte_tabnet_set)
    }

    # 3. Quinte Models: RF vs TabNet
    comparison['quinte_rf_vs_tabnet'] = {
        'shared': sorted(list(quinte_rf_set & quinte_tabnet_set)),
        'shared_count': len(quinte_rf_set & quinte_tabnet_set),
        'rf_only': sorted(list(quinte_rf_set - quinte_tabnet_set)),
        'rf_only_count': len(quinte_rf_set - quinte_tabnet_set),
        'tabnet_only': sorted(list(quinte_tabnet_set - quinte_rf_set)),
        'tabnet_only_count': len(quinte_tabnet_set - quinte_rf_set)
    }

    # 4. General Models: RF vs TabNet
    comparison['general_rf_vs_tabnet'] = {
        'shared': sorted(list(general_rf_set & general_tabnet_set)),
        'shared_count': len(general_rf_set & general_tabnet_set),
        'rf_only': sorted(list(general_rf_set - general_tabnet_set)),
        'rf_only_count': len(general_rf_set - general_tabnet_set),
        'tabnet_only': sorted(list(general_tabnet_set - general_rf_set)),
        'tabnet_only_count': len(general_tabnet_set - general_rf_set)
    }

    return comparison


def print_report(quinte_results, general_results, comparison):
    """Print text report to console"""
    print("\n" + "="*80)
    print("PERFORMANCE ASSESSMENT REPORT - ALL 4 MODELS")
    print("="*80)

    # Performance metrics
    print("\nQUINTE MODEL (Quinte+ races only)")
    print("-"*80)
    print(f"Races evaluated: {quinte_results['n_races']}")
    print(f"Total predictions: {quinte_results['n_predictions']}")
    print(f"Winner accuracy: {quinte_results['winner_accuracy']*100:.1f}%")
    print(f"Top 3 exact match: {quinte_results['top3_exact']*100:.1f}%")
    print(f"Top 5 exact match: {quinte_results['top5_exact']*100:.1f}%")
    print(f"Avg top 3 overlap: {quinte_results['avg_top3_overlap']:.2f} / 3")
    print(f"Avg top 5 overlap: {quinte_results['avg_top5_overlap']:.2f} / 5")
    print(f"MAE: {quinte_results['mae']:.3f}")
    print(f"RMSE: {quinte_results['rmse']:.3f}")
    print(f"Median error: {quinte_results['median_error']:.3f}")

    print("\nGENERAL MODEL (All race types)")
    print("-"*80)
    print(f"Races evaluated: {general_results['n_races']}")
    print(f"Total predictions: {general_results['n_predictions']}")
    print(f"Winner accuracy: {general_results['winner_accuracy']*100:.1f}%")
    print(f"Top 3 exact match: {general_results['top3_exact']*100:.1f}%")
    print(f"Top 5 exact match: {general_results['top5_exact']*100:.1f}%")
    print(f"Avg top 3 overlap: {general_results['avg_top3_overlap']:.2f} / 3")
    print(f"Avg top 5 overlap: {general_results['avg_top5_overlap']:.2f} / 5")
    print(f"MAE: {general_results['mae']:.3f}")
    print(f"RMSE: {general_results['rmse']:.3f}")
    print(f"Median error: {general_results['median_error']:.3f}")

    # Feature counts summary
    print("\n" + "="*80)
    print("FEATURE COUNTS - ALL 4 MODELS")
    print("="*80)
    counts = comparison['counts']
    print(f"Quinte RF:      {counts['quinte_rf']:3d} features")
    print(f"Quinte TabNet:  {counts['quinte_tabnet']:3d} features")
    print(f"General RF:     {counts['general_rf']:3d} features")
    print(f"General TabNet: {counts['general_tabnet']:3d} features")

    # 1. RF Models Comparison
    print("\n" + "="*80)
    print("COMPARISON 1: RF MODELS (Quinte RF vs General RF)")
    print("="*80)
    rf = comparison['rf_models']
    print(f"Shared features: {rf['shared_count']}")
    print(f"In Quinte RF only: {rf['in_quinte_only_count']}")
    print(f"  - Quinte-specific (expected): {rf['quinte_specific_count']}")
    print(f"  - Non-quinte missing from General: {rf['non_quinte_missing_count']}")
    print(f"In General RF only: {rf['in_general_only_count']}")

    if rf['quinte_specific_features']:
        print(f"\nQuinte-specific features (expected in Quinte only):")
        for feat in rf['quinte_specific_features'][:10]:
            print(f"  - {feat}")
        if len(rf['quinte_specific_features']) > 10:
            print(f"  ... and {len(rf['quinte_specific_features'])-10} more")

    if rf['non_quinte_missing_from_general']:
        print(f"\nNon-quinte features MISSING from General RF:")
        for feat in rf['non_quinte_missing_from_general'][:20]:
            print(f"  - {feat}")
        if len(rf['non_quinte_missing_from_general']) > 20:
            print(f"  ... and {len(rf['non_quinte_missing_from_general'])-20} more")

    # 2. TabNet Models Comparison
    print("\n" + "="*80)
    print("COMPARISON 2: TABNET MODELS (Quinte TabNet vs General TabNet)")
    print("="*80)
    tn = comparison['tabnet_models']
    print(f"Shared features: {tn['shared_count']}")
    print(f"In Quinte TabNet only: {tn['in_quinte_only_count']}")
    print(f"In General TabNet only: {tn['in_general_only_count']}")

    if tn['in_quinte_only_count'] > 0 and tn['in_quinte_only_count'] <= 20:
        print(f"\nFeatures in Quinte TabNet only:")
        for feat in tn['in_quinte_only']:
            print(f"  - {feat}")

    if tn['in_general_only_count'] > 0 and tn['in_general_only_count'] <= 20:
        print(f"\nFeatures in General TabNet only:")
        for feat in tn['in_general_only']:
            print(f"  - {feat}")

    # 3. Quinte: RF vs TabNet
    print("\n" + "="*80)
    print("COMPARISON 3: QUINTE MODELS (Quinte RF vs Quinte TabNet)")
    print("="*80)
    qcomp = comparison['quinte_rf_vs_tabnet']
    print(f"Shared features: {qcomp['shared_count']}")
    print(f"Quinte RF only: {qcomp['rf_only_count']}")
    print(f"Quinte TabNet only: {qcomp['tabnet_only_count']}")
    print(f"\nNote: TabNet typically uses an optimized subset of RF features")

    # 4. General: RF vs TabNet
    print("\n" + "="*80)
    print("COMPARISON 4: GENERAL MODELS (General RF vs General TabNet)")
    print("="*80)
    gcomp = comparison['general_rf_vs_tabnet']
    print(f"Shared features: {gcomp['shared_count']}")
    print(f"General RF only: {gcomp['rf_only_count']}")
    print(f"General TabNet only: {gcomp['tabnet_only_count']}")
    print(f"\nNote: TabNet typically uses an optimized subset of RF features")

    print("\n" + "="*80)


def save_results(quinte_results, general_results, comparison):
    """Save results to files"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Text report
    report_path = Path(OUTPUT_DIR) / "assessment_report.txt"
    with open(report_path, 'w') as f:
        f.write("PERFORMANCE ASSESSMENT REPORT - ALL 4 MODELS\n")
        f.write("="*80 + "\n\n")

        f.write("QUINTE MODEL (Quinte+ races only)\n")
        f.write("-"*80 + "\n")
        for key, value in sorted(quinte_results.items()):
            f.write(f"{key}: {value}\n")

        f.write("\nGENERAL MODEL (All race types)\n")
        f.write("-"*80 + "\n")
        for key, value in sorted(general_results.items()):
            f.write(f"{key}: {value}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("FEATURE COUNTS - ALL 4 MODELS\n")
        f.write("="*80 + "\n")
        counts = comparison['counts']
        f.write(f"Quinte RF:      {counts['quinte_rf']} features\n")
        f.write(f"Quinte TabNet:  {counts['quinte_tabnet']} features\n")
        f.write(f"General RF:     {counts['general_rf']} features\n")
        f.write(f"General TabNet: {counts['general_tabnet']} features\n")

        f.write("\n" + "="*80 + "\n")
        f.write("COMPARISON 1: RF MODELS (Quinte RF vs General RF)\n")
        f.write("="*80 + "\n")
        rf = comparison['rf_models']
        f.write(f"Shared features: {rf['shared_count']}\n")
        f.write(f"In Quinte RF only: {rf['in_quinte_only_count']}\n")
        f.write(f"  - Quinte-specific: {rf['quinte_specific_count']}\n")
        f.write(f"  - Non-quinte missing from General: {rf['non_quinte_missing_count']}\n")
        f.write(f"In General RF only: {rf['in_general_only_count']}\n\n")

        f.write("Quinte-specific features (expected):\n")
        for feat in rf['quinte_specific_features']:
            f.write(f"  {feat}\n")

        f.write("\nNon-quinte features MISSING from General RF:\n")
        for feat in rf['non_quinte_missing_from_general']:
            f.write(f"  {feat}\n")

        f.write("\nGeneral RF-only features:\n")
        for feat in rf['in_general_only'][:50]:
            f.write(f"  {feat}\n")
        if len(rf['in_general_only']) > 50:
            f.write(f"  ... and {len(rf['in_general_only']) - 50} more\n")

        f.write("\n" + "="*80 + "\n")
        f.write("COMPARISON 2: TABNET MODELS (Quinte TabNet vs General TabNet)\n")
        f.write("="*80 + "\n")
        tn = comparison['tabnet_models']
        f.write(f"Shared features: {tn['shared_count']}\n")
        f.write(f"In Quinte TabNet only: {tn['in_quinte_only_count']}\n")
        f.write(f"In General TabNet only: {tn['in_general_only_count']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("COMPARISON 3: QUINTE MODELS (RF vs TabNet)\n")
        f.write("="*80 + "\n")
        qcomp = comparison['quinte_rf_vs_tabnet']
        f.write(f"Shared: {qcomp['shared_count']}\n")
        f.write(f"Quinte RF only: {qcomp['rf_only_count']}\n")
        f.write(f"Quinte TabNet only: {qcomp['tabnet_only_count']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("COMPARISON 4: GENERAL MODELS (RF vs TabNet)\n")
        f.write("="*80 + "\n")
        gcomp = comparison['general_rf_vs_tabnet']
        f.write(f"Shared: {gcomp['shared_count']}\n")
        f.write(f"General RF only: {gcomp['rf_only_count']}\n")
        f.write(f"General TabNet only: {gcomp['tabnet_only_count']}\n")

    # JSON data - comprehensive dump
    data = {
        'performance': {
            'quinte_model': quinte_results,
            'general_model': general_results
        },
        'feature_comparison': comparison
    }

    json_path = Path(OUTPUT_DIR) / "assessment_data.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved:")
    print(f"  {report_path}")
    print(f"  {json_path}")


def main():
    """Main execution"""
    print("="*80)
    print("PERFORMANCE ASSESSMENT: ALL 4 MODELS")
    print("  1. Quinte RF")
    print("  2. Quinte TabNet")
    print("  3. General RF")
    print("  4. General TabNet")
    print("="*80)

    print("\nLoading predictions...")

    # Load predictions
    quinte_df = load_quinte_predictions_from_json()
    general_df = load_general_predictions_from_db()

    # Validate
    quinte_df = validate_predictions(quinte_df, "QUINTE")
    general_df = validate_predictions(general_df, "GENERAL")

    # Calculate metrics
    print("\nCalculating metrics...")
    quinte_results = calculate_metrics(quinte_df, "QUINTE")
    general_results = calculate_metrics(general_df, "GENERAL")

    # Load and compare features from all 4 models
    quinte_rf_features, quinte_tabnet_features, general_rf_features, general_tabnet_features = load_feature_lists()

    print("\nComparing features across all 4 models...")
    comparison = compare_features(
        quinte_rf_features, quinte_tabnet_features,
        general_rf_features, general_tabnet_features
    )

    # Print report
    print_report(quinte_results, general_results, comparison)

    # Save results
    save_results(quinte_results, general_results, comparison)

    print("\nAssessment complete.")


if __name__ == "__main__":
    main()
