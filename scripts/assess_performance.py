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
QUINTE_MODEL_FEATURES_PATH = "models/2025-10-26/2years_120713_quinte_rf/feature_columns.json"
GENERAL_FEATURES_EXTRACT_PATH = "X_general_features.json"
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
    """Load feature lists from saved models"""
    # Quinte features
    with open(QUINTE_MODEL_FEATURES_PATH) as f:
        quinte_features = json.load(f)

    # General features - extract from X_general_features.json
    with open(GENERAL_FEATURES_EXTRACT_PATH) as f:
        general_data = json.load(f)

    # Get feature names from first horse in first race
    if general_data['races'] and general_data['races'][0]['X_features']:
        first_horse = general_data['races'][0]['X_features'][0]
        general_features = list(first_horse.keys())
    else:
        general_features = []

    return quinte_features, general_features


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


def compare_features(quinte_features, general_features):
    """Compare feature lists between models"""
    quinte_set = set(quinte_features)
    general_set = set(general_features)

    comparison = {
        'quinte_count': len(quinte_features),
        'general_count': len(general_features),
        'shared_count': len(quinte_set & general_set),
        'quinte_only_count': len(quinte_set - general_set),
        'general_only_count': len(general_set - quinte_set),
        'quinte_only': sorted(list(quinte_set - general_set)),
        'general_only': sorted(list(general_set - quinte_set)),
        'shared': sorted(list(quinte_set & general_set))
    }

    return comparison


def print_report(quinte_results, general_results, feature_comparison):
    """Print text report to console"""
    print("\n" + "="*80)
    print("PERFORMANCE ASSESSMENT REPORT")
    print("="*80)

    print("\nQUINTE MODEL")
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

    print("\nGENERAL MODEL")
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

    print("\nFEATURE COMPARISON")
    print("-"*80)
    print(f"Quinte features: {feature_comparison['quinte_count']}")
    print(f"General features: {feature_comparison['general_count']}")
    print(f"Shared features: {feature_comparison['shared_count']}")
    print(f"Quinte-only features: {feature_comparison['quinte_only_count']}")
    print(f"General-only features: {feature_comparison['general_only_count']}")

    if feature_comparison['quinte_only']:
        print(f"\nTop Quinte-specific features:")
        for feat in feature_comparison['quinte_only'][:15]:
            print(f"  - {feat}")
        if len(feature_comparison['quinte_only']) > 15:
            print(f"  ... and {len(feature_comparison['quinte_only'])-15} more")

    print("\n" + "="*80)


def save_results(quinte_results, general_results, feature_comparison):
    """Save results to files"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Text report
    report_path = Path(OUTPUT_DIR) / "assessment_report.txt"
    with open(report_path, 'w') as f:
        f.write("PERFORMANCE ASSESSMENT REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("QUINTE MODEL\n")
        f.write("-"*80 + "\n")
        for key, value in sorted(quinte_results.items()):
            f.write(f"{key}: {value}\n")

        f.write("\nGENERAL MODEL\n")
        f.write("-"*80 + "\n")
        for key, value in sorted(general_results.items()):
            f.write(f"{key}: {value}\n")

        f.write("\nFEATURE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"Quinte features: {feature_comparison['quinte_count']}\n")
        f.write(f"General features: {feature_comparison['general_count']}\n")
        f.write(f"Shared: {feature_comparison['shared_count']}\n")
        f.write(f"Quinte-only: {feature_comparison['quinte_only_count']}\n")
        f.write(f"General-only: {feature_comparison['general_only_count']}\n\n")

        f.write("Quinte-specific features:\n")
        for feat in feature_comparison['quinte_only']:
            f.write(f"  {feat}\n")

        f.write("\nGeneral-specific features:\n")
        for feat in feature_comparison['general_only']:
            f.write(f"  {feat}\n")

    # JSON data
    data = {
        'quinte_model': quinte_results,
        'general_model': general_results,
        'feature_comparison': {
            'quinte_count': feature_comparison['quinte_count'],
            'general_count': feature_comparison['general_count'],
            'shared_count': feature_comparison['shared_count'],
            'quinte_only_count': feature_comparison['quinte_only_count'],
            'general_only_count': feature_comparison['general_only_count'],
            'quinte_only': feature_comparison['quinte_only'],
            'general_only': feature_comparison['general_only']
        }
    }

    json_path = Path(OUTPUT_DIR) / "assessment_data.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved:")
    print(f"  {report_path}")
    print(f"  {json_path}")


def main():
    """Main execution"""
    print("Loading predictions...")

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

    # Load and compare features
    print("\nComparing features...")
    quinte_features, general_features = load_feature_lists()
    feature_comparison = compare_features(quinte_features, general_features)

    # Print report
    print_report(quinte_results, general_results, feature_comparison)

    # Save results
    save_results(quinte_results, general_results, feature_comparison)

    print("\nAssessment complete.")


if __name__ == "__main__":
    main()
