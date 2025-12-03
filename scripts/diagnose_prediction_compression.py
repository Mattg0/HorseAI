"""
Diagnose RF and TabNet prediction compression issues
"""

import sqlite3
import json
import pandas as pd
import numpy as np

def diagnose_compression():
    """Analyze prediction compression across multiple races"""

    conn = sqlite3.connect('data/hippique2.db')

    # Get races with predictions and results
    query = """
    SELECT comp, jour, dist, typec, partant, prediction_results, actual_results
    FROM daily_race
    WHERE prediction_results IS NOT NULL
      AND actual_results IS NOT NULL
      AND actual_results != 'pending'
      AND jour >= date('now', '-7 days')
    ORDER BY jour DESC
    LIMIT 20
    """

    races = pd.read_sql_query(query, conn)
    conn.close()

    print("="*80)
    print("PREDICTION COMPRESSION ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(races)} recent races...")

    all_stats = []

    for idx, race in races.iterrows():
        try:
            pred_data = json.loads(race['prediction_results'])
            predictions = pred_data['predictions']

            # Extract all prediction types
            pred_positions = [p.get('predicted_position', 0) for p in predictions]
            pred_rf = [p.get('predicted_position_rf', 0) for p in predictions]
            pred_tabnet = [p.get('predicted_position_tabnet', 0) for p in predictions]

            # Calculate statistics
            stats = {
                'race_id': race['comp'],
                'field_size': race['partant'],
                'pred_pos_min': min(pred_positions),
                'pred_pos_max': max(pred_positions),
                'pred_pos_span': max(pred_positions) - min(pred_positions),
                'pred_rf_min': min(pred_rf),
                'pred_rf_max': max(pred_rf),
                'pred_rf_span': max(pred_rf) - min(pred_rf),
                'pred_tabnet_min': min(pred_tabnet) if pred_tabnet[0] > 0 else None,
                'pred_tabnet_max': max(pred_tabnet) if pred_tabnet[0] > 0 else None,
                'pred_tabnet_span': max(pred_tabnet) - min(pred_tabnet) if pred_tabnet[0] > 0 else None,
            }

            all_stats.append(stats)

        except Exception as e:
            print(f"Error processing race {race['comp']}: {e}")
            continue

    if not all_stats:
        print("No races to analyze!")
        return

    df = pd.DataFrame(all_stats)

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\n📊 PREDICTED_POSITION (Final blended predictions)")
    print(f"   Average span: {df['pred_pos_span'].mean():.2f} (Expected: ~{df['field_size'].mean() - 1:.1f})")
    print(f"   Min span: {df['pred_pos_span'].min():.2f}")
    print(f"   Max span: {df['pred_pos_span'].max():.2f}")
    print(f"   Typical range: {df['pred_pos_min'].mean():.2f} to {df['pred_pos_max'].mean():.2f}")

    if df['pred_pos_span'].mean() < (df['field_size'].mean() / 3):
        print(f"\n   ❌ SEVERE COMPRESSION! Span should be ~{df['field_size'].mean() - 1:.0f}, but is only {df['pred_pos_span'].mean():.1f}")

    print("\n📊 RF PREDICTIONS")
    print(f"   Average span: {df['pred_rf_span'].mean():.2f} (Expected: ~{df['field_size'].mean() - 1:.1f})")
    print(f"   Min span: {df['pred_rf_span'].min():.2f}")
    print(f"   Max span: {df['pred_rf_span'].max():.2f}")
    print(f"   Typical range: {df['pred_rf_min'].mean():.2f} to {df['pred_rf_max'].mean():.2f}")

    if df['pred_rf_span'].mean() < (df['field_size'].mean() / 3):
        print(f"\n   ❌ RF COMPRESSION! Span should be ~{df['field_size'].mean() - 1:.0f}, but is only {df['pred_rf_span'].mean():.1f}")
        print(f"   This indicates:")
        print(f"      - Isotonic calibration is compressing output, OR")
        print(f"      - RF model training issue, OR")
        print(f"      - Insufficient feature discrimination")

    print("\n📊 TABNET PREDICTIONS")
    tabnet_df = df.dropna(subset=['pred_tabnet_span'])
    if len(tabnet_df) > 0:
        print(f"   Average span: {tabnet_df['pred_tabnet_span'].mean():.2f} (Expected: ~{df['field_size'].mean() - 1:.1f})")
        print(f"   Min span: {tabnet_df['pred_tabnet_span'].min():.2f}")
        print(f"   Max span: {tabnet_df['pred_tabnet_span'].max():.2f}")
        print(f"   Typical range: {tabnet_df['pred_tabnet_min'].mean():.2f} to {tabnet_df['pred_tabnet_max'].mean():.2f}")

        if tabnet_df['pred_tabnet_max'].mean() > 30:
            print(f"\n   ⚠️  TABNET SCALE ISSUE! Values too high (avg max: {tabnet_df['pred_tabnet_max'].mean():.1f})")
            print(f"   TabNet may not be properly calibrated/scaled")
    else:
        print("   No TabNet predictions found")

    # Detail view of worst compression cases
    print(f"\n{'='*80}")
    print("WORST COMPRESSION CASES")
    print("="*80)

    worst_cases = df.nsmallest(5, 'pred_pos_span')

    print(f"\n{'Race':<12} {'Field':<7} {'Pred Span':<12} {'RF Span':<10} {'Status'}")
    print('-'*60)
    for _, row in worst_cases.iterrows():
        expected_span = row['field_size'] - 1
        compression_pct = (1 - row['pred_pos_span'] / expected_span) * 100
        status = f"{compression_pct:.0f}% compressed"
        print(f"{row['race_id']:<12} {row['field_size']:<7} {row['pred_pos_span']:<12.2f} {row['pred_rf_span']:<10.2f} {status}")

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print("="*80)

    avg_compression = (1 - df['pred_pos_span'].mean() / (df['field_size'].mean() - 1)) * 100

    if avg_compression > 50:
        print("\n🔧 CRITICAL: Predictions severely compressed")
        print("\nTo fix:")
        print("  1. Check adaptive calibrator - it may be over-smoothing")
        print("     → Disable in predict_quinte.py:567 temporarily")
        print("  2. Check RF model training data range")
        print("     → Review training outputs in logs")
        print("  3. Add logging to prediction pipeline:")
        print("     → Log raw_rf_preds before calibration")
        print("     → Log after isotonic calibration")
        print("     → Log after blending")
    elif avg_compression > 30:
        print("\n⚠️  MODERATE: Predictions moderately compressed")
        print("\nInvestigate:")
        print("  - Isotonic calibration settings")
        print("  - RF model confidence/uncertainty")
        print("  - Feature engineering discrimination power")
    else:
        print("\n✓ Predictions show reasonable spread")

    return df


if __name__ == "__main__":
    diagnose_compression()
