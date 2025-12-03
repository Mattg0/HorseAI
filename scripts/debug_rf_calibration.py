"""
Debug RF calibration to find where compression happens
"""

import sqlite3
import json
import pandas as pd
import numpy as np

def debug_rf_calibration():
    """
    Compare raw_rf_prediction vs predicted_position_rf to find compression point
    """

    conn = sqlite3.connect('data/hippique2.db')

    # Get races with predictions
    query = """
    SELECT comp, jour, dist, typec, partant, prediction_results, actual_results
    FROM daily_race
    WHERE prediction_results IS NOT NULL
      AND actual_results IS NOT NULL
      AND actual_results != 'pending'
      AND jour >= date('now', '-7 days')
    ORDER BY jour DESC
    LIMIT 5
    """

    races = pd.read_sql_query(query, conn)
    conn.close()

    print("="*80)
    print("RF CALIBRATION DEBUG")
    print("="*80)
    print(f"\nAnalyzing {len(races)} recent races...\n")

    for idx, race in races.iterrows():
        try:
            pred_data = json.loads(race['prediction_results'])
            predictions = pred_data['predictions']

            # Extract values
            raw_rf = [p.get('raw_rf_prediction', 0) for p in predictions]
            calibrated_rf = [p.get('predicted_position_rf', 0) for p in predictions]
            final_pred = [p.get('predicted_position', 0) for p in predictions]

            # Check if we have the data we need
            if raw_rf[0] == 0:
                print(f"⚠️  Race {race['comp']}: No raw_rf_prediction data")
                continue

            print(f"{'='*80}")
            print(f"Race {race['comp']} - {race['partant']} horses")
            print(f"{'='*80}")

            print(f"\n📊 RAW RF PREDICTIONS (before adaptive calibration):")
            print(f"   Range: {min(raw_rf):.2f} to {max(raw_rf):.2f}")
            print(f"   Span: {max(raw_rf) - min(raw_rf):.2f}")
            print(f"   Mean: {np.mean(raw_rf):.2f}")
            print(f"   Values: {[f'{x:.2f}' for x in sorted(raw_rf)[:5]]}...")

            print(f"\n📊 CALIBRATED RF (after adaptive calibration):")
            print(f"   Range: {min(calibrated_rf):.2f} to {max(calibrated_rf):.2f}")
            print(f"   Span: {max(calibrated_rf) - min(calibrated_rf):.2f}")
            print(f"   Mean: {np.mean(calibrated_rf):.2f}")
            print(f"   Values: {[f'{x:.2f}' for x in sorted(calibrated_rf)[:5]]}...")

            print(f"\n📊 FINAL BLENDED (after ensemble + competitive analysis):")
            print(f"   Range: {min(final_pred):.2f} to {max(final_pred):.2f}")
            print(f"   Span: {max(final_pred) - min(final_pred):.2f}")
            print(f"   Mean: {np.mean(final_pred):.2f}")

            # Calculate compression
            raw_span = max(raw_rf) - min(raw_rf)
            calib_span = max(calibrated_rf) - min(calibrated_rf)
            final_span = max(final_pred) - min(final_pred)

            if raw_span > 0:
                calib_compression = (1 - calib_span / raw_span) * 100
                final_compression = (1 - final_span / raw_span) * 100

                print(f"\n📉 COMPRESSION ANALYSIS:")
                print(f"   Adaptive calibrator compressed by: {calib_compression:.1f}%")
                print(f"   Total pipeline compressed by: {final_compression:.1f}%")

                if calib_compression > 50:
                    print(f"   ❌ ADAPTIVE CALIBRATOR IS THE PROBLEM!")
                elif final_compression - calib_compression > 30:
                    print(f"   ❌ BLENDING/COMPETITIVE ANALYSIS IS COMPRESSING!")
                else:
                    print(f"   ⚠️  Moderate compression at multiple stages")

            # Sample comparison
            print(f"\n🔍 SAMPLE COMPARISON (first 3 horses):")
            print(f"   {'Horse':<8} {'Raw RF':<12} {'Calibrated':<12} {'Final':<12} {'Actual'}")
            print(f"   {'-'*60}")

            actual_list = race['actual_results'].split('-')
            for i in range(min(3, len(predictions))):
                pred = predictions[i]
                numero = pred['numero']

                # Find actual position
                if str(numero) in actual_list:
                    actual_pos = actual_list.index(str(numero)) + 1
                else:
                    actual_pos = len(actual_list) + 1

                raw = pred.get('raw_rf_prediction', 0)
                calib = pred.get('predicted_position_rf', 0)
                final = pred.get('predicted_position', 0)

                print(f"   #{numero:<7} {raw:<12.2f} {calib:<12.2f} {final:<12.2f} {actual_pos}")

            print()

        except Exception as e:
            print(f"Error processing race {race['comp']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\nIf adaptive calibrator is compressing >50%:")
    print("  → The isotonic calibration is over-smoothing predictions")
    print("  → Check calibrator training data and parameters")
    print("  → Consider disabling adaptive calibration temporarily")
    print("\nIf raw RF predictions are already compressed:")
    print("  → RF model training issue or feature discrimination problem")
    print("  → Review model training logs and feature importance")


if __name__ == "__main__":
    debug_rf_calibration()
