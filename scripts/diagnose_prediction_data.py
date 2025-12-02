"""
Diagnostic script to investigate prediction data in database
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta

def diagnose_prediction_storage(db_path='data/hippique2.db'):
    """
    Investigate what's actually stored in prediction_results
    """

    print("="*80)
    print("PREDICTION DATA DIAGNOSTIC")
    print("="*80)

    conn = sqlite3.connect(db_path)

    # Get a sample race with predictions and results
    query = """
    SELECT
        comp,
        jour,
        dist,
        typec,
        partant,
        prediction_results,
        actual_results
    FROM daily_race
    WHERE prediction_results IS NOT NULL
      AND actual_results IS NOT NULL
      AND actual_results != ''
    ORDER BY jour DESC
    LIMIT 1
    """

    cursor = conn.execute(query)
    row = cursor.fetchone()

    if not row:
        print("No races with predictions and results found!")
        return

    comp, jour, dist, typec, partant, pred_json, actual_results = row

    print(f"\n📊 Sample Race:")
    print(f"   Race ID: {comp}")
    print(f"   Date: {jour}")
    print(f"   Distance: {dist}m, Type: {typec}, Field: {partant}")
    print(f"   Actual results: {actual_results}")

    # Parse prediction JSON
    pred_data = json.loads(pred_json)

    print(f"\n📋 Prediction JSON Structure:")
    print(f"   Keys: {list(pred_data.keys())}")

    if 'predictions' in pred_data:
        predictions = pred_data['predictions']
        print(f"   Number of predictions: {len(predictions)}")

        if len(predictions) > 0:
            print(f"\n🔍 First prediction entry:")
            first = predictions[0]
            print(f"   Keys: {list(first.keys())}")
            print(f"   Full entry: {json.dumps(first, indent=4)}")

            print(f"\n📈 All predictions for this race:")
            print(f"   {'Numero':<8} {'Field Name':<25} {'Value':<15} {'Type'}")
            print(f"   {'-'*70}")

            # Collect all values to analyze
            prediction_values = []
            for pred in predictions[:5]:  # Show first 5
                numero = pred.get('numero', '?')

                # Check different possible field names
                fields_to_check = [
                    'predicted_position',
                    'final_prediction',
                    'predicted_rank',
                    'ensemble_prediction',
                    'calibrated_prediction'
                ]

                for field in fields_to_check:
                    if field in pred:
                        value = pred[field]
                        value_type = type(value).__name__
                        print(f"   {numero:<8} {field:<25} {value:<15} {value_type}")
                        if field in ['predicted_position', 'final_prediction']:
                            prediction_values.append(value)

            if len(predictions) > 5:
                print(f"   ... and {len(predictions) - 5} more horses")

            # Analyze all prediction values
            if prediction_values:
                print(f"\n📊 Statistics for prediction values:")
                values_array = pd.Series(prediction_values)
                print(f"   Count: {len(values_array)}")
                print(f"   Mean: {values_array.mean():.2f}")
                print(f"   Min: {values_array.min():.2f}")
                print(f"   Max: {values_array.max():.2f}")
                print(f"   Std: {values_array.std():.2f}")

                print(f"\n   Sample values: {list(values_array.head(10).values)}")

                # Diagnose what type of values these are
                print(f"\n🎯 Diagnosis:")
                mean_val = values_array.mean()
                max_val = values_array.max()

                if mean_val < 2 and max_val < 3:
                    print(f"   ❌ These look like RANKS (1, 2, 3...)")
                    print(f"      But mean is {mean_val:.2f}, which is too low!")
                    print(f"      Expected mean ~{partant/2:.1f} for field of {partant}")
                elif 1 <= mean_val <= 20 and max_val <= 25:
                    print(f"   ✓ These look like continuous predictions or ranks")
                    print(f"      Mean {mean_val:.2f} is reasonable for field of {partant}")
                elif mean_val > 20:
                    print(f"   ❌ These values are TOO HIGH!")
                    print(f"      Mean {mean_val:.2f} >> expected ~{partant/2:.1f}")
                    print(f"      This is NOT position data!")
                else:
                    print(f"   ⚠️  Unclear what these values represent")

    # Check actual results parsing
    actual_list = actual_results.split('-')
    print(f"\n🏁 Actual Results:")
    print(f"   Arrival order: {actual_list[:5]}{'...' if len(actual_list) > 5 else ''}")
    print(f"   Number of finishers: {len(actual_list)}")

    # Now let's check what the calibration script would extract
    print(f"\n🔧 What calibration script extracts:")
    try:
        predictions = pred_data.get('predictions', [])
        actual_results_list = actual_results.split('-')

        for i, pred in enumerate(predictions[:3]):
            numero = int(pred['numero'])
            predicted_pos = pred.get('predicted_position', pred.get('final_prediction', 99))

            # Find actual position
            if str(numero) in actual_results_list:
                actual_pos = actual_results_list.index(str(numero)) + 1
            else:
                actual_pos = len(actual_results_list) + 1

            error = predicted_pos - actual_pos

            print(f"   Horse #{numero}: predicted={predicted_pos:.2f}, actual={actual_pos}, error={error:+.2f}")
    except Exception as e:
        print(f"   Error: {e}")

    conn.close()

    print(f"\n{'='*80}")
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/hippique2.db'
    diagnose_prediction_storage(db_path)
