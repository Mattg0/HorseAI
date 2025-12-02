"""
Model Calibration Script

Main script for detecting biases and calibrating predictions.
Supports both initial calibration and incremental updates.

Usage:
    # Initial calibration
    python scripts/calibrate_models.py

    # Check and update if needed
    python scripts/calibrate_models.py --check

    # Force update
    python scripts/calibrate_models.py --force
"""

import sys
import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.calibration.bias_detector import BiasDetector
from core.calibration.prediction_calibrator import PredictionCalibrator
from core.calibration.incremental_updater import IncrementalCalibrationUpdater


def load_recent_predictions(db_path, model_type='general', days=90):
    """
    Load recent predictions with actual results

    Args:
        db_path: Path to SQLite database
        model_type: 'general' or 'quinte'
        days: Number of days to look back

    Returns:
        DataFrame with predictions and results
    """

    print(f"\nLoading {model_type} predictions from last {days} days...")

    conn = sqlite3.connect(db_path)

    if model_type == 'quinte':
        # Load from quinte_predictions table
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # NOTE: For quinté, final_prediction should be the UNCALIBRATED prediction
        # If bias calibration has been applied, this will be the pre-calibration value
        query = f"""
        SELECT
            qp.race_id,
            qp.horse_number as numero,
            qp.horse_id,
            qp.final_prediction as predicted_position,
            dr.actual_results,
            dr.dist,
            dr.typec,
            dr.partant
        FROM quinte_predictions qp
        JOIN daily_race dr ON qp.race_id = dr.comp
        WHERE qp.race_date >= '{cutoff_date}'
          AND dr.actual_results IS NOT NULL
          AND dr.actual_results != ''
          AND dr.actual_results != 'pending'
        ORDER BY qp.race_date DESC
        """

        df = pd.read_sql_query(query, conn)

        # Parse actual_results to get actual_position
        def get_actual_position(row):
            try:
                actual_str = row['actual_results']

                # Additional safety check for invalid results format
                if not actual_str or actual_str in ['pending', 'null', 'None', '']:
                    return None

                results = actual_str.split('-')

                # Validate that results contains numbers
                if not results or len(results) < 3:  # Need at least top 3 finishers
                    return None

                if str(row['numero']) in results:
                    return results.index(str(row['numero'])) + 1
                else:
                    return len(results) + 1  # Finished outside top positions
            except:
                return None

        df['actual_position'] = df.apply(get_actual_position, axis=1)
        df = df.dropna(subset=['actual_position'])

        # Add cotedirect (odds) - load from participants
        odds_query = """
        SELECT dr.comp as race_id, dr.participants
        FROM daily_race dr
        WHERE dr.participants IS NOT NULL
        """
        odds_df = pd.read_sql_query(odds_query, conn)

        # Parse participants JSON to get odds
        def extract_odds(row):
            try:
                participants = json.loads(row['participants'])
                odds_map = {p['numero']: p.get('cotedirect', 5.0) for p in participants}
                return odds_map
            except:
                return {}

        odds_df['odds_map'] = odds_df.apply(extract_odds, axis=1)

        # Merge odds
        df = df.merge(odds_df[['race_id', 'odds_map']], on='race_id', how='left')
        df['cotedirect'] = df.apply(lambda row: row['odds_map'].get(row['numero'], 5.0), axis=1)
        df = df.drop(columns=['odds_map', 'actual_results'])

        # Rename dist to distance for consistency with bias detector
        df = df.rename(columns={'dist': 'distance'})

    else:
        # Load from race_predictions or daily_race.prediction_results
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        query = f"""
        SELECT
            dr.comp as race_id,
            dr.jour as race_date,
            dr.prediction_results,
            dr.actual_results,
            dr.dist,
            dr.typec,
            dr.partant,
            dr.participants
        FROM daily_race dr
        WHERE dr.jour >= '{cutoff_date}'
          AND dr.prediction_results IS NOT NULL
          AND dr.actual_results IS NOT NULL
          AND dr.actual_results != ''
          AND dr.actual_results != 'pending'
        ORDER BY dr.jour DESC
        """

        races_df = pd.read_sql_query(query, conn)

        # Expand predictions
        all_predictions = []
        for _, race in races_df.iterrows():
            try:
                pred_data = json.loads(race['prediction_results'])
                predictions = pred_data.get('predictions', [])
                actual_results = race['actual_results'].split('-')

                # Parse participants for odds
                participants = json.loads(race['participants']) if race['participants'] else []
                odds_map = {p['numero']: p.get('cotedirect', 5.0) for p in participants}

                for pred in predictions:
                    numero = int(pred['numero'])

                    # Find actual position
                    if str(numero) in actual_results:
                        actual_position = actual_results.index(str(numero)) + 1
                    else:
                        actual_position = len(actual_results) + 1

                    # IMPORTANT: Use uncalibrated predictions for calibration training
                    # Try multiple fields in order of preference:
                    # 1. predicted_position_uncalibrated (if bias calibration was applied)
                    # 2. predicted_position (blended prediction, usually uncalibrated in historical data)
                    # 3. predicted_position_rf (raw RF prediction as fallback)
                    predicted_pos = pred.get('predicted_position_uncalibrated',
                                             pred.get('predicted_position',
                                                     pred.get('predicted_position_rf',
                                                             pred.get('final_prediction', 99))))

                    all_predictions.append({
                        'race_id': race['race_id'],
                        'numero': numero,
                        'predicted_position': predicted_pos,
                        'actual_position': actual_position,
                        'cotedirect': odds_map.get(numero, 5.0),
                        'distance': race['dist'],
                        'typec': race['typec'],
                        'partant': race['partant']
                    })

            except Exception as e:
                print(f"Error parsing race {race['race_id']}: {e}")
                continue

        df = pd.DataFrame(all_predictions)

    conn.close()

    print(f"Loaded {len(df)} predictions")
    print(f"  Races: {df['race_id'].nunique()}")
    print(f"  Date range: {df['race_date'].min() if 'race_date' in df.columns else 'N/A'} to {df['race_date'].max() if 'race_date' in df.columns else 'N/A'}")

    # CRITICAL VALIDATION: Filter out races with garbage predicted_position values
    # This happens when older predictions stored wrong values in predicted_position_uncalibrated
    if len(df) > 0:
        # Calculate mean predicted_position per race
        race_stats = df.groupby('race_id')['predicted_position'].agg(['mean', 'max', 'count']).reset_index()

        # Flag races where predicted positions are suspiciously high
        # Normal races have mean ~6-10, max ~20
        bad_races = race_stats[
            (race_stats['mean'] > 15) |  # Mean way too high
            (race_stats['max'] > 35)     # Max way too high
        ]['race_id'].tolist()

        if len(bad_races) > 0:
            print(f"\n⚠️  Filtering out {len(bad_races)} races with garbage prediction values")
            print(f"   These races have mean predicted_position > 15 or max > 35")
            print(f"   Sample bad races: {bad_races[:5]}")

            # Remove bad races
            df_before = len(df)
            df = df[~df['race_id'].isin(bad_races)]
            df_after = len(df)

            print(f"   Removed {df_before - df_after} predictions from {len(bad_races)} races")
            print(f"   Remaining: {df_after} predictions from {df['race_id'].nunique()} races")

    # VALIDATION: Check for bad data
    if len(df) > 0:
        pred_mean = df['predicted_position'].mean()
        pred_max = df['predicted_position'].max()
        actual_mean = df['actual_position'].mean()
        actual_max = df['actual_position'].max()

        print(f"\nData quality check:")
        print(f"  Predicted positions: mean={pred_mean:.2f}, max={pred_max:.2f}")
        print(f"  Actual positions: mean={actual_mean:.2f}, max={actual_max:.2f}")

        # Flag suspicious data
        if pred_mean > 20 or pred_max > 40:
            print(f"\n⚠️  WARNING: Predicted positions look wrong!")
            print(f"     Expected mean ~6-10, max ~25, but got mean={pred_mean:.2f}, max={pred_max:.2f}")
            print(f"     This will create invalid calibration. Check prediction_results format in database.")

        if actual_mean > 10 or actual_max > 25:
            print(f"\n⚠️  WARNING: Actual positions look wrong!")
            print(f"     Expected mean ~8, max ~20, but got mean={actual_mean:.2f}, max={actual_max:.2f}")

    return df


def calibrate_model(db_path, model_type='general', validation_split=0.2):
    """
    Build calibration for a model

    Args:
        db_path: Path to SQLite database
        model_type: 'general' or 'quinte'
        validation_split: Fraction of data to use for validation
    """

    print("\n" + "="*80)
    print(f"CALIBRATING {model_type.upper()} MODEL")
    print("="*80)

    # Load predictions
    predictions_df = load_recent_predictions(db_path, model_type, days=90)

    if len(predictions_df) < 100:
        print(f"\nInsufficient data for calibration ({len(predictions_df)} predictions)")
        print("Need at least 100 predictions with results")
        return None, None

    # Validate data quality (abort if predictions are garbage)
    pred_mean = predictions_df['predicted_position'].mean()
    if pred_mean > 20:
        print(f"\n❌ ABORTING: Invalid prediction data detected!")
        print(f"   Predicted positions have mean={pred_mean:.2f}, which is WAY too high")
        print(f"   Expected mean ~6-10 for typical races")
        print(f"\n   This usually means:")
        print(f"   1. Wrong field being extracted from prediction_results JSON")
        print(f"   2. Predictions haven't been stored correctly in database")
        print(f"   3. Database contains test/garbage data")
        print(f"\n   Fix the data source before running calibration!")
        return None, None

    # Split into train/validation
    split_idx = int(len(predictions_df) * (1 - validation_split))
    train_df = predictions_df.iloc[:split_idx]
    val_df = predictions_df.iloc[split_idx:]

    print(f"\nTrain set: {len(train_df)} predictions")
    print(f"Validation set: {len(val_df)} predictions")

    # DEBUG: Show sample of training data before bias detection
    print(f"\n{'='*80}")
    print("DEBUG: Sample of training data (first 10 rows)")
    print(f"{'='*80}")
    sample_cols = ['race_id', 'numero', 'predicted_position', 'actual_position', 'cotedirect', 'distance', 'typec', 'partant']
    available_cols = [col for col in sample_cols if col in train_df.columns]
    print(train_df[available_cols].head(10).to_string())

    print(f"\nTraining data statistics:")
    print(f"  predicted_position: mean={train_df['predicted_position'].mean():.2f}, "
          f"median={train_df['predicted_position'].median():.2f}, "
          f"min={train_df['predicted_position'].min():.2f}, "
          f"max={train_df['predicted_position'].max():.2f}")
    print(f"  actual_position: mean={train_df['actual_position'].mean():.2f}, "
          f"median={train_df['actual_position'].median():.2f}, "
          f"min={train_df['actual_position'].min():.2f}, "
          f"max={train_df['actual_position'].max():.2f}")
    print(f"  Error (predicted - actual): mean={train_df['predicted_position'].sub(train_df['actual_position']).mean():.2f}")
    print(f"{'='*80}\n")

    # Detect biases
    detector = BiasDetector()
    biases = detector.analyze_biases(train_df)

    if not biases:
        print("\nNo significant biases detected")
        print("Calibration not needed at this time")
        return None, None

    # Build calibration
    calibrator = PredictionCalibrator()
    calibrator.fit(biases, validation_df=val_df)

    # Check if calibration helps or hurts
    validation_results = calibrator.calibrations.get('validation', {})
    improvement = validation_results.get('improvement', 0)

    output_path = Path(project_root) / f'models/calibration/{model_type}_calibration.json'

    if improvement < 0:
        # Calibration makes things WORSE - don't save it!
        print(f"\n" + "="*80)
        print(f"⚠️  CALIBRATION REJECTED")
        print(f"="*80)
        print(f"Calibration would DECREASE performance by {abs(improvement):.3f} MAE ({validation_results.get('improvement_pct', 0):.1f}%)")
        print(f"\nThis usually means:")
        print(f"  1. Model is already well-calibrated")
        print(f"  2. Detected biases don't generalize to new data")
        print(f"  3. Sample size too small for reliable calibration")
        print(f"\nNO calibration file will be saved.")
        print(f"Models will continue using uncalibrated predictions.")

        # Delete existing calibration file if it exists
        if output_path.exists():
            backup_path = output_path.with_suffix('.json.rejected')
            output_path.rename(backup_path)
            print(f"\nExisting calibration moved to: {backup_path.name}")

        return None, None

    # Save calibration (only if it helps)
    calibrator.save(output_path)

    # Generate report
    report_path = Path(project_root) / f'models/calibration/{model_type}_calibration_report.txt'
    with open(report_path, 'w') as f:
        f.write("CALIBRATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Training samples: {len(train_df)}\n")
        f.write(f"Validation samples: {len(val_df)}\n\n")

        f.write("DETECTED BIASES:\n")
        f.write("-"*80 + "\n")
        for bias_type, bias_info in biases.items():
            if bias_info.get('significant'):
                f.write(f"\n{bias_type.upper()}:\n")
                f.write(f"  Severity: {bias_info['severity']}\n")
                f.write(f"  Impact: {bias_info['impact']:.3f} MAE\n")
                f.write(f"  Description: {bias_info['description']}\n")

        if 'validation' in calibrator.calibrations:
            val_results = calibrator.calibrations['validation']
            f.write("\n\nVALIDATION RESULTS:\n")
            f.write("-"*80 + "\n")
            f.write(f"MAE before calibration: {val_results['mae_before']:.3f}\n")
            f.write(f"MAE after calibration: {val_results['mae_after']:.3f}\n")
            f.write(f"Improvement: {val_results['improvement']:+.3f} ({val_results['improvement_pct']:+.1f}%)\n")

    print(f"\nReport saved to: {report_path}")

    return calibrator, detector


def check_and_update_calibration(db_path, model_type='general', force=False):
    """
    Check if calibration needs updating and update if needed

    Args:
        db_path: Path to SQLite database
        model_type: 'general' or 'quinte'
        force: Force update even if not needed
    """

    print("\n" + "="*80)
    print(f"CALIBRATION HEALTH CHECK: {model_type.upper()}")
    print("="*80)

    # Load existing calibration
    calibration_path = Path(project_root) / f'models/calibration/{model_type}_calibration.json'

    if not calibration_path.exists():
        print("\nNo existing calibration found. Running initial calibration...")
        calibrator, detector = calibrate_model(db_path, model_type)

        # Return consistent 3-tuple format
        if calibrator is None:
            return None, None, {'reason': 'Insufficient data for calibration'}
        else:
            return calibrator, detector, {'reason': 'Initial calibration created'}

    calibrator = PredictionCalibrator(calibration_path)
    detector = BiasDetector()

    # Load recent predictions (last 30 days)
    recent_df = load_recent_predictions(db_path, model_type, days=30)

    if len(recent_df) == 0:
        print("\nNo recent predictions found")
        return calibrator, detector, {'reason': 'No recent data'}

    # Apply existing calibration
    recent_df = calibrator.transform(recent_df)

    # Check if update needed
    updater = IncrementalCalibrationUpdater(
        calibrator, detector,
        lookback_days=30,
        min_samples=50
    )

    should_update, metrics = updater.check_if_update_needed(recent_df)

    if should_update or force:
        print("\nUpdating calibration...")
        updater.update(recent_df, save_path=calibration_path)
    else:
        print("\nCalibration healthy, no update needed")

    return calibrator, detector, metrics


def main():
    """
    Main calibration workflow
    """

    parser = argparse.ArgumentParser(description='Model Calibration System')
    parser.add_argument('--check', action='store_true', help='Check and update if needed')
    parser.add_argument('--force', action='store_true', help='Force update')
    parser.add_argument('--model', choices=['general', 'quinte', 'both'], default='both', help='Model to calibrate')
    parser.add_argument('--db', default='data/hippique2.db', help='Database path')

    args = parser.parse_args()

    db_path = Path(project_root) / args.db

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    print("\n" + "="*80)
    print("MODEL CALIBRATION WORKFLOW")
    print("="*80)

    models_to_process = ['general', 'quinte'] if args.model == 'both' else [args.model]

    for model_type in models_to_process:
        print(f"\n{'='*80}")
        print(f"{model_type.upper()} MODEL")
        print("="*80)

        try:
            if args.check or args.force:
                calibrator, detector, metrics = check_and_update_calibration(db_path, model_type, force=args.force)
                print(f"\nResult: {metrics.get('reason', 'Unknown')}")
            else:
                calibrator, detector = calibrate_model(db_path, model_type)
                if calibrator:
                    print(f"\nCalibration complete for {model_type}")

        except Exception as e:
            print(f"\nError calibrating {model_type}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("CALIBRATION WORKFLOW COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
