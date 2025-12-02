#!/usr/bin/env python3
"""
Update Adaptive Calibrators from Race Results

This script updates the isotonic regression calibrators based on recent
race predictions and actual results. Run this daily after race results are available.

Usage:
    python scripts/update_calibrators.py --days 30 --verbose
    python scripts/update_calibrators.py --days 60  # Use more historical data
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.env_setup import AppConfig, get_sqlite_dbpath
from model_training.regressions.adaptive_calibrator import (
    AdaptiveCalibratorManager,
    fetch_predictions_with_results
)


def main():
    parser = argparse.ArgumentParser(description='Update adaptive calibrators from race results')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days of history to use (default: 30)')
    parser.add_argument('--min-samples', type=int, default=100,
                       help='Minimum number of samples required (default: 100)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without saving')
    parser.add_argument('--calibrator-dir', type=str, default='models/calibrators',
                       help='Directory to store calibrators')

    args = parser.parse_args()

    # Initialize
    config = AppConfig(args.config)
    db_type = config._config.base.active_db
    db_path = get_sqlite_dbpath(db_type)

    print("=" * 70)
    print("ADAPTIVE CALIBRATOR UPDATE")
    print("=" * 70)
    print(f"Database: {db_path}")
    print(f"History window: {args.days} days")
    print(f"Minimum samples: {args.min_samples}")
    print(f"Calibrator directory: {args.calibrator_dir}")
    if args.dry_run:
        print("DRY RUN MODE - No changes will be saved")
    print()

    # Initialize calibrator manager
    manager = AdaptiveCalibratorManager(calibrator_dir=args.calibrator_dir)

    # Fetch predictions with results
    print(f"Loading predictions from last {args.days} days...")
    try:
        df = fetch_predictions_with_results(db_path, days=args.days)
    except Exception as e:
        print(f"❌ Error fetching predictions: {e}")
        return 1

    if df.empty:
        print("❌ No predictions with results found!")
        print(f"   Make sure:")
        print(f"   1. Predictions have been made (raw_rf_prediction, raw_tabnet_prediction)")
        print(f"   2. Race results have been updated (actual_result)")
        print(f"   3. Data exists within the last {args.days} days")
        return 1

    print(f"✅ Found {len(df)} predictions with results")
    print(f"   Date range: {df['prediction_date'].min()} to {df['prediction_date'].max()}")
    print(f"   Unique races: {df['race_id'].nunique()}")
    print()

    # Check for required columns
    required_cols = ['raw_rf_prediction', 'raw_tabnet_prediction', 'actual_result']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   These columns should be in the race_predictions table.")
        print(f"   Run predictions first to populate raw prediction columns.")
        return 1

    # Update RF calibrator
    print("-" * 70)
    print("RF CALIBRATOR")
    print("-" * 70)

    try:
        rf_mask = ~(pd.isna(df['raw_rf_prediction']) | pd.isna(df['actual_result']))
        rf_predictions = df.loc[rf_mask, 'raw_rf_prediction'].values
        rf_actuals = df.loc[rf_mask, 'actual_result'].values

        print(f"Training samples: {len(rf_predictions)}")

        if len(rf_predictions) < args.min_samples:
            print(f"⚠️  Not enough samples ({len(rf_predictions)} < {args.min_samples})")
            print(f"   Skipping RF calibrator update")
            rf_calibrator = None
            rf_metadata = None
        else:
            rf_calibrator, rf_metadata = manager.update_calibrator(
                rf_predictions, rf_actuals, 'rf'
            )

            print(f"✅ RF Calibrator trained successfully")
            print(f"   Raw MAE: {rf_metadata['raw_mae']:.3f}")
            print(f"   Calibrated MAE: {rf_metadata['calibrated_mae']:.3f}")
            print(f"   Improvement: {rf_metadata['mae_improvement_pct']:.2f}%")
            print(f"   Raw RMSE: {rf_metadata['raw_rmse']:.3f}")
            print(f"   Calibrated RMSE: {rf_metadata['calibrated_rmse']:.3f}")
            print(f"   RMSE Improvement: {rf_metadata['rmse_improvement_pct']:.2f}%")

            if not args.dry_run:
                manager.save_calibrator(rf_calibrator, 'rf', rf_metadata)
            else:
                print(f"   (DRY RUN - not saved)")

    except Exception as e:
        print(f"❌ Error updating RF calibrator: {e}")
        rf_calibrator = None
        rf_metadata = None

    print()

    # Update TabNet calibrator
    print("-" * 70)
    print("TABNET CALIBRATOR")
    print("-" * 70)

    try:
        tabnet_mask = ~(pd.isna(df['raw_tabnet_prediction']) | pd.isna(df['actual_result']))
        tabnet_predictions = df.loc[tabnet_mask, 'raw_tabnet_prediction'].values
        tabnet_actuals = df.loc[tabnet_mask, 'actual_result'].values

        print(f"Training samples: {len(tabnet_predictions)}")

        if len(tabnet_predictions) < args.min_samples:
            print(f"⚠️  Not enough samples ({len(tabnet_predictions)} < {args.min_samples})")
            print(f"   Skipping TabNet calibrator update")
            tabnet_calibrator = None
            tabnet_metadata = None
        else:
            tabnet_calibrator, tabnet_metadata = manager.update_calibrator(
                tabnet_predictions, tabnet_actuals, 'tabnet'
            )

            print(f"✅ TabNet Calibrator trained successfully")
            print(f"   Raw MAE: {tabnet_metadata['raw_mae']:.3f}")
            print(f"   Calibrated MAE: {tabnet_metadata['calibrated_mae']:.3f}")
            print(f"   Improvement: {tabnet_metadata['mae_improvement_pct']:.2f}%")
            print(f"   Raw RMSE: {tabnet_metadata['raw_rmse']:.3f}")
            print(f"   Calibrated RMSE: {tabnet_metadata['calibrated_rmse']:.3f}")
            print(f"   RMSE Improvement: {tabnet_metadata['rmse_improvement_pct']:.2f}%")

            if not args.dry_run:
                manager.save_calibrator(tabnet_calibrator, 'tabnet', tabnet_metadata)
            else:
                print(f"   (DRY RUN - not saved)")

    except Exception as e:
        print(f"❌ Error updating TabNet calibrator: {e}")
        tabnet_calibrator = None
        tabnet_metadata = None

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if rf_calibrator and tabnet_calibrator:
        avg_improvement = (rf_metadata['mae_improvement_pct'] + tabnet_metadata['mae_improvement_pct']) / 2
        print(f"✅ Both calibrators updated successfully")
        print(f"   Average MAE improvement: {avg_improvement:.2f}%")

        if not args.dry_run:
            print()
            print(f"📅 Next update recommended: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}")
            print(f"   Run: python scripts/update_calibrators.py --days {args.days}")
    elif rf_calibrator or tabnet_calibrator:
        print(f"⚠️  Only one calibrator was updated")
    else:
        print(f"❌ No calibrators were updated (not enough data)")
        return 1

    if args.dry_run:
        print()
        print(f"🔍 DRY RUN complete - no changes were saved")
        print(f"   Remove --dry-run flag to actually update calibrators")

    return 0


if __name__ == '__main__':
    sys.exit(main())
