#!/usr/bin/env python3
"""
Predict All Quinté Races Using General Model

This script finds all quinté races in the daily_race table and generates
predictions using the general model. This allows comparing general model
performance against the specialized quinté model.

Usage:
    # Predict all quinté races (with or without existing predictions)
    python race_prediction/predict_quinte_with_general.py

    # Only predict races without existing predictions
    python race_prediction/predict_quinte_with_general.py --skip-existing

    # Predict quinté races for a specific date
    python race_prediction/predict_quinte_with_general.py --date 2025-10-26

    # Predict with verbose output
    python race_prediction/predict_quinte_with_general.py --verbose
"""

import argparse
import sqlite3
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from core.orchestrators.prediction_orchestrator import PredictionOrchestrator
from utils.env_setup import AppConfig, get_sqlite_dbpath


class QuinteGeneralPredictor:
    """
    Generates general model predictions for quinté races.
    Stores predictions in race_predictions table for later comparison.
    """

    def __init__(self, db_name: str = None, verbose: bool = False):
        """
        Initialize the quinté general predictor.

        Args:
            db_name: Database name (defaults to active_db from config)
            verbose: Whether to print verbose output
        """
        self.config = AppConfig()
        self.verbose = verbose

        # Get database configuration
        if db_name is None:
            db_name = self.config._config.base.active_db

        self.db_name = db_name
        self.db_path = get_sqlite_dbpath(self.db_name)

        # Initialize prediction orchestrator
        self.orchestrator = PredictionOrchestrator(
            db_name=db_name,
            verbose=verbose
        )

        self.log_info(f"Initialized QuinteGeneralPredictor with database: {self.db_name}")

    def log_info(self, message: str):
        """Log informational message."""
        if self.verbose:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] {message}")

    def get_quinte_races(self, date: str = None, skip_existing: bool = False) -> List[Dict]:
        """
        Get all quinté races from daily_race table.

        Args:
            date: Optional date filter (YYYY-MM-DD)
            skip_existing: If True, skip races that already have predictions

        Returns:
            List of race dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query
        query = """
            SELECT
                dr.comp,
                dr.jour,
                dr.hippo,
                dr.prix,
                dr.typec,
                dr.partant,
                dr.dist,
                dr.quinte,
                CASE WHEN dr.prediction_results IS NOT NULL
                     AND dr.prediction_results != ''
                     THEN 1 ELSE 0 END as has_predictions
            FROM daily_race dr
            WHERE dr.quinte = 1
        """

        params = []

        # Add date filter if specified
        if date:
            query += " AND dr.jour = ?"
            params.append(date)

        # Add existing predictions filter if specified
        if skip_existing:
            query += """ AND NOT EXISTS (
                SELECT 1 FROM race_predictions rp
                WHERE rp.race_id = dr.comp
            )"""

        query += " ORDER BY dr.jour DESC, dr.comp"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to list of dicts
        races = [dict(row) for row in rows]

        conn.close()

        return races

    def predict_quinte_race(self, comp: str) -> Dict:
        """
        Generate general model prediction for a quinté race.

        Args:
            comp: Race comp ID

        Returns:
            Prediction result dictionary
        """
        self.log_info(f"Predicting quinté race {comp} with general model...")

        try:
            # Use orchestrator to predict the race
            # This will store predictions in race_predictions table
            result = self.orchestrator.predict_race(comp)

            if result['status'] == 'success':
                self.log_info(f"✅ Successfully predicted race {comp}")
                return result
            else:
                self.log_info(f"❌ Failed to predict race {comp}: {result.get('error', 'Unknown error')}")
                return result

        except Exception as e:
            self.log_info(f"❌ Error predicting race {comp}: {str(e)}")
            return {
                'status': 'error',
                'comp': comp,
                'error': str(e)
            }

    def predict_all_quinte_races(self, date: str = None, skip_existing: bool = False) -> Dict:
        """
        Predict all quinté races using general model.

        Args:
            date: Optional date filter (YYYY-MM-DD)
            skip_existing: If True, skip races that already have predictions

        Returns:
            Summary dictionary with results
        """
        start_time = datetime.now()

        # Get quinté races
        self.log_info("Fetching quinté races from database...")
        races = self.get_quinte_races(date=date, skip_existing=skip_existing)

        if not races:
            self.log_info("No quinté races found matching criteria")
            return {
                'status': 'no_races',
                'message': 'No quinté races found',
                'date': date,
                'total_races': 0
            }

        self.log_info(f"Found {len(races)} quinté races to predict")

        # Predict each race
        results = []
        success_count = 0
        error_count = 0

        for i, race in enumerate(races, 1):
            comp = race['comp']
            jour = race['jour']
            hippo = race['hippo']
            prix = race['prix']

            self.log_info(f"\n{'='*80}")
            self.log_info(f"[{i}/{len(races)}] Race {comp} - {jour} - {hippo} - {prix}")
            self.log_info(f"{'='*80}")

            result = self.predict_quinte_race(comp)
            results.append(result)

            if result['status'] == 'success':
                success_count += 1
            else:
                error_count += 1

        # Calculate elapsed time
        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Create summary
        summary = {
            'status': 'completed',
            'date_filter': date,
            'skip_existing': skip_existing,
            'total_races': len(races),
            'successful_predictions': success_count,
            'failed_predictions': error_count,
            'elapsed_time_seconds': elapsed_time,
            'results': results
        }

        # Print summary
        self.log_info(f"\n{'='*80}")
        self.log_info("PREDICTION SUMMARY")
        self.log_info(f"{'='*80}")
        self.log_info(f"Total quinté races: {len(races)}")
        self.log_info(f"Successful predictions: {success_count}")
        self.log_info(f"Failed predictions: {error_count}")
        self.log_info(f"Success rate: {success_count/len(races)*100:.1f}%")
        self.log_info(f"Total time: {elapsed_time:.1f}s")
        self.log_info(f"Average time per race: {elapsed_time/len(races):.1f}s")
        self.log_info(f"{'='*80}\n")

        return summary


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Predict quinté races using general model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict all quinté races
  python race_prediction/predict_quinte_with_general.py

  # Only predict races without existing predictions
  python race_prediction/predict_quinte_with_general.py --skip-existing

  # Predict quinté races for a specific date
  python race_prediction/predict_quinte_with_general.py --date 2025-10-26

  # Predict with verbose output
  python race_prediction/predict_quinte_with_general.py --verbose
        """
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Only predict quinté races for this date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip races that already have predictions in race_predictions table'
    )

    parser.add_argument(
        '--db-name',
        type=str,
        default=None,
        help='Database name (defaults to active_db from config)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Create predictor
    predictor = QuinteGeneralPredictor(
        db_name=args.db_name,
        verbose=args.verbose
    )

    # Run predictions
    summary = predictor.predict_all_quinte_races(
        date=args.date,
        skip_existing=args.skip_existing
    )

    # Print summary (always, even if not verbose)
    if not args.verbose:
        print(f"\n{'='*80}")
        print("PREDICTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total quinté races: {summary['total_races']}")
        print(f"Successful predictions: {summary['successful_predictions']}")
        print(f"Failed predictions: {summary['failed_predictions']}")
        if summary['total_races'] > 0:
            print(f"Success rate: {summary['successful_predictions']/summary['total_races']*100:.1f}%")
            print(f"Total time: {summary['elapsed_time_seconds']:.1f}s")
            print(f"Average time per race: {summary['elapsed_time_seconds']/summary['total_races']:.1f}s")
        print(f"{'='*80}\n")

    # Return exit code based on results
    if summary['status'] == 'no_races':
        return 0
    elif summary['failed_predictions'] == 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())
