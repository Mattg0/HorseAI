#!/usr/bin/env python3
"""
Test Different Competitive Weighting Configurations

This script tests various competitive analysis weighting configurations
to find the optimal balance between base model predictions and competitive adjustments.

Usage:
    python race_prediction/test_competitive_weightings.py --predictions predictions/file.csv
"""

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from utils.env_setup import AppConfig, get_sqlite_dbpath


class CompetitiveWeightingTester:
    """
    Tests different competitive weighting configurations to optimize predictions.
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = True):
        """Initialize the weighting tester."""
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        self.db_path = get_sqlite_dbpath(self.db_type)

        self.log_info(f"Initialized CompetitiveWeightingTester with database: {self.db_type}")

    def log_info(self, message):
        """Simple logging method."""
        if self.verbose:
            print(f"[WeightTest] {message}")

    def find_latest_prediction_file(self, predictions_dir: str = 'predictions') -> Optional[str]:
        """
        Find the most recent quinté prediction file.

        Args:
            predictions_dir: Directory containing prediction files

        Returns:
            Path to the latest prediction file, or None if not found
        """
        pred_path = Path(predictions_dir)

        if not pred_path.exists():
            self.log_info(f"Predictions directory not found: {predictions_dir}")
            return None

        # Look for quinté prediction files (CSV or JSON)
        csv_files = list(pred_path.glob("quinte_predictions_*.csv"))
        json_files = list(pred_path.glob("quinte_predictions_*.json"))

        all_files = csv_files + json_files

        if not all_files:
            self.log_info(f"No quinté prediction files found in {predictions_dir}")
            return None

        # Sort by modification time and get the latest
        latest_file = max(all_files, key=lambda f: f.stat().st_mtime)

        return str(latest_file)

    def load_predictions(self, prediction_file: str) -> pd.DataFrame:
        """
        Load predictions from CSV or JSON file.

        Args:
            prediction_file: Path to prediction file

        Returns:
            DataFrame with predictions
        """
        file_path = Path(prediction_file)

        if not file_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

        # Load based on file extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            # Load JSON and check if it has nested predictions structure
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Check if data is a list of races with nested predictions
            if isinstance(data, list) and len(data) > 0 and 'predictions' in data[0]:
                # Flatten nested structure
                all_predictions = []
                for race in data:
                    if 'predictions' in race and isinstance(race['predictions'], list):
                        all_predictions.extend(race['predictions'])

                df = pd.DataFrame(all_predictions)
                self.log_info(f"Flattened {len(data)} races into {len(df)} horse predictions")
            else:
                # Standard flat JSON structure
                df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        self.log_info(f"Loaded {len(df)} predictions from {file_path.name}")

        return df

    def load_actual_results(self) -> Dict[str, Dict[int, int]]:
        """
        Load actual race results from quinte_results table.

        Returns:
            Dict mapping race_comp to dict of {horse_numero: finish_position}
        """
        self.log_info("Loading actual results from quinte_results table...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT comp, actual_results FROM daily_race WHERE quinte=&")
        rows = cursor.fetchall()
        conn.close()

        results_map = {}
        for comp, ordre_arrivee in rows:
            try:
                results_data = json.loads(ordre_arrivee)
                horse_positions = {}
                for result in results_data:
                    narrivee = result.get('narrivee', '')
                    cheval = result.get('cheval', 0)

                    if str(narrivee).isdigit() and cheval > 0:
                        position = int(narrivee)
                        horse_positions[int(cheval)] = position

                if horse_positions:
                    results_map[comp] = horse_positions

            except (json.JSONDecodeError, ValueError, AttributeError, TypeError) as e:
                continue

        self.log_info(f"Loaded results for {len(results_map)} quinté races")

        return results_map

    def test_weighting(self, df_predictions: pd.DataFrame, competitive_weight: float,
                      actual_results: Dict[str, Dict[int, int]]) -> Dict:
        """
        Test a specific competitive weighting configuration.

        Args:
            df_predictions: DataFrame with predictions
            competitive_weight: Weight for competitive adjustment (0.0 to 1.0)
            actual_results: Actual race results

        Returns:
            Dict with metrics for this weighting
        """
        df = df_predictions.copy()

        # Calculate base prediction if not present
        if 'predicted_position_base' not in df.columns:
            # Base is TabNet (weight 1.0) + RF (weight 0.0) as per predict_quinte.py
            if 'predicted_position_tabnet' in df.columns and 'predicted_position_rf' in df.columns:
                df['predicted_position_base'] = (
                    df['predicted_position_tabnet'] * 1.0 +
                    df['predicted_position_rf'] * 0.0
                )
            elif 'predicted_position_tabnet' in df.columns:
                df['predicted_position_base'] = df['predicted_position_tabnet']
            elif 'predicted_position_rf' in df.columns:
                df['predicted_position_base'] = df['predicted_position_rf']
            else:
                raise ValueError("No base prediction columns found")

        # Calculate competitive adjustment if not present
        if 'competitive_adjustment' not in df.columns:
            # Competitive adjustment is the difference between final and base prediction
            if 'predicted_position' in df.columns:
                df['competitive_adjustment'] = df['predicted_position'] - df['predicted_position_base']
            else:
                df['competitive_adjustment'] = 0.0

        # Recalculate prediction with new weighting
        df['predicted_position'] = (
            df['predicted_position_base'] +
            df['competitive_adjustment'] * competitive_weight
        )

        # Calculate predicted rank
        df['predicted_rank'] = df.groupby('comp')['predicted_position'].rank(method='first').astype(int)

        # Add actual positions
        df['actual_position'] = np.nan
        for idx, row in df.iterrows():
            race_comp = row['comp']
            horse_numero = int(row['numero'])

            if race_comp in actual_results and horse_numero in actual_results[race_comp]:
                df.at[idx, 'actual_position'] = actual_results[race_comp][horse_numero]

        # Filter to races with results
        df_with_results = df[df['actual_position'].notna()].copy()

        if len(df_with_results) == 0:
            return {
                'competitive_weight': competitive_weight,
                'total_races': 0,
                'error': 'No results found'
            }

        # Calculate metrics
        metrics = self._calculate_metrics(df_with_results, competitive_weight)

        return metrics

    def _calculate_metrics(self, df: pd.DataFrame, competitive_weight: float) -> Dict:
        """
        Calculate performance metrics.

        Args:
            df: DataFrame with predictions and actual results
            competitive_weight: Weight used for this test

        Returns:
            Dict with calculated metrics
        """
        metrics = {
            'competitive_weight': competitive_weight,
            'total_races': df['comp'].nunique(),
            'total_horses': len(df)
        }

        # MAE and RMSE - only calculate for horses with valid finish positions
        # Filter out NaN, infinity, and invalid positions (horses that didn't finish)
        valid_predictions = df[
            df['actual_position'].notna() &
            np.isfinite(df['actual_position']) &
            np.isfinite(df['predicted_position']) &
            (df['actual_position'] > 0)
        ].copy()

        if len(valid_predictions) > 0:
            errors = valid_predictions['actual_position'] - valid_predictions['predicted_position']
            metrics['mae'] = float(np.mean(np.abs(errors)))
            metrics['rmse'] = float(np.sqrt(np.mean(errors ** 2)))
            metrics['horses_with_valid_position'] = len(valid_predictions)
        else:
            metrics['mae'] = float('inf')
            metrics['rmse'] = float('inf')
            metrics['horses_with_valid_position'] = 0

        # Per-race metrics
        race_results = []
        for race_comp, race_df in df.groupby('comp'):
            race_actual = race_df['actual_position'].values
            race_predicted_rank = race_df['predicted_rank'].values

            # Actual top 5
            actual_top3 = set([n for n, p in zip(race_df['numero'], race_actual) if p <= 3])
            actual_top4 = set([n for n, p in zip(race_df['numero'], race_actual) if p <= 4])
            actual_top5 = set([n for n, p in zip(race_df['numero'], race_actual) if p <= 5])

            # Predicted top 5 and top 6
            predicted_top5 = set([n for n, r in zip(race_df['numero'], race_predicted_rank) if r <= 5])
            predicted_top6 = set([n for n, r in zip(race_df['numero'], race_predicted_rank) if r <= 6])

            # Calculate wins
            race_results.append({
                'comp': race_comp,
                # Top 5 metrics
                'top5_quinte_desordre': int(len(actual_top5 & predicted_top5) == 5),
                'top5_bonus3': int(len(actual_top3 & predicted_top5) == 3),
                'top5_bonus4': int(len(actual_top4 & predicted_top5) == 4),
                # Top 6 metrics
                'top6_quinte_desordre': int(len(actual_top5 & predicted_top6) == 5),
                'top6_bonus3': int(len(actual_top3 & predicted_top6) == 3),
                'top6_bonus4': int(len(actual_top4 & predicted_top6) == 4),
            })

        # Aggregate race results
        total_races = len(race_results)

        metrics['top5_quinte_desordre'] = sum([r['top5_quinte_desordre'] for r in race_results])
        metrics['top5_bonus3'] = sum([r['top5_bonus3'] for r in race_results])
        metrics['top5_bonus4'] = sum([r['top5_bonus4'] for r in race_results])

        metrics['top6_quinte_desordre'] = sum([r['top6_quinte_desordre'] for r in race_results])
        metrics['top6_bonus3'] = sum([r['top6_bonus3'] for r in race_results])
        metrics['top6_bonus4'] = sum([r['top6_bonus4'] for r in race_results])

        # Percentages
        metrics['top5_quinte_desordre_pct'] = (metrics['top5_quinte_desordre'] / total_races * 100) if total_races > 0 else 0
        metrics['top5_bonus3_pct'] = (metrics['top5_bonus3'] / total_races * 100) if total_races > 0 else 0
        metrics['top5_bonus4_pct'] = (metrics['top5_bonus4'] / total_races * 100) if total_races > 0 else 0

        metrics['top6_quinte_desordre_pct'] = (metrics['top6_quinte_desordre'] / total_races * 100) if total_races > 0 else 0
        metrics['top6_bonus3_pct'] = (metrics['top6_bonus3'] / total_races * 100) if total_races > 0 else 0
        metrics['top6_bonus4_pct'] = (metrics['top6_bonus4'] / total_races * 100) if total_races > 0 else 0

        return metrics

    def test_multiple_weightings(self, df_predictions: pd.DataFrame,
                                 weight_range: Tuple[float, float, float] = (0.0, 1.0, 0.1)) -> pd.DataFrame:
        """
        Test multiple competitive weighting configurations.

        Args:
            df_predictions: DataFrame with predictions
            weight_range: Tuple of (start, stop, step) for weights to test

        Returns:
            DataFrame with results for all tested weights
        """
        self.log_info("Testing multiple competitive weightings...")

        # Load actual results
        actual_results = self.load_actual_results()

        # Generate weights to test
        start, stop, step = weight_range
        weights = np.arange(start, stop + step, step)

        self.log_info(f"Testing {len(weights)} different weights: {weights}")

        # Test each weight
        results = []
        for weight in weights:
            self.log_info(f"Testing competitive weight: {weight:.2f}")
            metrics = self.test_weighting(df_predictions, weight, actual_results)
            results.append(metrics)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def print_results(self, results_df: pd.DataFrame):
        """
        Print results in a formatted table.

        Args:
            results_df: DataFrame with test results
        """
        print("\n" + "=" * 140)
        print("COMPETITIVE WEIGHTING TEST RESULTS")
        print("=" * 140)
        print(f"{'Weight':<8} {'MAE':<8} {'RMSE':<8} {'5-QD':<8} {'5-B3':<8} {'5-B4':<8} {'6-QD':<8} {'6-B3':<8} {'6-B4':<8}")
        print("-" * 140)

        for _, row in results_df.iterrows():
            print(f"{row['competitive_weight']:<8.2f} "
                  f"{row['mae']:<8.3f} "
                  f"{row['rmse']:<8.3f} "
                  f"{row['top5_quinte_desordre']:<3d} {row['top5_quinte_desordre_pct']:>3.1f}% "
                  f"{row['top5_bonus3']:<3d} {row['top5_bonus3_pct']:>3.1f}% "
                  f"{row['top5_bonus4']:<3d} {row['top5_bonus4_pct']:>3.1f}% "
                  f"{row['top6_quinte_desordre']:<3d} {row['top6_quinte_desordre_pct']:>3.1f}% "
                  f"{row['top6_bonus3']:<3d} {row['top6_bonus3_pct']:>3.1f}% "
                  f"{row['top6_bonus4']:<3d} {row['top6_bonus4_pct']:>3.1f}%")

        print("-" * 140)

        # Find best configurations
        print("\n" + "=" * 80)
        print("BEST CONFIGURATIONS")
        print("=" * 80)

        best_mae = results_df.loc[results_df['mae'].idxmin()]
        print(f"\nBest MAE: {best_mae['mae']:.3f} at weight {best_mae['competitive_weight']:.2f}")

        best_quinte_top5 = results_df.loc[results_df['top5_quinte_desordre'].idxmax()]
        print(f"Best Quinté Désordre (Top 5): {best_quinte_top5['top5_quinte_desordre']} "
              f"({best_quinte_top5['top5_quinte_desordre_pct']:.1f}%) at weight {best_quinte_top5['competitive_weight']:.2f}")

        best_quinte_top6 = results_df.loc[results_df['top6_quinte_desordre'].idxmax()]
        print(f"Best Quinté Désordre (Top 6): {best_quinte_top6['top6_quinte_desordre']} "
              f"({best_quinte_top6['top6_quinte_desordre_pct']:.1f}%) at weight {best_quinte_top6['competitive_weight']:.2f}")

        best_bonus3 = results_df.loc[results_df['top5_bonus3'].idxmax()]
        print(f"Best Bonus 3: {best_bonus3['top5_bonus3']} "
              f"({best_bonus3['top5_bonus3_pct']:.1f}%) at weight {best_bonus3['competitive_weight']:.2f}")

        best_bonus4 = results_df.loc[results_df['top5_bonus4'].idxmax()]
        print(f"Best Bonus 4: {best_bonus4['top5_bonus4']} "
              f"({best_bonus4['top5_bonus4_pct']:.1f}%) at weight {best_bonus4['competitive_weight']:.2f}")

        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)

        # Calculate composite score (weighted combination of metrics)
        results_df['composite_score'] = (
            results_df['top5_quinte_desordre'] * 2.0 +  # Most important
            results_df['top5_bonus3'] * 1.0 +
            results_df['top5_bonus4'] * 1.0 -
            results_df['mae'] * 5.0  # Penalize high error
        )

        best_overall = results_df.loc[results_df['composite_score'].idxmax()]
        print(f"\nRecommended competitive weight: {best_overall['competitive_weight']:.2f}")
        print(f"  - MAE: {best_overall['mae']:.3f}")
        print(f"  - Quinté Désordre: {best_overall['top5_quinte_desordre']} ({best_overall['top5_quinte_desordre_pct']:.1f}%)")
        print(f"  - Bonus 3: {best_overall['top5_bonus3']} ({best_overall['top5_bonus3_pct']:.1f}%)")
        print(f"  - Bonus 4: {best_overall['top5_bonus4']} ({best_overall['top5_bonus4_pct']:.1f}%)")
        print(f"  - Composite Score: {best_overall['composite_score']:.2f}")

        print("\n" + "=" * 80)

    def save_results(self, results_df: pd.DataFrame, output_dir: str = 'analysis'):
        """
        Save test results to CSV.

        Args:
            results_df: DataFrame with test results
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_path / f"competitive_weighting_test_{timestamp}.csv"

        results_df.to_csv(filename, index=False)
        self.log_info(f"Results saved to: {filename}")


def main():
    """Main function to run competitive weighting tests."""
    parser = argparse.ArgumentParser(description='Test different competitive weighting configurations')
    parser.add_argument('--predictions', type=str, required=False, default=None,
                       help='Path to prediction file (CSV or JSON). If not specified, auto-finds latest.')
    parser.add_argument('--start', type=float, default=0.0,
                       help='Starting weight (default: 0.0)')
    parser.add_argument('--stop', type=float, default=1.0,
                       help='Ending weight (default: 1.0)')
    parser.add_argument('--step', type=float, default=0.1,
                       help='Weight step size (default: 0.1)')
    parser.add_argument('--output-dir', type=str, default='analysis',
                       help='Output directory for results (default: analysis)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Initialize tester
    tester = CompetitiveWeightingTester(verbose=args.verbose)

    # Auto-find predictions if not specified
    if args.predictions:
        prediction_file = args.predictions
        print(f"Using specified prediction file: {prediction_file}")
    else:
        print("No prediction file specified, auto-detecting latest...")
        prediction_file = tester.find_latest_prediction_file()
        if not prediction_file:
            print("❌ No quinté prediction files found in predictions/ directory")
            print("   Please run predict_quinte.py first or specify --predictions")
            return
        print(f"✓ Auto-detected prediction file: {prediction_file}")

    # Load predictions
    df_predictions = tester.load_predictions(prediction_file)

    # Test multiple weightings
    weight_range = (args.start, args.stop, args.step)
    results_df = tester.test_multiple_weightings(df_predictions, weight_range)

    # Print results
    tester.print_results(results_df)

    # Save results
    tester.save_results(results_df, args.output_dir)

    print("\nTesting complete!")


if __name__ == "__main__":
    main()
