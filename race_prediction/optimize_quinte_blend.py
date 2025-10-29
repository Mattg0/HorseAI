#!/usr/bin/env python3
"""
Quinté RF/TabNet Blend Optimization Script

Tests different blend weights between RF and TabNet predictions (from CSV/JSON)
to find the optimal combination that maximizes prediction accuracy.

Usage:
    # Use latest prediction file
    python race_prediction/optimize_quinte_blend.py

    # Use specific prediction file
    python race_prediction/optimize_quinte_blend.py --predictions predictions/file.csv

    # Test custom weight range
    python race_prediction/optimize_quinte_blend.py --step 0.05
"""

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.env_setup import AppConfig, get_sqlite_dbpath


class QuinteBlendOptimizer:
    """
    Optimizes RF/TabNet blend weights for quinté predictions from existing prediction files.

    Tests various weight combinations and evaluates performance using:
    - Quinté désordre accuracy (all 5 horses in top 5)
    - Bonus 3 win rate
    - Bonus 4 win rate
    - Avg horses in quinté
    - MAE (Mean Absolute Error)
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = True):
        """Initialize the blend optimizer."""
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        self.db_path = get_sqlite_dbpath(self.db_type)

        self.log_info(f"Initialized QuinteBlendOptimizer with database: {self.db_type}")

    def log_info(self, message):
        """Log informational message."""
        if self.verbose:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] {message}")

    def find_latest_prediction_file(self, predictions_dir: str = 'predictions') -> str:
        """
        Find the latest quinté prediction file.

        Args:
            predictions_dir: Directory containing prediction files

        Returns:
            Path to latest prediction file
        """
        pred_path = Path(predictions_dir)

        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

        # Look for quinté prediction files
        csv_files = list(pred_path.glob("quinte_predictions_*.csv"))
        json_files = list(pred_path.glob("quinte_predictions_*.json"))

        all_files = csv_files + json_files

        if not all_files:
            raise FileNotFoundError(f"No quinté prediction files found in {predictions_dir}")

        # Get most recent file
        latest_file = max(all_files, key=lambda f: f.stat().st_mtime)

        return str(latest_file)

    def load_predictions(self, file_path: str) -> pd.DataFrame:
        """
        Load predictions from CSV or JSON file.

        Args:
            file_path: Path to prediction file

        Returns:
            DataFrame with predictions including rf_predicted_position and tabnet_predicted_position
        """
        self.log_info(f"Loading predictions from {file_path}...")

        file_path_obj = Path(file_path)

        if file_path_obj.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path_obj.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)

            # JSON structure is list of races, each with 'predictions' array
            # Flatten to get all predictions
            all_predictions = []
            if isinstance(data, list):
                for race in data:
                    if 'predictions' in race:
                        all_predictions.extend(race['predictions'])
                    else:
                        # Fallback: if no 'predictions' key, assume race itself is the prediction
                        all_predictions.append(race)
            else:
                # If data is a dict or something else, try to convert directly
                all_predictions = data

            df = pd.DataFrame(all_predictions)
        else:
            raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")

        # Check for required columns (support both naming conventions)
        required_base = ['comp', 'numero']

        # Check base columns first
        missing_base = [col for col in required_base if col not in df.columns]
        if missing_base:
            raise ValueError(f"Missing required columns: {missing_base}")

        # Check for RF predictions (try both column names)
        if 'rf_predicted_position' not in df.columns:
            if 'predicted_position_rf' in df.columns:
                df = df.rename(columns={'predicted_position_rf': 'rf_predicted_position'})
            else:
                raise ValueError("Missing RF prediction column (expected 'rf_predicted_position' or 'predicted_position_rf')")

        # Check for TabNet predictions (try both column names)
        if 'tabnet_predicted_position' not in df.columns:
            if 'predicted_position_tabnet' in df.columns:
                df = df.rename(columns={'predicted_position_tabnet': 'tabnet_predicted_position'})
            else:
                raise ValueError("Missing TabNet prediction column (expected 'tabnet_predicted_position' or 'predicted_position_tabnet')")

        self.log_info(f"Loaded {len(df)} predictions for {df['comp'].nunique()} races")

        return df

    def load_actual_results(self, race_comps: List[str]) -> Dict[str, Dict[int, int]]:
        """
        Load actual results from database.

        Args:
            race_comps: List of race comp IDs

        Returns:
            Dict mapping comp -> {horse_number: position}
        """
        self.log_info(f"Loading actual results for {len(race_comps)} races...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert to strings
        race_comps_str = [str(comp) for comp in race_comps]

        placeholders = ','.join(['?' for _ in race_comps_str])
        query = f"""
        SELECT comp, actual_results FROM daily_race
        WHERE comp IN ({placeholders})
        AND actual_results IS NOT NULL
        """

        cursor.execute(query, race_comps_str)
        rows = cursor.fetchall()
        conn.close()

        results_map = {}
        for comp, actual_results in rows:
            try:
                if not actual_results or actual_results.strip() == '':
                    continue

                # Parse actual_results format: "14-16-3-5-7"
                horse_numbers = actual_results.strip().split('-')

                # Create mapping of horse_numero -> finish_position
                horse_positions = {}
                for position, horse_num_str in enumerate(horse_numbers, start=1):
                    try:
                        horse_num = int(horse_num_str.strip())
                        if horse_num > 0:
                            horse_positions[horse_num] = position
                    except (ValueError, AttributeError):
                        continue

                if horse_positions:
                    results_map[str(comp)] = horse_positions

            except (ValueError, AttributeError, TypeError) as e:
                self.log_info(f"Warning: Could not parse results for race {comp}: {e}")
                continue

        self.log_info(f"Loaded results for {len(results_map)} races")

        return results_map

    def blend_predictions(self, df: pd.DataFrame, rf_weight: float, tabnet_weight: float) -> pd.DataFrame:
        """
        Create blended predictions from RF and TabNet predictions.

        Args:
            df: DataFrame with rf_predicted_position and tabnet_predicted_position
            rf_weight: Weight for RF model (0.0 to 1.0)
            tabnet_weight: Weight for TabNet model (0.0 to 1.0)

        Returns:
            DataFrame with blended predictions and ranks
        """
        result_df = df.copy()

        # Blend predictions
        result_df['predicted_position'] = (
            rf_weight * result_df['rf_predicted_position'] +
            tabnet_weight * result_df['tabnet_predicted_position']
        )

        # Calculate predicted rank within each race
        result_df['predicted_rank'] = result_df.groupby('comp')['predicted_position'].rank(method='first')

        return result_df

    def calculate_metrics(self, df_predictions: pd.DataFrame,
                         actual_results: Dict[str, Dict[int, int]]) -> Dict:
        """
        Calculate performance metrics for predictions.

        Args:
            df_predictions: DataFrame with predictions and ranks
            actual_results: Dict mapping comp -> {horse_number: position}

        Returns:
            Dict with various performance metrics
        """
        metrics = {
            'total_races': 0,
            'quinte_desordre': 0,  # All 5 predicted horses in top 5
            'bonus3': 0,  # All 3 actual top finishers in predicted top 5
            'bonus4': 0,  # All 4 actual top finishers in predicted top 5
            'mae_values': [],
            'horses_in_quinte': [],  # Count of predicted horses in actual top 5
        }

        for comp, race_df in df_predictions.groupby('comp'):
            comp_str = str(comp)

            if comp_str not in actual_results:
                continue

            metrics['total_races'] += 1
            actual_positions = actual_results[comp_str]

            # Get predicted top 5
            predicted_top5 = set(race_df[race_df['predicted_rank'] <= 5]['numero'].values)

            # Get actual top 5
            actual_top5 = set([num for num, pos in actual_positions.items() if pos <= 5])
            actual_top3 = set([num for num, pos in actual_positions.items() if pos <= 3])
            actual_top4 = set([num for num, pos in actual_positions.items() if pos <= 4])

            # Quinté désordre: all 5 predicted horses in actual top 5
            horses_in_top5 = len(predicted_top5 & actual_top5)
            metrics['horses_in_quinte'].append(horses_in_top5)

            if horses_in_top5 == 5:
                metrics['quinte_desordre'] += 1

            # Bonus 3: all actual top 3 in predicted top 5
            if len(actual_top3 & predicted_top5) == 3:
                metrics['bonus3'] += 1

            # Bonus 4: all actual top 4 in predicted top 5
            if len(actual_top4 & predicted_top5) == 4:
                metrics['bonus4'] += 1

            # MAE for horses with actual results
            for _, row in race_df.iterrows():
                horse_num = int(row['numero'])
                if horse_num in actual_positions:
                    predicted_pos = row['predicted_position']
                    actual_pos = actual_positions[horse_num]
                    metrics['mae_values'].append(abs(predicted_pos - actual_pos))

        # Calculate percentages and averages
        if metrics['total_races'] > 0:
            metrics['quinte_desordre_pct'] = (metrics['quinte_desordre'] / metrics['total_races']) * 100
            metrics['bonus3_pct'] = (metrics['bonus3'] / metrics['total_races']) * 100
            metrics['bonus4_pct'] = (metrics['bonus4'] / metrics['total_races']) * 100
            metrics['avg_horses_in_quinte'] = np.mean(metrics['horses_in_quinte'])
        else:
            metrics['quinte_desordre_pct'] = 0.0
            metrics['bonus3_pct'] = 0.0
            metrics['bonus4_pct'] = 0.0
            metrics['avg_horses_in_quinte'] = 0.0

        if len(metrics['mae_values']) > 0:
            metrics['mae'] = np.mean(metrics['mae_values'])
        else:
            metrics['mae'] = 0.0

        return metrics

    def optimize(self, prediction_file: str, weight_step: float = 0.1) -> pd.DataFrame:
        """
        Test different blend weights and find optimal combination.

        Args:
            prediction_file: Path to prediction CSV or JSON file
            weight_step: Step size for weight testing (e.g., 0.1 = test 0.0, 0.1, 0.2, ...)

        Returns:
            DataFrame with results for each weight combination
        """
        self.log_info(f"Starting blend optimization with step size {weight_step}")

        # Load predictions
        df_predictions = self.load_predictions(prediction_file)

        # Load actual results
        race_comps = df_predictions['comp'].unique().tolist()
        actual_results = self.load_actual_results(race_comps)

        # Filter to only races with actual results
        races_with_results = set(actual_results.keys())
        df_predictions = df_predictions[df_predictions['comp'].astype(str).isin(races_with_results)].copy()

        self.log_info(f"Testing on {len(races_with_results)} races with actual results")

        # Generate weight combinations
        # RF weight from 0.0 to 1.0, TabNet weight = 1.0 - RF weight
        rf_weights = np.arange(0.0, 1.0 + weight_step, weight_step)

        results = []

        for rf_weight in tqdm(rf_weights, desc="Testing blend weights"):
            tabnet_weight = 1.0 - rf_weight

            self.log_info(f"\nTesting RF={rf_weight:.2f}, TabNet={tabnet_weight:.2f}")

            # Blend predictions
            df_blended = self.blend_predictions(df_predictions, rf_weight, tabnet_weight)

            # Calculate metrics
            metrics = self.calculate_metrics(df_blended, actual_results)

            # Store results
            result = {
                'rf_weight': round(rf_weight, 2),
                'tabnet_weight': round(tabnet_weight, 2),
                'total_races': metrics['total_races'],
                'quinte_desordre': metrics['quinte_desordre'],
                'quinte_desordre_pct': round(metrics['quinte_desordre_pct'], 2),
                'bonus3': metrics['bonus3'],
                'bonus3_pct': round(metrics['bonus3_pct'], 2),
                'bonus4': metrics['bonus4'],
                'bonus4_pct': round(metrics['bonus4_pct'], 2),
                'avg_horses_in_quinte': round(metrics['avg_horses_in_quinte'], 2),
                'mae': round(metrics['mae'], 3),
            }

            results.append(result)

            self.log_info(f"  Quinté désordre: {result['quinte_desordre_pct']:.2f}%")
            self.log_info(f"  Bonus 3: {result['bonus3_pct']:.2f}%")
            self.log_info(f"  Bonus 4: {result['bonus4_pct']:.2f}%")
            self.log_info(f"  MAE: {result['mae']:.3f}")

        return pd.DataFrame(results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Optimize RF/TabNet blend weights for quinté predictions'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        help='Path to prediction CSV or JSON file (auto-finds latest if not specified)'
    )
    parser.add_argument(
        '--step',
        type=float,
        default=0.1,
        help='Weight step size (default: 0.1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions',
        help='Output directory for results (default: predictions)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize optimizer
    optimizer = QuinteBlendOptimizer(config_path=args.config)

    # Find or use specified prediction file
    if args.predictions:
        prediction_file = args.predictions
    else:
        prediction_file = optimizer.find_latest_prediction_file(args.output)

    print(f"\nUsing prediction file: {prediction_file}")

    # Run optimization
    df_results = optimizer.optimize(
        prediction_file=prediction_file,
        weight_step=args.step
    )

    # Sort by quinté désordre percentage (primary), then bonus 4 (secondary)
    df_results_sorted = df_results.sort_values(
        by=['quinte_desordre_pct', 'bonus4_pct', 'bonus3_pct'],
        ascending=False
    )

    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = output_dir / f"quinte_blend_optimization_{timestamp}.csv"

    # Save results
    df_results_sorted.to_csv(csv_file, index=False)

    print("\n" + "="*80)
    print("BLEND OPTIMIZATION RESULTS")
    print("="*80)
    print(f"\nTested {len(df_results)} blend combinations on {df_results['total_races'].iloc[0]} races")
    print(f"\nTop 10 performing blends:")
    print(df_results_sorted.head(10).to_string(index=False))

    print(f"\n\nBest blend for Quinté Désordre:")
    best_quinte = df_results_sorted.iloc[0]
    print(f"  RF Weight: {best_quinte['rf_weight']:.2f}")
    print(f"  TabNet Weight: {best_quinte['tabnet_weight']:.2f}")
    print(f"  Quinté Désordre: {best_quinte['quinte_desordre_pct']:.2f}% ({best_quinte['quinte_desordre']} wins)")
    print(f"  Bonus 3: {best_quinte['bonus3_pct']:.2f}% ({best_quinte['bonus3']} wins)")
    print(f"  Bonus 4: {best_quinte['bonus4_pct']:.2f}% ({best_quinte['bonus4']} wins)")
    print(f"  MAE: {best_quinte['mae']:.3f}")

    print(f"\n\nResults saved to: {csv_file}")
    print("="*80)


if __name__ == '__main__':
    main()
