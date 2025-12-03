#!/usr/bin/env python3

import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

from utils.env_setup import AppConfig, get_sqlite_dbpath


class CompetitiveWeightChecker:
    """Simple competitive weight checker using DB actual results and prediction files."""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the checker."""
        self.config = AppConfig(config_path)
        self.db_type = self.config._config.base.active_db

        # Get database path - use direct path since get_sqlite_dbpath may be incorrect
        db_path = get_sqlite_dbpath(self.db_type)

        # Fix path if it's pointing to wrong location
        if 'horseai/databases' in db_path and not Path(db_path).exists():
            # Try data/ directory instead
            db_name = Path(db_path).name
            alternative_path = f'data/{db_name}'
            if Path(alternative_path).exists():
                self.db_path = alternative_path
                print(f"Using database: {self.db_path}")
            else:
                self.db_path = db_path
        else:
            self.db_path = db_path

    def get_available_prediction_dates(self) -> List[str]:
        """Get list of dates with predictions in quinte_predictions table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
        SELECT DISTINCT race_date
        FROM quinte_predictions
        WHERE race_date IS NOT NULL
        ORDER BY race_date DESC
        """
        cursor.execute(query)
        dates = [row[0] for row in cursor.fetchall()]
        conn.close()

        print(f"   Found predictions for {len(dates)} dates")
        return dates

    def load_predictions(self, race_date: str = None, race_comps: List[str] = None) -> pd.DataFrame:
        """
        Load predictions from quinte_predictions table.

        Args:
            race_date: Specific date to load predictions for (YYYY-MM-DD)
            race_comps: Specific race IDs to load

        Returns:
            DataFrame with columns: comp, numero, predicted_position_base, competitive_adjustment
        """
        conn = sqlite3.connect(self.db_path)

        if race_comps:
            # Load specific races
            race_comps_str = [str(rc) for rc in race_comps]
            placeholders = ','.join('?' * len(race_comps_str))
            query = f"""
            SELECT
                race_id as comp,
                horse_number as numero,
                quinte_tabnet_prediction as predicted_position_base,
                competitive_adjustment,
                final_prediction,
                predicted_rank
            FROM quinte_predictions
            WHERE race_id IN ({placeholders})
            ORDER BY race_id, predicted_rank
            """
            df = pd.read_sql_query(query, conn, params=race_comps_str)
        elif race_date:
            # Load specific date
            query = """
            SELECT
                race_id as comp,
                horse_number as numero,
                quinte_tabnet_prediction as predicted_position_base,
                competitive_adjustment,
                final_prediction,
                predicted_rank
            FROM quinte_predictions
            WHERE race_date = ?
            ORDER BY race_id, predicted_rank
            """
            df = pd.read_sql_query(query, conn, params=(race_date,))
        else:
            # Load all predictions
            query = """
            SELECT
                race_id as comp,
                horse_number as numero,
                quinte_tabnet_prediction as predicted_position_base,
                competitive_adjustment,
                final_prediction,
                predicted_rank
            FROM quinte_predictions
            ORDER BY race_date DESC, race_id, predicted_rank
            """
            df = pd.read_sql_query(query, conn)

        conn.close()

        # Fill missing adjustments with 0.0
        df['competitive_adjustment'] = df['competitive_adjustment'].fillna(0.0)

        print(f"   Loaded {len(df)} predictions from quinte_predictions table")

        return df[['comp', 'numero', 'predicted_position_base', 'competitive_adjustment']]

    def load_actual_results(self, race_comps: List[str]) -> Dict[str, Dict[str, Set[int]]]:
        """
        Load actual top 3, 4, and 5 finishers from daily_race table.

        Args:
            race_comps: List of race comp IDs to load

        Returns:
            Dict mapping comp to dict with 'top3', 'top4', 'top5' sets of horse numbers
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query with placeholders
        race_comps_str = [str(rc) for rc in race_comps]
        placeholders = ','.join('?' * len(race_comps_str))
        query = f"SELECT comp, actual_results FROM daily_race WHERE comp IN ({placeholders}) AND actual_results IS NOT NULL"

        cursor.execute(query, race_comps_str)
        rows = cursor.fetchall()
        conn.close()

        results = {}
        for comp, actual_results in rows:
            try:
                if not actual_results or actual_results.strip() == '':
                    continue

                # Parse actual_results format: "13-10-7-1-8-3-4-2-14-16"
                # This is the order of arrival: first number finished 1st, etc.
                horse_numbers = actual_results.strip().split('-')

                top3 = set()
                top4 = set()
                top5 = set()

                for position, horse_num_str in enumerate(horse_numbers, start=1):
                    try:
                        horse_num = int(horse_num_str.strip())
                        if horse_num > 0:
                            if position <= 3:
                                top3.add(horse_num)
                            if position <= 4:
                                top4.add(horse_num)
                            if position <= 5:
                                top5.add(horse_num)
                    except ValueError:
                        continue

                if len(top5) == 5:  # Only include races with complete top 5
                    results[comp] = {
                        'top3': top3,
                        'top4': top4,
                        'top5': top5
                    }

            except (ValueError, AttributeError):
                continue

        return results

    def apply_weight_and_rank(self, df_predictions: pd.DataFrame, weight: float) -> pd.DataFrame:
        """
        Apply competitive weight and calculate predicted rankings.

        Args:
            df_predictions: DataFrame with base predictions and adjustments
            weight: Competitive weight to apply (0.0 to 1.0)

        Returns:
            DataFrame with predicted_position and predicted_rank columns
        """
        df = df_predictions.copy()

        # Apply weight to competitive adjustment
        df['predicted_position'] = (
            df['predicted_position_base'] +
            df['competitive_adjustment'] * weight
        )

        # Rank within each race (lower position = better rank)
        df['predicted_rank'] = df.groupby('comp')['predicted_position'].rank(method='first').astype(int)

        return df

    def calculate_wins(self, df_predictions: pd.DataFrame,
                      actual_results: Dict[str, Dict[str, Set[int]]]) -> Dict[str, int]:
        """
        Calculate quinté désordre, bonus 3, and bonus 4 wins.

        Args:
            df_predictions: DataFrame with predictions and ranks
            actual_results: Dict of comp -> dict with 'top3', 'top4', 'top5' sets

        Returns:
            Dict with counts for quinte_desordre, bonus3, and bonus4
        """
        quinte_wins = 0
        bonus3_wins = 0
        bonus4_wins = 0

        for comp, actual_sets in actual_results.items():
            race_df = df_predictions[df_predictions['comp'] == comp]

            if len(race_df) == 0:
                continue

            # Get predicted top 5
            predicted_top5 = set(race_df[race_df['predicted_rank'] <= 5]['numero'].values)

            actual_top3 = actual_sets['top3']
            actual_top4 = actual_sets['top4']
            actual_top5 = actual_sets['top5']

            # Quinté désordre: all 5 predicted horses in actual top 5
            if len(predicted_top5 & actual_top5) == 5:
                quinte_wins += 1

            # Bonus 3: all 3 of actual top 3 are in predicted top 5
            if len(actual_top3 & predicted_top5) == 3:
                bonus3_wins += 1

            # Bonus 4: all 4 of actual top 4 are in predicted top 5
            if len(actual_top4 & predicted_top5) == 4:
                bonus4_wins += 1

        return {
            'quinte_desordre': quinte_wins,
            'bonus3': bonus3_wins,
            'bonus4': bonus4_wins
        }

    def test_weights(self, race_date: str = None, race_comps: List[str] = None,
                    weight_start: float = 0.0, weight_stop: float = 1.0, weight_step: float = 0.1):
        """
        Test different competitive weights and display results.

        Args:
            race_date: Specific date to load predictions for (YYYY-MM-DD)
            race_comps: Specific race IDs to load
            weight_start: Starting weight
            weight_stop: Ending weight (inclusive)
            weight_step: Step size
        """
        print("=" * 80)
        print("COMPETITIVE WEIGHTING CHECK")
        print("=" * 80)

        # Load data
        print("\n📊 Loading data from database...")

        # Load predictions from quinte_predictions table
        df_predictions = self.load_predictions(race_date=race_date, race_comps=race_comps)
        race_comps = df_predictions['comp'].unique().tolist()
        print(f"   Races in predictions: {len(race_comps)}")

        # Load actual results
        actual_results = self.load_actual_results(race_comps)
        print(f"   Races with actual top 5: {len(actual_results)}")

        # Check for competitive adjustments
        has_adjustments = (df_predictions['competitive_adjustment'].abs() > 0.001).any()
        print(f"\n📌 Base Predictions: Using TabNet model (predicted_position_tabnet)")
        if not has_adjustments:
            print("⚠️  WARNING: All competitive adjustments are 0.0")
            print("   This means competitive analysis is not in the database.")
            print("   Results will be identical for all weights.\n")
        else:
            adj_nonzero = (df_predictions['competitive_adjustment'].abs() > 0.001).sum()
            print(f"✓  Found competitive adjustments for {adj_nonzero}/{len(df_predictions)} horses")
            print(f"   Weight 0.0 = Pure TabNet predictions (no competitive adjustment)")
            print(f"   Weight 1.0 = TabNet + Full competitive adjustment from DB\n")

        # Test each weight
        print("\n" + "=" * 100)
        print("TESTING WEIGHTS")
        print("=" * 100)
        print(f"{'Weight':<8} {'Quinté Désordre':<20} {'Bonus 3':<20} {'Bonus 4':<20}")
        print("-" * 100)

        results = []
        weights = np.arange(weight_start, weight_stop + weight_step/2, weight_step)
        total_races = len(actual_results)

        for weight in weights:
            # Apply weight and rank
            df_ranked = self.apply_weight_and_rank(df_predictions, weight)

            # Calculate wins
            wins = self.calculate_wins(df_ranked, actual_results)

            quinte_rate = (wins['quinte_desordre'] / total_races * 100) if total_races > 0 else 0.0
            bonus3_rate = (wins['bonus3'] / total_races * 100) if total_races > 0 else 0.0
            bonus4_rate = (wins['bonus4'] / total_races * 100) if total_races > 0 else 0.0

            results.append({
                'weight': weight,
                'quinte_desordre': wins['quinte_desordre'],
                'bonus3': wins['bonus3'],
                'bonus4': wins['bonus4'],
                'quinte_rate': quinte_rate,
                'bonus3_rate': bonus3_rate,
                'bonus4_rate': bonus4_rate,
                'total': total_races
            })

            print(f"{weight:<8.1f} "
                  f"{wins['quinte_desordre']:>3d}/{total_races:<3d} ({quinte_rate:>5.1f}%)  "
                  f"{wins['bonus3']:>3d}/{total_races:<3d} ({bonus3_rate:>5.1f}%)  "
                  f"{wins['bonus4']:>3d}/{total_races:<3d} ({bonus4_rate:>5.1f}%)")

        print("-" * 100)

        # Find best weights
        best_quinte = max(results, key=lambda x: x['quinte_desordre'])
        best_bonus3 = max(results, key=lambda x: x['bonus3'])
        best_bonus4 = max(results, key=lambda x: x['bonus4'])

        print(f"\n🏆 Best Weights:")
        print(f"   Quinté Désordre: {best_quinte['weight']:.1f} with {best_quinte['quinte_desordre']} wins ({best_quinte['quinte_rate']:.1f}%)")
        print(f"   Bonus 3:         {best_bonus3['weight']:.1f} with {best_bonus3['bonus3']} wins ({best_bonus3['bonus3_rate']:.1f}%)")
        print(f"   Bonus 4:         {best_bonus4['weight']:.1f} with {best_bonus4['bonus4']} wins ({best_bonus4['bonus4_rate']:.1f}%)")

        # Show adjustment statistics
        print("\n" + "=" * 80)
        print("COMPETITIVE ADJUSTMENT STATISTICS")
        print("=" * 80)
        adj_stats = df_predictions['competitive_adjustment'].describe()
        print(f"Mean:   {adj_stats['mean']:>8.4f}")
        print(f"Std:    {adj_stats['std']:>8.4f}")
        print(f"Min:    {adj_stats['min']:>8.4f}")
        print(f"Max:    {adj_stats['max']:>8.4f}")
        print(f"Median: {adj_stats['50%']:>8.4f}")

        return results


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Test competitive weighting configurations from database')
    parser.add_argument('--date', type=str, help='Race date to analyze (YYYY-MM-DD). If not specified, uses all predictions.')
    parser.add_argument('--race-ids', nargs='+', help='Specific race IDs to analyze (space-separated)')
    parser.add_argument('--start', type=float, default=0.0, help='Starting weight (default: 0.0)')
    parser.add_argument('--stop', type=float, default=1.0, help='Ending weight (default: 1.0)')
    parser.add_argument('--step', type=float, default=0.1, help='Weight step (default: 0.1)')

    args = parser.parse_args()

    checker = CompetitiveWeightChecker()
    checker.test_weights(
        race_date=args.date,
        race_comps=args.race_ids,
        weight_start=args.start,
        weight_stop=args.stop,
        weight_step=args.step
    )


if __name__ == '__main__':
    main()
