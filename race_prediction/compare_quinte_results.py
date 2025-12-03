#!/usr/bin/env python3
"""
Quinté Result Comparison Script

Compares quinté predictions against actual results to evaluate model performance.

Usage:
    # Auto-find and compare latest prediction file
    python race_prediction/compare_quinte_results.py

    # Compare specific prediction file
    python race_prediction/compare_quinte_results.py --predictions predictions/file.csv
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


class QuinteResultComparator:
    """
    Compares quinté predictions with actual results.

    Evaluates prediction accuracy using various metrics:
    - Exact position accuracy
    - Top 5 accuracy (quinté placing)
    - Winner accuracy
    - Mean Absolute Error (MAE)
    - Rank correlation
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = True):
        """Initialize the result comparator."""
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        self.db_path = get_sqlite_dbpath(self.db_type)

        self.log_info(f"Initialized QuinteResultComparator with database: {self.db_type}")

    def log_info(self, message):
        """Simple logging method."""
        if self.verbose:
            print(f"[QuinteCompare] {message}")

    def get_available_prediction_dates(self) -> List[str]:
        """
        Get list of dates with predictions in quinte_predictions table.

        Returns:
            List of prediction dates (sorted descending)
        """
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

        self.log_info(f"Found predictions for {len(dates)} dates")
        return dates

    def load_predictions(self, race_date: Optional[str] = None, race_comps: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load predictions from quinte_predictions table.

        Args:
            race_date: Specific date to load predictions for (YYYY-MM-DD)
            race_comps: Specific race IDs to load

        Returns:
            DataFrame with predictions
        """
        self.log_info(f"Loading predictions from quinte_predictions table...")

        conn = sqlite3.connect(self.db_path)

        if race_comps:
            # Load specific races
            race_comps_str = [str(rc) for rc in race_comps]
            placeholders = ','.join('?' * len(race_comps_str))
            query = f"""
            SELECT
                race_id as comp,
                race_date as jour,
                track as hippo,
                race_name as prixnom,
                horse_number as numero,
                horse_id as idche,
                final_prediction as predicted_position,
                predicted_rank,
                quinte_rf_prediction as rf_predicted_position,
                quinte_tabnet_prediction as tabnet_predicted_position,
                competitive_adjustment
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
                race_date as jour,
                track as hippo,
                race_name as prixnom,
                horse_number as numero,
                horse_id as idche,
                final_prediction as predicted_position,
                predicted_rank,
                quinte_rf_prediction as rf_predicted_position,
                quinte_tabnet_prediction as tabnet_predicted_position,
                competitive_adjustment
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
                race_date as jour,
                track as hippo,
                race_name as prixnom,
                horse_number as numero,
                horse_id as idche,
                final_prediction as predicted_position,
                predicted_rank,
                quinte_rf_prediction as rf_predicted_position,
                quinte_tabnet_prediction as tabnet_predicted_position,
                competitive_adjustment
            FROM quinte_predictions
            ORDER BY race_date DESC, race_id, predicted_rank
            """
            df = pd.read_sql_query(query, conn)

        conn.close()

        self.log_info(f"Loaded {len(df)} predictions for {df['comp'].nunique()} races")

        return df

    def load_general_model_predictions(self, race_comps: List[str]) -> Dict[str, Dict]:
        """
        Load general model predictions from daily_race.prediction_results.

        Args:
            race_comps: List of race identifiers

        Returns:
            Dict mapping race_comp to dict with 'top5_str' and 'top5_numbers' and 'positions'
        """
        self.log_info("Loading general model predictions from daily_race...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        race_comps_str = [str(rc) for rc in race_comps]
        placeholders = ','.join('?' * len(race_comps_str))
        query = f"SELECT comp, prediction_results FROM daily_race WHERE comp IN ({placeholders}) AND prediction_results IS NOT NULL"
        cursor.execute(query, race_comps_str)

        rows = cursor.fetchall()
        conn.close()

        general_predictions = {}
        for comp, prediction_results in rows:
            try:
                pred_data = json.loads(prediction_results)

                # Extract predictions array sorted by predicted_rank
                predictions = pred_data.get('predictions', [])
                sorted_preds = sorted(predictions, key=lambda x: x.get('predicted_rank', 999))

                # Get top 5
                top5 = sorted_preds[:5]
                top5_numbers = [int(p['numero']) for p in top5]
                top5_str = ','.join([str(n) for n in top5_numbers])

                # Create position mapping for calculating metrics
                positions = {int(p['numero']): p.get('predicted_rank', 999) for p in predictions}

                general_predictions[str(comp)] = {
                    'top5_str': top5_str,
                    'top5_numbers': top5_numbers,
                    'positions': positions
                }

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.log_info(f"Warning: Could not parse general predictions for race {comp}: {e}")
                continue

        self.log_info(f"Loaded general model predictions for {len(general_predictions)} races")

        return general_predictions

    def load_actual_results(self, race_date: Optional[str] = None, race_comps: Optional[List[str]] = None) -> Dict[str, Dict[int, int]]:
        """
        Load actual results from daily_race table.

        Args:
            race_date: Specific date to load results for
            race_comps: Specific race identifiers to load

        Returns:
            Dict mapping race_comp to {horse_numero: finish_position}
        """
        self.log_info("Loading actual results from daily_race table...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query for daily_race table
        if race_comps:
            # Convert race_comps to strings for SQL IN clause
            race_comps_str = [str(rc) for rc in race_comps]
            placeholders = ','.join('?' * len(race_comps_str))
            query = f"SELECT comp, actual_results FROM daily_race WHERE comp IN ({placeholders}) AND actual_results IS NOT NULL"
            cursor.execute(query, race_comps_str)
        elif race_date:
            query = "SELECT comp, actual_results FROM daily_race WHERE jour = ? AND actual_results IS NOT NULL"
            cursor.execute(query, (race_date,))
        else:
            # Load all races with actual results
            query = "SELECT comp, actual_results FROM daily_race WHERE actual_results IS NOT NULL"
            cursor.execute(query)

        rows = cursor.fetchall()
        conn.close()

        results_map = {}
        for comp, actual_results in rows:
            try:
                if not actual_results or actual_results.strip() == '':
                    continue

                # Parse actual_results format: "13-10-7-1-8-3-4-2-14-16"
                # This is the order of arrival: first number finished 1st, second number finished 2nd, etc.
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

    def merge_predictions_and_results(self, df_predictions: pd.DataFrame,
                                     results_map: Dict[str, Dict[int, int]]) -> pd.DataFrame:
        """
        Merge predictions with actual results.

        Args:
            df_predictions: DataFrame with predictions
            results_map: Dict with actual results

        Returns:
            DataFrame with both predictions and actual positions
        """
        self.log_info("Merging predictions with actual results...")

        result_df = df_predictions.copy()
        result_df['actual_position'] = np.nan

        matched_count = 0
        for idx, row in result_df.iterrows():
            race_comp = str(row['comp'])  # Ensure comp is string for matching
            horse_numero = int(row['numero']) if pd.notna(row['numero']) else 0

            if race_comp in results_map and horse_numero > 0:
                if horse_numero in results_map[race_comp]:
                    position = results_map[race_comp][horse_numero]
                    result_df.at[idx, 'actual_position'] = position
                    matched_count += 1

        self.log_info(f"Matched {matched_count}/{len(result_df)} predictions with actual results")

        # DO NOT filter out horses without actual_position here!
        # We need ALL horses from predictions to calculate predicted top 5
        # The actual_position will be NaN for horses that didn't finish in recorded positions
        # result_df = result_df[result_df['actual_position'].notna()].copy()

        return result_df

    def calculate_metrics(self, df_merged: pd.DataFrame, general_predictions: Dict = None) -> Dict:
        """
        Calculate performance metrics.

        Args:
            df_merged: DataFrame with predictions and actual results
            general_predictions: Dict with general model predictions (optional)

        Returns:
            Dict with various metrics
        """
        self.log_info("Calculating performance metrics...")

        if len(df_merged) == 0:
            return {
                'error': 'No data to calculate metrics'
            }

        metrics = {}

        # Overall metrics - only use rows with actual results (not NaN)
        valid_mask = df_merged['actual_position'].notna()
        predicted = df_merged.loc[valid_mask, 'predicted_position'].values
        actual = df_merged.loc[valid_mask, 'actual_position'].values

        if len(actual) > 0:
            # Mean Absolute Error
            mae = np.mean(np.abs(predicted - actual))
            metrics['mae'] = float(mae)

            # Root Mean Squared Error
            rmse = np.sqrt(np.mean((predicted - actual) ** 2))
            metrics['rmse'] = float(rmse)

            # Exact position accuracy
            exact_correct = np.sum(np.round(predicted) == actual)
            metrics['exact_accuracy'] = float(exact_correct / len(actual))
            metrics['exact_correct'] = int(exact_correct)
        else:
            metrics['mae'] = 0.0
            metrics['rmse'] = 0.0
            metrics['exact_accuracy'] = 0.0
            metrics['exact_correct'] = 0

        # Winner prediction (position 1) - per-race calculation
        # Need to calculate per race, not across all horses
        # This will be calculated from race_metrics instead
        metrics['winner_accuracy'] = 0.0
        metrics['winners_correct'] = 0
        metrics['total_winners'] = 0

        # Top 5 accuracy (quinté placing)
        top5_mask = actual <= 5
        if top5_mask.sum() > 0:
            top5_predicted = (np.round(predicted[top5_mask]) <= 5).sum()
            metrics['top5_accuracy'] = float(top5_predicted / top5_mask.sum())
            metrics['top5_correct'] = int(top5_predicted)
            metrics['total_top5'] = int(top5_mask.sum())
        else:
            metrics['top5_accuracy'] = 0.0
            metrics['top5_correct'] = 0
            metrics['total_top5'] = 0

        # Per-race metrics with detailed quinté analysis
        race_metrics = []
        for race_comp, race_df in df_merged.groupby('comp'):
            race_predicted = race_df['predicted_position'].values
            race_actual = race_df['actual_position'].values
            race_predicted_rank = race_df['predicted_rank'].values if 'predicted_rank' in race_df.columns else np.argsort(race_predicted) + 1

            # Calculate MAE only for horses with actual positions (filter out NaN)
            valid_mask = ~np.isnan(race_actual)
            if valid_mask.sum() > 0:
                race_mae = np.mean(np.abs(race_predicted[valid_mask] - race_actual[valid_mask]))
            else:
                race_mae = np.nan

            # Rank correlation (Spearman) - only for horses with actual positions
            from scipy.stats import spearmanr
            try:
                if valid_mask.sum() > 1:  # Need at least 2 points for correlation
                    corr, _ = spearmanr(race_predicted[valid_mask], race_actual[valid_mask])
                else:
                    corr = 0.0
            except:
                corr = 0.0

            # Get top 5 predicted horses (quinté prediction)
            # IMPORTANT: Use the predicted_rank from our CSV, not recalculated
            # Convert to int to ensure proper comparison
            race_predicted_rank_int = race_predicted_rank.astype(int) if hasattr(race_predicted_rank, 'astype') else race_predicted_rank
            predicted_top5_mask = race_predicted_rank_int <= 5
            predicted_top5_numbers = race_df[predicted_top5_mask]['numero'].values

            # Get actual top 5 horses (quinté result)
            # Use actual_position to get horses that actually finished in top 5
            actual_top5_mask = race_actual <= 5
            actual_top5_numbers = race_df[actual_top5_mask]['numero'].values

            # Calculate quinté metrics
            # % horses in quinté in exact order (checking positions 1-5)
            exact_order_count = 0
            for pos in range(1, 6):
                pred_horse_at_pos = race_df[race_predicted_rank == pos]['numero'].values
                actual_horse_at_pos = race_df[race_actual == pos]['numero'].values
                if len(pred_horse_at_pos) > 0 and len(actual_horse_at_pos) > 0:
                    if pred_horse_at_pos[0] == actual_horse_at_pos[0]:
                        exact_order_count += 1

            pct_exact_order = (exact_order_count / 5.0) * 100 if len(actual_top5_numbers) >= 5 else 0.0

            # % horses in quinté regardless of order (intersection)
            # This checks: of the 5 horses we predicted, how many actually finished in top 5?
            horses_in_quinte = len(set(predicted_top5_numbers) & set(actual_top5_numbers))
            pct_in_quinte = (horses_in_quinte / 5.0) * 100 if len(predicted_top5_numbers) >= 5 else 0.0

            # Top 5 metrics (standard quinté)
            actual_top3 = set([n for n, p in zip(race_df['numero'], race_actual) if p <= 3])
            actual_top4 = set([n for n, p in zip(race_df['numero'], race_actual) if p <= 4])
            actual_top5 = set([n for n, p in zip(race_df['numero'], race_actual) if p <= 5])

            predicted_top5 = set([n for n, r in zip(race_df['numero'], race_predicted_rank) if r <= 5])
            predicted_top6 = set([n for n, r in zip(race_df['numero'], race_predicted_rank) if r <= 6])

            # Top 5 (standard quinté) - Bonus 3 and Bonus 4
            bonus3_top5_count = len(actual_top3 & predicted_top5)
            bonus3_top5_win = (bonus3_top5_count == 3)

            bonus4_top5_count = len(actual_top4 & predicted_top5)
            bonus4_top5_win = (bonus4_top5_count == 4)

            # Quinté désordre with top 5
            quinte_desordre_top5_count = len(actual_top5 & predicted_top5)
            quinte_desordre_top5_win = (quinte_desordre_top5_count == 5)

            # Top 6 (combiné with 6th horse) - Bonus 3 and Bonus 4
            bonus3_top6_count = len(actual_top3 & predicted_top6)
            bonus3_top6_win = (bonus3_top6_count == 3)

            bonus4_top6_count = len(actual_top4 & predicted_top6)
            bonus4_top6_win = (bonus4_top6_count == 4)

            # Quinté désordre with top 6
            quinte_desordre_top6_count = len(actual_top5 & predicted_top6)
            quinte_desordre_top6_win = (quinte_desordre_top6_count == 5)

            # Format predicted arrivals - top 5 and top 6
            horses_with_pred_rank = [(int(n), int(r)) for n, r in zip(race_df['numero'], race_predicted_rank)]
            horses_with_pred_rank.sort(key=lambda x: x[1])  # Sort by predicted rank
            predicted_top5_list = [n for n, r in horses_with_pred_rank[:5]]
            predicted_top6_list = [n for n, r in horses_with_pred_rank[:6]]
            predicted_arrive_top5 = ','.join([str(n) for n in predicted_top5_list])
            predicted_arrive_top6 = ','.join([str(n) for n in predicted_top6_list])

            # Format actual arrivals - simple approach
            # Build list of (horse_number, actual_position) for all horses that have actual positions
            # Filter out NaN values first
            horses_with_actual_pos = [(int(n), int(p)) for n, p in zip(race_df['numero'], race_actual) if not np.isnan(p)]
            horses_with_actual_pos.sort(key=lambda x: x[1])  # Sort by actual position
            actual_top5_list = [n for n, p in horses_with_actual_pos[:5]]
            actual_arrive = ','.join([str(n) for n in actual_top5_list])

            # Winner prediction - check if predicted rank 1 matches actual position 1
            predicted_winner_idx = race_predicted_rank == 1
            actual_winner_idx = race_actual == 1

            winner_correct = 0
            if predicted_winner_idx.sum() > 0 and actual_winner_idx.sum() > 0:
                # Get the horse number predicted to win
                predicted_winner_numero = race_df[predicted_winner_idx]['numero'].values
                # Get the horse number that actually won
                actual_winner_numero = race_df[actual_winner_idx]['numero'].values

                # Check if they match
                if len(predicted_winner_numero) > 0 and len(actual_winner_numero) > 0:
                    if predicted_winner_numero[0] == actual_winner_numero[0]:
                        winner_correct = 1

            race_metrics.append({
                'comp': race_comp,
                'jour': race_df['jour'].iloc[0] if 'jour' in race_df.columns else '',
                'hippo': race_df['hippo'].iloc[0],
                'prixnom': race_df['prixnom'].iloc[0],
                'predicted_top5': predicted_arrive_top5,
                'predicted_top6': predicted_arrive_top6,
                'actual_arrive': actual_arrive,
                'horses': len(race_df),
                'pct_exact_order': float(pct_exact_order),
                'pct_in_quinte': float(pct_in_quinte),
                'horses_in_quinte': int(horses_in_quinte),
                # Top 5 metrics
                'top5_bonus3': int(bonus3_top5_win),
                'top5_bonus4': int(bonus4_top5_win),
                'top5_quinte_desordre': int(quinte_desordre_top5_win),
                'top5_bonus3_count': int(bonus3_top5_count),
                'top5_bonus4_count': int(bonus4_top5_count),
                'top5_quinte_count': int(quinte_desordre_top5_count),
                # Top 6 metrics
                'top6_bonus3': int(bonus3_top6_win),
                'top6_bonus4': int(bonus4_top6_win),
                'top6_quinte_desordre': int(quinte_desordre_top6_win),
                'top6_bonus3_count': int(bonus3_top6_count),
                'top6_bonus4_count': int(bonus4_top6_count),
                'top6_quinte_count': int(quinte_desordre_top6_count),
                'mae': float(race_mae),
                'correlation': float(corr),
                'winner_correct': winner_correct
            })

        metrics['race_metrics'] = race_metrics
        metrics['total_horses'] = len(df_merged)
        metrics['total_races'] = len(race_metrics)

        # Calculate overall winner accuracy from race_metrics
        if len(race_metrics) > 0:
            winners_correct = sum([r['winner_correct'] for r in race_metrics])
            metrics['winners_correct'] = winners_correct
            metrics['total_winners'] = len(race_metrics)  # Each race has 1 winner
            metrics['winner_accuracy'] = float(winners_correct / len(race_metrics))
        else:
            metrics['winner_accuracy'] = 0.0
            metrics['winners_correct'] = 0
            metrics['total_winners'] = 0

        return metrics

    def print_summary(self, metrics: Dict, df_merged: pd.DataFrame):
        """
        Print a summary of results.

        Args:
            metrics: Dict with calculated metrics
            df_merged: DataFrame with merged data
        """
        print("\n" + "=" * 80)
        print("QUINTÉ PREDICTION EVALUATION")
        print("=" * 80)

        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
            return

        # Calculate summary metrics
        total_races = metrics['total_races']
        top5_quinte_desordre_wins = sum([r['top5_quinte_desordre'] for r in metrics['race_metrics']])
        top5_bonus4_wins = sum([r['top5_bonus4'] for r in metrics['race_metrics']])
        top5_bonus3_wins = sum([r['top5_bonus3'] for r in metrics['race_metrics']])

        # SUMMARY BOX (KEY METRICS)
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 30 + "SUMMARY" + " " * 41 + "║")
        print("╠" + "═" + "═" * 78 + "╣")
        print(f"║  Winner Accuracy:        {metrics['winner_accuracy']:>6.1%}   ({metrics['winners_correct']}/{metrics['total_winners']} races)" + " " * (78 - 46 - len(f"{metrics['winners_correct']}/{metrics['total_winners']} races")) + "║")
        print(f"║  Podium Accuracy (Top 3): {top5_bonus3_wins/total_races:>6.1%}   ({top5_bonus3_wins}/{total_races} races)" + " " * (78 - 49 - len(f"{top5_bonus3_wins}/{total_races} races")) + "║")
        print(f"║  Quinté Désordre:        {top5_quinte_desordre_wins/total_races:>6.1%}   ({top5_quinte_desordre_wins}/{total_races} races)" + " " * (78 - 46 - len(f"{top5_quinte_desordre_wins}/{total_races} races")) + "║")
        print(f"║  Mean Absolute Error:    {metrics['mae']:>6.3f}   positions" + " " * (78 - 40) + "║")
        print(f"║  Quinté Races Evaluated: {total_races:>6d}   races" + " " * (78 - 37) + "║")
        print("╚" + "═" * 78 + "╝")

        print(f"\nOverall Performance:")
        print(f"  Total Races: {metrics['total_races']}")
        print(f"  Total Horses: {metrics['total_horses']}")
        print(f"  Mean Absolute Error: {metrics['mae']:.3f} positions")
        print(f"  RMSE: {metrics['rmse']:.3f} positions")

        print(f"\nAccuracy Metrics:")
        print(f"  Exact Position: {metrics['exact_accuracy']:.1%} ({metrics['exact_correct']}/{metrics['total_horses']})")
        print(f"  Winner Prediction: {metrics['winner_accuracy']:.1%} ({metrics['winners_correct']}/{metrics['total_winners']})")
        print(f"  Top 5 (Quinté): {metrics['top5_accuracy']:.1%} ({metrics['top5_correct']}/{metrics['total_top5']})")

        # Calculate overall quinté accuracy
        avg_exact = np.mean([r['pct_exact_order'] for r in metrics['race_metrics']])
        avg_in_quinte = np.mean([r['pct_in_quinte'] for r in metrics['race_metrics']])
        total_races = metrics['total_races']

        # Top 5 metrics
        top5_bonus3_wins = sum([r['top5_bonus3'] for r in metrics['race_metrics']])
        top5_bonus4_wins = sum([r['top5_bonus4'] for r in metrics['race_metrics']])
        top5_quinte_desordre_wins = sum([r['top5_quinte_desordre'] for r in metrics['race_metrics']])

        # Top 6 metrics
        top6_bonus3_wins = sum([r['top6_bonus3'] for r in metrics['race_metrics']])
        top6_bonus4_wins = sum([r['top6_bonus4'] for r in metrics['race_metrics']])
        top6_quinte_desordre_wins = sum([r['top6_quinte_desordre'] for r in metrics['race_metrics']])

        print(f"\n{'='*80}")
        print(f"QUINTÉ PREDICTIONS WITH TOP 5 (Standard)")
        print(f"{'='*80}")
        print(f"  Avg % Exact Order: {avg_exact:.1f}% (correct horses in correct positions)")
        print(f"  Avg % In Quinté: {avg_in_quinte:.1f}% (correct horses regardless of order)")
        print(f"  Quinté Désordre: {top5_quinte_desordre_wins}/{total_races} ({top5_quinte_desordre_wins/total_races*100:.1f}%) - All 5 correct in any order")
        print(f"  Bonus 3: {top5_bonus3_wins}/{total_races} ({top5_bonus3_wins/total_races*100:.1f}%) - Top 3 in any order")
        print(f"  Bonus 4: {top5_bonus4_wins}/{total_races} ({top5_bonus4_wins/total_races*100:.1f}%) - Top 4 in any order")

        print(f"\n{'='*80}")
        print(f"QUINTÉ PREDICTIONS WITH TOP 6 (Combiné with 6th horse)")
        print(f"{'='*80}")
        print(f"  Quinté Désordre: {top6_quinte_desordre_wins}/{total_races} ({top6_quinte_desordre_wins/total_races*100:.1f}%) - All 5 correct in any order")
        print(f"  Bonus 3: {top6_bonus3_wins}/{total_races} ({top6_bonus3_wins/total_races*100:.1f}%) - Top 3 in any order")
        print(f"  Bonus 4: {top6_bonus4_wins}/{total_races} ({top6_bonus4_wins/total_races*100:.1f}%) - Top 4 in any order")

        print(f"\n{'='*80}")
        print(f"IMPACT OF ADDING 6TH HORSE (Improvement)")
        print(f"{'='*80}")
        quinte_improvement = top6_quinte_desordre_wins - top5_quinte_desordre_wins
        bonus3_improvement = top6_bonus3_wins - top5_bonus3_wins
        bonus4_improvement = top6_bonus4_wins - top5_bonus4_wins

        print(f"  Quinté Désordre: {quinte_improvement:+d} wins ({quinte_improvement/total_races*100:+.1f}%)")
        print(f"  Bonus 3: {bonus3_improvement:+d} wins ({bonus3_improvement/total_races*100:+.1f}%)")
        print(f"  Bonus 4: {bonus4_improvement:+d} wins ({bonus4_improvement/total_races*100:+.1f}%)")

        if quinte_improvement > 0:
            print(f"\n  ✓ Adding 6th horse improves results!")
        elif quinte_improvement == 0:
            print(f"\n  = No change by adding 6th horse")
        else:
            print(f"\n  ✗ Adding 6th horse does not help")

        print(f"\n{'='*170}")
        print(f"PER-RACE QUINTÉ COMPARISON (Top 5 vs Top 6)")
        print(f"{'='*170}")
        print(f"{'Date':<12} {'Race':<12} {'Top 5 Pred':<25} {'Top 6 Pred':<30} {'Actual':<20} {'%Ord':<6} {'5-QD':<5} {'5-B3':<5} {'5-B4':<5} {'6-QD':<5} {'6-B3':<5} {'6-B4':<5} {'MAE':<6}")
        print(f"{'-'*170}")

        # Sort by race comp ID in descending order (newest first)
        sorted_races = sorted(metrics['race_metrics'], key=lambda x: x['jour'], reverse=True)

        for race in sorted_races:
            top5_pred = race['predicted_top5']
            top6_pred = race['predicted_top6']
            actual = race['actual_arrive']
            jour = race.get('jour', '')

            # Top 5 indicators
            t5_qd = "✓" if race['top5_quinte_desordre'] else "✗"
            t5_b3 = "✓" if race['top5_bonus3'] else "✗"
            t5_b4 = "✓" if race['top5_bonus4'] else "✗"

            # Top 6 indicators
            t6_qd = "✓" if race['top6_quinte_desordre'] else "✗"
            t6_b3 = "✓" if race['top6_bonus3'] else "✗"
            t6_b4 = "✓" if race['top6_bonus4'] else "✗"

            print(f"{jour:<12} {race['comp']:<12} {top5_pred:<25} {top6_pred:<30} {actual:<20} "
                  f"{race['pct_exact_order']:<5.1f}% {t5_qd:<5} {t5_b3:<5} {t5_b4:<5} "
                  f"{t6_qd:<5} {t6_b3:<5} {t6_b4:<5} "
                  f"{race['mae']:<6.2f}")

        print(f"{'-'*170}")
        print(f"\nLegend:")
        print(f"  QD = Quinté Désordre (all 5 correct in any order)")
        print(f"  B3 = Bonus 3 (top 3 correct in any order)")
        print(f"  B4 = Bonus 4 (top 4 correct in any order)")
        print(f"  5-* = Using top 5 predictions, 6-* = Using top 6 predictions")

        # Show best and worst predictions
        print(f"\nBest Predictions (lowest error):")
        df_with_error = df_merged.copy()
        df_with_error['error'] = abs(df_with_error['predicted_position'] - df_with_error['actual_position'])
        best = df_with_error.nsmallest(5, 'error')[['comp', 'numero', 'predicted_position', 'actual_position', 'error']]
        print(best.to_string(index=False))

        print(f"\nWorst Predictions (highest error):")
        worst = df_with_error.nlargest(5, 'error')[['comp', 'numero', 'predicted_position', 'actual_position', 'error']]
        print(worst.to_string(index=False))

    def save_comparison(self, df_merged: pd.DataFrame, metrics: Dict, output_dir: str = 'predictions'):
        """
        Save comparison results to files.

        Args:
            df_merged: DataFrame with merged predictions and results
            metrics: Dict with calculated metrics
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed comparison CSV (all horses)
        comparison_file = output_path / f"quinte_comparison_{timestamp}.csv"
        df_merged['error'] = abs(df_merged['predicted_position'] - df_merged['actual_position'])
        df_merged.to_csv(comparison_file, index=False)
        self.log_info(f"✓ Saved detailed comparison to {comparison_file}")

        # Save per-race quinté summary CSV
        race_summary_file = output_path / f"quinte_race_summary_{timestamp}.csv"
        race_summary_df = pd.DataFrame(metrics['race_metrics'])
        # Reorder columns for clarity
        column_order = ['comp', 'quinte_predicted', 'general_predicted', 'actual_arrive',
                       'quinte_pct_exact_order', 'quinte_pct_in_quinte', 'quinte_horses_in_quinte',
                       'quinte_bonus3', 'quinte_bonus4',
                       'general_pct_exact_order', 'general_pct_in_quinte', 'general_horses_in_quinte',
                       'general_bonus3', 'general_bonus4',
                       'horses', 'mae', 'correlation', 'winner_correct', 'hippo', 'prixnom']
        race_summary_df = race_summary_df[[col for col in column_order if col in race_summary_df.columns]]
        race_summary_df.to_csv(race_summary_file, index=False)
        self.log_info(f"✓ Saved race summary to {race_summary_file}")

        # Save metrics JSON
        metrics_file = output_path / f"quinte_metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        self.log_info(f"✓ Saved metrics to {metrics_file}")

    def compare(self, race_date: Optional[str] = None, race_comps: Optional[List[str]] = None,
                output_dir: str = 'predictions') -> Dict:
        """
        Complete comparison workflow.

        Args:
            race_date: Date to load predictions for (YYYY-MM-DD), or None for all
            race_comps: Specific race IDs to compare, or None to use race_date
            output_dir: Directory to save comparison results

        Returns:
            Dict with comparison results
        """
        self.log_info("=" * 60)
        self.log_info("STARTING QUINTÉ RESULT COMPARISON")
        self.log_info("=" * 60)

        # Step 1: Load predictions from database
        df_predictions = self.load_predictions(race_date=race_date, race_comps=race_comps)

        if len(df_predictions) == 0:
            return {
                'status': 'no_predictions',
                'message': 'No predictions found in database'
            }

        # Step 2: Load actual results for ALL races in predictions
        race_comps = df_predictions['comp'].unique().tolist()
        results_map = self.load_actual_results(race_comps=race_comps)

        # Step 4: Load general model predictions from daily_race
        general_predictions = self.load_general_model_predictions(race_comps)

        # Step 5: Merge predictions and results
        df_merged = self.merge_predictions_and_results(df_predictions, results_map)

        if len(df_merged) == 0:
            self.log_info("⚠ No matching results found")
            return {
                'status': 'no_results',
                'message': 'No matching results found'
            }

        # Step 6: Calculate metrics (including general model comparison)
        metrics = self.calculate_metrics(df_merged, general_predictions)

        # Step 5: Print summary
        self.print_summary(metrics, df_merged)

        # Step 6: Save comparison
        self.save_comparison(df_merged, metrics, output_dir)

        self.log_info("=" * 60)
        self.log_info("COMPARISON COMPLETE")
        self.log_info("=" * 60)

        return {
            'status': 'success',
            'metrics': metrics,
            'comparison_data': df_merged
        }


def main():
    """Main entry point for result comparison script."""
    parser = argparse.ArgumentParser(
        description='Compare quinté predictions with actual results from database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all predictions in database
  python race_prediction/compare_quinte_results.py

  # Compare specific date
  python race_prediction/compare_quinte_results.py --date 2025-11-03

  # Compare specific race IDs
  python race_prediction/compare_quinte_results.py --race-ids 1621325 1621326

  # With verbose output
  python race_prediction/compare_quinte_results.py --date 2025-11-03 --verbose
        """
    )
    parser.add_argument('--date', type=str, help='Race date to compare (YYYY-MM-DD). If not specified, compares all predictions.')
    parser.add_argument('--race-ids', nargs='+', help='Specific race IDs to compare (space-separated)')
    parser.add_argument('--output-dir', type=str, default='predictions', help='Directory for saving comparison results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Run comparison
    comparator = QuinteResultComparator(verbose=args.verbose)
    result = comparator.compare(
        race_date=args.date,
        race_comps=args.race_ids,
        output_dir=args.output_dir
    )

    if result['status'] == 'success':
        print("\n✓ Comparison complete!")
    elif result['status'] == 'no_predictions':
        print("\n✗ No predictions found in database. Run predict_quinte.py first.")
    elif result['status'] == 'no_results':
        print("\n✗ No matching results found in database.")
    else:
        print(f"\n✗ Comparison failed: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
