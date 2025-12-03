#!/usr/bin/env python3
"""
Comprehensive Quinté Prediction Failure Analysis

Analyzes why the model excels at winner prediction (32.8%) but fails at
Quinté Désordre (1.5%) and Bonus 4 (6.1%) despite good overall accuracy.

Focus: Identify specific failure patterns in positions 4-5 predictions.

Usage:
    python scripts/analyze_quinte_failures.py
    python scripts/analyze_quinte_failures.py --date 2025-11-03
    python scripts/analyze_quinte_failures.py --min-races 50
"""

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from scipy import stats

# Optional visualization dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    print("⚠ Warning: matplotlib/seaborn not installed. Visualizations will be skipped.")
    print("  Install with: pip install matplotlib seaborn")

from utils.env_setup import AppConfig, get_sqlite_dbpath


class QuinteFailureAnalyzer:
    """
    Deep analysis of quinté prediction failures.

    Identifies:
    - Why positions 4-5 fail despite good winner accuracy
    - Positional drift patterns
    - Favorite vs outsider prediction bias
    - Race characteristics that cause failures
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = True):
        """Initialize the failure analyzer."""
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        self.db_path = get_sqlite_dbpath(self.db_type)

        self.log_info(f"Initialized QuinteFailureAnalyzer with database: {self.db_type}")

    def log_info(self, message: str):
        """Simple logging method."""
        if self.verbose:
            print(f"[FailureAnalysis] {message}")

    def load_predictions_with_results(self,
                                     race_date: Optional[str] = None,
                                     min_races: int = 30) -> pd.DataFrame:
        """
        Load predictions with actual results and additional horse metadata.

        Args:
            race_date: Specific date to analyze (YYYY-MM-DD)
            min_races: Minimum number of races to analyze

        Returns:
            DataFrame with predictions, results, and metadata
        """
        self.log_info("Loading predictions with results from database...")

        conn = sqlite3.connect(self.db_path)

        # Query to join predictions with actual results
        # We need: predictions, actual positions, horse metadata (odds, name, trainer)
        query = """
        WITH race_results AS (
            SELECT
                comp,
                actual_results,
                jour,
                hippo,
                prixnom
            FROM daily_race
            WHERE actual_results IS NOT NULL
        ),
        predictions AS (
            SELECT
                qp.race_id as comp,
                qp.race_date as jour,
                qp.track as hippo,
                qp.race_name as prixnom,
                qp.horse_number as numero,
                qp.horse_id as idche,
                qp.final_prediction as predicted_position,
                qp.predicted_rank,
                qp.quinte_rf_prediction as rf_predicted_position,
                qp.quinte_tabnet_prediction as tabnet_predicted_position,
                qp.competitive_adjustment
            FROM quinte_predictions qp
            {date_filter}
            ORDER BY qp.race_date DESC, qp.race_id, qp.predicted_rank
        )
        SELECT
            p.*,
            rr.actual_results
        FROM predictions p
        LEFT JOIN race_results rr ON p.comp = rr.comp
        WHERE rr.actual_results IS NOT NULL
        """

        if race_date:
            date_filter = f"WHERE qp.race_date = '{race_date}'"
        else:
            date_filter = ""

        query = query.format(date_filter=date_filter)
        df = pd.read_sql_query(query, conn)

        if len(df) == 0:
            self.log_info("⚠ No predictions with results found")
            conn.close()
            return pd.DataFrame()

        # Parse actual_results string to get actual positions
        df['actual_position'] = df.apply(self._parse_actual_position, axis=1)

        # Get additional horse metadata (odds, names) from partant table via daily_race
        self._enrich_with_horse_metadata(df, conn)

        conn.close()

        n_races = df['comp'].nunique()
        self.log_info(f"Loaded {len(df)} predictions from {n_races} races")

        if n_races < min_races:
            self.log_info(f"⚠ Warning: Only {n_races} races found (minimum recommended: {min_races})")

        return df

    def _parse_actual_position(self, row) -> Optional[int]:
        """Parse actual position from actual_results string."""
        try:
            if pd.isna(row['actual_results']) or row['actual_results'].strip() == '':
                return None

            # actual_results format: "13-10-7-1-8-3-4-2-14-16"
            # Position in list = finish position
            horse_numbers = row['actual_results'].strip().split('-')
            horse_num = int(row['numero'])

            for position, num_str in enumerate(horse_numbers, start=1):
                if int(num_str.strip()) == horse_num:
                    return position

            return None  # Horse didn't finish in recorded positions
        except (ValueError, AttributeError, TypeError):
            return None

    def _enrich_with_horse_metadata(self, df: pd.DataFrame, conn: sqlite3.Connection):
        """
        Enrich DataFrame with horse metadata (odds, names, trainer).

        Modifies df in-place by parsing participants JSON from daily_race.
        """
        self.log_info("Enriching with horse metadata (odds, names, trainer)...")

        # Get unique race comps
        race_comps = df['comp'].unique().tolist()

        # Query participants JSON for these races
        placeholders = ','.join('?' * len(race_comps))
        query = f"""
        SELECT comp, participants
        FROM daily_race
        WHERE comp IN ({placeholders}) AND participants IS NOT NULL
        """

        cursor = conn.cursor()
        cursor.execute(query, race_comps)
        rows = cursor.fetchall()

        # Parse JSON and create metadata mapping
        metadata_map = {}  # {(comp, numero): {cheval_nom, cotedirect, trainer}}

        for comp, participants_json in rows:
            try:
                participants = json.loads(participants_json)
                for horse in participants:
                    numero = int(horse.get('numero', 0))
                    if numero > 0:
                        key = (str(comp), numero)
                        metadata_map[key] = {
                            'cheval_nom': horse.get('cheval', 'Unknown'),
                            'cotedirect': float(horse.get('cotedirect', 0)) if horse.get('cotedirect') else None,
                            'trainer': horse.get('entraineur', 'Unknown')
                        }
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                self.log_info(f"Warning: Could not parse participants for race {comp}: {e}")
                continue

        # Enrich dataframe with metadata
        df['cheval_nom'] = df.apply(
            lambda row: metadata_map.get((str(row['comp']), int(row['numero'])), {}).get('cheval_nom', 'Unknown'),
            axis=1
        )
        df['cotedirect'] = df.apply(
            lambda row: metadata_map.get((str(row['comp']), int(row['numero'])), {}).get('cotedirect', None),
            axis=1
        )
        df['trainer'] = df.apply(
            lambda row: metadata_map.get((str(row['comp']), int(row['numero'])), {}).get('trainer', 'Unknown'),
            axis=1
        )

        enriched_count = df['cheval_nom'].notna().sum()
        self.log_info(f"Enriched {enriched_count}/{len(df)} predictions with metadata")

    def analyze_positional_drift(self, df: pd.DataFrame) -> Dict:
        """
        Analyze how predicted positions drift from actual positions.

        For each predicted position 1-5:
        - Distribution of actual positions
        - Average drift (positive = finished worse than predicted)
        - Success rate (% that finish in top 5)

        Args:
            df: DataFrame with predictions and results

        Returns:
            Dict with positional drift analysis
        """
        self.log_info("\n" + "="*80)
        self.log_info("POSITIONAL DRIFT ANALYSIS")
        self.log_info("="*80)

        drift_analysis = {}

        # Analyze each predicted position
        for pred_pos in range(1, 6):
            # Filter to horses predicted at this position
            pred_mask = df['predicted_rank'] == pred_pos
            pred_df = df[pred_mask].copy()

            # Only analyze horses with actual positions
            pred_df = pred_df[pred_df['actual_position'].notna()]

            if len(pred_df) == 0:
                continue

            # Calculate drift (positive = finished worse than predicted)
            pred_df['drift'] = pred_df['actual_position'] - pred_pos

            # Calculate metrics
            actual_positions = pred_df['actual_position'].values
            drifts = pred_df['drift'].values

            # Success rate: % that finish in top 5
            success_rate = (actual_positions <= 5).mean()

            # Average drift
            avg_drift = drifts.mean()
            median_drift = np.median(drifts)
            std_drift = drifts.std()

            # Distribution of actual positions
            actual_pos_dist = Counter(actual_positions)

            # Common drift destinations
            common_actual = Counter(actual_positions).most_common(5)

            drift_analysis[pred_pos] = {
                'n_predictions': len(pred_df),
                'success_rate': float(success_rate),
                'avg_drift': float(avg_drift),
                'median_drift': float(median_drift),
                'std_drift': float(std_drift),
                'actual_pos_distribution': dict(actual_pos_dist),
                'common_actual_positions': common_actual,
                'pct_in_top5': float((actual_positions <= 5).mean() * 100),
                'pct_in_top3': float((actual_positions <= 3).mean() * 100),
                'pct_exact': float((actual_positions == pred_pos).mean() * 100),
            }

            # Print summary
            print(f"\n{'─'*80}")
            print(f"PREDICTED POSITION {pred_pos}")
            print(f"{'─'*80}")
            print(f"  Predictions analyzed: {len(pred_df)}")
            print(f"  Success rate (finished top 5): {success_rate:.1%}")
            print(f"  Exact position accuracy: {drift_analysis[pred_pos]['pct_exact']:.1f}%")
            print(f"  Average drift: {avg_drift:+.2f} positions (+ = finished worse)")
            print(f"  Median drift: {median_drift:+.1f} positions")
            print(f"  Std deviation: {std_drift:.2f} positions")
            print(f"\n  Most common actual positions:")
            for actual_pos, count in common_actual:
                pct = (count / len(pred_df)) * 100
                print(f"    Position {int(actual_pos):2d}: {count:3d} times ({pct:5.1f}%)")

        return drift_analysis

    def analyze_missed_quintes(self, df: pd.DataFrame) -> Dict:
        """
        Analyze races where we got the winner but missed quinté.

        Focus on:
        - Which predicted positions most often miss (4th? 5th?)
        - What horses replace them in actual top 5
        - Are they favorites or outsiders?

        Args:
            df: DataFrame with predictions and results

        Returns:
            Dict with missed quinté analysis
        """
        self.log_info("\n" + "="*80)
        self.log_info("MISSED QUINTÉ ANALYSIS")
        self.log_info("="*80)

        missed_analysis = {
            'races_analyzed': 0,
            'winner_correct_quinte_missed': 0,
            'races_detail': []
        }

        # Group by race
        for race_comp, race_df in df.groupby('comp'):
            race_df = race_df.copy()

            # Get predicted top 5 and actual top 5
            predicted_top5 = set(race_df[race_df['predicted_rank'] <= 5]['numero'].values)

            # Get actual top 5 (horses that actually finished in top 5)
            actual_top5_df = race_df[race_df['actual_position'] <= 5]
            actual_top5 = set(actual_top5_df['numero'].values)

            # Check if we got the winner
            predicted_winner = race_df[race_df['predicted_rank'] == 1]['numero'].values
            actual_winner = race_df[race_df['actual_position'] == 1]['numero'].values

            if len(predicted_winner) == 0 or len(actual_winner) == 0:
                continue

            winner_correct = (predicted_winner[0] == actual_winner[0])
            quinte_correct = (predicted_top5 == actual_top5)

            missed_analysis['races_analyzed'] += 1

            # Focus on races where winner was correct but quinté was missed
            if winner_correct and not quinte_correct:
                missed_analysis['winner_correct_quinte_missed'] += 1

                # Identify which horses we predicted but didn't finish top 5
                false_positives = predicted_top5 - actual_top5

                # Identify which horses finished top 5 but we didn't predict
                false_negatives = actual_top5 - predicted_top5

                # Get details of false positives (horses we overestimated)
                fp_details = []
                for numero in false_positives:
                    horse = race_df[race_df['numero'] == numero].iloc[0]
                    fp_details.append({
                        'numero': int(numero),
                        'nom': horse.get('cheval_nom', 'Unknown'),
                        'predicted_rank': int(horse['predicted_rank']),
                        'predicted_position': float(horse['predicted_position']),
                        'actual_position': int(horse['actual_position']) if pd.notna(horse['actual_position']) else None,
                        'cotedirect': float(horse['cotedirect']) if pd.notna(horse.get('cotedirect')) else None,
                        'drift': int(horse['actual_position'] - horse['predicted_rank']) if pd.notna(horse['actual_position']) else None
                    })

                # Get details of false negatives (horses we underestimated)
                fn_details = []
                for numero in false_negatives:
                    horse = race_df[race_df['numero'] == numero].iloc[0]
                    fn_details.append({
                        'numero': int(numero),
                        'nom': horse.get('cheval_nom', 'Unknown'),
                        'predicted_rank': int(horse['predicted_rank']),
                        'predicted_position': float(horse['predicted_position']),
                        'actual_position': int(horse['actual_position']) if pd.notna(horse['actual_position']) else None,
                        'cotedirect': float(horse['cotedirect']) if pd.notna(horse.get('cotedirect')) else None,
                        'rank_error': int(horse['predicted_rank'] - horse['actual_position']) if pd.notna(horse['actual_position']) else None
                    })

                missed_analysis['races_detail'].append({
                    'comp': race_comp,
                    'jour': race_df['jour'].iloc[0],
                    'hippo': race_df['hippo'].iloc[0],
                    'prixnom': race_df['prixnom'].iloc[0],
                    'predicted_top5': sorted(list(predicted_top5)),
                    'actual_top5': sorted(list(actual_top5)),
                    'false_positives': fp_details,
                    'false_negatives': fn_details,
                    'n_correct_in_top5': len(predicted_top5 & actual_top5)
                })

        # Summary statistics
        if missed_analysis['races_analyzed'] > 0:
            pct_winner_correct = (missed_analysis['winner_correct_quinte_missed'] /
                                 missed_analysis['races_analyzed']) * 100
        else:
            pct_winner_correct = 0

        print(f"\n{'─'*80}")
        print(f"RACES WITH WINNER CORRECT BUT QUINTÉ MISSED")
        print(f"{'─'*80}")
        print(f"  Total races analyzed: {missed_analysis['races_analyzed']}")
        print(f"  Winner correct, quinté missed: {missed_analysis['winner_correct_quinte_missed']} "
              f"({pct_winner_correct:.1f}%)")

        # Analyze patterns in false positives and false negatives
        all_fp = []
        all_fn = []
        for race in missed_analysis['races_detail']:
            all_fp.extend(race['false_positives'])
            all_fn.extend(race['false_negatives'])

        if len(all_fp) > 0:
            print(f"\n  FALSE POSITIVES (predicted top 5, finished 6+):")
            print(f"    Total: {len(all_fp)} horses")

            # Average predicted rank of false positives
            fp_ranks = [h['predicted_rank'] for h in all_fp]
            fp_rank_dist = Counter(fp_ranks)
            print(f"    Predicted rank distribution:")
            for rank in sorted(fp_rank_dist.keys()):
                count = fp_rank_dist[rank]
                pct = (count / len(all_fp)) * 100
                print(f"      Rank {rank}: {count} ({pct:.1f}%)")

            # Average odds of false positives
            fp_odds = [h['cotedirect'] for h in all_fp if h['cotedirect'] is not None]
            if len(fp_odds) > 0:
                avg_fp_odds = np.mean(fp_odds)
                median_fp_odds = np.median(fp_odds)
                print(f"    Average odds: {avg_fp_odds:.2f} (median: {median_fp_odds:.2f})")

            # Average actual position
            fp_actual = [h['actual_position'] for h in all_fp if h['actual_position'] is not None]
            if len(fp_actual) > 0:
                avg_fp_actual = np.mean(fp_actual)
                print(f"    Average actual position: {avg_fp_actual:.1f}")

        if len(all_fn) > 0:
            print(f"\n  FALSE NEGATIVES (finished top 5, predicted 6+):")
            print(f"    Total: {len(all_fn)} horses")

            # Average predicted rank of false negatives
            fn_ranks = [h['predicted_rank'] for h in all_fn]
            if len(fn_ranks) > 0:
                avg_fn_rank = np.mean(fn_ranks)
                median_fn_rank = np.median(fn_ranks)
                print(f"    Average predicted rank: {avg_fn_rank:.1f} (median: {median_fn_rank:.1f})")

            # Average odds of false negatives
            fn_odds = [h['cotedirect'] for h in all_fn if h['cotedirect'] is not None]
            if len(fn_odds) > 0:
                avg_fn_odds = np.mean(fn_odds)
                median_fn_odds = np.median(fn_odds)
                print(f"    Average odds: {avg_fn_odds:.2f} (median: {median_fn_odds:.2f})")

            # Average actual position
            fn_actual = [h['actual_position'] for h in all_fn if h['actual_position'] is not None]
            if len(fn_actual) > 0:
                avg_fn_actual = np.mean(fn_actual)
                print(f"    Average actual position: {avg_fn_actual:.1f}")

        return missed_analysis

    def analyze_favorite_vs_outsider(self, df: pd.DataFrame) -> Dict:
        """
        Analyze prediction performance for favorites vs outsiders.

        Using cotedirect (odds):
        - Do we overpredict favorites (low odds)?
        - Do we miss outsiders (high odds) that place well?

        Args:
            df: DataFrame with predictions and results

        Returns:
            Dict with favorite vs outsider analysis
        """
        self.log_info("\n" + "="*80)
        self.log_info("FAVORITE VS OUTSIDER ANALYSIS")
        self.log_info("="*80)

        # Filter to horses with odds data
        df_odds = df[df['cotedirect'].notna()].copy()

        if len(df_odds) == 0:
            print("⚠ No odds data available for analysis")
            return {}

        # Categorize horses by odds
        # Favorite: odds <= 5
        # Mid-range: 5 < odds <= 15
        # Outsider: odds > 15
        df_odds['odds_category'] = pd.cut(
            df_odds['cotedirect'],
            bins=[0, 5, 15, float('inf')],
            labels=['Favorite (≤5)', 'Mid-range (5-15)', 'Outsider (>15)']
        )

        analysis = {}

        print(f"\n{'─'*80}")
        print(f"PREDICTION ACCURACY BY ODDS CATEGORY")
        print(f"{'─'*80}")

        for category in ['Favorite (≤5)', 'Mid-range (5-15)', 'Outsider (>15)']:
            cat_df = df_odds[df_odds['odds_category'] == category].copy()

            if len(cat_df) == 0:
                continue

            # Only analyze horses with actual positions
            cat_df = cat_df[cat_df['actual_position'].notna()]

            if len(cat_df) == 0:
                continue

            # Calculate metrics
            avg_predicted_rank = cat_df['predicted_rank'].mean()
            avg_actual_position = cat_df['actual_position'].mean()
            avg_drift = (cat_df['actual_position'] - cat_df['predicted_rank']).mean()

            # Top 5 accuracy
            predicted_top5_mask = cat_df['predicted_rank'] <= 5
            actual_top5_mask = cat_df['actual_position'] <= 5

            n_predicted_top5 = predicted_top5_mask.sum()
            n_actual_top5 = actual_top5_mask.sum()
            n_correct_top5 = (predicted_top5_mask & actual_top5_mask).sum()

            # Precision: of horses we predicted top 5, how many actually finished top 5?
            precision = n_correct_top5 / n_predicted_top5 if n_predicted_top5 > 0 else 0

            # Recall: of horses that finished top 5, how many did we predict top 5?
            recall = n_correct_top5 / n_actual_top5 if n_actual_top5 > 0 else 0

            analysis[category] = {
                'n_horses': len(cat_df),
                'avg_predicted_rank': float(avg_predicted_rank),
                'avg_actual_position': float(avg_actual_position),
                'avg_drift': float(avg_drift),
                'n_predicted_top5': int(n_predicted_top5),
                'n_actual_top5': int(n_actual_top5),
                'n_correct_top5': int(n_correct_top5),
                'precision': float(precision),
                'recall': float(recall)
            }

            print(f"\n{category}:")
            print(f"  Horses analyzed: {len(cat_df)}")
            print(f"  Avg predicted rank: {avg_predicted_rank:.2f}")
            print(f"  Avg actual position: {avg_actual_position:.2f}")
            print(f"  Avg drift: {avg_drift:+.2f} (+ = we overestimated)")
            print(f"  Top 5 prediction:")
            print(f"    Predicted top 5: {n_predicted_top5}")
            print(f"    Actually top 5: {n_actual_top5}")
            print(f"    Correct: {n_correct_top5}")
            print(f"    Precision: {precision:.1%} (of predicted top 5, % actually top 5)")
            print(f"    Recall: {recall:.1%} (of actual top 5, % we predicted)")

        return analysis

    def analyze_position_4_5_failures(self, df: pd.DataFrame) -> Dict:
        """
        Deep dive into positions 4-5 specifically.

        Why do these positions fail?
        - Characteristics of horses predicted 4-5
        - Characteristics of horses that actually finish 4-5
        - What differentiates them?

        Args:
            df: DataFrame with predictions and results

        Returns:
            Dict with position 4-5 analysis
        """
        self.log_info("\n" + "="*80)
        self.log_info("POSITION 4-5 DEEP DIVE ANALYSIS")
        self.log_info("="*80)

        analysis = {}

        # Analyze predicted position 4-5
        predicted_4_5 = df[df['predicted_rank'].isin([4, 5])].copy()
        predicted_4_5 = predicted_4_5[predicted_4_5['actual_position'].notna()]

        # Analyze actual position 4-5
        actual_4_5 = df[df['actual_position'].isin([4, 5])].copy()

        print(f"\n{'─'*80}")
        print(f"PREDICTED POSITION 4-5 ANALYSIS")
        print(f"{'─'*80}")

        if len(predicted_4_5) > 0:
            # Success rate
            success_4_5 = (predicted_4_5['actual_position'] <= 5).mean()
            exact_4_5 = predicted_4_5['actual_position'].isin([4, 5]).mean()

            print(f"  Total predictions at positions 4-5: {len(predicted_4_5)}")
            print(f"  Actually finished top 5: {success_4_5:.1%}")
            print(f"  Actually finished 4-5: {exact_4_5:.1%}")

            # Where do they actually finish?
            actual_dist = Counter(predicted_4_5['actual_position'].values)
            print(f"\n  Actual finish positions:")
            for pos in sorted(actual_dist.keys()):
                count = actual_dist[pos]
                pct = (count / len(predicted_4_5)) * 100
                print(f"    Position {int(pos):2d}: {count:3d} ({pct:5.1f}%)")

            # Odds analysis
            if 'cotedirect' in predicted_4_5.columns:
                odds_4_5 = predicted_4_5['cotedirect'].dropna()
                if len(odds_4_5) > 0:
                    print(f"\n  Odds statistics:")
                    print(f"    Mean: {odds_4_5.mean():.2f}")
                    print(f"    Median: {odds_4_5.median():.2f}")
                    print(f"    Min: {odds_4_5.min():.2f}")
                    print(f"    Max: {odds_4_5.max():.2f}")

            analysis['predicted_4_5'] = {
                'n_predictions': len(predicted_4_5),
                'success_rate': float(success_4_5),
                'exact_rate': float(exact_4_5),
                'actual_position_dist': dict(actual_dist)
            }

        print(f"\n{'─'*80}")
        print(f"ACTUAL POSITION 4-5 ANALYSIS")
        print(f"{'─'*80}")

        if len(actual_4_5) > 0:
            # What ranks did we predict for these horses?
            pred_rank_dist = Counter(actual_4_5['predicted_rank'].values)

            print(f"  Total horses that finished 4-5: {len(actual_4_5)}")
            print(f"\n  Our predicted ranks for these horses:")
            for rank in sorted(pred_rank_dist.keys()):
                count = pred_rank_dist[rank]
                pct = (count / len(actual_4_5)) * 100
                indicator = "✓" if rank in [4, 5] else "✗"
                print(f"    Rank {int(rank):2d}: {count:3d} ({pct:5.1f}%) {indicator}")

            # How many did we correctly identify?
            correct_4_5 = actual_4_5['predicted_rank'].isin([4, 5]).sum()
            correct_top5 = actual_4_5['predicted_rank'].isin([1, 2, 3, 4, 5]).sum()

            print(f"\n  Correctly predicted at rank 4-5: {correct_4_5}/{len(actual_4_5)} ({correct_4_5/len(actual_4_5):.1%})")
            print(f"  Predicted in top 5: {correct_top5}/{len(actual_4_5)} ({correct_top5/len(actual_4_5):.1%})")

            # Odds analysis
            if 'cotedirect' in actual_4_5.columns:
                odds_actual_4_5 = actual_4_5['cotedirect'].dropna()
                if len(odds_actual_4_5) > 0:
                    print(f"\n  Odds statistics for horses that actually finished 4-5:")
                    print(f"    Mean: {odds_actual_4_5.mean():.2f}")
                    print(f"    Median: {odds_actual_4_5.median():.2f}")
                    print(f"    Min: {odds_actual_4_5.min():.2f}")
                    print(f"    Max: {odds_actual_4_5.max():.2f}")

            analysis['actual_4_5'] = {
                'n_horses': len(actual_4_5),
                'predicted_rank_dist': dict(pred_rank_dist),
                'correct_4_5': int(correct_4_5),
                'correct_top5': int(correct_top5),
                'recall_4_5': float(correct_4_5 / len(actual_4_5)) if len(actual_4_5) > 0 else 0,
                'recall_top5': float(correct_top5 / len(actual_4_5)) if len(actual_4_5) > 0 else 0
            }

        return analysis

    def create_visualizations(self, df: pd.DataFrame, output_dir: str = 'analysis_output'):
        """
        Create visualizations for failure analysis.

        Args:
            df: DataFrame with predictions and results
            output_dir: Directory to save visualizations
        """
        self.log_info("\n" + "="*80)
        self.log_info("CREATING VISUALIZATIONS")
        self.log_info("="*80)

        if not VISUALIZATIONS_AVAILABLE:
            print("⚠ Skipping visualizations (matplotlib/seaborn not installed)")
            print("  Install with: pip install matplotlib seaborn")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Filter to horses with actual positions
        df_valid = df[df['actual_position'].notna()].copy()

        if len(df_valid) == 0:
            print("⚠ No valid data for visualizations")
            return

        # Set style
        sns.set_style("whitegrid")

        # 1. Positional drift heatmap
        self._plot_positional_drift_heatmap(df_valid, output_path)

        # 2. Success rate by predicted position
        self._plot_success_rate_by_position(df_valid, output_path)

        # 3. Odds distribution comparison
        self._plot_odds_distribution(df_valid, output_path)

        # 4. Position 4-5 specific analysis
        self._plot_position_4_5_analysis(df_valid, output_path)

        self.log_info(f"✓ Visualizations saved to {output_path}")

    def _plot_positional_drift_heatmap(self, df: pd.DataFrame, output_path: Path):
        """Create heatmap showing drift from predicted to actual positions."""
        # Create confusion matrix: predicted rank vs actual position
        predicted_ranks = []
        actual_positions = []

        for _, row in df.iterrows():
            pred_rank = int(row['predicted_rank'])
            actual_pos = int(row['actual_position'])

            # Limit to top 10 for visualization
            if pred_rank <= 10 and actual_pos <= 10:
                predicted_ranks.append(pred_rank)
                actual_positions.append(actual_pos)

        # Create confusion matrix
        matrix = np.zeros((10, 10))
        for pred, actual in zip(predicted_ranks, actual_positions):
            matrix[pred-1, actual-1] += 1

        # Normalize by row (each predicted rank sums to 100%)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_pct = np.divide(matrix, row_sums, where=row_sums!=0) * 100

        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix_pct, annot=True, fmt='.1f', cmap='RdYlGn_r',
                   xticklabels=range(1, 11), yticklabels=range(1, 11),
                   cbar_kws={'label': '% of predictions'})
        plt.xlabel('Actual Position', fontsize=12)
        plt.ylabel('Predicted Rank', fontsize=12)
        plt.title('Positional Drift: Predicted Rank vs Actual Position\n(% of horses from each predicted rank)',
                 fontsize=14, fontweight='bold')

        # Highlight the diagonal (exact predictions)
        for i in range(10):
            plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=2))

        plt.tight_layout()
        plt.savefig(output_path / 'positional_drift_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved positional drift heatmap")

    def _plot_success_rate_by_position(self, df: pd.DataFrame, output_path: Path):
        """Plot success rate (finished top 5) by predicted position."""
        success_rates = []
        positions = range(1, 11)

        for pos in positions:
            pos_df = df[df['predicted_rank'] == pos]
            if len(pos_df) > 0:
                success_rate = (pos_df['actual_position'] <= 5).mean() * 100
                success_rates.append(success_rate)
            else:
                success_rates.append(0)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(positions, success_rates, color=['#2ecc71' if sr >= 70 else '#e74c3c' for sr in success_rates])

        # Add value labels on bars
        for i, (pos, sr) in enumerate(zip(positions, success_rates)):
            plt.text(pos, sr + 2, f'{sr:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Add horizontal line at 80% (ideal success rate for top 5)
        plt.axhline(y=80, color='blue', linestyle='--', linewidth=2, label='Target: 80%')

        plt.xlabel('Predicted Rank', fontsize=12)
        plt.ylabel('Success Rate (% Finished Top 5)', fontsize=12)
        plt.title('Success Rate by Predicted Position\n(Positions 4-5 are critical for Quinté success)',
                 fontsize=14, fontweight='bold')
        plt.xticks(positions)
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'success_rate_by_position.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved success rate by position")

    def _plot_odds_distribution(self, df: pd.DataFrame, output_path: Path):
        """Compare odds distribution for predicted top 5 vs actual top 5."""
        if 'cotedirect' not in df.columns or df['cotedirect'].isna().all():
            print("  ⚠ Skipping odds distribution (no data)")
            return

        df_odds = df[df['cotedirect'].notna()].copy()

        # Get odds for predicted top 5 and actual top 5
        predicted_top5_odds = df_odds[df_odds['predicted_rank'] <= 5]['cotedirect'].values
        actual_top5_odds = df_odds[df_odds['actual_position'] <= 5]['cotedirect'].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Histogram comparison
        axes[0].hist(predicted_top5_odds, bins=30, alpha=0.6, label='Predicted Top 5', color='blue', edgecolor='black')
        axes[0].hist(actual_top5_odds, bins=30, alpha=0.6, label='Actual Top 5', color='green', edgecolor='black')
        axes[0].set_xlabel('Odds (cotedirect)', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('Odds Distribution: Predicted vs Actual Top 5', fontweight='bold')
        axes[0].legend()
        axes[0].set_xlim(0, 50)

        # Plot 2: Box plot comparison
        box_data = [predicted_top5_odds, actual_top5_odds]
        bp = axes[1].boxplot(box_data, labels=['Predicted Top 5', 'Actual Top 5'],
                            patch_artist=True, showmeans=True)

        # Color the boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')

        axes[1].set_ylabel('Odds (cotedirect)', fontsize=11)
        axes[1].set_title('Odds Distribution Summary', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        # Add mean values as text
        mean_pred = np.mean(predicted_top5_odds)
        mean_actual = np.mean(actual_top5_odds)
        axes[1].text(1, mean_pred + 2, f'μ={mean_pred:.1f}', ha='center', fontweight='bold', color='blue')
        axes[1].text(2, mean_actual + 2, f'μ={mean_actual:.1f}', ha='center', fontweight='bold', color='green')

        plt.tight_layout()
        plt.savefig(output_path / 'odds_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved odds distribution comparison")

    def _plot_position_4_5_analysis(self, df: pd.DataFrame, output_path: Path):
        """Detailed analysis plot for positions 4-5."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Where predicted 4-5 actually finish
        predicted_4_5 = df[df['predicted_rank'].isin([4, 5])]
        actual_pos_counts = Counter(predicted_4_5['actual_position'].values)
        positions = sorted([int(p) for p in actual_pos_counts.keys() if p <= 15])
        counts = [actual_pos_counts[p] for p in positions]

        colors = ['#2ecc71' if p <= 5 else '#e74c3c' for p in positions]
        axes[0, 0].bar(positions, counts, color=colors)
        axes[0, 0].set_xlabel('Actual Position', fontsize=10)
        axes[0, 0].set_ylabel('Count', fontsize=10)
        axes[0, 0].set_title('Predicted Rank 4-5: Where They Actually Finish', fontweight='bold')
        axes[0, 0].axvline(x=5.5, color='blue', linestyle='--', linewidth=2, label='Top 5 cutoff')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Plot 2: What we predicted for horses that actually finished 4-5
        actual_4_5 = df[df['actual_position'].isin([4, 5])]
        pred_rank_counts = Counter(actual_4_5['predicted_rank'].values)
        ranks = sorted([int(r) for r in pred_rank_counts.keys() if r <= 15])
        counts = [pred_rank_counts[r] for r in ranks]

        colors = ['#2ecc71' if r in [4, 5] else '#f39c12' if r <= 5 else '#e74c3c' for r in ranks]
        axes[0, 1].bar(ranks, counts, color=colors)
        axes[0, 1].set_xlabel('Predicted Rank', fontsize=10)
        axes[0, 1].set_ylabel('Count', fontsize=10)
        axes[0, 1].set_title('Actual Position 4-5: What We Predicted', fontweight='bold')
        axes[0, 1].axvline(x=5.5, color='blue', linestyle='--', linewidth=2, label='Top 5 cutoff')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Plot 3: Drift analysis for predicted 4-5
        if len(predicted_4_5) > 0:
            predicted_4_5_copy = predicted_4_5.copy()
            predicted_4_5_copy['drift'] = predicted_4_5_copy['actual_position'] - predicted_4_5_copy['predicted_rank']
            drift_counts = Counter(predicted_4_5_copy['drift'].values)
            drifts = sorted([int(d) for d in drift_counts.keys() if -5 <= d <= 10])
            counts = [drift_counts[d] for d in drifts]

            colors = ['#2ecc71' if d == 0 else '#f39c12' if abs(d) <= 2 else '#e74c3c' for d in drifts]
            axes[1, 0].bar(drifts, counts, color=colors)
            axes[1, 0].set_xlabel('Drift (Actual - Predicted)', fontsize=10)
            axes[1, 0].set_ylabel('Count', fontsize=10)
            axes[1, 0].set_title('Predicted 4-5: Position Drift\n(Positive = finished worse)', fontweight='bold')
            axes[1, 0].axvline(x=0, color='blue', linestyle='--', linewidth=2, label='No drift')
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)

        # Plot 4: Success rate by predicted position (zoomed to 1-10)
        success_rates = []
        positions = range(1, 11)

        for pos in positions:
            pos_df = df[df['predicted_rank'] == pos]
            if len(pos_df) > 0:
                success_rate = (pos_df['actual_position'] <= 5).mean() * 100
                success_rates.append(success_rate)
            else:
                success_rates.append(0)

        colors = ['#2ecc71' if pos <= 3 else '#f39c12' if pos in [4, 5] else '#95a5a6' for pos in positions]
        bars = axes[1, 1].bar(positions, success_rates, color=colors)

        # Highlight positions 4-5
        for i, pos in enumerate(positions):
            if pos in [4, 5]:
                axes[1, 1].text(pos, success_rates[i] + 2, f'{success_rates[i]:.0f}%',
                              ha='center', va='bottom', fontweight='bold', fontsize=10)

        axes[1, 1].axhline(y=80, color='blue', linestyle='--', linewidth=2, label='Target: 80%')
        axes[1, 1].set_xlabel('Predicted Rank', fontsize=10)
        axes[1, 1].set_ylabel('Success Rate (%)', fontsize=10)
        axes[1, 1].set_title('Success Rate by Predicted Position\n(Focus on positions 4-5)', fontweight='bold')
        axes[1, 1].set_xticks(positions)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(output_path / 'position_4_5_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved position 4-5 detailed analysis")

    def generate_report(self, df: pd.DataFrame, output_dir: str = 'analysis_output'):
        """
        Generate comprehensive failure analysis report.

        Args:
            df: DataFrame with predictions and results
            output_dir: Directory to save report
        """
        self.log_info("\n" + "="*80)
        self.log_info("GENERATING COMPREHENSIVE REPORT")
        self.log_info("="*80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f'quinte_failure_analysis_{timestamp}.txt'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("QUINTÉ PREDICTION FAILURE ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Races analyzed: {df['comp'].nunique()}\n")
            f.write(f"Total predictions: {len(df)}\n")
            f.write("\n")

            # Run all analyses
            drift_analysis = self.analyze_positional_drift(df)
            missed_analysis = self.analyze_missed_quintes(df)
            fav_out_analysis = self.analyze_favorite_vs_outsider(df)
            pos_4_5_analysis = self.analyze_position_4_5_failures(df)

            # Create visualizations
            self.create_visualizations(df, output_dir)

            # Summary and recommendations
            f.write("\n" + "="*80 + "\n")
            f.write("KEY FINDINGS AND RECOMMENDATIONS\n")
            f.write("="*80 + "\n")

            # Analyze the data to provide recommendations
            if 4 in drift_analysis and 5 in drift_analysis:
                success_4 = drift_analysis[4].get('success_rate', 0)
                success_5 = drift_analysis[5].get('success_rate', 0)
                drift_4 = drift_analysis[4].get('avg_drift', 0)
                drift_5 = drift_analysis[5].get('avg_drift', 0)

                f.write(f"\n1. POSITION 4-5 PERFORMANCE:\n")
                f.write(f"   - Position 4 success rate: {success_4:.1%} (drift: {drift_4:+.2f})\n")
                f.write(f"   - Position 5 success rate: {success_5:.1%} (drift: {drift_5:+.2f})\n")

                if success_4 < 0.7 or success_5 < 0.7:
                    f.write(f"   ⚠ CRITICAL: Positions 4-5 have low success rates (<70%)\n")
                    f.write(f"   → Consider increasing margin/uncertainty for these positions\n")
                    f.write(f"   → May need separate model or additional features for mid-pack positions\n")

            if 'winner_correct_quinte_missed' in missed_analysis:
                n_races = missed_analysis['races_analyzed']
                n_missed = missed_analysis['winner_correct_quinte_missed']
                if n_races > 0:
                    pct_missed = (n_missed / n_races) * 100
                    f.write(f"\n2. WINNER CORRECT BUT QUINTÉ MISSED:\n")
                    f.write(f"   - Occurs in {n_missed}/{n_races} races ({pct_missed:.1f}%)\n")
                    f.write(f"   ✓ Good winner prediction but struggling with full top 5\n")
                    f.write(f"   → Focus on improving positions 2-5, not position 1\n")

            if fav_out_analysis:
                f.write(f"\n3. FAVORITE VS OUTSIDER BIAS:\n")
                for category, metrics in fav_out_analysis.items():
                    drift = metrics.get('avg_drift', 0)
                    precision = metrics.get('precision', 0)
                    recall = metrics.get('recall', 0)
                    f.write(f"   {category}:\n")
                    f.write(f"     - Drift: {drift:+.2f} (+ = overestimate)\n")
                    f.write(f"     - Precision: {precision:.1%}, Recall: {recall:.1%}\n")

                # Check for bias
                if 'Favorite (≤5)' in fav_out_analysis and 'Outsider (>15)' in fav_out_analysis:
                    fav_drift = fav_out_analysis['Favorite (≤5)'].get('avg_drift', 0)
                    out_drift = fav_out_analysis['Outsider (>15)'].get('avg_drift', 0)

                    if fav_drift > 1:
                        f.write(f"   ⚠ WARNING: Over-predicting favorites (drift: {fav_drift:+.2f})\n")
                        f.write(f"   → Reduce weight on recent form, increase weight on race conditions\n")

                    if out_drift < -1:
                        f.write(f"   ⚠ WARNING: Under-predicting outsiders (drift: {out_drift:+.2f})\n")
                        f.write(f"   → Consider adding upset potential features\n")

            f.write(f"\n4. RECOMMENDED ACTIONS:\n")
            f.write(f"   a) Implement position-specific models or calibration\n")
            f.write(f"   b) Add features that capture mid-pack dynamics (positions 4-8)\n")
            f.write(f"   c) Consider ensemble approach: winner model + placer model\n")
            f.write(f"   d) Investigate race-specific patterns (track, distance, surface)\n")
            f.write(f"   e) Add uncertainty quantification to identify risky predictions\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        self.log_info(f"✓ Report saved to {report_file}")

        return report_file

    def analyze(self, race_date: Optional[str] = None,
               min_races: int = 30,
               output_dir: str = 'analysis_output') -> Dict:
        """
        Run complete failure analysis.

        Args:
            race_date: Specific date to analyze (YYYY-MM-DD), or None for all
            min_races: Minimum number of races required
            output_dir: Directory to save analysis results

        Returns:
            Dict with analysis results
        """
        self.log_info("="*80)
        self.log_info("STARTING QUINTÉ FAILURE ANALYSIS")
        self.log_info("="*80)

        # Load data
        df = self.load_predictions_with_results(race_date, min_races)

        if len(df) == 0:
            return {
                'status': 'no_data',
                'message': 'No prediction data found'
            }

        # Generate comprehensive report
        report_file = self.generate_report(df, output_dir)

        self.log_info("="*80)
        self.log_info("ANALYSIS COMPLETE")
        self.log_info(f"Report: {report_file}")
        self.log_info("="*80)

        return {
            'status': 'success',
            'report_file': str(report_file),
            'n_races': df['comp'].nunique(),
            'n_predictions': len(df)
        }


def main():
    """Main entry point for failure analysis script."""
    parser = argparse.ArgumentParser(
        description='Comprehensive quinté prediction failure analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all predictions
  python scripts/analyze_quinte_failures.py

  # Analyze specific date
  python scripts/analyze_quinte_failures.py --date 2025-11-03

  # Require minimum 50 races
  python scripts/analyze_quinte_failures.py --min-races 50

  # Custom output directory
  python scripts/analyze_quinte_failures.py --output-dir my_analysis
        """
    )
    parser.add_argument('--date', type=str,
                       help='Race date to analyze (YYYY-MM-DD). If not specified, analyzes all predictions.')
    parser.add_argument('--min-races', type=int, default=30,
                       help='Minimum number of races required for analysis (default: 30)')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Directory for saving analysis results (default: analysis_output)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Run analysis
    analyzer = QuinteFailureAnalyzer(verbose=args.verbose)
    result = analyzer.analyze(
        race_date=args.date,
        min_races=args.min_races,
        output_dir=args.output_dir
    )

    if result['status'] == 'success':
        print(f"\n✓ Analysis complete!")
        print(f"  Races analyzed: {result['n_races']}")
        print(f"  Total predictions: {result['n_predictions']}")
        print(f"  Report: {result['report_file']}")
    elif result['status'] == 'no_data':
        print(f"\n✗ No prediction data found. Run predict_quinte.py first.")
    else:
        print(f"\n✗ Analysis failed: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
