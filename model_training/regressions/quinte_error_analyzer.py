#!/usr/bin/env python3
"""
Quinté Error Analyzer

Analyzes quinté prediction failures and identifies patterns to guide incremental training.

Key Metrics:
- Quinté désordre: All 5 predicted horses in actual top 5 (any order)
- Bonus 4: All actual top 4 in predicted top 5
- Bonus 3: All actual top 3 in predicted top 5
- MAE: Mean absolute error in position predictions

Failure Patterns:
- Missed favorites (low odds horses not predicted)
- Missed longshots (high odds horses that placed)
- Track condition bias (failures on specific surfaces)
- Field size bias (failures in large/small fields)
- Competitive field issues (under/over-weighting)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
import json


class QuinteErrorAnalyzer:
    """
    Analyzes quinté prediction failures and identifies patterns.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the error analyzer.

        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose

        # Failure weights for training (higher = more important)
        self.failure_weights = {
            'quinte_desordre_miss': 10.0,  # Biggest payout impact
            'bonus_4_miss': 5.0,            # Decent payout
            'bonus_3_miss': 3.0,            # Decent payout
            'high_mae': 2.0,                # Continuous optimization
            'success': 1.0                  # Keep for balance
        }

    def analyze_race_prediction(self, race_data: Dict) -> Dict:
        """
        Analyze a single quinté race prediction.

        Args:
            race_data: Dictionary containing:
                - predicted_top5: List of predicted horse numbers [1, 2, 3, 4, 5]
                - actual_results: String like "13-10-7-1-8" (finish order)
                - predictions_df: DataFrame with all predictions and actuals
                - race_metadata: Race context (field_size, track, etc.)

        Returns:
            Analysis dictionary with:
                - failure_type: 'quinte_desordre', 'bonus4', 'bonus3', 'success'
                - quinte_desordre: Boolean (all 5 in top 5)
                - bonus_4: Boolean (all actual top 4 in predicted top 5)
                - bonus_3: Boolean (all actual top 3 in predicted top 5)
                - missed_horses: List of horses that should have been predicted
                - false_positives: List of horses predicted but didn't place
                - mae: Mean absolute error
                - failure_weight: Weight for incremental training
                - pattern_insights: Dictionary of detected patterns
        """
        # Parse actual results
        actual_top5 = self._parse_actual_results(race_data['actual_results'])
        predicted_top5 = set(race_data['predicted_top5'])

        # Calculate metrics
        quinte_desordre = len(predicted_top5 & actual_top5) == 5
        bonus_4 = len(predicted_top5 & set(list(actual_top5)[:4])) == 4
        bonus_3 = len(predicted_top5 & set(list(actual_top5)[:3])) == 3

        # Determine failure type
        if quinte_desordre:
            failure_type = 'success'
            failure_weight = self.failure_weights['success']
        elif bonus_4:
            failure_type = 'bonus_4_miss'
            failure_weight = self.failure_weights['bonus_4_miss']
        elif bonus_3:
            failure_type = 'bonus_3_miss'
            failure_weight = self.failure_weights['bonus_3_miss']
        else:
            failure_type = 'quinte_desordre_miss'
            failure_weight = self.failure_weights['quinte_desordre_miss']

        # Find missed horses and false positives
        missed_horses = list(actual_top5 - predicted_top5)
        false_positives = list(predicted_top5 - actual_top5)

        # Calculate MAE
        mae = self._calculate_mae(race_data.get('predictions_df'))
        if mae > 3.0:  # High error threshold
            failure_weight = max(failure_weight, self.failure_weights['high_mae'])

        # Analyze patterns
        pattern_insights = self._analyze_failure_patterns(
            race_data, missed_horses, false_positives
        )

        return {
            'failure_type': failure_type,
            'quinte_desordre': quinte_desordre,
            'bonus_4': bonus_4,
            'bonus_3': bonus_3,
            'missed_horses': missed_horses,
            'false_positives': false_positives,
            'mae': mae,
            'failure_weight': failure_weight,
            'pattern_insights': pattern_insights,
            'actual_top5': list(actual_top5),
            'predicted_top5': list(predicted_top5)
        }

    def _parse_actual_results(self, actual_results: str) -> Set[int]:
        """
        Parse actual results string to get top 5 horses.

        Args:
            actual_results: String like "13-10-7-1-8-3-4-2-14-16"

        Returns:
            Set of top 5 horse numbers
        """
        if not actual_results or actual_results == 'pending':
            return set()

        try:
            horse_numbers = [int(x) for x in actual_results.strip().split('-')]
            return set(horse_numbers[:5])  # Top 5 finishers
        except Exception as e:
            if self.verbose:
                print(f"Error parsing actual results '{actual_results}': {e}")
            return set()

    def _calculate_mae(self, predictions_df: Optional[pd.DataFrame]) -> float:
        """
        Calculate mean absolute error for position predictions.

        Args:
            predictions_df: DataFrame with 'predicted_position' and 'actual_position'

        Returns:
            MAE value
        """
        if predictions_df is None or predictions_df.empty:
            return 0.0

        if 'predicted_position' not in predictions_df.columns or 'actual_position' not in predictions_df.columns:
            return 0.0

        # Filter out NaN values (horses with no actual position)
        valid_mask = predictions_df['actual_position'].notna()
        if valid_mask.sum() == 0:
            return 0.0

        mae = np.mean(np.abs(
            predictions_df.loc[valid_mask, 'predicted_position'] -
            predictions_df.loc[valid_mask, 'actual_position']
        ))

        return float(mae)

    def _analyze_failure_patterns(self, race_data: Dict,
                                   missed_horses: List[int],
                                   false_positives: List[int]) -> Dict:
        """
        Analyze patterns in prediction failures.

        Args:
            race_data: Full race data
            missed_horses: Horses that should have been predicted
            false_positives: Horses incorrectly predicted

        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            'missed_favorite': False,
            'missed_longshot': False,
            'predicted_favorite': False,
            'over_weighted_favorite': False,
            'track_condition': race_data.get('race_metadata', {}).get('track_condition', 'unknown'),
            'field_size': race_data.get('race_metadata', {}).get('field_size', 0),
            'field_size_category': self._categorize_field_size(
                race_data.get('race_metadata', {}).get('field_size', 0)
            )
        }

        predictions_df = race_data.get('predictions_df')
        if predictions_df is None or predictions_df.empty:
            return patterns

        # Analyze missed horses
        for horse_num in missed_horses:
            horse_data = predictions_df[predictions_df['numero'] == horse_num]
            if not horse_data.empty:
                odds = horse_data['cotedirect'].iloc[0] if 'cotedirect' in horse_data.columns else None
                if odds is not None:
                    if odds < 5.0:
                        patterns['missed_favorite'] = True
                    elif odds > 20.0:
                        patterns['missed_longshot'] = True

        # Analyze false positives
        for horse_num in false_positives:
            horse_data = predictions_df[predictions_df['numero'] == horse_num]
            if not horse_data.empty:
                odds = horse_data['cotedirect'].iloc[0] if 'cotedirect' in horse_data.columns else None
                if odds is not None and odds < 5.0:
                    patterns['predicted_favorite'] = True
                    patterns['over_weighted_favorite'] = True

        return patterns

    def _categorize_field_size(self, field_size: int) -> str:
        """Categorize field size"""
        if field_size < 14:
            return 'small'
        elif field_size <= 16:
            return 'medium'
        else:
            return 'large'

    def identify_failure_patterns(self, failed_races: List[Dict]) -> Dict:
        """
        Identify common patterns across multiple failed quinté predictions.

        Args:
            failed_races: List of race analysis dictionaries from analyze_race_prediction()

        Returns:
            Summary of common failure patterns with counts
        """
        if not failed_races:
            return {}

        pattern_counts = {
            'total_failures': len(failed_races),
            'quinte_desordre_misses': 0,
            'bonus_4_misses': 0,
            'bonus_3_misses': 0,
            'missed_favorites_count': 0,
            'missed_longshots_count': 0,
            'over_weighted_favorites_count': 0,
            'track_condition_distribution': {},
            'field_size_distribution': {},
            'high_mae_count': 0,
            'avg_mae': 0.0
        }

        maes = []

        for race in failed_races:
            # Count failure types
            if race['failure_type'] == 'quinte_desordre_miss':
                pattern_counts['quinte_desordre_misses'] += 1
            elif race['failure_type'] == 'bonus_4_miss':
                pattern_counts['bonus_4_misses'] += 1
            elif race['failure_type'] == 'bonus_3_miss':
                pattern_counts['bonus_3_misses'] += 1

            # Count patterns
            patterns = race.get('pattern_insights', {})
            if patterns.get('missed_favorite'):
                pattern_counts['missed_favorites_count'] += 1
            if patterns.get('missed_longshot'):
                pattern_counts['missed_longshots_count'] += 1
            if patterns.get('over_weighted_favorite'):
                pattern_counts['over_weighted_favorites_count'] += 1

            # Track conditions
            track = patterns.get('track_condition', 'unknown')
            pattern_counts['track_condition_distribution'][track] = \
                pattern_counts['track_condition_distribution'].get(track, 0) + 1

            # Field sizes
            field_cat = patterns.get('field_size_category', 'unknown')
            pattern_counts['field_size_distribution'][field_cat] = \
                pattern_counts['field_size_distribution'].get(field_cat, 0) + 1

            # MAE
            mae = race.get('mae', 0.0)
            maes.append(mae)
            if mae > 3.0:
                pattern_counts['high_mae_count'] += 1

        # Calculate average MAE
        if maes:
            pattern_counts['avg_mae'] = float(np.mean(maes))

        # Calculate percentages
        total = pattern_counts['total_failures']
        pattern_counts['missed_favorites_pct'] = (pattern_counts['missed_favorites_count'] / total * 100) if total > 0 else 0
        pattern_counts['missed_longshots_pct'] = (pattern_counts['missed_longshots_count'] / total * 100) if total > 0 else 0
        pattern_counts['over_weighted_favorites_pct'] = (pattern_counts['over_weighted_favorites_count'] / total * 100) if total > 0 else 0

        return pattern_counts

    def calculate_failure_weights(self, race_analyses: List[Dict]) -> pd.DataFrame:
        """
        Calculate sample weights for incremental training based on failure severity.

        Args:
            race_analyses: List of race analysis dictionaries

        Returns:
            DataFrame with race_id, failure_weight, and failure_type
        """
        weights_data = []

        for analysis in race_analyses:
            weights_data.append({
                'race_id': analysis.get('race_id', ''),
                'failure_weight': analysis['failure_weight'],
                'failure_type': analysis['failure_type'],
                'mae': analysis['mae']
            })

        return pd.DataFrame(weights_data)

    def generate_correction_suggestions(self, pattern_summary: Dict) -> List[str]:
        """
        Generate actionable suggestions based on failure patterns.

        Args:
            pattern_summary: Output from identify_failure_patterns()

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Missed favorites
        if pattern_summary.get('missed_favorites_pct', 0) > 30:
            suggestions.append(
                f"⚠️ Missing favorites in {pattern_summary['missed_favorites_pct']:.1f}% of failures. "
                "Consider: Reducing competitive weight or increasing base model weight."
            )

        # Missed longshots
        if pattern_summary.get('missed_longshots_pct', 0) > 30:
            suggestions.append(
                f"⚠️ Missing longshots in {pattern_summary['missed_longshots_pct']:.1f}% of failures. "
                "Consider: Increasing competitive analysis weight."
            )

        # Over-weighted favorites
        if pattern_summary.get('over_weighted_favorites_pct', 0) > 40:
            suggestions.append(
                f"⚠️ Over-predicting favorites in {pattern_summary['over_weighted_favorites_pct']:.1f}% of failures. "
                "Consider: Adjusting odds-based features or reducing favorite bias."
            )

        # High MAE
        if pattern_summary.get('avg_mae', 0) > 3.5:
            suggestions.append(
                f"⚠️ High average MAE ({pattern_summary['avg_mae']:.2f}). "
                "Consider: Retraining on recent data or feature engineering."
            )

        # Track conditions
        track_dist = pattern_summary.get('track_condition_distribution', {})
        if track_dist:
            max_track = max(track_dist, key=track_dist.get)
            max_count = track_dist[max_track]
            if max_count / pattern_summary.get('total_failures', 1) > 0.5:
                suggestions.append(
                    f"⚠️ {max_count} failures on '{max_track}' track condition ({max_count/pattern_summary['total_failures']*100:.1f}%). "
                    "Consider: Adding more weight to track condition features."
                )

        # Field size
        field_dist = pattern_summary.get('field_size_distribution', {})
        if field_dist:
            max_field = max(field_dist, key=field_dist.get)
            max_count = field_dist[max_field]
            if max_count / pattern_summary.get('total_failures', 1) > 0.5:
                suggestions.append(
                    f"⚠️ {max_count} failures in '{max_field}' field sizes ({max_count/pattern_summary['total_failures']*100:.1f}%). "
                    "Consider: Adjusting field size-specific features."
                )

        if not suggestions:
            suggestions.append("✅ No clear patterns detected. Model performance is relatively balanced.")

        return suggestions


if __name__ == "__main__":
    # Example usage
    analyzer = QuinteErrorAnalyzer(verbose=True)

    # Test with sample data
    sample_race = {
        'predicted_top5': [1, 2, 3, 4, 5],
        'actual_results': "13-10-7-1-8-3-4-2",
        'predictions_df': pd.DataFrame({
            'numero': [1, 2, 3, 4, 5, 13, 10, 7, 8],
            'predicted_position': [1, 2, 3, 4, 5, 10, 9, 8, 7],
            'actual_position': [4, 8, 6, 7, 9, 1, 2, 3, 5],
            'cotedirect': [3.5, 5.0, 8.0, 12.0, 15.0, 25.0, 18.0, 10.0, 20.0]
        }),
        'race_metadata': {
            'field_size': 16,
            'track_condition': 'PH'
        }
    }

    analysis = analyzer.analyze_race_prediction(sample_race)
    print("\n=== Race Analysis ===")
    print(json.dumps(analysis, indent=2, default=str))

    # Test pattern identification
    failed_races = [analysis] * 10  # Simulate 10 similar failures
    patterns = analyzer.identify_failure_patterns(failed_races)
    print("\n=== Failure Patterns ===")
    print(json.dumps(patterns, indent=2))

    # Generate suggestions
    suggestions = analyzer.generate_correction_suggestions(patterns)
    print("\n=== Correction Suggestions ===")
    for suggestion in suggestions:
        print(f"  {suggestion}")
