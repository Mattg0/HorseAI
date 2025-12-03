#!/usr/bin/env python3
"""
Quinté Correction Strategy

Implements specific correction strategies for quinté prediction failures.
Analyzes failure patterns and suggests adjustments to improve model performance.

Strategies:
- Competitive weight adjustment (based on favorite/longshot patterns)
- Feature importance reweighting (based on correlated failures)
- Hard example mining (worst 20% of failures)
- Hyperparameter suggestions (model adjustments)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class QuinteCorrectionStrategy:
    """
    Implements correction strategies for quinté prediction failures.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the correction strategy.

        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose

        # Default competitive weights
        self.default_competitive_weight = 0.3

        # Adjustment ranges
        self.competitive_weight_range = (0.0, 0.6)
        self.max_adjustment = 0.15  # Maximum adjustment per iteration

    def adjust_competitive_weighting(self, failure_patterns: Dict) -> Dict[str, float]:
        """
        Adjust competitive field analysis weight based on failure patterns.

        Strategy:
        - If missing favorites → Increase base model weight (reduce competitive)
        - If missing longshots → Increase competitive weight
        - If over-weighted favorites → Reduce odds-based weighting

        Args:
            failure_patterns: Dictionary from QuinteErrorAnalyzer.identify_failure_patterns()

        Returns:
            Dictionary with:
                - current_weight: Current competitive weight
                - suggested_weight: Suggested new weight
                - adjustment: Amount of adjustment
                - reason: Explanation of adjustment
        """
        current_weight = self.default_competitive_weight
        adjustment = 0.0
        reasons = []

        # Check for missed favorites
        missed_fav_pct = failure_patterns.get('missed_favorites_pct', 0)
        if missed_fav_pct > 30:
            # Too many missed favorites → reduce competitive weight
            reduction = min(self.max_adjustment, missed_fav_pct / 200)  # Scale by severity
            adjustment -= reduction
            reasons.append(f"Reducing competitive weight by {reduction:.3f} (missing {missed_fav_pct:.1f}% favorites)")

        # Check for missed longshots
        missed_long_pct = failure_patterns.get('missed_longshots_pct', 0)
        if missed_long_pct > 30:
            # Too many missed longshots → increase competitive weight
            increase = min(self.max_adjustment, missed_long_pct / 200)
            adjustment += increase
            reasons.append(f"Increasing competitive weight by {increase:.3f} (missing {missed_long_pct:.1f}% longshots)")

        # Check for over-weighted favorites
        over_fav_pct = failure_patterns.get('over_weighted_favorites_pct', 0)
        if over_fav_pct > 40:
            # Too many false positive favorites → reduce competitive weight slightly
            reduction = min(self.max_adjustment * 0.5, over_fav_pct / 400)
            adjustment -= reduction
            reasons.append(f"Reducing competitive weight by {reduction:.3f} (over-predicting {over_fav_pct:.1f}% favorites)")

        # Calculate suggested weight
        suggested_weight = current_weight + adjustment
        suggested_weight = max(self.competitive_weight_range[0],
                               min(self.competitive_weight_range[1], suggested_weight))

        return {
            'current_weight': current_weight,
            'suggested_weight': float(suggested_weight),
            'adjustment': float(adjustment),
            'reason': '; '.join(reasons) if reasons else 'No adjustment needed',
            'within_range': self.competitive_weight_range[0] <= suggested_weight <= self.competitive_weight_range[1]
        }

    def reweight_features(self, failure_analysis: List[Dict]) -> Dict[str, float]:
        """
        Suggest feature importance adjustments based on failure correlations.

        Args:
            failure_analysis: List of race analysis dictionaries

        Returns:
            Dictionary mapping feature categories to suggested weight multipliers
        """
        feature_adjustments = {}

        if not failure_analysis:
            return feature_adjustments

        # Analyze track condition patterns
        track_failures = defaultdict(int)
        track_total = 0
        for analysis in failure_analysis:
            track = analysis.get('pattern_insights', {}).get('track_condition', 'unknown')
            if analysis.get('failure_type') != 'success':
                track_failures[track] += 1
            track_total += 1

        # If specific track has >50% failures, suggest increasing track features
        for track, count in track_failures.items():
            if track != 'unknown' and count / track_total > 0.5:
                feature_adjustments['track_condition_features'] = 1.3  # 30% increase
                if self.verbose:
                    print(f"Suggesting 30% increase in track condition features (high failures on {track})")

        # Analyze field size patterns
        field_failures = defaultdict(int)
        field_total = 0
        for analysis in failure_analysis:
            field_cat = analysis.get('pattern_insights', {}).get('field_size_category', 'unknown')
            if analysis.get('failure_type') != 'success':
                field_failures[field_cat] += 1
            field_total += 1

        # If specific field size has >50% failures, suggest increasing field size features
        for field_cat, count in field_failures.items():
            if field_cat != 'unknown' and count / field_total > 0.5:
                feature_adjustments['field_size_features'] = 1.25  # 25% increase
                if self.verbose:
                    print(f"Suggesting 25% increase in field size features (high failures in {field_cat} fields)")

        # Analyze quinté-specific features
        # If many failures have high MAE, suggest increasing position-prediction features
        high_mae_count = sum(1 for a in failure_analysis if a.get('mae', 0) > 3.5)
        if high_mae_count / len(failure_analysis) > 0.4:
            feature_adjustments['quinte_career_features'] = 1.2  # 20% increase
            if self.verbose:
                print(f"Suggesting 20% increase in quinté career features (high MAE in {high_mae_count} races)")

        return feature_adjustments

    def generate_hard_examples_dataset(self, failure_data: pd.DataFrame,
                                       race_analyses: List[Dict],
                                       top_pct: float = 0.2) -> pd.DataFrame:
        """
        Create focused training dataset from worst failures.

        Selection criteria:
        - Top 20% worst quinté désordre failures
        - Races where predicted and actual have 0 overlap
        - Recent failures (higher weight to last 30 days)

        Args:
            failure_data: DataFrame with all failure data
            race_analyses: List of race analysis dictionaries
            top_pct: Percentage of worst failures to include (default: 20%)

        Returns:
            DataFrame with hard examples and boosted weights
        """
        if failure_data.empty or not race_analyses:
            return pd.DataFrame()

        # Create mapping of race_id to failure metrics
        race_metrics = {}
        for analysis in race_analyses:
            race_id = analysis.get('race_id')
            if race_id:
                race_metrics[race_id] = {
                    'failure_weight': analysis.get('failure_weight', 1.0),
                    'mae': analysis.get('mae', 0.0),
                    'quinte_desordre': analysis.get('quinte_desordre', True),
                    'overlap': len(set(analysis.get('actual_top5', [])) &
                                  set(analysis.get('predicted_top5', [])))
                }

        # Score races by severity (higher = worse)
        def calculate_severity(race_id):
            if race_id not in race_metrics:
                return 0.0

            metrics = race_metrics[race_id]
            severity = metrics['failure_weight']  # Base weight

            # Boost if zero overlap
            if metrics['overlap'] == 0:
                severity *= 2.0

            # Boost if high MAE
            if metrics['mae'] > 3.5:
                severity *= 1.5

            return severity

        # Add severity scores
        failure_data['severity_score'] = failure_data['race_id'].apply(calculate_severity)

        # Sort by severity and take top percentage
        failure_data = failure_data.sort_values('severity_score', ascending=False)

        n_hard_examples = int(len(race_analyses) * top_pct)
        hard_example_races = set(failure_data['race_id'].unique()[:n_hard_examples])

        # Filter to hard examples
        hard_examples = failure_data[failure_data['race_id'].isin(hard_example_races)].copy()

        # Boost weights for hard examples
        hard_examples['failure_weight'] = hard_examples['failure_weight'] * 2.0

        if self.verbose:
            print(f"Selected {len(hard_example_races)} hard example races")
            print(f"  Total samples: {len(hard_examples)}")
            print(f"  Average severity: {hard_examples['severity_score'].mean():.2f}")

        return hard_examples

    def suggest_model_adjustments(self, error_analysis: Dict,
                                  pattern_summary: Dict) -> List[Dict[str, str]]:
        """
        Generate actionable suggestions for model improvement.

        Args:
            error_analysis: Individual race error analysis
            pattern_summary: Summary from identify_failure_patterns()

        Returns:
            List of suggestion dictionaries with:
                - category: Type of adjustment
                - priority: high/medium/low
                - suggestion: Detailed recommendation
                - expected_impact: Estimated improvement
        """
        suggestions = []

        # Competitive weighting adjustment
        comp_adjustment = self.adjust_competitive_weighting(pattern_summary)
        if abs(comp_adjustment['adjustment']) > 0.05:
            suggestions.append({
                'category': 'competitive_weight',
                'priority': 'high',
                'suggestion': f"Adjust competitive weight from {comp_adjustment['current_weight']:.2f} to {comp_adjustment['suggested_weight']:.2f}",
                'reason': comp_adjustment['reason'],
                'expected_impact': f"{abs(comp_adjustment['adjustment'])*20:.1f}% improvement in quinté désordre rate"
            })

        # Feature reweighting
        feature_weights = self.reweight_features([error_analysis] if isinstance(error_analysis, dict) else error_analysis)
        if feature_weights:
            for feature_cat, multiplier in feature_weights.items():
                suggestions.append({
                    'category': 'feature_importance',
                    'priority': 'medium',
                    'suggestion': f"Increase {feature_cat} importance by {(multiplier-1)*100:.0f}%",
                    'reason': f"High failure correlation with {feature_cat}",
                    'expected_impact': f"{(multiplier-1)*10:.1f}% improvement"
                })

        # High MAE suggestions
        avg_mae = pattern_summary.get('avg_mae', 0)
        if avg_mae > 3.5:
            suggestions.append({
                'category': 'model_architecture',
                'priority': 'high',
                'suggestion': "Retrain model with focus on position accuracy",
                'reason': f"Average MAE is high ({avg_mae:.2f})",
                'expected_impact': "15-20% MAE reduction"
            })

        # Track condition specific training
        track_dist = pattern_summary.get('track_condition_distribution', {})
        if track_dist:
            max_track = max(track_dist, key=track_dist.get)
            max_pct = track_dist[max_track] / pattern_summary.get('total_failures', 1) * 100
            if max_pct > 50:
                suggestions.append({
                    'category': 'data_augmentation',
                    'priority': 'medium',
                    'suggestion': f"Add more training data for '{max_track}' track condition",
                    'reason': f"{max_pct:.0f}% of failures on this surface",
                    'expected_impact': "10-15% improvement on this surface"
                })

        # Field size specific training
        field_dist = pattern_summary.get('field_size_distribution', {})
        if field_dist:
            max_field = max(field_dist, key=field_dist.get)
            max_pct = field_dist[max_field] / pattern_summary.get('total_failures', 1) * 100
            if max_pct > 50:
                suggestions.append({
                    'category': 'data_augmentation',
                    'priority': 'medium',
                    'suggestion': f"Add more training data for '{max_field}' field sizes",
                    'reason': f"{max_pct:.0f}% of failures in this category",
                    'expected_impact': "10-15% improvement in this category"
                })

        # TabNet hyperparameter suggestions
        quinte_miss_rate = pattern_summary.get('quinte_desordre_misses', 0) / pattern_summary.get('total_failures', 1)
        if quinte_miss_rate > 0.5:
            suggestions.append({
                'category': 'hyperparameters',
                'priority': 'medium',
                'suggestion': "Increase TabNet attention dimension (n_d, n_a) from 64 to 128",
                'reason': "High quinté désordre miss rate suggests need for better feature interactions",
                'expected_impact': "5-10% improvement in désordre rate"
            })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order.get(x['priority'], 3))

        return suggestions

    def apply_corrections_to_config(self, suggestions: List[Dict[str, str]],
                                   current_config: Dict) -> Dict:
        """
        Apply correction suggestions to model configuration.

        Args:
            suggestions: List from suggest_model_adjustments()
            current_config: Current model/training configuration

        Returns:
            Updated configuration dictionary
        """
        new_config = current_config.copy()

        for suggestion in suggestions:
            category = suggestion['category']

            if category == 'competitive_weight':
                # Extract suggested weight from suggestion text
                if 'suggested_weight' in suggestion:
                    new_config['competitive_weight'] = suggestion['suggested_weight']
                elif 'to' in suggestion['suggestion']:
                    # Parse from text: "... to 0.35"
                    parts = suggestion['suggestion'].split('to')
                    if len(parts) > 1:
                        try:
                            new_weight = float(parts[1].strip())
                            new_config['competitive_weight'] = new_weight
                        except ValueError:
                            pass

            elif category == 'hyperparameters':
                # TabNet dimension adjustments
                if 'attention dimension' in suggestion['suggestion']:
                    if 'tabnet_params' not in new_config:
                        new_config['tabnet_params'] = {}
                    new_config['tabnet_params']['n_d'] = 128
                    new_config['tabnet_params']['n_a'] = 128

            elif category == 'feature_importance':
                # Feature reweighting
                if 'feature_weights' not in new_config:
                    new_config['feature_weights'] = {}

                # Extract feature category and multiplier
                if 'track_condition' in suggestion['suggestion']:
                    new_config['feature_weights']['track_condition'] = 1.3
                elif 'field_size' in suggestion['suggestion']:
                    new_config['feature_weights']['field_size'] = 1.25
                elif 'career' in suggestion['suggestion']:
                    new_config['feature_weights']['quinte_career'] = 1.2

        return new_config

    def generate_training_report(self, baseline_metrics: Dict,
                                improved_metrics: Dict,
                                suggestions_applied: List[Dict]) -> str:
        """
        Generate a comprehensive training report.

        Args:
            baseline_metrics: Metrics before training
            improved_metrics: Metrics after training
            suggestions_applied: List of suggestions that were applied

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("QUINTÉ INCREMENTAL TRAINING REPORT")
        report.append("=" * 60)
        report.append("")

        # Baseline vs Improved
        report.append("PERFORMANCE METRICS")
        report.append("-" * 60)

        metrics = [
            ('Quinté Désordre Rate', 'quinte_desordre_rate', '%'),
            ('Bonus 4 Rate', 'bonus_4_rate', '%'),
            ('Bonus 3 Rate', 'bonus_3_rate', '%'),
            ('Average MAE', 'avg_mae', '')
        ]

        for metric_name, metric_key, unit in metrics:
            baseline = baseline_metrics.get(metric_key, 0)
            improved = improved_metrics.get(metric_key, 0)

            if unit == '%':
                baseline_str = f"{baseline*100:.1f}%"
                improved_str = f"{improved*100:.1f}%"
                change = (improved - baseline) * 100
                change_str = f"{'+' if change >= 0 else ''}{change:.1f}%"
            else:
                baseline_str = f"{baseline:.3f}"
                improved_str = f"{improved:.3f}"
                change = improved - baseline
                change_str = f"{'+' if change >= 0 else ''}{change:.3f}"

            report.append(f"{metric_name:25} {baseline_str:>10} → {improved_str:>10} ({change_str})")

        report.append("")

        # Suggestions Applied
        if suggestions_applied:
            report.append("CORRECTIONS APPLIED")
            report.append("-" * 60)
            for i, suggestion in enumerate(suggestions_applied, 1):
                report.append(f"{i}. [{suggestion['priority'].upper()}] {suggestion['category']}")
                report.append(f"   {suggestion['suggestion']}")
                report.append(f"   Expected: {suggestion.get('expected_impact', 'N/A')}")
                report.append("")

        report.append("=" * 60)

        return '\n'.join(report)


if __name__ == "__main__":
    # Example usage
    strategy = QuinteCorrectionStrategy(verbose=True)

    # Sample failure patterns
    patterns = {
        'total_failures': 20,
        'quinte_desordre_misses': 15,
        'missed_favorites_pct': 45.0,
        'missed_longshots_pct': 25.0,
        'over_weighted_favorites_pct': 35.0,
        'avg_mae': 3.8,
        'track_condition_distribution': {'PH': 12, 'PS': 5, 'DUR': 3},
        'field_size_distribution': {'large': 13, 'medium': 5, 'small': 2}
    }

    # Test competitive weight adjustment
    print("\n=== Competitive Weight Adjustment ===")
    comp_adj = strategy.adjust_competitive_weighting(patterns)
    print(f"Current: {comp_adj['current_weight']:.3f}")
    print(f"Suggested: {comp_adj['suggested_weight']:.3f}")
    print(f"Adjustment: {comp_adj['adjustment']:.3f}")
    print(f"Reason: {comp_adj['reason']}")

    # Test model adjustment suggestions
    print("\n=== Model Adjustment Suggestions ===")
    suggestions = strategy.suggest_model_adjustments({}, patterns)
    for i, sug in enumerate(suggestions, 1):
        print(f"\n{i}. [{sug['priority'].upper()}] {sug['category']}")
        print(f"   {sug['suggestion']}")
        print(f"   Reason: {sug['reason']}")
        print(f"   Expected Impact: {sug.get('expected_impact', 'N/A')}")

    # Test report generation
    print("\n=== Training Report ===")
    baseline = {
        'quinte_desordre_rate': 0.14,
        'bonus_4_rate': 0.28,
        'bonus_3_rate': 0.35,
        'avg_mae': 3.8
    }
    improved = {
        'quinte_desordre_rate': 0.19,
        'bonus_4_rate': 0.32,
        'bonus_3_rate': 0.38,
        'avg_mae': 3.5
    }

    report = strategy.generate_training_report(baseline, improved, suggestions[:3])
    print(report)
