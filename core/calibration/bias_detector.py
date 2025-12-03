"""
Bias Detection System for Horse Racing Predictions

Detects systematic biases in model predictions across multiple dimensions:
- Odds-based bias (favorites vs long-shots)
- Post position bias
- Field size bias
- Distance bias
- Race type bias
- Systematic errors
"""

import pandas as pd
import numpy as np
from scipy import stats


class BiasDetector:
    """
    Detect systematic biases in model predictions
    """

    def __init__(self):
        self.bias_patterns = {}

    def analyze_biases(self, predictions_df):
        """
        Comprehensive bias analysis

        Args:
            predictions_df: DataFrame with columns:
                - predicted_position, actual_position
                - cotedirect (odds)
                - numero (post position)
                - distance, typec, partant
                - race_id, horse_id

        Returns:
            dict: Detected biases with severity and correction factors
        """

        print("\n" + "="*80)
        print("BIAS DETECTION ANALYSIS")
        print("="*80)

        # Make a copy to avoid modifying the original
        predictions_df = predictions_df.copy()
        predictions_df['error'] = predictions_df['predicted_position'] - predictions_df['actual_position']

        biases = {}

        # 1. ODDS-BASED BIAS (most common)
        print("\n1. Odds-based bias analysis...")
        odds_bias = self._analyze_odds_bias(predictions_df)
        if odds_bias['significant']:
            biases['odds'] = odds_bias

        # 2. POST POSITION BIAS
        print("\n2. Post position bias analysis...")
        position_bias = self._analyze_position_bias(predictions_df)
        if position_bias['significant']:
            biases['post_position'] = position_bias

        # 3. FIELD SIZE BIAS
        print("\n3. Field size bias analysis...")
        field_bias = self._analyze_field_size_bias(predictions_df)
        if field_bias['significant']:
            biases['field_size'] = field_bias

        # 4. DISTANCE BIAS
        print("\n4. Distance bias analysis...")
        distance_bias = self._analyze_distance_bias(predictions_df)
        if distance_bias['significant']:
            biases['distance'] = distance_bias

        # 5. RACE TYPE BIAS
        print("\n5. Race type bias analysis...")
        typec_bias = self._analyze_typec_bias(predictions_df)
        if typec_bias['significant']:
            biases['typec'] = typec_bias

        # 6. SYSTEMATIC ERROR PATTERNS
        print("\n6. Systematic error patterns...")
        systematic = self._analyze_systematic_errors(predictions_df)
        if systematic['significant']:
            biases['systematic'] = systematic

        self.bias_patterns = biases

        print("\n" + "="*80)
        print(f"DETECTED {len(biases)} SIGNIFICANT BIASES")
        print("="*80)

        for bias_type, details in biases.items():
            print(f"\n{bias_type.upper()}:")
            print(f"  Severity: {details['severity']}")
            print(f"  Impact: {details['impact']:.3f} MAE")
            print(f"  Description: {details['description']}")

        return biases

    def _analyze_odds_bias(self, df):
        """
        Check if model systematically over/under predicts based on odds
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Create odds buckets
        df['odds_bucket'] = pd.cut(
            df['cotedirect'],
            bins=[0, 3, 5, 10, 20, 100],
            labels=['favorite', 'second_choice', 'mid_odds', 'long_shot', 'extreme']
        )

        results = []
        for bucket, group in df.groupby('odds_bucket', observed=True):
            if len(group) < 20:
                continue

            mean_error = group['error'].mean()
            std_error = group['error'].std()
            n = len(group)

            # T-test: is mean error significantly different from 0?
            t_stat, p_value = stats.ttest_1samp(group['error'], 0)

            results.append({
                'bucket': bucket,
                'n': n,
                'mean_error': mean_error,
                'std_error': std_error,
                'p_value': p_value,
                'significant': p_value < 0.05 and abs(mean_error) > 0.3
            })

        results_df = pd.DataFrame(results)

        # Check if pattern exists
        significant_buckets = results_df[results_df['significant']]

        if len(significant_buckets) > 0:
            # Calculate correction factors
            corrections = {}
            for _, row in results_df.iterrows():
                corrections[row['bucket']] = -row['mean_error']  # Correction is negative of error

            print(f"  Found bias in {len(significant_buckets)} odds buckets:")
            for _, row in significant_buckets.iterrows():
                direction = "over-predicting" if row['mean_error'] > 0 else "under-predicting"
                print(f"    {row['bucket']}: {direction} by {abs(row['mean_error']):.2f} positions")

            return {
                'significant': True,
                'severity': 'HIGH' if len(significant_buckets) > 2 else 'MEDIUM',
                'impact': significant_buckets['mean_error'].abs().mean(),
                'description': f"Model systematically biased across {len(significant_buckets)} odds ranges",
                'corrections': corrections,
                'details': results_df.to_dict('records')
            }

        return {'significant': False}

    def _analyze_position_bias(self, df):
        """
        Check for post position bias
        """
        if 'numero' not in df.columns:
            return {'significant': False}

        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Group by post position
        df['position_bucket'] = pd.cut(
            df['numero'],
            bins=[0, 3, 6, 10, 20],
            labels=['inside', 'mid_inside', 'mid_outside', 'outside']
        )

        results = []
        for bucket, group in df.groupby('position_bucket', observed=True):
            if len(group) < 30:
                continue

            mean_error = group['error'].mean()
            t_stat, p_value = stats.ttest_1samp(group['error'], 0)

            results.append({
                'bucket': bucket,
                'n': len(group),
                'mean_error': mean_error,
                'p_value': p_value,
                'significant': p_value < 0.05 and abs(mean_error) > 0.3
            })

        results_df = pd.DataFrame(results)
        significant_buckets = results_df[results_df['significant']]

        if len(significant_buckets) > 0:
            corrections = {}
            for _, row in results_df.iterrows():
                corrections[row['bucket']] = -row['mean_error']

            print(f"  Found bias in {len(significant_buckets)} post positions:")
            for _, row in significant_buckets.iterrows():
                print(f"    {row['bucket']}: error {row['mean_error']:+.2f}")

            return {
                'significant': True,
                'severity': 'MEDIUM',
                'impact': significant_buckets['mean_error'].abs().mean(),
                'description': f"Post position bias detected",
                'corrections': corrections,
                'details': results_df.to_dict('records')
            }

        return {'significant': False}

    def _analyze_field_size_bias(self, df):
        """
        Check if errors correlate with field size
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        df['field_bucket'] = pd.cut(
            df['partant'],
            bins=[0, 10, 14, 18, 30],
            labels=['small', 'medium', 'large', 'xlarge']
        )

        results = []
        for bucket, group in df.groupby('field_bucket', observed=True):
            if len(group) < 30:
                continue

            mean_error = group['error'].mean()
            t_stat, p_value = stats.ttest_1samp(group['error'], 0)

            results.append({
                'bucket': bucket,
                'n': len(group),
                'mean_error': mean_error,
                'p_value': p_value,
                'significant': p_value < 0.05 and abs(mean_error) > 0.3
            })

        results_df = pd.DataFrame(results)
        significant_buckets = results_df[results_df['significant']]

        if len(significant_buckets) > 0:
            corrections = {}
            for _, row in results_df.iterrows():
                corrections[row['bucket']] = -row['mean_error']

            print(f"  Found bias in {len(significant_buckets)} field sizes:")
            for _, row in significant_buckets.iterrows():
                print(f"    {row['bucket']}: error {row['mean_error']:+.2f}")

            return {
                'significant': True,
                'severity': 'MEDIUM',
                'impact': significant_buckets['mean_error'].abs().mean(),
                'description': f"Field size bias detected",
                'corrections': corrections,
                'details': results_df.to_dict('records')
            }

        return {'significant': False}

    def _analyze_distance_bias(self, df):
        """
        Check for distance-related bias
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        df['distance_bucket'] = pd.cut(
            df['distance'],
            bins=[0, 1600, 2000, 2800, 10000],
            labels=['sprint', 'mile', 'middle', 'long']
        )

        results = []
        for bucket, group in df.groupby('distance_bucket', observed=True):
            if len(group) < 30:
                continue

            mean_error = group['error'].mean()
            t_stat, p_value = stats.ttest_1samp(group['error'], 0)

            results.append({
                'bucket': bucket,
                'n': len(group),
                'mean_error': mean_error,
                'p_value': p_value,
                'significant': p_value < 0.05 and abs(mean_error) > 0.3
            })

        results_df = pd.DataFrame(results)
        significant_buckets = results_df[results_df['significant']]

        if len(significant_buckets) > 0:
            corrections = {}
            for _, row in results_df.iterrows():
                corrections[row['bucket']] = -row['mean_error']

            return {
                'significant': True,
                'severity': 'LOW',
                'impact': significant_buckets['mean_error'].abs().mean(),
                'description': f"Distance bias detected",
                'corrections': corrections,
                'details': results_df.to_dict('records')
            }

        return {'significant': False}

    def _analyze_typec_bias(self, df):
        """
        Check for race type bias
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        results = []
        for typec, group in df.groupby('typec', observed=True):
            if len(group) < 30:
                continue

            mean_error = group['error'].mean()
            t_stat, p_value = stats.ttest_1samp(group['error'], 0)

            results.append({
                'bucket': typec,
                'n': len(group),
                'mean_error': mean_error,
                'p_value': p_value,
                'significant': p_value < 0.05 and abs(mean_error) > 0.4
            })

        results_df = pd.DataFrame(results)
        significant_buckets = results_df[results_df['significant']]

        if len(significant_buckets) > 0:
            corrections = {}
            for _, row in results_df.iterrows():
                corrections[row['bucket']] = -row['mean_error']

            return {
                'significant': True,
                'severity': 'MEDIUM',
                'impact': significant_buckets['mean_error'].abs().mean(),
                'description': f"Race type bias detected",
                'corrections': corrections,
                'details': results_df.to_dict('records')
            }

        return {'significant': False}

    def _analyze_systematic_errors(self, df):
        """
        Check for overall systematic errors (over/under prediction)
        """
        overall_error = df['error'].mean()
        t_stat, p_value = stats.ttest_1samp(df['error'], 0)

        if p_value < 0.05 and abs(overall_error) > 0.2:
            direction = "over-predicting" if overall_error > 0 else "under-predicting"

            print(f"  Model systematically {direction} by {abs(overall_error):.2f} positions")

            return {
                'significant': True,
                'severity': 'HIGH' if abs(overall_error) > 0.5 else 'MEDIUM',
                'impact': abs(overall_error),
                'description': f"Systematic {direction} by {abs(overall_error):.2f}",
                'correction': -overall_error,
                'p_value': p_value
            }

        return {'significant': False}
