"""
Prediction Calibration System

Applies calibration corrections to predictions based on detected biases.
Supports multiple bias types and maintains calibration history.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class PredictionCalibrator:
    """
    Apply calibration corrections to predictions based on detected biases
    """

    def __init__(self, calibration_path=None):
        self.calibrations = {}
        self.calibration_history = []

        if calibration_path and Path(calibration_path).exists():
            self.load(calibration_path)

    def fit(self, biases, validation_df=None):
        """
        Create calibration from detected biases

        Args:
            biases: Output from BiasDetector.analyze_biases()
            validation_df: Optional validation set to tune corrections
        """

        print("\n" + "="*80)
        print("BUILDING CALIBRATION")
        print("="*80)

        self.calibrations = {
            'version': datetime.now().isoformat(),
            'biases': biases,
            'corrections': {}
        }

        # Build correction functions
        for bias_type, bias_info in biases.items():
            if not bias_info.get('significant'):
                continue

            if bias_type == 'systematic':
                # Simple offset
                self.calibrations['corrections']['systematic_offset'] = bias_info['correction']
                print(f"\nSystematic correction: {bias_info['correction']:+.3f}")

            elif 'corrections' in bias_info:
                # Category-based corrections
                self.calibrations['corrections'][bias_type] = bias_info['corrections']
                print(f"\n{bias_type.upper()} corrections:")
                for category, correction in bias_info['corrections'].items():
                    if abs(correction) > 0.1:
                        print(f"  {category}: {correction:+.3f}")

        # Validate if validation set provided
        if validation_df is not None:
            print("\n" + "="*80)
            print("VALIDATION")
            print("="*80)

            # Before calibration
            before_mae = np.abs(validation_df['predicted_position'] - validation_df['actual_position']).mean()

            # Apply calibration
            calibrated = self.transform(validation_df)

            # After calibration
            after_mae = np.abs(calibrated['calibrated_prediction'] - validation_df['actual_position']).mean()

            improvement = before_mae - after_mae
            improvement_pct = (improvement / before_mae) * 100

            print(f"\nMAE before calibration: {before_mae:.3f}")
            print(f"MAE after calibration: {after_mae:.3f}")
            print(f"Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")

            self.calibrations['validation'] = {
                'mae_before': before_mae,
                'mae_after': after_mae,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            }

        return self

    def transform(self, predictions_df):
        """
        Apply calibration to predictions

        Args:
            predictions_df: DataFrame with predictions and race characteristics

        Returns:
            DataFrame with 'calibrated_prediction' column added
        """

        df = predictions_df.copy()
        df['calibrated_prediction'] = df['predicted_position']

        if not self.calibrations.get('corrections'):
            return df

        corrections = self.calibrations['corrections']

        # Apply systematic offset
        if 'systematic_offset' in corrections:
            df['calibrated_prediction'] += corrections['systematic_offset']

        # Apply odds correction
        if 'odds' in corrections and 'cotedirect' in df.columns:
            df['odds_bucket'] = pd.cut(
                df['cotedirect'],
                bins=[0, 3, 5, 10, 20, 100],
                labels=['favorite', 'second_choice', 'mid_odds', 'long_shot', 'extreme']
            )

            for bucket, correction in corrections['odds'].items():
                mask = df['odds_bucket'] == bucket
                df.loc[mask, 'calibrated_prediction'] += correction

        # Apply post position correction
        if 'post_position' in corrections and 'numero' in df.columns:
            df['position_bucket'] = pd.cut(
                df['numero'],
                bins=[0, 3, 6, 10, 20],
                labels=['inside', 'mid_inside', 'mid_outside', 'outside']
            )

            for bucket, correction in corrections['post_position'].items():
                mask = df['position_bucket'] == bucket
                df.loc[mask, 'calibrated_prediction'] += correction

        # Apply field size correction
        if 'field_size' in corrections and 'partant' in df.columns:
            df['field_bucket'] = pd.cut(
                df['partant'],
                bins=[0, 10, 14, 18, 30],
                labels=['small', 'medium', 'large', 'xlarge']
            )

            for bucket, correction in corrections['field_size'].items():
                mask = df['field_bucket'] == bucket
                df.loc[mask, 'calibrated_prediction'] += correction

        # Apply distance correction
        if 'distance' in corrections and 'distance' in df.columns:
            df['distance_bucket'] = pd.cut(
                df['distance'],
                bins=[0, 1600, 2000, 2800, 10000],
                labels=['sprint', 'mile', 'middle', 'long']
            )

            for bucket, correction in corrections['distance'].items():
                mask = df['distance_bucket'] == bucket
                df.loc[mask, 'calibrated_prediction'] += correction

        # Apply race type correction
        if 'typec' in corrections and 'typec' in df.columns:
            for typec, correction in corrections['typec'].items():
                mask = df['typec'] == typec
                df.loc[mask, 'calibrated_prediction'] += correction

        # Ensure predictions stay in valid range (1 to partant)
        df['calibrated_prediction'] = df['calibrated_prediction'].clip(lower=1)
        if 'partant' in df.columns:
            df['calibrated_prediction'] = df.apply(
                lambda row: min(row['calibrated_prediction'], row['partant']),
                axis=1
            )

        return df

    def save(self, path):
        """Save calibration to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        save_data = {
            'version': self.calibrations.get('version'),
            'corrections': self.calibrations.get('corrections', {}),
            'validation': self.calibrations.get('validation', {}),
            'history': self.calibration_history
        }

        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\nCalibration saved to: {path}")

    def load(self, path):
        """Load calibration from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        self.calibrations = {
            'version': data.get('version'),
            'corrections': data.get('corrections', {}),
            'validation': data.get('validation', {})
        }
        self.calibration_history = data.get('history', [])

        print(f"\nCalibration loaded from: {path}")
        print(f"  Version: {self.calibrations['version']}")
        print(f"  Corrections: {len(self.calibrations['corrections'])}")
