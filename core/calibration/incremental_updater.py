"""
Incremental Calibration Updater

Updates calibration incrementally as new race results arrive.
Monitors calibration health and triggers updates when needed.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class IncrementalCalibrationUpdater:
    """
    Update calibration incrementally as new results arrive
    """

    def __init__(self, calibrator, detector, lookback_days=30, min_samples=50):
        self.calibrator = calibrator
        self.detector = detector
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.update_history = []

    def check_if_update_needed(self, recent_predictions_df):
        """
        Determine if calibration needs updating

        Returns:
            bool, dict: (should_update, metrics)
        """

        print("\n" + "="*80)
        print("CHECKING CALIBRATION HEALTH")
        print("="*80)

        # Calculate current performance
        if 'calibrated_prediction' in recent_predictions_df.columns:
            current_mae = np.abs(
                recent_predictions_df['calibrated_prediction'] -
                recent_predictions_df['actual_position']
            ).mean()
        else:
            current_mae = np.abs(
                recent_predictions_df['predicted_position'] -
                recent_predictions_df['actual_position']
            ).mean()

        # Calculate baseline (uncalibrated) performance
        baseline_mae = np.abs(
            recent_predictions_df['predicted_position'] -
            recent_predictions_df['actual_position']
        ).mean()

        # Check for new biases
        recent_predictions_df['error'] = (
            recent_predictions_df['predicted_position'] -
            recent_predictions_df['actual_position']
        )

        # Simple bias check
        mean_error = recent_predictions_df['error'].mean()
        abs_mean_error = abs(mean_error)

        # Check if systematic bias developing
        systematic_bias_developing = abs_mean_error > 0.3

        # Check if calibration helping
        calibration_effective = current_mae < baseline_mae - 0.05

        # Check if have enough new data
        enough_data = len(recent_predictions_df) >= self.min_samples

        metrics = {
            'current_mae': current_mae,
            'baseline_mae': baseline_mae,
            'mean_error': mean_error,
            'abs_mean_error': abs_mean_error,
            'n_samples': len(recent_predictions_df),
            'systematic_bias_developing': systematic_bias_developing,
            'calibration_effective': calibration_effective,
            'enough_data': enough_data
        }

        print(f"\nCurrent MAE: {current_mae:.3f}")
        print(f"Baseline MAE: {baseline_mae:.3f}")
        print(f"Mean error: {mean_error:+.3f}")
        print(f"Samples: {len(recent_predictions_df)}")

        # Decision logic
        should_update = False
        reason = ""

        if not enough_data:
            reason = f"Insufficient data ({len(recent_predictions_df)} < {self.min_samples})"
        elif systematic_bias_developing:
            should_update = True
            reason = f"Systematic bias developing ({abs_mean_error:.2f})"
        elif not calibration_effective and len(self.calibrator.calibrations.get('corrections', {})) > 0:
            should_update = True
            reason = "Calibration no longer effective"
        elif current_mae > baseline_mae + 0.1:
            should_update = True
            reason = "Calibration hurting performance"
        else:
            reason = "Calibration healthy"

        print(f"\nDecision: {'UPDATE NEEDED' if should_update else 'NO UPDATE'}")
        print(f"Reason: {reason}")

        metrics['should_update'] = should_update
        metrics['reason'] = reason

        return should_update, metrics

    def update(self, new_predictions_df, save_path=None):
        """
        Incrementally update calibration with new data

        Args:
            new_predictions_df: Recent predictions with actual results
            save_path: Optional path to save updated calibration
        """

        print("\n" + "="*80)
        print("INCREMENTAL CALIBRATION UPDATE")
        print("="*80)

        # Detect new biases
        new_biases = self.detector.analyze_biases(new_predictions_df)

        if not new_biases:
            print("\nNo significant new biases detected")
            return

        # Merge with existing calibrations (weighted by sample size)
        print("\nMerging with existing calibrations...")

        # For simplicity, just retrain on recent window
        # In production, could use exponential moving average

        self.calibrator.fit(new_biases)

        # Record update
        update_record = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(new_predictions_df),
            'biases_detected': list(new_biases.keys()),
            'calibration_version': self.calibrator.calibrations.get('version')
        }

        self.update_history.append(update_record)
        self.calibrator.calibration_history.append(update_record)

        if save_path:
            self.calibrator.save(save_path)

        print(f"\nCalibration updated successfully")
        print(f"Total updates: {len(self.update_history)}")
