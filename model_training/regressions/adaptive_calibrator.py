"""
Adaptive Calibrator - Learn from daily race predictions

Provides utilities for:
- Loading/saving calibrators
- Updating calibrators with new race results
- Evaluating calibrator performance
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


class AdaptiveCalibratorManager:
    """
    Manages adaptive calibrators that learn from daily race predictions.
    """

    def __init__(self, calibrator_dir: str = "models/calibrators"):
        """
        Initialize the calibrator manager.

        Args:
            calibrator_dir: Directory to store calibrators
        """
        self.calibrator_dir = Path(calibrator_dir)
        self.calibrator_dir.mkdir(parents=True, exist_ok=True)

    def load_calibrator(self, model_type: str) -> Optional[IsotonicRegression]:
        """
        Load the most recent calibrator for a model type.

        Args:
            model_type: 'rf' or 'tabnet'

        Returns:
            IsotonicRegression calibrator or None if not found
        """
        calibrator_path = self.calibrator_dir / f"{model_type}_calibrator.joblib"

        if not calibrator_path.exists():
            return None

        try:
            calibrator = joblib.load(calibrator_path)
            metadata = self.load_metadata(model_type)

            # Check if calibrator is too old (>90 days)
            if metadata:
                last_update = datetime.fromisoformat(metadata['last_updated'])
                days_old = (datetime.now() - last_update).days

                if days_old > 90:
                    print(f"⚠️  {model_type} calibrator is {days_old} days old (>90), not using")
                    return None

                # Check if enough data
                if metadata['data_points'] < 100:
                    print(f"⚠️  {model_type} calibrator has only {metadata['data_points']} data points (<100), not using")
                    return None

            return calibrator

        except Exception as e:
            print(f"⚠️  Could not load {model_type} calibrator: {e}")
            return None

    def save_calibrator(self, calibrator: IsotonicRegression, model_type: str,
                       metadata: Dict[str, Any]) -> None:
        """
        Save a calibrator and its metadata.

        Args:
            calibrator: Fitted IsotonicRegression
            model_type: 'rf' or 'tabnet'
            metadata: Performance metrics and info
        """
        # Save calibrator
        calibrator_path = self.calibrator_dir / f"{model_type}_calibrator.joblib"
        joblib.dump(calibrator, calibrator_path)

        # Save metadata
        metadata['last_updated'] = datetime.now().isoformat()
        metadata['model_type'] = model_type

        metadata_path = self.calibrator_dir / f"{model_type}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved {model_type} calibrator to {calibrator_path}")

    def load_metadata(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Load calibrator metadata.

        Args:
            model_type: 'rf' or 'tabnet'

        Returns:
            Metadata dict or None
        """
        metadata_path = self.calibrator_dir / f"{model_type}_metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load {model_type} metadata: {e}")
            return None

    def apply_calibration(self, predictions: np.ndarray,
                         calibrator: Optional[IsotonicRegression]) -> np.ndarray:
        """
        Apply calibration if calibrator exists, otherwise return raw predictions.

        Args:
            predictions: Raw predictions
            calibrator: IsotonicRegression or None

        Returns:
            Calibrated predictions (or raw if no calibrator)
        """
        if calibrator is None:
            return predictions

        try:
            return calibrator.predict(predictions)
        except Exception as e:
            print(f"⚠️  Calibration failed: {e}, returning raw predictions")
            return predictions

    def update_calibrator(self, raw_predictions: np.ndarray, actual_results: np.ndarray,
                         model_type: str) -> Tuple[IsotonicRegression, Dict[str, Any]]:
        """
        Create or update a calibrator with new data.

        Args:
            raw_predictions: Raw model predictions
            actual_results: Actual race positions
            model_type: 'rf' or 'tabnet'

        Returns:
            Tuple of (calibrator, metadata)
        """
        # Remove any NaN values
        valid_mask = ~(np.isnan(raw_predictions) | np.isnan(actual_results))
        raw_predictions = raw_predictions[valid_mask]
        actual_results = actual_results[valid_mask]

        if len(raw_predictions) < 50:
            raise ValueError(f"Not enough data to train calibrator: {len(raw_predictions)} points (need ≥50)")

        # Create isotonic regression calibrator
        calibrator = IsotonicRegression(
            out_of_bounds='clip',
            y_min=1.0,
            y_max=20.0  # Reasonable max position
        )

        calibrator.fit(raw_predictions, actual_results)

        # Evaluate calibrator performance
        calibrated_predictions = calibrator.predict(raw_predictions)

        raw_mae = mean_absolute_error(actual_results, raw_predictions)
        calibrated_mae = mean_absolute_error(actual_results, calibrated_predictions)
        raw_rmse = np.sqrt(mean_squared_error(actual_results, raw_predictions))
        calibrated_rmse = np.sqrt(mean_squared_error(actual_results, calibrated_predictions))

        mae_improvement = ((raw_mae - calibrated_mae) / raw_mae * 100) if raw_mae > 0 else 0
        rmse_improvement = ((raw_rmse - calibrated_rmse) / raw_rmse * 100) if raw_rmse > 0 else 0

        metadata = {
            'data_points': len(raw_predictions),
            'raw_mae': float(raw_mae),
            'calibrated_mae': float(calibrated_mae),
            'mae_improvement_pct': float(mae_improvement),
            'raw_rmse': float(raw_rmse),
            'calibrated_rmse': float(calibrated_rmse),
            'rmse_improvement_pct': float(rmse_improvement),
            'prediction_range': {
                'min': float(raw_predictions.min()),
                'max': float(raw_predictions.max()),
                'mean': float(raw_predictions.mean())
            }
        }

        return calibrator, metadata

    def evaluate_calibrator(self, calibrator: IsotonicRegression,
                           test_predictions: np.ndarray,
                           test_actuals: np.ndarray) -> Dict[str, float]:
        """
        Evaluate calibrator on test data.

        Args:
            calibrator: Fitted calibrator
            test_predictions: Raw predictions
            test_actuals: Actual results

        Returns:
            Dict with evaluation metrics
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(test_predictions) | np.isnan(test_actuals))
        test_predictions = test_predictions[valid_mask]
        test_actuals = test_actuals[valid_mask]

        if len(test_predictions) == 0:
            return {
                'raw_mae': 0.0,
                'calibrated_mae': 0.0,
                'improvement_pct': 0.0,
                'n_samples': 0
            }

        # Get calibrated predictions
        calibrated = calibrator.predict(test_predictions)

        # Calculate metrics
        raw_mae = mean_absolute_error(test_actuals, test_predictions)
        calibrated_mae = mean_absolute_error(test_actuals, calibrated)
        improvement = ((raw_mae - calibrated_mae) / raw_mae * 100) if raw_mae > 0 else 0

        return {
            'raw_mae': float(raw_mae),
            'calibrated_mae': float(calibrated_mae),
            'improvement_pct': float(improvement),
            'n_samples': len(test_predictions)
        }

    def get_calibrator_status(self) -> Dict[str, Any]:
        """
        Get status of all calibrators.

        Returns:
            Dict with status for each model type
        """
        status = {}

        for model_type in ['rf', 'tabnet']:
            metadata = self.load_metadata(model_type)
            calibrator_path = self.calibrator_dir / f"{model_type}_calibrator.joblib"

            if metadata and calibrator_path.exists():
                last_update = datetime.fromisoformat(metadata['last_updated'])
                days_old = (datetime.now() - last_update).days

                status[model_type] = {
                    'exists': True,
                    'last_updated': metadata['last_updated'],
                    'days_old': days_old,
                    'data_points': metadata['data_points'],
                    'mae_improvement': metadata['mae_improvement_pct'],
                    'is_valid': days_old <= 90 and metadata['data_points'] >= 100
                }
            else:
                status[model_type] = {
                    'exists': False,
                    'is_valid': False
                }

        return status


def fetch_predictions_with_results(db_path: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch predictions that have actual results from the database.

    Args:
        db_path: Path to SQLite database
        days: Number of days to look back

    Returns:
        DataFrame with predictions and actual results
    """
    import sqlite3

    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    query = """
    SELECT
        race_id,
        horse_id,
        prediction_date,
        raw_rf_prediction,
        raw_tabnet_prediction,
        rf_prediction,
        tabnet_prediction,
        final_prediction,
        actual_result
    FROM race_predictions
    WHERE actual_result IS NOT NULL
      AND prediction_date >= ?
      AND raw_rf_prediction IS NOT NULL
      AND raw_tabnet_prediction IS NOT NULL
    ORDER BY prediction_date DESC
    """

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, params=(cutoff_date,))
    conn.close()

    return df


# Convenience functions for quick usage
def load_calibrator(model_type: str, calibrator_dir: str = "models/calibrators") -> Optional[IsotonicRegression]:
    """Quick function to load a calibrator."""
    manager = AdaptiveCalibratorManager(calibrator_dir)
    return manager.load_calibrator(model_type)


def apply_calibration(predictions: np.ndarray, model_type: str,
                     calibrator_dir: str = "models/calibrators") -> np.ndarray:
    """Quick function to apply calibration."""
    manager = AdaptiveCalibratorManager(calibrator_dir)
    calibrator = manager.load_calibrator(model_type)
    return manager.apply_calibration(predictions, calibrator)
