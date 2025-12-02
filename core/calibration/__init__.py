"""
Incremental Calibration System for Horse Racing Predictions

Detects and corrects systematic biases in model predictions without retraining.
"""

from .bias_detector import BiasDetector
from .prediction_calibrator import PredictionCalibrator
from .incremental_updater import IncrementalCalibrationUpdater

__all__ = [
    'BiasDetector',
    'PredictionCalibrator',
    'IncrementalCalibrationUpdater'
]

__version__ = '1.0.0'
