"""
Comprehensive utilities for regression models, including:
- Model calibration
- Performance evaluation
- Result visualization
- Data processing utilities

This module provides a complete suite of tools for training, evaluating,
and improving regression models for race prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, List
import time
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# Import scikit-learn components
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


#------------------------------------------------------------------------------
# DATA PROCESSING UTILITIES
#------------------------------------------------------------------------------

def convert_race_results_to_numeric(results_array, drop_empty=True):
    """
    Convert race result values to numeric, preserving meaning of non-numeric codes.

    Args:
        results_array: Array or Series of race results (mix of numeric positions and status codes)
        drop_empty: Whether to drop empty strings or convert them to a numeric value

    Returns:
        if drop_empty=True: tuple of (numeric_array, valid_mask)
        if drop_empty=False: numeric array with all values converted
    """
    # First convert to pandas Series for easier handling
    results = pd.Series(results_array)

    # Create a mask for empty strings
    empty_mask = results.astype(str).str.strip() == ''
    if empty_mask.sum() > 0:
        print(f"Found {empty_mask.sum()} empty result values")

    # Get current max numeric value to use as base for non-finishers
    try:
        numeric_results = pd.to_numeric(results, errors='coerce')
        max_position = numeric_results.max()
        # Use a safe default if max is NaN
        max_position = 20 if pd.isna(max_position) else max_position
    except:
        max_position = 20  # Default if we can't determine max

    # Create a dictionary for mapping special codes
    special_codes = {
        # Empty values (if not dropping them)
        '': max_position + 50,

        # Disqualifications (least bad of non-finishers)
        'D': max_position + 10,
        'DI': max_position + 10,
        'DP': max_position + 10,
        'DAI': max_position + 10,
        'DIS': max_position + 10,

        # Retired/Fell (medium bad)
        'RET': max_position + 20,
        'TOM': max_position + 20,
        'ARR': max_position + 20,
        'FER': max_position + 20,

        # Never started (worst outcome)
        'NP': max_position + 30,
        'ABS': max_position + 30
    }

    # Apply conversions
    def convert_value(val):
        # Check for empty strings
        if isinstance(val, str) and val.strip() == '':
            return np.nan if drop_empty else special_codes['']

        # If it's already numeric, return as is
        try:
            return float(val)
        except (ValueError, TypeError):
            # If it's a recognized code, map it
            if isinstance(val, str):
                for code, value in special_codes.items():
                    if code in val.upper():  # Case insensitive matching
                        return value
            # Default for any other unrecognized string
            return max_position + 40

    # Apply conversion to each value
    numeric_results = np.array([convert_value(val) for val in results])

    if drop_empty:
        # Create mask of valid (non-empty) entries
        valid_mask = ~np.isnan(numeric_results)
        return numeric_results, valid_mask
    else:
        # Return just the numeric results
        return numeric_results


def select_important_features(X: pd.DataFrame, y: np.ndarray,
                             k: int = 20, method: str = 'f_regression') -> List[str]:
    """
    Select the most important features for regression.

    Args:
        X: Feature DataFrame
        y: Target values
        k: Number of features to select
        method: Feature selection method ('f_regression' or 'mutual_info')

    Returns:
        List of selected feature names
    """
    # Adjust k if it's larger than the number of features
    k = min(k, X.shape[1])

    # Select the feature selection method
    if method == 'f_regression':
        selector = SelectKBest(f_regression, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_regression, k=k)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'f_regression' or 'mutual_info'.")

    # Apply feature selection
    selector.fit(X, y)

    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get feature names
    selected_features = X.columns[selected_indices].tolist()

    return selected_features


def calculate_race_specific_error(predictions: np.ndarray, actuals: np.ndarray,
                                race_ids: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate error metrics by race.

    Args:
        predictions: Model predictions
        actuals: Actual values
        race_ids: Race identifiers for each prediction

    Returns:
        Dictionary with race-specific error metrics
    """
    # Initialize results
    race_metrics = {}

    # Group by race
    unique_races = np.unique(race_ids)

    for race_id in unique_races:
        # Get data for this race
        race_mask = race_ids == race_id
        race_preds = predictions[race_mask]
        race_actual = actuals[race_mask]

        # Calculate metrics
        if len(race_preds) > 0:
            # Mean squared error
            mse = np.mean((race_preds - race_actual) ** 2)

            # Mean absolute error
            mae = np.mean(np.abs(race_preds - race_actual))

            # Winner correctly predicted
            # Find indices of minimum predicted and actual
            pred_winner_idx = np.argmin(race_preds)
            actual_winner_idx = np.argmin(race_actual)
            winner_correct = pred_winner_idx == actual_winner_idx

            # Podium accuracy (top 3)
            # Get indices of top 3 for both predictions and actuals
            n_podium = min(3, len(race_preds))
            pred_podium = np.argsort(race_preds)[:n_podium]
            actual_podium = np.argsort(race_actual)[:n_podium]
            podium_overlap = len(set(pred_podium) & set(actual_podium))
            podium_accuracy = podium_overlap / n_podium

            # Store results
            race_metrics[str(race_id)] = {
                'mse': mse,
                'mae': mae,
                'winner_correct': winner_correct,
                'podium_accuracy': podium_accuracy,
                'num_participants': len(race_preds)
            }

    # Calculate averages across all races
    all_mse = np.mean([m['mse'] for m in race_metrics.values()])
    all_mae = np.mean([m['mae'] for m in race_metrics.values()])
    all_winner = np.mean([m['winner_correct'] for m in race_metrics.values()])
    all_podium = np.mean([m['podium_accuracy'] for m in race_metrics.values()])

    # Add overall metrics
    race_metrics['overall'] = {
        'avg_mse': all_mse,
        'avg_mae': all_mae,
        'winner_accuracy': all_winner,
        'podium_accuracy': all_podium,
        'race_count': len(unique_races)
    }

    return race_metrics


#------------------------------------------------------------------------------
# CALIBRATION UTILITIES
#------------------------------------------------------------------------------

def isotonic_calibration(predictions: np.ndarray, targets: np.ndarray,
                         out_of_bounds: str = 'clip', y_min: float = None,
                         y_max: float = None) -> IsotonicRegression:
    """
    Calibrate predictions using isotonic regression.

    Args:
        predictions: Uncalibrated predictions
        targets: True target values
        out_of_bounds: Strategy for handling predictions outside bounds ('clip' or 'nan')
        y_min: Minimum output value
        y_max: Maximum output value

    Returns:
        Fitted isotonic regressor
    """
    # Create isotonic regression model
    isotonic = IsotonicRegression(out_of_bounds=out_of_bounds, y_min=y_min, y_max=y_max)

    # Fit the model
    isotonic.fit(predictions, targets)

    return isotonic


def apply_calibration(model: Any, calibrator: IsotonicRegression,
                      X: np.ndarray) -> np.ndarray:
    """
    Apply calibration to model predictions.

    Args:
        model: Trained regression model with predict method
        calibrator: Fitted isotonic regression calibrator
        X: Input features to predict with

    Returns:
        Calibrated predictions
    """
    # Get raw predictions
    raw_predictions = model.predict(X)

    # Apply calibration
    calibrated_predictions = calibrator.predict(raw_predictions)

    return calibrated_predictions


class CalibratedRegressor(BaseEstimator, RegressorMixin):
    """
    A regression model wrapper that automatically applies calibration.

    Combines a base regressor with an isotonic regression calibrator to
    provide more accurate predictions.
    """

    def __init__(self, base_regressor, clip_min: float = None, clip_max: float = None):
        """
        Initialize a calibrated regressor.

        Args:
            base_regressor: The base regression model
            clip_min: Minimum value for calibrated predictions
            clip_max: Maximum value for calibrated predictions
        """
        self.base_regressor = base_regressor
        self.calibrator = None
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, X_calib: Optional[np.ndarray] = None,
            y_calib: Optional[np.ndarray] = None, **fit_params) -> 'CalibratedRegressor':
        """
        Fit the base regressor and calibrator.

        Args:
            X: Training features
            y: Training targets
            X_calib: Features for calibration (if None, uses X)
            y_calib: Targets for calibration (if None, uses y)
            **fit_params: Additional parameters to pass to base regressor

        Returns:
            Fitted CalibratedRegressor instance
        """
        # If no separate calibration set, use training data
        if X_calib is None:
            X_calib = X
        if y_calib is None:
            y_calib = y

        # Fit the base regressor
        self.base_regressor.fit(X, y, **fit_params)

        # Get uncalibrated predictions on calibration set
        uncalibrated_preds = self.base_regressor.predict(X_calib)

        # Fit the calibrator
        self.calibrator = isotonic_calibration(
            uncalibrated_preds, y_calib,
            out_of_bounds='clip',
            y_min=self.clip_min,
            y_max=self.clip_max
        )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate calibrated predictions.

        Args:
            X: Input features to predict with

        Returns:
            Calibrated predictions
        """
        if not self.is_fitted:
            raise ValueError("This CalibratedRegressor instance is not fitted yet.")

        # Get raw predictions from base model
        raw_predictions = self.base_regressor.predict(X)

        # Apply calibration
        calibrated_predictions = self.calibrator.predict(raw_predictions)

        return calibrated_predictions

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Generate uncalibrated predictions directly from the base model.

        Args:
            X: Input features to predict with

        Returns:
            Uncalibrated predictions
        """
        if not self.is_fitted:
            raise ValueError("This CalibratedRegressor instance is not fitted yet.")

        return self.base_regressor.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate both raw and calibrated predictions.

        Args:
            X: Features
            y: True target values

        Returns:
            Dictionary with evaluation metrics
        """
        # Get raw and calibrated predictions
        raw_pred = self.predict_raw(X)
        calibrated_pred = self.predict(X)

        # Calculate metrics
        raw_metrics = calculate_metrics(y, raw_pred)
        calibrated_metrics = calculate_metrics(y, calibrated_pred)

        # Format results
        results = {}
        for k, v in raw_metrics.items():
            results[f"raw_{k}"] = v
        for k, v in calibrated_metrics.items():
            results[f"calibrated_{k}"] = v

        # Calculate improvement
        for k in raw_metrics.keys():
            if k in ['mse', 'rmse', 'mae']:
                # Lower is better
                improvement = (raw_metrics[k] - calibrated_metrics[k]) / raw_metrics[k] * 100
            else:
                # Higher is better
                improvement = (calibrated_metrics[k] - raw_metrics[k]) / raw_metrics[k] * 100

            results[f"{k}_improvement"] = improvement

        return results

    def save(self, filepath: str) -> None:
        """
        Save the calibrated regressor to a file.

        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'base_regressor': self.base_regressor,
            'calibrator': self.calibrator,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'is_fitted': self.is_fitted,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'CalibratedRegressor':
        """
        Load a calibrated regressor from a file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded CalibratedRegressor instance
        """
        model_data = joblib.load(filepath)

        # Create new instance
        calibrated_model = cls(
            base_regressor=model_data['base_regressor'],
            clip_min=model_data['clip_min'],
            clip_max=model_data['clip_max']
        )

        # Set fitted attributes
        calibrated_model.calibrator = model_data['calibrator']
        calibrated_model.is_fitted = model_data['is_fitted']

        return calibrated_model


#------------------------------------------------------------------------------
# EVALUATION UTILITIES
#------------------------------------------------------------------------------

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary with regression metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }

    # Add specialized racing metrics
    metrics.update({
        'top1_accuracy': np.mean((y_true == 1) == (np.round(y_pred) == 1)),
        'top3_accuracy': np.mean((y_true <= 3) == (np.round(y_pred) <= 3)),
        'exact_match': np.mean(np.round(y_pred) == y_true),
        'within_one': np.mean(abs(np.round(y_pred) - y_true) <= 1)
    })

    return metrics


def evaluate_regression_model(model: Any, X: np.ndarray, y: np.ndarray,
                              dataset_name: str = "Evaluation") -> Dict[str, float]:
    """
    Evaluate a regression model on a dataset.

    Args:
        model: Regression model with predict method
        X: Features
        y: True target values
        dataset_name: Name of the dataset for reporting

    Returns:
        Dictionary with evaluation metrics
    """
    # Generate predictions
    predictions = model.predict(X)

    # Calculate metrics
    metrics = calculate_metrics(y, predictions)

    # Add dataset name to metrics
    labeled_metrics = {f"{dataset_name}_{k}": v for k, v in metrics.items()}

    return labeled_metrics


def regression_metrics_report(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                             X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
                             output_file: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Generate a comprehensive evaluation report for a regression model.

    Args:
        model: Regression model with predict method
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        X_test: Test features (optional)
        y_test: Test targets (optional)
        output_file: Path to save the report (optional)

    Returns:
        Dictionary with metrics for each dataset
    """
    # Initialize results dictionary
    results = {}

    # Evaluate on training data
    train_metrics = evaluate_regression_model(model, X_train, y_train, "train")
    results["train"] = {k.replace("train_", ""): v for k, v in train_metrics.items()}

    # Evaluate on validation data if provided
    if X_val is not None and y_val is not None:
        val_metrics = evaluate_regression_model(model, X_val, y_val, "val")
        results["val"] = {k.replace("val_", ""): v for k, v in val_metrics.items()}

    # Evaluate on test data if provided
    if X_test is not None and y_test is not None:
        test_metrics = evaluate_regression_model(model, X_test, y_test, "test")
        results["test"] = {k.replace("test_", ""): v for k, v in test_metrics.items()}

    # Generate report string
    report = ["=== Regression Model Evaluation ===\n"]

    for dataset_name, metrics in results.items():
        report.append(f"\n{dataset_name.upper()} SET METRICS:")
        for metric_name, value in metrics.items():
            report.append(f"  {metric_name}: {value:.4f}")

    report_str = "\n".join(report)

    # Print or save the report
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_str)
    else:
        print(report_str)

    return results


#------------------------------------------------------------------------------
# VISUALIZATION UTILITIES
#------------------------------------------------------------------------------

def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                             title: str = "Prediction vs Actual",
                             save_path: Optional[str] = None) -> None:
    """
    Plot predicted values against actual values.

    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot (None to display)
    """
    plt.figure(figsize=(10, 6))

    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Add labels and title
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)

    # Add metrics annotation
    metrics = calculate_metrics(y_true, y_pred)
    metrics_text = f"RMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nR²: {metrics['r2']:.4f}"
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Save or display
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_calibration_effect(model_raw_preds: np.ndarray, model_calibrated_preds: np.ndarray,
                          y_true: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Plot the effect of calibration on model predictions.

    Args:
        model_raw_preds: Uncalibrated model predictions
        model_calibrated_preds: Calibrated model predictions
        y_true: True target values
        save_path: Path to save the plot (None to display)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot uncalibrated predictions
    ax1.scatter(y_true, model_raw_preds, alpha=0.5)
    min_val = min(np.min(y_true), np.min(model_raw_preds))
    max_val = max(np.max(y_true), np.max(model_raw_preds))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Raw Predictions")
    ax1.set_title("Before Calibration")

    raw_metrics = calculate_metrics(y_true, model_raw_preds)
    raw_metrics_text = f"RMSE: {raw_metrics['rmse']:.4f}\nMAE: {raw_metrics['mae']:.4f}\nR²: {raw_metrics['r2']:.4f}"
    ax1.annotate(raw_metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Plot calibrated predictions
    ax2.scatter(y_true, model_calibrated_preds, alpha=0.5)
    min_val = min(np.min(y_true), np.min(model_calibrated_preds))
    max_val = max(np.max(y_true), np.max(model_calibrated_preds))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Calibrated Predictions")
    ax2.set_title("After Calibration")

    cal_metrics = calculate_metrics(y_true, model_calibrated_preds)
    cal_metrics_text = f"RMSE: {cal_metrics['rmse']:.4f}\nMAE: {cal_metrics['mae']:.4f}\nR²: {cal_metrics['r2']:.4f}"
    ax2.annotate(cal_metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()

    # Save or display
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                           title: str = "Feature Importance", top_n: int = 20,
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance values from a model.

    Args:
        feature_names: List of feature names
        importances: Array of importance values corresponding to features
        title: Plot title
        top_n: Number of top features to show (default 20)
        save_path: Path to save the plot (None to display)
    """
    # Create DataFrame for easier sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance and take top N
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(importance_df['Feature'], importance_df['Importance'])

    # Add values to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center')

    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()

    # Save or display
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_histogram_of_errors(y_true: np.ndarray, y_pred: np.ndarray,
                            title: str = "Distribution of Prediction Errors",
                            bins: int = 30, save_path: Optional[str] = None) -> None:
    """
    Plot histogram of prediction errors.

    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
        bins: Number of histogram bins
        save_path: Path to save the plot (None to display)
    """
    # Calculate errors
    errors = y_pred - y_true

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=bins, alpha=0.75, edgecolor='black')

    # Add mean and std lines
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=1,
                label=f'Mean Error: {mean_error:.4f}')
    plt.axvline(mean_error + std_error, color='green', linestyle='dashed', linewidth=1,
                label=f'Mean + 1 StdDev: {mean_error + std_error:.4f}')
    plt.axvline(mean_error - std_error, color='green', linestyle='dashed', linewidth=1,
                label=f'Mean - 1 StdDev: {mean_error - std_error:.4f}')

    # Add labels and title
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    # Save or display
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Direct exports for cleaner imports
__all__ = [
    # Calibration
    'isotonic_calibration', 'apply_calibration', 'CalibratedRegressor',

    # Evaluation
    'calculate_metrics', 'evaluate_regression_model', 'regression_metrics_report',

    # Data processing
    'convert_race_results_to_numeric', 'select_important_features', 'calculate_race_specific_error',

    # Visualization
    'plot_prediction_vs_actual', 'plot_calibration_effect', 'plot_feature_importance',
    'plot_histogram_of_errors'
]