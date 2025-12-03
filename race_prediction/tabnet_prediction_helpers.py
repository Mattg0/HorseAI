"""
TabNet Prediction Helpers

Helper functions for making predictions with TabNet models that use feature selection.
Integrates seamlessly with existing prediction pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from pytorch_tabnet.tab_model import TabNetRegressor
from core.feature_selection.tabnet_feature_selector import TabNetFeatureSelector
from core.calculators.static_feature_calculator import FeatureCalculator
from core.calculators.quinte_feature_calculator import QuinteFeatureCalculator


def load_tabnet_with_selector(model_path: str) -> tuple:
    """
    Load TabNet model and its feature selector

    Args:
        model_path: Path to model directory

    Returns:
        Tuple of (model, feature_selector, feature_list)
    """
    model_dir = Path(model_path)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Load TabNet model
    model = TabNetRegressor()
    model_file = model_dir / 'tabnet_model.zip'

    if not model_file.exists():
        raise FileNotFoundError(f"TabNet model not found: {model_file}")

    model.load_model(str(model_dir / 'tabnet_model'))
    print(f"✓ Loaded TabNet model from {model_file}")

    # Load feature selector
    selector_file = model_dir / 'feature_selector.json'

    if not selector_file.exists():
        # Fallback: try to load from feature_columns.json
        print(f"Warning: feature_selector.json not found, trying feature_columns.json...")
        features_file = model_dir / 'feature_columns.json'

        if features_file.exists():
            with open(features_file, 'r') as f:
                feature_list = json.load(f)

            # Create a simple selector with just the feature list
            selector = TabNetFeatureSelector()
            selector.selected_features = feature_list
            print(f"✓ Loaded {len(feature_list)} features from feature_columns.json")
        else:
            raise FileNotFoundError(f"Neither feature_selector.json nor feature_columns.json found")
    else:
        selector = TabNetFeatureSelector()
        selector.load(str(selector_file))

    feature_list = selector.selected_features

    return model, selector, feature_list


def predict_race_tabnet(race_data: pd.DataFrame,
                         model_path: str,
                         model_type: str = 'general',
                         db_path: Optional[str] = None) -> np.ndarray:
    """
    Predict race positions using TabNet with automatic feature selection

    This function:
    1. Calculates ALL features (same as training)
    2. Loads feature selector
    3. Applies same feature selection as training
    4. Makes predictions

    Args:
        race_data: DataFrame with race data
        model_path: Path to model directory
        model_type: 'general' or 'quinte'
        db_path: Optional database path for temporal features

    Returns:
        Predictions array
    """
    # 1. Calculate ALL features (just like training)
    print(f"\nCalculating features for {len(race_data)} horses...")

    df = race_data.copy()

    # Calculate standard features
    df = FeatureCalculator.calculate_all_features(
        df,
        use_temporal=True if db_path else False,
        db_path=db_path
    )

    # Calculate quinte features if needed
    if model_type == 'quinte' and db_path:
        quinte_calc = QuinteFeatureCalculator(db_path)
        df = quinte_calc.calculate_quinte_features(df)

    print(f"  Calculated {len(df.columns)} features")

    # 2. Load model and feature selector
    model, selector, feature_list = load_tabnet_with_selector(model_path)

    # 3. Select features (apply same selection as training)
    # Get only the features that were selected during training
    available_features = [f for f in feature_list if f in df.columns]
    missing_features = [f for f in feature_list if f not in df.columns]

    if missing_features:
        print(f"Warning: {len(missing_features)} selected features missing in race data")
        print(f"  Examples: {missing_features[:3]}")

    X_selected = df[available_features]
    print(f"  Selected {len(available_features)} features for prediction")

    # 4. Make predictions
    predictions = model.predict(X_selected.values)

    return predictions.flatten()


def batch_predict_races_tabnet(races: List[pd.DataFrame],
                                model_path: str,
                                model_type: str = 'general',
                                db_path: Optional[str] = None) -> List[np.ndarray]:
    """
    Predict multiple races in batch using TabNet

    Args:
        races: List of race DataFrames
        model_path: Path to model directory
        model_type: 'general' or 'quinte'
        db_path: Optional database path

    Returns:
        List of prediction arrays
    """
    # Load model once
    model, selector, feature_list = load_tabnet_with_selector(model_path)

    predictions_list = []

    for i, race_data in enumerate(races):
        print(f"\nPredicting race {i+1}/{len(races)}...")

        # Calculate features
        df = race_data.copy()
        df = FeatureCalculator.calculate_all_features(
            df,
            use_temporal=True if db_path else False,
            db_path=db_path
        )

        if model_type == 'quinte' and db_path:
            quinte_calc = QuinteFeatureCalculator(db_path)
            df = quinte_calc.calculate_quinte_features(df)

        # Select features
        available_features = [f for f in feature_list if f in df.columns]
        X_selected = df[available_features]

        # Predict
        predictions = model.predict(X_selected.values)
        predictions_list.append(predictions.flatten())

    return predictions_list


def compare_rf_tabnet_predictions(race_data: pd.DataFrame,
                                   rf_model_path: str,
                                   tabnet_model_path: str,
                                   model_type: str = 'general',
                                   db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare predictions from RF and TabNet models

    Args:
        race_data: DataFrame with race data
        rf_model_path: Path to RF model
        tabnet_model_path: Path to TabNet model
        model_type: 'general' or 'quinte'
        db_path: Optional database path

    Returns:
        Dictionary with both predictions and comparison
    """
    import joblib

    # Calculate features once
    df = race_data.copy()
    df = FeatureCalculator.calculate_all_features(
        df,
        use_temporal=True if db_path else False,
        db_path=db_path
    )

    if model_type == 'quinte' and db_path:
        quinte_calc = QuinteFeatureCalculator(db_path)
        df = quinte_calc.calculate_quinte_features(df)

    # RF Prediction (uses all features)
    rf_model = joblib.load(rf_model_path)
    rf_feature_file = Path(rf_model_path).parent / 'feature_columns.json'

    with open(rf_feature_file, 'r') as f:
        rf_features = json.load(f)

    rf_available = [f for f in rf_features if f in df.columns]
    rf_predictions = rf_model.predict(df[rf_available].values)

    # TabNet Prediction (uses selected features)
    tabnet_model, tabnet_selector, tabnet_features = load_tabnet_with_selector(tabnet_model_path)
    tabnet_available = [f for f in tabnet_features if f in df.columns]
    tabnet_predictions = tabnet_model.predict(df[tabnet_available].values).flatten()

    # Compare
    correlation = np.corrcoef(rf_predictions, tabnet_predictions)[0, 1]
    mean_diff = np.mean(np.abs(rf_predictions - tabnet_predictions))

    return {
        'rf_predictions': rf_predictions,
        'tabnet_predictions': tabnet_predictions,
        'rf_features_used': len(rf_available),
        'tabnet_features_used': len(tabnet_available),
        'correlation': float(correlation),
        'mean_absolute_difference': float(mean_diff),
        'rf_mean': float(np.mean(rf_predictions)),
        'tabnet_mean': float(np.mean(tabnet_predictions))
    }


def get_tabnet_feature_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about TabNet model's feature selection

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary with feature selection info
    """
    model_dir = Path(model_path)

    # Try to load feature selector
    selector_file = model_dir / 'feature_selector.json'

    if not selector_file.exists():
        return {'error': 'feature_selector.json not found'}

    with open(selector_file, 'r') as f:
        selector_data = json.load(f)

    metadata = selector_data.get('metadata', {})

    return {
        'selected_features': selector_data.get('selected_features', []),
        'selected_count': len(selector_data.get('selected_features', [])),
        'original_count': metadata.get('original_count', None),
        'removed_constant': len(metadata.get('removed_constant', [])),
        'removed_sparse': len(metadata.get('removed_sparse', [])),
        'removed_correlated': len(metadata.get('removed_correlated', [])),
        'sparse_threshold': metadata.get('sparse_threshold', None),
        'correlation_threshold': metadata.get('correlation_threshold', None)
    }
