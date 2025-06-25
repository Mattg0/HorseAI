import os
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple
import logging
from datetime import datetime

# Import the optimized orchestrator and utilities
from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from utils.model_manager import get_model_manager


class RacePredictor:
    """
    Optimized race predictor that uses the same data preparation pipeline as training.
    Leverages the hybrid RF+LSTM model with optimized blend weights.
    """

    def __init__(self, model_path: str = None, db_name: str = None, verbose: bool = False):
        """
        Initialize the optimized race predictor.

        Args:
            model_path: Path to the trained model (if None, uses latest from config)
            db_name: Database name from config (defaults to active_db)
            verbose: Whether to print verbose output
        """
        # Initialize config
        self.config = AppConfig()
        self.verbose = verbose

        # Set database
        if db_name is None:
            self.db_name = self.config._config.base.active_db
        else:
            self.db_name = db_name

        # Get database path
        self.db_path = self.config.get_sqlite_dbpath(self.db_name)

        # Initialize the SAME orchestrator used in training
        self.orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=self.db_path,
            verbose=verbose
        )

        # Load model configuration and determine model path
        if model_path is None:
            self.model_manager = get_model_manager()
            model_path = self.model_manager.get_model_path()

        self.model_path = model_path
        # Load models and configuration
        self._load_models()

        if self.verbose:
            print(f"OptimizedRacePredictor initialized")
            print(f"  Model: {self.model_path}")
            print(f"  Database: {self.db_path}")
            print(f"  Blend weight: {getattr(self, 'blend_weight', 0.9)}")

    def _determine_model_path(self, model_path):
        """Determine the model path to use."""
        if model_path is None:
            # Use latest model from config
            try:
                latest_model = self.config._config.models.latest_hybrid_model
                model_paths = self.config.get_model_paths(
                    self.config._config,
                    model_name='hybrid_model'
                )
                self.model_path = Path(model_paths['model_path']) / latest_model
            except:
                # Fallback to manual path construction
                model_dir = self.config._config.models.model_dir
                db_type = self.config._config.base.active_db
                self.model_path = Path(model_dir) / db_type / "hybrid_model"

                # Find latest version
                versions = [d for d in self.model_path.iterdir()
                            if d.is_dir() and d.name.startswith('v')]
                if versions:
                    versions.sort(reverse=True)
                    self.model_path = versions[0]
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path {self.model_path} does not exist")

    def _load_models(self):
        """Load the trained models and configuration."""
        # Load model configuration
        config_path = self.model_path / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
                # Get blend weight from model config or use default
                training_results = self.model_config.get('training_results', {})
                self.blend_weight = training_results.get('blend_weight', 0.9)
        else:
            self.model_config = {}
            self.blend_weight = 0.9

        # Load feature engineering state to match training
        feature_config_path = self.model_path / "hybrid_feature_engineer.joblib"
        if feature_config_path.exists():
            feature_config = joblib.load(feature_config_path)

            # Update orchestrator with same parameters used in training
            if isinstance(feature_config, dict):
                if 'preprocessing_params' in feature_config:
                    self.orchestrator.preprocessing_params.update(
                        feature_config['preprocessing_params']
                    )
                if 'embedding_dim' in feature_config:
                    self.orchestrator.embedding_dim = feature_config['embedding_dim']

        # Load RF model
        rf_model_path = self.model_path / "hybrid_rf_model.joblib"
        if rf_model_path.exists():
            self.rf_model = joblib.load(rf_model_path)
            if self.verbose:
                print(f"Loaded RF model: {type(self.rf_model)}")
        else:
            self.rf_model = None
            print("Warning: RF model not found")

        # Load LSTM model
        lstm_model_path = self.model_path / "hybrid_lstm_model"
        if lstm_model_path.exists():
            try:
                from tensorflow.keras.models import load_model
                self.lstm_model = load_model(lstm_model_path)
                if self.verbose:
                    print(f"Loaded LSTM model")
            except Exception as e:
                print(f"Warning: Could not load LSTM model: {str(e)}")
                self.lstm_model = None
        else:
            self.lstm_model = None
            if self.verbose:
                print("LSTM model not found - will use RF only")

    def prepare_race_data(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare race data using the SAME pipeline as training.

        Args:
            race_df: DataFrame with race and participant data

        Returns:
            DataFrame with processed features ready for prediction
        """
        if self.verbose:
            print(f"Preparing race data: {len(race_df)} participants")

        # Step 1: Add missing required columns with defaults
        required_columns = {
            'idche': 0,
            'idJockey': 0,
            'numero': 0,
            'typec': 'P',  # Default to Plat
            'final_position': np.nan,  # This will be missing during prediction
            'jour': datetime.now().strftime('%Y-%m-%d')
        }

        for col, default_val in required_columns.items():
            if col not in race_df.columns:
                race_df[col] = default_val

        # Step 2: Ensure numeric columns are properly typed
        numeric_fields = [
            'age', 'cotedirect', 'coteprob', 'pourcVictChevalHippo',
            'pourcPlaceChevalHippo', 'pourcVictJockHippo', 'pourcPlaceJockHippo',
            'victoirescheval', 'placescheval', 'coursescheval', 'dist',
            'temperature', 'forceVent', 'idche', 'idJockey', 'numero'
        ]

        for field in numeric_fields:
            if field in race_df.columns:
                race_df[field] = pd.to_numeric(race_df[field], errors='coerce').fillna(0)

        # Step 3: Use the SAME data preparation pipeline as training
        # First prepare features (handle missing values, categorical encoding, etc.)
        processed_df = self.orchestrator.prepare_features(race_df, use_cache=False)

        # Step 4: Apply embeddings (this will fit embeddings if needed)
        # Use a larger sample for fitting if this is the first time
        if not self.orchestrator.embeddings_fitted:
            if self.verbose:
                print("Fitting embeddings from historical data...")
            # Load some historical data to fit embeddings
            historical_df = self.orchestrator.load_historical_races(
                limit=1000, use_cache=True
            )
            self.orchestrator.fit_embeddings(historical_df, use_cache=True)

        # Apply embeddings to the race data
        embedded_df = self.orchestrator.apply_embeddings(processed_df, use_cache=False)

        if self.verbose:
            print(f"Data preparation complete: {len(embedded_df.columns)} features")

        return embedded_df

    def predict_with_rf(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the RF model."""
        if self.rf_model is None:
            raise ValueError("RF model not loaded")

        # Ensure feature alignment with training
        expected_features = None
        if hasattr(self.rf_model, 'feature_names_in_'):
            expected_features = self.rf_model.feature_names_in_
        elif hasattr(self.rf_model, 'base_regressor'):
            if hasattr(self.rf_model.base_regressor, 'feature_names_in_'):
                expected_features = self.rf_model.base_regressor.feature_names_in_

        if expected_features is not None:
            # Align features with training
            aligned_X = pd.DataFrame(0, index=range(len(X)), columns=expected_features)
            common_features = set(X.columns) & set(expected_features)

            for feature in common_features:
                aligned_X[feature] = X[feature].values

            X_for_prediction = aligned_X

            if self.verbose:
                print(f"RF prediction: {len(common_features)}/{len(expected_features)} features aligned")
        else:
            X_for_prediction = X

        return self.rf_model.predict(X_for_prediction)

    def predict_with_lstm(self, embedded_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the LSTM model."""
        if self.lstm_model is None:
            return None

        try:
            # Use the same sequence preparation as training
            X_sequences, X_static, _ = self.orchestrator.extract_lstm_features(embedded_df)

            if X_sequences is None or X_static is None:
                if self.verbose:
                    print("Warning: Could not prepare LSTM sequences")
                return None

            # Generate predictions
            lstm_preds = self.lstm_model.predict([X_sequences, X_static], verbose=0)

            if len(lstm_preds.shape) > 1:
                lstm_preds = lstm_preds.flatten()

            if self.verbose:
                print(f"LSTM prediction: {len(lstm_preds)} predictions generated")

            return lstm_preds

        except Exception as e:
            if self.verbose:
                print(f"Warning: LSTM prediction failed: {str(e)}")
            return None

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict race outcome using the optimized hybrid model.

        Args:
            race_df: DataFrame with race and participant data

        Returns:
            DataFrame with predictions and rankings
        """
        if self.verbose:
            print(f"Predicting race with {len(race_df)} participants")

        # Step 1: Prepare data using the same pipeline as training
        embedded_df = self.prepare_race_data(race_df)

        # Step 2: Extract RF features and generate RF predictions
        X_rf, _ = self.orchestrator.extract_rf_features(embedded_df)
        rf_predictions = self.predict_with_rf(X_rf)

        # Step 3: Generate LSTM predictions if model available
        lstm_predictions = self.predict_with_lstm(embedded_df)

        # Step 4: Blend predictions using optimized weights
        if lstm_predictions is not None and len(lstm_predictions) == len(rf_predictions):
            # Use the optimized blend weight from training
            final_predictions = (rf_predictions * self.blend_weight +
                                 lstm_predictions * (1 - self.blend_weight))
            if self.verbose:
                print(f"Blended predictions using weight {self.blend_weight}")
        else:
            final_predictions = rf_predictions
            if self.verbose:
                print("Using RF predictions only")

        # Step 5: Create result DataFrame
        result_df = race_df.copy()
        result_df['predicted_position'] = final_predictions

        # Sort by predicted position (lower is better)
        result_df = result_df.sort_values('predicted_position')
        result_df['predicted_rank'] = range(1, len(result_df) + 1)

        # Create arrival string format
        numeros_ordered = result_df['numero'].astype(str).tolist()
        predicted_arriv = '-'.join(numeros_ordered)
        result_df['predicted_arriv'] = predicted_arriv

        if self.verbose:
            print("Prediction complete")
            print(f"Top 3 predicted: {numeros_ordered[:3]}")

        return result_df


def predict_race_simple(race_data: Union[pd.DataFrame, List[Dict], str],
                        model_path: str = None,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Simple function to predict a race - perfect for IDE usage.

    Args:
        race_data: Race data as DataFrame, list of dicts, or JSON string
        model_path: Path to model (None = use latest from config)
        verbose: Whether to print progress

    Returns:
        DataFrame with predictions and rankings
    """
    # Convert input to DataFrame if needed
    if isinstance(race_data, str):
        race_df = pd.DataFrame(json.loads(race_data))
    elif isinstance(race_data, list):
        race_df = pd.DataFrame(race_data)
    else:
        race_df = race_data.copy()

    # Create predictor and generate predictions
    predictor = RacePredictor(model_path=model_path, verbose=verbose)
    result = predictor.predict_race(race_df)

    return result


# Example usage for IDE
def example_prediction():
    """Example function showing how to use the predictor from IDE."""

    # Sample race data (you would get this from your API or database)
    sample_race_data = [
        {
            'numero': 1, 'cheval': 'Horse A', 'idche': 101, 'idJockey': 201,
            'age': 5, 'cotedirect': 3.2, 'victoirescheval': 2, 'placescheval': 5,
            'coursescheval': 12, 'pourcVictChevalHippo': 16.7, 'pourcPlaceChevalHippo': 41.7
        },
        {
            'numero': 2, 'cheval': 'Horse B', 'idche': 102, 'idJockey': 202,
            'age': 6, 'cotedirect': 5.1, 'victoirescheval': 1, 'placescheval': 3,
            'coursescheval': 8, 'pourcVictChevalHippo': 12.5, 'pourcPlaceChevalHippo': 37.5
        },
        {
            'numero': 3, 'cheval': 'Horse C', 'idche': 103, 'idJockey': 203,
            'age': 4, 'cotedirect': 7.8, 'victoirescheval': 0, 'placescheval': 2,
            'coursescheval': 6, 'pourcVictChevalHippo': 0.0, 'pourcPlaceChevalHippo': 33.3
        }
    ]

    # Add race context information
    race_info = {
        'typec': 'P', 'dist': 1600, 'natpis': 'PSF', 'meteo': 'BEAU',
        'temperature': 15, 'hippo': 'VINCENNES'
    }

    # Add race info to each participant
    for participant in sample_race_data:
        participant.update(race_info)

    # Generate predictions
    print("Running example prediction...")
    results = predict_race_simple(sample_race_data, verbose=True)

    # Display results
    print("\nPrediction Results:")
    print("=" * 50)
    for _, horse in results.head(3).iterrows():
        print(f"{horse['predicted_rank']}. {horse['cheval']} (#{horse['numero']}) - "
              f"Predicted position: {horse['predicted_position']:.2f}")

    print(f"\nPredicted arrival order: {results['predicted_arriv'].iloc[0]}")

    return results


if __name__ == "__main__":
    # Run example when executed directly from IDE
    example_prediction()