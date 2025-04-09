import os
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple
import logging
from datetime import datetime

# Import from existing code
from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from model_training.regressions.isotonic_calibration import CalibratedRegressor


class RacePredictor:
    """
    Race predictor that loads trained models and applies them to new race data.
    Handles both static models (Random Forest) and sequence models (LSTM) if available.
    """

    def __init__(self, model_path: str = None, db_name: str = "dev",
                 use_latest_base: bool = False, verbose: bool = False):
        """
        Initialize the race predictor.

        Args:
            model_path: Path to saved model directory (use None with use_latest_base=True to use latest from config)
            db_name: Database configuration name from config
            use_latest_base: Whether to use the latest base model from config
            verbose: Whether to print verbose output
        """
        # Initialize config
        self.config = AppConfig()

        # If use_latest_base is True, try to get latest model from config
        if use_latest_base:
            try:
                if hasattr(self.config._config.models, 'latest_base_model'):
                    latest_model = self.config._config.models.latest_base_model
                    # Construct full path to model
                    base_model_dir = os.path.join(
                        self.config._config.models.model_dir,
                        'hybrid'  # Default model name
                    )
                    self.model_path = Path(base_model_dir) / latest_model
                    print(f"Using latest base model from config: {self.model_path}")
                else:
                    # Fall back to provided model_path
                    self.model_path = Path(model_path) if model_path else None
                    print("No latest_base_model found in config, using specified model path")
            except (AttributeError, TypeError) as e:
                print(f"Error loading latest_base_model from config: {str(e)}")
                self.model_path = Path(model_path) if model_path else None
        else:
            # Use specified model path
            self.model_path = Path(model_path) if model_path else None

        # Check if model path exists
        if self.model_path is None:
            raise ValueError("No model path provided and couldn't get latest_base_model from config")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path {self.model_path} does not exist")

        self.verbose = verbose

        # Get database path from config
        self.db_path = get_sqlite_dbpath(db_name)

        # Initialize orchestrator for feature embedding
        self.orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=self.db_path,
            verbose=verbose
        )

        # Set up logging
        self._setup_logging()

        # Load models and configuration
        self._load_models()

        self.log_info(f"Race predictor initialized with model at {self.model_path}")
        self.log_info(f"Using database: {self.db_path}")

    def _setup_logging(self):
        """Set up logging."""
        # Create logs directory if it doesn't exist
        log_dir = self.model_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Set up logger
        self.logger = logging.getLogger("RacePredictor")
        self.logger.setLevel(logging.INFO if not self.verbose else logging.DEBUG)

        # Add file handler
        log_file = log_dir / f"race_predictor_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)

    def log_info(self, message):
        """Log an info message."""
        if hasattr(self, 'logger'):
            self.logger.info(message)
        else:
            print(message)

    def log_error(self, message):
        """Log an error message."""
        if hasattr(self, 'logger'):
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")

    def _load_models(self):
        """Load the trained models using the new directory structure."""
        # Get model manager
        from utils.model_manager import get_model_manager
        model_manager = get_model_manager()

        # Extract db_type from model_path
        db_type = self.model_path.parts[-3] if len(self.model_path.parts) >= 3 else None

        if not db_type:
            # Try to get db_type from config
            db_type = self.config._config.base.active_db
            self.log_info(f"Using database type from config: {db_type}")

        # Determine which models to load based on the specified path
        model_type = self.model_path.parts[-2] if len(self.model_path.parts) >= 2 else 'hybrid'

        self.log_info(f"Loading models of type '{model_type}' for database '{db_type}'")

        # Get the latest date for the specified model type
        # This could be extracted from directory listing or config
        date = None

        # Try to get latest date from config
        try:
            if model_type == 'rf' and hasattr(self.config._config.models, 'latest_rf_model'):
                latest_model = self.config._config.models.latest_rf_model
                date = latest_model.split('_')[1]  # Extract date from "db_date"
            elif model_type == 'lstm' and hasattr(self.config._config.models, 'latest_lstm_model'):
                latest_model = self.config._config.models.latest_lstm_model
                date = latest_model.split('_')[1]  # Extract date from "db_date"
            elif model_type == 'hybrid' and hasattr(self.config._config.models, 'latest_hybrid_model'):
                latest_model = self.config._config.models.latest_hybrid_model
                date = latest_model.split('_')[1]  # Extract date from "db_date"
        except (AttributeError, IndexError):
            # If we can't get date from config, find latest file
            pass

        if not date:
            # Find latest file in the directory
            model_dir = model_manager.get_model_path(db_type=db_type, model_type=model_type)
            if model_dir.exists():
                # List files and sort by modification time
                files = list(model_dir.glob(f"*.*"))
                if files:
                    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    # Try to extract date from filename
                    filename = files[0].name
                    date_match = re.search(r'_(\d{8})\.', filename)
                    if date_match:
                        date = date_match.group(1)
                    else:
                        # Use current date as fallback
                        date = datetime.now().strftime('%Y%m%d')

        self.log_info(f"Using date: {date}")

        # Load models based on model_type
        if model_type == 'rf' or model_type == 'hybrid':
            # Load RF model
            rf_path = model_manager.get_model_path(db_type=db_type, model_type='rf')
            rf_model_file = rf_path / f"model_{date}.joblib"

            if rf_model_file.exists():
                try:
                    # Load the RF model
                    rf_data = joblib.load(rf_model_file)

                    # Handle different model formats
                    if isinstance(rf_data, dict) and 'model' in rf_data:
                        self.rf_model = rf_data['model']
                    elif hasattr(rf_data, 'predict') and callable(getattr(rf_data, 'predict')):
                        self.rf_model = rf_data
                    else:
                        # Try to find model component
                        if hasattr(rf_data, 'base_regressor'):
                            self.rf_model = rf_data
                        else:
                            self.rf_model = rf_data

                    self.log_info(f"Loaded RF model from {rf_model_file}")

                    # Load feature configuration
                    feature_config_file = rf_path / f"feature_config_{date}.joblib"
                    if feature_config_file.exists():
                        self.feature_config = joblib.load(feature_config_file)
                        self.log_info(f"Loaded feature configuration from {feature_config_file}")
                except Exception as e:
                    self.log_error(f"Error loading RF model: {str(e)}")

        if model_type == 'lstm' or model_type == 'hybrid':
            # Load LSTM model
            lstm_path = model_manager.get_model_path(db_type=db_type, model_type='lstm')
            lstm_model_dir = lstm_path / f"model_{date}"

            if lstm_model_dir.exists():
                try:
                    from tensorflow.keras.models import load_model
                    self.lstm_model = load_model(lstm_model_dir)
                    self.log_info(f"Loaded LSTM model from {lstm_model_dir}")

                    # Load history if available
                    history_file = lstm_path / f"history_{date}.joblib"
                    if history_file.exists():
                        self.history = joblib.load(history_file)
                        self.log_info(f"Loaded LSTM training history from {history_file}")
                except Exception as e:
                    self.log_error(f"Error loading LSTM model: {str(e)}")

        if model_type == 'hybrid':
            # Load hybrid configuration
            hybrid_path = model_manager.get_model_path(db_type=db_type, model_type='hybrid')
            config_file = hybrid_path / f"config_{date}.json"

            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        self.model_config = json.load(f)
                    self.log_info(f"Loaded hybrid configuration from {config_file}")
                except Exception as e:
                    self.log_error(f"Error loading hybrid configuration: {str(e)}")

    def prepare_race_data(self, race_df: pd.DataFrame) -> Tuple[
        pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare race data for prediction by applying feature engineering and embeddings.

        Args:
            race_df: DataFrame with race and participant data

        Returns:
            Tuple of (processed features DataFrame, sequence features, static features)
        """
        # Make a copy to avoid modifying the original
        df = race_df.copy()

        # Add missing columns that are expected by the feature engineering pipeline
        missing_cols = []
        expected_cols = ['idche', 'idJockey', 'cheval', 'cotedirect', 'numero']

        for col in expected_cols:
            if col not in df.columns:
                missing_cols.append(col)
                df[col] = None

        if missing_cols:
            self.log_info(f"Added missing columns for feature engineering: {missing_cols}")

        # Convert numeric fields
        numeric_fields = [
            'age', 'cotedirect', 'coteprob', 'pourcVictChevalHippo',
            'pourcPlaceChevalHippo', 'pourcVictJockHippo', 'pourcPlaceJockHippo',
            'victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere',
            'gainsAnneeEnCours', 'nbCourseCouple', 'nbVictCouple', 'nbPlaceCouple',
            'TxVictCouple', 'recence', 'dist', 'temperature', 'forceVent',
            'ratio_victoires', 'ratio_places', 'gains_par_course',
            'efficacite_couple', 'regularite_couple', 'progression_couple',
            'perf_cheval_hippo', 'perf_jockey_hippo'
        ]

        for field in numeric_fields:
            if field in df.columns and not pd.api.types.is_numeric_dtype(df[field]):
                df[field] = pd.to_numeric(df[field], errors='coerce')

        # Prepare features using orchestrator
        self.log_info("Preparing features...")

        # First apply feature calculator for static features
        try:
            from core.calculators.static_feature_calculator import FeatureCalculator
            df = FeatureCalculator.calculate_all_features(df)
            self.log_info(f"Applied static feature calculations")
        except Exception as e:
            self.log_error(f"Error applying static feature calculations: {str(e)}")

        # Apply embeddings
        try:
            df = self.orchestrator.apply_embeddings(df, clean_after_embedding=False)
            self.log_info(f"Applied embeddings")
        except Exception as e:
            self.log_error(f"Error applying embeddings: {str(e)}")

        # Clean up features for RF model
        X = self.orchestrator.drop_embedded_raw_features(df)
        self.log_info(f"Prepared {len(X)} samples with {X.shape[1]} features for RF model")

        # Prepare sequence data for LSTM model if available
        X_seq = None
        X_static = None

        if self.lstm_model is not None and hasattr(self.orchestrator, 'prepare_sequence_data'):
            try:
                # Get sequence length from model config or use default
                sequence_length = self.feature_config.get('sequence_length', 5) if self.feature_config else 5

                self.log_info(f"Preparing sequence data with length {sequence_length}...")
                X_seq, X_static, _ = self.orchestrator.prepare_sequence_data(
                    df, sequence_length=sequence_length
                )
                self.log_info(
                    f"Prepared sequence data with shape {X_seq.shape} and static data with shape {X_static.shape}")
            except Exception as e:
                self.log_error(f"Error preparing sequence data: {str(e)}")
                X_seq = None
                X_static = None

        return X, X_seq, X_static

    def predict_with_rf(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the Random Forest model with feature name alignment.

        Args:
            X: Feature DataFrame

        Returns:
            NumPy array with predictions
        """
        if self.rf_model is None:
            self.log_error("Random Forest model not loaded")
            return np.zeros(len(X))

        try:
            # Check if we have feature names from training
            expected_features = None

            # Try to get expected feature names from different places
            if hasattr(self.feature_config, 'preprocessing_params') and 'feature_columns' in self.feature_config[
                'preprocessing_params']:
                expected_features = self.feature_config['preprocessing_params']['feature_columns']
                self.log_info(f"Found {len(expected_features)} expected feature columns from config")
            elif hasattr(self.rf_model, 'feature_names_in_'):
                expected_features = self.rf_model.feature_names_in_
                self.log_info(f"Found {len(expected_features)} feature names from model")
            elif isinstance(self.rf_model, dict) and 'base_regressor' in self.rf_model:
                if hasattr(self.rf_model['base_regressor'], 'feature_names_in_'):
                    expected_features = self.rf_model['base_regressor'].feature_names_in_
                    self.log_info(f"Found {len(expected_features)} feature names from base_regressor")

            # If we have expected features, align the input data
            if expected_features is not None:
                # Create a DataFrame with expected feature columns
                aligned_X = pd.DataFrame(0, index=range(len(X)), columns=expected_features)

                # Copy values for columns that exist in both DataFrames
                common_features = set(X.columns) & set(expected_features)
                self.log_info(f"Found {len(common_features)} common features out of {len(expected_features)} expected")

                for feature in common_features:
                    aligned_X[feature] = X[feature].values

                # Use the aligned DataFrame for prediction
                X_for_prediction = aligned_X
            else:
                # No expected features found, use original data
                self.log_info("No expected feature list found, using original features")
                X_for_prediction = X

            # Make prediction with the appropriate model
            if isinstance(self.rf_model, dict) and 'base_regressor' in self.rf_model:
                self.log_info("Using base_regressor from dictionary for prediction")
                preds = self.rf_model['base_regressor'].predict(X_for_prediction)
            elif hasattr(self.rf_model, 'predict') and callable(getattr(self.rf_model, 'predict')):
                self.log_info("Using model's predict method")
                preds = self.rf_model.predict(X_for_prediction)
            else:
                self.log_error("No valid prediction method found in model")
                return np.zeros(len(X))

            self.log_info(f"Generated predictions for {len(X)} samples")
            return preds

        except Exception as e:
            self.log_error(f"Error generating RF predictions: {str(e)}")
            import traceback
            self.log_error(traceback.format_exc())
            return np.zeros(len(X))

    def predict_with_lstm(self, X_seq: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """
        Make predictions using the LSTM model.

        Args:
            X_seq: Sequence features
            X_static: Static features

        Returns:
            NumPy array with predictions
        """
        if self.lstm_model is None:
            self.log_error("LSTM model not loaded")
            return np.zeros(len(X_seq))

        if X_seq is None or X_static is None:
            self.log_error("Sequence data not available")
            return np.zeros(len(X_seq) if X_seq is not None else 0)

        try:
            preds = self.lstm_model.predict([X_seq, X_static], verbose=0)

            # Flatten predictions if needed
            if len(preds.shape) > 1:
                preds = preds.flatten()

            self.log_info(f"Generated LSTM predictions for {len(preds)} samples")
            return preds
        except Exception as e:
            self.log_error(f"Error generating LSTM predictions: {str(e)}")
            return np.zeros(len(X_seq))

    def predict(self, X: pd.DataFrame, X_seq: Optional[np.ndarray] = None,
                X_static: Optional[np.ndarray] = None, blend_weight: float = 0.7) -> np.ndarray:
        """
        Make predictions using available models, optionally blending results.

        Args:
            X: Feature DataFrame for RF model
            X_seq: Sequence features for LSTM model (optional)
            X_static: Static features for LSTM model (optional)
            blend_weight: Weight for RF model in blended predictions (0-1)

        Returns:
            NumPy array with predictions
        """
        # Get RF predictions
        rf_preds = self.predict_with_rf(X)

        # Get LSTM predictions if possible
        lstm_preds = None
        if self.lstm_model is not None and X_seq is not None and X_static is not None:
            lstm_preds = self.predict_with_lstm(X_seq, X_static)

            # Make sure shapes match
            if len(lstm_preds) != len(rf_preds):
                self.log_error(f"Shape mismatch: RF {len(rf_preds)}, LSTM {len(lstm_preds)}")
                lstm_preds = None

        # Blend predictions if both models are available
        if lstm_preds is not None:
            self.log_info(f"Blending predictions with weight {blend_weight} for RF")
            final_preds = rf_preds * blend_weight + lstm_preds * (1 - blend_weight)
        else:
            final_preds = rf_preds

        return final_preds

    def predict_race(self, race_df: pd.DataFrame, blend_weight: float = 0.7) -> pd.DataFrame:
        """
        Predict race outcome with arrival string format.

        Args:
            race_df: DataFrame with race data
            blend_weight: Weight for RF model in blended predictions (0-1)

        Returns:
            DataFrame with predictions
        """
        # Prepare data for prediction
        X, X_seq, X_static = self.prepare_race_data(race_df)

        # Make predictions
        predictions = self.predict(X, X_seq, X_static, blend_weight)

        # Add predictions to original data
        result_df = race_df.copy()
        result_df['predicted_position'] = predictions

        # Sort by predicted position (ascending, better positions first)
        result_df = result_df.sort_values('predicted_position')

        # Add rank column
        result_df['predicted_rank'] = range(1, len(result_df) + 1)

        # Create arrival string format (numero-numero-numero)
        # This follows the same format as 'arriv' in actual results
        numeros_ordered = result_df['numero'].astype(str).tolist()
        predicted_arriv = '-'.join(numeros_ordered)

        # Add the predicted_arriv as a column to each row
        result_df['predicted_arriv'] = predicted_arriv

        return result_df