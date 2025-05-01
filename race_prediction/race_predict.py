import os
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple
import logging
from datetime import datetime
import sqlite3

# Import from existing code
from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from model_training.regressions.isotonic_calibration import CalibratedRegressor
from core.calculators.static_feature_calculator import FeatureCalculator

class RacePredictor:
    """
    Race predictor that loads trained models and applies them to new race data.
    Handles both static models (Random Forest) and sequence models (LSTM) if available.
    """

    def __init__(self, model_path: str = None, db_name: str = None,
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

        # Set up logging with proper verbose control
        self._setup_logging()

        # Load models and configuration
        self._load_models()

        if self.verbose:
            self.log_info(f"Race predictor initialized with model at {self.model_path}")
            self.log_info(f"Using database: {self.db_path}")

    def _setup_logging(self):
        """Set up logging with proper verbose control."""
        # Create logs directory if it doesn't exist
        log_dir = self.model_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Get or create logger
        self.logger = logging.getLogger("RacePredictor")

        # Remove any existing handlers to avoid duplicates
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)

        # Set level based on verbose flag
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Add file handler (always log to file)
        log_file = log_dir / f"race_predictor_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        # Add console handler only if verbose is True
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)

    def log_info(self, message):
        """Log an info message."""
        if hasattr(self, 'logger'):
            self.logger.info(message)
        elif self.verbose:  # Only print if verbose when logger not initialized
            print(message)

    def log_error(self, message):
        """Log an error message."""
        if hasattr(self, 'logger'):
            self.logger.error(message)
        else:  # Always print errors even without logger
            print(f"ERROR: {message}")
    def _load_models(self):
        # Get model manager
        from utils.model_manager import get_model_manager
        model_manager = get_model_manager()

        # Load models
        artifacts = model_manager.load_model_artifacts(
            base_path=self.model_path,
            load_rf=True,
            load_lstm=True,
            load_feature_config=True
        )

        # Set model attributes from loaded artifacts
        if 'rf_model' in artifacts:
            self.rf_model = artifacts['rf_model']
            self.log_info(f"Loaded RF model")
        else:
            self.rf_model = None
            self.log_info("RF model not available")

        if 'lstm_model' in artifacts:
            self.lstm_model = artifacts['lstm_model']
            self.log_info(f"Loaded LSTM model")
        else:
            self.lstm_model = None
            self.log_info("LSTM model not available")

        if 'feature_config' in artifacts:
            self.feature_config = artifacts['feature_config']
            self.log_info(f"Loaded feature configuration")
        else:
            self.feature_config = {}

        if 'model_config' in artifacts:
            self.model_config = artifacts['model_config']
            self.log_info(f"Loaded model configuration")

    def prepare_rf_features(self, race_df: pd.DataFrame) -> Tuple[
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

        df = FeatureCalculator.calculate_all_features(df)
        df = self.orchestrator.apply_embeddings(df, clean_after_embedding=False)

        # Clean up features for RF model
        X = self.orchestrator.drop_embedded_raw_features(df)
        self.log_info(f"Prepared {len(X)} samples with {X.shape[1]} features for RF model")

        return X

    def prepare_lstm_race_data(self, race_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare race data specifically for LSTM prediction.

        Args:
            race_df: DataFrame with race and participant data

        Returns:
            Tuple of (sequence features, static features)
        """
        # Use sequence_length from the model
        sequence_length = 5  # From model inspection

        # Define the sequence and static features the model expects
        seq_features = ['cotedirect', 'horse_emb_0', 'horse_emb_1', 'horse_emb_2',
                        'jockey_emb_0', 'jockey_emb_1', 'jockey_emb_2']

        static_features = ['couple_emb_0', 'couple_emb_1', 'couple_emb_2',
                           'course_emb_0', 'course_emb_1', 'course_emb_2']

        # Get embedded data
        embedded_df = self.orchestrator.apply_embeddings(
            race_df.copy(),
            clean_after_embedding=True,
            keep_identifiers=True
        )

        # Ensure all required features exist
        for col in seq_features + static_features:
            if col not in embedded_df.columns:
                embedded_df[col] = 0.0

        # Create tensors with the correct shape
        n_samples = len(embedded_df)

        # Initialize arrays
        X_seq = np.zeros((n_samples, sequence_length, len(seq_features)), dtype=np.float32)
        X_static = np.zeros((n_samples, len(static_features)), dtype=np.float32)

        # Fill static features from current race data
        for i, (_, row) in enumerate(embedded_df.iterrows()):
            for j, feat in enumerate(static_features):
                if pd.notna(row[feat]):
                    X_static[i, j] = row[feat]

        # For sequence features, use fetch_historical_data if implemented,
        # otherwise use current data repeated
        historical_data = self.fetch_historical_data(embedded_df, sequence_length)

        if historical_data is not None:
            # Use historical data for sequences
            X_seq = historical_data
        else:
            # Use current data repeated for each timestep
            for i, (_, row) in enumerate(embedded_df.iterrows()):
                for t in range(sequence_length):
                    for j, feat in enumerate(seq_features):
                        if pd.notna(row[feat]):
                            X_seq[i, t, j] = row[feat]

        return X_seq, X_static

    def fetch_historical_data(self, race_df, sequence_length):
        """
        Fetch historical race data for each horse to create proper sequences.

        Args:
            race_df: DataFrame with current race data
            sequence_length: Length of sequences to create

        Returns:
            Array of shape (n_samples, sequence_length, n_features) or None if not enough data
        """
        # Check if we have horse IDs
        if 'idche' not in race_df.columns:
            return None

        # Sequence features we need to extract
        seq_features = ['cotedirect', 'horse_emb_0', 'horse_emb_1', 'horse_emb_2',
                        'jockey_emb_0', 'jockey_emb_1', 'jockey_emb_2']

        # Initialize result array
        n_samples = len(race_df)
        X_seq = np.zeros((n_samples, sequence_length, len(seq_features)), dtype=np.float32)

        # Connect to database
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Current race date for filtering
        current_date = None
        if 'jour' in race_df.columns:
            current_date = race_df['jour'].iloc[0]

        # Track how many horses we process successfully
        success_count = 0

        # Process each horse
        for i, (_, row) in enumerate(race_df.iterrows()):
            if 'idche' not in row or pd.isna(row['idche']):
                continue

            horse_id = int(row['idche'])

            # Build query to fetch historical races for this horse
            query = """
            SELECT hr.comp, hr.jour, hr.participants 
            FROM historical_races hr
            WHERE hr.participants LIKE ?
            """

            # Add date filter if available
            params = [f'%"idche": {horse_id}%']
            if current_date and isinstance(current_date, str):
                query += " AND hr.jour < ?"
                params.append(current_date)

            # Order by date (newest first) and limit to get enough for sequence
            query += " ORDER BY hr.jour DESC LIMIT ?"
            params.append(sequence_length * 2)  # Get extra in case some can't be processed

            # Execute query
            cursor.execute(query, params)
            races = cursor.fetchall()

            # Skip if not enough races
            if not races or len(races) < sequence_length:
                continue

            # Process historical races
            seq_data = []

            for race in races:
                # Unpack row data (access by index)
                comp = race[0]
                jour = race[1]
                participants_json = race[2]

                try:
                    # Parse participants JSON
                    participants = json.loads(participants_json)

                    # Find this horse in participants
                    horse_entry = None
                    for p in participants:
                        if p.get('idche') == horse_id or str(p.get('idche')) == str(horse_id):
                            horse_entry = p
                            break

                    if horse_entry:
                        # Extract sequence features
                        features = []
                        for feat in seq_features:
                            if feat in horse_entry and pd.notna(horse_entry[feat]):
                                features.append(float(horse_entry[feat]))
                            else:
                                features.append(0.0)

                        seq_data.append(features)
                except:
                    # Skip this race if any error occurs
                    continue

            # Check if we have enough sequence data
            if len(seq_data) >= sequence_length:
                # Take the first sequence_length entries
                for t in range(sequence_length):
                    if t < len(seq_data):
                        X_seq[i, t] = seq_data[t]

                success_count += 1

        conn.close()

        # Return sequences if we have enough successful horses
        if success_count >= n_samples // 2:  # At least half
            return X_seq
        else:
            return None
    def predict_with_rf(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the Random Forest model with feature name alignment.
        """
        if self.rf_model is None:
            self.log_error("Random Forest model not loaded")
            return np.zeros(len(X))

        try:
            # Check if we have feature names from training
            expected_features = None

            # Try to get expected feature names from different places
            if hasattr(self.feature_config,
                       'preprocessing_params') and 'feature_columns' in self.feature_config.preprocessing_params:
                expected_features = self.feature_config.preprocessing_params['feature_columns']
                self.log_info(f"Found {len(expected_features)} expected feature columns from config")
            elif hasattr(self.rf_model, 'feature_names_in_'):
                expected_features = self.rf_model.feature_names_in_
                self.log_info(f"Found {len(expected_features)} feature names from model")
            elif hasattr(self.rf_model, 'base_regressor') and hasattr(self.rf_model.base_regressor,
                                                                      'feature_names_in_'):
                expected_features = self.rf_model.base_regressor.feature_names_in_
                self.log_info(f"Found {len(expected_features)} feature names from base_regressor")

            # If we have expected features, align the input data
            if expected_features is not None:
                # Create a DataFrame with expected feature columns filled with zeros
                aligned_X = pd.DataFrame(0, index=range(len(X)), columns=expected_features)

                # Copy values for columns that exist in both DataFrames
                common_features = set(X.columns) & set(expected_features)
                self.log_info(f"Found {len(common_features)} common features out of {len(expected_features)} expected")

                for feature in common_features:
                    aligned_X[feature] = X[feature].values

                # Use the aligned DataFrame for prediction
                X_for_prediction = aligned_X
            else:
                # No expected features found, use original data (will likely fail)
                self.log_info("No expected feature list found, using original features")
                X_for_prediction = X

            # Make prediction with appropriate model
            preds = self.rf_model.predict(X_for_prediction)
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
                X_static: Optional[np.ndarray] = None, horse_ids: Optional[np.ndarray] = None,
                blend_weight: float = 0.7) -> np.ndarray:
        """
        Make predictions using available models, optionally blending results.
        Enhanced to handle mapping LSTM predictions to the correct horses.

        Args:
            X: Feature DataFrame for RF model
            X_seq: Sequence features for LSTM model (optional)
            X_static: Static features for LSTM model (optional)
            horse_ids: Horse IDs corresponding to LSTM sequences (optional)
            blend_weight: Weight for RF model in blended predictions (0-1)

        Returns:
            NumPy array with predictions
        """
        # If LSTM model is None, no blending will occur
        if self.lstm_model is None:
            self.log_info("LSTM model is None, no blending will occur")

        # If sequence data is None, no blending will occur
        if X_seq is None or X_static is None:
            self.log_info("Sequence data is None, no blending will occur")

        # Get RF predictions
        rf_preds = self.predict_with_rf(X)

        # Initialize final predictions with RF predictions
        final_preds = rf_preds.copy()

        # Get LSTM predictions if possible
        lstm_preds = None
        if self.lstm_model is not None and X_seq is not None and X_static is not None:
            lstm_preds = self.predict_with_lstm(X_seq, X_static)
            self.log_info(f"LSTM predictions generated: {lstm_preds}")

        # Handle mapping LSTM predictions to horses when we have horse IDs
        if lstm_preds is not None and horse_ids is not None:
            try:
                # Create mapping from horse IDs to indices in X DataFrame
                horse_id_to_idx = {}

                # Try to extract horse IDs from X DataFrame
                if 'idche' in X.columns:
                    for i, horse_id in enumerate(X['idche']):
                        try:
                            horse_id_to_idx[int(horse_id)] = i
                        except (ValueError, TypeError):
                            # Skip if horse_id can't be converted to int
                            pass

                # Map LSTM predictions to the correct horses and blend with RF predictions
                for i, horse_id in enumerate(horse_ids):
                    if horse_id in horse_id_to_idx:
                        idx = horse_id_to_idx[horse_id]
                        # Blend RF and LSTM predictions
                        final_preds[idx] = rf_preds[idx] * blend_weight + lstm_preds[i] * (1 - blend_weight)
                        self.log_info(
                            f"Blending for horse {horse_id}: RF={rf_preds[idx]:.2f}, LSTM={lstm_preds[i]:.2f}, Final={final_preds[idx]:.2f}")

                # Log how many horses were blended
                blended_count = sum(1 for horse_id in horse_ids if horse_id in horse_id_to_idx)
                self.log_info(
                    f"Blended predictions for {blended_count}/{len(rf_preds)} horses using weight {blend_weight}")

            except Exception as e:
                self.log_error(f"Error mapping LSTM predictions to horses: {str(e)}")
                import traceback
                self.log_error(traceback.format_exc())

        # Direct blending when we don't have horse IDs but sequence shapes match
        elif lstm_preds is not None and len(lstm_preds) == len(rf_preds):
            self.log_info(f"Direct blending of predictions with weight {blend_weight}")
            final_preds = rf_preds * blend_weight + lstm_preds * (1 - blend_weight)

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

        # Step 1: Prepare and predict with Random Forest model
        rf_features = self.prepare_rf_features(race_df)
        rf_predictions = self.predict_with_rf(rf_features)

        # Step 2: Prepare and predict with LSTM model (if available)
        lstm_predictions = None
        if self.lstm_model is not None:
            try:
                X_seq, X_static = self.prepare_lstm_race_data(race_df)
                if X_seq is not None and X_static is not None:
                    lstm_predictions = self.predict_with_lstm(X_seq, X_static)
                    if lstm_predictions is not None:
                        # Create a simple mapping to log
                        lstm_pred_map = {}
                        for i, row in enumerate(race_df.itertuples()):
                            if hasattr(row, 'idche') and i < len(lstm_predictions):
                                lstm_pred_map[int(row.idche)] = float(lstm_predictions[i])

                        # Log the mapping
                        self.log_info(f"LSTM predictions by horse: {lstm_pred_map}")
            except Exception as e:
                self.log_error(f"Error in LSTM prediction: {str(e)}")


        # Step 3: Blend predictions if both models produced results
        final_predictions = rf_predictions
        if lstm_predictions is not None and len(lstm_predictions) == len(rf_predictions):
            self.log_info(f"Blending predictions with weight {blend_weight} for RF")
            final_predictions = rf_predictions * blend_weight + lstm_predictions * (1 - blend_weight)

        # Step 4: Add predictions to result DataFrame and format
        result_df = race_df.copy()
        result_df['predicted_position'] = final_predictions

        # Sort by predicted position (ascending, better positions first)
        result_df = result_df.sort_values('predicted_position')

        # Add rank column
        result_df['predicted_rank'] = range(1, len(result_df) + 1)

        # Create arrival string format (numero-numero-numero)
        numeros_ordered = result_df['numero'].astype(str).tolist()
        predicted_arriv = '-'.join(numeros_ordered)

        # Add the predicted_arriv as a column to each row
        result_df['predicted_arriv'] = predicted_arriv

        return result_df
