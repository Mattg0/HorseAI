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

    def prepare_race_data(self, race_df: pd.DataFrame) -> Tuple[
        pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare race data for prediction by applying feature engineering and embeddings.
        """
        # Prepare RF features
        X = self._prepare_rf_features(race_df)  # Extract current RF preparation logic to this method

        # Prepare LSTM features if the model is available
        X_seq, X_static = None, None
        if self.lstm_model is not None:
            X_seq, X_static = self.prepare_lstm_race_data(race_df)

        return X, X_seq, X_static
    def _prepare_rf_features(self, race_df: pd.DataFrame) -> Tuple[
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
        self.log_info(f"X shape: {X.shape if X is not None else 'None'}")
        self.log_info(f"X_seq shape: {X_seq.shape if X_seq is not None else 'None'}")
        self.log_info(f"X_static shape: {X_static.shape if X_static is not None else 'None'}")
        self.log_info(f"LSTM model available: {self.lstm_model is not None}")
        return X, X_seq, X_static

    def fetch_horse_sequences(self, race_df, sequence_length=None):
        """
        Fetch historical race sequences for all horses in a race to enable LSTM prediction.

        Args:
            race_df: DataFrame with the current race data (containing horses to predict)
            sequence_length: Length of sequence to retrieve (uses default if None)

        Returns:
            Tuple of (X_seq, X_static, horse_ids) for sequence prediction
        """
        if sequence_length is None:
            sequence_length = self.feature_config.get('sequence_length', 5) if self.feature_config else 5

        self.log_info(f"Fetching historical sequences of length {sequence_length} for horses in race")

        # Get all horse IDs from the race
        horse_ids = []
        if 'idche' in race_df.columns:
            # Filter out missing or invalid IDs
            horse_ids = [int(h) for h in race_df['idche'] if pd.notna(h)]

        if not horse_ids:
            self.log_error("No valid horse IDs found in race data")
            return None, None, None

        self.log_info(f"Found {len(horse_ids)} horses to fetch sequences for")

        # Connect to the database
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # This enables column access by name
            cursor = conn.cursor()

            # Define sequence and static features (match training configuration)
            # These should match what was used in training
            sequential_features = [
                'final_position', 'cotedirect', 'dist',
                # Include embeddings if available
                'horse_emb_0', 'horse_emb_1', 'horse_emb_2',
                'jockey_emb_0', 'jockey_emb_1', 'jockey_emb_2',
                # Add musique-derived features
                'che_global_avg_pos', 'che_global_recent_perf', 'che_global_consistency', 'che_global_pct_top3',
                'che_weighted_avg_pos', 'che_weighted_recent_perf', 'che_weighted_consistency', 'che_weighted_pct_top3'
            ]

            static_features = [
                'age', 'temperature', 'natpis', 'typec', 'meteo', 'corde',
                'couple_emb_0', 'couple_emb_1', 'couple_emb_2',
                'course_emb_0', 'course_emb_1', 'course_emb_2'
            ]

            # Prepare containers for sequences
            all_sequences = []
            all_static_features = []
            all_horse_ids = []

            # For each horse, retrieve its historical races
            for horse_id in horse_ids:
                self.log_info(f"Processing historical data for horse {horse_id}")

                # Fetch historical races for this horse
                # Note: SQL query filters races before the current race date
                # to prevent data leakage

                # First get the current race date to use as a cutoff
                current_race_date = None
                if 'jour' in race_df.columns:
                    # Try to get the date from the current race data
                    current_race_date = race_df['jour'].iloc[0] if len(race_df) > 0 else None

                # Determine SQL date filter based on current race date
                date_filter = ""
                if current_race_date:
                    # Convert to proper date format if needed
                    if isinstance(current_race_date, str):
                        date_filter = f" AND hr.jour < '{current_race_date}'"
                    else:
                        # Try to format as a date string
                        try:
                            date_str = current_race_date.strftime('%Y-%m-%d')
                            date_filter = f" AND hr.jour < '{date_str}'"
                        except:
                            # If formatting fails, don't use a date filter
                            pass

                # Create query to find races with this horse
                query = f"""
                SELECT hr.* 
                FROM historical_races hr
                WHERE hr.participants LIKE ?
                {date_filter}
                ORDER BY hr.jour DESC
                LIMIT 20
                """

                # Execute query
                cursor.execute(query, (f'%"idche": {horse_id}%',))
                horse_races = cursor.fetchall()

                if not horse_races:
                    self.log_info(f"No historical races found for horse {horse_id}")
                    continue

                self.log_info(f"Found {len(horse_races)} historical races for horse {horse_id}")

                # Extract and process participant data for this horse
                horse_data = []
                for race in horse_races:
                    try:
                        # Parse participant JSON
                        participants = json.loads(race['participants'])

                        # Find this horse in the participants
                        horse_entry = next((p for p in participants if int(p.get('idche', 0)) == horse_id), None)

                        if horse_entry:
                            # Add race attributes to horse entry
                            for key in ['jour', 'hippo', 'dist', 'typec', 'temperature', 'natpis', 'meteo', 'corde']:
                                if key in race:
                                    horse_entry[key] = race[key]

                            # If there's a race result, try to get the final position for this horse
                            if 'ordre_arrivee' in race and race['ordre_arrivee']:
                                try:
                                    results = json.loads(race['ordre_arrivee'])
                                    # Find this horse's position
                                    horse_result = next((r for r in results if int(r.get('cheval', 0)) == horse_id),
                                                        None)
                                    if horse_result:
                                        horse_entry['final_position'] = horse_result.get('narrivee')
                                except:
                                    # If we can't parse results, skip
                                    pass

                            horse_data.append(horse_entry)
                    except:
                        # If we can't parse participants, skip this race
                        continue

                # Convert to DataFrame for easier processing
                if not horse_data:
                    self.log_info(f"No historical data could be extracted for horse {horse_id}")
                    continue

                horse_df = pd.DataFrame(horse_data)

                # Sort by date
                if 'jour' in horse_df.columns:
                    horse_df['jour'] = pd.to_datetime(horse_df['jour'], errors='coerce')
                    horse_df = horse_df.sort_values('jour', ascending=False)

                # Apply feature engineering to get embeddings
                try:
                    # Use orchestrator to process features
                    processed_df = self.orchestrator.prepare_features(horse_df)
                    embedded_df = self.orchestrator.apply_embeddings(processed_df, clean_after_embedding=False)

                    # Check if we have enough races for a sequence
                    if len(embedded_df) >= sequence_length:
                        # Extract sequential features (only keep those that exist in the DataFrame)
                        seq_features = [f for f in sequential_features if f in embedded_df.columns]

                        if not seq_features:
                            self.log_info(f"No sequential features found for horse {horse_id}")
                            continue

                        # Get sequence data (take first sequence_length races)
                        seq_data = embedded_df[seq_features].head(sequence_length).values.astype(np.float32)

                        # Make sure we have the right sequence length
                        if len(seq_data) < sequence_length:
                            # Pad with zeros if needed
                            padding = np.zeros((sequence_length - len(seq_data), len(seq_features)), dtype=np.float32)
                            seq_data = np.vstack([seq_data, padding])

                        # Get static features from current race for this horse
                        current_horse = race_df[race_df['idche'] == horse_id]

                        if len(current_horse) > 0:
                            # Start with empty array for static features
                            static_data = np.zeros(len(static_features), dtype=np.float32)

                            # Fill in available static features from current race
                            for i, feature in enumerate(static_features):
                                if feature in current_horse.columns:
                                    try:
                                        val = current_horse[feature].iloc[0]
                                        if pd.notna(val):
                                            static_data[i] = float(val)
                                    except:
                                        pass

                            # Add to output containers
                            all_sequences.append(seq_data)
                            all_static_features.append(static_data)
                            all_horse_ids.append(horse_id)

                            self.log_info(f"Successfully created sequence for horse {horse_id}")
                        else:
                            self.log_info(f"Horse {horse_id} not found in current race data")
                    else:
                        self.log_info(
                            f"Not enough historical races for horse {horse_id} (found {len(embedded_df)}, need {sequence_length})")
                except Exception as e:
                    self.log_error(f"Error processing horse {horse_id}: {str(e)}")
                    import traceback
                    self.log_error(traceback.format_exc())

            conn.close()

            # Convert to numpy arrays
            if not all_sequences:
                self.log_info(f"No valid sequences could be created for any horse")
                return None, None, None

            X_sequences = np.array(all_sequences, dtype=np.float32)
            X_static = np.array(all_static_features, dtype=np.float32)
            sequence_horse_ids = np.array(all_horse_ids)

            self.log_info(f"Created {len(X_sequences)} sequences for {len(np.unique(sequence_horse_ids))} horses")
            self.log_info(f"Sequence shape: {X_sequences.shape}, Static shape: {X_static.shape}")

            return X_sequences, X_static, sequence_horse_ids

        except Exception as e:
            self.log_error(f"Error in fetch_horse_sequences: {str(e)}")
            import traceback
            self.log_error(traceback.format_exc())
            return None, None, None

        except Exception as e:
            self.log_error(f"Error in fetch_horse_sequences: {str(e)}")
            import traceback
            self.log_error(traceback.format_exc())
            return None, None, None

    def prepare_lstm_race_data(self, race_df: pd.DataFrame) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare race data specifically for LSTM prediction.
        Enhanced to fetch historical sequences for horses.

        Args:
            race_df: DataFrame with race and participant data

        Returns:
            Tuple of (sequence features, static features, horse_ids)
        """
        # Make a copy to avoid modifying the original
        df = race_df.copy()

        # Apply basic feature engineering but preserve 'jour' and 'idche'
        df = self.orchestrator.apply_embeddings(df, clean_after_embedding=True, keep_identifiers=True)

        # Try preparing sequence data using the traditional method first (for backward compatibility)
        try:
            # Get sequence length from model config or use default
            sequence_length = self.feature_config.get('sequence_length', 5) if self.feature_config else 5

            self.log_info(f"Trying standard sequence preparation with length {sequence_length}...")
            X_seq, X_static, _ = self.orchestrator.prepare_sequence_data(
                df, sequence_length=sequence_length
            )
            self.log_info(
                f"Successfully prepared LSTM sequence data with shape {X_seq.shape} and static data with shape {X_static.shape}")

            # This succeeded, so return the results without horse IDs (since the standard method doesn't provide them)
            return X_seq, X_static, None

        except Exception as e:
            self.log_info(
                f"Standard sequence preparation failed: {str(e)} - Attempting to fetch historical horse sequences")

        # If the standard method failed, try our new approach that fetches historical data
        try:
            # Get sequence length from model config or use default
            sequence_length = self.feature_config.get('sequence_length', 5) if self.feature_config else 5

            self.log_info(f"Fetching historical horse sequences with length {sequence_length}...")
            X_seq, X_static, horse_ids = self.fetch_horse_sequences(df, sequence_length=sequence_length)

            if X_seq is not None and X_static is not None:
                self.log_info(
                    f"Successfully fetched LSTM sequence data with shape {X_seq.shape} and static data with shape {X_static.shape}")
                return X_seq, X_static, horse_ids
            else:
                self.log_error("Could not create sequences from historical data")
                return None, None, None

        except Exception as e:
            self.log_error(f"Error preparing LSTM sequence data: {str(e)}")
            import traceback
            self.log_error(traceback.format_exc())
            return None, None, None
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
        Enhanced to use historical horse sequences when possible.

        Args:
            race_df: DataFrame with race data
            blend_weight: Weight for RF model in blended predictions (0-1)

        Returns:
            DataFrame with predictions
        """
        # Prepare data for prediction
        X, X_seq, X_static = self._prepare_rf_features(race_df)

        # Prepare LSTM data if model is available
        horse_ids = None
        if self.lstm_model is not None:
            try:
                # Try to get LSTM data with historical sequences
                lstm_X_seq, lstm_X_static, horse_ids = self.prepare_lstm_race_data(race_df)

                # If we got valid LSTM data, use it
                if lstm_X_seq is not None and lstm_X_static is not None:
                    X_seq = lstm_X_seq
                    X_static = lstm_X_static
                    self.log_info(f"Using LSTM data: {X_seq.shape}, {X_static.shape}")
            except Exception as e:
                self.log_error(f"Error preparing LSTM data: {str(e)}")

        # Make predictions
        predictions = self.predict(X, X_seq, X_static, horse_ids, blend_weight)

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