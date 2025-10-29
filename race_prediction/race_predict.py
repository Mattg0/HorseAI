import os
import numpy as np
import pandas as pd
import json
import joblib
import time
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple
import logging
from datetime import datetime

# Import the optimized orchestrator and utilities
from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from utils.model_manager import get_model_manager
from core.calculators.static_feature_calculator import FeatureCalculator
from sklearn.preprocessing import StandardScaler
from .enhanced_prediction_blender import EnhancedPredictionBlender

# TabNet imports
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

# Alternative models imports
try:
    from model_training.models import TransformerModel, EnsembleModel
    ALTERNATIVE_MODELS_AVAILABLE = True
except ImportError:
    ALTERNATIVE_MODELS_AVAILABLE = False

# Prediction storage imports
from .simple_prediction_storage import SimplePredictionStorage, extract_prediction_data_from_competitive_analysis


class RacePredictor:
    """
    Enhanced race predictor that supports RF and TabNet models with intelligent blending.
    Uses the same data preparation pipeline as training for consistency.
    """

    def __init__(self, model_path: str = None, db_name: str = None, verbose: bool = False,
                 enable_prediction_storage: bool = True, enable_feature_export: bool = False,
                 feature_export_min_races: int = 5):
        """
        Initialize the optimized race predictor.

        Args:
            model_path: Path to the trained model (if None, uses latest from config)
            db_name: Database name from config (defaults to active_db)
            verbose: Whether to print verbose output
            enable_prediction_storage: Whether to enable comprehensive prediction storage
            enable_feature_export: Whether to export X features for analysis
            feature_export_min_races: Minimum races to accumulate before exporting (default: 5)
        """
        # Initialize config
        self.config = AppConfig()
        self.verbose = verbose

        # Feature export settings
        self.enable_feature_export = enable_feature_export
        self.feature_export_min_races = feature_export_min_races
        self.feature_buffer = []  # Buffer to accumulate features from multiple races

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

        # Load model configuration and determine model paths
        self.model_manager = get_model_manager()
        if model_path is None:
            # Get all model paths from config
            self.model_paths = self.model_manager.get_all_model_paths()
            # Use RF path as primary for backward compatibility
            rf_path = self.model_paths.get('rf')
            if rf_path:
                self.model_path = Path(rf_path)
            else:
                raise ValueError("No RF model path found - RF model is required")
        else:
            self.model_path = Path(model_path)
            # Get all model paths first, then override RF path if specified
            self.model_paths = self.model_manager.get_all_model_paths()
            self.model_paths['rf'] = self.model_path

        # Initialize enhanced prediction blender with database path for historical data
        self.blender = EnhancedPredictionBlender(verbose=self.verbose, db_path=str(self.db_path))

        # Initialize prediction storage if enabled
        self.enable_prediction_storage = enable_prediction_storage
        if enable_prediction_storage:
            # Use the same database as race data for easy joins
            self.simple_storage = SimplePredictionStorage(db_path=str(self.db_path), verbose=self.verbose)
        else:
            self.simple_storage = None

        # Load models and configuration
        self._load_models()
        self._load_alternative_models()

        # Legacy blend weights (for backward compatibility)
        self._load_blend_weights()

        if self.verbose:
            print(f"RacePredictor initialized")
            print(f"  Model: {self.model_path}")
            print(f"  Database: {self.db_path}")
            print(f"  Model weights: RF={self.rf_weight:.1f}, TabNet={self.tabnet_weight:.1f}")
            print(f"  Legacy models loaded: RF={self.rf_model is not None}, TabNet={self.tabnet_model is not None}")
            print(f"  Prediction storage: {'Enabled' if enable_prediction_storage else 'Disabled'}")
            if hasattr(self, 'alternative_models') and self.alternative_models:
                alt_loaded = [name for name, model in self.alternative_models.items() if model is not None]
                print(f"  Alternative models loaded: {alt_loaded if alt_loaded else 'None'}")

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
                # Get blend weights from model config or use default
                training_results = self.model_config.get('training_results', {})
                # Check if we have RF and TabNet weights
                if 'rf_weight' in training_results:
                    self.rf_weight = training_results.get('rf_weight', 0.7)
                    self.tabnet_weight = training_results.get('tabnet_weight', 0.3)
                    # LSTM weight ignored (no longer used)
                    self.lstm_weight = 0.0
                else:
                    # Legacy: use RF for old blend_weight
                    old_blend_weight = training_results.get('blend_weight', 0.7)
                    self.rf_weight = old_blend_weight
                    self.tabnet_weight = 1.0 - old_blend_weight
                    self.lstm_weight = 0.0
        else:
            self.model_config = {}
            # Use default blend weights if not in config (RF + TabNet only)
            self.rf_weight = 0.7
            self.tabnet_weight = 0.3
            self.lstm_weight = 0.0

        # Load feature engineering state to match training (no hybrid prefix)
        feature_config_path = self.model_path / "feature_engineer.joblib"
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

        # Load RF model (REQUIRED) - no hybrid prefix
        rf_model_path = self.model_path / "rf_model.joblib"
        if rf_model_path.exists():
            self.rf_model = joblib.load(rf_model_path)
            if self.verbose:
                print(f"Loaded RF model: {type(self.rf_model)}")
        else:
            self.rf_model = None
            raise FileNotFoundError(f"RF model not found at {rf_model_path} - required for predictions")

        # LSTM model removed - using RF + TabNet only
        self.lstm_model = None
                
        # Load TabNet model
        self._load_tabnet_model()

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
            'temperature', 'forceVent', 'idche', 'idJockey', 'numero', 'gainsCarriere'
        ]

        # CRITICAL FIX: Log values before/after conversion to debug data loss
        for field in numeric_fields:
            if field in race_df.columns:
                # DEBUG: Log original values for critical fields
                if field in ['cotedirect', 'coteprob', 'gainsCarriere'] and self.verbose:
                    original_values = race_df[field].head(5).tolist()
                    print(f"  DEBUG {field} BEFORE conversion: {original_values}")

                # Convert to numeric, coercing errors to NaN
                race_df[field] = pd.to_numeric(race_df[field], errors='coerce').fillna(0)

                # DEBUG: Log converted values
                if field in ['cotedirect', 'coteprob', 'gainsCarriere'] and self.verbose:
                    converted_values = race_df[field].head(5).tolist()
                    print(f"  DEBUG {field} AFTER conversion: {converted_values}")

        # Ensure target column is NOT present during prediction
        if 'final_position' in race_df.columns:
            race_df = race_df.drop('final_position', axis=1)
            if self.verbose:
                print("  ⚠️  Removed 'final_position' column from prediction data")

        # Step 3: Apply FeatureCalculator WITH TEMPORAL MODE (matching training pipeline exactly)
        # Training uses: FeatureCalculator.calculate_all_features(df_historical, use_temporal=True)
        # CRITICAL: use_temporal=True prevents data leakage in career statistics
        if self.verbose:
            print("  🔧 Applying FeatureCalculator with temporal calculations (no leakage)...")

        prep_start = time.time()
        # CRITICAL FIX: Use temporal calculations to prevent data leakage
        df_with_features = FeatureCalculator.calculate_all_features(
            race_df,
            use_temporal=True,  # SAME AS TRAINING - prevents data leakage
            db_path=self.db_path
        )
        prep_time = time.time() - prep_start

        if self.verbose:
            print(f"  ✅ FeatureCalculator applied (temporal mode): {prep_time:.2f}s")

        # Step 3.5: Apply feature cleanup (same as training)
        if self.verbose:
            print("  🔧 Applying feature cleanup (removing leaking features)...")

        from core.data_cleaning.feature_cleanup import FeatureCleaner
        cleaner = FeatureCleaner()

        # Clean features (removes leaking career stats)
        df_with_features = cleaner.clean_features(df_with_features)

        # Apply transformations (log transforms)
        df_with_features = cleaner.apply_transformations(df_with_features)

        if self.verbose:
            print(f"  ✅ Feature cleanup applied: {len(df_with_features.columns)} features")

        # Step 4: Use orchestrator's TabNet preparation (matching training pipeline exactly)
        # Training uses: orchestrator.prepare_tabnet_features(df_with_features)
        if self.verbose:
            print("  🔧 Preparing TabNet features (same as training)...")

        tabnet_start = time.time()
        embedded_df = self.orchestrator.prepare_tabnet_features(
            df_with_features,
            use_cache=False  # Don't cache predictions
        )
        tabnet_time = time.time() - tabnet_start

        if self.verbose:
            print(f"  ✅ TabNet features prepared: {tabnet_time:.2f}s")
            print(f"Data preparation complete: {len(embedded_df.columns)} features")

        return embedded_df

    def predict_with_rf(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the RF model."""
        if X is None or X.empty:
            if self.verbose:
                print("Warning: RF input data is None or empty")
            return None

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
                # Clean empty strings and non-numeric values before assignment
                feature_values = X[feature].replace(['', None, 'NULL'], 0).fillna(0)
                try:
                    aligned_X[feature] = pd.to_numeric(feature_values, errors='coerce').fillna(0)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not convert feature {feature} to numeric, using 0s: {e}")
                    aligned_X[feature] = 0

            X_for_prediction = aligned_X

            if self.verbose:
                print(f"RF prediction: {len(common_features)}/{len(expected_features)} features aligned")
        else:
            X_for_prediction = X

        predictions = self.rf_model.predict(X_for_prediction)

        if self.verbose:
            print(f"RF prediction: {len(predictions)} predictions generated")

        return predictions

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

    def _load_blend_weights(self):
        """Load optimal blend weights from config."""
        try:
            blend_config = self.config._config.blend
            self.rf_weight = blend_config.get('rf_weight', 0.0)
            self.lstm_weight = blend_config.get('lstm_weight', 0.0)  # Default to 0 if not present
            self.tabnet_weight = blend_config.get('tabnet_weight', 1.0)

            # Validate weights sum to 1
            total_weight = self.rf_weight + self.lstm_weight + self.tabnet_weight
            if abs(total_weight - 1.0) > 1e-6:
                if self.verbose:
                    print(f"Warning: Blend weights don't sum to 1.0: {total_weight}, normalizing...")
                # Normalize weights
                self.rf_weight = self.rf_weight / total_weight
                self.lstm_weight = self.lstm_weight / total_weight
                self.tabnet_weight = self.tabnet_weight / total_weight

        except (AttributeError, KeyError) as e:
            # Use default optimal weights if not in config
            if self.verbose:
                print(f"Error loading blend weights ({e}), using default: 0/0/1 (TabNet only)")
            self.rf_weight = 0.0
            self.lstm_weight = 0.0
            self.tabnet_weight = 1.0

    def get_optimal_weights(self, race_metadata: Dict) -> Tuple[float, float]:
        """
        Dynamic model weighting based on race characteristics.

        Args:
            race_metadata: Dictionary with race characteristics including:
                - typec: Race type (Monté, Attelé, etc.)
                - field_size or partant: Number of participants
                - dist or distance: Race distance in meters

        Returns:
            Tuple of (rf_weight, tabnet_weight)
        """
        try:
            blend_config = self.config._config.blend

            # Check if dynamic weights are enabled
            if not blend_config.get('use_dynamic_weights', False):
                if self.verbose:
                    print("Dynamic weights disabled, using static weights")
                return (self.rf_weight, self.tabnet_weight)

            # Get race characteristics
            typec = race_metadata.get('typec', '')
            field_size = race_metadata.get('field_size', race_metadata.get('partant', 0))
            distance = race_metadata.get('dist', race_metadata.get('distance', 0))

            if self.verbose:
                print(f"Evaluating dynamic weights for: typec={typec}, field_size={field_size}, distance={distance}")

            # Get dynamic weight rules from config
            dynamic_weights = blend_config.get('dynamic_weights', [])

            # Evaluate each rule in order (first match wins)
            for rule in dynamic_weights:
                if not isinstance(rule, dict) or 'condition' not in rule:
                    continue

                condition = rule['condition']
                matched = True

                # Check typec condition
                if 'typec' in condition:
                    if typec != condition['typec']:
                        matched = False

                # Check field size conditions
                if matched and 'partant_min' in condition:
                    if field_size < condition['partant_min']:
                        matched = False

                if matched and 'partant_max' in condition:
                    if field_size > condition['partant_max']:
                        matched = False

                # Check distance conditions
                if matched and 'dist_min' in condition:
                    if distance < condition['dist_min']:
                        matched = False

                if matched and 'dist_max' in condition:
                    if distance > condition['dist_max']:
                        matched = False

                # If all conditions matched, return the weights
                if matched:
                    weights = rule.get('weights', {})
                    rf_w = weights.get('rf_weight', 0.5)
                    tabnet_w = weights.get('tabnet_weight', 0.5)

                    if self.verbose:
                        description = rule.get('description', 'No description')
                        accuracy = rule.get('accuracy', 'N/A')
                        print(f"✅ Matched rule: {description} (accuracy: {accuracy}%)")
                        print(f"   Weights: RF={rf_w:.1f}, TabNet={tabnet_w:.1f}")

                    return (rf_w, tabnet_w)

            # No rule matched, use default weights
            default_weights = blend_config.get('default_weights', None)
            if default_weights and isinstance(default_weights, dict):
                rf_w = default_weights.get('rf_weight', 0.0)
                tabnet_w = default_weights.get('tabnet_weight', 1.0)

                if self.verbose:
                    description = default_weights.get('description', 'Default weights')
                    accuracy = default_weights.get('accuracy', 'N/A')
                    print(f"Using default weights: {description} (accuracy: {accuracy}%)")
                    print(f"   Weights: RF={rf_w:.1f}, TabNet={tabnet_w:.1f}")

                return (rf_w, tabnet_w)

            # Fallback to static weights
            if self.verbose:
                print(f"No dynamic rules matched, using static weights: RF={self.rf_weight:.1f}, TabNet={self.tabnet_weight:.1f}")
            return (self.rf_weight, self.tabnet_weight)

        except Exception as e:
            if self.verbose:
                print(f"Error in get_optimal_weights: {e}, falling back to static weights")
            return (self.rf_weight, self.tabnet_weight)

    def _load_tabnet_model(self):
        """Load TabNet model and associated files."""
        self.tabnet_model = None
        self.tabnet_scaler = None
        self.tabnet_feature_columns = None

        if not TABNET_AVAILABLE:
            if self.verbose:
                print("TabNet not available - install pytorch-tabnet")
            return

        try:
            # Get TabNet model path from model_paths
            tabnet_path = None
            if hasattr(self, 'model_paths') and 'tabnet' in self.model_paths:
                tabnet_path = Path(self.model_paths['tabnet'])
            else:
                # Fallback to looking in RF model path
                tabnet_path = self.model_path

            if tabnet_path is None:
                if self.verbose:
                    print("No TabNet model path found")
                return

            # Look for TabNet model files
            tabnet_model_file = tabnet_path / "tabnet_model.zip"
            if not tabnet_model_file.exists():
                tabnet_model_file = tabnet_path / "tabnet_model.zip.zip"

            if tabnet_model_file.exists():
                self.tabnet_model = TabNetRegressor()
                self.tabnet_model.load_model(str(tabnet_model_file))

                # Load scaler
                scaler_file = tabnet_path / "tabnet_scaler.joblib"
                if scaler_file.exists():
                    self.tabnet_scaler = joblib.load(scaler_file)

                # Load feature configuration
                config_file = tabnet_path / "tabnet_config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        self.tabnet_feature_columns = config.get('feature_columns', [])

                if self.verbose:
                    print(f"Loaded TabNet model from: {tabnet_model_file}")
                    if self.tabnet_feature_columns:
                        print(f"TabNet expects {len(self.tabnet_feature_columns)} features")
            else:
                if self.verbose:
                    print(f"TabNet model not found at: {tabnet_path}")

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load TabNet model: {str(e)}")
            self.tabnet_model = None
            self.tabnet_scaler = None
            self.tabnet_feature_columns = None
    
    def _load_alternative_models(self):
        """Load alternative models (TabNet only for 2-model system)."""
        self.alternative_models = {}
        
        if not ALTERNATIVE_MODELS_AVAILABLE:
            if self.verbose:
                print("Alternative models not available (import failed)")
            return
        
        try:
            # Get alternative models configuration
            alt_config = getattr(self.config._config, 'alternative_models', {})
            enabled_models = alt_config.get('model_selection', [])
            models_dir = Path("models")
            
            if not models_dir.exists():
                if self.verbose:
                    print("Models directory not found - alternative models not loaded")
                return
            
            # Load Transformer model
            if 'transformer' in enabled_models:
                transformer_files = list(models_dir.glob("*transformer*.h5")) + list(models_dir.glob("*transformer*.keras"))
                if transformer_files:
                    try:
                        config = alt_config.get('transformer', {})
                        model = TransformerModel(config, verbose=False)
                        model.model = model.load_model(str(transformer_files[0]))
                        self.alternative_models['transformer'] = model
                        if self.verbose:
                            print(f"Loaded transformer model from {transformer_files[0]}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Failed to load transformer model: {e}")
                        self.alternative_models['transformer'] = None
                else:
                    self.alternative_models['transformer'] = None
            
            # Load Ensemble model
            if 'ensemble' in enabled_models:
                ensemble_files = list(models_dir.glob("*ensemble*.pkl")) + list(models_dir.glob("*ensemble*.joblib"))
                if ensemble_files:
                    try:
                        config = alt_config.get('ensemble', {})
                        model = EnsembleModel(config, verbose=False)
                        model.load_ensemble(str(ensemble_files[0]))
                        self.alternative_models['ensemble'] = model
                        if self.verbose:
                            print(f"Loaded ensemble model from {ensemble_files[0]}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Failed to load ensemble model: {e}")
                        self.alternative_models['ensemble'] = None
                else:
                    self.alternative_models['ensemble'] = None
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading alternative models: {e}")
            self.alternative_models = {}

    def prepare_tabnet_features(self, race_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features specifically for TabNet prediction.
        FIXED: Now uses saved feature columns from training for exact alignment.

        Training pipeline:
        1. FeatureCalculator.calculate_all_features(df_historical)
        2. orchestrator.prepare_tabnet_features(df_with_features)
        3. Select features matching saved tabnet_feature_columns
        """
        if self.verbose:
            print("Preparing TabNet features using saved feature alignment...")

        # Ensure target column is NOT present during prediction
        race_df_clean = race_df.copy()
        if 'final_position' in race_df_clean.columns:
            race_df_clean = race_df_clean.drop('final_position', axis=1)
            if self.verbose:
                print("  ⚠️  Removed 'final_position' column from prediction data")

        # Step 1: Apply FeatureCalculator FIRST (matching training exactly)
        prep_start = time.time()
        df_with_features = FeatureCalculator.calculate_all_features(race_df_clean)
        calc_time = time.time() - prep_start

        if self.verbose:
            print(f"  ✅ FeatureCalculator applied: {calc_time:.2f}s")

        # Step 2: Use orchestrator's TabNet preparation (matching training exactly)
        tabnet_start = time.time()
        complete_df = self.orchestrator.prepare_tabnet_features(
            df_with_features,
            use_cache=False  # Don't cache predictions
        )
        tabnet_time = time.time() - tabnet_start

        if self.verbose:
            print(f"  ✅ TabNet features prepared: {tabnet_time:.2f}s")

        # Step 3: CRITICAL FIX - Use saved feature columns instead of extract_tabnet_features
        # The extract_tabnet_features method calls prepare_training_dataset which excludes columns
        # But we need to match EXACTLY what was saved during training
        if self.tabnet_feature_columns:
            # Filter to only features that exist in complete_df
            available_features = [f for f in self.tabnet_feature_columns if f in complete_df.columns]
            missing_features = [f for f in self.tabnet_feature_columns if f not in complete_df.columns]

            if missing_features and self.verbose:
                print(f"  ⚠️  Missing {len(missing_features)} features from training:")
                for feat in missing_features[:10]:  # Show first 10
                    print(f"     - {feat}")

            # Create aligned DataFrame with exact training features
            X_tabnet = complete_df[available_features].copy()

            if self.verbose:
                print(f"  ✅ TabNet features aligned: {X_tabnet.shape[1]} features from saved config")
                print(f"     Match ratio: {len(available_features)}/{len(self.tabnet_feature_columns)} ({len(available_features)/len(self.tabnet_feature_columns)*100:.1f}%)")
        else:
            # Fallback: use feature_selector (less reliable)
            if self.verbose:
                print("  ⚠️  No saved feature columns - using feature_selector (may mismatch)")

            tabnet_features = self.orchestrator.feature_selector.get_model_features('tabnet', complete_df)
            available_features = [f for f in tabnet_features if f in complete_df.columns]
            X_tabnet = complete_df[available_features].copy()

        return X_tabnet, complete_df
    
    def predict_with_tabnet(self, race_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using TabNet model with domain features.
        NEW: Uses same domain feature extraction as training.
        """
        if self.tabnet_model is None:
            if self.verbose:
                print("TabNet model not available")
            return None
            
        try:
            # Prepare TabNet-specific features using new pipeline
            X_tabnet, complete_df = self.prepare_tabnet_features(race_df)
            if X_tabnet is None or X_tabnet.empty:
                if self.verbose:
                    print("Warning: No TabNet features prepared")
                return None
            
            # Validate feature consistency with training
            if self.tabnet_feature_columns:
                available_features = [col for col in self.tabnet_feature_columns if col in X_tabnet.columns]
                missing_features = [col for col in self.tabnet_feature_columns if col not in X_tabnet.columns]
                
                if missing_features and self.verbose:
                    print(f"Warning: Missing TabNet features: {len(missing_features)}/{len(self.tabnet_feature_columns)}")

                if not available_features:
                    if self.verbose:
                        print("Warning: No matching TabNet features found")
                    return None

                # Check if we have enough features (at least 70% match)
                feature_match_ratio = len(available_features) / len(self.tabnet_feature_columns)

                if feature_match_ratio < 0.7:  # Less than 70% feature match
                    if self.verbose:
                        print(f"Warning: Insufficient feature match ({feature_match_ratio:.2f} < 0.70)")
                    return None
                    
                # Create aligned DataFrame with all training features
                # Fill missing features with zeros to maintain exact feature order from training
                aligned_X = pd.DataFrame(0.0, index=range(len(X_tabnet)), columns=self.tabnet_feature_columns)

                # DEBUG: Log critical features before alignment
                if self.verbose:
                    for feat in ['cotedirect', 'coteprob', 'gainsAnneeEnCours']:
                        if feat in X_tabnet.columns:
                            print(f"  DEBUG: X_tabnet[{feat}] before align: {X_tabnet[feat].head(3).tolist()}")
                        if feat in available_features:
                            print(f"  DEBUG: {feat} IN available_features ✅")
                        else:
                            print(f"  DEBUG: {feat} NOT in available_features ❌")

                # Fill available features with actual values
                for feature in available_features:
                    aligned_X[feature] = X_tabnet[feature]

                # DEBUG: Log critical features after alignment
                if self.verbose:
                    for feat in ['cotedirect', 'coteprob', 'gainsAnneeEnCours']:
                        if feat in aligned_X.columns:
                            print(f"  DEBUG: aligned_X[{feat}] after align: {aligned_X[feat].head(3).tolist()}")

                if self.verbose:
                    print(f"TabNet features aligned: {aligned_X.shape}")
                X_df = aligned_X
                X = X_df.values
            else:
                # Use all available features - keep as DataFrame for scaler
                X_df = X_tabnet
                X = X_df.values

            # Scale features using training scaler
            if self.tabnet_scaler is not None:
                X_scaled = self.tabnet_scaler.transform(X_df)
                if self.verbose:
                    print(f"TabNet features scaled using training scaler")
            else:
                X_scaled = X
                if self.verbose:
                    print("Warning: No TabNet scaler available")
                
            # Generate predictions with diagnostics
            tabnet_preds = self.tabnet_model.predict(X_scaled)

            # CRITICAL DEBUGGING: Log raw prediction values AND features
            if self.verbose:
                print(f"🔍 TABNET PREDICTION DIAGNOSTICS:")
                print(f"   Input features shape: {X_scaled.shape}")
                print(f"   Input features range: {X_scaled.min():.3f} to {X_scaled.max():.3f}")

                # Find which features have extreme values
                max_vals_per_feature = np.max(np.abs(X_scaled), axis=0)
                extreme_feature_idx = np.where(max_vals_per_feature > 1000)[0]
                if len(extreme_feature_idx) > 0:
                    print(f"   🚨 EXTREME FEATURE VALUES FOUND:")
                    for idx in extreme_feature_idx:
                        feature_name = self.tabnet_feature_columns[idx] if self.tabnet_feature_columns and idx < len(self.tabnet_feature_columns) else f"feature_{idx}"
                        print(f"      Feature {idx} ({feature_name}): max_abs = {max_vals_per_feature[idx]:.0f}")
                        print(f"      Values: {X_scaled[:, idx]}")

                print(f"   Raw predictions shape: {tabnet_preds.shape}")
                print(f"   Raw predictions range: {tabnet_preds.min():.3f} to {tabnet_preds.max():.3f}")
                print(f"   Raw predictions sample: {tabnet_preds.flatten()[:5]}")

                # Check for extreme values
                extreme_count = np.sum(np.abs(tabnet_preds) > 100)
                if extreme_count > 0:
                    print(f"   ⚠️  WARNING: {extreme_count} predictions have absolute value > 100")
                    print(f"   ⚠️  Extreme values: {tabnet_preds[np.abs(tabnet_preds) > 100]}")

            if len(tabnet_preds.shape) > 1:
                tabnet_preds = tabnet_preds.flatten()

            if self.verbose:
                print(f"   Final predictions after flatten: {tabnet_preds[:5]}")
                print(f"TabNet prediction: {len(tabnet_preds)} predictions generated")

            return tabnet_preds
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: TabNet prediction failed: {str(e)}")
            return None
    
    def _predict_with_alternative_models(self, race_df: pd.DataFrame, embedded_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions using alternative models.
        
        Args:
            race_df: Original race DataFrame
            embedded_df: DataFrame with embedded features
            
        Returns:
            Dictionary of predictions from alternative models
        """
        predictions = {}
        
        if not hasattr(self, 'alternative_models') or not self.alternative_models:
            return predictions
        
        # Prepare data for alternative models (similar to LSTM format)
        # This is a simplified version - in production you'd want proper feature engineering
        try:
            n_samples = len(race_df)
            n_features = 20  # Should match training
            n_static = 10
            sequence_length = 5
            
            # Generate placeholder features that match alternative model expectations
            # In production, replace with proper feature engineering from race_df
            X_sequences = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
            X_static = np.random.randn(n_samples, n_static).astype(np.float32)
            
            # Generate predictions for each available alternative model
            for model_name, model in self.alternative_models.items():
                if model is not None:
                    try:
                        model_predictions = model.predict(X_sequences, X_static)
                        predictions[model_name] = model_predictions
                        if self.verbose:
                            print(f"Generated {model_name} predictions: shape {model_predictions.shape}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Error in {model_name} prediction: {e}")
                        predictions[model_name] = None
                else:
                    predictions[model_name] = None
                    
        except Exception as e:
            if self.verbose:
                print(f"Error preparing alternative model predictions: {e}")
        
        return predictions

    def _capture_features_for_export(self, race_df: pd.DataFrame, X_tabnet: pd.DataFrame,
                                     race_metadata: dict):
        """
        Capture TabNet features from current race for later export.

        Args:
            race_df: Original race data
            X_tabnet: TabNet feature matrix (unscaled)
            race_metadata: Race metadata (comp, hippo, etc.)
        """
        if not self.enable_feature_export:
            return

        race_id = race_metadata.get('comp', 'unknown')

        # Prepare feature record with metadata (TabNet features only)
        feature_record = {
            'race_id': race_id,
            'hippo': race_metadata.get('hippo', ''),
            'date': race_metadata.get('date', ''),
            'partants': len(race_df),
            'X_features': X_tabnet.to_dict('records') if X_tabnet is not None else [],
            'feature_names': list(X_tabnet.columns) if X_tabnet is not None else [],
            'feature_count': len(X_tabnet.columns) if X_tabnet is not None else 0
        }

        self.feature_buffer.append(feature_record)

        if self.verbose:
            feature_count = feature_record['feature_count']
            print(f"📊 Captured {feature_count} TabNet features for race {race_id} ({len(self.feature_buffer)}/{self.feature_export_min_races})")

        # Export if we've reached the minimum
        if len(self.feature_buffer) >= self.feature_export_min_races:
            self._export_features()

    def _export_features(self):
        """Export accumulated features to JSON file."""
        if not self.feature_buffer:
            return

        try:
            output_file = 'X_general_features.json'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            export_data = {
                'export_timestamp': timestamp,
                'total_races': len(self.feature_buffer),
                'total_horses': sum(record['partants'] for record in self.feature_buffer),
                'races': self.feature_buffer
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            if self.verbose:
                print(f"✅ Exported {len(self.feature_buffer)} races features to {output_file}")

            # Clear buffer after export
            self.feature_buffer = []

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error exporting features: {e}")

    def get_feature_buffer_status(self) -> dict:
        """Get current status of feature buffer."""
        return {
            'enabled': self.enable_feature_export,
            'buffered_races': len(self.feature_buffer),
            'min_races': self.feature_export_min_races,
            'ready_to_export': len(self.feature_buffer) >= self.feature_export_min_races
        }

    def force_export_features(self):
        """Force export of features even if minimum not reached."""
        if self.feature_buffer:
            self._export_features()

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict race outcome using the optimized hybrid model.

        Args:
            race_df: DataFrame with race and participant data

        Returns:
            DataFrame with predictions and rankings
        """
        start_time = time.time()

        if race_df is None:
            raise ValueError("race_df cannot be None")

        if self.verbose:
            print(f"🏇 Predicting race with {len(race_df)} participants")

        # Step 1: Prepare data using the same pipeline as training
        step_start = time.time()
        embedded_df = self.prepare_race_data(race_df)
        step_time = time.time() - step_start
        if self.verbose:
            print(f"⏱️  Step 1 (Data Preparation): {step_time:.2f}s")

        if embedded_df is None:
            raise ValueError("prepare_race_data returned None")

        # Step 2: Extract RF features and generate RF predictions
        step_start = time.time()
        X_rf, _ = self.orchestrator.extract_rf_features(embedded_df)
        feature_time = time.time() - step_start

        pred_start = time.time()
        rf_predictions = self.predict_with_rf(X_rf)
        rf_pred_time = time.time() - pred_start

        if self.verbose:
            print(f"⏱️  Step 2a (RF Feature Extraction): {feature_time:.2f}s")
            print(f"⏱️  Step 2b (RF Prediction): {rf_pred_time:.2f}s")

        # Step 3: LSTM predictions (removed - using RF + TabNet only)
        step_start = time.time()
        lstm_predictions = None
        lstm_time = time.time() - step_start
        if self.verbose:
            print(f"⏱️  Step 3 (LSTM Prediction): {lstm_time:.2f}s")

        # Step 4: Generate TabNet predictions
        step_start = time.time()
        tabnet_predictions = self.predict_with_tabnet(race_df)
        tabnet_time = time.time() - step_start
        if self.verbose:
            print(f"⏱️  Step 4 (TabNet Prediction): {tabnet_time:.2f}s")
        
        # Step 5: Generate alternative model predictions
        step_start = time.time()
        alternative_predictions = self._predict_with_alternative_models(race_df, embedded_df)
        alt_time = time.time() - step_start
        if self.verbose:
            print(f"⏱️  Step 5 (Alternative Models): {alt_time:.2f}s")
        
        # Step 6: Collect all predictions for blending (RF and TabNet only)
        all_predictions = {
            'rf': rf_predictions,
            'tabnet': tabnet_predictions
        }

        # Add alternative model predictions
        all_predictions.update(alternative_predictions)

        # Debug: Print prediction status (only show available models)
        if self.verbose:
            for model, preds in all_predictions.items():
                if preds is not None:
                    print(f"✅ {model}: {type(preds)} shape {preds.shape}")
                # Don't show None models to avoid confusion

        # Step 6.5: Capture features for export (if enabled)
        race_metadata = {
            'distance': race_df.get('dist', [0]).iloc[0] if 'dist' in race_df.columns else 1600,
            'dist': race_df.get('dist', [0]).iloc[0] if 'dist' in race_df.columns else 1600,
            'typec': race_df.get('typec', ['P']).iloc[0] if 'typec' in race_df.columns else 'P',
            'field_size': len(race_df),
            'hippo': race_df.get('hippo', ['']).iloc[0] if 'hippo' in race_df.columns else '',
            'comp': race_df.get('comp', ['']).iloc[0] if 'comp' in race_df.columns else '',
            'date': race_df.get('jour', ['']).iloc[0] if 'jour' in race_df.columns else ''
        }

        # Capture features for analysis (accumulates 5+ races before export)
        if self.enable_feature_export:
            # Get TabNet features before scaling (TabNet is the primary model)
            X_tabnet_unscaled, _ = self.prepare_tabnet_features(race_df)
            self._capture_features_for_export(race_df, X_tabnet_unscaled, race_metadata)

        # Step 7: Prepare race metadata and apply dynamic weights

        # Apply dynamic weights based on race characteristics
        if self.verbose:
            print(f"\n=== DYNAMIC WEIGHTING ===")

        rf_weight, tabnet_weight = self.get_optimal_weights(race_metadata)

        # Update blender's default weights for this race
        self.blender.default_weights['rf'] = rf_weight
        self.blender.default_weights['tabnet'] = tabnet_weight
        self.blender.default_weights['lstm'] = 0.0  # LSTM not used

        if self.verbose:
            print(f"Dynamic weights applied: RF={rf_weight:.2f}, TabNet={tabnet_weight:.2f}")
            print(f"=== END DYNAMIC WEIGHTING ===\n")

        # Debug: Show race data structure before competitive analysis
        if self.verbose:
            print(f"\n=== RACE DATA DEBUG BEFORE COMPETITIVE ANALYSIS ===")
            print(f"race_df shape: {race_df.shape}")
            print(f"race_df columns ({len(race_df.columns)}): {list(race_df.columns)}")

            # Check critical competitive columns
            critical_cols = ['recordG', 'hippo', 'coursescheval', 'victoirescheval', 'placescheval', 'gainsCarriere', 'age']
            print(f"\n=== CRITICAL COLUMNS CHECK ===")
            for col in critical_cols:
                if col in race_df.columns:
                    non_null = race_df[col].notna().sum()
                    print(f"✅ {col}: {non_null}/{len(race_df)} non-null ({non_null/len(race_df)*100:.1f}%)")
                    if non_null > 0:
                        sample = race_df[col].dropna().head(2).tolist()
                        print(f"   Sample: {sample}")
                else:
                    print(f"❌ {col}: NOT FOUND")

            # Show first few rows with key columns
            key_cols = ['numero', 'cheval'] + [col for col in critical_cols if col in race_df.columns]
            available_key_cols = [col for col in key_cols if col in race_df.columns]
            if available_key_cols:
                print(f"\n=== SAMPLE RACE DATA (first 3 horses) ===")
                sample_data = race_df[available_key_cols].head(3)
                for i, (idx, row) in enumerate(sample_data.iterrows()):
                    print(f"Horse {i+1}: {dict(row)}")

            print(f"=== END RACE DATA DEBUG ===\n")

        # Step 8: Use competitive analysis enhanced blending
        step_start = time.time()
        final_predictions, blend_info = self.blender.blend_with_competitive_analysis(
            predictions=all_predictions,
            race_data=race_df,  # Use original race data for competitive analysis
            race_metadata=race_metadata
        )
        blend_time = time.time() - step_start
        if self.verbose:
            print(f"⏱️  Step 8 (Competitive Blending): {blend_time:.2f}s")

        # Extract competitive analysis results for simple storage
        competitive_results = {}
        if blend_info.get('competitive_analysis_applied', False):
            # Extract the actual competitive analysis data from blend_info
            # Now competitive_scores are available directly in blend_info
            competitive_results = {
                'competitive_analysis': {
                    'competitive_scores': blend_info.get('competitive_scores', {}),
                    'field_statistics': blend_info.get('field_statistics', {}),
                    'top_contenders': blend_info.get('top_contenders', [])
                },
                'audit_trail': {
                    'adjustment_summary': blend_info.get('competitive_summary', {})
                }
            }
        
        if self.verbose:
            print(f"Enhanced blending used {blend_info['total_models']} models: {blend_info['models_used']}")
            print(f"Applied weights: {blend_info['weights_applied']}")
            print(f"Blend method: {blend_info['blend_method']}")

            # Log competitive analysis results if applied
            if blend_info.get('competitive_analysis_applied', False):
                field_stats = blend_info.get('field_statistics', {})
                avg_score = field_stats.get('average_competitive_score', 0)
                competitiveness = field_stats.get('field_competitiveness', 'unknown')
                print(f"Competitive analysis applied - Field competitiveness: {competitiveness}, Avg score: {avg_score:.3f}")

                top_contenders = blend_info.get('top_contenders', [])
                if top_contenders:
                    print(f"Top contenders: {[c['horse_index'] for c in top_contenders[:3]]}")

                performance_expectations = blend_info.get('performance_expectations', {})
                for model, expectations in performance_expectations.items():
                    r2_improvement = expectations.get('estimated_r2_improvement', 0)
                    print(f"{model} expected R² improvement: +{r2_improvement:.3f}")
            else:
                print("Competitive analysis: Not applied (insufficient race data)")

        # Extract base predictions from all_predictions (needed for simple storage)
        base_predictions = {k: v for k, v in all_predictions.items() if v is not None}

        # Step 9: Store prediction data for competitive analysis
        if self.verbose:
            print(f"DEBUG: simple_storage is {'not None' if self.simple_storage is not None else 'None'}")

        if self.simple_storage is not None:
            try:
                if self.verbose:
                    print(f"DEBUG: Attempting to store prediction data...")
                # Use comp as race_id to match daily_race table
                race_id = race_metadata.get('comp')

                # Extract prediction data using helper function
                if self.verbose:
                    print(f"DEBUG: competitive_results keys: {list(competitive_results.keys())}")
                    print(f"DEBUG: blend_info keys: {list(blend_info.keys())}")

                predictions_data = extract_prediction_data_from_competitive_analysis(
                    race_id=race_id,
                    race_data=race_df,
                    base_predictions=base_predictions,
                    competitive_results=competitive_results,
                    final_predictions=final_predictions,
                    blend_weights=blend_info.get('weights_applied', {})
                )

                if self.verbose:
                    print(f"DEBUG: predictions_data length: {len(predictions_data)}")
                    if predictions_data:
                        print(f"DEBUG: First prediction data: {predictions_data[0]}")

                # Store simple prediction data
                stored_count = self.simple_storage.store_race_predictions(race_id, predictions_data)

                if self.verbose:
                    print(f"Simple prediction storage: {stored_count} horses stored for race {race_id}")

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to store simple prediction data: {e}")

        # Step 10: Create result DataFrame
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

    def re_blend_existing_predictions_with_dynamic_weights(self, race_id: Optional[str] = None,
                                                           date: Optional[str] = None,
                                                           all_races: bool = True,
                                                           verbose: bool = None) -> Dict[str, Any]:
        """
        Re-blend existing predictions with dynamic weights without re-predicting.
        This is much faster than re-running predictions from scratch.

        Args:
            race_id: Specific race_id to re-blend. If provided, only that race is processed.
            date: Specific date (YYYY-MM-DD) to re-blend. Ignored if race_id or all_races=True.
            all_races: If True, re-blends ALL races with predictions (default). Overrides date.
            verbose: Override instance verbosity

        Returns:
            Dictionary with summary of re-blending results
        """
        if verbose is None:
            verbose = self.verbose

        if verbose:
            print(f"🔄 Re-blending predictions with dynamic weights...")

        # Connect to database
        import sqlite3
        conn = sqlite3.connect(self.db_path)

        try:
            # Get races to re-blend
            if race_id:
                # Specific race only (now includes quinté races)
                race_query = """
                    SELECT DISTINCT comp, typec, partant, dist
                    FROM daily_race
                    WHERE comp = ?
                """
                races = pd.read_sql_query(race_query, conn, params=(race_id,))
            elif all_races:
                # All races that have predictions (now includes quinté races)
                race_query = """
                    SELECT DISTINCT dr.comp, dr.typec, dr.partant, dr.dist
                    FROM daily_race dr
                    JOIN race_predictions rp ON dr.comp = rp.race_id
                """
                races = pd.read_sql_query(race_query, conn)
            elif date:
                # Specific date (now includes quinté races)
                race_query = """
                    SELECT DISTINCT comp, typec, partant, dist
                    FROM daily_race
                    WHERE jour = ?
                """
                races = pd.read_sql_query(race_query, conn, params=(date,))
            else:
                # Default to all races with predictions (now includes quinté races)
                race_query = """
                    SELECT DISTINCT dr.comp, dr.typec, dr.partant, dr.dist
                    FROM daily_race dr
                    JOIN race_predictions rp ON dr.comp = rp.race_id
                """
                races = pd.read_sql_query(race_query, conn)

            if len(races) == 0:
                if verbose:
                    print("No races found to re-blend")
                return {'races_processed': 0, 'horses_updated': 0}

            if verbose:
                print(f"Found {len(races)} races to re-blend")

            # Process each race
            total_updated = 0
            races_updated = []

            for idx, race_row in races.iterrows():
                race_comp = race_row['comp']

                # Get race metadata for dynamic weights
                race_metadata = {
                    'typec': race_row['typec'],
                    'field_size': race_row['partant'],
                    'partant': race_row['partant'],
                    'dist': race_row['dist'],
                    'distance': race_row['dist']
                }

                # Get optimal weights for this race
                rf_weight, tabnet_weight = self.get_optimal_weights(race_metadata)
                use_dynamic = self.config._config.blend.get('use_dynamic_weights', False)

                if verbose:
                    print(f"\nRace {race_comp}: typec={race_row['typec']}, partant={race_row['partant']}, dist={race_row['dist']}")
                    print(f"  Dynamic weights: RF={rf_weight:.2f}, TabNet={tabnet_weight:.2f}")

                # Get existing predictions for this race (now includes quinté)
                pred_query = """
                    SELECT id, horse_id, rf_prediction, tabnet_prediction,
                           ensemble_weight_rf, ensemble_weight_tabnet
                    FROM race_predictions
                    WHERE race_id = ?
                """
                predictions = pd.read_sql_query(pred_query, conn, params=(race_comp,))

                if len(predictions) == 0:
                    if verbose:
                        print(f"  No predictions found for race {race_comp}, skipping")
                    continue

                # Re-blend predictions with new weights
                updates = []
                for _, pred_row in predictions.iterrows():
                    rf_pred = pred_row['rf_prediction']
                    tabnet_pred = pred_row['tabnet_prediction']

                    # Skip if either prediction is missing
                    if pd.isna(rf_pred) or pd.isna(tabnet_pred):
                        continue

                    # Calculate new ensemble prediction
                    new_ensemble_pred = rf_pred * rf_weight + tabnet_pred * tabnet_weight

                    # Store update info
                    updates.append({
                        'id': pred_row['id'],
                        'ensemble_weight_rf': rf_weight,
                        'ensemble_weight_tabnet': tabnet_weight,
                        'ensemble_prediction': new_ensemble_pred,
                        'final_prediction': new_ensemble_pred  # Update final too
                    })

                # Apply updates to race_predictions
                cursor = conn.cursor()
                for update in updates:
                    cursor.execute("""
                        UPDATE race_predictions
                        SET ensemble_weight_rf = ?,
                            ensemble_weight_tabnet = ?,
                            ensemble_prediction = ?,
                            final_prediction = ?
                        WHERE id = ?
                    """, (
                        update['ensemble_weight_rf'],
                        update['ensemble_weight_tabnet'],
                        update['ensemble_prediction'],
                        update['final_prediction'],
                        update['id']
                    ))

                # Update prediction_results in daily_race using participants mapping
                import json

                # Get updated predictions ordered by final_prediction (now includes quinté)
                cursor.execute("""
                    SELECT
                        json_extract(p.value, '$.numero') as numero,
                        rp.final_prediction
                    FROM daily_race dr,
                         json_each(dr.participants) p
                    LEFT JOIN race_predictions rp ON rp.race_id = dr.comp
                        AND rp.horse_id = json_extract(p.value, '$.idche')
                    WHERE dr.comp = ?
                    ORDER BY rp.final_prediction ASC
                """, (race_comp,))

                # Build new predicted_arriv from numeros sorted by final_prediction
                numeros_ordered = [str(row[0]) for row in cursor.fetchall()]
                new_predicted_arriv = '-'.join(numeros_ordered)

                # Update the prediction_results JSON
                cursor.execute("SELECT prediction_results FROM daily_race WHERE comp = ?", (race_comp,))
                result = cursor.fetchone()

                if result and result[0]:
                    try:
                        pred_json = json.loads(result[0])

                        # Update predicted_arriv
                        pred_json['predicted_arriv'] = new_predicted_arriv

                        # Update metadata
                        if 'metadata' not in pred_json:
                            pred_json['metadata'] = {}
                        pred_json['metadata']['blend_weight_rf'] = rf_weight
                        pred_json['metadata']['blend_weight_tabnet'] = tabnet_weight
                        pred_json['metadata']['reblend_method'] = 'dynamic_weights' if use_dynamic else 'static_weights'

                        # Update predictions array with new positions and ranks
                        cursor.execute("""
                            SELECT
                                json_extract(p.value, '$.idche') as idche,
                                json_extract(p.value, '$.numero') as numero,
                                rp.final_prediction
                            FROM daily_race dr,
                                 json_each(dr.participants) p
                            LEFT JOIN race_predictions rp ON rp.race_id = dr.comp
                                AND rp.horse_id = json_extract(p.value, '$.idche')
                            WHERE dr.comp = ?
                            ORDER BY rp.final_prediction ASC
                        """, (race_comp,))

                        # Create mapping of idche -> (predicted_position, predicted_rank)
                        updated_data = {}
                        for rank, row in enumerate(cursor.fetchall(), 1):
                            idche, numero, final_pred = row
                            updated_data[idche] = {
                                'predicted_position': final_pred,
                                'predicted_rank': rank
                            }

                        # Update each prediction in the array
                        for pred in pred_json.get('predictions', []):
                            idche = pred.get('idche')
                            if idche and idche in updated_data:
                                pred['predicted_position'] = updated_data[idche]['predicted_position']
                                pred['predicted_rank'] = updated_data[idche]['predicted_rank']

                        # Save updated JSON
                        cursor.execute("""
                            UPDATE daily_race
                            SET prediction_results = ?
                            WHERE comp = ?
                        """, (json.dumps(pred_json), race_comp))

                        if verbose:
                            print(f"  Updated predicted_arriv: {new_predicted_arriv}")

                    except json.JSONDecodeError:
                        if verbose:
                            print(f"  Warning: Could not parse prediction_results for {race_comp}")

                conn.commit()
                total_updated += len(updates)
                races_updated.append({
                    'race_id': race_comp,
                    'horses_updated': len(updates),
                    'rf_weight': rf_weight,
                    'tabnet_weight': tabnet_weight
                })

                if verbose:
                    print(f"  Updated {len(updates)} horses")

            if verbose:
                print(f"\n✅ Re-blending complete!")
                print(f"   Races processed: {len(races_updated)}")
                print(f"   Total horses updated: {total_updated}")

            return {
                'races_processed': len(races_updated),
                'horses_updated': total_updated,
                'races_detail': races_updated
            }

        finally:
            conn.close()

    def get_model_info(self) -> Dict:
        """
        Get information about all loaded models including alternative models.
        
        Returns:
            Dictionary with comprehensive model information
        """
        # Alternative models info
        alt_models_info = {}
        if hasattr(self, 'alternative_models') and self.alternative_models:
            alt_models_info = {
                name: model is not None 
                for name, model in self.alternative_models.items()
            }
        
        return {
            'model_type': 'Enhanced RacePredictor (Legacy + Alternative Models)',
            'model_path': str(self.model_path),
            'legacy_models': {
                'rf_weight': self.rf_weight,
                'lstm_weight': self.lstm_weight,
                'tabnet_weight': self.tabnet_weight,
                'models_loaded': {
                    'rf': self.rf_model is not None,
                    'tabnet': self.tabnet_model is not None
                }
            },
            'alternative_models': {
                'available': ALTERNATIVE_MODELS_AVAILABLE,
                'models_loaded': alt_models_info,
                'total_loaded': sum(1 for loaded in alt_models_info.values() if loaded)
            },
            'blending': {
                'method': 'enhanced_blender',
                'supports_adaptive': True,
                'current_weights': self.blender.get_model_weights() if hasattr(self, 'blender') else {}
            },
            'database': self.db_path,
            'database_name': self.db_name
        }


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
    if race_data is None:
        raise ValueError("race_data cannot be None")
    elif isinstance(race_data, str):
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


def re_blend_predictions_with_dynamic_weights(race_id: Optional[str] = None,
                                               date: Optional[str] = None,
                                               all_races: bool = True,
                                               db_name: str = None,
                                               verbose: bool = True) -> Dict[str, Any]:
    """
    Standalone function to re-blend existing predictions with dynamic weights.
    Can be called from UI or scripts without re-predicting.

    Args:
        race_id: Specific race_id to re-blend. If provided, only that race is processed.
        date: Specific date (YYYY-MM-DD) to re-blend. Ignored if race_id or all_races=True.
        all_races: If True, re-blends ALL races with predictions (default). Overrides date.
        db_name: Database name (defaults to active_db from config)
        verbose: Whether to print progress

    Returns:
        Dictionary with summary of re-blending results

    Example:
        # Re-blend ALL races with predictions (default)
        result = re_blend_predictions_with_dynamic_weights()

        # Re-blend specific date only
        result = re_blend_predictions_with_dynamic_weights(date='2025-10-07', all_races=False)

        # Re-blend specific race
        result = re_blend_predictions_with_dynamic_weights(race_id='1606874', all_races=False)
    """
    # Initialize predictor (lightweight - no model loading needed)
    predictor = RacePredictor(db_name=db_name, verbose=verbose)

    # Call re-blend method
    result = predictor.re_blend_existing_predictions_with_dynamic_weights(
        race_id=race_id,
        date=date,
        all_races=all_races,
        verbose=verbose
    )

    return result


if __name__ == "__main__":
    # Run example when executed directly from IDE
    example_prediction()