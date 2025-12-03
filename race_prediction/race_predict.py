import os
import numpy as np
import pandas as pd
import json
import joblib
import time
import gc
import sqlite3
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple
import logging
from datetime import datetime

# Optional psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import the optimized orchestrator and utilities
from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from utils.model_manager import get_model_manager
from core.calculators.static_feature_calculator import FeatureCalculator
from sklearn.preprocessing import StandardScaler
from race_prediction.enhanced_prediction_blender import EnhancedPredictionBlender

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
from race_prediction.simple_prediction_storage import SimplePredictionStorage, extract_prediction_data_from_competitive_analysis


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        # Return dummy values if psutil not available
        return {
            'rss_mb': 0.0,
            'vms_mb': 0.0,
            'percent': 0.0
        }

    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }


def cleanup_memory():
    """Aggressively cleanup memory."""
    gc.collect()
    # Force garbage collection of all generations
    for _ in range(3):
        gc.collect(generation=2)


# Global predictor for multiprocessing workers (one per process)
_worker_predictor = None
_worker_db_name = None
_worker_verbose = False


def _init_worker(db_name: str, verbose: bool):
    """Initialize a worker process with its own predictor instance."""
    global _worker_predictor, _worker_db_name, _worker_verbose
    _worker_db_name = db_name
    _worker_verbose = verbose
    # Create predictor instance for this worker
    _worker_predictor = None  # Lazy initialization on first use


def _worker_predict_race(race_df: pd.DataFrame) -> pd.DataFrame:
    """Worker function for multiprocessing - uses global predictor."""
    global _worker_predictor

    # Lazy initialization: create predictor on first use
    if _worker_predictor is None:
        _worker_predictor = RacePredictor(
            db_name=_worker_db_name,
            verbose=False,  # Disable verbose in workers
            enable_prediction_storage=False  # Disable storage in workers (main process will handle it)
        )

    # Predict the race
    try:
        result = _worker_predictor.predict_race(race_df)
        return result
    except Exception as e:
        # Return empty result on error
        print(f"Worker error: {e}")
        return race_df.copy()


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

        # Cache for prepared features to avoid duplicate work
        self._cached_tabnet_features = None

        # Cache for feature alignment to avoid repeated set operations
        self._rf_common_features_cache = None
        self._tabnet_available_features_cache = None

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

        # Convert numeric fields
        for field in numeric_fields:
            if field in race_df.columns:
                # Convert to numeric, coercing errors to NaN
                race_df[field] = pd.to_numeric(race_df[field], errors='coerce').fillna(0)

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
            # Align features with training - OPTIMIZED VERSION
            # Use cached common features if available (avoids set operations)
            if self._rf_common_features_cache is None:
                self._rf_common_features_cache = list(set(expected_features))

            # Filter to available features
            common_features = [f for f in self._rf_common_features_cache if f in X.columns]

            # Use reindex for fast alignment (much faster than DataFrame init + loop)
            aligned_X = X[common_features].reindex(columns=expected_features, fill_value=0)

            # Clean all columns at once (vectorized)
            aligned_X = aligned_X.replace(['', None, 'NULL'], 0).fillna(0)

            # Ensure numeric dtypes (vectorized)
            for col in aligned_X.columns:
                aligned_X[col] = pd.to_numeric(aligned_X[col], errors='coerce').fillna(0)

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
        self.tabnet_feature_selector = None

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

                # Load feature selector if it exists
                try:
                    from core.feature_selection.tabnet_feature_selector import TabNetFeatureSelector
                    feature_selector_file = tabnet_path / "feature_selector.json"
                    if feature_selector_file.exists():
                        self.tabnet_feature_selector = TabNetFeatureSelector.load(str(feature_selector_file))
                        if self.verbose:
                            print(f"Loaded TabNet feature selector: {len(self.tabnet_feature_selector.selected_features)} selected features")
                    else:
                        if self.verbose:
                            print(f"No feature selector found (using all features)")
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not load feature selector: {e}")

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
            self.tabnet_feature_selector = None
    
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

            # Cache features for potential reuse (e.g., feature export)
            self._cached_tabnet_features = (X_tabnet, complete_df)

            if X_tabnet is None or X_tabnet.empty:
                if self.verbose:
                    print("Warning: No TabNet features prepared")
                return None
            
            # Determine which features to use based on whether we have feature selection
            if self.tabnet_feature_selector is not None:
                # Use feature selection pipeline
                if self.verbose:
                    print(f"Using TabNet feature selection pipeline...")

                # Get original features from selector (features before selection)
                expected_features = self.tabnet_feature_selector.original_features
                available_features = [col for col in expected_features if col in X_tabnet.columns]
                missing_features = [col for col in expected_features if col not in X_tabnet.columns]

                if missing_features and self.verbose:
                    print(f"Warning: Missing {len(missing_features)}/{len(expected_features)} original features")

                # Create aligned DataFrame with original features - OPTIMIZED
                aligned_X = X_tabnet[available_features].reindex(columns=expected_features, fill_value=0.0)

                if self.verbose:
                    print(f"TabNet features before selection: {aligned_X.shape}")

                # Apply feature selection
                X_df = self.tabnet_feature_selector.transform(aligned_X)
                if self.verbose:
                    print(f"TabNet features after selection: {X_df.shape} ({len(expected_features)} → {len(X_df.columns)})")

            elif self.tabnet_feature_columns:
                # No feature selection - use model's expected features directly
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

                # Create aligned DataFrame with all training features - OPTIMIZED
                aligned_X = X_tabnet[available_features].reindex(columns=self.tabnet_feature_columns, fill_value=0.0)

                if self.verbose:
                    print(f"TabNet features aligned: {aligned_X.shape}")
                X_df = aligned_X
            else:
                # Use all available features
                X_df = X_tabnet
                if self.verbose:
                    print(f"Using all available features: {X_df.shape}")

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
                
            # Generate predictions
            tabnet_preds = self.tabnet_model.predict(X_scaled)

            if len(tabnet_preds.shape) > 1:
                tabnet_preds = tabnet_preds.flatten()

            if self.verbose:
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

    def load_races_from_db(self, race_ids: List[str] = None, date: str = None,
                           limit: int = None) -> List[pd.DataFrame]:
        """
        Efficiently load multiple races from database.

        Args:
            race_ids: List of specific race IDs to load
            date: Load all races from specific date (YYYY-MM-DD)
            limit: Maximum number of races to load

        Returns:
            List of race DataFrames
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)

        try:
            if race_ids:
                # Load specific races
                placeholders = ','.join(['?'] * len(race_ids))
                query = f"""
                    SELECT * FROM daily_race
                    WHERE comp IN ({placeholders})
                """
                params = race_ids
            elif date:
                # Load races from specific date
                query = "SELECT * FROM daily_race WHERE jour = ?"
                params = (date,)
            else:
                # Load all races (with optional limit)
                query = "SELECT * FROM daily_race"
                params = ()

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql_query(query, conn, params=params)

            if len(df) == 0:
                return []

            # Expand participants for each race
            races = []
            for _, race_row in df.iterrows():
                try:
                    participants_json = race_row.get('participants', '[]')
                    participants = json.loads(participants_json)

                    # Create race dataframe with participant data
                    race_data = []
                    for participant in participants:
                        row = {**race_row.to_dict(), **participant}
                        row.pop('participants', None)
                        race_data.append(row)

                    if race_data:
                        race_df = pd.DataFrame(race_data)
                        races.append(race_df)

                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not parse race {race_row.get('comp')}: {e}")
                    continue

            if self.verbose:
                print(f"Loaded {len(races)} races from database")

            return races

        finally:
            conn.close()

    def predict_races_batch(self, races: List[pd.DataFrame], n_jobs: int = 4,
                            chunk_size: int = 50, max_memory_mb: float = 4096,
                            verbose: bool = None, progress_callback=None) -> List[pd.DataFrame]:
        """
        Predict multiple races in parallel using multiprocessing with memory management.

        Args:
            races: List of race DataFrames
            n_jobs: Number of parallel jobs (default: 4, use -1 for all cores)
            chunk_size: Number of races to process per chunk (default: 50)
            max_memory_mb: Maximum memory usage in MB before forcing cleanup (default: 4GB)
            verbose: Override instance verbosity
            progress_callback: Optional callback(percent, message) for progress updates

        Returns:
            List of prediction DataFrames

        Example:
            races = [race1_df, race2_df, race3_df, ...]
            results = predictor.predict_races_batch(races, n_jobs=8, chunk_size=50)
        """
        if verbose is None:
            verbose = self.verbose

        if verbose:
            mem_usage = get_memory_usage()
            print(f"🚀 Batch predicting {len(races)} races with {n_jobs} workers...")
            print(f"   Initial memory: {mem_usage['rss_mb']:.1f} MB ({mem_usage['percent']:.1f}%)")
            print(f"   Chunk size: {chunk_size} races")

        import multiprocessing as mp

        # Determine number of workers
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        n_jobs = min(n_jobs, mp.cpu_count())

        if n_jobs == 1 or len(races) == 1:
            # Sequential processing for single job or single race
            results = []
            for idx, race in enumerate(races):
                if progress_callback:
                    progress = 20 + int((idx / len(races)) * 75)
                    progress_callback(progress, f"Processing race {idx + 1}/{len(races)}...")
                results.append(self.predict_race(race))
            return results

        # Process in chunks to manage memory
        start_time = time.time()
        all_results = []

        # Calculate number of chunks
        num_chunks = (len(races) + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, len(races))
            chunk_races = races[chunk_start:chunk_end]

            # Update progress: 20% to 95% based on chunk completion
            if progress_callback:
                progress = 20 + int((chunk_idx / num_chunks) * 75)
                mem_info = get_memory_usage()
                mem_str = f" (Mem: {mem_info['rss_mb']:.0f}MB)" if PSUTIL_AVAILABLE else ""
                progress_callback(progress, f"Processing chunk {chunk_idx + 1}/{num_chunks}{mem_str}...")

            if verbose:
                print(f"\n📦 Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_races)} races)...")

            # Check memory before processing
            mem_before = get_memory_usage()
            if mem_before['rss_mb'] > max_memory_mb and PSUTIL_AVAILABLE:
                if verbose:
                    print(f"⚠️  Memory usage high ({mem_before['rss_mb']:.1f} MB), forcing cleanup...")
                cleanup_memory()
                mem_after = get_memory_usage()
                if verbose:
                    freed = mem_before['rss_mb'] - mem_after['rss_mb']
                    print(f"✅ Freed {freed:.1f} MB (now at {mem_after['rss_mb']:.1f} MB)")

            # Parallel processing for chunk
            chunk_start_time = time.time()

            # Limit workers to chunk size for small chunks
            chunk_workers = min(n_jobs, len(chunk_races))

            # Use pool with initializer for efficient worker setup
            with mp.Pool(
                processes=chunk_workers,
                initializer=_init_worker,
                initargs=(self.db_name, False)
            ) as pool:
                chunk_results = pool.map(_worker_predict_race, chunk_races)

            chunk_time = time.time() - chunk_start_time
            all_results.extend(chunk_results)

            if verbose:
                mem_after = get_memory_usage()
                print(f"✅ Chunk complete: {len(chunk_races)} races in {chunk_time:.2f}s")
                if PSUTIL_AVAILABLE:
                    print(f"   Memory: {mem_after['rss_mb']:.1f} MB ({mem_after['percent']:.1f}%)")

            # Cleanup after each chunk
            cleanup_memory()

        total_time = time.time() - start_time

        if verbose:
            mem_final = get_memory_usage()
            print(f"\n{'='*60}")
            print(f"✅ Batch prediction complete!")
            print(f"   Total: {len(races)} races in {total_time:.2f}s")
            print(f"   Average: {total_time/len(races):.2f}s per race")
            if PSUTIL_AVAILABLE:
                print(f"   Final memory: {mem_final['rss_mb']:.1f} MB ({mem_final['percent']:.1f}%)")
            print(f"{'='*60}")

        return all_results

    def predict_from_db(self, race_ids: List[str] = None, date: str = None,
                       limit: int = None, n_jobs: int = 4,
                       chunk_size: int = 50, max_memory_mb: float = 4096,
                       store_to_db: bool = True, progress_callback=None) -> List[pd.DataFrame]:
        """
        Load races from database and predict in batch with memory management.

        Args:
            race_ids: List of specific race IDs to predict
            date: Predict all races from specific date
            limit: Maximum number of races to predict
            n_jobs: Number of parallel workers (use -1 for all cores)
            chunk_size: Number of races to process per chunk (default: 50)
            max_memory_mb: Maximum memory usage in MB (default: 4GB)
            store_to_db: Whether to store predictions back to database
            progress_callback: Optional callback(percent, message) for progress updates

        Returns:
            List of prediction DataFrames

        Example:
            # Predict all races from a date with memory management
            results = predictor.predict_from_db(date='2025-10-15', n_jobs=8, chunk_size=50)

            # Predict 1000 races with strict memory limit
            results = predictor.predict_from_db(limit=1000, n_jobs=8,
                                               chunk_size=30, max_memory_mb=2048)
        """
        start_time = time.time()

        if progress_callback:
            progress_callback(5, "Loading races from database...")

        if self.verbose:
            mem_start = get_memory_usage()
            print(f"🔥 Memory-safe batch prediction pipeline starting...")
            if PSUTIL_AVAILABLE:
                print(f"   Initial memory: {mem_start['rss_mb']:.1f} MB ({mem_start['percent']:.1f}%)")

        # Step 1: Load races from database in streaming mode
        load_start = time.time()
        races = self.load_races_from_db(race_ids=race_ids, date=date, limit=limit)
        load_time = time.time() - load_start

        if not races:
            if self.verbose:
                print("No races found")
            if progress_callback:
                progress_callback(100, "No races found")
            return []

        if progress_callback:
            progress_callback(15, f"Loaded {len(races)} races, starting predictions...")

        if self.verbose:
            mem_after_load = get_memory_usage()
            print(f"✅ Loaded {len(races)} races in {load_time:.2f}s")
            if PSUTIL_AVAILABLE:
                print(f"   Memory after load: {mem_after_load['rss_mb']:.1f} MB")

        # Step 2: Batch predict with multiprocessing and memory management
        results = self.predict_races_batch(
            races,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            max_memory_mb=max_memory_mb,
            verbose=self.verbose,
            progress_callback=progress_callback  # Pass through progress callback
        )

        # Clear races from memory
        del races
        cleanup_memory()

        # Step 3: Store to database if requested (in chunks to save memory)
        if store_to_db and results:
            if progress_callback:
                progress_callback(95, "Storing predictions to database...")

            store_start = time.time()
            self._store_predictions_batch(results)
            store_time = time.time() - store_start

            if self.verbose:
                print(f"✅ Stored {len(results)} races in {store_time:.2f}s")

        total_time = time.time() - start_time

        if progress_callback:
            progress_callback(100, f"Complete! Processed {len(results)} races in {total_time:.1f}s")

        if self.verbose:
            mem_final = get_memory_usage()
            print(f"\n{'='*60}")
            print(f"🎉 BATCH PREDICTION COMPLETE")
            print(f"{'='*60}")
            print(f"  Races processed: {len(results)}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average: {total_time/len(results):.3f}s per race")
            print(f"  Throughput: {len(results)/total_time:.1f} races/second")
            if PSUTIL_AVAILABLE:
                print(f"  Final memory: {mem_final['rss_mb']:.1f} MB ({mem_final['percent']:.1f}%)")
            print(f"{'='*60}\n")

        return results

    def _store_predictions_batch(self, predictions: List[pd.DataFrame]):
        """Store predictions for multiple races in batch (optimized)."""
        if not self.simple_storage:
            return

        stored = 0
        failed = 0
        for pred_df in predictions:
            try:
                # Get race metadata
                race_id = pred_df['comp'].iloc[0] if 'comp' in pred_df.columns else None
                if not race_id:
                    failed += 1
                    continue

                # Extract prediction data for race_predictions table
                predictions_data = []
                for _, row in pred_df.iterrows():
                    horse_data = {
                        'horse_id': row.get('idche'),
                        'rf_prediction': row.get('predicted_position_rf'),
                        'tabnet_prediction': row.get('predicted_position_tabnet'),
                        'ensemble_prediction': row.get('predicted_position'),
                        'final_prediction': row.get('predicted_position'),
                        'ensemble_weight_rf': row.get('ensemble_weight_rf', 0.0),
                        'ensemble_weight_tabnet': row.get('ensemble_weight_tabnet', 0.0)
                    }
                    predictions_data.append(horse_data)

                # Store to race_predictions table
                self.simple_storage.store_race_predictions(race_id, predictions_data)

                # Also update daily_race.prediction_results field
                # Extract columns for daily_race storage
                output_columns = [
                    'numero', 'idche', 'nom', 'predicted_position',
                    'predicted_position_rf', 'predicted_position_tabnet',
                    'predicted_position_uncalibrated',  # For calibration debugging
                    'raw_rf_prediction',  # For debugging RF calibration compression
                    'predicted_rank', 'ensemble_weight_rf', 'ensemble_weight_tabnet'
                ]

                # Only include columns that exist in pred_df
                available_columns = [col for col in output_columns if col in pred_df.columns]
                prediction_results = pred_df[available_columns].to_dict(orient='records')

                # Get predicted_arriv
                predicted_arriv = pred_df['predicted_arriv'].iloc[0] if 'predicted_arriv' in pred_df.columns else None

                # Create metadata
                metadata = {
                    'comp': race_id,
                    'hippo': pred_df['hippo'].iloc[0] if 'hippo' in pred_df.columns else None,
                    'prix': pred_df['prix'].iloc[0] if 'prix' in pred_df.columns else None,
                    'jour': pred_df['jour'].iloc[0] if 'jour' in pred_df.columns else None,
                    'typec': pred_df['typec'].iloc[0] if 'typec' in pred_df.columns else None,
                    'participants_count': len(prediction_results),
                    'predicted_arriv': predicted_arriv
                }

                # Create full prediction data structure
                prediction_data = {
                    'metadata': metadata,
                    'predictions': prediction_results,
                    'predicted_arriv': predicted_arriv
                }

                # Update daily_race table
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE daily_race SET prediction_results = ?, updated_at = ? WHERE comp = ?",
                    (json.dumps(prediction_data), datetime.now().isoformat(), race_id)
                )
                conn.commit()
                conn.close()

                stored += 1

            except Exception as e:
                failed += 1
                if self.verbose:
                    print(f"Warning: Failed to store predictions for race {race_id}: {e}")
                    import traceback
                    print(traceback.format_exc())

        if self.verbose or failed > 0:
            print(f"Storage: {stored}/{len(predictions)} races successful, {failed} failed")

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
            # Reuse cached TabNet features to avoid duplicate preparation
            if self._cached_tabnet_features is not None:
                X_tabnet_unscaled, _ = self._cached_tabnet_features
                self._capture_features_for_export(race_df, X_tabnet_unscaled, race_metadata)
            elif self.verbose:
                print("Warning: No cached TabNet features available for export")

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

        # Add individual model predictions for storage
        if rf_predictions is not None:
            result_df['predicted_position_rf'] = rf_predictions
        if tabnet_predictions is not None:
            result_df['predicted_position_tabnet'] = tabnet_predictions

        # Add blend weights for storage
        result_df['ensemble_weight_rf'] = rf_weight
        result_df['ensemble_weight_tabnet'] = tabnet_weight

        # Step 10.5: Apply bias-based calibration (general model)
        from core.calibration.prediction_calibrator import PredictionCalibrator
        from pathlib import Path as CalibPath

        calibration_path = CalibPath('models/calibration/general_calibration.json')
        if calibration_path.exists():
            try:
                calibrator = PredictionCalibrator(calibration_path)

                # Prepare DataFrame for calibration - build it properly
                calib_df = result_df[['predicted_position', 'numero']].copy()

                # Add cotedirect column (with default if missing)
                if 'cotedirect' in result_df.columns:
                    calib_df['cotedirect'] = result_df['cotedirect'].values
                else:
                    calib_df['cotedirect'] = 5.0  # Broadcast to all rows

                # Add race characteristics (broadcast scalars to all rows)
                calib_df['distance'] = race_metadata.get('distance', race_metadata.get('dist', 1600))
                calib_df['typec'] = race_metadata.get('typec', 'P')
                calib_df['partant'] = race_metadata.get('field_size', len(result_df))

                if self.verbose:
                    print(f"\nBefore calibration: {result_df['predicted_position'].head().values}")

                # Apply bias calibration
                calibrated_df = calibrator.transform(calib_df)
                result_df['predicted_position_uncalibrated'] = result_df['predicted_position'].values
                result_df['predicted_position'] = calibrated_df['calibrated_prediction'].values

                if self.verbose:
                    print(f"After calibration: {result_df['predicted_position'].head().values}")
                    print(f"✓ Applied general bias calibration")

            except Exception as e:
                if self.verbose:
                    print(f"⚠ Bias calibration failed: {e}")
                    import traceback
                    traceback.print_exc()
        elif self.verbose:
            print(f"⚠ No general bias calibration available")

        # Sort by predicted position (lower is better)
        result_df = result_df.sort_values('predicted_position')
        result_df['predicted_rank'] = range(1, len(result_df) + 1)

        # Create arrival string format
        numeros_ordered = result_df['numero'].astype(str).tolist()
        predicted_arriv = '-'.join(numeros_ordered)
        result_df['predicted_arriv'] = predicted_arriv

        # Clear cached features and cleanup memory
        self._cached_tabnet_features = None

        # Free memory from large intermediate objects
        del embedded_df, X_rf
        # Don't delete predictions yet - they're in the result_df

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

                # Re-blend predictions with new weights - VECTORIZED (much faster!)
                # Filter out rows with missing predictions
                valid_preds = predictions.dropna(subset=['rf_prediction', 'tabnet_prediction'])

                if len(valid_preds) == 0:
                    if verbose:
                        print(f"  No valid predictions for race {race_comp}, skipping")
                    continue

                # Vectorized calculation (no loop!)
                valid_preds['ensemble_prediction'] = (
                    valid_preds['rf_prediction'] * rf_weight +
                    valid_preds['tabnet_prediction'] * tabnet_weight
                )
                valid_preds['final_prediction'] = valid_preds['ensemble_prediction']
                valid_preds['ensemble_weight_rf'] = rf_weight
                valid_preds['ensemble_weight_tabnet'] = tabnet_weight

                # Batch update to database (much faster than individual updates!)
                cursor = conn.cursor()

                # Convert to list of tuples efficiently (no iterrows!)
                update_columns = ['ensemble_weight_rf', 'ensemble_weight_tabnet',
                                  'ensemble_prediction', 'final_prediction', 'id']
                update_data = [tuple(row) for row in valid_preds[update_columns].values]

                # Use executemany for batch update (much faster!)
                cursor.executemany("""
                    UPDATE race_predictions
                    SET ensemble_weight_rf = ?,
                        ensemble_weight_tabnet = ?,
                        ensemble_prediction = ?,
                        final_prediction = ?
                    WHERE id = ?
                """, update_data)

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
                total_updated += len(update_data)
                races_updated.append({
                    'race_id': race_comp,
                    'horses_updated': len(update_data),
                    'rf_weight': rf_weight,
                    'tabnet_weight': tabnet_weight
                })

                if verbose:
                    print(f"  Updated {len(update_data)} horses")

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


def predict_races_fast(race_ids: List[str] = None, date: str = None,
                       limit: int = None, n_jobs: int = -1,
                       chunk_size: int = 50, max_memory_mb: float = 4096,
                       db_name: str = None, verbose: bool = True,
                       progress_callback=None) -> List[pd.DataFrame]:
    """
    Fast batch prediction with memory management for 1000+ races.

    Args:
        race_ids: List of specific race IDs
        date: Predict all races from date (YYYY-MM-DD)
        limit: Maximum races to predict
        n_jobs: Number of parallel workers (-1 = all cores, default: -1)
        chunk_size: Races per chunk for memory management (default: 50)
        max_memory_mb: Max memory in MB before cleanup (default: 4GB)
        db_name: Database name (defaults to active_db)
        verbose: Print progress
        progress_callback: Optional callback(percent, message) for progress updates

    Returns:
        List of prediction DataFrames

    Example:
        # Predict 1000 races with memory management
        results = predict_races_fast(limit=1000, n_jobs=-1, chunk_size=50)

        # Predict with strict memory limit (2GB)
        results = predict_races_fast(date='2025-10-15', n_jobs=8,
                                     chunk_size=30, max_memory_mb=2048)
    """
    predictor = RacePredictor(db_name=db_name, verbose=verbose)
    return predictor.predict_from_db(
        race_ids=race_ids,
        date=date,
        limit=limit,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        max_memory_mb=max_memory_mb,
        store_to_db=True,
        progress_callback=progress_callback
    )


if __name__ == "__main__":
    # Run example when executed directly from IDE
    example_prediction()