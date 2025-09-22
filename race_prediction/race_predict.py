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


class RacePredictor:
    """
    Enhanced race predictor that supports RF and TabNet models with intelligent blending.
    Uses the same data preparation pipeline as training for consistency.
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

        self.model_path = Path(model_path)
        
        # Initialize enhanced prediction blender
        self.blender = EnhancedPredictionBlender(verbose=self.verbose)
        
        # Load models and configuration
        self._load_models()
        self._load_alternative_models()
        
        # Legacy blend weights (for backward compatibility)
        self._load_blend_weights()

        if self.verbose:
            print(f"RacePredictor initialized")
            print(f"  Model: {self.model_path}")
            print(f"  Database: {self.db_path}")
            print(f"  Legacy weights: RF={self.rf_weight:.1f}, LSTM={self.lstm_weight:.1f}, TabNet={self.tabnet_weight:.1f}")
            print(f"  Legacy models loaded: RF={self.rf_model is not None}, LSTM={self.lstm_model is not None}, TabNet={self.tabnet_model is not None}")
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
                # Check if we have three-model weights
                if 'rf_weight' in training_results:
                    self.rf_weight = training_results.get('rf_weight', 0.8)
                    self.lstm_weight = training_results.get('lstm_weight', 0.1)
                    self.tabnet_weight = training_results.get('tabnet_weight', 0.1)
                else:
                    # Legacy: convert old blend_weight to RF/LSTM split
                    old_blend_weight = training_results.get('blend_weight', 0.9)
                    self.rf_weight = old_blend_weight
                    self.lstm_weight = 1.0 - old_blend_weight
                    self.tabnet_weight = 0.0
        else:
            self.model_config = {}
            # Use default blend weights if not in config
            self.rf_weight = 0.8
            self.lstm_weight = 0.1
            self.tabnet_weight = 0.1

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

        # Load RF model (REQUIRED)
        rf_model_path = self.model_path / "hybrid_rf_model.joblib"
        if rf_model_path.exists():
            self.rf_model = joblib.load(rf_model_path)
            if self.verbose:
                print(f"Loaded RF model: {type(self.rf_model)}")
        else:
            self.rf_model = None
            raise FileNotFoundError(f"RF model not found at {rf_model_path} - required for predictions")

        # Load LSTM model (REQUIRED)
        lstm_model_path = self.model_path / "hybrid_lstm_model.keras"
        if lstm_model_path.exists():
            try:
                from tensorflow.keras.models import load_model
                self.lstm_model = load_model(lstm_model_path)
                if self.verbose:
                    print(f"Loaded LSTM model")
            except Exception as e:
                raise RuntimeError(f"Failed to load LSTM model at {lstm_model_path}: {str(e)}")
        else:
            raise FileNotFoundError(f"LSTM model not found at {lstm_model_path} - required for predictions")
                
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

    def _load_blend_weights(self):
        """Load optimal blend weights from config."""
        try:
            blend_config = self.config._config.blend
            self.rf_weight = blend_config.rf_weight
            self.lstm_weight = blend_config.lstm_weight
            self.tabnet_weight = blend_config.tabnet_weight
            
            # Validate weights sum to 1
            total_weight = self.rf_weight + self.lstm_weight + self.tabnet_weight
            if abs(total_weight - 1.0) > 1e-6:
                if self.verbose:
                    print(f"Warning: Blend weights don't sum to 1.0: {total_weight}, normalizing...")
                # Normalize weights
                self.rf_weight = self.rf_weight / total_weight
                self.lstm_weight = self.lstm_weight / total_weight
                self.tabnet_weight = self.tabnet_weight / total_weight
                
        except (AttributeError, KeyError):
            # Use default optimal weights if not in config
            if self.verbose:
                print("Using default optimal blend weights: 80/10/10")
            self.rf_weight = 0.8
            self.lstm_weight = 0.1
            self.tabnet_weight = 0.1

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
            # Look for TabNet model files
            tabnet_model_file = self.model_path / "tabnet_model.zip"
            if not tabnet_model_file.exists():
                tabnet_model_file = self.model_path / "tabnet_model.zip.zip"
            
            if tabnet_model_file.exists():
                self.tabnet_model = TabNetRegressor()
                self.tabnet_model.load_model(str(tabnet_model_file))
                
                # Load scaler
                scaler_file = self.model_path / "tabnet_scaler.joblib"
                if scaler_file.exists():
                    self.tabnet_scaler = joblib.load(scaler_file)
                    
                # Load feature configuration
                config_file = self.model_path / "tabnet_config.json"
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
                    print("TabNet model not found")
                    
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
        Prepare features specifically for TabNet prediction using domain features.
        NEW: Uses ModelFeatureSelector for consistency with training.
        """
        if self.verbose:
            print("Preparing domain features for TabNet prediction...")
            
        # Apply feature calculator for musique preprocessing
        df_with_features = FeatureCalculator.calculate_all_features(race_df)
        
        # Apply embeddings to get complete feature set
        complete_df = self.orchestrator.apply_embeddings(df_with_features)
        
        # Extract TabNet features using the same pipeline as training
        X_tabnet, _ = self.orchestrator.extract_tabnet_features(complete_df)
        
        if self.verbose:
            print(f"TabNet prediction features: {X_tabnet.shape[1]} domain features")
        
        return X_tabnet, complete_df
    
    def predict_with_tabnet(self, race_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using TabNet model with domain features.
        NEW: Uses same domain feature extraction as training.
        """
        if self.tabnet_model is None:
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
                    print(f"Warning: Missing TabNet features: {missing_features}")
                    
                if not available_features:
                    if self.verbose:
                        print("Warning: No matching TabNet features found")
                    return None
                    
                # Use training feature order
                X = X_tabnet[available_features].values
            else:
                # Use all available features
                X = X_tabnet.values
                
            # Scale features using training scaler
            if self.tabnet_scaler is not None:
                X_scaled = self.tabnet_scaler.transform(X)
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
        
        # Step 4: Generate TabNet predictions
        tabnet_predictions = self.predict_with_tabnet(race_df)
        
        # Step 5: Generate alternative model predictions
        alternative_predictions = self._predict_with_alternative_models(race_df, embedded_df)
        
        # Step 6: Collect all predictions for blending
        all_predictions = {
            'rf': rf_predictions,
            'lstm': lstm_predictions,
            'tabnet': tabnet_predictions
        }
        
        # Add alternative model predictions
        all_predictions.update(alternative_predictions)
        
        # Step 7: Use enhanced blender for intelligent blending
        race_metadata = {
            'distance': race_df.get('dist', [0]).iloc[0] if 'dist' in race_df.columns else 1600,
            'typec': race_df.get('typec', ['P']).iloc[0] if 'typec' in race_df.columns else 'P',
            'field_size': len(race_df),
            'hippo': race_df.get('hippo', ['']).iloc[0] if 'hippo' in race_df.columns else ''
        }
        
        final_predictions, blend_info = self.blender.blend_predictions(
            predictions=all_predictions,
            race_metadata=race_metadata
        )
        
        if self.verbose:
            print(f"Enhanced blending used {blend_info['total_models']} models: {blend_info['models_used']}")
            print(f"Applied weights: {blend_info['weights_applied']}")
            print(f"Blend method: {blend_info['blend_method']}")

        # Step 8: Create result DataFrame
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
                    'lstm': self.lstm_model is not None,
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