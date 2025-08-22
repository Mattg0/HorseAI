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
from race_prediction.prediction_blender import PredictionBlender

# TabNet imports
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False


class RacePredictor:
    """
    Enhanced race predictor that uses RF + LSTM + TabNet models with optimal blend weights.
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
        
        # Initialize prediction blender for configurable blending
        self.blender = PredictionBlender()
        
        # Load optimal blend weights from config as fallback
        self._load_optimal_blend_weights()
        
        # Load models and configuration
        self._load_models()

        if self.verbose:
            print(f"RacePredictor initialized")
            print(f"  Model: {self.model_path}")
            print(f"  Database: {self.db_path}")
            print(f"  Models loaded: RF={self.rf_model is not None}, TabNet={self.tabnet_model is not None}")
            print(f"  Available blending rules: {', '.join(self.blender.get_available_rules())}")

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
        else:
            self.model_config = {}

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

        # LSTM removed - no longer used
        self.lstm_model = None
                
        # Load TabNet model
        self._load_tabnet_model()

    def prepare_race_data(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare race data using the SAME pipeline as training.
        This is the generic preparation used by RF and LSTM models.

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
            'typec': 'P',
            'natpis': 'P',
            'meteo': 'Sec',
            'hippo': 'Unknown',
            'final_position': np.nan,
            'jour': datetime.now().strftime('%Y-%m-%d'),
            'temperature': 15.0,
            'corde': 1,
            'musiqueche': '',  # Required for TabNet feature calculation
            'musiquejoc': '',  # Required for jockey musique features
            'coursescheval': 0,  # Required for feature calculation
            'victoirescheval': 0,  # Required for feature calculation
            'placescheval': 0,  # Required for feature calculation
            'gainsCarriere': 0  # Required for feature calculation
        }

        for col, default_val in required_columns.items():
            if col not in race_df.columns:
                race_df[col] = default_val

        # Step 2: Ensure numeric columns are properly typed
        numeric_fields = [
            'age', 'cotedirect', 'coteprob', 'pourcVictChevalHippo',
            'pourcPlaceChevalHippo', 'pourcVictJockHippo', 'pourcPlaceJockHippo',
            'victoirescheval', 'placescheval', 'coursescheval', 'dist',
            'temperature', 'forceVent', 'idche', 'idJockey', 'numero', 'corde'
        ]

        for field in numeric_fields:
            if field in race_df.columns:
                race_df[field] = pd.to_numeric(race_df[field], errors='coerce').fillna(0)

        # Step 3: Apply standard feature preparation and embeddings
        processed_df = self.orchestrator.prepare_features(race_df, use_cache=False)
        
        # Fit embeddings if not already fitted
        if not self.orchestrator.embeddings_fitted:
            # Use small historical sample for embedding fitting
            historical_sample = self.orchestrator.load_historical_races(limit=100, use_cache=True)
            self.orchestrator.fit_embeddings(historical_sample, use_cache=True)
        
        # Apply embeddings
        embedded_df = self.orchestrator.apply_embeddings(processed_df, use_cache=False)

        if self.verbose:
            print(f"Data preparation complete: {len(embedded_df.columns)} features")
            # Debug: Check if we have embeddings or raw features
            has_embeddings = any(col.startswith(('horse_emb_', 'jockey_emb_', 'couple_emb_', 'course_emb_')) for col in embedded_df.columns)
            has_raw_features = any(col in embedded_df.columns for col in ['TxVictCouple', 'efficacite_couple', 'nbPlaceCouple'])
            print(f"Debug - Has embeddings: {has_embeddings}, Has raw features: {has_raw_features}")

        return embedded_df

    def prepare_rf_data(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data specifically for RF model to match the EXACT 72 features it was trained with.
        """
        if self.verbose:
            print(f"Preparing RF data to match model's expected features...")

        # Get the expected feature names from the RF model
        expected_features = self.rf_model.base_regressor.feature_names_in_
        
        # Add ALL required columns with appropriate defaults
        required_columns = {
            # Basic identifiers
            'idche': 0, 'idJockey': 0, 'numero': 0,
            # Race info
            'typec': 'Plat', 'natpis': 'PSF', 'meteo': 'BEAU', 'hippo': 'Unknown',
            'dist': 1600, 'temperature': 15.0, 'corde': 1, 'partant': len(race_df),
            'jour': datetime.now().strftime('%Y-%m-%d'),
            # Performance data
            'cotedirect': 5.0, 'coteprob': 5.0, 'age': 5,
            'victoirescheval': 0, 'placescheval': 0, 'coursescheval': 0,
            'pourcVictChevalHippo': 0.0, 'pourcPlaceChevalHippo': 0.0,
            'gainsCarriere': 0, 'gainsAnneeEnCours': 0,
            # Technical fields
            'handicapDistance': 0, 'handicapPoids': 0, 'poidmont': 0, 'recence': 0,
            'reunion': 1, 'comp': 1, 'final_position': np.nan,
            # Musique data
            'musiqueche': '', 'musiquejoc': ''
        }

        for col, default_val in required_columns.items():
            if col not in race_df.columns:
                race_df[col] = default_val

        # Ensure all numeric fields are properly typed
        numeric_fields = [
            'age', 'cotedirect', 'coteprob', 'numero', 'idche', 'idJockey', 'dist', 
            'temperature', 'corde', 'partant', 'victoirescheval', 'placescheval', 
            'coursescheval', 'pourcVictChevalHippo', 'pourcPlaceChevalHippo',
            'gainsCarriere', 'gainsAnneeEnCours', 'handicapDistance', 'handicapPoids',
            'poidmont', 'recence', 'reunion', 'comp'
        ]

        for field in numeric_fields:
            if field in race_df.columns:
                race_df[field] = pd.to_numeric(race_df[field], errors='coerce').fillna(0)

        # Step 1: Generate musique features using FeatureCalculator
        from core.calculators.static_feature_calculator import FeatureCalculator
        df_with_features = FeatureCalculator.calculate_all_features(race_df)
        
        # Step 2: Apply core feature preparation (embeddings + basic processing)
        processed_df = self.orchestrator.prepare_features(df_with_features, use_cache=False)
        
        # Step 3: Fit embeddings if needed
        if not self.orchestrator.embeddings_fitted:
            historical_sample = self.orchestrator.load_historical_races(limit=100, use_cache=True)
            self.orchestrator.fit_embeddings(historical_sample, use_cache=True)
        
        # Step 4: Apply embeddings  
        embedded_df = self.orchestrator.apply_embeddings(processed_df, use_cache=False)
        
        # Step 5: Create feature matrix matching EXACTLY the 72 expected features
        X_rf = pd.DataFrame(index=race_df.index)
        
        for feature_name in expected_features:
            if feature_name in embedded_df.columns:
                X_rf[feature_name] = embedded_df[feature_name]
            else:
                # Feature missing - fill with appropriate default
                if 'emb_' in feature_name:
                    X_rf[feature_name] = 0.0  # Embedding defaults
                elif feature_name in ['year', 'month', 'dayofweek']:
                    # Temporal features
                    if feature_name == 'year':
                        X_rf[feature_name] = 2025
                    elif feature_name == 'month':
                        X_rf[feature_name] = datetime.now().month
                    else:  # dayofweek
                        X_rf[feature_name] = datetime.now().weekday()
                elif feature_name in ['course_code', 'corde_code']:
                    X_rf[feature_name] = 1  # Code defaults
                else:
                    # Other missing features - use 0 as default
                    X_rf[feature_name] = 0.0
                    
                if self.verbose and feature_name not in embedded_df.columns:
                    print(f"Warning: Missing feature '{feature_name}', using default")
        
        # Ensure correct column order matching training
        X_rf = X_rf[expected_features]
        
        if self.verbose:
            print(f"RF data prepared: {X_rf.shape} (matches expected {len(expected_features)} features)")
            missing_features = [f for f in expected_features if f not in embedded_df.columns]
            if missing_features:
                print(f"Defaulted {len(missing_features)} missing features: {missing_features[:5]}...")
        
        return X_rf

# LSTM data preparation removed - no longer needed

# LSTM historical data preparation removed - no longer needed

    def prepare_tabnet_data(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data specifically for TabNet model using EXACT same workflow as TabNetTrainer.
        This replicates the EXACT sequence of calls that TabNetTrainer.prepare_features_for_training() uses.
        """
        if self.verbose:
            print(f"Preparing TabNet data using EXACT TabNetTrainer sequence...")

        # EXACT Step 1: Apply FeatureCalculator (same as TabNetTrainer line 112)
        from core.calculators.static_feature_calculator import FeatureCalculator
        df_with_features = FeatureCalculator.calculate_all_features(race_df)

        # EXACT Step 2: Use orchestrator's prepare_tabnet_features (same as TabNetTrainer line 116)
        complete_df = self.orchestrator.prepare_tabnet_features(
            df_with_features,
            use_cache=False
        )

        # EXACT Step 3: Use TabNetTrainer._select_tabnet_features logic (same as TabNetTrainer line 141)
        if self.tabnet_feature_columns and len(self.tabnet_feature_columns) > 0:
            # Use the EXACT feature columns that were saved during training
            available_features = [col for col in self.tabnet_feature_columns if col in complete_df.columns]
            missing_features = [col for col in self.tabnet_feature_columns if col not in complete_df.columns]
            
            if missing_features:
                if self.verbose:
                    print(f"Missing features from training: {missing_features}")
                    print(f"Available columns in complete_df: {sorted(complete_df.columns)}")
                    
            if len(available_features) < len(self.tabnet_feature_columns):
                if self.verbose:
                    print(f"WARNING: Only {len(available_features)}/{len(self.tabnet_feature_columns)} training features available")
                    print(f"Missing: {missing_features}")
                    
            # EXACT Step 4: Create feature matrix (same as TabNetTrainer line 155)
            X_tabnet = complete_df[available_features].copy()
            
        else:
            if self.verbose:
                print("No saved feature columns - this should not happen in production!")
            # Emergency fallback - recreate the feature selection from TabNetTrainer
            feature_columns = self._recreate_tabnet_feature_selection(complete_df)
            available_features = [col for col in feature_columns if col in complete_df.columns]
            X_tabnet = complete_df[available_features].copy()

        # EXACT Step 5: Fill NaN values with 0 (same as training)
        X_tabnet = X_tabnet.fillna(0)

        if self.verbose:
            print(f"TabNet data prepared: {X_tabnet.shape} - features: {list(X_tabnet.columns)}")

        return X_tabnet

    def _select_tabnet_features_for_prediction(self, df: pd.DataFrame) -> list:
        """
        Select appropriate features for TabNet prediction using the EXACT same logic as TabNetTrainer.
        This ensures perfect feature alignment between training and prediction.
        """
        # Musique-derived features (performance statistics)
        musique_features = [
            col for col in df.columns 
            if any(prefix in col for prefix in ['che_global_', 'che_weighted_', 'che_bytype_', 
                                               'joc_global_', 'joc_weighted_', 'joc_bytype_'])
        ]

        # Static race features (same as TabNet training)
        static_features = [
            'age', 'dist', 'temperature', 'cotedirect', 'corde', 
            'typec_code', 'natpis_code', 'meteo_code', 'partant', 'forceVent', 'directionVent', 'nebulosite'
        ]

        # Performance statistics (same as TabNet training)
        performance_features = [
            'victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere',
            'ratio_victoires', 'ratio_places', 'gains_par_course',
            'pourcVictChevalHippo', 'pourcPlaceChevalHippo', 'gainsAnneeEnCours',
            'pourcVictJockHippo', 'pourcPlaceJockHippo',
            'efficacite_couple', 'regularite_couple', 'progression_couple',
            'perf_cheval_hippo', 'perf_jockey_hippo'
        ]

        # Temporal features (same as TabNet training)
        temporal_features = ['year', 'month', 'dayofweek']

        # Combine all feature types (exact same order as training)
        all_features = musique_features + static_features + performance_features + temporal_features
        
        # Filter to only include features that exist in the dataframe
        available_features = [col for col in all_features if col in df.columns]
        
        if self.verbose:
            print(f"Selected {len(available_features)} features for TabNet prediction:")
            print(f"  - Musique features: {len([f for f in available_features if any(p in f for p in ['che_', 'joc_'])])}")
            print(f"  - Static features: {len([f for f in available_features if f in static_features])}")
            print(f"  - Performance features: {len([f for f in available_features if f in performance_features])}")
            print(f"  - Temporal features: {len([f for f in available_features if f in temporal_features])}")
            
            # Debug: Show some actual feature names for troubleshooting
            if len(available_features) < 20:
                print(f"  - All features: {available_features}")
            else:
                print(f"  - Sample features: {available_features[:10]}...")

        return available_features

    def _recreate_tabnet_feature_selection(self, df: pd.DataFrame) -> list:
        """
        Emergency fallback to recreate the feature selection from TabNetTrainer.
        This should match TabNetTrainer._select_tabnet_features() exactly.
        """
        # Musique-derived features (performance statistics)
        musique_features = [
            col for col in df.columns 
            if any(prefix in col for prefix in ['che_global_', 'che_weighted_', 'che_bytype_', 
                                               'joc_global_', 'joc_weighted_', 'joc_bytype_'])
        ]

        # Static race features (match TabNetTrainer exactly)
        static_features = [
            'age', 'dist', 'temperature', 'cotedirect', 'corde', 
            'typec', 'natpis', 'meteo', 'nbprt', 'forceVent', 'directionVent', 'nebulosite'
        ]

        # Performance statistics
        performance_features = [
            'victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere',
            'ratio_victoires', 'ratio_places', 'gains_par_course',
            'pourcVictChevalHippo', 'pourcPlaceChevalHippo', 'gainsAnneeEnCours',
            'pourcVictJockHippo', 'pourcPlaceJockHippo',
            'efficacite_couple', 'regularite_couple', 'progression_couple',
            'perf_cheval_hippo', 'perf_jockey_hippo'
        ]

        # Temporal features
        temporal_features = ['year', 'month', 'dayofweek']

        # CONSERVATIVE APPROACH: The saved scaler expects exactly 9 features
        # Instead of recreating the full comprehensive list, limit to core features
        core_features = [
            'age', 'dist', 'temperature', 'cotedirect',  # Essential static features
            'victoirescheval', 'placescheval', 'coursescheval',  # Core performance
            'pourcVictChevalHippo', 'pourcPlaceChevalHippo'  # Essential percentages
        ]
        
        # Filter to only include features that exist in the dataframe and limit to 9
        available_features = [col for col in core_features if col in df.columns][:9]
        
        if self.verbose:
            print(f"Recreated TabNet features (conservative): {len(available_features)} features")
            print(f"Features: {available_features}")
            
        return available_features

    def predict_with_rf(self, race_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the RF model with RF-specific data preparation."""
        if self.rf_model is None:
            raise ValueError("RF model not loaded")
            
        # Prepare data specifically for RF using EXACT training pipeline
        X_rf = self.prepare_rf_data(race_df)
        
        if self.verbose:
            print(f"RF prediction input: {X_rf.shape} features")
            print(f"RF feature sample: {list(X_rf.columns)[:10]}...")
            
        # X_rf is already properly formatted by prepare_rf_data (using exact training pipeline)
        # No additional cleaning needed since extract_rf_features already did it
        predictions = self.rf_model.predict(X_rf)
        
        if self.verbose:
            print(f"RF prediction: {len(predictions)} predictions generated")
        
        return predictions

    def predict_with_lstm(self, race_df: pd.DataFrame) -> np.ndarray:
        """LSTM predictions removed - always returns None."""
        return None


    def _load_tabnet_model(self):
        """Load TabNet model and associated files with fallback to previous model directories."""
        self.tabnet_model = None
        self.tabnet_scaler = None
        self.tabnet_feature_columns = None
        self.tabnet_fallback_path = None
        
        if not TABNET_AVAILABLE:
            if self.verbose:
                print("TabNet not available - install pytorch-tabnet")
            return
        
        try:
            # First try current model path
            tabnet_success = self._try_load_tabnet_from_path(self.model_path)
            
            if not tabnet_success:
                # Fallback: search for TabNet in previous model directories
                if self.verbose:
                    print("TabNet not found in current model directory, searching previous models...")
                
                # Get model base directory and search for TabNet models
                models_base_dir = self.model_path.parent.parent
                tabnet_success = self._fallback_to_previous_tabnet_models(models_base_dir)
                    
            if not tabnet_success and self.verbose:
                print("TabNet model not found in any available model directory")
                    
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load TabNet model: {str(e)}")
            self.tabnet_model = None
            self.tabnet_scaler = None
            self.tabnet_feature_columns = None
            
    def _try_load_tabnet_from_path(self, model_path: Path) -> bool:
        """Try to load TabNet model from a specific path with enhanced fallback mechanisms."""
        try:
            # Try to find TabNet model in the same directory first
            tabnet_path = model_path / "tabnet_model.zip"
            tabnet_path_double = model_path / "tabnet_model.zip"  # Handle double extension
            scaler_path = model_path / "tabnet_scaler.joblib"
            config_path = model_path / "tabnet_config.json"

            # Check for double extension first, then single
            if tabnet_path_double.exists():
                tabnet_path = tabnet_path_double
                if self.verbose:
                    print(f"Found TabNet model with double extension: {tabnet_path}")

            # If not found, look in sibling directories of the same date
            if not tabnet_path.exists():
                date_dir = model_path.parent
                for subdir in date_dir.iterdir():
                    if subdir.is_dir():
                        potential_tabnet = subdir / "tabnet_model.zip"
                        potential_tabnet_zip = subdir / "tabnet_model.zip"  # Handle double extension
                        potential_scaler = subdir / "tabnet_scaler.joblib"
                        potential_config = subdir / "tabnet_config.json"

                        if potential_tabnet.exists() and potential_scaler.exists():
                            tabnet_path = potential_tabnet
                            scaler_path = potential_scaler
                            config_path = potential_config
                            break
                        elif potential_tabnet_zip.exists() and potential_scaler.exists():
                            tabnet_path = potential_tabnet_zip
                            scaler_path = potential_scaler
                            config_path = potential_config
                            break
            
            # Load TabNet model if found (scaler and config are optional)
            if tabnet_path.exists():
                try:
                    self.tabnet_model = TabNetRegressor()
                    self.tabnet_model.load_model(str(tabnet_path))

                    # Load scaler if available, create new one if missing
                    if scaler_path.exists():
                        self.tabnet_scaler = joblib.load(scaler_path)
                        if self.verbose:
                            print(f"  Loaded TabNet scaler: {scaler_path}")
                    else:
                        self.tabnet_scaler = None
                        if self.verbose:
                            print(f"  TabNet scaler not found, will create new one during prediction")

                    # Load feature configuration if available, use defaults if missing
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            self.tabnet_config = json.load(f)
                        self.tabnet_feature_columns = self.tabnet_config.get('feature_columns', [])
                        if self.verbose:
                            print(f"  Loaded TabNet config: {len(self.tabnet_feature_columns)} features")
                    else:
                        self.tabnet_config = {}
                        self.tabnet_feature_columns = []
                        if self.verbose:
                            print(f"  TabNet config not found, will use dynamic feature selection")
                    
                    # Track if we used a fallback path
                    if model_path != self.model_path:
                        self.tabnet_fallback_path = str(model_path)
                            
                    if self.verbose:
                        source = f" (fallback from {model_path})" if model_path != self.model_path else ""
                        print(f"✅ Loaded TabNet model from: {tabnet_path}{source}")
                    
                    return True
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to load TabNet model: {str(e)}")
                    self.tabnet_model = None
                    self.tabnet_scaler = None
                    self.tabnet_feature_columns = None
                    return False
            
            return False
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to load TabNet from {model_path}: {str(e)}")
            return False
            
    def _fallback_to_previous_tabnet_models(self, models_base_dir: Path) -> bool:
        """Search for TabNet models in previous model directories, starting with most recent."""
        try:
            # Get all model directories sorted by date (most recent first)
            model_dirs = []
            for date_dir in models_base_dir.iterdir():
                if date_dir.is_dir():
                    for model_dir in date_dir.iterdir():
                        if model_dir.is_dir():
                            # Check if this directory has TabNet files
                            has_tabnet = (model_dir / "tabnet_model.zip").exists() or \
                                        (model_dir / "tabnet_model.zip").exists()
                            if has_tabnet:
                                model_dirs.append(model_dir)
            
            # Sort by modification time (most recent first)
            model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Try loading from each directory until success
            for model_dir in model_dirs:
                if self.verbose:
                    print(f"Trying TabNet fallback from: {model_dir}")
                
                if self._try_load_tabnet_from_path(model_dir):
                    return True
                    
            return False
            
        except Exception as e:
            if self.verbose:
                print(f"Error during TabNet fallback search: {str(e)}")
            return False

    def prepare_tabnet_features(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features specifically for TabNet prediction."""
        if self.verbose:
            print("Preparing features for TabNet prediction...")
            
        # Apply feature calculator for musique preprocessing
        df_with_features = FeatureCalculator.calculate_all_features(race_df)
        
        # Prepare TabNet-specific features using the orchestrator
        tabnet_df = self.orchestrator.prepare_tabnet_features(df_with_features, use_cache=False)
        
        return tabnet_df
    
    def predict_with_tabnet(self, race_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the TabNet model."""
        if self.tabnet_model is None:
            return None
            
        try:
            # Prepare TabNet-specific data
            X_tabnet = self.prepare_tabnet_data(race_df)
            
            # Scale features if scaler available, create new scaler if needed
            if self.tabnet_scaler is not None:
                X_scaled = self.tabnet_scaler.transform(X_tabnet)
            else:
                # Create and fit new scaler if missing
                from sklearn.preprocessing import StandardScaler
                self.tabnet_scaler = StandardScaler()
                X_scaled = self.tabnet_scaler.fit_transform(X_tabnet)
                if self.verbose:
                    print(f"Created new TabNet scaler during prediction")

            # Generate predictions
            tabnet_preds = self.tabnet_model.predict(X_scaled.astype(np.float32))
            tabnet_preds = tabnet_preds.flatten()

            if self.verbose:
                print(f"TabNet prediction: {len(tabnet_preds)} predictions generated")

            return tabnet_preds
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: TabNet prediction failed: {str(e)}")
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

        # Create clean copies for each model to avoid DataFrame contamination
        race_df_clean = race_df.copy()

        # Generate predictions from RF and TabNet models (LSTM removed)
        rf_predictions = self.predict_with_rf(race_df_clean.copy())
        tabnet_predictions = self.predict_with_tabnet(race_df_clean.copy())
        
        # Step 5: Create race metadata for blending
        race_metadata = self._extract_race_metadata(race_df)
        
        # Prepare predictions dictionary for blending (LSTM removed)
        predictions_dict = {
            'rf': rf_predictions,
            'lstm': np.zeros_like(rf_predictions),  # LSTM removed - use zeros
            'tabnet': tabnet_predictions if tabnet_predictions is not None else np.zeros_like(rf_predictions)
        }
        
        # Handle available predictions (LSTM removed)
        available_models = []
        if rf_predictions is not None:
            available_models.append('rf')
        if tabnet_predictions is not None and len(tabnet_predictions) == len(rf_predictions):
            available_models.append('tabnet')
            
        # Use PredictionBlender for race-specific blending
        try:
            blending_result = self.blender.blend_predictions(predictions_dict, race_metadata)
            final_predictions = blending_result['blended_predictions']
            weights_used = blending_result['weights_used']
            applied_rule = blending_result['applied_rule']
            
            if self.verbose:
                print(f"Applied blending rule: {applied_rule}")
                print(f"Weights used: RF={weights_used['rf']:.3f}, LSTM={weights_used['lstm']:.3f}, TabNet={weights_used['tabnet']:.3f}")
                print(f"Available models: {', '.join(available_models)}")
                
        except Exception as e:
            # Fallback to blender's default weights if race-specific blending fails
            if self.verbose:
                print(f"Warning: PredictionBlender failed ({str(e)}), using default blend weights")
            final_predictions = self._apply_default_blend_weights(predictions_dict, available_models)
            applied_rule = "default_blend_weights_fallback"
            weights_used = self.blender.blending_config['default']

        # Step 7: Create result DataFrame
        result_df = race_df.copy()
        result_df['predicted_position'] = final_predictions
        
        # Add individual model predictions for analysis (LSTM removed)
        result_df['rf_prediction'] = rf_predictions
        result_df['lstm_prediction'] = None  # LSTM removed
        result_df['tabnet_prediction'] = tabnet_predictions if tabnet_predictions is not None else None
        
        # Add blending information
        result_df['blending_rule_applied'] = applied_rule
        result_df['rf_weight_used'] = weights_used['rf']
        result_df['lstm_weight_used'] = weights_used['lstm'] 
        result_df['tabnet_weight_used'] = weights_used['tabnet']
        
        # Calculate confidence score based on model agreement (LSTM removed)
        if tabnet_predictions is not None:
            # Use RF-TabNet agreement for confidence
            model_predictions = np.column_stack([rf_predictions, tabnet_predictions])
            prediction_std = np.std(model_predictions, axis=1)
            # Convert to confidence: lower std = higher confidence (scale 0.1 to 0.9)
            max_std = np.max(prediction_std) if np.max(prediction_std) > 0 else 1.0
            confidence_scores = 0.9 - (prediction_std / max_std) * 0.8
            result_df['ensemble_confidence_score'] = np.clip(confidence_scores, 0.1, 0.9)
        else:
            # Single model (RF only) - use moderate confidence
            result_df['ensemble_confidence_score'] = 0.7

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

    def _load_optimal_blend_weights(self):
        """Load optimal blend weights from config or use tested defaults."""
        try:
            blend_config = self.config._config.blend
            self.optimal_rf_weight = blend_config.rf_weight
            self.optimal_lstm_weight = blend_config.lstm_weight
            self.optimal_tabnet_weight = blend_config.tabnet_weight
            
            # Validate weights sum to 1
            total_weight = self.optimal_rf_weight + self.optimal_lstm_weight + self.optimal_tabnet_weight
            if abs(total_weight - 1.0) > 1e-6:
                if self.verbose:
                    print(f"Warning: Blend weights don't sum to 1.0: {total_weight}, normalizing...")
                # Normalize weights
                self.optimal_rf_weight /= total_weight
                self.optimal_lstm_weight /= total_weight
                self.optimal_tabnet_weight /= total_weight
                
        except AttributeError:
            # Fallback to tested optimal weights if not in config
            if self.verbose:
                print("Using default optimal blend weights: 80/10/10")
            self.optimal_rf_weight = 0.8
            self.optimal_lstm_weight = 0.1
            self.optimal_tabnet_weight = 0.1
    
    def _apply_default_blend_weights(self, predictions_dict: Dict[str, np.ndarray], available_models: List[str]) -> np.ndarray:
        """Apply blender's default weights, adjusting for missing models."""
        default_weights = self.blender.blending_config['default']
        rf_predictions = predictions_dict.get('rf')
        lstm_predictions = predictions_dict.get('lstm')
        tabnet_predictions = predictions_dict.get('tabnet')
        
        # Adjust weights based on available models
        if len(available_models) == 3:
            # All models available - use default weights from config
            final_predictions = (
                default_weights['rf'] * rf_predictions +
                default_weights['lstm'] * lstm_predictions +
                default_weights['tabnet'] * tabnet_predictions
            )
        elif len(available_models) == 2:
            if 'rf' in available_models and 'lstm' in available_models:
                # RF + LSTM: normalize RF and LSTM weights
                total_weight = default_weights['rf'] + default_weights['lstm']
                rf_weight = default_weights['rf'] / total_weight
                lstm_weight = default_weights['lstm'] / total_weight
                final_predictions = rf_weight * rf_predictions + lstm_weight * lstm_predictions
            elif 'rf' in available_models and 'tabnet' in available_models:
                # RF + TabNet: normalize RF and TabNet weights
                total_weight = default_weights['rf'] + default_weights['tabnet']
                rf_weight = default_weights['rf'] / total_weight
                tabnet_weight = default_weights['tabnet'] / total_weight
                final_predictions = rf_weight * rf_predictions + tabnet_weight * tabnet_predictions
            else:
                # LSTM + TabNet: normalize LSTM and TabNet weights
                total_weight = default_weights['lstm'] + default_weights['tabnet']
                lstm_weight = default_weights['lstm'] / total_weight
                tabnet_weight = default_weights['tabnet'] / total_weight
                final_predictions = lstm_weight * lstm_predictions + tabnet_weight * tabnet_predictions
        else:
            # Single model - use that model's predictions
            if 'rf' in available_models:
                final_predictions = rf_predictions
            elif 'lstm' in available_models:
                final_predictions = lstm_predictions
            else:
                final_predictions = tabnet_predictions
        
        if self.verbose:
            weights_info = f"RF={default_weights['rf']:.1f}, LSTM={default_weights['lstm']:.1f}, TabNet={default_weights['tabnet']:.1f}"
            print(f"Using default blend weights: {weights_info} (adjusted for {len(available_models)} available models)")
        
        return final_predictions

    def _extract_race_metadata(self, race_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract race characteristics for blending rule selection.
        
        Args:
            race_df: DataFrame with race data
            
        Returns:
            Dictionary with race metadata
        """
        # Get race characteristics from first row (assuming all horses in same race)
        if len(race_df) == 0:
            return {}
            
        first_row = race_df.iloc[0]
        
        metadata = {}
        
        # Race type
        if 'typec' in race_df.columns:
            metadata['typec'] = first_row.get('typec', 'Plat')
            
        # Distance
        if 'dist' in race_df.columns:
            metadata['distance'] = pd.to_numeric(first_row.get('dist', 1600), errors='coerce')
            
        # Number of participants
        metadata['partant'] = len(race_df)
        
        # Additional metadata that might be useful
        if 'hippo' in race_df.columns:
            metadata['hippo'] = first_row.get('hippo', '')
            
        if 'natpis' in race_df.columns:
            metadata['natpis'] = first_row.get('natpis', '')
            
        if 'meteo' in race_df.columns:
            metadata['meteo'] = first_row.get('meteo', '')
            
        if self.verbose:
            print(f"Race metadata: {metadata}")
            
        return metadata

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        # Check which models are loaded (LSTM removed)
        models_loaded = {
            'rf': self.rf_model is not None,
            'lstm': False,  # LSTM removed
            'tabnet': self.tabnet_model is not None
        }
        
        # Generate model status details
        model_status = {}
        model_errors = {}
        
        # RF Model Status
        if self.rf_model is not None:
            model_status['rf'] = 'loaded'
        else:
            model_status['rf'] = 'missing'
            model_errors['rf'] = 'RF model file not found or failed to load'
        
        # LSTM Model Status - removed
        model_status['lstm'] = 'removed'
        model_errors['lstm'] = 'LSTM model removed from system'
            
        # TabNet Model Status
        if not TABNET_AVAILABLE:
            model_status['tabnet'] = 'unavailable'
            model_errors['tabnet'] = 'pytorch-tabnet not installed'
        elif self.tabnet_model is not None:
            if hasattr(self, 'tabnet_fallback_path') and self.tabnet_fallback_path:
                model_status['tabnet'] = 'loaded_fallback'
                model_errors['tabnet'] = f"Using TabNet from previous model: {self.tabnet_fallback_path}"
            else:
                model_status['tabnet'] = 'loaded'
        else:
            model_status['tabnet'] = 'missing'
            # Check specific TabNet files in current directory
            tabnet_files = {
                'model': self.model_path / "tabnet_model.zip",
                'model_alt': self.model_path / "tabnet_model.zip",
                'scaler': self.model_path / "tabnet_scaler.joblib", 
                'config': self.model_path / "tabnet_config.json"
            }
            existing_files = [name for name, path in tabnet_files.items() if path.exists()]
            missing_files = [name for name, path in tabnet_files.items() if not path.exists()]
            
            if not existing_files:
                model_errors['tabnet'] = f"No TabNet files found in current model directory. Use fallback mechanism to load from previous models."
            elif 'model' in existing_files or 'model_alt' in existing_files:
                # Have model file but missing scaler/config
                if 'scaler' not in existing_files and 'config' not in existing_files:
                    model_errors['tabnet'] = f"TabNet model found but missing scaler and config files. Will create new scaler during prediction."
                elif 'scaler' not in existing_files:
                    model_errors['tabnet'] = f"TabNet model and config found but missing scaler file. Will create new scaler during prediction."
                elif 'config' not in existing_files:
                    model_errors['tabnet'] = f"TabNet model and scaler found but missing config file. Will use dynamic feature selection."
                else:
                    model_errors['tabnet'] = f"TabNet files found but failed to load due to corruption or version mismatch. Existing: {', '.join(existing_files)}."
            else:
                model_errors['tabnet'] = f"TabNet files found but failed to load. Existing: {', '.join(existing_files)}. Missing critical model file."
        
        # Get available models
        available_models = [name for name, loaded in models_loaded.items() if loaded]
        
        # Get blending configuration
        default_weights = self.blender.blending_config['default']
        available_rules = self.blender.get_available_rules()
        
        return {
            'model_type': 'Enhanced RacePredictor (RF + TabNet) with Configurable Blending',
            'model_path': str(self.model_path),
            'blending_info': {
                'default_weights': default_weights,
                'available_rules': available_rules,
                'rule_count': len(available_rules)
            },
            'models_loaded': models_loaded,
            'model_status': model_status,
            'model_errors': model_errors,
            'available_models': available_models,
            'available_models_count': len(available_models),
            'tabnet_available': TABNET_AVAILABLE,
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
        'typec': 'Plat', 'dist': 1600, 'natpis': 'PSF', 'meteo': 'BEAU',
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