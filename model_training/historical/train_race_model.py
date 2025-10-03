import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import platform
import os

# Disable GPU on M1 processors to avoid hanging
if platform.processor() == 'arm' or 'arm64' in platform.machine().lower():
    print("[DEBUG-GPU] M1/ARM processor detected, disabling GPU for TensorFlow")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from model_training.regressions.isotonic_calibration import CalibratedRegressor
from utils.model_manager import ModelManager

# Import TabNet model
try:
    from model_training.tabnet.tabnet_model import TabNetModel
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False


class HorseRaceModel:
    """
    Enhanced horse race training orchestrator that trains RF and TabNet models
    using historical data. Focuses on the two best-performing models for optimal
    prediction accuracy.
    
    Optimized 2-model system:
    - Random Forest: Tree-based ensemble (R² = 0.2081, reliable baseline)
    - TabNet: Attention-based neural network (R² = 0.2118, best performance)
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = False):
        """Initialize the model with configuration."""
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        db_path = get_sqlite_dbpath(self.db_type)
        self.model_manager = ModelManager()
        # Get model configuration - FIX: Remove the config parameter
        #model_path = manager.get_model_path()
        #paths = manager.save_model_artifacts(
        #    model_config={"version": "1.0"},
        #    db_type= self.db_type
        #)

        # Initialize data orchestrator
        embedding_dim = self.config.get_default_embedding_dim()
        self.orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=db_path,
            verbose=verbose
        )

        # 2-model system configuration (RF + TabNet only)
        # Alternative models disabled - focus on best performers

        # Model containers (2-model system: RF + TabNet)
        self.rf_model = None
        self.tabnet_model = None
        self.training_results = None

        # Data containers
        self.rf_data = None
        self.lstm_data = None
        self.tabnet_data = None
        
        # 2-model system: TabNet is a core component, not alternative
        self.alt_models_enabled = True  # TabNet is always enabled in 2-model system

        self.log_info(f"Initialized HorseRaceModel with database: {self.db_type}")
        if TABNET_AVAILABLE and self.alt_models_enabled:
            self.log_info(f"TabNet model enabled: {self.alt_models_enabled}")
        else:
            self.log_info("TabNet model: Not available or not configured")

    def log_info(self, message):
        """Simple logging method."""
        if self.verbose:
            print(f"[HorseRaceModel] {message}")

    def load_and_prepare_data(self, limit: Optional[int] = None,
                              race_filter: Optional[str] = None,
                              date_filter: Optional[str] = None) -> Dict[str, Any]:
        """Load and prepare complete dataset once with memory-efficient batch processing."""

        # Check total records and decide on processing strategy
        total_records = self.orchestrator.get_total_record_count(race_filter, date_filter)
        if limit:
            total_records = min(total_records, limit)
            
        self.log_info(f"Processing strategy decision: {total_records:,} records")
        
        if self.orchestrator.should_use_batch_processing(total_records):
            self.log_info(f"Using memory-efficient batch processing for {total_records:,} records")
            
            # Load historical data with batch processing
            df_historical = self.orchestrator.load_historical_races_batched(
                limit=limit,
                race_filter=race_filter,
                date_filter=date_filter,
                use_cache=True
            )
            
            # Prepare complete dataset with batch processing
            self.log_info("Preparing complete feature set with batch processing...")
            self.complete_df = self.orchestrator.prepare_complete_dataset_batched(
                df_historical,
                use_cache=True
            )
            
        else:
            self.log_info("Using standard processing for smaller dataset")
            
            # Load historical data (standard way)
            df_historical = self.orchestrator.load_historical_races(
                limit=limit,
                race_filter=race_filter,
                date_filter=date_filter,
                use_cache=True
            )
            
            # Prepare complete dataset (standard way)
            self.log_info("Preparing complete feature set...")
            self.complete_df = self.orchestrator.prepare_complete_dataset(
                df_historical,
                use_cache=True
            )
        
        # Log memory usage summary if available
        memory_summary = self.orchestrator.get_memory_summary()
        if memory_summary:
            self.log_info(f"Memory usage summary: Peak {memory_summary.get('peak_memory_mb', 0):.1f}MB")

        self.log_info(
            f"Complete dataset prepared: {len(self.complete_df)} records, {len(self.complete_df.columns)} features")

        return {
            'status': 'success',
            'records': len(self.complete_df),
            'features': len(self.complete_df.columns),
            'memory_summary': memory_summary
#            'races': self.complete_df['comp'].nunique()
        }

    def train(self, limit: Optional[int] = None,
              race_filter: Optional[str] = None,
              date_filter: Optional[str] = None,
              test_size: float = 0.2,
              random_state: int = 42) -> Dict[str, Any]:
        """
        Complete training workflow: load data, train both models, generate predictions.
        """
        start_time = datetime.now()
        self.log_info("Starting complete training workflow...")

        # Step 1: Load and prepare data once
        data_prep_results = self.load_and_prepare_data(limit, race_filter, date_filter)

        # FIX: Check for complete_df instead of rf_data and lstm_data
        if not hasattr(self, 'complete_df') or self.complete_df is None or len(self.complete_df) == 0:
            raise ValueError("Data preparation failed - no data available for training")

        self.log_info("Splitting data for training and testing...")

        # Step 2: Extract RF features just before training
        X_rf, y_rf = self.orchestrator.extract_rf_features(self.complete_df)
        X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
            X_rf, y_rf, test_size=test_size, random_state=random_state
        )

        # Step 3: Train Random Forest model
        self.log_info("Training Random Forest model...")
        rf_results = self._train_rf_model(X_rf_train, y_rf_train, X_rf_test, y_rf_test)

        # Step 4: Train TabNet model (standalone with domain features)
        tabnet_results = {}
        if TABNET_AVAILABLE and self.alt_models_enabled:
            self.log_info("Training TabNet model with domain features...")
            print(f"[DEBUG-TABNET] Starting TabNet training with domain features...")
            tabnet_results = self._train_tabnet_model_domain(
                self.complete_df, test_size, random_state
            )
            print(f"[DEBUG-TABNET] TabNet training completed")
        else:
            print(f"[DEBUG-TABNET] TabNet skipped - Available: {TABNET_AVAILABLE}, Enabled: {'tabnet' in self.alt_models_enabled if self.alt_models_enabled else False}")

        # Step 5: Compile complete results for 2-model system
        training_time = (datetime.now() - start_time).total_seconds()

        self.training_results = {
            'status': 'success',
            'training_time': training_time,
            'model_type': '2_model_system',  # RF + TabNet (optimized)
            'data_preparation': data_prep_results,
            'rf_results': rf_results,
            'tabnet_results': tabnet_results,
            'training_config': {
                'test_size': test_size,
                'random_state': random_state,
                'db_type': self.db_type
            }
        }

        self.log_info(f"Training completed in {training_time:.2f} seconds")
        self.log_info(f"RF Test MAE: {rf_results['test_mae']:.4f}")
        
        # Log TabNet results if available
        if tabnet_results and tabnet_results.get('status') == 'success':
            self.log_info(f"TabNet Test MAE: {tabnet_results['test_mae']:.4f}")
        
        # 2-model system complete - RF and TabNet results logged above
        
        return self.training_results

    def _train_rf_model(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train Random Forest model with provided split data."""

        # Create calibrated RF model
        base_rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )

        self.rf_model = CalibratedRegressor(
            base_regressor=base_rf,
            clip_min=1.0  # Race positions start at 1
        )

        # Train the model with provided data
        self.rf_model.fit(X_train, y_train)

        # Generate predictions
        train_preds = self.rf_model.predict(X_train)
        test_preds = self.rf_model.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)

        # Get feature importance from the base RF model
        feature_importance = None
        feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        if hasattr(self.rf_model, 'base_regressor') and hasattr(self.rf_model.base_regressor, 'feature_importances_'):
            feature_importance = self.rf_model.base_regressor.feature_importances_.tolist()

        return {
            'model_type': 'RandomForest',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(X_train.columns) if hasattr(X_train, 'columns') else X_train.shape[1],
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'test_r2': float(test_r2),
            'test_predictions': test_preds.tolist(),
            'test_targets': y_test.tolist(),
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'model_params': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
            }
        }

    
    def _train_tabnet_model_domain(self, complete_df, test_size=0.2, random_state=42) -> Dict[str, Any]:
        """Train TabNet model using domain features (no embeddings)."""
        try:
            print(f"[DEBUG-TABNET] Initializing TabNet model with domain features...")
            
            # Extract TabNet-specific features using new pipeline
            X_tabnet, y_tabnet = self.orchestrator.extract_tabnet_features(complete_df)
            
            print(f"[DEBUG-TABNET] TabNet feature extraction completed")
            print(f"[DEBUG-TABNET] Features shape: {X_tabnet.shape}")
            print(f"[DEBUG-TABNET] Target shape: {y_tabnet.shape}")
            
            # Get TabNet config from legacy_alternative_models section
            import yaml
            try:
                with open('config.yaml', 'r') as f:
                    config_data = yaml.safe_load(f)
                legacy_models = config_data.get('legacy_alternative_models', {})
                tabnet_config = legacy_models.get('tabnet', {})
            except Exception as e:
                print(f"[DEBUG-TABNET] Could not load TabNet config: {e}, using defaults")
                tabnet_config = {}
            
            # Only override learning rate if specified, let TabNet use its defaults
            tabnet_params = {}
            if 'learning_rate' in tabnet_config:
                tabnet_params['optimizer_params'] = {'lr': tabnet_config['learning_rate']}
            
            tabnet_model = TabNetModel(verbose=self.verbose)
            
            # Split data for TabNet
            X_train, X_test, y_train, y_test = train_test_split(
                X_tabnet, y_tabnet, test_size=test_size, random_state=random_state
            )
            
            print(f"[DEBUG-TABNET] Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
            print(f"[DEBUG-TABNET] Test set: {X_test.shape[0]:,} samples")
            
            # Scale the data for TabNet manually since we're using pre-processed data
            print(f"[DEBUG-TABNET] Scaling features for TabNet...")
            from sklearn.preprocessing import StandardScaler
            tabnet_model.scaler = StandardScaler()
            X_train_scaled = tabnet_model.scaler.fit_transform(X_train)
            X_test_scaled = tabnet_model.scaler.transform(X_test)
            
            # Set feature columns for the TabNet model (needed for saving)
            # Use actual feature names from domain features
            feature_columns = list(X_tabnet.columns) if hasattr(X_tabnet, 'columns') else [f"feature_{i}" for i in range(X_tabnet.shape[1])]
            tabnet_model.feature_columns = feature_columns
            
            print(f"[DEBUG-TABNET] TabNet features: {feature_columns}")
            
            # Train TabNet using its internal _train_tabnet_model method with scaled data
            print(f"[DEBUG-TABNET] Starting TabNet training...")
            training_result = tabnet_model._train_tabnet_model(
                X_train_scaled, y_train, X_test_scaled, y_test, tabnet_params=tabnet_params
            )
            
            # Set training results for saving
            tabnet_model.training_results = training_result
            
            # Make predictions using trained model on scaled test data
            print(f"[DEBUG-TABNET] Generating TabNet predictions...")
            predictions = tabnet_model.tabnet_model.predict(X_test_scaled)
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            print(f"[DEBUG-TABNET] === TABNET DOMAIN FEATURES PERFORMANCE ===")
            print(f"[DEBUG-TABNET] Features used: {len(feature_columns)} domain features")
            print(f"[DEBUG-TABNET] Test MAE: {mae:.4f}")
            print(f"[DEBUG-TABNET] Test RMSE: {rmse:.4f}")
            print(f"[DEBUG-TABNET] Test R²: {r2:.4f}")
            print(f"[DEBUG-TABNET] ==========================================")
            
            # Store TabNet model
            self.tabnet_model = tabnet_model
            
            return {
                'status': 'success',
                'model_type': 'TabNet_Domain',
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': len(feature_columns),
                'features': len(feature_columns),  # Compatibility field
                'n_features': len(feature_columns),  # Compatibility field
                'n_epochs': training_result.get('n_epochs', 'N/A'),  # Pass through epoch info
                'feature_names': feature_columns,
                'test_mae': float(mae),
                'test_rmse': float(rmse),
                'test_r2': float(r2),
                'training_result': training_result,
                'test_predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                'test_targets': y_test.tolist()
            }
            
        except Exception as e:
            print(f"[DEBUG-TABNET] TabNet training failed: {e}")
            import traceback
            print(f"[DEBUG-TABNET] Full error: {traceback.format_exc()}")
            return {
                'status': 'failed',
                'message': str(e)
            }

    def _train_tabnet_model_old(self, X_sequences, X_static, y, test_size=0.2, random_state=42) -> Dict[str, Any]:
        """Train TabNet model as standalone component."""
        try:
            print(f"[DEBUG-TABNET] Initializing TabNet model...")
            
            # Get TabNet config - use minimal override to avoid conflicts
            tabnet_config = self.alt_models_config.get('tabnet', {})
            
            # Only override learning rate if specified, let TabNet use its defaults
            tabnet_params = {}
            if 'learning_rate' in tabnet_config:
                tabnet_params['optimizer_params'] = {'lr': tabnet_config['learning_rate']}
            
            tabnet_model = TabNetModel(verbose=self.verbose)
            
            # Prepare TabNet data (requires flattened sequential + static)
            print(f"[DEBUG-TABNET] Preparing TabNet data...")
            print(f"[DEBUG-TABNET] Input shapes: X_seq={X_sequences.shape}, X_static={X_static.shape}, y={y.shape}")
            
            # Flatten sequential data and combine with static
            X_seq_flat = X_sequences.reshape(X_sequences.shape[0], -1)
            X_tabnet_combined = np.concatenate([X_seq_flat, X_static], axis=1)
            
            print(f"[DEBUG-TABNET] TabNet input shape: {X_tabnet_combined.shape}")
            
            # Split data for TabNet (using the same split as other models)
            X_train, X_test, y_train, y_test = train_test_split(
                X_tabnet_combined, y, test_size=test_size, random_state=random_state
            )
            
            print(f"[DEBUG-TABNET] Training set: {X_train.shape[0]:,} samples")
            print(f"[DEBUG-TABNET] Test set: {X_test.shape[0]:,} samples")
            
            # Scale the data for TabNet manually since we're using pre-processed data
            print(f"[DEBUG-TABNET] Scaling features for TabNet...")
            from sklearn.preprocessing import StandardScaler
            tabnet_model.scaler = StandardScaler()
            X_train_scaled = tabnet_model.scaler.fit_transform(X_train)
            X_test_scaled = tabnet_model.scaler.transform(X_test)
            
            # Set feature columns for the TabNet model (needed for saving)
            # Use the same features as we have in the flattened data
            feature_columns = [f"feature_{i}" for i in range(X_train.shape[1])]
            tabnet_model.feature_columns = feature_columns
            
            # Train TabNet using its internal _train_tabnet_model method with scaled data
            print(f"[DEBUG-TABNET] Starting TabNet training...")
            training_result = tabnet_model._train_tabnet_model(
                X_train_scaled, y_train, X_test_scaled, y_test, tabnet_params=tabnet_params
            )
            
            # Set training results for saving
            tabnet_model.training_results = training_result
            
            # Make predictions using trained model on scaled test data
            print(f"[DEBUG-TABNET] Generating TabNet predictions...")
            predictions = tabnet_model.tabnet_model.predict(X_test_scaled)
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            print(f"[DEBUG-TABNET] === TABNET PERFORMANCE ===")
            print(f"[DEBUG-TABNET] Test MAE: {mae:.4f}")
            print(f"[DEBUG-TABNET] Test RMSE: {rmse:.4f}")
            print(f"[DEBUG-TABNET] Test R²: {r2:.4f}")
            print(f"[DEBUG-TABNET] ========================")
            
            # Store TabNet model
            self.tabnet_model = tabnet_model
            
            return {
                'status': 'success',
                'model_type': 'TabNet',
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'test_mae': float(mae),
                'test_rmse': float(rmse),
                'test_r2': float(r2),
                'training_result': training_result,
                'test_predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                'test_targets': y_test.tolist()
            }
            
        except Exception as e:
            print(f"[DEBUG-TABNET] TabNet training failed: {e}")
            import traceback
            print(f"[DEBUG-TABNET] Full error: {traceback.format_exc()}")
            return {
                'status': 'failed',
                'message': str(e)
            }
    


    def save_models(self, orchestrator=None):
        """
        Save all trained models (RF and TabNet models).

        Args:
            orchestrator: Orchestrator with feature state (optional)

        Returns:
            Dictionary with paths to saved artifacts
        """
        from utils.model_manager import get_model_manager

        print("===== SAVING ALL TRAINED MODELS =====")

        # Prepare feature state if orchestrator provided
        feature_state = None
        if orchestrator and hasattr(orchestrator, 'preprocessing_params'):
            feature_state = {
                'preprocessing_params': orchestrator.preprocessing_params,
                'embedding_dim': getattr(orchestrator, 'embedding_dim', 16)
            }

        # Get the model manager and save models
        model_manager = get_model_manager()
        saved_paths = model_manager.save_models(
            rf_model=self.rf_model,
            feature_state=feature_state
        )
        
        # Save TabNet model separately (standalone)
        if self.tabnet_model is not None:
            try:
                print("Saving TabNet model...")
                tabnet_saved_paths = self.tabnet_model.save_model()
                saved_paths['tabnet_model'] = tabnet_saved_paths
                print(f"✅ TabNet model saved: {tabnet_saved_paths}")
            except Exception as e:
                print(f"❌ Failed to save TabNet model: {e}")
                saved_paths['tabnet_model'] = f"Failed: {e}"
        

        print(f"All models saved successfully")
        return saved_paths


def main(progress_callback=None):
    """
    Main function to train the hybrid model from IDE.
    Modify the parameters below as needed.
    """
    if progress_callback:
        progress_callback(5, "Initializing model...")

    # Initialize the model
    model = HorseRaceModel(verbose=False)

    if progress_callback:
        progress_callback(10, "Loading and preparing data...")

    # Train the model with your desired parameters
    results = model.train(
        limit=None,  # Set to a number like 1000 to limit races for testing
        race_filter=None,  # Set to 'A' for Attele, 'P' for Plat, etc.
        date_filter=None,  # Set to "jour > '2023-01-01'" to limit date range
        test_size=0.2,  # 20% for testing
        random_state=42  # For reproducible results
    )

    if progress_callback:
        progress_callback(90, "Saving trained models...")

    # Save all trained models
    saved_paths = model.save_models(model.orchestrator)

    if progress_callback:
        progress_callback(100, "Training completed successfully!")

    # Print detailed summary results
    print("\n" + "=" * 60)
    print("MULTI-MODEL TRAINING COMPLETED")
    print("=" * 60)
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Model type: {results['model_type']}")
    
    # RF model results
    print(f"\n--- Random Forest Results ---")
    rf_results = results['rf_results']
    print(f"Features used: {rf_results['features']}")
    print(f"Train samples: {rf_results['train_samples']}")
    print(f"Test samples: {rf_results['test_samples']}")
    print(f"Test MAE: {rf_results['test_mae']:.4f}")
    print(f"Test RMSE: {rf_results['test_rmse']:.4f}")
    print(f"Test R²: {rf_results['test_r2']:.4f}")
    
    # TabNet model results
    tabnet_results = results.get('tabnet_results', {})
    if tabnet_results and tabnet_results.get('status') == 'success':
        print(f"\n--- TabNet Results ---")
        print(f"Train samples: {tabnet_results.get('train_samples', 'N/A')}")
        print(f"Test samples: {tabnet_results.get('test_samples', 'N/A')}")
        print(f"Features: {tabnet_results.get('features_used', tabnet_results.get('features', 'N/A'))}")
        print(f"Test MAE: {tabnet_results['test_mae']:.4f}")
        print(f"Test RMSE: {tabnet_results['test_rmse']:.4f}")
        print(f"Test R²: {tabnet_results['test_r2']:.4f}")
        print(f"Training epochs: {tabnet_results.get('n_epochs', 'N/A')}")
    
    # Alternative model results
    alt_results = results.get('alternative_models', {})
    if alt_results:
        print(f"\n--- Alternative Models Results ---")
        for model_name, model_result in alt_results.items():
            if model_result.get('status') == 'success':
                evaluation = model_result.get('evaluation', {})
                print(f"{model_name.title()}: MAE={evaluation.get('mae', 'N/A'):.4f}, "
                      f"RMSE={evaluation.get('rmse', 'N/A'):.4f}, "
                      f"Top3 Acc={evaluation.get('top3_accuracy', 'N/A'):.3f}")
            else:
                print(f"{model_name.title()}: {model_result.get('message', 'Failed')}")
    else:
        print(f"\n--- Alternative Models: Not enabled ---")
    
    print(f"\nModel saved to: {saved_paths.get('rf_model', 'N/A')}")
    
    # Display top feature importance for RF model
    if results['rf_results'].get('feature_importance') and results['rf_results'].get('feature_names'):
        print("\nTop 10 Most Important Features (Random Forest):")
        print("-" * 50)
        feature_importance = results['rf_results']['feature_importance']
        feature_names = results['rf_results']['feature_names']
        
        # Create list of (importance, name) pairs and sort by importance
        importance_pairs = list(zip(feature_importance, feature_names))
        importance_pairs.sort(key=lambda x: x[0], reverse=True)
        
        for i, (importance, name) in enumerate(importance_pairs[:10]):
            print(f"{name}: {importance:.4f}")


if __name__ == "__main__":
    main()