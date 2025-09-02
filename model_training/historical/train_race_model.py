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

# Import alternative models
try:
    from model_training.models import FeedforwardModel, TransformerModel, EnsembleModel
    ALTERNATIVE_MODELS_AVAILABLE = True
except ImportError:
    ALTERNATIVE_MODELS_AVAILABLE = False


class HorseRaceModel:
    """
    Enhanced horse race training orchestrator that trains RF, LSTM, TabNet, and alternative models
    (Feedforward, Transformer, Ensemble) using historical data. Focuses on individual model training
    without blending - blending is handled in the prediction pipeline.
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

        # Alternative models configuration
        self.alt_models_config = getattr(self.config._config, 'alternative_models', {})
        self.alt_models_enabled = self.alt_models_config.get('model_selection', [])

        # Model containers
        self.rf_model = None
        self.lstm_model = None
        self.tabnet_model = None
        self.alternative_models = {}
        self.training_results = None

        # Data containers
        self.rf_data = None
        self.lstm_data = None
        self.tabnet_data = None

        self.log_info(f"Initialized HorseRaceModel with database: {self.db_type}")
        if ALTERNATIVE_MODELS_AVAILABLE and self.alt_models_enabled:
            self.log_info(f"Alternative models enabled: {self.alt_models_enabled}")
        else:
            self.log_info("Alternative models: Not available or not configured")

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

        # Step 3: Extract LSTM features just before training
        X_sequences, X_static, y_lstm = self.orchestrator.extract_lstm_features(self.complete_df)
        X_seq_train, X_seq_test, X_static_train, X_static_test, y_lstm_train, y_lstm_test = train_test_split(
            X_sequences, X_static, y_lstm, test_size=test_size, random_state=random_state
        )

        # Step 4: Train Random Forest model
        self.log_info("Training Random Forest model...")
        rf_results = self._train_rf_model(X_rf_train, y_rf_train, X_rf_test, y_rf_test)

        # Step 5: Train LSTM model
        self.log_info("Training LSTM model...")
        lstm_results = self._train_lstm_model(
            X_seq_train, X_static_train, y_lstm_train,
            X_seq_test, X_static_test, y_lstm_test
        )

        # Step 6: Train alternative models (if enabled)
        alternative_results = {}
        if ALTERNATIVE_MODELS_AVAILABLE and self.alt_models_enabled:
            self.log_info("Training alternative models...")
            print(f"[DEBUG-ALT] Starting alternative models training with models: {self.alt_models_enabled}")
            alternative_results = self._train_alternative_models(
                X_sequences, X_static, y_lstm, test_size, random_state
            )
            print(f"[DEBUG-ALT] Alternative models training completed")

        # Step 7: Compile complete results
        training_time = (datetime.now() - start_time).total_seconds()

        self.training_results = {
            'status': 'success',
            'training_time': training_time,
            'model_type': 'multi_model_ensemble',
            'data_preparation': data_prep_results,
            'rf_results': rf_results,
            'lstm_results': lstm_results,
            'alternative_models': alternative_results,
            'training_config': {
                'test_size': test_size,
                'random_state': random_state,
                'db_type': self.db_type,
                'alternative_models_enabled': self.alt_models_enabled
            }
        }

        self.log_info(f"Training completed in {training_time:.2f} seconds")
        self.log_info(f"RF Test MAE: {rf_results['test_mae']:.4f}")
        self.log_info(f"LSTM Test MAE: {lstm_results['test_mae']:.4f}")
        
        # Log alternative model results
        if alternative_results:
            for model_name, result in alternative_results.items():
                if result.get('status') == 'success':
                    mae = result.get('evaluation', {}).get('mae', 'N/A')
                    self.log_info(f"{model_name.title()} Test MAE: {mae}")
                else:
                    self.log_info(f"{model_name.title()}: {result.get('message', 'Training failed')}")

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

    def _train_lstm_model(self, X_seq_train, X_static_train, y_train,
                          X_seq_test, X_static_test, y_test) -> Dict[str, Any]:
        """Train LSTM model with provided split data."""

        # Create LSTM model
        lstm_model_components = self.orchestrator.create_hybrid_model(
            sequence_shape=X_seq_train.shape,
            static_shape=X_static_train.shape,
            lstm_units=64,
            dropout_rate=0.2
        )

        if lstm_model_components is None:
            self.log_info("LSTM model creation failed - TensorFlow may not be available")
            return {
                'status': 'failed',
                'error': 'LSTM model creation failed'
            }

        self.lstm_model = lstm_model_components['lstm']

        # Configure training callbacks with advanced plateau and overfitting management
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
        
        # Custom callback to monitor training progress and detect issues
        class TrainingMonitor(Callback):
            def __init__(self):
                super().__init__()
                self.epoch_start_time = None
                self.stalled_epochs = 0
                self.previous_loss = float('inf')
                
            def on_epoch_begin(self, epoch, logs=None):
                import time
                self.epoch_start_time = time.time()
                print(f"[DEBUG-LSTM] Starting epoch {epoch + 1}...")
                
            def on_epoch_end(self, epoch, logs=None):
                import time
                epoch_time = time.time() - self.epoch_start_time
                current_loss = logs.get('val_loss', logs.get('loss', 0))
                current_mae = logs.get('val_mae', logs.get('mae', 0))
                
                print(f"[DEBUG-LSTM] Epoch {epoch + 1} completed in {epoch_time:.2f}s")
                print(f"[DEBUG-LSTM] Loss: {current_loss:.4f}, MAE: {current_mae:.4f}")
                
                # Check for stalled training
                if abs(current_loss - self.previous_loss) < 0.0001:
                    self.stalled_epochs += 1
                    if self.stalled_epochs >= 3:
                        print(f"[DEBUG-LSTM] Warning: Training may be stalled (loss unchanged for {self.stalled_epochs} epochs)")
                else:
                    self.stalled_epochs = 0
                    
                self.previous_loss = current_loss
                
            def on_train_begin(self, logs=None):
                print("[DEBUG-LSTM] Training started successfully!")
                
            def on_train_end(self, logs=None):
                print("[DEBUG-LSTM] Training completed!")
        
        training_monitor = TrainingMonitor()
        
        callbacks = [
            # Add our custom training monitor first
            training_monitor,
            
            # Early stopping with improved patience and monitoring
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience for better convergence
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001  # Minimum change to qualify as improvement
            ),
            
            # Reduce learning rate when loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Reduce LR by half
                patience=5,   # Wait 5 epochs before reducing
                min_lr=1e-7,  # Minimum learning rate
                verbose=1,
                min_delta=0.001
            ),
            
            # Additional early stopping for overfitting detection
            EarlyStopping(
                monitor='val_mae',
                patience=20,  # Monitor MAE separately with higher patience
                restore_best_weights=False,  # Don't restore for this one
                verbose=0,
                min_delta=0.01,
                mode='min'
            )
        ]
        
        # Add model checkpointing if we can determine a save path
        try:
            from pathlib import Path
            model_checkpoint_path = Path("models") / "lstm_best_checkpoint.keras"
            model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            callbacks.append(
                ModelCheckpoint(
                    filepath=str(model_checkpoint_path),
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                    mode='min'
                )
            )
            print(f"[DEBUG-LSTM] Added model checkpoint: {model_checkpoint_path}")
        except Exception as e:
            print(f"[DEBUG-LSTM] Could not add model checkpoint: {e}")
        

        # Train the model with provided data
        print(f"[DEBUG-LSTM] Starting model training...")
        print(f"[DEBUG-LSTM] Training data shapes: X_seq={X_seq_train.shape}, X_static={X_static_train.shape}, y={y_train.shape}")
        print(f"[DEBUG-LSTM] Validation data shapes: X_seq={X_seq_test.shape}, X_static={X_static_test.shape}, y={y_test.shape}")
        print(f"[DEBUG-LSTM] Using batch_size=32, epochs=50...")
        
        history = self.lstm_model.fit(
            [X_seq_train, X_static_train], y_train,
            validation_data=([X_seq_test, X_static_test], y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=2  # Show progress per epoch
        )

        print(f"[DEBUG-LSTM] Training completed successfully!")
        print(f"[DEBUG-LSTM] Total epochs trained: {len(history.history['loss'])}")
        print(f"[DEBUG-LSTM] Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"[DEBUG-LSTM] Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"[DEBUG-LSTM] Best validation loss: {min(history.history['val_loss']):.4f}")
        
        # Check if early stopping was triggered
        if len(history.history['loss']) < 50:
            print(f"[DEBUG-LSTM] Early stopping triggered at epoch {len(history.history['loss'])}")
        
        # Generate predictions
        print(f"[DEBUG-LSTM] Generating predictions on training data...")
        train_preds = self.lstm_model.predict([X_seq_train, X_static_train], verbose=0)
        print(f"[DEBUG-LSTM] Training predictions shape: {train_preds.shape}")
        
        print(f"[DEBUG-LSTM] Generating predictions on test data...")
        test_preds = self.lstm_model.predict([X_seq_test, X_static_test], verbose=0)
        print(f"[DEBUG-LSTM] Test predictions shape: {test_preds.shape}")

        # Flatten predictions if needed
        print(f"[DEBUG-LSTM] Flattening predictions...")
        print(f"[DEBUG-LSTM] Pre-flatten shapes: train={train_preds.shape}, test={test_preds.shape}")
        train_preds = train_preds.flatten()
        test_preds = test_preds.flatten()
        print(f"[DEBUG-LSTM] Post-flatten shapes: train={train_preds.shape}, test={test_preds.shape}")

        # Calculate metrics
        print(f"[DEBUG-LSTM] Calculating performance metrics...")
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)
        
        print(f"[DEBUG-LSTM] === FINAL LSTM PERFORMANCE ===")
        print(f"[DEBUG-LSTM] Training MAE: {train_mae:.4f}")
        print(f"[DEBUG-LSTM] Test MAE: {test_mae:.4f}")
        print(f"[DEBUG-LSTM] Training RMSE: {train_rmse:.4f}")
        print(f"[DEBUG-LSTM] Test RMSE: {test_rmse:.4f}")
        print(f"[DEBUG-LSTM] Test R²: {test_r2:.4f}")
        print(f"[DEBUG-LSTM] ================================")

        return {
            'model_type': 'LSTM',
            'train_samples': len(X_seq_train),
            'test_samples': len(X_seq_test),
            'sequence_length': X_seq_train.shape[1],
            'sequential_features': X_seq_train.shape[2],
            'static_features': X_static_train.shape[1],
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'test_r2': float(test_r2),
            'test_predictions': test_preds.tolist(),
            'test_targets': y_test.tolist(),
            'training_history': {
                'epochs': len(history.history['loss']),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
        }
        
        return lstm_results
    
    def _train_alternative_models(self, X_sequences, X_static, y, test_size=0.2, random_state=42) -> Dict[str, Any]:
        """Train alternative models (Feedforward, Transformer, Ensemble) using LSTM-compatible data."""
        
        if not ALTERNATIVE_MODELS_AVAILABLE:
            return {"status": "skipped", "message": "Alternative models not available"}
        
        if not self.alt_models_enabled:
            return {"status": "skipped", "message": "Alternative models not enabled in configuration"}
        
        results = {}
        
        # Split data for alternative models
        X_seq_train, X_seq_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
            X_sequences, X_static, y, test_size=test_size, random_state=random_state
        )
        
        # Train Feedforward model
        if 'feedforward' in self.alt_models_enabled:
            try:
                self.log_info("Training Feedforward model...")
                ff_config = self.alt_models_config.get('feedforward', {})
                ff_model = FeedforwardModel(ff_config, verbose=self.verbose)
                
                ff_training_result = ff_model.train(X_seq_train, X_static_train, y_train, validation_split=0.2)
                ff_evaluation = ff_model.evaluate(X_seq_test, X_static_test, y_test)
                
                results['feedforward'] = {
                    'status': 'success',
                    'model': ff_model,
                    'training_result': ff_training_result,
                    'evaluation': ff_evaluation
                }
                
                # Store model
                self.alternative_models['feedforward'] = ff_model
                
            except Exception as e:
                self.log_info(f"Feedforward model training failed: {e}")
                results['feedforward'] = {
                    'status': 'failed',
                    'message': str(e)
                }
        
        # Train Transformer model
        if 'transformer' in self.alt_models_enabled:
            try:
                self.log_info("Training Transformer model...")
                trans_config = self.alt_models_config.get('transformer', {})
                trans_model = TransformerModel(trans_config, verbose=self.verbose)
                
                trans_training_result = trans_model.train(X_seq_train, X_static_train, y_train, validation_split=0.2)
                trans_evaluation = trans_model.evaluate(X_seq_test, X_static_test, y_test)
                
                results['transformer'] = {
                    'status': 'success',
                    'model': trans_model,
                    'training_result': trans_training_result,
                    'evaluation': trans_evaluation
                }
                
                # Store model
                self.alternative_models['transformer'] = trans_model
                
            except Exception as e:
                self.log_info(f"Transformer model training failed: {e}")
                results['transformer'] = {
                    'status': 'failed',
                    'message': str(e)
                }
        
        # Train Ensemble model  
        if 'ensemble' in self.alt_models_enabled:
            try:
                self.log_info("Training Ensemble model...")
                ens_config = self.alt_models_config.get('ensemble', {})
                ens_model = EnsembleModel(ens_config, verbose=self.verbose)
                
                ens_training_result = ens_model.train(X_seq_train, X_static_train, y_train, validation_split=0.0)
                ens_evaluation = ens_model.evaluate(X_seq_test, X_static_test, y_test)
                
                results['ensemble'] = {
                    'status': 'success',
                    'model': ens_model,
                    'training_result': ens_training_result,
                    'evaluation': ens_evaluation
                }
                
                # Store model
                self.alternative_models['ensemble'] = ens_model
                
            except Exception as e:
                self.log_info(f"Ensemble model training failed: {e}")
                results['ensemble'] = {
                    'status': 'failed',
                    'message': str(e)
                }
        
        return results

    def _old_generate_blended_predictions(self, rf_results, lstm_results,
                                      X_rf_test, y_rf_test,
                                      X_seq_test, X_static_test, y_lstm_test) -> Dict[str, Any]:
        """DEPRECATED: Generate blended predictions from both models."""

        if lstm_results.get('status') == 'failed':
            # If LSTM failed, return RF results only
            return {
                'status': 'rf_only',
                'message': 'LSTM training failed, using RF predictions only',
                'test_mae': rf_results['test_mae'],
                'test_rmse': rf_results['test_rmse'],
                'test_r2': rf_results['test_r2'],
                'test_predictions': rf_results['test_predictions'],
                'test_targets': rf_results['test_targets']
            }

        # Get predictions from both models
        rf_preds = np.array(rf_results['test_predictions'])
        lstm_preds = np.array(lstm_results['test_predictions'])

        # Handle different prediction lengths
        if len(rf_preds) != len(lstm_preds):
            self.log_info(f"Warning: Prediction lengths differ (RF: {len(rf_preds)}, LSTM: {len(lstm_preds)})")
            # Use the shorter length
            min_len = min(len(rf_preds), len(lstm_preds))
            rf_preds = rf_preds[:min_len]
            lstm_preds = lstm_preds[:min_len]
            y_test = np.array(y_rf_test)[:min_len]
        else:
            y_test = np.array(y_rf_test)

        # Blend predictions
        blended_preds = rf_preds * self.blend_weight + lstm_preds * (1 - self.blend_weight)

        # Calculate metrics
        test_mae = mean_absolute_error(y_test, blended_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, blended_preds))
        test_r2 = r2_score(y_test, blended_preds)

        return {
            'status': 'success',
            'blend_weight': self.blend_weight,
            'rf_weight': self.blend_weight,
            'lstm_weight': 1 - self.blend_weight,
            'test_samples': len(blended_preds),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
            'test_r2': float(test_r2),
            'test_predictions': blended_preds.tolist(),
            'test_targets': y_test.tolist(),
            'individual_predictions': {
                'rf_predictions': rf_preds.tolist(),
                'lstm_predictions': lstm_preds.tolist()
            }
        }

    def save_models(self, orchestrator=None):
        """
        Save all trained models (RF, LSTM, and alternative models).

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
                'embedding_dim': getattr(orchestrator, 'embedding_dim', 16),
                'sequence_length': getattr(orchestrator, 'sequence_length', 5)
            }

        # Get the model manager and save legacy models
        model_manager = get_model_manager()
        saved_paths = model_manager.save_models(
            rf_model=self.rf_model,
            lstm_model=self.lstm_model,
            feature_state=feature_state
        )
        
        # Save alternative models if available
        if self.alternative_models:
            alt_model_paths = {}
            models_dir = Path("models")
            
            for model_name, model in self.alternative_models.items():
                if model is not None:
                    try:
                        # Create a timestamped directory for alternative models
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_file = models_dir / f"{model_name}_{timestamp}.h5"
                        
                        if model_name in ['feedforward', 'transformer']:
                            # Save Keras models
                            model.save_model(str(model_file))
                        elif model_name == 'ensemble':
                            # Save ensemble model (pickle)
                            model_file = models_dir / f"{model_name}_{timestamp}.pkl"
                            model.save_ensemble(str(model_file))
                        
                        alt_model_paths[model_name] = str(model_file)
                        print(f"Saved {model_name} model to {model_file}")
                        
                    except Exception as e:
                        print(f"Failed to save {model_name} model: {e}")
                        
            saved_paths['alternative_models'] = alt_model_paths

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
    
    # LSTM model results
    lstm_results = results['lstm_results']
    if lstm_results.get('status') != 'failed':
        print(f"\n--- LSTM Results ---")
        print(f"Train samples: {lstm_results['train_samples']}")
        print(f"Test samples: {lstm_results['test_samples']}")
        print(f"Sequence length: {lstm_results['sequence_length']}")
        print(f"Sequential features: {lstm_results['sequential_features']}")
        print(f"Static features: {lstm_results['static_features']}")
        print(f"Test MAE: {lstm_results['test_mae']:.4f}")
        print(f"Test RMSE: {lstm_results['test_rmse']:.4f}")
        print(f"Test R²: {lstm_results['test_r2']:.4f}")
        print(f"Training epochs: {lstm_results['training_history']['epochs']}")
    
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