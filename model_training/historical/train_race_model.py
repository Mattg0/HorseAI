from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import argparse
import time
import json
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Now - simply import from the package
from model_training.regressions.isotonic_calibration import CalibratedRegressor, regression_metrics_report, plot_calibration_effect
# Import consolidated orchestrator
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from utils.env_setup import AppConfig, get_sqlite_dbpath
from utils.model_manager import get_model_manager


class HorseRaceModel:
    """Horse race prediction model that combines random forest and LSTM for predictions."""

    def __init__(self, config_path: str = 'config.yaml', model_name: str = 'hybrid',
                 sequence_length: int = 5, embedding_dim: int = None,
                 verbose: bool = False):
        """Initialize the model with configuration."""
        self.config = AppConfig(config_path)
        self.model_name = model_name
        self.model_paths = self.config.get_model_paths()
        self.sequence_length = sequence_length
        self.verbose = verbose

        # Initialize model components
        self.models = {}
        self.rf_model = None
        self.lstm_model = None
        self.history = None

        # Get active database configuration
        self.db_type = self.config._config.base.active_db
        db_path = get_sqlite_dbpath(self.db_type)

        # Initialize data orchestrator for all data preparation
        self.data_orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=db_path,
            embedding_dim=embedding_dim or self.config._config.features.embedding_dim,
            sequence_length=sequence_length,
            verbose=verbose
        )

        # Load training configuration
        self.training_config = self._load_training_config()

        self.log_info(f"Initialized HorseRaceModel with database: {self.db_type}")
        self.log_info(f"Model paths: {self.model_paths}")

    def log_info(self, message):
        """Simple logging method."""
        if self.verbose:
            print(message)

    def _load_training_config(self) -> Dict:
        """Load training configuration parameters."""
        config = self.config._config

        # Default configuration
        training_config = {
            # RF parameters
            'rf_params': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'n_jobs': -1,
                'random_state': 42
            },
            # LSTM parameters
            'lstm_params': {
                'lstm_units': 64,
                'dropout_rate': 0.2,
                'epochs': 100,
                'batch_size': 32,
                'validation_split': 0.2,
                'early_stopping_patience': 10,
                'lr_reduction_factor': 0.5,
                'lr_patience': 5,
                'min_lr': 0.00001
            },
            # Data parameters
            'data_params': {
                'target_field': 'final_position',
                'test_size': 0.2,
                'val_size': 0.1,
                'random_state': 42,
                'normalize_features': True
            }
        }

        # Override with config values if they exist
        if hasattr(config, 'training'):
            if hasattr(config.training, 'rf_params'):
                training_config['rf_params'].update(config.training.rf_params)
            if hasattr(config.training, 'lstm_params'):
                training_config['lstm_params'].update(config.training.lstm_params)
            if hasattr(config.training, 'data_params'):
                training_config['data_params'].update(config.training.data_params)

        return training_config

    def train_rf_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """
        Train the Random Forest model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        self.log_info("\n===== TRAINING RANDOM FOREST MODEL =====")

        # Get RF parameters from configuration
        rf_params = self.training_config['rf_params']

        # Fix the max_features parameter
        if rf_params.get('max_features') == 'auto':
            self.log_info("Warning: 'max_features=auto' is deprecated, using 'sqrt' instead")
            max_features = 'sqrt'
        else:
            max_features = rf_params.get('max_features', 'sqrt')  # Default to 'sqrt'

        # Create Random Forest model
        from sklearn.ensemble import RandomForestRegressor
        base_rf = RandomForestRegressor(
            n_estimators=rf_params.get('n_estimators', 100),
            max_depth=rf_params.get('max_depth', None),
            min_samples_split=rf_params.get('min_samples_split', 2),
            min_samples_leaf=rf_params.get('min_samples_leaf', 1),
            max_features=max_features,
            n_jobs=rf_params.get('n_jobs', -1),
            random_state=rf_params.get('random_state', 42)
        )

        # Create calibrated regressor (will automatically handle calibration)
        self.rf_model = CalibratedRegressor(
            base_regressor=base_rf,
            # Clip predictions between 1 and highest position
            clip_min=1.0,
            clip_max=None  # Will be determined during fitting
        )

        # Train the model with calibration
        self.log_info(f"Training RF model on {len(X_train)} samples with {X_train.shape[1]} features...")
        start_time = time.time()

        # If no validation set, use a portion of training data for calibration
        if X_val is None or y_val is None:
            self.rf_model.fit(X_train, y_train)
        else:
            # Use validation set for calibration
            self.rf_model.fit(X_train, y_train, X_calib=X_val, y_calib=y_val)

        training_time = time.time() - start_time
        self.log_info(f"RF model training completed in {training_time:.2f} seconds")

        # Evaluate performance
        train_metrics = self.rf_model.evaluate(X_train, y_train)
        self.log_info("\nTraining set performance:")
        self.log_info(f"  Uncalibrated - RMSE: {train_metrics['raw_rmse']:.4f}, MAE: {train_metrics['raw_mae']:.4f}")
        self.log_info(
            f"  Calibrated - RMSE: {train_metrics['calibrated_rmse']:.4f}, MAE: {train_metrics['calibrated_mae']:.4f}")

        if X_val is not None and y_val is not None:
            val_metrics = self.rf_model.evaluate(X_val, y_val)
            self.log_info("\nValidation set performance:")
            self.log_info(f"  Uncalibrated - RMSE: {val_metrics['raw_rmse']:.4f}, MAE: {val_metrics['raw_mae']:.4f}")
            self.log_info(
                f"  Calibrated - RMSE: {val_metrics['calibrated_rmse']:.4f}, MAE: {val_metrics['calibrated_mae']:.4f}")

            # Generate calibration effect plot
            logs_path = Path(self.model_paths['logs'])
            logs_path.mkdir(parents=True, exist_ok=True)
            plot_path = logs_path / f'calibration_effect_{self.db_type}.png'

            # Get raw and calibrated predictions
            raw_preds = self.rf_model.predict_raw(X_val)
            cal_preds = self.rf_model.predict(X_val)

            # Create plot
            plot_calibration_effect(
                raw_preds, cal_preds, y_val.values,
                save_path=str(plot_path)
            )
            self.log_info(f"Calibration effect plot saved to {plot_path}")

        # Calculate and display feature importance
        base_regressor = self.rf_model.base_regressor
        if hasattr(base_regressor, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': base_regressor.feature_importances_
            }).sort_values('importance', ascending=False)

            self.log_info("\nTop feature importance for RF model:")
            for i, (feature, importance) in enumerate(
                    zip(feature_importance['feature'][:10], feature_importance['importance'][:10])
            ):
                self.log_info(f"{i + 1}. {feature}: {importance:.4f}")

            # Save feature importance
            importance_path = Path(self.model_paths['logs']) / f'rf_feature_importance_{self.db_type}.csv'
            importance_path.parent.mkdir(parents=True, exist_ok=True)
            feature_importance.to_csv(importance_path, index=False)
            self.log_info(f"Feature importance saved to {importance_path}")

        # Store model in models dictionary
        self.models['rf'] = self.rf_model

        # Store performance metrics
        self.models['rf_metrics'] = {
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics if X_val is not None else None,
        }

    def train_lstm_model(self, X_sequences, X_static, y_targets) -> None:
        """
        Train the LSTM model for sequence prediction.

        Args:
            X_sequences: Sequential input features
            X_static: Static input features
            y_targets: Target values
        """
        self.log_info("\n===== TRAINING LSTM MODEL =====")

        # Get LSTM parameters from configuration
        lstm_params = self.training_config['lstm_params']

        # Split into train/validation sets
        from sklearn.model_selection import train_test_split

        # Use validation_split from parameters
        val_split = lstm_params.get('validation_split', 0.2)
        random_state = self.training_config['data_params'].get('random_state', 42)

        X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
            X_sequences, X_static, y_targets,
            test_size=val_split,
            random_state=random_state
        )

        self.log_info(f"Training set: {len(X_seq_train)} sequences")
        self.log_info(f"Validation set: {len(X_seq_val)} sequences")

        # Create LSTM model if not already created
        if self.lstm_model is None:
            # Get model from orchestrator
            hybrid_model = self.data_orchestrator.create_hybrid_model(
                sequence_shape=X_sequences.shape,
                static_shape=X_static.shape,
                lstm_units=lstm_params.get('lstm_units', 64),
                dropout_rate=lstm_params.get('dropout_rate', 0.2)
            )

            if hybrid_model is None:
                self.log_info("Failed to create LSTM model. TensorFlow may not be available.")
                return

            self.lstm_model = hybrid_model['lstm']
            self.models['lstm'] = self.lstm_model

        # Configure callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=lstm_params.get('early_stopping_patience', 10),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=lstm_params.get('lr_reduction_factor', 0.5),
                patience=lstm_params.get('lr_patience', 5),
                min_lr=lstm_params.get('min_lr', 0.00001)
            )
        ]

        # Add model checkpoint
        if self.model_paths['logs']:
            checkpoint_path = Path(self.model_paths['logs']) / f'best_lstm_model_{self.db_type}.keras'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    str(checkpoint_path),
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                )
            )

        # Train LSTM model
        self.log_info(f"Training LSTM model...")
        start_time = time.time()

        history = self.lstm_model.fit(
            [X_seq_train, X_static_train],
            y_train,
            epochs=lstm_params.get('epochs', 100),
            batch_size=lstm_params.get('batch_size', 32),
            validation_data=([X_seq_val, X_static_val], y_val),
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )

        training_time = time.time() - start_time
        self.log_info(f"LSTM model training completed in {training_time:.2f} seconds")

        # Store training history
        self.history = history.history

        # Evaluate on training and validation sets
        train_loss, train_mae = self.lstm_model.evaluate(
            [X_seq_train, X_static_train], y_train, verbose=0
        )
        val_loss, val_mae = self.lstm_model.evaluate(
            [X_seq_val, X_static_val], y_val, verbose=0
        )

        self.log_info(f"LSTM Training metrics - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
        self.log_info(f"LSTM Validation metrics - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

        # Store performance metrics
        self.models['lstm_metrics'] = {
            'train_loss': train_loss,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
            'training_time': training_time
        }

        # Plot training history
        self._plot_lstm_history(history)

    def _plot_lstm_history(self, history):
        """
        Plot LSTM training history.

        Args:
            history: History object from model training
        """
        try:
            # Create logs directory if it doesn't exist
            logs_path = Path(self.model_paths['logs'])
            logs_path.mkdir(parents=True, exist_ok=True)

            # Plot loss
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')

            # Plot MAE
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'])
            plt.plot(history.history['val_mae'])
            plt.title('Model MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')

            # Save figure
            plt.tight_layout()
            plt.savefig(logs_path / f'lstm_history_{self.db_type}.png')
            plt.close()

            self.log_info(f"Training history plot saved to {logs_path}/lstm_history_{self.db_type}.png")
        except Exception as e:
            self.log_info(f"Error plotting LSTM history: {str(e)}")

    def train_rf_model_pipeline(self, limit=None, race_filter=None, date_filter=None, use_cache=True):
        """
        Train and evaluate the Random Forest model.

        Args:
            limit: Optional limit for number of races to load
            race_filter: Optional filter for specific race types
            date_filter: Optional date filter
            use_cache: Whether to use cached results when available

        Returns:
            Dictionary with training results and evaluation metrics
        """
        start_time = time.time()
        self.log_info(f"\nStarting Random Forest training for {self.db_type} database...")

        # Get prepared data through the orchestrator
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_orchestrator.run_pipeline(
            limit=limit,
            race_filter=race_filter,
            date_filter=date_filter,
            use_cache=use_cache,
            clean_embeddings=True
        )

        # Train the RF model
        self.train_rf_model(X_train, y_train, X_val, y_val)

        # Evaluate on test set
        rf_results = self.evaluate_rf_model(X_test, y_test)

        # Calculate training time
        training_time = time.time() - start_time

        results = {
            'model_type': 'random_forest',
            'training_time': training_time,
            'evaluation': rf_results
        }

        self.log_info(f"Random Forest training completed in {training_time:.2f} seconds")
        return results

    def train_lstm_model_pipeline(self, limit=None, race_filter=None, date_filter=None, use_cache=True):
        """
        Train and evaluate the LSTM model.

        Args:
            limit: Optional limit for number of races to load
            race_filter: Optional filter for specific race types
            date_filter: Optional date filter
            use_cache: Whether to use cached results when available

        Returns:
            Dictionary with training results and evaluation metrics
        """
        start_time = time.time()
        self.log_info(f"\nStarting LSTM training for {self.db_type} database...")

        try:
            # Get data for sequential model
            df_raw = self.data_orchestrator.load_historical_races(
                limit=limit,
                race_filter=race_filter,
                date_filter=date_filter,
                use_cache=use_cache
            )
            # Apply embeddings with lstm_mode=True to preserve idche and jour
            df_features = self.data_orchestrator.apply_embeddings(
                df_raw,
                clean_after_embedding=True,
                keep_identifiers=True,
                lstm_mode=True  # This will ensure idche and jour are preserved
            )

            # Now prepare sequence data
            X_sequences, X_static, y, horse_ids, race_dates = self.data_orchestrator.prepare_lstm_sequence_features(
                df_features,
                sequence_length=self.sequence_length
            )

            # Split into train/test sets
            from sklearn.model_selection import train_test_split

            # Split with test_size from parameters
            test_size = self.training_config['data_params'].get('test_size', 0.2)
            random_state = self.training_config['data_params'].get('random_state', 42)

            X_seq_train, X_seq_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
                X_sequences, X_static, y,
                test_size=test_size,
                random_state=random_state
            )

            # Train the LSTM model
            self.train_lstm_model(X_seq_train, X_static_train, y_train)

            # Evaluate on test set
            lstm_results = self.evaluate_lstm_model(X_seq_test, X_static_test, y_test)

            # Calculate training time
            training_time = time.time() - start_time

            results = {
                'model_type': 'lstm',
                'training_time': training_time,
                'evaluation': lstm_results,
                'sequence_length': self.sequence_length
            }

            self.log_info(f"LSTM training completed in {training_time:.2f} seconds")
            return results

        except Exception as e:
            self.log_info(f"Error during LSTM model training: {str(e)}")
            import traceback
            self.log_info(traceback.format_exc())

            return {
                'model_type': 'lstm',
                'status': 'error',
                'error': str(e)
            }

    def evaluate_rf_model(self, X_test, y_test):
        """
        Evaluate the Random Forest model.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if self.rf_model is None:
            self.log_info("RF model not available for evaluation")
            return {'status': 'error', 'error': 'Model not available'}

        self.log_info("\n===== EVALUATING RANDOM FOREST MODEL =====")

        try:
            rf_pred = self.rf_model.predict(X_test)

            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_mae = mean_absolute_error(y_test, rf_pred)
            rf_r2 = r2_score(y_test, rf_pred)

            self.log_info(f"RF Test metrics - RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")

            return {
                'status': 'success',
                'rmse': float(rf_rmse),
                'mae': float(rf_mae),
                'r2': float(rf_r2),
                'sample_count': len(y_test)
            }

        except Exception as e:
            self.log_info(f"Error evaluating RF model: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def evaluate_lstm_model(self, X_seq_test, X_static_test, y_test):
        """
        Evaluate the LSTM model.

        Args:
            X_seq_test: Test sequence features
            X_static_test: Test static features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if self.lstm_model is None:
            self.log_info("LSTM model not available for evaluation")
            return {'status': 'error', 'error': 'Model not available'}

        self.log_info("\n===== EVALUATING LSTM MODEL =====")

        try:
            lstm_loss, lstm_mae = self.lstm_model.evaluate(
                [X_seq_test, X_static_test], y_test, verbose=0
            )

            # Get predictions for R² calculation
            lstm_pred = self.lstm_model.predict([X_seq_test, X_static_test], verbose=0)
            lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
            lstm_r2 = r2_score(y_test, lstm_pred)

            self.log_info(
                f"LSTM Test metrics - Loss: {lstm_loss:.4f}, MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}, R²: {lstm_r2:.4f}")

            return {
                'status': 'success',
                'loss': float(lstm_loss),
                'mae': float(lstm_mae),
                'rmse': float(lstm_rmse),
                'r2': float(lstm_r2),
                'sample_count': len(y_test)
            }

        except Exception as e:
            self.log_info(f"Error evaluating LSTM model: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def train_hybrid_model_pipeline(self, limit=None, race_filter=None, date_filter=None, use_cache=True):
        """
        Train both RF and LSTM models as a hybrid approach.

        Args:
            limit: Optional limit for number of races to load
            race_filter: Optional filter for specific race types
            date_filter: Optional date filter
            use_cache: Whether to use cached results when available

        Returns:
            Dictionary with training results for both models
        """
        start_time = time.time()
        self.log_info(f"\nStarting Hybrid model training for {self.db_type} database...")

        # Train RF model
        rf_results = self.train_rf_model_pipeline(
            limit=limit,
            race_filter=race_filter,
            date_filter=date_filter,
            use_cache=use_cache
        )

        # Train LSTM model
        lstm_results = self.train_lstm_model_pipeline(
            limit=limit,
            race_filter=race_filter,
            date_filter=date_filter,
            use_cache=use_cache
        )

        # Calculate total training time
        total_time = time.time() - start_time

        # Combine results
        results = {
            'model_type': 'hybrid',
            'training_time': total_time,
            'rf_results': rf_results,
            'lstm_results': lstm_results
        }

        self.log_info(f"\nHybrid model training completed in {total_time:.2f} seconds")

        # Save models
        self.save_models(save_rf=True, save_lstm=True)

        return results

    def evaluate_models(self, X_test=None, y_test=None, X_seq_test=None, X_static_test=None):
        """
        Evaluate trained models on test data.

        This is a wrapper around the specialized evaluation functions. Consider using
        evaluate_rf_model() or evaluate_lstm_model() directly for specific model evaluation.

        Args:
            X_test: Test features for RF model (can be None if only evaluating LSTM)
            y_test: Test targets
            X_seq_test: Test sequential features for LSTM (can be None if only evaluating RF)
            X_static_test: Test static features for LSTM (can be None if only evaluating RF)

        Returns:
            Dictionary with evaluation metrics
        """
        results = {}

        # Verify we have targets to evaluate against
        if y_test is None:
            self.log_info("Cannot evaluate models: y_test is None")
            return {"error": "No target data provided for evaluation"}

        # Evaluate RF model if we have the right data
        if X_test is not None:
            rf_results = self.evaluate_rf_model(X_test, y_test)
            results['rf'] = rf_results

        # Evaluate LSTM model if we have the right data
        if X_seq_test is not None and X_static_test is not None:
            lstm_results = self.evaluate_lstm_model(X_seq_test, X_static_test, y_test)
            results['lstm'] = lstm_results

        # Store evaluation results
        self.models['test_evaluation'] = results
        return results

    def train(self, limit=None, race_filter=None, date_filter=None, use_cache=True, model_type='hybrid'):
        """
        Train models based on the specified model type.

        Args:
            limit: Optional limit for number of races to load
            race_filter: Optional filter for specific race types
            date_filter: Optional date filter
            use_cache: Whether to use cached results when available
            model_type: Type of model to train ('rf', 'lstm', or 'hybrid')

        Returns:
            Dictionary with training results
        """
        # Normalize model type string
        model_type = model_type.lower()

        if model_type == 'rf' or model_type == 'random_forest':
            results = self.train_rf_model_pipeline(limit, race_filter, date_filter, use_cache)
            self.save_models(save_rf=True, save_lstm=False)

        elif model_type == 'lstm':
            results = self.train_lstm_model_pipeline(limit, race_filter, date_filter, use_cache)
            self.save_models(save_rf=False, save_lstm=True)

        elif model_type == 'hybrid':
            results = self.train_hybrid_model_pipeline(limit, race_filter, date_filter, use_cache)
            self.save_models(save_rf=True, save_lstm=True)

        else:
            self.log_info(f"Unknown model type: {model_type}. Valid options are 'rf', 'lstm', or 'hybrid'")
            return {
                'status': 'error',
                'error': f"Unknown model type: {model_type}. Valid options are 'rf', 'lstm', or 'hybrid'"
            }

        self.log_info("\nTraining process completed!")
        return results

    def save_models(self, save_rf: bool = True, save_lstm: bool = True) -> None:
        """
        Save the trained models and metadata.

        Args:
            save_rf: Whether to save RF model
            save_lstm: Whether to save LSTM model
        """
        # Get the model manager
        model_manager = get_model_manager()

        # Create version string based on date, database type, and training type
        version = model_manager.get_version_path(self.db_type, train_type='full')

        # Resolve the base path
        save_dir = model_manager.get_model_path(self.model_name) / version

        self.log_info(f"Saving models to: {save_dir}")

        # Prepare orchestrator state
        orchestrator_state = {
            'preprocessing_params': self.data_orchestrator.preprocessing_params,
            'embedding_dim': self.data_orchestrator.embedding_dim,
            'sequence_length': self.data_orchestrator.sequence_length,
            'target_info': self.data_orchestrator.target_info
        }

        # Prepare model configuration
        model_config = {
            'version': version,
            'model_name': self.model_name,
            'db_type': self.db_type,
            'train_type': 'full',  # Explicitly mark as full training
            'sequence_length': self.sequence_length,
            'embedding_dim': self.data_orchestrator.embedding_dim,
            'training_config': self.training_config,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_trained': {
                'rf': save_rf and self.rf_model is not None,
                'lstm': save_lstm and self.lstm_model is not None
            },
            'evaluation_results': self.models.get('test_evaluation', {})
        }

        # Save all artifacts at once
        saved_paths = model_manager.save_model_artifacts(
            base_path=save_dir,
            rf_model=self.rf_model if save_rf else None,
            lstm_model=self.lstm_model if save_lstm else None,
            orchestrator_state=orchestrator_state,
            history=self.history,
            model_config=model_config,
            db_type=self.db_type,
            train_type='full',
            update_config=True  # Update config.yaml with reference to this model
        )

        self.log_info(f"All models and components saved successfully to {save_dir}")
        return saved_paths

    def _update_config_with_latest_model(self, version: str) -> None:
        """
        Update config.yaml with the latest full model information.

        Args:
            version: Version string of the model
        """
        try:
            config_path = "config.yaml"

            # Read existing config
            with open(config_path, 'r') as f:
                config_content = f.read()

            # Parse YAML
            import yaml
            config_data = yaml.safe_load(config_content)

            # Update or add the latest_base_model field
            if 'models' not in config_data:
                config_data['models'] = {}

            config_data['models']['latest_base_model'] = version

            # Write updated config
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)

            self.log_info(f"Updated config.yaml with latest_base_model: {version}")
        except Exception as e:
            self.log_info(f"Warning: Failed to update config.yaml: {str(e)}")

    def load_models(self, model_dir=None, version=None, use_latest_base=False):
        """
        Load saved models.

        Args:
            model_dir: Path to model directory, if None uses default from config
            version: Specific version to load, if None uses latest
            use_latest_base: Whether to load the latest base model from config

        Returns:
            Whether loading was successful
        """
        # Determine model directory
        if model_dir is None:
            model_dir = self.model_paths['model_path']

        model_path = Path(model_dir)

        # If use_latest_base is True, try to get the latest_base_model from config
        if use_latest_base:
            try:
                latest_base = self.config._config.models.latest_base_model
                if latest_base:
                    # Construct the path to the latest base model
                    base_model_dir = Path(self.config._config.models.model_dir) / self.model_name
                    version = latest_base
                    model_path = base_model_dir
                    self.log_info(f"Using latest base model from config: {latest_base}")
            except (AttributeError, KeyError):
                self.log_info("No latest_base_model found in config, using specified model path")

        # If version is specified, use it directly
        if version is not None:
            version_path = model_path / version
            if not version_path.exists():
                self.log_info(f"Version {version} not found in {model_path}")
                return False
        else:
            # Find available versions
            versions = [d for d in model_path.iterdir() if
                        d.is_dir() and d.name.startswith(('v', 'full_v', '2years_v', '5years_v', 'dev_v'))]

            if not versions:
                self.log_info(f"No model versions found in {model_path}")
                return False

            # Sort versions (newest first)
            versions.sort(reverse=True)
            version_path = versions[0]  # Latest version

        self.log_info(f"Loading models from {version_path}")

        success = True

        # Load RF model if available
        rf_path = version_path / self.model_paths['artifacts']['rf_model']
        if rf_path.exists():
            try:
                # Try to load with CalibratedRegressor.load first
                try:
                    from model_training.regressions.isotonic_calibration import CalibratedRegressor
                    self.rf_model = CalibratedRegressor.load(rf_path)
                    self.models['rf'] = self.rf_model
                    self.log_info(f"Loaded RF model with CalibratedRegressor from {rf_path}")
                except:
                    # Fall back to joblib load if CalibratedRegressor.load fails
                    rf_metadata = joblib.load(rf_path)

                    if isinstance(rf_metadata, dict) and 'model' in rf_metadata:
                        self.rf_model = rf_metadata['model']
                        self.models['rf'] = self.rf_model
                        self.models['rf_metadata'] = rf_metadata
                        self.log_info(f"Loaded RF model from metadata dictionary at {rf_path}")
                    else:
                        # Direct model object
                        self.rf_model = rf_metadata
                        self.models['rf'] = self.rf_model
                        self.log_info(f"Loaded RF model directly from {rf_path}")
            except Exception as e:
                self.log_info(f"Error loading RF model: {str(e)}")
                success = False

        # Load LSTM model if available
        lstm_path = version_path / self.model_paths['artifacts']['lstm_model']
        if lstm_path.exists():
            try:
                from tensorflow.keras.models import load_model
                self.lstm_model = load_model(lstm_path)
                self.models['lstm'] = self.lstm_model
                self.log_info(f"Loaded LSTM model from {lstm_path}")

                # Try to load history
                history_path = version_path / 'lstm_history.joblib'
                if history_path.exists():
                    self.history = joblib.load(history_path)
            except Exception as e:
                self.log_info(f"Error loading LSTM model: {str(e)}")
                success = False

        # Load orchestrator state
        feature_path = version_path / self.model_paths['artifacts']['feature_engineer']
        if feature_path.exists():
            try:
                orchestrator_state = joblib.load(feature_path)

                # Update orchestrator with loaded state
                if isinstance(orchestrator_state, dict):
                    if 'preprocessing_params' in orchestrator_state:
                        self.data_orchestrator.preprocessing_params.update(
                            orchestrator_state['preprocessing_params']
                        )
                    if 'target_info' in orchestrator_state:
                        self.data_orchestrator.target_info = orchestrator_state['target_info']

                    # Update embedding_dim and sequence_length if available
                    if 'embedding_dim' in orchestrator_state:
                        self.data_orchestrator.embedding_dim = orchestrator_state['embedding_dim']
                    if 'sequence_length' in orchestrator_state:
                        self.data_orchestrator.sequence_length = orchestrator_state['sequence_length']

                self.log_info(f"Loaded feature engineering state from {feature_path}")
            except Exception as e:
                self.log_info(f"Error loading feature engineering state: {str(e)}")

        # Load model config if available
        model_config_path = version_path / 'model_config.json'
        if model_config_path.exists():
            try:
                with open(model_config_path, 'r') as f:
                    self.model_config = json.load(f)
                self.log_info(f"Loaded model configuration from {model_config_path}")
            except Exception as e:
                self.log_info(f"Error loading model configuration: {str(e)}")

        return success


def main():
    """
    Main function to run the training process from command line.
    """
    parser = argparse.ArgumentParser(description='Train horse race prediction model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--db', type=str, default=None, help='Database to use (overrides config)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of races to load')
    parser.add_argument('--race-type', type=str, default=None, help='Filter by race type (e.g., "A" for Attele)')
    parser.add_argument('--date-filter', type=str, default=None, help='Date filter (e.g., "jour > \'2023-01-01\'")')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of intermediate data')
    parser.add_argument('--sequence-length', type=int, default=5, help='Sequence length for LSTM')
    parser.add_argument('--embedding-dim', type=int, default=None, help='Dimension for entity embeddings')
    parser.add_argument('--model-name', type=str, default='hybrid', help='Model architecture name')
    parser.add_argument('--model-type', type=str, default='hybrid', choices=['rf', 'lstm', 'hybrid'],
                        help='Type of model to train')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Update AppConfig with specified database if provided
    if args.db is not None:
        config = AppConfig(args.config)
        config._config.base.active_db = args.db

    # Create and train the model
    trainer = HorseRaceModel(
        config_path=args.config,
        model_name=args.model_name,
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        verbose=args.verbose
    )

    trainer.train(
        limit=args.limit,
        race_filter=args.race_type,
        date_filter=args.date_filter,
        use_cache=args.no_cache,
        model_type=args.model_type
    )


if __name__ == "__main__":
    main()