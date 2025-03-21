from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import argparse
import time
import json
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import consolidated orchestrator
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from utils.env_setup import AppConfig, get_sqlite_dbpath


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

        # Fix the max_features parameter - 'auto' is deprecated, use 'sqrt' instead
        if rf_params.get('max_features') == 'auto':
            self.log_info("Warning: 'max_features=auto' is deprecated, using 'sqrt' instead")
            max_features = 'sqrt'
        else:
            max_features = rf_params.get('max_features', 'sqrt')  # Default to 'sqrt'

        # Create Random Forest model
        self.rf_model = RandomForestRegressor(
            n_estimators=rf_params.get('n_estimators', 100),
            max_depth=rf_params.get('max_depth', None),
            min_samples_split=rf_params.get('min_samples_split', 2),
            min_samples_leaf=rf_params.get('min_samples_leaf', 1),
            max_features=max_features,  # Use the fixed value
            n_jobs=rf_params.get('n_jobs', -1),
            random_state=rf_params.get('random_state', 42)
        )

        # Train the model
        self.log_info(f"Training RF model on {len(X_train)} samples with {X_train.shape[1]} features...")
        start_time = time.time()
        print(X_train.columns.values)
        self.rf_model.fit(X_train, y_train)

        training_time = time.time() - start_time
        self.log_info(f"RF model training completed in {training_time:.2f} seconds")

        # Calculate performance metrics
        train_pred = self.rf_model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)

        self.log_info(f"RF Training metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")

        # Calculate validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            val_pred = self.rf_model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)

            self.log_info(f"RF Validation metrics - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")

        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.rf_model.feature_importances_
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

        # Store model in models dictionary
        self.models['rf'] = self.rf_model

        # Store performance metrics
        self.models['rf_metrics'] = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_rmse': val_rmse if X_val is not None else None,
            'val_mae': val_mae if X_val is not None else None,
            'val_r2': val_r2 if X_val is not None else None,
            'feature_importance': feature_importance.to_dict(),
            'training_time': training_time
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

    def evaluate_models(self, X_test, y_test, X_seq_test=None, X_static_test=None):
        """
        Evaluate trained models on test data.

        Args:
            X_test: Test features for RF model
            y_test: Test targets
            X_seq_test: Test sequential features for LSTM
            X_static_test: Test static features for LSTM

        Returns:
            Dictionary with evaluation metrics
        """
        results = {}

        # Evaluate RF model if available
        if self.rf_model is not None:
            self.log_info("\n===== EVALUATING RANDOM FOREST MODEL =====")
            rf_pred = self.rf_model.predict(X_test)

            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_mae = mean_absolute_error(y_test, rf_pred)
            rf_r2 = r2_score(y_test, rf_pred)

            self.log_info(f"RF Test metrics - RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")

            results['rf'] = {
                'rmse': rf_rmse,
                'mae': rf_mae,
                'r2': rf_r2
            }

        # Evaluate LSTM model if available
        if self.lstm_model is not None and X_seq_test is not None and X_static_test is not None:
            self.log_info("\n===== EVALUATING LSTM MODEL =====")
            lstm_loss, lstm_mae = self.lstm_model.evaluate(
                [X_seq_test, X_static_test], y_test, verbose=0
            )

            # Get predictions for R² calculation
            lstm_pred = self.lstm_model.predict([X_seq_test, X_static_test], verbose=0)
            lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
            lstm_r2 = r2_score(y_test, lstm_pred)

            self.log_info(
                f"LSTM Test metrics - Loss: {lstm_loss:.4f}, MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}, R²: {lstm_r2:.4f}")

            results['lstm'] = {
                'loss': lstm_loss,
                'mae': lstm_mae,
                'rmse': lstm_rmse,
                'r2': lstm_r2
            }

        # Store evaluation results
        self.models['test_evaluation'] = results
        return results

    def train(self, limit=None, race_filter=None, date_filter=None,
              use_cache=True, train_rf=True, train_lstm=True) -> None:
        """
        Train either or both models based on parameters.

        Args:
            limit: Optional limit for number of races to load
            race_filter: Optional filter for specific race types
            date_filter: Optional date filter
            use_cache: Whether to use cached results when available
            train_rf: Whether to train the Random Forest model
            train_lstm: Whether to train the LSTM model
        """
        start_time = time.time()
        self.log_info(f"\nStarting training process for {self.db_type} database...")

        # Train RF model if requested
        if train_rf:
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
            self.evaluate_models(X_test, y_test)

        # Train LSTM model if requested
        if train_lstm:
            try:
                # Get data for sequential model
                df_features, _ = self.data_orchestrator.load_or_prepare_data(
                    use_cache=use_cache,
                    limit=limit,
                    race_filter=race_filter,
                    date_filter=date_filter
                )

                # Prepare sequence data
                X_sequences, X_static, y = self.data_orchestrator.prepare_sequence_data(
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

                # Train the LSTM model (includes further train/val split)
                self.train_lstm_model(X_seq_train, X_static_train, y_train)

                # Evaluate on test set
                self.evaluate_models(None, y_test, X_seq_test, X_static_test)

            except Exception as e:
                self.log_info(f"Error during LSTM model training: {str(e)}")
                import traceback
                self.log_info(traceback.format_exc())

        # Calculate total training time
        total_time = time.time() - start_time
        self.log_info(f"\nTotal training time: {total_time:.2f} seconds")

        # Save models if any were trained
        if train_rf or train_lstm:
            self.save_models(save_rf=train_rf, save_lstm=train_lstm)

        self.log_info("\nTraining process completed!")

    def save_models(self, save_rf: bool = True, save_lstm: bool = True) -> None:
        """
        Save the trained models and metadata.

        Args:
            save_rf: Whether to save RF model
            save_lstm: Whether to save LSTM model
        """
        self.log_info("\n===== SAVING MODELS =====")

        # Create version string
        version = f"v{time.strftime('%Y%m%d')}"
        save_dir = Path(self.model_paths['model_path']) / version
        save_dir.mkdir(parents=True, exist_ok=True)

        self.log_info(f"Saving models to: {save_dir}")

        # Save RF model
        if save_rf and self.rf_model is not None:
            rf_path = save_dir / self.model_paths['artifacts']['rf_model']

            # Add model metadata
            rf_metadata = {
                'model': self.rf_model,
                'version': version,
                'features': self.data_orchestrator.preprocessing_params.get('feature_columns', []),
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': self.models.get('rf_metrics', {})
            }
            joblib.dump(rf_metadata, rf_path)
            self.log_info(f"Saved RF model with metadata to: {rf_path}")

        # Save LSTM model
        if save_lstm and self.lstm_model is not None:
            lstm_path = save_dir / self.model_paths['artifacts']['lstm_model']
            self.lstm_model.save(lstm_path)
            self.log_info(f"Saved LSTM model to: {lstm_path}")

            # Save LSTM training history separately
            if self.history:
                history_path = save_dir / 'lstm_history.joblib'
                joblib.dump(self.history, history_path)
                self.log_info(f"Saved LSTM training history to: {history_path}")

        # Save orchestrator state for reproducibility
        feature_path = save_dir / self.model_paths['artifacts']['feature_engineer']
        orchestrator_state = {
            'preprocessing_params': self.data_orchestrator.preprocessing_params,
            'embedding_dim': self.data_orchestrator.embedding_dim,
            'sequence_length': self.data_orchestrator.sequence_length,
            'target_info': self.data_orchestrator.target_info
        }
        joblib.dump(orchestrator_state, feature_path)
        self.log_info(f"Saved feature engineering state to: {feature_path}")

        # Save model configuration
        config_path = save_dir / 'model_config.json'
        model_config = {
            'version': version,
            'model_name': self.model_name,
            'db_type': self.db_type,
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

        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2, default=str)
        self.log_info(f"Saved model configuration to: {config_path}")

        self.log_info(f"All models and components saved successfully to {save_dir}")

    def load_models(self, model_dir=None, version=None):
        """
        Load saved models.

        Args:
            model_dir: Path to model directory, if None uses default from config
            version: Specific version to load, if None uses latest

        Returns:
            Whether loading was successful
        """
        # Determine model directory
        if model_dir is None:
            model_dir = self.model_paths['model_path']

        model_path = Path(model_dir)

        # Find available versions
        versions = [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith('v')]

        if not versions:
            self.log_info(f"No model versions found in {model_path}")
            return False

        # Sort versions (newest first)
        versions.sort(reverse=True)

        # Select version to load
        if version is not None:
            version_path = model_path / version
            if not version_path.exists():
                self.log_info(f"Version {version} not found in {model_path}")
                return False
        else:
            version_path = versions[0]  # Latest version

        self.log_info(f"Loading models from {version_path}")

        success = True

        # Load RF model if available
        rf_path = version_path / self.model_paths['artifacts']['rf_model']
        if rf_path.exists():
            try:
                rf_metadata = joblib.load(rf_path)
                self.rf_model = rf_metadata.get('model')
                self.models['rf'] = self.rf_model
                self.models['rf_metadata'] = rf_metadata
                self.log_info(f"Loaded RF model from {rf_path}")
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

                self.log_info(f"Loaded feature engineering state from {feature_path}")
            except Exception as e:
                self.log_info(f"Error loading feature engineering state: {str(e)}")

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
    parser.add_argument('--rf-only', action='store_true', help='Train only Random Forest model')
    parser.add_argument('--lstm-only', action='store_true', help='Train only LSTM model')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Update AppConfig with specified database if provided
    if args.db is not None:
        config = AppConfig(args.config)
        config._config.base.active_db = args.db

    # Determine which models to train
    train_rf = not args.lstm_only  # Train RF unless LSTM-only flag is set
    train_lstm = not args.rf_only  # Train LSTM unless RF-only flag is set

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
        use_cache=not args.no_cache,
        train_rf=train_rf,
        train_lstm=train_lstm
    )


if __name__ == "__main__":
    main()