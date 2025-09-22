import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple
import time
import psutil
import platform

from .base_model import BaseModel


class FeedforwardModel(BaseModel):
    """
    Feedforward Neural Network for horse racing prediction.
    
    This model processes flat domain features (58 features like TabNet) using
    a simple feedforward architecture. Uses the same domain-specific features
    that work successfully for Random Forest and TabNet models.
    """
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the Feedforward model.
        
        Args:
            config: Model configuration dictionary
            verbose: Whether to print verbose output
        """
        super().__init__(config, verbose)
        
        # Extract feedforward-specific config with performance optimizations
        self.hidden_units = config.get('hidden_units', 96)
        self.learning_rate = config.get('learning_rate', 0.005)
        
        # Optimize batch size for hardware
        default_batch_size = self._get_optimal_batch_size()
        self.batch_size = config.get('batch_size', default_batch_size)
        
        self.epochs = config.get('epochs', 100)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.early_stopping_patience = config.get('early_stopping_patience', 15)
        self.position_encoding_dim = config.get('position_encoding_dim', 16)
        
        # Performance monitoring
        self.enable_profiling = config.get('enable_profiling', True)
        self._cached_features = None
        
        if self.verbose:
            print(f"FeedforwardModel initialized with:")
            print(f"  Hidden units: {self.hidden_units}")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Position encoding dim: {self.position_encoding_dim}")
    
    def _get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available hardware."""
        try:
            # Check if GPU is available and get memory
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # RTX 4090 has 24GB - use larger batches
                return 4096
            else:
                # CPU or M1 - use smaller batches
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                if available_memory_gb > 12:  # M1 Pro 16GB
                    return 1024
                else:
                    return 512
        except:
            return 512  # Safe fallback
    
    def _profile_step(self, step_name: str, start_time: float = None):
        """Profile training steps for performance optimization."""
        if not self.enable_profiling:
            return time.time()
            
        current_time = time.time()
        if start_time is not None:
            duration = current_time - start_time
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"[PERF] {step_name}: {duration:.3f}s, Memory: {memory_mb:.1f}MB")
        
        return current_time
    
    def assert_data_quality_flat(self, X_flat: np.ndarray, y: np.ndarray = None):
        """
        Assert data quality for flat feature input (TabNet-style).
        
        Args:
            X_flat: Flat training features (samples x 58_features)
            y: Optional target values for training
        """
        # Basic shape validations
        assert X_flat.ndim == 2, f"X_flat must be 2D, got shape {X_flat.shape}"
        if y is not None:
            assert y.ndim == 1, f"y must be 1D, got shape {y.shape}"
            assert len(X_flat) == len(y), f"X_flat and y must have same length: {len(X_flat)} vs {len(y)}"
        
        # Expected 58 domain features (same as TabNet)
        expected_features = 58
        assert X_flat.shape[1] == expected_features, f"Expected {expected_features} features, got {X_flat.shape[1]}"
        
        # Check for data quality issues
        nan_count = np.isnan(X_flat).sum()
        inf_count = np.isinf(X_flat).sum()
        
        if nan_count > 0:
            nan_percentage = (nan_count / X_flat.size) * 100
            print(f"[WARNING-FF] Found {nan_count:,} NaN values ({nan_percentage:.2f}% of data)")
            if nan_percentage > 50:
                raise ValueError(f"Too many NaN values: {nan_percentage:.1f}% of data")
        
        if inf_count > 0:
            print(f"[WARNING-FF] Found {inf_count:,} inf values")
        
        if self.verbose:
            print(f"[DEBUG-FF] Data quality check passed: {X_flat.shape[0]:,} samples x {X_flat.shape[1]} features")
            print(f"[DEBUG-FF] Data ranges: min={X_flat.min():.3f}, max={X_flat.max():.3f}, mean={X_flat.mean():.3f}")
    
    def prepare_features(self, X_sequences: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """
        Legacy method for BaseModel compatibility. 
        Feedforward model now uses flat features directly via train(X_flat, y).
        
        Args:
            X_sequences: Sequential features (not used in flat feature mode)
            X_static: Static features (not used in flat feature mode)
            
        Returns:
            Pass-through for compatibility
        """
        # This method exists for BaseModel abstract compliance
        # The actual Feedforward model uses flat features via train(X_flat, y)
        print("[WARNING] prepare_features called on Feedforward model - use flat feature interface instead")
        return X_sequences, X_static
    
    def build_model(self, input_shape: int) -> Model:
        """
        Build the feedforward model architecture for flat features.
        
        Args:
            input_shape: Number of flat features (58 for racing domain)
            
        Returns:
            Compiled Keras model
        """
        # Single flat feature input (58 domain features like TabNet)
        feature_input = Input(shape=(input_shape,), name='flat_features')
        
        # First hidden layer
        hidden1 = Dense(
            self.hidden_units,
            activation='relu',
            name='hidden_layer_1'
        )(feature_input)
        
        # Dropout for regularization
        hidden1 = Dropout(self.dropout_rate, name='dropout_1')(hidden1)
        
        # Second hidden layer (smaller)
        hidden2 = Dense(
            self.hidden_units // 2,
            activation='relu',
            name='hidden_layer_2'
        )(hidden1)
        
        # Second dropout
        hidden2 = Dropout(self.dropout_rate / 2, name='dropout_2')(hidden2)
        
        # Output layer (regression for position prediction)
        output = Dense(1, activation='linear', name='position_output')(hidden2)
        
        # Create model
        model = Model(
            inputs=feature_input,
            outputs=output,
            name='FeedforwardHorseRacingModel'
        )
        
        # Compile model with optimizations
        optimizer = Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae'],
            run_eagerly=False
        )
        
        if self.verbose:
            print("Feedforward model architecture (flat features):")
            model.summary()
        
        return model
    
    def train(self, X_flat: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Optimized training with comprehensive profiling and performance monitoring.
        Uses flat features like TabNet (58 domain features) instead of sequences.
        
        Args:
            X_flat: Flat training features (same as TabNet: samples x 58_features)
            y: Target values (horse positions)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results dictionary
        """
        training_start = self._profile_step("Training initialization")
        
        # Assert data quality for flat features
        self.assert_data_quality_flat(X_flat, y)
        
        # Performance diagnostics
        print(f"[DEBUG-FF] Starting feedforward training with optimizations:")
        print(f"[DEBUG-FF] Dataset: {len(y):,} samples, {X_flat.shape[1]} flat features")
        print(f"[DEBUG-FF] Memory usage: {psutil.Process().memory_info().rss / 1024**2:.1f} MB")
        print(f"[DEBUG-FF] Batch size: {self.batch_size} (optimized for hardware)")
        
        # GPU utilization check
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[DEBUG-FF] GPU available: {len(gpus)} device(s)")
            try:
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"[DEBUG-FF] GPU memory: {gpu_info['peak'] / 1024**2:.1f} MB peak")
            except:
                print(f"[DEBUG-FF] GPU memory info unavailable")
        else:
            print(f"[DEBUG-FF] Running on CPU")
        
        # Use flat features directly (no complex preparation needed)
        feature_start = self._profile_step("Feature preparation")
        X_prepared = X_flat.astype(np.float32)  # Ensure consistent dtype
        self._profile_step("Feature preparation completed", feature_start)
        
        # Critical shape validation before training
        print(f"[DEBUG-FF] Shape validation before model.fit():")
        print(f"[DEBUG-FF] X_prepared: {X_prepared.shape}")
        print(f"[DEBUG-FF] y: {y.shape}")
        
        # Assert all arrays have same number of samples
        expected_samples = len(y)
        assert X_prepared.shape[0] == expected_samples, f"X_prepared has {X_prepared.shape[0]} samples, expected {expected_samples}"
        
        # Build model if not already built
        if self.model is None:
            input_shape = X_prepared.shape[1]  # Number of flat features (58)
            self.model = self.build_model(input_shape)
        
        # Split data for training and validation
        split_start = self._profile_step("Data splitting")
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_prepared, y,
                test_size=validation_split,
                random_state=42
            )
            validation_data = (X_val, y_val)
        else:
            X_train, y_train = X_prepared, y
            validation_data = None
        
        self._profile_step("Data splitting completed", split_start)
        print(f"[DEBUG-FF] Training set: {len(y_train):,} samples")
        if validation_data:
            print(f"[DEBUG-FF] Validation set: {len(y_val):,} samples")
        
        # Check for any NaN or inf values
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print(f"[DEBUG-FF] WARNING: X_train contains NaN or inf values")
        
        # Ensure arrays are contiguous and properly shaped
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        y_train = np.ascontiguousarray(y_train, dtype=np.float32)
        
        print(f"[DEBUG-FF] Final training data:")
        print(f"[DEBUG-FF]   X_train: {X_train.shape}, {X_train.dtype}")
        print(f"[DEBUG-FF]   y_train: {y_train.shape}, {y_train.dtype}")
        
        if validation_data:
            X_val, y_val = validation_data
            X_val = np.ascontiguousarray(X_val, dtype=np.float32)
            y_val = np.ascontiguousarray(y_val, dtype=np.float32)
            validation_data = (X_val, y_val)
            print(f"[DEBUG-FF]   X_val: {X_val.shape}, {X_val.dtype}")
            print(f"[DEBUG-FF]   y_val: {y_val.shape}, {y_val.dtype}")
        
        # Define optimized callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1 if self.verbose else 0
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1 if self.verbose else 0
            )
        ]
        
        # Train model with optimized settings
        fit_start = self._profile_step("Model fitting")
        print(f"[DEBUG-FF] Starting training loop...")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        self._profile_step("Model fitting completed", fit_start)
        
        self.is_trained = True
        self.training_history = {
            'loss': history.history['loss'],
            'mae': history.history['mae'],
            'epochs_trained': len(history.history['loss'])
        }
        
        if validation_data:
            self.training_history.update({
                'val_loss': history.history['val_loss'],
                'val_mae': history.history['val_mae']
            })
        
        # Calculate final training metrics
        final_loss = history.history['loss'][-1]
        final_mae = history.history['mae'][-1]
        
        if validation_data:
            final_val_loss = history.history['val_loss'][-1]
            final_val_mae = history.history['val_mae'][-1]
        else:
            final_val_loss = None
            final_val_mae = None
        
        # Performance summary
        total_training_time = time.time() - training_start
        print(f"[DEBUG-FF] === FEEDFORWARD TRAINING COMPLETED ===")
        print(f"[DEBUG-FF] Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} minutes)")
        print(f"[DEBUG-FF] Samples per second: {len(y_train)/total_training_time:.1f}")
        print(f"[DEBUG-FF] Epochs trained: {len(history.history['loss'])}")
        
        # Assert learning progress
        initial_loss = history.history['loss'][0]
        assert final_loss < initial_loss, f"Model failed to learn: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"
        
        results = {
            'status': 'success',
            'model_type': 'feedforward',
            'epochs_trained': len(history.history['loss']),
            'final_loss': float(final_loss),
            'final_mae': float(final_mae),
            'final_val_loss': float(final_val_loss) if final_val_loss else None,
            'final_val_mae': float(final_val_mae) if final_val_mae else None,
            'training_samples': len(X_train),
            'validation_samples': len(X_val) if validation_data else 0,
            'total_training_time': total_training_time,
            'samples_per_second': len(y_train)/total_training_time
        }
        
        print(f"[DEBUG-FF] Final MAE: {final_mae:.4f}")
        if final_val_mae:
            print(f"[DEBUG-FF] Final Val MAE: {final_val_mae:.4f}")
        print(f"[DEBUG-FF] Performance: {results['samples_per_second']:.1f} samples/sec")
        print(f"[DEBUG-FF] ======================================")
        
        return results
    
    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained feedforward model.
        
        Args:
            X_flat: Flat features for prediction (same format as TabNet)
            
        Returns:
            Predicted horse positions
        """
        # Assert model is trained
        assert self.is_trained, "Model must be trained before making predictions"
        assert self.model is not None, "Model is not available"
        
        # Assert data quality (no y for prediction)
        self.assert_data_quality_flat(X_flat)
        
        # Prepare features
        X_prepared = X_flat.astype(np.float32)
        
        # Make predictions
        predictions = self.model.predict(X_prepared, verbose=0)
        
        # Flatten predictions and ensure positive values
        predictions = predictions.flatten()
        predictions = np.maximum(predictions, 1.0)  # Minimum position is 1
        
        return predictions
    
    def evaluate(self, X_flat: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained feedforward model on test data.
        
        Args:
            X_flat: Flat test features (same format as TabNet)
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Assert model is trained
        assert self.is_trained, "Model must be trained before evaluation"
        assert self.model is not None, "Model is not available"
        
        # Assert data quality
        self.assert_data_quality_flat(X_flat, y)
        
        # Make predictions
        predictions = self.predict(X_flat)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        evaluation_results = {
            'mae': float(mae),
            'mse': float(mse), 
            'rmse': float(rmse),
            'r2': float(r2),
            'samples_evaluated': len(y)
        }
        
        if self.verbose:
            print(f"[DEBUG-FF] Evaluation Results:")
            print(f"[DEBUG-FF]   MAE: {mae:.4f}")
            print(f"[DEBUG-FF]   RMSE: {rmse:.4f}")
            print(f"[DEBUG-FF]   R²: {r2:.4f}")
            print(f"[DEBUG-FF]   Samples: {len(y):,}")
        
        return evaluation_results
    
    def save_model(self, path: str = None) -> Dict[str, str]:
        """
        Save the trained feedforward model with proper config updating.
        
        Args:
            path: Optional path to save the model (if None, uses default structure)
            
        Returns:
            Dictionary of saved file paths
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        from pathlib import Path
        from datetime import datetime
        import joblib
        import json
        from utils.model_manager import ModelManager
        
        # Create save directory structure similar to TabNet
        if path is None:
            model_manager = ModelManager()
            models_dir = Path(model_manager.model_dir)
            date_str = datetime.now().strftime('%Y-%m-%d')
            timestamp_str = datetime.now().strftime('%H%M%S')
            
            # Get database type from config
            from utils.env_setup import AppConfig
            config = AppConfig()
            db_type = config._config.base.active_db
            
            save_path = models_dir / date_str / f"{db_type}_{timestamp_str}"
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = Path(path).parent
            save_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save Keras model
        keras_model_path = save_path / "feedforward_model.keras"
        self.model.save(str(keras_model_path))
        saved_paths['feedforward_model'] = str(keras_model_path)
        
        # Save config and training results
        config_data = {
            'model_type': 'Feedforward',
            'config': self.config,
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics,
            'created_at': datetime.now().isoformat(),
            'db_type': db_type if path is None else 'unknown'
        }
        
        # Save config as JSON
        config_path = save_path / "feedforward_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        saved_paths['feedforward_config'] = str(config_path)
        
        # Update config.yaml with latest Feedforward model path
        if path is None:  # Only update config if using default path structure
            relative_path = save_path.relative_to(models_dir)
            self._update_config_feedforward(str(relative_path))
        
        if self.verbose:
            print(f"[PERF] Feedforward model saved:")
            print(f"  Keras model: {saved_paths['feedforward_model']}")
            print(f"  Config: {saved_paths['feedforward_config']}")
        
        return saved_paths
    
    def _update_config_feedforward(self, model_path: str):
        """Update config.yaml with latest Feedforward model path."""
        import yaml
        
        try:
            with open('config.yaml', 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update Feedforward model path
            if 'models' not in config_data:
                config_data['models'] = {}
            if 'latest_models' not in config_data['models']:
                config_data['models']['latest_models'] = {}
            
            config_data['models']['latest_models']['feedforward'] = model_path
            
            with open('config.yaml', 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
                
            if self.verbose:
                print(f"[PERF] Updated config.yaml with Feedforward model path: {model_path}")
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not update config.yaml: {e}")