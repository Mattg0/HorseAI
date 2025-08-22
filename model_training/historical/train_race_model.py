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

from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from model_training.regressions.isotonic_calibration import CalibratedRegressor
from utils.model_manager import ModelManager


class HorseRaceModel:
    """
    3-Model Ensemble horse race prediction model that trains RF, LSTM, and TabNet models
    in a single streamlined workflow with fail-fast approach.
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

        # Model containers
        self.rf_model = None
        self.lstm_model = None
        self.tabnet_model = None
        self.training_results = None

        # Data containers
        self.rf_data = None
        self.lstm_data = None
        self.tabnet_data = None

        self.log_info(f"Initialized 3-Model training pipeline with database: {self.db_type}")
        self.log_info("Individual models will be trained independently - blending occurs during prediction")

    def log_info(self, message):
        """Simple logging method."""
        if self.verbose:
            print(f"[HorseRaceModel] {message}")

    def load_and_prepare_data(self, limit: Optional[int] = None,
                              race_filter: Optional[str] = None,
                              date_filter: Optional[str] = None) -> Dict[str, Any]:
        """Load and prepare complete dataset once."""

        self.log_info("Loading historical race data...")

        # Load historical data
        df_historical = self.orchestrator.load_historical_races(
            limit=limit,
            race_filter=race_filter,
            date_filter=date_filter,
            use_cache=True
        )

        # Prepare complete dataset with ALL features
        self.log_info("Preparing complete feature set...")
        self.complete_df = self.orchestrator.prepare_complete_dataset(
            df_historical,
            use_cache=True
        )

        self.log_info(
            f"Complete dataset prepared: {len(self.complete_df)} records, {len(self.complete_df.columns)} features")

        return {
            'status': 'success',
            'records': len(self.complete_df),
            'features': len(self.complete_df.columns)
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

        # Step 3.5: Extract TabNet features just before training
        tabnet_df = self.orchestrator.prepare_tabnet_features(self.complete_df, use_cache=True)
        
        # Extract features and target from TabNet DataFrame
        if 'final_position' in tabnet_df.columns:
            X_tabnet = tabnet_df.drop('final_position', axis=1)
            y_tabnet = tabnet_df['final_position']
        else:
            raise ValueError("Target column 'final_position' not found in TabNet features")
            
        X_tabnet_train, X_tabnet_test, y_tabnet_train, y_tabnet_test = train_test_split(
            X_tabnet, y_tabnet, test_size=test_size, random_state=random_state
        )

        # Step 4: Train Random Forest model (fail-fast)
        self.log_info("Training Random Forest model...")
        rf_results = self._train_rf_model(X_rf_train, y_rf_train, X_rf_test, y_rf_test)
        if rf_results.get('status') == 'failed':
            raise RuntimeError(f"RF model training failed: {rf_results.get('error', 'Unknown error')}")

        # Step 5: Train LSTM model (fail-fast)
        self.log_info("Training LSTM model...")
        lstm_results = self._train_lstm_model(
            X_seq_train, X_static_train, y_lstm_train,
            X_seq_test, X_static_test, y_lstm_test
        )
        if lstm_results.get('status') == 'failed':
            raise RuntimeError(f"LSTM model training failed: {lstm_results.get('error', 'Unknown error')}")

        # Step 6: Train TabNet model (fail-fast)
        self.log_info("Training TabNet model...")
        tabnet_results = self._train_tabnet_model(
            X_tabnet_train, y_tabnet_train, X_tabnet_test, y_tabnet_test
        )
        if tabnet_results.get('status') == 'failed':
            raise RuntimeError(f"TabNet model training failed: {tabnet_results.get('error', 'Unknown error')}")

        # Step 7: Training completed - no blending during training

        # Step 6: Compile complete results
        training_time = (datetime.now() - start_time).total_seconds()

        self.training_results = {
            'status': 'success',
            'training_time': training_time,
            'model_type': 'individual_models',
            'data_preparation': data_prep_results,
            'rf_results': rf_results,
            'lstm_results': lstm_results,
            'tabnet_results': tabnet_results,
            'training_config': {
                'test_size': test_size,
                'random_state': random_state,
                'db_type': self.db_type
            }
        }

        self.log_info(f"3-Model training completed in {training_time:.2f} seconds")
        self.log_info(f"RF Test MAE: {rf_results['test_mae']:.4f}")
        self.log_info(f"LSTM Test MAE: {lstm_results['test_mae']:.4f}")
        self.log_info(f"TabNet Test MAE: {tabnet_results['test_mae']:.4f}")
        self.log_info("Individual models trained successfully - blending will occur during prediction")

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

        # Configure training
        from tensorflow.keras.callbacks import EarlyStopping

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]

        # Train the model with provided data
        history = self.lstm_model.fit(
            [X_seq_train, X_static_train], y_train,
            validation_data=([X_seq_test, X_static_test], y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0 if not self.verbose else 1
        )

        # Generate predictions
        train_preds = self.lstm_model.predict([X_seq_train, X_static_train], verbose=0)
        test_preds = self.lstm_model.predict([X_seq_test, X_static_test], verbose=0)

        # Flatten predictions if needed
        train_preds = train_preds.flatten()
        test_preds = test_preds.flatten()

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)

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

    def _train_tabnet_model(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train TabNet model with provided split data."""
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Import torch optimizer functions
            import torch.optim as optim
            
            # TabNet parameters optimized for horse race prediction
            tabnet_params = {
                'n_d': 32,  # Dimension of the feature transformer
                'n_a': 32,  # Dimension of the attention transformer
                'n_steps': 5,  # Number of decision steps
                'gamma': 1.5,  # Coefficient for feature reusage regularization
                'n_independent': 2,  # Number of independent GLU layers
                'n_shared': 2,  # Number of shared GLU layers
                'lambda_sparse': 1e-4,  # Sparsity regularization coefficient
                'optimizer_fn': optim.Adam,  # Use torch optimizer function
                'optimizer_params': {'lr': 2e-2},
                'mask_type': 'entmax',
                'scheduler_params': {'step_size': 30, 'gamma': 0.95},
                'scheduler_fn': optim.lr_scheduler.StepLR,  # Use torch scheduler function
                'verbose': 1 if self.verbose else 0,
                'device_name': 'cpu'  # Use CPU for better compatibility
            }
            
            # Create and train TabNet regressor
            print(f"TabNet using device: {tabnet_params['device_name']}")
            self.tabnet_model = TabNetRegressor(**tabnet_params)
            
            # Convert pandas Series to numpy arrays and ensure float32 for MPS compatibility
            X_train_np = X_train_scaled if isinstance(X_train_scaled, np.ndarray) else X_train_scaled.values
            X_test_np = X_test_scaled if isinstance(X_test_scaled, np.ndarray) else X_test_scaled.values
            y_train_np = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train.values if hasattr(y_train, 'values') else y_train
            y_test_np = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test.values if hasattr(y_test, 'values') else y_test
            
            # Convert to float32 for compatibility
            X_train_np = X_train_np.astype(np.float32)
            X_test_np = X_test_np.astype(np.float32)
            y_train_np = y_train_np.astype(np.float32)
            y_test_np = y_test_np.astype(np.float32)
            
            # Train the model with numpy arrays
            self.tabnet_model.fit(
                X_train=X_train_np,
                y_train=y_train_np.reshape(-1, 1),  # 2D array for TabNet
                eval_set=[(X_test_np, y_test_np.reshape(-1, 1))],
                max_epochs=100,
                patience=15,
                batch_size=256,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
            
            # Generate predictions using numpy arrays
            train_preds = self.tabnet_model.predict(X_train_np).flatten()
            test_preds = self.tabnet_model.predict(X_test_np).flatten()
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_preds)
            test_mae = mean_absolute_error(y_test, test_preds)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            test_r2 = r2_score(y_test, test_preds)
            
            return {
                'model_type': 'TabNet',
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X_train.shape[1],
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'test_r2': float(test_r2),
                'test_predictions': test_preds.tolist(),
                'test_targets': y_test.tolist(),
                'training_history': {'epochs': len(self.tabnet_model.history['loss'])}
            }
            
        except Exception as e:
            self.log_info(f"TabNet model training failed: {str(e)}")
            return {
                'status': 'failed',
                'error': f'TabNet model training failed: {str(e)}'
            }
    

    def save_incremental_model(self, rf_model, lstm_model=None, tabnet_model=None, orchestrator=None):
        """
        Save all three models trained incrementally on daily data.

        Args:
            rf_model: Random Forest model to save (required)
            lstm_model: LSTM model to save (required)
            tabnet_model: TabNet model to save (required)
            orchestrator: Orchestrator with feature state (optional)

        Returns:
            Dictionary with paths to saved artifacts
        """
        # Fail-fast validation
        if rf_model is None:
            raise ValueError("RF model is required for 3-model ensemble")
        if lstm_model is None:
            raise ValueError("LSTM model is required for 3-model ensemble")
        if tabnet_model is None:
            raise ValueError("TabNet model is required for 3-model ensemble")
        from utils.model_manager import get_model_manager

        print("===== SAVING INCREMENTAL MODEL =====")

        # Prepare feature state if orchestrator provided
        feature_state = None
        if orchestrator and hasattr(orchestrator, 'preprocessing_params'):
            feature_state = {
                'preprocessing_params': orchestrator.preprocessing_params,
                'embedding_dim': getattr(orchestrator, 'embedding_dim', 16),
                'sequence_length': getattr(orchestrator, 'sequence_length', 5)
            }

        # Get the model manager and save all three models
        model_manager = get_model_manager()
        saved_paths = model_manager.save_models(
            rf_model=rf_model,
            lstm_model=lstm_model,
            tabnet_model=tabnet_model,
            feature_state=feature_state
        )

        print(f"Incremental model saved successfully")

        return saved_paths


def main(progress_callback=None):
    """
    Main function to train the hybrid model from IDE.
    Modify the parameters below as needed.
    """
    if progress_callback:
        progress_callback(5, "Initializing model...")

    # Initialize the model
    model = HorseRaceModel(verbose=True)

    if progress_callback:
        progress_callback(10, "Loading and preparing data...")

    # Train the model with all available data
    results = model.train(
        limit=None,  # Process ALL races (no limit)
        race_filter=None,  # Use all race types
        date_filter=None,  # Use all available dates
        test_size=0.2,  # 20% for testing
        random_state=42  # For reproducible results
    )

    if progress_callback:
        progress_callback(90, "Saving trained models...")

    # Save all three trained models (fail-fast approach)
    if model.rf_model is None:
        raise RuntimeError("RF model was not trained properly")
    if model.lstm_model is None:
        raise RuntimeError("LSTM model was not trained properly")
    if model.tabnet_model is None:
        raise RuntimeError("TabNet model was not trained properly")
        
    saved_paths = model.model_manager.save_models(
        rf_model=model.rf_model,
        lstm_model=model.lstm_model,
        tabnet_model=model.tabnet_model
    )

    if progress_callback:
        progress_callback(100, "Training completed successfully!")

    # Print detailed summary results
    print("\n" + "=" * 50)
    print("3-MODEL TRAINING COMPLETED")
    print("=" * 50)
    print(f"Training time: {results['training_time']:.2f} seconds")
    print("Individual models trained successfully - blending will occur during prediction")
    
    # RF model results
    print(f"\n--- Random Forest Results ---")
    print(f"Features used: {results['rf_results']['features']}")
    print(f"Train samples: {results['rf_results']['train_samples']}")
    print(f"Test samples: {results['rf_results']['test_samples']}")
    print(f"Test MAE: {results['rf_results']['test_mae']:.4f}")
    print(f"Test RMSE: {results['rf_results']['test_rmse']:.4f}")
    print(f"Test R²: {results['rf_results']['test_r2']:.4f}")
    
    # LSTM model results (must be successful due to fail-fast)
    print(f"\n--- LSTM Results ---")
    print(f"Train samples: {results['lstm_results']['train_samples']}")
    print(f"Test samples: {results['lstm_results']['test_samples']}")
    print(f"Sequence length: {results['lstm_results']['sequence_length']}")
    print(f"Sequential features: {results['lstm_results']['sequential_features']}")
    print(f"Static features: {results['lstm_results']['static_features']}")
    print(f"Test MAE: {results['lstm_results']['test_mae']:.4f}")
    print(f"Test RMSE: {results['lstm_results']['test_rmse']:.4f}")
    print(f"Test R²: {results['lstm_results']['test_r2']:.4f}")
    print(f"Training epochs: {results['lstm_results']['training_history']['epochs']}")
    
    # TabNet model results (must be successful due to fail-fast)
    print(f"\n--- TabNet Results ---")
    print(f"Train samples: {results['tabnet_results']['train_samples']}")
    print(f"Test samples: {results['tabnet_results']['test_samples']}")
    print(f"Features: {results['tabnet_results']['features']}")
    print(f"Test MAE: {results['tabnet_results']['test_mae']:.4f}")
    print(f"Test RMSE: {results['tabnet_results']['test_rmse']:.4f}")
    print(f"Test R²: {results['tabnet_results']['test_r2']:.4f}")
    
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