
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor

from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from utils.model_manager import ModelManager

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False


class TabNetModel:
    """
    TabNet-based horse race prediction model that uses raw features + musique-derived features.
    Follows the same pattern as HorseRaceModel but skips embeddings in favor of tabular learning.
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = False):
        """Initialize the TabNet model with configuration."""
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Check TabNet availability
        if not TABNET_AVAILABLE:
            raise ImportError(
                "pytorch_tabnet is required for TabNetModel. "
                "Install with: pip install pytorch-tabnet"
            )

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        db_path = get_sqlite_dbpath(self.db_type)
        self.model_manager = ModelManager()

        # Initialize data orchestrator
        self.orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=db_path,
            verbose=verbose
        )

        # Configure orchestrator to skip embeddings and keep raw features
        self.orchestrator.clean_after_embedding = False
        self.orchestrator.keep_identifiers = True

        # Model containers
        self.tabnet_model = None
        self.scaler = None
        self.training_results = None

        # Data containers
        self.complete_df = None
        self.feature_columns = None

        self.log_info(f"Initialized TabNetModel with database: {self.db_type}")

    def log_info(self, message):
        """Simple logging method."""
        if self.verbose:
            print(f"[TabNetModel] {message}")

    def load_and_prepare_data(self, limit: Optional[int] = None,
                              race_filter: Optional[str] = None,
                              date_filter: Optional[str] = None) -> Dict[str, Any]:
        """Load and prepare dataset with raw features + musique-derived features."""

        self.log_info("Loading historical race data...")

        # Load historical data
        df_historical = self.orchestrator.load_historical_races(
            limit=limit,
            race_filter=race_filter,
            date_filter=date_filter,
            use_cache=True
        )

        # Prepare features but skip embeddings
        self.log_info("Preparing raw features with musique-derived statistics...")
        features_df = self.orchestrator.prepare_features(df_historical)

        # Select relevant raw features for TabNet
        self.feature_columns = self._select_tabnet_features(features_df)
        
        # Create final dataset with selected features
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        missing_features = [col for col in self.feature_columns if col not in features_df.columns]
        
        if missing_features:
            self.log_info(f"Warning: Missing features: {missing_features}")

        # Create final dataset
        final_columns = available_features + ['final_position']
        self.complete_df = features_df[final_columns].dropna()

        self.log_info(
            f"Dataset prepared: {len(self.complete_df)} records, {len(available_features)} features"
        )

        return {
            'status': 'success',
            'records': len(self.complete_df),
            'features': len(available_features),
            'selected_features': available_features,
            'missing_features': missing_features
        }

    def _select_tabnet_features(self, df: pd.DataFrame) -> list:
        """Select appropriate features for TabNet model."""
        
        # Musique-derived features (performance statistics)
        musique_features = [
            col for col in df.columns 
            if any(prefix in col for prefix in ['che_global_', 'che_weighted_', 'che_bytype_', 
                                               'joc_global_', 'joc_weighted_', 'joc_bytype_'])
        ]

        # Static race features
        static_features = [
            'age', 'dist', 'temperature', 'cotedirect', 'corde', 
            'typec', 'natpis', 'meteo', 'nbprt'
        ]

        # Performance statistics
        performance_features = [
            'victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere',
            'ratio_victoires', 'ratio_places', 'gains_par_course'
        ]

        # Combine all feature types
        all_features = musique_features + static_features + performance_features
        
        # Filter to only include features that exist in the dataframe
        available_features = [col for col in all_features if col in df.columns]
        
        self.log_info(f"Selected {len(available_features)} features for TabNet:")
        self.log_info(f"  - Musique features: {len([f for f in available_features if any(p in f for p in ['che_', 'joc_'])])}")
        self.log_info(f"  - Static features: {len([f for f in available_features if f in static_features])}")
        self.log_info(f"  - Performance features: {len([f for f in available_features if f in performance_features])}")

        return available_features

    def train(self, limit: Optional[int] = None,
              race_filter: Optional[str] = None,
              date_filter: Optional[str] = None,
              test_size: float = 0.2,
              random_state: int = 42,
              tabnet_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete TabNet training workflow.
        """
        start_time = datetime.now()
        self.log_info("Starting TabNet training workflow...")

        # Step 1: Load and prepare data
        data_prep_results = self.load_and_prepare_data(limit, race_filter, date_filter)

        if self.complete_df is None or len(self.complete_df) == 0:
            raise ValueError("Data preparation failed - no data available for training")

        # Step 2: Prepare features and target
        X = self.complete_df.drop('final_position', axis=1)
        y = self.complete_df['final_position'].values

        self.log_info(f"Feature matrix shape: {X.shape}")
        self.log_info(f"Target vector shape: {y.shape}")

        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Step 4: Scale features
        self.log_info("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Step 5: Train TabNet model
        self.log_info("Training TabNet model...")
        tabnet_results = self._train_tabnet_model(
            X_train_scaled, y_train, X_test_scaled, y_test, tabnet_params
        )

        # Step 6: Compile results
        training_time = (datetime.now() - start_time).total_seconds()

        self.training_results = {
            'status': 'success',
            'training_time': training_time,
            'model_type': 'TabNet',
            'data_preparation': data_prep_results,
            'tabnet_results': tabnet_results,
            'training_config': {
                'test_size': test_size,
                'random_state': random_state,
                'db_type': self.db_type,
                'feature_scaling': True
            }
        }

        self.log_info(f"Training completed in {training_time:.2f} seconds")
        self.log_info(f"TabNet Test MAE: {tabnet_results['test_mae']:.4f}")
        self.log_info(f"TabNet Test R²: {tabnet_results['test_r2']:.4f}")

        return self.training_results

    def _train_tabnet_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           tabnet_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Train TabNet model with provided data."""

        # Default TabNet parameters optimized for horse race prediction
        default_params = {
            'n_d': 32,  # Dimension of the feature transformer
            'n_a': 32,  # Dimension of the attention transformer
            'n_steps': 5,  # Number of decision steps
            'gamma': 1.5,  # Coefficient for feature reusage regularization
            'n_independent': 2,  # Number of independent GLU layers at each step
            'n_shared': 2,  # Number of shared GLU layers at each step
            'lambda_sparse': 1e-4,  # Sparsity regularization coefficient
            'optimizer_fn': 'adam',
            'optimizer_params': {'lr': 2e-2},
            'mask_type': 'entmax',  # Type of mask function
            'scheduler_params': {'step_size': 30, 'gamma': 0.95},
            'scheduler_fn': 'StepLR',
            'verbose': 1 if self.verbose else 0,
            'device_name': 'cpu'
        }

        # Update with user-provided parameters
        if tabnet_params:
            default_params.update(tabnet_params)

        # Create TabNet regressor with CPU device
        print(f"TabNet using device: {default_params['device_name']}")
        self.tabnet_model = TabNetRegressor(**default_params)

        # Train the model
        self.tabnet_model.fit(
            X_train=X_train,
            y_train=y_train.reshape(-1, 1),
            eval_set=[(X_test, y_test.reshape(-1, 1))],
            max_epochs=200,
            patience=20,
            batch_size=1024,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False
        )

        # Generate predictions
        train_preds = self.tabnet_model.predict(X_train).flatten()
        test_preds = self.tabnet_model.predict(X_test).flatten()

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)

        # Get feature importance
        feature_importance = self.tabnet_model.feature_importances_
        
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
            'feature_importance': feature_importance.tolist(),
            'model_params': default_params
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from trained TabNet model."""
        if self.tabnet_model is None:
            raise ValueError("Model must be trained first")

        if self.feature_columns is None:
            raise ValueError("Feature columns not available")

        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.tabnet_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained TabNet model."""
        if self.tabnet_model is None:
            raise ValueError("Model must be trained first")

        if self.scaler is None:
            raise ValueError("Scaler not available - model must be trained first")

        # Select and scale features
        X_selected = X[self.feature_columns]
        X_scaled = self.scaler.transform(X_selected)

        # Generate predictions
        predictions = self.tabnet_model.predict(X_scaled)
        return predictions.flatten()

    def save_model(self) -> Dict[str, str]:
        """Save the trained TabNet model and associated artifacts."""
        if self.tabnet_model is None:
            raise ValueError("No trained model to save")

        if self.scaler is None:
            raise ValueError("No scaler available to save")

        # Create feature state for compatibility
        feature_state = {
            'model_type': 'TabNet',
            'training_results': self.training_results,
            'timestamp': datetime.now().isoformat(),
            'feature_columns': self.feature_columns
        }

        # Save using model manager with proper TabNet parameters
        saved_paths = self.model_manager.save_models(
            rf_model=None,  # No RF model for TabNet
            lstm_model=None,  # No LSTM model for TabNet
            tabnet_model=self.tabnet_model,
            tabnet_scaler=self.scaler,
            tabnet_feature_columns=self.feature_columns,
            feature_state=feature_state
        )

        self.log_info(f"TabNet model saved successfully with scaler and config")
        self.log_info(f"  Model: {saved_paths.get('tabnet_model', 'Not saved')}")
        self.log_info(f"  Scaler: {saved_paths.get('tabnet_scaler', 'Not saved')}")
        self.log_info(f"  Config: {saved_paths.get('tabnet_config', 'Not saved')}")
        return saved_paths


def main(progress_callback=None):
    """
    Main function to train the TabNet model from IDE.
    """
    if progress_callback:
        progress_callback(5, "Initializing TabNet model...")

    # Initialize the model
    model = TabNetModel(verbose=True)

    if progress_callback:
        progress_callback(10, "Loading and preparing data...")

    # Train the model
    results = model.train(
        limit=None,  # Set to limit races for testing
        race_filter=None,  # Set to 'A' for Attele, 'P' for Plat, etc.
        date_filter=None,  # Set date filter if needed
        test_size=0.2,
        random_state=42
    )

    if progress_callback:
        progress_callback(90, "Saving trained model...")

    # Save the trained model
    saved_paths = model.save_model()

    if progress_callback:
        progress_callback(100, "Training completed successfully!")

    # Print summary results
    print("\n" + "=" * 50)
    print("TABNET TRAINING COMPLETED")
    print("=" * 50)
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Features used: {results['tabnet_results']['features']}")
    print(f"Test MAE: {results['tabnet_results']['test_mae']:.4f}")
    print(f"Test RMSE: {results['tabnet_results']['test_rmse']:.4f}")
    print(f"Test R²: {results['tabnet_results']['test_r2']:.4f}")
    
    # Display top feature importance
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    feature_importance = model.get_feature_importance(top_n=10)
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()