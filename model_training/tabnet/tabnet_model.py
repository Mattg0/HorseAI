
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
import torch

from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.data_cleaning.tabnet_cleaner import TabNetDataCleaner
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

        # Create final dataset with only numeric features
        final_columns = available_features + ['final_position']
        temp_df = features_df[final_columns].dropna()
        
        # Filter out non-numeric columns (strings, objects) 
        numeric_columns = []
        for col in available_features:
            if col in temp_df.columns:
                # Check if column is numeric
                try:
                    pd.to_numeric(temp_df[col], errors='raise')
                    numeric_columns.append(col)
                    self.log_info(f"✅ {col}: numeric")
                except (ValueError, TypeError):
                    self.log_info(f"❌ {col}: non-numeric, skipping")
                    
        # Update feature columns to only include numeric ones
        self.feature_columns = numeric_columns
        available_features = numeric_columns
        
        # Create final dataset with only numeric features
        final_columns = available_features + ['final_position'] 
        self.complete_df = temp_df[final_columns]

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
        
        # Step 2.5: TabNet compatibility validation
        self.log_info("Validating TabNet compatibility...")
        cleaner = TabNetDataCleaner()
        is_compatible = cleaner.validate_tabnet_compatibility(X, verbose=True)
        
        if not is_compatible:
            self.log_info("Data is not TabNet compatible - applying cleaning...")
            X = cleaner.comprehensive_data_cleaning(X, verbose=True)
            self.log_info("Data cleaning completed - verifying compatibility...")
            cleaner.validate_tabnet_compatibility(X, verbose=True)
        
        # Step 2.5: Data validation and NaN handling
        self.log_info("Validating data quality...")
        
        # Check for NaN values in features
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            self.log_info(f"Found {nan_count} NaN values in features, filling with 0")
            X = X.fillna(0.0)
        
        # Check for infinite values in features
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            self.log_info(f"Found {inf_count} infinite values in features, replacing with 0")
            X = X.replace([np.inf, -np.inf], 0.0)
        
        # Check for NaN values in target
        target_nan_count = np.isnan(y).sum()
        if target_nan_count > 0:
            self.log_info(f"Found {target_nan_count} NaN values in target, filling with median")
            y_median = np.nanmedian(y)
            y = np.nan_to_num(y, nan=y_median)
        
        # Convert DataFrame to numpy array for consistent handling with float32 for compatibility
        X_array = X.values.astype(np.float32)
        y = y.astype(np.float32)
        self.log_info(f"Final feature matrix shape: {X_array.shape}, dtype: {X_array.dtype}")
        self.log_info(f"Final target vector shape: {y.shape}, dtype: {y.dtype}")

        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y, test_size=test_size, random_state=random_state
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

        # Extract fit() parameters if provided
        fit_params = {}
        if tabnet_params:
            # Extract max_epochs and patience for .fit() method
            fit_params['max_epochs'] = tabnet_params.pop('max_epochs', 200)
            fit_params['patience'] = tabnet_params.pop('patience', 20)
        else:
            fit_params['max_epochs'] = 200
            fit_params['patience'] = 20

        # Default TabNet parameters optimized for horse race prediction
        default_params = {
            'n_d': 32,  # Dimension of the feature transformer
            'n_a': 32,  # Dimension of the attention transformer
            'n_steps': 5,  # Number of decision steps
            'gamma': 1.5,  # Coefficient for feature reusage regularization
            'n_independent': 2,  # Number of independent GLU layers at each step
            'n_shared': 2,  # Number of shared GLU layers at each step
            'lambda_sparse': 1e-4,  # Sparsity regularization coefficient
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': {'lr': 2e-2},
            'mask_type': 'entmax',  # Type of mask function
            'scheduler_params': {'step_size': 30, 'gamma': 0.95},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'verbose': 1 if self.verbose else 0,
            'device_name': 'cpu'  # Use CPU to avoid MPS float64 compatibility issues
        }

        # Update with user-provided parameters (after removing fit params)
        if tabnet_params:
            default_params.update(tabnet_params)

        # Create TabNet regressor with MPS device
        print(f"TabNet using device: {default_params['device_name']}")
        self.tabnet_model = TabNetRegressor(**default_params)

        # Train the model
        # Convert pandas Series to numpy arrays before reshape
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

        # Train with history tracking
        history = self.tabnet_model.fit(
            X_train=X_train,
            y_train=y_train_array.reshape(-1, 1),
            eval_set=[(X_test, y_test_array.reshape(-1, 1))],
            max_epochs=fit_params['max_epochs'],
            patience=fit_params['patience'],
            batch_size=1024,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False
        )

        # Get actual number of epochs trained (TabNet stops early with patience)
        actual_epochs = len(history) if history is not None else 200

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
            'n_features': X_train.shape[1],  # Compatibility field
            'n_epochs': actual_epochs,  # Actual epochs trained
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

        # Import needed modules first
        from pathlib import Path
        from datetime import datetime
        import joblib
        import json
        
        # Create save directory
        models_dir = Path(self.model_manager.model_dir)
        date_str = datetime.now().strftime('%Y-%m-%d')
        timestamp_str = datetime.now().strftime('%H%M%S')
        db_type = self.config._config.base.active_db
        
        save_path = models_dir / date_str / f"{db_type}_{timestamp_str}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save TabNet model (TabNet automatically adds .zip extension)
        tabnet_model_path = save_path / "tabnet_model"
        self.tabnet_model.save_model(str(tabnet_model_path))
        # TabNet creates a .zip file automatically
        actual_model_path = str(tabnet_model_path) + ".zip"
        saved_paths['tabnet_model'] = actual_model_path
        
        # Save scaler
        tabnet_scaler_path = save_path / "tabnet_scaler.joblib"
        joblib.dump(self.scaler, tabnet_scaler_path)
        saved_paths['tabnet_scaler'] = str(tabnet_scaler_path)
        
        # Save feature columns and config
        tabnet_config_path = save_path / "tabnet_config.json"
        
        # Create JSON-safe training results
        def make_json_safe(obj):
            """Recursively convert objects to JSON-safe format."""
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {key: make_json_safe(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_safe(item) for item in obj]
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                return str(obj)
        
        json_safe_results = make_json_safe(self.training_results) if self.training_results else {}
        
        config_data = {
            'feature_columns': self.feature_columns,
            'training_results': json_safe_results,
            'db_type': db_type,
            'created_at': datetime.now().isoformat(),
            'model_type': 'TabNet'
        }

        with open(tabnet_config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        saved_paths['tabnet_config'] = str(tabnet_config_path)

        # Save feature selector if it exists (from automatic feature selection)
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            try:
                feature_selector_path = save_path / "feature_selector.json"
                self.feature_selector.save(str(feature_selector_path))
                saved_paths['feature_selector'] = str(feature_selector_path)
                self.log_info(f"  Feature selector: {feature_selector_path}")
            except Exception as e:
                self.log_info(f"  Warning: Could not save feature selector: {e}")

        # Update config.yaml with latest TabNet model path
        relative_path = save_path.relative_to(models_dir)
        self._update_config_tabnet(str(relative_path))

        self.log_info(f"TabNet model saved successfully with scaler and config")
        self.log_info(f"  Model: {saved_paths.get('tabnet_model', 'Not saved')}")
        self.log_info(f"  Scaler: {saved_paths.get('tabnet_scaler', 'Not saved')}")
        self.log_info(f"  Config: {saved_paths.get('tabnet_config', 'Not saved')}")
        return saved_paths
    
    def _update_config_tabnet(self, model_path: str):
        """Update config.yaml with latest TabNet model path."""
        import yaml
        
        try:
            with open('config.yaml', 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update TabNet model path
            if 'models' not in config_data:
                config_data['models'] = {}
            if 'latest_models' not in config_data['models']:
                config_data['models']['latest_models'] = {}
            
            config_data['models']['latest_models']['tabnet'] = model_path
            
            with open('config.yaml', 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
                
            self.log_info(f"Updated config.yaml with TabNet model path: {model_path}")
            
        except Exception as e:
            self.log_info(f"Warning: Could not update config.yaml: {e}")


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