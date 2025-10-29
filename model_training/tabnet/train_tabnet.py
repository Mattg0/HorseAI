#!/usr/bin/env python3
"""
TabNet Training Script for Horse Race Prediction

This script trains a TabNet model using the same data pipeline and train/test split
as the existing HorseRaceModel. It integrates with the FeatureCalculator (StaticFeatureCalculator)
for musique preprocessing while feeding raw features to TabNet.

Compatible with IDE workflow and follows the same patterns as train_race_model.py.
"""

import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Core imports
from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from core.calculators.static_feature_calculator import FeatureCalculator
from utils.model_manager import ModelManager

# TabNet model import
from model_training.tabnet.tabnet_model import TabNetModel

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False


class TabNetTrainer:
    """
    TabNet trainer that follows the same workflow as HorseRaceModel 
    but uses raw features + musique-derived features instead of embeddings.
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = False):
        """Initialize the TabNet trainer with configuration."""
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Check TabNet availability
        if not TABNET_AVAILABLE:
            raise ImportError(
                "pytorch_tabnet is required for TabNet training. "
                "Install with: pip install pytorch-tabnet"
            )

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        db_path = get_sqlite_dbpath(self.db_type)
        self.model_manager = ModelManager()

        # Initialize data orchestrator for raw feature preparation
        self.orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=db_path,
            verbose=verbose
        )

        # Model containers
        self.tabnet_model = None
        self.scaler = None
        self.training_results = None

        # Data containers
        self.complete_df = None
        self.feature_columns = None

        self.log_info(f"Initialized TabNetTrainer with database: {self.db_type}")

    def log_info(self, message):
        """Simple logging method."""
        if self.verbose:
            print(f"[TabNetTrainer] {message}")

    def load_and_prepare_data(self, limit: Optional[int] = None,
                              race_filter: Optional[str] = None,
                              date_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and prepare dataset using the same pipeline as HorseRaceModel 
        but with TabNet-specific feature preparation.
        """
        self.log_info("Loading historical race data...")

        # Load historical data using the orchestrator
        df_historical = self.orchestrator.load_historical_races(
            limit=limit,
            race_filter=race_filter,
            date_filter=date_filter,
            use_cache=True
        )

        self.log_info(f"Loaded {len(df_historical)} records from {df_historical['comp'].nunique() if 'comp' in df_historical.columns else 'unknown'} races")

        # Step 1: Apply FeatureCalculator with temporal calculations (leakage fix)
        self.log_info("Applying FeatureCalculator with temporal calculations...")
        db_path = get_sqlite_dbpath(self.db_type)
        df_with_features = FeatureCalculator.calculate_all_features(
            df_historical,
            use_temporal=True,
            db_path=db_path
        )

        # Step 2: Use orchestrator's TabNet feature preparation
        self.log_info("Preparing TabNet-specific features...")
        self.complete_df = self.orchestrator.prepare_tabnet_features(
            df_with_features,
            use_cache=True
        )

        self.log_info(
            f"TabNet dataset prepared: {len(self.complete_df)} records, {len(self.complete_df.columns)} features"
        )

        return {
            'status': 'success',
            'records': len(self.complete_df),
            'features': len(self.complete_df.columns),
            'races': self.complete_df['comp'].nunique() if 'comp' in self.complete_df.columns else 0
        }

    def prepare_features_for_training(self) -> tuple:
        """
        Prepare features and target for TabNet training.
        Uses the same feature selection logic as TabNetModel.
        """
        if self.complete_df is None:
            raise ValueError("Data must be loaded first. Call load_and_prepare_data().")

        # Define feature categories for TabNet
        self.feature_columns = self._select_tabnet_features(self.complete_df)

        # Create final dataset with selected features
        available_features = [col for col in self.feature_columns if col in self.complete_df.columns]
        missing_features = [col for col in self.feature_columns if col not in self.complete_df.columns]

        if missing_features:
            self.log_info(f"Warning: Missing features: {missing_features}")

        # Debug: Print available columns
        self.log_info(f"Available columns in complete_df: {list(self.complete_df.columns)}")
        self.log_info(f"Looking for target column 'final_position': {'final_position' in self.complete_df.columns}")

        # Apply ONLY transformations (NO cleanup - we want to keep ALL 90 features)
        from core.data_cleaning.feature_cleanup import FeatureCleaner
        cleaner = FeatureCleaner()

        # Separate features and target
        X = self.complete_df[available_features].copy()

        # SKIP clean_features() - we want ALL features including bytype, global, etc.
        # These are critical for model performance (che_bytype_dnf_rate is 20% importance!)
        # X = cleaner.clean_features(X)  # ← DISABLED

        # Only apply transformations (recence_log, cotedirect_log)
        X = cleaner.apply_transformations(X)

        # Update feature_columns to match the ACTUAL features after transformations
        self.feature_columns = list(X.columns)
        self.log_info(f"✓ Feature columns after transformations: {len(self.feature_columns)} features (keeping ALL features)")

        # FAIL-FAST: Validate we have the expected ~90 features (89 after recence/cotedirect transform)
        expected_min_features = 85  # Should be ~89, but allow some variation
        if len(self.feature_columns) < expected_min_features:
            error_msg = f"""
❌ FEATURE COUNT TOO LOW!

Created: {len(self.feature_columns)} features
Expected: ~89 features (minimum {expected_min_features})

This suggests the feature calculation is not creating all required features.
Check that:
1. FeatureCalculator.calculate_all_features() is creating musique features (che_*, joc_*)
2. QuinteFeatureCalculator is adding quinté-specific features
3. clean_features() is NOT being called (it removes important features)

Current features:
{self.feature_columns}
"""
            self.log_info(error_msg)
            raise ValueError(f"Feature count too low: {len(self.feature_columns)} < {expected_min_features}")

        # Verify critical features are present
        critical_features = ['che_bytype_dnf_rate', 'che_global_avg_pos', 'joc_bytype_avg_pos']
        missing_critical = [f for f in critical_features if f not in self.feature_columns]
        if missing_critical:
            error_msg = f"""
❌ CRITICAL FEATURES MISSING!

Missing: {missing_critical}

These features are essential for model performance. Check feature calculation pipeline.
"""
            self.log_info(error_msg)
            raise ValueError(f"Critical features missing: {missing_critical}")

        y = self.complete_df['final_position'].copy() if 'final_position' in self.complete_df.columns else None

        if y is None:
            # Try alternative target columns
            alternative_targets = ['cl', 'narrivee', 'position']
            for alt_target in alternative_targets:
                if alt_target in self.complete_df.columns:
                    self.log_info(f"Using alternative target column '{alt_target}' instead of 'final_position'")
                    y = self.complete_df[alt_target].copy()
                    break

            if y is None:
                raise ValueError(f"No target column found in dataset. Available columns: {list(self.complete_df.columns)}")

        # Remove rows with missing target values
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        self.log_info(f"Feature matrix shape: {X.shape}")
        self.log_info(f"Target vector shape: {y.shape}")
        self.log_info(f"Selected features (pre-transformations): {len(available_features)}")
        self.log_info(f"Final features (post-transformations): {len(self.feature_columns)}")
        self.log_info(f"✅ All validation checks passed!")

        return X, y

    def _select_tabnet_features(self, df: pd.DataFrame) -> list:
        """
        Select appropriate features for TabNet model.
        Mirrors the logic from TabNetModel._select_tabnet_features.
        """
        # Musique-derived features (performance statistics)
        musique_features = [
            col for col in df.columns 
            if any(prefix in col for prefix in ['che_global_', 'che_weighted_', 'che_bytype_', 
                                               'joc_global_', 'joc_weighted_', 'joc_bytype_'])
        ]

        # Static race features
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

        # Combine all feature types
        all_features = musique_features + static_features + performance_features + temporal_features
        
        # Filter to only include features that exist in the dataframe
        available_features = [col for col in all_features if col in df.columns]
        
        self.log_info(f"Selected {len(available_features)} features for TabNet:")
        self.log_info(f"  - Musique features: {len([f for f in available_features if any(p in f for p in ['che_', 'joc_'])])}")
        self.log_info(f"  - Static features: {len([f for f in available_features if f in static_features])}")
        self.log_info(f"  - Performance features: {len([f for f in available_features if f in performance_features])}")
        self.log_info(f"  - Temporal features: {len([f for f in available_features if f in temporal_features])}")

        return available_features

    def train(self, limit: Optional[int] = None,
              race_filter: Optional[str] = None,
              date_filter: Optional[str] = None,
              test_size: float = 0.2,
              random_state: int = 42,
              tabnet_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete TabNet training workflow following the same pattern as HorseRaceModel.train().
        """
        start_time = datetime.now()
        self.log_info("Starting TabNet training workflow...")

        # Step 1: Load and prepare data (same as HorseRaceModel)
        data_prep_results = self.load_and_prepare_data(limit, race_filter, date_filter)

        # Step 2: Prepare features and target
        X, y = self.prepare_features_for_training()

        # Step 3: Split data (same random_state and test_size as HorseRaceModel)
        self.log_info("Splitting data for training and testing...")
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

        # Step 6: Compile results (same format as HorseRaceModel)
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
                'feature_scaling': True,
                'musique_preprocessing': True
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
        import torch.optim as optim
        from pytorch_tabnet.callbacks import EarlyStopping
        
        default_params = {
            'n_d': 32,  # Dimension of the feature transformer
            'n_a': 32,  # Dimension of the attention transformer
            'n_steps': 5,  # Number of decision steps
            'gamma': 1.5,  # Coefficient for feature reusage regularization
            'n_independent': 2,  # Number of independent GLU layers at each step
            'n_shared': 2,  # Number of shared GLU layers at each step
            'lambda_sparse': 1e-4,  # Sparsity regularization coefficient
            'optimizer_fn': optim.Adam,
            'optimizer_params': {'lr': 2e-2},
            'mask_type': 'entmax',  # Type of mask function
            'scheduler_params': {'step_size': 30, 'gamma': 0.95},
            'scheduler_fn': optim.lr_scheduler.StepLR,
            'verbose': 1 if self.verbose else 0,
            'device_name': 'cpu'
        }

        # Update with user-provided parameters
        if tabnet_params:
            default_params.update(tabnet_params)

        # Create TabNet regressor
        self.tabnet_model = TabNetRegressor(**default_params)

        # Convert data to float32 for compatibility
        X_train_f32 = X_train.astype(np.float32)
        X_test_f32 = X_test.astype(np.float32)
        y_train_f32 = y_train.values.astype(np.float32).reshape(-1, 1) if hasattr(y_train, 'values') else y_train.astype(np.float32).reshape(-1, 1)
        y_test_f32 = y_test.values.astype(np.float32).reshape(-1, 1) if hasattr(y_test, 'values') else y_test.astype(np.float32).reshape(-1, 1)

        # Create early stopping callback
        early_stopping = EarlyStopping(
            early_stopping_metric='val_0_mse',
            is_maximize=False,  # We want to minimize MSE
            tol=0.5,  # Minimum improvement threshold
            patience=8  # Number of epochs to wait
        )

        # Train the model with early stopping
        self.tabnet_model.fit(
            X_train=X_train_f32,
            y_train=y_train_f32,
            eval_set=[(X_test_f32, y_test_f32)],
            max_epochs=200,
            patience=20,  # Keep original patience as fallback
            batch_size=1024,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False,
            callbacks=[early_stopping]
        )

        # Generate predictions using float32 versions
        train_preds = self.tabnet_model.predict(X_train_f32).flatten()
        test_preds = self.tabnet_model.predict(X_test_f32).flatten()

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
            'feature_names': self.feature_columns,  # Store feature names for prediction
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

    def save_model(self) -> Dict[str, str]:
        """Save the trained TabNet model using ModelManager for consistency."""
        if self.tabnet_model is None:
            raise ValueError("No trained model to save")
        
        if self.scaler is None:
            raise ValueError("No scaler to save")

        # Use ModelManager to save TabNet with all required files
        saved_paths = self.model_manager.save_models(
            tabnet_model=self.tabnet_model,
            tabnet_scaler=self.scaler,
            tabnet_feature_columns=self.feature_columns,
            feature_state={
                'model_type': 'TabNet',
                'training_results': self.training_results,
                'timestamp': datetime.now().isoformat(),
                'musique_preprocessing': True,
                'db_type': self.db_type
            }
        )

        self.log_info(f"TabNet model saved using ModelManager:")
        self.log_info(f"  Model: {saved_paths.get('tabnet_model', 'Missing')}")
        self.log_info(f"  Scaler: {saved_paths.get('tabnet_scaler', 'Missing')}")
        self.log_info(f"  Config: {saved_paths.get('tabnet_config', 'Missing')}")
        
        return saved_paths

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


def main(progress_callback=None):
    """
    Main function to train the TabNet model from IDE.
    Follows the same pattern as train_race_model.py main() function.
    """
    if progress_callback:
        progress_callback(5, "Initializing TabNet trainer...")

    # Initialize the trainer
    trainer = TabNetTrainer(verbose=True)

    if progress_callback:
        progress_callback(10, "Loading and preparing data...")

    # Train the model with same parameters as HorseRaceModel
    results = trainer.train(
        limit=None,  # Set to limit races for testing
        race_filter=None,  # Set to 'A' for Attele, 'P' for Plat, etc.
        date_filter=None,  # Set date filter if needed
        test_size=0.2,  # Same as HorseRaceModel
        random_state=42  # Same as HorseRaceModel
    )

    if progress_callback:
        progress_callback(90, "Saving trained model...")

    # Save the trained model
    saved_paths = trainer.save_model()

    if progress_callback:
        progress_callback(100, "Training completed successfully!")

    # Print summary results (same format as HorseRaceModel)
    print("\n" + "=" * 50)
    print("TABNET TRAINING COMPLETED")
    print("=" * 50)
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Features used: {results['tabnet_results']['features']}")
    print(f"Train samples: {results['tabnet_results']['train_samples']}")
    print(f"Test samples: {results['tabnet_results']['test_samples']}")
    print(f"Test MAE: {results['tabnet_results']['test_mae']:.4f}")
    print(f"Test RMSE: {results['tabnet_results']['test_rmse']:.4f}")
    print(f"Test R²: {results['tabnet_results']['test_r2']:.4f}")
    print(f"Model saved to: {saved_paths.get('tabnet_model', 'N/A')}")
    
    # Display top feature importance
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    try:
        feature_importance = trainer.get_feature_importance(top_n=10)
        for idx, row in feature_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        print(f"Could not display feature importance: {e}")

    return results


if __name__ == "__main__":
    main()