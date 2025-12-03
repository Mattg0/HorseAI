"""
TabNet Trainer with Integrated Feature Selection

Trains TabNet models with automatic feature selection to optimize performance.
Uses a 3-phase training approach:
  1. Initial training on all features to get importance
  2. Feature selection based on sparsity, correlation, and importance
  3. Final training on selected features

Integrates seamlessly with existing pipeline - RF models remain unchanged.
"""

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
import torch

from pytorch_tabnet.tab_model import TabNetRegressor

from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.feature_selection.tabnet_feature_selector import TabNetFeatureSelector
from core.data_cleaning.tabnet_cleaner import TabNetDataCleaner
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from utils.model_manager import ModelManager


class TabNetTrainerWithSelection:
    """
    TabNet trainer with integrated automatic feature selection

    Key Features:
    - 3-phase training (importance → selection → final)
    - Automatic feature optimization
    - Saves both model and feature selector
    - Drop-in replacement for existing TabNet training
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = True):
        """
        Initialize the TabNet trainer

        Args:
            config_path: Path to config file
            verbose: Whether to print detailed progress
        """
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        db_path = get_sqlite_dbpath(self.db_type)
        self.model_manager = ModelManager()

        # Initialize data orchestrator
        self.orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=db_path,
            verbose=verbose
        )

        # Configure orchestrator to skip embeddings
        self.orchestrator.clean_after_embedding = False
        self.orchestrator.keep_identifiers = True

        # Model containers
        self.tabnet_model = None
        self.feature_selector = None
        self.scaler = None
        self.training_results = None

        # Data containers
        self.complete_df = None
        self.feature_columns = None

        self.log_info(f"Initialized TabNet Trainer with database: {self.db_type}")

    def log_info(self, message: str) -> None:
        """Simple logging method"""
        if self.verbose:
            print(f"[TabNetTrainer] {message}")

    def load_and_prepare_data(self,
                              limit: Optional[int] = None,
                              race_filter: Optional[str] = None,
                              date_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and prepare dataset with ALL features

        Args:
            limit: Maximum number of records to load
            race_filter: SQL filter for race types (e.g., "hippodrome = 'PAR'")
            date_filter: SQL filter for dates (e.g., "jour >= '2023-01-01'")

        Returns:
            Dictionary with preparation status
        """
        self.log_info("Loading historical race data...")

        # Load historical data
        df_historical = self.orchestrator.load_historical_races(
            limit=limit,
            race_filter=race_filter,
            date_filter=date_filter,
            use_cache=True
        )

        self.log_info("Preparing ALL features (selection happens during training)...")
        features_df = self.orchestrator.prepare_features(df_historical)

        # Get all numeric columns except target
        numeric_columns = []
        for col in features_df.columns:
            if col in ['final_position', 'cl']:
                continue
            try:
                pd.to_numeric(features_df[col], errors='raise')
                numeric_columns.append(col)
            except (ValueError, TypeError):
                pass

        self.feature_columns = numeric_columns
        final_columns = numeric_columns + ['final_position']
        self.complete_df = features_df[final_columns].dropna()

        self.log_info(
            f"Dataset prepared: {len(self.complete_df)} records, "
            f"{len(numeric_columns)} features (before selection)"
        )

        return {
            'status': 'success',
            'records': len(self.complete_df),
            'features': len(numeric_columns),
            'features_list': numeric_columns
        }

    def train(self,
              test_size: float = 0.2,
              validation_size: float = 0.1,
              sparse_threshold: float = 0.7,
              correlation_threshold: float = 0.95,
              target_features: int = 45,
              initial_epochs: int = 50,
              final_epochs: int = 200,
              batch_size: int = 256,
              virtual_batch_size: int = 128,
              **tabnet_params) -> Dict[str, Any]:
        """
        Train TabNet with 3-phase approach

        Phase 1: Quick training on ALL features to get importance
        Phase 2: Feature selection based on importance + correlation + sparsity
        Phase 3: Full training on SELECTED features

        Args:
            test_size: Fraction of data for test set
            validation_size: Fraction of training data for validation
            sparse_threshold: Remove features with >this fraction of zeros
            correlation_threshold: Remove highly correlated features above this threshold
            target_features: Target number of features to select
            initial_epochs: Epochs for phase 1 (importance)
            final_epochs: Epochs for phase 3 (final training)
            batch_size: Batch size for training
            virtual_batch_size: Virtual batch size for TabNet
            **tabnet_params: Additional TabNet parameters

        Returns:
            Dictionary with training results
        """
        if self.complete_df is None:
            raise ValueError("Must call load_and_prepare_data() before train()")

        print("\n" + "="*70)
        print("3-PHASE TABNET TRAINING WITH AUTOMATIC FEATURE SELECTION")
        print("="*70)

        # Prepare data
        X = self.complete_df[self.feature_columns]
        y = self.complete_df['final_position']

        print(f"\nTotal dataset: {len(X)} samples, {len(X.columns)} features")

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=42
        )

        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")

        # ===== PHASE 1: Initial Training for Importance =====
        print("\n" + "="*70)
        print("PHASE 1: INITIAL TRAINING FOR FEATURE IMPORTANCE")
        print("="*70)
        print(f"Training on ALL {len(X.columns)} features for {initial_epochs} epochs...")

        model_initial = TabNetRegressor(
            n_d=tabnet_params.get('n_d', 64),
            n_a=tabnet_params.get('n_a', 64),
            n_steps=tabnet_params.get('n_steps', 5),
            gamma=tabnet_params.get('gamma', 1.5),
            n_independent=tabnet_params.get('n_independent', 2),
            n_shared=tabnet_params.get('n_shared', 2),
            lambda_sparse=tabnet_params.get('lambda_sparse', 1e-4),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=tabnet_params.get('lr', 2e-2)),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=0,
            seed=42
        )

        model_initial.fit(
            X_train.values, y_train.values.reshape(-1, 1),
            eval_set=[(X_val.values, y_val.values.reshape(-1, 1))],
            max_epochs=initial_epochs,
            patience=10,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            eval_metric=['mae']
        )

        feature_importances = model_initial.feature_importances_
        initial_val_mae = mean_absolute_error(y_val, model_initial.predict(X_val.values))

        print(f"\n✓ Phase 1 complete:")
        print(f"  Validation MAE: {initial_val_mae:.3f}")
        print(f"  Feature importances extracted")

        # ===== PHASE 2: Feature Selection =====
        print("\n" + "="*70)
        print("PHASE 2: AUTOMATIC FEATURE SELECTION")
        print("="*70)

        self.feature_selector = TabNetFeatureSelector(
            sparse_threshold=sparse_threshold,
            correlation_threshold=correlation_threshold,
            target_features=target_features
        )

        X_train_selected = self.feature_selector.fit_transform(X_train, feature_importances)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)

        # ===== PHASE 3: Final Training on Selected Features =====
        print("\n" + "="*70)
        print("PHASE 3: FINAL TRAINING ON SELECTED FEATURES")
        print("="*70)
        print(f"Training on {len(X_train_selected.columns)} selected features for {final_epochs} epochs...")

        self.tabnet_model = TabNetRegressor(
            n_d=tabnet_params.get('n_d', 64),
            n_a=tabnet_params.get('n_a', 64),
            n_steps=tabnet_params.get('n_steps', 5),
            gamma=tabnet_params.get('gamma', 1.5),
            n_independent=tabnet_params.get('n_independent', 2),
            n_shared=tabnet_params.get('n_shared', 2),
            lambda_sparse=tabnet_params.get('lambda_sparse', 1e-4),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=tabnet_params.get('lr', 2e-2)),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=1,
            seed=42
        )

        self.tabnet_model.fit(
            X_train_selected.values, y_train.values.reshape(-1, 1),
            eval_set=[(X_val_selected.values, y_val.values.reshape(-1, 1))],
            max_epochs=final_epochs,
            patience=20,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            eval_metric=['mae']
        )

        # Evaluate on test set
        y_pred_test = self.tabnet_model.predict(X_test_selected.values).flatten()
        final_test_mae = mean_absolute_error(y_test, y_pred_test)
        final_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        final_test_r2 = r2_score(y_test, y_pred_test)

        print(f"\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"\nTest Set Performance:")
        print(f"  MAE:  {final_test_mae:.3f}")
        print(f"  RMSE: {final_test_rmse:.3f}")
        print(f"  R²:   {final_test_r2:.4f}")
        print(f"\nFeature Reduction:")
        print(f"  Original: {len(X.columns)} features")
        print(f"  Selected: {len(X_train_selected.columns)} features")
        print(f"  Reduction: {(1 - len(X_train_selected.columns)/len(X.columns))*100:.1f}%")

        self.training_results = {
            'test_mae': float(final_test_mae),
            'test_rmse': float(final_test_rmse),
            'test_r2': float(final_test_r2),
            'original_features': len(X.columns),
            'selected_features': len(X_train_selected.columns),
            'feature_reduction_pct': (1 - len(X_train_selected.columns)/len(X.columns))*100,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }

        return self.training_results

    def save_model(self, model_type: str = 'general', custom_path: Optional[str] = None) -> Path:
        """
        Save TabNet model and feature selector

        Args:
            model_type: Model type ('general' or 'quinte')
            custom_path: Optional custom save path

        Returns:
            Path where model was saved
        """
        if self.tabnet_model is None or self.feature_selector is None:
            raise ValueError("Must train model before saving")

        if custom_path:
            output_dir = Path(custom_path)
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d/%Hyears_%H%M%S")
            output_dir = Path(f'models/{timestamp}/{model_type}_tabnet')

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save TabNet model
        model_path = output_dir / 'tabnet_model'
        self.tabnet_model.save_model(str(model_path))
        self.log_info(f"✓ Saved TabNet model to {model_path}.zip")

        # Save feature selector
        selector_path = output_dir / 'feature_selector.json'
        self.feature_selector.save(str(selector_path))

        # Save feature list for compatibility
        features_path = output_dir / 'feature_columns.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_selector.selected_features, f, indent=2)
        self.log_info(f"✓ Saved feature list to {features_path}")

        # Save training config
        config_path = output_dir / 'tabnet_config.json'
        config_data = {
            'model_type': 'TabNet',
            'db_type': self.db_type,
            'created_at': datetime.now().isoformat(),
            'feature_columns': self.feature_selector.selected_features,
            'training_results': self.training_results,
            'feature_selection': self.feature_selector.get_feature_summary()
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        self.log_info(f"✓ Saved config to {config_path}")

        print(f"\n✓ Model saved to: {output_dir}")
        return output_dir

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using trained model with automatic feature selection

        Args:
            X: Features DataFrame (with ALL features)

        Returns:
            Predictions array
        """
        if self.tabnet_model is None or self.feature_selector is None:
            raise ValueError("Must train model before predicting")

        # Apply feature selection
        X_selected = self.feature_selector.transform(X)

        # Predict
        predictions = self.tabnet_model.predict(X_selected.values)

        return predictions.flatten()


def quick_train_tabnet(model_type: str = 'general',
                        limit: Optional[int] = None,
                        race_filter: Optional[str] = None,
                        target_features: int = 45,
                        config_path: str = 'config.yaml') -> tuple:
    """
    Quick convenience function to train TabNet with feature selection

    Args:
        model_type: 'general' or 'quinte'
        limit: Max records to load
        race_filter: SQL filter for races
        target_features: Target number of features
        config_path: Path to config file

    Returns:
        Tuple of (trainer, model_path)
    """
    trainer = TabNetTrainerWithSelection(config_path=config_path, verbose=True)

    # Load data
    trainer.load_and_prepare_data(
        limit=limit,
        race_filter=race_filter
    )

    # Train
    trainer.train(target_features=target_features)

    # Save
    model_path = trainer.save_model(model_type=model_type)

    return trainer, model_path
