#!/usr/bin/env python3
"""
Quinté Race Model Training Script

This script trains specialized models for Quinté+ races using:
- Data from historical_quinte and quinte_results tables
- Standard racing features from FeatureCalculator
- Specialized quinté features from QuinteFeatureCalculator
- TabNet and Random Forest models

The quinté-specific features capture patterns unique to quinté races:
- Larger field sizes (14-18+ horses)
- Higher purses and stakes
- Different betting dynamics
- Specialized horse performance in quinté conditions
"""

import pandas as pd
import numpy as np
import joblib
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import platform
import os

# Disable GPU on M1 processors to avoid hanging
if platform.processor() == 'arm' or 'arm64' in platform.machine().lower():
    print("[DEBUG-GPU] M1/ARM processor detected, disabling GPU for TensorFlow")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.calculators.static_feature_calculator import FeatureCalculator
from core.calculators.quinte_feature_calculator import QuinteFeatureCalculator, add_quinte_features_to_dataframe
from model_training.regressions.isotonic_calibration import CalibratedRegressor
from utils.model_manager import ModelManager

# Import TabNet model
try:
    from model_training.tabnet.tabnet_model import TabNetModel
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch.optim as optim
    from pytorch_tabnet.callbacks import EarlyStopping as TabNetEarlyStopping
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False


class QuinteRaceModel:
    """
    Specialized model trainer for Quinté+ races.

    Trains models specifically on quinté data with quinté-specific features:
    - Random Forest: Baseline model
    - TabNet: Attention-based neural network (best for quinté patterns)

    Uses historical_quinte table for training data and adds specialized features
    that capture quinté-specific performance patterns.
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = False):
        """Initialize the quinté model trainer."""
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        self.db_path = get_sqlite_dbpath(self.db_type)
        self.model_manager = ModelManager()

        # Model containers
        self.rf_model = None
        self.tabnet_model = None
        self.scaler = None
        self.training_results = None

        # Data containers
        self.complete_df = None
        self.feature_columns = None

        # Quinté feature calculator
        self.quinte_calculator = QuinteFeatureCalculator(self.db_path)

        self.log_info(f"Initialized QuinteRaceModel with database: {self.db_type}")
        if TABNET_AVAILABLE:
            self.log_info("TabNet model: Available")
        else:
            self.log_info("TabNet model: Not available")

    def log_info(self, message):
        """Simple logging method."""
        if self.verbose:
            print(f"[QuinteRaceModel] {message}")
        else:
            print(message)  # Always print for quinté training

    def load_quinté_races(self, limit: Optional[int] = None,
                         date_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Load quinté races from historical_quinte table.

        Args:
            limit: Maximum number of races to load
            date_filter: SQL date filter (e.g., "jour > '2023-01-01'")

        Returns:
            DataFrame with quinté race data
        """
        self.log_info("Loading quinté races from historical_quinte table...")

        conn = sqlite3.connect(self.db_path)

        # Build query
        query = "SELECT * FROM historical_quinte"

        where_clauses = []
        if date_filter:
            where_clauses.append(date_filter)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY jour DESC"

        if limit:
            query += f" LIMIT {limit}"

        df_races = pd.read_sql_query(query, conn)
        conn.close()

        self.log_info(f"Loaded {len(df_races)} quinté races")

        return df_races

    def expand_participants(self, df_races: pd.DataFrame) -> pd.DataFrame:
        """
        Expand participants JSON into individual rows.

        Args:
            df_races: DataFrame with quinté races

        Returns:
            DataFrame with one row per participant
        """
        self.log_info("Expanding participant data...")

        all_participants = []

        for _, race_row in df_races.iterrows():
            race_data = race_row.to_dict()
            participants_json = race_data.get('participants', '[]')

            try:
                participants = json.loads(participants_json)

                for participant in participants:
                    # Combine race-level and participant-level data
                    row = {**race_data, **participant}
                    # Remove the original participants JSON to avoid confusion
                    row.pop('participants', None)
                    all_participants.append(row)

            except (json.JSONDecodeError, TypeError) as e:
                self.log_info(f"Warning: Could not parse participants for race {race_data.get('comp')}: {e}")
                continue

        df_participants = pd.DataFrame(all_participants)
        self.log_info(f"Expanded to {len(df_participants)} participant records")

        return df_participants

    def load_quinté_results(self) -> Dict[str, Dict[int, int]]:
        """
        Load quinté results from quinte_results table.

        Returns:
            Dict mapping race_comp to dict of {horse_numero: finish_position}
            Example: {'R20250101C1': {7: 1, 16: 2, 1: 3, ...}}
        """
        self.log_info("Loading quinté results from quinte_results table...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT comp, ordre_arrivee FROM quinte_results")
        rows = cursor.fetchall()
        conn.close()

        results_map = {}
        for comp, ordre_arrivee in rows:
            try:
                # Parse ordre_arrivee as JSON array
                # Format: [{"narrivee": "1", "cheval": 7, "idche": 1248485}, ...]
                results_data = json.loads(ordre_arrivee)

                # Create mapping of horse_numero -> finish_position
                horse_positions = {}
                for result in results_data:
                    narrivee = result.get('narrivee', '')
                    cheval = result.get('cheval', 0)

                    # Only include numeric positions (skip "DAI", "ARR", etc.)
                    if str(narrivee).isdigit() and cheval > 0:
                        position = int(narrivee)
                        horse_positions[int(cheval)] = position

                if horse_positions:  # Only add if we have valid results
                    results_map[comp] = horse_positions

            except (json.JSONDecodeError, ValueError, AttributeError, TypeError) as e:
                self.log_info(f"Warning: Could not parse results for race {comp}: {e}")
                continue

        self.log_info(f"Loaded results for {len(results_map)} quinté races")

        return results_map

    def add_final_positions(self, df_participants: pd.DataFrame) -> pd.DataFrame:
        """
        Add final position to each participant based on quinte_results.

        Args:
            df_participants: DataFrame with participant data

        Returns:
            DataFrame with 'final_position' column added
        """
        self.log_info("Adding final positions from results...")

        results_map = self.load_quinté_results()

        # Debug: Check sample data
        if len(results_map) > 0:
            sample_comp = list(results_map.keys())[0]
            sample_results = results_map[sample_comp]
            self.log_info(f"DEBUG - Sample race comp: {sample_comp}")
            self.log_info(f"DEBUG - Sample horse positions: {sample_results}")

        if len(df_participants) > 0:
            sample_idx = df_participants.index[0]
            sample_comp = df_participants.at[sample_idx, 'comp']
            sample_numero = df_participants.at[sample_idx, 'numero']
            self.log_info(f"DEBUG - Sample participant comp: {sample_comp}")
            self.log_info(f"DEBUG - Sample participant numero: {sample_numero} (type: {type(sample_numero)})")

            # Check if comp exists in results
            if sample_comp in results_map:
                self.log_info(f"DEBUG - Results found for sample comp: {results_map[sample_comp]}")
            else:
                self.log_info(f"DEBUG - No results found for sample comp")
                # Show what comps we have
                sample_result_comps = list(results_map.keys())[:5]
                self.log_info(f"DEBUG - Sample result comps: {sample_result_comps}")
                sample_participant_comps = df_participants['comp'].unique()[:5].tolist()
                self.log_info(f"DEBUG - Sample participant comps: {sample_participant_comps}")

        # Initialize final_position column
        df_participants['final_position'] = np.nan

        matched_count = 0
        for idx, row in df_participants.iterrows():
            race_comp = row['comp']
            horse_numero = int(FeatureCalculator.safe_numeric(row.get('numero'), 0))

            if race_comp in results_map and horse_numero > 0:
                horse_positions = results_map[race_comp]

                # Direct lookup: horse_numero -> position
                if horse_numero in horse_positions:
                    position = horse_positions[horse_numero]
                    df_participants.at[idx, 'final_position'] = position
                    matched_count += 1

                    # Debug first match
                    if matched_count == 1:
                        self.log_info(f"DEBUG - First match: comp={race_comp}, numero={horse_numero}, position={position}")

        # Count how many have results
        with_results = df_participants['final_position'].notna().sum()
        self.log_info(f"Added final positions for {with_results}/{len(df_participants)} participants")

        # Filter to only participants with results for training
        df_with_results = df_participants[df_participants['final_position'].notna()].copy()
        self.log_info(f"Training dataset: {len(df_with_results)} participants with results")

        return df_with_results

    def prepare_quinté_features(self, df_participants: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare complete feature set for quinté races using batch loading:
        1. Standard racing features from FeatureCalculator
        2. Batch-load all quinté historical data once
        3. Calculate quinté features efficiently from pre-loaded data

        Args:
            df_participants: DataFrame with expanded participant data

        Returns:
            DataFrame with all features calculated
        """
        self.log_info("Calculating standard racing features with temporal calculations...")

        # Step 1: Calculate standard racing features with leakage fix
        df_with_features = FeatureCalculator.calculate_all_features(
            df_participants,
            use_temporal=True,
            db_path=self.db_path
        )

        self.log_info(f"Standard features calculated: {len(df_with_features.columns)} total columns")

        # Step 2: Batch-load all quinté historical data ONCE
        self.log_info("Batch-loading all quinté historical data...")
        all_quinte_data = self.quinte_calculator.batch_load_all_quinte_data()

        self.log_info(f"Loaded {len(all_quinte_data['races'])} historical races and {len(all_quinte_data['results'])} results")

        # Step 3: Add quinté features using batch-loaded data
        self.log_info("Calculating quinté features from batch-loaded data...")

        unique_races = df_with_features['comp'].unique()
        self.log_info(f"Processing {len(unique_races)} unique races...")

        # Process each race separately to add quinté features
        race_dfs = []
        for idx, race_comp in enumerate(unique_races):
            if idx % 100 == 0:
                self.log_info(f"  Processing race {idx}/{len(unique_races)}...")

            race_df = df_with_features[df_with_features['comp'] == race_comp].copy()

            if len(race_df) == 0:
                continue

            # Get race info from first row
            race_info = race_df.iloc[0].to_dict()
            race_date = race_info.get('jour')

            # Add quinté features using batch-loaded data
            try:
                race_df = self.quinte_calculator.add_batch_quinte_features(
                    df=race_df,
                    race_info=race_info,
                    before_date=race_date,
                    all_data=all_quinte_data  # Pass pre-loaded data
                )
                race_dfs.append(race_df)
            except Exception as e:
                self.log_info(f"Warning: Could not add quinté features for race {race_comp}: {e}")
                # Keep the race without quinté features
                race_dfs.append(race_df)

        # Combine all races
        df_complete = pd.concat(race_dfs, ignore_index=True)

        self.log_info(f"Complete feature set: {len(df_complete)} participants, {len(df_complete.columns)} features")

        return df_complete

    def select_tabnet_features(self, df: pd.DataFrame) -> list:
        """
        Select features for TabNet training.

        Includes:
        - Standard racing features
        - Quinté-specific features
        - Performance metrics
        - Temporal features

        Args:
            df: DataFrame with all features

        Returns:
            List of feature column names
        """
        # Quinté-specific features
        quinte_features = [
            'quinte_career_starts', 'quinte_win_rate', 'quinte_top5_rate',
            'avg_quinte_position', 'days_since_last_quinte',
            'quinte_handicap_specialist', 'quinte_conditions_specialist',
            'quinte_large_field_ability', 'quinte_track_condition_fit',
            'is_handicap_quinte', 'handicap_division', 'purse_level_category',
            'field_size_category', 'track_condition_PH', 'track_condition_DUR',
            'track_condition_PS', 'track_condition_PSF', 'weather_clear',
            'weather_rain', 'weather_cloudy', 'post_position_bias',
            'post_position_track_bias'
        ]

        # Musique features
        musique_features = [
            col for col in df.columns
            if any(prefix in col for prefix in ['che_global_', 'che_weighted_', 'che_bytype_',
                                               'joc_global_', 'joc_weighted_', 'joc_bytype_'])
        ]

        # Standard racing features (numeric only - categorical already encoded in quinte features)
        standard_features = [
            'age', 'dist', 'temperature', 'cotedirect', 'corde',
            'forceVent', 'directionVent',
            'victoirescheval', 'placescheval', 'coursescheval',
            'ratio_victoires', 'ratio_places', 'gains_par_course',
            'pourcVictChevalHippo', 'pourcPlaceChevalHippo',
            'pourcVictJockHippo', 'pourcPlaceJockHippo',
            'efficacite_couple', 'regularite_couple', 'progression_couple',
            'perf_cheval_hippo', 'perf_jockey_hippo',
            'recence', 'numero'
        ]

        # Combine all features
        all_features = quinte_features + musique_features + standard_features

        # Filter to only existing features and ensure they're numeric
        available_features = []
        for col in all_features:
            if col in df.columns:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    available_features.append(col)
                else:
                    self.log_info(f"Skipping non-numeric column: {col} (dtype: {df[col].dtype})")

        self.log_info(f"Selected {len(available_features)} features for training:")
        self.log_info(f"  - Quinté features: {len([f for f in available_features if f in quinte_features])}")
        self.log_info(f"  - Musique features: {len([f for f in available_features if any(p in f for p in ['che_', 'joc_'])])}")
        self.log_info(f"  - Standard features: {len([f for f in available_features if f in standard_features])}")

        return available_features

    def train(self, limit: Optional[int] = None,
              date_filter: Optional[str] = None,
              test_size: float = 0.2,
              random_state: int = 42) -> Dict[str, Any]:
        """
        Complete quinté model training workflow.

        Args:
            limit: Maximum number of races to load
            date_filter: SQL date filter
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Dict with training results
        """
        start_time = datetime.now()
        self.log_info("=" * 60)
        self.log_info("STARTING QUINTÉ MODEL TRAINING")
        self.log_info("=" * 60)

        # Step 1: Load quinté races
        df_races = self.load_quinté_races(limit, date_filter)

        # Step 2: Expand participants
        df_participants = self.expand_participants(df_races)

        # Step 3: Add final positions from results
        df_with_results = self.add_final_positions(df_participants)

        if len(df_with_results) == 0:
            raise ValueError("No participants with results found. Cannot train model.")

        # Step 4: Prepare features
        self.complete_df = self.prepare_quinté_features(df_with_results)

        # Step 5: Select features and target
        self.feature_columns = self.select_tabnet_features(self.complete_df)

        # Apply feature transformations (NO cleanup - keep ALL 90 features)
        from core.data_cleaning.feature_cleanup import FeatureCleaner
        cleaner = FeatureCleaner()

        X = self.complete_df[self.feature_columns].copy()

        # SKIP clean_features() - we want ALL features including bytype, global, etc.
        # These are critical for model performance (che_bytype_dnf_rate is 20% importance!)
        # X = cleaner.clean_features(X)  # ← DISABLED

        # Only apply transformations (recence_log, cotedirect_log)
        X = cleaner.apply_transformations(X)

        # Update feature_columns to reflect actual features after transformations
        self.feature_columns = list(X.columns)

        # FAIL-FAST: Validate we have the expected ~90 features
        expected_min_features = 85  # Should be ~89 after recence/cotedirect transform
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
{self.feature_columns[:20]}... (showing first 20)
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

        y = self.complete_df['final_position'].copy()

        # Handle missing values
        X = X.fillna(0)

        self.log_info(f"Feature matrix: {X.shape}")
        self.log_info(f"Target vector: {y.shape}")
        self.log_info(f"✓ Feature columns after transformations: {len(self.feature_columns)} features (keeping ALL features)")
        self.log_info(f"✅ All validation checks passed!")

        # Step 6: Split data
        self.log_info("Splitting data for training and testing...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Step 7: Train Random Forest
        self.log_info("Training Random Forest model...")
        rf_results = self._train_rf_model(X_train, y_train, X_test, y_test)

        # Step 8: Train TabNet
        tabnet_results = {}
        if TABNET_AVAILABLE:
            self.log_info("Training TabNet model...")
            tabnet_results = self._train_tabnet_model(X_train, y_train, X_test, y_test)
        else:
            self.log_info("TabNet not available - skipping")

        # Step 9: Compile results
        training_time = (datetime.now() - start_time).total_seconds()

        self.training_results = {
            'status': 'success',
            'training_time': training_time,
            'model_type': 'quinté_specialized',
            'data_stats': {
                'total_races': len(df_races),
                'total_participants': len(df_participants),
                'participants_with_results': len(df_with_results),
                'features': len(self.feature_columns)
            },
            'rf_results': rf_results,
            'tabnet_results': tabnet_results,
            'training_config': {
                'test_size': test_size,
                'random_state': random_state,
                'db_type': self.db_type,
                'date_filter': date_filter
            }
        }

        self.log_info("=" * 60)
        self.log_info(f"QUINTÉ TRAINING COMPLETED IN {training_time:.2f}s")
        self.log_info(f"RF Test MAE: {rf_results['test_mae']:.4f}")
        if tabnet_results and tabnet_results.get('status') == 'success':
            self.log_info(f"TabNet Test MAE: {tabnet_results['test_mae']:.4f}")
        self.log_info("=" * 60)

        return self.training_results

    def _train_rf_model(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train Random Forest model for quinté races."""

        base_rf = RandomForestRegressor(
            n_estimators=200,  # More trees for quinté complexity
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )

        self.rf_model = CalibratedRegressor(
            base_regressor=base_rf,
            clip_min=1.0
        )

        self.rf_model.fit(X_train, y_train)

        # Predictions
        train_preds = self.rf_model.predict(X_train)
        test_preds = self.rf_model.predict(X_test)

        # Metrics
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)

        # Feature importance
        feature_importance = None
        if hasattr(self.rf_model.base_regressor, 'feature_importances_'):
            feature_importance = self.rf_model.base_regressor.feature_importances_.tolist()

        return {
            'model_type': 'RandomForest_Quinté',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(self.feature_columns),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'test_r2': float(test_r2),
            'feature_importance': feature_importance,
            'feature_names': self.feature_columns
        }

    def _train_tabnet_model(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train TabNet model for quinté races."""

        try:
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # TabNet parameters (tuned for quinté)
            tabnet_params = {
                'n_d': 64,  # Larger for quinté complexity
                'n_a': 64,
                'n_steps': 6,  # More steps for larger fields
                'gamma': 1.5,
                'n_independent': 2,
                'n_shared': 2,
                'lambda_sparse': 1e-4,
                'optimizer_fn': optim.Adam,
                'optimizer_params': {'lr': 2e-2},
                'mask_type': 'entmax',
                'scheduler_params': {'step_size': 30, 'gamma': 0.95},
                'scheduler_fn': optim.lr_scheduler.StepLR,
                'verbose': 1,
                'device_name': 'cpu'
            }

            # Create model
            self.tabnet_model = TabNetRegressor(**tabnet_params)

            # Prepare data
            X_train_f32 = X_train_scaled.astype(np.float32)
            X_test_f32 = X_test_scaled.astype(np.float32)
            y_train_f32 = y_train.values.astype(np.float32).reshape(-1, 1) if hasattr(y_train, 'values') else y_train.astype(np.float32).reshape(-1, 1)
            y_test_f32 = y_test.values.astype(np.float32).reshape(-1, 1) if hasattr(y_test, 'values') else y_test.astype(np.float32).reshape(-1, 1)

            # Early stopping
            early_stopping = TabNetEarlyStopping(
                early_stopping_metric='val_0_mse',
                is_maximize=False,
                tol=0.5,
                patience=10
            )

            # Train
            self.tabnet_model.fit(
                X_train=X_train_f32,
                y_train=y_train_f32,
                eval_set=[(X_test_f32, y_test_f32)],
                max_epochs=200,
                patience=20,
                batch_size=512,  # Smaller for quinté
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
                callbacks=[early_stopping]
            )

            # Predictions
            train_preds = self.tabnet_model.predict(X_train_f32).flatten()
            test_preds = self.tabnet_model.predict(X_test_f32).flatten()

            # Metrics
            train_mae = mean_absolute_error(y_train, train_preds)
            test_mae = mean_absolute_error(y_test, test_preds)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            test_r2 = r2_score(y_test, test_preds)

            # Feature importance
            feature_importance = self.tabnet_model.feature_importances_

            return {
                'status': 'success',
                'model_type': 'TabNet_Quinté',
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': len(self.feature_columns),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'test_r2': float(test_r2),
                'feature_importance': feature_importance.tolist(),
                'feature_names': self.feature_columns
            }

        except Exception as e:
            self.log_info(f"TabNet training failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'failed',
                'message': str(e)
            }

    def save_models(self) -> Dict[str, str]:
        """
        Save trained quinté models using ModelManager.

        Returns:
            Dict with paths to saved models
        """
        self.log_info("Saving quinté models using ModelManager...")

        # Use ModelManager's save_quinte_models method
        saved_paths = self.model_manager.save_quinte_models(
            rf_model=self.rf_model,
            tabnet_model=self.tabnet_model,
            tabnet_scaler=self.scaler,
            feature_columns=self.feature_columns,
            training_results=self.training_results
        )

        return saved_paths


def main(progress_callback=None):
    """
    Main function to train quinté models from IDE.
    """
    if progress_callback:
        progress_callback(5, "Initializing quinté model...")

    # Initialize model
    model = QuinteRaceModel(verbose=True)

    if progress_callback:
        progress_callback(10, "Loading quinté data...")

    # Train the model
    results = model.train(
        limit=None,  # Set to limit races for testing
        date_filter=None,  # e.g., "jour > '2024-01-01'"
        test_size=0.2,
        random_state=42
    )

    if progress_callback:
        progress_callback(90, "Saving trained models...")

    # Save models
    saved_paths = model.save_models()

    if progress_callback:
        progress_callback(100, "Quinté training completed!")

    # Print summary
    print("\n" + "=" * 60)
    print("QUINTÉ MODEL TRAINING SUMMARY")
    print("=" * 60)
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Total races: {results['data_stats']['total_races']}")
    print(f"Total participants: {results['data_stats']['total_participants']}")
    print(f"Participants with results: {results['data_stats']['participants_with_results']}")
    print(f"Features: {results['data_stats']['features']}")

    # RF results
    print(f"\n--- Random Forest Results ---")
    rf_results = results['rf_results']
    print(f"Test MAE: {rf_results['test_mae']:.4f}")
    print(f"Test RMSE: {rf_results['test_rmse']:.4f}")
    print(f"Test R²: {rf_results['test_r2']:.4f}")

    # TabNet results
    tabnet_results = results.get('tabnet_results', {})
    if tabnet_results and tabnet_results.get('status') == 'success':
        print(f"\n--- TabNet Results ---")
        print(f"Test MAE: {tabnet_results['test_mae']:.4f}")
        print(f"Test RMSE: {tabnet_results['test_rmse']:.4f}")
        print(f"Test R²: {tabnet_results['test_r2']:.4f}")

    print(f"\nModels saved:")
    for model_name, path in saved_paths.items():
        print(f"  {model_name}: {path}")

    # Top features
    if rf_results.get('feature_importance') and rf_results.get('feature_names'):
        print("\nTop 15 Most Important Features (Random Forest):")
        print("-" * 50)
        importance_pairs = list(zip(rf_results['feature_importance'], rf_results['feature_names']))
        importance_pairs.sort(key=lambda x: x[0], reverse=True)

        for importance, name in importance_pairs[:15]:
            print(f"{name:50s}: {importance:.4f}")

    return results


if __name__ == "__main__":
    main()
